import csv
import io
from pathlib import Path

from django.db import transaction

from django.contrib import messages
from django.contrib.auth import authenticate, login, update_session_auth_hash
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib.auth.models import User
from django.db.models import Avg, Count, Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.utils import timezone

from .forms import SignupForm, StudentForm, AcademicRecordForm, TeacherDecisionForm, FeedbackForm, ProfileForm, ChangePasswordForm
from .models import Student, AcademicRecord, Prediction, SHAPExplanation, TeacherDecision, Feedback, Report, TeacherProfile
from .services.ml_service import predict_record, ModelNotReadyError
from .services.shap_service import explain_prediction
from .services.review_service import build_grounded_review


def landing(request):
    return render(request, "predictor/landing.html")


def signup(request):
    if request.user.is_authenticated:
        return redirect("dashboard")
    form = SignupForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        user = form.save()
        TeacherProfile.objects.create(user=user, department=form.cleaned_data.get("department", ""))
        login(request, user)
        messages.success(request, "Account created successfully.")
        return redirect("dashboard")
    return render(request, "predictor/signup.html", {"form": form})


def login_view(request):
    from django.contrib.auth.forms import AuthenticationForm
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == "POST" and form.is_valid():
        login(request, form.get_user())
        return redirect("dashboard")
    return render(request, "predictor/login.html", {"form": form})


def _risk_counts():
    latest = {}
    for prediction in Prediction.objects.select_related("student").order_by("-created_at"):
        latest.setdefault(prediction.student_id, prediction)
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for prediction in latest.values():
        counts[prediction.risk_level] = counts.get(prediction.risk_level, 0) + 1
    return counts, latest


@login_required
def dashboard(request):
    risk_counts, latest = _risk_counts()
    categories = {name: 0 for name in ["Excellent", "Good", "Average", "Needs Improvement", "At Risk"]}
    for prediction in latest.values():
        categories[prediction.predicted_category] = categories.get(prediction.predicted_category, 0) + 1

    recent_predictions = list(Prediction.objects.select_related("student").order_by("-created_at")[:8])
    context = {
        "total_predictions": Prediction.objects.count(),
        "avg_score": Prediction.objects.aggregate(avg=Avg("predicted_percentage"))["avg"],
        "needs_attention_count": categories.get("At Risk", 0),
        "recent_predictions": recent_predictions,
        "risk_counts": risk_counts,
        "category_counts": categories,
        "total_students": Student.objects.count(),
    }
    return render(request, "predictor/dashboard.html", context)


@login_required
def student_list(request):
    students = Student.objects.prefetch_related("academic_records", "predictions").all()
    rows = []
    for student in students:
        prediction = student.predictions.order_by("-created_at").first()
        record = student.academic_records.order_by("-semester").first()
        rows.append({
            "student_id": student.student_id,
            "student_name": student.full_name,
            "class_year": student.class_year,
            "predicted_percentage": prediction.predicted_percentage if prediction else None,
            "performance_category": prediction.predicted_category if prediction else "Not Predicted",
            "risk_level": prediction.risk_level if prediction else "—",
            "created_at": prediction.created_at if prediction else student.created_at,
            "attendance": record.attendance if record else None,
        })
    return render(request, "predictor/student_list.html", {"predictions": rows, "default_predictions": rows})


@login_required
def student_detail(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    records = student.academic_records.all()
    predictions = student.predictions.select_related("academic_record").all()
    return render(request, "predictor/student_detail.html", {"student": student, "records": records, "predictions": predictions})


@login_required
def student_create(request):
    form = StudentForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        student = form.save()
        messages.success(request, "Student added successfully.")
        return redirect("student_list")
    return render(request, "predictor/student_form.html", {"form": form, "is_edit": False})


@login_required
def student_update(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    form = StudentForm(request.POST or None, instance=student)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Student updated successfully.")
        return redirect("student_list")
    return render(request, "predictor/student_form.html", {"form": form, "student": student, "is_edit": True})


@login_required
def academic_record_edit(request, student_id, semester=1):
    student = get_object_or_404(Student, student_id=student_id)
    record, _ = AcademicRecord.objects.get_or_create(student=student, semester=semester)
    form = AcademicRecordForm(request.POST or None, instance=record)
    if request.method == "POST" and form.is_valid():
        record = form.save(commit=False)
        record.student = student
        record.semester = semester
        record.save()
        messages.success(request, "Academic record saved successfully.")
        return redirect("student_detail", student_id=student.student_id)
    return render(request, "predictor/academic_record_form.html", {"form": form, "student": student, "semester": semester, "record": record})


def _parse_float(value, field_name):
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number.") from exc


def _parse_int(value, field_name, default=0):
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a whole number.") from exc


CSV_FIELDS = [
    "student_id", "student_name", "class_year", "education_level", "age", "gender",
    "previous_total_marks", "previous_gained_marks", "internal_total_marks", "internal_gained_marks",
    "total_assignments", "completed_assignments", "backlogs", "past_failures", "attendance",
    "study_hours", "sleep_hours", "learning_mode", "tutoring_sessions", "internet_access",
    "participation", "motivation_level", "parental_involvement", "extracurricular",
]


def _import_csv_and_predict(request, uploaded_file):
    if not uploaded_file.name.lower().endswith(".csv"):
        raise ValueError("Only CSV files are supported by this prediction upload.")
    if uploaded_file.size > 10 * 1024 * 1024:
        raise ValueError("CSV file must be 10 MB or smaller.")

    raw = uploaded_file.read().decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(raw))
    if not reader.fieldnames:
        raise ValueError("The CSV file has no header row.")
    normalized = {name.strip().lower() for name in reader.fieldnames if name}
    missing = [name for name in CSV_FIELDS if name not in normalized]
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(missing))

    created = []
    errors = []
    for row_number, original in enumerate(reader, start=2):
        row = {str(k).strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in original.items()}
        try:
            student_id = row.get("student_id", "").strip()
            name = row.get("student_name", "").strip()
            if not student_id or not name:
                raise ValueError("student_id and student_name are required")
            with transaction.atomic():
                student, _ = Student.objects.update_or_create(
                    student_id=student_id,
                    defaults={
                        "full_name": name, "class_year": row.get("class_year", ""),
                        "education_level": row.get("education_level", ""),
                        "age": _parse_int(row.get("age"), "age", default=None),
                        "gender": row.get("gender", ""),
                    },
                )
                record, _ = AcademicRecord.objects.update_or_create(
                    student=student, semester=1,
                    defaults={
                        "previous_total_marks": _parse_float(row.get("previous_total_marks"), "previous_total_marks"),
                        "previous_gained_marks": _parse_float(row.get("previous_gained_marks"), "previous_gained_marks"),
                        "internal_total_marks": _parse_float(row.get("internal_total_marks"), "internal_total_marks"),
                        "internal_gained_marks": _parse_float(row.get("internal_gained_marks"), "internal_gained_marks"),
                        "total_assignments": _parse_int(row.get("total_assignments"), "total_assignments", default=0),
                        "completed_assignments": _parse_int(row.get("completed_assignments"), "completed_assignments", default=0),
                        "backlogs": _parse_int(row.get("backlogs"), "backlogs"),
                        "past_failures": _parse_int(row.get("past_failures"), "past_failures"),
                        "attendance": _parse_float(row.get("attendance"), "attendance"),
                        "study_hours": _parse_float(row.get("study_hours"), "study_hours"),
                        "sleep_hours": _parse_float(row.get("sleep_hours"), "sleep_hours"),
                        "learning_mode": row.get("learning_mode", ""),
                        "tutoring_sessions": _parse_int(row.get("tutoring_sessions"), "tutoring_sessions"),
                        "internet_access": row.get("internet_access", ""),
                        "participation": _parse_float(row.get("participation"), "participation"),
                        "motivation_level": row.get("motivation_level", ""),
                        "parental_involvement": row.get("parental_involvement", ""),
                        "extracurricular": row.get("extracurricular", ""),
                    },
                )
                result = predict_record(record)
                prediction = Prediction.objects.create(
                    student=student, academic_record=record,
                    predicted_category=result["category"], predicted_percentage=result["predicted_percentage"],
                    confidence=result["confidence"], risk_level=result["risk_level"],
                    probabilities=result["probabilities"], model_name=result["model_name"], created_by=request.user,
                )
                try:
                    SHAPExplanation.objects.create(prediction=prediction, **explain_prediction(result["model"], result["frame"]))
                except RuntimeError:
                    pass
                created.append(prediction)
        except Exception as exc:
            errors.append(f"Row {row_number}: {exc}")
    return created, errors


@login_required
def predict_performance(request):
    if request.method == "POST" and request.FILES.get("csv_file"):
        try:
            created, errors = _import_csv_and_predict(request, request.FILES["csv_file"])
            if created:
                messages.success(request, f"Processed {len(created)} student prediction(s) successfully.")
            if errors:
                messages.warning(request, "Some rows were not processed: " + " | ".join(errors[:5]))
            return redirect("student_list")
        except (ValueError, UnicodeDecodeError) as exc:
            messages.error(request, str(exc))
            return render(request, "predictor/predict_form.html")

    if request.method == "POST":
        data = request.POST
        try:
            student, _ = Student.objects.update_or_create(
                student_id=data.get("student_id", "").strip(),
                defaults={
                    "full_name": data.get("student_name", "").strip(),
                    "class_year": data.get("class_year", "").strip(),
                    "education_level": data.get("education_level", "").strip(),
                    "age": int(data["age"]) if data.get("age") else None,
                    "gender": data.get("gender", ""),
                },
            )
            if not student.student_id or not student.full_name:
                raise ValueError("Student ID and student name are required.")
            def f(name):
                value = data.get(name)
                return float(value) if value not in (None, "") else None
            def i(name):
                value = data.get(name)
                return int(value) if value not in (None, "") else 0
            previous_total = f("previous_total_marks")
            previous_gained = f("previous_gained_marks")
            internal_total = f("internal_total_marks")
            internal_gained = f("internal_gained_marks")
            total_assignments = i("total_assignments")
            completed_assignments = i("completed_assignments")
            record, _ = AcademicRecord.objects.update_or_create(
                student=student,
                semester=1,
                defaults={
                    "previous_total_marks": previous_total,
                    "previous_gained_marks": previous_gained,
                    "internal_total_marks": internal_total,
                    "internal_gained_marks": internal_gained,
                    "total_assignments": total_assignments,
                    "completed_assignments": completed_assignments,
                    "backlogs": i("backlogs"),
                    "past_failures": i("past_failures"),
                    "attendance": f("attendance"),
                    "study_hours": f("study_hours"),
                    "sleep_hours": f("sleep_hours"),
                    "learning_mode": data.get("learning_mode", ""),
                    "tutoring_sessions": i("tutoring_sessions"),
                    "internet_access": data.get("internet_access", ""),
                    "participation": f("participation"),
                    "motivation_level": data.get("motivation_level", ""),
                    "parental_involvement": data.get("parental_involvement", ""),
                    "extracurricular": data.get("extracurricular", ""),
                },
            )
            result = predict_record(record)
            prediction = Prediction.objects.create(
                student=student,
                academic_record=record,
                predicted_category=result["category"],
                predicted_percentage=result["predicted_percentage"],
                confidence=result["confidence"],
                risk_level=result["risk_level"],
                probabilities=result["probabilities"],
                model_name=result["model_name"],
                created_by=request.user,
            )
            try:
                shap_data = explain_prediction(result["model"], result["frame"])
                SHAPExplanation.objects.create(prediction=prediction, **shap_data)
            except RuntimeError as exc:
                messages.warning(request, str(exc))
            review = build_grounded_review(record, prediction)
            context = {"student_name": student.full_name, "student_id": student.student_id, "class_year": student.class_year, "prediction": prediction, "predicted_percentage": prediction.predicted_percentage, "performance_category": prediction.predicted_category, "risk_level": prediction.risk_level, "confidence": prediction.confidence, "probabilities": prediction.probabilities, "review": review}
            return render(request, "predictor/predict_result.html", context)
        except (ValueError, ModelNotReadyError) as exc:
            messages.error(request, str(exc))
            return render(request, "predictor/predict_form.html", {"form_data": request.POST})
    return render(request, "predictor/predict_form.html")


@login_required
def report_detail(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    prediction = student.predictions.select_related("academic_record").first()
    if not prediction:
        messages.info(request, "No prediction exists for this student yet.")
        return redirect("student_list")
    explanation = getattr(prediction, "shap_explanation", None)
    review = build_grounded_review(prediction.academic_record, prediction, explanation)
    factors = explanation.local_features if explanation else []
    context = {"student_id": student.student_id, "student_name": student.full_name, "class_year": student.class_year, "predicted_percentage": prediction.predicted_percentage, "performance_category": prediction.predicted_category, "risk_level": prediction.risk_level, "confidence": prediction.confidence, "factors": [{"name": f["name"], "impact": f["shap_value"], "bar_width": min(100, abs(f["shap_value"]) * 100)} for f in factors], "positive_points": review["strengths"], "improve_points": review["improvements"], "prediction": prediction}
    return render(request, "predictor/report_detail.html", context)


@login_required
def compare_students(request):
    students = Student.objects.order_by("student_id")
    return render(request, "predictor/compare_students.html", {"students": students})


@login_required
def compare_students_api(request):
    ids = [request.GET.get("a", "").strip(), request.GET.get("b", "").strip()]
    if not all(ids) or ids[0] == ids[1]:
        return JsonResponse({"error": "Enter two different student IDs."}, status=400)
    payload = []
    for sid in ids:
        student = get_object_or_404(Student, student_id=sid)
        record = student.academic_records.order_by("-semester").first()
        prediction = student.predictions.order_by("-created_at").first()
        if not record:
            return JsonResponse({"error": f"No academic record exists for {sid}."}, status=400)
        payload.append({"student_id": sid, "student_name": student.full_name, "class_year": student.class_year, "category": prediction.predicted_category if prediction else "Not Predicted", "score": prediction.predicted_percentage if prediction else None, "attendance": record.attendance, "previous_percent": record.previous_percent, "study_hours": record.study_hours, "backlogs": record.backlogs})
    return JsonResponse({"students": payload})


@login_required
def class_analytics(request):
    categories = Prediction.objects.values("predicted_category").annotate(count=Count("id"))
    total = sum(item["count"] for item in categories)
    default_categories = [{"name": item["predicted_category"], "count": item["count"], "percent": round(item["count"] * 100 / total, 1) if total else 0, "color_class": "bg-primary"} for item in categories]
    avg = Prediction.objects.aggregate(avg=Avg("predicted_percentage"))["avg"]
    context = {"default_categories": default_categories, "analytics": {"total": total, "average": avg}}
    return render(request, "predictor/class_analytics.html", context)


@login_required
def profile(request):
    profile_obj, _ = TeacherProfile.objects.get_or_create(user=request.user)
    if request.method == "POST":
        form = ProfileForm(request.POST)
        if form.is_valid():
            parts = form.cleaned_data["full_name"].strip().split(maxsplit=1)
            request.user.first_name = parts[0] if parts else ""
            request.user.last_name = parts[1] if len(parts) > 1 else ""
            request.user.save(update_fields=["first_name", "last_name"])
            profile_obj.department = form.cleaned_data["department"]
            profile_obj.phone = form.cleaned_data["phone"]
            profile_obj.institute = form.cleaned_data["institute"]
            profile_obj.save()
            messages.success(request, "Profile updated successfully.")
            return redirect("profile")
    else:
        form = ProfileForm(initial={"full_name": request.user.get_full_name(), "department": profile_obj.department, "phone": profile_obj.phone, "institute": profile_obj.institute})
    return render(request, "predictor/profile.html", {"form": form, "department": profile_obj.department, "phone": profile_obj.phone, "institute": profile_obj.institute})


@login_required
def change_password(request):
    form = ChangePasswordForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        if not request.user.check_password(form.cleaned_data["current_password"]):
            form.add_error("current_password", "Current password is incorrect.")
        else:
            request.user.set_password(form.cleaned_data["new_password"])
            request.user.save()
            update_session_auth_hash(request, request.user)
            messages.success(request, "Password changed successfully.")
            return redirect("profile")
    return render(request, "predictor/profile.html", {"password_form": form})


@login_required
def teacher_decision(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id)
    decision, _ = TeacherDecision.objects.get_or_create(prediction=prediction, teacher=request.user)
    form = TeacherDecisionForm(request.POST or None, instance=decision)
    if request.method == "POST" and form.is_valid():
        decision = form.save(commit=False)
        decision.prediction = prediction
        decision.teacher = request.user
        decision.save()
        messages.success(request, "Teacher decision saved.")
        return redirect("report_detail", prediction.student.student_id)
    return render(request, "predictor/teacher_decision.html", {"form": form, "prediction": prediction})


@login_required
def feedback(request, prediction_id):
    prediction = get_object_or_404(Prediction, id=prediction_id)
    form = FeedbackForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        entry = form.save(commit=False)
        entry.prediction = prediction
        entry.teacher = request.user
        entry.save()
        messages.success(request, "Feedback submitted successfully.")
        return redirect("report_detail", prediction.student.student_id)
    return render(request, "predictor/feedback_form.html", {"form": form, "prediction": prediction})


@login_required
def report_pdf(request, student_id):
    student = get_object_or_404(Student, student_id=student_id)
    prediction = student.predictions.select_related("academic_record").first()
    if not prediction:
        messages.error(request, "Generate a prediction before creating a report.")
        return redirect("student_list")
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        messages.error(request, "ReportLab is not installed. Install reportlab to generate PDFs.")
        return redirect("report_detail", student_id=student.student_id)

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{student.student_id}-hc-xai-report.pdf"'
    pdf = canvas.Canvas(response, pagesize=A4)
    y = 800
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(50, y, "HC-XAI Student Performance Report")
    y -= 35
    pdf.setFont("Helvetica", 11)
    lines = [
        f"Student ID: {student.student_id}", f"Student Name: {student.full_name}", f"Class/Year: {student.class_year or '—'}",
        f"Prediction: {prediction.predicted_category}", f"Risk Level: {prediction.risk_level}", f"Confidence: {prediction.confidence or '—'}%",
        f"Attendance: {prediction.academic_record.attendance or '—'}%", f"Study Hours: {prediction.academic_record.study_hours or '—'} / week",
        f"Backlogs: {prediction.academic_record.backlogs}",
    ]
    for line in lines:
        pdf.drawString(50, y, line)
        y -= 20
    pdf.setFont("Helvetica-Oblique", 9)
    pdf.drawString(50, 90, "AI output is decision support. The teacher remains the final decision-maker.")
    pdf.save()
    return response
