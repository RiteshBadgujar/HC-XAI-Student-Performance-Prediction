import csv
import io
from pathlib import Path
from django.urls import reverse

from django.db import transaction

from django.contrib import messages
from django.contrib.auth import authenticate, login, update_session_auth_hash
from django.contrib.auth.decorators import login_required
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
from .services.review_service import build_grounded_review, build_shap_sections


# ---------------------------------------------------------------------------
# Public views
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FIX 1: All helpers scoped to a user
# ---------------------------------------------------------------------------

def _user_students(user):
    """Base queryset: only students owned by this user."""
    return Student.objects.filter(owner=user)


def _user_predictions(user):
    """Base queryset: only predictions for students owned by this user."""
    return Prediction.objects.filter(student__owner=user)


def _risk_counts(user):
    """Latest prediction per student, scoped to the logged-in teacher."""
    latest = {}
    for prediction in _user_predictions(user).select_related("student").order_by("-created_at"):
        latest.setdefault(prediction.student_id, prediction)
    counts = {"Low": 0, "Medium": 0, "High": 0}
    for prediction in latest.values():
        counts[prediction.risk_level] = counts.get(prediction.risk_level, 0) + 1
    return counts, latest


# ---------------------------------------------------------------------------
# Dashboard — scoped to request.user
# ---------------------------------------------------------------------------

@login_required
def dashboard(request):
    risk_counts, latest = _risk_counts(request.user)
    categories = {name: 0 for name in ["Excellent", "Good", "Average", "Needs Improvement", "At Risk"]}
    for prediction in latest.values():
        categories[prediction.predicted_category] = categories.get(prediction.predicted_category, 0) + 1

    user_predictions = _user_predictions(request.user)
    recent_predictions = list(user_predictions.select_related("student").order_by("-created_at")[:8])
    context = {
        "total_predictions": user_predictions.count(),
        "avg_score": user_predictions.aggregate(avg=Avg("predicted_percentage"))["avg"],
        "needs_attention_count": categories.get("At Risk", 0),
        "recent_predictions": recent_predictions,
        "risk_counts": risk_counts,
        "category_counts": categories,
        "total_students": _user_students(request.user).count(),
    }
    return render(request, "predictor/dashboard.html", context)


# ---------------------------------------------------------------------------
# Student views — scoped to request.user
# ---------------------------------------------------------------------------

@login_required
def student_list(request):
    students = _user_students(request.user).prefetch_related("academic_records", "predictions")
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
    # FIX 1: restrict to owner
    student = get_object_or_404(Student, student_id=student_id, owner=request.user)
    records = student.academic_records.all()
    predictions = student.predictions.select_related("academic_record").all()
    return render(request, "predictor/student_detail.html", {"student": student, "records": records, "predictions": predictions})


@login_required
def student_create(request):
    form = StudentForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        student = form.save(commit=False)
        student.owner = request.user   # FIX 1: assign owner
        student.save()
        messages.success(request, "Student added successfully.")
        return redirect("student_list")
    return render(request, "predictor/student_form.html", {"form": form, "is_edit": False})


@login_required
def student_update(request, student_id):
    # FIX 1: restrict to owner
    student = get_object_or_404(Student, student_id=student_id, owner=request.user)
    form = StudentForm(request.POST or None, instance=student)
    if request.method == "POST" and form.is_valid():
        form.save()
        messages.success(request, "Student updated successfully.")
        return redirect("student_list")
    return render(request, "predictor/student_form.html", {"form": form, "student": student, "is_edit": True})


@login_required
def academic_record_edit(request, student_id, semester=1):
    # FIX 1: restrict to owner
    student = get_object_or_404(Student, student_id=student_id, owner=request.user)
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


# ---------------------------------------------------------------------------
# CSV parsing helpers (unchanged)
# ---------------------------------------------------------------------------

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
                # FIX 1: scope update_or_create to owner=request.user
                student, _ = Student.objects.update_or_create(
                    owner=request.user,
                    student_id=student_id,
                    defaults={
                        "full_name": name,
                        "class_year": row.get("class_year", ""),
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


# ---------------------------------------------------------------------------
# Predict — scoped to request.user
# ---------------------------------------------------------------------------

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
            # FIX 1: update_or_create scoped to owner=request.user
            student, _ = Student.objects.update_or_create(
                owner=request.user,
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

            record, _ = AcademicRecord.objects.update_or_create(
                student=student,
                semester=1,
                defaults={
                    "previous_total_marks": f("previous_total_marks"),
                    "previous_gained_marks": f("previous_gained_marks"),
                    "internal_total_marks": f("internal_total_marks"),
                    "internal_gained_marks": f("internal_gained_marks"),
                    "total_assignments": i("total_assignments"),
                    "completed_assignments": i("completed_assignments"),
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
            shap_data = None
            try:
                shap_data = explain_prediction(result["model"], result["frame"])
                SHAPExplanation.objects.create(prediction=prediction, **shap_data)
            except RuntimeError as exc:
                messages.warning(request, str(exc))

            review = build_grounded_review(record, prediction)

            # FIX 2: build teacher-facing SHAP sections using real values
            explanation_obj = getattr(prediction, "shap_explanation", None)
            shap_sections = build_shap_sections(record, explanation_obj)

            context = {
                "student_name": student.full_name,
                "student_id": student.student_id,
                "class_year": student.class_year,
                "prediction": prediction,
                "predicted_percentage": prediction.predicted_percentage,
                "performance_category": prediction.predicted_category,
                "risk_level": prediction.risk_level,
                "confidence": prediction.confidence,
                "probabilities": prediction.probabilities,
                "review": review,
                "shap_sections": shap_sections,   # FIX 2: added
            }
            return render(request, "predictor/predict_result.html", context)
        except (ValueError, ModelNotReadyError) as exc:
            messages.error(request, str(exc))
            return render(request, "predictor/predict_form.html", {"form_data": request.POST})
    return render(request, "predictor/predict_form.html")


# ---------------------------------------------------------------------------
# Report detail — scoped to request.user
# ---------------------------------------------------------------------------

@login_required
def report_detail(request, student_id):
    # FIX 1: restrict to owner
    student = get_object_or_404(Student, student_id=student_id, owner=request.user)
    prediction = student.predictions.select_related("academic_record").first()
    if not prediction:
        messages.info(request, "No prediction exists for this student yet.")
        return redirect("student_list")
    explanation = getattr(prediction, "shap_explanation", None)
    review = build_grounded_review(prediction.academic_record, prediction, explanation)

    # FIX 2: build teacher-facing SHAP sections
    shap_sections = build_shap_sections(prediction.academic_record, explanation)

    factors = explanation.local_features if explanation else []
    context = {
        "student_id": student.student_id,
        "student_name": student.full_name,
        "class_year": student.class_year,
        "predicted_percentage": prediction.predicted_percentage,
        "performance_category": prediction.predicted_category,
        "risk_level": prediction.risk_level,
        "confidence": prediction.confidence,
        "factors": [{"name": f["name"], "impact": f["shap_value"], "bar_width": min(100, abs(f["shap_value"]) * 100)} for f in factors],
        "positive_points": review["strengths"],
        "improve_points": review["improvements"],
        "prediction": prediction,
        "shap_sections": shap_sections,   # FIX 2: added
    }
    return render(request, "predictor/report_detail.html", context)


# ---------------------------------------------------------------------------
# Compare — scoped to request.user
# ---------------------------------------------------------------------------

@login_required
def compare_students(request):
    # FIX 1: only show logged-in teacher's students in the datalist
    students = _user_students(request.user).order_by("student_id")
    return render(request, "predictor/compare_students.html", {"students": students})


@login_required
def compare_students_api(request):
    ids = [request.GET.get("a", "").strip(), request.GET.get("b", "").strip()]
    if not all(ids) or ids[0] == ids[1]:
        return JsonResponse({"error": "Enter two different student IDs."}, status=400)
    payload = []
    for sid in ids:
        # FIX 1: restrict comparison to owner's students
        student = get_object_or_404(Student, student_id=sid, owner=request.user)
        record = student.academic_records.order_by("-semester").first()
        prediction = student.predictions.order_by("-created_at").first()
        if not record:
            return JsonResponse({"error": f"No academic record exists for {sid}."}, status=400)
        payload.append({
            "student_id": sid,
            "student_name": student.full_name,
            "class_year": student.class_year,
            "category": prediction.predicted_category if prediction else "Not Predicted",
            "score": prediction.predicted_percentage if prediction else None,
            "attendance": record.attendance,
            "previous_percent": record.previous_percent,
            "study_hours": record.study_hours,
            "backlogs": record.backlogs,
        })
    return JsonResponse({"students": payload})


# ---------------------------------------------------------------------------
# Analytics — scoped to request.user
# ---------------------------------------------------------------------------

# Replace your existing class_analytics view in predictor/views.py with this.
# Also add this import at the top of views.py:
#     from django.urls import reverse

@login_required
def class_analytics(request):
    """Analytics for the logged-in teacher's latest prediction per student."""

    # ---- Semester filter (?semester=1) -----------------------------------
    semesters = sorted(
        set(
            AcademicRecord.objects
            .filter(student__owner=request.user)
            .values_list("semester", flat=True)
        )
    )
    selected_semester = request.GET.get("semester", "").strip()
    if selected_semester and not selected_semester.isdigit():
        selected_semester = ""

    queryset = (
        _user_predictions(request.user)
        .select_related("student", "academic_record", "shap_explanation")
        .order_by("-created_at")
    )
    if selected_semester:
        queryset = queryset.filter(academic_record__semester=int(selected_semester))

    latest_predictions = {}
    for prediction in queryset:
        latest_predictions.setdefault(prediction.student_id, prediction)
    predictions = list(latest_predictions.values())

    category_names = ["Excellent", "Good", "Average", "Needs Improvement", "At Risk"]
    risk_names = ["Low", "Medium", "High"]

    def main_factor(prediction):
        """The factor pulling the prediction down the most (most negative SHAP)."""
        explanation = getattr(prediction, "shap_explanation", None)
        if not explanation:
            return "—"
        worst_name, worst_value = None, 0.0
        for factor in (explanation.local_features or []):
            try:
                value = float(factor.get("shap_value", 0))
            except (TypeError, ValueError):
                continue
            if value < worst_value:
                worst_name, worst_value = factor.get("name"), value
        return worst_name or "—"

    def build_analytics(items):
        category_counts = {n: sum(1 for p in items if p.predicted_category == n) for n in category_names}
        risk_counts = {n: sum(1 for p in items if p.risk_level == n) for n in risk_names}

        scores = [float(p.predicted_percentage) for p in items if p.predicted_percentage is not None]

        attendance_score, study_hours_score = [], []
        for p in items:
            record = p.academic_record
            if not record or p.predicted_percentage is None:
                continue
            if record.attendance is not None:
                attendance_score.append({
                    "x": float(record.attendance),
                    "y": float(p.predicted_percentage),
                    "student": p.student.full_name,
                })
            if record.study_hours is not None:
                study_hours_score.append({
                    "x": float(record.study_hours),
                    "y": float(p.predicted_percentage),
                    "student": p.student.full_name,
                })

        # Average absolute SHAP impact per factor
        shap_totals, shap_counts = {}, {}
        for p in items:
            explanation = getattr(p, "shap_explanation", None)
            if not explanation:
                continue
            for factor in (explanation.local_features or []):
                name = factor.get("name")
                if not name:
                    continue
                try:
                    value = abs(float(factor.get("shap_value", 0)))
                except (TypeError, ValueError):
                    continue
                shap_totals[name] = shap_totals.get(name, 0.0) + value
                shap_counts[name] = shap_counts.get(name, 0) + 1

        shap_factors = [
            {
                "name": name,
                "average_impact": round(total / shap_counts[name], 4),
                "students": shap_counts[name],
            }
            for name, total in shap_totals.items()
            if shap_counts.get(name)
        ]
        shap_factors.sort(key=lambda row: row["average_impact"], reverse=True)

        # Students needing attention: weak category or high risk, lowest score first
        flagged = [
            p for p in items
            if p.predicted_category in ("Needs Improvement", "At Risk") or p.risk_level == "High"
        ]
        flagged.sort(key=lambda p: (p.predicted_percentage is None, p.predicted_percentage or 0))
        attention = [
            {
                "student_id": p.student.student_id,
                "name": p.student.full_name,
                "class_name": (p.student.class_year or "").strip() or "Not Specified",
                "score": round(float(p.predicted_percentage), 2) if p.predicted_percentage is not None else None,
                "risk": p.risk_level,
                "main_factor": main_factor(p),
                "url": reverse("report_detail", args=[p.student.student_id]),
            }
            for p in flagged[:10]
        ]

        return {
            "total": len(items),
            "average_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "categories": [{"name": n, "count": category_counts[n]} for n in category_names],
            "risk_levels": [{"name": n, "count": risk_counts[n]} for n in risk_names],
            "attendance_score": attendance_score,
            "study_hours_score": study_hours_score,
            "shap_factors": shap_factors[:10],
            "attention": attention,
        }

    all_data = build_analytics(predictions)

    grouped = {}
    for prediction in predictions:
        class_name = (prediction.student.class_year or "").strip() or "Not Specified"
        grouped.setdefault(class_name, []).append(prediction)

    analytics_by_class = {
        class_name: build_analytics(items)
        for class_name, items in sorted(grouped.items(), key=lambda item: item[0].lower())
    }

    all_data["class_scores"] = [
        {"name": name, "score": data["average_score"]}
        for name, data in analytics_by_class.items()
        if data["total"] > 0
    ]
    analytics_by_class["all"] = all_data

    context = {
        "analytics_json": analytics_by_class,                      # must stay a dict
        "class_names": [n for n in analytics_by_class if n != "all"],
        "semesters": semesters,
        "selected_semester": selected_semester,
    }
    return render(request, "predictor/class_analytics.html", context)

# ---------------------------------------------------------------------------
# Profile / Password (no student data — unchanged)
# ---------------------------------------------------------------------------

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
        form = ProfileForm(initial={
            "full_name": request.user.get_full_name(),
            "department": profile_obj.department,
            "phone": profile_obj.phone,
            "institute": profile_obj.institute,
        })
    return render(request, "predictor/profile.html", {
        "form": form,
        "department": profile_obj.department,
        "phone": profile_obj.phone,
        "institute": profile_obj.institute,
    })


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


# ---------------------------------------------------------------------------
# Teacher decision — scoped to request.user
# ---------------------------------------------------------------------------

@login_required
def teacher_decision(request, prediction_id):
    # FIX 1: ensure prediction belongs to this user's student
    prediction = get_object_or_404(Prediction, id=prediction_id, student__owner=request.user)
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


# ---------------------------------------------------------------------------
# Feedback — scoped to request.user
# ---------------------------------------------------------------------------

@login_required
def feedback(request, prediction_id):
    # FIX 1: ensure prediction belongs to this user's student
    prediction = get_object_or_404(Prediction, id=prediction_id, student__owner=request.user)
    form = FeedbackForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        entry = form.save(commit=False)
        entry.prediction = prediction
        entry.teacher = request.user
        entry.save()
        messages.success(request, "Feedback submitted successfully.")
        return redirect("report_detail", prediction.student.student_id)
    return render(request, "predictor/feedback_form.html", {"form": form, "prediction": prediction})


# ---------------------------------------------------------------------------
# PDF Report — scoped to request.user
# ---------------------------------------------------------------------------

@login_required
def report_pdf(request, student_id):
    # FIX 1: restrict to owner
    student = get_object_or_404(Student, student_id=student_id, owner=request.user)

    prediction = (
        student.predictions
        .select_related("academic_record")
        .order_by("-created_at")
        .first()
    )

    if not prediction:
        messages.error(request, "Generate a prediction before creating a report.")
        return redirect("student_list")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except ImportError:
        messages.error(request, "ReportLab is not installed. Install it using: pip install reportlab")
        return redirect("report_detail", student_id=student.student_id)

    record = prediction.academic_record
    explanation = getattr(prediction, "shap_explanation", None)

    # FIX 2: build real SHAP sections for the PDF
    shap_sections = build_shap_sections(record, explanation)

    try:
        review = build_grounded_review(record, prediction, explanation)
        positive_points = review.get("strengths", [])
        improve_points = review.get("improvements", [])
    except Exception:
        positive_points = []
        improve_points = []

    response = HttpResponse(content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{student.student_id}-hc-xai-report.pdf"'

    doc = SimpleDocTemplate(
        response, pagesize=A4,
        rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title="HC-XAI Student Performance Report", author="HC-XAI",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("HCXAITitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, alignment=TA_CENTER, spaceAfter=6)
    subtitle_style = ParagraphStyle("HCXAISubtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=9, leading=12, alignment=TA_CENTER, spaceAfter=15)
    heading_style = ParagraphStyle("HCXAIHeading", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=12, spaceAfter=8)
    normal_style = ParagraphStyle("HCXAINormal", parent=styles["Normal"], fontName="Helvetica", fontSize=9.5, leading=13)
    small_style = ParagraphStyle("HCXAISmall", parent=styles["Normal"], fontName="Helvetica", fontSize=8.5, leading=12)
    bold_small = ParagraphStyle("HCXAIBoldSmall", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=9, leading=12)

    story = []

    # Title
    story.append(Paragraph("HC-XAI Student Performance Report", title_style))
    story.append(Paragraph("Human-Centered Explainable AI", subtitle_style))

    # Student Information
    story.append(Paragraph("Student Information", heading_style))
    student_data = [
        [Paragraph("<b>Student ID</b>", normal_style), Paragraph(str(student.student_id), normal_style)],
        [Paragraph("<b>Student Name</b>", normal_style), Paragraph(str(student.full_name), normal_style)],
        [Paragraph("<b>Class / Year</b>", normal_style), Paragraph(str(student.class_year or "—"), normal_style)],
        [Paragraph("<b>Education Level</b>", normal_style), Paragraph(str(student.education_level or "—"), normal_style)],
        [Paragraph("<b>Age</b>", normal_style), Paragraph(str(student.age or "—"), normal_style)],
        [Paragraph("<b>Gender</b>", normal_style), Paragraph(str(student.gender or "—"), normal_style)],
    ]
    _table_style = TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ])
    t = Table(student_data, colWidths=[55 * mm, 110 * mm])
    t.setStyle(_table_style)
    story.append(t)

    # Prediction Summary
    story.append(Paragraph("Prediction Summary", heading_style))
    confidence = f"{prediction.confidence}%" if prediction.confidence is not None else "—"
    pred_data = [
        [Paragraph("<b>Predicted Performance</b>", normal_style), Paragraph(f"{prediction.predicted_percentage}%", normal_style)],
        [Paragraph("<b>Performance Category</b>", normal_style), Paragraph(str(prediction.predicted_category), normal_style)],
        [Paragraph("<b>Risk Level</b>", normal_style), Paragraph(str(prediction.risk_level), normal_style)],
        [Paragraph("<b>Confidence</b>", normal_style), Paragraph(str(confidence), normal_style)],
    ]
    t2 = Table(pred_data, colWidths=[55 * mm, 110 * mm])
    t2.setStyle(_table_style)
    story.append(t2)

    # Academic Information
    story.append(Paragraph("Academic Information", heading_style))
    academic_data = [
        [Paragraph("<b>Previous Marks</b>", normal_style), Paragraph(f"{record.previous_gained_marks or '—'} / {record.previous_total_marks or '—'}", normal_style)],
        [Paragraph("<b>Internal Marks</b>", normal_style), Paragraph(f"{record.internal_gained_marks or '—'} / {record.internal_total_marks or '—'}", normal_style)],
        [Paragraph("<b>Assignment Completion</b>", normal_style), Paragraph(f"{record.completed_assignments} / {record.total_assignments}", normal_style)],
        [Paragraph("<b>Attendance</b>", normal_style), Paragraph(f"{record.attendance or '—'}%", normal_style)],
        [Paragraph("<b>Study Hours</b>", normal_style), Paragraph(f"{record.study_hours or '—'} hours/week", normal_style)],
        [Paragraph("<b>Sleep Hours</b>", normal_style), Paragraph(f"{record.sleep_hours or '—'} hours", normal_style)],
        [Paragraph("<b>Participation</b>", normal_style), Paragraph(str(record.participation or "—"), normal_style)],
        [Paragraph("<b>Backlogs</b>", normal_style), Paragraph(str(record.backlogs), normal_style)],
        [Paragraph("<b>Past Failures</b>", normal_style), Paragraph(str(record.past_failures), normal_style)],
        [Paragraph("<b>Tutoring Sessions</b>", normal_style), Paragraph(str(record.tutoring_sessions), normal_style)],
        [Paragraph("<b>Learning Mode</b>", normal_style), Paragraph(str(record.learning_mode or "—"), normal_style)],
        [Paragraph("<b>Internet Access</b>", normal_style), Paragraph(str(record.internet_access or "—"), normal_style)],
        [Paragraph("<b>Motivation Level</b>", normal_style), Paragraph(str(record.motivation_level or "—"), normal_style)],
        [Paragraph("<b>Parental Involvement</b>", normal_style), Paragraph(str(record.parental_involvement or "—"), normal_style)],
        [Paragraph("<b>Extracurricular</b>", normal_style), Paragraph(str(record.extracurricular or "—"), normal_style)],
    ]
    t3 = Table(academic_data, colWidths=[70 * mm, 95 * mm])
    t3.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(t3)

    # ── FIX 2: SHAP sections in PDF ──────────────────────────────────────
    story.append(Paragraph("Explainable AI – Factor Impact (SHAP)", heading_style))
    story.append(Paragraph(
        "SHAP values explain how individual factors contributed to the model prediction. "
        "Positive values indicate positive contribution, while negative values indicate negative contribution.",
        small_style,
    ))

    if not shap_sections["shap_available"]:
        story.append(Paragraph("Explanation unavailable — SHAP data was not generated for this prediction.", normal_style))
    else:
        # Raw SHAP table (existing design — preserved)
        factors = explanation.local_features if explanation else []
        if factors:
            shap_table_data = [
                [Paragraph("<b>Factor</b>", normal_style), Paragraph("<b>SHAP Impact</b>", normal_style), Paragraph("<b>Contribution</b>", normal_style)]
            ]
            shap_styles_list = [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
            for index, factor in enumerate(factors, start=1):
                shap_val = float(factor.get("shap_value", 0))
                impact_text = f"+{shap_val:.6f}" if shap_val > 0 else f"{shap_val:.6f}"
                contribution = "Positive" if shap_val > 0 else ("Negative" if shap_val < 0 else "Neutral")
                shap_table_data.append([
                    Paragraph(str(factor.get("name", "")), normal_style),
                    Paragraph(impact_text, normal_style),
                    Paragraph(contribution, normal_style),
                ])
                color = colors.green if shap_val > 0 else (colors.red if shap_val < 0 else colors.grey)
                shap_styles_list.append(("TEXTCOLOR", (2, index), (2, index), color))
            shap_t = Table(shap_table_data, colWidths=[70 * mm, 45 * mm, 50 * mm], repeatRows=1)
            shap_t.setStyle(TableStyle(shap_styles_list))
            story.append(shap_t)

        story.append(Spacer(1, 10))

        # ── FIX 2: Positive Factors section ──────────────────────────────
        story.append(Paragraph("Positive Factors", heading_style))
        if shap_sections["positive_factors"]:
            for fac in shap_sections["positive_factors"]:
                story.append(Paragraph(f"<b>{fac['label']}: {fac['value_str']}</b>", bold_small))
                story.append(Paragraph(f"Positive Impact  |  SHAP: +{fac['shap_value']:.4f}", small_style))
                story.append(Paragraph(fac["explanation"], small_style))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No positive factors identified.", normal_style))

        # ── FIX 2: Areas to Improve section ──────────────────────────────
        story.append(Paragraph("Areas to Improve", heading_style))
        if shap_sections["negative_factors"]:
            for fac in shap_sections["negative_factors"]:
                story.append(Paragraph(f"<b>{fac['label']}: {fac['value_str']}</b>", bold_small))
                story.append(Paragraph(f"Negative Impact  |  SHAP: {fac['shap_value']:.4f}", small_style))
                story.append(Paragraph(fac["explanation"], small_style))
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No specific improvement areas identified.", normal_style))

    # HC-XAI note (existing — unchanged)
    story.append(Spacer(1, 12))
    story.append(Paragraph("Human-Centered AI Decision Support", heading_style))
    story.append(Paragraph(
        "This prediction is intended to support teacher decision-making. "
        "The AI output should be considered alongside the student's academic context. "
        "The teacher remains the final decision-maker.",
        small_style,
    ))

    doc.build(story)
    return response
