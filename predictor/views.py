import json
from django.contrib import messages
from django.shortcuts import render

# Dummy per-student explanation data, keyed by student_id.
# This will be replaced with real SHAP output once the ML pipeline is built.
REPORT_DATA = {
    "ST001": {
        "student_name": "Rahul Verma",
        "class_year": "TE-B",
        "predicted_percentage": 82,
        "performance_category": "Good",
        "factors": [
            {"name": "Attendance", "impact": 8.2, "bar_width": 70},
            {"name": "Previous marks", "impact": 6.5, "bar_width": 55},
            {"name": "Study hours", "impact": 3.1, "bar_width": 30},
            {"name": "Backlogs", "impact": -4.8, "bar_width": 45},
        ],
        "positive_points": ["Good attendance", "Strong previous marks", "Consistent study hours"],
        "improve_points": ["Reduce backlogs", "Improve assignment completion"],
    },
    "ST002": {
        "student_name": "Priya Nair",
        "class_year": "TE-A",
        "predicted_percentage": 91,
        "performance_category": "Excellent",
        "factors": [
            {"name": "Attendance", "impact": 9.5, "bar_width": 85},
            {"name": "Previous marks", "impact": 8.9, "bar_width": 80},
            {"name": "Participation", "impact": 4.2, "bar_width": 40},
            {"name": "Backlogs", "impact": 0, "bar_width": 5},
        ],
        "positive_points": ["Excellent attendance", "High previous marks", "Strong participation"],
        "improve_points": ["No major concerns identified"],
    },
    "ST003": {
        "student_name": "Amit Joshi",
        "class_year": "TE-B",
        "predicted_percentage": 56,
        "performance_category": "Average",
        "factors": [
            {"name": "Attendance", "impact": 1.2, "bar_width": 20},
            {"name": "Previous marks", "impact": 2.0, "bar_width": 25},
            {"name": "Backlogs", "impact": -7.5, "bar_width": 65},
            {"name": "Assignment completion", "impact": -3.4, "bar_width": 40},
        ],
        "positive_points": ["Slight positive from attendance"],
        "improve_points": ["Reduce backlogs", "Improve assignment completion", "Increase study hours"],
    },
    "ST004": {
        "student_name": "Sneha Patil",
        "class_year": "TE-C",
        "predicted_percentage": 39,
        "performance_category": "Needs Attention",
        "factors": [
            {"name": "Attendance", "impact": -6.0, "bar_width": 55},
            {"name": "Backlogs", "impact": -9.1, "bar_width": 80},
            {"name": "Assignment completion", "impact": -5.5, "bar_width": 50},
            {"name": "Study hours", "impact": -2.0, "bar_width": 20},
        ],
        "positive_points": ["No significant positive factors identified"],
        "improve_points": ["Improve attendance urgently", "Clear pending backlogs", "Increase assignment completion"],
    },
}


def landing(request):
    return render(request, "predictor/landing.html", {})


def dashboard(request):
    dummy_predictions = [
        {"student_id": "ST001", "student_name": "Rahul Verma", "predicted_percentage": 82, "performance_category": "Good", "created_at": None},
        {"student_id": "ST002", "student_name": "Priya Nair", "predicted_percentage": 91, "performance_category": "Excellent", "created_at": None},
        {"student_id": "ST003", "student_name": "Amit Joshi", "predicted_percentage": 56, "performance_category": "Average", "created_at": None},
        {"student_id": "ST004", "student_name": "Sneha Patil", "predicted_percentage": 39, "performance_category": "Needs Attention", "created_at": None},
    ]
    context = {
        "total_predictions": len(dummy_predictions),
        "avg_score": round(sum(p["predicted_percentage"] for p in dummy_predictions) / len(dummy_predictions), 1),
        "needs_attention_count": sum(1 for p in dummy_predictions if p["performance_category"] == "Needs Attention"),
        "recent_predictions": dummy_predictions,
    }
    return render(request, "predictor/dashboard.html", context)


def signup(request):
    return render(request, "predictor/signup.html", {})


def predict_performance(request):
    if request.method == "POST":
        messages.success(request, "Prediction generated successfully.")
        return render(request, "predictor/predict_result.html", {})
    return render(request, "predictor/predict_form.html", {})


def report_detail(request, student_id="ST001"):
    data = REPORT_DATA.get(student_id, REPORT_DATA["ST001"])
    context = {"student_id": student_id, **data}
    return render(request, "predictor/report_detail.html", context)


def student_list(request):
    default_predictions = [
        {"student_id": "ST001", "student_name": "Rahul Verma", "class_year": "TE-B", "predicted_percentage": 82, "performance_category": "Good", "created_at": None},
        {"student_id": "ST002", "student_name": "Priya Nair", "class_year": "TE-A", "predicted_percentage": 91, "performance_category": "Excellent", "created_at": None},
        {"student_id": "ST003", "student_name": "Amit Joshi", "class_year": "TE-B", "predicted_percentage": 56, "performance_category": "Average", "created_at": None},
        {"student_id": "ST004", "student_name": "Sneha Patil", "class_year": "TE-C", "predicted_percentage": 39, "performance_category": "Needs Attention", "created_at": None},
    ]
    return render(request, "predictor/student_list.html", {"default_predictions": default_predictions})


def compare_students(request):
    default_factors_a = [
        {"name": "Attendance", "value": "78%"},
        {"name": "Previous marks", "value": "80%"},
        {"name": "Study hours/week", "value": "12"},
        {"name": "Backlogs", "value": "1"},
    ]
    default_factors_b = [
        {"name": "Attendance", "value": "95%"},
        {"name": "Previous marks", "value": "93%"},
        {"name": "Study hours/week", "value": "18"},
        {"name": "Backlogs", "value": "0"},
    ]
    return render(request, "predictor/compare_students.html", {
        "default_factors_a": default_factors_a,
        "default_factors_b": default_factors_b,
    })


def class_analytics(request):
    default_categories = [
        {"name": "Excellent", "count": 34, "percent": 27, "color_class": "bg-success"},
        {"name": "Good", "count": 48, "percent": 38, "color_class": "bg-primary"},
        {"name": "Average", "count": 27, "percent": 21, "color_class": "bg-warning"},
        {"name": "Needs Attention", "count": 19, "percent": 15, "color_class": "bg-danger"},
    ]
    default_common_factors = [
        "Attendance is the strongest positive driver across most students",
        "Backlogs are the most common negative factor",
        "Assignment completion shows the widest spread between top and bottom performers",
    ]
    return render(request, "predictor/class_analytics.html", {
        "default_categories": default_categories,
        "default_common_factors": default_common_factors,
    })

def profile(request):
    return render(request, "predictor/profile.html", {})