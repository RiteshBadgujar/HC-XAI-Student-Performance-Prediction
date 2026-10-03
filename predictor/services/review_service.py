"""
review_service.py — Grounded human-readable SHAP review builder.

FIX 2: Each SHAP factor now shows:
  - Feature name
  - Actual student value
  - Positive / Negative impact label
  - Actual SHAP value
  - Simple teacher-facing explanation using "contributing to the prediction" wording.

Real SHAP values and real student data are always used.
If SHAP is unavailable the caller receives an empty factors list and a clear message.
No values are invented.
"""

# Human-readable names for each ML feature
FEATURE_LABELS = {
    "previous_percent":   "Previous Marks",
    "internal_percent":   "Internal Marks",
    "assignment_percent": "Assignment Completion",
    "attendance":         "Attendance",
    "study_hours":        "Study Hours",
    "participation":      "Class Participation",
    "backlogs":           "Pending Backlogs",
    "past_failures":      "Past Failures",
    "sleep_hours":        "Sleep Hours",
    "tutoring_sessions":  "Tutoring Sessions",
}

# Units appended after the actual value for display
FEATURE_UNITS = {
    "previous_percent":   "%",
    "internal_percent":   "%",
    "assignment_percent": "%",
    "attendance":         "%",
    "study_hours":        " hrs/week",
    "participation":      "/10",
    "backlogs":           "",
    "past_failures":      "",
    "sleep_hours":        " hrs",
    "tutoring_sessions":  " sessions",
}

# Template explanations — kept factual, never causal
POSITIVE_EXPLANATIONS = {
    "previous_percent":   "Strong previous marks are positively contributing to the model's prediction.",
    "internal_percent":   "Good internal assessment scores are positively contributing to the prediction.",
    "assignment_percent": "High assignment completion is positively contributing to the model's prediction.",
    "attendance":         "Good attendance is positively contributing to the model's prediction.",
    "study_hours":        "Adequate study hours are positively contributing to the prediction.",
    "participation":      "Active class participation is positively contributing to the model's prediction.",
    "backlogs":           "Low or no pending backlogs are positively contributing to the prediction.",
    "past_failures":      "Few or no past failures are positively contributing to the model's prediction.",
    "sleep_hours":        "Healthy sleep hours are positively contributing to the prediction.",
    "tutoring_sessions":  "Tutoring sessions attended are positively contributing to the model's prediction.",
}

NEGATIVE_EXPLANATIONS = {
    "previous_percent": (
        "Previous academic performance is negatively contributing to the prediction. "
        "Review weaker subjects and focus on targeted exam preparation and revision."
    ),
    "internal_percent": (
        "Internal assessment performance is negatively contributing to the prediction. "
        "Focus on class tests, internal assessments, and regular preparation."
    ),
    "assignment_percent": (
        "Assignment completion is negatively contributing to the prediction. "
        "Complete pending assignments and follow a consistent submission schedule."
    ),
    "attendance": (
        "Attendance is negatively contributing to the prediction. "
        "Try to maintain regular attendance and avoid unnecessary absences."
    ),
    "study_hours": (
        "Study time is negatively contributing to the prediction. "
        "Increase focused study time gradually and follow a structured weekly study plan."
    ),
    "participation": (
        "Class participation is negatively contributing to the prediction. "
        "Participate more actively in discussions, activities, and classroom learning."
    ),
    "backlogs": (
        "Pending backlogs are negatively contributing to the prediction. "
        "Prioritize the pending subjects and create a focused plan to clear them."
    ),
    "past_failures": (
        "Past academic failures are negatively contributing to the prediction. "
        "Identify the subjects that need attention and provide targeted academic support."
    ),
    "sleep_hours": (
        "Sleep hours are negatively contributing to the prediction. "
        "Maintain a consistent sleep routine to support regular study and learning."
    ),
    "tutoring_sessions": (
        "Tutoring support is negatively contributing to the prediction. "
        "Consider additional academic guidance or tutoring for difficult subjects."
    ),
}
def _get_improvement_suggestion(feature_name):
    suggestions = {
        "previous_percent": "Focus on revision and weaker subjects.",
        "internal_percent": "Prepare regularly for internal assessments and class tests.",
        "assignment_percent": "Complete pending assignments and submit them on time.",
        "attendance": "Maintain regular attendance and avoid unnecessary absences.",
        "study_hours": "Increase focused study time gradually with a weekly study plan.",
        "participation": "Participate more actively in classroom activities and discussions.",
        "backlogs": "Prioritize pending subjects and create a plan to clear them.",
        "past_failures": "Identify weak subjects and provide targeted academic support.",
        "sleep_hours": "Maintain a consistent and adequate sleep routine.",
        "tutoring_sessions": "Consider additional tutoring or academic guidance when needed.",
    }

    return suggestions.get(
        feature_name,
        "Review this factor with the student and plan appropriate academic support."
    )


def _get_student_value(record, feature_name):
    """Return the actual student value for a feature from the AcademicRecord."""
    value_map = {
        "previous_percent":   record.previous_percent,
        "internal_percent":   record.internal_percent,
        "assignment_percent": record.assignment_percent,
        "attendance":         record.attendance,
        "study_hours":        record.study_hours,
        "participation":      record.participation,
        "backlogs":           record.backlogs,
        "past_failures":      record.past_failures,
        "sleep_hours":        record.sleep_hours,
        "tutoring_sessions":  record.tutoring_sessions,
    }
    return value_map.get(feature_name)


def build_shap_sections(record, explanation, top_n=5):
    """
    Build teacher-facing SHAP sections from real SHAP data.

    Returns:
        {
          "positive_factors": [...],   # factors with positive SHAP contribution
          "negative_factors": [...],   # factors with negative SHAP contribution
          "shap_available": True/False,
        }

    Each factor dict:
        {
          "label":       "Attendance",
          "value_str":   "86%",
          "impact":      "positive" | "negative",
          "shap_value":  0.143200,
          "explanation": "Good attendance is positively contributing ...",
        }
    """
    if explanation is None or not explanation.local_features:
        return {
            "positive_factors": [],
            "negative_factors": [],
            "shap_available": False,
        }

    # local_features is already sorted by |shap_value| desc from shap_service
    all_features = explanation.local_features  # list of dicts with name, shap_value, impact

    positive_factors = []
    negative_factors = []

    for f in all_features:
        name = f.get("name", "")
        shap_val = f.get("shap_value", 0)
        impact = "positive" if shap_val >= 0 else "negative"

        label = FEATURE_LABELS.get(name, name.replace("_", " ").title())
        unit = FEATURE_UNITS.get(name, "")

        raw_value = _get_student_value(record, name)
        if raw_value is None:
            value_str = "N/A"
        else:
            # Format: integers as int, floats with 1 decimal
            if isinstance(raw_value, float) and raw_value == int(raw_value):
                value_str = f"{int(raw_value)}{unit}"
            else:
                value_str = f"{raw_value:.1f}{unit}" if isinstance(raw_value, float) else f"{raw_value}{unit}"

        if impact == "positive":
            explanation_text = POSITIVE_EXPLANATIONS.get(
                name, f"{label} is positively contributing to the model's prediction."
            )
            positive_factors.append({
                "label": label,
                "value_str": value_str,
                "impact": "positive",
                "shap_value": shap_val,
                "explanation": explanation_text,
            })
        else:
            explanation_text = NEGATIVE_EXPLANATIONS.get(
                name, f"{label} is negatively contributing to the prediction."
            )
            negative_factors.append({
                "label": label,
                "value_str": value_str,
                "impact": "negative",
                "shap_value": shap_val,
                "explanation": explanation_text,
                "suggestion": _get_improvement_suggestion(name),
            })

    # Limit to top_n each, already sorted by magnitude
    return {
        "positive_factors": positive_factors[:top_n],
        "negative_factors": negative_factors[:top_n],
        "shap_available": True,
    }


def build_grounded_review(record, prediction, explanation=None):
    """Existing function — unchanged logic, kept for backward compatibility."""
    strengths = []
    improvements = []
    if record.attendance is not None:
        (strengths if record.attendance >= 75 else improvements).append(
            f"Attendance is {record.attendance:.1f}%"
        )
    if record.assignment_percent is not None:
        (strengths if record.assignment_percent >= 75 else improvements).append(
            f"Assignment completion is {record.assignment_percent:.1f}%"
        )
    if record.study_hours is not None:
        (strengths if record.study_hours >= 10 else improvements).append(
            f"Study time is {record.study_hours:.1f} hours/week"
        )
    if record.backlogs:
        improvements.append(f"There are {record.backlogs} pending backlog subject(s)")
    if not strengths:
        strengths.append("No strong positive indicator was identified from the available academic data.")
    if not improvements:
        improvements.append("No major concern was identified from the available academic data.")

    actions = {
        "teacher": "Review the highest-impact factors and discuss an appropriate academic support plan with the student.",
        "student": "Maintain the strongest habits and address the listed improvement areas consistently.",
    }
    return {
        "strengths": strengths,
        "improvements": improvements,
        "teacher_action": actions["teacher"],
        "student_action": actions["student"],
        "summary": f"The model classified the student as {prediction.predicted_category} with {prediction.confidence or 0:.1f}% confidence and {prediction.risk_level} risk.",
    }
