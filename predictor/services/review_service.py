def build_grounded_review(record, prediction, explanation=None):
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
