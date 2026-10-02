from pathlib import Path
from django.conf import settings

FEATURES = [
    "previous_percent",
    "internal_percent",
    "assignment_percent",
    "attendance",
    "study_hours",
    "participation",
    "backlogs",
    "past_failures",
    "sleep_hours",
    "tutoring_sessions",
]

MODEL_PATH = Path(settings.BASE_DIR) / "predictor" / "artifacts" / "student_performance_model.joblib"


class ModelNotReadyError(RuntimeError):
    pass


def load_model():
    if not MODEL_PATH.exists():
        raise ModelNotReadyError(
            "The ML model has not been trained yet. Train it with the train_model management command."
        )
    import joblib
    return joblib.load(MODEL_PATH)


def record_to_frame(record):
    values = {
        "previous_percent": record.previous_percent,
        "internal_percent": record.internal_percent,
        "assignment_percent": record.assignment_percent,
        "attendance": record.attendance,
        "study_hours": record.study_hours,
        "participation": record.participation,
        "backlogs": record.backlogs,
        "past_failures": record.past_failures,
        "sleep_hours": record.sleep_hours,
        "tutoring_sessions": record.tutoring_sessions,
    }
    missing = [name for name in FEATURES if values.get(name) is None]
    if missing:
        raise ValueError("Missing required prediction fields: " + ", ".join(missing))
    import pandas as pd
    return pd.DataFrame([values], columns=FEATURES)


def predict_record(record):
    model = load_model()
    frame = record_to_frame(record)
    predicted = model.predict(frame)[0]
    probabilities = {}
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(frame)[0]
        probabilities = {str(label): round(float(value) * 100, 2) for label, value in zip(model.classes_, proba)}
        confidence = round(max(probabilities.values()), 2) if probabilities else None

    category = str(predicted)
    risk = "High" if category == "At Risk" else "Medium" if category == "Needs Improvement" else "Low"
    predicted_percentage = probabilities.get(category) if probabilities else None
    return {
        "category": category,
        "confidence": confidence,
        "risk_level": risk,
        "probabilities": probabilities,
        "predicted_percentage": predicted_percentage,
        "model_name": type(model).__name__,
        "model": model,
        "frame": frame,
    }
