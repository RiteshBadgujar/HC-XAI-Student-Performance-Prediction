from .ml_service import FEATURES


def explain_prediction(model, frame):
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Install the project requirements before using explanations.") from exc

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(frame)
    predicted_index = None
    if hasattr(model, "predict") and hasattr(model, "classes_"):
        predicted_label = model.predict(frame)[0]
        classes = list(model.classes_)
        predicted_index = classes.index(predicted_label)

    if isinstance(raw, list):
        values = raw[predicted_index if predicted_index is not None else 0][0]
    else:
        import numpy as np
        array = np.asarray(raw)
        if array.ndim == 3:
            values = array[0, :, predicted_index if predicted_index is not None else 0]
        else:
            values = array[0]

    local = []
    for feature, value in zip(FEATURES, values):
        numeric = float(value)
        local.append({"name": feature, "shap_value": round(numeric, 6), "impact": "positive" if numeric >= 0 else "negative"})
    local.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

    summary = "The explanation ranks the factors that contributed most strongly to this individual prediction."
    return {"local_features": local, "values": local, "global_importance": local, "summary": summary}
