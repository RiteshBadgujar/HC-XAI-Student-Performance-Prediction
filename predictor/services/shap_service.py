from .ml_service import FEATURES


def explain_prediction(model, frame):
    """
    Generate a teacher-friendly local SHAP explanation for one prediction.

    Returns:
        {
            "local_features": [
                {
                    "name": feature_name,
                    "shap_value": numeric_shap_value,
                    "impact": "positive" or "negative"
                }
            ],
            "values": [...],
            "global_importance": [...],
            "summary": "..."
        }
    """

    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "SHAP is not installed. Install the project requirements "
            "before using explanations."
        ) from exc

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "NumPy is not installed. Install the project requirements "
            "before using explanations."
        ) from exc

    # ---------------------------------------------------------
    # Create SHAP Tree Explainer
    # ---------------------------------------------------------
    try:
        explainer = shap.TreeExplainer(model)
        raw = explainer.shap_values(frame)
    except Exception as exc:
        raise RuntimeError(
            f"SHAP explanation could not be generated: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Find the class predicted by the model
    # ---------------------------------------------------------
    predicted_index = None

    try:
        if hasattr(model, "predict") and hasattr(model, "classes_"):
            predicted_label = model.predict(frame)[0]
            classes = list(model.classes_)

            if predicted_label in classes:
                predicted_index = classes.index(predicted_label)
    except Exception:
        predicted_index = None

    # ---------------------------------------------------------
    # Extract SHAP values safely
    # ---------------------------------------------------------
    try:
        if isinstance(raw, list):
            # Older SHAP versions may return:
            # [class_1_values, class_2_values, ...]
            if predicted_index is None:
                predicted_index = 0

            class_values = raw[predicted_index]

            array = np.asarray(class_values)

            if array.ndim == 2:
                values = array[0]
            else:
                values = array.reshape(-1)

        else:
            array = np.asarray(raw)

            if array.ndim == 3:
                # Possible format:
                # samples × features × classes
                if predicted_index is None:
                    predicted_index = 0

                values = array[0, :, predicted_index]

            elif array.ndim == 2:
                # samples × features
                values = array[0]

            elif array.ndim == 1:
                values = array

            else:
                values = array.reshape(-1)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to process SHAP values: {exc}"
        ) from exc

    # ---------------------------------------------------------
    # Ensure feature count matches
    # ---------------------------------------------------------
    values = np.asarray(values).reshape(-1)

    if len(values) != len(FEATURES):
        raise RuntimeError(
            f"SHAP returned {len(values)} values, but the model expects "
            f"{len(FEATURES)} features."
        )

    # ---------------------------------------------------------
    # Build local feature explanation
    # ---------------------------------------------------------
    local = []

    for feature, value in zip(FEATURES, values):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = 0.0

        local.append(
            {
                "name": feature,
                "shap_value": round(numeric, 6),
                "impact": (
                    "positive"
                    if numeric >= 0
                    else "negative"
                ),
            }
        )

    # ---------------------------------------------------------
    # Sort by absolute SHAP impact
    # ---------------------------------------------------------
    local.sort(
        key=lambda item: abs(item["shap_value"]),
        reverse=True,
    )

    # ---------------------------------------------------------
    # Human-readable summary
    # ---------------------------------------------------------
    summary = (
        "The explanation ranks the factors that contributed most "
        "strongly to this individual prediction."
    )

    # ---------------------------------------------------------
    # Return data used by review_service.py and templates
    # ---------------------------------------------------------
    return {
        "local_features": local,
        "values": local,
        "global_importance": local,
        "summary": summary,
    }