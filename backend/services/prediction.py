import os
import joblib
import numpy as np


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "models",
    "cyber_attack_model.joblib"
)


def load_model():
    """Load the trained machine learning model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_risk(features):
    """
    Predict cyber incident risk from input features.

    features should be a list or array of numerical values.
    """

    model = load_model()

    input_data = np.array(features).reshape(1, -1)

    prediction = model.predict(input_data)[0]

    result = {
        "prediction": str(prediction)
    }

    # If the model supports probability prediction
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0]
        result["probability"] = float(np.max(probability))

    return result