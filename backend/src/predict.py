from pathlib import Path

import joblib
import numpy as np
import pandas as pd


# Find the project folder automatically
BASE_DIR = Path(__file__).resolve().parents[2]

# Correct model location
MODEL_PATH = BASE_DIR / "models" / "cyber_attack_model.joblib"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_risk(features):
    model = load_model()

    # The model needs exactly 77 features
    if len(features) != model.n_features_in_:
        raise ValueError(
            f"Expected {model.n_features_in_} features, but received {len(features)}."
        )

    # Give the model the correct feature names
    X = pd.DataFrame(
        [features],
        columns=model.feature_names_in_
    )

    prediction = model.predict(X)[0]

    result = {
        "prediction": str(prediction)
    }

    # Get prediction probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        result["probability"] = float(np.max(probabilities))

    return result