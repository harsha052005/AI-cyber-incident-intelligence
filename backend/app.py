from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from backend.services.prediction import predict_risk
import pandas as pd
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "cyber_attack_model.joblib"


app = FastAPI(
    title="AI Cyber Incident Intelligence & Attack Risk Prediction Platform",
    description="A platform for analyzing cyber incidents and predicting attack risk.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {
        "message": "AI Cyber Incident Intelligence Platform is running!",
        "status": "success"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    result = predict_risk(request.features)

    return {
        "status": "success",
        "prediction": result["prediction"],
        "probability": result.get("probability")
    }


@app.get("/sample")
def get_sample():
    model = joblib.load(MODEL_PATH)

    df = pd.read_parquet(
        BASE_DIR / "data" / "raw" / "CIC-IDS2017" / "Benign-Monday-no-metadata.parquet"
    )

    features = df[model.feature_names_in_].iloc[0].astype(float).tolist()

    return {
        "features": features
    }
@app.get("/attack-sample")
def get_attack_sample():
    import joblib
    import pandas as pd

    model = joblib.load(MODEL_PATH)

    df = pd.read_parquet(
        BASE_DIR / "data" / "raw" / "CIC-IDS2017" / "DDoS-Friday-no-metadata.parquet"
    )

    row = df[df["Label"] == "DDoS"].iloc[0]

    features = row[model.feature_names_in_].astype(float).tolist()

    return {
        "actual_label": "DDoS",
        "features": features
    }
@app.get("/dos-sample")
def get_dos_sample():
    import joblib
    import pandas as pd

    model = joblib.load(MODEL_PATH)

    df = pd.read_parquet(
        BASE_DIR / "data" / "raw" / "CIC-IDS2017" / "DoS-Wednesday-no-metadata.parquet"
    )

    row = df[df["Label"] == "DoS Hulk"].iloc[0]

    features = row[model.feature_names_in_].astype(float).tolist()

    return {
        "actual_label": "DoS Hulk",
        "features": features
    }
@app.get("/portscan-sample")
def get_portscan_sample():
    import joblib
    import pandas as pd

    model = joblib.load(MODEL_PATH)

    df = pd.read_parquet(
        BASE_DIR / "data" / "raw" / "CIC-IDS2017" / "Portscan-Friday-no-metadata.parquet"
    )

    row = df[df["Label"] == "PortScan"].iloc[0]

    features = row[model.feature_names_in_].astype(float).tolist()

    return {
        "actual_label": "PortScan",
        "features": features
    }
