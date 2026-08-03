# credit_risk_api/main.py
import joblib
import shap
import numpy as np
from fastapi import FastAPI
from schemas import ApplicantInput, PredictionOutput, SHAPDriver
from preprocess import preprocess

app = FastAPI(title="Credit Risk Decisioning API")

# Load artifacts
model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")
explainer = joblib.load("models/shap_explainer.pkl")
feature_columns = joblib.load("models/feature_columns.pkl")

THRESHOLD = 0.48

@app.get("/")
def health_check():
    return {"status": "ok", "model": "XGBoost (tuned)", "threshold": THRESHOLD}

@app.get("/threshold")
def get_threshold():
    return {"threshold": THRESHOLD, "fn_cost": 500, "fp_cost": 50}

@app.post("/predict", response_model=PredictionOutput)
def predict(applicant: ApplicantInput):
    # Preprocess input
    X = preprocess(applicant.model_dump(), scaler, feature_columns)

    # Risk score
    risk_score = float(model.predict_proba(X)[:, 1][0])
    decision = "Reject" if risk_score >= THRESHOLD else "Approve"

    # SHAP explanation
    shap_values = explainer.shap_values(X)
    shap_series = dict(zip(feature_columns, shap_values[0]))
    top_drivers = sorted(shap_series.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    return PredictionOutput(
        risk_score=round(risk_score, 4),
        decision=decision,
        threshold=THRESHOLD,
        top_shap_drivers=[
            SHAPDriver(feature=f, impact=round(float(v), 4))
            for f, v in top_drivers
        ]
    )