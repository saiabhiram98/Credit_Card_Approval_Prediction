# credit_risk_api/main.py
"""FastAPI scoring service.

A thin HTTP layer over scoring.py — all model loading and decision logic lives
there, shared with the Streamlit UI so the two cannot disagree.
"""

from fastapi import FastAPI

import scoring
from schemas import ApplicantInput, PredictionOutput, SHAPDriver

app = FastAPI(
    title="Credit Risk Decisioning API",
    description="Scores credit card applicants and returns a cost-optimized approve/reject decision with SHAP explanations.",
    version="1.0.0",
)


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "model": scoring.MODEL_NAME,
        "threshold": scoring.THRESHOLD,
    }


@app.get("/threshold")
def get_threshold():
    return {
        "threshold": scoring.THRESHOLD,
        "fn_cost": scoring.FN_COST,
        "fp_cost": scoring.FP_COST,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(applicant: ApplicantInput):
    result = scoring.score_applicant(applicant.model_dump())

    return PredictionOutput(
        risk_score=result["risk_score"],
        decision=result["decision"],
        threshold=result["threshold"],
        top_shap_drivers=[SHAPDriver(**driver) for driver in result["top_shap_drivers"]],
    )


if __name__ == "__main__":
    # Allows `python main.py` in addition to `uvicorn main:app --reload`.
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
