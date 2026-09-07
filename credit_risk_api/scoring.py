"""Shared scoring core.

Owns artifact loading and the scoring/decision logic so the two interfaces —
the FastAPI service (main.py) and the Streamlit UI (streamlit_app.py) — cannot
drift apart. Either can be deployed independently and both produce identical
decisions.

Paths resolve relative to this file, not the working directory, so the module
works whether it is imported from this folder, the repository root (as on
Streamlit Community Cloud), or inside the container.
"""

import sys
from pathlib import Path

import joblib

_HERE = Path(__file__).resolve().parent

# Make sibling modules importable no matter how this package is entered
# (script, `streamlit run` from the repo root, or `uvicorn` from this folder).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from preprocess import preprocess  # noqa: E402  (needs the sys.path entry above)

MODEL_DIR = _HERE / "models"

# Cost assumptions behind the decision threshold. A false negative (approving an
# applicant who defaults) is assumed 10x more expensive than a false positive
# (rejecting an applicant who would have repaid).
FN_COST = 500
FP_COST = 50

model = joblib.load(MODEL_DIR / "xgb_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
explainer = joblib.load(MODEL_DIR / "shap_explainer.pkl")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

# Cost-optimal cutoff chosen in the notebook by sweeping thresholds, loaded from
# the training artifact rather than hardcoded so it cannot fall out of sync with
# the deployed model.
THRESHOLD = round(float(joblib.load(MODEL_DIR / "decision_threshold.pkl")), 2)

MODEL_NAME = "XGBoost (tuned)"


def score_applicant(applicant: dict, top_n: int = 5) -> dict:
    """Score one applicant.

    Returns the risk score, the approve/reject decision at the cost-optimal
    threshold, and the features that moved this particular prediction most.
    """
    X = preprocess(applicant, scaler, feature_columns)

    risk_score = float(model.predict_proba(X)[:, 1][0])
    decision = "Reject" if risk_score >= THRESHOLD else "Approve"

    # XGBoost binary classifiers return a single 2-D array (rows x features),
    # unlike RandomForest which returns one array per class.
    shap_values = explainer.shap_values(X)
    contributions = zip(feature_columns, shap_values[0])
    top_drivers = sorted(contributions, key=lambda kv: abs(kv[1]), reverse=True)[:top_n]

    return {
        "risk_score": round(risk_score, 4),
        "decision": decision,
        "threshold": THRESHOLD,
        "top_shap_drivers": [
            {"feature": name, "impact": round(float(value), 4)}
            for name, value in top_drivers
        ],
    }
