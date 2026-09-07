"""Streamlit UI for the credit risk decisioning model.

Scores in-process via the shared scoring module (the same code the FastAPI
service uses), so this app can be deployed standalone to Streamlit Community
Cloud with no separate backend to host.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

# Works whether Streamlit is launched from this folder or the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring  # noqa: E402

st.set_page_config(page_title="Credit Risk Decisioning", layout="wide")

st.title("Credit Risk Decisioning Service")
# Dollar signs are escaped: Streamlit renders markdown, where a bare "$" opens
# a LaTeX math span and silently swallows the symbol.
st.caption(
    f"{scoring.MODEL_NAME} · Decision threshold {scoring.THRESHOLD:.2f} · "
    f"Assumed costs: \\${scoring.FN_COST} per missed default, "
    f"\\${scoring.FP_COST} per rejected good applicant"
)

with st.expander("How this works"):
    st.markdown(
        f"""
Only **1.67%** of applicants in the training data became delinquent, so accuracy
is a poor guide here — always predicting "no default" scores ~98%. This model is
selected on **PR-AUC** instead (0.25 vs. a 0.017 random baseline).

The approve/reject cutoff is **not** the usual 0.5. Because a missed default is
assumed to cost 10x more than a wrongly rejected applicant, every threshold was
swept to find the one minimizing total expected cost — **{scoring.THRESHOLD:.2f}** —
which cuts expected cost ~30% versus approving everyone.

*Caveat: PR-AUC ranged 0.14–0.25 across random seeds during development; this is
the best of those. Cost figures rest on the illustrative 10:1 assumption, not
real loss data.*
"""
    )

with st.form("applicant_form"):
    st.subheader("Applicant Information")
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["M", "F"])
        own_car = st.selectbox("Owns a Car", ["Y", "N"])
        own_realty = st.selectbox("Owns Realty", ["Y", "N"])
        children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
        fam_members = st.number_input("Family Members", min_value=1, max_value=10, value=2)

    with col2:
        income = st.number_input(
            "Annual Income ($)", min_value=10000, max_value=1000000, value=150000, step=5000
        )
        income_type = st.selectbox(
            "Income Type",
            ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
        )
        education = st.selectbox(
            "Education Level",
            [
                "Lower secondary",
                "Secondary / secondary special",
                "Incomplete higher",
                "Higher education",
                "Academic degree",
            ],
            index=3,
        )
        family_status = st.selectbox(
            "Family Status",
            ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        )
        housing = st.selectbox(
            "Housing Type",
            [
                "House / apartment",
                "Rented apartment",
                "With parents",
                "Municipal apartment",
                "Office apartment",
            ],
        )

    with col3:
        occupation = st.selectbox(
            "Occupation Type",
            [
                "Unknown", "Laborers", "Core staff", "Sales staff", "Managers",
                "Drivers", "High skill tech staff", "Accountants", "Medicine staff",
                "Cooking staff", "Security staff", "Cleaning staff",
                "Private service staff", "Low-skill Laborers", "Waiters/barmen staff",
                "Secretaries", "HR staff", "Realty agents", "IT staff",
            ],
        )
        age_years = st.number_input("Age (years)", min_value=18, max_value=70, value=35)
        is_pensioner = st.checkbox("Pensioner (not currently employed)")
        employed_years = st.number_input(
            "Years Employed", min_value=0, max_value=50, value=5, disabled=is_pensioner
        )
        credit_months = st.number_input(
            "Months of Credit History", min_value=0, max_value=60, value=24
        )

    submitted = st.form_submit_button("Score Applicant", use_container_width=True)

if submitted:
    applicant = {
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": float(income),
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": -(age_years * 365),
        # 365243 is the sentinel the source data uses for "not employed".
        "DAYS_EMPLOYED": 365243 if is_pensioner else -(employed_years * 365),
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": float(fam_members),
        "lowest_balance_months": credit_months,
    }

    with st.spinner("Scoring applicant..."):
        result = scoring.score_applicant(applicant)

    st.divider()
    col_score, col_decision = st.columns(2)

    with col_score:
        st.metric(
            "Risk Score",
            f"{result['risk_score']:.4f}",
            help="Estimated probability of delinquency (0 = safe, 1 = high risk)",
        )
        st.progress(min(result["risk_score"], 1.0))

    with col_decision:
        if result["decision"] == "Approve":
            st.success(f"Decision: APPROVE — score below the {result['threshold']:.2f} threshold")
        else:
            st.error(f"Decision: REJECT — score at or above the {result['threshold']:.2f} threshold")

    st.subheader("Why: Top Risk Drivers (SHAP)")
    drivers = result["top_shap_drivers"]
    features = [d["feature"] for d in drivers]
    impacts = [d["impact"] for d in drivers]

    fig, ax = plt.subplots(figsize=(8, 3))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in impacts]
    ax.barh(features[::-1], impacts[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP impact  (right / red = raises risk, left / blue = lowers risk)")
    ax.set_title("Feature contributions for this applicant")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
