import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Credit Risk Decisioning", layout="wide")
st.title("Credit Risk Decisioning Service")
st.caption("Powered by XGBoost · Threshold: 0.48 · FN cost $500 · FP cost $50")

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
        income = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=150000, step=5000)
        income_type = st.selectbox("Income Type", ["Working", "Commercial associate", "Pensioner", "State servant", "Student"])
        education = st.selectbox("Education Level", ["Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"])
        family_status = st.selectbox("Family Status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"])
        housing = st.selectbox("Housing Type", ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment"])

    with col3:
        occupation = st.selectbox("Occupation Type", [
            "Unknown", "Laborers", "Core staff", "Sales staff", "Managers",
            "Drivers", "High skill tech staff", "Accountants", "Medicine staff",
            "Cooking staff", "Security staff", "Cleaning staff",
            "Private service staff", "Low-skill Laborers", "Waiters/barmen staff",
            "Secretaries", "HR staff", "Realty agents", "IT staff"
        ])
        age_years = st.number_input("Age (years)", min_value=18, max_value=70, value=35)
        is_pensioner = st.checkbox("Pensioner (not currently employed)")
        employed_years = st.number_input("Years Employed", min_value=0, max_value=50, value=5, disabled=is_pensioner)
        credit_months = st.number_input("Months of Credit History", min_value=0, max_value=60, value=24)

    submitted = st.form_submit_button("Score Applicant", use_container_width=True)

if submitted:
    days_birth = -(age_years * 365)
    days_employed = 365243 if is_pensioner else -(employed_years * 365)

    payload = {
        "CODE_GENDER": gender,
        "FLAG_OWN_CAR": own_car,
        "FLAG_OWN_REALTY": own_realty,
        "CNT_CHILDREN": children,
        "AMT_INCOME_TOTAL": float(income),
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing,
        "DAYS_BIRTH": days_birth,
        "DAYS_EMPLOYED": days_employed,
        "OCCUPATION_TYPE": occupation,
        "CNT_FAM_MEMBERS": float(fam_members),
        "lowest_balance_months": credit_months
    }

    with st.spinner("Scoring applicant..."):
        response = requests.post(f"{API_URL}/predict", json=payload)

    if response.status_code == 200:
        result = response.json()
        score = result["risk_score"]
        decision = result["decision"]
        drivers = result["top_shap_drivers"]

        st.divider()
        col_score, col_decision = st.columns(2)

        with col_score:
            st.metric("Risk Score", f"{score:.4f}", help="Probability of default (0 = safe, 1 = high risk)")
            st.progress(score)

        with col_decision:
            if decision == "Approve":
                st.success(f"Decision: APPROVE (score below threshold {result['threshold']})")
            else:
                st.error(f"Decision: REJECT (score above threshold {result['threshold']})")

        st.subheader("Top Risk Drivers (SHAP)")
        features = [d["feature"] for d in drivers]
        impacts = [d["impact"] for d in drivers]

        fig, ax = plt.subplots(figsize=(8, 3))
        colors = ["red" if v > 0 else "steelblue" for v in impacts]
        ax.barh(features[::-1], impacts[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Impact (positive = increases risk)")
        ax.set_title("Feature Contributions for This Applicant")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.error(f"API error: {response.status_code} — {response.text}")