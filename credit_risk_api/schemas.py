# credit_risk_api/schemas.py
from pydantic import BaseModel

class ApplicantInput(BaseModel):
    CODE_GENDER: str                    # "M" or "F"
    FLAG_OWN_CAR: str                   # "Y" or "N"
    FLAG_OWN_REALTY: str                # "Y" or "N"
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    NAME_INCOME_TYPE: str               # "Working", "Commercial associate", "Pensioner", "State servant", "Student"
    NAME_EDUCATION_TYPE: str            # "Lower secondary", "Secondary / secondary special", "Incomplete higher", "Higher education", "Academic degree"
    NAME_FAMILY_STATUS: str             # "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
    NAME_HOUSING_TYPE: str              # "House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment"
    DAYS_BIRTH: int                     # negative integer (days since birth)
    DAYS_EMPLOYED: int                  # negative for employed, 365243 for pensioner
    OCCUPATION_TYPE: str                # "Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff", "High skill tech staff", "Medicine staff", "Cooking staff", "Security staff", "Cleaning staff", "Private service staff", "Low-skill Laborers"
    CNT_FAM_MEMBERS: float
    lowest_balance_months: int          # months of credit history

class SHAPDriver(BaseModel):
    feature: str
    impact: float

class PredictionOutput(BaseModel):
    risk_score: float
    decision: str
    threshold: float
    top_shap_drivers: list[SHAPDriver]