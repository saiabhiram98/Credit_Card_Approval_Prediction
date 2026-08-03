# credit_risk_api/preprocess.py
import pandas as pd
import numpy as np
import joblib

EDUCATION_MAP = {
    'Lower secondary': 1,
    'Secondary / secondary special': 2,
    'Incomplete higher': 3,
    'Higher education': 4,
    'Academic degree': 5
}

GENDER_MAP = {'F': 0, 'M': 1}
CAR_MAP = {'N': 0, 'Y': 1}
REALTY_MAP = {'N': 0, 'Y': 1}

def preprocess(data: dict, scaler, feature_columns: list) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # Binary encoding
    df['CODE_GENDER'] = df['CODE_GENDER'].map(GENDER_MAP)
    df['FLAG_OWN_CAR'] = df['FLAG_OWN_CAR'].map(CAR_MAP)
    df['FLAG_OWN_REALTY'] = df['FLAG_OWN_REALTY'].map(REALTY_MAP)

    # Ordinal encoding for education
    df['NAME_EDUCATION_TYPE'] = df['NAME_EDUCATION_TYPE'].map(EDUCATION_MAP)

    # Days to positive
    df['DAYS_BIRTH'] = abs(df['DAYS_BIRTH'])

    # Retirement flag + sentinel replacement
    df['Retirement_Status'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, 0)
    df['DAYS_EMPLOYED'] = abs(df['DAYS_EMPLOYED'])

    # Fill missing occupation
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')

    # One-hot encode nominal columns
    ohe_cols = ['NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Align columns with training features — fills missing OHE columns with 0
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale
    df_scaled = pd.DataFrame(scaler.transform(df), columns=feature_columns)

    return df_scaled