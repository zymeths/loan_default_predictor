"""
features.py
-----------
Feature engineering and encoding for the loan default predictor.
All transforms are designed to be applied inside CV folds to prevent leakage.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder


# ── Columns to drop after engineering (low importance from feature analysis)
DROP_COLS = ['credit_band', 'interest_band', 'grade_letter', 'grade_number',
             'gender', 'marital_status']

# ── Ordinal mappings
GRADE_ORDER     = [f'{l}{n}' for l in 'ABCDEF' for n in range(1, 6)]
EDU_ORDER       = [['High School', "Bachelor's", "Master's", 'PhD']]
TARGET_ENCODE_COLS = ['loan_purpose', 'employment_status']
LABEL_ENCODE_COLS  = ['gender', 'marital_status']


def encode_base(df_train: pd.DataFrame,
                df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply ordinal and label encoding to both train and test.
    Safe to apply before the CV loop as these encoders do not use the target.
    Returns encoded copies of df_train and df_test.
    """
    train = df_train.copy()
    test  = df_test.copy()

    # ── Ordinal: grade_subgrade (A1 < A2 < ... < F5)
    oe_grade = OrdinalEncoder(
        categories=[GRADE_ORDER],
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    train['grade_subgrade'] = oe_grade.fit_transform(train[['grade_subgrade']])
    test['grade_subgrade']  = oe_grade.transform(test[['grade_subgrade']])

    # ── Ordinal: education_level
    oe_edu = OrdinalEncoder(
        categories=EDU_ORDER,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    train['education_level'] = oe_edu.fit_transform(train[['education_level']])
    test['education_level']  = oe_edu.transform(test[['education_level']])

    # ── Label: gender, marital_status
    le = LabelEncoder()
    for col in LABEL_ENCODE_COLS:
        combined = pd.concat([train[col], test[col]])
        le.fit(combined)
        train[col] = le.transform(train[col])
        test[col]  = le.transform(test[col])

    return train, test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio and interaction features.
    Must be called after base encoding and target encoding.
    """
    df = df.copy()

    df['loan_to_income']     = df['loan_amount'] / (df['annual_income'] + 1)
    df['interest_burden']    = df['loan_amount'] * df['interest_rate'] / 100
    df['implied_total_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['loan_to_credit']     = df['loan_amount'] / (df['credit_score'] + 1)

    df['credit_band'] = pd.cut(
        df['credit_score'],
        bins=[0, 580, 669, 739, 799, 900],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    df['interest_band'] = pd.cut(
        df['interest_rate'],
        bins=[0, 8, 13, 18, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    df['grade_letter'] = df['grade_subgrade'].astype(str).str[0]
    df['grade_number'] = pd.to_numeric(
        df['grade_subgrade'].astype(str).str[1], errors='coerce'
    ).fillna(0).astype(int)

    return df
