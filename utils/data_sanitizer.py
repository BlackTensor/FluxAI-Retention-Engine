"""
FluxAI — Data Sanitizer
Cleans and validates DataFrames before they reach the ML model,
preserving prediction accuracy by catching common data-entry issues.
"""
import re
import numpy as np
import pandas as pd

# Canonical column names the model expects
NUMERIC_COLS = {'tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen'}
BINARY_MAP = {
    'Yes': 'Yes', 'No': 'No',
    'yes': 'Yes', 'no': 'No',
    'Y': 'Yes', 'N': 'No',
    'y': 'Yes', 'n': 'No',
    '1': 'Yes', '0': 'No',
    '1.0': 'Yes', '0.0': 'No',
    True: 'Yes', False: 'No',
}
SENIOR_MAP = {'0': 'No', '1': 'Yes', '0.0': 'No', '1.0': 'Yes', 0: 'No', 1: 'Yes'}
GENDER_MAP = {
    'M': 'Male', 'F': 'Female',
    'm': 'Male', 'f': 'Female',
    'male': 'Male', 'female': 'Female',
    'MALE': 'Male', 'FEMALE': 'Female',
}


def sanitize_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Clean and validate a customer DataFrame.

    Returns
    -------
    df_clean : pd.DataFrame   — sanitized copy
    warnings : list[str]      — non-blocking issues (user can still proceed)
    errors   : list[str]      — blocking issues (must fix before proceeding)
    """
    df = df.copy()
    warnings: list[str] = []
    errors: list[str] = []

    # ── 1. Strip whitespace from all string columns ───────────────────
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        # Replace literal 'nan' / 'None' strings with actual NaN
        df[col] = df[col].replace({'nan': np.nan, 'None': np.nan, 'null': np.nan, '': np.nan})

    # ── 2. Coerce numeric columns ─────────────────────────────────────
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        original = df[col].copy()
        df[col] = pd.to_numeric(df[col], errors='coerce')
        bad_mask = original.notna() & df[col].isna()
        bad_count = bad_mask.sum()
        if bad_count:
            bad_vals = original[bad_mask].unique()[:3]
            warnings.append(
                f"**{col}**: {bad_count} value(s) couldn't be parsed as numbers "
                f"(e.g. `{', '.join(str(v) for v in bad_vals)}`). "
                f"They were set to 0."
            )
            df[col] = df[col].fillna(0)

    # ── 3. Map SeniorCitizen 0/1 → No/Yes ────────────────────────────
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].replace(SENIOR_MAP)
        df['SeniorCitizen'] = df['SeniorCitizen'].astype(str)

    # ── 4. Normalise gender aliases ───────────────────────────────────
    if 'gender' in df.columns:
        mapped = df['gender'].map(lambda x: GENDER_MAP.get(str(x).strip(), x))
        changed = (mapped != df['gender']).sum()
        if changed:
            warnings.append(f"**gender**: {changed} value(s) normalised to Male/Female.")
        df['gender'] = mapped

    # ── 5. Normalise binary Yes/No columns ───────────────────────────
    binary_cols = [
        'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    ]
    for col in binary_cols:
        if col not in df.columns:
            continue
        df[col] = df[col].map(lambda x: BINARY_MAP.get(str(x).strip(), x))

    # ── 6. Remove fully duplicate rows ────────────────────────────────
    dupes = df.duplicated().sum()
    if dupes:
        df = df.drop_duplicates()
        warnings.append(f"**{dupes} duplicate row(s)** removed.")

    # ── 7. Check minimum row count ────────────────────────────────────
    if len(df) == 0:
        errors.append("No valid rows found after cleaning. Please check your data.")

    # ── 8. Check required columns exist ──────────────────────────────
    required_cols = {'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract'}
    missing = required_cols - set(df.columns)
    if missing:
        errors.append(
            f"Missing required columns: **{', '.join(sorted(missing))}**. "
            "Download the template for the correct format."
        )

    # ── 9. Check for fully-empty critical columns ─────────────────────
    for col in ['MonthlyCharges', 'tenure']:
        if col in df.columns and df[col].isna().all():
            errors.append(f"Column **{col}** is entirely empty — cannot run predictions.")

    return df, warnings, errors
