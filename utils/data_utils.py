"""
FluxAI — Data Utilities (Hardened)
Template generation, data validation, fuzzy column matching, and sample data loading.
"""
import os
import io
import re
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Expected columns for upload validation
REQUIRED_COLUMNS = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract'
]

EXPECTED_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
    'PaymentMethod', 'MonthlyCharges', 'TotalCharges'
]

# Fuzzy column mapping: common misspellings / variations → canonical name
COLUMN_ALIASES = {
    # tenure variants
    'tenure_months': 'tenure', 'tenuremonths': 'tenure', 'tenure_month': 'tenure',
    'months': 'tenure', 'tenure (months)': 'tenure', 'customer_tenure': 'tenure',
    # MonthlyCharges variants
    'monthly_charges': 'MonthlyCharges', 'monthlycharge': 'MonthlyCharges',
    'monthly_charge': 'MonthlyCharges', 'monthly charges': 'MonthlyCharges',
    'monthlycharges': 'MonthlyCharges', 'monthly_cost': 'MonthlyCharges',
    'monthlyfee': 'MonthlyCharges', 'monthly_fee': 'MonthlyCharges',
    # TotalCharges variants
    'total_charges': 'TotalCharges', 'totalcharge': 'TotalCharges',
    'total_charge': 'TotalCharges', 'total charges': 'TotalCharges',
    'totalcharges': 'TotalCharges', 'total_cost': 'TotalCharges',
    # Contract variants
    'contract_type': 'Contract', 'contracttype': 'Contract',
    'contract type': 'Contract',
    # customerID variants
    'customer_id': 'customerID', 'customerid': 'customerID',
    'customer id': 'customerID', 'cust_id': 'customerID', 'id': 'customerID',
    # InternetService variants
    'internet_service': 'InternetService', 'internetservice': 'InternetService',
    'internet': 'InternetService', 'internet_type': 'InternetService',
    # TechSupport variants
    'tech_support': 'TechSupport', 'techsupport': 'TechSupport',
    'technical_support': 'TechSupport',
    # SeniorCitizen variants
    'senior_citizen': 'SeniorCitizen', 'seniorcitizen': 'SeniorCitizen',
    'is_senior': 'SeniorCitizen',
    # PhoneService variants
    'phone_service': 'PhoneService', 'phoneservice': 'PhoneService',
    # PaperlessBilling variants
    'paperless_billing': 'PaperlessBilling', 'paperlessbilling': 'PaperlessBilling',
    # PaymentMethod variants
    'payment_method': 'PaymentMethod', 'paymentmethod': 'PaymentMethod',
    'payment': 'PaymentMethod',
    # Other service variants
    'online_security': 'OnlineSecurity', 'onlinesecurity': 'OnlineSecurity',
    'online_backup': 'OnlineBackup', 'onlinebackup': 'OnlineBackup',
    'device_protection': 'DeviceProtection', 'deviceprotection': 'DeviceProtection',
    'streaming_tv': 'StreamingTV', 'streamingtv': 'StreamingTV',
    'streaming_movies': 'StreamingMovies', 'streamingmovies': 'StreamingMovies',
    'multiple_lines': 'MultipleLines', 'multiplelines': 'MultipleLines',
}

DATA_DICTIONARY = {
    'customerID': 'Unique customer identifier (e.g., CUST-00001)',
    'gender': 'Customer gender: Male or Female',
    'SeniorCitizen': 'Is the customer a senior citizen? 0 = No, 1 = Yes',
    'Partner': 'Does the customer have a partner? Yes or No',
    'Dependents': 'Does the customer have dependents? Yes or No',
    'tenure': 'Number of months the customer has stayed with the company',
    'PhoneService': 'Does the customer have phone service? Yes or No',
    'MultipleLines': 'Does the customer have multiple lines? Yes / No / No phone service',
    'InternetService': 'Type of internet service: DSL / Fiber optic / No',
    'OnlineSecurity': 'Does the customer have online security? Yes / No / No internet service',
    'OnlineBackup': 'Does the customer have online backup? Yes / No / No internet service',
    'DeviceProtection': 'Does the customer have device protection? Yes / No / No internet service',
    'TechSupport': 'Does the customer have tech support? Yes / No / No internet service',
    'StreamingTV': 'Does the customer stream TV? Yes / No / No internet service',
    'StreamingMovies': 'Does the customer stream movies? Yes / No / No internet service',
    'Contract': 'Contract type: Month-to-month / One year / Two year',
    'PaperlessBilling': 'Does the customer use paperless billing? Yes or No',
    'PaymentMethod': 'Payment method: Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)',
    'MonthlyCharges': 'Monthly charge amount in dollars (e.g., 29.85)',
    'TotalCharges': 'Total charges to date in dollars (e.g., 1889.50)',
}


def _fuzzy_rename_columns(df):
    """
    Attempt to rename columns using fuzzy matching.
    Returns (renamed_df, list_of_renames_applied).
    """
    renames = {}
    applied = []
    existing_lower = {col.lower().strip(): col for col in df.columns}

    for raw_col in df.columns:
        normalized = raw_col.lower().strip().replace(' ', '_')
        # Check alias map
        if normalized in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[normalized]
            if canonical not in df.columns:  # Don't overwrite existing correct column
                renames[raw_col] = canonical
                applied.append(f"'{raw_col}' → '{canonical}'")
        # Also try without underscores
        no_underscore = normalized.replace('_', '')
        if no_underscore in COLUMN_ALIASES:
            canonical = COLUMN_ALIASES[no_underscore]
            if canonical not in df.columns and raw_col not in renames:
                renames[raw_col] = canonical
                applied.append(f"'{raw_col}' → '{canonical}'")

    if renames:
        df = df.rename(columns=renames)

    return df, applied


def generate_template():
    """
    Generate a downloadable Excel template with:
    - Sheet 1: Data (with example rows)
    - Sheet 2: Data Dictionary (column descriptions)
    Returns BytesIO buffer.
    """
    buffer = io.BytesIO()

    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        # Sheet 1: Data template with examples
        example_rows = [
            {
                'customerID': 'CUST-00001', 'gender': 'Female', 'SeniorCitizen': 0,
                'Partner': 'Yes', 'Dependents': 'No', 'tenure': 24,
                'PhoneService': 'Yes', 'MultipleLines': 'No',
                'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
                'OnlineBackup': 'Yes', 'DeviceProtection': 'No',
                'TechSupport': 'No', 'StreamingTV': 'Yes', 'StreamingMovies': 'No',
                'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
                'PaymentMethod': 'Electronic check',
                'MonthlyCharges': 79.85, 'TotalCharges': 1889.50,
            },
            {
                'customerID': 'CUST-00002', 'gender': 'Male', 'SeniorCitizen': 0,
                'Partner': 'No', 'Dependents': 'Yes', 'tenure': 48,
                'PhoneService': 'Yes', 'MultipleLines': 'Yes',
                'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
                'OnlineBackup': 'No', 'DeviceProtection': 'Yes',
                'TechSupport': 'Yes', 'StreamingTV': 'No', 'StreamingMovies': 'Yes',
                'Contract': 'Two year', 'PaperlessBilling': 'No',
                'PaymentMethod': 'Bank transfer (automatic)',
                'MonthlyCharges': 56.30, 'TotalCharges': 2680.20,
            },
        ]
        data_df = pd.DataFrame(example_rows)
        data_df.to_excel(writer, sheet_name='Customer Data', index=False)

        # Sheet 2: Data Dictionary
        dict_df = pd.DataFrame([
            {'Column Name': k, 'Description': v, 'Required': '✓' if k in REQUIRED_COLUMNS else ''}
            for k, v in DATA_DICTIONARY.items()
        ])
        dict_df.to_excel(writer, sheet_name='Data Dictionary', index=False)

    buffer.seek(0)
    return buffer


def validate_upload(df):
    """
    Validate an uploaded DataFrame with fuzzy column matching.
    Returns (is_valid, cleaned_df_or_None, errors_list, warnings_list).
    """
    errors = []
    warnings = []

    # --- Guard: None or empty ---
    if df is None:
        return False, None, ["The uploaded file is empty or could not be read."], warnings

    if isinstance(df, pd.DataFrame) and df.empty:
        return False, None, ["The uploaded file contains no data rows."], warnings

    if len(df) == 0:
        return False, None, ["The uploaded file contains no data rows."], warnings

    # --- Fuzzy column rename ---
    df, renames = _fuzzy_rename_columns(df)
    if renames:
        warnings.append(f"🔄 Auto-mapped columns: {', '.join(renames)}")

    # --- Check for required columns (after fuzzy rename) ---
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        hint = "Tip: We support common variations like 'Tenure_Months', 'monthly_charges', etc."
        errors.append(f"Missing required columns: {', '.join(missing)}\n\n{hint}")
        return False, None, errors, warnings

    # --- Clean up ---
    cleaned = df.copy()

    # Coerce TotalCharges to numeric (handles whitespace, text like "Free")
    if 'TotalCharges' in cleaned.columns:
        cleaned['TotalCharges'] = pd.to_numeric(cleaned['TotalCharges'], errors='coerce')
        bad_tc = cleaned['TotalCharges'].isna().sum()
        if bad_tc > 0:
            warnings.append(f"⚠️ {bad_tc} rows had non-numeric TotalCharges (set to 0).")
        cleaned['TotalCharges'] = cleaned['TotalCharges'].fillna(0)

    # Coerce MonthlyCharges to numeric (handles text like "Free", "N/A")
    if 'MonthlyCharges' in cleaned.columns:
        cleaned['MonthlyCharges'] = pd.to_numeric(cleaned['MonthlyCharges'], errors='coerce')
        bad_mc = cleaned['MonthlyCharges'].isna().sum()
        if bad_mc > 0:
            warnings.append(f"⚠️ {bad_mc} rows had non-numeric MonthlyCharges (set to 0).")
        cleaned['MonthlyCharges'] = cleaned['MonthlyCharges'].fillna(0)

    # Coerce tenure to numeric
    if 'tenure' in cleaned.columns:
        cleaned['tenure'] = pd.to_numeric(cleaned['tenure'], errors='coerce')
        bad_ten = cleaned['tenure'].isna().sum()
        if bad_ten > 0:
            warnings.append(f"⚠️ {bad_ten} rows had non-numeric tenure (set to 0).")
        cleaned['tenure'] = cleaned['tenure'].fillna(0).astype(int)

    # Handle SeniorCitizen
    if 'SeniorCitizen' in cleaned.columns:
        cleaned['SeniorCitizen'] = cleaned['SeniorCitizen'].astype(str).replace({
            '0': 'No', '1': 'Yes', '0.0': 'No', '1.0': 'Yes'
        })

    # Fill missing optional columns with defaults
    defaults = {
        'customerID': [f'CUST-{i:05d}' for i in range(len(cleaned))],
        'gender': 'Unknown',
        'SeniorCitizen': 'No',
        'Partner': 'No',
        'Dependents': 'No',
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'No',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Electronic check',
    }

    for col, default_val in defaults.items():
        if col not in cleaned.columns:
            cleaned[col] = default_val

    # Large dataset warning
    if len(cleaned) > 50000:
        warnings.append("⚠️ Large dataset detected (>50,000 rows). Processing may be slow.")

    return True, cleaned, errors, warnings


def load_sample_data():
    """Load the pre-built sample dataset for demo mode."""
    sample_path = os.path.join(DATA_DIR, "sample_data.csv")
    if os.path.exists(sample_path):
        return pd.read_csv(sample_path)

    # If sample doesn't exist, generate on the fly
    np.random.seed(123)
    n = 1000

    df = pd.DataFrame({
        'customerID': [f'DEMO-{i:04d}' for i in range(n)],
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'tenure': np.random.randint(1, 73, n),
        'PhoneService': np.random.choice(['Yes', 'No'], n, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.21, 0.24]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice([
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ], n),
        'MonthlyCharges': np.round(np.random.uniform(18, 120, n), 2),
    })

    df['TotalCharges'] = np.round(df['tenure'] * df['MonthlyCharges'] * np.random.uniform(0.8, 1.2, n), 2)
    return df
