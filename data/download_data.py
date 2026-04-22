"""
Download the IBM Telco Customer Churn dataset from a public source.
Run this script once to populate data/telco_churn.csv
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_telco_dataset():
    """Download the IBM Telco Churn dataset from a public URL."""
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "telco_churn.csv")

    if os.path.exists(output_path):
        print(f"[✓] Dataset already exists at {output_path}")
        return output_path

    # Public mirror of the IBM Telco Churn dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    print(f"[↓] Downloading dataset from {url}...")

    try:
        df = pd.read_csv(url)
        df.to_csv(output_path, index=False)
        print(f"[✓] Saved {len(df)} rows to {output_path}")
        return output_path
    except Exception as e:
        print(f"[✗] Download failed: {e}")
        print("[i] Generating synthetic dataset instead...")
        return generate_synthetic_dataset(output_path)


def generate_synthetic_dataset(output_path):
    """Generate a synthetic telco churn dataset if download fails."""
    np.random.seed(42)
    n = 7043

    df = pd.DataFrame({
        'customerID': [f'CUST-{i:05d}' for i in range(n)],
        'gender': np.random.choice(['Male', 'Female'], n),
        'SeniorCitizen': np.random.choice([0, 1], n, p=[0.84, 0.16]),
        'Partner': np.random.choice(['Yes', 'No'], n),
        'Dependents': np.random.choice(['Yes', 'No'], n, p=[0.3, 0.7]),
        'tenure': np.random.randint(0, 73, n),
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
        'TotalCharges': None,
        'Churn': None,
    })

    df['TotalCharges'] = np.round(df['tenure'] * df['MonthlyCharges'] * np.random.uniform(0.8, 1.2, n), 2)
    df.loc[df['tenure'] == 0, 'TotalCharges'] = ' '

    # Generate churn based on realistic factors
    churn_prob = np.zeros(n)
    churn_prob += (df['Contract'] == 'Month-to-month').astype(float) * 0.3
    churn_prob += (df['InternetService'] == 'Fiber optic').astype(float) * 0.15
    churn_prob += (df['tenure'] < 12).astype(float) * 0.2
    churn_prob += (df['MonthlyCharges'] > 70).astype(float) * 0.1
    churn_prob += (df['TechSupport'] == 'No').astype(float) * 0.1
    churn_prob += (df['OnlineSecurity'] == 'No').astype(float) * 0.1
    churn_prob = np.clip(churn_prob, 0, 0.85)
    df['Churn'] = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')

    df.to_csv(output_path, index=False)
    print(f"[✓] Generated synthetic dataset with {n} rows at {output_path}")
    return output_path


def generate_sample_data():
    """Generate a 1,000-row sample dataset for demo mode."""
    os.makedirs(DATA_DIR, exist_ok=True)
    output_path = os.path.join(DATA_DIR, "sample_data.csv")

    np.random.seed(123)
    n = 1000

    # Company names for flavor
    companies = [f"Customer #{i+1}" for i in range(n)]

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
        'TotalCharges': None,
        'Churn': None,
    })

    df['TotalCharges'] = np.round(df['tenure'] * df['MonthlyCharges'] * np.random.uniform(0.8, 1.2, n), 2)

    churn_prob = np.zeros(n)
    churn_prob += (df['Contract'] == 'Month-to-month').astype(float) * 0.3
    churn_prob += (df['InternetService'] == 'Fiber optic').astype(float) * 0.15
    churn_prob += (df['tenure'] < 12).astype(float) * 0.2
    churn_prob += (df['MonthlyCharges'] > 70).astype(float) * 0.1
    churn_prob += (df['TechSupport'] == 'No').astype(float) * 0.1
    churn_prob += (df['OnlineSecurity'] == 'No').astype(float) * 0.1
    churn_prob = np.clip(churn_prob, 0, 0.85)
    df['Churn'] = np.where(np.random.random(n) < churn_prob, 'Yes', 'No')

    df.to_csv(output_path, index=False)
    print(f"[✓] Generated sample dataset with {n} rows at {output_path}")
    return output_path


if __name__ == "__main__":
    download_telco_dataset()
    generate_sample_data()
