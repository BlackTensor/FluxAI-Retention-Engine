"""
FluxAI — XGBoost Churn Model Training Pipeline
Trains on IBM Telco Churn dataset, saves model + preprocessor + SHAP explainer.
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix, f1_score
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap

warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_and_clean_data(path):
    """Load and clean the IBM Telco Churn dataset."""
    print("[1/6] Loading dataset...")
    df = pd.read_csv(path)
    print(f"       → {df.shape[0]} rows, {df.shape[1]} columns")

    # Drop customer ID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # Fix TotalCharges (some have whitespace instead of numbers)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Convert SeniorCitizen to categorical for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

    # Encode target
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    print(f"       → Churn rate: {df['Churn'].mean():.1%}")
    return df


def build_preprocessor(df):
    """Build a ColumnTransformer for categorical + numerical features."""
    print("[2/6] Building preprocessor...")

    # Identify column types
    target = 'Churn'
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    num_cols = [c for c in num_cols if c != target]

    print(f"       → {len(cat_cols)} categorical, {len(num_cols)} numerical features")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols),
        ],
        remainder='drop'
    )

    return preprocessor, cat_cols, num_cols


def train_model(df, preprocessor, cat_cols, num_cols):
    """Train XGBoost with SMOTE and evaluate."""
    print("[3/6] Splitting data...")
    target = 'Churn'
    feature_cols = num_cols + cat_cols
    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"       → Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Preprocess
    print("[4/6] Preprocessing & applying SMOTE...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    print(f"       → After SMOTE: {X_train_resampled.shape[0]} training samples")

    # Train XGBoost
    print("[5/6] Training XGBoost classifier...")
    model = XGBClassifier(
        max_depth=5,
        learning_rate=0.1,
        n_estimators=300,
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=[(X_test_processed, y_test)],
        verbose=False,
    )

    # Evaluate
    y_pred = model.predict(X_test_processed)
    y_prob = model.predict_proba(X_test_processed)[:, 1]

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=['Retain', 'Churn']))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    print(f"F1 Score:      {f1_score(y_test, y_pred):.4f}")
    print("=" * 60)

    return model, X_test_processed, feature_cols


def save_artifacts(model, preprocessor, feature_cols):
    """Save model, preprocessor, SHAP explainer, and feature names."""
    print("\n[6/6] Saving model artifacts...")
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(MODELS_DIR, "xgb_model.pkl")
    joblib.dump(model, model_path)
    print(f"       → Model saved to {model_path}")

    # Save preprocessor
    prep_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
    joblib.dump(preprocessor, prep_path)
    print(f"       → Preprocessor saved to {prep_path}")

    # Save feature names
    feat_path = os.path.join(MODELS_DIR, "feature_names.pkl")
    joblib.dump(feature_cols, feat_path)
    print(f"       → Feature names saved to {feat_path}")

    # Create and save SHAP explainer
    print("       → Creating SHAP TreeExplainer (this may take a moment)...")
    explainer = shap.TreeExplainer(model)
    exp_path = os.path.join(MODELS_DIR, "shap_explainer.pkl")
    joblib.dump(explainer, exp_path)
    print(f"       → SHAP explainer saved to {exp_path}")

    print("\n[✓] All artifacts saved successfully!")


def main():
    # Check if dataset exists, if not, download it
    if not os.path.exists(DATA_PATH):
        print("[!] Dataset not found. Downloading...")
        sys.path.insert(0, os.path.join(BASE_DIR, "data"))
        from download_data import download_telco_dataset, generate_sample_data
        download_telco_dataset()
        generate_sample_data()

    df = load_and_clean_data(DATA_PATH)
    preprocessor, cat_cols, num_cols = build_preprocessor(df)
    model, X_test, feature_cols = train_model(df, preprocessor, cat_cols, num_cols)
    save_artifacts(model, preprocessor, feature_cols)


if __name__ == "__main__":
    main()
