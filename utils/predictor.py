"""
FluxAI — Predictor Module
Loads trained model artifacts and provides prediction + SHAP explanation functions.
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class ChurnPredictor:
    """Encapsulates model loading, prediction, and SHAP explanation."""

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.explainer = None
        self.feature_names = None
        self._loaded = False

    def load(self):
        """Load all model artifacts from disk."""
        if self._loaded:
            return True

        try:
            self.model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.pkl"))
            self.preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
            self.explainer = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
            self.feature_names = joblib.load(os.path.join(MODELS_DIR, "feature_names.pkl"))
            self._loaded = True
            return True
        except FileNotFoundError as e:
            print(f"[✗] Model artifacts not found: {e}")
            print("[i] Run 'python training/train_model.py' first.")
            return False

    def predict(self, df):
        """
        Run churn prediction on a DataFrame.
        Returns the original DataFrame with added columns:
          - Churn_Probability (0-100%)
          - Risk_Level (Low / Medium / High / Critical)
        """
        if not self._loaded:
            if not self.load():
                raise FileNotFoundError("Model artifacts not found. Run training first.")

        # Prepare features — use only the columns the model was trained on
        feature_df = self._prepare_features(df)
        X = self.preprocessor.transform(feature_df)

        # Predict probabilities
        probs = self.model.predict_proba(X)[:, 1]

        # Add to result DataFrame
        result = df.copy()
        result['Churn_Probability'] = np.round(probs * 100, 2)
        # Use string Risk_Level to avoid Categorical comparison issues in filtering
        risk_labels = []
        for p in result['Churn_Probability']:
            if p >= 75:
                risk_labels.append('Critical')
            elif p >= 50:
                risk_labels.append('High')
            elif p >= 25:
                risk_labels.append('Medium')
            else:
                risk_labels.append('Low')
        result['Risk_Level'] = risk_labels

        return result

    def explain(self, df, idx):
        """
        Get SHAP explanation for a single customer (by DataFrame position index).
        Returns dict with:
          - feature_names: list of feature names
          - shap_values: list of SHAP values
          - base_value: the base prediction value
        """
        if not self._loaded:
            if not self.load():
                return None

        feature_df = self._prepare_features(df)
        X = self.preprocessor.transform(feature_df)

        # Clamp idx to valid range
        idx = max(0, min(idx, X.shape[0] - 1))

        # Get SHAP values for the specific row
        try:
            shap_values = self.explainer.shap_values(X[idx:idx+1])
        except Exception:
            return None

        # Build readable feature names from the preprocessor
        readable_names = self._get_readable_feature_names()

        # Handle different SHAP output formats
        try:
            if isinstance(shap_values, list):
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0]
            elif isinstance(shap_values, np.ndarray):
                sv = shap_values[0]
            else:
                sv = shap_values[0]
            sv_list = sv.tolist() if hasattr(sv, 'tolist') else list(sv)
        except Exception:
            sv_list = [0.0] * len(readable_names)

        try:
            if isinstance(self.explainer.expected_value, (int, float, np.floating)):
                base = float(self.explainer.expected_value)
            elif isinstance(self.explainer.expected_value, np.ndarray):
                base = float(self.explainer.expected_value[-1])  # Take churn class
            else:
                base = float(self.explainer.expected_value)
        except Exception:
            base = 0.0

        return {
            'feature_names': readable_names,
            'shap_values': sv_list,
            'base_value': base,
        }

    def get_global_shap(self, df, max_samples=200):
        """Get global SHAP values for feature importance summary."""
        if not self._loaded:
            self.load()

        feature_df = self._prepare_features(df)
        sample_df = feature_df.head(max_samples)
        X = self.preprocessor.transform(sample_df)

        shap_values = self.explainer.shap_values(X)
        readable_names = self._get_readable_feature_names()

        # Mean absolute SHAP value per feature
        mean_abs = np.abs(shap_values).mean(axis=0)
        importance = sorted(
            zip(readable_names, mean_abs),
            key=lambda x: x[1],
            reverse=True
        )

        return importance

    def get_top_drivers(self, df, max_rows=500) -> pd.Series:
        """
        Batch-compute the top churn driver for each customer.

        Returns a pd.Series (aligned with df) of human-readable strings,
        e.g. 'High Monthly Charges', 'Low Tenure', 'Month-to-Month Contract'.
        Capped at max_rows for performance; remaining rows get 'See Deep Dive'.
        """
        if not self._loaded:
            self.load()

        readable_names = self._get_readable_feature_names()
        n = min(len(df), max_rows)

        try:
            feature_df = self._prepare_features(df.head(n))
            X = self.preprocessor.transform(feature_df)
            shap_values = self.explainer.shap_values(X)   # (n, features)

            # Handle list output (binary classifiers sometimes return [neg, pos])
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            # For each row pick the feature with highest |SHAP| value
            top_idx = np.argmax(np.abs(sv), axis=1)        # shape (n,)
            top_shap = sv[np.arange(n), top_idx]           # shape (n,)

            def _label(feat_name: str, shap_val: float) -> str:
                """Turn a feature + SHAP direction into a readable label."""
                fn = feat_name.strip()
                direction = 'High' if shap_val > 0 else 'Low'

                # Contract type — just use it directly
                if fn.lower() == 'contract':
                    return f'Contract Type'
                if fn.lower() in ('internetservice', 'internet service'):
                    return f'Internet Service'
                if fn.lower() in ('tenure',):
                    return f'{direction} Tenure'
                if fn.lower() in ('monthlycharges', 'monthly charges'):
                    return f'{direction} Monthly Charges'
                if fn.lower() in ('totalcharges', 'total charges'):
                    return f'{direction} Total Charges'
                if fn.lower() in ('techsupport', 'tech support'):
                    return 'No Tech Support' if shap_val > 0 else 'Has Tech Support'
                if fn.lower() in ('onlinesecurity', 'online security'):
                    return 'No Online Security' if shap_val > 0 else 'Has Online Security'
                if fn.lower() in ('seniorcitizen', 'senior citizen'):
                    return 'Senior Citizen' if shap_val > 0 else 'Not Senior'
                # Generic fallback
                return f'{direction} {fn}'

            labels = [
                _label(readable_names[top_idx[i]], top_shap[i])
                for i in range(n)
            ]

        except Exception:
            labels = ['See Deep Dive'] * n

        # Pad remaining rows (if df > max_rows) with a fallback label
        fallback = ['See Deep Dive'] * (len(df) - n)
        result = pd.Series(labels + fallback, index=df.index)
        return result

    def _prepare_features(self, df):
        """Prepare a DataFrame to match the training feature set."""
        feature_df = df.copy()

        # Ensure all expected columns exist
        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0  # Default for missing columns

        # Handle numeric columns — coerce text like 'Free', 'N/A' to 0
        for num_col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
            if num_col in feature_df.columns:
                feature_df[num_col] = pd.to_numeric(
                    feature_df[num_col], errors='coerce'
                ).fillna(0)

        # Handle SeniorCitizen — convert to string if needed
        if 'SeniorCitizen' in feature_df.columns:
            feature_df['SeniorCitizen'] = feature_df['SeniorCitizen'].astype(str)
            feature_df['SeniorCitizen'] = feature_df['SeniorCitizen'].replace({
                '0': 'No', '1': 'Yes',
                '0.0': 'No', '1.0': 'Yes'
            })

        return feature_df[self.feature_names]

    def _get_readable_feature_names(self):
        """Get human-readable feature names from the preprocessor."""
        names = []
        for name, transformer, cols in self.preprocessor.transformers_:
            if name == 'remainder':
                continue
            names.extend(cols)
        return names


# Singleton instance
_predictor = None


def get_predictor():
    """Get or create the singleton ChurnPredictor."""
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor()
        _predictor.load()
    return _predictor
