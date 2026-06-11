"""
FluxAI — Predictor Module
Loads trained model artifacts and provides prediction + SHAP explanation functions.
"""
import os
import threading
import numpy as np
import pandas as pd
import joblib
import shap

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def _unwrap_shap(shap_values):
    """
    Normalise SHAP output to a single 2-D ndarray (samples × features).

    TreeExplainer returns a list [neg_class, pos_class] for binary classifiers
    and a plain ndarray for regressors / single-output models.  We always want
    the positive (churn) class.
    """
    if isinstance(shap_values, list):
        return shap_values[1] if len(shap_values) > 1 else shap_values[0]
    return shap_values


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
            print(f"[ERROR] Model artifacts not found: {e}")
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

        feature_df = self._prepare_features(df)
        X = self.preprocessor.transform(feature_df)

        probs = self.model.predict_proba(X)[:, 1]

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

        Only the target row is preprocessed and scored — the full DataFrame is
        never transformed, keeping this O(1) in dataset size.

        Returns dict with:
          - feature_names: list of feature names
          - shap_values: list of SHAP values
          - base_value: the base prediction value
        """
        if not self._loaded:
            if not self.load():
                return None

        # Clamp idx before slicing so _prepare_features receives exactly one row
        idx = max(0, min(idx, len(df) - 1))
        feature_df = self._prepare_features(df.iloc[[idx]])
        X = self.preprocessor.transform(feature_df)

        try:
            raw = self.explainer.shap_values(X)
        except Exception:
            return None

        readable_names = self._get_readable_feature_names()

        try:
            sv = _unwrap_shap(raw)
            # sv is now shape (1, n_features); take the single row
            sv_list = sv[0].tolist() if hasattr(sv[0], 'tolist') else list(sv[0])
        except Exception:
            sv_list = [0.0] * len(readable_names)

        try:
            ev = self.explainer.expected_value
            if isinstance(ev, (int, float, np.floating)):
                base = float(ev)
            elif isinstance(ev, np.ndarray):
                base = float(ev[-1])  # churn (positive) class
            else:
                base = float(ev)
        except Exception:
            base = 0.0

        return {
            'feature_names': readable_names,
            'shap_values': sv_list,
            'base_value': base,
        }

    def get_global_shap(self, df, max_samples=200):
        """
        Get global SHAP feature importance using a random sample of rows.

        Random sampling avoids positional bias that arises when data is sorted
        (e.g., by risk score or customer ID).
        """
        if not self._loaded:
            if not self.load():
                raise FileNotFoundError("Model artifacts not found. Run training first.")

        feature_df = self._prepare_features(df)

        n = min(len(feature_df), max_samples)
        sample_df = feature_df.sample(n=n, random_state=42)

        X = self.preprocessor.transform(sample_df)
        raw = self.explainer.shap_values(X)

        # Unwrap list output from binary classifiers before aggregating
        sv = _unwrap_shap(raw)

        readable_names = self._get_readable_feature_names()
        mean_abs = np.abs(sv).mean(axis=0)

        importance = sorted(
            zip(readable_names, mean_abs),
            key=lambda x: x[1],
            reverse=True,
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
            raw = self.explainer.shap_values(X)

            sv = _unwrap_shap(raw)   # shape (n, features)

            top_idx = np.argmax(np.abs(sv), axis=1)   # shape (n,)
            top_shap = sv[np.arange(n), top_idx]       # shape (n,)

            def _label(feat_name: str, shap_val: float) -> str:
                fn = feat_name.strip()
                direction = 'High' if shap_val > 0 else 'Low'

                if fn.lower() == 'contract':
                    return 'Contract Type'
                if fn.lower() in ('internetservice', 'internet service'):
                    return 'Internet Service'
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
                return f'{direction} {fn}'

            labels = [
                _label(readable_names[top_idx[i]], top_shap[i])
                for i in range(n)
            ]

        except Exception:
            labels = ['See Deep Dive'] * n

        fallback = ['See Deep Dive'] * (len(df) - n)
        return pd.Series(labels + fallback, index=df.index)

    def _prepare_features(self, df):
        """Prepare a DataFrame to match the training feature set."""
        feature_df = df.copy()

        for col in self.feature_names:
            if col not in feature_df.columns:
                feature_df[col] = 0

        for num_col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
            if num_col in feature_df.columns:
                feature_df[num_col] = pd.to_numeric(
                    feature_df[num_col], errors='coerce'
                ).fillna(0)

        if 'SeniorCitizen' in feature_df.columns:
            feature_df['SeniorCitizen'] = feature_df['SeniorCitizen'].astype(str)
            feature_df['SeniorCitizen'] = feature_df['SeniorCitizen'].replace({
                '0': 'No', '1': 'Yes',
                '0.0': 'No', '1.0': 'Yes',
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


# ── Singleton ────────────────────────────────────────────────────────────────

_predictor: "ChurnPredictor | None" = None
_predictor_lock = threading.Lock()


def get_predictor() -> ChurnPredictor:
    """
    Return the process-wide ChurnPredictor, creating it on first call.

    Double-checked locking ensures only one instance is ever created even when
    multiple threads (e.g. the recovery playbook thread) call this simultaneously.
    """
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:          # re-check inside the lock
                instance = ChurnPredictor()
                instance.load()
                _predictor = instance       # assign only after full initialisation
    return _predictor
