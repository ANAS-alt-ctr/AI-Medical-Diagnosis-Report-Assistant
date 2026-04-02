"""
risk_predictor.py
─────────────────
XGBoost-based disease risk predictor with SHAP explainability.
Trains on symptoms_data.csv (or generates synthetic training data).
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple


SYMPTOM_FEATURES = [
    "fever", "cough", "shortness_of_breath", "chest_pain",
    "fatigue", "headache", "nausea", "skin_rash",
    "joint_pain", "weight_loss", "night_sweats", "dizziness",
]

DISEASE_RISK_BASELINE = {
    "Pneumonia":            0.72,
    "Pneumothorax":         0.88,
    "Edema":                0.65,
    "Consolidation":        0.60,
    "Mass":                 0.78,
    "Melanoma":             0.82,
    "Basal Cell Carcinoma": 0.70,
    "Cardiomegaly":         0.65,
    "Atelectasis":          0.55,
    "Fibrosis":             0.60,
    "Infiltration":         0.52,
    "Nodule":               0.48,
    "Emphysema":            0.58,
    "Pleural Thickening":   0.50,
    "No Finding":           0.05,
    "Unknown":              0.40,
}

MODEL_PATH = Path("models/risk_model.pkl")


def _symptom_key(symptom: str) -> str:
    """Normalise a display symptom name to a feature key."""
    return symptom.lower().replace(" ", "_").replace("-", "_")


def _build_feature_vector(patient_info: Dict, disease: str) -> np.ndarray:
    """Convert patient data into a numeric feature vector."""
    age     = float(patient_info.get("age", 35))
    gender  = 1.0 if patient_info.get("gender", "Male") == "Male" else 0.0
    dur_map = {
        "< 1 week": 1, "1-2 weeks": 2, "2-4 weeks": 3,
        "1-3 months": 4, "> 3 months": 5
    }
    duration = float(dur_map.get(patient_info.get("duration", "< 1 week"), 1))
    baseline = DISEASE_RISK_BASELINE.get(disease, 0.4)

    active_keys = {_symptom_key(s) for s in patient_info.get("symptoms", [])}
    sym_vec = np.array([1.0 if s in active_keys else 0.0 for s in SYMPTOM_FEATURES])

    feat = np.concatenate([
        [age / 100.0, gender, duration / 5.0, baseline],
        sym_vec,
    ])
    return feat.reshape(1, -1)


def _generate_synthetic_data(n: int = 2000) -> pd.DataFrame:
    """Create synthetic patient data for training."""
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        age      = rng.integers(5, 90)
        gender   = rng.integers(0, 2)
        duration = rng.integers(1, 6)
        baseline = rng.uniform(0.05, 0.90)
        symptoms = rng.integers(0, 2, size=len(SYMPTOM_FEATURES))
        risk_raw = (
            0.25 * (age / 100)
            + 0.20 * baseline
            + 0.15 * (symptoms.sum() / len(SYMPTOM_FEATURES))
            + 0.10 * (duration / 5)
            + rng.uniform(-0.1, 0.1)
        )
        target = int(np.clip(risk_raw, 0, 1) > 0.45)
        rows.append([age / 100, gender, duration / 5, baseline, *symptoms, target])

    cols = ["age", "gender", "duration", "baseline_risk", *SYMPTOM_FEATURES, "target"]
    return pd.DataFrame(rows, columns=cols)


def _train_and_save() -> Any:
    """Train XGBoost on CSV data (or synthetic) and save to disk."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score

        data_path = Path("data/symptoms_data.csv")
        if data_path.exists():
            df = pd.read_csv(data_path)
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values
            y = df["target"].values
        else:
            df = _generate_synthetic_data(3000)
            feature_cols = [c for c in df.columns if c != "target"]
            X = df[feature_cols].values
            y = df["target"].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        MODEL_PATH.parent.mkdir(exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": model, "feature_cols": feature_cols}, f)

        return model, feature_cols

    except ImportError:
        return None, None


def _load_model():
    """Load saved model or train a new one."""
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                bundle = pickle.load(f)
            return bundle["model"], bundle.get("feature_cols", [])
        except Exception:
            pass
    return _train_and_save()


def _shap_explanation(model, feature_vector: np.ndarray, feature_names: List[str]) -> List[Tuple[str, float]]:
    """Compute SHAP values and return top 3 contributing features."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(feature_vector)
        if isinstance(sv, list):
            sv = sv[1]
        sv_flat = sv.flatten()
        pairs = list(zip(feature_names, sv_flat))
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        def _human(k):
            return k.replace("_", " ").replace("baseline risk", "disease baseline").title()
        return [(_human(k), float(v)) for k, v in pairs[:3]]
    except Exception:
        return []


def predict_risk(patient_info: Dict, disease: str = "Unknown") -> Dict[str, Any]:
    """
    Predict patient risk score using XGBoost.

    Returns:
        {
            "risk_score":  float (0-100),
            "risk_level":  str   (LOW/MEDIUM/HIGH/CRITICAL),
            "top_factors": list[tuple[str, float]],
            "probability": float (0-1),
        }
    """
    result = {
        "risk_score":  0.0,
        "risk_level":  "UNKNOWN",
        "top_factors": [],
        "probability": 0.0,
    }

    feature_vector = _build_feature_vector(patient_info, disease)

    model, feature_cols = _load_model()

    if model is not None:
        try:
            if feature_vector.shape[1] != len(feature_cols):
                fv = feature_vector
            else:
                fv = feature_vector

            prob = float(model.predict_proba(fv)[0][1])
            result["probability"] = prob
            result["risk_score"]  = round(prob * 100, 1)
            result["top_factors"] = _shap_explanation(model, fv, feature_cols or
                ["age","gender","duration","baseline_risk"]+SYMPTOM_FEATURES)
        except Exception as e:
            result["error_model"] = str(e)

    if result["probability"] == 0.0:
        age        = float(patient_info.get("age", 35))
        n_symptoms = len(patient_info.get("symptoms", []))
        baseline   = DISEASE_RISK_BASELINE.get(disease, 0.4)
        dur_map    = {"< 1 week":1,"1-2 weeks":2,"2-4 weeks":3,"1-3 months":4,"> 3 months":5}
        duration   = dur_map.get(patient_info.get("duration","< 1 week"), 1)

        prob = (
            0.25 * min(age / 80, 1.0)
            + 0.30 * baseline
            + 0.25 * min(n_symptoms / 6, 1.0)
            + 0.10 * min(duration / 5, 1.0)
            + 0.10 * (1.0 if patient_info.get("gender") == "Male" else 0.5)
        )
        prob = float(np.clip(prob, 0.01, 0.99))
        result["probability"] = prob
        result["risk_score"]  = round(prob * 100, 1)

        factors = []
        if baseline > 0.6:
            factors.append(("Disease Baseline Risk", baseline - 0.4))
        if age > 60:
            factors.append(("Age", (age - 40) / 80))
        if n_symptoms >= 3:
            factors.append(("Symptom Count", n_symptoms / 12))
        if not factors:
            factors = [("Duration", duration / 10), ("No Critical Factor", 0.0)]
        result["top_factors"] = [(k, float(v)) for k, v in factors[:3]]

    score = result["risk_score"]
    if score < 25:
        result["risk_level"] = "LOW"
    elif score < 50:
        result["risk_level"] = "MEDIUM"
    elif score < 75:
        result["risk_level"] = "HIGH"
    else:
        result["risk_level"] = "CRITICAL"

    return result
