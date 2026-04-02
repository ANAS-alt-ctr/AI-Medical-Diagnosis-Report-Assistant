"""
train_model.py
──────────────
Standalone script to train and save the XGBoost risk model.
Run:  python train_model.py
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

SYMPTOM_FEATURES = [
    "fever", "cough", "shortness_of_breath", "chest_pain",
    "fatigue", "headache", "nausea", "skin_rash",
    "joint_pain", "weight_loss", "night_sweats", "dizziness",
]

MODEL_PATH = Path("models/risk_model.pkl")


def generate_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic but realistic symptom dataset."""
    rng = np.random.default_rng(seed)

    ages       = rng.integers(5, 90, size=n)
    genders    = rng.integers(0, 2, size=n)          
    durations  = rng.integers(1, 6, size=n)           
    baselines  = rng.uniform(0.05, 0.95, size=n)
    symptoms   = rng.integers(0, 2, size=(n, len(SYMPTOM_FEATURES)))

    risk_raw = (
        0.20 * (ages / 90)
        + 0.25 * baselines
        + 0.20 * (symptoms.sum(axis=1) / len(SYMPTOM_FEATURES))
        + 0.15 * (durations / 5)
        + 0.10 * genders
        + rng.uniform(-0.08, 0.08, size=n)
    )
    target = (np.clip(risk_raw, 0, 1) > 0.45).astype(int)

    df = pd.DataFrame({
        "age":           ages / 100,
        "gender":        genders.astype(float),
        "duration":      durations / 5.0,
        "baseline_risk": baselines,
        **{feat: symptoms[:, i].astype(float) for i, feat in enumerate(SYMPTOM_FEATURES)},
        "target":        target,
    })
    return df


def train(n_samples: int = 5000):
    """Train XGBoost model and save to disk."""
    try:
        import xgboost as xgb
        from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
        from sklearn.metrics import roc_auc_score, classification_report
    except ImportError as e:
        print(f"ERROR: Required package not found — {e}")
        print("Install with: pip install xgboost scikit-learn")
        return

    data_path = Path("data/symptoms_data.csv")
    if data_path.exists():
        print(f"Loading data from {data_path}…")
        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c != "target"]
    else:
        print(f"No data/symptoms_data.csv found — generating {n_samples} synthetic samples…")
        df = generate_dataset(n_samples)
        feature_cols = [c for c in df.columns if c != "target"]
        data_path.parent.mkdir(exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"  ✓ Saved synthetic dataset to {data_path}")

    X = df[feature_cols].values
    y = df["target"].values
    print(f"Dataset: {len(df)} rows, {len(feature_cols)} features, {y.mean():.1%} positive rate")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )

    print("Training XGBoost model…")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred       = (y_pred_proba >= 0.5).astype(int)
    auc          = roc_auc_score(y_test, y_pred_proba)
    print(f"\n── Evaluation ──────────────────────────────")
    print(f"  ROC-AUC : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Low Risk","High Risk"]))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    bundle = {"model": model, "feature_cols": feature_cols}
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)
    print(f"  ✓ Model saved to {MODEL_PATH}")

    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\n── Top 5 Feature Importances ───────────────")
    for feat, imp in feat_imp[:5]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22} {imp:.4f}  {bar}")


if __name__ == "__main__":
    train()
