"""
XGBoost Risk Prediction Training Pipeline
Trains a binary classifier: 0 = LOW RISK, 1 = HIGH RISK
Saves model artifact to models/xgb_risk_model.joblib
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
try:
    import xgboost as xgb
    USE_XGB = True
except ImportError:
    USE_XGB = False
    print("ℹ  XGBoost not installed — using sklearn GradientBoostingClassifier (equivalent for hackathon)")
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))
from feature_engineering import compute_features, create_training_labels, get_feature_columns

MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/xgb_risk_model.joblib")
METADATA_PATH = os.path.join(os.path.dirname(__file__), "../models/model_metadata.json")
FEATURE_IMPORTANCE_PLOT = os.path.join(os.path.dirname(__file__), "../models/feature_importance.png")


def build_training_dataset(hist_df: pd.DataFrame, lookback_weeks: int = 4):
    """
    Build (X, y) training pairs by sliding a window over historical weeks.
    For each week, compute features from data up to that week and label
    from that week's outcome.
    """
    X_rows, y_rows = [], []
    months = sorted(hist_df["month"].unique())
    weeks_order = [(m, w) for m in months for w in sorted(hist_df[hist_df["month"] == m]["week"].unique())]

    for idx, (month, week) in enumerate(weeks_order):
        if idx < lookback_weeks:
            continue  # Need enough history

        # Get features from data BEFORE this period
        features_df = compute_features(hist_df, lookback_weeks, current_month=month, current_week=week)
        if features_df.empty:
            continue

        # Get labels FROM this week's actual outcomes
        week_data = hist_df[(hist_df["month"] == month) & (hist_df["week"] == week)]
        if week_data.empty:
            continue

        # Label: bug density this week above median → high risk
        week_density = week_data.set_index("module_name").apply(
            lambda r: r["bugs_found"] / max(r["commits"], 1), axis=1
        )
        median_density = week_density.median()
        week_labels = (week_density >= median_density).astype(int)

        # Merge features with labels
        for _, row in features_df.iterrows():
            module = row["module_name"]
            if module in week_labels.index:
                X_rows.append(row.to_dict())
                y_rows.append(week_labels[module])

    X = pd.DataFrame(X_rows)
    y = pd.Series(y_rows, name="module_risk")
    return X, y


def train_model(hist_df: pd.DataFrame, save: bool = True) -> tuple:
    """
    Train XGBoost classifier on historical verification data.

    Returns:
        (pipeline, feature_columns, metrics_dict)
    """
    print("🔧 Building training dataset...")
    X, y = build_training_dataset(hist_df)

    feature_cols = get_feature_columns()
    X_feat = X[feature_cols].fillna(0)

    print(f"   Training samples: {len(X_feat)}, Class distribution: {dict(y.value_counts())}")

    # ── Model Pipeline ────────────────────────────────────────────────────────
    if USE_XGB:
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    else:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, min_samples_leaf=3, random_state=42,
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    # ── Cross-Validation ──────────────────────────────────────────────────────
    print("📊 Running 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_feat, y, cv=cv, scoring="roc_auc")
    print(f"   ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Final Training ────────────────────────────────────────────────────────
    pipeline.fit(X_feat, y)

    # ── Evaluation on full training set ──────────────────────────────────────
    y_pred = pipeline.predict(X_feat)
    y_prob = pipeline.predict_proba(X_feat)[:, 1]
    auc = roc_auc_score(y, y_prob)

    print("\n📋 Classification Report:")
    print(classification_report(y, y_pred, target_names=["LOW RISK", "HIGH RISK"]))

    metrics = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "train_roc_auc": float(auc),
        "n_training_samples": len(X_feat),
        "class_distribution": dict(y.value_counts().astype(str)),
        "feature_columns": feature_cols,
    }

    # ── Feature Importance Plot ───────────────────────────────────────────────
    clf_model = pipeline.named_steps["clf"]
    if USE_XGB:
        booster = clf_model.get_booster()
        importance_dict = booster.get_score(importance_type="gain")
        importance_df = pd.DataFrame(
            [(feature_cols[int(k[1:])] if k.startswith("f") else k, v)
             for k, v in importance_dict.items()],
            columns=["feature", "importance"]
        ).sort_values("importance", ascending=True)
    else:
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": clf_model.feature_importances_,
        }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#2563eb")
    ax.set_title("XGBoost Feature Importance (Gain)", fontweight="bold")
    ax.set_xlabel("Gain")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PLOT), exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Feature importance plot → {FEATURE_IMPORTANCE_PLOT}")

    # ── Save Model ────────────────────────────────────────────────────────────
    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(pipeline, MODEL_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n💾 Model saved → {MODEL_PATH}")
        print(f"💾 Metadata  → {METADATA_PATH}")

    return pipeline, feature_cols, metrics


def load_model() -> tuple:
    """Load saved model and metadata."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model() first."
        )
    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    return pipeline, metadata["feature_columns"], metadata


def predict_risk(pipeline, feature_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Predict risk scores for a set of modules.

    Args:
        pipeline: Trained sklearn Pipeline.
        feature_df: DataFrame with module features.
        feature_cols: List of feature column names.

    Returns:
        DataFrame with module_name, risk_score, risk_level columns.
    """
    X = feature_df[feature_cols].fillna(0)
    proba = pipeline.predict_proba(X)[:, 1]

    results = feature_df[["module_name"]].copy()
    results["risk_score"] = proba.round(4)
    results["risk_level"] = results["risk_score"].apply(
        lambda s: "HIGH" if s >= 0.65 else ("MEDIUM" if s >= 0.40 else "LOW")
    )
    results = results.sort_values("risk_score", ascending=False).reset_index(drop=True)
    return results


if __name__ == "__main__":
    hist_path = os.path.join(os.path.dirname(__file__), "../data/historical_verification_data.csv")
    if not os.path.exists(hist_path):
        print("⚠  Historical data not found. Run data/generate_datasets.py first.")
        sys.exit(1)

    hist_df = pd.read_csv(hist_path)
    pipeline, feature_cols, metrics = train_model(hist_df)
    print("\n✅ Training complete!")
    print(f"   CV ROC-AUC: {metrics['cv_roc_auc_mean']:.3f} ± {metrics['cv_roc_auc_std']:.3f}")
