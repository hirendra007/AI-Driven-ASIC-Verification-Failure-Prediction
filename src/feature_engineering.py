"""
Feature Engineering Pipeline for AI-Driven ASIC Verification Failure Prediction
Computes: code_churn, historical_bug_density, coverage_trend, module_instability, regression_cost
"""

import pandas as pd
import numpy as np
from typing import Optional


def compute_features(
    hist_df: pd.DataFrame,
    lookback_weeks: int = 4,
    current_month: Optional[int] = None,
    current_week: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compute per-module feature vectors from historical verification data.

    Args:
        hist_df: Historical verification dataframe.
        lookback_weeks: Number of past weeks to use for rolling metrics.
        current_month: Month of the current commit (for filtering).
        current_week: Week of the current commit (for filtering).

    Returns:
        DataFrame with one row per module containing all ML features.
    """
    # Filter to data available BEFORE the current commit
    if current_month is not None and current_week is not None:
        # Use all data strictly before current week/month
        mask = (hist_df["month"] < current_month) | (
            (hist_df["month"] == current_month) & (hist_df["week"] < current_week)
        )
        df = hist_df[mask].copy()
    else:
        df = hist_df.copy()

    if df.empty:
        return pd.DataFrame()

    features = []

    for module in df["module_name"].unique():
        mdf = df[df["module_name"] == module].copy()

        # Sort by month, week for rolling computation
        mdf = mdf.sort_values(["month", "week"]).reset_index(drop=True)

        # Use last N rows for rolling metrics
        recent = mdf.tail(lookback_weeks)
        all_data = mdf

        # ── Feature 1: code_churn (recent average) ──────────────────────────
        code_churn = recent["loc_changed"].mean() if not recent.empty else 0

        # ── Feature 2: historical_bug_density (bugs per commit) ─────────────
        total_commits = all_data["commits"].sum()
        total_bugs = all_data["bugs_found"].sum()
        historical_bug_density = total_bugs / total_commits if total_commits > 0 else 0

        # ── Feature 3: coverage_trend (slope over recent weeks) ─────────────
        if len(recent) >= 2:
            x = np.arange(len(recent))
            y = recent["coverage_percent"].values
            coverage_trend = float(np.polyfit(x, y, 1)[0])  # slope (positive = improving)
        else:
            coverage_trend = 0.0

        # ── Feature 4: module_instability (variance of bugs) ─────────────────
        module_instability = float(all_data["bugs_found"].var()) if len(all_data) > 1 else 0.0

        # ── Feature 5: regression_cost (mean runtime) ────────────────────────
        regression_cost = recent["regression_runtime"].mean() if not recent.empty else 0

        # ── Feature 6: recent_bug_rate (last 4 weeks normalized) ─────────────
        recent_bug_rate = recent["bugs_found"].mean() if not recent.empty else 0

        # ── Feature 7: avg_coverage_recent ───────────────────────────────────
        avg_coverage = recent["coverage_percent"].mean() if not recent.empty else 85.0

        # ── Feature 8: commit_frequency ──────────────────────────────────────
        commit_frequency = all_data["commits"].mean() if not all_data.empty else 1.0

        # ── Feature 9: weeks_with_bugs (reliability proxy) ───────────────────
        weeks_with_bugs = int((all_data["bugs_found"] > 0).sum())

        # ── Feature 10: max_weekly_bugs ──────────────────────────────────────
        max_weekly_bugs = int(all_data["bugs_found"].max()) if not all_data.empty else 0

        features.append({
            "module_name": module,
            "code_churn": round(code_churn, 2),
            "historical_bug_density": round(historical_bug_density, 4),
            "coverage_trend": round(coverage_trend, 4),
            "module_instability": round(module_instability, 4),
            "regression_cost": round(regression_cost, 2),
            "recent_bug_rate": round(recent_bug_rate, 4),
            "avg_coverage": round(avg_coverage, 2),
            "commit_frequency": round(commit_frequency, 4),
            "weeks_with_bugs": weeks_with_bugs,
            "max_weekly_bugs": max_weekly_bugs,
        })

    feature_df = pd.DataFrame(features)
    return feature_df


def create_training_labels(hist_df: pd.DataFrame, threshold_percentile: float = 65) -> pd.Series:
    """
    Derive binary risk labels from historical bug density.
    Modules above the threshold percentile → 1 (HIGH RISK), else 0.

    Args:
        hist_df: Historical verification dataframe.
        threshold_percentile: Percentile cutoff for high-risk classification.

    Returns:
        Series with module_name index and binary label.
    """
    # Compute per-module aggregate bug density
    agg = hist_df.groupby("module_name").apply(
        lambda g: g["bugs_found"].sum() / max(g["commits"].sum(), 1)
    )
    threshold = np.percentile(agg.values, threshold_percentile)
    labels = (agg >= threshold).astype(int)
    return labels


def engineer_commit_features(
    commit_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    commit_id: str,
    lookback_weeks: int = 4,
) -> pd.DataFrame:
    """
    For a specific incoming commit, compute feature vectors for changed modules.

    Args:
        commit_df: Full git commit history dataframe.
        hist_df: Historical verification dataframe.
        commit_id: The specific commit to analyze.
        lookback_weeks: Lookback window for rolling features.

    Returns:
        Feature DataFrame for changed modules in this commit.
    """
    commit_rows = commit_df[commit_df["commit_id"] == commit_id]
    if commit_rows.empty:
        raise ValueError(f"Commit '{commit_id}' not found in commit log.")

    changed_modules = commit_rows["module_name"].unique().tolist()

    # Infer temporal context from commit timestamp
    latest_date = pd.to_datetime(commit_rows["timestamp"].max())
    approx_month = latest_date.month % 3 or 3  # rough milestone month
    approx_week = (latest_date.day - 1) // 7 + 1  # rough week of month

    # Get historical features for changed modules
    base_features = compute_features(
        hist_df,
        lookback_weeks=lookback_weeks,
        current_month=approx_month,
        current_week=approx_week,
    )

    # Add commit-specific churn data
    commit_churn = commit_rows.groupby("module_name").agg(
        commit_loc_added=("loc_added", "sum"),
        commit_loc_deleted=("loc_deleted", "sum"),
        commit_files_changed=("files_changed", "count"),
    ).reset_index()
    commit_churn["commit_code_churn"] = (
        commit_churn["commit_loc_added"] + commit_churn["commit_loc_deleted"]
    )

    # Merge historical features with commit-specific churn
    merged = base_features[base_features["module_name"].isin(changed_modules)].merge(
        commit_churn, on="module_name", how="left"
    ).fillna(0)

    # Override code_churn with actual commit churn
    merged["code_churn"] = merged["commit_code_churn"]

    return merged


def get_feature_columns() -> list:
    """Return ordered list of feature columns used by ML model."""
    return [
        "code_churn",
        "historical_bug_density",
        "coverage_trend",
        "module_instability",
        "regression_cost",
        "recent_bug_rate",
        "avg_coverage",
        "commit_frequency",
        "weeks_with_bugs",
        "max_weekly_bugs",
    ]


if __name__ == "__main__":
    import os
    # Quick smoke test
    hist_path = os.path.join(os.path.dirname(__file__), "../data/historical_verification_data.csv")
    hist_df = pd.read_csv(hist_path)

    print("📊 Computing feature vectors from historical data...\n")
    features = compute_features(hist_df)
    print(features.to_string(index=False))

    print("\n🏷  Deriving training labels...")
    labels = create_training_labels(hist_df)
    print(labels)
