"""
Bug Trend Prediction
Predicts expected bugs in the current milestone using historical burn-down data.
Uses Prophet / linear regression for time-series forecasting.
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


MODULES = ["ALU", "Decoder", "Cache", "DMA", "AXI", "FIFO", "BranchUnit", "ControlUnit"]


def get_milestone_context(
    bug_trend_df: pd.DataFrame,
    current_month: int,
    current_week: int,
) -> tuple:
    """
    Determine which historical data to use for milestone prediction.

    Rule: Use all complete months + completed weeks of current month.
    Example: Commit in Week 3, Month 3 → use Month 1, Month 2, Week 1, Week 2.
    """
    # Data from all complete months before current
    complete_months = bug_trend_df[bug_trend_df["month"] < current_month].copy()

    # Data from current month but only weeks before current week
    partial_month = bug_trend_df[
        (bug_trend_df["month"] == current_month) & (bug_trend_df["week"] < current_week)
    ].copy()

    context = pd.concat([complete_months, partial_month], ignore_index=True)
    return context, complete_months, partial_month


def predict_milestone_bugs(
    bug_trend_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    current_month: int,
    current_week: int,
) -> dict:
    """
    Predict total bugs expected in the current milestone.

    Returns:
        {
            "predicted_bugs": int,
            "confidence_interval": (low, high),
            "trend": "INCREASING" | "DECREASING" | "STABLE",
            "weekly_forecast": list,
            "module_breakdown": dict,
        }
    """
    context, complete_months, partial = get_milestone_context(
        bug_trend_df, current_month, current_week
    )

    if context.empty:
        return _default_prediction()

    # ── Overall trend prediction ──────────────────────────────────────────────
    weekly_bugs = context["total_bugs"].values
    X = np.arange(len(weekly_bugs)).reshape(-1, 1)

    # Polynomial regression for trend
    model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))
    model.fit(X, weekly_bugs)

    # Predict remaining weeks in milestone
    weeks_in_milestone = 4  # Assume 4-week milestone
    weeks_completed = current_week - 1 + (current_month - 1) * 4
    weeks_elapsed_in_milestone = (current_month - 1) * 4 + current_week - 1
    weeks_remaining = max(1, weeks_in_milestone - (current_week - 1))

    forecast_weeks = []
    for i in range(1, weeks_remaining + 1):
        x_pred = np.array([[len(weekly_bugs) + i]])
        predicted = max(0, float(model.predict(x_pred)[0]))
        forecast_weeks.append(round(predicted, 1))

    # Bugs already found this milestone
    bugs_so_far = context[context["month"] == current_month]["total_bugs"].sum()
    predicted_total = bugs_so_far + sum(forecast_weeks)

    # Confidence interval (±15% heuristic based on variance)
    std_dev = float(np.std(weekly_bugs)) if len(weekly_bugs) > 1 else 1.0
    ci_low = max(0, int(predicted_total - 1.5 * std_dev))
    ci_high = int(predicted_total + 1.5 * std_dev)

    # Trend direction
    if len(weekly_bugs) >= 3:
        recent_slope = np.polyfit(np.arange(3), weekly_bugs[-3:], 1)[0]
        if recent_slope > 0.5:
            trend = "INCREASING"
        elif recent_slope < -0.5:
            trend = "DECREASING"
        else:
            trend = "STABLE"
    else:
        trend = "STABLE"

    # ── Per-module breakdown ──────────────────────────────────────────────────
    module_breakdown = _predict_module_bugs(hist_df, current_month, current_week)

    return {
        "predicted_bugs": round(predicted_total),
        "bugs_found_so_far": int(bugs_so_far),
        "confidence_interval": (ci_low, ci_high),
        "trend": trend,
        "weekly_forecast": forecast_weeks,
        "module_breakdown": module_breakdown,
        "weeks_used": len(context),
        "context_summary": {
            "complete_months": int(current_month - 1),
            "partial_weeks": int(current_week - 1),
        },
    }


def _predict_module_bugs(
    hist_df: pd.DataFrame,
    current_month: int,
    current_week: int,
) -> dict:
    """Predict bug count per module for the current milestone."""
    context = hist_df[
        (hist_df["month"] < current_month) |
        ((hist_df["month"] == current_month) & (hist_df["week"] < current_week))
    ]

    if context.empty:
        return {m: 0 for m in MODULES}

    module_predictions = {}
    for module in MODULES:
        mdata = context[context["module_name"] == module]["bugs_found"].values
        if len(mdata) == 0:
            module_predictions[module] = 0
            continue

        # Use weighted average (recent weeks weighted more)
        weights = np.exp(np.linspace(0, 1, len(mdata)))
        weights /= weights.sum()
        avg_weekly = float(np.average(mdata, weights=weights))

        # Scale to remaining milestone weeks
        milestone_week = current_week
        weeks_remaining = max(1, 4 - milestone_week)
        predicted = avg_weekly * (weeks_remaining + 1)

        module_predictions[module] = max(0, round(predicted, 1))

    return module_predictions


def generate_weekly_summaries(
    hist_df: pd.DataFrame,
    current_month: int,
    current_week: int,
) -> dict:
    """
    Aggregate developer feedback weekly and monthly for LLM input.
    Implements rate-limiting strategy: weekly → monthly summaries.
    """
    summaries = {}

    # Build weekly summaries (feedback aggregated by week)
    for month in sorted(hist_df["month"].unique()):
        if month > current_month:
            break
        for week in sorted(hist_df[hist_df["month"] == month]["week"].unique()):
            if month == current_month and week >= current_week:
                break
            week_data = hist_df[
                (hist_df["month"] == month) & (hist_df["week"] == week)
            ]
            if week_data.empty:
                continue

            top_bugs = week_data.nlargest(3, "bugs_found")
            feedback_items = top_bugs["developer_feedback"].tolist()
            total_bugs = int(week_data["bugs_found"].sum())
            avg_coverage = round(week_data["coverage_percent"].mean(), 1)
            high_churn_modules = week_data.nlargest(2, "loc_changed")["module_name"].tolist()

            key = f"month_{month}_week_{week}"
            summaries[key] = {
                "period": f"Month {month}, Week {week}",
                "total_bugs": total_bugs,
                "avg_coverage": avg_coverage,
                "high_churn_modules": high_churn_modules,
                "top_feedback": feedback_items,
            }

    # Build monthly summaries
    for month in sorted(hist_df["month"].unique()):
        if month >= current_month:
            break
        month_data = hist_df[hist_df["month"] == month]
        if month_data.empty:
            continue

        total_bugs = int(month_data["bugs_found"].sum())
        avg_coverage = round(month_data["coverage_percent"].mean(), 1)
        worst_modules = month_data.groupby("module_name")["bugs_found"].sum().nlargest(3)
        feedback = month_data.nlargest(5, "bugs_found")["developer_feedback"].tolist()

        summaries[f"month_{month}_summary"] = {
            "period": f"Month {month} Summary",
            "total_bugs": total_bugs,
            "avg_coverage": avg_coverage,
            "worst_modules": worst_modules.to_dict(),
            "top_feedback": feedback,
        }

    return summaries


def _default_prediction() -> dict:
    return {
        "predicted_bugs": 3,
        "bugs_found_so_far": 0,
        "confidence_interval": (1, 5),
        "trend": "STABLE",
        "weekly_forecast": [1, 1],
        "module_breakdown": {m: 0 for m in MODULES},
        "weeks_used": 0,
        "context_summary": {"complete_months": 0, "partial_weeks": 0},
    }


def format_prediction_report(pred: dict) -> str:
    """Format bug prediction results for display."""
    ci_low, ci_high = pred["confidence_interval"]
    lines = [
        "📈 BUG TREND PREDICTION",
        "=" * 50,
        f"  Predicted Milestone Bugs:  {pred['predicted_bugs']}",
        f"  Confidence Interval:       {ci_low} – {ci_high}",
        f"  Bugs Found So Far:         {pred['bugs_found_so_far']}",
        f"  Trend:                     {pred['trend']}",
        f"  Data Points Used:          {pred['weeks_used']} weeks",
        "",
        "  Module Breakdown:",
    ]
    for module, count in sorted(pred["module_breakdown"].items(), key=lambda x: -x[1]):
        bar = "█" * max(1, int(count))
        lines.append(f"    {module:15s} {count:4.1f}  {bar}")
    lines.append("=" * 50)
    return "\n".join(lines)


if __name__ == "__main__":
    hist_path = os.path.join(os.path.dirname(__file__), "../data/historical_verification_data.csv")
    trend_path = os.path.join(os.path.dirname(__file__), "../data/bug_trend_data.csv")

    hist_df = pd.read_csv(hist_path)
    trend_df = pd.read_csv(trend_path)

    pred = predict_milestone_bugs(trend_df, hist_df, current_month=3, current_week=3)
    print(format_prediction_report(pred))

    print("\n📋 Weekly Summaries for LLM:")
    summaries = generate_weekly_summaries(hist_df, current_month=3, current_week=3)
    for k, v in list(summaries.items())[:3]:
        print(f"  [{k}] Bugs: {v['total_bugs']}, Coverage: {v['avg_coverage']}%")
