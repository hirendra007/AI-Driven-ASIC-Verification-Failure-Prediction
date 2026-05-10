"""
End-to-End AI Verification Pipeline
Orchestrates: Commit Parse → Feature Engineering → XGBoost Risk → Regression → Bug Trend → LLM Explain
"""

import os
import sys
import json
import argparse
from datetime import datetime
from typing import Optional
import pandas as pd

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "regression"))

from src.commit_parser import CommitParser, format_commit_summary
from src.feature_engineering import compute_features, get_feature_columns
from src.train_model import load_model, predict_risk, train_model
from src.bug_trend import predict_milestone_bugs, generate_weekly_summaries, format_prediction_report
from src.llm_explainer import LLMExplainer
from regression.orchestrator import RegressionOrchestrator, VerilatorSimulator


DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")


class VerificationPipeline:
    """Full verification pipeline from git commit to risk report."""

    def __init__(
        self,
        llm_provider: str = "mock",
        llm_api_key: Optional[str] = None,
        simulation_mode: bool = True,
    ):
        # Load data
        self.hist_df = pd.read_csv(os.path.join(DATA_DIR, "historical_verification_data.csv"))
        self.trend_df = pd.read_csv(os.path.join(DATA_DIR, "bug_trend_data.csv"))
        self.commit_df = pd.read_csv(os.path.join(DATA_DIR, "git_commits.csv"))

        # Initialize components
        self.commit_parser = CommitParser(
            commit_csv_path=os.path.join(DATA_DIR, "git_commits.csv")
        )
        self.orchestrator = RegressionOrchestrator(
            simulator=VerilatorSimulator(simulation_mode=simulation_mode),
            dep_csv_path=os.path.join(DATA_DIR, "module_dependencies.csv"),
        )
        self.explainer = LLMExplainer(provider=llm_provider, api_key=llm_api_key)

        # Load or train model
        try:
            self.pipeline, self.feature_cols, self.model_meta = load_model()
            print(f"✅ Loaded XGBoost model (CV AUC: {self.model_meta.get('cv_roc_auc_mean', 'N/A'):.3f})")
        except FileNotFoundError:
            print("⚠  No trained model found. Training now...")
            self.pipeline, self.feature_cols, self.model_meta = train_model(self.hist_df)

    def run(self, commit_id: str, verbose: bool = True) -> dict:
        """
        Execute the full pipeline for a given commit.

        Returns:
            Complete pipeline result dictionary.
        """
        print(f"\n{'='*65}")
        print(f"🚀 AI VERIFICATION PIPELINE — Commit: {commit_id}")
        print(f"{'='*65}")

        # ── Step 1: Parse Commit ────────────────────────────────────────────
        print("\n[1/7] 📝 Parsing commit...")
        commit_info = self.commit_parser.parse_commit(commit_id)
        if verbose:
            print(format_commit_summary(commit_info))
        changed_modules = commit_info["changed_modules"]

        # ── Step 2: Feature Engineering ────────────────────────────────────
        print("\n[2/7] 🔧 Computing feature vectors...")
        feature_df = compute_features(self.hist_df)
        feature_df_changed = feature_df[feature_df["module_name"].isin(changed_modules)]

        # Inject commit-specific churn data
        commit_rows = self.commit_df[self.commit_df["commit_id"] == commit_id]
        churn_by_module = commit_rows.groupby("module_name").apply(
            lambda g: g["loc_added"].sum() + g["loc_deleted"].sum()
        )
        for idx, row in feature_df_changed.iterrows():
            m = row["module_name"]
            if m in churn_by_module.index:
                feature_df.loc[idx, "code_churn"] = churn_by_module[m]

        if verbose:
            print(f"   Features computed for: {changed_modules}")

        # ── Step 3: XGBoost Risk Prediction ────────────────────────────────
        print("\n[3/7] 🤖 Running XGBoost risk prediction...")
        risk_df = predict_risk(self.pipeline, feature_df_changed, self.feature_cols)

        print("\n   ┌─────────────────────────────────────────┐")
        print(   "   │  Module Risk Scores                      │")
        print(   "   ├──────────────┬────────────┬─────────────┤")
        print(   "   │ Module       │ Risk Score │ Risk Level  │")
        print(   "   ├──────────────┼────────────┼─────────────┤")
        for _, row in risk_df.iterrows():
            icon = "🔴" if row["risk_level"] == "HIGH" else ("🟡" if row["risk_level"] == "MEDIUM" else "🟢")
            print(f"   │ {row['module_name']:12s} │   {row['risk_score']:.2f}     │ {icon} {row['risk_level']:8s} │")
        print(   "   └──────────────┴────────────┴─────────────┘")

        # ── Step 4: Regression Execution ───────────────────────────────────
        print("\n[4/7] 🔬 Executing selective regression...")
        reg_report = self.orchestrator.run(
            commit_id=commit_id,
            changed_modules=changed_modules,
            risk_df=risk_df,
        )
        print(self.orchestrator.summarize(reg_report))

        # ── Step 5: Bug Trend Prediction ────────────────────────────────────
        print("\n[5/7] 📈 Predicting bug trends...")
        # Infer current milestone context
        latest_commit_ts = pd.to_datetime(commit_rows["timestamp"].iloc[0])
        current_month = min(6, max(1, (latest_commit_ts.month - 7) % 6 + 1))
        current_week = min(4, max(1, (latest_commit_ts.day - 1) // 7 + 1))

        bug_pred = predict_milestone_bugs(
            self.trend_df, self.hist_df, current_month, current_week
        )
        if verbose:
            print(format_prediction_report(bug_pred))

        # ── Step 6: LLM Risk Explanations ──────────────────────────────────
        print("\n[6/7] 💬 Generating LLM risk explanations...")
        summaries = generate_weekly_summaries(self.hist_df, current_month, current_week)
        reg_results = {r.module: r.__dict__ for r in reg_report.results}

        explanations = {}
        for _, row in risk_df.iterrows():
            module = row["module_name"]
            module_features = feature_df[feature_df["module_name"] == module]
            if module_features.empty:
                continue
            feat_dict = module_features.iloc[0].to_dict()
            module_pred = bug_pred["module_breakdown"].get(module, 0)
            ci = (
                max(0, int(module_pred * 0.7)),
                int(module_pred * 1.3) + 1
            )

            explanation = self.explainer.explain_module_risk(
                module=module,
                risk_score=float(row["risk_score"]),
                risk_level=row["risk_level"],
                features=feat_dict,
                bug_prediction=round(module_pred),
                ci=ci,
                summaries=summaries,
                regression_result=reg_results.get(module),
            )
            explanations[module] = explanation

            if verbose:
                print(f"\n   📌 {module} ({row['risk_level']}):")
                print(f"   {explanation}")

        # ── Step 7: Project Summary ─────────────────────────────────────────
        print("\n[7/7] 📋 Generating project health summary...")
        project_summary = self.explainer.generate_project_summary(risk_df, bug_pred, summaries)
        if verbose:
            print(f"\n   {project_summary}")

        # ── Build Result ────────────────────────────────────────────────────
        result = {
            "commit_id": commit_id,
            "timestamp": datetime.now().isoformat(),
            "changed_modules": changed_modules,
            "risk_predictions": risk_df.to_dict("records"),
            "regression_report": reg_report.to_dict(),
            "bug_prediction": bug_pred,
            "llm_explanations": explanations,
            "project_summary": project_summary,
        }

        # Save result
        output_path = os.path.join(PROJECT_ROOT, "outputs", f"report_{commit_id}.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n✅ Full report saved → {output_path}")

        return result


def main():
    parser = argparse.ArgumentParser(description="AI-Driven ASIC Verification Failure Prediction")
    parser.add_argument("--commit", type=str, help="Commit ID to analyze", default=None)
    parser.add_argument("--train", action="store_true", help="Re-train the XGBoost model")
    parser.add_argument("--generate-data", action="store_true", help="Regenerate synthetic datasets")
    parser.add_argument("--llm", type=str, default="mock", choices=["gemini", "openai", "claude", "mock"],
                        help="LLM provider for risk explanations")
    parser.add_argument("--api-key", type=str, default=None, help="API key for LLM provider")
    args = parser.parse_args()

    # Generate data if requested
    if args.generate_data:
        from data.generate_datasets import (
            generate_historical_data, generate_bug_trend_data,
            generate_git_commits, generate_module_dependency_map
        )
        os.chdir(PROJECT_ROOT)
        generate_historical_data(n_weeks=26)
        generate_bug_trend_data()
        generate_git_commits(n_commits=400)
        generate_module_dependency_map()
        print("✅ Data generation complete!")
        return

    # Train if requested
    if args.train:
        hist_df = pd.read_csv(os.path.join(DATA_DIR, "historical_verification_data.csv"))
        train_model(hist_df)
        print("✅ Model training complete!")
        if not args.commit:
            return

    # Run pipeline on a commit
    pipeline = VerificationPipeline(
        llm_provider=args.llm,
        llm_api_key=args.api_key,
    )

    # Use provided commit or pick a recent one
    if args.commit:
        commit_id = args.commit
    else:
        commits = pipeline.commit_parser.get_recent_commits(5)
        if commits.empty:
            print("❌ No commits found. Run with --generate-data first.")
            return
        commit_id = commits["commit_id"].iloc[0]
        print(f"ℹ  No commit specified. Using most recent: {commit_id}")

    result = pipeline.run(commit_id)
    print(f"\n{'='*65}")
    print("✅ PIPELINE COMPLETE")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
