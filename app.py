"""
AI-Driven ASIC Verification Dashboard
Streamlit app for visualizing module risk, bug trends, coverage, and commit impact.
"""

import os
import sys
import json
import glob
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "regression"))

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Verification Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .risk-high { color: #ef4444; font-weight: bold; }
    .risk-med  { color: #f59e0b; font-weight: bold; }
    .risk-low  { color: #22c55e; font-weight: bold; }
    .metric-card {
        background: #1e293b; border-radius: 8px; padding: 16px;
        border: 1px solid #334155; margin: 4px;
    }
    h1, h2, h3 { color: #f1f5f9; }
    .stTabs [data-baseweb="tab"] { color: #94a3b8; }
    .stTabs [aria-selected="true"] { color: #3b82f6; border-bottom-color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RISK_COLORS = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#22c55e"}
MODULES = ["ALU", "Decoder", "Cache", "DMA", "AXI", "FIFO", "BranchUnit", "ControlUnit"]


@st.cache_data
def load_data():
    hist = pd.read_csv(os.path.join(DATA_DIR, "historical_verification_data.csv"))
    trend = pd.read_csv(os.path.join(DATA_DIR, "bug_trend_data.csv"))
    commits = pd.read_csv(os.path.join(DATA_DIR, "git_commits.csv"))
    return hist, trend, commits


@st.cache_data
def load_latest_report():
    reports = sorted(glob.glob(os.path.join(OUTPUTS_DIR, "report_*.json")))
    if not reports:
        return None
    with open(reports[-1]) as f:
        return json.load(f)


def run_pipeline_cached(commit_id: str, llm_provider: str):
    """Run the verification pipeline and return results."""
    from src.feature_engineering import compute_features, get_feature_columns
    from src.train_model import load_model, predict_risk
    from src.bug_trend import predict_milestone_bugs, generate_weekly_summaries
    from src.llm_explainer import LLMExplainer
    from src.commit_parser import CommitParser
    from regression.orchestrator import RegressionOrchestrator, VerilatorSimulator

    hist_df, trend_df, commit_df = load_data()

    parser = CommitParser(commit_csv_path=os.path.join(DATA_DIR, "git_commits.csv"))
    commit_info = parser.parse_commit(commit_id)
    changed_modules = commit_info["changed_modules"]

    feature_df = compute_features(hist_df)
    feature_df_changed = feature_df[feature_df["module_name"].isin(changed_modules)]

    try:
        pipeline, feature_cols, meta = load_model()
    except:
        from src.train_model import train_model
        pipeline, feature_cols, meta = train_model(hist_df, save=True)

    risk_df = predict_risk(pipeline, feature_df_changed, feature_cols)

    orchestrator = RegressionOrchestrator(
        simulator=VerilatorSimulator(simulation_mode=True),
        dep_csv_path=os.path.join(DATA_DIR, "module_dependencies.csv"),
    )
    reg_report = orchestrator.run(commit_id, changed_modules, risk_df)

    commit_rows = commit_df[commit_df["commit_id"] == commit_id]
    latest_ts = pd.to_datetime(commit_rows["timestamp"].iloc[0])
    current_month = min(6, max(1, (latest_ts.month - 7) % 6 + 1))
    current_week = min(4, max(1, (latest_ts.day - 1) // 7 + 1))

    bug_pred = predict_milestone_bugs(trend_df, hist_df, current_month, current_week)
    summaries = generate_weekly_summaries(hist_df, current_month, current_week)

    explainer = LLMExplainer(provider=llm_provider)
    explanations = {}
    for _, row in risk_df.iterrows():
        module = row["module_name"]
        feat_row = feature_df[feature_df["module_name"] == module]
        if feat_row.empty:
            continue
        feat_dict = feat_row.iloc[0].to_dict()
        module_pred = bug_pred["module_breakdown"].get(module, 0)
        ci = (max(0, int(module_pred * 0.7)), int(module_pred * 1.3) + 1)
        reg_result = next((r.__dict__ for r in reg_report.results if r.module == module), None)
        explanations[module] = explainer.explain_module_risk(
            module, float(row["risk_score"]), row["risk_level"],
            feat_dict, round(module_pred), ci, summaries, reg_result
        )

    project_summary = explainer.generate_project_summary(risk_df, bug_pred, summaries)

    return {
        "commit_info": commit_info,
        "risk_df": risk_df,
        "reg_report": reg_report,
        "bug_pred": bug_pred,
        "explanations": explanations,
        "project_summary": project_summary,
        "feature_df": feature_df,
    }


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/circuit.png", width=60)
    st.title("🔬 AI Verification")
    st.caption("ASIC Failure Prediction System")
    st.divider()

    try:
        hist_df, trend_df, commit_df = load_data()
        data_loaded = True
    except:
        data_loaded = False
        st.error("⚠ Data not found. Run setup first.")

    if data_loaded:
        st.subheader("⚙ Run Analysis")
        recent_commits = commit_df.sort_values("timestamp", ascending=False).drop_duplicates("commit_id").head(20)
        commit_options = recent_commits["commit_id"].tolist()
        selected_commit = st.selectbox("Select Commit", commit_options,
                                        format_func=lambda c: f"{c} ({recent_commits[recent_commits['commit_id']==c]['timestamp'].iloc[0][:10]})")

        llm_provider = st.selectbox("LLM Provider", ["mock", "gemini", "openai", "claude"],
                                     help="'mock' uses rule-based explanations (no API key needed)")

        run_btn = st.button("🚀 Run Pipeline", type="primary", use_container_width=True)

        st.divider()
        st.subheader("📊 View Mode")
        view_mode = st.radio("", ["Live Results", "Historical Analysis"], label_visibility="collapsed")

        if os.path.exists(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "model_metadata.json")):
            with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
                meta = json.load(f)
            st.divider()
            st.caption("Model Info")
            st.metric("CV ROC-AUC", f"{meta.get('cv_roc_auc_mean', 0):.3f}")
            st.metric("Training Samples", meta.get('n_training_samples', 'N/A'))


# ── Main Content ──────────────────────────────────────────────────────────────
st.title("🔬 AI-Driven ASIC Verification Dashboard")
st.caption("Predicts high-risk RTL modules and bug trends using ML on historical verification data")

if not data_loaded:
    st.warning("Please generate datasets first: `python pipeline.py --generate-data && python pipeline.py --train`")
    st.stop()

# ── Run Pipeline ──────────────────────────────────────────────────────────────
results = None
if 'run_btn' in dir() and run_btn:
    with st.spinner(f"⚙ Running pipeline for commit {selected_commit}..."):
        try:
            results = run_pipeline_cached(selected_commit, llm_provider)
            st.session_state["last_results"] = results
            st.success(f"✅ Pipeline complete for commit: {selected_commit}")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            import traceback; st.code(traceback.format_exc())

if "last_results" in st.session_state:
    results = st.session_state["last_results"]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Risk Overview", "📈 Bug Trends", "📡 Coverage", "💻 Commit Impact", "💬 LLM Explanations"
])

# ── Tab 1: Risk Overview ──────────────────────────────────────────────────────
with tab1:
    if results:
        risk_df = results["risk_df"]
        commit_info = results["commit_info"]
        reg_report = results["reg_report"]

        # KPI row
        col1, col2, col3, col4 = st.columns(4)
        high_count = len(risk_df[risk_df["risk_level"] == "HIGH"])
        med_count = len(risk_df[risk_df["risk_level"] == "MEDIUM"])
        executed = len(reg_report.executed_modules)
        saved = len(reg_report.skipped_modules)
        efficiency = round(saved / max(1, executed + saved) * 100)

        col1.metric("🔴 High Risk Modules", high_count)
        col2.metric("🟡 Medium Risk Modules", med_count)
        col3.metric("⚡ Regressions Run", executed)
        col4.metric("⏭ Simulations Saved", f"{efficiency}%")

        st.divider()
        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.subheader("Module Risk Heatmap")
            heatmap_data = risk_df.pivot_table(
                values="risk_score", index="module_name", columns=["risk_level"],
                aggfunc="mean", fill_value=0
            )
            # Simple bar chart as heatmap alternative
            fig = px.bar(
                risk_df.sort_values("risk_score", ascending=True),
                x="risk_score", y="module_name", orientation="h",
                color="risk_level",
                color_discrete_map=RISK_COLORS,
                text="risk_score",
                title="Module Risk Scores",
                labels={"risk_score": "Risk Score (0–1)", "module_name": "Module"},
            )
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            fig.update_layout(
                plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
                font_color="#f1f5f9", showlegend=True,
                xaxis=dict(range=[0, 1.1])
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Regression Results")
            reg_data = []
            for r in reg_report.results:
                reg_data.append({
                    "Module": r.module,
                    "Coverage %": r.coverage_percent,
                    "Failures": r.fail_count,
                    "Runtime (s)": r.simulation_runtime,
                    "Status": r.status,
                })
            reg_df_display = pd.DataFrame(reg_data)

            def color_status(val):
                if val == "PASS": return "color: #22c55e; font-weight: bold"
                if val == "FAIL": return "color: #ef4444; font-weight: bold"
                return ""

            if not reg_df_display.empty:
                st.dataframe(
                    reg_df_display.style.applymap(color_status, subset=["Status"]),
                    use_container_width=True, hide_index=True,
                )

            # Execution summary
            st.info(
                f"✅ Executed: {', '.join(reg_report.executed_modules)}\n\n"
                f"⏭ Skipped: {', '.join(reg_report.skipped_modules) or 'None'}\n\n"
                f"🔄 Fallback: {'Triggered' if reg_report.fallback_triggered else 'Not needed'}"
            )

    else:
        # Show historical risk preview
        st.info("👈 Select a commit and click 'Run Pipeline' to see live risk analysis")
        hist_df, _, _ = load_data()
        from src.feature_engineering import compute_features
        features = compute_features(hist_df)
        if not features.empty:
            st.subheader("📊 Historical Bug Density by Module")
            agg = hist_df.groupby("module_name")["bugs_found"].mean().reset_index()
            agg.columns = ["module_name", "avg_bugs_per_week"]
            fig = px.bar(agg.sort_values("avg_bugs_per_week", ascending=False),
                         x="module_name", y="avg_bugs_per_week",
                         color="avg_bugs_per_week", color_continuous_scale="RdYlGn_r",
                         title="Average Bugs Per Week by Module")
            fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
            st.plotly_chart(fig, use_container_width=True)

# ── Tab 2: Bug Trends ─────────────────────────────────────────────────────────
with tab2:
    hist_df, trend_df, _ = load_data()
    st.subheader("📈 Bug Trend Analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Weekly bug trend
        trend_fig = px.line(
            trend_df, x="week_offset", y="total_bugs",
            title="Weekly Bug Count Over Project Timeline",
            markers=True, labels={"week_offset": "Week", "total_bugs": "Bugs Found"},
            color_discrete_sequence=["#3b82f6"]
        )
        trend_fig.add_scatter(
            x=trend_df["week_offset"], y=trend_df["cumulative_bugs"],
            name="Cumulative", line=dict(color="#f59e0b", dash="dash"), mode="lines"
        )
        trend_fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(trend_fig, use_container_width=True)

    with col2:
        if results:
            bug_pred = results["bug_pred"]
            ci_low, ci_high = bug_pred["confidence_interval"]
            st.metric("🔮 Predicted Milestone Bugs", bug_pred["predicted_bugs"],
                      delta=f"CI: {ci_low}–{ci_high}")
            st.metric("📊 Trend Direction", bug_pred["trend"])
            st.metric("✅ Bugs Found So Far", bug_pred["bugs_found_so_far"])

            # Module breakdown
            st.subheader("Module Bug Forecast")
            breakdown = bug_pred.get("module_breakdown", {})
            if breakdown:
                bd_df = pd.DataFrame(list(breakdown.items()), columns=["Module", "Predicted Bugs"])
                bd_df = bd_df.sort_values("Predicted Bugs", ascending=False)
                fig_bd = px.bar(bd_df, x="Module", y="Predicted Bugs",
                                color="Predicted Bugs", color_continuous_scale="RdYlGn_r")
                fig_bd.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
                st.plotly_chart(fig_bd, use_container_width=True)
        else:
            # Monthly aggregate
            monthly = trend_df.groupby("month")["total_bugs"].sum().reset_index()
            fig_m = px.bar(monthly, x="month", y="total_bugs", title="Bugs by Milestone Month",
                           color="total_bugs", color_continuous_scale="RdYlGn_r")
            fig_m.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
            st.plotly_chart(fig_m, use_container_width=True)

# ── Tab 3: Coverage ───────────────────────────────────────────────────────────
with tab3:
    hist_df, _, _ = load_data()
    st.subheader("📡 Coverage Trends by Module")

    selected_modules = st.multiselect("Select Modules", MODULES, default=MODULES[:4])

    if selected_modules:
        filtered = hist_df[hist_df["module_name"].isin(selected_modules)].copy()
        filtered["period"] = filtered["month"].astype(str) + "-W" + filtered["week"].astype(str)

        cov_fig = px.line(
            filtered, x="period", y="coverage_percent", color="module_name",
            title="Coverage % Over Time by Module",
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        cov_fig.add_hline(y=80, line_dash="dash", line_color="#f59e0b",
                           annotation_text="80% Target", annotation_position="right")
        cov_fig.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(cov_fig, use_container_width=True)

        # Coverage heatmap by month
        st.subheader("Coverage Heatmap (Module × Month)")
        pivot = hist_df.groupby(["module_name", "month"])["coverage_percent"].mean().unstack()
        fig_heat = px.imshow(
            pivot, color_continuous_scale="RdYlGn",
            title="Average Coverage % (Module × Month)",
            labels=dict(x="Month", y="Module", color="Coverage %"),
            zmin=60, zmax=99,
        )
        fig_heat.update_layout(paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig_heat, use_container_width=True)

# ── Tab 4: Commit Impact ──────────────────────────────────────────────────────
with tab4:
    hist_df, _, commit_df = load_data()
    st.subheader("💻 Commit Impact Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Code churn vs bugs
        churn_bugs = hist_df.groupby("module_name").agg(
            total_churn=("loc_changed", "sum"),
            total_bugs=("bugs_found", "sum"),
            avg_coverage=("coverage_percent", "mean"),
        ).reset_index()

        fig_scatter = px.scatter(
            churn_bugs, x="total_churn", y="total_bugs",
            size="avg_coverage", color="module_name",
            title="Code Churn vs Bugs Found (size = avg coverage)",
            labels={"total_churn": "Total LOC Changed", "total_bugs": "Total Bugs"},
            text="module_name",
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        if results:
            commit_info = results["commit_info"]
            risk_df = results["risk_df"]

            # Churn breakdown for this commit
            churn_data = [
                {"Module": d["module"], "LOC Added": d["loc_added"],
                 "LOC Deleted": d["loc_deleted"], "Total Churn": d["code_churn"]}
                for d in commit_info["module_details"]
            ]
            if churn_data:
                churn_df = pd.DataFrame(churn_data)
                fig_churn = px.bar(
                    churn_df, x="Module", y=["LOC Added", "LOC Deleted"],
                    title=f"Commit {commit_info['commit_id']} Churn",
                    barmode="group",
                    color_discrete_map={"LOC Added": "#22c55e", "LOC Deleted": "#ef4444"}
                )
                fig_churn.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
                st.plotly_chart(fig_churn, use_container_width=True)
        else:
            # Commits per module over time
            commit_agg = commit_df.groupby(["module_name", "timestamp"]).agg(
                total_churn=("loc_added", lambda x: x.sum() + commit_df.loc[x.index, "loc_deleted"].sum())
            ).reset_index()
            monthly_churn = commit_df.groupby("module_name").agg(
                avg_churn=("loc_added", "mean")
            ).reset_index()
            fig_cm = px.bar(monthly_churn.sort_values("avg_churn", ascending=False),
                            x="module_name", y="avg_churn",
                            title="Average LOC Added per Commit by Module",
                            color="avg_churn", color_continuous_scale="Blues")
            fig_cm.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
            st.plotly_chart(fig_cm, use_container_width=True)

    # Predicted vs actual (verification efficiency)
    st.subheader("🎯 Verification Efficiency")
    efficiency_data = hist_df.groupby("module_name").agg(
        actual_bugs=("bugs_found", "mean"),
        avg_runtime=("regression_runtime", "mean"),
        commits=("commits", "sum"),
    ).reset_index()
    efficiency_data["bugs_per_runtime_hour"] = efficiency_data["actual_bugs"] / (efficiency_data["avg_runtime"] / 60)

    fig_eff = px.bar(
        efficiency_data.sort_values("bugs_per_runtime_hour", ascending=False),
        x="module_name", y="bugs_per_runtime_hour",
        title="Bug Discovery Efficiency (Bugs Found per Runtime Hour)",
        color="bugs_per_runtime_hour", color_continuous_scale="RdYlGn_r",
    )
    fig_eff.update_layout(plot_bgcolor="#1e293b", paper_bgcolor="#0f172a", font_color="#f1f5f9")
    st.plotly_chart(fig_eff, use_container_width=True)

# ── Tab 5: LLM Explanations ───────────────────────────────────────────────────
with tab5:
    if results:
        st.subheader("💬 AI Risk Explanations")
        st.info(f"🤖 LLM Provider: `{llm_provider}` | Commit: `{results['commit_info']['commit_id']}`")

        # Project summary
        with st.expander("📋 Project Health Summary", expanded=True):
            st.write(results["project_summary"])

        st.divider()

        # Per-module explanations
        risk_df = results["risk_df"]
        for _, row in risk_df.iterrows():
            module = row["module_name"]
            if module not in results["explanations"]:
                continue

            level = row["risk_level"]
            icon = "🔴" if level == "HIGH" else ("🟡" if level == "MEDIUM" else "🟢")
            color = RISK_COLORS[level]

            with st.expander(f"{icon} {module} — {level} RISK (score: {row['risk_score']:.2f})", expanded=(level == "HIGH")):
                st.markdown(f"""
                <div style="border-left: 4px solid {color}; padding-left: 12px; color: #e2e8f0;">
                {results["explanations"][module]}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("👈 Run the pipeline to generate LLM risk explanations")
        st.markdown("""
        ### How the LLM Explanation Works

        The system aggregates verification history into weekly and monthly summaries,
        then sends them to the LLM with module-specific metrics to generate:

        - **Why** the module is high risk (code churn, coverage decay, bug density)
        - **What** types of RTL bugs are likely (overflow, FSM deadlock, AXI violations)
        - **How many** bugs to expect in the milestone
        - **Where** to focus verification effort

        Rate limiting is handled by using pre-aggregated summaries instead of raw data.
        """)
