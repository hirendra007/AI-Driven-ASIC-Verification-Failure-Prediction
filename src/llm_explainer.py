"""
LLM Risk Explanation Generator
Uses Google Gemini or OpenAI GPT to generate natural-language risk reports.
Implements rate-limiting via weekly/monthly summary aggregation.
"""

import os
import json
import time
from typing import Optional
import pandas as pd

import google.generativeai as genai  # <-- Add this here
# ── Prompt Templates ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior ASIC/SoC verification engineer with deep expertise in RTL design.
You analyze verification data and explain risks in precise technical terms that hardware engineers understand.
Your explanations are concise, actionable, and reference specific RTL failure modes.
Always cite specific bug types (e.g., arithmetic overflow, FSM deadlock, AXI handshake violation).
Keep responses under 200 words per module."""


def build_module_risk_prompt(
    module: str,
    risk_score: float,
    risk_level: str,
    features: dict,
    bug_prediction: int,
    ci_low: int,
    ci_high: int,
    historical_summaries: dict,
    regression_result: Optional[dict] = None,
) -> str:
    """
    Build the LLM prompt for a single module risk explanation.

    Args are aggregated summaries (not raw data) to minimize API calls.
    """
    # Serialize only the most recent summaries to avoid token bloat
    recent_summaries = dict(list(historical_summaries.items())[-6:])
    summary_text = json.dumps(recent_summaries, indent=2)

    reg_text = ""
    if regression_result:
        reg_text = f"""
Regression Results:
- Coverage: {regression_result.get('coverage_percent', 'N/A')}%
- Failing Tests: {regression_result.get('fail_count', 0)}
- Status: {regression_result.get('status', 'N/A')}
"""

    prompt = f"""
ASIC Module Risk Analysis Request

Module: {module}
Risk Score: {risk_score:.2f} / 1.00
Risk Level: {risk_level}

Current Metrics:
- Code Churn: {features.get('code_churn', 0):.0f} LOC changed
- Historical Bug Density: {features.get('historical_bug_density', 0):.3f} bugs/commit
- Coverage Trend: {features.get('coverage_trend', 0):+.2f}% per week
- Module Instability (variance): {features.get('module_instability', 0):.2f}
- Recent Bug Rate: {features.get('recent_bug_rate', 0):.2f} bugs/week
- Average Coverage: {features.get('avg_coverage', 85):.1f}%
{reg_text}
Historical Summary (last 6 weeks/months):
{summary_text}

Bug Prediction for Current Milestone:
- Expected bugs: {bug_prediction} (range: {ci_low}–{ci_high})

Generate a technical risk explanation that covers:
1. Why this module is {risk_level} risk (cite specific metrics)
2. What types of RTL bugs are most likely (be specific — e.g., "arithmetic overflow in the multiply-accumulate path")
3. Which areas of the module to prioritize in verification
4. Predicted bug count with brief justification

Format as a single paragraph, 3-4 sentences maximum.
"""
    return prompt.strip()


def build_project_summary_prompt(
    risk_df: pd.DataFrame,
    bug_prediction: dict,
    historical_summaries: dict,
) -> str:
    """Build prompt for overall project health summary."""
    high_risk = risk_df[risk_df["risk_level"] == "HIGH"]["module_name"].tolist()
    med_risk = risk_df[risk_df["risk_level"] == "MEDIUM"]["module_name"].tolist()

    pred_bugs = bug_prediction.get("predicted_bugs", "N/A")
    ci = bug_prediction.get("confidence_interval", (0, 0))
    trend = bug_prediction.get("trend", "STABLE")

    month_summaries = {k: v for k, v in historical_summaries.items() if "summary" in k}
    recent_months = dict(list(month_summaries.items())[-2:])
    partial = {k: v for k, v in historical_summaries.items() if "summary" not in k}
    recent_partial = dict(list(partial.items())[-3:])

    context = {**recent_months, **recent_partial}

    prompt = f"""
ASIC Project Verification Health Report

Risk Summary:
- HIGH RISK modules: {', '.join(high_risk) or 'None'}
- MEDIUM RISK modules: {', '.join(med_risk) or 'None'}
- LOW RISK modules: {', '.join(risk_df[risk_df['risk_level'] == 'LOW']['module_name'].tolist()) or 'None'}

Bug Trend Prediction:
- Expected bugs this milestone: {pred_bugs} (range: {ci[0]}–{ci[1]})
- Trend: {trend}

Historical Context (monthly summaries + recent weeks):
{json.dumps(context, indent=2)}

Generate a 4-5 sentence project health summary for a verification manager that:
1. Highlights the most critical risk areas and why they're risky
2. Summarizes the expected bug trajectory for this milestone
3. Recommends 2-3 specific verification actions to reduce risk
4. Provides confidence in the current test coverage strategy
"""
    return prompt.strip()

class LLMExplainer:
    """
    LLM-powered risk explanation generator.
    Supports: Google Gemini, OpenAI GPT, Anthropic Claude.
    Falls back to rule-based templates if no API key is configured.
    """

    def __init__(self, provider: str = "gemini", api_key: Optional[str] = None):
        """
        Args:
            provider: "gemini" | "openai" | "claude" | "mock"
            api_key: API key (reads from env if not provided)
        """
        self.provider = provider
        self.api_key = api_key or os.getenv(
            {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "claude": "ANTHROPIC_API_KEY"}.get(provider, "")
        )
        self._call_count = 0
        self._last_call_time = 0

    def _rate_limit(self, min_interval: float = 1.0):
        """Enforce minimum interval between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()
        self._call_count += 1

    def explain_module_risk(
        self,
        module: str,
        risk_score: float,
        risk_level: str,
        features: dict,
        bug_prediction: int,
        ci: tuple,
        summaries: dict,
        regression_result: Optional[dict] = None,
    ) -> str:
        """Generate risk explanation for a single module."""
        prompt = build_module_risk_prompt(
            module=module,
            risk_score=risk_score,
            risk_level=risk_level,
            features=features,
            bug_prediction=bug_prediction,
            ci_low=ci[0],
            ci_high=ci[1],
            historical_summaries=summaries,
            regression_result=regression_result,
        )

        if not self.api_key or self.provider == "mock":
            print(f" ⚙️  [INFO] Using rule-based fallback for {module} (No API key found or provider='mock')")
            return "[⚙️ Rule-Based Fallback] " + self._rule_based_explanation(module, risk_score, risk_level, features, bug_prediction, ci)

        self._rate_limit()
        try:
            if self.provider == "gemini":
                res = self._call_gemini(prompt)
                print(f" ✅ [SUCCESS] Gemini successfully generated explanation for {module}")
                return res
            elif self.provider == "openai":
                res = self._call_openai(prompt)
                print(f" ✅ [SUCCESS] OpenAI successfully generated explanation for {module}")
                return "[🤖 Powered by OpenAI] " + res
            elif self.provider == "claude":
                res = self._call_claude(prompt)
                print(f" ✅ [SUCCESS] Claude successfully generated explanation for {module}")
                return "[🤖 Powered by Claude] " + res
        except Exception as e:
            print(f" ⚠ [ERROR] LLM API error: {e}. Falling back to rule-based explanation.")
            return "[⚙️ Rule-Based Fallback] " + self._rule_based_explanation(module, risk_score, risk_level, features, bug_prediction, ci)

    def generate_project_summary(
        self,
        risk_df: pd.DataFrame,
        bug_prediction: dict,
        summaries: dict,
    ) -> str:
        """Generate overall project health summary."""
        prompt = build_project_summary_prompt(risk_df, bug_prediction, summaries)

        if not self.api_key or self.provider == "mock":
            print(f" ⚙️  [INFO] Using rule-based fallback for Project Summary")
            return "[⚙️ Rule-Based Fallback] " + self._rule_based_project_summary(risk_df, bug_prediction)

        self._rate_limit()
        try:
            if self.provider == "gemini":
                res = self._call_gemini(prompt)
                print(f" ✅ [SUCCESS] Gemini successfully generated Project Summary")
                return   res
            elif self.provider == "openai":
                res = self._call_openai(prompt)
                print(f" ✅ [SUCCESS] OpenAI successfully generated Project Summary")
                return "[🤖 Powered by OpenAI] " + res
            elif self.provider == "claude":
                res = self._call_claude(prompt)
                print(f" ✅ [SUCCESS] Claude successfully generated Project Summary")
                return "[🤖 Powered by Claude] " + res
        except Exception as e:
            print(f" ⚠ [ERROR] LLM API error on Project Summary: {e}. Falling back to rule-based explanation.")
            return "[⚙️ Rule-Based Fallback] " + self._rule_based_project_summary(risk_df, bug_prediction)

    def _call_gemini(self, prompt: str) -> str:
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        response = model.generate_content(prompt)
        return response.text.strip()

    def _call_openai(self, prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    def _call_claude(self, prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    # ── Rule-based fallback (no API key required) ────────────────────────────
    # (Keep your existing BUG_TYPES, _rule_based_explanation, and _rule_based_project_summary methods exactly as they are down here)

if __name__ == "__main__":
    explainer = LLMExplainer(provider="mock")

    explanation = explainer.explain_module_risk(
        module="ALU",
        risk_score=0.84,
        risk_level="HIGH",
        features={
            "code_churn": 72, "historical_bug_density": 2.1,
            "coverage_trend": -1.2, "module_instability": 3.4,
            "recent_bug_rate": 1.8, "avg_coverage": 78,
        },
        bug_prediction=5,
        ci=(4, 7),
        summaries={"month_1_summary": {"total_bugs": 15, "avg_coverage": 80}},
    )
    print("📝 Module Risk Explanation:")
    print(explanation)


