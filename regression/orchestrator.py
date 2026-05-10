"""
Regression Orchestration Engine
Manages selective Verilator regression execution based on ML risk predictions.
Implements: dependency expansion, coverage-based validation, fallback logic.
"""

import os
import subprocess
import tempfile
import time
import random
import json
import pandas as pd
import numpy as np
from typing import Optional
from dataclasses import dataclass, field


# ── Module Dependency Graph ────────────────────────────────────────────────────
DEFAULT_DEPENDENCIES = {
    "ALU":         {"ControlUnit"},
    "Decoder":     {"ControlUnit"},
    "BranchUnit":  {"Decoder", "ALU"},
    "Cache":       {"DMA", "AXI"},
    "DMA":         {"AXI"},
    "FIFO":        {"DMA"},
    "ControlUnit": {"FIFO"},
    "AXI":         set(),
}


@dataclass
class RegressionResult:
    module: str
    coverage_percent: float
    simulation_runtime: float
    pass_count: int
    fail_count: int
    status: str  # "PASS" | "FAIL" | "SKIPPED"
    log: str = ""


@dataclass
class RegressionReport:
    commit_id: str
    high_risk_modules: list
    executed_modules: list
    skipped_modules: list
    results: list = field(default_factory=list)
    fallback_triggered: bool = False
    total_runtime: float = 0.0

    def to_dict(self):
        return {
            "commit_id": self.commit_id,
            "high_risk_modules": self.high_risk_modules,
            "executed_modules": self.executed_modules,
            "skipped_modules": self.skipped_modules,
            "fallback_triggered": self.fallback_triggered,
            "total_runtime": round(self.total_runtime, 2),
            "results": [
                {
                    "module": r.module,
                    "coverage_percent": r.coverage_percent,
                    "simulation_runtime": r.simulation_runtime,
                    "pass_count": r.pass_count,
                    "fail_count": r.fail_count,
                    "status": r.status,
                }
                for r in self.results
            ],
        }


class DependencyResolver:
    """Resolves RTL module dependencies for regression expansion."""

    def __init__(self, dep_csv_path: Optional[str] = None):
        if dep_csv_path and os.path.exists(dep_csv_path):
            dep_df = pd.read_csv(dep_csv_path)
            self.deps = {}
            for _, row in dep_df.iterrows():
                m = row["module"]
                d = row["depends_on"]
                self.deps.setdefault(m, set()).add(d)
        else:
            self.deps = DEFAULT_DEPENDENCIES

    def get_dependent_modules(self, modules: list) -> set:
        """
        Given a set of changed modules, return all modules that depend on them
        (i.e., modules that should be re-verified because their dependencies changed).
        """
        affected = set(modules)
        for module, deps in self.deps.items():
            if deps.intersection(set(modules)):
                affected.add(module)
        return affected

    def expand_for_regression(self, high_risk_modules: list, all_changed: list) -> list:
        """
        Return the full set of modules to run regression for:
        - High risk modules themselves
        - All modules that depend on high-risk modules
        """
        expanded = set(high_risk_modules)
        for module, deps in self.deps.items():
            if deps.intersection(set(high_risk_modules)):
                expanded.add(module)
        return sorted(expanded)


class VerilatorSimulator:
    """
    Verilator-based regression simulator.
    In production: executes actual Verilator commands.
    In simulation mode: generates realistic synthetic results.
    """

    # Realistic RTL module simulation profiles
    SIMULATION_PROFILES = {
        "ALU":         {"base_cov": 78, "base_rt": 18, "fail_rate": 0.35},
        "Decoder":     {"base_cov": 91, "base_rt": 10, "fail_rate": 0.12},
        "Cache":       {"base_cov": 72, "base_rt": 32, "fail_rate": 0.42},
        "DMA":         {"base_cov": 76, "base_rt": 25, "fail_rate": 0.38},
        "AXI":         {"base_cov": 85, "base_rt": 20, "fail_rate": 0.20},
        "FIFO":        {"base_cov": 89, "base_rt": 12, "fail_rate": 0.15},
        "BranchUnit":  {"base_cov": 81, "base_rt": 16, "fail_rate": 0.28},
        "ControlUnit": {"base_cov": 87, "base_rt": 14, "fail_rate": 0.18},
    }

    def __init__(self, rtl_dir: Optional[str] = None, simulation_mode: bool = True):
        self.rtl_dir = rtl_dir
        self.simulation_mode = simulation_mode

    def run_module(self, module: str, risk_score: float = 0.5) -> RegressionResult:
        """Run regression for a single module."""
        if self.simulation_mode:
            return self._simulate_module(module, risk_score)
        else:
            return self._run_verilator(module)

    def _simulate_module(self, module: str, risk_score: float) -> RegressionResult:
        """Generate synthetic but realistic regression results."""
        profile = self.SIMULATION_PROFILES.get(module, {"base_cov": 82, "base_rt": 15, "fail_rate": 0.25})

        # Higher risk score → lower coverage, higher fail probability
        cov_penalty = risk_score * 12
        coverage = max(40, min(99, profile["base_cov"] - cov_penalty + np.random.normal(0, 3)))

        runtime = profile["base_rt"] * (1 + risk_score * 0.3) + np.random.normal(0, 2)
        runtime = max(5, runtime)

        # Determine pass/fail
        fail_prob = profile["fail_rate"] * (0.5 + risk_score)
        total_tests = random.randint(80, 200)
        fail_count = int(total_tests * fail_prob * np.random.uniform(0.8, 1.2))
        fail_count = max(0, min(fail_count, total_tests // 2))
        pass_count = total_tests - fail_count
        status = "FAIL" if fail_count > 0 else "PASS"

        log = (
            f"[Verilator] Module: {module}\n"
            f"  Tests: {total_tests} | Pass: {pass_count} | Fail: {fail_count}\n"
            f"  Coverage: {coverage:.1f}% | Runtime: {runtime:.1f}s\n"
            f"  Status: {status}\n"
        )
        if fail_count > 0:
            log += f"  ⚠ {fail_count} test(s) failed — review {module} implementation\n"

        time.sleep(0.05)  # Simulate execution delay

        return RegressionResult(
            module=module,
            coverage_percent=round(coverage, 1),
            simulation_runtime=round(runtime, 2),
            pass_count=pass_count,
            fail_count=fail_count,
            status=status,
            log=log,
        )

    def _run_verilator(self, module: str) -> RegressionResult:
        """Run actual Verilator simulation (requires RTL files)."""
        rtl_file = os.path.join(self.rtl_dir or ".", f"{module.lower()}.v")
        if not os.path.exists(rtl_file):
            return RegressionResult(
                module=module, coverage_percent=0.0, simulation_runtime=0.0,
                pass_count=0, fail_count=0, status="SKIPPED",
                log=f"RTL file not found: {rtl_file}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                t0 = time.time()
                compile_cmd = [
                    "verilator", "--binary", "-j", "0",
                    "--coverage", "--assert",
                    "-Mdir", tmpdir,
                    "--top-module", module.lower(),
                    rtl_file,
                ]
                subprocess.run(compile_cmd, check=True, capture_output=True)
                sim_binary = os.path.join(tmpdir, f"V{module.lower()}")
                result = subprocess.run([sim_binary], capture_output=True, text=True, timeout=120)
                runtime = time.time() - t0

                # Parse coverage from .dat file if available
                coverage = 85.0  # Default
                cov_file = os.path.join(tmpdir, "coverage.dat")
                if os.path.exists(cov_file):
                    with open(cov_file) as f:
                        content = f.read()
                    match = __import__("re").search(r"coverage.*?(\d+\.?\d*)", content, __import__("re").I)
                    if match:
                        coverage = float(match.group(1))

                status = "PASS" if result.returncode == 0 else "FAIL"
                return RegressionResult(
                    module=module, coverage_percent=coverage,
                    simulation_runtime=round(runtime, 2),
                    pass_count=1 if status == "PASS" else 0,
                    fail_count=0 if status == "PASS" else 1,
                    status=status, log=result.stdout + result.stderr,
                )
            except Exception as e:
                return RegressionResult(
                    module=module, coverage_percent=0.0, simulation_runtime=0.0,
                    pass_count=0, fail_count=1, status="FAIL",
                    log=f"Verilator error: {str(e)}"
                )


class RegressionOrchestrator:
    """
    Orchestrates selective regression execution based on ML risk predictions.

    Strategy:
    1. Run regression for HIGH/MEDIUM risk modules + their dependents
    2. Evaluate coverage results
    3. If predicted modules pass cleanly → run remaining changed modules
    4. Identify actual failing modules
    """

    COVERAGE_THRESHOLD = 75.0  # Below this = still high risk after regression

    def __init__(
        self,
        simulator: Optional[VerilatorSimulator] = None,
        dep_csv_path: Optional[str] = None,
        simulation_mode: bool = True,
    ):
        self.simulator = simulator or VerilatorSimulator(simulation_mode=simulation_mode)
        self.resolver = DependencyResolver(dep_csv_path)

    def run(
        self,
        commit_id: str,
        changed_modules: list,
        risk_df: pd.DataFrame,
    ) -> RegressionReport:
        """
        Execute regression orchestration for a commit.

        Args:
            commit_id: Git commit identifier.
            changed_modules: Modules changed in the commit.
            risk_df: DataFrame with columns [module_name, risk_score, risk_level].

        Returns:
            RegressionReport with all results.
        """
        print(f"\n🔬 Starting regression orchestration for commit: {commit_id}")

        report = RegressionReport(
            commit_id=commit_id,
            high_risk_modules=[],
            executed_modules=[],
            skipped_modules=[],
        )

        # ── Step 1: Identify high/medium risk modules from ML predictions ────
        risk_map = dict(zip(risk_df["module_name"], zip(risk_df["risk_score"], risk_df["risk_level"])))
        high_risk = [m for m, (s, l) in risk_map.items() if l in ("HIGH", "MEDIUM")]
        low_risk = [m for m, (s, l) in risk_map.items() if l == "LOW"]

        # Only consider modules that were changed in this commit
        high_risk_changed = [m for m in high_risk if m in changed_modules]
        low_risk_changed = [m for m in low_risk if m in changed_modules]

        report.high_risk_modules = high_risk_changed

        # ── Step 2: Expand with dependent modules ────────────────────────────
        priority_modules = self.resolver.expand_for_regression(high_risk_changed, changed_modules)
        print(f"   Priority modules (high/med risk + dependents): {priority_modules}")

        # ── Step 3: Run priority regression ──────────────────────────────────
        t_start = time.time()
        priority_results = []

        for module in priority_modules:
            risk_score = risk_map.get(module, (0.5, "MEDIUM"))[0]
            print(f"   ▶ Running regression: {module} (risk={risk_score:.2f})")
            result = self.simulator.run_module(module, risk_score)
            priority_results.append(result)
            report.results.append(result)
            report.executed_modules.append(module)
            print(f"     ✓ Coverage: {result.coverage_percent}% | Status: {result.status}")

        # ── Step 4: Coverage-based validation ────────────────────────────────
        low_coverage_modules = [
            r for r in priority_results
            if r.coverage_percent < self.COVERAGE_THRESHOLD or r.status == "FAIL"
        ]
        all_priority_stable = len(low_coverage_modules) == 0

        if all_priority_stable and low_risk_changed:
            # ── Step 5: Fallback — run remaining changed low-risk modules ─────
            print(f"\n   ✅ Priority modules stable. Running fallback for: {low_risk_changed}")
            report.fallback_triggered = True

            for module in low_risk_changed:
                risk_score = risk_map.get(module, (0.2, "LOW"))[0]
                print(f"   ▶ Fallback regression: {module}")
                result = self.simulator.run_module(module, risk_score)
                report.results.append(result)
                report.executed_modules.append(module)
        else:
            # Skip low-risk modules — priority modules need attention
            skipped = [m for m in changed_modules if m not in report.executed_modules]
            report.skipped_modules = skipped
            if skipped:
                print(f"\n   ⏭  Skipped low-risk modules (saving time): {skipped}")

        report.total_runtime = time.time() - t_start
        print(f"\n   ⏱  Total regression runtime: {report.total_runtime:.1f}s")

        return report

    def summarize(self, report: RegressionReport) -> str:
        """Generate human-readable regression summary."""
        lines = [
            f"\n{'='*60}",
            f"REGRESSION SUMMARY — Commit: {report.commit_id}",
            f"{'='*60}",
            f"High-Risk Modules:  {', '.join(report.high_risk_modules) or 'None'}",
            f"Executed:           {', '.join(report.executed_modules)}",
            f"Skipped:            {', '.join(report.skipped_modules) or 'None'}",
            f"Fallback Triggered: {'Yes' if report.fallback_triggered else 'No'}",
            f"Total Runtime:      {report.total_runtime:.1f}s",
            f"",
            f"Module Results:",
        ]
        for r in report.results:
            status_icon = "✅" if r.status == "PASS" else "❌"
            lines.append(
                f"  {status_icon} {r.module:15s} "
                f"Coverage: {r.coverage_percent:5.1f}%  "
                f"Fail: {r.fail_count:3d}  "
                f"RT: {r.simulation_runtime:.1f}s"
            )
        lines.append("=" * 60)
        return "\n".join(lines)


if __name__ == "__main__":
    simulator = VerilatorSimulator(simulation_mode=True)
    orchestrator = RegressionOrchestrator(simulator=simulator)

    # Mock risk predictions
    risk_df = pd.DataFrame([
        {"module_name": "ALU",     "risk_score": 0.84, "risk_level": "HIGH"},
        {"module_name": "Cache",   "risk_score": 0.63, "risk_level": "MEDIUM"},
        {"module_name": "Decoder", "risk_score": 0.19, "risk_level": "LOW"},
    ])

    report = orchestrator.run(
        commit_id="b13d2xx",
        changed_modules=["ALU", "Cache", "Decoder"],
        risk_df=risk_df,
    )
    print(orchestrator.summarize(report))
