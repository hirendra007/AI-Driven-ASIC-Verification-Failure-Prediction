"""
Synthetic Dataset Generator for AI-Driven ASIC Verification Failure Prediction
Generates: historical_verification_data.csv, bug_trend_data.csv, git_commits.csv
"""

import pandas as pd
import numpy as np
import random
import string
from datetime import datetime, timedelta
import os

random.seed(42)
np.random.seed(42)

MODULES = ["ALU", "Decoder", "Cache", "DMA", "AXI", "FIFO", "BranchUnit", "ControlUnit"]

# Module-specific risk profiles (realistic ASIC characteristics)
MODULE_PROFILES = {
    "ALU":         {"base_bug_density": 2.1, "base_coverage": 78, "churn_factor": 1.4, "runtime_base": 18},
    "Decoder":     {"base_bug_density": 0.8, "base_coverage": 91, "churn_factor": 0.7, "runtime_base": 10},
    "Cache":       {"base_bug_density": 2.8, "base_coverage": 72, "churn_factor": 1.6, "runtime_base": 32},
    "DMA":         {"base_bug_density": 2.3, "base_coverage": 76, "churn_factor": 1.3, "runtime_base": 25},
    "AXI":         {"base_bug_density": 1.5, "base_coverage": 85, "churn_factor": 1.1, "runtime_base": 20},
    "FIFO":        {"base_bug_density": 0.9, "base_coverage": 89, "churn_factor": 0.8, "runtime_base": 12},
    "BranchUnit":  {"base_bug_density": 1.8, "base_coverage": 81, "churn_factor": 1.2, "runtime_base": 16},
    "ControlUnit": {"base_bug_density": 1.2, "base_coverage": 87, "churn_factor": 0.9, "runtime_base": 14},
}

DEVELOPER_FEEDBACK = {
    "ALU":         ["overflow edge case in multiplier", "pipeline stall during signed division",
                    "carry propagation bug", "arithmetic shift right incorrect", "saturate mode failure"],
    "Decoder":     ["decode mismatch on illegal opcode", "control signal glitch", "instruction decode timing",
                    "opcode extension handling", "NOP decode issue"],
    "Cache":       ["memory stall issue", "cache coherency violation", "eviction policy bug",
                    "write-back inconsistency", "tag array mismatch", "dirty bit not set"],
    "DMA":         ["burst transfer hang", "descriptor fetch issue", "channel arbitration bug",
                    "interrupt not asserted", "address wrap issue"],
    "AXI":         ["AXI handshake violation", "outstanding transaction overflow", "AWLEN mismatch",
                    "RRESP error not propagated", "burst boundary crossing"],
    "FIFO":        ["overflow under stress", "read pointer wrap", "empty flag glitch",
                    "synchronizer metastability", "depth count off by one"],
    "BranchUnit":  ["branch misprediction on ret", "BTB miss under loop", "speculative fetch error",
                    "conditional jump timing", "indirect branch resolution"],
    "ControlUnit": ["FSM deadlock state", "reset sequence incomplete", "clock gate enable glitch",
                    "power state transition hang", "mode register not latched"],
}


def generate_historical_data(n_weeks=26, output_path="data/historical_verification_data.csv"):
    """Generate 6 months of weekly historical verification data."""
    rows = []
    start_date = datetime(2024, 7, 1)

    for week_offset in range(n_weeks):
        week_num = (week_offset % 4) + 1
        month_num = (week_offset // 4) + 1
        current_date = start_date + timedelta(weeks=week_offset)

        # Milestone pressure: bugs increase near end of milestone (every 4 weeks)
        milestone_pressure = 1.0 + 0.4 * np.sin(np.pi * (week_offset % 4) / 3)

        for module in MODULES:
            profile = MODULE_PROFILES[module]

            # Simulate natural variation + milestone effects
            commits = max(1, int(np.random.poisson(2.5) * (1 + 0.2 * milestone_pressure)))
            loc_changed = max(5, int(np.random.normal(
                40 * profile["churn_factor"], 15 * profile["churn_factor"]
            )))

            # Bug density with seasonal/milestone variation
            bug_density = profile["base_bug_density"] * milestone_pressure
            bugs_found = max(0, int(np.random.poisson(bug_density * commits * 0.6)))

            # Coverage degrades with churn, improves with low churn weeks
            churn_effect = -0.1 * (loc_changed / 100)
            coverage = min(99, max(50, profile["base_coverage"] + np.random.normal(churn_effect * 10, 3)))

            # Runtime correlated with loc and coverage
            runtime = max(5, profile["runtime_base"] + int(loc_changed * 0.15 + np.random.normal(0, 3)))

            # Pick relevant developer feedback
            feedback = random.choice(DEVELOPER_FEEDBACK[module]) if bugs_found > 0 else "clean run"

            rows.append({
                "week": week_num,
                "month": month_num,
                "module_name": module,
                "loc_changed": loc_changed,
                "commits": commits,
                "bugs_found": bugs_found,
                "coverage_percent": round(coverage, 1),
                "regression_runtime": runtime,
                "developer_feedback": feedback,
            })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Historical data: {len(df)} rows → {output_path}")
    return df


def generate_bug_trend_data(output_path="data/bug_trend_data.csv"):
    """Generate weekly bug burn-down trend data for milestone prediction."""
    rows = []
    start_date = datetime(2024, 7, 1)
    cumulative_bugs = {m: 0 for m in MODULES}

    for week_offset in range(26):
        week_num = (week_offset % 4) + 1
        month_num = (week_offset // 4) + 1

        total_bugs_week = 0
        for module in MODULES:
            profile = MODULE_PROFILES[module]
            pressure = 1.0 + 0.4 * np.sin(np.pi * (week_offset % 4) / 3)
            weekly_bugs = max(0, int(np.random.poisson(profile["base_bug_density"] * pressure)))
            cumulative_bugs[module] += weekly_bugs
            total_bugs_week += weekly_bugs

        rows.append({
            "week": week_num,
            "month": month_num,
            "week_offset": week_offset + 1,
            "total_bugs": total_bugs_week,
            "cumulative_bugs": sum(cumulative_bugs.values()),
            "open_bugs": max(0, sum(cumulative_bugs.values()) - int(sum(cumulative_bugs.values()) * 0.7)),
            "closed_bugs": int(sum(cumulative_bugs.values()) * 0.7),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Bug trend data: {len(df)} rows → {output_path}")
    return df


def _random_commit_id():
    return ''.join(random.choices(string.hexdigits[:16], k=7)).lower()


def generate_git_commits(n_commits=300, output_path="data/git_commits.csv"):
    """Generate synthetic git commit log with RTL file changes."""
    rows = []
    start_date = datetime(2024, 7, 1)
    end_date = datetime(2025, 3, 1)
    total_days = (end_date - start_date).days

    module_files = {
        "ALU":         ["alu.v", "alu_mul.v", "alu_div.v"],
        "Decoder":     ["decoder.v", "inst_decode.v"],
        "Cache":       ["cache.v", "cache_ctrl.v", "tag_array.v"],
        "DMA":         ["dma.v", "dma_ctrl.v", "desc_fetch.v"],
        "AXI":         ["axi_master.v", "axi_slave.v", "axi_arbiter.v"],
        "FIFO":        ["fifo.v", "fifo_sync.v"],
        "BranchUnit":  ["branch_unit.v", "btb.v", "bht.v"],
        "ControlUnit": ["ctrl_unit.v", "fsm.v", "power_ctrl.v"],
    }

    for _ in range(n_commits):
        commit_id = _random_commit_id()
        day_offset = random.randint(0, total_days)
        timestamp = (start_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")

        # A commit may touch 1–3 modules
        n_modules_touched = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
        touched_modules = random.sample(MODULES, n_modules_touched)

        for module in touched_modules:
            files = random.choice(module_files[module])
            profile = MODULE_PROFILES[module]
            loc_added = max(0, int(np.random.normal(20 * profile["churn_factor"], 10)))
            loc_deleted = max(0, int(np.random.normal(8 * profile["churn_factor"], 5)))

            rows.append({
                "commit_id": commit_id,
                "timestamp": timestamp,
                "module_name": module,
                "files_changed": files,
                "loc_added": loc_added,
                "loc_deleted": loc_deleted,
            })

    df = pd.DataFrame(rows)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Git commits: {len(df)} rows → {output_path}")
    return df


def generate_module_dependency_map(output_path="data/module_dependencies.csv"):
    """Define realistic RTL module dependency graph."""
    deps = [
        {"module": "ALU",         "depends_on": "ControlUnit"},
        {"module": "Decoder",     "depends_on": "ControlUnit"},
        {"module": "BranchUnit",  "depends_on": "Decoder"},
        {"module": "BranchUnit",  "depends_on": "ALU"},
        {"module": "Cache",       "depends_on": "DMA"},
        {"module": "DMA",         "depends_on": "AXI"},
        {"module": "Cache",       "depends_on": "AXI"},
        {"module": "FIFO",        "depends_on": "DMA"},
        {"module": "ControlUnit", "depends_on": "FIFO"},
    ]
    df = pd.DataFrame(deps)
    df.to_csv(output_path, index=False)
    print(f"✅ Module dependencies: {len(df)} rows → {output_path}")
    return df


if __name__ == "__main__":
    print("🔧 Generating synthetic ASIC verification datasets...\n")
    generate_historical_data(n_weeks=26, output_path="data/historical_verification_data.csv")
    generate_bug_trend_data(output_path="data/bug_trend_data.csv")
    generate_git_commits(n_commits=400, output_path="data/git_commits.csv")
    generate_module_dependency_map(output_path="data/module_dependencies.csv")
    print("\n✅ All datasets generated successfully!")
