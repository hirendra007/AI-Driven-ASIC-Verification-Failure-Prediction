"""
Git Commit Parser
Parses git commit logs and maps changed files to RTL modules.
Supports both live git repos and CSV-based simulation.
"""

import os
import re
import subprocess
import pandas as pd
from datetime import datetime
from typing import Optional


# ── File → Module mapping ──────────────────────────────────────────────────────
FILE_TO_MODULE = {
    # ALU
    "alu.v": "ALU", "alu_mul.v": "ALU", "alu_div.v": "ALU",
    # Decoder
    "decoder.v": "Decoder", "inst_decode.v": "Decoder",
    # Cache
    "cache.v": "Cache", "cache_ctrl.v": "Cache", "tag_array.v": "Cache",
    # DMA
    "dma.v": "DMA", "dma_ctrl.v": "DMA", "desc_fetch.v": "DMA",
    # AXI
    "axi_master.v": "AXI", "axi_slave.v": "AXI", "axi_arbiter.v": "AXI",
    # FIFO
    "fifo.v": "FIFO", "fifo_sync.v": "FIFO",
    # BranchUnit
    "branch_unit.v": "BranchUnit", "btb.v": "BranchUnit", "bht.v": "BranchUnit",
    # ControlUnit
    "ctrl_unit.v": "ControlUnit", "fsm.v": "ControlUnit", "power_ctrl.v": "ControlUnit",
}

KNOWN_MODULES = list(set(FILE_TO_MODULE.values()))


def resolve_module_from_file(filepath: str) -> Optional[str]:
    """Map a file path to its RTL module name."""
    filename = os.path.basename(filepath).lower()
    # Direct lookup
    if filename in FILE_TO_MODULE:
        return FILE_TO_MODULE[filename]
    # Pattern-based inference
    for pattern, module in [
        (r"alu", "ALU"), (r"decoder|decode", "Decoder"),
        (r"cache", "Cache"), (r"dma", "DMA"),
        (r"axi", "AXI"), (r"fifo", "FIFO"),
        (r"branch", "BranchUnit"), (r"ctrl|control|fsm", "ControlUnit"),
    ]:
        if re.search(pattern, filename):
            return module
    return None


class CommitParser:
    """Parse git commits to identify changed RTL modules."""

    def __init__(self, commit_csv_path: Optional[str] = None, repo_path: Optional[str] = None):
        """
        Args:
            commit_csv_path: Path to synthetic git_commits.csv (simulation mode).
            repo_path: Path to actual git repository (live mode).
        """
        self.commit_csv_path = commit_csv_path
        self.repo_path = repo_path
        self._commit_df: Optional[pd.DataFrame] = None

        if commit_csv_path and os.path.exists(commit_csv_path):
            self._commit_df = pd.read_csv(commit_csv_path)

    def parse_commit(self, commit_id: str) -> dict:
        """
        Parse a commit and return changed modules with churn metrics.

        Returns:
            {
                "commit_id": str,
                "timestamp": str,
                "changed_modules": list[str],
                "module_details": [{"module": str, "files": list, "loc_added": int, "loc_deleted": int}],
                "total_churn": int,
            }
        """
        if self._commit_df is not None:
            return self._parse_from_csv(commit_id)
        elif self.repo_path:
            return self._parse_from_git(commit_id)
        else:
            raise RuntimeError("No data source configured. Provide commit_csv_path or repo_path.")

    def _parse_from_csv(self, commit_id: str) -> dict:
        """Parse commit from synthetic CSV dataset."""
        rows = self._commit_df[self._commit_df["commit_id"] == commit_id]
        if rows.empty:
            # Try prefix match
            rows = self._commit_df[self._commit_df["commit_id"].str.startswith(commit_id)]
        if rows.empty:
            raise ValueError(f"Commit '{commit_id}' not found.")

        timestamp = rows["timestamp"].iloc[0]
        module_details = []
        seen_modules = set()

        for _, row in rows.iterrows():
            module = row["module_name"]
            if module not in seen_modules:
                seen_modules.add(module)
                module_details.append({
                    "module": module,
                    "files": [row["files_changed"]],
                    "loc_added": int(row["loc_added"]),
                    "loc_deleted": int(row["loc_deleted"]),
                    "code_churn": int(row["loc_added"]) + int(row["loc_deleted"]),
                })
            else:
                # Accumulate churn for same module
                for detail in module_details:
                    if detail["module"] == module:
                        detail["files"].append(row["files_changed"])
                        detail["loc_added"] += int(row["loc_added"])
                        detail["loc_deleted"] += int(row["loc_deleted"])
                        detail["code_churn"] += int(row["loc_added"]) + int(row["loc_deleted"])

        return {
            "commit_id": commit_id,
            "timestamp": timestamp,
            "changed_modules": list(seen_modules),
            "module_details": module_details,
            "total_churn": sum(d["code_churn"] for d in module_details),
        }

    def _parse_from_git(self, commit_id: str) -> dict:
        """Parse commit from actual git repository."""
        try:
            # Get timestamp
            ts_cmd = ["git", "-C", self.repo_path, "log", "-1",
                      "--format=%ci", commit_id]
            timestamp = subprocess.check_output(ts_cmd, text=True).strip()

            # Get changed files with diff stats
            diff_cmd = ["git", "-C", self.repo_path, "diff", "--numstat",
                        f"{commit_id}~1", commit_id]
            diff_output = subprocess.check_output(diff_cmd, text=True)

            module_map = {}
            for line in diff_output.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                added, deleted, filepath = parts
                module = resolve_module_from_file(filepath)
                if module:
                    if module not in module_map:
                        module_map[module] = {"files": [], "loc_added": 0, "loc_deleted": 0}
                    module_map[module]["files"].append(filepath)
                    try:
                        module_map[module]["loc_added"] += int(added)
                        module_map[module]["loc_deleted"] += int(deleted)
                    except ValueError:
                        pass

            module_details = [
                {
                    "module": m,
                    "files": v["files"],
                    "loc_added": v["loc_added"],
                    "loc_deleted": v["loc_deleted"],
                    "code_churn": v["loc_added"] + v["loc_deleted"],
                }
                for m, v in module_map.items()
            ]

            return {
                "commit_id": commit_id,
                "timestamp": timestamp,
                "changed_modules": list(module_map.keys()),
                "module_details": module_details,
                "total_churn": sum(d["code_churn"] for d in module_details),
            }

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Git command failed: {e}")

    def get_all_commit_ids(self) -> list:
        """Return all available commit IDs."""
        if self._commit_df is not None:
            return self._commit_df["commit_id"].unique().tolist()
        return []

    def get_recent_commits(self, n: int = 10) -> pd.DataFrame:
        """Return the most recent N commits."""
        if self._commit_df is not None:
            latest = (
                self._commit_df.sort_values("timestamp", ascending=False)
                .drop_duplicates("commit_id")
                .head(n)
            )
            return latest[["commit_id", "timestamp", "module_name"]].reset_index(drop=True)
        return pd.DataFrame()


def format_commit_summary(parsed: dict) -> str:
    """Pretty-print a parsed commit."""
    lines = [
        f"┌─ Commit: {parsed['commit_id']}",
        f"│  Timestamp: {parsed['timestamp']}",
        f"│  Total Churn: {parsed['total_churn']} LOC",
        "│",
        "│  Changed Modules:",
    ]
    for detail in parsed["module_details"]:
        lines.append(
            f"│    • {detail['module']:15s} "
            f"+{detail['loc_added']} / -{detail['loc_deleted']} LOC  "
            f"({', '.join(detail['files'])})"
        )
    lines.append("└" + "─" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    csv_path = os.path.join(os.path.dirname(__file__), "../data/git_commits.csv")
    parser = CommitParser(commit_csv_path=csv_path)

    recent = parser.get_recent_commits(5)
    print("📝 Recent Commits:\n", recent.to_string(index=False))
    print()

    if not recent.empty:
        cid = recent["commit_id"].iloc[0]
        parsed = parser.parse_commit(cid)
        print(format_commit_summary(parsed))
