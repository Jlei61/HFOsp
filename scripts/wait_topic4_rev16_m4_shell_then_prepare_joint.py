#!/usr/bin/env python3
"""Wait coarsely for M4-shell workers, aggregate, then prepare joint config."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_STATUS = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "m4_shell_coordinate_atlas/status/m3_replication_controller.json"
)
DEFAULT_ANALYSIS = ROOT / "config/topic4_rev16_joint_m3_m4_response_analysis.json"
DEFAULT_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "m4_shell_coordinate_atlas/analysis/joint_m3_m4_response_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_joint_m3_m4_candidates.json"
COMPLETE = "REV16_M4_SHELL_QUEUE_COMPLETE"
FAIL_CLOSED = {
    "REV16_M4_SHELL_EMERGENCY_STOPPED",
    "REV16_M4_SHELL_DRAINING_AFTER_FAILURE",
    "REV16_M4_SHELL_QUEUE_FAILED",
}


def classify(payload: Mapping[str, object]) -> str:
    status = payload.get("status")
    if status == COMPLETE:
        if (
            int(payload.get("n_complete", -1)) == 120
            and int(payload.get("n_jobs", -2)) == 120
            and int(payload.get("n_failed", 1)) == 0
            and int(payload.get("n_invalid_artifact", 1)) == 0
        ):
            return "complete"
        return "failed"
    if status in FAIL_CLOSED:
        return "failed"
    return "wait"


def _require_clean_commit(expected_commit: str) -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).strip()
    if head != expected_commit or status:
        raise RuntimeError("rev16 analysis waiter worktree changed while waiting")


def _notify(message: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "Topic 4 rev16 Node", message], check=False)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if args.interval_seconds != 600:
        raise ValueError("rev16 waiter interval must remain 600 s")
    while True:
        _require_clean_commit(args.expected_commit)
        if args.status.is_file():
            controller = json.loads(args.status.read_text())
            state = classify(controller)
            if state == "failed":
                raise RuntimeError(
                    f"rev16 M4-shell queue failed closed: {controller.get('status')}"
                )
            if state == "complete":
                subprocess.run([
                    sys.executable,
                    str(ROOT / "scripts/aggregate_topic4_rev16_joint_m3_m4_response.py"),
                    "--config", str(args.analysis_config.resolve()),
                    "--artifact-root", str(args.artifact_root.resolve()),
                ], cwd=ROOT, check=True)
                aggregate = json.loads(args.aggregate.read_text())
                if (
                    aggregate.get("status") != "COMPLETE"
                    or aggregate.get("inventory", {}).get("present_validated") != 120
                    or aggregate.get("joint_response_tensor") is None
                    or aggregate.get("robust_directions") is None
                ):
                    raise RuntimeError("rev16 joint response aggregate did not close")
                if args.output.exists():
                    raise RuntimeError("rev16 joint candidate config already exists")
                subprocess.run([
                    sys.executable,
                    str(ROOT / "scripts/prepare_topic4_rev16_joint_candidate_config.py"),
                    "--aggregate", str(args.aggregate.resolve()),
                    "--analysis-config", str(args.analysis_config.resolve()),
                    "--artifact-root", str(args.artifact_root.resolve()),
                    "--output", str(args.output.resolve()),
                ], cwd=ROOT, check=True)
                prepared = json.loads(args.output.read_text())
                _notify(
                    "M4-shell 120/120 完成；joint 3x48 候选配置已生成，等待提交和冻结。"
                )
                print(json.dumps({
                    "status": "REV16_M4_SHELL_WAIT_AND_PREPARE_COMPLETE",
                    "present_validated": 120,
                    "output": str(args.output.resolve()),
                    "candidate_count": prepared["field_design"]["candidate_count"],
                    "snn_simulation_run": False,
                }, indent=2), flush=True)
                return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
