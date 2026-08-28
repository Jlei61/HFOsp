#!/usr/bin/env python3
"""Wait coarsely for the frozen atlas, then run its analysis-only aggregate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev15_m3_multinetwork_response_analysis.json"
DEFAULT_STATUS = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_multinetwork_atlas/status/m3_replication_controller.json"
)
COMPLETE = "REV15_M3_MULTINETWORK_QUEUE_COMPLETE"
FAIL_CLOSED = {
    "REV15_M3_MULTINETWORK_EMERGENCY_STOPPED",
    "REV15_M3_MULTINETWORK_DRAINING_AFTER_FAILURE",
    "REV15_M3_MULTINETWORK_QUEUE_FAILED",
}


def classify(payload: dict) -> str:
    status = payload.get("status")
    if status == COMPLETE:
        if (
            int(payload.get("n_complete", -1)) == int(payload.get("n_jobs", -2))
            and int(payload.get("n_failed", 1)) == 0
            and int(payload.get("n_invalid_artifact", 1)) == 0
        ):
            return "complete"
        return "failed"
    if status in FAIL_CLOSED:
        return "failed"
    return "wait"


def _require_clean_commit(expected_commit: str) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).strip()
    if head != expected_commit or status:
        raise RuntimeError("analysis waiter worktree changed while waiting")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--status", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if args.interval_seconds != 600:
        raise ValueError("rev15 multinetwork waiter interval must remain 600 s")
    while True:
        _require_clean_commit(args.expected_commit)
        if args.status.is_file():
            payload = json.loads(args.status.read_text())
            state = classify(payload)
            if state == "failed":
                raise RuntimeError(
                    f"multinetwork queue failed closed: {payload.get('status')}"
                )
            if state == "complete":
                command = [
                    sys.executable,
                    str(ROOT / "scripts/aggregate_topic4_rev15_m3_multinetwork_response.py"),
                    "--config", str(args.config.resolve()),
                    "--artifact-root", str(args.artifact_root.resolve()),
                ]
                subprocess.run(command, cwd=ROOT, check=True)
                print(json.dumps({
                    "status": "REV15_M3_MULTINETWORK_WAIT_AND_AGGREGATE_COMPLETE",
                    "controller_status": payload["status"],
                    "n_complete": payload["n_complete"],
                }, indent=2), flush=True)
                return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
