#!/usr/bin/env python3
"""Wait coarsely for the response tensor, then prepare (not freeze) candidates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_multinetwork_atlas/analysis/m3_multinetwork_response_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev15_m3_robust_candidates.json"


def _require_clean_commit(expected_commit: str) -> None:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).strip()
    if head != expected_commit or status:
        raise RuntimeError("robust-config waiter worktree changed while waiting")


def classify(payload: dict) -> str:
    status = payload.get("status")
    inventory = payload.get("inventory", {})
    if status == "COMPLETE":
        if (
            inventory.get("complete_cartesian_product") is True
            and int(inventory.get("present_validated", -1)) == 116
            and not inventory.get("missing")
            and not inventory.get("invalid_artifact")
            and payload.get("response_tensor") is not None
            and payload.get("robust_directions") is not None
        ):
            return "complete"
        return "failed"
    if status in {"INVALID_INPUT", "INVALID_PROVENANCE"}:
        return "failed"
    return "wait"


def _notify(message: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(
            ["notify-send", "Topic 4 rev15 Node", message], check=False,
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args(argv)
    if args.interval_seconds != 600:
        raise ValueError("rev15 robust-config waiter interval must remain 600 s")
    while True:
        _require_clean_commit(args.expected_commit)
        if args.aggregate.is_file():
            payload = json.loads(args.aggregate.read_text())
            state = classify(payload)
            if state == "failed":
                raise RuntimeError(
                    f"multinetwork response aggregate failed closed: {payload.get('status')}"
                )
            if state == "complete":
                if args.output.exists():
                    raise RuntimeError("robust candidate config already exists")
                command = [
                    sys.executable,
                    str(ROOT / "scripts/prepare_topic4_rev15_m3_robust_candidate_config.py"),
                    "--aggregate", str(args.aggregate.resolve()),
                    "--artifact-root", str(args.artifact_root.resolve()),
                    "--output", str(args.output.resolve()),
                ]
                subprocess.run(command, cwd=ROOT, check=True)
                config = json.loads(args.output.read_text())
                _notify(
                    "三网络响应张量已闭合；robust Node config 已生成，等待提交和冻结。"
                )
                print(json.dumps({
                    "status": "REV15_M3_RESPONSE_WAIT_AND_PREPARE_COMPLETE",
                    "aggregate_status": payload["status"],
                    "present_validated": payload["inventory"]["present_validated"],
                    "output": str(args.output.resolve()),
                    "candidate_count": config["m3_design"]["candidate_count"],
                    "snn_simulation_run": False,
                }, indent=2), flush=True)
                return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
