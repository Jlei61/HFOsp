#!/usr/bin/env python3
"""Wait coarsely for the rev16 response atlas, then freeze and launch selection."""
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
DEFAULT_CONTROLLER = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "m4_shell_coordinate_atlas/status/m3_replication_controller.json"
)
DEFAULT_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "m4_shell_coordinate_atlas/analysis/joint_m3_m4_response_aggregate.json"
)
DEFAULT_ANALYSIS_CONFIG = ROOT / "config/topic4_rev16_joint_m3_m4_response_analysis.json"
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_joint_m3_m4_candidates.json"
QUEUE_COMPLETE = "REV16_M4_SHELL_QUEUE_COMPLETE"
QUEUE_FAILURES = {
    "REV16_M4_SHELL_EMERGENCY_STOPPED",
    "REV16_M4_SHELL_DRAINING_AFTER_FAILURE",
    "REV16_M4_SHELL_QUEUE_FAILED",
}
AGGREGATE_SCHEMA = "topic4_rev16_joint_m3_m4_response_aggregate_v1"


def classify(controller: Mapping[str, object], aggregate: Mapping[str, object] | None) -> str:
    status = controller.get("status")
    if status in QUEUE_FAILURES:
        return "failed"
    if status != QUEUE_COMPLETE:
        return "wait"
    if not (
        int(controller.get("n_jobs", -1)) == 120
        and int(controller.get("n_complete", -1)) == 120
        and int(controller.get("n_failed", 1)) == 0
        and int(controller.get("n_invalid_artifact", 1)) == 0
    ):
        return "failed"
    if aggregate is None or aggregate.get("status") != "COMPLETE":
        return "wait"
    inventory = aggregate.get("inventory")
    if not isinstance(inventory, Mapping) or not (
        aggregate.get("schema_id") == AGGREGATE_SCHEMA
        and int(inventory.get("present_validated", -1)) == 120
        and inventory.get("complete_cartesian_product") is True
        and aggregate.get("joint_response_tensor") is not None
        and aggregate.get("robust_directions") is not None
        and aggregate.get("provenance", {}).get("formal_ready") is True
    ):
        return "failed"
    return "ready"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _require_clean_commit(expected_commit: str) -> None:
    if _git("rev-parse", "HEAD") != expected_commit or _git(
        "status", "--porcelain", "--untracked-files=all"
    ):
        raise RuntimeError("rev16 stage-transition worktree changed while waiting")


def _notify(message: str) -> None:
    if shutil.which("notify-send"):
        subprocess.run(["notify-send", "Topic 4 rev16 Node", message], check=False)


def _load_optional(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, default=DEFAULT_CONTROLLER)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--worker-cap", type=int, default=6)
    args = parser.parse_args(argv)
    if args.interval_seconds != 600:
        raise ValueError("rev16 stage-transition waiter interval must remain 600 s")
    if args.worker_cap != 6:
        raise ValueError("rev16 selection worker cap must remain 6")
    if args.output.exists():
        raise RuntimeError("rev16 joint candidate config already exists")

    while True:
        _require_clean_commit(args.expected_commit)
        controller = _load_optional(args.controller)
        state = "wait" if controller is None else classify(
            controller, _load_optional(args.aggregate),
        )
        if state == "failed":
            raise RuntimeError("rev16 response atlas or aggregate failed closed")
        if state == "ready":
            break
        time.sleep(args.interval_seconds)

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/prepare_topic4_rev16_joint_candidate_config.py"),
        "--aggregate", str(args.aggregate.resolve()),
        "--analysis-config", str(args.analysis_config.resolve()),
        "--artifact-root", str(args.artifact_root.resolve()),
        "--output", str(args.output.resolve()),
    ], cwd=ROOT, check=True)
    relative = str(args.output.resolve().relative_to(ROOT))
    changed = _git("status", "--porcelain", "--untracked-files=all").splitlines()
    if changed != [f"?? {relative}"]:
        raise RuntimeError(f"unexpected paths changed during rev16 preparation: {changed}")
    subprocess.run(["git", "add", "--", relative], cwd=ROOT, check=True)
    subprocess.run([
        "git", "commit", "-m", "Freeze rev16 joint Node candidates",
    ], cwd=ROOT, check=True)
    freeze_commit = _git("rev-parse", "HEAD")
    if _git("status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("rev16 worktree is dirty after candidate config commit")

    subprocess.run([
        sys.executable,
        str(ROOT / "scripts/freeze_topic4_rev16_joint_m3_m4_candidates.py"),
        "--config", str(args.output.resolve()),
        "--expected-commit", freeze_commit,
        "--artifact-root", str(args.artifact_root.resolve()),
    ], cwd=ROOT, check=True)
    launch = subprocess.run([
        sys.executable,
        str(ROOT / "scripts/launch_topic4_rev16_joint_m3_m4_candidates.py"),
        "--config", str(args.output.resolve()),
        "--expected-commit", freeze_commit,
        "--artifact-root", str(args.artifact_root.resolve()),
        "--worker-cap", str(args.worker_cap), "--execute",
    ], cwd=ROOT, check=True, text=True, capture_output=True)
    launched = json.loads(launch.stdout)
    _notify(
        "M4 atlas 与 joint tensor 完成；2351–2353 Node 候选选择已冻结并启动。"
    )
    print(json.dumps({
        "status": "REV16_JOINT_SELECTION_LAUNCHED",
        "freeze_commit": freeze_commit,
        "controller_unit": launched["unit"],
        "n_jobs": launched["n_jobs"],
        "worker_cap": args.worker_cap,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
