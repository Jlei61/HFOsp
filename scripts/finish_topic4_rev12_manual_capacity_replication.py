#!/usr/bin/env python3
"""Wait coarsely for Stage-AD and aggregate the capacity control once."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    output_root = artifact_root / config["output_root"]
    controller = output_root / "status" / "fit_controller.json"
    interval = int(config["resources"]["monitor_interval_seconds"])
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    while True:
        if controller.exists():
            payload = json.loads(controller.read_text())
            if payload.get("status") != "REV12ND_NODE_WORKER_QUEUE_COMPLETE":
                raise RuntimeError("Stage-AD controller did not complete")
            if payload.get("expected_git_commit") != expected:
                raise RuntimeError("Stage-AD controller commit changed")
            break
        time.sleep(interval)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/aggregate_topic4_rev12_soft_global_fit.py"),
        "--config", str(config_path), "--artifact-root", str(artifact_root),
        "--seed-pool", "fit",
    ], cwd=ROOT, check=True)
    aggregate = output_root / "aggregate" / "fit_soft_global_summary.json"
    result = json.loads(aggregate.read_text())
    complete = {
        "status": "REV12ND_MANUAL_CAPACITY_REPLICATION_COMPLETE",
        "git_commit": expected,
        "fit_status": result["status"],
        "n_candidates": len(result["rows"]),
        "n_networks": len(result["requested_seeds"]),
        "selection_eligible": False,
        "automatic_nominees": result["nomination_decision"]["candidate_ids"],
        "aggregate": str(aggregate),
    }
    if complete["automatic_nominees"]:
        raise RuntimeError("non-selectable capacity control entered nomination")
    output = output_root / "status" / "stage_ad_complete.json"
    _atomic_json(output, complete)
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
