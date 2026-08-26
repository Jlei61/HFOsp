#!/usr/bin/env python3
"""Wait coarsely for Stage-AK, then aggregate and analyze once."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


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
                raise RuntimeError("Stage-AK controller ended without completion")
            if payload.get("expected_git_commit") != expected:
                raise RuntimeError("Stage-AK controller commit changed")
            break
        time.sleep(interval)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/aggregate_topic4_rev12_soft_global_fit.py"),
        "--config", str(config_path), "--artifact-root", str(artifact_root),
        "--seed-pool", "fit",
    ], cwd=ROOT, check=True)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/analyze_topic4_rev12_dual_node_channel_canary.py"),
        "--config", str(config_path), "--artifact-root", str(artifact_root),
    ], cwd=ROOT, check=True)
    result_path = output_root / "analysis" / "dual_node_channel_result.json"
    result = json.loads(result_path.read_text())
    complete_path = output_root / "status" / "stage_ak_complete.json"
    complete_path.write_text(json.dumps({
        "status": "REV12ND_DUAL_NODE_CHANNEL_CANARY_COMPLETE",
        "git_commit": expected,
        "scientific_status": result["status"],
        "balanced_candidate_ids": result["balanced_candidate_ids"],
        "result": str(result_path),
    }, indent=2) + "\n")
    print(complete_path.read_text(), flush=True)


if __name__ == "__main__":
    main()
