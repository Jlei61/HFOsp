#!/usr/bin/env python3
"""Wait coarsely for Stage-X, then aggregate and analyze it once."""
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
    parser.add_argument("--stage-config", required=True, type=Path)
    parser.add_argument("--analysis-config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--wait-seconds", type=int, default=600)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    stage_config = args.stage_config.resolve()
    analysis_config = args.analysis_config.resolve()
    stage = json.loads(stage_config.read_text())
    artifact_root = args.artifact_root.resolve()
    output_root = artifact_root / stage["output_root"]
    controller = output_root / "status" / "fit_controller.json"
    wait_seconds = max(60, int(args.wait_seconds))
    while not controller.exists():
        time.sleep(wait_seconds)
    status = json.loads(controller.read_text())
    if status.get("status") != "REV12ND_NODE_WORKER_QUEUE_COMPLETE":
        raise RuntimeError("Stage-X controller ended without queue completion")
    manifest = json.loads((artifact_root / stage["candidate_manifest"]).read_text())
    expected_workers = len(manifest["candidates"]) * len(
        stage["search"]["fit_network_seeds"]
    )
    worker_files = list((output_root / "workers").glob("*.json"))
    complete = sum(
        json.loads(path.read_text()).get("status") == "REV12ND_NODE_WORKER_COMPLETE"
        for path in worker_files
    )
    if complete != expected_workers:
        raise RuntimeError(
            f"Stage-X worker count differs: {complete}/{expected_workers}"
        )
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/aggregate_topic4_rev12_cascade_fit.py"),
        "--config", str(stage_config), "--seed-pool", "fit",
        "--artifact-root", str(artifact_root),
    ], cwd=ROOT, check=True)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/analyze_topic4_rev12_paired_field_replication.py"),
        "--config", str(analysis_config),
        "--expected-commit", str(args.expected_commit),
        "--artifact-root", str(artifact_root),
    ], cwd=ROOT, check=True)
    analysis_path = artifact_root / json.loads(analysis_config.read_text())["output"]
    analysis = json.loads(analysis_path.read_text())
    final = output_root / "status" / "postprocess.json"
    final.write_text(json.dumps({
        "status": "REV12ND_STAGE_X_POSTPROCESS_COMPLETE",
        "controller": str(controller),
        "aggregate": str(output_root / "aggregate" / "fit_cascade_summary.json"),
        "analysis": str(analysis_path),
        "decision": analysis["decision"],
        "failed_aggregate_endpoints": analysis["failed_aggregate_endpoints"],
        "failed_sign_endpoints": analysis["failed_sign_endpoints"],
    }, indent=2) + "\n")
    print(json.dumps(json.loads(final.read_text()), indent=2))


if __name__ == "__main__":
    main()
