#!/usr/bin/env python3
"""Wait coarsely for Stage-U, then aggregate and analyze it once."""
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
    stage = json.loads(args.stage_config.resolve().read_text())
    output_root = args.artifact_root.resolve() / stage["output_root"]
    controller = output_root / "status" / "fit_controller.json"
    wait_seconds = max(60, int(args.wait_seconds))
    while not controller.exists():
        time.sleep(wait_seconds)
    status = json.loads(controller.read_text())
    if status.get("status") != "REV12ND_NODE_WORKER_QUEUE_COMPLETE":
        raise RuntimeError("Stage-U controller ended without queue completion")
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/aggregate_topic4_rev12_cascade_fit.py"),
        "--config", str(args.stage_config.resolve()), "--seed-pool", "fit",
        "--artifact-root", str(args.artifact_root.resolve()),
    ], cwd=ROOT, check=True)
    subprocess.run([
        str(PYTHON), str(ROOT / "scripts/analyze_topic4_rev12_orthogonal_free_field_screen.py"),
        "--config", str(args.analysis_config.resolve()),
        "--expected-commit", str(args.expected_commit),
        "--artifact-root", str(args.artifact_root.resolve()),
    ], cwd=ROOT, check=True)
    final = output_root / "status" / "postprocess.json"
    final.write_text(json.dumps({
        "status": "REV12ND_STAGE_U_POSTPROCESS_COMPLETE",
        "controller": str(controller),
        "aggregate": str(output_root / "aggregate" / "fit_cascade_summary.json"),
        "analysis": str(output_root / "analysis" / "orthogonal_mode_response.json"),
    }, indent=2) + "\n")
    print(json.dumps(json.loads(final.read_text()), indent=2))


if __name__ == "__main__":
    main()
