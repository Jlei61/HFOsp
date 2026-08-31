#!/usr/bin/env python3
"""Wait without busy polling, then aggregate the completed rev17 atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
CONFIG = ROOT / "config/topic4_rev17_dual_field_residual_atlas.json"
CONTROLLER = "codex-t4-r17-dual-atlas-controller-586bb11b.service"


def _state(unit: str) -> str:
    result = subprocess.run(
        ["systemctl", "--user", "show", "-p", "ActiveState", "--value", unit],
        text=True, capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "inactive"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--controller", default=CONTROLLER)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    expected = int(config["dual_field_residual"]["expected_candidate_count"]) * len(
        config["search"]["fit_network_seeds"]
    )
    worker_root = ARTIFACT_ROOT / config["output_root"] / "workers"
    while True:
        present = len(list(worker_root.glob("*.json"))) if worker_root.exists() else 0
        state = _state(args.controller)
        print(json.dumps({
            "controller_state": state, "present_json": present,
            "expected_json": expected,
        }), flush=True)
        if state not in {"active", "activating"}:
            if present != expected:
                raise RuntimeError(
                    f"rev17 controller exited with {present}/{expected} worker JSONs"
                )
            break
        time.sleep(max(60, int(args.interval_seconds)))
    subprocess.run([
        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python",
        str(ROOT / "scripts/aggregate_topic4_rev17_dual_field_residual_atlas.py"),
        "--config", str(CONFIG), "--artifact-root", str(ARTIFACT_ROOT),
    ], cwd=ROOT, check=True)
    print(json.dumps({
        "status": "REV17_ATLAS_WAIT_AND_AGGREGATE_COMPLETE",
        "present_json": expected,
    }), flush=True)


if __name__ == "__main__":
    main()
