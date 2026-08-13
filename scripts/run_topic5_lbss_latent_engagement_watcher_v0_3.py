#!/usr/bin/env python3
"""Wait for formal LBSS training, then run the target-free zero-H audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    snapshot = args.snapshot.resolve()
    status_path = out_root / "LATENT_ENGAGEMENT_WAIT_STATUS.json"
    failure = out_root / "LATENT_ENGAGEMENT_FAILED.json"

    primary = out_root
    while not (primary / "PIPELINE_COMPLETE.json").exists():
        pointer = out_root / "PRIMARY_ARTIFACT_POINTER.json"
        if pointer.exists():
            primary = Path(json.loads(pointer.read_text())["artifact_root"]).resolve()
        if (out_root / "SPATIAL_DECISION_FAILED.json").exists():
            raise RuntimeError("spatial decision failed before latent engagement audit")
        if (primary / "PIPELINE_FAILED.json").exists() or (out_root / "FORMAL_TRAINING_FAILED.json").exists():
            raise RuntimeError("primary pipeline failed before latent engagement audit")
        done = len(list((out_root / "per_fit").glob("*/*/seed*/DONE.json")))
        failed = len(list((out_root / "per_fit").glob("*/*/seed*/FAILED.json")))
        write_json(status_path, {
            "status": "WAITING_FOR_PRIMARY_PIPELINE",
            "done": done,
            "scheduled": 465,
            "failed": failed,
            "target_values_read": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "primary_artifact_root": str(primary),
        })
        time.sleep(max(5, int(args.poll_seconds)))

    decision_path = out_root / "SPATIAL_DECISION_COMPLETE.json"
    if not decision_path.exists():
        raise RuntimeError("primary pipeline completed without a spatial-model decision")
    decision = json.loads(decision_path.read_text())
    model_snapshot = (
        Path(decision["search_snapshot"]).resolve()
        if decision.get("selected_contract") == "FULL_COHORT_SELECTED_SPATIAL_CONFIG"
        else out_root / "run_snapshot"
    )

    command = [
        sys.executable,
        str(snapshot / "analyse_topic5_lbss_latent_engagement_v0_3.py"),
        "--out-root", str(primary),
        "--model-snapshot", str(model_snapshot),
        "--arms", "L3_LOCAL_PLUS_LEARNED_LR",
        "--device", args.device,
    ]
    write_json(status_path, {
        "status": "RUNNING",
        "command": command,
        "primary_artifact_root": str(primary),
        "target_values_read": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    result = subprocess.run(command, text=True)
    if result.returncode != 0:
        write_json(failure, {
            "status": "FAIL",
            "returncode": int(result.returncode),
            "command": command,
            "target_values_read": False,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        raise SystemExit(result.returncode)
    write_json(status_path, {
        "status": "COMPLETE",
        "primary_artifact_root": str(primary),
        "target_values_read": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
