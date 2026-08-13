#!/usr/bin/env python3
"""Wait for v0.3 training, then advance the immutable postprocess chain."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--python", default="/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--through-target",
        action="store_true",
        help=(
            "Continue from frozen interictal/field/attenuation artifacts into "
            "target authorization and early-ictal scoring. By default the "
            "pipeline stops before target access so target-free model selection "
            "can be completed first."
        ),
    )
    args = parser.parse_args()
    out = args.out_root.resolve()
    snapshot = Path(__file__).resolve().parents[1]
    lock = out / "POSTPROCESS_PIPELINE.lock"
    if lock.exists():
        payload = json.loads(lock.read_text())
        try:
            os.kill(int(payload["pid"]), 0)
        except (ProcessLookupError, PermissionError, KeyError, ValueError):
            lock.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"postprocess pipeline already active: {payload}")
    atomic(lock, {"pid": os.getpid(), "created_at": now(), "snapshot": str(snapshot)})
    log_root = out / "run_logs" / "postprocess_v0_3"
    log_root.mkdir(parents=True, exist_ok=True)
    try:
        while not (out / "FORMAL_TRAINING_COMPLETE.json").exists():
            if (out / "FORMAL_TRAINING_FAILED.json").exists():
                raise RuntimeError("formal training failed; postprocess will not start")
            complete = len(list((out / "per_fit").glob("*/*/seed*/DONE.json")))
            failed = len(list((out / "per_fit").glob("*/*/seed*/FAILED.json")))
            oom = len(list((out / "per_fit").glob("*/*/seed*/OOM.json")))
            atomic(out / "POSTPROCESS_WAIT_STATUS.json", {
                "status": "WAITING_FOR_FORMAL_TRAINING",
                "complete_units": complete,
                "failed_units": failed,
                "oom_units": oom,
                "scheduled_units": 465,
                "updated_at": now(),
                "pid": os.getpid(),
            })
            time.sleep(min(max(5, args.poll_seconds), 30))

        pretarget_steps = (
            ("D_interictal", "INTERICTAL_ANALYSIS_COMPLETE.json",
             ["scripts/analyse_topic5_lbss_full_tissue_interictal_v0_3.py"]),
            ("E_fields", "MODEL_FIELDS_FROZEN.json",
             ["scripts/build_topic5_lbss_fields_v0_2.py"]),
            ("E_pathways", "PATHWAY_ANALYSIS_COMPLETE.json",
             ["scripts/analyse_topic5_lbss_pathways_v0_2.py", "--device", "cuda:0", "--workers", "4",
              "--representative", "epilepsiae_1146"]),
            ("F_attenuation", "ATTENUATION_COMPLETE.json",
             ["scripts/run_topic5_lbss_attenuation_v0_2.py", "--device", "cuda:0", "--workers", "12"]),
        )
        target_steps = (
            ("G_authorize", "TARGET_UNSEAL_AUTHORIZATION.json",
             ["scripts/prepare_topic5_lbss_full_tissue_target_unseal_v0_3.py"]),
            ("G_early_ictal", "EARLY_ICTAL_SCORING_COMPLETE.json",
             ["scripts/score_topic5_lbss_full_tissue_early_ictal_v0_3.py", "--n-perm", "1000"]),
        )
        completed = []
        for label, marker, relative in pretarget_steps:
            marker_path = out / marker
            if marker_path.exists():
                completed.append({"step": label, "status": "already_complete", "marker": str(marker_path)})
                continue
            command = [args.python, str(snapshot / relative[0]), "--out-root", str(out), *relative[1:]]
            started = time.time()
            atomic(out / "POSTPROCESS_STATUS.json", {
                "status": "RUNNING",
                "step": label,
                "command": command,
                "completed_steps": completed,
                "started_at": now(),
                "pid": os.getpid(),
            })
            log = log_root / f"{label}.log"
            with log.open("w") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
            result = {
                "step": label,
                "returncode": process.returncode,
                "seconds": round(time.time() - started, 2),
                "log": str(log),
                "marker": str(marker_path),
                "marker_exists": marker_path.exists(),
            }
            completed.append(result)
            if process.returncode != 0 or not marker_path.exists():
                atomic(out / "PIPELINE_FAILED.json", {
                    "status": "FAILED",
                    "failed_step": label,
                    "completed_steps": completed,
                    "updated_at": now(),
                    "snapshot": str(snapshot),
                })
                raise RuntimeError(f"postprocess failed at {label}; see {log}")

        atomic(out / "INTERICTAL_POSTPROCESS_PRETARGET_COMPLETE.json", {
            "status": "COMPLETE",
            "completed_steps": completed,
            "updated_at": now(),
            "target_values_read": False,
            "snapshot": str(snapshot),
        })
        if not args.through_target:
            atomic(out / "POSTPROCESS_STATUS.json", {
                "status": "PAUSED_BEFORE_TARGET",
                "reason": "TARGET_FREE_SPATIAL_MODEL_DECISION_REQUIRED",
                "completed_steps": completed,
                "updated_at": now(),
                "target_values_read": False,
                "pid": os.getpid(),
            })
            return

        for label, marker, relative in target_steps:
            marker_path = out / marker
            if marker_path.exists():
                completed.append({"step": label, "status": "already_complete", "marker": str(marker_path)})
                continue
            command = [args.python, str(snapshot / relative[0]), "--out-root", str(out), *relative[1:]]
            started = time.time()
            atomic(out / "POSTPROCESS_STATUS.json", {
                "status": "RUNNING",
                "step": label,
                "command": command,
                "completed_steps": completed,
                "started_at": now(),
                "pid": os.getpid(),
            })
            log = log_root / f"{label}.log"
            with log.open("w") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
            result = {
                "step": label,
                "returncode": process.returncode,
                "seconds": round(time.time() - started, 2),
                "log": str(log),
                "marker": str(marker_path),
                "marker_exists": marker_path.exists(),
            }
            completed.append(result)
            if process.returncode != 0 or not marker_path.exists():
                atomic(out / "PIPELINE_FAILED.json", {
                    "status": "FAILED",
                    "failed_step": label,
                    "completed_steps": completed,
                    "updated_at": now(),
                    "snapshot": str(snapshot),
                })
                raise RuntimeError(f"postprocess failed at {label}; see {log}")
        atomic(out / "PIPELINE_COMPLETE.json", {
            "status": "COMPLETE",
            "completed_steps": completed,
            "updated_at": now(),
            "target_values_read": True,
            "snapshot": str(snapshot),
        })
        (out / "PIPELINE_FAILED.json").unlink(missing_ok=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
