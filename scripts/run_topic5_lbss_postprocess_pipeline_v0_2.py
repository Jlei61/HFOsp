#!/usr/bin/env python3
"""Wait for formal LBSS training and advance the immutable postprocess chain."""
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
    log_root = out / "run_logs" / "postprocess"; log_root.mkdir(parents=True, exist_ok=True)
    try:
        while not (out / "FORMAL_TRAINING_COMPLETE.json").exists():
            if (out / "FORMAL_TRAINING_FAILED.json").exists():
                raise RuntimeError("formal training failed; postprocess will not start")
            complete = len(list((out / "per_fit").glob("*/*/seed*/DONE.json")))
            atomic(out / "POSTPROCESS_WAIT_STATUS.json", {
                "status": "WAITING_FOR_FORMAL_TRAINING", "complete_units": complete,
                "scheduled_units": 465, "updated_at": now(), "pid": os.getpid(),
            })
            time.sleep(min(max(5, args.poll_seconds), 30))

        steps = (
            ("D_interictal", "INTERICTAL_ANALYSIS_COMPLETE.json",
             ["scripts/analyse_topic5_lbss_interictal_v0_2.py"]),
            ("E_fields", "MODEL_FIELDS_FROZEN.json",
             ["scripts/build_topic5_lbss_fields_v0_2.py"]),
            ("E_pathways", "PATHWAY_ANALYSIS_COMPLETE.json",
             ["scripts/analyse_topic5_lbss_pathways_v0_2.py", "--device", "cuda:0", "--workers", "6"]),
            ("F_attenuation", "ATTENUATION_COMPLETE.json",
             ["scripts/run_topic5_lbss_attenuation_v0_2.py", "--device", "cuda:0", "--workers", "6"]),
            ("G_authorize", "TARGET_UNSEAL_AUTHORIZATION.json",
             ["scripts/prepare_topic5_lbss_early_ictal_unseal_v0_2.py"]),
            ("G_early_ictal", "EARLY_ICTAL_SCORING_COMPLETE.json",
             ["scripts/score_topic5_lbss_early_ictal_v0_2.py", "--n-perm", "5000"]),
            ("G_claims", "LBSS_CLAIM_ADJUDICATION_COMPLETE.json",
             ["scripts/summarize_topic5_lbss_claims_v0_2.py"]),
            ("H_figure", "figures/topic5_figure6_lbss_rnn.png",
             ["scripts/plot_topic5_lbss_figure6_v0_2.py"]),
        )
        completed = []
        for label, marker, relative in steps:
            marker_path = out / marker
            if marker_path.exists():
                completed.append({"step": label, "status": "already_complete", "marker": str(marker_path)})
                continue
            command = [args.python, str(snapshot / relative[0]), "--out-root", str(out), *relative[1:]]
            started = time.time()
            atomic(out / "POSTPROCESS_STATUS.json", {
                "status": "RUNNING", "step": label, "command": command,
                "completed_steps": completed, "started_at": now(), "pid": os.getpid(),
            })
            log = log_root / f"{label}.log"
            with log.open("w") as stream:
                process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, text=True)
            result = {
                "step": label, "returncode": process.returncode,
                "seconds": round(time.time() - started, 2), "log": str(log),
                "marker": str(marker_path), "marker_exists": marker_path.exists(),
            }
            completed.append(result)
            if process.returncode != 0 or not marker_path.exists():
                atomic(out / "PIPELINE_FAILED.json", {
                    "status": "FAILED", "failed_step": label, "completed_steps": completed,
                    "updated_at": now(), "snapshot": str(snapshot),
                })
                raise RuntimeError(f"postprocess failed at {label}; see {log}")
        atomic(out / "PIPELINE_COMPLETE.json", {
            "status": "COMPLETE", "completed_steps": completed, "updated_at": now(),
            "target_values_read": True, "snapshot": str(snapshot),
        })
        (out / "PIPELINE_FAILED.json").unlink(missing_ok=True)
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
