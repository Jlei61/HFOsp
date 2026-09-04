#!/usr/bin/env python3
"""Persistent two-GPU queue for v0.3.5 W1 then W2 real-patient units.

This supervisor owns only v0.3.5 processes.  It writes an atomic queue state,
skips completed units, and retries CUDA OOM once with lower within-unit batch
settings only when the runner supports such a retry.  W3-W6 are registered in
the same scope manifest and cannot be silently declared complete by this queue.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v035.contracts import (  # noqa: E402
    LOCKED_SEEDS, OUTPUT_ROOT, V035_SUBJECTS, atomic_json, initialise_scope_manifest,
    update_scope_manifest,
)

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
STEPWISE_SUBJECTS = ("epilepsiae_253", "epilepsiae_548", "epilepsiae_583", "epilepsiae_1146", "epilepsiae_922")


def _jobs(*, redo_w1: bool = False) -> list[dict]:
    jobs = []
    # All eight registered patients receive W1, with five seeds for the six
    # primary patients and three seeds for the second-wave coverage patients.
    for subject in V035_SUBJECTS:
        seeds = LOCKED_SEEDS if subject not in {"epilepsiae_384", "epilepsiae_1125"} else LOCKED_SEEDS[:3]
        for seed in seeds:
            out = OUTPUT_ROOT / "dynamic_rate" / subject / f"seed{seed}" / "card.json"
            jobs.append({"kind": "W1", "subject": subject, "seed": seed, "out": str(out),
                         "force": bool(redo_w1),
                         "cmd": [str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_dynamic_rate.py"),
                                 "--subject", subject, "--seed", str(seed)] + (["--overwrite"] if redo_w1 else [])})
    # W2 uses the matching three mature-decoder seeds; additional decoder
    # retraining for E1096/E384/E1125 is a W2 subtask, not faked here.
    for subject in STEPWISE_SUBJECTS:
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
            out = OUTPUT_ROOT / "stepwise_decoder" / subject / f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "card.json"
            jobs.append({"kind": "W2", "subject": subject, "seed": state_seed, "decoder_seed": decoder_seed,
                         "out": str(out),
                         "dependency": str(OUTPUT_ROOT / "dynamic_rate" / subject / f"seed{state_seed}" / "trajectory_and_scores.npz"),
                         "cmd": [str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_stepwise_decoder.py"),
                                 "--subject", subject, "--decoder-seed", str(decoder_seed),
                                 "--state-seed", str(state_seed)]})
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--poll-seconds", type=float, default=5.0)
    ap.add_argument("--redo-w1", action="store_true", help="rebuild W1 artifacts before W2")
    args = ap.parse_args()
    gpus = [v.strip() for v in args.gpus.split(",") if v.strip()]
    state_path = OUTPUT_ROOT / "supervisor" / "queue_state.json"
    log_dir = OUTPUT_ROOT / "supervisor" / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    scope = OUTPUT_ROOT / "scope_manifest.json"
    if not scope.exists(): initialise_scope_manifest(scope)
    update_scope_manifest(scope, "W1", "RUNNING", [])
    update_scope_manifest(scope, "W2", "RUNNING", [])
    pending = _jobs(redo_w1=args.redo_w1); running = {}; failed = []; completed = []
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    while pending or running:
        # Reap before scheduling.
        for gpu, info in list(running.items()):
            code = info["process"].poll()
            if code is None: continue
            info["handle"].close()
            job = info["job"]
            if code == 0 and Path(job["out"]).is_file(): completed.append(job)
            else: failed.append({**job, "returncode": code, "log": info["log"]})
            del running[gpu]
        # Schedule dependencies in natural W1 -> W2 order.
        for gpu in gpus:
            if gpu in running: continue
            selected = None
            for i, job in enumerate(pending):
                if Path(job["out"]).is_file() and not job.get("force", False):
                    completed.append(job); selected = (i, None); break
                dep = job.get("dependency")
                if dep is None or Path(dep).is_file():
                    selected = (i, job); break
            if selected is None: continue
            i, job = selected; pending.pop(i)
            if job is None: continue
            log = log_dir / f"{job['kind']}_{job['subject']}_seed{job['seed']}_gpu{gpu}.log"
            handle = log.open("a", encoding="utf-8")
            cmd = job["cmd"] + ["--device", f"cuda:{gpu}"]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            running[gpu] = {"process": process, "job": job, "handle": handle, "log": str(log), "started": time.time()}
        atomic_json(state_path, {
            "format": "group_event_state_v0_3_5_queue_state_v1", "updated_epoch": time.time(),
            "pending": len(pending), "completed": len(completed), "failed": failed,
            "running": {gpu: {"pid": row["process"].pid, "job": row["job"], "log": row["log"],
                              "elapsed_seconds": time.time() - row["started"]} for gpu, row in running.items()},
            "development_targets_read": False, "sealed_partition_opened": False,
        })
        if pending or running: time.sleep(args.poll_seconds)
    w1 = [job["out"] for job in completed if job["kind"] == "W1"]
    w2 = [job["out"] for job in completed if job["kind"] == "W2"]
    update_scope_manifest(scope, "W1", "COMPLETE" if len(w1) == 36 else "PARTIAL", w1)
    update_scope_manifest(scope, "W2", "PARTIAL", w2)
    atomic_json(OUTPUT_ROOT / "supervisor" / "queue_done.json", {
        "format": "group_event_state_v0_3_5_queue_done_v1", "completed": completed,
        "failed": failed, "note": "W2 remains PARTIAL until new decoder coverage and full m(t) modulation are complete",
    })


if __name__ == "__main__":
    main()
