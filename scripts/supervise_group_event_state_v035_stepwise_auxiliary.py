#!/usr/bin/env python3
"""Persistent OOM-safe queue for W4 frozen per-step auxiliary heads."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v035.contracts import LOCKED_SEEDS, OUTPUT_ROOT, atomic_json, update_scope_manifest  # noqa: E402

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SUBJECTS = ("epilepsiae_253", "epilepsiae_548", "epilepsiae_583")


def jobs() -> list[dict]:
    output = []
    for subject in SUBJECTS:
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
            dep = (OUTPUT_ROOT / "full_mark_state" / subject /
                   f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "state_trajectory.npz")
            card = (OUTPUT_ROOT / "stepwise_auxiliary" / subject /
                    f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "card.json")
            output.append({"subject": subject, "decoder_seed": decoder_seed, "state_seed": state_seed,
                           "dependency": str(dep), "out": str(card), "batch_events": 96,
                           "retries": 0})
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="0,1")
    args = ap.parse_args()
    gpus = [v.strip() for v in args.gpus.split(",") if v.strip()]
    root = OUTPUT_ROOT / "stepwise_auxiliary_supervisor"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    scope = OUTPUT_ROOT / "scope_manifest.json"
    update_scope_manifest(scope, "W4", "RUNNING", [])
    pending, running, complete, failed = jobs(), {}, [], []
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    while pending or running:
        for gpu, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["handle"].close()
            job = row["job"]
            log = Path(row["log"])
            if code == 0 and Path(job["out"]).is_file():
                complete.append(job)
            else:
                body = log.read_text(encoding="utf-8", errors="replace")[-20000:] if log.exists() else ""
                if "out of memory" in body.lower() and job["batch_events"] > 12 and job["retries"] < 3:
                    job["batch_events"] //= 2
                    job["retries"] += 1
                    pending.insert(0, job)
                else:
                    failed.append({**job, "returncode": code, "log": str(log), "tail": body[-3000:]})
            del running[gpu]
        for gpu in gpus:
            if gpu in running:
                continue
            chosen = None
            for i, job in enumerate(pending):
                if Path(job["out"]).is_file():
                    complete.append(job)
                    chosen = (i, None)
                    break
                if Path(job["dependency"]).is_file():
                    chosen = (i, job)
                    break
            if chosen is None:
                continue
            i, job = chosen
            pending.pop(i)
            if job is None:
                continue
            log = logs / f"{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_gpu{gpu}.log"
            handle = log.open("a", encoding="utf-8")
            command = [str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_stepwise_auxiliary.py"),
                       "--subject", job["subject"], "--decoder-seed", str(job["decoder_seed"]),
                       "--state-seed", str(job["state_seed"]), "--batch-events", str(job["batch_events"]),
                       "--device", f"cuda:{gpu}"]
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            running[gpu] = {"job": job, "process": process, "handle": handle,
                            "log": str(log), "started": time.time()}
        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_stepwise_auxiliary_queue_v1",
            "updated_epoch": time.time(), "pending": len(pending), "complete": len(complete),
            "failed": failed,
            "running": {gpu: {"pid": row["process"].pid, "job": row["job"], "log": row["log"],
                              "elapsed_seconds": time.time() - row["started"]}
                        for gpu, row in running.items()},
        })
        if pending or running:
            time.sleep(15)
    evidence = [job["out"] for job in complete]
    update_scope_manifest(scope, "W4", "COMPLETE" if len(complete) == len(jobs()) else "PARTIAL", evidence)
    atomic_json(root / "queue_done.json", {
        "format": "group_event_state_v0_3_5_stepwise_auxiliary_done_v1",
        "complete": complete, "failed": failed,
    })


if __name__ == "__main__":
    main()
