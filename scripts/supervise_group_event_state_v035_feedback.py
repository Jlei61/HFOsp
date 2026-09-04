#!/usr/bin/env python3
"""Persistent CPU W6 queue; depends only on frozen W3 trajectories."""

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
SUBJECTS = ("epilepsiae_253", "epilepsiae_548", "epilepsiae_583", "epilepsiae_1146")


def jobs() -> list[dict]:
    out = []
    for subject in SUBJECTS:
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
            dep = OUTPUT_ROOT / "full_mark_state" / subject / f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "state_trajectory.npz"
            card = OUTPUT_ROOT / "feedback_models" / subject / f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "card.json"
            out.append({"subject": subject, "decoder_seed": decoder_seed, "state_seed": state_seed,
                        "dependency": str(dep), "out": str(card)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--workers", type=int, default=2); args = ap.parse_args()
    root = OUTPUT_ROOT / "feedback_supervisor"; logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    scope = OUTPUT_ROOT / "scope_manifest.json"; update_scope_manifest(scope, "W6", "RUNNING", [])
    pending, running, complete, failed = jobs(), {}, [], []
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): env[name] = "1"
    while pending or running:
        for slot, row in list(running.items()):
            code = row["process"].poll()
            if code is None: continue
            row["handle"].close(); job = row["job"]
            if code == 0 and Path(job["out"]).is_file(): complete.append(job)
            else: failed.append({**job, "returncode": code, "log": row["log"]})
            del running[slot]
        upstream_done = (OUTPUT_ROOT / "functional_supervisor" / "queue_done.json").exists()
        for slot in range(args.workers):
            if slot in running: continue
            chosen = None
            for i, job in enumerate(pending):
                if Path(job["out"]).is_file(): complete.append(job); chosen = (i, None); break
                if Path(job["dependency"]).is_file(): chosen = (i, job); break
                if upstream_done:
                    failed.append({**job, "returncode": None, "log": None,
                                   "reason": "exact frozen W3 trajectory unavailable"})
                    chosen = (i, None); break
            if chosen is None: continue
            i, job = chosen; pending.pop(i)
            if job is None: continue
            log = logs / f"{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_slot{slot}.log"
            handle = log.open("a", encoding="utf-8")
            cmd = [str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_feedback_models.py"),
                   "--subject", job["subject"], "--decoder-seed", str(job["decoder_seed"]),
                   "--state-seed", str(job["state_seed"])]
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            running[slot] = {"job": job, "process": process, "handle": handle,
                             "log": str(log), "started": time.time()}
        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_feedback_queue_v1", "updated_epoch": time.time(),
            "pending": len(pending), "complete": len(complete), "failed": failed,
            "running": {str(slot): {"pid": row["process"].pid, "job": row["job"], "log": row["log"],
                                    "elapsed_seconds": time.time() - row["started"]} for slot, row in running.items()},
        })
        if pending or running: time.sleep(15)
    evidence = [job["out"] for job in complete]
    update_scope_manifest(scope, "W6", "COMPLETE" if len(complete) == len(jobs()) else "PARTIAL", evidence)
    atomic_json(root / "queue_done.json", {"format": "group_event_state_v0_3_5_feedback_done_v1",
                "complete": complete, "failed": failed})


if __name__ == "__main__": main()
