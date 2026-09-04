#!/usr/bin/env python3
"""Persistent OOM-safe W3 queue; starts each real unit when its W2 input exists."""

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
    LOCKED_SEEDS, OUTPUT_ROOT, atomic_json, initialise_scope_manifest, update_scope_manifest,
)

PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SUBJECTS = ("epilepsiae_253", "epilepsiae_548", "epilepsiae_583", "epilepsiae_1146", "epilepsiae_922")


def jobs() -> list[dict]:
    out = []
    for subject in SUBJECTS:
        for decoder_seed, state_seed in enumerate(LOCKED_SEEDS[:3]):
            dep = OUTPUT_ROOT / "stepwise_decoder" / subject / f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "adapter.pt"
            card = OUTPUT_ROOT / "full_mark_state" / subject / f"decoder_seed{decoder_seed}_state_seed{state_seed}" / "card.json"
            out.append({"subject": subject, "decoder_seed": decoder_seed, "state_seed": state_seed,
                        "dependency": str(dep), "out": str(card), "chunk_events": 256, "retries": 0})
    return out


def command(job: dict, gpu: str) -> list[str]:
    return [str(PYTHON), str(ROOT / "scripts/run_group_event_state_v035_full_mark_state.py"),
            "--subject", job["subject"], "--decoder-seed", str(job["decoder_seed"]),
            "--state-seed", str(job["state_seed"]), "--chunk-events", str(job["chunk_events"]),
            "--device", f"cuda:{gpu}"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--gpus", default="0,1"); args = ap.parse_args()
    gpus = [v.strip() for v in args.gpus.split(",") if v.strip()]
    root = OUTPUT_ROOT / "full_mark_supervisor"; logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    scope = OUTPUT_ROOT / "scope_manifest.json"
    if not scope.exists(): initialise_scope_manifest(scope)
    update_scope_manifest(scope, "W3", "RUNNING", [])
    pending, running, complete, failed = jobs(), {}, [], []
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): env[name] = "1"
    while pending or running:
        for gpu, row in list(running.items()):
            code = row["process"].poll()
            if code is None: continue
            row["handle"].close(); job = row["job"]; log = Path(row["log"])
            if code == 0 and Path(job["out"]).is_file():
                complete.append(job)
            else:
                text = log.read_text(encoding="utf-8", errors="replace")[-20000:] if log.exists() else ""
                if "out of memory" in text.lower() and job["chunk_events"] > 32 and job["retries"] < 3:
                    job["chunk_events"] //= 2; job["retries"] += 1; pending.insert(0, job)
                else:
                    failed.append({**job, "returncode": code, "log": str(log), "tail": text[-2000:]})
            del running[gpu]
        parent_done = (OUTPUT_ROOT / "supervisor" / "queue_done.json").exists()
        for gpu in gpus:
            if gpu in running: continue
            chosen = None
            for i, job in enumerate(pending):
                if Path(job["out"]).is_file(): complete.append(job); chosen = (i, None); break
                if Path(job["dependency"]).is_file(): chosen = (i, job); break
                if parent_done:
                    failed.append({**job, "returncode": None, "log": None, "tail": "W2 dependency unavailable after parent queue ended"})
                    chosen = (i, None); break
            if chosen is None: continue
            i, job = chosen; pending.pop(i)
            if job is None: continue
            log = logs / f"{job['subject']}_decoder{job['decoder_seed']}_state{job['state_seed']}_gpu{gpu}.log"
            handle = log.open("a", encoding="utf-8")
            process = subprocess.Popen(command(job, gpu), cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            running[gpu] = {"job": job, "process": process, "handle": handle, "log": str(log), "started": time.time()}
        atomic_json(root / "queue_state.json", {
            "format": "group_event_state_v0_3_5_full_mark_queue_v1", "updated_epoch": time.time(),
            "pending": len(pending), "complete": len(complete), "failed": failed,
            "running": {gpu: {"pid": row["process"].pid, "job": row["job"], "log": row["log"],
                              "elapsed_seconds": time.time() - row["started"]} for gpu, row in running.items()},
        })
        if pending or running: time.sleep(10)
    evidence = [job["out"] for job in complete]
    update_scope_manifest(scope, "W3", "COMPLETE" if len(complete) == len(jobs()) else "PARTIAL", evidence)
    atomic_json(root / "queue_done.json", {"format": "group_event_state_v0_3_5_full_mark_done_v1",
                "complete": complete, "failed": failed})


if __name__ == "__main__": main()
