#!/usr/bin/env python3
"""Detached multi-GPU supervisor for formal v0.3.7 H1 units."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RUNNER = REPO / "scripts/run_group_event_state_v037_h1.py"
ROOT = Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon")
SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    args = parser.parse_args()
    supervisor = ROOT / "supervisor"
    log_root = supervisor / "logs"; log_root.mkdir(parents=True, exist_ok=True)
    # Interleave patients so an expensive high-contact representation build
    # cannot monopolise both devices at queue start.
    pending = [(subject, seed) for seed in SEEDS for subject in SUBJECTS]
    slots = [(gpu, slot) for gpu in (0, 1) for slot in range(args.workers_per_gpu)]
    running: dict[tuple[int, int], tuple[subprocess.Popen, str, int, object]] = {}
    failures: list[dict] = []
    complete = 0
    while pending or running:
        for key in list(running):
            process, subject, seed, handle = running[key]
            code = process.poll()
            if code is None: continue
            handle.close(); del running[key]
            if code == 0 and (ROOT / subject / f"seed{seed}" / "card.json").exists():
                complete += 1
            else:
                failures.append({"subject": subject, "seed": seed, "returncode": code, "physical_gpu": key[0]})
        for key in slots:
            if key in running or not pending: continue
            subject, seed = pending.pop(0)
            log = (log_root / f"{subject}__seed{seed}.log").open("a", encoding="utf-8")
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(key[0])
            env.setdefault("OMP_NUM_THREADS", "2")
            command = [
                str(PYTHON), str(RUNNER), "--subject", subject, "--seed", str(seed),
                "--device", "cuda:0", "--out-root", str(ROOT),
            ]
            process = subprocess.Popen(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[key] = (process, subject, seed, log)
        write_json(supervisor / "queue_status.json", {
            "format": "group_event_state_v0_3_7_h1_equal_horizon_queue_v2",
            "status": "RUNNING" if pending or running else ("FAILED" if failures else "COMPLETE"),
            "total": len(SUBJECTS) * len(SEEDS),
            "complete": complete,
            "pending": len(pending),
            "running": [
                {"subject": subject, "seed": seed, "pid": process.pid,
                 "physical_gpu": gpu, "slot_on_gpu": slot}
                for (gpu, slot), (process, subject, seed, _handle) in sorted(running.items())
            ],
            "failures": failures,
            "development_targets_read": False,
            "seizure_targets_read": False,
            "sealed_partition_opened": False,
        })
        if pending or running: time.sleep(args.poll_seconds)
    write_json(supervisor / "queue_status.json", {
        "format": "group_event_state_v0_3_7_h1_equal_horizon_queue_v2",
        "status": "FAILED" if failures else "COMPLETE",
        "total": len(SUBJECTS) * len(SEEDS), "complete": complete,
        "pending": 0, "running": [], "failures": failures,
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__": main()
