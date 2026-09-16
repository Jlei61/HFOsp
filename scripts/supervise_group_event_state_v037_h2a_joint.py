#!/usr/bin/env python3
"""Run the preregistered event-only H2a joint sensitivity on 4x5 cells."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RUNNER = ROOT / "scripts/run_group_event_state_v037_h2a_joint.py"
SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--h1-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h1_shared_equal_horizon"))
    parser.add_argument("--primary-h2a-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2a_frozen_decoder_equal_horizon"))
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h2a_joint_sensitivity"))
    args = parser.parse_args()
    gpus = tuple(int(v) for v in args.gpus.split(",") if v.strip())
    jobs = [(subject, seed) for subject in SUBJECTS for seed in SEEDS]
    pending = [job for job in jobs if not (args.out_root / job[0] / f"seed{job[1]}" / "card.json").exists()]
    running = {}; failures = []
    logs = args.out_root / "supervisor/logs"; logs.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for slot in list(running):
            process, job, handle = running[slot]
            code = process.poll()
            if code is None:
                continue
            handle.close(); del running[slot]
            expected = args.out_root / job[0] / f"seed{job[1]}" / "card.json"
            if code != 0 or not expected.exists():
                failures.append({"job": job, "returncode": code})
        for slot in range(int(args.workers)):
            if slot in running or not pending:
                continue
            subject, seed = pending.pop(0); gpu = gpus[slot % len(gpus)]
            handle = (logs / f"{subject}__seed{seed}.log").open("a", encoding="utf-8")
            cmd = [str(PYTHON), str(RUNNER), "--subject", subject, "--seed", str(seed),
                   "--device", f"cuda:{gpu}", "--out-root", str(args.out_root),
                   "--h1-root", str(args.h1_root),
                   "--primary-h2a-root", str(args.primary_h2a_root)]
            env = dict(os.environ); env.setdefault("OMP_NUM_THREADS", "2")
            running[slot] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT), (subject, seed), handle)
        _write(args.out_root / "supervisor/queue_status.json", {
            "format": "group_event_state_v0_3_7_h2a_joint_queue_v1",
            "status": "FAILED" if failures else ("COMPLETE" if not pending and not running else "RUNNING"),
            "total": len(jobs), "complete": len(jobs) - len(pending) - len(running) - len(failures),
            "pending": len(pending),
            "running": [{"slot": slot, "gpu": gpus[slot % len(gpus)], "pid": item[0].pid,
                         "subject": item[1][0], "seed": item[1][1]} for slot, item in sorted(running.items())],
            "failures": failures, "scope": "event-only supportive sensitivity",
            "mandatory_h1_reevaluation_in_every_card": True,
            "development_targets_read": False, "seizure_targets_read": False,
            "sealed_partition_opened": False,
        })
        if failures:
            for process, _job, handle in running.values(): process.terminate(); handle.close()
            raise SystemExit(1)
        time.sleep(20.0)


if __name__ == "__main__":
    main()
