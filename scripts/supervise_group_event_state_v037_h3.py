#!/usr/bin/env python3
"""Wait for the H3 instrument marker, then schedule human M0/M1/M2 fits."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RUN = ROOT / "scripts/run_group_event_state_v037_h3.py"
SUBJECTS = ("epilepsiae_253", "epilepsiae_958", "epilepsiae_1077", "epilepsiae_1125")
SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--out-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h3_independent_generative"))
    parser.add_argument("--instrument-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_7/h3_instrument"))
    args = parser.parse_args()
    supervisor = args.out_root / "supervisor"; logs = supervisor / "logs"; logs.mkdir(parents=True, exist_ok=True)
    pending = [(subject, seed) for seed in SEEDS for subject in SUBJECTS if not (
        args.out_root / subject / f"seed{seed}" / "card.json"
    ).exists()]
    running = {}; failures = []; total = len(SUBJECTS) * len(SEEDS)
    slots = [(gpu, slot) for gpu in (0, 1) for slot in range(args.workers_per_gpu)]
    while pending or running:
        for key in list(running):
            process, subject, seed, handle = running[key]
            code = process.poll()
            if code is None:
                continue
            handle.close(); del running[key]
            if code != 0 or not (args.out_root / subject / f"seed{seed}" / "card.json").exists():
                failures.append({"subject": subject, "seed": seed, "returncode": code, "gpu": key[0]})
        released = (args.instrument_root / "HUMAN_RELEASED").exists()
        if released:
            for key in slots:
                if key in running or not pending:
                    continue
                subject, seed = pending.pop(0)
                handle = (logs / f"{subject}__seed{seed}.log").open("a", encoding="utf-8")
                env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(key[0]); env.setdefault("OMP_NUM_THREADS", "2")
                process = subprocess.Popen(
                    [str(PYTHON), str(RUN), "--subject", subject, "--seed", str(seed),
                     "--device", "cuda:0", "--out-root", str(args.out_root)],
                    cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT,
                )
                running[key] = (process, subject, seed, handle)
        status = "WAITING_FOR_H3_INSTRUMENT" if pending and not released else "RUNNING"
        _write(supervisor / "queue_status.json", {
            "format": "group_event_state_v0_3_7_h3_independent_queue_v1",
            "status": status, "total": total,
            "complete": total - len(pending) - len(running) - len(failures),
            "pending": len(pending),
            "running": [{"subject": s, "seed": z, "pid": p.pid, "gpu": gpu, "slot": slot}
                        for (gpu, slot), (p, s, z, _h) in sorted(running.items())],
            "failures": failures, "observer_checkpoint_used_as_jump": False,
            "development_targets_read": False, "seizure_targets_read": False,
            "sealed_partition_opened": False,
        })
        time.sleep(args.poll_seconds)
    _write(supervisor / "queue_status.json", {
        "format": "group_event_state_v0_3_7_h3_independent_queue_v1",
        "status": "FAILED" if failures else "COMPLETE", "total": total,
        "complete": total - len(failures), "pending": 0, "running": [],
        "failures": failures, "observer_checkpoint_used_as_jump": False,
        "development_targets_read": False, "seizure_targets_read": False,
        "sealed_partition_opened": False,
    })
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

