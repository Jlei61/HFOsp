#!/usr/bin/env python3
"""Run the v0.3.7 strict-prefix decoder matrix as resumable GPU workers."""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v037.contracts import atomic_json  # noqa: E402


DEFAULT_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/decoder_strict")
ARM = "L3_LOCAL_PLUS_LEARNED_LR"


def _jobs(out_root: Path, seeds: tuple[int, ...]) -> list[tuple[str, int]]:
    manifest = json.loads((out_root / "INPUT_CACHE_MANIFEST.json").read_text(encoding="utf-8"))
    fits = sorted(manifest["fits"])
    return [(fit, seed) for fit in fits for seed in seeds]


def _done(out_root: Path, fit: str, seed: int) -> Path:
    return out_root / "formal_units" / fit / ARM / f"seed{seed}" / "DONE.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--gpus", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--workers-per-gpu", type=int, default=3)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    seeds = tuple(int(value) for value in args.seeds)
    slots = [gpu for gpu in args.gpus for _ in range(int(args.workers_per_gpu))]
    if not slots:
        raise ValueError("at least one GPU worker slot is required")
    pending = deque(
        (fit, seed) for fit, seed in _jobs(out_root, seeds)
        if not _done(out_root, fit, seed).exists()
    )
    all_jobs = _jobs(out_root, seeds)
    logs = out_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    running: dict[int, tuple[subprocess.Popen, object, str, int, Path]] = {}
    failures: list[dict] = []
    status_path = out_root / "supervisor" / "queue_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    def status(label: str) -> None:
        complete = sum(_done(out_root, fit, seed).exists() for fit, seed in all_jobs)
        atomic_json(status_path, {
            "format": "group_event_state_v0_3_7_strict_decoder_queue_v1",
            "status": label,
            "total": len(all_jobs),
            "complete": complete,
            "pending": len(pending),
            "running": [
                {"gpu": gpu, "pid": proc.pid, "fit_id": fit, "seed": seed}
                for gpu, (proc, _handle, fit, seed, _path) in running.items()
            ],
            "failures": failures,
            "development_targets_read": False,
            "seizure_targets_read": False,
            "sealed_partition_opened": False,
        })

    while pending or running:
        for slot_index, gpu in enumerate(slots):
            if slot_index in running or not pending:
                continue
            fit, seed = pending.popleft()
            log_path = logs / f"{fit}__seed{seed}.log"
            handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                str(ROOT / "scripts/train_topic5_lbss_unit_v0_2.py"),
                "--fit-id", fit,
                "--arm", ARM,
                "--seed", str(seed),
                "--out-root", str(out_root),
                "--unit-root-name", "formal_units",
                "--contract-label", "group_event_state_v0_3_7_strict_anatomy_decoder",
                "--device", f"cuda:{gpu}",
            ]
            environment = os.environ.copy()
            environment.update({
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            })
            process = subprocess.Popen(
                command, cwd=ROOT, env=environment,
                stdout=handle, stderr=subprocess.STDOUT,
            )
            running[slot_index] = (process, handle, fit, seed, log_path)
        status("RUNNING")
        time.sleep(float(args.poll_seconds))
        for slot_index, (process, handle, fit, seed, log_path) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            if code != 0 or not _done(out_root, fit, seed).exists():
                failures.append({
                    "fit_id": fit, "seed": seed, "exit_code": code,
                    "log": str(log_path),
                })
            del running[slot_index]

    status("FAILED" if failures else "COMPLETE")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
