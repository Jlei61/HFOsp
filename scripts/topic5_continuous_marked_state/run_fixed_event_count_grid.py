#!/usr/bin/env python3
"""Recoverable cohort orchestrator for fixed event-count H3-S0 arms."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.topic5_continuous_marked_state import contract


PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"


def _environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(contract.REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _status(path, stage: str, **extra) -> None:
    row = {
        "contract": contract.REVISION,
        "fit_revision": contract.FIT_REVISION,
        "stage": stage,
        "pid": os.getpid(),
        "updated": time.time(),
        "sealed_opened": False,
        **extra,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(row, indent=2, sort_keys=True))
    os.replace(temporary, path)


def _run(subject: str, kind: str, memory: float,
         clock: str) -> tuple[str, str, float, str, int]:
    log_root = contract.RESULT_ROOT / "logs/exposure_event_count"
    log_root.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON,
        "scripts/topic5_continuous_marked_state/run_fixed_event_count_screen.py",
        "--subject", subject,
        "--exposure-kind", kind,
        "--memory-events", str(memory),
        "--decay-clock", clock,
    ]
    log_path = log_root / f"{subject}__{kind}__{clock}__N{memory:g}.log"
    with log_path.open("a") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        done = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=_environment(),
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return subject, kind, memory, clock, int(done.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--memories", nargs="+", type=float,
                        default=(25.0, 50.0, 100.0, 200.0, 400.0))
    parser.add_argument("--kinds", nargs="+", choices=("load", "participation"),
                        default=("load", "participation"))
    parser.add_argument("--decay-clocks", nargs="+",
                        choices=("event_count", "physical_time"),
                        default=("event_count",))
    parser.add_argument("--status-tag", default="event_count")
    args = parser.parse_args()
    subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    jobs = [(subject, kind, memory, clock) for subject in subjects
            for kind in args.kinds for memory in args.memories
            for clock in args.decay_clocks]
    path = contract.RESULT_ROOT / (
        "EVENT_COUNT_GRID_STATUS.json" if args.status_tag == "event_count"
        else f"FIXED_MEMORY_CLOCK_GRID_STATUS_{args.status_tag}.json"
    )
    completed = 0
    failures = []
    _status(path, "RUNNING", n_jobs=len(jobs), n_completed=0, failures=[],
            memories_events=list(args.memories), exposure_kinds=list(args.kinds),
            decay_clocks=list(args.decay_clocks), status_tag=args.status_tag)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_run, *job): job for job in jobs}
        for future in as_completed(futures):
            subject, kind, memory, clock, code = future.result()
            completed += 1
            if code:
                failures.append({
                    "subject": subject, "exposure_kind": kind,
                    "memory_events": memory, "decay_clock": clock,
                    "exit_code": code,
                })
            _status(path, "RUNNING", n_jobs=len(jobs), n_completed=completed,
                    failures=failures, memories_events=list(args.memories),
                    exposure_kinds=list(args.kinds),
                    decay_clocks=list(args.decay_clocks), status_tag=args.status_tag)
    _status(path, "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
            n_jobs=len(jobs), n_completed=completed, failures=failures,
            memories_events=list(args.memories), exposure_kinds=list(args.kinds),
            decay_clocks=list(args.decay_clocks), status_tag=args.status_tag)


if __name__ == "__main__":
    main()
