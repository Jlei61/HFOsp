#!/usr/bin/env python3
"""Recoverable CPU orchestrator for pilot or cohort H3-S0 grids."""
from __future__ import annotations

import json
import os
import subprocess
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.topic5_continuous_marked_state import contract


PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
LOG_ROOT = contract.RESULT_ROOT / "logs/exposure"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(contract.REPO_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["LD_LIBRARY_PATH"] = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:" + env.get("LD_LIBRARY_PATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def write_status(status_path, stage: str, **extra) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract": contract.REVISION, "fit_revision": contract.FIT_REVISION,
        "pid": os.getpid(), "stage": stage, "updated": time.time(),
        "sealed_opened": False, **extra,
    }
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, status_path)


def run_one(subject: str, tau: float, kind: str,
            decay_clock: str) -> tuple[str, float, str, int]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{subject}__{kind}__{decay_clock}__tau{tau:g}m.log"
    command = [
        PYTHON, "scripts/topic5_continuous_marked_state/run_exposure_screen.py",
        "--subject", subject, "--tau-minutes", str(tau), "--epochs", "300",
        "--exposure-kind", kind,
        "--decay-clock", decay_clock,
    ]
    with log_path.open("a") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        done = subprocess.run(command, cwd=contract.REPO_ROOT, env=environment(),
                              stdout=log, stderr=subprocess.STDOUT, check=False)
    return subject, tau, kind, int(done.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("pilot", "cohort"), default="pilot")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--kinds", nargs="+", choices=("load", "participation"),
                        default=("load", "participation"))
    parser.add_argument("--taus", nargs="+", type=float,
                        default=(1.0, 10.0, 60.0, 360.0))
    parser.add_argument("--status-tag", default="core")
    parser.add_argument("--decay-clock", choices=("physical_time", "event_count"),
                        default="physical_time")
    args = parser.parse_args()
    if not args.status_tag.replace("_", "").isalnum():
        raise ValueError("status-tag must be alphanumeric/underscore")
    status_path = contract.RESULT_ROOT / (
        "EXPOSURE_RUN_STATUS.json" if args.status_tag == "core"
        else f"EXPOSURE_RUN_STATUS_{args.status_tag}.json"
    )
    if args.scope == "pilot":
        subjects = contract.PILOT_SUBJECTS
    else:
        subjects = tuple(json.loads(contract.SPLIT_MANIFEST.read_text())["subjects"])
    jobs = [(subject, tau, kind, args.decay_clock) for subject in subjects
            for kind in args.kinds for tau in args.taus]
    write_status(
        status_path, "RUNNING", n_jobs=len(jobs), n_completed=0,
        taus_minutes=[float(value) for value in args.taus],
        exposure_kinds=list(args.kinds), status_tag=args.status_tag,
        decay_clock=args.decay_clock,
    )
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for future in as_completed(futures):
            subject, tau, kind, code = future.result()
            completed += 1
            if code:
                failures.append({"subject": subject, "tau_minutes": tau,
                                 "exposure_kind": kind, "exit_code": code})
            write_status(
                status_path, "RUNNING", n_jobs=len(jobs), n_completed=completed,
                failures=failures, taus_minutes=[float(value) for value in args.taus],
                exposure_kinds=list(args.kinds), status_tag=args.status_tag,
                decay_clock=args.decay_clock,
            )
    if args.decay_clock == "physical_time":
        subprocess.run([
            PYTHON, "scripts/topic5_continuous_marked_state/aggregate_exposure_screen.py"
        ], cwd=contract.REPO_ROOT, env=environment(), check=False)
    write_status(
        status_path,
        "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
        n_jobs=len(jobs), n_completed=completed, failures=failures,
        taus_minutes=[float(value) for value in args.taus],
        exposure_kinds=list(args.kinds), status_tag=args.status_tag,
        decay_clock=args.decay_clock,
    )


if __name__ == "__main__":
    main()
