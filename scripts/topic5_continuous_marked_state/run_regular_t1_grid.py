#!/usr/bin/env python3
"""Recoverable T0/T1 grid for pilots whose regular observations are ready."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.regular_t1 import REGULAR_T1_REVISION


PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
LOG_ROOT = contract.RESULT_ROOT / "logs/regular_t1"


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
        "contract": contract.REVISION, "pid": os.getpid(), "stage": stage,
        "regular_t1_revision": REGULAR_T1_REVISION,
        "updated": time.time(), "sealed_opened": False, **extra,
    }
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, status_path)


def run_one(subject: str, arm: str, seed: int, epochs: int,
            observation_variant: str, state_dim: int) -> tuple[str, str, int, int]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    # Keep sensitivity variants physically separate.  Reusing the spectral
    # filename would not alter the result JSON, but it would silently mix two
    # different observation contracts in the audit log.
    variant_suffix = "" if observation_variant == "spectral" else f"__{observation_variant}_e0"
    log_path = LOG_ROOT / f"{subject}__{arm}__s{seed}{variant_suffix}.log"
    command = [
        PYTHON, "scripts/topic5_continuous_marked_state/run_regular_t1.py",
        "--subject", subject, "--arm", arm, "--seed", str(seed),
        "--epochs", str(epochs),
        "--observation-variant", observation_variant,
        "--state-dim", str(state_dim),
    ]
    with log_path.open("a") as log:
        log.write("COMMAND " + " ".join(command) + "\n")
        log.flush()
        done = subprocess.run(command, cwd=contract.REPO_ROOT, env=environment(),
                              stdout=log, stderr=subprocess.STDOUT, check=False)
    return subject, arm, seed, int(done.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", required=True,
                        choices=contract.PILOT_SUBJECTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--observation-variant", choices=("spectral", "raw", "both"),
                        default="spectral")
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument(
        "--status-tag", default="",
        help="optional suffix so concurrent recovery grids do not overwrite status",
    )
    args = parser.parse_args()
    if args.state_dim < 1:
        raise ValueError("state-dim must be positive")
    variant = "" if args.observation_variant == "spectral" else f"_{args.observation_variant}_e0"
    tag = f"_{args.status_tag}" if args.status_tag else ""
    status_path = contract.RESULT_ROOT / f"REGULAR_T1_RUN_STATUS{variant}{tag}.json"
    feature_root = (
        contract.RESULT_ROOT / "regular_observation/features"
        if args.observation_variant == "spectral"
        else contract.RESULT_ROOT / f"regular_observation/features_{args.observation_variant}"
    )
    ready = [subject for subject in args.subjects if (
        feature_root / f"{subject}.npz"
    ).exists() and (
        contract.RESULT_ROOT / "long_sequence/features" / f"{subject}.npz"
    ).exists()]
    jobs = [
        (subject, arm, seed, args.epochs, args.observation_variant, args.state_dim)
        for subject in ready
        for seed in args.seeds
        for arm in ("t0_no_observation_state", "t1_regular_observation")
    ]
    write_status(status_path, "RUNNING", ready=ready, n_jobs=len(jobs), n_completed=0,
                 observation_variant=args.observation_variant,
                 state_dim=args.state_dim, status_tag=args.status_tag)
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run_one, *job): job for job in jobs}
        for future in as_completed(futures):
            subject, arm, seed, code = future.result()
            completed += 1
            if code:
                failures.append({"subject": subject, "arm": arm,
                                 "seed": seed, "exit_code": code})
            write_status(status_path, "RUNNING", ready=ready, n_jobs=len(jobs),
                         n_completed=completed, failures=failures,
                         observation_variant=args.observation_variant,
                         state_dim=args.state_dim, status_tag=args.status_tag)
    subprocess.run([
        PYTHON, "scripts/topic5_continuous_marked_state/aggregate_regular_t1.py",
        "--observation-variant", args.observation_variant,
    ], cwd=contract.REPO_ROOT, env=environment(), check=False)
    write_status(status_path, "COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
                 ready=ready, n_jobs=len(jobs), n_completed=completed,
                 failures=failures, observation_variant=args.observation_variant,
                 state_dim=args.state_dim, status_tag=args.status_tag)


if __name__ == "__main__":
    main()
