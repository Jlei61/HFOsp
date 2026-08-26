#!/usr/bin/env python3
"""Recoverable 8--10 h development runner for Bridge-E0 and state smokes."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.topic5_continuous_marked_state import contract

PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
ROOT = contract.REPO_ROOT
LOG_ROOT = contract.RESULT_ROOT / "logs"
STATUS_PATH = contract.RESULT_ROOT / "RUN_STATUS.json"


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:" + env.get("LD_LIBRARY_PATH", "")
    env["PYTHONPATH"] = str(ROOT) + ":" + env.get("PYTHONPATH", "")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    return env


def _write_status(stage: str, **extra) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    previous = {}
    if STATUS_PATH.exists():
        try:
            previous = json.loads(STATUS_PATH.read_text())
        except Exception:
            previous = {}
    history = list(previous.get("history", []))
    history.append({"time": time.time(), "stage": stage, **extra})
    payload = {
        "contract": contract.REVISION,
        "pid": os.getpid(),
        "stage": stage,
        "updated": time.time(),
        "sealed_opened": False,
        "history": history[-100:],
        **extra,
    }
    tmp = STATUS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, STATUS_PATH)


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _run(name: str, command: list[str]) -> tuple[str, int]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{name}.log"
    with log_path.open("a") as log:
        log.write("\nCOMMAND " + " ".join(command) + "\n")
        log.flush()
        done = subprocess.run(command, cwd=ROOT, env=_env(), stdout=log,
                              stderr=subprocess.STDOUT, check=False)
    return name, int(done.returncode)


def _feature_ready(subject: str) -> bool:
    base = contract.RESULT_ROOT / "bridge/features" / f"{subject}.npz"
    return base.exists() and base.with_suffix(".manifest.json").exists()


def _build_yuquan(subjects: list[str]) -> None:
    missing = [s for s in subjects if not _feature_ready(s)]
    if not missing:
        return
    command = [
        PYTHON, "scripts/topic5_continuous_marked_state/build_bridge_features.py",
        "--subjects", *missing, "--jobs", "1", "--max-train", "6000",
        "--max-validation", "2500",
    ]
    name, code = _run("build_bridge_yuquan", command)
    if code:
        raise RuntimeError(f"{name} failed with exit {code}")


def _bridge_subject(subject: str, seeds: list[int]) -> tuple[str, list[tuple[str, int]]]:
    outcomes = []
    for seed in seeds:
        for arm in ("b0_history", "b1_spectral", "b2_raw", "b3_both"):
            job = f"bridge_{subject}_{arm}_s{seed}"
            outcomes.append(_run(job, [
                PYTHON, "scripts/topic5_continuous_marked_state/run_bridge.py",
                "--subject", subject, "--arm", arm, "--seed", str(seed),
                "--epochs", "300",
            ]))
    return subject, outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, default=0,
                        help="old disk-heavy GPU job; Yuquan reads wait for it")
    parser.add_argument("--max-hours", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    started = time.time()
    deadline = started + args.max_hours * 3600.0
    _write_status("WAIT_OLD_GPU", wait_pid=args.wait_pid, deadline=deadline)
    while _alive(args.wait_pid) and time.time() < deadline:
        time.sleep(30)
    _write_status("BUILD_YUQUAN_FEATURES", old_gpu_alive=_alive(args.wait_pid))
    _build_yuquan(["yuquan_huanghanwen", "yuquan_zhangjiaqi", "yuquan_hanyuxuan"])

    _write_status("WAIT_ALL_FEATURES")
    while time.time() < deadline:
        missing = [s for s in contract.PILOT_SUBJECTS if not _feature_ready(s)]
        if not missing:
            break
        time.sleep(30)
    missing = [s for s in contract.PILOT_SUBJECTS if not _feature_ready(s)]
    ready = [s for s in contract.PILOT_SUBJECTS if s not in missing]
    _write_status("RUN_BRIDGE", ready=ready, missing=missing)

    failures = []
    if ready:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(ready))) as pool:
            # The Bridge head is a zero-initialised convex full-batch LBFGS fit.
            # Seed 0/1 gave bit-identical metrics in the pre-run audit, so
            # repeating all arms across seeds would create pseudo-replication,
            # not uncertainty.  Spend the budget across patients instead.
            futures = {pool.submit(_bridge_subject, s, [0]): s for s in ready}
            for future in as_completed(futures):
                subject, outcomes = future.result()
                bad = [name for name, code in outcomes if code]
                if bad:
                    failures.extend(bad)
                _write_status("RUN_BRIDGE", ready=ready, missing=missing,
                              last_subject=subject, failures=failures)

    _run("aggregate_bridge", [
        PYTHON, "scripts/topic5_continuous_marked_state/aggregate_bridge.py"
    ])
    smoke_subject = "yuquan_huanghanwen" if _feature_ready("yuquan_huanghanwen") else ready[0]
    _run("state_smoke", [
        PYTHON, "scripts/topic5_continuous_marked_state/run_state_smoke.py",
        "--subject", smoke_subject,
    ])
    _write_status("COMPLETE" if not failures else "COMPLETE_WITH_FAILURES",
                  ready=ready, missing=missing, failures=failures,
                  elapsed_hours=(time.time() - started) / 3600.0)


if __name__ == "__main__":
    main()
