#!/usr/bin/env python3
"""Monitor transient data mounts for H2b v0.2 without holding a terminal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / (
    "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2"
)
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-checks", type=int, default=288)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    status_path = output_root / "RUNTIME_MONITOR_STATUS.json"
    log_path = output_root / "logs/runtime_monitor_census.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(REPO)
    env["LD_LIBRARY_PATH"] = (
        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib"
        + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    )
    command = [
        str(PYTHON),
        str(REPO / "scripts/topic5_continuous_marked_state_h2b/build_v02_support_census.py"),
        "--output-root", str(output_root),
    ]
    for index in range(int(args.max_checks)):
        started = datetime.now(timezone.utc).isoformat()
        with log_path.open("a", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        census_path = RESULT_ROOT / "manifests/support_census.json"
        census = json.loads(census_path.read_text()) if census_path.is_file() else {}
        runnable = int(census.get("n_subjects_runnable_now", 0))
        status = {
            "status": "READY_FOR_COHORT_RUN" if runnable else "WAITING_FOR_DATA_MOUNTS",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "check_started_utc": started,
            "check_index": index + 1,
            "max_checks": int(args.max_checks),
            "interval_seconds": int(args.interval_seconds),
            "census_returncode": completed.returncode,
            "n_subjects_runnable_now": runnable,
            "raw_mounts_present": census.get("raw_mounts_present"),
            "formal_test_partition_opened": False,
            "sealed_opened": False,
            "h3_or_t2_run": False,
            "next_action": (
                "run final raw-reader support census and cohort queue"
                if runnable else "continue monitoring transient mounts"
            ),
        }
        atomic_json(status_path, status)
        if runnable:
            return
        if index + 1 < int(args.max_checks):
            time.sleep(max(30, int(args.interval_seconds)))
    final = json.loads(status_path.read_text())
    final["status"] = "MONITOR_WINDOW_EXHAUSTED_MOUNTS_STILL_UNAVAILABLE"
    final["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(status_path, final)


if __name__ == "__main__":
    main()
