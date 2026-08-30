#!/usr/bin/env python3
"""Monitor transient data mounts for H2b v0.2 without holding a terminal."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


REPO = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO / (
    "results/epi_prssm/continuous_marked_state/h2b_cross_task/v0_2"
)
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
CANONICAL_RESULT_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/epi_prssm/continuous_marked_state/"
    "h2b_cross_task/v0_2"
)


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


def sync_results(source: Path, target: Path) -> None:
    if source.resolve() == target.resolve():
        return
    target.mkdir(parents=True, exist_ok=True)
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        destination = target / path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".sync_tmp")
        shutil.copy2(path, temporary)
        os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--max-checks", type=int, default=288)
    parser.add_argument("--output-root", type=Path, default=RESULT_ROOT)
    parser.add_argument(
        "--canonical-result-root", type=Path, default=CANONICAL_RESULT_ROOT,
    )
    parser.add_argument("--launch-cohort-queue", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    canonical_root = args.canonical_result_root.resolve()
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
        census_path = output_root / "manifests/support_census.json"
        census = json.loads(census_path.read_text()) if census_path.is_file() else {}
        runnable = int(census.get("n_subjects_runnable_now", 0))
        raw_required = int(census.get("n_subjects_requiring_raw_for_primary_h2b", 0))
        raw_required_ready = int(census.get("n_required_subjects_with_raw_cache", 0))
        mounts = census.get("raw_mounts_present") or {}
        all_mounts_ready = bool(mounts) and all(bool(value) for value in mounts.values())
        ready_to_launch = bool(
            all_mounts_ready and raw_required > 0
            and raw_required_ready == raw_required
        )
        status = {
            "status": (
                "READY_FOR_COHORT_RUN" if ready_to_launch
                else "MOUNTS_PRESENT_RAW_CACHES_INCOMPLETE" if all_mounts_ready
                else "WAITING_FOR_ALL_DATA_MOUNTS"
            ),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "monitor_pid": int(os.getpid()),
            "tmux_session": os.environ.get("H2B_MONITOR_SESSION"),
            "check_started_utc": started,
            "check_index": index + 1,
            "max_checks": int(args.max_checks),
            "interval_seconds": int(args.interval_seconds),
            "census_returncode": completed.returncode,
            "n_subjects_runnable_now": runnable,
            "n_subjects_requiring_raw_for_primary_h2b": raw_required,
            "n_required_subjects_with_raw_cache": raw_required_ready,
            "raw_mounts_present": mounts,
            "formal_test_partition_opened": False,
            "sealed_opened": False,
            "h3_or_t2_run": False,
            "next_action": (
                "run final raw-reader support census and cohort queue"
                if ready_to_launch
                else "continue monitoring until both mounts and all required raw caches are ready"
            ),
        }
        if ready_to_launch and args.launch_cohort_queue:
            queue_log = output_root / "logs/cohort_queue.nohup.log"
            queue_log.parent.mkdir(parents=True, exist_ok=True)
            queue_command = [
                str(PYTHON),
                str(REPO / "scripts/topic5_continuous_marked_state_h2b/run_v02_cohort_queue.py"),
                "--result-root", str(output_root),
                "--canonical-result-root", str(canonical_root),
            ]
            with queue_log.open("a", encoding="utf-8") as handle:
                process = subprocess.Popen(
                    queue_command, cwd=REPO, env=env,
                    stdin=subprocess.DEVNULL, stdout=handle,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
            status.update({
                "status": "COHORT_QUEUE_LAUNCHED",
                "queue_pid": int(process.pid),
                "queue_command": queue_command,
                "queue_log": str(queue_log),
                "next_action": "cohort queue owns execution and canonical sync",
            })
        atomic_json(status_path, status)
        sync_results(output_root, canonical_root)
        if ready_to_launch:
            return
        if index + 1 < int(args.max_checks):
            time.sleep(max(30, int(args.interval_seconds)))
    final = json.loads(status_path.read_text())
    final["status"] = "MONITOR_WINDOW_EXHAUSTED_MOUNTS_STILL_UNAVAILABLE"
    final["updated_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(status_path, final)
    sync_results(output_root, canonical_root)


if __name__ == "__main__":
    main()
