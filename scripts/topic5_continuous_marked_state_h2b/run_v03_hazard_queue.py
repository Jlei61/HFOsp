#!/usr/bin/env python3
"""Bounded resumable queue for H2b v0.3 A3--A5 hazard cells."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.topic5_continuous_marked_state_h2b.contract import (  # noqa: E402
    CANONICAL_V0_2_RESULT_ROOT,
    CANONICAL_V0_3_RESULT_ROOT,
    atomic_json,
    sha256_file,
    utc_now,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
CELL_SCRIPT = REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_hazard_cell.py"
HAZARD_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_hazard.py"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _complete(root: Path, subject: str, seed: int, cache_path: Path) -> bool:
    path = root / "hazard/by_cell" / subject / f"seed_{seed}/result.json"
    if not path.is_file():
        return False
    try:
        payload = _json(path)
        source = payload["source"]
        return bool(
            payload.get("status") == "COMPLETE_EXPLORATORY"
            and payload.get("revision") == "h2b_v0_3_hazard_cell_v1"
            and payload.get("subject") == subject and int(payload.get("seed")) == seed
            and source.get("state_cache_sha256") == sha256_file(cache_path)
            and source.get("producer_sha256") == sha256_file(CELL_SCRIPT)
            and source.get("hazard_module_sha256") == sha256_file(HAZARD_MODULE)
        )
    except Exception:
        return False


def _run(task: tuple[str, int, Path], *, v02: Path, root: Path,
         log_root: Path) -> dict:
    subject, seed, _ = task
    log = log_root / f"{subject}_seed_{seed}.log"
    command = [
        str(PYTHON), str(CELL_SCRIPT), "--subject", subject, "--seed", str(seed),
        "--v0-2-root", str(v02), "--result-root", str(root),
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=REPO, env=env, stdout=handle,
            stderr=subprocess.STDOUT, text=True,
        )
    return {
        "subject": subject, "seed": seed, "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started, "log": str(log),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--cpu-workers", type=int, default=8)
    args = parser.parse_args()
    v02, root = args.v0_2_root.resolve(), args.result_root.resolve()
    lock_path = root / "hazard/.queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another v0.3 hazard queue owns the lock") from error
    tasks = []
    for manifest in sorted((v02 / "state_cache").glob(
        "*/seed_*/states.manifest.json"
    )):
        subject = manifest.parents[1].name
        seed = int(manifest.parent.name.replace("seed_", ""))
        cache = manifest.parent / "states.npz"
        tasks.append((subject, seed, cache))
    pending = [task for task in tasks if not _complete(
        root, task[0], task[1], task[2],
    )]
    available = _mem_available_bytes()
    memory_workers = max(1, int(0.70 * available // int(0.25 * 1024 ** 3)))
    workers = max(1, min(
        int(args.cpu_workers), len(pending) or 1, memory_workers,
        max(1, (os.cpu_count() or 1) // 2),
    ))
    status_path = root / "hazard/QUEUE_STATUS.json"
    status = {
        "status": "RUNNING", "created_utc": utc_now(),
        "revision": "h2b_v0_3_hazard_queue_v1",
        "requested_tasks": len(tasks), "already_complete": len(tasks) - len(pending),
        "pending_tasks": len(pending), "cpu_workers": workers,
        "configured_cpu_workers": int(args.cpu_workers),
        "mem_available_bytes_at_start": available,
        "per_worker_memory_budget_bytes": int(0.25 * 1024 ** 3),
        "thread_limits": 1, "formal_test_partition_opened": False,
        "sealed_opened": False, "h3_or_t2_run": False,
    }
    atomic_json(status_path, status)
    log_root = root / "logs/hazard"
    log_root.mkdir(parents=True, exist_ok=True)
    completed_rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future = {
            pool.submit(_run, task, v02=v02, root=root, log_root=log_root): task
            for task in pending
        }
        for item in as_completed(future):
            row = item.result()
            completed_rows.append(row)
            if row["returncode"] != 0:
                failures.append(row)
            status.update({
                "updated_utc": utc_now(), "completed_this_run": len(completed_rows),
                "failed_this_run": len(failures),
            })
            atomic_json(status_path, status)
    status.update({
        "status": "COMPLETE" if not failures else "FAILED",
        "updated_utc": utc_now(), "completed_this_run": len(completed_rows),
        "failed_this_run": len(failures), "failures": failures,
        "task_rows": completed_rows,
    })
    atomic_json(status_path, status)
    if failures:
        raise SystemExit(1)
    print(f"COMPLETE tasks={len(tasks)} new={len(completed_rows)} workers={workers}")


if __name__ == "__main__":
    main()
