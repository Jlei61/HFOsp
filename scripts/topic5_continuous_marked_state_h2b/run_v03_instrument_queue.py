#!/usr/bin/env python3
"""Bounded CPU queue for H2b v0.3 interictal instrument cells."""
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
from scripts.topic5_continuous_marked_state_h2b.run_v03_instrument_cell import (  # noqa: E402
    _resolve_interictal_design,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
CELL_SCRIPT = REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_instrument_cell.py"
INSTRUMENT_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_instrument.py"
NUISANCE_MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_nuisance.py"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is unavailable")


def _complete(root: Path, subject: str, seed: int, checkpoint_sha256: str,
              n_null_permutations: int) -> bool:
    path = root / "instrument/by_cell" / subject / f"seed_{seed}" / "instrument_manifest.json"
    if not path.is_file():
        return False
    try:
        payload = _json(path)
        trace = Path(payload["trace_path"])
        return bool(
            payload.get("status") == "COMPLETE"
            and payload.get("revision") == "h2b_v0_3_interictal_instrument_cell_v4"
            and payload.get("subject") == subject
            and int(payload.get("seed", -1)) == seed
            and payload.get("source", {}).get("checkpoint", {}).get("checkpoint_sha256")
            == checkpoint_sha256
            and int(payload.get("instrument_config", {}).get(
                "n_null_permutations", -1
            )) == int(n_null_permutations)
            and payload.get("source", {}).get("producer_script_sha256")
            == sha256_file(CELL_SCRIPT)
            and payload.get("source", {}).get("instrument_module_sha256")
            == sha256_file(INSTRUMENT_MODULE)
            and payload.get("source", {}).get("nuisance_module_sha256")
            == sha256_file(NUISANCE_MODULE)
            and trace.is_file()
            and payload.get("trace_sha256") == sha256_file(trace)
        )
    except Exception:
        return False


def _run(task: tuple[str, int], *, v02: Path, result: Path, log_root: Path,
         n_null_permutations: int) -> dict:
    subject, seed = task
    log = log_root / f"{subject}_seed_{seed}.log"
    command = [
        str(PYTHON), str(CELL_SCRIPT), "--subject", subject, "--seed", str(seed),
        "--v0-2-root", str(v02), "--result-root", str(result),
        "--n-null-permutations", str(int(n_null_permutations)),
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
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--n-null-permutations", type=int, default=100)
    args = parser.parse_args()
    v02, result = args.v0_2_root.resolve(), args.result_root.resolve()
    lock_path = result / "instrument/.queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another v0.3 instrument queue owns the lock") from error

    inventory = _json(v02 / "manifests/r1_7_checkpoint_inventory.json")
    requested = set(map(str, args.subjects or []))
    candidate_tasks = []
    entry_by_task = {}
    for entry in inventory["entries"]:
        subject, seed = str(entry["subject"]), int(entry["seed"])
        if not bool(entry.get("checkpoint_available")):
            continue
        if requested and subject not in requested:
            continue
        key = (subject, seed)
        candidate_tasks.append(key)
        entry_by_task[key] = entry
    candidate_tasks = sorted(set(candidate_tasks))
    unavailable_designs = {}
    for subject in sorted({task[0] for task in candidate_tasks}):
        try:
            _resolve_interictal_design(
                subject, v02_root=v02, result_root=result,
            )
        except FileNotFoundError as error:
            unavailable_designs[subject] = str(error)
    tasks = [task for task in candidate_tasks if task[0] not in unavailable_designs]
    pending = [task for task in tasks if not _complete(
        result, task[0], task[1], entry_by_task[task]["checkpoint_sha256"],
        int(args.n_null_permutations),
    )]
    available = _mem_available_bytes()
    # One measured cell peaked below 0.5 GiB; budget 1.25 GiB per worker and
    # retain 30% of available memory.  The configured cap remains authoritative.
    memory_workers = max(1, int(0.70 * available // int(1.25 * 1024 ** 3)))
    workers = max(1, min(
        int(args.cpu_workers), len(pending) or 1, memory_workers,
        max(1, (os.cpu_count() or 1) // 2),
    ))
    status_path = result / "instrument/QUEUE_STATUS.json"
    status = {
        "status": "RUNNING", "revision": "h2b_v0_3_instrument_queue_v3",
        "created_utc": utc_now(), "pid": os.getpid(), "pgid": os.getpgid(0),
        "requested_tasks": len(candidate_tasks), "eligible_tasks": len(tasks),
        "excluded_design_unavailable_tasks": len(candidate_tasks) - len(tasks),
        "design_unavailable_subjects": unavailable_designs,
        "already_complete": len(tasks) - len(pending),
        "pending_tasks": len(pending), "cpu_workers": workers,
        "configured_cpu_workers": int(args.cpu_workers),
        "mem_available_bytes_at_start": available,
        "per_worker_memory_budget_bytes": int(1.25 * 1024 ** 3),
        "n_null_permutations": int(args.n_null_permutations),
        "thread_limits": 1, "cuda_visible_devices": "",
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(status_path, status)
    log_root = result / "logs/instrument"
    log_root.mkdir(parents=True, exist_ok=True)
    completed_rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future = {
            pool.submit(
                _run, task, v02=v02, result=result, log_root=log_root,
                n_null_permutations=int(args.n_null_permutations),
            ): task
            for task in pending
        }
        for item in as_completed(future):
            row = item.result()
            completed_rows.append(row)
            if int(row["returncode"]) != 0:
                failures.append(row)
            status.update({
                "updated_utc": utc_now(),
                "completed_this_run": len(completed_rows),
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
