#!/usr/bin/env python3
"""Resumable CPU queue for H2b v0.3 full-grid OOS geometry cells."""
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
CELL_SCRIPT = REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_geometry_cell.py"
MODULE = REPO / "src/topic5_continuous_marked_state_h2b/v03_geometry.py"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _complete(root: Path, subject: str, seed: int, cache: Path,
              *, exploratory: bool) -> bool:
    path = root / "geometry/by_cell" / subject / f"seed_{seed}/result.json"
    if not path.is_file():
        return False
    try:
        payload = _json(path)
        source = payload["source"]
        expected = (
            "EXPLORATORY_A1_EMPTY_ASSAY_NOT_SENSITIVE_FULL_GRID"
            if exploratory else "CLAIM_ROUTE_RELEASED_DEVELOPMENT_ONLY"
        )
        assay = root / "assay" / (
            "type1_power_summary_smoke.json" if exploratory
            else "type1_power_summary.json"
        )
        return bool(
            payload.get("revision") == "h2b_v0_3_oos_geometry_cell_v2"
            and payload.get("subject") == subject
            and int(payload.get("seed")) == seed
            and payload.get("claim_status") == expected
            and payload.get("common_extraction_domain") is True
            and source.get("state_cache_sha256") == sha256_file(cache)
            and source.get("producer_sha256") == sha256_file(CELL_SCRIPT)
            and source.get("geometry_module_sha256") == sha256_file(MODULE)
            and assay.is_file()
            and source.get("assay_summary_sha256") == sha256_file(assay)
        )
    except Exception:
        return False


def _run(task: tuple[str, int, Path], *, v02: Path, root: Path,
         exploratory: bool, log_root: Path) -> dict:
    subject, seed, _ = task
    command = [
        str(PYTHON), str(CELL_SCRIPT), "--subject", subject, "--seed", str(seed),
        "--v0-2-root", str(v02), "--result-root", str(root),
    ]
    if exploratory:
        command.append("--allow-diagnostic-exploration")
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    log = log_root / f"{subject}_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
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
    parser.add_argument("--exploratory-all-frozen", action="store_true")
    args = parser.parse_args()
    v02, root = args.v0_2_root.resolve(), args.result_root.resolve()
    exploratory = bool(args.exploratory_all_frozen)
    qualification = _json(root / "qualification/state_qualified_manifest.json")
    qualified = set(map(str, qualification.get("subjects", [])))
    final_assay = root / "assay/type1_power_summary.json"
    if not exploratory and (not qualified or not final_assay.is_file()):
        atomic_json(root / "geometry/QUEUE_STATUS.json", {
            "status": "NOT_RELEASED_A1_OR_A2", "created_utc": utc_now(),
            "tasks_started": 0, "formal_test_partition_opened": False,
            "sealed_opened": False, "h3_or_t2_run": False,
        })
        print("NOT_RELEASED_A1_OR_A2 tasks=0")
        return
    lock_path = root / "geometry/.queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another v0.3 geometry queue owns the lock") from error
    tasks = []
    for manifest in sorted((root / "full_grid/state_cache").glob(
        "*/seed_*/states.manifest.json"
    )):
        subject = manifest.parents[1].name
        if not exploratory and subject not in qualified:
            continue
        seed = int(manifest.parent.name.replace("seed_", ""))
        tasks.append((subject, seed, manifest.parent / "states.npz"))
    pending = [task for task in tasks if not _complete(
        root, task[0], task[1], task[2], exploratory=exploratory,
    )]
    workers = max(1, min(
        int(args.cpu_workers), len(pending) or 1, max(1, (os.cpu_count() or 1) // 2),
    ))
    status_path = root / "geometry/QUEUE_STATUS.json"
    status = {
        "status": "RUNNING", "created_utc": utc_now(),
        "revision": "h2b_v0_3_oos_geometry_queue_v1",
        "requested_tasks": len(tasks), "pending_tasks": len(pending),
        "already_complete": len(tasks) - len(pending), "cpu_workers": workers,
        "diagnostic_exploration": exploratory, "thread_limits": 1,
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False,
    }
    atomic_json(status_path, status)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run, task, v02=v02, root=root, exploratory=exploratory,
                log_root=root / "logs/geometry",
            ): task for task in pending
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            if row["returncode"] != 0:
                failures.append(row)
            status.update({
                "updated_utc": utc_now(), "completed_this_run": len(rows),
                "failed_this_run": len(failures),
            })
            atomic_json(status_path, status)
    status.update({
        "status": "COMPLETE" if not failures else "FAILED",
        "updated_utc": utc_now(), "completed_this_run": len(rows),
        "failed_this_run": len(failures), "failures": failures,
        "task_rows": rows,
    })
    atomic_json(status_path, status)
    if failures:
        raise SystemExit(1)
    print(f"COMPLETE tasks={len(tasks)} new={len(rows)} workers={workers}")


if __name__ == "__main__":
    main()
