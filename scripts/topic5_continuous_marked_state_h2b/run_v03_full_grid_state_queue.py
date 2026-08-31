#!/usr/bin/env python3
"""Resumable CPU queue for frozen full-development-grid state extraction."""
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

import numpy as np

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
EXTRACTOR = REPO / "scripts/topic5_continuous_marked_state_h2b/extract_states.py"
INVENTORY = CANONICAL_V0_2_RESULT_ROOT / "manifests/r1_7_checkpoint_inventory.json"
DEFAULT_SUBJECTS = (
    "epilepsiae_1073", "epilepsiae_1077", "epilepsiae_1125",
    "epilepsiae_1146", "epilepsiae_1150", "epilepsiae_253",
    "epilepsiae_442", "epilepsiae_548", "epilepsiae_635",
    "yuquan_xuxinyi",
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def _task_metadata(
    subject: str, seed: int, *, v02: Path, root: Path,
) -> tuple[list[str], Path, Path]:
    existing_root = v02 / "state_cache" / subject / f"seed_{seed}"
    existing = _json(existing_root / "states.manifest.json")
    input_manifest = _json(v02 / "risk_sets" / subject / "input_manifest.json")
    design = Path(input_manifest["design_path"])
    design_root = design.parents[2]
    baseline = design_root / "baselines" / subject / "seed_0/models.pt"
    query = root / "full_grid/queries" / f"{subject}.csv"
    output = root / "full_grid/state_cache" / subject / f"seed_{seed}/states.npz"
    command = [
        str(PYTHON), str(EXTRACTOR), "--subject", subject, "--seed", str(seed),
        "--checkpoint", existing["checkpoint"], "--checkpoint-sha256",
        existing["checkpoint_sha256"], "--allow-unstable-complete",
        "--queries", str(query), "--global-exclusions",
        str(v02 / "risk_sets" / subject / "global_exclusions.csv"),
        "--source-repo-root", "/home/honglab/leijiaxin/HFOsp",
        "--design-path", str(design), "--design-sha256",
        input_manifest["design_sha256"], "--design-manifest",
        input_manifest["design_manifest_path"], "--coverage-path",
        input_manifest["coverage_path"], "--coverage-sha256",
        input_manifest["coverage_sha256"], "--history-baseline-path",
        str(baseline), "--history-baseline-sha256",
        existing["source_hashes"]["history_baseline"],
        "--explicit-scaler-result", existing["explicit_scaler_result"],
        "--explicit-scaler-result-sha256",
        existing["explicit_scaler_result_sha256"],
        "--checkpoint-inventory", str(INVENTORY),
        "--checkpoint-inventory-sha256", sha256_file(INVENTORY),
        "--h2b-revision", "continuous_marked_state_h2b_cross_task_v0_2",
        "--embedding-batch-size", "128", "--output", str(output),
    ]
    return command, query, output


def _augment_and_audit(output: Path, query: Path) -> dict:
    manifest_path = output.with_suffix(".manifest.json")
    manifest = _json(manifest_path)
    query_manifest_path = query.with_suffix(".manifest.json")
    query_manifest = _json(query_manifest_path)
    if manifest.get("status") != "COMPLETE":
        raise ValueError("full-grid state extraction manifest is incomplete")
    if manifest.get("query_input") != str(query.resolve()):
        raise ValueError("full-grid state extraction used a different query table")
    if manifest.get("source_hashes", {}).get("query_csv") != sha256_file(query):
        raise ValueError("full-grid query SHA256 drift")
    with np.load(output, allow_pickle=False) as data:
        anchor = np.asarray(data["anchor_time_epoch"], dtype=np.float64)
        if len(anchor) != int(query_manifest["n_queries"]):
            raise ValueError("full-grid cache/query denominator mismatch")
        if not np.all(anchor < float(query_manifest["development_end_epoch"])):
            raise ValueError("full-grid cache crossed the development boundary")
        if not bool(np.asarray(data["observation_available"], dtype=bool).all()):
            raise ValueError("full-grid cache has unavailable observations")
        if np.any(np.asarray(data["max_source_time_epoch"]) > anchor + 1e-9):
            raise ValueError("full-grid cache contains a future observation")
    manifest.update({
        "full_recorded_five_minute_grid": True,
        "full_recorded_scope": "all_admissible_development_coverage_segments",
        "grid_spacing_seconds": 300.0,
        "development_only": True,
        "query_manifest": str(query_manifest_path),
        "query_manifest_sha256": sha256_file(query_manifest_path),
        "full_grid_queue_producer_sha256": sha256_file(Path(__file__).resolve()),
        "formal": False, "sealed": False,
    })
    atomic_json(manifest_path, manifest)
    return {
        "status": "COMPLETE", "cache": str(output),
        "cache_sha256": sha256_file(output), "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "n_queries": int(query_manifest["n_queries"]),
    }


def _complete(output: Path, query: Path) -> bool:
    manifest_path = output.with_suffix(".manifest.json")
    if not output.is_file() or not manifest_path.is_file():
        return False
    try:
        manifest = _json(manifest_path)
        return bool(
            manifest.get("status") == "COMPLETE"
            and manifest.get("full_recorded_five_minute_grid") is True
            and manifest.get("cache_sha256") == sha256_file(output)
            and manifest.get("source_hashes", {}).get("query_csv") == sha256_file(query)
            and manifest.get("full_grid_queue_producer_sha256")
            == sha256_file(Path(__file__).resolve())
        )
    except Exception:
        return False


def _run(task: tuple[str, int, list[str], Path, Path], *, log_root: Path) -> dict:
    subject, seed, command, query, output = task
    log = log_root / f"{subject}_seed_{seed}.log"
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
    })
    started = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    row = {
        "subject": subject, "seed": seed, "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started, "log": str(log),
    }
    if completed.returncode == 0:
        row.update(_augment_and_audit(output, query))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional seed subset for resource smoke or partial resumable runs.",
    )
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    args = parser.parse_args()
    v02, root = args.v0_2_root.resolve(), args.result_root.resolve()
    lock_path = root / "full_grid/.state_queue.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another full-grid state queue owns the lock") from error
    tasks = []
    for manifest in sorted((v02 / "state_cache").glob("*/seed_*/states.manifest.json")):
        subject = manifest.parents[1].name
        if subject not in set(args.subjects):
            continue
        seed = int(manifest.parent.name.replace("seed_", ""))
        if args.seeds is not None and seed not in set(args.seeds):
            continue
        command, query, output = _task_metadata(
            subject, seed, v02=v02, root=root,
        )
        tasks.append((subject, seed, command, query, output))
    pending = [task for task in tasks if not _complete(task[4], task[3])]
    available = _available_memory()
    # A single epilepsiae_442/seed_0 smoke peaked at 9.27 GiB RSS, while the
    # first mixed-patient 8-worker batch showed a 22.5 GiB live high-water
    # process.  Schedule at 28 GiB/cell so the resumable follow-up batch keeps
    # headroom for larger contact sets and loader transients.
    measured_budget = int(28.0 * 1024 ** 3)
    workers = max(1, min(
        int(args.cpu_workers), len(pending) or 1,
        max(1, int(0.70 * available // measured_budget)),
        max(1, (os.cpu_count() or 1) // 2),
    ))
    status_path = root / "full_grid/STATE_QUEUE_STATUS.json"
    status = {
        "status": "RUNNING", "created_utc": utc_now(),
        "revision": "h2b_v0_3_full_grid_state_queue_v3",
        "requested_tasks": len(tasks), "pending_tasks": len(pending),
        "already_complete": len(tasks) - len(pending), "cpu_workers": workers,
        "configured_cpu_workers": int(args.cpu_workers),
        "mem_available_bytes_at_start": available,
        "per_worker_memory_budget_bytes": measured_budget,
        "resource_smoke_peak_rss_bytes": 9_274_196 * 1024,
        "resource_smoke_subject_seed": "epilepsiae_442/seed_0",
        "mixed_batch_observed_live_high_water_rss_bytes": int(22.5 * 1024 ** 3),
        "thread_limits": 1, "formal_test_partition_opened": False,
        "sealed_opened": False, "h3_or_t2_run": False,
    }
    atomic_json(status_path, status)
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run, task, log_root=root / "logs/full_grid_state",
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
