#!/usr/bin/env python3
"""Durable A3--A6/A8 follow-up after all full-grid state caches complete."""
from __future__ import annotations

import argparse
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


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(command: list[str], *, log: Path, environment: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{utc_now()}] {' '.join(command)}\n")
        handle.flush()
        completed = subprocess.run(
            command, cwd=REPO, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"follow-up command failed ({completed.returncode}): {command}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v0-2-root", type=Path, default=CANONICAL_V0_2_RESULT_ROOT)
    parser.add_argument("--result-root", type=Path, default=CANONICAL_V0_3_RESULT_ROOT)
    parser.add_argument("--expected-cells", type=int, default=46)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()
    v02, root = args.v0_2_root.resolve(), args.result_root.resolve()
    lock_path = root / "full_grid/.followup.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("another full-grid follow-up owns the lock") from error
    status_path = root / "full_grid/FOLLOWUP_STATUS.json"
    status = {
        "status": "WAITING_FOR_STATE_GRID", "created_utc": utc_now(),
        "revision": "h2b_v0_3_full_grid_followup_v1",
        "expected_cells": int(args.expected_cells), "stage": "state_grid",
        "formal_test_partition_opened": False, "sealed_opened": False,
        "h3_or_t2_run": False, "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(status_path, status)
    queue_path = root / "full_grid/STATE_QUEUE_STATUS.json"
    while True:
        queue = _json(queue_path) if queue_path.is_file() else {}
        cells = list((root / "full_grid/state_cache").glob(
            "*/seed_*/states.manifest.json"
        ))
        status.update({
            "updated_utc": utc_now(), "observed_state_cells": len(cells),
            "state_queue_status": queue.get("status"),
        })
        atomic_json(status_path, status)
        if queue.get("status") == "FAILED":
            raise RuntimeError("full-grid state queue failed")
        if queue.get("status") == "COMPLETE" and len(cells) == int(args.expected_cells):
            break
        time.sleep(float(args.poll_seconds))
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            "/usr/local/cuda-12.4/lib64"
        ),
    })
    state_root = root / "full_grid/state_cache"
    commands = [
        ("hazard_cells", [
            str(PYTHON), str(REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_hazard_queue.py"),
            "--v0-2-root", str(v02), "--result-root", str(root),
            "--state-cache-root", str(state_root), "--output-subdir", "hazard_full_grid",
            "--cpu-workers", "12", "--exploratory-all-frozen",
        ]),
        ("hazard_aggregate", [
            str(PYTHON), str(REPO / "scripts/topic5_continuous_marked_state_h2b/aggregate_v03_hazard.py"),
            "--result-root", str(root), "--analysis-subdir", "hazard_full_grid",
            "--include-diagnostic-exploration",
        ]),
        ("geometry_cells", [
            str(PYTHON), str(REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_geometry_queue.py"),
            "--v0-2-root", str(v02), "--result-root", str(root),
            "--cpu-workers", "12", "--exploratory-all-frozen",
        ]),
        ("geometry_aggregate", [
            str(PYTHON), str(REPO / "scripts/topic5_continuous_marked_state_h2b/aggregate_v03_geometry.py"),
            "--result-root", str(root),
        ]),
        ("continuous_phenotype", [
            str(PYTHON), str(REPO / "scripts/topic5_continuous_marked_state_h2b/run_v03_continuous_phenotype.py"),
            "--v0-2-root", str(v02), "--result-root", str(root),
        ]),
    ]
    for stage, command in commands:
        status.update({"status": "RUNNING", "stage": stage, "updated_utc": utc_now()})
        atomic_json(status_path, status)
        _run(command, log=root / f"logs/full_grid_followup/{stage}.log",
             environment=environment)
    status.update({
        "status": "COMPLETE", "stage": "complete", "updated_utc": utc_now(),
        "hazard_summary": str(root / "hazard_full_grid/patient_first_summary.json"),
        "geometry_summary": str(root / "geometry/patient_first_summary.json"),
        "phenotype_summary": str(root / "phenotype_continuous/summary.json"),
    })
    atomic_json(status_path, status)
    print("COMPLETE full-grid follow-up")


if __name__ == "__main__":
    main()
