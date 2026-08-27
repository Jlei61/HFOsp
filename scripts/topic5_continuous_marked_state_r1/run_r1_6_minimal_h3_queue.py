#!/usr/bin/env python3
"""Recoverable queue for the conditional frozen R1.6 minimal H3 cells."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.h3_long import SOURCES
from src.topic5_continuous_marked_state_r1.optimizer_h3 import (
    R1_6_MINIMAL_H3_REVISION,
)


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY",
        "CUDA_VISIBLE_DEVICES": "0",
    })
    return value


def available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0 / 1024.0
    return 0.0


def gpu_free_mib() -> float:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ], text=True)
        return min(float(value.strip()) for value in output.splitlines())
    except Exception:
        return 0.0


def wait_for_resources() -> None:
    while available_gib() < 32.0 or gpu_free_mib() < 2500.0:
        time.sleep(15.0)


def valid(path: Path, subject: str, seed: int, source: str) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == R1_6_MINIMAL_H3_REVISION
        and value.get("subject") == subject
        and value.get("seed") == seed
        and value.get("source") == source
        and value.get("scale_events") == 1000
        and value.get("t1", {}).get("seed_stable_t1") is True
        and value.get("formal_test_partition_opened") is False
        and value.get("sealed_opened") is False
    )


def run_cell(root: Path, subject: str, seed: int, source: str) -> dict:
    output = (
        root / "minimal_h3" / subject / source
        / f"seed_{seed}_n_1000/result.json"
    )
    if valid(output, subject, seed, source):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    wait_for_resources()
    log = root / "logs/minimal_h3" / subject / f"{source}_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_6_minimal_h3_cell.py",
        "--subject", subject,
        "--seed", str(seed),
        "--source", source,
        "--device", "cuda",
        "--optimizer-root", str(root),
    ]
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command,
            cwd=contract.REPO_ROOT,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    return {
        "status": "COMPLETE" if (
            process.returncode == 0 and valid(output, subject, seed, source)
        ) else "FAIL",
        "returncode": int(process.returncode),
        "subject": subject,
        "seed": int(seed),
        "source": source,
        "output": str(output),
        "log": str(log),
        "started": started,
        "finished": now(),
    }


def write_status(root: Path, *, stage: str, tasks: list[tuple],
                 rows: list[dict] | None = None) -> None:
    completed = sum(
        valid(
            root / "minimal_h3" / subject / source
            / f"seed_{seed}_n_1000/result.json",
            subject, seed, source,
        )
        for subject, seed, source in tasks
    )
    contract.atomic_json(root / "MINIMAL_H3_STATUS.json", {
        "status": "COMPLETE" if stage == "complete" else "RUNNING",
        "stage": stage,
        "revision": R1_6_MINIMAL_H3_REVISION,
        "tasks": [
            {"subject": subject, "seed": seed, "source": source}
            for subject, seed, source in tasks
        ],
        "completed": int(completed),
        "expected": len(tasks),
        "last_rows": rows or [],
        "formal_test_partition_opened": False,
        "sealed_opened": False,
        "updated_at": now(),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    summary_path = args.root / "reports/optimizer_confirmation_summary.json"
    summary = json.loads(summary_path.read_text())
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("development_validation_used_for_selection") is not False
        or summary.get("formal_test_partition_opened") is not False
        or summary.get("sealed_opened") is not False
    ):
        raise ValueError("R1.6 confirmation summary is not admissible")
    eligible = set(summary["stable_t1_subjects_for_minimal_h3"])
    tasks = []
    for row in summary["seed_rows"]:
        if row["subject"] in eligible and row["stable_checkpoint"]:
            for source in SOURCES:
                tasks.append((row["subject"], int(row["seed"]), source))
    if not tasks:
        write_status(args.root, stage="complete", tasks=[])
        return
    lock = (args.root / "minimal_h3_queue.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("R1.6 minimal H3 queue is already running") from error
    lock.write(f"pid={os.getpid()} started={now()}\n")
    lock.flush()
    write_status(args.root, stage="running", tasks=tasks)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_cell, args.root, subject, seed, source):
            (subject, seed, source)
            for subject, seed, source in tasks
        }
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({
                    "status": "FAIL",
                    "task": list(futures[future]),
                    "error": repr(error),
                })
    if any(row.get("status") != "COMPLETE" for row in rows):
        write_status(args.root, stage="fail", tasks=tasks, rows=rows)
        raise RuntimeError("R1.6 minimal H3 queue failed")
    write_status(args.root, stage="complete", tasks=tasks, rows=rows)


if __name__ == "__main__":
    main()
