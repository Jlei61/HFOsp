#!/usr/bin/env python3
"""Wait for R1.7A, aggregate it, then run only eligible D_mechanism T2 cells."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from src.topic5_continuous_marked_state_r1 import contract


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")


def env() -> dict[str, str]:
    value = os.environ.copy(); value.update({
        "PYTHONPATH": str(contract.REPO_ROOT), "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "0",
    }); return value


def run(command: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as handle:
        handle.write("\n" + " ".join(command) + "\n"); handle.flush()
        return subprocess.run(
            command, cwd=contract.REPO_ROOT, env=env(), stdout=handle,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True, text=True,
        ).returncode


def cell(root: Path, output: Path, subject: str, seed: int, source: str) -> dict:
    result = output / subject / f"{source}_seed_{seed}_n_100/result.json"
    if result.exists():
        try:
            if json.loads(result.read_text()).get("status") == "COMPLETE":
                return {"status": "COMPLETE", "skipped": True, "result": str(result)}
        except Exception:
            pass
    while True:
        try:
            free = float(subprocess.check_output([
                "nvidia-smi", "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ], text=True).splitlines()[0])
        except Exception:
            free = 0.0
        if free >= 6500.0:
            break
        time.sleep(20)
    command = [
        str(PYTHON), "scripts/topic5_continuous_marked_state_r1/run_r1_7_t2_r2_cell.py",
        "--subject", subject, "--seed", str(seed), "--source", source,
        "--device", "cuda", "--r1-7-root", str(root),
        "--output-root", str(output),
    ]
    code = run(command, root / "logs/t2" / f"{subject}_{source}_seed_{seed}.log")
    return {"status": "COMPLETE" if code == 0 and result.exists() else "FAIL",
            "returncode": code, "result": str(result)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--root", type=Path, default=contract.RESULT_ROOT / "r1_7a")
    args = parser.parse_args()
    status_path = args.root / "QUEUE_STATUS.json"
    while True:
        if status_path.exists():
            value = json.loads(status_path.read_text())
            if value.get("status") == "FAIL":
                raise RuntimeError("R1.7A failed before T2")
            if value.get("stage") == "R1.7A_COMPLETE":
                break
        time.sleep(30)
    code = run([
        str(PYTHON), "scripts/topic5_continuous_marked_state_r1/aggregate_r1_7a.py",
        "--root", str(args.root),
    ], args.root / "logs/aggregate.log")
    if code:
        raise RuntimeError("R1.7A aggregation failed")
    summary = json.loads((args.root / "reports/r1_7a_summary.json").read_text())
    tasks = []
    for subject in summary["t2_run_subjects"]:
        for seed in range(5):
            result = json.loads((args.root / "fits" / subject / f"seed_{seed}/result.json").read_text())
            if result["stable_checkpoint"]:
                for source in ("load", "participation"):
                    tasks.append((args.root, args.root / "t2_r2", subject, seed, source))
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(cell, *task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    final = {
        "status": "COMPLETE" if all(row["status"] == "COMPLETE" for row in rows) else "FAIL",
        "stage": "T2_R2_COMPLETE", "updated_utc": datetime.now(timezone.utc).isoformat(),
        "scheduled_cells": len(tasks), "rows": rows,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(args.root / "T2_QUEUE_STATUS.json", final)
    if final["status"] != "COMPLETE":
        raise RuntimeError("R1.7A T2 failed")
    code = run([
        str(PYTHON), "scripts/topic5_continuous_marked_state_r1/aggregate_r1_7_t2_r2.py",
        "--root", str(args.root),
    ], args.root / "logs/t2_aggregate.log")
    if code:
        raise RuntimeError("R1.7A T2 aggregation failed")


if __name__ == "__main__":
    main()
