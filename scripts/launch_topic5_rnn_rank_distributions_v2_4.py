#!/usr/bin/env python3
"""Launch and freeze 14-patient x 3-seed interictal rank distributions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
BUILDER = ROOT / "scripts/build_topic5_rnn_rank_distributions_v2_4.py"
FINALIZER = ROOT / "scripts/finalize_topic5_rnn_rank_distributions_v2_4.py"
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
AUDIT = BASE / "input_audit/INPUT_AUDIT_STATUS.json"
REP = BASE / "representations"
SEEDS = (17, 29, 43)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute(task: tuple[str, int]) -> dict[str, Any]:
    subject, seed = task
    log_dir = REP / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{subject}_seed{seed}.log"
    command = [
        str(PYTHON),
        str(BUILDER),
        "--subject",
        subject,
        "--seed",
        str(seed),
        "--n-rollouts",
        "5000",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "subject": subject,
        "seed": seed,
        "returncode": result.returncode,
        "runtime_seconds": time.time() - started,
        "log": str(log_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("target_values_read"):
        raise SystemExit("target seal failed")
    tasks = [
        (subject, seed)
        for subject in audit["target_metadata_eligible_patients"]
        for seed in SEEDS
    ]
    if len(tasks) != 42:
        raise SystemExit("representation task denominator drifted")
    state_path = REP / "REPRESENTATION_LAUNCHER_STATE.json"
    lock = threading.Lock()
    completed = []
    started = time.time()

    def update(result: dict[str, Any]) -> None:
        with lock:
            completed.append(result)
            atomic_json(
                state_path,
                {
                    "status": "RUNNING",
                    "n_tasks_total": len(tasks),
                    "n_tasks_finished": len(completed),
                    "n_tasks_failed": sum(
                        row["returncode"] != 0 for row in completed
                    ),
                    "workers": args.workers,
                    "updated_unix": time.time(),
                    "target_values_read": False,
                    "completed": completed,
                },
            )
            print(
                f"[{len(completed):02d}/{len(tasks)}] "
                f"{result['subject']} seed={result['seed']} "
                f"rc={result['returncode']}",
                flush=True,
            )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            update(future.result())
    failures = sum(row["returncode"] != 0 for row in completed)
    finalizer_returncode = None
    if failures == 0:
        finalizer_returncode = subprocess.run(
            [str(PYTHON), str(FINALIZER)], cwd=ROOT, check=False
        ).returncode
    final_status = (
        "COMPLETE"
        if failures == 0 and finalizer_returncode == 0
        else "FAILED"
    )
    atomic_json(
        state_path,
        {
            "status": final_status,
            "n_tasks_total": len(tasks),
            "n_tasks_finished": len(completed),
            "n_tasks_failed": failures,
            "workers": args.workers,
            "runtime_seconds": time.time() - started,
            "finalizer_returncode": finalizer_returncode,
            "target_values_read": False,
            "completed": completed,
        },
    )
    raise SystemExit(0 if final_status == "COMPLETE" else 1)


if __name__ == "__main__":
    main()
