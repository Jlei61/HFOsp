#!/usr/bin/env python3
"""Launch the 9-patient x 3-seed RNN candidate-axis search."""
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
TRAINER = ROOT / "scripts/train_topic5_rnn_candidate_axis_v2_4.py"
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
FORMAL = BASE / "formal"
AUDIT = BASE / "input_audit/INPUT_AUDIT_STATUS.json"
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
    log_dir = FORMAL / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{subject}_seed{seed}.log"
    command = [
        str(PYTHON),
        str(TRAINER),
        "--subject",
        subject,
        "--seed",
        str(seed),
        "--device",
        "cpu",
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[launcher] start={started:.6f} command={' '.join(command)}\n"
        )
        handle.flush()
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        handle.write(
            f"[launcher] finish={time.time():.6f} returncode={result.returncode}\n"
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
    if not 1 <= args.workers <= 16:
        raise SystemExit("--workers must be in [1,16]")
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("target_values_read"):
        raise SystemExit("target seal failed")
    subjects = list(map(str, audit["axis_positive_primary_patients"]))
    tasks = [(subject, seed) for subject in subjects for seed in SEEDS]
    if len(tasks) != 27:
        raise SystemExit("candidate-axis task denominator drifted")

    state_path = FORMAL / "AXIS_SEARCH_LAUNCHER_STATE.json"
    lock = threading.Lock()
    completed: list[dict[str, Any]] = []
    started = time.time()

    def update(result: dict[str, Any]) -> None:
        with lock:
            completed.append(result)
            failures = sum(row["returncode"] != 0 for row in completed)
            atomic_json(
                state_path,
                {
                    "status": "RUNNING",
                    "started_unix": started,
                    "updated_unix": time.time(),
                    "n_tasks_total": len(tasks),
                    "n_tasks_finished": len(completed),
                    "n_tasks_failed": failures,
                    "workers": args.workers,
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

    FORMAL.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            update(future.result())
    failures = sum(row["returncode"] != 0 for row in completed)
    atomic_json(
        state_path,
        {
            "status": "COMPLETE" if failures == 0 else "FAILED",
            "started_unix": started,
            "finished_unix": time.time(),
            "runtime_seconds": time.time() - started,
            "n_tasks_total": len(tasks),
            "n_tasks_finished": len(completed),
            "n_tasks_failed": failures,
            "workers": args.workers,
            "target_values_read": False,
            "completed": completed,
        },
    )
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
