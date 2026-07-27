#!/usr/bin/env python3
"""Launch 22 patients x 3 seeds with one GPU and bounded CPU workers."""
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

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
TRAINER = ROOT / "scripts/train_topic5_competitive_propagation_formal_v2_3.py"
BASE = (
    ROOT
    / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
    / "formal"
)
AUDIT = (
    ROOT
    / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
    / "input_audit"
)
SEEDS = (17, 29, 43)


def atomic_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def execute(
    task: tuple[str, int],
    *,
    device: str,
) -> dict[str, Any]:
    subject, seed = task
    log_dir = BASE / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{subject}_seed{seed}_{device}.log"
    command = [
        str(PYTHON),
        str(TRAINER),
        "--subject",
        subject,
        "--seed",
        str(seed),
        "--device",
        device,
    ]
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "4" if device == "cpu" else "2"
    environment["MKL_NUM_THREADS"] = environment["OMP_NUM_THREADS"]
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if device.startswith("cuda"):
        environment["CUDA_VISIBLE_DEVICES"] = "0"
    started = time.time()
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[launcher] start={started:.6f} device={device} command="
            f"{' '.join(command)}\n"
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
        "device": device,
        "returncode": result.returncode,
        "runtime_seconds": time.time() - started,
        "log": str(log_path.relative_to(ROOT)),
    }


def run_queue(
    tasks: list[tuple[str, int]],
    *,
    device: str,
    workers: int,
    callback: Any,
) -> list[dict[str, Any]]:
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute, task, device=device): task for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            callback(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-workers", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.cpu_workers <= 12:
        raise SystemExit("--cpu-workers must be in [1, 12]")

    inventory = pd.read_csv(AUDIT / "subject_denominator_inventory.csv")
    physical = inventory.loc[
        inventory.physical_axis_formal.astype(bool)
    ].sort_values("n_events_total", ascending=False)
    subjects = list(map(str, physical.subject))
    if len(subjects) != 22:
        raise SystemExit("physical-axis cohort drifted")
    smallest = subjects[-1]
    gpu_tasks = [(smallest, seed) for seed in SEEDS]
    cpu_tasks = [
        (subject, seed)
        for subject in subjects
        for seed in SEEDS
        if (subject, seed) not in gpu_tasks
    ]
    if len(cpu_tasks) != 63 or len(gpu_tasks) != 3:
        raise SystemExit("formal task assignment drifted")

    BASE.mkdir(parents=True, exist_ok=True)
    state_path = BASE / "LAUNCHER_STATE.json"
    lock = threading.Lock()
    completed: list[dict[str, Any]] = []
    started = time.time()

    def update(result: dict[str, Any]) -> None:
        with lock:
            completed.append(result)
            failures = sum(row["returncode"] != 0 for row in completed)
            atomic_state(
                state_path,
                {
                    "status": "RUNNING",
                    "started_unix": started,
                    "updated_unix": time.time(),
                    "n_tasks_total": 66,
                    "n_tasks_finished": len(completed),
                    "n_tasks_failed": failures,
                    "gpu_workers": 1,
                    "cpu_workers": args.cpu_workers,
                    "target_values_read": False,
                    "completed": completed,
                },
            )
            print(
                f"[{len(completed):02d}/66] {result['subject']} "
                f"seed={result['seed']} {result['device']} "
                f"rc={result['returncode']}",
                flush=True,
            )

    atomic_state(
        state_path,
        {
            "status": "RUNNING",
            "started_unix": started,
            "n_tasks_total": 66,
            "n_tasks_finished": 0,
            "n_tasks_failed": 0,
            "gpu_workers": 1,
            "cpu_workers": args.cpu_workers,
            "target_values_read": False,
        },
    )
    with ThreadPoolExecutor(max_workers=2) as queues:
        gpu = queues.submit(
            run_queue,
            gpu_tasks,
            device="cuda",
            workers=1,
            callback=update,
        )
        cpu = queues.submit(
            run_queue,
            cpu_tasks,
            device="cpu",
            workers=args.cpu_workers,
            callback=update,
        )
        results = gpu.result() + cpu.result()
    failures = [row for row in results if row["returncode"] != 0]
    atomic_state(
        state_path,
        {
            "status": "FAILED" if failures else "COMPLETE",
            "started_unix": started,
            "finished_unix": time.time(),
            "runtime_seconds": time.time() - started,
            "n_tasks_total": 66,
            "n_tasks_finished": len(results),
            "n_tasks_failed": len(failures),
            "gpu_workers": 1,
            "cpu_workers": args.cpu_workers,
            "target_values_read": False,
            "completed": sorted(
                results, key=lambda row: (row["subject"], row["seed"])
            ),
        },
    )
    if failures:
        raise SystemExit(f"{len(failures)} formal tasks failed")
    print("formal interictal training complete", flush=True)


if __name__ == "__main__":
    main()
