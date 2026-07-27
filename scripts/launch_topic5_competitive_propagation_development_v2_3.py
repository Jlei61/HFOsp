#!/usr/bin/env python3
"""Launch the frozen 36-run v2.3 development grid with bounded resources."""
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
TRAINER = ROOT / "scripts/train_topic5_competitive_propagation_development_v2_3.py"
OUT = (
    ROOT
    / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
    / "development"
)
SUBJECTS = (
    "epilepsiae_1077",
    "epilepsiae_1146",
    "yuquan_chengshuai",
)
PERSISTENCE = ("p025_c050", "p050_c075", "p050_c090")
LEARNING_RATES = (0.003, 0.01)
SEEDS = (17, 29)


def atomic_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def task_name(task: tuple[str, str, float, int]) -> str:
    subject, persistence, learning_rate, seed = task
    learning = f"lr{learning_rate:g}".replace(".", "p")
    return f"{subject}_{persistence}_{learning}_seed{seed}"


def execute(
    task: tuple[str, str, float, int],
    *,
    device: str,
    force: bool,
) -> dict[str, Any]:
    subject, persistence, learning_rate, seed = task
    name = task_name(task)
    log_dir = OUT / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{device.replace(':', '')}.log"
    command = [
        str(PYTHON),
        str(TRAINER),
        "--subject",
        subject,
        "--persistence",
        persistence,
        "--learning-rate",
        str(learning_rate),
        "--seed",
        str(seed),
        "--batch-size",
        "2048",
        "--device",
        device,
    ]
    if force:
        command.append("--force")
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
        "task": name,
        "device": device,
        "returncode": result.returncode,
        "runtime_seconds": time.time() - started,
        "log": str(log_path.relative_to(ROOT)),
    }


def run_queue(
    tasks: list[tuple[str, str, float, int]],
    *,
    device: str,
    workers: int,
    force: bool,
    callback: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(execute, task, device=device, force=force): task
            for task in tasks
        }
        for future in as_completed(futures):
            result = future.result()
            rows.append(result)
            callback(result)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.cpu_workers <= 12:
        raise SystemExit("--cpu-workers must be in [1, 12]")

    tasks = [
        (subject, persistence, learning_rate, seed)
        for subject in SUBJECTS
        for persistence in PERSISTENCE
        for learning_rate in LEARNING_RATES
        for seed in SEEDS
    ]
    if len(tasks) != 36:
        raise SystemExit("development grid drifted")
    gpu_tasks = [task for index, task in enumerate(tasks) if index % 6 == 0]
    cpu_tasks = [task for task in tasks if task not in gpu_tasks]
    if len(gpu_tasks) != 6 or len(cpu_tasks) != 30:
        raise SystemExit("resource assignment drifted")

    OUT.mkdir(parents=True, exist_ok=True)
    state_path = OUT / "LAUNCHER_STATE.json"
    lock = threading.Lock()
    completed: list[dict[str, Any]] = []
    started = time.time()

    def update(result: dict[str, Any]) -> None:
        with lock:
            completed.append(result)
            atomic_state(
                state_path,
                {
                    "status": "RUNNING",
                    "started_unix": started,
                    "updated_unix": time.time(),
                    "n_tasks_total": len(tasks),
                    "n_tasks_finished": len(completed),
                    "n_tasks_failed": sum(
                        row["returncode"] != 0 for row in completed
                    ),
                    "gpu_workers": 1,
                    "cpu_workers": args.cpu_workers,
                    "target_values_read": False,
                    "completed": completed,
                },
            )
            print(
                f"[{len(completed):02d}/{len(tasks)}] "
                f"{result['task']} {result['device']} rc={result['returncode']}",
                flush=True,
            )

    atomic_state(
        state_path,
        {
            "status": "RUNNING",
            "started_unix": started,
            "n_tasks_total": len(tasks),
            "n_tasks_finished": 0,
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
            force=args.force,
            callback=update,
        )
        cpu = queues.submit(
            run_queue,
            cpu_tasks,
            device="cpu",
            workers=args.cpu_workers,
            force=args.force,
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
            "n_tasks_total": len(tasks),
            "n_tasks_finished": len(results),
            "n_tasks_failed": len(failures),
            "gpu_workers": 1,
            "cpu_workers": args.cpu_workers,
            "target_values_read": False,
            "completed": sorted(results, key=lambda row: row["task"]),
        },
    )
    if failures:
        raise SystemExit(f"{len(failures)} development tasks failed")
    print("development grid complete", flush=True)


if __name__ == "__main__":
    main()
