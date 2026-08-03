#!/usr/bin/env python3
"""Resume-safe multi-worker launcher for patient-specific target-free units."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_topic5_patient_specific_rnn_unit_v0_1.py"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def own_target_subjects(cache_root: Path) -> list[str]:
    subjects = []
    for directory in sorted(cache_root.glob("outer_*")):
        subject = directory.name.removeprefix("outer_")
        if list(directory.glob(f"{subject}__*.npz")):
            subjects.append(subject)
    return subjects


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--models", nargs="*", default=None)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output = ROOT / config["output_root"]
    cache = ROOT / config["target_cache_root"]
    subjects = sorted(args.subjects or own_target_subjects(cache))
    models = args.models or [
        config["models"]["primary"],
        config["models"]["sensitivity"],
        "rank_shuffle_gru",
    ]
    seeds = list(map(int, config["training"]["seeds"]))
    tasks = [(subject, model, seed) for subject in subjects for model in models for seed in seeds]
    workers = int(args.workers or config["resources"]["default_workers"])
    state_path = output / "watchers/launcher_state.json"
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    completed = []
    started = time.time()

    def execute(task):
        subject, model, seed = task
        done = output / "units" / subject / model / f"seed_{seed}/DONE.json"
        if done.exists() and json.loads(done.read_text()).get("status") == "COMPLETE":
            return {"subject": subject, "model": model, "seed": seed, "returncode": 0, "resumed": True}
        log = logs / f"{subject}__{model}__seed{seed}.log"
        command = [
            config["resources"]["python"], str(RUNNER), "--config", str(config_path),
            "--subject", subject, "--model", model, "--seed", str(seed),
        ]
        environment = os.environ.copy()
        environment.update({
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES", "0"),
        })
        unit_started = time.time()
        with log.open("a") as handle:
            result = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=handle,
                stderr=subprocess.STDOUT, check=False,
            )
        return {
            "subject": subject, "model": model, "seed": seed,
            "returncode": result.returncode, "resumed": False,
            "runtime_seconds": time.time() - unit_started,
            "log": str(log.relative_to(ROOT)),
        }

    def update(row):
        with lock:
            completed.append(row)
            atomic_json(state_path, {
                "status": "RUNNING", "n_total": len(tasks), "n_finished": len(completed),
                "n_failed": sum(item["returncode"] != 0 for item in completed),
                "workers": workers, "subjects": subjects, "models": models,
                "updated_unix": time.time(), "completed": completed,
            })
            print(f"[{len(completed)}/{len(tasks)}] {row['subject']} {row['model']} seed={row['seed']} rc={row['returncode']}", flush=True)

    atomic_json(state_path, {
        "status": "RUNNING", "n_total": len(tasks), "n_finished": 0,
        "n_failed": 0, "workers": workers, "subjects": subjects, "models": models,
        "updated_unix": time.time(), "completed": [],
    })
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            update(future.result())
    failed = sum(row["returncode"] != 0 for row in completed)
    atomic_json(state_path, {
        "status": "COMPLETE" if not failed else "FAILED", "n_total": len(tasks),
        "n_finished": len(completed), "n_failed": failed, "workers": workers,
        "runtime_seconds": time.time() - started, "subjects": subjects, "models": models,
        "completed": completed,
    })
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
