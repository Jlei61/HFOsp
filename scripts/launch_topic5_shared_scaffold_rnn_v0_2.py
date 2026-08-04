#!/usr/bin/env python3
"""Resume-safe multi-GPU launcher for v0.2 patient-specific training units."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_topic5_shared_scaffold_rnn_unit_v0_2.py"


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _process_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def acquire_launcher_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}
        if (
            existing.get("hostname") == socket.gethostname()
            and _process_alive(int(existing.get("pid", -1)))
        ):
            raise RuntimeError(
                f"another launcher is active (pid={existing['pid']}): {path}"
            )
        path.unlink(missing_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "started_unix": time.time(),
            },
            handle,
        )
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    dataset_artifact_root = Path(config["dataset_artifact_root"]).resolve()
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / config["output_root"]
    )
    unit_root = output_root / ("smoke" if args.smoke else "")
    if args.smoke:
        subjects = list(args.subjects or config["smoke"]["subjects"])
        models = list(args.models or config["smoke"]["models"])
        seeds = list(args.seeds or [int(config["smoke"]["seed"])])
    else:
        source_manifest = json.loads(
            (
                dataset_artifact_root
                / config["dataset_root"]
                / "dataset_manifest.json"
            ).read_text()
        )
        subjects = list(args.subjects or source_manifest["cohort_subjects"])
        models = list(args.models or config["models"]["names"])
        seeds = list(args.seeds or map(int, config["training"]["seeds"]))
    unknown_models = sorted(set(models).difference(config["models"]["names"]))
    if unknown_models:
        raise ValueError(f"unknown models: {unknown_models}")
    devices = [str(item) for item in (args.devices or config["resources"]["cuda_devices"])]
    if not devices and config["resources"]["device"] == "cuda":
        raise ValueError("at least one CUDA device is required")
    workers = int(args.workers or config["resources"]["default_workers"])
    if workers < 1:
        raise ValueError("workers must be positive")

    tasks = [
        (index, subject, model, int(seed))
        for index, (subject, model, seed) in enumerate(
            (
                (subject, model, seed)
                for subject in sorted(subjects)
                for model in models
                for seed in seeds
            )
        )
    ]
    monitor_root = output_root / "monitor"
    state_path = monitor_root / ("smoke_launcher_state.json" if args.smoke else "launcher_state.json")
    lock_path = monitor_root / ("smoke_launcher.lock" if args.smoke else "launcher.lock")
    acquire_launcher_lock(lock_path)
    log_root = output_root / "logs" / ("smoke" if args.smoke else "formal")
    log_root.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    state_lock = threading.Lock()
    started = time.time()

    def execute(task: tuple[int, str, str, int]) -> dict[str, Any]:
        task_index, subject, model, seed = task
        run_dir = unit_root / "per_subject" / subject / model / f"seed_{seed}"
        done_path = run_dir / "DONE.json"
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if done.get("status") == "COMPLETE":
                return {
                    "subject": subject,
                    "model": model,
                    "seed": seed,
                    "returncode": 0,
                    "resumed_complete": True,
                    "runtime_seconds": 0.0,
                }
        device = devices[task_index % len(devices)] if devices else "cpu"
        log_path = log_root / f"{subject}__{model}__seed{seed}.log"
        command = [
            str(config["resources"]["python"]),
            str(RUNNER),
            "--config",
            str(config_path),
            "--subject",
            subject,
            "--model",
            model,
            "--seed",
            str(seed),
        ]
        if args.output_root:
            command.extend(["--output-root", str(args.output_root.resolve())])
        if args.resume:
            command.append("--resume")
        if args.smoke:
            command.append("--smoke")
        environment = os.environ.copy()
        if config["resources"]["device"] == "cuda":
            environment["CUDA_VISIBLE_DEVICES"] = device
            command.extend(["--device", "cuda:0"])
        else:
            command.extend(["--device", "cpu"])
        environment.update(
            {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            }
        )
        unit_started = time.time()
        with log_path.open("a") as handle:
            handle.write(
                f"\n[launcher] unix={unit_started:.3f} device={device} command={' '.join(command)}\n"
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
        return {
            "subject": subject,
            "model": model,
            "seed": seed,
            "device": device,
            "returncode": int(result.returncode),
            "resumed_complete": False,
            "runtime_seconds": time.time() - unit_started,
            "log": str(log_path),
        }

    def write_state(status: str) -> None:
        atomic_json(
            state_path,
            {
                "status": status,
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "smoke": bool(args.smoke),
                "n_total": len(tasks),
                "n_finished": len(completed),
                "n_failed": sum(row["returncode"] != 0 for row in completed),
                "workers": workers,
                "devices": devices,
                "subjects": subjects,
                "models": models,
                "seeds": seeds,
                "resume": bool(args.resume),
                "started_unix": started,
                "updated_unix": time.time(),
                "completed": completed,
            },
        )

    try:
        write_state("RUNNING")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(execute, task) for task in tasks]
            for future in as_completed(futures):
                row = future.result()
                with state_lock:
                    completed.append(row)
                    write_state("RUNNING")
                    print(
                        f"[{len(completed)}/{len(tasks)}] {row['subject']} "
                        f"{row['model']} seed={row['seed']} rc={row['returncode']}",
                        flush=True,
                    )
        failed = sum(row["returncode"] != 0 for row in completed)
        write_state("COMPLETE" if not failed else "FAILED")
        raise SystemExit(1 if failed else 0)
    finally:
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
