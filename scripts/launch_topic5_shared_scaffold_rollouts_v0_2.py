#!/usr/bin/env python3
"""Parallel resume-safe launcher for exact-k frozen rollout fields."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import time

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/freeze_topic5_shared_scaffold_rollout_subject_v0_2.py"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--devices", nargs="*", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--source-pool-rule",
        choices=("learned_axis", "normalized_laplacian"),
        default="learned_axis",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / config["output_root"]
    )
    manifest_path = (
        Path(config["dataset_artifact_root"]).resolve()
        / config["dataset_root"]
        / "dataset_manifest.json"
    )
    subjects = list(
        args.subjects or json.loads(manifest_path.read_text())["cohort_subjects"]
    )
    models = list(args.models or ("structured", "ordinary_gru"))
    devices = [str(item) for item in (args.devices or config["resources"]["cuda_devices"])]
    if int(args.workers) < 1:
        raise ValueError("workers must be positive")
    freeze_dir = (
        "field_freeze"
        if args.source_pool_rule == "learned_axis"
        else "field_freeze_diffusion_graph_sensitivity"
    )
    logs = output_root / "logs" / freeze_dir
    logs.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "monitor" / f"rollout_launcher_state_{args.source_pool_rule}.json"
    started = time.time()
    completed = []

    def execute(task_index: int, subject: str) -> dict:
        done_path = output_root / freeze_dir / "per_subject" / subject / "DONE.json"
        if done_path.exists():
            done = json.loads(done_path.read_text())
            if done.get("status") == "COMPLETE" and set(done.get("models", [])) == set(models):
                return {"subject": subject, "returncode": 0, "resumed_complete": True}
        device = devices[task_index % len(devices)] if devices else "cpu"
        command = [
            str(config["resources"]["python"]),
            str(RUNNER),
            "--config",
            str(config_path),
            "--subject",
            subject,
            "--models",
            *models,
            "--source-pool-rule",
            args.source_pool_rule,
        ]
        if args.output_root:
            command.extend(["--output-root", str(args.output_root.resolve())])
        if args.n_rollouts is not None:
            command.extend(["--n-rollouts", str(args.n_rollouts)])
        if args.batch_size is not None:
            command.extend(["--batch-size", str(args.batch_size)])
        if args.resume:
            command.append("--resume")
        environment = os.environ.copy()
        if config["resources"]["device"] == "cuda":
            environment["CUDA_VISIBLE_DEVICES"] = device
            command.extend(["--device", "cuda:0"])
        else:
            command.extend(["--device", "cpu"])
        environment.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        log_path = logs / f"{subject}.log"
        subject_started = time.time()
        with log_path.open("a") as handle:
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
            "returncode": int(result.returncode),
            "resumed_complete": False,
            "device": device,
            "runtime_seconds": time.time() - subject_started,
            "log": str(log_path),
        }

    atomic_json(
        state_path,
        {
            "status": "RUNNING",
            "n_total": len(subjects),
            "n_finished": 0,
            "n_failed": 0,
            "subjects": subjects,
            "models": models,
            "workers": int(args.workers),
            "devices": devices,
            "started_unix": started,
            "completed": [],
        },
    )
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {
            pool.submit(execute, index, subject): subject
            for index, subject in enumerate(subjects)
        }
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            atomic_json(
                state_path,
                {
                    "status": "RUNNING",
                    "n_total": len(subjects),
                    "n_finished": len(completed),
                    "n_failed": sum(item["returncode"] != 0 for item in completed),
                    "subjects": subjects,
                    "models": models,
                    "workers": int(args.workers),
                    "devices": devices,
                    "started_unix": started,
                    "updated_unix": time.time(),
                    "completed": completed,
                },
            )
            print(
                f"[{len(completed)}/{len(subjects)}] {row['subject']} rc={row['returncode']}",
                flush=True,
            )
    failed = sum(item["returncode"] != 0 for item in completed)
    atomic_json(
        state_path,
        {
            "status": "COMPLETE" if not failed else "FAILED",
            "n_total": len(subjects),
            "n_finished": len(completed),
            "n_failed": failed,
            "subjects": subjects,
            "models": models,
            "workers": int(args.workers),
            "devices": devices,
            "runtime_seconds": time.time() - started,
            "completed": completed,
        },
    )
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
