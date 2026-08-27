#!/usr/bin/env python3
"""Recoverable prefix-core optimizer search for R1.6."""
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

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.optimizer_audit import R1_6_REVISION


PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
SUBJECTS = (
    "yuquan_zhangkexuan", "epilepsiae_384",
    "epilepsiae_1096", "yuquan_zhangjiaqi",
)
SEEDS = (0, 1, 2)
CONFIGS = {
    "prefix_current_e4_c256_v2": {
        "optimizer": "adamw", "lr": 3e-4, "weight_decay": 1e-3,
        "warmup": 0.0, "clip": 1.0, "epochs": 4, "chunk": 256,
        "min_delta": 0.0, "patience": 0,
    },
    "prefix_more_passes_e12_c256": {
        "optimizer": "adamw", "lr": 3e-4, "weight_decay": 1e-3,
        "warmup": 0.0, "clip": 1.0, "epochs": 12, "chunk": 256,
        "min_delta": 0.0, "patience": 0,
    },
    "prefix_more_steps_e4_c64": {
        "optimizer": "adamw", "lr": 3e-4, "weight_decay": 1e-3,
        "warmup": 0.0, "clip": 1.0, "epochs": 4, "chunk": 64,
        "min_delta": 0.0, "patience": 0,
    },
    "prefix_no_decay_warm_e8_c128": {
        "optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.0,
        "warmup": 0.1, "clip": 5.0, "epochs": 8, "chunk": 128,
        "min_delta": 0.0, "patience": 0,
    },
    "prefix_low_lr_e12_c128": {
        "optimizer": "adamw", "lr": 1e-4, "weight_decay": 0.0,
        "warmup": 0.1, "clip": 5.0, "epochs": 12, "chunk": 128,
        "min_delta": 0.0, "patience": 0,
    },
    "prefix_high_lr_e8_c128": {
        "optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.0,
        "warmup": 0.1, "clip": 5.0, "epochs": 8, "chunk": 128,
        "min_delta": 0.0, "patience": 0,
    },
    "prefix_adam_diagnostic_e8_c128": {
        "optimizer": "adam", "lr": 3e-4, "weight_decay": 0.0,
        "warmup": 0.1, "clip": 5.0, "epochs": 8, "chunk": 128,
        "min_delta": 0.0, "patience": 0,
    },
    "prefix_patience_diagnostic_e12_c128": {
        "optimizer": "adamw", "lr": 3e-4, "weight_decay": 0.0,
        "warmup": 0.1, "clip": 5.0, "epochs": 12, "chunk": 128,
        "min_delta": 1e-4, "patience": 3,
    },
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment() -> dict[str, str]:
    value = os.environ.copy()
    value.update({
        "PYTHONPATH": str(contract.REPO_ROOT),
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
        "CUDA_MODULE_LOADING": "LAZY", "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": (
            "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib:"
            + value.get("LD_LIBRARY_PATH", "")
        ),
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


def valid(path: Path, config_id: str, subject: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    trace = value.get("trace", {})
    trajectory = trace.get("trajectory", [])
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == R1_6_REVISION
        and value.get("stage") == "prefix_initialisation"
        and value.get("config_id") == config_id
        and value.get("subject") == subject and value.get("seed") == seed
        and value.get("epoch_zero_seen_alignment_selection") is False
        and value.get("formal_test_partition_opened") is False
        and value.get("sealed_opened") is False
        and 1 <= len(trajectory) <= int(trace.get("epochs_budget", -1)) + 1
        and all("evaluated_train_metrics" in row for row in trajectory)
        and all("optimizer_steps" in row for row in trajectory)
    )


def run_cell(root: Path, config_id: str, subject: str, seed: int) -> dict:
    config = CONFIGS[config_id]
    output = (
        root / "prefix_initialisation" / config_id
        / subject / f"seed_{seed}/result.json"
    )
    if valid(output, config_id, subject, seed):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    wait_for_resources()
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/build_r1_6_prefix_initialisation.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--epochs", str(config["epochs"]),
        "--learning-rate", str(config["lr"]),
        "--weight-decay", str(config["weight_decay"]),
        "--warmup-fraction", str(config["warmup"]),
        "--grad-clip-norm", str(config["clip"]),
        "--chunk-anchors", str(config["chunk"]),
        "--optimizer", str(config["optimizer"]),
        "--selection-min-delta", str(config["min_delta"]),
        "--early-stopping-patience", str(config["patience"]),
        "--config-id", config_id, "--output-root", str(root),
    ]
    log = root / "logs/prefix_tuning" / config_id / f"{subject}_seed_{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, start_new_session=True,
        )
    return {
        "status": "COMPLETE" if (
            process.returncode == 0 and valid(output, config_id, subject, seed)
        ) else "FAIL",
        "returncode": int(process.returncode), "command": command,
        "log": str(log), "output": str(output), "started": started,
        "finished": now(),
    }


def write_status(root: Path, stage: str, rows: list[dict] | None = None) -> None:
    completed = sum(
        valid(
            root / "prefix_initialisation" / config_id
            / subject / f"seed_{seed}/result.json",
            config_id, subject, seed,
        )
        for config_id in CONFIGS for subject in SUBJECTS for seed in SEEDS
    )
    contract.atomic_json(root / "PREFIX_TUNING_STATUS.json", {
        "status": "COMPLETE" if stage == "complete" else "RUNNING",
        "stage": stage, "revision": R1_6_REVISION,
        "configs": CONFIGS, "subjects": list(SUBJECTS), "seeds": list(SEEDS),
        "completed": int(completed),
        "expected": len(CONFIGS) * len(SUBJECTS) * len(SEEDS),
        "last_rows": rows or [], "updated_at": now(),
        "development_validation_scored": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


def aggregate(root: Path) -> dict:
    rows = []
    for config_id in CONFIGS:
        for subject in SUBJECTS:
            for seed in SEEDS:
                path = (
                    root / "prefix_initialisation" / config_id
                    / subject / f"seed_{seed}/result.json"
                )
                value = json.loads(path.read_text())
                trace = value["trace"]
                trajectory = trace["trajectory"]
                selected = int(trace["selected_epoch"])
                train_values = [
                    float(row["evaluated_train_metrics"]["joint_nll_per_event"])
                    for row in trajectory
                ]
                rows.append({
                    "config_id": config_id, "subject": subject, "seed": seed,
                    "selected_epoch": selected,
                    "base_select_improvement": float(
                        trajectory[0]["base_select_joint_nll"]
                        - trace["base_select_joint_nll"]
                    ),
                    "train_improvement": float(
                        train_values[0] - min(train_values)
                    ),
                    "terminal_train_improvement": float(
                        train_values[0] - train_values[-1]
                    ),
                    "selected_train_improvement": float(
                        train_values[0] - train_values[selected]
                    ),
                    "optimizer_steps_to_selected": int(sum(
                        int(row.get("optimizer_steps", 0))
                        for row in trajectory[1:selected + 1]
                    )),
                })
    by_config = {}
    for config_id in CONFIGS:
        local = [row for row in rows if row["config_id"] == config_id]
        by_subject = {}
        for subject in SUBJECTS:
            values = [row for row in local if row["subject"] == subject]
            by_subject[subject] = {
                "median_base_select_improvement": float(np.median([
                    row["base_select_improvement"] for row in values
                ])),
                "favourable_seeds": int(sum(
                    row["base_select_improvement"] > 0 for row in values
                )),
                "train_favourable_seeds": int(sum(
                    row["train_improvement"] > 0 for row in values
                )),
            }
        by_config[config_id] = {
            "stable_patients": int(sum(
                value["favourable_seeds"] >= 2
                for value in by_subject.values()
            )),
            "median_stable_patient_base_select_improvement": (
                float(np.median([
                    value["median_base_select_improvement"]
                    for value in by_subject.values()
                    if value["favourable_seeds"] >= 2
                ]))
                if any(value["favourable_seeds"] >= 2
                       for value in by_subject.values())
                else None
            ),
            "median_patient_base_select_improvement": float(np.median([
                value["median_base_select_improvement"]
                for value in by_subject.values()
            ])),
            "by_subject": by_subject,
        }
    ranking = sorted(
        CONFIGS,
        key=lambda key: (
            by_config[key]["stable_patients"],
            (
                by_config[key][
                    "median_stable_patient_base_select_improvement"
                ]
                if by_config[key][
                    "median_stable_patient_base_select_improvement"
                ] is not None else -float("inf")
            ),
            by_config[key]["median_patient_base_select_improvement"],
            -list(CONFIGS).index(key),
        ),
        reverse=True,
    )
    result = {
        "status": "COMPLETE", "revision": R1_6_REVISION,
        "selected_prefix_config": ranking[0], "ranking": ranking,
        "by_config": by_config, "rows": rows,
        "selection_uses_development_validation": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(root / "reports/prefix_tuning_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.root / "prefix_tuning_queue.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("R1.6 prefix tuning queue is already running") from error
    lock_handle.write(f"pid={os.getpid()} started={now()}\n")
    lock_handle.flush()
    write_status(args.root, "running")
    tasks = [
        (args.root, config_id, subject, seed)
        for config_id in CONFIGS for subject in SUBJECTS for seed in SEEDS
    ]
    rows = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(run_cell, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({
                    "status": "FAIL", "task": list(futures[future]),
                    "error": repr(error),
                })
    if any(row.get("status") != "COMPLETE" for row in rows):
        write_status(args.root, "fail", rows)
        raise RuntimeError("R1.6 prefix optimizer queue failed")
    write_status(args.root, "aggregate", rows)
    aggregate(args.root)
    write_status(args.root, "complete", rows)


if __name__ == "__main__":
    main()
