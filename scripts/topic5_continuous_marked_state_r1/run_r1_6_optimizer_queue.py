#!/usr/bin/env python3
"""Recoverable multi-process R1.6 prefix and optimizer-selection queue."""
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
PREFIX_CONFIG = "prefix_adamw_lr3e-4_wd1e-3"
OVERFIT_CONFIG = "overfit_warm_lr1e-3"
TUNING_SUBJECTS = (
    "yuquan_zhangkexuan",
    "epilepsiae_384",
    "epilepsiae_1096",
    "yuquan_zhangjiaqi",
)
OVERFIT_SUBJECTS = TUNING_SUBJECTS
TUNING_SEEDS = (0, 1, 2)
CONFIGS = {
    "nested_current": {
        "state_lr": 3e-4, "observer_ratio": 0.1,
        "weight_decay": 1e-3, "warmup": 0.0, "clip": 1.0,
        "optimizer": "adamw", "chunk": 8,
    },
    "nested_chunk32_current": {
        "state_lr": 3e-4, "observer_ratio": 0.1,
        "weight_decay": 1e-3, "warmup": 0.0, "clip": 1.0,
        "optimizer": "adamw", "chunk": 32,
    },
    "nested_low_lr": {
        "state_lr": 1e-4, "observer_ratio": 0.1,
        "weight_decay": 1e-3, "warmup": 0.1, "clip": 1.0,
        "optimizer": "adamw", "chunk": 32,
    },
    "nested_no_decay": {
        "state_lr": 3e-4, "observer_ratio": 0.1,
        "weight_decay": 0.0, "warmup": 0.0, "clip": 1.0,
        "optimizer": "adamw", "chunk": 32,
    },
    "nested_warm_clip5": {
        "state_lr": 3e-4, "observer_ratio": 0.1,
        "weight_decay": 0.0, "warmup": 0.1, "clip": 5.0,
        "optimizer": "adamw", "chunk": 32,
    },
    "nested_high_warm": {
        "state_lr": 1e-3, "observer_ratio": 0.1,
        "weight_decay": 0.0, "warmup": 0.1, "clip": 5.0,
        "optimizer": "adamw", "chunk": 32,
    },
    "nested_slow_observer": {
        "state_lr": 3e-4, "observer_ratio": 0.03,
        "weight_decay": 0.0, "warmup": 0.1, "clip": 5.0,
        "optimizer": "adamw", "chunk": 32,
    },
    "nested_very_low": {
        "state_lr": 3e-5, "observer_ratio": 0.1,
        "weight_decay": 0.0, "warmup": 0.1, "clip": 5.0,
        "optimizer": "adamw", "chunk": 32,
    },
    "nested_adam_diagnostic": {
        "state_lr": 3e-4, "observer_ratio": 0.1,
        "weight_decay": 0.0, "warmup": 0.1, "clip": 5.0,
        "optimizer": "adam", "chunk": 32,
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


def run(command: list[str], log: Path) -> dict:
    wait_for_resources()
    log.parent.mkdir(parents=True, exist_ok=True)
    started = now()
    with log.open("a") as handle:
        handle.write(f"\n[{started}] {' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(
            command, cwd=contract.REPO_ROOT, env=environment(),
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            text=True, start_new_session=True,
        )
    return {
        "command": command, "log": str(log), "started": started,
        "finished": now(), "returncode": int(process.returncode),
    }


def read_complete(path: Path, *, stage: str, config_id: str,
                  subject: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == R1_6_REVISION
        and value.get("stage") == stage
        and value.get("config_id") == config_id
        and value.get("subject") == subject
        and value.get("seed") == seed
        and value.get("formal_test_partition_opened") is False
        and value.get("sealed_opened") is False
        and (
            stage != "optimizer_selection"
            or (
                value.get("development_validation_scored") is False
                and value.get("epoch_zero_seen_alignment_selection") is False
            )
        )
    )


def prefix_task(root: Path, subject: str, seed: int) -> dict:
    output = (
        root / "prefix_initialisation" / PREFIX_CONFIG
        / subject / f"seed_{seed}/result.json"
    )
    if read_complete(
        output, stage="prefix_initialisation", config_id=PREFIX_CONFIG,
        subject=subject, seed=seed,
    ):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/build_r1_6_prefix_initialisation.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--epochs", "4", "--learning-rate", "0.0003",
        "--weight-decay", "0.001", "--chunk-anchors", "256",
        "--config-id", PREFIX_CONFIG, "--output-root", str(root),
    ]
    value = run(command, root / "logs/prefix" / f"{subject}_seed_{seed}.log")
    value["output"] = str(output)
    value["status"] = "COMPLETE" if (
        value["returncode"] == 0 and read_complete(
            output, stage="prefix_initialisation", config_id=PREFIX_CONFIG,
            subject=subject, seed=seed,
        )
    ) else "FAIL"
    return value


def selection_task(root: Path, config_id: str,
                   subject: str, seed: int) -> dict:
    config = CONFIGS[config_id]
    output = (
        root / "selection_cells" / config_id
        / subject / f"seed_{seed}/result.json"
    )
    if read_complete(
        output, stage="optimizer_selection", config_id=config_id,
        subject=subject, seed=seed,
    ):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_6_optimizer_cell.py",
        "--subject", subject, "--seed", str(seed),
        "--config-id", config_id, "--prefix-config-id", PREFIX_CONFIG,
        "--device", "cuda", "--observer-epochs", "4",
        "--joint-epochs", "4", "--state-learning-rate",
        str(config["state_lr"]), "--observer-lr-ratio",
        str(config["observer_ratio"]), "--weight-decay",
        str(config["weight_decay"]), "--warmup-fraction",
        str(config["warmup"]), "--grad-clip-norm", str(config["clip"]),
        "--optimizer", str(config["optimizer"]),
        "--chunk-anchors", str(config["chunk"]),
        "--output-root", str(root),
    ]
    value = run(
        command,
        root / "logs/selection" / config_id / f"{subject}_seed_{seed}.log",
    )
    value["output"] = str(output)
    value["status"] = "COMPLETE" if (
        value["returncode"] == 0 and read_complete(
            output, stage="optimizer_selection", config_id=config_id,
            subject=subject, seed=seed,
        )
    ) else "FAIL"
    return value


def overfit_task(root: Path, subject: str, seed: int) -> dict:
    output = (
        root / "overfit" / OVERFIT_CONFIG
        / subject / f"seed_{seed}/result.json"
    )
    if read_complete(
        output, stage="short_segment_overfit", config_id=OVERFIT_CONFIG,
        subject=subject, seed=seed,
    ):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_6_optimizer_overfit.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--epochs", "20", "--maximum-anchors", "64",
        "--state-learning-rate", "0.001", "--observer-lr-ratio", "0.1",
        "--weight-decay", "0", "--warmup-fraction", "0.1",
        "--grad-clip-norm", "5", "--chunk-anchors", "8",
        "--config-id", OVERFIT_CONFIG, "--output-root", str(root),
    ]
    value = run(command, root / "logs/overfit" / f"{subject}_seed_{seed}.log")
    value["output"] = str(output)
    value["status"] = "COMPLETE" if (
        value["returncode"] == 0 and read_complete(
            output, stage="short_segment_overfit", config_id=OVERFIT_CONFIG,
            subject=subject, seed=seed,
        )
    ) else "FAIL"
    return value


def parallel(function, tasks: list[tuple], workers: int) -> list[dict]:
    rows = []
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        futures = {pool.submit(function, *task): task for task in tasks}
        for future in as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                rows.append({
                    "status": "FAIL", "task": list(futures[future]),
                    "error": repr(error),
                })
    return rows


def write_status(root: Path, stage: str, rows: list[dict] | None = None) -> None:
    completed_prefix = sum(
        read_complete(
            root / "prefix_initialisation" / PREFIX_CONFIG
            / subject / f"seed_{seed}/result.json",
            stage="prefix_initialisation", config_id=PREFIX_CONFIG,
            subject=subject, seed=seed,
        )
        for subject in TUNING_SUBJECTS for seed in TUNING_SEEDS
    )
    completed_selection = sum(
        read_complete(
            root / "selection_cells" / config_id
            / subject / f"seed_{seed}/result.json",
            stage="optimizer_selection", config_id=config_id,
            subject=subject, seed=seed,
        )
        for config_id in CONFIGS for subject in TUNING_SUBJECTS
        for seed in TUNING_SEEDS
    )
    completed_overfit = sum(
        read_complete(
            root / "overfit" / OVERFIT_CONFIG
            / subject / f"seed_{seed}/result.json",
            stage="short_segment_overfit", config_id=OVERFIT_CONFIG,
            subject=subject, seed=seed,
        )
        for subject in OVERFIT_SUBJECTS for seed in TUNING_SEEDS
    )
    contract.atomic_json(root / "STATUS.json", {
        "status": "COMPLETE" if stage == "complete" else "RUNNING",
        "stage": stage, "revision": R1_6_REVISION,
        "prefix_config": PREFIX_CONFIG,
        "configs": CONFIGS,
        "tuning_subjects": list(TUNING_SUBJECTS),
        "tuning_seeds": list(TUNING_SEEDS),
        "completed_prefix": int(completed_prefix),
        "expected_prefix": len(TUNING_SUBJECTS) * len(TUNING_SEEDS),
        "completed_overfit": int(completed_overfit),
        "expected_overfit": len(OVERFIT_SUBJECTS) * len(TUNING_SEEDS),
        "completed_selection": int(completed_selection),
        "expected_selection": (
            len(CONFIGS) * len(TUNING_SUBJECTS) * len(TUNING_SEEDS)
        ),
        "last_rows": rows or [], "updated_at": now(),
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


def aggregate(root: Path) -> dict:
    rows = []
    for config_id in CONFIGS:
        for subject in TUNING_SUBJECTS:
            for seed in TUNING_SEEDS:
                path = (
                    root / "selection_cells" / config_id
                    / subject / f"seed_{seed}/result.json"
                )
                value = json.loads(path.read_text())
                trajectory = value["fit_trace"]["trajectory"]
                epoch_zero = float(trajectory[0]["joint_nll"])
                best = float(value["fit_trace"]["inner_validation_joint_nll"])
                train_zero = float(trajectory[0]["evaluated_train_joint_nll"])
                selected_epoch = int(value["fit_trace"]["selected_total_epoch"])
                selected_row = trajectory[selected_epoch]
                rows.append({
                    "config_id": config_id, "subject": subject, "seed": seed,
                    "selected_epoch": selected_epoch,
                    "inner_improvement": epoch_zero - best,
                    "train_improvement": (
                        train_zero
                        - float(selected_row["evaluated_train_joint_nll"])
                    ),
                    "clip_fraction_at_selected": selected_row.get("clip_fraction"),
                    "optimizer_steps_at_selected_epoch": int(sum(
                        int(row.get("optimizer_steps", 0))
                        for row in trajectory[1:selected_epoch + 1]
                    )),
                })
    by_config = {}
    for config_id in CONFIGS:
        local = [row for row in rows if row["config_id"] == config_id]
        patient = {}
        for subject in TUNING_SUBJECTS:
            values = [row for row in local if row["subject"] == subject]
            patient[subject] = {
                "median_inner_improvement": float(np.median([
                    row["inner_improvement"] for row in values
                ])),
                "favourable_seeds": int(sum(
                    row["inner_improvement"] > 0 for row in values
                )),
                "train_favourable_seeds": int(sum(
                    row["train_improvement"] > 0 for row in values
                )),
            }
        stable_patients = int(sum(
            value["favourable_seeds"] >= 2 for value in patient.values()
        ))
        by_config[config_id] = {
            "stable_patients": stable_patients,
            "median_patient_inner_improvement": float(np.median([
                value["median_inner_improvement"] for value in patient.values()
            ])),
            "by_subject": patient,
        }
    ranking = sorted(
        CONFIGS,
        key=lambda key: (
            by_config[key]["stable_patients"],
            by_config[key]["median_patient_inner_improvement"],
            -list(CONFIGS).index(key),
        ),
        reverse=True,
    )
    result = {
        "status": "COMPLETE", "revision": R1_6_REVISION,
        "selection_uses_development_validation": False,
        "selected_config": ranking[0], "ranking": ranking,
        "by_config": by_config, "rows": rows,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(root / "reports/tuning_summary.json", result)
    return result


def require(rows: list[dict], stage: str, root: Path) -> None:
    if any(row.get("status") != "COMPLETE" for row in rows):
        write_status(root, f"{stage}_fail", rows)
        raise RuntimeError(f"R1.6 {stage} failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.root / "queue.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("R1.6 queue is already running") from error
    lock_handle.write(f"pid={os.getpid()} started={now()}\n")
    lock_handle.flush()

    write_status(args.root, "prefix")
    rows = parallel(
        prefix_task,
        [(args.root, subject, seed) for subject in TUNING_SUBJECTS
         for seed in TUNING_SEEDS],
        args.workers,
    )
    require(rows, "prefix", args.root)
    write_status(args.root, "overfit", rows)
    rows = parallel(
        overfit_task,
        [(args.root, subject, seed) for subject in OVERFIT_SUBJECTS
         for seed in TUNING_SEEDS],
        args.workers,
    )
    require(rows, "overfit", args.root)
    write_status(args.root, "selection", rows)
    rows = parallel(
        selection_task,
        [(args.root, config_id, subject, seed) for config_id in CONFIGS
         for subject in TUNING_SUBJECTS for seed in TUNING_SEEDS],
        args.workers,
    )
    require(rows, "selection", args.root)
    write_status(args.root, "aggregate", rows)
    aggregate(args.root)
    write_status(args.root, "complete", rows)


if __name__ == "__main__":
    main()
