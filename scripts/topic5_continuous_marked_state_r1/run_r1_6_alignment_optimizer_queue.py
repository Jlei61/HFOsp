#!/usr/bin/env python3
"""Target-alignment optimizer search after prefix configuration is frozen."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.optimizer_audit import R1_6_REVISION
from scripts.topic5_continuous_marked_state_r1.run_r1_6_optimizer_queue import (
    CONFIGS,
    PYTHON,
    TUNING_SEEDS,
    TUNING_SUBJECTS,
    parallel,
    run,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def effective_id(prefix_config: str, config_id: str) -> str:
    return f"{config_id}__prefix__{prefix_config}"


def valid_selection(path: Path, expected_id: str,
                    subject: str, seed: int) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except Exception:
        return False
    trace = value.get("fit_trace", {})
    trajectory = trace.get("trajectory", [])
    return bool(
        value.get("status") == "COMPLETE"
        and value.get("revision") == R1_6_REVISION
        and value.get("stage") == "optimizer_selection"
        and value.get("config_id") == expected_id
        and value.get("subject") == subject and value.get("seed") == seed
        and value.get("development_validation_scored") is False
        and value.get("epoch_zero_seen_alignment_selection") is False
        and value.get("formal_test_partition_opened") is False
        and value.get("sealed_opened") is False
        and len(trajectory) == 9
        and all("evaluated_train_metrics" in row for row in trajectory)
        and all("optimizer_steps" in row for row in trajectory)
    )


def valid_overfit(path: Path, expected_id: str,
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
        and value.get("stage") == "short_segment_overfit"
        and value.get("config_id") == expected_id
        and value.get("subject") == subject and value.get("seed") == seed
        and value.get("development_validation_scored") is False
        and value.get("formal_test_partition_opened") is False
        and value.get("sealed_opened") is False
    )


def overfit_task(root: Path, prefix_config: str,
                 subject: str, seed: int) -> dict:
    config_id = f"overfit__prefix__{prefix_config}"
    output = root / "overfit" / config_id / subject / f"seed_{seed}/result.json"
    if valid_overfit(output, config_id, subject, seed):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_6_optimizer_overfit.py",
        "--subject", subject, "--seed", str(seed), "--device", "cuda",
        "--epochs", "20", "--maximum-anchors", "64",
        "--state-learning-rate", "0.001", "--observer-lr-ratio", "0.1",
        "--weight-decay", "0", "--warmup-fraction", "0.1",
        "--grad-clip-norm", "5", "--chunk-anchors", "8",
        "--prefix-config-id", prefix_config,
        "--config-id", config_id, "--output-root", str(root),
    ]
    value = run(
        command,
        root / "logs/alignment_overfit" / prefix_config
        / f"{subject}_seed_{seed}.log",
    )
    value["output"] = str(output)
    value["status"] = "COMPLETE" if (
        value["returncode"] == 0
        and valid_overfit(output, config_id, subject, seed)
    ) else "FAIL"
    return value


def selection_task(root: Path, prefix_config: str, config_id: str,
                   subject: str, seed: int) -> dict:
    config = CONFIGS[config_id]
    run_id = effective_id(prefix_config, config_id)
    output = (
        root / "selection_cells" / run_id
        / subject / f"seed_{seed}/result.json"
    )
    if valid_selection(output, run_id, subject, seed):
        return {"status": "COMPLETE", "skipped": True, "output": str(output)}
    command = [
        str(PYTHON),
        "scripts/topic5_continuous_marked_state_r1/run_r1_6_optimizer_cell.py",
        "--subject", subject, "--seed", str(seed),
        "--config-id", run_id, "--prefix-config-id", prefix_config,
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
        root / "logs/alignment_tuning" / prefix_config / config_id
        / f"{subject}_seed_{seed}.log",
    )
    value["output"] = str(output)
    value["status"] = "COMPLETE" if (
        value["returncode"] == 0
        and valid_selection(output, run_id, subject, seed)
    ) else "FAIL"
    return value


def write_status(root: Path, stage: str, prefix_config: str,
                 rows: list[dict] | None = None) -> None:
    completed_overfit = sum(
        valid_overfit(
            root / "overfit" / f"overfit__prefix__{prefix_config}"
            / subject / f"seed_{seed}/result.json",
            f"overfit__prefix__{prefix_config}", subject, seed,
        )
        for subject in TUNING_SUBJECTS for seed in TUNING_SEEDS
    )
    completed_selection = sum(
        valid_selection(
            root / "selection_cells" / effective_id(prefix_config, config_id)
            / subject / f"seed_{seed}/result.json",
            effective_id(prefix_config, config_id), subject, seed,
        )
        for config_id in CONFIGS for subject in TUNING_SUBJECTS
        for seed in TUNING_SEEDS
    )
    contract.atomic_json(root / "ALIGNMENT_TUNING_STATUS.json", {
        "status": "COMPLETE" if stage == "complete" else "RUNNING",
        "stage": stage, "revision": R1_6_REVISION,
        "selected_prefix_config": prefix_config,
        "configs": CONFIGS, "subjects": list(TUNING_SUBJECTS),
        "seeds": list(TUNING_SEEDS),
        "completed_overfit": int(completed_overfit),
        "expected_overfit": len(TUNING_SUBJECTS) * len(TUNING_SEEDS),
        "completed_selection": int(completed_selection),
        "expected_selection": (
            len(CONFIGS) * len(TUNING_SUBJECTS) * len(TUNING_SEEDS)
        ),
        "last_rows": rows or [], "updated_at": now(),
        "development_validation_scored": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    })


def require(rows: list[dict], stage: str,
            root: Path, prefix_config: str) -> None:
    if any(row.get("status") != "COMPLETE" for row in rows):
        write_status(root, f"{stage}_fail", prefix_config, rows)
        raise RuntimeError(f"R1.6 alignment optimizer {stage} failed")


def aggregate(root: Path, prefix_config: str) -> dict:
    rows = []
    for config_id in CONFIGS:
        run_id = effective_id(prefix_config, config_id)
        for subject in TUNING_SUBJECTS:
            for seed in TUNING_SEEDS:
                value = json.loads((
                    root / "selection_cells" / run_id
                    / subject / f"seed_{seed}/result.json"
                ).read_text())
                trajectory = value["fit_trace"]["trajectory"]
                selected = int(value["fit_trace"]["selected_total_epoch"])
                rows.append({
                    "config_id": config_id, "run_id": run_id,
                    "subject": subject, "seed": seed,
                    "selected_epoch": selected,
                    "inner_improvement": float(
                        trajectory[0]["joint_nll"]
                        - value["fit_trace"]["inner_validation_joint_nll"]
                    ),
                    "train_improvement": float(
                        trajectory[0]["evaluated_train_joint_nll"]
                        - trajectory[selected]["evaluated_train_joint_nll"]
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
        for subject in TUNING_SUBJECTS:
            values = [row for row in local if row["subject"] == subject]
            by_subject[subject] = {
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
        by_config[config_id] = {
            "stable_patients": int(sum(
                value["favourable_seeds"] >= 2
                for value in by_subject.values()
            )),
            "median_patient_inner_improvement": float(np.median([
                value["median_inner_improvement"]
                for value in by_subject.values()
            ])),
            "by_subject": by_subject,
        }
    ranking = sorted(
        CONFIGS,
        key=lambda key: (
            by_config[key]["stable_patients"],
            by_config[key]["median_patient_inner_improvement"],
            -list(CONFIGS).index(key),
        ), reverse=True,
    )
    overfit_rows = []
    overfit_id = f"overfit__prefix__{prefix_config}"
    for subject in TUNING_SUBJECTS:
        for seed in TUNING_SEEDS:
            value = json.loads((
                root / "overfit" / overfit_id
                / subject / f"seed_{seed}/result.json"
            ).read_text())
            overfit_rows.append({
                "subject": subject, "seed": seed,
                "joint_nll_improvement": value["joint_nll_improvement"],
            })
    result = {
        "status": "COMPLETE", "revision": R1_6_REVISION,
        "selected_prefix_config": prefix_config,
        "selected_config": ranking[0], "ranking": ranking,
        "by_config": by_config, "rows": rows,
        "overfit_rows": overfit_rows,
        "overfit_patient_pass": {
            subject: int(sum(
                row["joint_nll_improvement"] > 0
                for row in overfit_rows if row["subject"] == subject
            ))
            for subject in TUNING_SUBJECTS
        },
        "selection_uses_development_validation": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    contract.atomic_json(root / "reports/tuning_summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--root", type=Path,
        default=contract.RESULT_ROOT / "optimizer_identifiability_r1_6",
    )
    args = parser.parse_args()
    prefix_summary = json.loads(
        (args.root / "reports/prefix_tuning_summary.json").read_text()
    )
    if (prefix_summary.get("status") != "COMPLETE"
            or prefix_summary.get("selection_uses_development_validation") is not False):
        raise ValueError("R1.6 prefix tuning is not admissible")
    prefix_config = str(prefix_summary["selected_prefix_config"])
    lock_handle = (args.root / "alignment_tuning_queue.lock").open("w")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("R1.6 alignment queue is already running") from error
    lock_handle.write(f"pid={os.getpid()} started={now()}\n")
    lock_handle.flush()
    write_status(args.root, "overfit", prefix_config)
    rows = parallel(
        overfit_task,
        [(args.root, prefix_config, subject, seed)
         for subject in TUNING_SUBJECTS for seed in TUNING_SEEDS],
        args.workers,
    )
    require(rows, "overfit", args.root, prefix_config)
    write_status(args.root, "selection", prefix_config, rows)
    rows = parallel(
        selection_task,
        [(args.root, prefix_config, config_id, subject, seed)
         for config_id in CONFIGS for subject in TUNING_SUBJECTS
         for seed in TUNING_SEEDS],
        args.workers,
    )
    require(rows, "selection", args.root, prefix_config)
    write_status(args.root, "aggregate", prefix_config, rows)
    aggregate(args.root, prefix_config)
    write_status(args.root, "complete", prefix_config, rows)


if __name__ == "__main__":
    main()
