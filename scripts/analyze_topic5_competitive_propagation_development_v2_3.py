#!/usr/bin/env python3
"""Freeze v2.3 development hyperparameters without reading heldout20."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = (
    ROOT
    / "results/topic5_symmetric_axis_competitive_propagation_v2_3"
    / "development"
)
SUBJECTS = (
    "epilepsiae_1077",
    "epilepsiae_1146",
    "yuquan_chengshuai",
)
PERSISTENCE_ORDER = {
    "p025_c050": 0,
    "p050_c075": 1,
    "p050_c090": 2,
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    launcher = json.loads(
        (BASE / "LAUNCHER_STATE.json").read_text(encoding="utf-8")
    )
    if launcher.get("status") != "COMPLETE" or launcher.get("n_tasks_failed"):
        raise SystemExit("development launcher is not complete")
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted((BASE / "grid").glob("**/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        resolved = json.loads(
            (metrics_path.parent / "resolved_config.json").read_text(
                encoding="utf-8"
            )
        )
        if payload.get("target_values_read") or "heldout20_sealed" in payload.get(
            "metrics", {}
        ):
            raise SystemExit(f"target/heldout leak in {metrics_path}")
        validation = payload["metrics"]["validation20"]
        rows.append(
            {
                "subject": payload["subject"],
                "persistence_label": payload["persistence_label"],
                "rho_propagation": payload["rho_propagation"],
                "rho_competition": payload["rho_competition"],
                "learning_rate": payload["learning_rate"],
                "seed": payload["seed"],
                "best_epoch": payload["best_epoch"],
                "epochs_completed": payload["epochs_completed"],
                "validation_full_nll": validation["full_categorical_nll"],
                "validation_node_nll": validation["node_categorical_nll"],
                "validation_full_over_node_benefit": validation[
                    "full_over_node_benefit"
                ],
                "finite": validation["finite"],
                "runtime_seconds": payload["resource"]["runtime_seconds"],
                "peak_rss_gb": payload["resource"]["peak_rss_gb"],
                "peak_cuda_allocated_gb": payload["resource"][
                    "peak_cuda_allocated_gb"
                ],
                "batch_size": resolved["batch_size"],
                "target_values_read": False,
                "metrics_path": str(metrics_path.relative_to(ROOT)),
            }
        )
    table = pd.DataFrame(rows)
    if (
        len(table) != 36
        or set(table.subject) != set(SUBJECTS)
        or not table.finite.all()
        or table.target_values_read.any()
    ):
        raise SystemExit("development grid artifact count/schema drifted")
    if table.groupby(
        ["subject", "persistence_label", "learning_rate"]
    ).seed.nunique().min() != 2:
        raise SystemExit("development seed coverage is incomplete")

    table.to_csv(BASE / "development_run_inventory.csv", index=False)
    per_subject_seed = (
        table.groupby(
            ["subject", "persistence_label", "learning_rate"], as_index=False
        )
        .agg(
            validation_full_nll=("validation_full_nll", "mean"),
            validation_node_nll=("validation_node_nll", "mean"),
            validation_full_over_node_benefit=(
                "validation_full_over_node_benefit",
                "mean",
            ),
            seed_sd=("validation_full_nll", "std"),
            median_best_epoch=("best_epoch", "median"),
        )
    )
    grid = (
        per_subject_seed.groupby(
            ["persistence_label", "learning_rate"], as_index=False
        )
        .agg(
            patient_first_mean_validation_nll=(
                "validation_full_nll",
                "mean",
            ),
            patient_first_median_validation_nll=(
                "validation_full_nll",
                "median",
            ),
            patient_first_mean_benefit=(
                "validation_full_over_node_benefit",
                "mean",
            ),
            median_best_epoch=("median_best_epoch", "median"),
            max_seed_sd=("seed_sd", "max"),
        )
    )
    grid["persistence_order"] = grid.persistence_label.map(PERSISTENCE_ORDER)
    grid = grid.sort_values(
        [
            "patient_first_mean_validation_nll",
            "learning_rate",
            "persistence_order",
        ],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    grid.to_csv(BASE / "development_hyperparameter_summary.csv", index=False)
    selected = grid.iloc[0]
    freeze = {
        "contract": "topic5_symmetric_axis_competitive_propagation_v2_3",
        "status": "FROZEN",
        "selection_endpoint": (
            "mean_across_three_patients_of_seed_mean_validation_categorical_nll"
        ),
        "selected_persistence_label": selected["persistence_label"],
        "rho_propagation": float(
            table.loc[
                table.persistence_label == selected["persistence_label"],
                "rho_propagation",
            ].iloc[0]
        ),
        "rho_competition": float(
            table.loc[
                table.persistence_label == selected["persistence_label"],
                "rho_competition",
            ].iloc[0]
        ),
        "learning_rate": float(selected["learning_rate"]),
        "batch_size": 2048,
        "optimizer": "AdamW",
        "weight_decay": 1.0e-4,
        "gradient_clip": 5.0,
        "maximum_epochs": 200,
        "patience": 20,
        "development_patients": list(SUBJECTS),
        "development_seeds": [17, 29],
        "n_runs": len(table),
        "all_runs_finite": bool(table.finite.all()),
        "heldout20_read_for_selection": False,
        "ab_labels_read": False,
        "early_ictal_target_values_read": False,
        "scientific_gate_applied": False,
        "permission": "formal_interictal_training_only",
        "selected_validation_summary": {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else str(value)
            )
            for key, value in selected.to_dict().items()
            if key != "persistence_order"
        },
    }
    atomic_json(BASE / "DEVELOPMENT_FREEZE.json", freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
