#!/usr/bin/env python3
"""Analyze teacher-forced versus free-rollout static fields."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    load_target,
    ordinary_model_fields,
    strict_clinical_inventory,
)
from scripts.analyze_topic5_static_scaffold_fixed_readout_phase1 import (  # noqa: E402
    INDEX_NULLS,
    N_DRAWS,
    bootstrap_summary,
    bh_fdr,
    collapse_seed_scores,
    load_coords,
)
from src.topic5_static_scaffold_validation import coherent_index_null  # noqa: E402


OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
TF_ROOT = OUT / "teacher_forced_fields/per_seed"
BASELINE_ROOT = OUT / "target_free_baselines/per_subject"
SEEDS = (20260725, 20260726, 20260727)
TF_MODELS = {
    "teacher_forced_full_gru": "full_history_gru",
    "teacher_forced_rank_shuffle_gru": "rank_shuffle_gru",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_teacher_fields(
    subject: str, control: str, expected_names: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    union_fields = []
    summed_fields = []
    for seed in SEEDS:
        path = TF_ROOT / f"{subject}_seed{seed}_{control}.npz"
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["target_values_read"]:
            raise RuntimeError(f"{path}: target seal failed")
        if metadata["output_npz_sha256"] != sha256(path):
            raise RuntimeError(f"{path}: output fingerprint drifted")
        with np.load(path, allow_pickle=False) as data:
            names = np.asarray(data["contact_names"]).astype(str)
            if not np.array_equal(names, expected_names):
                raise RuntimeError(
                    f"{subject}/{seed}/{control}: contact ordering drifted"
                )
            union_fields.append(
                np.asarray(data["union_participation"], dtype=np.float64)
            )
            summed_fields.append(
                np.asarray(data["summed_next_probability"], dtype=np.float64)
            )
    return union_fields, summed_fields


def main() -> None:
    status_files = sorted(TF_ROOT.parent.glob("SHARD_*_STATUS.json"))
    if len(status_files) != 3:
        raise RuntimeError("expected three teacher-forced shard statuses")
    if any(
        json.loads(path.read_text())["status"] != "COMPLETE"
        for path in status_files
    ):
        raise RuntimeError("teacher-forced extraction is incomplete")
    inventory = strict_clinical_inventory()
    rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    for patient_index, (subject, seizures) in enumerate(inventory.items()):
        names, ordinary = ordinary_model_fields(subject)
        keep, target, used = load_target(subject, seizures, names)
        joined_names = names[keep]
        coords = load_coords(subject, names)[keep]
        with np.load(
            BASELINE_ROOT / f"{subject}.npz", allow_pickle=False
        ) as baseline_data:
            baseline_names = np.asarray(
                baseline_data["contact_names"]
            ).astype(str)
            if not np.array_equal(names, baseline_names):
                raise RuntimeError(f"{subject}: baseline ordering drifted")
            raw = np.asarray(
                baseline_data["raw_train80_participation"],
                dtype=np.float64,
            )
            best = np.asarray(
                baseline_data["best_validation_regularized_participation"],
                dtype=np.float64,
            )
        free_fields = {
            control: [
                np.asarray(seed["participation"], dtype=np.float64)
                for seed in ordinary[control]
            ]
            for control in ("full_history_gru", "rank_shuffle_gru")
        }
        teacher_fields = {}
        for label, control in TF_MODELS.items():
            union, summed = load_teacher_fields(subject, control, names)
            teacher_fields[label] = union
            for seed_index, seed in enumerate(SEEDS):
                field_rows.extend(
                    [
                        {
                            "subject": subject,
                            "seed": seed,
                            "comparison": f"{label}__vs__free_{control}",
                            "spearman": float(
                                spearmanr(
                                    union[seed_index],
                                    free_fields[control][seed_index],
                                ).statistic
                            ),
                        },
                        {
                            "subject": subject,
                            "seed": seed,
                            "comparison": f"{label}__vs__raw_train80",
                            "spearman": float(
                                spearmanr(
                                    union[seed_index], raw
                                ).statistic
                            ),
                        },
                        {
                            "subject": subject,
                            "seed": seed,
                            "comparison": f"{label}__vs__best_regularized",
                            "spearman": float(
                                spearmanr(
                                    union[seed_index], best
                                ).statistic
                            ),
                        },
                        {
                            "subject": subject,
                            "seed": seed,
                            "comparison": f"{label}__union_vs_sum",
                            "spearman": float(
                                spearmanr(
                                    union[seed_index], summed[seed_index]
                                ).statistic
                            ),
                        },
                    ]
                )
        for seed_index, seed in enumerate(SEEDS):
            field_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "comparison": (
                        "teacher_forced_full_gru__vs__"
                        "teacher_forced_rank_shuffle_gru"
                    ),
                    "spearman": float(
                        spearmanr(
                            teacher_fields["teacher_forced_full_gru"][
                                seed_index
                            ],
                            teacher_fields[
                                "teacher_forced_rank_shuffle_gru"
                            ][seed_index],
                        ).statistic
                    ),
                }
            )

        index_nulls = {
            mode: coherent_index_null(
                joined_names,
                n_draws=N_DRAWS,
                seed=2026076100 + patient_index * 10 + null_index,
                mode=mode,
            )
            for null_index, mode in enumerate(INDEX_NULLS)
        }
        geometry_eligible = bool(
            len(joined_names) >= 6 and np.all(np.isfinite(coords))
        )
        geometry_normal = (
            [
                np.random.default_rng(
                    2026077100 + patient_index * 10 + seed_index
                ).normal(size=(N_DRAWS, len(joined_names)))
                for seed_index in range(3)
            ]
            if geometry_eligible
            else None
        )
        for model, fields in teacher_fields.items():
            joined_fields = [field[keep] for field in fields]
            for null_mode, (indices, null_audit) in index_nulls.items():
                eligible = bool(null_audit["eligible"])
                if eligible:
                    score, lengthscales = collapse_seed_scores(
                        joined_fields, target, index=indices
                    )
                else:
                    score = {
                        "observed_signed": np.nan,
                        "observed_absolute": np.nan,
                        "null_signed": np.full(N_DRAWS, np.nan),
                        "null_absolute": np.full(N_DRAWS, np.nan),
                    }
                    lengthscales = []
                null_signed = np.asarray(score["null_signed"], dtype=np.float64)
                null_absolute = np.asarray(
                    score["null_absolute"], dtype=np.float64
                )
                observed_signed = float(score["observed_signed"])
                observed_absolute = float(score["observed_absolute"])
                rows.append(
                    {
                        "subject": subject,
                        "model": model,
                        "field": "teacher_forced_union_participation",
                        "score_direction": "positive_signed",
                        "null_mode": null_mode,
                        "eligible": eligible,
                        "n_contacts": len(joined_names),
                        "n_seizures": len(used),
                        "n_shafts": null_audit["n_shafts"],
                        "movable_fraction": null_audit["movable_fraction"],
                        "observed_signed_rho": observed_signed,
                        "null_signed_median": (
                            float(np.median(null_signed))
                            if eligible
                            else np.nan
                        ),
                        "signed_margin": (
                            float(observed_signed - np.median(null_signed))
                            if eligible
                            else np.nan
                        ),
                        "signed_empirical_p": (
                            float(
                                (
                                    1
                                    + np.count_nonzero(
                                        null_signed >= observed_signed
                                    )
                                )
                                / (N_DRAWS + 1)
                            )
                            if eligible
                            else np.nan
                        ),
                        "observed_absolute_rho": observed_absolute,
                        "null_absolute_median": (
                            float(np.median(null_absolute))
                            if eligible
                            else np.nan
                        ),
                        "absolute_margin": (
                            float(
                                observed_absolute
                                - np.median(null_absolute)
                            )
                            if eligible
                            else np.nan
                        ),
                        "absolute_empirical_p": (
                            float(
                                (
                                    1
                                    + np.count_nonzero(
                                        null_absolute >= observed_absolute
                                    )
                                )
                                / (N_DRAWS + 1)
                            )
                            if eligible
                            else np.nan
                        ),
                        "rbf_lengthscale_median": (
                            float(np.median(lengthscales))
                            if lengthscales
                            else np.nan
                        ),
                    }
                )
            if geometry_eligible:
                score, lengthscales = collapse_seed_scores(
                    joined_fields,
                    target,
                    coords=coords,
                    standard_normal=geometry_normal,
                )
                null_signed = np.asarray(score["null_signed"], dtype=np.float64)
                null_absolute = np.asarray(
                    score["null_absolute"], dtype=np.float64
                )
                observed_signed = float(score["observed_signed"])
                observed_absolute = float(score["observed_absolute"])
                rows.append(
                    {
                        "subject": subject,
                        "model": model,
                        "field": "teacher_forced_union_participation",
                        "score_direction": "positive_signed",
                        "null_mode": "geometry_smooth_rbf",
                        "eligible": True,
                        "n_contacts": len(joined_names),
                        "n_seizures": len(used),
                        "n_shafts": np.nan,
                        "movable_fraction": 1.0,
                        "observed_signed_rho": observed_signed,
                        "null_signed_median": float(np.median(null_signed)),
                        "signed_margin": float(
                            observed_signed - np.median(null_signed)
                        ),
                        "signed_empirical_p": float(
                            (
                                1
                                + np.count_nonzero(
                                    null_signed >= observed_signed
                                )
                            )
                            / (N_DRAWS + 1)
                        ),
                        "observed_absolute_rho": observed_absolute,
                        "null_absolute_median": float(
                            np.median(null_absolute)
                        ),
                        "absolute_margin": float(
                            observed_absolute - np.median(null_absolute)
                        ),
                        "absolute_empirical_p": float(
                            (
                                1
                                + np.count_nonzero(
                                    null_absolute >= observed_absolute
                                )
                            )
                            / (N_DRAWS + 1)
                        ),
                        "rbf_lengthscale_median": float(
                            np.median(lengthscales)
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "subject": subject,
                        "model": model,
                        "field": "teacher_forced_union_participation",
                        "score_direction": "positive_signed",
                        "null_mode": "geometry_smooth_rbf",
                        "eligible": False,
                        "n_contacts": len(joined_names),
                        "n_seizures": len(used),
                    }
                )
        print(f"phase3 {patient_index + 1}/16 {subject}", flush=True)

    patient = pd.DataFrame(rows).sort_values(
        ["null_mode", "model", "subject"]
    )
    patient.to_csv(
        OUT / "phase3_teacher_forced_patient_metrics.csv", index=False
    )
    field = pd.DataFrame(field_rows).sort_values(
        ["comparison", "subject", "seed"]
    )
    field.to_csv(
        OUT / "phase3_teacher_free_field_similarity_per_seed.csv", index=False
    )
    collapsed_field = (
        field.groupby(["comparison", "subject"], as_index=False)
        .spearman.median()
    )
    collapsed_field.to_csv(
        OUT / "phase3_teacher_free_field_similarity_patient.csv", index=False
    )
    field_summary = {
        comparison: bootstrap_summary(
            group.spearman.to_numpy(float), 2026080300 + index
        )
        for index, (comparison, group) in enumerate(
            collapsed_field.groupby("comparison")
        )
    }
    cohort: dict[str, Any] = {}
    for (model, null_mode), group in patient.loc[patient.eligible].groupby(
        ["model", "null_mode"]
    ):
        for metric in (
            "observed_signed_rho",
            "signed_margin",
            "absolute_margin",
        ):
            cohort[f"{model}__{null_mode}__{metric}"] = bootstrap_summary(
                group[metric].to_numpy(float),
                2026078300 + len(cohort),
            )

    phase1 = pd.read_csv(OUT / "phase1_existing_fields_patient_metrics.csv")
    phase2 = pd.read_csv(
        OUT / "phase2_regularized_baseline_patient_metrics.csv"
    )
    references = pd.concat(
        [
            phase1.loc[
                phase1.model.isin(
                    ["full_history_gru", "rank_shuffle_gru"]
                )
            ],
            phase2.loc[
                phase2.model == "best_validation_regularized_participation"
            ],
        ],
        ignore_index=True,
    )
    comparison_specs = (
        ("full_history_gru", "teacher_forced_full_gru"),
        ("teacher_forced_full_gru", "teacher_forced_rank_shuffle_gru"),
        (
            "teacher_forced_full_gru",
            "best_validation_regularized_participation",
        ),
    )
    combined = pd.concat([patient, references], ignore_index=True)
    comparisons: list[dict[str, Any]] = []
    for null_mode in sorted(patient.null_mode.unique()):
        subset = combined.loc[
            (combined.null_mode == null_mode) & combined.eligible
        ]
        for metric in (
            "observed_signed_rho",
            "signed_margin",
            "absolute_margin",
        ):
            family_start = len(comparisons)
            for left_name, right_name in comparison_specs:
                left = subset.loc[
                    subset.model == left_name
                ].set_index("subject")
                right = subset.loc[
                    subset.model == right_name
                ].set_index("subject")
                common = left.index.intersection(right.index)
                difference = (
                    left.loc[common, metric] - right.loc[common, metric]
                ).to_numpy(float)
                comparisons.append(
                    {
                        "null_mode": null_mode,
                        "metric": metric,
                        "left": left_name,
                        "right": right_name,
                        **bootstrap_summary(
                            difference,
                            2026079300 + len(comparisons),
                        ),
                    }
                )
            q_values = bh_fdr(
                [
                    comparisons[index]["wilcoxon_greater_p"]
                    for index in range(family_start, len(comparisons))
                ]
            )
            for index, q_value in zip(
                range(family_start, len(comparisons)), q_values
            ):
                comparisons[index]["family_bh_fdr_q"] = q_value
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(
        OUT / "phase3_teacher_free_paired_comparisons.csv", index=False
    )
    result = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "phase": "teacher_forced_free_rollout_decomposition",
        "status": "COMPLETE",
        "teacher_forced_definition": (
            "event-first mean union probability along observed heldout20 "
            "prefixes; includes terminal STOP competition"
        ),
        "not_free_running": True,
        "n_cells": 96,
        "n_patients": 16,
        "n_seizures": 106,
        "cohort_metrics": cohort,
        "field_similarity": field_summary,
        "paired_comparisons": comparisons,
    }
    atomic_json(OUT / "PHASE3_TEACHER_FORCED_SUMMARY.json", result)
    atomic_json(
        OUT / "RUN_STATUS.json",
        {
            "status": "PHASE3_COMPLETE_CONFOUND_AUDIT_PENDING",
            "input_audit": "INPUT_AUDIT.json",
            "phase1_summary": "PHASE1_EXISTING_FIELDS_SUMMARY.json",
            "baseline_freeze": "BASELINE_FREEZE.json",
            "phase2_summary": "PHASE2_REGULARIZED_BASELINE_SUMMARY.json",
            "phase3_summary": "PHASE3_TEACHER_FORCED_SUMMARY.json",
        },
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "n_patient_rows": len(patient),
                "n_comparisons": len(comparison),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
