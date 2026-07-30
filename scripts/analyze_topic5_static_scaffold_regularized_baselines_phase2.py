#!/usr/bin/env python3
"""Evaluate frozen target-free baselines against the fixed early-ictal target."""
from __future__ import annotations

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
PER_SUBJECT = OUT / "target_free_baselines/per_subject"
BASELINES = (
    "raw_train80_participation",
    "beta_binomial_participation",
    "shaft_laplacian_participation",
    "geometry_laplacian_participation",
    "dirichlet_rank_participation",
    "low_rank_logit_participation",
    "best_validation_regularized_participation",
)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    freeze = json.loads((OUT / "BASELINE_FREEZE.json").read_text())
    if (
        freeze["status"] != "COMPLETE"
        or freeze["target_values_read"]
        or freeze["early_ictal_arrays_deserialized"]
    ):
        raise RuntimeError("regularized baselines were not target-blind frozen")
    inventory = strict_clinical_inventory()
    rows: list[dict[str, Any]] = []
    similarities: list[dict[str, Any]] = []
    for patient_index, (subject, seizures) in enumerate(inventory.items()):
        ordinary_names, ordinary = ordinary_model_fields(subject)
        baseline_path = PER_SUBJECT / f"{subject}.npz"
        with np.load(baseline_path, allow_pickle=False) as data:
            baseline_names = np.asarray(data["contact_names"]).astype(str)
            baseline_fields = {
                baseline: np.asarray(data[baseline], dtype=np.float64)
                for baseline in BASELINES
            }
        if not np.array_equal(ordinary_names, baseline_names):
            raise RuntimeError(f"{subject}: baseline/model contact ordering drifted")
        keep, target, used = load_target(
            subject, seizures, baseline_names
        )
        joined_names = baseline_names[keep]
        coords = load_coords(subject, baseline_names)[keep]
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
                    2026077100 + patient_index * 10
                ).normal(size=(N_DRAWS, len(joined_names)))
            ]
            if geometry_eligible
            else None
        )
        full_field = np.median(
            np.row_stack(
                [
                    np.asarray(seed["participation"], dtype=np.float64)
                    for seed in ordinary["full_history_gru"]
                ]
            ),
            axis=0,
        )
        for baseline, field in baseline_fields.items():
            finite_field = bool(np.all(np.isfinite(field)))
            similarities.append(
                {
                    "subject": subject,
                    "baseline": baseline,
                    "eligible": finite_field,
                    "full_gru_field_spearman": (
                        float(spearmanr(full_field, field).statistic)
                        if finite_field
                        else np.nan
                    ),
                }
            )
            seed_fields = [field[keep]]
            for null_mode, (indices, null_audit) in index_nulls.items():
                eligible = bool(null_audit["eligible"] and finite_field)
                if eligible:
                    score, lengthscales = collapse_seed_scores(
                        seed_fields, target, index=indices
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
                        "model": baseline,
                        "field": "participation",
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
            geometry_field_eligible = bool(
                geometry_eligible and finite_field
            )
            if geometry_field_eligible:
                score, lengthscales = collapse_seed_scores(
                    seed_fields,
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
                        "model": baseline,
                        "field": "participation",
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
                        "model": baseline,
                        "field": "participation",
                        "score_direction": "positive_signed",
                        "null_mode": "geometry_smooth_rbf",
                        "eligible": False,
                        "n_contacts": len(joined_names),
                        "n_seizures": len(used),
                    }
                )
        print(f"phase2 {patient_index + 1}/16 {subject}", flush=True)

    patient = pd.DataFrame(rows).sort_values(
        ["null_mode", "model", "subject"]
    )
    patient.to_csv(
        OUT / "phase2_regularized_baseline_patient_metrics.csv", index=False
    )
    similarity = pd.DataFrame(similarities).sort_values(
        ["baseline", "subject"]
    )
    similarity.to_csv(
        OUT / "phase2_regularized_baseline_field_similarity.csv", index=False
    )
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
                2026078200 + len(cohort),
            )

    phase1 = pd.read_csv(OUT / "phase1_existing_fields_patient_metrics.csv")
    full = phase1.loc[
        (phase1.model == "full_history_gru") & phase1.eligible
    ]
    comparisons: list[dict[str, Any]] = []
    for null_mode in sorted(patient.null_mode.unique()):
        left = full.loc[full.null_mode == null_mode].set_index("subject")
        right_all = patient.loc[
            (patient.null_mode == null_mode) & patient.eligible
        ]
        for metric in (
            "observed_signed_rho",
            "signed_margin",
            "absolute_margin",
        ):
            family_start = len(comparisons)
            for baseline, right_group in right_all.groupby("model"):
                right = right_group.set_index("subject")
                common = left.index.intersection(right.index)
                difference = (
                    left.loc[common, metric] - right.loc[common, metric]
                ).to_numpy(float)
                comparisons.append(
                    {
                        "null_mode": null_mode,
                        "metric": metric,
                        "left": "full_history_gru",
                        "right": baseline,
                        **bootstrap_summary(
                            difference,
                            2026079200 + len(comparisons),
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
        OUT / "phase2_full_gru_vs_regularized_baselines.csv", index=False
    )
    similarity_summary = {
        baseline: bootstrap_summary(
            group.loc[group.eligible, "full_gru_field_spearman"].to_numpy(float),
            2026080200 + index,
        )
        for index, (baseline, group) in enumerate(
            similarity.groupby("baseline")
        )
    }
    result = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "phase": "regularized_baseline_fixed_target_evaluation",
        "status": "COMPLETE",
        "target_context": (
            "same 16-patient/106-seizure target previously opened; baseline "
            "choice was frozen target-free before this phase"
        ),
        "primary_field": "participation",
        "primary_metric": "positive signed Spearman",
        "n_draws": N_DRAWS,
        "n_patients": 16,
        "n_seizures": 106,
        "cohort_metrics": cohort,
        "full_gru_paired_comparisons": comparisons,
        "field_similarity_to_full_gru": similarity_summary,
    }
    atomic_json(OUT / "PHASE2_REGULARIZED_BASELINE_SUMMARY.json", result)
    atomic_json(
        OUT / "RUN_STATUS.json",
        {
            "status": "PHASE2_COMPLETE_TEACHER_FORCED_PENDING",
            "input_audit": "INPUT_AUDIT.json",
            "phase1_summary": "PHASE1_EXISTING_FIELDS_SUMMARY.json",
            "baseline_freeze": "BASELINE_FREEZE.json",
            "phase2_summary": "PHASE2_REGULARIZED_BASELINE_SUMMARY.json",
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
