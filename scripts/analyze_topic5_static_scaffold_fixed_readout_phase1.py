#!/usr/bin/env python3
"""Phase 1 fixed signed participation readout for existing frozen models."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    load_target,
    ordinary_model_fields,
    strict_clinical_inventory,
)
from src.topic5_static_scaffold_validation import (  # noqa: E402
    coherent_index_null,
    geometry_smooth_surrogates,
    score_signed_field,
)


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
OUT = ROOT / "results/topic5_static_scaffold_fixed_readout_validation"
MODELS = (
    "empirical_rank_distribution",
    "static_contact_hazard",
    "unordered_prefix",
    "last_set_first_order",
    "rank_shuffle_gru",
    "full_history_gru",
)
INDEX_NULLS = (
    "all_contact",
    "within_shaft_circular",
    "within_shaft_dihedral",
    "equal_size_shaft_profile",
)
N_DRAWS = 5000


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bootstrap_summary(values: np.ndarray, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0}
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(values, size=(20_000, len(values)), replace=True)
    nonzero = values[values != 0.0]
    p = (
        1.0
        if not len(nonzero)
        else float(
            wilcoxon(
                nonzero,
                alternative="greater",
                method="exact" if len(nonzero) <= 20 else "approx",
            ).pvalue
        )
    )
    return {
        "n": int(len(values)),
        "median": float(np.median(values)),
        "bootstrap_ci95": np.quantile(
            np.median(sampled, axis=1), [0.025, 0.975]
        ).tolist(),
        "n_positive": int(np.count_nonzero(values > 0)),
        "wilcoxon_greater_p": p,
    }


def bh_fdr(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def load_coords(subject: str, expected_names: np.ndarray) -> np.ndarray:
    path = DATASET / "per_subject" / f"{subject}.npz"
    with np.load(path, allow_pickle=False) as data:
        names = np.asarray(data["contact_names"]).astype(str)
        coords = np.asarray(data["contact_coords"], dtype=np.float64)
    if not np.array_equal(names, expected_names):
        raise RuntimeError(f"{subject}: dataset/model contact ordering drifted")
    return coords


def collapse_seed_scores(
    seed_fields: list[np.ndarray],
    target: np.ndarray,
    *,
    index: np.ndarray | None = None,
    coords: np.ndarray | None = None,
    standard_normal: list[np.ndarray] | None = None,
) -> tuple[dict[str, float | np.ndarray], list[float]]:
    scores = []
    lengthscales = []
    for seed_index, field in enumerate(seed_fields):
        if index is not None:
            null_fields = np.asarray(field, dtype=np.float64)[index]
        else:
            if coords is None or standard_normal is None:
                raise ValueError("geometry null inputs missing")
            null_fields, scale = geometry_smooth_surrogates(
                field,
                coords,
                standard_normal=standard_normal[seed_index],
            )
            lengthscales.append(float(scale))
        scores.append(score_signed_field(field, target, null_fields))
    return (
        {
            "observed_signed": float(
                np.median([score["observed_signed"] for score in scores])
            ),
            "observed_absolute": float(
                np.median([score["observed_absolute"] for score in scores])
            ),
            "null_signed": np.median(
                np.column_stack([score["null_signed"] for score in scores]),
                axis=1,
            ),
            "null_absolute": np.median(
                np.column_stack([score["null_absolute"] for score in scores]),
                axis=1,
            ),
        },
        lengthscales,
    )


def main() -> None:
    audit = json.loads((OUT / "INPUT_AUDIT.json").read_text())
    if audit["target_values_read"] or audit["early_ictal_arrays_deserialized"]:
        raise RuntimeError("metadata input audit was not target blind")
    inventory = strict_clinical_inventory()
    rows = []
    for patient_index, (subject, seizures) in enumerate(inventory.items()):
        names, fields = ordinary_model_fields(subject)
        keep, target, used = load_target(subject, seizures, names)
        joined_names = names[keep]
        coords = load_coords(subject, names)[keep]
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
        for model in MODELS:
            seed_fields = [
                np.asarray(seed["participation"], dtype=np.float64)[keep]
                for seed in fields[model]
            ]
            for null_mode, (indices, null_audit) in index_nulls.items():
                eligible = bool(null_audit["eligible"])
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
                null_absolute = np.asarray(score["null_absolute"], dtype=np.float64)
                observed_signed = float(score["observed_signed"])
                observed_absolute = float(score["observed_absolute"])
                null_signed_median = (
                    float(np.median(null_signed)) if eligible else np.nan
                )
                null_absolute_median = (
                    float(np.median(null_absolute)) if eligible else np.nan
                )
                rows.append(
                    {
                        "subject": subject,
                        "model": model,
                        "field": "participation",
                        "score_direction": "positive_signed",
                        "null_mode": null_mode,
                        "eligible": eligible,
                        "n_contacts": len(joined_names),
                        "n_seizures": len(used),
                        "n_shafts": null_audit["n_shafts"],
                        "movable_fraction": null_audit["movable_fraction"],
                        "observed_signed_rho": observed_signed,
                        "null_signed_median": null_signed_median,
                        "signed_margin": (
                            float(observed_signed - null_signed_median)
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
                        "null_absolute_median": null_absolute_median,
                        "absolute_margin": (
                            float(observed_absolute - null_absolute_median)
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
                    seed_fields,
                    target,
                    coords=coords,
                    standard_normal=geometry_normal,
                )
                null_signed = np.asarray(score["null_signed"], dtype=np.float64)
                null_absolute = np.asarray(score["null_absolute"], dtype=np.float64)
                observed_signed = float(score["observed_signed"])
                observed_absolute = float(score["observed_absolute"])
                rows.append(
                    {
                        "subject": subject,
                        "model": model,
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
                            (1 + np.count_nonzero(null_signed >= observed_signed))
                            / (N_DRAWS + 1)
                        ),
                        "observed_absolute_rho": observed_absolute,
                        "null_absolute_median": float(np.median(null_absolute)),
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
                        "field": "participation",
                        "score_direction": "positive_signed",
                        "null_mode": "geometry_smooth_rbf",
                        "eligible": False,
                        "n_contacts": len(joined_names),
                        "n_seizures": len(used),
                    }
                )
        print(f"phase1 {patient_index + 1}/16 {subject}", flush=True)

    patient = pd.DataFrame(rows).sort_values(["null_mode", "model", "subject"])
    OUT.mkdir(parents=True, exist_ok=True)
    patient.to_csv(OUT / "phase1_existing_fields_patient_metrics.csv", index=False)
    cohort = {}
    for (model, null_mode), group in patient.loc[patient.eligible].groupby(
        ["model", "null_mode"]
    ):
        for metric in ("observed_signed_rho", "signed_margin", "absolute_margin"):
            cohort[f"{model}__{null_mode}__{metric}"] = bootstrap_summary(
                group[metric].to_numpy(float),
                2026078100 + len(cohort),
            )

    comparisons = []
    controls = tuple(model for model in MODELS if model != "full_history_gru")
    for null_mode, group in patient.loc[patient.eligible].groupby("null_mode"):
        for metric in (
            "observed_signed_rho",
            "signed_margin",
            "absolute_margin",
        ):
            wide = group.pivot(index="subject", columns="model", values=metric)
            family_start = len(comparisons)
            for control in controls:
                difference = (
                    wide.full_history_gru - wide[control]
                ).dropna()
                comparisons.append(
                    {
                        "null_mode": null_mode,
                        "metric": metric,
                        "left": "full_history_gru",
                        "right": control,
                        **bootstrap_summary(
                            difference.to_numpy(float),
                            2026079100 + len(comparisons),
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
        OUT / "phase1_existing_fields_paired_comparisons.csv", index=False
    )
    result = {
        "contract": "topic5_static_scaffold_fixed_readout_validation_v0_1",
        "phase": "existing_frozen_fields_fixed_signed_readout",
        "status": "COMPLETE",
        "target_context": (
            "same 16-patient/106-seizure target previously opened; strict "
            "internal validation, not independent confirmation"
        ),
        "primary_field": "participation",
        "primary_metric": "positive signed Spearman",
        "n_draws": N_DRAWS,
        "n_patients": 16,
        "n_seizures": 106,
        "cohort_metrics": cohort,
        "paired_comparisons": comparisons,
        "next_phase": (
            "target-free regularized nonrecurrent baselines and "
            "teacher-forced/free-rollout decomposition"
        ),
    }
    atomic_json(OUT / "PHASE1_EXISTING_FIELDS_SUMMARY.json", result)
    atomic_json(
        OUT / "RUN_STATUS.json",
        {
            "status": "PHASE1_COMPLETE_NEXT_BASELINES_PENDING",
            "input_audit": "INPUT_AUDIT.json",
            "phase1_summary": "PHASE1_EXISTING_FIELDS_SUMMARY.json",
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_patient_rows": len(patient),
                "n_comparisons": len(comparison),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
