#!/usr/bin/env python3
"""Strict clinical-onset read-back after target-blind hidden-direction freeze."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    N_PERM,
    ordinary_model_fields,
    strict_clinical_inventory,
    load_target,
)
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402


BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
V25 = ROOT / "results/topic5_rnn_bidirectional_cross_model_audit_v2_5"
SEED_DIRS = ("seed_20260725", "seed_20260726", "seed_20260727")
FIXED_FIELDS = ("participation", "endpoint_joint_mass")
ORDINARY_MODELS = (
    "full_history_gru",
    "rank_shuffle_gru",
    "unordered_prefix",
    "last_set_first_order",
    "static_contact_hazard",
    "empirical_rank_distribution",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def centered_rank(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(np.asarray(values, dtype=np.float64))
    return ranked - ranked.mean()


def permutation_indices(
    names: np.ndarray, *, n_perm: int, seed: int, within_shaft: bool
) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(int(seed))
    names = np.asarray(names).astype(str)
    if not within_shaft:
        return (
            np.row_stack([rng.permutation(len(names)) for _ in range(n_perm)]),
            {
                "eligible": True,
                "n_shafts": int(len({parse_shaft(name)[0] for name in names})),
                "shufflable_fraction": 1.0,
            },
        )
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        groups.setdefault(parse_shaft(name)[0], []).append(index)
    shufflable = sum(len(indices) for indices in groups.values() if len(indices) >= 2)
    result = np.tile(np.arange(len(names)), (n_perm, 1))
    for draw in range(n_perm):
        for indices in groups.values():
            if len(indices) >= 2:
                result[draw, indices] = rng.permutation(indices)
    return (
        result,
        {
            "eligible": bool(shufflable >= 4 and shufflable / len(names) >= 0.5),
            "n_shafts": int(len(groups)),
            "shufflable_fraction": float(shufflable / len(names)),
        },
    )


def score_fixed_field(
    seed_fields: list[np.ndarray],
    target: np.ndarray,
    permutations: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    target_rank = np.row_stack([centered_rank(row) for row in target])
    target_norm = np.linalg.norm(target_rank, axis=1)
    observed_signed = []
    observed_absolute = []
    null_by_seed = []
    for field in seed_fields:
        field_rank = centered_rank(field)
        field_norm = np.linalg.norm(field_rank)
        if field_norm <= 0:
            return float("nan"), float("nan"), np.full(len(permutations), np.nan)
        correlations = (
            target_rank @ field_rank
        ) / np.maximum(target_norm * field_norm, 1.0e-12)
        observed_signed.append(float(np.median(correlations)))
        observed_absolute.append(float(np.median(np.abs(correlations))))
        per_seizure = []
        for seizure_index in range(len(target_rank)):
            shuffled = target_rank[seizure_index][permutations]
            rho = (
                shuffled @ field_rank
            ) / max(target_norm[seizure_index] * field_norm, 1.0e-12)
            per_seizure.append(np.abs(rho))
        null_by_seed.append(np.median(np.column_stack(per_seizure), axis=1))
    null = np.median(np.column_stack(null_by_seed), axis=1)
    return (
        float(np.median(observed_signed)),
        float(np.median(observed_absolute)),
        null,
    )


def internal_fields(
    subject: str, contact_names: np.ndarray
) -> dict[str, dict[str, list[np.ndarray]]]:
    frame = pd.read_csv(BASE / "interictal_direction_contact_fields.csv")
    frame = frame.loc[
        (frame.subject == subject)
        & (frame.control.isin(("full_history_gru", "rank_shuffle_gru")))
        & np.isclose(frame.amplitude_sd, 0.5)
        & (frame.event_half == "all")
    ].copy()
    output: dict[str, dict[str, list[np.ndarray]]] = {}
    for key, group in frame.groupby(
        ["control", "direction_type", "direction_index"]
    ):
        control, direction_type, direction_index = key
        model = f"internal_{control}_{direction_type}{int(direction_index)}"
        output[model] = {
            "probability_contrast": [],
            "probability_contrast_residual_participation": [],
        }
        for seed_dir in SEED_DIRS:
            seed = group.loc[group.seed_dir == seed_dir].sort_values(
                "contact_index"
            )
            if not np.array_equal(
                seed.contact_name.astype(str).to_numpy(), contact_names
            ):
                raise RuntimeError(f"{subject}/{model}: contact ordering drifted")
            field = seed.probability_contrast.to_numpy(float)
            participation = seed.train80_participation.to_numpy(float)
            design = np.column_stack([np.ones(len(field)), participation])
            coefficient, *_ = np.linalg.lstsq(design, field, rcond=None)
            residual = field - design @ coefficient
            output[model]["probability_contrast"].append(field)
            output[model][
                "probability_contrast_residual_participation"
            ].append(residual)
    return output


def bootstrap_summary(values: np.ndarray, seed: int, alternative: str) -> dict:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"n": 0}
    rng = np.random.default_rng(int(seed))
    sampled = rng.choice(data, size=(20_000, len(data)), replace=True)
    p = (
        1.0
        if np.allclose(data, 0.0)
        else float(wilcoxon(data, alternative=alternative).pvalue)
    )
    return {
        "n": int(len(data)),
        "median": float(np.median(data)),
        "bootstrap_ci95": np.quantile(
            np.median(sampled, axis=1), [0.025, 0.975]
        ).tolist(),
        "n_positive": int(np.count_nonzero(data > 0)),
        "wilcoxon_p": p,
        "alternative": alternative,
    }


def bh_fdr(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p)
    adjusted = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    args = parser.parse_args()
    if args.n_perm != 5000:
        raise SystemExit("formal read-back freezes n_perm=5000")
    freeze = json.loads((BASE / "INTERICTAL_FREEZE.json").read_text())
    if freeze.get("status") != "FROZEN":
        raise SystemExit("interictal directions are not frozen")
    if freeze.get("target_values_read") or freeze.get(
        "early_ictal_arrays_deserialized"
    ):
        raise SystemExit("interictal freeze is not target blind")

    inventory = strict_clinical_inventory()
    rows = []
    for patient_index, (subject, seizures) in enumerate(inventory.items()):
        names, ordinary = ordinary_model_fields(subject)
        keep, target, used = load_target(subject, seizures, names)
        joined_names = names[keep]
        models: dict[str, dict[str, list[np.ndarray]]] = {}
        for model in ORDINARY_MODELS:
            models[model] = {
                field: [seed[field][keep] for seed in ordinary[model]]
                for field in FIXED_FIELDS
            }
        for model, fields in internal_fields(subject, names).items():
            models[model] = {
                field_name: [field[keep] for field in seed_fields]
                for field_name, seed_fields in fields.items()
            }
        all_perm, all_audit = permutation_indices(
            joined_names,
            n_perm=args.n_perm,
            seed=2026072800 + patient_index,
            within_shaft=False,
        )
        shaft_perm, shaft_audit = permutation_indices(
            joined_names,
            n_perm=args.n_perm,
            seed=2026073800 + patient_index,
            within_shaft=True,
        )
        for model, fields in models.items():
            for field, seed_fields in fields.items():
                target_splits = {
                    "all": target,
                    "alternating_a": target[::2],
                    "alternating_b": target[1::2],
                }
                for seizure_split, split_target in target_splits.items():
                    if len(split_target) == 0:
                        continue
                    signed, absolute, all_null = score_fixed_field(
                        seed_fields, split_target, all_perm
                    )
                    _, _, shaft_null = score_fixed_field(
                        seed_fields, split_target, shaft_perm
                    )
                    rows.append(
                        {
                            "subject": subject,
                            "model": model,
                            "field": field,
                            "seizure_split": seizure_split,
                            "n_contacts": int(len(keep)),
                            "n_seizures": int(len(split_target)),
                            "signed_rho": signed,
                            "absolute_rho": absolute,
                            "all_contact_null_median": float(
                                np.nanmedian(all_null)
                            ),
                            "all_contact_margin": float(
                                absolute - np.nanmedian(all_null)
                            ),
                            "all_contact_empirical_p": float(
                                (
                                    1
                                    + np.count_nonzero(
                                        np.asarray(all_null) >= absolute
                                    )
                                )
                                / (args.n_perm + 1)
                            ),
                            "within_shaft_eligible": shaft_audit["eligible"],
                            "within_shaft_n_shafts": shaft_audit["n_shafts"],
                            "within_shaft_shufflable_fraction": shaft_audit[
                                "shufflable_fraction"
                            ],
                            "within_shaft_null_median": float(
                                np.nanmedian(shaft_null)
                            ),
                            "within_shaft_margin": float(
                                absolute - np.nanmedian(shaft_null)
                            ),
                            "within_shaft_empirical_p": float(
                                (
                                    1
                                    + np.count_nonzero(
                                        np.asarray(shaft_null) >= absolute
                                    )
                                )
                                / (args.n_perm + 1)
                            ),
                        }
                    )
        print(f"readback {patient_index + 1}/16 {subject}", flush=True)

    metrics = pd.DataFrame(rows).sort_values(["model", "field", "subject"])
    metrics.to_csv(BASE / "early_ictal_fixed_readback_patient_metrics.csv", index=False)
    cohort = {}
    primary_metrics = metrics.loc[metrics.seizure_split == "all"].copy()
    for (model, field), group in primary_metrics.groupby(["model", "field"]):
        for metric in ("absolute_rho", "all_contact_margin"):
            cohort[f"{model}__{field}__{metric}"] = bootstrap_summary(
                group[metric].to_numpy(float),
                2026074000 + len(cohort),
                "greater",
            )
        shaft = group.loc[group.within_shaft_eligible]
        cohort[f"{model}__{field}__within_shaft_margin"] = bootstrap_summary(
            shaft.within_shaft_margin.to_numpy(float),
            2026074000 + len(cohort),
            "greater",
        )

    comparisons = []
    ordinary_pairs = (
        ("full_history_gru", "static_contact_hazard"),
        ("full_history_gru", "unordered_prefix"),
        ("full_history_gru", "last_set_first_order"),
        ("full_history_gru", "rank_shuffle_gru"),
        ("full_history_gru", "empirical_rank_distribution"),
    )
    for field in FIXED_FIELDS:
        for metric in ("absolute_rho", "all_contact_margin", "within_shaft_margin"):
            wide = primary_metrics.loc[
                (primary_metrics.field == field)
                & (
                    primary_metrics.within_shaft_eligible
                    if metric == "within_shaft_margin"
                    else True
                )
            ].pivot(index="subject", columns="model", values=metric)
            family_start = len(comparisons)
            for left, right in ordinary_pairs:
                difference = (wide[left] - wide[right]).dropna()
                result = bootstrap_summary(
                    difference.to_numpy(float),
                    2026075000 + len(comparisons),
                    "greater",
                )
                comparisons.append(
                    {
                        "field": field,
                        "metric": metric,
                        "left": left,
                        "right": right,
                        **result,
                    }
                )
            q = bh_fdr(
                [
                    comparisons[index]["wilcoxon_p"]
                    for index in range(family_start, len(comparisons))
                ]
            )
            for index, value in zip(
                range(family_start, len(comparisons)), q
            ):
                comparisons[index]["family_bh_fdr_q"] = value
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(
        BASE / "early_ictal_fixed_readback_paired_comparisons.csv", index=False
    )
    internal_comparisons = []
    internal = primary_metrics.loc[
        primary_metrics.model.str.startswith("internal_")
    ].copy()
    for field in sorted(internal.field.unique()):
        for direction_type in ("pca", "output_coupled"):
            for direction_index in (1, 2):
                full = (
                    f"internal_full_history_gru_"
                    f"{direction_type}{direction_index}"
                )
                shuffled = (
                    f"internal_rank_shuffle_gru_"
                    f"{direction_type}{direction_index}"
                )
                for metric in (
                    "absolute_rho",
                    "all_contact_margin",
                    "within_shaft_margin",
                ):
                    subset = internal.loc[
                        (internal.field == field)
                        & (
                            internal.within_shaft_eligible
                            if metric == "within_shaft_margin"
                            else True
                        )
                    ]
                    wide = subset.pivot(
                        index="subject", columns="model", values=metric
                    )
                    difference = (wide[full] - wide[shuffled]).dropna()
                    result = bootstrap_summary(
                        difference.to_numpy(float),
                        2026075500 + len(internal_comparisons),
                        "greater",
                    )
                    internal_comparisons.append(
                        {
                            "field": field,
                            "direction_type": direction_type,
                            "direction_index": direction_index,
                            "metric": metric,
                            "left": full,
                            "right": shuffled,
                            **result,
                        }
                    )
    internal_comparison_frame = pd.DataFrame(internal_comparisons)
    for (_, _), indices in internal_comparison_frame.groupby(
        ["field", "metric"]
    ).groups.items():
        ordered_indices = list(indices)
        q_values = bh_fdr(
            internal_comparison_frame.loc[
                ordered_indices, "wilcoxon_p"
            ].astype(float).tolist()
        )
        internal_comparison_frame.loc[
            ordered_indices, "direction_family_bh_fdr_q"
        ] = q_values
        for index, q_value in zip(ordered_indices, q_values):
            internal_comparisons[int(index)][
                "direction_family_bh_fdr_q"
            ] = float(q_value)
    internal_comparison_frame.to_csv(
        BASE / "early_ictal_internal_full_vs_rank_shuffle.csv", index=False
    )
    split_rows = []
    split_metrics = metrics.loc[
        metrics.seizure_split.isin(("alternating_a", "alternating_b"))
        & (
            metrics.field
            == "probability_contrast_residual_participation"
        )
        & metrics.model.str.contains("_pca")
    ]
    for direction_index in (1, 2):
        full = f"internal_full_history_gru_pca{direction_index}"
        shuffled = f"internal_rank_shuffle_gru_pca{direction_index}"
        patient_halves = {}
        for seizure_split in ("alternating_a", "alternating_b"):
            subset = split_metrics.loc[
                split_metrics.seizure_split == seizure_split
            ]
            wide = subset.pivot(
                index="subject",
                columns="model",
                values="all_contact_margin",
            )
            difference = (wide[full] - wide[shuffled]).dropna()
            patient_halves[seizure_split] = difference
            result_half = bootstrap_summary(
                difference.to_numpy(float),
                2026075800 + len(split_rows),
                "greater",
            )
            split_rows.append(
                {
                    "direction_index": direction_index,
                    "seizure_split": seizure_split,
                    "metric": "all_contact_margin",
                    **result_half,
                }
            )
        paired = pd.concat(patient_halves, axis=1).dropna()
        split_rows.append(
            {
                "direction_index": direction_index,
                "seizure_split": "a_vs_b_patient_spearman",
                "metric": "all_contact_margin",
                "n": int(len(paired)),
                "median": float(
                    pd.Series(paired.iloc[:, 0]).corr(
                        pd.Series(paired.iloc[:, 1]), method="spearman"
                    )
                ),
                "n_positive": int(
                    np.count_nonzero(
                        (paired.iloc[:, 0] > 0) & (paired.iloc[:, 1] > 0)
                    )
                ),
            }
        )
    split_frame = pd.DataFrame(split_rows)
    split_frame.to_csv(
        BASE / "early_ictal_alternating_seizure_validation.csv", index=False
    )
    v25_summary = json.loads((V25 / "STATIC_TRANSFER_SUMMARY.json").read_text())
    result = {
        "contract": "topic5_rnn_internal_state_reduction_v0_1_early_ictal_readback",
        "status": "COMPLETE",
        "strict_clinical_onset_cohort": {
            "n_patients": len(inventory),
            "n_seizures": int(sum(map(len, inventory.values()))),
            "dataset": "epilepsiae",
            "window_sec": [0, 10],
            "band_hz": [1, 150],
            "yuquan_eeg_onset_in_primary": False,
        },
        "fixed_readouts": list(FIXED_FIELDS),
        "primary_null": "5000 coherent within-patient all-contact permutations",
        "anatomical_sensitivity": (
            "5000 coherent within-shaft permutations; eligibility reported per patient"
        ),
        "target_opening_context": (
            "same strict target was already opened in v2.5; this is mechanism "
            "decomposition, not independent confirmation"
        ),
        "five_field_omnibus_sensitivity_source": (
            "results/topic5_rnn_bidirectional_cross_model_audit_v2_5/"
            "STATIC_TRANSFER_SUMMARY.json"
        ),
        "five_field_omnibus_full_history": v25_summary["model_summaries"][
            "full_history_gru"
        ],
        "cohort_metrics": cohort,
        "paired_comparisons": comparisons,
        "internal_full_vs_rank_shuffle": internal_comparisons,
        "alternating_seizure_validation": split_rows,
    }
    atomic_json(BASE / "EARLY_ICTAL_READBACK_SUMMARY.json", result)
    atomic_json(
        BASE / "RUN_STATUS.json",
        {
            "status": "ANALYSES_COMPLETE",
            "interictal": json.loads((BASE / "INTERICTAL_SUMMARY.json").read_text()),
            "early_ictal": result,
        },
    )
    print(json.dumps(result["strict_clinical_onset_cohort"], indent=2))


if __name__ == "__main__":
    main()
