#!/usr/bin/env python3
"""Test early-ictal increment of frozen ordered-history fields."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_rnn_bidirectional_cross_model_v2_5 import (  # noqa: E402
    load_target,
    strict_clinical_inventory,
)
from src.topic5_static_scaffold_validation import partial_rank_score  # noqa: E402


BASELINE = (
    ROOT
    / "results/topic5_static_scaffold_fixed_readout_validation/"
    "target_free_baselines/per_subject"
)
SEEDS = (20260725, 20260726, 20260727)


def centered_rank(values: np.ndarray) -> np.ndarray:
    ranked = rankdata(np.asarray(values, dtype=float))
    return ranked - ranked.mean()


def cohort(values: np.ndarray, seed: int) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {
            "n_patients": 0,
            "median": np.nan,
            "bootstrap_ci95": [np.nan, np.nan],
            "n_positive": 0,
            "wilcoxon_greater_p": np.nan,
        }
    rng = np.random.default_rng(seed)
    bootstrap = rng.choice(x, (20000, len(x)), replace=True)
    try:
        p = float(wilcoxon(x, alternative="greater").pvalue)
    except ValueError:
        p = 1.0
    return {
        "n_patients": int(len(x)),
        "median": float(np.median(x)),
        "bootstrap_ci95": np.quantile(
            np.median(bootstrap, axis=1), [0.025, 0.975]
        ).tolist(),
        "n_positive": int(np.count_nonzero(x > 0)),
        "wilcoxon_greater_p": p,
    }


def simple_score(field: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    values = np.asarray(
        [spearmanr(field, seizure).statistic for seizure in target],
        dtype=float,
    )
    return float(np.median(values)), float(np.median(np.abs(values)))


def residual_fraction(field: np.ndarray, covariates: np.ndarray) -> float:
    y = centered_rank(field)
    design = np.column_stack([np.ones(len(y)), covariates])
    fitted = design @ (np.linalg.pinv(design) @ y)
    denominator = float(np.sum(y * y))
    return (
        float(np.sum((y - fitted) ** 2) / denominator)
        if denominator > 0
        else np.nan
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    intervention_root = (
        args.intervention_root
        if args.intervention_root.is_absolute()
        else ROOT / args.intervention_root
    )
    output = args.output if args.output.is_absolute() else ROOT / args.output
    selection = json.loads(
        (
            ROOT
            / "results/topic5_ordered_history_architecture_audit/analysis/"
            "ARCHITECTURE_SUMMARY.json"
        ).read_text()
    )
    if not selection["target_blind_best_non_gru"][
        "matched_within_event_rank_shuffle_complete"
    ]:
        raise RuntimeError("matched shuffled model is not frozen")
    selected = selection["target_blind_best_non_gru"]["control"]
    inventory = strict_clinical_inventory()
    rows = []
    for patient_index, (subject, seizures) in enumerate(inventory.items()):
        seed_fields: dict[str, list[np.ndarray]] = {
            "selected_ordered": [],
            "selected_rank_shuffle": [],
            "unordered_prefix": [],
            "full_history_gru": [],
        }
        names = None
        for seed in SEEDS:
            path = (
                intervention_root
                / f"seed_{seed}"
                / subject
                / "teacher_forced_fields.npz"
            )
            with np.load(path, allow_pickle=False) as data:
                current_names = np.asarray(data["contact_names"]).astype(str)
                if names is None:
                    names = current_names
                elif not np.array_equal(names, current_names):
                    raise RuntimeError(f"{subject}: contact ordering drifted")
                for label in seed_fields:
                    seed_fields[label].append(
                        np.asarray(
                            data[f"{label}_union_participation"],
                            dtype=np.float64,
                        )
                    )
        if names is None:
            raise RuntimeError(f"{subject}: no frozen fields")
        keep, target, used = load_target(subject, seizures, names)
        fields = {
            label: np.median(np.row_stack(values), axis=0)[keep]
            for label, values in seed_fields.items()
        }
        with np.load(BASELINE / f"{subject}.npz", allow_pickle=False) as data:
            baseline_names = np.asarray(data["contact_names"]).astype(str)
            if not np.array_equal(names, baseline_names):
                raise RuntimeError(f"{subject}: baseline contact ordering drifted")
            raw = np.asarray(
                data["raw_train80_participation"], dtype=np.float64
            )[keep]
            regularized = np.asarray(
                data["best_validation_regularized_participation"],
                dtype=np.float64,
            )[keep]
        covariate_sets = {
            "static_only": np.column_stack(
                [centered_rank(raw), centered_rank(regularized)]
            ),
            "static_plus_unordered": np.column_stack(
                [
                    centered_rank(raw),
                    centered_rank(regularized),
                    centered_rank(fields["unordered_prefix"]),
                ]
            ),
            "static_unordered_plus_matched_shuffle": np.column_stack(
                [
                    centered_rank(raw),
                    centered_rank(regularized),
                    centered_rank(fields["unordered_prefix"]),
                    centered_rank(fields["selected_rank_shuffle"]),
                ]
            ),
        }
        for field_label, field in fields.items():
            simple_signed, simple_absolute = simple_score(field, target)
            for covariate_label, covariates in covariate_sets.items():
                result = partial_rank_score(
                    field,
                    target,
                    covariates,
                    min_residual_df=3,
                    n_null_draws=5000,
                    null_seed=(
                        20261100
                        + patient_index * 100
                        + list(fields).index(field_label) * 10
                        + list(covariate_sets).index(covariate_label)
                    ),
                )
                rows.append(
                    {
                        "subject": subject,
                        "selected_architecture": selected,
                        "field": field_label,
                        "conditioning": covariate_label,
                        "eligible": bool(result["eligible"]),
                        "n_contacts": len(keep),
                        "n_seizures": len(used),
                        "simple_signed_rho": simple_signed,
                        "simple_absolute_rho": simple_absolute,
                        "rank_residual_fraction": residual_fraction(
                            field, covariates
                        ),
                        **{
                            key: value
                            for key, value in result.items()
                            if key != "per_seizure_signed_rho"
                        },
                    }
                )
    patient = pd.DataFrame(rows)
    output.mkdir(parents=True, exist_ok=True)
    patient.to_csv(
        output / "early_ictal_conditional_patient_metrics.csv", index=False
    )
    summaries = {}
    eligible = patient.loc[patient.eligible]
    for index, ((field, conditioning), group) in enumerate(
        eligible.groupby(["field", "conditioning"])
    ):
        for metric in (
            "signed_rho",
            "absolute_rho",
            "signed_margin",
            "absolute_margin",
            "rank_residual_fraction",
        ):
            summaries[f"{field}__{conditioning}__{metric}"] = cohort(
                group[metric].to_numpy(float), 20261200 + index * 10
            )
    paired_rows = []
    for conditioning in (
        "static_only",
        "static_plus_unordered",
        "static_unordered_plus_matched_shuffle",
    ):
        subset = eligible.loc[eligible.conditioning.eq(conditioning)]
        wide_abs = subset.pivot(
            index="subject", columns="field", values="absolute_rho"
        )
        wide_signed = subset.pivot(
            index="subject", columns="field", values="signed_rho"
        )
        for metric, wide in (
            ("absolute_rho", wide_abs),
            ("signed_rho", wide_signed),
        ):
            for reference in (
                "selected_rank_shuffle",
                "unordered_prefix",
                "full_history_gru",
            ):
                # A field explicitly included in the conditioning block has
                # zero residual by construction and is ineligible. Do not
                # manufacture a paired comparison against that field.
                if (
                    "selected_ordered" not in wide.columns
                    or reference not in wide.columns
                ):
                    continue
                common = wide.index[
                    wide[["selected_ordered", reference]].notna().all(axis=1)
                ]
                difference = (
                    wide.loc[common, "selected_ordered"]
                    - wide.loc[common, reference]
                )
                paired_rows.append(
                    {
                        "conditioning": conditioning,
                        "metric": metric,
                        "left": "selected_ordered",
                        "right": reference,
                        **cohort(
                            difference.to_numpy(float),
                            20261300 + len(paired_rows),
                        ),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(
        output / "early_ictal_conditional_paired_comparisons.csv", index=False
    )
    primary = summaries.get(
        "selected_ordered__static_plus_unordered__absolute_margin", {}
    )
    matched = paired.loc[
        paired.conditioning.eq("static_plus_unordered")
        & paired.metric.eq("absolute_rho")
        & paired.right.eq("selected_rank_shuffle")
    ].iloc[0].to_dict()
    supported = bool(
        primary.get("median", -np.inf) > 0
        and primary.get("wilcoxon_greater_p", 1.0) < 0.05
        and matched.get("median", -np.inf) > 0
        and matched.get("wilcoxon_greater_p", 1.0) < 0.05
    )
    summary = {
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "status": "COMPLETE",
        "selected_architecture": selected,
        "n_patients": 16,
        "n_seizures": 106,
        "target": {
            "time_reference": "clinical_onset",
            "window_sec": [0, 10],
            "band_hz": [1, 150],
            "target_reused_not_independent_confirmation": True,
        },
        "estimand": (
            "partial rank correspondence of the frozen ordered teacher-forced "
            "contact field after static participation and unordered-prefix "
            "fields; operational proxy, not a direct mutual-information estimate"
        ),
        "conditional_early_ictal_increment_status": (
            "SUPPORTED_WITHIN_REUSED_TARGET"
            if supported
            else "NOT_ESTABLISHED"
        ),
        "cohort_summaries": summaries,
        "paired_comparisons": paired_rows,
        "across_event_history_target_branch": (
            "NOT_RUN_AS_PRIMARY: only 46 distinct causal histories among 106 "
            "seizures and only 6 patients with at least three distinct histories"
        ),
        "claim_boundary": (
            "static cross-state contact morphology only; no seizure-specific "
            "forecast, continuous-time dynamics, or biological slow-state claim"
        ),
    }
    (output / "EARLY_ICTAL_CONDITIONAL_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
