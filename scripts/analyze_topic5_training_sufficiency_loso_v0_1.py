#!/usr/bin/env python3
"""Analyse the leave-one-patient-out phases (B1c confirmation, C, D).

Statistics follow one fixed order: metric inside a seed, seeds merged inside a
patient, patient as the unit of analysis, Epilepsiae and Yuquan reported
separately and combined.  Effect size, bootstrap CI, the number of improved
patients and the paired test are always reported together; a P value alone is
never the output.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: endpoint -> True when a larger value is better
ENDPOINTS = {
    "transition_correlation": True,
    "transition_mae": False,
    "suffix_rank_wasserstein": False,
    "suffix_precedence_correlation": True,
    "suffix_precedence_mae": False,
    "suffix_participation_mae": False,
    "whole_participation_mae": False,
    "whole_rank_wasserstein": False,
    "event_length_wasserstein": False,
    "stop_hazard_mae": False,
    "likelihood_contact_choice_nll": False,
    "likelihood_event_total_nll": False,
    "likelihood_stop_contribution_nll": False,
}
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260730


def _load(root: Path) -> pd.DataFrame:
    frames = []
    for done in sorted(root.rglob("DONE.json")):
        cell = done.parent
        path = cell / "cell_metrics.csv"
        if not path.is_file():
            raise RuntimeError(f"{cell}: DONE.json without cell_metrics.csv")
        frames.append(pd.read_csv(path))
    if not frames:
        raise RuntimeError(f"no completed LOSO cells under {root}")
    return pd.concat(frames, ignore_index=True)


def _patient_values(frame: pd.DataFrame, endpoint: str) -> dict:
    """Seed-mean inside a patient; patients weighted equally."""
    if endpoint not in frame.columns:
        return {}
    subset = frame[["subject", "seed", endpoint]].dropna()
    if subset.empty:
        return {}
    return {
        str(subject): float(group[endpoint].mean())
        for subject, group in subset.groupby("subject")
    }


def _bootstrap_median_ci(values: np.ndarray) -> tuple[float, float]:
    if values.size < 3:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, values.size), replace=True)
    medians = np.median(draws, axis=1)
    return (
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def _paired(
    treatment: dict,
    reference: dict,
    *,
    higher_is_better: bool,
) -> dict:
    subjects = sorted(set(treatment) & set(reference))
    if not subjects:
        return {"n_patients": 0}
    left = np.asarray([treatment[key] for key in subjects], dtype=float)
    right = np.asarray([reference[key] for key in subjects], dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = left[valid], right[valid]
    subjects = [key for key, keep in zip(subjects, valid) if keep]
    if left.size < 3:
        return {"n_patients": int(left.size)}
    # positive gain always means "treatment is better"
    gain = (left - right) if higher_is_better else (right - left)
    low, high = _bootstrap_median_ci(gain)
    nonzero = gain[gain != 0]
    if nonzero.size:
        wilcoxon = stats.wilcoxon(nonzero, alternative="two-sided")
        ranks = stats.rankdata(np.abs(nonzero))
        positive = float(np.sum(ranks[nonzero > 0]))
        negative = float(np.sum(ranks[nonzero < 0]))
        rank_biserial = (positive - negative) / (positive + negative)
        p_value = float(wilcoxon.pvalue)
    else:
        rank_biserial = 0.0
        p_value = 1.0
    return {
        "n_patients": int(left.size),
        "median_gain": float(np.median(gain)),
        "mean_gain": float(np.mean(gain)),
        "bootstrap_ci_median_gain": [low, high],
        "n_improved": int(np.sum(gain > 0)),
        "n_worse": int(np.sum(gain < 0)),
        "fraction_improved": float(np.mean(gain > 0)),
        "rank_biserial_correlation": float(rank_biserial),
        "wilcoxon_p": p_value,
        "treatment_median": float(np.median(left)),
        "reference_median": float(np.median(right)),
        "subjects": subjects,
    }


def _stratified_paired(
    frame: pd.DataFrame,
    treatment_mask: pd.Series,
    reference_mask: pd.Series,
    endpoint: str,
) -> dict:
    higher = ENDPOINTS[endpoint]
    out = {}
    for stratum in ("combined", "epilepsiae", "yuquan"):
        if stratum == "combined":
            subset = frame
        else:
            subset = frame[frame.dataset == stratum]
        treatment = _patient_values(subset[treatment_mask.reindex(subset.index, fill_value=False)], endpoint)
        reference = _patient_values(subset[reference_mask.reindex(subset.index, fill_value=False)], endpoint)
        out[stratum] = _paired(treatment, reference, higher_is_better=higher)
    return out


def _descriptive(frame: pd.DataFrame, endpoints) -> list[dict]:
    rows = []
    for (condition, rollout), group in frame.groupby(
        ["condition", "rollout_condition"], dropna=False
    ):
        row = {
            "condition": condition,
            "rollout_condition": rollout,
            "n_patients": int(group.subject.nunique()),
            "n_seeds": int(group.seed.nunique()),
            "n_cells": int(len(group)),
        }
        for endpoint in endpoints:
            values = np.asarray(
                list(_patient_values(group, endpoint).values()), dtype=float
            )
            values = values[np.isfinite(values)]
            row[f"{endpoint}__patient_median"] = (
                float(np.median(values)) if values.size else float("nan")
            )
            row[f"{endpoint}__patient_mean"] = (
                float(np.mean(values)) if values.size else float("nan")
            )
        rows.append(row)
    return rows


def analyse_b1c(frame: pd.DataFrame, out: Path) -> dict:
    frame = frame[frame.rollout_condition == "none"].copy()
    frame["arm"] = (
        frame.condition.astype(str) + "__offset" + frame.offset_cycles.astype(str)
    )
    endpoints = [
        "likelihood_contact_choice_nll",
        "likelihood_event_total_nll",
        "likelihood_stop_contribution_nll",
    ]
    rows = []
    for arm, group in frame.groupby("arm"):
        for endpoint in endpoints:
            values = np.asarray(
                list(_patient_values(group, endpoint).values()), dtype=float
            )
            rows.append(
                {
                    "arm": arm,
                    "condition": str(group.condition.iloc[0]),
                    "cycles": int(group.cycles.iloc[0]),
                    "updates_per_patient": int(group.updates_per_patient.iloc[0]),
                    "offset_cycles": int(group.offset_cycles.iloc[0]),
                    "endpoint": endpoint,
                    "n_patients": int(np.sum(np.isfinite(values))),
                    "patient_median": float(np.median(values[np.isfinite(values)])),
                    "patient_mean": float(np.mean(values[np.isfinite(values)])),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out / "b1c_arm_summary.csv", index=False)

    primary = table[table.endpoint == "likelihood_contact_choice_nll"]
    best = primary.loc[primary.patient_median.idxmin()]
    comparisons = {}
    for arm in sorted(frame.arm.unique()):
        if arm == best.arm:
            continue
        comparisons[f"{best.arm}__vs__{arm}"] = _paired(
            _patient_values(frame[frame.arm == best.arm], "likelihood_contact_choice_nll"),
            _patient_values(frame[frame.arm == arm], "likelihood_contact_choice_nll"),
            higher_is_better=False,
        )
    shared_only = primary.groupby(["condition", "offset_cycles"]).patient_median.first()
    return {
        "phase": "b1c",
        "arms": table.to_dict("records"),
        "best_arm": {
            "arm": str(best.arm),
            "condition": str(best.condition),
            "cycles": int(best.cycles),
            "updates_per_patient": int(best.updates_per_patient),
            "offset_cycles": int(best.offset_cycles),
            "patient_median_contact_choice_nll": float(best.patient_median),
        },
        "paired_vs_best": comparisons,
        "shared_budget_dominates_offset_budget": {
            "note": (
                "shared model selection has priority over offset calibration; "
                "offset budgets are reported for completeness"
            ),
            "table": {str(key): float(value) for key, value in shared_only.items()},
        },
    }


def analyse_generation(frame: pd.DataFrame, out: Path, *, phase: str) -> dict:
    endpoints = [key for key in ENDPOINTS if key in frame.columns]
    rollout = frame[frame.rollout_condition == "full_constructive"].copy()
    static = frame[frame.rollout_condition == "static_only"].copy()
    likelihood = frame[frame.rollout_condition == "none"].copy()

    descriptive = _descriptive(frame, endpoints)
    pd.DataFrame(descriptive).to_csv(out / f"{phase}_condition_summary.csv", index=False)

    patient_rows = []
    for (condition, rollout_condition), group in frame.groupby(
        ["condition", "rollout_condition"], dropna=False
    ):
        for endpoint in endpoints:
            for subject, value in _patient_values(group, endpoint).items():
                patient_rows.append(
                    {
                        "condition": condition,
                        "rollout_condition": rollout_condition,
                        "subject": subject,
                        "dataset": str(
                            group.loc[group.subject == subject, "dataset"].iloc[0]
                        ),
                        "endpoint": endpoint,
                        "value": value,
                    }
                )
    pd.DataFrame(patient_rows).to_csv(
        out / f"{phase}_patient_metrics.csv", index=False
    )

    native = frame[frame.rollout_condition == "native_model"].copy()
    conditions = sorted(rollout.condition.unique())
    reference = (
        "objective_teacher_forced_one_step"
        if "objective_teacher_forced_one_step" in conditions
        else (
            "current_teacher_forced_reference"
            if "current_teacher_forced_reference" in conditions
            else conditions[0]
        )
    )
    pairs = [
        (condition, reference) for condition in conditions if condition != reference
    ]
    pairs += [
        (left, right)
        for index, left in enumerate(conditions)
        for right in conditions[index + 1 :]
        if reference not in (left, right)
    ]
    # the primary rollout is the constructive generator, matching the previous
    # round; the model's own rollout is reported alongside so that a null is
    # not explained away by a training/evaluation sampling mismatch
    generators = {"full_constructive": rollout}
    if not native.empty:
        generators["native_model"] = native
    tests: dict = {}
    for generator_name, generator_frame in generators.items():
        block: dict = {}
        for treatment, baseline in pairs:
            key = f"{treatment}__vs__{baseline}"
            block[key] = {}
            for endpoint in endpoints:
                source = (
                    likelihood
                    if endpoint.startswith("likelihood_")
                    else generator_frame
                )
                block[key][endpoint] = _stratified_paired(
                    source,
                    source.condition == treatment,
                    source.condition == baseline,
                    endpoint,
                )
        tests[generator_name] = block
    # the generator contrast itself, within each trained condition
    generator_tests: dict = {}
    if not native.empty:
        for condition in conditions:
            merged = pd.concat(
                [
                    native[native.condition == condition].assign(arm="native"),
                    rollout[rollout.condition == condition].assign(arm="constructive"),
                ],
                ignore_index=True,
            )
            generator_tests[f"native_model__vs__full_constructive::{condition}"] = {
                endpoint: _stratified_paired(
                    merged, merged.arm == "native", merged.arm == "constructive", endpoint
                )
                for endpoint in endpoints
                if not endpoint.startswith("likelihood_")
            }
    # every trained condition also compared against the static-only rollout
    static_tests: dict = {}
    for condition in conditions:
        static_tests[f"{condition}__vs__static_only"] = {}
        merged = pd.concat(
            [
                rollout[rollout.condition == condition].assign(arm="model"),
                static[static.condition == condition].assign(arm="static"),
            ],
            ignore_index=True,
        )
        for endpoint in endpoints:
            if endpoint.startswith("likelihood_"):
                continue
            static_tests[f"{condition}__vs__static_only"][endpoint] = _stratified_paired(
                merged, merged.arm == "model", merged.arm == "static", endpoint
            )

    static_identity = {}
    for endpoint in endpoints:
        if endpoint.startswith("likelihood_"):
            continue
        pivot = (
            static.pivot_table(
                index=["subject", "seed"], columns="condition", values=endpoint
            )
            .dropna()
        )
        if pivot.shape[1] > 1:
            static_identity[endpoint] = float(
                np.max(np.abs(pivot.to_numpy() - pivot.to_numpy()[:, [0]]))
            )
    payload = {
        "phase": phase,
        "reference_condition": reference,
        "conditions": conditions,
        "n_patients": int(frame.subject.nunique()),
        "n_seeds": int(frame.seed.nunique()),
        "n_cells": int(len(frame.groupby(["condition", "subject", "seed"]))),
        "descriptive": descriptive,
        "primary_generator": "full_constructive",
        "secondary_generator": "native_model" if not native.empty else None,
        "paired_vs_reference": tests["full_constructive"],
        "paired_vs_reference_by_generator": tests,
        "paired_generator_contrast": generator_tests,
        "paired_vs_static_only": static_tests,
        "static_only_identical_across_conditions": static_identity,
        "statistics_order": (
            "metric within seed, seeds merged within patient, patient as unit, "
            "Epilepsiae/Yuquan stratified plus combined"
        ),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("b1c", "c", "d"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/topic5_rnn_training_sufficiency_v0_1/analysis",
    )
    args = parser.parse_args()

    root = args.root if args.root.is_absolute() else ROOT / args.root
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    frame = _load(root)
    frame.to_csv(out / f"{args.phase}_cell_metrics.csv", index=False)
    if args.phase == "b1c":
        payload = analyse_b1c(frame, out)
    else:
        payload = analyse_generation(frame, out, phase=args.phase)
    payload["ictal_target_read"] = False
    payload["outer_heldout_read"] = args.phase == "d"
    (out / f"{args.phase}_paired_tests.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    written = out / f"{args.phase}_paired_tests.json"
    print(
        json.dumps(
            {
                "phase": args.phase,
                "n_cells": int(len(frame)),
                "written": str(
                    written.relative_to(ROOT)
                    if written.is_relative_to(ROOT)
                    else written
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
