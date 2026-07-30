#!/usr/bin/env python3
"""Aggregate and gate Topic 5 constructive generation sufficiency v0.1."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_constructive_readback import (  # noqa: E402
    axis_distribution_errors,
    evaluate_axis_readback,
    evaluate_mode_readback,
    fit_train_axis_readback,
    fit_train_mode_readback,
    mode_distribution_errors,
    transition_errors,
)


CONDITIONS = [
    "full_constructive",
    "static_only",
    "static_shuffle",
    "history_h1",
    "history_h2",
    "constant_stop",
    "no_termination",
]


def _strict_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _strict_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _strict_jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        scalar = float(value)
        return scalar if np.isfinite(scalar) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _equal_halves(size: int) -> tuple[np.ndarray, np.ndarray]:
    midpoint = int(size) // 2
    count = min(midpoint, int(size) - midpoint)
    if count < 1:
        raise RuntimeError("heldout split is too small")
    return (
        np.arange(midpoint - count, midpoint, dtype=int),
        np.arange(midpoint, midpoint + count, dtype=int),
    )


def _paired_test(
    patient: pd.DataFrame,
    *,
    metric: str,
    reference: str,
    higher_is_better: bool = False,
    eligible: str | None = None,
) -> dict[str, object]:
    pivot = patient.pivot(index="subject", columns="condition", values=metric)
    if "full_constructive" not in pivot or reference not in pivot:
        raise RuntimeError(f"missing paired condition for {metric}: {reference}")
    if eligible is not None:
        eligibility = (
            patient.groupby("subject", as_index=True)[eligible].max().astype(bool)
        )
        pivot = pivot.loc[pivot.index.intersection(eligibility[eligibility].index)]
    left = pivot["full_constructive"].to_numpy(float)
    right = pivot[reference].to_numpy(float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    benefit = left - right if higher_is_better else right - left
    if benefit.size == 0:
        p_value = float("nan")
    elif np.allclose(benefit, 0):
        p_value = 1.0
    else:
        p_value = float(wilcoxon(benefit, zero_method="wilcox").pvalue)
    return {
        "metric": metric,
        "reference": reference,
        "higher_is_better": bool(higher_is_better),
        "n": int(benefit.size),
        "median_benefit": (
            float(np.median(benefit)) if benefit.size else float("nan")
        ),
        "n_full_better": int(np.sum(benefit > 0)),
        "p_two_sided": p_value,
        "benefit_by_subject": {
            str(subject): float(value)
            for subject, value in zip(pivot.index[valid], benefit)
        },
    }


def _contiguous_ranks(groups: np.ndarray) -> bool:
    for event in np.asarray(groups, dtype=int):
        valid = event[event >= 0]
        if valid.size and not np.array_equal(
            np.unique(valid), np.arange(np.max(valid) + 1)
        ):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--formal-root",
        type=Path,
        default=ROOT / "results/topic5_constructive_event_generation/formal_v0_1",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/topic5_constructive_event_generation/analysis_v0_1",
    )
    args = parser.parse_args()
    formal_root = args.formal_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    records = load_records(args.dataset_root.resolve())
    summaries = sorted(formal_root.glob("seed_*/*/run_summary.json"))
    if len(summaries) != 102:
        raise RuntimeError(f"expected 102 complete cells, found {len(summaries)}")
    local_metrics = pd.concat(
        [pd.read_csv(path.parent / "condition_metrics.csv") for path in summaries],
        ignore_index=True,
    )
    if local_metrics.shape[0] != 102 * len(CONDITIONS):
        raise RuntimeError("condition-metric row count is incomplete")

    cell_rows: list[dict[str, object]] = []
    empirical_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    engineering_errors: list[str] = []

    by_subject: dict[str, list[Path]] = {}
    for path in summaries:
        summary = json.loads(path.read_text())
        if summary.get("status") != "COMPLETE":
            engineering_errors.append(f"{path}: incomplete")
        if summary.get("ictal_target_read") is not False:
            engineering_errors.append(f"{path}: ictal target was read")
        if summary.get("ab_or_axis_used_during_rollout") is not False:
            engineering_errors.append(f"{path}: read-back leaked into rollout")
        if summary.get("source_rows_identical_across_conditions") is not True:
            engineering_errors.append(f"{path}: source mismatch")
        if summary.get("uniforms_identical_across_conditions") is not True:
            engineering_errors.append(f"{path}: uniform mismatch")
        by_subject.setdefault(str(summary["subject"]), []).append(path)

    for subject, record in sorted(records.items()):
        paths = sorted(by_subject.get(subject, []))
        if len(paths) != 3:
            raise RuntimeError(f"{subject}: expected 3 seeds, found {len(paths)}")
        with np.load(record.path, allow_pickle=False) as source:
            coords = np.asarray(source["contact_coords"], dtype=float)
        train_groups = np.asarray(record.group_ids[record.train_indices], dtype=int)
        train_count = np.asarray(record.group_count[record.train_indices], dtype=int)
        heldout_groups = np.asarray(record.group_ids[record.eval_indices], dtype=int)
        heldout_count = np.asarray(record.group_count[record.eval_indices], dtype=int)
        first, second = _equal_halves(heldout_groups.shape[0])

        mode = fit_train_mode_readback(train_groups)
        heldout_mode = evaluate_mode_readback(mode, heldout_groups)
        first_mode = evaluate_mode_readback(mode, heldout_groups[first])
        second_mode = evaluate_mode_readback(mode, heldout_groups[second])
        empirical_mode = mode_distribution_errors(first_mode, second_mode)

        axis = fit_train_axis_readback(train_groups, train_count, coords)
        heldout_axis = evaluate_axis_readback(
            axis, heldout_groups, heldout_count, coords
        )
        first_axis = evaluate_axis_readback(
            axis, heldout_groups[first], heldout_count[first], coords
        )
        second_axis = evaluate_axis_readback(
            axis, heldout_groups[second], heldout_count[second], coords
        )
        empirical_axis = axis_distribution_errors(first_axis, second_axis)
        empirical_transition = transition_errors(
            heldout_groups[first], heldout_groups[second]
        )
        axis_side_support = bool(
            int(heldout_axis["positive_count"]) >= 20
            and int(heldout_axis["negative_count"]) >= 20
        )
        axis_eligible = bool(axis.reliable and axis_side_support)
        global_eligible = bool(mode.reliable and axis_eligible)

        reference = json.loads((paths[0].parent / "empirical_reference.json").read_text())
        empirical_rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                **reference,
                **{f"empirical_{key}": value for key, value in empirical_mode.items()},
                **{
                    f"empirical_{key}": value
                    for key, value in empirical_transition.items()
                },
                **{f"empirical_{key}": value for key, value in empirical_axis.items()},
                "mode_readback_reliable": mode.reliable,
                "axis_readback_reliable": axis.reliable,
                "axis_side_support": axis_side_support,
                "global_readback_eligible": global_eligible,
            }
        )
        subject_rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_train_events": int(record.train_indices.size),
                "n_heldout_events": int(record.eval_indices.size),
                "n_contacts": int(record.group_ids.shape[1]),
                "mode_silhouette": mode.silhouette,
                "mode_minimum_cluster_fraction": mode.minimum_cluster_fraction,
                "mode_cross_half_ari": mode.cross_half_ari,
                "train_template_correlation": mode.train_template_correlation,
                "mode_readback_reliable": mode.reliable,
                "axis_explained_variance_fraction": (
                    axis.explained_variance_fraction
                ),
                "axis_n_train_vectors": axis.n_train_vectors,
                "axis_readback_reliable": axis.reliable,
                "heldout_axis_positive_count": int(
                    heldout_axis["positive_count"]
                ),
                "heldout_axis_negative_count": int(
                    heldout_axis["negative_count"]
                ),
                "axis_side_support": axis_side_support,
                "global_readback_eligible": global_eligible,
            }
        )

        for path in paths:
            summary = json.loads(path.read_text())
            seed = int(summary["seed"])
            with np.load(path.parent / "constructive_rollouts.npz") as rollouts:
                observed = np.asarray(rollouts["observed_group_ids"], dtype=int)
                source_mask = np.asarray(
                    rollouts["revealed_source_mask"], dtype=bool
                )
                if not np.array_equal(observed, heldout_groups):
                    engineering_errors.append(f"{path}: observed rows drifted")
                for condition in CONDITIONS:
                    generated = np.asarray(
                        rollouts[f"{condition}__event_group_ids"], dtype=int
                    )
                    generated_count = np.asarray(
                        rollouts[f"{condition}__event_group_count"], dtype=int
                    )
                    if not np.all(generated[source_mask] == 0):
                        engineering_errors.append(
                            f"{path}:{condition}: source not retained"
                        )
                    if not _contiguous_ranks(generated):
                        engineering_errors.append(
                            f"{path}:{condition}: non-contiguous ranks"
                        )
                    generated_mode = evaluate_mode_readback(mode, generated)
                    mode_error = mode_distribution_errors(
                        heldout_mode, generated_mode
                    )
                    generated_axis = evaluate_axis_readback(
                        axis, generated, generated_count, coords
                    )
                    axis_error = axis_distribution_errors(
                        heldout_axis, generated_axis
                    )
                    direction_denominator = int(
                        generated_axis["positive_count"]
                    ) + int(generated_axis["negative_count"])
                    minimum_side = max(
                        10, int(math.ceil(0.10 * direction_denominator))
                    )
                    generated_direction_support = bool(
                        int(generated_axis["positive_count"]) >= minimum_side
                        and int(generated_axis["negative_count"]) >= minimum_side
                    )
                    cell_rows.append(
                        {
                            "subject": subject,
                            "dataset": record.dataset,
                            "seed": seed,
                            "condition": condition,
                            **mode_error,
                            **transition_errors(heldout_groups, generated),
                            **axis_error,
                            "generated_axis_concentration": generated_axis[
                                "axis_concentration"
                            ],
                            "generated_axis_positive_fraction": generated_axis[
                                "positive_fraction"
                            ],
                            "generated_direction_support": (
                                generated_direction_support
                            ),
                            "mode_readback_reliable": mode.reliable,
                            "axis_readback_reliable": axis.reliable,
                            "axis_side_support": axis_side_support,
                            "global_readback_eligible": global_eligible,
                        }
                    )

    readback_cells = pd.DataFrame(cell_rows)
    merged = local_metrics.merge(
        readback_cells,
        on=["subject", "dataset", "seed", "condition"],
        validate="one_to_one",
    )
    empirical = pd.DataFrame(empirical_rows)
    subject_inventory = pd.DataFrame(subject_rows)
    numeric = [
        column
        for column in merged.columns
        if column not in {"subject", "dataset", "condition"}
        and pd.api.types.is_numeric_dtype(merged[column])
    ]
    patient = (
        merged.groupby(["subject", "dataset", "condition"], as_index=False)[numeric]
        .median()
    )
    for column in [
        "mode_readback_reliable",
        "axis_readback_reliable",
        "axis_side_support",
        "global_readback_eligible",
        "generated_direction_support",
    ]:
        patient[column] = patient[column].astype(bool)

    tests = {}
    for metric, reference, higher in [
        ("suffix_participation_mae", "static_only", False),
        ("suffix_rank_wasserstein", "static_only", False),
        ("suffix_precedence_mae", "static_only", False),
        ("suffix_precedence_correlation", "static_only", True),
        ("transition_mae", "static_only", False),
        ("transition_correlation", "static_only", True),
        ("suffix_rank_wasserstein", "history_h1", False),
        ("suffix_rank_wasserstein", "history_h2", False),
        ("event_length_wasserstein", "constant_stop", False),
        ("stop_hazard_mae", "constant_stop", False),
        ("event_length_wasserstein", "no_termination", False),
        ("stop_hazard_mae", "no_termination", False),
        ("template_error", "static_only", False),
        ("signed_axis_wasserstein", "static_only", False),
    ]:
        key = f"{metric}__vs__{reference}"
        eligible = (
            "global_readback_eligible"
            if metric in {"template_error", "signed_axis_wasserstein"}
            else None
        )
        tests[key] = _paired_test(
            patient,
            metric=metric,
            reference=reference,
            higher_is_better=higher,
            eligible=eligible,
        )

    full = patient[patient.condition == "full_constructive"].set_index("subject")
    empirical_by_subject = empirical.set_index("subject")
    within_rows = []
    for subject in sorted(full.index):
        row = {"subject": subject}
        for metric in [
            "suffix_participation_mae",
            "suffix_rank_wasserstein",
            "suffix_precedence_mae",
        ]:
            reference = float(empirical_by_subject.loc[subject, metric])
            value = float(full.loc[subject, metric])
            row[f"{metric}__within_empirical"] = bool(
                np.isfinite(value)
                and np.isfinite(reference)
                and value <= 1.10 * reference
            )
        row["n_local_endpoints_within_empirical"] = int(
            sum(
                row[f"{metric}__within_empirical"]
                for metric in [
                    "suffix_participation_mae",
                    "suffix_rank_wasserstein",
                    "suffix_precedence_mae",
                ]
            )
        )
        template_ref = float(
            empirical_by_subject.loc[subject, "empirical_template_error"]
        )
        axis_ref = float(
            empirical_by_subject.loc[
                subject, "empirical_signed_axis_wasserstein"
            ]
        )
        row["template_within_empirical"] = bool(
            np.isfinite(full.loc[subject, "template_error"])
            and np.isfinite(template_ref)
            and full.loc[subject, "template_error"] <= 1.10 * template_ref
        )
        row["axis_within_empirical"] = bool(
            np.isfinite(full.loc[subject, "signed_axis_wasserstein"])
            and np.isfinite(axis_ref)
            and full.loc[subject, "signed_axis_wasserstein"] <= 1.10 * axis_ref
        )
        within_rows.append(row)
    within = pd.DataFrame(within_rows)
    patient = patient.merge(within, on="subject", how="left", validate="many_to_one")

    b_rank = tests["suffix_rank_wasserstein__vs__static_only"]
    b_precedence = tests[
        "suffix_precedence_correlation__vs__static_only"
    ]
    gate_b1 = bool(
        (
            b_rank["median_benefit"] > 0
            and b_rank["p_two_sided"] < 0.05
        )
        or (
            b_precedence["median_benefit"] > 0
            and b_precedence["p_two_sided"] < 0.05
        )
    )
    n_local_within = int(
        np.sum(
            within["n_local_endpoints_within_empirical"].to_numpy(int) >= 2
        )
    )
    gate_b2 = bool(n_local_within >= math.ceil(len(within) / 2))
    termination_conditions = {}
    for reference in ["constant_stop", "no_termination"]:
        length = tests[f"event_length_wasserstein__vs__{reference}"]
        stop = tests[f"stop_hazard_mae__vs__{reference}"]
        termination_conditions[reference] = bool(
            length["median_benefit"] > 0
            and length["p_two_sided"] < 0.05
            and stop["median_benefit"] > 0
            and stop["p_two_sided"] < 0.05
        )
    gate_b3 = bool(any(termination_conditions.values()))
    gate_b = bool(gate_b1 and gate_b2 and gate_b3)

    eligible_subjects = subject_inventory.loc[
        subject_inventory.global_readback_eligible, "subject"
    ].tolist()
    eligible_within = within[within.subject.isin(eligible_subjects)]
    template_test = tests["template_error__vs__static_only"]
    axis_test = tests["signed_axis_wasserstein__vs__static_only"]
    gate_c1 = bool(
        template_test["n"] >= 5
        and axis_test["n"] >= 5
        and template_test["median_benefit"] > 0
        and template_test["p_two_sided"] < 0.05
        and axis_test["median_benefit"] > 0
        and axis_test["p_two_sided"] < 0.05
    )
    n_global_within = int(
        np.sum(
            eligible_within["template_within_empirical"]
            & eligible_within["axis_within_empirical"]
        )
    )
    required_global = (
        math.ceil(len(eligible_subjects) / 2) if eligible_subjects else 1
    )
    gate_c2 = bool(
        len(eligible_subjects) > 0 and n_global_within >= required_global
    )
    eligible_full = full.loc[full.index.intersection(eligible_subjects)]
    n_direction_support = int(
        eligible_full["generated_direction_support"].sum()
        if not eligible_full.empty
        else 0
    )
    gate_c3 = bool(
        len(eligible_subjects) > 0
        and n_direction_support >= required_global
    )
    gate_c_numeric = bool(gate_c1 and gate_c2 and gate_c3)
    gate_c_status = (
        "PASS" if gate_b and gate_c_numeric else
        "FAIL" if gate_b else
        "LOCKED_NOT_EVALUATED"
    )

    gate_a = bool(len(engineering_errors) == 0)
    if not gate_a:
        gate_b_status = "BLOCKED_BY_GATE_A"
        gate_c_status = "BLOCKED_BY_GATE_A"
    else:
        gate_b_status = "PASS" if gate_b else "FAIL"
    snn_status = (
        "OPEN_FOR_INVENTORY"
        if gate_a and gate_b and gate_c_numeric
        else "LOCKED_BY_HUMAN_SUFFICIENCY_GATE"
    )
    gates = {
        "contract": "topic5_constructive_event_generation_v0_1",
        "gate_a": {
            "status": "PASS" if gate_a else "FAIL",
            "complete_cells": len(summaries),
            "engineering_errors": engineering_errors,
        },
        "gate_b": {
            "status": gate_b_status,
            "b1_ordered_over_static": gate_b1,
            "b2_empirical_fidelity": gate_b2,
            "b3_termination_necessity": gate_b3,
            "n_patients_two_of_three_local_endpoints_within_empirical": (
                n_local_within
            ),
            "required_patients": math.ceil(len(within) / 2),
            "termination_conditions": termination_conditions,
        },
        "gate_c": {
            "status": gate_c_status,
            "diagnostic_numeric_pass": gate_c_numeric,
            "c1_full_over_static": gate_c1,
            "c2_within_empirical": gate_c2,
            "c3_bidirectional_support": gate_c3,
            "n_eligible": len(eligible_subjects),
            "eligible_subjects": eligible_subjects,
            "n_both_template_axis_within_empirical": n_global_within,
            "n_generated_bidirectional_support": n_direction_support,
        },
        "snn_fingerprint": {"status": snn_status},
        "safe_claim": (
            "algorithmic_sufficiency_supported"
            if gate_a and gate_b and gate_c_numeric
            else "local_generation_supported_global_modes_not_supported"
            if gate_a and gate_b
            else "one_step_order_information_does_not_establish_free_running_generation"
            if gate_a
            else "no_scientific_claim"
        ),
    }

    merged.to_csv(output_root / "cell_condition_metrics.csv", index=False)
    patient.to_csv(output_root / "patient_condition_metrics.csv", index=False)
    empirical.to_csv(output_root / "empirical_variability_reference.csv", index=False)
    subject_inventory.to_csv(output_root / "readback_subject_inventory.csv", index=False)
    (output_root / "paired_tests.json").write_text(
        json.dumps(_strict_jsonable(tests), indent=2, allow_nan=False) + "\n"
    )
    (output_root / "gate_verdict.json").write_text(
        json.dumps(_strict_jsonable(gates), indent=2, allow_nan=False) + "\n"
    )
    acceptance = {
        "status": "PASS" if gate_a else "FAIL",
        "expected_cells": 102,
        "complete_cells": len(summaries),
        "expected_condition_rows": 714,
        "condition_rows": int(merged.shape[0]),
        "patients": int(merged.subject.nunique()),
        "seeds": sorted(int(value) for value in merged.seed.unique()),
        "conditions": sorted(merged.condition.unique().tolist()),
        "ictal_target_read": False,
        "ab_or_axis_used_during_rollout": False,
        "readback_computed_after_frozen_rollout": True,
        "engineering_errors": engineering_errors,
    }
    (output_root / "machine_acceptance.json").write_text(
        json.dumps(_strict_jsonable(acceptance), indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(_strict_jsonable(gates), indent=2), flush=True)


if __name__ == "__main__":
    main()
