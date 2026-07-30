#!/usr/bin/env python3
"""Aggregate tolerance, cross-dataset, and inter-event feasibility audits."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE = ROOT / "results/topic5_minimal_sequence_kernel_closeout"
COMPONENTS = (
    "event_total_nll",
    "event_contact_choice_nll",
    "event_contact_contribution_nll",
    "event_stop_contribution_nll",
)


def _jsonable(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stats(values: np.ndarray, seed: int) -> dict:
    data = np.asarray(values, float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {"n": 0}
    rng = np.random.default_rng(seed)
    draws = np.median(
        data[rng.integers(0, len(data), size=(10_000, len(data)))], axis=1
    )
    nonzero = data[data != 0]
    return {
        "n": int(len(data)),
        "median_nats": float(np.median(data)),
        "median_bits": float(np.median(data) / np.log(2.0)),
        "positive": int(np.sum(data > 0)),
        "sign_p_two_sided": float(
            binomtest(int(np.sum(data > 0)), len(data), 0.5).pvalue
        ),
        "wilcoxon_p_two_sided": float(
            wilcoxon(nonzero, alternative="two-sided").pvalue
        )
        if len(nonzero)
        else 1.0,
        "bootstrap_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
    }


def _tolerance() -> dict:
    root = BASE / "rank_tolerance_v0_2"
    paths = sorted(root.glob("*/tolerance_metrics.csv"))
    if len(paths) != 34:
        raise RuntimeError(f"expected 34 tolerance subjects, found {len(paths)}")
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics["event_contact_contribution_nll"] = (
        metrics.event_total_nll - metrics.event_stop_contribution_nll
    )
    cardinality = pd.concat(
        [pd.read_csv(path.parent / "tolerance_cardinality.csv") for path in paths],
        ignore_index=True,
    )
    metrics.to_csv(root / "tolerance_metrics_all.csv", index=False)
    cardinality.to_csv(root / "tolerance_cardinality_all.csv", index=False)
    indexed = metrics.set_index(
        ["subject", "dataset", "seed", "tolerance_ms", "condition"]
    )
    rows = []
    for condition in ("history_3", "linear_state"):
        reference = indexed.xs("unordered_prefix", level="condition")
        model = indexed.xs(condition, level="condition")
        reference, model = reference.align(model, join="inner", axis=0)
        for component in COMPONENTS:
            for key, value in (reference[component] - model[component]).items():
                subject, dataset, seed, tolerance = key
                rows.append(
                    {
                        "subject": subject,
                        "dataset": dataset,
                        "seed": int(seed),
                        "tolerance_ms": float(tolerance),
                        "condition": condition,
                        "component": component,
                        "gain_nats": float(value),
                    }
                )
    gains = pd.DataFrame(rows)
    collapsed = (
        gains.groupby(
            ["subject", "dataset", "tolerance_ms", "condition", "component"],
            as_index=False,
        )
        .gain_nats.median()
    )
    collapsed["gain_bits"] = collapsed.gain_nats / np.log(2.0)
    collapsed.to_csv(root / "patient_tolerance_gains.csv", index=False)
    stat_rows = []
    for keys, frame in collapsed.groupby(["tolerance_ms", "condition", "component"]):
        tolerance, condition, component = keys
        for dataset, subset in [
            ("all", frame),
            *[(name, group) for name, group in frame.groupby("dataset")],
        ]:
            stat_rows.append(
                {
                    "tolerance_ms": tolerance,
                    "condition": condition,
                    "component": component,
                    "dataset": dataset,
                    **_stats(
                        subset.gain_nats.to_numpy(),
                        20260730 + int(tolerance * 100) + sum(map(ord, condition + component + dataset)),
                    ),
                }
            )
    statistics = pd.DataFrame(stat_rows)
    statistics.to_csv(root / "tolerance_gain_statistics.csv", index=False)
    zero = collapsed.loc[collapsed.tolerance_ms == 0].rename(
        columns={"gain_nats": "zero_gain"}
    )
    delta = collapsed.merge(
        zero[
            ["subject", "dataset", "condition", "component", "zero_gain"]
        ],
        on=["subject", "dataset", "condition", "component"],
        how="left",
    )
    delta["gain_change_from_zero"] = delta.gain_nats - delta.zero_gain
    delta.to_csv(root / "patient_tolerance_change_from_zero.csv", index=False)
    return {
        "n_subjects": int(collapsed.subject.nunique()),
        "tolerances_ms": sorted(collapsed.tolerance_ms.unique().tolist()),
        "statistics": statistics.to_dict(orient="records"),
        "total_rank_sets_by_tolerance": {
            str(tolerance): int(frame.n_eval_rank_sets.sum())
            for tolerance, frame in cardinality.groupby("tolerance_ms")
        },
        "tied_rank_sets_by_tolerance": {
            str(tolerance): int(frame.n_eval_tied_rank_sets.sum())
            for tolerance, frame in cardinality.groupby("tolerance_ms")
        },
        "zero_tolerance_source": "frozen_group_ids_not_float32_lag_reconstruction",
    }


def _cross_dataset() -> dict:
    root = BASE / "cross_dataset_v0_2"
    paths = sorted(root.glob("*_to_other/seed_*/target_patient_metrics.csv"))
    if len(paths) != 6:
        raise RuntimeError(f"expected 6 cross-dataset cells, found {len(paths)}")
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics["event_contact_contribution_nll"] = (
        metrics.event_total_nll - metrics.event_stop_contribution_nll
    )
    metrics.to_csv(root / "target_patient_metrics_all.csv", index=False)
    indexed = metrics.set_index(
        ["subject", "dataset", "source_dataset", "target_dataset", "seed", "condition"]
    )
    rows = []
    for condition in ("linear_source_frozen", "fir_h3_source_frozen"):
        reference = indexed.xs("unordered_source_frozen", level="condition")
        model = indexed.xs(condition, level="condition")
        reference, model = reference.align(model, join="inner", axis=0)
        for component in COMPONENTS:
            for key, value in (reference[component] - model[component]).items():
                subject, dataset, source_dataset, target_dataset, seed = key
                rows.append(
                    {
                        "subject": subject,
                        "dataset": dataset,
                        "source_dataset": source_dataset,
                        "target_dataset": target_dataset,
                        "seed": int(seed),
                        "condition": condition,
                        "component": component,
                        "gain_nats": float(value),
                    }
                )
    gains = pd.DataFrame(rows)
    collapsed = (
        gains.groupby(
            [
                "subject",
                "dataset",
                "source_dataset",
                "target_dataset",
                "condition",
                "component",
            ],
            as_index=False,
        )
        .gain_nats.median()
    )
    collapsed["gain_bits"] = collapsed.gain_nats / np.log(2.0)
    collapsed.to_csv(root / "patient_cross_dataset_gains.csv", index=False)
    stat_rows = []
    for keys, frame in collapsed.groupby(
        ["source_dataset", "target_dataset", "condition", "component"]
    ):
        source_dataset, target_dataset, condition, component = keys
        stat_rows.append(
            {
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "condition": condition,
                "component": component,
                **_stats(
                    frame.gain_nats.to_numpy(),
                    20260731 + sum(map(ord, "".join(keys))),
                ),
            }
        )
    statistics = pd.DataFrame(stat_rows)
    statistics.to_csv(root / "cross_dataset_gain_statistics.csv", index=False)
    confirmation = {}
    for condition in ("linear_source_frozen", "fir_h3_source_frozen"):
        rows = statistics.loc[
            (statistics.condition == condition)
            & (statistics.component == "event_contact_choice_nll")
        ]
        directional_pass = {}
        for row in rows.itertuples():
            interval = row.bootstrap_ci95
            if isinstance(interval, str):
                interval = json.loads(interval)
            direction = f"{row.source_dataset}_to_{row.target_dataset}"
            directional_pass[direction] = bool(
                row.median_nats > 0 and interval[0] > 0
            )
        confirmation[condition] = {
            "directional_pass": directional_pass,
            "bidirectional_pass": bool(
                len(directional_pass) == 2 and all(directional_pass.values())
            ),
        }
    return {
        "n_source_seed_cells": len(paths),
        "n_target_patients": int(collapsed.subject.nunique()),
        "status": "new_endpoint_cross_dataset_confirmation_not_untouched_external_validation",
        "confirmation": confirmation,
        "statistics": statistics.to_dict(orient="records"),
    }


def _gate1() -> dict:
    root = BASE / "when_gate1_inter_event_v0_2"
    paths = sorted(root.glob("*/model_metrics.csv"))
    if len(paths) != 34:
        raise RuntimeError(f"expected 34 Gate-1 subjects, found {len(paths)}")
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    summaries = pd.DataFrame(
        [json.loads((path.parent / "summary.json").read_text()) for path in paths]
    )
    metrics.to_csv(root / "model_metrics_all.csv", index=False)
    summaries.to_csv(root / "patient_null_summary.csv", index=False)
    pivot = metrics.pivot(
        index=["subject", "dataset"],
        columns="model",
        values="mean_contact_bce",
    )
    rows = []
    for model in ("last_event", "recent_unordered", "scalar_context", "time_state"):
        gain = pivot.static_prior - pivot[model]
        for (subject, dataset), value in gain.items():
            rows.append(
                {
                    "subject": subject,
                    "dataset": dataset,
                    "comparison": f"{model}_minus_static",
                    "gain_nats": float(value),
                }
            )
    best_control = pivot[
        ["static_prior", "last_event", "recent_unordered", "scalar_context"]
    ].min(axis=1)
    for (subject, dataset), value in (best_control - pivot.time_state).items():
        rows.append(
            {
                "subject": subject,
                "dataset": dataset,
                "comparison": "time_state_minus_best_nonstate_control",
                "gain_nats": float(value),
            }
        )
    gains = pd.DataFrame(rows)
    gains["gain_bits"] = gains.gain_nats / np.log(2.0)
    gains.to_csv(root / "patient_gate1_gains.csv", index=False)
    stat_rows = []
    for comparison, frame in gains.groupby("comparison"):
        for dataset, subset in [
            ("all", frame),
            *[(name, group) for name, group in frame.groupby("dataset")],
        ]:
            stat_rows.append(
                {
                    "comparison": comparison,
                    "dataset": dataset,
                    **_stats(
                        subset.gain_nats.to_numpy(),
                        20260732 + sum(map(ord, comparison + dataset)),
                    ),
                }
            )
    statistics = pd.DataFrame(stat_rows)
    statistics.to_csv(root / "gate1_gain_statistics.csv", index=False)
    primary = statistics.loc[
        (statistics.comparison == "time_state_minus_best_nonstate_control")
        & (statistics.dataset == "all")
    ].iloc[0]
    dataset_primary = statistics.loc[
        (statistics.comparison == "time_state_minus_best_nonstate_control")
        & (statistics.dataset != "all")
    ]
    null_pass = (
        (summaries.circular_null_p_greater < 0.05)
        & (summaries.block_null_p_greater < 0.05)
    )
    ci = primary.bootstrap_ci95
    if isinstance(ci, str):
        ci = json.loads(ci)
    cohort_screen_positive = bool(
        primary.median_nats > 0
        and ci[0] > 0
        and primary.wilcoxon_p_two_sided < 0.05
        and np.sum(null_pass) > len(null_pass) / 2
    )
    dataset_replication = True
    for row in dataset_primary.itertuples():
        dataset_ci = row.bootstrap_ci95
        if isinstance(dataset_ci, str):
            dataset_ci = json.loads(dataset_ci)
        dataset_replication = bool(
            dataset_replication
            and row.median_nats > 0
            and dataset_ci[0] > 0
        )
    gate_pass = bool(cohort_screen_positive and dataset_replication)
    return {
        "n_subjects": 34,
        "target": "next_event_contact_participation",
        "statistics": statistics.to_dict(orient="records"),
        "patients_passing_both_pairing_nulls": int(np.sum(null_pass)),
        "cohort_screen_positive": cohort_screen_positive,
        "replicated_in_both_datasets": dataset_replication,
        "gate1_pass": gate_pass,
        "gate1_verdict": (
            "PASS"
            if gate_pass
            else "PROVISIONAL_COHORT_SIGNAL_NOT_REPLICATED_ACROSS_DATASETS"
            if cohort_screen_positive
            else "NO_COHORT_SIGNAL"
        ),
        "interpretation": (
            "A fixed linear feasibility screen only; pass does not identify a "
            "recurrent biological state and fail does not test seizure timing."
        ),
    }


def main() -> None:
    payload = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "rank_tolerance": _tolerance(),
        "cross_dataset": _cross_dataset(),
        "when_gate1": _gate1(),
        "when_gate0": json.loads(
            (
                BASE
                / "when_gate0_early_ictal_reliability_v0_2/GATE0_SUMMARY.json"
            ).read_text()
        ),
    }
    output = BASE / "MINIMAL_SEQUENCE_AUXILIARY_SUMMARY.json"
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
