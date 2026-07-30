#!/usr/bin/env python3
"""Aggregate exact loss decomposition, lag kernels and FIR-H3 results."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402


COMPARISONS = {
    "history2_minus_history1": ("history_1", "history_2"),
    "history3_minus_history2": ("history_2", "history_3"),
    "full_minus_history3": ("history_3", "full_history"),
    "history3_minus_rank_shuffle": (
        "history_3_rank_shuffle",
        "history_3",
    ),
    "linear_minus_unordered": ("unordered_prefix", "linear_state"),
    "linear_minus_rank_shuffle": (
        "linear_state_rank_shuffle",
        "linear_state",
    ),
    "lag0_contribution": ("lag0_removed", "linear_state"),
    "lag1_contribution": ("lag1_removed", "linear_state"),
    "lag2_contribution": ("lag2_removed", "linear_state"),
    "lag3plus_contribution": ("lag3plus_removed", "linear_state"),
}
COMPONENTS = (
    "event_total_nll",
    "event_contact_choice_nll",
    "event_contact_contribution_nll",
    "event_stop_contribution_nll",
    "event_continue_nll",
    "event_terminal_stop_nll",
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


def _bootstrap_median(values: np.ndarray, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    draws = np.median(
        data[rng.integers(0, len(data), size=(5000, len(data)))], axis=1
    )
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def _paired_stats(values: np.ndarray, seed: int) -> dict:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if not len(data):
        return {
            "n": 0,
            "median_nats": np.nan,
            "median_bits": np.nan,
            "positive": 0,
            "sign_p_two_sided": np.nan,
            "wilcoxon_p_two_sided": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }
    nonzero = data[data != 0]
    wilcoxon_p = (
        float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        if len(nonzero)
        else 1.0
    )
    low, high = _bootstrap_median(data, seed)
    return {
        "n": int(len(data)),
        "median_nats": float(np.median(data)),
        "median_bits": float(np.median(data) / np.log(2.0)),
        "positive": int(np.sum(data > 0)),
        "sign_p_two_sided": float(
            binomtest(int(np.sum(data > 0)), len(data), 0.5).pvalue
        ),
        "wilcoxon_p_two_sided": wilcoxon_p,
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def _decision_contract_digest(path: Path) -> tuple[str, int]:
    """Hash only decision identity, denominator and candidate-mask columns."""

    digest = hashlib.sha256()
    count = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        keys = (
            "event_index",
            "event_source_index",
            "prediction_step",
            "terminal",
            "n_candidates",
            "target_set_size",
            "candidate_mask_hex",
        )
        for row in reader:
            digest.update(
                ("|".join(row[key] for key in keys) + "\n").encode()
            )
            count += 1
    return digest.hexdigest(), count


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 0 else np.nan


def _collect_rescore(root: Path, *, verify_decisions: bool) -> dict:
    metric_paths = sorted(root.glob("seed_*/*/component_metrics.csv"))
    if len(metric_paths) != 102:
        raise RuntimeError(f"expected 102 rescore cells, found {len(metric_paths)}")
    frames = [pd.read_csv(path) for path in metric_paths]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(root / "component_metrics_all.csv", index=False)
    expected_conditions = {
        "unordered_prefix",
        "history_1",
        "history_2",
        "history_3",
        "full_history",
        "history_3_rank_shuffle",
        "linear_state",
        "linear_state_rank_shuffle",
        "lag0_removed",
        "lag1_removed",
        "lag2_removed",
        "lag3plus_removed",
    }
    counts = metrics.groupby(["subject", "seed"]).condition.nunique()
    if len(counts) != 102 or not np.all(counts == len(expected_conditions)):
        raise RuntimeError("incomplete condition inventory")
    if set(metrics.condition) != expected_conditions:
        raise RuntimeError("unexpected condition inventory")
    if float(metrics.maximum_event_nll_reconstruction_error.max()) > 2e-5:
        raise RuntimeError("likelihood reconstruction exceeds tolerance")

    contract_rows = []
    if verify_decisions:
        for cell in sorted(path.parent for path in metric_paths):
            digests = {}
            for condition in sorted(expected_conditions):
                path = cell / f"{condition}_decisions.csv.gz"
                digest, count = _decision_contract_digest(path)
                digests[condition] = digest
                contract_rows.append(
                    {
                        "subject": cell.name,
                        "seed": int(cell.parent.name.split("_")[-1]),
                        "condition": condition,
                        "decision_contract_sha256": digest,
                        "n_decisions": count,
                    }
                )
            if len(set(digests.values())) != 1:
                raise RuntimeError(f"{cell}: decision contract mismatch")
    contract = pd.DataFrame(contract_rows)
    if len(contract):
        contract.to_csv(root / "decision_contract_audit.csv", index=False)

    indexed = metrics.set_index(["subject", "dataset", "seed", "condition"])
    paired_rows = []
    for name, (reference, model) in COMPARISONS.items():
        left = indexed.xs(reference, level="condition")
        right = indexed.xs(model, level="condition")
        left, right = left.align(right, join="inner", axis=0)
        if len(left) != 102:
            raise RuntimeError(f"{name}: incomplete paired cells")
        for component in COMPONENTS:
            difference = left[component] - right[component]
            for key, value in difference.items():
                subject, dataset, seed = key
                paired_rows.append(
                    {
                        "subject": subject,
                        "dataset": dataset,
                        "seed": int(seed),
                        "comparison": name,
                        "component": component,
                        "gain_nats": float(value),
                        "gain_bits": float(value / np.log(2.0)),
                    }
                )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(root / "patient_seed_component_gains.csv", index=False)
    collapsed = (
        paired.groupby(
            ["subject", "dataset", "comparison", "component"], as_index=False
        )
        .gain_nats.median()
    )
    collapsed["gain_bits"] = collapsed.gain_nats / np.log(2.0)
    collapsed.to_csv(root / "patient_component_gains.csv", index=False)

    stats_rows = []
    for (comparison, component), frame in collapsed.groupby(
        ["comparison", "component"]
    ):
        for dataset, subset in [
            ("all", frame),
            *[(name, group) for name, group in frame.groupby("dataset")],
        ]:
            stats_rows.append(
                {
                    "comparison": comparison,
                    "component": component,
                    "dataset": dataset,
                    **_paired_stats(
                        subset.gain_nats.to_numpy(),
                        20260730
                        + sum(map(ord, comparison + component + dataset)),
                    ),
                }
            )
    statistics = pd.DataFrame(stats_rows)
    statistics.to_csv(root / "component_gain_statistics.csv", index=False)

    kernel_paths = sorted(root.glob("seed_*/*/linear_state_lag_kernels.npz"))
    if len(kernel_paths) != 102:
        raise RuntimeError(f"expected 102 kernel files, found {len(kernel_paths)}")
    stability_rows = []
    for subject_dir in sorted({path.parent.name for path in kernel_paths}):
        paths = sorted(root.glob(f"seed_*/{subject_dir}/linear_state_lag_kernels.npz"))
        if len(paths) != 3:
            raise RuntimeError(f"{subject_dir}: kernel seed count is not 3")
        payloads = [np.load(path, allow_pickle=False) for path in paths]
        for left_index, right_index in combinations(range(3), 2):
            for lag in range(6):
                stability_rows.append(
                    {
                        "subject": subject_dir,
                        "seed_left": int(
                            paths[left_index].parents[1].name.split("_")[-1]
                        ),
                        "seed_right": int(
                            paths[right_index].parents[1].name.split("_")[-1]
                        ),
                        "lag": lag,
                        "contact_kernel_cosine": _cosine(
                            payloads[left_index]["contact_kernels"][lag],
                            payloads[right_index]["contact_kernels"][lag],
                        ),
                        "stop_kernel_cosine": _cosine(
                            payloads[left_index]["stop_kernels"][lag],
                            payloads[right_index]["stop_kernels"][lag],
                        ),
                    }
                )
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(root / "kernel_seed_stability.csv", index=False)
    kernel_summary = pd.concat(
        [
            pd.read_csv(path)
            for path in sorted(root.glob("seed_*/*/kernel_summary.csv"))
        ],
        ignore_index=True,
    )
    kernel_summary.to_csv(root / "kernel_summary_all.csv", index=False)

    records = load_records(
        ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
    )
    inventory_rows = []
    for subject, record in records.items():
        tie_sets = 0
        rank_sets = 0
        for event, count in zip(record.group_ids, record.group_count):
            for step in range(int(count)):
                rank_sets += 1
                tie_sets += int(np.sum(event == step) > 1)
        inventory_rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_contacts": int(record.group_ids.shape[1]),
                "n_events": int(len(record.group_ids)),
                "n_train_events": int(len(record.train_indices)),
                "n_heldout_events": int(len(record.eval_indices)),
                "mean_recruitment_groups": float(np.mean(record.group_count)),
                "n_rank_sets": int(rank_sets),
                "n_tied_rank_sets": int(tie_sets),
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(root / "data_cardinality_inventory.csv", index=False)

    primary = collapsed[
        (collapsed.comparison == "linear_minus_unordered")
        & (collapsed.component == "event_contact_choice_nll")
    ].merge(inventory, on=["subject", "dataset"], how="inner")
    heterogeneity_rows = []
    for covariate in (
        "n_events",
        "n_contacts",
        "mean_recruitment_groups",
    ):
        for dataset, subset in [
            ("all", primary),
            *[(name, group) for name, group in primary.groupby("dataset")],
        ]:
            result = spearmanr(
                subset.gain_nats, subset[covariate], nan_policy="omit"
            )
            heterogeneity_rows.append(
                {
                    "dataset": dataset,
                    "covariate": covariate,
                    "n": int(len(subset)),
                    "spearman_rho": float(result.statistic),
                    "p_two_sided": float(result.pvalue),
                }
            )
    heterogeneity = pd.DataFrame(heterogeneity_rows)
    heterogeneity.to_csv(root / "effect_heterogeneity.csv", index=False)

    return {
        "n_cells": 102,
        "n_subjects": 34,
        "decision_contract_verified": bool(verify_decisions),
        "maximum_nll_reconstruction_error": float(
            metrics.maximum_event_nll_reconstruction_error.max()
        ),
        "rank_sets": int(inventory.n_rank_sets.sum()),
        "tied_rank_sets": int(inventory.n_tied_rank_sets.sum()),
        "tied_rank_set_fraction": float(
            inventory.n_tied_rank_sets.sum() / inventory.n_rank_sets.sum()
        ),
        "statistics": statistics.to_dict(orient="records"),
        "kernel_seed_stability_median": {
            str(lag): {
                "contact": float(
                    stability.loc[
                        stability.lag == lag, "contact_kernel_cosine"
                    ].median()
                ),
                "stop": float(
                    stability.loc[
                        stability.lag == lag, "stop_kernel_cosine"
                    ].median()
                ),
            }
            for lag in range(6)
        },
        "hankel": {
            "contact_rank90_median": float(
                kernel_summary.contact_hankel_rank90.median()
            ),
            "contact_rank95_median": float(
                kernel_summary.contact_hankel_rank95.median()
            ),
            "contact_effective_order_median": float(
                kernel_summary.contact_hankel_effective_order.median()
            ),
            "combined_rank90_median": float(
                kernel_summary.combined_hankel_rank90.median()
            ),
            "combined_rank95_median": float(
                kernel_summary.combined_hankel_rank95.median()
            ),
            "combined_effective_order_median": float(
                kernel_summary.combined_hankel_effective_order.median()
            ),
        },
    }


def _collect_fir(root: Path, rescore_root: Path) -> dict:
    paths = sorted(root.glob("seed_*/*/component_metrics.csv"))
    if len(paths) != 102:
        raise RuntimeError(f"expected 102 FIR cells, found {len(paths)}")
    metrics = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    metrics.to_csv(root / "component_metrics_all.csv", index=False)
    indexed = metrics.set_index(["subject", "dataset", "seed", "condition"])
    baseline = indexed.xs("unordered_retrained", level="condition")
    fir = indexed.xs("fir_h3_residual", level="condition")
    baseline, fir = baseline.align(fir, join="inner", axis=0)
    rescore_path = rescore_root / "component_metrics_all.csv"
    if rescore_path.exists():
        rescore_metrics = pd.read_csv(rescore_path)
    else:
        rescore_metrics = pd.concat(
            [
                pd.read_csv(path)
                for path in sorted(
                    rescore_root.glob("seed_*/*/component_metrics.csv")
                )
            ],
            ignore_index=True,
        )
    rescore_indexed = rescore_metrics.set_index(
        ["subject", "dataset", "seed", "condition"]
    )
    selected_linear = rescore_indexed.xs("linear_state", level="condition")
    frozen_unordered = rescore_indexed.xs(
        "unordered_prefix", level="condition"
    )
    for comparison, left, right in (
        ("fir_minus_retrained_unordered", baseline, fir),
        ("fir_minus_selected_linear", selected_linear, fir),
        ("retrained_minus_frozen_unordered", frozen_unordered, baseline),
    ):
        left_aligned, right_aligned = left.align(right, join="inner", axis=0)
        if len(left_aligned) != 102:
            raise RuntimeError(f"{comparison}: incomplete paired cells")
        for count_key in ("n_events", "n_decisions", "n_nonterminal_decisions"):
            if not np.array_equal(
                left_aligned[count_key].to_numpy(),
                right_aligned[count_key].to_numpy(),
            ):
                raise RuntimeError(f"{comparison}: decision count drift")
    rows = []
    for comparison, left, right in (
        ("fir_minus_retrained_unordered", baseline, fir),
        ("fir_minus_selected_linear", selected_linear, fir),
        ("retrained_minus_frozen_unordered", frozen_unordered, baseline),
    ):
        left, right = left.align(right, join="inner", axis=0)
        for component in COMPONENTS:
            difference = left[component] - right[component]
            for (subject, dataset, seed), value in difference.items():
                rows.append(
                    {
                        "subject": subject,
                        "dataset": dataset,
                        "seed": int(seed),
                        "comparison": comparison,
                        "component": component,
                        "gain_nats": float(value),
                        "gain_bits": float(value / np.log(2.0)),
                    }
                )
    paired = pd.DataFrame(rows)
    paired.to_csv(root / "patient_seed_fir_gains.csv", index=False)
    collapsed = (
        paired.groupby(
            ["subject", "dataset", "comparison", "component"], as_index=False
        )
        .gain_nats.median()
    )
    collapsed["gain_bits"] = collapsed.gain_nats / np.log(2.0)
    collapsed.to_csv(root / "patient_fir_gains.csv", index=False)
    statistics_rows = []
    for (comparison, component), frame in collapsed.groupby(
        ["comparison", "component"]
    ):
        for dataset, subset in [
            ("all", frame),
            *[(name, group) for name, group in frame.groupby("dataset")],
        ]:
            statistics_rows.append(
                {
                    "comparison": comparison,
                    "component": component,
                    "dataset": dataset,
                    **_paired_stats(
                        subset.gain_nats.to_numpy(),
                        20260731
                        + sum(map(ord, comparison + component + dataset)),
                    ),
                }
            )
    statistics = pd.DataFrame(statistics_rows)
    statistics.to_csv(root / "fir_gain_statistics.csv", index=False)
    summaries = [
        json.loads(path.read_text())
        for path in sorted(root.glob("seed_*/*/summary.json"))
    ]
    if not all(
        summary.get("baseline_frozen_during_fir") is True
        and summary.get("target_values_read") is False
        for summary in summaries
    ):
        raise RuntimeError("FIR freeze or target seal audit failed")
    return {
        "n_cells": len(paths),
        "n_subjects": int(collapsed.subject.nunique()),
        "baseline_frozen_all_cells": True,
        "decision_counts_match_all_comparisons": True,
        "statistics": statistics.to_dict(orient="records"),
        "runtime_seconds_median": float(
            np.median(
                [summary["resources"]["runtime_seconds"] for summary in summaries]
            )
        ),
        "gpu_peak_reserved_bytes_max": int(
            max(
                summary["resources"]["gpu_peak_reserved_bytes"]
                for summary in summaries
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rescore-root",
        type=Path,
        default=ROOT
        / "results/topic5_minimal_sequence_kernel_closeout/formal_v0_2",
    )
    parser.add_argument(
        "--fir-root",
        type=Path,
        default=ROOT
        / "results/topic5_minimal_sequence_kernel_closeout/fir_h3_formal_v0_2",
    )
    parser.add_argument("--skip-fir", action="store_true")
    parser.add_argument("--skip-decision-verification", action="store_true")
    args = parser.parse_args()
    rescore_root = (
        args.rescore_root
        if args.rescore_root.is_absolute()
        else ROOT / args.rescore_root
    )
    fir_root = (
        args.fir_root if args.fir_root.is_absolute() else ROOT / args.fir_root
    )
    rescore = _collect_rescore(
        rescore_root,
        verify_decisions=not args.skip_decision_verification,
    )
    payload = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "target_values_read": False,
        "rescore": rescore,
    }
    if not args.skip_fir:
        payload["fir_h3"] = _collect_fir(fir_root, rescore_root)
    output = (
        ROOT
        / "results/topic5_minimal_sequence_kernel_closeout/"
        "MINIMAL_SEQUENCE_KERNEL_SUMMARY.json"
    )
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(_jsonable(payload), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
