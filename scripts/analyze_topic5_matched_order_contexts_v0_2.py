#!/usr/bin/env python3
"""Observed-data matched-context test for recent within-event order."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402


def _set_mask(groups: np.ndarray, step: int) -> int:
    mask = 0
    for contact in np.flatnonzero(groups == step):
        mask |= 1 << int(contact)
    return mask


def _prefix_mask(groups: np.ndarray, step: int) -> int:
    mask = 0
    for contact in np.flatnonzero((groups >= 0) & (groups < step)):
        mask |= 1 << int(contact)
    return mask


def _training_tables(record, horizon: int, min_support: int):
    n_contacts = record.group_ids.shape[1]
    context_counts = defaultdict(lambda: np.zeros(n_contacts, dtype=np.float64))
    history_counts = defaultdict(lambda: np.zeros(n_contacts, dtype=np.float64))
    history_support = defaultdict(int)
    histories_by_context = defaultdict(set)
    for event_index in record.train_indices:
        groups = record.group_ids[event_index]
        count = int(record.group_count[event_index])
        for step in range(int(horizon), count):
            prefix = _prefix_mask(groups, step)
            context = (prefix, step)
            history = tuple(
                _set_mask(groups, rank)
                for rank in range(step - int(horizon), step)
            )
            target = np.flatnonzero(groups == step)
            if not len(target):
                continue
            weight = 1.0 / len(target)
            context_counts[context][target] += weight
            history_counts[(context, history)][target] += weight
            history_support[(context, history)] += 1
            histories_by_context[context].add(history)
    eligible_contexts = set()
    eligible_histories = set()
    for context, histories in histories_by_context.items():
        supported = [
            history for history in histories
            if history_support[(context, history)] >= int(min_support)
        ]
        if len(supported) >= 2:
            eligible_contexts.add(context)
            eligible_histories.update((context, history) for history in supported)
    return (
        context_counts,
        history_counts,
        history_support,
        eligible_contexts,
        eligible_histories,
    )


def _evaluate(record, horizon: int, min_support: int, alpha: float) -> dict:
    (
        context_counts,
        history_counts,
        history_support,
        eligible_contexts,
        eligible_histories,
    ) = _training_tables(record, horizon, min_support)
    n_contacts = record.group_ids.shape[1]
    event_unordered = defaultdict(list)
    event_ordered = defaultdict(list)
    evaluated_contexts = set()
    evaluated_histories = set()
    for event_index in record.eval_indices:
        groups = record.group_ids[event_index]
        count = int(record.group_count[event_index])
        for step in range(int(horizon), count):
            prefix = _prefix_mask(groups, step)
            context = (prefix, step)
            history = tuple(
                _set_mask(groups, rank)
                for rank in range(step - int(horizon), step)
            )
            pair = (context, history)
            if context not in eligible_contexts or pair not in eligible_histories:
                continue
            target = np.flatnonzero(groups == step)
            if not len(target):
                continue
            candidate = np.asarray(
                [not (prefix & (1 << contact)) for contact in range(n_contacts)],
                dtype=bool,
            )
            pooled = context_counts[context].copy()
            ordered = history_counts[pair].copy()
            pooled_probability = np.zeros(n_contacts, dtype=np.float64)
            ordered_probability = np.zeros(n_contacts, dtype=np.float64)
            pooled_probability[candidate] = (
                pooled[candidate] + float(alpha)
            )
            ordered_probability[candidate] = (
                ordered[candidate] + float(alpha)
            )
            pooled_probability /= pooled_probability.sum()
            ordered_probability /= ordered_probability.sum()
            pooled_mass = float(np.sum(pooled_probability[target]))
            ordered_mass = float(np.sum(ordered_probability[target]))
            event_unordered[int(event_index)].append(
                -np.log(max(pooled_mass, 1.0e-12))
            )
            event_ordered[int(event_index)].append(
                -np.log(max(ordered_mass, 1.0e-12))
            )
            evaluated_contexts.add(context)
            evaluated_histories.add(pair)
    common_events = sorted(set(event_unordered) & set(event_ordered))
    unordered_event = np.asarray(
        [np.mean(event_unordered[event]) for event in common_events],
        dtype=np.float64,
    )
    ordered_event = np.asarray(
        [np.mean(event_ordered[event]) for event in common_events],
        dtype=np.float64,
    )
    decisions = int(sum(len(event_unordered[event]) for event in common_events))
    return {
        "subject": record.subject,
        "dataset": record.dataset,
        "history_horizon": int(horizon),
        "min_train_per_order": int(min_support),
        "dirichlet_alpha": float(alpha),
        "n_train_eligible_contexts": int(len(eligible_contexts)),
        "n_train_eligible_order_histories": int(len(eligible_histories)),
        "n_heldout_evaluated_contexts": int(len(evaluated_contexts)),
        "n_heldout_evaluated_order_histories": int(len(evaluated_histories)),
        "n_heldout_evaluated_events": int(len(common_events)),
        "n_heldout_evaluated_decisions": decisions,
        "unordered_contact_choice_nll": (
            float(np.mean(unordered_event)) if len(common_events) else np.nan
        ),
        "ordered_contact_choice_nll": (
            float(np.mean(ordered_event)) if len(common_events) else np.nan
        ),
        "ordered_gain_nats": (
            float(np.mean(unordered_event - ordered_event))
            if len(common_events)
            else np.nan
        ),
        "ordered_gain_bits": (
            float(np.mean(unordered_event - ordered_event) / np.log(2.0))
            if len(common_events)
            else np.nan
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT
        / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "results/topic5_minimal_sequence_kernel_closeout/matched_contexts_v0_2",
    )
    parser.add_argument("--min-train-per-order", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    output = (
        args.output_dir
        if args.output_dir.is_absolute()
        else ROOT / args.output_dir
    )
    output.mkdir(parents=True, exist_ok=False)
    records = load_records(dataset)
    rows = []
    for subject in sorted(records):
        for horizon in (2, 3):
            row = _evaluate(
                records[subject],
                horizon,
                args.min_train_per_order,
                args.alpha,
            )
            rows.append(row)
            print(
                json.dumps(
                    {
                        "subject": subject,
                        "horizon": horizon,
                        "decisions": row["n_heldout_evaluated_decisions"],
                    }
                ),
                flush=True,
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "matched_context_patient_results.csv", index=False)
    support = frame.n_heldout_evaluated_decisions >= 50
    summary_rows = []
    for horizon, subset in frame.groupby("history_horizon"):
        eligible = subset[
            subset.n_heldout_evaluated_decisions >= 50
        ].dropna(subset=["ordered_gain_nats"])
        values = eligible.ordered_gain_nats.to_numpy()
        if len(values):
            nonzero = values[values != 0]
            p_wilcoxon = (
                float(wilcoxon(nonzero, alternative="two-sided").pvalue)
                if len(nonzero)
                else 1.0
            )
            p_sign = float(
                binomtest(int(np.sum(values > 0)), len(values), 0.5).pvalue
            )
        else:
            p_wilcoxon = p_sign = np.nan
        summary_rows.append(
            {
                "history_horizon": int(horizon),
                "n_patients_total": int(len(subset)),
                "n_patients_ge_50_decisions": int(len(eligible)),
                "support_gate_pass": bool(len(eligible) >= 20),
                "median_ordered_gain_nats": (
                    float(np.median(values)) if len(values) else np.nan
                ),
                "median_ordered_gain_bits": (
                    float(np.median(values) / np.log(2.0))
                    if len(values)
                    else np.nan
                ),
                "positive_patients": int(np.sum(values > 0)),
                "wilcoxon_p_two_sided": p_wilcoxon,
                "sign_p_two_sided": p_sign,
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(output / "matched_context_summary.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "target_values_read": False,
        "support_rule": "at_least_20_patients_with_50_heldout_decisions",
        "support": summary_rows,
    }
    (output / "MATCHED_CONTEXT_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
