#!/usr/bin/env python3
"""Re-score generated events against the previous round's own acceptance bar.

The previous constructive-generation round asked: for how many patients does a
freely generated event distribution sit within ``+10%`` of the error you get by
comparing one half of the *observed* held-out events against the other half?
It required at least 17/34 patients to clear at least two of the three
whole-event endpoints, and reported 9/34.

This script recomputes that empirical split-half reference from the frozen
dataset (and cross-checks it against the archived table), then applies the same
rule to every Phase D condition and to both generators.  Nothing here selects a
model; the reference is a fixed property of the observed data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import load_records  # noqa: E402
from src.topic5_constructive_event_generator import remove_revealed_source  # noqa: E402
from src.topic5_rank_distribution import distribution_errors  # noqa: E402

ARCHIVED_REFERENCE = (
    ROOT
    / "results/topic5_constructive_event_generation/analysis_v0_1"
    / "empirical_variability_reference.csv"
)
#: the three whole-event endpoints of the preregistered rule, all errors
ENDPOINTS = (
    "suffix_participation_mae",
    "suffix_rank_wasserstein",
    "suffix_precedence_mae",
)
TOLERANCE = 1.10
REQUIRED_ENDPOINTS = 2
PREREGISTERED_BAR = 17


def _equal_halves(n_events: int):
    indices = np.arange(int(n_events))
    midpoint = len(indices) // 2
    half = min(midpoint, len(indices) - midpoint)
    if half < 1:
        raise RuntimeError("held-out split has too few events")
    return indices[midpoint - half : midpoint], indices[midpoint : midpoint + half]


def empirical_reference(dataset_dir: Path) -> pd.DataFrame:
    """Observed half versus observed half, on the frozen outer heldout events."""
    rows = []
    for subject, record in sorted(load_records(dataset_dir).items()):
        indices = np.asarray(record.eval_indices, dtype=int)
        groups = np.asarray(record.group_ids[indices], dtype=np.int16)
        counts = np.asarray(record.group_count[indices], dtype=np.int16)
        source = groups == 0
        suffix = remove_revealed_source(groups, source)
        suffix_count = np.maximum(counts - 1, 0)
        first, second = _equal_halves(len(indices))
        errors = distribution_errors(
            suffix[first], suffix_count[first], suffix[second], suffix_count[second],
            bins=10,
        )
        rows.append(
            {
                "subject": subject,
                "dataset": record.dataset,
                "n_events": int(len(indices)),
                "n_events_per_half": int(len(first)),
                **{f"suffix_{key}": value for key, value in errors.items()},
            }
        )
    return pd.DataFrame(rows)


def _cross_check(reference: pd.DataFrame) -> dict:
    if not ARCHIVED_REFERENCE.is_file():
        return {"archived_reference_present": False}
    archived = pd.read_csv(ARCHIVED_REFERENCE)
    merged = reference.merge(archived, on="subject", suffixes=("", "_archived"))
    report = {"archived_reference_present": True, "n_subjects": int(len(merged))}
    for endpoint in ENDPOINTS:
        column = f"{endpoint}_archived"
        if column not in merged:
            continue
        report[endpoint] = float(
            np.nanmax(np.abs(merged[endpoint] - merged[column]))
        )
    report["max_absolute_difference"] = float(
        max(value for key, value in report.items() if key in ENDPOINTS)
    )
    report["matches_archived"] = report["max_absolute_difference"] < 1e-9
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "results/topic5_rnn_training_sufficiency_v0_1/analysis/d_cell_metrics.csv",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results/topic5_rnn_training_sufficiency_v0_1/analysis",
    )
    args = parser.parse_args()

    out = args.out if args.out.is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    reference = empirical_reference(
        args.dataset_root if args.dataset_root.is_absolute() else ROOT / args.dataset_root
    )
    reference.to_csv(out / "empirical_variability_reference.csv", index=False)
    cross_check = _cross_check(reference)

    metrics = pd.read_csv(
        args.metrics if args.metrics.is_absolute() else ROOT / args.metrics
    )
    metrics = metrics[metrics.rollout_condition != "none"].copy()
    reference_map = reference.set_index("subject")

    rows = []
    for (condition, generator, subject), group in metrics.groupby(
        ["condition", "rollout_condition", "subject"]
    ):
        if subject not in reference_map.index:
            continue
        record = {"condition": condition, "generator": generator, "subject": subject,
                  "dataset": str(group.dataset.iloc[0])}
        cleared = 0
        for endpoint in ENDPOINTS:
            generated = float(group[endpoint].mean())  # seeds averaged first
            empirical = float(reference_map.loc[subject, endpoint])
            within = bool(generated <= empirical * TOLERANCE)
            record[f"{endpoint}__generated"] = generated
            record[f"{endpoint}__empirical"] = empirical
            record[f"{endpoint}__ratio"] = (
                generated / empirical if empirical else float("nan")
            )
            record[f"{endpoint}__within_tolerance"] = within
            cleared += int(within)
        record["n_endpoints_within_tolerance"] = cleared
        record["patient_clears_rule"] = bool(cleared >= REQUIRED_ENDPOINTS)
        rows.append(record)
    per_patient = pd.DataFrame(rows)
    per_patient.to_csv(out / "d_empirical_variability_rescore.csv", index=False)

    summary = []
    for (condition, generator), group in per_patient.groupby(["condition", "generator"]):
        entry = {
            "condition": condition,
            "generator": generator,
            "n_patients": int(len(group)),
            "n_patients_clearing_rule": int(group.patient_clears_rule.sum()),
            "preregistered_bar": PREREGISTERED_BAR,
            "clears_preregistered_bar": bool(
                group.patient_clears_rule.sum() >= PREREGISTERED_BAR
            ),
        }
        for endpoint in ENDPOINTS:
            entry[f"{endpoint}__n_within"] = int(
                group[f"{endpoint}__within_tolerance"].sum()
            )
            entry[f"{endpoint}__median_ratio"] = float(
                group[f"{endpoint}__ratio"].median()
            )
        for stratum in ("epilepsiae", "yuquan"):
            subset = group[group.dataset == stratum]
            entry[f"n_clearing_{stratum}"] = int(subset.patient_clears_rule.sum())
            entry[f"n_patients_{stratum}"] = int(len(subset))
        summary.append(entry)
    summary_frame = pd.DataFrame(summary).sort_values(["generator", "condition"])
    summary_frame.to_csv(out / "d_empirical_variability_summary.csv", index=False)

    payload = {
        "contract": "topic5_rnn_training_sufficiency_v0_1_empirical_variability_rescore",
        "question": (
            "for how many patients does a freely generated event distribution sit "
            "within +10% of the error between two halves of the observed held-out "
            "events, on at least two of three whole-event endpoints?"
        ),
        "endpoints": list(ENDPOINTS),
        "tolerance": TOLERANCE,
        "required_endpoints_per_patient": REQUIRED_ENDPOINTS,
        "preregistered_bar": PREREGISTERED_BAR,
        "previous_round_result": {
            "generator": "full_constructive",
            "n_patients_clearing_rule": 9,
            "source": (
                "docs/archive/topic5/constructive_event_generation_sufficiency_v0_1"
                "_report_2026-07-30.md section 4.2"
            ),
        },
        "reference_cross_check": cross_check,
        "summary": summary_frame.to_dict("records"),
    }
    (out / "d_empirical_variability_rescore.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps({
        "reference_matches_archived": cross_check.get("matches_archived"),
        "summary": [
            {
                "condition": row["condition"],
                "generator": row["generator"],
                "clearing": f"{row['n_patients_clearing_rule']}/{row['n_patients']}",
                "bar": row["preregistered_bar"],
                "clears": row["clears_preregistered_bar"],
            }
            for row in summary_frame.to_dict("records")
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
