#!/usr/bin/env python3
"""Aggregate extended held-out ECoG metrics without treating events as replicates."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1")
DISTANCE_KEYS = (
    "up_down_left_right", "diagonal", "two_grid_steps", "farther_than_two_grid_steps",
)


def holm_adjust(p_values: list[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    unit_rows: list[dict[str, Any]] = []
    for subject in ("958", "1084"):
        for path in sorted((ROOT / "training" / subject).glob("*/heldout_extended_metrics.json")):
            data = json.loads(path.read_text())
            metrics = data["metrics"]
            graph_id = str(data["graph_id"])
            graph_index = int(graph_id.rsplit("_", 1)[-1]) if graph_id.rsplit("_", 1)[-1].isdigit() else -1
            row: dict[str, Any] = {
                "subject": subject,
                "family": data["family"],
                "graph_id": graph_id,
                "graph_index": graph_index,
                "seed_index": int(data["seed_index"]),
                "top1_any_next_contact": metrics["top1_any_next_contact"],
                "top_observed_cardinality_recall": metrics["top_observed_cardinality_recall"],
                "stop_brier": metrics["stop_brier"],
                "stop_bce": metrics["stop_bce"],
                "contact_nll_per_true_contact": metrics["contact_nll_per_true_contact"],
                "path": str(path),
            }
            for key in DISTANCE_KEYS:
                row[f"distance_{key}_n"] = metrics["distance_strata"][key]["n"]
                row[f"distance_{key}_nll"] = metrics["distance_strata"][key]["mean_target_contact_nll"]
            unit_rows.append(row)
    if len(unit_rows) != 384:
        raise RuntimeError(f"need 384 extended metric units, found {len(unit_rows)}")
    write_csv(ROOT / "summary/HELDOUT_EXTENDED_UNIT_RESULTS.csv", unit_rows)

    graph_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []
    for subject in ("958", "1084"):
        rows = [row for row in unit_rows if row["subject"] == subject]
        true = sorted([row for row in rows if row["family"] == "TRUE_GRID"], key=lambda row: row["seed_index"])
        suffix = sorted([row for row in rows if row["family"] == "SUFFIX_SHUFFLED"], key=lambda row: row["seed_index"])
        if len(true) != 3 or len(suffix) != 3:
            raise RuntimeError(f"{subject}: missing true/suffix metrics")
        metric_names = [
            "top1_any_next_contact", "top_observed_cardinality_recall", "stop_brier", "stop_bce",
            "contact_nll_per_true_contact",
            *[f"distance_{key}_nll" for key in DISTANCE_KEYS],
        ]
        result: dict[str, Any] = {"subject": subject}
        for metric in metric_names:
            true_values = np.asarray([float(row[metric]) for row in true])
            suffix_values = np.asarray([float(row[metric]) for row in suffix])
            result[f"true_{metric}_median_seed"] = float(np.median(true_values))
            result[f"true_minus_suffix_{metric}_median_seed"] = float(np.median(true_values - suffix_values))
            for family in ("WRONG_GRID", "DEGREE_RANDOM"):
                effects = []
                for graph_index in range(31):
                    matched = sorted([
                        row for row in rows
                        if row["family"] == family and row["graph_index"] == graph_index
                    ], key=lambda row: row["seed_index"])
                    if len(matched) != 3:
                        raise RuntimeError(f"{subject} {family} {graph_index}: missing extended metric seed")
                    control_values = np.asarray([float(row[metric]) for row in matched])
                    effect = float(np.median(true_values) - np.median(control_values))
                    effects.append(effect)
                    graph_rows.append({
                        "subject": subject, "family": family, "graph_index": graph_index,
                        "metric": metric, "true_minus_control": effect,
                    })
                result[f"true_minus_{family.lower()}_{metric}_median_graph"] = float(np.median(effects))
                control_statistics = np.asarray([
                    float(np.median([
                        float(row[metric]) for row in rows
                        if row["family"] == family and row["graph_index"] == graph_index
                    ]))
                    for graph_index in range(31)
                ])
                true_statistic = float(np.median(true_values))
                higher_is_better = metric in (
                    "top1_any_next_contact", "top_observed_cardinality_recall",
                )
                if higher_is_better:
                    better_count = int(np.sum(true_statistic > control_statistics))
                    exact_p = float((1 + np.sum(control_statistics >= true_statistic)) / 32)
                else:
                    better_count = int(np.sum(true_statistic < control_statistics))
                    exact_p = float((1 + np.sum(control_statistics <= true_statistic)) / 32)
                prefix = f"true_vs_{family.lower()}_{metric}"
                result[f"{prefix}_better_count"] = better_count
                result[f"{prefix}_exact_p_one_sided"] = exact_p
        for family in ("WRONG_GRID", "DEGREE_RANDOM"):
            prefixes = [
                f"true_vs_{family.lower()}_distance_{key}_nll" for key in DISTANCE_KEYS
            ]
            adjusted = holm_adjust([
                float(result[f"{prefix}_exact_p_one_sided"]) for prefix in prefixes
            ])
            for prefix, value in zip(prefixes, adjusted):
                result[f"{prefix}_holm_across_four_distance_bins"] = value
        patient_rows.append(result)
    write_csv(ROOT / "summary/HELDOUT_EXTENDED_GRAPH_EFFECTS.csv", graph_rows)
    write_csv(ROOT / "summary/HELDOUT_EXTENDED_PATIENT_RESULTS.csv", patient_rows)
    payload = {
        "schema": "topic5_ecog_heldout_extended_summary_v0.1",
        "complete": True,
        "n_units": len(unit_rows),
        "patient_results": patient_rows,
        "distance_definition": (
            "Each true next contact is binned by Euclidean grid distance to the closest contact "
            "in the current rank set; losses are averaged over true contacts within each bin."
        ),
    }
    (ROOT / "summary/HELDOUT_EXTENDED_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"complete": True, "n_units": len(unit_rows)}, indent=2))


if __name__ == "__main__":
    main()
