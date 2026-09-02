#!/usr/bin/env python3
"""Summarize the one-microstep true-vs-isomorphic-wrong-grid sensitivity."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path("results/topic5_ecog_physical_neighborhood_rnn_v0_1")


def exact_lower(observed: float, null: np.ndarray) -> float:
    return float((1 + np.sum(np.asarray(null) <= observed)) / (len(null) + 1))


def main() -> None:
    rows = []
    for subject in ("958", "1084"):
        for path in sorted((ROOT / "training_one_microstep" / subject).glob("*/summary.json")):
            data = json.loads(path.read_text())
            if int(data.get("microsteps", -1)) != 1:
                raise RuntimeError(f"not a one-microstep unit: {path}")
            graph_id = str(data["graph_id"])
            rows.append({
                "subject": subject,
                "family": data["family"],
                "graph_id": graph_id,
                "graph_index": int(graph_id.rsplit("_", 1)[-1])
                if graph_id.rsplit("_", 1)[-1].isdigit() else -1,
                "seed_index": int(data["seed_index"]),
                "test_contact_nll": float(data["test"]["contact_nll"]),
            })
    if len(rows) != 192:
        raise RuntimeError(f"need 192 one-microstep units, found {len(rows)}")
    patient = []
    graph_rows = []
    for subject in ("958", "1084"):
        true = sorted(
            [row for row in rows if row["subject"] == subject and row["family"] == "TRUE_GRID"],
            key=lambda row: row["seed_index"],
        )
        true_values = np.asarray([row["test_contact_nll"] for row in true])
        controls = []
        for graph_index in range(31):
            matched = sorted([
                row for row in rows
                if row["subject"] == subject and row["family"] == "WRONG_GRID"
                and row["graph_index"] == graph_index
            ], key=lambda row: row["seed_index"])
            values = np.asarray([row["test_contact_nll"] for row in matched])
            controls.append(float(np.median(values)))
            graph_rows.append({
                "subject": subject,
                "graph_index": graph_index,
                "true_minus_wrong_nll": float(np.median(true_values - values)),
            })
        true_median = float(np.median(true_values))
        controls_array = np.asarray(controls)
        patient.append({
            "subject": subject,
            "true_nll_median_seed": true_median,
            "wrong_nll_median_graph": float(np.median(controls_array)),
            "true_minus_wrong_nll_median_graph": float(np.median(true_median - controls_array)),
            "true_better_count": int(np.sum(true_median < controls_array)),
            "exact_one_sided_p": exact_lower(true_median, controls_array),
        })
    output = ROOT / "summary"
    output.mkdir(parents=True, exist_ok=True)
    for path, values in (
        (output / "ONE_MICROSTEP_GRAPH_EFFECTS.csv", graph_rows),
        (output / "ONE_MICROSTEP_PATIENT_RESULTS.csv", patient),
    ):
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    payload = {
        "schema": "topic5_ecog_one_microstep_summary_v0.1",
        "complete": True,
        "n_units": 192,
        "patient_results": patient,
    }
    (output / "ONE_MICROSTEP_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
