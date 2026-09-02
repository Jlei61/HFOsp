#!/usr/bin/env python3
"""Summarize the frozen ECoG graph benchmark at graph and patient levels."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_lower_tail(observed: float, null: np.ndarray) -> float:
    """Plus-one exact p for a smaller observed loss than exchangeable null graphs."""
    values = np.asarray(null, dtype=float)
    return float((1 + np.sum(values <= float(observed))) / (len(values) + 1))


def bootstrap_median_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    data = np.asarray(values, dtype=float)
    if data.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    sample = rng.choice(data, size=(20000, data.size), replace=True)
    return tuple(float(value) for value in np.quantile(np.median(sample, axis=1), [0.025, 0.975]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/summary"
    ))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    unit_rows: list[dict[str, Any]] = []
    for subject in ("958", "1084"):
        for path in sorted((args.training_root / subject).glob("*/summary.json")):
            summary = json.loads(path.read_text())
            if bool(summary.get("smoke", False)):
                continue
            graph_index = -1
            graph_id = str(summary["graph_id"])
            if "_" in graph_id and graph_id.rsplit("_", 1)[-1].isdigit():
                graph_index = int(graph_id.rsplit("_", 1)[-1])
            unit_rows.append({
                "subject": subject,
                "family": summary["family"],
                "graph_id": graph_id,
                "graph_index": graph_index,
                "seed_index": int(summary["seed_index"]),
                "test_contact_nll": float(summary["test"]["contact_nll"]),
                "test_top1": float(summary["test"]["top1"]),
                "validation_contact_nll": float(summary["validation"]["contact_nll"]),
                "best_epoch": int(summary["best_epoch"]),
                "epochs_completed": int(summary["epochs_completed"]),
                "runtime_sec": float(summary["runtime_sec"]),
                "initial_parameter_sha256": summary["initial_parameter_sha256"],
                "best_parameter_sha256": summary["best_parameter_sha256"],
                "summary_path": str(path),
            })
    expected = 384
    if len(unit_rows) != expected and not args.allow_incomplete:
        raise RuntimeError(f"need {expected} formal units, found {len(unit_rows)}")
    write_csv(args.output_root / "TRAINING_UNIT_RESULTS.csv", unit_rows)

    patient_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for subject in ("958", "1084"):
        rows = [row for row in unit_rows if row["subject"] == subject]
        if not rows:
            continue
        initial_by_seed: dict[int, set[str]] = {}
        for row in rows:
            initial_by_seed.setdefault(int(row["seed_index"]), set()).add(str(row["initial_parameter_sha256"]))
        initial_match = all(len(values) == 1 for values in initial_by_seed.values())
        true = sorted(
            [row for row in rows if row["family"] == "TRUE_GRID"],
            key=lambda row: int(row["seed_index"]),
        )
        suffix = sorted(
            [row for row in rows if row["family"] == "SUFFIX_SHUFFLED"],
            key=lambda row: int(row["seed_index"]),
        )
        if len(true) != 3 or len(suffix) != 3:
            if args.allow_incomplete:
                continue
            raise RuntimeError(f"{subject}: true/suffix seed count mismatch")
        true_nll = np.asarray([row["test_contact_nll"] for row in true])
        true_top1 = np.asarray([row["test_top1"] for row in true])
        family_payload: dict[str, dict[str, Any]] = {}
        for family in ("WRONG_GRID", "DEGREE_RANDOM"):
            graph_values = []
            graph_top1 = []
            for graph_index in range(31):
                matched = sorted(
                    [row for row in rows if row["family"] == family and row["graph_index"] == graph_index],
                    key=lambda row: int(row["seed_index"]),
                )
                if len(matched) != 3:
                    if args.allow_incomplete:
                        continue
                    raise RuntimeError(f"{subject} {family} graph {graph_index}: need 3 seeds")
                values = np.asarray([row["test_contact_nll"] for row in matched])
                top1 = np.asarray([row["test_top1"] for row in matched])
                graph_values.append(float(np.median(values)))
                graph_top1.append(float(np.median(top1)))
                graph_rows.append({
                    "subject": subject, "family": family, "graph_index": graph_index,
                    "true_contact_nll_median_seed": float(np.median(true_nll)),
                    "control_contact_nll_median_seed": float(np.median(values)),
                    "true_minus_control_nll": float(np.median(true_nll) - np.median(values)),
                    "true_top1_median_seed": float(np.median(true_top1)),
                    "control_top1_median_seed": float(np.median(top1)),
                    "true_minus_control_top1": float(np.median(true_top1) - np.median(top1)),
                })
            family_payload[family] = {
                "nll": np.asarray(graph_values), "top1": np.asarray(graph_top1),
            }

        true_median = float(np.median(true_nll))
        suffix_delta = np.asarray([
            true[seed]["test_contact_nll"] - suffix[seed]["test_contact_nll"] for seed in range(3)
        ])
        wrong_delta = true_median - family_payload["WRONG_GRID"]["nll"]
        random_delta = true_median - family_payload["DEGREE_RANDOM"]["nll"]
        wrong_ci = bootstrap_median_ci(wrong_delta, 202608161)
        random_ci = bootstrap_median_ci(random_delta, 202608162)
        patient_rows.append({
            "subject": subject,
            "n_formal_units": len(rows),
            "initial_trainable_parameters_identical_within_seed": initial_match,
            "true_grid_contact_nll_median_seed": true_median,
            "true_grid_top1_median_seed": float(np.median(true_top1)),
            "wrong_grid_contact_nll_median_graph": float(np.median(family_payload["WRONG_GRID"]["nll"])),
            "true_minus_wrong_grid_nll_median": float(np.median(wrong_delta)),
            "true_minus_wrong_grid_nll_ci95_low": wrong_ci[0],
            "true_minus_wrong_grid_nll_ci95_high": wrong_ci[1],
            "true_better_than_wrong_grid_count": int(np.sum(wrong_delta < 0)),
            "true_vs_wrong_grid_exact_p_lower": exact_lower_tail(
                true_median, family_payload["WRONG_GRID"]["nll"]
            ),
            "degree_random_contact_nll_median_graph": float(np.median(family_payload["DEGREE_RANDOM"]["nll"])),
            "true_minus_degree_random_nll_median": float(np.median(random_delta)),
            "true_minus_degree_random_nll_ci95_low": random_ci[0],
            "true_minus_degree_random_nll_ci95_high": random_ci[1],
            "true_better_than_degree_random_count": int(np.sum(random_delta < 0)),
            "true_vs_degree_random_exact_p_lower": exact_lower_tail(
                true_median, family_payload["DEGREE_RANDOM"]["nll"]
            ),
            "true_minus_suffix_shuffled_nll_median_seed": float(np.median(suffix_delta)),
            "true_better_than_suffix_seed_count": int(np.sum(suffix_delta < 0)),
        })
    write_csv(args.output_root / "GRAPH_LEVEL_EFFECTS.csv", graph_rows)
    write_csv(args.output_root / "PATIENT_RESULTS.csv", patient_rows)
    payload = {
        "schema": "topic5_ecog_graph_training_summary_v0.1",
        "expected_units": expected,
        "observed_formal_units": len(unit_rows),
        "complete": len(unit_rows) == expected,
        "patient_results": patient_rows,
        "scientific_boundary": (
            "Negative true-minus-control NLL supports a useful physical-neighbour training constraint; "
            "it does not establish post-training necessity, which is adjudicated separately by patch attenuation."
        ),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "GRAPH_TRAINING_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
