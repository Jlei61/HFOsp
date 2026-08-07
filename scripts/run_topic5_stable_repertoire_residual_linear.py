#!/usr/bin/env python3
"""Fit the nested R1 + ordered linear-history correction (v2.3.1)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_stable_repertoire_r4 import prepare  # noqa: E402
from src.topic5_stable_repertoire_event_rnn import (  # noqa: E402
    circularly_shift_targets,
    fit_residual_linear_event_state,
    score_predictions,
    shuffled_histories,
)


DEFAULT_CONFIG = ROOT / "config/topic5_stable_repertoire_event_rnn_v2_3.yaml"


def score_dict(score):
    return {key: float(value) for key, value in asdict(score).items()}


def fit_condition(datasets, config, seed):
    n_modes = int(config["n_modes"])
    n_contacts = (datasets["train"].targets.shape[1] - n_modes) // 2
    model = fit_residual_linear_event_state(
        datasets["train"],
        datasets["validation"],
        dimension_grid=config["linear_dimension_grid"],
        decay_grid=config["linear_decay_grid"],
        alpha_grid=config["ridge_alpha_grid"],
        n_modes=n_modes,
        n_contacts=n_contacts,
        seed=int(seed),
    )
    prediction = model.predict(datasets["test"])
    score = score_predictions(
        datasets["test"].targets,
        prediction,
        n_modes=n_modes,
        n_contacts=n_contacts,
    )
    return {
        "seed": int(seed),
        "dimension": int(model.dimension),
        "decay": float(model.decay),
        "base_alpha": float(model.base_alpha),
        "correction_alpha": float(model.correction_alpha),
        "validation_score": score_dict(model.validation_score),
        "test_score": score_dict(score),
    }, prediction


def median_score(runs):
    return {
        field: float(np.median([run["test_score"][field] for run in runs]))
        for field in ("composite", "occupancy", "rank", "participation")
    }


def output_root(config):
    stem = "v2_3_h40_residual_linear" if int(config["horizon"]) == 40 else "v2_3_1_residual_linear"
    return ROOT / "results/topic5_stable_repertoire_event_rnn/development" / stem


def run_patient(subject, config, output):
    datasets, _ = prepare(subject, config)
    runs = {"ordered": [], "shuffle": [], "circular": []}
    predictions = {key: [] for key in runs}
    for seed in map(int, config["seeds"]):
        conditions = {
            "ordered": datasets,
            "shuffle": {
                name: shuffled_histories(dataset, seed + 30_000)
                for name, dataset in datasets.items()
            },
            "circular": {
                name: circularly_shift_targets(dataset)
                for name, dataset in datasets.items()
            },
        }
        for name, values in conditions.items():
            info, prediction = fit_condition(values, config, seed)
            runs[name].append(info)
            predictions[name].append(prediction)
    medians = {name: median_score(value) for name, value in runs.items()}
    previous_path = ROOT / config["output_root"] / "per_subject" / f"{subject}.json"
    previous = json.load(previous_path.open())
    r1 = previous["r1_recent_ridge"]["test_score"]["composite"]
    r1_long = previous["r1_long_history_summary_ridge"]["test_score"]["composite"]
    strongest_unordered = min(r1, r1_long)
    direct = previous["r3_linear_event_state"]["median_test_score"]["composite"]
    ordered = medians["ordered"]["composite"]
    result = {
        "contract": "topic5_stable_repertoire_event_rnn_v2_3_1_nested_residual",
        "subject": subject,
        "n_test_windows": int(len(datasets["test"])),
        "ordered": {"runs": runs["ordered"], "median_test_score": medians["ordered"]},
        "within_history_shuffle": {"runs": runs["shuffle"], "median_test_score": medians["shuffle"]},
        "circular_pairing": {"runs": runs["circular"], "median_test_score": medians["circular"]},
        "comparators": {
            "r1_recent": float(r1),
            "r1_long_history_summary": float(r1_long),
            "strongest_unordered": float(strongest_unordered),
            "r3_direct": float(direct),
        },
        "gates": {
            "nested_linear_beats_r1": bool(ordered < r1),
            "nested_linear_beats_strongest_unordered": bool(ordered < strongest_unordered),
            "nested_linear_beats_direct_r3": bool(ordered < direct),
            "nested_linear_beats_shuffle": bool(ordered < medians["shuffle"]["composite"]),
            "nested_linear_beats_circular": bool(ordered < medians["circular"]["composite"]),
        },
        "provenance": {
            "old_heldout20_entered": False,
            "forbidden_labels_entered": False,
            "geometry_soz_ictal_snn_entered": False,
            "common_r0_r3_artifact": str(previous_path.resolve()),
        },
    }
    (output / "per_subject").mkdir(parents=True, exist_ok=True)
    json.dump(result, (output / "per_subject" / f"{subject}.json").open("w"), indent=2, sort_keys=True)
    np.savez_compressed(
        output / "per_subject" / f"{subject}_predictions.npz",
        target=datasets["test"].targets.astype(np.float32),
        ordered=np.mean(predictions["ordered"], axis=0).astype(np.float32),
        shuffle=np.mean(predictions["shuffle"], axis=0).astype(np.float32),
        circular=np.mean(predictions["circular"], axis=0).astype(np.float32),
        target_event_indices=datasets["test"].target_event_indices,
        history_event_indices=datasets["test"].history_event_indices,
    )
    return result


def aggregate(results, output, config):
    rows = []
    for result in results:
        ordered = result["ordered"]["median_test_score"]["composite"]
        rows.append({
            "subject": result["subject"],
            "n_test_windows": result["n_test_windows"],
            "r1_recent": result["comparators"]["r1_recent"],
            "r1_long_history_summary": result["comparators"]["r1_long_history_summary"],
            "strongest_unordered": result["comparators"]["strongest_unordered"],
            "r3_direct": result["comparators"]["r3_direct"],
            "nested_linear": ordered,
            "nested_minus_r1": ordered - result["comparators"]["r1_recent"],
            "nested_minus_strongest_unordered": ordered - result["comparators"]["strongest_unordered"],
            "nested_minus_direct": ordered - result["comparators"]["r3_direct"],
            "shuffle": result["within_history_shuffle"]["median_test_score"]["composite"],
            "circular": result["circular_pairing"]["median_test_score"]["composite"],
            **result["gates"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    state = {
        "contract": "topic5_stable_repertoire_event_rnn_v2_3_1_nested_residual",
        "horizon": int(config["horizon"]),
        "status": "COMPLETE",
        "n_patients": int(len(frame)),
        "n_beats_r1": int(frame["nested_linear_beats_r1"].sum()),
        "n_beats_strongest_unordered": int(frame["nested_linear_beats_strongest_unordered"].sum()),
        "n_beats_direct_r3": int(frame["nested_linear_beats_direct_r3"].sum()),
        "n_beats_shuffle": int(frame["nested_linear_beats_shuffle"].sum()),
        "n_beats_circular": int(frame["nested_linear_beats_circular"].sum()),
        "median_nested_minus_r1": float(frame["nested_minus_r1"].median()),
        "median_nested_minus_strongest_unordered": float(
            frame["nested_minus_strongest_unordered"].median()
        ),
        "old_heldout20_entered": False,
        "forbidden_labels_entered": False,
        "geometry_soz_ictal_snn_entered": False,
    }
    json.dump(state, (output / "STATE.json").open("w"), indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.open())
    subjects = args.subjects or config["pilot_subjects"]
    output = output_root(config)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for subject in subjects:
        print(f"[v2.3.1 residual linear] {subject}", flush=True)
        results.append(run_patient(subject, config, output))
    aggregate(results, output, config)


if __name__ == "__main__":
    main()
