#!/usr/bin/env python3
"""Fit the nested unordered-history + ordered GRU correction (v2.3.1)."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_stable_repertoire_r4 import prepare  # noqa: E402
from src.topic5_stable_repertoire_event_rnn import (  # noqa: E402
    circularly_shift_targets,
    fit_residual_gru_event_state,
    score_predictions,
    shuffled_histories,
)


DEFAULT_CONFIG = ROOT / "config/topic5_stable_repertoire_event_rnn_v2_3.yaml"


def score_dict(score):
    return {key: float(value) for key, value in asdict(score).items()}


def output_root(config):
    name = "v2_3_h40_residual_gru" if int(config["horizon"]) == 40 else "v2_3_1_residual_gru"
    return ROOT / "results/topic5_stable_repertoire_event_rnn/development" / name


def fit_condition(datasets, config, seed):
    n_modes = int(config["n_modes"])
    n_contacts = (datasets["train"].targets.shape[1] - n_modes) // 2
    model = fit_residual_gru_event_state(
        datasets["train"],
        datasets["validation"],
        hidden_size_grid=config["gru_hidden_size_grid"],
        weight_decay_grid=config["gru_weight_decay_grid"],
        alpha_grid=config["ridge_alpha_grid"],
        learning_rate=float(config["gru_learning_rate"]),
        batch_size=int(config["gru_batch_size"]),
        maximum_epochs=int(config["gru_maximum_epochs"]),
        patience=int(config["gru_patience"]),
        n_modes=n_modes,
        n_contacts=n_contacts,
        seed=int(seed),
    )
    prediction = model.predict(datasets["test"])
    score = score_predictions(
        datasets["test"].targets, prediction, n_modes=n_modes, n_contacts=n_contacts
    )
    info = {
        "seed": int(seed),
        "hidden_size": int(model.hidden_size),
        "weight_decay": float(model.weight_decay),
        "base_alpha": float(model.base_alpha),
        "best_epoch": int(model.best_epoch),
        "training_adequate": bool(
            int(model.best_epoch) < int(config["gru_maximum_epochs"]) - 1
        ),
        "n_parameters": int(model.n_parameters),
        "validation_score": score_dict(model.best_validation_score),
        "test_score": score_dict(score),
    }
    return model, info, prediction


def save_checkpoint(path, model, info, condition):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "contract": "topic5_stable_repertoire_event_rnn_v2_3_1_nested_residual",
        "condition": condition,
        "model_state_dict": model.model.state_dict(),
        "token_mean": model.token_mean,
        "token_scale": model.token_scale,
        "hidden_size": model.hidden_size,
        "n_modes": model.n_modes,
        "base_coef": model.base_model.coef_,
        "base_intercept": model.base_model.intercept_,
        "run_info": info,
    }, path)


def median_score(runs):
    return {
        field: float(np.median([run["test_score"][field] for run in runs]))
        for field in ("composite", "occupancy", "rank", "participation")
    }


def run_patient(subject, config, output):
    datasets, _ = prepare(subject, config)
    runs = {"ordered": [], "shuffle": [], "circular": []}
    predictions = {key: [] for key in runs}
    for seed in map(int, config["seeds"]):
        conditions = {
            "ordered": datasets,
            "shuffle": {
                name: shuffled_histories(dataset, seed + 40_000)
                for name, dataset in datasets.items()
            },
            "circular": {
                name: circularly_shift_targets(dataset)
                for name, dataset in datasets.items()
            },
        }
        for condition, values in conditions.items():
            model, info, prediction = fit_condition(values, config, seed)
            runs[condition].append(info)
            predictions[condition].append(prediction)
            save_checkpoint(
                output / "checkpoints" / subject / f"{condition}_seed{seed}.pt",
                model, info, condition,
            )
    medians = {name: median_score(value) for name, value in runs.items()}
    common = json.load((ROOT / config["output_root"] / "per_subject" / f"{subject}.json").open())
    linear_root = (
        "v2_3_h40_residual_linear" if int(config["horizon"]) == 40
        else "v2_3_1_residual_linear"
    )
    linear_path = (
        ROOT / "results/topic5_stable_repertoire_event_rnn/development"
        / linear_root / "per_subject" / f"{subject}.json"
    )
    linear = json.load(linear_path.open())
    unordered = min(
        common["r1_recent_ridge"]["test_score"]["composite"],
        common["r1_long_history_summary_ridge"]["test_score"]["composite"],
    )
    linear_score = linear["ordered"]["median_test_score"]["composite"]
    ordered = medians["ordered"]["composite"]
    result = {
        "contract": "topic5_stable_repertoire_event_rnn_v2_3_1_nested_residual",
        "subject": subject,
        "n_test_windows": int(len(datasets["test"])),
        "ordered": {"runs": runs["ordered"], "median_test_score": medians["ordered"]},
        "within_history_shuffle": {"runs": runs["shuffle"], "median_test_score": medians["shuffle"]},
        "circular_pairing": {"runs": runs["circular"], "median_test_score": medians["circular"]},
        "comparators": {
            "strongest_unordered": float(unordered),
            "nested_linear": float(linear_score),
        },
        "gates": {
            "nested_gru_beats_strongest_unordered": bool(ordered < unordered),
            "nested_gru_beats_nested_linear": bool(ordered < linear_score),
            "nested_gru_beats_shuffle": bool(ordered < medians["shuffle"]["composite"]),
            "nested_gru_beats_circular": bool(ordered < medians["circular"]["composite"]),
            "all_runs_training_adequate": bool(
                all(run["training_adequate"] for values in runs.values() for run in values)
            ),
        },
        "provenance": {
            "old_heldout20_entered": False,
            "forbidden_labels_entered": False,
            "geometry_soz_ictal_snn_entered": False,
            "linear_comparator": str(linear_path.resolve()),
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
        unordered = result["comparators"]["strongest_unordered"]
        linear = result["comparators"]["nested_linear"]
        rows.append({
            "subject": result["subject"],
            "n_test_windows": result["n_test_windows"],
            "strongest_unordered": unordered,
            "nested_linear": linear,
            "nested_gru": ordered,
            "gru_minus_unordered": ordered - unordered,
            "gru_minus_linear": ordered - linear,
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
        "n_beats_strongest_unordered": int(frame["nested_gru_beats_strongest_unordered"].sum()),
        "n_beats_nested_linear": int(frame["nested_gru_beats_nested_linear"].sum()),
        "n_beats_shuffle": int(frame["nested_gru_beats_shuffle"].sum()),
        "n_beats_circular": int(frame["nested_gru_beats_circular"].sum()),
        "n_all_runs_training_adequate": int(frame["all_runs_training_adequate"].sum()),
        "median_gru_minus_unordered": float(frame["gru_minus_unordered"].median()),
        "median_gru_minus_linear": float(frame["gru_minus_linear"].median()),
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
    torch.set_num_threads(1)
    results = []
    for subject in subjects:
        print(f"[v2.3.1 residual GRU] {subject}", flush=True)
        results.append(run_patient(subject, config, output))
    aggregate(results, output, config)


if __name__ == "__main__":
    main()
