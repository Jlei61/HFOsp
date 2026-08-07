#!/usr/bin/env python3
"""Run R4 GRU after the R0--R3 stable-repertoire pilot is complete."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_stable_repertoire_event_rnn import (  # noqa: E402
    build_future_window_dataset,
    chronological_source_partition,
    circularly_shift_targets,
    fit_gru_event_state,
    fit_stable_templates,
    score_predictions,
    shuffled_histories,
)
from scripts.run_topic5_stable_repertoire_r0_r3 import file_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic5_stable_repertoire_event_rnn_v2_3.yaml"


def score_dict(score) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(score).items()}


def prepare(subject: str, config: dict[str, Any]):
    data_path = ROOT / config["dataset_root"] / f"{subject}.npz"
    map_path = ROOT / config["source_mapping_root"] / f"{subject}.npz"
    raw = np.load(data_path, allow_pickle=False)
    mapping = np.load(map_path, allow_pickle=False)
    rank = np.asarray(raw["event_local_rank"], float)
    participation = np.asarray(raw["event_participation"], bool)
    event_time = np.asarray(raw["event_abs_time"], float)
    source = np.asarray(mapping["event_source_block_id"], int)
    eligible = np.flatnonzero(np.asarray(raw["event_split"], int) == 0)
    partition = chronological_source_partition(
        source, event_time, eligible, fractions=config["source_split"]
    )
    split_indices = {
        name: partition.indices(source, name, eligible)
        for name in ("train", "validation", "test")
    }
    encoder = fit_stable_templates(
        rank,
        participation,
        split_indices["train"],
        n_modes=int(config["n_modes"]),
        seed=0,
    )
    tokens, modes = encoder.event_tokens(rank, participation)
    datasets = {
        name: build_future_window_dataset(
            tokens,
            modes,
            rank,
            participation,
            source,
            event_time,
            indices,
            encoder,
            history_length=int(config["history_length"]),
            horizon=int(config["horizon"]),
        )
        for name, indices in split_indices.items()
    }
    return datasets, rank.shape[1]


def run_condition(train, validation, test, config, seed: int):
    n_modes = int(config["n_modes"])
    n_contacts = (train.targets.shape[1] - n_modes) // 2
    fitted = fit_gru_event_state(
        train,
        validation,
        hidden_size_grid=config["gru_hidden_size_grid"],
        weight_decay_grid=config["gru_weight_decay_grid"],
        learning_rate=float(config["gru_learning_rate"]),
        batch_size=int(config["gru_batch_size"]),
        maximum_epochs=int(config["gru_maximum_epochs"]),
        patience=int(config["gru_patience"]),
        n_modes=n_modes,
        n_contacts=n_contacts,
        seed=int(seed),
    )
    prediction = fitted.predict(test.histories)
    score = score_predictions(
        test.targets, prediction, n_modes=n_modes, n_contacts=n_contacts
    )
    info = {
        "seed": int(seed),
        "hidden_size": int(fitted.hidden_size),
        "weight_decay": float(fitted.weight_decay),
        "best_epoch": int(fitted.best_epoch),
        "training_adequate": bool(
            int(fitted.best_epoch) < int(config["gru_maximum_epochs"]) - 1
        ),
        "n_parameters": int(fitted.n_parameters),
        "best_validation_score": score_dict(fitted.best_validation_score),
        "test_score": score_dict(score),
    }
    return fitted, info, prediction


def save_checkpoint(path: Path, fitted, info: dict[str, Any], condition: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": "topic5_stable_repertoire_event_rnn_v2_3",
            "condition": condition,
            "model_state_dict": fitted.model.state_dict(),
            "token_mean": fitted.token_mean,
            "token_scale": fitted.token_scale,
            "hidden_size": fitted.hidden_size,
            "n_modes": fitted.n_modes,
            "run_info": info,
        },
        path,
    )


def median_score(runs: list[dict[str, Any]]) -> dict[str, float]:
    return {
        field: float(np.median([run["test_score"][field] for run in runs]))
        for field in ("composite", "occupancy", "rank", "participation")
    }


def run_patient(subject: str, config: dict[str, Any], output: Path) -> dict[str, Any]:
    datasets, n_contacts = prepare(subject, config)
    all_runs: dict[str, list[dict[str, Any]]] = {
        "ordered": [], "within_history_shuffle": [], "circular_pairing": []
    }
    all_predictions: dict[str, list[np.ndarray]] = {key: [] for key in all_runs}
    for seed in map(int, config["seeds"]):
        conditions = {
            "ordered": datasets,
            "within_history_shuffle": {
                name: shuffled_histories(dataset, seed + 20_000)
                for name, dataset in datasets.items()
            },
            "circular_pairing": {
                name: circularly_shift_targets(dataset)
                for name, dataset in datasets.items()
            },
        }
        for condition, values in conditions.items():
            fitted, info, prediction = run_condition(
                values["train"], values["validation"], values["test"], config, seed
            )
            all_runs[condition].append(info)
            all_predictions[condition].append(prediction)
            save_checkpoint(
                output / "checkpoints" / subject / f"{condition}_seed{seed}.pt",
                fitted,
                info,
                condition,
            )
    medians = {condition: median_score(runs) for condition, runs in all_runs.items()}
    r0_r3_path = ROOT / config["output_root"] / "per_subject" / f"{subject}.json"
    with r0_r3_path.open() as stream:
        previous = json.load(stream)
    strongest_baseline = min(
        previous["r0_static"]["test_score"]["composite"],
        previous["r1_recent_ridge"]["test_score"]["composite"],
        previous["r1_long_history_summary_ridge"]["test_score"]["composite"],
        previous["r2_discrete_switching"]["test_score"]["composite"],
    )
    r3 = previous["r3_linear_event_state"]["median_test_score"]["composite"]
    ordered = medians["ordered"]["composite"]
    result = {
        "contract": config["contract"],
        "subject": subject,
        "n_contacts": int(n_contacts),
        "n_prediction_windows": {
            name: int(len(dataset)) for name, dataset in datasets.items()
        },
        "r4_gru_event_state": {
            "runs": all_runs["ordered"],
            "median_test_score": medians["ordered"],
        },
        "r4_within_history_shuffle": {
            "runs": all_runs["within_history_shuffle"],
            "median_test_score": medians["within_history_shuffle"],
        },
        "r4_circular_pairing_null": {
            "runs": all_runs["circular_pairing"],
            "median_test_score": medians["circular_pairing"],
        },
        "comparators": {
            "strongest_r0_r2_composite": float(strongest_baseline),
            "r3_linear_composite": float(r3),
        },
        "gates": {
            "r4_beats_strongest_r0_r2": bool(ordered < strongest_baseline),
            "r4_beats_r3": bool(ordered < r3),
            "r4_beats_shuffle": bool(ordered < medians["within_history_shuffle"]["composite"]),
            "r4_beats_circular": bool(ordered < medians["circular_pairing"]["composite"]),
            "all_runs_finite": bool(
                all(
                    np.isfinite(run["test_score"]["composite"])
                    for runs in all_runs.values() for run in runs
                )
            ),
            "all_runs_training_adequate": bool(
                all(run["training_adequate"] for runs in all_runs.values() for run in runs)
            ),
        },
        "provenance": {
            "old_heldout20_entered": False,
            "forbidden_labels_entered": False,
            "geometry_soz_ictal_snn_entered": False,
            "r0_r3_path": str(r0_r3_path.resolve()),
            "r0_r3_sha256": file_sha256(r0_r3_path),
        },
    }
    (output / "per_subject").mkdir(parents=True, exist_ok=True)
    with (output / "per_subject" / f"{subject}.json").open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    np.savez_compressed(
        output / "per_subject" / f"{subject}_predictions.npz",
        target=datasets["test"].targets.astype(np.float32),
        r4=np.mean(all_predictions["ordered"], axis=0).astype(np.float32),
        r4_shuffle=np.mean(all_predictions["within_history_shuffle"], axis=0).astype(np.float32),
        r4_circular=np.mean(all_predictions["circular_pairing"], axis=0).astype(np.float32),
        target_event_indices=datasets["test"].target_event_indices,
        history_event_indices=datasets["test"].history_event_indices,
    )
    return result


def aggregate(results: list[dict[str, Any]], output: Path, config_path: Path) -> None:
    rows = []
    for result in results:
        ordered = result["r4_gru_event_state"]["median_test_score"]["composite"]
        r3 = result["comparators"]["r3_linear_composite"]
        baseline = result["comparators"]["strongest_r0_r2_composite"]
        rows.append(
            {
                "subject": result["subject"],
                "n_test_windows": result["n_prediction_windows"]["test"],
                "strongest_r0_r2_composite": baseline,
                "r3_composite": r3,
                "r4_composite": ordered,
                "r4_minus_strongest": ordered - baseline,
                "r4_minus_r3": ordered - r3,
                "r4_shuffle_composite": result["r4_within_history_shuffle"]["median_test_score"]["composite"],
                "r4_circular_composite": result["r4_circular_pairing_null"]["median_test_score"]["composite"],
                **result["gates"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    state = {
        "contract": "topic5_stable_repertoire_event_rnn_v2_3",
        "status": "R4_COMPLETE",
        "n_patients": int(len(frame)),
        "n_r4_beats_strongest_r0_r2": int(frame["r4_beats_strongest_r0_r2"].sum()),
        "n_r4_beats_r3": int(frame["r4_beats_r3"].sum()),
        "n_r4_beats_shuffle": int(frame["r4_beats_shuffle"].sum()),
        "n_r4_beats_circular": int(frame["r4_beats_circular"].sum()),
        "n_all_runs_finite": int(frame["all_runs_finite"].sum()),
        "n_all_runs_training_adequate": int(frame["all_runs_training_adequate"].sum()),
        "median_r4_minus_strongest": float(frame["r4_minus_strongest"].median()),
        "median_r4_minus_r3": float(frame["r4_minus_r3"].median()),
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "source_sha256": file_sha256(Path(__file__)),
        "old_heldout20_entered": False,
        "forbidden_labels_entered": False,
        "geometry_soz_ictal_snn_entered": False,
    }
    with (output / "R4_STATE.json").open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config_path = args.config.resolve()
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    subjects = args.subjects or config["pilot_subjects"]
    output = ROOT / config["r4_output_root"]
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    results = []
    for subject in subjects:
        print(f"[v2.3 R4] {subject}", flush=True)
        results.append(run_patient(subject, config, output))
    aggregate(results, output, config_path)


if __name__ == "__main__":
    main()
