#!/usr/bin/env python3
"""Run the frozen R0--R3 event-indexed stable-repertoire pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_stable_repertoire_event_rnn import (  # noqa: E402
    build_future_window_dataset,
    chronological_source_partition,
    circularly_shift_targets,
    concatenate_window_datasets,
    estimate_transition_matrix,
    fit_linear_event_state,
    fit_history_summary_ridge,
    fit_recent_ridge,
    fit_stable_templates,
    future_window_split_half_reliability,
    mode_conditioned_descriptors,
    project_descriptor,
    refit_linear_event_state,
    score_predictions,
    shuffled_histories,
    transition_predictions,
    train_to_partition_template_stability,
    verify_dataset_contract,
)


DEFAULT_CONFIG = ROOT / "config/topic5_stable_repertoire_event_rnn_v2_3.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score_dict(score) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(score).items()}


def patient_split_indices(
    source: np.ndarray,
    partition,
    eligible_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        name: partition.indices(source, name, eligible_indices)
        for name in ("train", "validation", "test")
    }


def fit_final_recent(train, validation, selected_alpha: float, n_modes: int):
    combined = concatenate_window_datasets(train, validation)
    model = Ridge(alpha=float(selected_alpha)).fit(combined.recent_descriptors, combined.targets)
    return lambda dataset: project_descriptor(model.predict(dataset.recent_descriptors), n_modes)


def fit_final_history_summary(train, validation, selected_alpha: float, n_modes: int):
    combined = concatenate_window_datasets(train, validation)
    model = Ridge(alpha=float(selected_alpha)).fit(combined.history_descriptors, combined.targets)
    return lambda dataset: project_descriptor(model.predict(dataset.history_descriptors), n_modes)


def run_linear_condition(
    train,
    validation,
    test,
    config: dict[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = fit_linear_event_state(
        train,
        validation,
        dimension_grid=config["linear_dimension_grid"],
        decay_grid=config["linear_decay_grid"],
        alpha_grid=config["ridge_alpha_grid"],
        n_modes=int(config["n_modes"]),
        n_contacts=(train.targets.shape[1] - int(config["n_modes"])) // 2,
        seed=int(seed),
    )
    final = refit_linear_event_state(
        concatenate_window_datasets(train, validation),
        dimension=selected.dimension,
        decay=selected.decay,
        alpha=selected.alpha,
        n_modes=int(config["n_modes"]),
        seed=int(seed),
    )
    prediction = final.predict(test.histories)
    score = score_predictions(
        test.targets,
        prediction,
        n_modes=int(config["n_modes"]),
        n_contacts=(test.targets.shape[1] - int(config["n_modes"])) // 2,
    )
    return {
        "seed": int(seed),
        "dimension": int(selected.dimension),
        "decay": float(selected.decay),
        "alpha": float(selected.alpha),
        "validation_score": score_dict(selected.validation_score),
        "test_score": score_dict(score),
    }, prediction


def median_seed_score(runs: list[dict[str, Any]]) -> dict[str, float]:
    fields = ("composite", "occupancy", "rank", "participation")
    return {
        field: float(np.median([item["test_score"][field] for item in runs]))
        for field in fields
    }


def run_patient(subject: str, config: dict[str, Any], output: Path) -> dict[str, Any]:
    data_path = ROOT / config["dataset_root"] / f"{subject}.npz"
    mapping_path = ROOT / config["source_mapping_root"] / f"{subject}.npz"
    raw = np.load(data_path, allow_pickle=False)
    mapping = np.load(mapping_path, allow_pickle=False)
    rank = np.asarray(raw["event_local_rank"], float)
    participation = np.asarray(raw["event_participation"], bool)
    event_time = np.asarray(raw["event_abs_time"], float)
    event_split = np.asarray(raw["event_split"], int)
    source = np.asarray(mapping["event_source_block_id"], int)
    eligible = np.flatnonzero(event_split == 0)
    partition = chronological_source_partition(
        source,
        event_time,
        eligible,
        fractions=config["source_split"],
    )
    split_indices = patient_split_indices(source, partition, eligible)
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
    n_modes = int(config["n_modes"])
    n_contacts = rank.shape[1]
    train_validation = concatenate_window_datasets(datasets["train"], datasets["validation"])

    static_mean = np.mean(train_validation.targets, axis=0, keepdims=True)
    pred_r0 = np.repeat(static_mean, len(datasets["test"]), axis=0)
    score_r0 = score_predictions(datasets["test"].targets, pred_r0, n_modes=n_modes, n_contacts=n_contacts)

    selected_r1 = fit_recent_ridge(
        datasets["train"],
        datasets["validation"],
        alpha_grid=config["ridge_alpha_grid"],
        n_modes=n_modes,
        n_contacts=n_contacts,
    )
    predict_r1 = fit_final_recent(
        datasets["train"], datasets["validation"], selected_r1.alpha, n_modes
    )
    pred_r1 = predict_r1(datasets["test"])
    score_r1 = score_predictions(datasets["test"].targets, pred_r1, n_modes=n_modes, n_contacts=n_contacts)

    selected_r1_long = fit_history_summary_ridge(
        datasets["train"],
        datasets["validation"],
        alpha_grid=config["ridge_alpha_grid"],
        n_modes=n_modes,
        n_contacts=n_contacts,
    )
    predict_r1_long = fit_final_history_summary(
        datasets["train"], datasets["validation"], selected_r1_long.alpha, n_modes
    )
    pred_r1_long = predict_r1_long(datasets["test"])
    score_r1_long = score_predictions(
        datasets["test"].targets, pred_r1_long, n_modes=n_modes, n_contacts=n_contacts
    )

    train_validation_indices = np.concatenate(
        [split_indices["train"], split_indices["validation"]]
    )
    transition = estimate_transition_matrix(
        modes,
        source,
        train_validation_indices,
        n_modes=n_modes,
    )
    mode_descriptors = mode_conditioned_descriptors(
        train_validation_indices, modes, rank, participation, encoder
    )
    pred_r2 = transition_predictions(
        datasets["test"].last_mode,
        transition,
        mode_descriptors,
        int(config["horizon"]),
    )
    score_r2 = score_predictions(datasets["test"].targets, pred_r2, n_modes=n_modes, n_contacts=n_contacts)

    ordered_runs: list[dict[str, Any]] = []
    shuffled_runs: list[dict[str, Any]] = []
    shifted_runs: list[dict[str, Any]] = []
    ordered_predictions = []
    shuffled_predictions = []
    shifted_predictions = []
    for seed in map(int, config["seeds"]):
        info, prediction = run_linear_condition(
            datasets["train"], datasets["validation"], datasets["test"], config, seed=seed
        )
        ordered_runs.append(info)
        ordered_predictions.append(prediction)

        shuffled = {
            name: shuffled_histories(dataset, seed + 10_000)
            for name, dataset in datasets.items()
        }
        info, prediction = run_linear_condition(
            shuffled["train"], shuffled["validation"], shuffled["test"], config, seed=seed
        )
        shuffled_runs.append(info)
        shuffled_predictions.append(prediction)

        shifted = {
            name: circularly_shift_targets(dataset)
            for name, dataset in datasets.items()
        }
        info, prediction = run_linear_condition(
            shifted["train"], shifted["validation"], shifted["test"], config, seed=seed
        )
        shifted_runs.append(info)
        shifted_predictions.append(prediction)

    mode_occupancy = {
        name: (np.bincount(modes[indices], minlength=n_modes) / len(indices)).tolist()
        for name, indices in split_indices.items()
    }
    template_stability = {
        name: train_to_partition_template_stability(
            rank,
            participation,
            split_indices["train"],
            split_indices[name],
            encoder,
            seed=0,
        )
        for name in ("validation", "test")
    }
    rank_templates = mode_descriptors[:, n_modes : n_modes + n_contacts]
    template_corr = float(spearmanr(rank_templates[0], rank_templates[1]).statistic)
    strongest_baseline = min(
        score_r0.composite, score_r1.composite, score_r1_long.composite, score_r2.composite
    )
    ordered_median = median_seed_score(ordered_runs)
    shuffled_median = median_seed_score(shuffled_runs)
    shifted_median = median_seed_score(shifted_runs)
    minimum_occupancy = float(
        min(min(values) for values in mode_occupancy.values())
    )
    future_window_reliability = future_window_split_half_reliability(
        datasets["validation"],
        modes,
        rank,
        participation,
        encoder,
        repeats=10,
        seed=17,
    )
    contract_checks = {
        name: dict(verify_dataset_contract(dataset, int(config["horizon"]), source))
        for name, dataset in datasets.items()
    }
    contract_checks.update(
        {
            "partition_sources_disjoint": bool(
                set(partition.train_sources).isdisjoint(partition.validation_sources)
                and set(partition.train_sources).isdisjoint(partition.test_sources)
                and set(partition.validation_sources).isdisjoint(partition.test_sources)
            ),
            "old_heldout20_excluded": bool(np.all(event_split[eligible] == 0)),
            "all_final_split_indices_train80_only": bool(
                all(np.all(event_split[indices] == 0) for indices in split_indices.values())
            ),
            "template_fit_indices_train_only": bool(
                set(split_indices["train"]).isdisjoint(split_indices["validation"])
                and set(split_indices["train"]).isdisjoint(split_indices["test"])
            ),
        }
    )
    result = {
        "contract": config["contract"],
        "subject": subject,
        "n_events_train80": int(len(eligible)),
        "n_contacts": int(n_contacts),
        "n_source_records": {
            "train": int(len(partition.train_sources)),
            "validation": int(len(partition.validation_sources)),
            "test": int(len(partition.test_sources)),
        },
        "n_prediction_windows": {name: int(len(dataset)) for name, dataset in datasets.items()},
        "mode_occupancy": mode_occupancy,
        "minimum_partition_mode_occupancy": minimum_occupancy,
        "template_rank_spearman": template_corr,
        "train_to_partition_template_stability": template_stability,
        "validation_future_window_reliability": future_window_reliability,
        "contract_checks": contract_checks,
        "r0_static": {"test_score": score_dict(score_r0)},
        "r1_recent_ridge": {
            "alpha": float(selected_r1.alpha),
            "validation_score": score_dict(selected_r1.validation_score),
            "test_score": score_dict(score_r1),
        },
        "r1_long_history_summary_ridge": {
            "alpha": float(selected_r1_long.alpha),
            "validation_score": score_dict(selected_r1_long.validation_score),
            "test_score": score_dict(score_r1_long),
        },
        "r2_discrete_switching": {
            "transition": transition.tolist(),
            "test_score": score_dict(score_r2),
        },
        "r3_linear_event_state": {
            "runs": ordered_runs,
            "median_test_score": ordered_median,
        },
        "r3_within_history_shuffle": {
            "runs": shuffled_runs,
            "median_test_score": shuffled_median,
        },
        "r3_circular_pairing_null": {
            "runs": shifted_runs,
            "median_test_score": shifted_median,
        },
        "gates": {
            "c0_engineering": bool(
                all(
                    value
                    for check in contract_checks.values()
                    for value in (check.values() if isinstance(check, dict) else [check])
                )
            ),
            "c1_stable_repertoire_readback": bool(
                minimum_occupancy >= float(config["minimum_mode_occupancy"])
                and all(value["grade"] != "weak" for value in template_stability.values())
            ),
            "r3_beats_strongest_baseline": bool(ordered_median["composite"] < strongest_baseline),
            "r3_beats_shuffle": bool(ordered_median["composite"] < shuffled_median["composite"]),
            "r3_beats_circular": bool(ordered_median["composite"] < shifted_median["composite"]),
        },
        "provenance": {
            "dataset_path": str(data_path.resolve()),
            "dataset_sha256": file_sha256(data_path),
            "source_mapping_path": str(mapping_path.resolve()),
            "source_mapping_sha256": file_sha256(mapping_path),
            "template_train_index_sha256": hashlib.sha256(
                np.asarray(split_indices["train"], np.int64).tobytes()
            ).hexdigest(),
            "template_centers_sha256": hashlib.sha256(
                np.asarray(encoder.centers, np.float64).tobytes()
            ).hexdigest(),
            "old_heldout20_entered": False,
            "forbidden_labels_entered": False,
            "geometry_soz_ictal_snn_entered": False,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_subject").mkdir(exist_ok=True)
    with (output / "per_subject" / f"{subject}.json").open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    np.savez_compressed(
        output / "per_subject" / f"{subject}_predictions.npz",
        target=datasets["test"].targets.astype(np.float32),
        r0=pred_r0.astype(np.float32),
        r1=pred_r1.astype(np.float32),
        r1_long=pred_r1_long.astype(np.float32),
        r2=pred_r2.astype(np.float32),
        r3=np.mean(ordered_predictions, axis=0).astype(np.float32),
        r3_shuffle=np.mean(shuffled_predictions, axis=0).astype(np.float32),
        r3_circular=np.mean(shifted_predictions, axis=0).astype(np.float32),
        test_source=datasets["test"].source_ids,
        target_start=datasets["test"].target_start,
        target_stop=datasets["test"].target_stop,
        target_event_indices=datasets["test"].target_event_indices,
        history_event_indices=datasets["test"].history_event_indices,
    )
    return result


def aggregate(results: list[dict[str, Any]], config: dict[str, Any], output: Path, config_path: Path) -> None:
    rows = []
    for result in results:
        baselines = {
            "r0": result["r0_static"]["test_score"]["composite"],
            "r1": result["r1_recent_ridge"]["test_score"]["composite"],
            "r1_long": result["r1_long_history_summary_ridge"]["test_score"]["composite"],
            "r2": result["r2_discrete_switching"]["test_score"]["composite"],
        }
        strongest_name = min(baselines, key=baselines.get)
        r3 = result["r3_linear_event_state"]["median_test_score"]["composite"]
        rows.append(
            {
                "subject": result["subject"],
                "n_test_windows": result["n_prediction_windows"]["test"],
                "minimum_mode_occupancy": result["minimum_partition_mode_occupancy"],
                "template_rank_spearman": result["template_rank_spearman"],
                "r0_composite": baselines["r0"],
                "r1_composite": baselines["r1"],
                "r2_composite": baselines["r2"],
                "strongest_baseline": strongest_name,
                "strongest_baseline_composite": baselines[strongest_name],
                "r3_composite": r3,
                "r3_minus_strongest": r3 - baselines[strongest_name],
                "r3_shuffle_composite": result["r3_within_history_shuffle"]["median_test_score"]["composite"],
                "r3_circular_composite": result["r3_circular_pairing_null"]["median_test_score"]["composite"],
                **result["gates"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    state = {
        "contract": config["contract"],
        "status": "R0_R3_COMPLETE",
        "n_patients": int(len(frame)),
        "n_c0_pass": int(frame["c0_engineering"].sum()),
        "n_c1_pass": int(frame["c1_stable_repertoire_readback"].sum()),
        "n_r3_beats_strongest": int(frame["r3_beats_strongest_baseline"].sum()),
        "n_r3_beats_shuffle": int(frame["r3_beats_shuffle"].sum()),
        "n_r3_beats_circular": int(frame["r3_beats_circular"].sum()),
        "median_r3_minus_strongest": float(frame["r3_minus_strongest"].median()),
        "r4_authorized": bool(frame["c0_engineering"].all() and frame["c1_stable_repertoire_readback"].all()),
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "source_sha256": file_sha256(Path(__file__)),
        "old_heldout20_entered": False,
        "forbidden_labels_entered": False,
        "geometry_soz_ictal_snn_entered": False,
    }
    with (output / "R0_R3_STATE.json").open("w") as stream:
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
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for subject in subjects:
        print(f"[v2.3 R0-R3] {subject}", flush=True)
        results.append(run_patient(subject, config, output))
    aggregate(results, config, output, config_path)


if __name__ == "__main__":
    main()
