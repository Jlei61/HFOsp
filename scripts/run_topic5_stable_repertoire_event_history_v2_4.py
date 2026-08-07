#!/usr/bin/env python3
"""Run the frozen v2.4 event-history ladder on development or extension patients."""
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
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_stable_repertoire_event_rnn import (  # noqa: E402
    chronological_source_partition,
    fit_stable_templates,
    train_to_partition_template_stability,
)
from src.topic5_stable_repertoire_event_history_v2_4 import (  # noqa: E402
    EventHistoryDataset,
    FamilyScales,
    build_event_history_dataset,
    chronological_sequences,
    dataset_time_audit,
    family_scales_from_train,
    feature_matrix,
    fit_array_ridge_predict,
    fit_feature_ridge,
    fit_low_dimensional_state,
    fit_matched_recency_baselines,
    random_equal_count_features,
    refit_feature_ridge,
    safe_circular_target_pairing,
    score_v24,
    source_coherent_block_shuffle,
    split_half_reliability_v24,
    verify_event_history_contract,
    verify_target_values,
)


DEFAULT_CONFIG = ROOT / "config/topic5_stable_repertoire_event_history_v2_4.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).view(np.uint8)).hexdigest()


def score_dict(score) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(score).items()}


def _median_score(scores: list[dict[str, float]]) -> dict[str, float]:
    return {
        field: float(np.median([score[field] for score in scores]))
        for field in scores[0]
    }


def _static_prediction(
    train: EventHistoryDataset,
    validation: EventHistoryDataset,
    test: EventHistoryDataset,
    n_modes: int,
) -> np.ndarray:
    mean = np.mean(np.concatenate([train.targets, validation.targets]), axis=0)
    prediction = np.repeat(mean[None, :], len(test), axis=0)
    prediction[:, :n_modes] /= np.sum(prediction[:, :n_modes], axis=1, keepdims=True)
    return prediction


def _fit_named_feature(
    datasets: dict[str, EventHistoryDataset],
    *,
    name: str,
    rank_prior: np.ndarray,
    alpha_grid,
    n_modes: int,
    n_contacts: int,
    scales: FamilyScales,
) -> tuple[dict[str, Any], np.ndarray]:
    selected = fit_feature_ridge(
        datasets["train"],
        datasets["validation"],
        feature_name=name,
        rank_prior=rank_prior,
        decay=None,
        alpha_grid=alpha_grid,
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
    )
    final = refit_feature_ridge(datasets["train"], datasets["validation"], selected)
    prediction = final.predict(datasets["test"])
    score = score_v24(
        datasets["test"].targets,
        prediction,
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
    )
    return {
        "feature_name": name,
        "alpha": float(selected.alpha),
        "validation_score": score_dict(selected.validation_score),
        "test_score": score_dict(score),
    }, prediction


def _fit_random_equal_count(
    datasets: dict[str, EventHistoryDataset],
    *,
    rank_prior: np.ndarray,
    n_modes: int,
    n_contacts: int,
    alpha_grid,
    seeds,
    scales: FamilyScales,
) -> tuple[dict[str, Any], np.ndarray]:
    horizon = datasets["test"].target_event_indices.shape[1]
    runs = []
    predictions = []
    for seed in map(int, seeds):
        features = {
            split: random_equal_count_features(
                dataset,
                rank_prior,
                n_modes,
                count=horizon,
                seed=seed + 10_000 * index,
            )
            for index, (split, dataset) in enumerate(datasets.items())
        }
        prediction, alpha, validation_score = fit_array_ridge_predict(
            features["train"],
            features["validation"],
            features["test"],
            datasets["train"].targets,
            datasets["validation"].targets,
            alpha_grid=alpha_grid,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        score = score_v24(
            datasets["test"].targets,
            prediction,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        runs.append(
            {
                "seed": seed,
                "alpha": alpha,
                "validation_score": score_dict(validation_score),
                "test_score": score_dict(score),
            }
        )
        predictions.append(prediction)
    return {
        "runs": runs,
        "median_test_score": _median_score([run["test_score"] for run in runs]),
    }, np.mean(predictions, axis=0)


def fit_primary_models(
    datasets: dict[str, EventHistoryDataset],
    config: dict[str, Any],
    rank_prior: np.ndarray,
    scales: FamilyScales,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_modes = int(config["n_modes"])
    n_contacts = len(rank_prior)
    candidates, selected = fit_matched_recency_baselines(
        datasets["train"],
        datasets["validation"],
        rank_prior=rank_prior,
        decay_grid=config["decay_grid"],
        alpha_grid=config["ridge_alpha_grid"],
        n_modes=n_modes,
        n_contacts=n_contacts,
        scales=scales,
    )
    candidate_results = {}
    candidate_predictions = {}
    for name, model in candidates.items():
        final = refit_feature_ridge(datasets["train"], datasets["validation"], model)
        prediction = final.predict(datasets["test"])
        score = score_v24(
            datasets["test"].targets,
            prediction,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        candidate_results[name] = {
            "decay": None if model.decay is None else float(model.decay),
            "alpha": float(model.alpha),
            "validation_score": score_dict(model.validation_score),
            "test_score": score_dict(score),
        }
        candidate_predictions[name] = prediction
    state_runs = []
    state_predictions = []
    for seed in map(int, config["state_seeds"]):
        state = fit_low_dimensional_state(
            datasets["train"],
            datasets["validation"],
            base_model=selected,
            dimension_grid=config["dimension_grid"],
            decay_grid=config["decay_grid"],
            alpha_grid=config["ridge_alpha_grid"],
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
            seed=seed,
        )
        prediction = state.predict(datasets["test"])
        score = score_v24(
            datasets["test"].targets,
            prediction,
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        state_runs.append(
            {
                "seed": seed,
                "base_feature": selected.feature_name,
                "base_decay": selected.decay,
                "dimension": int(state.dimension),
                "decay": float(state.decay),
                "alpha": float(state.alpha),
                "validation_score": score_dict(state.validation_score),
                "test_score": score_dict(score),
            }
        )
        state_predictions.append(prediction)
    selected_name = selected.feature_name
    selected_score = candidate_results[selected_name]["test_score"]
    state_score = _median_score([run["test_score"] for run in state_runs])
    result = {
        "matched_recency_candidates": candidate_results,
        "validation_selected_matched_baseline": selected_name,
        "selected_matched_test_score": selected_score,
        "low_dimensional_state": {
            "runs": state_runs,
            "median_test_score": state_score,
        },
        "state_minus_matched": {
            field: float(state_score[field] - selected_score[field])
            for field in state_score
        },
        "matched_minus_state_gain": {
            field: float(selected_score[field] - state_score[field])
            for field in state_score
        },
    }
    predictions = {
        **candidate_predictions,
        "selected_matched": candidate_predictions[selected_name],
        "low_dimensional_state": np.mean(state_predictions, axis=0),
    }
    return result, predictions


def fit_full_ladder(
    datasets: dict[str, EventHistoryDataset],
    config: dict[str, Any],
    rank_prior: np.ndarray,
    scales: FamilyScales,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    n_modes = int(config["n_modes"])
    n_contacts = len(rank_prior)
    result = {}
    predictions = {}
    static = _static_prediction(
        datasets["train"], datasets["validation"], datasets["test"], n_modes
    )
    result["b0_static"] = {
        "test_score": score_dict(
            score_v24(
                datasets["test"].targets,
                static,
                n_modes=n_modes,
                n_contacts=n_contacts,
                scales=scales,
            )
        )
    }
    predictions["b0_static"] = static
    for name, key in (
        ("recent_h", "b1_last_h"),
        ("unordered_l", "b2_unordered_l"),
        ("first_h", "b3_first_h"),
        ("time_nuisance", "b6_time_iei_nuisance"),
    ):
        info, prediction = _fit_named_feature(
            datasets,
            name=name,
            rank_prior=rank_prior,
            alpha_grid=config["ridge_alpha_grid"],
            n_modes=n_modes,
            n_contacts=n_contacts,
            scales=scales,
        )
        result[key] = info
        predictions[key] = prediction
    random_info, random_prediction = _fit_random_equal_count(
        datasets,
        rank_prior=rank_prior,
        n_modes=n_modes,
        n_contacts=n_contacts,
        alpha_grid=config["ridge_alpha_grid"],
        seeds=config["equal_count_random_seeds"],
        scales=scales,
    )
    result["b3_random_h"] = random_info
    predictions["b3_random_h"] = random_prediction
    primary, primary_predictions = fit_primary_models(
        datasets, config, rank_prior, scales
    )
    result.update(primary)
    predictions.update(primary_predictions)
    return result, predictions


def _build_datasets(
    tokens,
    modes,
    rank,
    participation,
    event_time,
    encoder,
    split_sequences,
    config,
    surrogate_kind,
) -> dict[str, EventHistoryDataset]:
    return {
        split: build_event_history_dataset(
            tokens,
            modes,
            rank,
            participation,
            event_time,
            encoder,
            sequences,
            history_length=int(config["history_length"]),
            horizon=int(config["horizon"]),
            surrogate_kind=surrogate_kind,
        )
        for split, sequences in split_sequences.items()
    }


def _minimum_windows_pass(datasets, config) -> bool:
    return all(
        len(datasets[split]) >= int(config["minimum_windows"][split])
        for split in ("train", "validation", "test")
    )


def _contract_audit(
    datasets,
    *,
    source,
    event_time,
    eligible,
    horizon,
    require_future,
    modes,
    rank,
    participation,
    encoder,
):
    output = {}
    for split, dataset in datasets.items():
        checks = verify_event_history_contract(
            dataset,
            raw_source_ids=source,
            raw_event_time=event_time,
            eligible_indices=eligible,
            horizon=horizon,
            require_future=require_future,
        )
        checks["target_values_match_shifted_indices"] = verify_target_values(
            dataset, modes, rank, participation, encoder
        )
        output[split] = checks
    return output


def _all_checks_pass(audit) -> bool:
    return all(value for split in audit.values() for value in split.values())


def _null_index_hashes(datasets):
    return {
        split: {
            "history_event_indices_sha256": array_sha256(dataset.history_event_indices),
            "target_event_indices_sha256": array_sha256(dataset.target_event_indices),
            "origin_rows_sha256": array_sha256(dataset.origin_rows),
            "donor_rows_sha256": array_sha256(dataset.donor_rows),
            "n_windows": len(dataset),
        }
        for split, dataset in datasets.items()
    }


def _source_time_contract(split_sequences, event_time):
    times = np.asarray(event_time, float)
    all_sequences = [sequence for split in split_sequences.values() for sequence in split.values()]
    order = sorted(all_sequences, key=lambda sequence: float(np.min(times[sequence])))
    gaps = [float(np.min(times[right]) - np.max(times[left])) for left, right in zip(order[:-1], order[1:])]
    return {
        "source_semantics": "canonical_source_record_block_cross_record_continuity_unverified",
        "state_reset_at_source_boundary": True,
        "n_source_records": len(order),
        "cross_source_gap_seconds": {
            "minimum": float(np.min(gaps)) if gaps else None,
            "median": float(np.median(gaps)) if gaps else None,
            "maximum": float(np.max(gaps)) if gaps else None,
        },
    }


def run_patient(subject: str, config: dict[str, Any], output: Path) -> dict[str, Any]:
    data_path = ROOT / config["dataset_root"] / f"{subject}.npz"
    map_path = ROOT / config["source_mapping_root"] / f"{subject}.npz"
    raw = np.load(data_path, allow_pickle=False)
    mapping = np.load(map_path, allow_pickle=False)
    rank = np.asarray(raw["event_local_rank"], float)
    participation = np.asarray(raw["event_participation"], bool)
    event_time = np.asarray(raw["event_abs_time"], float)
    event_split = np.asarray(raw["event_split"], int)
    source = np.asarray(mapping["event_source_block_id"], int)
    eligible = np.flatnonzero(event_split == 0)
    partition = chronological_source_partition(
        source, event_time, eligible, fractions=config["source_split"]
    )
    split_indices = {
        split: partition.indices(source, split, eligible)
        for split in ("train", "validation", "test")
    }
    encoder = fit_stable_templates(
        rank,
        participation,
        split_indices["train"],
        n_modes=int(config["n_modes"]),
        seed=0,
    )
    tokens, modes = encoder.event_tokens(rank, participation)
    split_sequences = {
        split: chronological_sequences(source, event_time, indices)
        for split, indices in split_indices.items()
    }
    datasets = _build_datasets(
        tokens,
        modes,
        rank,
        participation,
        event_time,
        encoder,
        split_sequences,
        config,
        "true_chronology",
    )
    if not _minimum_windows_pass(datasets, config):
        raise ValueError(
            f"{subject}: insufficient prediction windows "
            + str({split: len(dataset) for split, dataset in datasets.items()})
        )
    true_audit = _contract_audit(
        datasets,
        source=source,
        event_time=event_time,
        eligible=eligible,
        horizon=int(config["horizon"]),
        require_future=True,
        modes=modes,
        rank=rank,
        participation=participation,
        encoder=encoder,
    )
    if not _all_checks_pass(true_audit):
        raise RuntimeError(f"{subject}: true chronology contract failed")
    scales = family_scales_from_train(
        datasets["train"].targets,
        n_modes=int(config["n_modes"]),
        n_contacts=rank.shape[1],
    )
    ladder, predictions = fit_full_ladder(datasets, config, encoder.rank_prior, scales)
    true_gain = ladder["matched_minus_state_gain"]

    block_results = []
    block_metadata = {}
    for seed in map(int, config["null_block_seeds"]):
        shuffled_sequences = {}
        permutations = {}
        for split, sequences in split_sequences.items():
            shuffled_sequences[split], permutations[split] = source_coherent_block_shuffle(
                sequences, block_size=int(config["null_block_size"]), seed=seed
            )
        null_datasets = _build_datasets(
            tokens,
            modes,
            rank,
            participation,
            event_time,
            encoder,
            shuffled_sequences,
            config,
            f"source_block_shuffle_seed_{seed}",
        )
        audit = _contract_audit(
            null_datasets,
            source=source,
            event_time=event_time,
            eligible=eligible,
            horizon=int(config["horizon"]),
            require_future=True,
            modes=modes,
            rank=rank,
            participation=participation,
            encoder=encoder,
        )
        for split in audit:
            modeled_sources = np.unique(datasets[split].source_ids)
            audit[split]["pseudo_sequence_differs_for_all_modeled_sources"] = bool(
                all(
                    not np.array_equal(
                        shuffled_sequences[split][source_id],
                        split_sequences[split][source_id],
                    )
                    for source_id in modeled_sources
                )
            )
        if not _all_checks_pass(audit):
            raise RuntimeError(f"{subject}: block-null contract failed for seed {seed}")
        fitted, _ = fit_primary_models(null_datasets, config, encoder.rank_prior, scales)
        block_results.append(
            {
                "seed": seed,
                "selected_matched": fitted["validation_selected_matched_baseline"],
                "matched_test_score": fitted["selected_matched_test_score"],
                "state_test_score": fitted["low_dimensional_state"]["median_test_score"],
                "matched_minus_state_gain": fitted["matched_minus_state_gain"],
                "contract_checks": audit,
                "index_hashes": _null_index_hashes(null_datasets),
            }
        )
        block_metadata[str(seed)] = permutations

    circular_results = []
    circular_index_arrays = {}
    for fraction_index, fraction in enumerate(config["safe_circular_shift_fractions"]):
        try:
            circular_datasets = {}
            shifts = {}
            for split, dataset in datasets.items():
                circular_datasets[split], shifts[split] = safe_circular_target_pairing(
                    dataset,
                    shift_fraction=float(fraction),
                    horizon=int(config["horizon"]),
                )
            if not _minimum_windows_pass(circular_datasets, config):
                raise ValueError("safe pairing left too few windows")
            audit = _contract_audit(
                circular_datasets,
                source=source,
                event_time=event_time,
                eligible=eligible,
                horizon=int(config["horizon"]),
                require_future=False,
                modes=modes,
                rank=rank,
                participation=participation,
                encoder=encoder,
            )
            if not _all_checks_pass(audit):
                raise RuntimeError("safe circular contract failed")
            fitted, _ = fit_primary_models(
                circular_datasets, config, encoder.rank_prior, scales
            )
            circular_results.append(
                {
                    "fraction": float(fraction),
                    "shifts": shifts,
                    "selected_matched": fitted["validation_selected_matched_baseline"],
                    "matched_test_score": fitted["selected_matched_test_score"],
                    "state_test_score": fitted["low_dimensional_state"]["median_test_score"],
                    "matched_minus_state_gain": fitted["matched_minus_state_gain"],
                    "contract_checks": audit,
                    "index_hashes": _null_index_hashes(circular_datasets),
                }
            )
            prefix = f"fraction_{fraction_index}"
            for split, dataset in circular_datasets.items():
                circular_index_arrays[f"{prefix}_{split}_history"] = dataset.history_event_indices
                circular_index_arrays[f"{prefix}_{split}_target"] = dataset.target_event_indices
                circular_index_arrays[f"{prefix}_{split}_origin_row"] = dataset.origin_rows
                circular_index_arrays[f"{prefix}_{split}_donor_row"] = dataset.donor_rows
        except ValueError as error:
            circular_results.append(
                {"fraction": float(fraction), "status": "INSUFFICIENT_SAFE_WINDOWS", "reason": str(error)}
            )
    valid_circular = [result for result in circular_results if "matched_minus_state_gain" in result]
    if not valid_circular:
        raise RuntimeError(f"{subject}: no valid safe circular replicate")

    reliability = split_half_reliability_v24(
        datasets["validation"],
        modes,
        rank,
        participation,
        encoder,
        train_target_mean=np.mean(datasets["train"].targets, axis=0),
        repeats=int(config["reliability_repeats"]),
        seed=17,
    )
    template_stability = {
        split: train_to_partition_template_stability(
            rank,
            participation,
            split_indices["train"],
            split_indices[split],
            encoder,
            seed=0,
        )
        for split in ("validation", "test")
    }
    block_gain = float(
        np.median([result["matched_minus_state_gain"]["propagation"] for result in block_results])
    )
    circular_gain = float(
        np.median([result["matched_minus_state_gain"]["propagation"] for result in valid_circular])
    )
    result = {
        "contract": config["contract"],
        "subject": subject,
        "dataset": subject.split("_", 1)[0],
        "n_contacts": int(rank.shape[1]),
        "n_events_train80": int(len(eligible)),
        "n_prediction_windows": {split: len(dataset) for split, dataset in datasets.items()},
        "family_scales_train_only": asdict(scales),
        "true_chronology": ladder,
        "coherent_block_shuffle": block_results,
        "safe_circular_pairing": circular_results,
        "chronology_gain": {
            "true_matched_minus_state_propagation": float(true_gain["propagation"]),
            "block_null_median_matched_minus_state_propagation": block_gain,
            "circular_null_median_matched_minus_state_propagation": circular_gain,
            "true_minus_block_null_gain": float(true_gain["propagation"] - block_gain),
            "true_minus_circular_null_gain": float(true_gain["propagation"] - circular_gain),
        },
        "validation_future_window_reliability": reliability,
        "train_to_partition_template_stability": template_stability,
        "time_scale_audit": {
            split: dataset_time_audit(dataset) for split, dataset in datasets.items()
        },
        "source_time_contract": _source_time_contract(split_sequences, event_time),
        "contract_checks": {
            "true": true_audit,
            "partition_sources_disjoint": bool(
                set(partition.train_sources).isdisjoint(partition.validation_sources)
                and set(partition.train_sources).isdisjoint(partition.test_sources)
                and set(partition.validation_sources).isdisjoint(partition.test_sources)
            ),
            "all_final_indices_train80_only": bool(
                all(np.all(event_split[indices] == 0) for indices in split_indices.values())
            ),
            "minimum_windows_pass": _minimum_windows_pass(datasets, config),
            "all_block_nulls_pass": all(
                _all_checks_pass(result["contract_checks"]) for result in block_results
            ),
            "all_valid_circular_nulls_pass": all(
                _all_checks_pass(result["contract_checks"]) for result in valid_circular
            ),
        },
        "provenance": {
            "dataset_path": str(data_path.resolve()),
            "dataset_sha256": sha256(data_path),
            "source_mapping_path": str(map_path.resolve()),
            "source_mapping_sha256": sha256(map_path),
            "eligible_index_sha256": array_sha256(np.asarray(eligible, np.int64)),
            "template_train_index_sha256": array_sha256(
                np.asarray(split_indices["train"], np.int64)
            ),
            "template_centers_sha256": array_sha256(
                np.asarray(encoder.centers, np.float64)
            ),
            "old_heldout20_entered": False,
            "forbidden_labels_entered": False,
            "geometry_soz_ictal_snn_entered": False,
        },
    }
    patient_root = output / "per_subject"
    patient_root.mkdir(parents=True, exist_ok=True)
    with (patient_root / f"{subject}.json").open("w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    prediction_arrays = {
        "target": datasets["test"].targets.astype(np.float32),
        "test_source": datasets["test"].source_ids,
        "history_event_indices": datasets["test"].history_event_indices,
        "target_event_indices": datasets["test"].target_event_indices,
        **{name: values.astype(np.float32) for name, values in predictions.items()},
    }
    np.savez_compressed(patient_root / f"{subject}_predictions.npz", **prediction_arrays)
    np.savez_compressed(
        patient_root / f"{subject}_safe_circular_indices.npz", **circular_index_arrays
    )
    with (patient_root / f"{subject}_block_permutations.json").open("w") as stream:
        json.dump(block_metadata, stream, indent=2, sort_keys=True)
    return result


def aggregate(results: list[dict[str, Any]], output: Path, config_path: Path, cohort: str):
    rows = []
    for result in results:
        true = result["true_chronology"]
        row = {
            "subject": result["subject"],
            "dataset": result["dataset"],
            "n_test_windows": result["n_prediction_windows"]["test"],
            "selected_matched_baseline": true["validation_selected_matched_baseline"],
            "matched_propagation": true["selected_matched_test_score"]["propagation"],
            "state_propagation": true["low_dimensional_state"]["median_test_score"]["propagation"],
            "state_minus_matched_propagation": true["state_minus_matched"]["propagation"],
            "state_minus_matched_recruitment": true["state_minus_matched"]["recruitment"],
            "state_minus_matched_repertoire": true["state_minus_matched"]["repertoire"],
            **result["chronology_gain"],
            "validation_dynamic_occupancy_reliability": result["validation_future_window_reliability"]["occupancy"]["train_mean_residualized"]["variance_reliability_median"],
            "validation_dynamic_rank_reliability": result["validation_future_window_reliability"]["rank"]["train_mean_residualized"]["variance_reliability_median"],
            "validation_dynamic_participation_reliability": result["validation_future_window_reliability"]["participation"]["train_mean_residualized"]["variance_reliability_median"],
            "validation_template_grade": result["train_to_partition_template_stability"]["validation"]["grade"],
            "test_template_grade": result["train_to_partition_template_stability"]["test"]["grade"],
        }
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    config = yaml.safe_load(config_path.open())
    state = {
        "contract": config["contract"],
        "cohort": cohort,
        "status": "COMPLETE",
        "n_requested": len(
            config["development_subjects"]
            if cohort == "development"
            else config["locked_extension_subjects"]
        ),
        "n_completed": int(len(frame)),
        "n_state_beats_matched_propagation": int(
            np.sum(frame["state_minus_matched_propagation"] < 0)
        ),
        "n_true_gain_beats_block_null": int(
            np.sum(frame["true_minus_block_null_gain"] > 0)
        ),
        "n_true_gain_beats_circular_null": int(
            np.sum(frame["true_minus_circular_null_gain"] > 0)
        ),
        "median_state_minus_matched_propagation": float(
            frame["state_minus_matched_propagation"].median()
        ),
        "median_true_minus_block_null_gain": float(
            frame["true_minus_block_null_gain"].median()
        ),
        "median_true_minus_circular_null_gain": float(
            frame["true_minus_circular_null_gain"].median()
        ),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "spec_sha256": sha256(
            ROOT / "docs/superpowers/specs/2026-08-02-topic5-stable-repertoire-event-history-v2_4.md"
        ),
        "module_sha256": sha256(
            ROOT / "src/topic5_stable_repertoire_event_history_v2_4.py"
        ),
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
        "forbidden_labels_entered": False,
        "geometry_soz_ictal_snn_entered": False,
    }
    with (output / "STATE.json").open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cohort", choices=("development", "extension"), default="development")
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.open())
    frozen = (
        config["development_subjects"]
        if args.cohort == "development"
        else config["locked_extension_subjects"]
    )
    subjects = args.subjects or frozen
    if args.cohort == "extension" and args.subjects:
        unknown = sorted(set(args.subjects) - set(config["locked_extension_subjects"]))
        if unknown:
            raise ValueError(f"subjects outside locked extension: {unknown}")
    output = ROOT / config["output_root"] / args.cohort
    output.mkdir(parents=True, exist_ok=True)
    results = []
    failures = []
    for subject in subjects:
        print(f"[v2.4 {args.cohort}] {subject}", flush=True)
        try:
            results.append(run_patient(subject, config, output))
        except (ValueError, RuntimeError) as error:
            failures.append({"subject": subject, "error": str(error)})
            if args.cohort == "development":
                raise
    with (output / "failures.json").open("w") as stream:
        json.dump(failures, stream, indent=2, sort_keys=True)
    aggregate(results, output, config_path, args.cohort)


if __name__ == "__main__":
    main()
