#!/usr/bin/env python3
"""Screen and run the trainable event-level RNN v2.5 on all 34 patients."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_stable_repertoire_event_history_v2_4 import (  # noqa: E402
    EventHistoryDataset,
    chronological_sequences,
    family_scales_from_train,
    score_v24,
)
from src.topic5_stable_repertoire_event_rnn import fit_stable_templates  # noqa: E402
from src.topic5_trainable_event_rnn_v2_5 import (  # noqa: E402
    RecurrentProfile,
    fit_fixed_feature_baseline,
    fit_trainable_residual_rnn,
    partition_indices,
    profile_from_mapping,
    trace_to_dict,
    window_balanced_source_partition,
)


DEFAULT_CONFIG = ROOT / "config/topic5_trainable_event_rnn_v2_5.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def score_dict(score):
    return {key: float(value) for key, value in asdict(score).items()}


def _training_stride(sequences: dict[object, np.ndarray], length: int, horizon: int, target: int) -> int:
    dense = sum(max(0, len(sequence) - int(length) - int(horizon) + 1) for sequence in sequences.values())
    if dense <= 0:
        raise ValueError("training sources contain no recurrent windows")
    return max(1, min(int(horizon), int(np.ceil(dense / max(int(target), 1)))))


def _cap_dataset(dataset: EventHistoryDataset, maximum: int) -> EventHistoryDataset:
    if len(dataset) <= int(maximum):
        return dataset
    rows = np.unique(np.linspace(0, len(dataset) - 1, int(maximum)).round().astype(int))
    return dataset.take(rows)


def _build_dataset(tokens, modes, rank, participation, event_time, encoder, sequences, config, *, train):
    from src.topic5_stable_repertoire_event_history_v2_4 import build_event_history_dataset

    if not sequences:
        return None, None
    stride = (
        _training_stride(
            sequences,
            int(config["history_length"]),
            int(config["horizon"]),
            int(config["training_target_windows"]),
        )
        if train
        else int(config["horizon"])
    )
    dataset = build_event_history_dataset(
        tokens,
        modes,
        rank,
        participation,
        event_time,
        encoder,
        sequences,
        history_length=int(config["history_length"]),
        horizon=int(config["horizon"]),
        stride=stride,
    )
    if train:
        dataset = _cap_dataset(dataset, int(config["maximum_training_windows"]))
    return dataset, stride


def load_subject(subject: str, config: dict[str, Any]):
    data_path = ROOT / config["dataset_root"] / f"{subject}.npz"
    map_path = ROOT / config["source_mapping_root"] / f"{subject}.npz"
    raw = np.load(data_path, allow_pickle=False)
    mapping = np.load(map_path, allow_pickle=False)
    values = {
        "rank": np.asarray(raw["event_local_rank"], float),
        "participation": np.asarray(raw["event_participation"], bool),
        "event_time": np.asarray(raw["event_abs_time"], float),
        "event_split": np.asarray(raw["event_split"], int),
        "source": np.asarray(mapping["event_source_block_id"]),
        "data_path": data_path,
        "map_path": map_path,
    }
    values["eligible"] = np.flatnonzero(values["event_split"] == 0)
    return values


def prepare_subject(subject: str, config: dict[str, Any], *, final_fit: bool):
    raw = load_subject(subject, config)
    partition = window_balanced_source_partition(
        raw["source"],
        raw["event_time"],
        raw["eligible"],
        history_length=int(config["history_length"]),
        horizon=int(config["horizon"]),
    )
    indices = {
        split: partition_indices(raw["source"], raw["eligible"], partition, split)
        for split in ("train", "validation", "test")
    }
    inner_validation = False
    if not final_fit and len(indices["validation"]) == 0:
        train_order = indices["train"][
            np.argsort(raw["event_time"][indices["train"]], kind="mergesort")
        ]
        cut = int(np.floor(0.7 * len(train_order)))
        minimum = int(config["history_length"]) + int(config["horizon"])
        if cut < minimum or len(train_order) - cut < minimum:
            raise ValueError("two-source inner validation lacks history+horizon events")
        indices["train"] = train_order[:cut]
        indices["validation"] = train_order[cut:]
        inner_validation = True
    fit_splits = ["train"] + (["validation"] if final_fit and len(indices["validation"]) else [])
    encoder_indices = np.concatenate([indices[name] for name in fit_splits])
    encoder = fit_stable_templates(
        raw["rank"],
        raw["participation"],
        encoder_indices,
        n_modes=int(config["n_modes"]),
        seed=0,
    )
    tokens, modes = encoder.event_tokens(raw["rank"], raw["participation"])
    sequence_by_split = {
        split: chronological_sequences(
            raw["source"], raw["event_time"], indices[split]
        )
        for split in indices
    }
    if final_fit:
        fitting_sequences = dict(sequence_by_split["train"])
        fitting_sequences.update(sequence_by_split["validation"])
        train, train_stride = _build_dataset(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            raw["event_time"],
            encoder,
            fitting_sequences,
            config,
            train=True,
        )
        test, test_stride = _build_dataset(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            raw["event_time"],
            encoder,
            sequence_by_split["test"],
            config,
            train=False,
        )
        datasets = {"train": train, "validation": None, "test": test}
        strides = {"train": train_stride, "validation": None, "test": test_stride}
    else:
        train, train_stride = _build_dataset(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            raw["event_time"],
            encoder,
            sequence_by_split["train"],
            config,
            train=True,
        )
        validation = validation_stride = None
        if sequence_by_split["validation"]:
            validation, validation_stride = _build_dataset(
                tokens,
                modes,
                raw["rank"],
                raw["participation"],
                raw["event_time"],
                encoder,
                sequence_by_split["validation"],
                config,
                train=False,
            )
        test, test_stride = _build_dataset(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            raw["event_time"],
            encoder,
            sequence_by_split["test"],
            config,
            train=False,
        )
        datasets = {"train": train, "validation": validation, "test": test}
        strides = {"train": train_stride, "validation": validation_stride, "test": test_stride}
    all_used = [
        dataset.history_event_indices.ravel()
        for dataset in datasets.values()
        if dataset is not None
    ] + [
        dataset.target_event_indices.ravel()
        for dataset in datasets.values()
        if dataset is not None
    ]
    used = np.concatenate(all_used)
    audit = {
        "all_indices_train80_only": bool(np.all(raw["event_split"][used] == 0)),
        "history_strictly_before_target": bool(
            all(
                np.all(dataset.history_stop <= dataset.target_start)
                for dataset in datasets.values()
                if dataset is not None
            )
        ),
        "formal_test_targets_nonoverlap": bool(
            all(
                np.intersect1d(
                    datasets["test"].target_event_indices[left],
                    datasets["test"].target_event_indices[right],
                ).size == 0
                for source_id in np.unique(datasets["test"].source_ids)
                for left, right in zip(
                    np.flatnonzero(datasets["test"].source_ids == source_id)[:-1],
                    np.flatnonzero(datasets["test"].source_ids == source_id)[1:],
                )
            )
        ),
        "split_sources_disjoint_or_two_source_inner_validation": bool(
            (
                set(partition.train_sources).isdisjoint(partition.validation_sources)
                and set(partition.train_sources).isdisjoint(partition.test_sources)
                and set(partition.validation_sources).isdisjoint(partition.test_sources)
            )
            if not inner_validation
            else (
                np.intersect1d(indices["train"], indices["validation"]).size == 0
                and np.intersect1d(
                    np.concatenate([indices["train"], indices["validation"]]),
                    indices["test"],
                ).size == 0
            )
        ),
        "two_source_inner_validation_contract": bool(
            (not inner_validation)
            or (
                np.max(raw["event_time"][indices["train"]])
                < np.min(raw["event_time"][indices["validation"]])
                and np.max(raw["event_time"][indices["validation"]])
                < np.min(raw["event_time"][indices["test"]])
            )
        ),
    }
    if not all(audit.values()):
        raise RuntimeError(f"{subject}: data contract failed: {audit}")
    return raw, encoder, datasets, partition, strides, audit


def baseline_candidates(config):
    for profile in config["baseline_profiles"]:
        for alpha in config["ridge_alpha_grid"]:
            yield {
                "feature_name": profile["feature_name"],
                "decay": profile.get("decay"),
                "alpha": float(alpha),
            }


def baseline_key(profile):
    return f"{profile['feature_name']}|{profile.get('decay')}|{profile['alpha']}"


def fit_baseline(dataset, profile, encoder, n_modes):
    return fit_fixed_feature_baseline(
        dataset,
        feature_name=profile["feature_name"],
        decay=profile.get("decay"),
        alpha=float(profile["alpha"]),
        rank_prior=encoder.rank_prior,
        n_modes=n_modes,
    )


def merge_profile(base: dict[str, Any], update: dict[str, Any]) -> tuple[str, RecurrentProfile]:
    values = dict(base)
    name = str(update.get("name", "profile"))
    values.update({key: value for key, value in update.items() if key != "name"})
    return name, profile_from_mapping(values)


def screen(config: dict[str, Any], config_path: Path, output: Path):
    screen_root = output / "development_screen"
    screen_root.mkdir(parents=True, exist_ok=True)
    subjects = list(config["development_subjects"])
    baseline_rows = []
    for subject in subjects:
        print(f"[v2.5 baseline screen] {subject}", flush=True)
        _, encoder, datasets, partition, strides, audit = prepare_subject(subject, config, final_fit=False)
        if datasets["validation"] is None:
            raise RuntimeError(f"development subject {subject} has no validation sources")
        scales = family_scales_from_train(
            datasets["train"].targets,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
        )
        for candidate in baseline_candidates(config):
            model = fit_baseline(datasets["train"], candidate, encoder, int(config["n_modes"]))
            score = score_v24(
                datasets["validation"].targets,
                model.predict(datasets["validation"]),
                n_modes=int(config["n_modes"]),
                n_contacts=len(encoder.rank_prior),
                scales=scales,
            )
            baseline_rows.append({"subject": subject, **candidate, **score_dict(score)})
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_frame.to_csv(screen_root / "baseline_screen.csv", index=False)
    baseline_summary = (
        baseline_frame.groupby(["feature_name", "decay", "alpha"], dropna=False)["propagation"]
        .median()
        .reset_index()
        .sort_values(["propagation", "alpha", "feature_name"])
    )
    baseline_summary.to_csv(screen_root / "baseline_screen_summary.csv", index=False)
    best_row = baseline_summary.iloc[0]
    selected_baseline = {
        "feature_name": str(best_row["feature_name"]),
        "decay": None if pd.isna(best_row["decay"]) else float(best_row["decay"]),
        "alpha": float(best_row["alpha"]),
    }

    training_rows = []
    base_arch = dict(config["screen_base_architecture"])
    for subject in subjects:
        print(f"[v2.5 training screen] {subject}", flush=True)
        _, encoder, datasets, _, _, _ = prepare_subject(subject, config, final_fit=False)
        scales = family_scales_from_train(
            datasets["train"].targets,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
        )
        baseline = fit_baseline(datasets["train"], selected_baseline, encoder, int(config["n_modes"]))
        baseline_score = score_v24(
            datasets["validation"].targets,
            baseline.predict(datasets["validation"]),
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        for values in config["training_profiles"]:
            name, profile = merge_profile(base_arch, values)
            started = time.time()
            fitted = fit_trainable_residual_rnn(
                datasets["train"],
                baseline=baseline,
                profile=profile,
                scales=scales,
                n_modes=int(config["n_modes"]),
                n_contacts=len(encoder.rank_prior),
                seed=int(config["screen_seed"]),
                maximum_epochs=int(config["maximum_epochs"]),
                patience=int(config["patience"]),
                minimum_epochs=int(config["minimum_epochs"]),
                validation=datasets["validation"],
            )
            score = fitted.validation_score
            training_rows.append({
                "subject": subject,
                "profile": name,
                "baseline_propagation": baseline_score.propagation,
                "rnn_propagation": score.propagation,
                "gain": baseline_score.propagation - score.propagation,
                "best_epoch": fitted.trace.best_epoch,
                "stopped_epoch": fitted.trace.stopped_epoch,
                "n_parameters": fitted.n_parameters,
                "finite": fitted.trace.finite,
                "runtime_seconds": time.time() - started,
                **asdict(profile),
            })
    training_frame = pd.DataFrame(training_rows)
    training_frame.to_csv(screen_root / "training_profile_screen.csv", index=False)
    training_summary = (
        training_frame.groupby("profile")
        .agg(median_gain=("gain", "median"), median_score=("rnn_propagation", "median"), median_epoch=("best_epoch", "median"), all_finite=("finite", "all"))
        .reset_index()
        .sort_values(["median_gain", "median_score", "profile"], ascending=[False, True, True])
    )
    training_summary.to_csv(screen_root / "training_profile_summary.csv", index=False)
    selected_training_name = str(training_summary.iloc[0]["profile"])
    selected_training_values = next(item for item in config["training_profiles"] if item["name"] == selected_training_name)

    architecture_rows = []
    training_only = {key: value for key, value in selected_training_values.items() if key != "name"}
    for subject in subjects:
        print(f"[v2.5 architecture screen] {subject}", flush=True)
        _, encoder, datasets, _, _, _ = prepare_subject(subject, config, final_fit=False)
        scales = family_scales_from_train(
            datasets["train"].targets,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
        )
        baseline = fit_baseline(datasets["train"], selected_baseline, encoder, int(config["n_modes"]))
        baseline_score = score_v24(
            datasets["validation"].targets,
            baseline.predict(datasets["validation"]),
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        for architecture in config["architecture_profiles"]:
            name, profile = merge_profile(training_only, architecture)
            started = time.time()
            fitted = fit_trainable_residual_rnn(
                datasets["train"],
                baseline=baseline,
                profile=profile,
                scales=scales,
                n_modes=int(config["n_modes"]),
                n_contacts=len(encoder.rank_prior),
                seed=int(config["screen_seed"]),
                maximum_epochs=int(config["maximum_epochs"]),
                patience=int(config["patience"]),
                minimum_epochs=int(config["minimum_epochs"]),
                validation=datasets["validation"],
            )
            score = fitted.validation_score
            architecture_rows.append({
                "subject": subject,
                "profile": name,
                "baseline_propagation": baseline_score.propagation,
                "rnn_propagation": score.propagation,
                "gain": baseline_score.propagation - score.propagation,
                "best_epoch": fitted.trace.best_epoch,
                "stopped_epoch": fitted.trace.stopped_epoch,
                "n_parameters": fitted.n_parameters,
                "finite": fitted.trace.finite,
                "runtime_seconds": time.time() - started,
                **asdict(profile),
            })
    architecture_frame = pd.DataFrame(architecture_rows)
    architecture_frame.to_csv(screen_root / "architecture_screen.csv", index=False)
    architecture_summary = (
        architecture_frame.groupby("profile")
        .agg(median_gain=("gain", "median"), median_score=("rnn_propagation", "median"), median_epoch=("best_epoch", "median"), median_parameters=("n_parameters", "median"), all_finite=("finite", "all"))
        .reset_index()
        .sort_values(
            ["median_gain", "median_score", "median_parameters", "profile"],
            ascending=[False, True, True, True],
        )
    )
    architecture_summary.to_csv(screen_root / "architecture_summary.csv", index=False)
    selected_architecture_name = str(architecture_summary.iloc[0]["profile"])
    selected_architecture = next(item for item in config["architecture_profiles"] if item["name"] == selected_architecture_name)
    _, selected_profile = merge_profile(training_only, selected_architecture)

    joint_rows = []
    selected_profile_values = asdict(selected_profile)
    for subject in subjects:
        print(f"[v2.5 joint refinement] {subject}", flush=True)
        _, encoder, datasets, _, _, _ = prepare_subject(subject, config, final_fit=False)
        scales = family_scales_from_train(
            datasets["train"].targets,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
        )
        baseline = fit_baseline(
            datasets["train"], selected_baseline, encoder, int(config["n_modes"])
        )
        baseline_score = score_v24(
            datasets["validation"].targets,
            baseline.predict(datasets["validation"]),
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        for refinement in config["joint_refinement_profiles"]:
            name, profile = merge_profile(selected_profile_values, refinement)
            started = time.time()
            fitted = fit_trainable_residual_rnn(
                datasets["train"],
                baseline=baseline,
                profile=profile,
                scales=scales,
                n_modes=int(config["n_modes"]),
                n_contacts=len(encoder.rank_prior),
                seed=int(config["screen_seed"]),
                maximum_epochs=int(config["maximum_epochs"]),
                patience=int(config["patience"]),
                minimum_epochs=int(config["minimum_epochs"]),
                validation=datasets["validation"],
            )
            score = fitted.validation_score
            joint_rows.append({
                "subject": subject,
                "profile": name,
                "baseline_propagation": baseline_score.propagation,
                "rnn_propagation": score.propagation,
                "gain": baseline_score.propagation - score.propagation,
                "best_epoch": fitted.trace.best_epoch,
                "stopped_epoch": fitted.trace.stopped_epoch,
                "n_parameters": fitted.n_parameters,
                "finite": fitted.trace.finite,
                "runtime_seconds": time.time() - started,
                **asdict(profile),
            })
    joint_frame = pd.DataFrame(joint_rows)
    joint_frame.to_csv(screen_root / "joint_refinement_screen.csv", index=False)
    joint_summary = (
        joint_frame.groupby("profile")
        .agg(
            median_gain=("gain", "median"),
            median_score=("rnn_propagation", "median"),
            median_epoch=("best_epoch", "median"),
            median_parameters=("n_parameters", "median"),
            all_finite=("finite", "all"),
        )
        .reset_index()
        .sort_values(
            ["median_gain", "median_score", "median_parameters", "profile"],
            ascending=[False, True, True, True],
        )
    )
    joint_summary.to_csv(screen_root / "joint_refinement_summary.csv", index=False)
    selected_joint_name = str(joint_summary.iloc[0]["profile"])
    selected_joint = next(
        item for item in config["joint_refinement_profiles"]
        if item["name"] == selected_joint_name
    )
    _, selected_profile = merge_profile(selected_profile_values, selected_joint)
    selected_runs = joint_frame[joint_frame["profile"] == selected_joint_name]
    median_best_epoch_plus_one = int(
        max(1, round(float(selected_runs["best_epoch"].median())) + 1)
    )
    frozen = {
        "contract": config["contract"],
        "status": "FROZEN_AFTER_SIX_PATIENT_VALIDATION_SCREEN",
        "development_subjects": subjects,
        "selected_baseline": selected_baseline,
        "selected_training_profile": selected_training_name,
        "selected_architecture_profile": selected_architecture_name,
        "selected_joint_refinement_profile": selected_joint_name,
        "recurrent_profile": asdict(selected_profile),
        "development_median_best_epoch_plus_one_diagnostic": median_best_epoch_plus_one,
        "cohort_checkpoint_rule": "best_patient_validation_propagation_no_refit",
        "screen_seed": int(config["screen_seed"]),
        "median_development_validation_gain": float(selected_runs["gain"].median()),
        "n_positive_development_validation_gain": int(np.sum(selected_runs["gain"] > 0)),
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_trainable_event_rnn_v2_5.py"),
        "runner_sha256": sha256(Path(__file__)),
        "execution_environment": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_num_threads": int(config["torch_num_threads"]),
        },
        "old_heldout20_entered": False,
    }
    with (screen_root / "FROZEN_PROFILE.json").open("w") as stream:
        json.dump(frozen, stream, indent=2, sort_keys=True)
    print(json.dumps(frozen, indent=2, sort_keys=True))
    return frozen


def support_grade(n_train: int, n_test: int) -> str:
    if n_train >= 100 and n_test >= 12:
        return "high"
    if n_train >= 20 and n_test >= 4:
        return "moderate"
    return "low"


def save_checkpoint(path: Path, fitted, subject: str, seed: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": "topic5_trainable_event_rnn_v2_5",
            "subject": subject,
            "seed": int(seed),
            "model_state_dict": fitted.model.state_dict(),
            "feature_mean": fitted.feature_mean,
            "feature_scale": fitted.feature_scale,
            "profile": asdict(fitted.profile),
            "trace": trace_to_dict(fitted.trace),
            "baseline_feature_name": fitted.baseline.feature_name,
            "baseline_decay": fitted.baseline.decay,
            "baseline_alpha": fitted.baseline.alpha,
        },
        path,
    )


def run_subject(subject: str, config: dict[str, Any], frozen: dict[str, Any], output: Path):
    raw, encoder, datasets, partition, strides, audit = prepare_subject(subject, config, final_fit=False)
    scales = family_scales_from_train(
        datasets["train"].targets,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    baseline = fit_baseline(
        datasets["train"], frozen["selected_baseline"], encoder, int(config["n_modes"])
    )
    baseline_prediction = baseline.predict(datasets["test"])
    baseline_score = score_v24(
        datasets["test"].targets,
        baseline_prediction,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        scales=scales,
    )
    profile = profile_from_mapping(frozen["recurrent_profile"])
    runs = []
    predictions = []
    for seed in map(int, config["final_seeds"]):
        started = time.time()
        fitted = fit_trainable_residual_rnn(
            datasets["train"],
            baseline=baseline,
            profile=profile,
            scales=scales,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            seed=seed,
            maximum_epochs=int(config["maximum_epochs"]),
            patience=int(config["patience"]),
            minimum_epochs=int(config["minimum_epochs"]),
            validation=datasets["validation"],
        )
        prediction = fitted.predict(datasets["test"])
        score = score_v24(
            datasets["test"].targets,
            prediction,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        runs.append({
            "seed": seed,
            "score": score_dict(score),
            "trace": trace_to_dict(fitted.trace),
            "n_parameters": fitted.n_parameters,
            "runtime_seconds": time.time() - started,
        })
        predictions.append(prediction)
        save_checkpoint(output / "checkpoints" / subject / f"seed_{seed}.pt", fitted, subject, seed)
    median_score = {
        field: float(np.median([run["score"][field] for run in runs]))
        for field in runs[0]["score"]
    }
    result = {
        "contract": config["contract"],
        "subject": subject,
        "dataset": subject.split("_", 1)[0],
        "development_subject": subject in config["development_subjects"],
        "n_contacts": int(raw["rank"].shape[1]),
        "n_events_train80": int(len(raw["eligible"])),
        "partition": {
            **jsonable(asdict(partition)),
            "train_sources": partition.train_sources.tolist(),
            "validation_sources": partition.validation_sources.tolist(),
            "test_sources": partition.test_sources.tolist(),
        },
        "n_windows": {
            "train_dense": len(datasets["train"]),
            "test_formal": len(datasets["test"]),
        },
        "strides": strides,
        "support_grade": support_grade(len(datasets["train"]), len(datasets["test"])),
        "selected_baseline": frozen["selected_baseline"],
        "baseline_test_score": score_dict(baseline_score),
        "recurrent_profile": frozen["recurrent_profile"],
        "checkpoint_rule": "best_patient_validation_propagation_no_refit",
        "recurrent_runs": runs,
        "recurrent_median_test_score": median_score,
        "rnn_minus_baseline": {
            field: float(median_score[field] - score_dict(baseline_score)[field])
            for field in median_score
        },
        "contract_checks": audit,
        "provenance": {
            "dataset_path": str(raw["data_path"].resolve()),
            "dataset_sha256": sha256(raw["data_path"]),
            "source_mapping_path": str(raw["map_path"].resolve()),
            "source_mapping_sha256": sha256(raw["map_path"]),
            "eligible_indices_sha256": array_sha256(raw["eligible"].astype(np.int64)),
            "test_history_indices_sha256": array_sha256(datasets["test"].history_event_indices.astype(np.int64)),
            "test_target_indices_sha256": array_sha256(datasets["test"].target_event_indices.astype(np.int64)),
            "old_heldout20_entered": False,
            "forbidden_labels_entered": False,
        },
    }
    subject_root = output / "per_subject"
    subject_root.mkdir(parents=True, exist_ok=True)
    with (subject_root / f"{subject}.json").open("w") as stream:
        json.dump(jsonable(result), stream, indent=2, sort_keys=True)
    np.savez_compressed(
        subject_root / f"{subject}_predictions.npz",
        target=datasets["test"].targets.astype(np.float32),
        baseline=baseline_prediction.astype(np.float32),
        recurrent=np.mean(predictions, axis=0).astype(np.float32),
        history_event_indices=datasets["test"].history_event_indices.astype(np.int64),
        target_event_indices=datasets["test"].target_event_indices.astype(np.int64),
    )
    return result


def patient_inference(values: np.ndarray):
    finite = np.asarray(values, float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {}
    rng = np.random.default_rng(20260802)
    boot = np.median(rng.choice(finite, (10000, len(finite)), replace=True), axis=1)
    nonzero = finite[finite != 0]
    try:
        signed = wilcoxon(finite, alternative="less", zero_method="wilcox")
        wp = float(signed.pvalue)
    except ValueError:
        wp = float("nan")
    better = int(np.sum(finite < 0))
    return {
        "n": int(len(finite)),
        "median_rnn_minus_baseline": float(np.median(finite)),
        "bootstrap_median_ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        "n_rnn_better": better,
        "wilcoxon_one_sided_less_p": wp,
        "sign_test_one_sided_p": float(binomtest(better, len(finite), 0.5, alternative="greater").pvalue),
    }


def aggregate(results, failures, config, config_path, frozen, output):
    rows = []
    for result in results:
        rows.append({
            "subject": result["subject"],
            "dataset": result["dataset"],
            "development_subject": result["development_subject"],
            "support_grade": result["support_grade"],
            "split_strategy": result["partition"]["strategy"],
            "n_events_train80": result["n_events_train80"],
            "n_train_windows": result["n_windows"]["train_dense"],
            "n_test_windows": result["n_windows"]["test_formal"],
            "baseline_propagation": result["baseline_test_score"]["propagation"],
            "rnn_propagation": result["recurrent_median_test_score"]["propagation"],
            "rnn_minus_baseline_propagation": result["rnn_minus_baseline"]["propagation"],
            "rnn_minus_baseline_recruitment": result["rnn_minus_baseline"]["recruitment"],
            "rnn_minus_baseline_repertoire": result["rnn_minus_baseline"]["repertoire"],
            "all_runs_finite": all(run["trace"]["finite"] for run in result["recurrent_runs"]),
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    failure_frame = pd.DataFrame(failures)
    failure_frame.to_csv(output / "denominator_failures.csv", index=False)
    extension = frame[~frame["development_subject"]]
    state = {
        "contract": config["contract"],
        "status": "REPAIRED_TRUE_CHRONOLOGY_34_PATIENT_RUN_COMPLETE" if len(frame) == 34 else "INCOMPLETE_DENOMINATOR",
        "n_subjects_attempted": 34,
        "n_subjects_completed": int(len(frame)),
        "n_subjects_failed": int(len(failures)),
        "n_development": int(frame["development_subject"].sum()) if len(frame) else 0,
        "n_extension": int((~frame["development_subject"]).sum()) if len(frame) else 0,
        "support_grade_counts": frame["support_grade"].value_counts().to_dict() if len(frame) else {},
        "all34_descriptive_propagation": patient_inference(frame["rnn_minus_baseline_propagation"].to_numpy()) if len(frame) else {},
        "extension_primary_propagation": patient_inference(extension["rnn_minus_baseline_propagation"].to_numpy()) if len(extension) else {},
        "extension_secondary_recruitment": patient_inference(extension["rnn_minus_baseline_recruitment"].to_numpy()) if len(extension) else {},
        "frozen_profile": frozen,
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_trainable_event_rnn_v2_5.py"),
        "runner_sha256": sha256(Path(__file__)),
        "execution_environment": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_num_threads": int(config["torch_num_threads"]),
        },
        "old_heldout20_entered": False,
    }
    with (output / "TRUE_CHRONOLOGY_STATE.json").open("w") as stream:
        json.dump(jsonable(state), stream, indent=2, sort_keys=True)
    print(json.dumps(jsonable(state), indent=2, sort_keys=True))
    return state


def cohort(config, config_path, output, frozen_path, subjects=None):
    frozen = json.load(frozen_path.open())
    expected = {
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_trainable_event_rnn_v2_5.py"),
        "runner_sha256": sha256(Path(__file__)),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"frozen profile hash mismatch for {key}")
    all_subjects = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    selected = all_subjects if not subjects else list(subjects)
    results = []
    failures = []
    output.mkdir(parents=True, exist_ok=True)
    for index, subject in enumerate(selected, 1):
        print(f"[v2.5 cohort {index}/{len(selected)}] {subject}", flush=True)
        try:
            results.append(run_subject(subject, config, frozen, output))
        except Exception as error:
            failures.append({"subject": subject, "error_type": type(error).__name__, "reason": str(error)})
            print(f"[v2.5 failure] {subject}: {type(error).__name__}: {error}", flush=True)
    return aggregate(results, failures, config, config_path, frozen, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase", choices=("screen", "cohort", "all"), default="all")
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.open())
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["torch_num_threads"]))
    frozen_path = output / "development_screen/FROZEN_PROFILE.json"
    if args.phase in {"screen", "all"}:
        screen(config, config_path, output)
    if args.phase in {"cohort", "all"}:
        cohort(config, config_path, output, frozen_path, subjects=args.subjects)


if __name__ == "__main__":
    main()
