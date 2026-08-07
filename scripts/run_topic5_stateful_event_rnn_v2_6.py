#!/usr/bin/env python3
"""Screen and execute the continuous stateful event-sequence RNN v2.6."""
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
    chronological_sequences,
    score_v24,
)
from src.topic5_stable_repertoire_event_rnn import (  # noqa: E402
    fit_stable_templates,
    project_descriptor,
)
from src.topic5_trainable_event_rnn_v2_5 import (  # noqa: E402
    partition_indices,
    window_balanced_source_partition,
)
from src.topic5_stateful_event_rnn_v2_6 import (  # noqa: E402
    StatefulProfile,
    build_stateful_sequences,
    family_scales_from_sequences,
    fit_continuous_ewma_ridge,
    fit_stateful_event_rnn,
    mean_future_descriptor,
    profile_from_mapping,
    trace_to_dict,
)


DEFAULT_CONFIG = ROOT / "config/topic5_stateful_event_rnn_v2_6.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(str(values.shape).encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def score_dict(score):
    return {key: float(value) for key, value in asdict(score).items()}


def load_subject(subject: str, config: dict[str, Any]):
    data_path = ROOT / config["dataset_root"] / f"{subject}.npz"
    mapping_path = ROOT / config["source_mapping_root"] / f"{subject}.npz"
    data = np.load(data_path, allow_pickle=False)
    mapping = np.load(mapping_path, allow_pickle=False)
    raw = {
        "rank": np.asarray(data["event_local_rank"], float),
        "participation": np.asarray(data["event_participation"], bool),
        "event_time": np.asarray(data["event_abs_time"], float),
        "event_split": np.asarray(data["event_split"], int),
        "source": np.asarray(mapping["event_source_block_id"]),
        "data_path": data_path,
        "mapping_path": mapping_path,
    }
    raw["eligible"] = np.flatnonzero(raw["event_split"] == 0)
    return raw


def _formal_targets_nonoverlap(sequences):
    for sequence in sequences:
        rows = np.flatnonzero(sequence.formal_mask)
        for left, right in zip(rows[:-1], rows[1:]):
            if np.intersect1d(
                sequence.target_event_indices[left],
                sequence.target_event_indices[right],
            ).size:
                return False
    return True


def prepare_subject(subject: str, config: dict[str, Any]):
    raw = load_subject(subject, config)
    partition = window_balanced_source_partition(
        raw["source"],
        raw["event_time"],
        raw["eligible"],
        history_length=int(config["warmup_events"]),
        horizon=int(config["horizon"]),
    )
    indices = {
        split: partition_indices(raw["source"], raw["eligible"], partition, split)
        for split in ("train", "validation", "test")
    }
    inner_validation = False
    if len(indices["validation"]) == 0:
        ordered = indices["train"][
            np.argsort(raw["event_time"][indices["train"]], kind="mergesort")
        ]
        cut = int(np.floor(0.7 * len(ordered)))
        minimum = int(config["warmup_events"]) + int(config["horizon"])
        if cut < minimum or len(ordered) - cut < minimum:
            raise ValueError("two-source patient lacks inner train/validation sequences")
        indices["train"] = ordered[:cut]
        indices["validation"] = ordered[cut:]
        inner_validation = True
    encoder = fit_stable_templates(
        raw["rank"],
        raw["participation"],
        indices["train"],
        n_modes=int(config["n_modes"]),
        seed=0,
    )
    tokens, modes = encoder.event_tokens(raw["rank"], raw["participation"])
    sequence_indices = {
        split: chronological_sequences(
            raw["source"], raw["event_time"], indices[split]
        )
        for split in indices
    }
    datasets = {
        split: build_stateful_sequences(
            tokens,
            modes,
            raw["rank"],
            raw["participation"],
            encoder,
            sequence_indices[split],
            horizon=int(config["horizon"]),
            warmup_events=int(config["warmup_events"]),
        )
        for split in sequence_indices
    }
    used = np.concatenate(
        [item.event_indices for split in datasets.values() for item in split]
        + [
            item.target_event_indices[item.formal_mask].ravel()
            for split in datasets.values()
            for item in split
        ]
    )
    used = used[used >= 0]
    audit = {
        "all_indices_train80_only": bool(np.all(raw["event_split"][used] == 0)),
        "formal_validation_targets_nonoverlap": _formal_targets_nonoverlap(
            datasets["validation"]
        ),
        "formal_test_targets_nonoverlap": _formal_targets_nonoverlap(datasets["test"]),
        "source_event_time_monotonic": bool(
            all(
                np.all(np.diff(raw["event_time"][item.event_indices]) >= 0)
                for split in datasets.values()
                for item in split
            )
        ),
        "state_reset_only_at_source_or_split_boundary": True,
        "split_event_indices_disjoint": bool(
            np.intersect1d(indices["train"], indices["validation"]).size == 0
            and np.intersect1d(indices["train"], indices["test"]).size == 0
            and np.intersect1d(indices["validation"], indices["test"]).size == 0
        ),
        "two_source_inner_validation_chronological": bool(
            (not inner_validation)
            or (
                np.max(raw["event_time"][indices["train"]])
                < np.min(raw["event_time"][indices["validation"]])
                < np.min(raw["event_time"][indices["test"]])
            )
        ),
    }
    if not all(audit.values()):
        raise RuntimeError(f"{subject}: stateful data contract failed: {audit}")
    return raw, encoder, datasets, partition, indices, audit


def merge_profile(base: dict[str, Any], update: dict[str, Any]):
    values = dict(base)
    name = str(update.get("name", "profile"))
    values.update({key: value for key, value in update.items() if key != "name"})
    return name, profile_from_mapping(values)


def fit_profile(subject, profile, datasets, encoder, config, scales, seed):
    started = time.time()
    fitted = fit_stateful_event_rnn(
        datasets["train"],
        datasets["validation"],
        profile=profile,
        scales=scales,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        seed=int(seed),
        maximum_epochs=int(config["maximum_epochs"]),
        minimum_epochs=int(config["minimum_epochs"]),
        patience=int(config["patience"]),
        carry_state=True,
    )
    return fitted, time.time() - started


def screen_subject(subject: str, config: dict[str, Any], output: Path):
    raw, encoder, datasets, partition, indices, audit = prepare_subject(subject, config)
    scales = family_scales_from_sequences(
        datasets["train"],
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    architecture_rows = []
    for values in config["architecture_profiles"]:
        name, profile = merge_profile(config["base_profile"], values)
        fitted, runtime = fit_profile(
            subject, profile, datasets, encoder, config, scales, config["screen_seed"]
        )
        architecture_rows.append(
            {
                "stage": "architecture",
                "profile": name,
                "trained_validation_propagation": fitted.trained_validation_score.propagation,
                "nested_validation_propagation": fitted.nested_validation_score.propagation,
                "best_trained_epoch": fitted.trace.best_trained_epoch,
                "best_nested_epoch": fitted.trace.best_nested_epoch,
                "n_parameters": fitted.n_parameters,
                "runtime_seconds": runtime,
                **asdict(profile),
            }
        )
    architecture_frame = pd.DataFrame(architecture_rows).sort_values(
        ["trained_validation_propagation", "n_parameters", "profile"]
    )
    best_architecture = architecture_frame.iloc[0]
    selected_architecture = {
        key: best_architecture[key] for key in StatefulProfile.__dataclass_fields__
    }

    refinement_rows = []
    for values in config["training_refinements"]:
        name, profile = merge_profile(selected_architecture, values)
        fitted, runtime = fit_profile(
            subject, profile, datasets, encoder, config, scales, config["screen_seed"]
        )
        refinement_rows.append(
            {
                "stage": "refinement",
                "profile": name,
                "trained_validation_propagation": fitted.trained_validation_score.propagation,
                "nested_validation_propagation": fitted.nested_validation_score.propagation,
                "best_trained_epoch": fitted.trace.best_trained_epoch,
                "best_nested_epoch": fitted.trace.best_nested_epoch,
                "n_parameters": fitted.n_parameters,
                "runtime_seconds": runtime,
                **asdict(profile),
            }
        )
    refinement_frame = pd.DataFrame(refinement_rows).sort_values(
        ["trained_validation_propagation", "n_parameters", "profile"]
    )
    best = refinement_frame.iloc[0]
    selected_profile = {
        key: jsonable(best[key]) for key in StatefulProfile.__dataclass_fields__
    }
    record = {
        "contract": config["contract"],
        "subject": subject,
        "status": "PATIENT_VALIDATION_PROFILE_FROZEN",
        "n_contacts": int(raw["rank"].shape[1]),
        "n_events_train80": int(len(raw["eligible"])),
        "partition_strategy": partition.strategy,
        "n_sequences": {key: len(value) for key, value in datasets.items()},
        "n_dense_targets": {
            key: int(sum(np.sum(item.valid_mask) for item in value))
            for key, value in datasets.items()
        },
        "n_formal_targets": {
            key: int(sum(np.sum(item.formal_mask) for item in value))
            for key, value in datasets.items()
        },
        "selected_architecture": str(best_architecture["profile"]),
        "selected_refinement": str(best["profile"]),
        "selected_profile": selected_profile,
        "selected_training_budget": {
            "maximum_epochs": int(config["maximum_epochs"]),
            "minimum_epochs": int(config["minimum_epochs"]),
            "patience": int(config["patience"]),
        },
        "selected_validation_propagation": float(
            best["trained_validation_propagation"]
        ),
        "architecture_screen": architecture_frame.to_dict("records"),
        "refinement_screen": refinement_frame.to_dict("records"),
        "contract_checks": audit,
        "provenance": {
            "dataset_sha256": sha256(raw["data_path"]),
            "source_mapping_sha256": sha256(raw["mapping_path"]),
            "eligible_indices_sha256": array_sha256(raw["eligible"].astype(np.int64)),
            "old_heldout20_entered": False,
        },
    }
    subject_root = output / "validation_screen/per_subject"
    subject_root.mkdir(parents=True, exist_ok=True)
    with (subject_root / f"{subject}.json").open("w") as stream:
        json.dump(jsonable(record), stream, indent=2, sort_keys=True)
    return record


def freeze_screen(config, config_path: Path, output: Path, subjects=None):
    available = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    selected = available if not subjects else list(subjects)
    records = []
    failures = []
    for subject in selected:
        path = output / "validation_screen/per_subject" / f"{subject}.json"
        if not path.exists():
            failures.append({"subject": subject, "error_type": "MissingArtifact", "reason": str(path)})
            continue
        record = json.load(path.open())
        boundary = record.get("epoch_boundary_audit")
        budget = record.get("selected_training_budget")
        if boundary is None or boundary.get("test_results_read") is not False:
            failures.append(
                {
                    "subject": subject,
                    "error_type": "BoundaryAuditMissing",
                    "reason": "validation-only epoch boundary audit is not complete",
                }
            )
            continue
        if not isinstance(budget, dict) or not {
            "maximum_epochs",
            "minimum_epochs",
            "patience",
        }.issubset(budget):
            failures.append(
                {
                    "subject": subject,
                    "error_type": "TrainingBudgetMissing",
                    "reason": "selected profile has no frozen training budget",
                }
            )
            continue
        if not all(record.get("contract_checks", {}).values()):
            failures.append(
                {
                    "subject": subject,
                    "error_type": "DataContractFailure",
                    "reason": "patient validation artifact contains a failed contract check",
                }
            )
            continue
        records.append(record)
    rows = [
        {
            "subject": item["subject"],
            "n_contacts": item["n_contacts"],
            "n_events_train80": item["n_events_train80"],
            "selected_architecture": item["selected_architecture"],
            "selected_refinement": item["selected_refinement"],
            "selected_cell": item["selected_profile"]["cell"],
            "selected_hidden_size": item["selected_profile"]["hidden_size"],
            "selected_tbptt_length": item["selected_profile"]["tbptt_length"],
            "selected_maximum_epochs": item.get(
                "selected_training_budget", {}
            ).get("maximum_epochs", int(config["maximum_epochs"])),
            "selected_validation_propagation": item["selected_validation_propagation"],
        }
        for item in records
    ]
    root = output / "validation_screen"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "patient_profile_summary.csv", index=False)
    pd.DataFrame(failures, columns=("subject", "error_type", "reason")).to_csv(
        root / "failures.csv", index=False
    )
    state = {
        "contract": config["contract"],
        "status": "ALL_PATIENT_VALIDATION_PROFILES_FROZEN" if len(records) == 34 else "INCOMPLETE",
        "n_attempted": len(selected),
        "n_completed": len(records),
        "n_failed": len(failures),
        "test_results_read_during_selection": False,
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_stateful_event_rnn_v2_6.py"),
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }
    with (root / "FROZEN_VALIDATION_STATE.json").open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))
    return state


def screen(config, config_path: Path, output: Path, subjects=None):
    available = sorted(path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz"))
    selected = available if not subjects else list(subjects)
    for index, subject in enumerate(selected, 1):
        print(f"[v2.6 screen {index}/{len(selected)}] {subject}", flush=True)
        screen_subject(subject, config, output)
    return freeze_screen(config, config_path, output, selected)


def _static_predictions(sequences, mean):
    targets = []
    predictions = []
    metadata = []
    for sequence in sequences:
        rows = np.flatnonzero(sequence.formal_mask)
        targets.append(sequence.targets[rows])
        predictions.append(np.tile(mean, (len(rows), 1)))
        metadata.extend(
            {
                "source_id": str(sequence.source_id),
                "event_index": int(sequence.event_indices[row]),
                "target_event_indices": sequence.target_event_indices[row].tolist(),
            }
            for row in rows
        )
    return np.concatenate(predictions), np.concatenate(targets), metadata


def save_checkpoint(path, fitted, encoder, ewma, subject, seed, training_budget):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "contract": "topic5_stateful_event_sequence_rnn_v2_6",
            "subject": subject,
            "seed": int(seed),
            "trained_model_state_dict": fitted.trained_model.state_dict(),
            "nested_model_state_dict": fitted.nested_model.state_dict(),
            "feature_mean": fitted.feature_mean,
            "feature_scale": fitted.feature_scale,
            "profile": asdict(fitted.profile),
            "training_budget": {
                key: int(value) for key, value in training_budget.items()
            },
            "trace": trace_to_dict(fitted.trace),
            "encoder": {
                "centers": encoder.centers,
                "feature_mean": encoder.feature_mean,
                "feature_scale": encoder.feature_scale,
                "rank_prior": encoder.rank_prior,
                "n_modes": encoder.n_modes,
            },
            "ewma": {
                "decay": ewma.decay,
                "alpha": ewma.alpha,
                "feature_mean": ewma.feature_mean,
                "feature_scale": ewma.feature_scale,
                "ridge_coef": ewma.ridge.coef_,
                "ridge_intercept": ewma.ridge.intercept_,
            },
        },
        path,
    )


def run_subject(subject, config, output):
    profile_record = json.load(
        (output / "validation_screen/per_subject" / f"{subject}.json").open()
    )
    raw, encoder, datasets, partition, indices, audit = prepare_subject(subject, config)
    scales = family_scales_from_sequences(
        datasets["train"],
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
    )
    ewma = fit_continuous_ewma_ridge(
        datasets["train"],
        decay=float(config["ewma_decay"]),
        alpha=float(config["ewma_alpha"]),
        n_modes=int(config["n_modes"]),
    )
    ewma_prediction, test_target, metadata = ewma.predict(
        datasets["test"], formal=True
    )
    ewma_score = score_v24(
        test_target,
        ewma_prediction,
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        scales=scales,
    )
    static_mean = mean_future_descriptor(datasets["train"])
    static_prediction, static_target, _ = _static_predictions(
        datasets["test"], static_mean
    )
    if not np.array_equal(static_target, test_target):
        raise RuntimeError("stateful test target mismatch between controls")
    static_score = score_v24(
        test_target,
        project_descriptor(static_prediction, int(config["n_modes"])),
        n_modes=int(config["n_modes"]),
        n_contacts=len(encoder.rank_prior),
        scales=scales,
    )
    profile = profile_from_mapping(profile_record["selected_profile"])
    training_config = dict(config)
    training_config.update(profile_record.get("selected_training_budget", {}))
    runs = []
    trained_predictions = []
    nested_predictions = []
    states = []
    for seed in map(int, config["final_seeds"]):
        fitted, runtime = fit_profile(
            subject, profile, datasets, encoder, training_config, scales, seed
        )
        trained_prediction, trained_target, _, hidden = fitted.predict(
            datasets["test"],
            checkpoint="trained",
            formal=True,
            return_states=True,
        )
        nested_prediction, nested_target, _ = fitted.predict(
            datasets["test"], checkpoint="nested", formal=True
        )
        if not (
            np.array_equal(trained_target, test_target)
            and np.array_equal(nested_target, test_target)
        ):
            raise RuntimeError("stateful recurrent target mismatch")
        trained_score = score_v24(
            test_target,
            trained_prediction,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        nested_score = score_v24(
            test_target,
            nested_prediction,
            n_modes=int(config["n_modes"]),
            n_contacts=len(encoder.rank_prior),
            scales=scales,
        )
        runs.append(
            {
                "seed": seed,
                "trained_test_score": score_dict(trained_score),
                "nested_test_score": score_dict(nested_score),
                "trained_validation_score": score_dict(
                    fitted.trained_validation_score
                ),
                "nested_validation_score": score_dict(
                    fitted.nested_validation_score
                ),
                "trace": trace_to_dict(fitted.trace),
                "n_parameters": fitted.n_parameters,
                "runtime_seconds": runtime,
                "test_state_norm_mean": float(
                    np.mean(np.linalg.norm(hidden, axis=1))
                ),
                "test_state_norm_max": float(
                    np.max(np.linalg.norm(hidden, axis=1))
                ),
            }
        )
        trained_predictions.append(trained_prediction)
        nested_predictions.append(nested_prediction)
        states.append(hidden)
        save_checkpoint(
            output / "checkpoints" / subject / f"seed_{seed}.pt",
            fitted,
            encoder,
            ewma,
            subject,
            seed,
            profile_record.get(
                "selected_training_budget",
                {
                    "maximum_epochs": config["maximum_epochs"],
                    "minimum_epochs": config["minimum_epochs"],
                    "patience": config["patience"],
                },
            ),
        )
    trained_median = {
        key: float(np.median([run["trained_test_score"][key] for run in runs]))
        for key in runs[0]["trained_test_score"]
    }
    nested_median = {
        key: float(np.median([run["nested_test_score"][key] for run in runs]))
        for key in runs[0]["nested_test_score"]
    }
    result = {
        "contract": config["contract"],
        "subject": subject,
        "dataset": subject.split("_", 1)[0],
        "n_contacts": int(raw["rank"].shape[1]),
        "n_events_train80": int(len(raw["eligible"])),
        "partition_strategy": partition.strategy,
        "selected_profile": profile_record["selected_profile"],
        "selected_training_budget": {
            key: int(value)
            for key, value in profile_record.get(
                "selected_training_budget",
                {
                    "maximum_epochs": config["maximum_epochs"],
                    "minimum_epochs": config["minimum_epochs"],
                    "patience": config["patience"],
                },
            ).items()
        },
        "selected_validation_propagation": profile_record[
            "selected_validation_propagation"
        ],
        "n_formal_test_targets": int(len(test_target)),
        "static_test_score": score_dict(static_score),
        "ewma_test_score": score_dict(ewma_score),
        "trained_recurrent_median_test_score": trained_median,
        "nested_recurrent_median_test_score": nested_median,
        "trained_rnn_minus_ewma": {
            key: trained_median[key] - score_dict(ewma_score)[key]
            for key in trained_median
        },
        "nested_rnn_minus_ewma": {
            key: nested_median[key] - score_dict(ewma_score)[key]
            for key in nested_median
        },
        "recurrent_runs": runs,
        "contract_checks": audit,
        "provenance": {
            "dataset_sha256": sha256(raw["data_path"]),
            "source_mapping_sha256": sha256(raw["mapping_path"]),
            "test_event_indices_sha256": array_sha256(
                np.asarray([item["event_index"] for item in metadata], np.int64)
            ),
            "test_target_indices_sha256": array_sha256(
                np.asarray([item["target_event_indices"] for item in metadata], np.int64)
            ),
            "old_heldout20_entered": False,
        },
    }
    subject_root = output / "per_subject"
    subject_root.mkdir(parents=True, exist_ok=True)
    with (subject_root / f"{subject}.json").open("w") as stream:
        json.dump(jsonable(result), stream, indent=2, sort_keys=True)
    np.savez_compressed(
        subject_root / f"{subject}_predictions.npz",
        target=test_target.astype(np.float32),
        static=static_prediction.astype(np.float32),
        ewma=ewma_prediction.astype(np.float32),
        trained_rnn=np.mean(trained_predictions, axis=0).astype(np.float32),
        nested_rnn=np.mean(nested_predictions, axis=0).astype(np.float32),
        hidden_state=np.mean(states, axis=0).astype(np.float32),
        event_indices=np.asarray([item["event_index"] for item in metadata], np.int64),
        target_event_indices=np.asarray(
            [item["target_event_indices"] for item in metadata], np.int64
        ),
    )
    return result


def patient_inference(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    favorable = int(np.sum(values < 0))
    nonzero = values[values != 0]
    rng = np.random.default_rng(20260802)
    boot = np.median(
        rng.choice(values, (10000, len(values)), replace=True), axis=1
    )
    try:
        wp = float(wilcoxon(values, alternative="less").pvalue)
    except ValueError:
        wp = None
    return {
        "n": int(len(values)),
        "median_rnn_minus_ewma": float(np.median(values)),
        "bootstrap_median_ci95": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "n_rnn_better": favorable,
        "wilcoxon_one_sided_less_p": wp,
        "n_non_ties": int(len(nonzero)),
        "tie_excluded_sign_p": (
            float(
                binomtest(
                    int(np.sum(nonzero < 0)),
                    len(nonzero),
                    0.5,
                    alternative="greater",
                ).pvalue
            )
            if len(nonzero)
            else None
        ),
    }


def aggregate(results, failures, config, config_path, output):
    rows = [
        {
            "subject": item["subject"],
            "dataset": item["dataset"],
            "n_events_train80": item["n_events_train80"],
            "n_contacts": item["n_contacts"],
            "n_formal_test_targets": item["n_formal_test_targets"],
            "cell": item["selected_profile"]["cell"],
            "hidden_size": item["selected_profile"]["hidden_size"],
            "tbptt_length": item["selected_profile"]["tbptt_length"],
            "optimizer": item["selected_profile"]["optimizer"],
            "learning_rate": item["selected_profile"]["learning_rate"],
            "trained_rnn_minus_ewma_propagation": item["trained_rnn_minus_ewma"][
                "propagation"
            ],
            "trained_rnn_minus_ewma_recruitment": item["trained_rnn_minus_ewma"][
                "recruitment"
            ],
            "nested_rnn_minus_ewma_propagation": item["nested_rnn_minus_ewma"][
                "propagation"
            ],
        }
        for item in results
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "patient_summary.csv", index=False)
    pd.DataFrame(failures, columns=("subject", "error_type", "reason")).to_csv(
        output / "failures.csv", index=False
    )
    state = {
        "contract": config["contract"],
        "status": "STATEFUL_34_PATIENT_TEST_COMPLETE" if len(results) == 34 else "INCOMPLETE",
        "n_attempted": 34,
        "n_completed": len(results),
        "n_failed": len(failures),
        "trained_primary_propagation": patient_inference(
            frame["trained_rnn_minus_ewma_propagation"]
        ) if len(frame) else {},
        "nested_secondary_propagation": patient_inference(
            frame["nested_rnn_minus_ewma_propagation"]
        ) if len(frame) else {},
        "trained_secondary_recruitment": patient_inference(
            frame["trained_rnn_minus_ewma_recruitment"]
        ) if len(frame) else {},
        "selected_cell_counts": frame["cell"].value_counts().to_dict() if len(frame) else {},
        "selected_tbptt_counts": frame["tbptt_length"].value_counts().to_dict() if len(frame) else {},
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_stateful_event_rnn_v2_6.py"),
        "runner_sha256": sha256(Path(__file__)),
        "old_heldout20_entered": False,
    }
    with (output / "STATEFUL_TEST_STATE.json").open("w") as stream:
        json.dump(jsonable(state), stream, indent=2, sort_keys=True)
    print(json.dumps(jsonable(state), indent=2, sort_keys=True))
    return state


def cohort(config, config_path, output, subjects=None):
    frozen = json.load((output / "validation_screen/FROZEN_VALIDATION_STATE.json").open())
    expected = {
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(ROOT / "src/topic5_stateful_event_rnn_v2_6.py"),
        "runner_sha256": sha256(Path(__file__)),
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise RuntimeError(f"v2.6 frozen validation hash mismatch: {key}")
    available = sorted(
        path.stem for path in (ROOT / config["dataset_root"]).glob("*.npz")
    )
    selected = available if not subjects else list(subjects)
    results = []
    failures = []
    for index, subject in enumerate(selected, 1):
        print(f"[v2.6 test {index}/{len(selected)}] {subject}", flush=True)
        try:
            results.append(run_subject(subject, config, output))
        except Exception as error:
            failures.append(
                {"subject": subject, "error_type": type(error).__name__, "reason": str(error)}
            )
            print(f"[v2.6 test failure] {subject}: {error}", flush=True)
    return aggregate(results, failures, config, config_path, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--phase",
        choices=("screen", "screen-patients", "freeze-screen", "cohort", "all"),
        default="all",
    )
    parser.add_argument("--subjects", nargs="*")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.open())
    output = ROOT / config["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(int(config["torch_num_threads"]))
    if args.phase in {"screen", "all"}:
        screen(config, config_path, output, args.subjects)
    if args.phase == "screen-patients":
        if not args.subjects:
            raise ValueError("screen-patients requires --subjects")
        for subject in args.subjects:
            print(f"[v2.6 screen patient] {subject}", flush=True)
            screen_subject(subject, config, output)
    if args.phase == "freeze-screen":
        freeze_screen(config, config_path, output, args.subjects)
    if args.phase in {"cohort", "all"}:
        cohort(config, config_path, output, args.subjects)


if __name__ == "__main__":
    main()
