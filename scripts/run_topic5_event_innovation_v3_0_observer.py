#!/usr/bin/env python3
"""Select the future-blind observer and cross-fit rank innovations on train/validation.

No human test target is read.  Every model step is one complete event.  The
output is an innovation-validity handoff, not an impulse or shaping result.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_event_innovation_v3_0_phase1_measurement import (  # noqa: E402
    _prepare,
    train_contact_backbone,
    unit_balanced_dense_fields,
)
from scripts.audit_topic5_event_innovation_v3_0_phase0 import sha256  # noqa: E402
from src.topic5_event_innovation_data import (  # noqa: E402
    ContinuitySequence,
    build_blocked_chronological_crossfit_folds,
    build_single_event_anchor_splits,
    resolve_single_event_anchor,
)
from src.topic5_event_innovation_observer_v3_0 import (  # noqa: E402
    blocked_innovation_validity,
    concatenate_feature_ladder,
    fit_standardized_masked_observer,
    masked_rank_mse,
)
from src.topic5_event_innovation_v3_0 import (  # noqa: E402
    RankStateBasis,
    fit_rank_state_basis,
    rolling_past_rank_fields,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"
INTERVALS = {
    "pre20": (0, 20),
    "pre40": (0, 40),
    "pre80": (0, 80),
    "lag0_20": (0, 20),
    "lag20_40": (20, 40),
    "lag40_60": (40, 60),
    "lag60_80": (60, 80),
}
LADDER_HISTORY = {
    "pre20": 20,
    "pre20_40_80": 80,
    "four_lag_bins": 80,
    "four_lag_bins_plus_time": 80,
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def sequence_metadata(
    sequences: Sequence[ContinuitySequence],
    n_events: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return sequence code, position and finite time nuisances per event."""

    group = np.full(n_events, -1, dtype=np.int32)
    position = np.full(n_events, -1, dtype=np.int32)
    nuisance = np.zeros((n_events, 3), dtype=np.float32)
    for sequence_index, sequence in enumerate(sequences):
        indices = np.asarray(sequence.event_indices, dtype=np.int64)
        times = np.asarray(sequence.event_times, dtype=float)
        group[indices] = sequence_index
        position[indices] = np.arange(len(indices), dtype=np.int32)
        nuisance[indices, 0] = np.linspace(0.0, 1.0, len(indices), dtype=np.float32)
        if len(indices) > 1:
            gap = np.maximum(np.diff(times), 0.0)
            nuisance[indices[1:], 1] = np.log1p(gap).astype(np.float32)
        if len(indices) > 20:
            duration = np.maximum(times[20:] - times[:-20], 1e-6)
            nuisance[indices[20:], 2] = np.log1p(20.0 / duration).astype(np.float32)
    return group, position, nuisance


def history_fields(
    ranks: np.ndarray,
    participation: np.ndarray,
    sequences: Sequence[ContinuitySequence],
) -> dict[str, np.ndarray]:
    indices = [np.asarray(sequence.event_indices, dtype=np.int64) for sequence in sequences]
    return {
        name: rolling_past_rank_fields(
            ranks,
            participation,
            indices,
            start_offset=start,
            stop_offset=stop,
        )[0]
        for name, (start, stop) in INTERVALS.items()
    }


def projected_ladder(
    basis: RankStateBasis,
    fields: Mapping[str, np.ndarray],
    nuisance: np.ndarray,
) -> dict[str, np.ndarray]:
    projected = {name: basis.transform(values) for name, values in fields.items()}
    return concatenate_feature_ladder(projected, nuisance)


def observable_ladder(
    basis: RankStateBasis,
    fields: Mapping[str, np.ndarray],
    nuisance: np.ndarray,
) -> dict[str, np.ndarray]:
    filled = {
        name: np.where(np.isfinite(values), values, basis.backbone[None, :])
        for name, values in fields.items()
    }
    return concatenate_feature_ladder(filled, nuisance)


def balanced_row_weights(indices: np.ndarray, groups: np.ndarray) -> np.ndarray:
    selected_groups = groups[np.asarray(indices, dtype=np.int64)]
    if np.any(selected_groups < 0):
        raise ValueError("selected event lacks a continuity-unit code")
    weight = np.zeros(len(indices), dtype=float)
    for group in np.unique(selected_groups):
        mask = selected_groups == group
        weight[mask] = 1.0 / np.sum(mask)
    return weight


def formal_innovation_events(anchors, sequences: Sequence[ContinuitySequence]) -> np.ndarray:
    return np.asarray(
        [resolve_single_event_anchor(anchors, row, sequences)[1] for row in range(len(anchors))],
        dtype=np.int64,
    )


def select_validation_candidate_per_ladder(
    candidates: Sequence[Mapping[str, Any]],
    ladder_order: Sequence[str],
) -> list[dict[str, Any]]:
    """Freeze one validation-selected configuration at every observer rung."""

    selected = []
    for ladder_name in ladder_order:
        ladder_candidates = [
            dict(row) for row in candidates if row["ladder"] == ladder_name
        ]
        if not ladder_candidates:
            continue
        selected.append(
            min(
                ladder_candidates,
                key=lambda row: (
                    row["validation_rank_mse"],
                    row["dimension"],
                    row["alpha"],
                ),
            )
        )
    return selected


def select_observer(
    raw: Mapping[str, Any],
    sequences: Mapping[str, Sequence[ContinuitySequence]],
    split_indices: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    horizon = int(config["primary_horizon"])
    anchors = build_single_event_anchor_splits(
        sequences,
        pre_events=int(config["primary_pre_events"]),
        horizon=horizon,
    )
    validation_formal = formal_innovation_events(anchors.validation, sequences["validation"])
    train_group, train_position, train_nuisance = sequence_metadata(
        sequences["train"], len(raw["rank"])
    )
    validation_group, validation_position, validation_nuisance = sequence_metadata(
        sequences["validation"], len(raw["rank"])
    )
    train_fields_map = history_fields(
        raw["rank"], raw["participation"], sequences["train"]
    )
    validation_fields_map = history_fields(
        raw["rank"], raw["participation"], sequences["validation"]
    )
    dense_fields, _, dense_weight = unit_balanced_dense_fields(
        raw["rank"], raw["participation"], sequences["train"], window=horizon
    )

    full_validation = validation_formal[validation_position[validation_formal] >= 80]
    minimum_validation = max(3, min(20, int(config["observer_minimum_observations"])))
    full_ladder_available = len(full_validation) >= minimum_validation
    candidate_ladders = (
        list(LADDER_HISTORY)
        if full_ladder_available
        else ["pre20"]
    )
    comparison_history = 80 if full_ladder_available else 20
    validation_rows = validation_formal[
        validation_position[validation_formal] >= comparison_history
    ]
    train_rows = np.asarray(split_indices["train"], dtype=np.int64)
    train_rows = train_rows[train_position[train_rows] >= comparison_history]
    if len(validation_rows) < 3 or len(train_rows) < 20:
        raise ValueError("insufficient common-support rows for observer selection")
    train_weight = balanced_row_weights(train_rows, train_group)
    validation_weight = balanced_row_weights(validation_rows, validation_group)

    candidates = []
    for dimension in map(int, config["observer_dimensions"]):
        if dimension > min(dense_fields.shape):
            continue
        basis = fit_rank_state_basis(
            dense_fields, dimension, sample_weight=dense_weight
        )
        train_ladder = projected_ladder(basis, train_fields_map, train_nuisance)
        validation_ladder = projected_ladder(
            basis, validation_fields_map, validation_nuisance
        )
        for ladder_name in candidate_ladders:
            for alpha in map(float, config["observer_alphas"]):
                observer = fit_standardized_masked_observer(
                    train_ladder[ladder_name][train_rows],
                    raw["rank"][train_rows],
                    raw["participation"][train_rows],
                    alpha=alpha,
                    feature_name=ladder_name,
                    minimum_observations=int(config["observer_minimum_observations"]),
                    sample_weight=train_weight,
                )
                prediction = observer.predict(
                    validation_ladder[ladder_name][validation_rows]
                )
                score = masked_rank_mse(
                    prediction,
                    raw["rank"][validation_rows],
                    raw["participation"][validation_rows],
                    sample_weight=validation_weight,
                )
                candidates.append(
                    {
                        "dimension": dimension,
                        "ladder": ladder_name,
                        "alpha": alpha,
                        "validation_rank_mse": score,
                        "n_train_rows": len(train_rows),
                        "n_validation_rows": len(validation_rows),
                    }
                )
    if not candidates:
        raise ValueError("observer selection produced no fitted candidate")
    candidates.sort(
        key=lambda row: (
            list(LADDER_HISTORY).index(row["ladder"]),
            row["validation_rank_mse"],
            row["dimension"],
            row["alpha"],
        )
    )
    backbone, _ = train_contact_backbone(
        raw["rank"], raw["participation"], split_indices["train"]
    )
    backbone_prediction = np.broadcast_to(
        backbone, raw["rank"][validation_rows].shape
    )
    validation_backbone_mse = masked_rank_mse(
        backbone_prediction,
        raw["rank"][validation_rows],
        raw["participation"][validation_rows],
        sample_weight=validation_weight,
    )
    selected_path = []
    for selected in select_validation_candidate_per_ladder(
        candidates, candidate_ladders
    ):
        ladder_candidates = [
            row for row in candidates if row["ladder"] == selected["ladder"]
        ]
        selected["full_ladder_available"] = full_ladder_available
        selected["comparison_history_events"] = comparison_history
        selected["validation_backbone_mse"] = validation_backbone_mse
        selected["validation_delta_vs_backbone"] = (
            validation_backbone_mse - selected["validation_rank_mse"]
        )
        selected["selection_rule"] = (
            "frozen_ladder_first_innovation_valid; dimension_and_alpha_"
            "selected_by_validation_within_ladder"
        )
        selected["candidate_table"] = ladder_candidates
        selected_path.append(selected)
    if not selected_path:
        raise ValueError("observer ladder produced no selected configuration")
    return selected_path


def _subsample_validity_rows(groups: np.ndarray, maximum: int) -> np.ndarray:
    if len(groups) <= int(maximum):
        return np.arange(len(groups), dtype=np.int64)
    selected = []
    unique = np.unique(groups)
    per_group = max(1, int(maximum) // len(unique))
    for group in unique:
        rows = np.flatnonzero(groups == group)
        take = min(len(rows), per_group)
        selected.append(rows[np.linspace(0, len(rows) - 1, take, dtype=int)])
    output = np.unique(np.concatenate(selected))
    return output[: int(maximum)]


def crossfit_innovations(
    raw: Mapping[str, Any],
    sequences: Sequence[ContinuitySequence],
    selected: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    folds = build_blocked_chronological_crossfit_folds(
        sequences,
        n_splits=int(config["crossfit_splits"]),
        embargo_events=int(config["crossfit_embargo_events"]),
        minimum_train_events=int(config["crossfit_minimum_train_events"]),
        minimum_validation_events=int(config["crossfit_minimum_validation_events"]),
    )
    dimension = int(selected["dimension"])
    ladder_name = str(selected["ladder"])
    history = int(LADDER_HISTORY[ladder_name])
    full_ladder = bool(selected["full_ladder_available"])
    validity_ladder_name = (
        "four_lag_bins_plus_time" if full_ladder else ladder_name
    )
    comparison_history = 80 if full_ladder else history
    event_parts = []
    predicted_parts = []
    residual_parts = []
    valid_parts = []
    state_residual_parts = []
    state_valid_parts = []
    feature_parts = []
    group_parts = []
    fold_records = []
    for fold_number, fold in enumerate(folds):
        sequence = sequences[int(fold.sequence_index)]
        indices = np.asarray(sequence.event_indices, dtype=np.int64)
        one_sequence = [sequence]
        _, position, nuisance = sequence_metadata(one_sequence, len(raw["rank"]))
        fields_map = history_fields(raw["rank"], raw["participation"], one_sequence)
        basis_rows = indices[max(20, fold.train_start) : fold.train_stop]
        if len(basis_rows) < max(20, dimension + 1):
            continue
        try:
            basis = fit_rank_state_basis(fields_map["pre20"][basis_rows], dimension)
        except ValueError:
            continue
        projected = projected_ladder(basis, fields_map, nuisance)
        observable = observable_ladder(basis, fields_map, nuisance)
        # Every rung is evaluated on common support.  In particular, a pre20
        # residual may not certify itself using only pre20 predictors while
        # longer past-only summaries remain available.
        train_rows = indices[max(comparison_history, fold.train_start) : fold.train_stop]
        validation_rows = indices[
            max(comparison_history, fold.validation_start) : fold.validation_stop
        ]
        if len(train_rows) < 20 or len(validation_rows) < 3:
            continue
        observer = fit_standardized_masked_observer(
            projected[ladder_name][train_rows],
            raw["rank"][train_rows],
            raw["participation"][train_rows],
            alpha=float(selected["alpha"]),
            feature_name=ladder_name,
            minimum_observations=int(config["observer_minimum_observations"]),
        )
        prediction = observer.predict(projected[ladder_name][validation_rows])
        valid = raw["participation"][validation_rows] & np.isfinite(
            raw["rank"][validation_rows]
        )
        residual = np.where(valid, raw["rank"][validation_rows] - prediction, 0.0)
        state_residual = residual @ basis.loadings
        state_valid = np.repeat((valid.sum(axis=1) >= 2)[:, None], dimension, axis=1)
        event_parts.append(validation_rows)
        predicted_parts.append(prediction.astype(np.float32))
        residual_parts.append(residual.astype(np.float32))
        valid_parts.append(valid)
        state_residual_parts.append(state_residual.astype(np.float32))
        state_valid_parts.append(state_valid)
        feature_parts.append(
            observable[validity_ladder_name][validation_rows].astype(np.float32)
        )
        group_parts.append(
            np.full(len(validation_rows), int(fold.sequence_index), dtype=np.int32)
        )
        fold_records.append(
            {
                "fold": fold_number,
                "sequence_index": int(fold.sequence_index),
                "n_train": len(train_rows),
                "n_validation": len(validation_rows),
                "first_event": int(validation_rows[0]),
                "last_event": int(validation_rows[-1]),
            }
        )
    if not event_parts:
        raise ValueError("no blocked cross-fit fold produced innovations")
    arrays = {
        "event_index": np.concatenate(event_parts),
        "predicted_rank": np.vstack(predicted_parts),
        "rank_residual": np.vstack(residual_parts),
        "rank_valid": np.vstack(valid_parts),
        "state_residual": np.vstack(state_residual_parts),
        "state_valid": np.vstack(state_valid_parts),
        "observer_features": np.vstack(feature_parts),
        "fold_group": np.concatenate(group_parts),
    }
    if len(np.unique(arrays["event_index"])) != len(arrays["event_index"]):
        raise RuntimeError("cross-fit validation events overlap across folds")
    validity_rows = _subsample_validity_rows(
        arrays["fold_group"], int(config["innovation_validity_max_rows"])
    )
    validity = blocked_innovation_validity(
        arrays["observer_features"][validity_rows],
        arrays["rank_residual"][validity_rows],
        arrays["rank_valid"][validity_rows],
        arrays["fold_group"][validity_rows],
        block_size=int(config["innovation_null_block_events"]),
        n_null=int(config["innovation_null_draws"]),
        seed=int(config["innovation_null_seed"]),
    )
    status = (
        "INNOVATION_VALID"
        if full_ladder and validity["valid"]
        else "UNRESOLVED_INSUFFICIENT_HISTORY"
        if not full_ladder
        else "UNRESOLVED_STATE_RESIDUAL"
    )
    summary = {
        "status": status,
        "n_crossfit_rows": len(arrays["event_index"]),
        "n_crossfit_folds": len(fold_records),
        "n_validity_rows": len(validity_rows),
        "validity_coordinate": "contact_rank",
        "validity_group": "continuity_sequence",
        "folds": fold_records,
        "validity_feature_ladder": validity_ladder_name,
        "whiteness": validity,
    }
    return summary, arrays


def run_subject(
    subject: str,
    config: Mapping[str, Any],
    phase0_root: Path,
    output: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, split_indices, sequences, phase0_path = _prepare(
        subject, config, phase0_root
    )
    selected_path = select_observer(raw, sequences, split_indices, config)
    innovation_path = []
    innovation = None
    arrays = None
    selected = None
    for candidate in selected_path:
        candidate_innovation, candidate_arrays = crossfit_innovations(
            raw, sequences["train"], candidate, config
        )
        innovation_path.append(
            {
                "ladder": candidate["ladder"],
                "dimension": candidate["dimension"],
                "alpha": candidate["alpha"],
                "status": candidate_innovation["status"],
                "whiteness": candidate_innovation["whiteness"],
            }
        )
        selected = candidate
        innovation = candidate_innovation
        arrays = candidate_arrays
        if candidate_innovation["status"] == "INNOVATION_VALID":
            break
    if selected is None or innovation is None or arrays is None:
        raise RuntimeError("observer ladder did not produce a cross-fitted innovation")
    artifact = output / "crossfit" / f"{subject}.npz"
    _atomic_npz(artifact, **arrays)
    result = {
        "contract": str(config["contract"]),
        "status": innovation["status"],
        "subject": subject,
        "one_step_is_one_complete_event": True,
        "observer_selection": selected,
        "observer_ladder_trace": innovation_path,
        "crossfit_innovation": innovation,
        "crossfit_artifact": str(artifact),
        "crossfit_artifact_sha256": sha256(artifact),
        "phase0_path": str(phase0_path),
        "phase0_sha256": sha256(phase0_path),
        "human_test_outcomes_read": False,
        "impulse_response_test_run": False,
        "within_event_next_rank_model_fit": False,
        "forbidden_inputs_read": False,
    }
    patient_path = output / "per_subject" / f"{subject}.json"
    atomic_write_json(patient_path, _jsonable(result))
    row = {
        "subject": subject,
        "status": innovation["status"],
        "dimension": selected["dimension"],
        "ladder": selected["ladder"],
        "alpha": selected["alpha"],
        "full_ladder_available": selected["full_ladder_available"],
        "validation_rank_mse": selected["validation_rank_mse"],
        "validation_backbone_mse": selected["validation_backbone_mse"],
        "validation_delta_vs_backbone": selected["validation_delta_vs_backbone"],
        "n_validation_rows": selected["n_validation_rows"],
        "n_crossfit_rows": innovation["n_crossfit_rows"],
        "observed_max_abs_correlation": innovation["whiteness"]["observed_max_abs_correlation"],
        "null_q95": innovation["whiteness"]["null_q95"],
    }
    return result, row


def unresolved_history_result(
    subject: str,
    config: Mapping[str, Any],
    phase0_root: Path,
    output: Path,
    reason: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Record an expected support limitation without failing the cohort run."""

    phase0_path = phase0_root / "per_subject" / f"{subject}.json"
    result = {
        "contract": str(config["contract"]),
        "status": "UNRESOLVED_INSUFFICIENT_HISTORY",
        "subject": subject,
        "reason": str(reason),
        "one_step_is_one_complete_event": True,
        "observer_selection": None,
        "observer_ladder_trace": [],
        "crossfit_innovation": None,
        "crossfit_artifact": None,
        "phase0_path": str(phase0_path),
        "phase0_sha256": sha256(phase0_path),
        "human_test_outcomes_read": False,
        "impulse_response_test_run": False,
        "within_event_next_rank_model_fit": False,
        "forbidden_inputs_read": False,
    }
    atomic_write_json(output / "per_subject" / f"{subject}.json", result)
    row = {
        "subject": subject,
        "status": result["status"],
        "dimension": None,
        "ladder": None,
        "alpha": None,
        "full_ladder_available": False,
        "validation_rank_mse": None,
        "validation_backbone_mse": None,
        "validation_delta_vs_backbone": None,
        "n_validation_rows": 0,
        "n_crossfit_rows": 0,
        "observed_max_abs_correlation": None,
        "null_q95": None,
    }
    return result, row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase0_root = ROOT / str(config["output_root"])
    measurement_root = ROOT / str(config["phase1_output_root"])
    measurement = json.loads(
        (measurement_root / "state_reliability.json").read_text(encoding="utf-8")
    )
    if (
        measurement.get("status") != "STATE_MEASUREMENT_COMPLETE_INNOVATION_PENDING"
        or measurement.get("n_pass") != 34
        or measurement.get("human_test_outcomes_read") is not False
    ):
        raise SystemExit("validation-only 34-patient state measurement is not complete")
    output = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else ROOT / (args.output_dir or Path(str(config["innovation_output_root"])))
    )
    cohort = [str(patient["subject"]) for patient in measurement["patients"]]
    subjects = cohort if not args.subjects else list(map(str, args.subjects))
    if sorted(set(subjects) - set(cohort)):
        raise SystemExit("requested subject is outside validation-only state measurement")
    full_cohort = subjects == cohort
    results = []
    rows = []
    failures = []
    for subject in subjects:
        try:
            result, row = run_subject(subject, config, phase0_root, output)
        except ValueError as exc:
            if "insufficient common-support rows" not in str(exc):
                failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
                print(subject, "FAIL", exc, flush=True)
                continue
            result, row = unresolved_history_result(
                subject, config, phase0_root, output, str(exc)
            )
        except Exception as exc:
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(subject, "FAIL", exc, flush=True)
            continue
        results.append(result)
        rows.append(row)
        print(subject, result["status"], flush=True)
    if rows:
        _atomic_csv(output / "observer_selection.csv", pd.DataFrame(rows))
    if failures:
        _atomic_csv(output / "failures.csv", pd.DataFrame(failures))
    elif (output / "failures.csv").exists():
        (output / "failures.csv").unlink()
    state_status = (
        "INNOVATION_VALIDITY_FAIL_CLOSED"
        if failures
        else "INNOVATION_VALIDITY_COMPLETE"
        if full_cohort
        else "INNOVATION_VALIDITY_PARTIAL_AUDIT"
    )
    state = {
        "contract": str(config["contract"]),
        "status": state_status,
        "cohort_scope": "full_34" if full_cohort else "explicit_partial_audit",
        "n_requested": len(subjects),
        "n_pass": len(results),
        "n_failed": len(failures),
        "n_innovation_valid": sum(row["status"] == "INNOVATION_VALID" for row in rows),
        "n_unresolved_state": sum(row["status"] == "UNRESOLVED_STATE_RESIDUAL" for row in rows),
        "n_unresolved_history": sum(row["status"] == "UNRESOLVED_INSUFFICIENT_HISTORY" for row in rows),
        "patients": results,
        "failures": failures,
        "measurement_state_path": str(measurement_root / "state_reliability.json"),
        "measurement_state_sha256": sha256(measurement_root / "state_reliability.json"),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "observer_module_sha256": sha256(ROOT / "src/topic5_event_innovation_observer_v3_0.py"),
        "core_module_sha256": sha256(ROOT / "src/topic5_event_innovation_v3_0.py"),
        "data_module_sha256": sha256(ROOT / "src/topic5_event_innovation_data.py"),
        "human_test_outcomes_read": False,
        "impulse_response_test_run": False,
        "within_event_next_rank_model_fit": False,
        "forbidden_inputs_read": False,
    }
    atomic_write_json(output / "innovation_validity.json", _jsonable(state))
    print(json.dumps({"status": state_status, "n_pass": len(results), "n_failed": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
