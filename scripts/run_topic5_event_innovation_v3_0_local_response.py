#!/usr/bin/env python3
"""Validation-only multi-horizon local response for Topic 5 v3.0.

This runner consumes the frozen train-only cross-fitted observer residuals and
never reads the human test split.  One row/event is one complete interictal
event; no within-event next-rank model is fitted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
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

from scripts.audit_topic5_event_innovation_v3_0_phase0 import sha256  # noqa: E402
from scripts.run_topic5_event_innovation_v3_0_observer import (  # noqa: E402
    LADDER_HISTORY,
    balanced_row_weights,
    history_fields,
    projected_ladder,
    sequence_metadata,
)
from scripts.run_topic5_event_innovation_v3_0_phase1_measurement import (  # noqa: E402
    _prepare,
    unit_balanced_dense_fields,
)
from src.topic5_event_innovation_data import (  # noqa: E402
    ContinuitySequence,
    build_single_event_anchors,
    build_single_event_anchor_splits,
    resolve_single_event_anchor,
)
from src.topic5_event_innovation_observer_v3_0 import (  # noqa: E402
    coherent_block_permutation,
    fit_standardized_masked_observer,
)
from src.topic5_event_innovation_response_v3_0 import (  # noqa: E402
    fit_weighted_local_projection,
    future_precedence_brier,
    masked_innovation_projection,
    masked_rank_field_mse,
    masked_state_projection,
    observable_propagation_gain,
)
from src.topic5_event_innovation_v3_0 import (  # noqa: E402
    RankStateBasis,
    fit_rank_state_basis,
    masked_window_rank_field,
)
from src.topic5_resource_guard import atomic_write_json, pin_thread_environment  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"


@dataclass(frozen=True)
class ResponseRows:
    event_index: np.ndarray
    group: np.ndarray
    pre_state: np.ndarray
    future_state: np.ndarray
    past_state: np.ndarray
    innovation_state: np.ndarray
    nuisance: np.ndarray
    observed_future_field: np.ndarray
    future_support: np.ndarray
    future_windows: list[np.ndarray]


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


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


def innovation_lookup(
    event_index: np.ndarray,
    residual: np.ndarray,
    valid: np.ndarray,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    index = np.asarray(event_index, dtype=np.int64)
    values = np.asarray(residual, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or mask.shape != values.shape or len(index) != len(values):
        raise ValueError("innovation lookup arrays are not aligned")
    if len(np.unique(index)) != len(index):
        raise ValueError("innovation event indices must be unique")
    return {int(event): (values[row], mask[row]) for row, event in enumerate(index)}


def fit_final_observer_innovations(
    raw: Mapping[str, Any],
    split_indices: Mapping[str, np.ndarray],
    sequences: Mapping[str, Sequence[ContinuitySequence]],
    basis: RankStateBasis,
    selected: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Fit on train and emit future-blind validation event residuals."""

    ladder_name = str(selected["ladder"])
    history = int(LADDER_HISTORY[ladder_name])
    train_group, train_position, train_nuisance = sequence_metadata(
        sequences["train"], len(raw["rank"])
    )
    _, validation_position, validation_nuisance = sequence_metadata(
        sequences["validation"], len(raw["rank"])
    )
    train_fields = history_fields(raw["rank"], raw["participation"], sequences["train"])
    validation_fields = history_fields(
        raw["rank"], raw["participation"], sequences["validation"]
    )
    train_ladder = projected_ladder(basis, train_fields, train_nuisance)
    validation_ladder = projected_ladder(basis, validation_fields, validation_nuisance)
    train_rows = np.asarray(split_indices["train"], dtype=np.int64)
    train_rows = train_rows[train_position[train_rows] >= history]
    validation_rows = np.concatenate(
        [np.asarray(sequence.event_indices, dtype=np.int64) for sequence in sequences["validation"]]
    )
    validation_rows = validation_rows[validation_position[validation_rows] >= history]
    observer = fit_standardized_masked_observer(
        train_ladder[ladder_name][train_rows],
        raw["rank"][train_rows],
        raw["participation"][train_rows],
        alpha=float(selected["alpha"]),
        feature_name=ladder_name,
        minimum_observations=int(config["observer_minimum_observations"]),
        sample_weight=balanced_row_weights(train_rows, train_group),
    )
    predicted = observer.predict(validation_ladder[ladder_name][validation_rows])
    valid = raw["participation"][validation_rows] & np.isfinite(raw["rank"][validation_rows])
    residual = np.where(valid, raw["rank"][validation_rows] - predicted, 0.0)
    return innovation_lookup(validation_rows, residual, valid)


def build_response_rows(
    raw: Mapping[str, Any],
    anchors,
    sequences: Sequence[ContinuitySequence],
    basis: RankStateBasis,
    innovations: Mapping[int, tuple[np.ndarray, np.ndarray]],
    nuisance_by_event: np.ndarray,
) -> ResponseRows:
    event_index = []
    group = []
    pre_fields = []
    pre_valid = []
    future_fields = []
    future_support = []
    past_fields = []
    past_valid = []
    innovation_values = []
    innovation_valid = []
    nuisance = []
    future_windows = []
    for row in range(len(anchors)):
        sequence_index = int(anchors.sequence_index[row])
        sequence = sequences[sequence_index]
        indices = np.asarray(sequence.event_indices, dtype=np.int64)
        pre, event, future = resolve_single_event_anchor(anchors, row, sequences)
        if event not in innovations:
            continue
        pre_position = int(anchors.pre_start[row])
        if pre_position < int(anchors.pre_events):
            continue
        past = indices[pre_position - int(anchors.pre_events) : pre_position]
        pre_field, pre_support = masked_window_rank_field(
            raw["rank"], raw["participation"], pre
        )
        future_field, support = masked_window_rank_field(
            raw["rank"], raw["participation"], future
        )
        past_field, past_support = masked_window_rank_field(
            raw["rank"], raw["participation"], past
        )
        innovation, valid = innovations[event]
        event_index.append(event)
        group.append(sequence_index)
        pre_fields.append(pre_field)
        pre_valid.append(pre_support > 0)
        future_fields.append(future_field)
        future_support.append(support)
        past_fields.append(past_field)
        past_valid.append(past_support > 0)
        innovation_values.append(innovation)
        innovation_valid.append(valid)
        nuisance.append(nuisance_by_event[event])
        future_windows.append(future)
    if not event_index:
        raise ValueError("no response anchor has an available innovation")
    pre_state, pre_ok = masked_state_projection(
        np.vstack(pre_fields), np.vstack(pre_valid), basis
    )
    future_state, future_ok = masked_state_projection(
        np.vstack(future_fields), np.vstack(future_support) > 0, basis
    )
    past_state, past_ok = masked_state_projection(
        np.vstack(past_fields), np.vstack(past_valid), basis
    )
    innovation_state, innovation_ok = masked_innovation_projection(
        np.vstack(innovation_values), np.vstack(innovation_valid), basis
    )
    keep = pre_ok & future_ok & past_ok & innovation_ok
    if np.sum(keep) < max(10, 3 * basis.dimension):
        raise ValueError("insufficient estimable response anchors")
    selected_rows = np.flatnonzero(keep)
    return ResponseRows(
        event_index=np.asarray(event_index, dtype=np.int64)[keep],
        group=np.asarray(group, dtype=np.int32)[keep],
        pre_state=pre_state[keep],
        future_state=future_state[keep],
        past_state=past_state[keep],
        innovation_state=innovation_state[keep],
        nuisance=np.vstack(nuisance)[keep],
        observed_future_field=np.vstack(future_fields)[keep],
        future_support=np.vstack(future_support)[keep],
        future_windows=[future_windows[index] for index in selected_rows],
    )


def group_balanced_weights(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups)
    weight = np.zeros(len(values), dtype=float)
    for group in np.unique(values):
        selected = values == group
        weight[selected] = 1.0 / np.sum(selected)
    return weight


def state_matched_donor(
    pre_state: np.ndarray,
    innovation: np.ndarray,
    group: np.ndarray,
    progress: np.ndarray,
    *,
    seed: int,
    top_k: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Choose non-self donor innovations matched on pre-state and progress."""

    pools, pool_distance = state_matched_donor_pool(
        pre_state, group, progress, top_k=top_k
    )
    event = np.asarray(innovation, dtype=float)
    rng = np.random.default_rng(int(seed))
    chosen = np.asarray([rng.choice(pool) for pool in pools], dtype=np.int64)
    counts = np.bincount(chosen, minlength=len(event))
    return event[chosen], {
        "mean_state_distance": float(np.mean(pool_distance)),
        "maximum_donor_reuse": int(np.max(counts)),
        "eligible_anchor_fraction": 1.0,
    }


def state_matched_donor_pool(
    pre_state: np.ndarray,
    group: np.ndarray,
    progress: np.ndarray,
    *,
    top_k: int = 5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Precompute nearest eligible donors once for all null draws."""

    pre = np.asarray(pre_state, dtype=float)
    groups = np.asarray(group)
    phase = np.asarray(progress, dtype=float)
    if pre.ndim != 2 or groups.shape != (len(pre),) or phase.shape != (len(pre),):
        raise ValueError("donor arrays are not aligned")
    scale = np.std(pre, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    phase_scale = max(float(np.std(phase)), 1e-8)
    pools = []
    distance = []
    for row in range(len(pre)):
        candidates = np.flatnonzero(groups == groups[row])
        candidates = candidates[candidates != row]
        if not len(candidates):
            candidates = np.flatnonzero(np.arange(len(pre)) != row)
        if not len(candidates):
            raise ValueError("state-matched donor needs at least two rows")
        d = np.sum(((pre[candidates] - pre[row]) / scale) ** 2, axis=1)
        d += ((phase[candidates] - phase[row]) / phase_scale) ** 2
        order = candidates[np.argsort(d, kind="stable")[: max(1, min(int(top_k), len(candidates)))]]
        pools.append(order)
        distance.append(float(np.min(np.sqrt(d))))
    return pools, np.asarray(distance, dtype=float)


def _state_mse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((np.asarray(observed) - np.asarray(predicted)) ** 2))


def contiguous_group_codes(group: np.ndarray, event_index: np.ndarray) -> np.ndarray:
    """Split continuity units again wherever estimable dense anchors have a gap."""

    groups = np.asarray(group)
    events = np.asarray(event_index, dtype=np.int64)
    if groups.shape != events.shape:
        raise ValueError("group/event arrays are not aligned")
    output = np.full(len(groups), -1, dtype=np.int32)
    code = 0
    for original in np.unique(groups):
        rows = np.flatnonzero(groups == original)
        order = rows[np.argsort(events[rows], kind="stable")]
        if not len(order):
            continue
        output[order[0]] = code
        for previous, current in zip(order[:-1], order[1:]):
            if events[current] != events[previous] + 1:
                code += 1
            output[current] = code
        code += 1
    return output


def chronology_nulls(
    fit,
    rows: ResponseRows,
    *,
    block_sizes: Sequence[int],
    safe_shift_events: Sequence[int],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Run exact-event block and no-wrap shift nulls on dense validation anchors."""

    groups = contiguous_group_codes(rows.group, rows.event_index)
    output: dict[str, Any] = {"block": {}, "safe_shift": {}}
    rng = np.random.default_rng(int(seed))
    for raw_size in map(int, block_sizes):
        eligible_groups = [
            group
            for group in np.unique(groups)
            if np.sum(groups == group) >= 2 * raw_size
        ]
        eligible = np.isin(groups, eligible_groups)
        if np.sum(eligible) < 10:
            output["block"][str(raw_size)] = {
                "status": "INSUFFICIENT_SUPPORT",
                "n_eligible": int(np.sum(eligible)),
            }
            continue
        true_prediction = fit.predict(
            rows.pre_state[eligible],
            rows.innovation_state[eligible],
            rows.nuisance[eligible],
        )
        true_error = _state_mse(rows.future_state[eligible], true_prediction)
        gains = []
        for _ in range(int(draws)):
            permuted, _ = coherent_block_permutation(
                rows.innovation_state[eligible],
                np.ones_like(rows.innovation_state[eligible], dtype=bool),
                groups[eligible],
                block_size=raw_size,
                rng=rng,
            )
            prediction = fit.predict(
                rows.pre_state[eligible], permuted, rows.nuisance[eligible]
            )
            gains.append(_state_mse(rows.future_state[eligible], prediction) - true_error)
        output["block"][str(raw_size)] = {
            "status": "COMPLETE",
            "n_eligible": int(np.sum(eligible)),
            "median_true_minus_null_gain": float(np.median(gains)),
            "q05_q95": np.quantile(gains, [0.05, 0.95]).tolist(),
        }
    for lag in map(int, safe_shift_events):
        selected = []
        donor = []
        for group in np.unique(groups):
            group_rows = np.flatnonzero(groups == group)
            order = group_rows[np.argsort(rows.event_index[group_rows], kind="stable")]
            if len(order) <= lag:
                continue
            selected.extend(order[lag:].tolist())
            donor.extend(order[:-lag].tolist())
        selected = np.asarray(selected, dtype=np.int64)
        donor = np.asarray(donor, dtype=np.int64)
        if len(selected) < 10:
            output["safe_shift"][str(lag)] = {
                "status": "INSUFFICIENT_SUPPORT",
                "n_eligible": len(selected),
            }
            continue
        true_prediction = fit.predict(
            rows.pre_state[selected],
            rows.innovation_state[selected],
            rows.nuisance[selected],
        )
        shifted_prediction = fit.predict(
            rows.pre_state[selected],
            rows.innovation_state[donor],
            rows.nuisance[selected],
        )
        output["safe_shift"][str(lag)] = {
            "status": "COMPLETE",
            "n_eligible": len(selected),
            "true_minus_shift_gain": _state_mse(
                rows.future_state[selected], shifted_prediction
            )
            - _state_mse(rows.future_state[selected], true_prediction),
            "no_wraparound": True,
        }
    return output


def evaluate_horizon(
    raw: Mapping[str, Any],
    basis: RankStateBasis,
    train: ResponseRows,
    validation: ResponseRows,
    validation_dense: ResponseRows,
    alphas: Sequence[float],
    *,
    donor_draws: int,
    donor_seed: int,
    block_sizes: Sequence[int],
    safe_shift_multipliers: Sequence[int],
) -> dict[str, Any]:
    candidates = []
    train_weight = group_balanced_weights(train.group)
    for alpha in map(float, alphas):
        fit = fit_weighted_local_projection(
            train.pre_state,
            train.future_state,
            train.innovation_state,
            nuisance=train.nuisance,
            alpha=alpha,
            sample_weight=train_weight,
        )
        prediction = fit.predict(
            validation.pre_state,
            validation.innovation_state,
            validation.nuisance,
        )
        candidates.append(
            {"alpha": alpha, "validation_state_mse": _state_mse(validation.future_state, prediction), "fit": fit}
        )
    selected = min(candidates, key=lambda row: (row["validation_state_mse"], row["alpha"]))
    fit = selected["fit"]
    true_prediction = fit.predict(
        validation.pre_state, validation.innovation_state, validation.nuisance
    )
    autonomous_prediction = fit.predict(
        validation.pre_state,
        np.zeros_like(validation.innovation_state),
        validation.nuisance,
    )
    observable = observable_propagation_gain(
        basis,
        validation.observed_future_field,
        validation.future_support,
        validation.future_windows,
        raw["rank"],
        raw["participation"],
        raw["rank"],
        autonomous_prediction,
        true_prediction,
    )
    train_backbone = np.broadcast_to(
        basis.backbone, train.observed_future_field.shape
    )
    rank_scale = masked_rank_field_mse(
        train_backbone, train.observed_future_field, train.future_support
    )
    pair_scale = future_precedence_brier(
        train_backbone,
        train.future_windows,
        raw["rank"],
        raw["participation"],
        raw["rank"],
    )
    observable["rank_training_scale"] = rank_scale
    observable["precedence_training_scale"] = pair_scale
    observable["rank_gain_standardized"] = observable["rank_gain"] / max(rank_scale, 1e-12)
    observable["precedence_gain_standardized"] = observable["precedence_gain"] / max(pair_scale, 1e-12)
    observable["propagation_gain_standardized"] = 0.5 * (
        observable["rank_gain_standardized"]
        + observable["precedence_gain_standardized"]
    )
    state_gain = _state_mse(validation.future_state, autonomous_prediction) - _state_mse(
        validation.future_state, true_prediction
    )

    past_fit = fit_weighted_local_projection(
        train.pre_state,
        train.past_state,
        train.innovation_state,
        nuisance=train.nuisance,
        alpha=float(selected["alpha"]),
        sample_weight=train_weight,
    )
    past_true = past_fit.predict(
        validation.pre_state, validation.innovation_state, validation.nuisance
    )
    past_auto = past_fit.predict(
        validation.pre_state,
        np.zeros_like(validation.innovation_state),
        validation.nuisance,
    )
    past_gain = _state_mse(validation.past_state, past_auto) - _state_mse(
        validation.past_state, past_true
    )

    donor_gains = []
    donor_audit = []
    true_state_error = _state_mse(validation.future_state, true_prediction)
    donor_pools, donor_distance = state_matched_donor_pool(
        validation.pre_state,
        validation.group,
        validation.nuisance[:, 0],
        top_k=5,
    )
    for draw in range(int(donor_draws)):
        rng = np.random.default_rng(int(donor_seed) + draw)
        chosen = np.asarray([rng.choice(pool) for pool in donor_pools], dtype=np.int64)
        donor = validation.innovation_state[chosen]
        counts = np.bincount(chosen, minlength=len(validation.event_index))
        audit = {
            "mean_state_distance": float(np.mean(donor_distance)),
            "maximum_donor_reuse": int(np.max(counts)),
            "eligible_anchor_fraction": 1.0,
        }
        donor_prediction = fit.predict(
            validation.pre_state, donor, validation.nuisance
        )
        donor_gains.append(
            _state_mse(validation.future_state, donor_prediction)
            - true_state_error
        )
        donor_audit.append(audit)
    nulls = chronology_nulls(
        fit,
        validation_dense,
        block_sizes=block_sizes,
        safe_shift_events=[int(value) * int(len(validation.future_windows[0])) for value in safe_shift_multipliers],
        draws=int(donor_draws),
        seed=int(donor_seed) + 10000,
    )
    return {
        "selected_alpha": float(selected["alpha"]),
        "validation_state_mse": float(selected["validation_state_mse"]),
        "state_gain_over_autonomous": state_gain,
        "past_state_gain": past_gain,
        "future_minus_past_state_gain": state_gain - past_gain,
        "true_minus_state_matched_null_gain": float(np.median(donor_gains)),
        "state_matched_null_score_coordinate": "orthogonal_rotation_invariant_state_mse",
        "state_matched_null_draws": int(len(donor_gains)),
        "state_matched_null_q05_q95": np.quantile(donor_gains, [0.05, 0.95]).tolist(),
        "state_matched_mean_state_distance": float(np.mean([row["mean_state_distance"] for row in donor_audit])),
        "state_matched_maximum_donor_reuse": int(max(row["maximum_donor_reuse"] for row in donor_audit)),
        "n_train_anchors": int(len(train.event_index)),
        "n_validation_anchors": int(len(validation.event_index)),
        "observable": observable,
        "chronology_nulls": nulls,
        "candidate_table": [
            {"alpha": row["alpha"], "validation_state_mse": row["validation_state_mse"]}
            for row in candidates
        ],
    }


def run_subject(
    subject: str,
    config: Mapping[str, Any],
    phase0_root: Path,
    innovation_root: Path,
) -> dict[str, Any]:
    observer_record = json.loads(
        (innovation_root / "per_subject" / f"{subject}.json").read_text(encoding="utf-8")
    )
    if observer_record.get("status") != "INNOVATION_VALID":
        return {
            "subject": subject,
            "status": str(observer_record.get("status")),
            "eligible": False,
        }
    raw, split_indices, sequences, phase0_path = _prepare(subject, config, phase0_root)
    selected = observer_record["observer_selection"]
    dimension = int(selected["dimension"])
    dense_fields, _, dense_weight = unit_balanced_dense_fields(
        raw["rank"], raw["participation"], sequences["train"], window=int(config["primary_horizon"])
    )
    basis = fit_rank_state_basis(dense_fields, dimension, sample_weight=dense_weight)
    crossfit_path = Path(observer_record["crossfit_artifact"])
    with np.load(crossfit_path, allow_pickle=False) as data:
        train_innovations = innovation_lookup(
            data["event_index"], data["rank_residual"], data["rank_valid"]
        )
    validation_innovations = fit_final_observer_innovations(
        raw, split_indices, sequences, basis, selected, config
    )
    _, _, train_nuisance = sequence_metadata(sequences["train"], len(raw["rank"]))
    _, _, validation_nuisance = sequence_metadata(
        sequences["validation"], len(raw["rank"])
    )
    horizons = {}
    for horizon in map(int, config["horizons"]):
        anchors = build_single_event_anchor_splits(
            sequences,
            pre_events=int(config["primary_pre_events"]),
            horizon=horizon,
        )
        try:
            train_rows = build_response_rows(
                raw,
                anchors.train,
                sequences["train"],
                basis,
                train_innovations,
                train_nuisance,
            )
            validation_rows = build_response_rows(
                raw,
                anchors.validation,
                sequences["validation"],
                basis,
                validation_innovations,
                validation_nuisance,
            )
            dense_validation_anchors = build_single_event_anchors(
                sequences["validation"],
                pre_events=int(config["primary_pre_events"]),
                horizon=horizon,
                stride=1,
            )
            validation_dense = build_response_rows(
                raw,
                dense_validation_anchors,
                sequences["validation"],
                basis,
                validation_innovations,
                validation_nuisance,
            )
            horizons[str(horizon)] = evaluate_horizon(
                raw,
                basis,
                train_rows,
                validation_rows,
                validation_dense,
                config.get("local_projection_alphas", [0.1, 1.0, 10.0, 100.0]),
                donor_draws=int(config.get("local_response_null_draws", 100)),
                donor_seed=int(config.get("local_response_null_seed", 7401)) + horizon,
                block_sizes=config.get("local_response_block_sizes", [1, 2, 5, 10, 20, 40]),
                safe_shift_multipliers=config.get("local_response_safe_shift_multipliers", [2, 3, 4]),
            )
        except ValueError as exc:
            horizons[str(horizon)] = {"status": "INSUFFICIENT_SUPPORT", "reason": str(exc)}
    primary = horizons[str(int(config["primary_horizon"]))]
    primary_eligible = "observable" in primary
    return {
        "contract": str(config["contract"]),
        "subject": subject,
        "status": "LOCAL_RESPONSE_VALIDATION_COMPLETE" if primary_eligible else "LOCAL_RESPONSE_PRIMARY_UNAVAILABLE",
        "eligible": bool(primary_eligible),
        "dimension": dimension,
        "observer_ladder": selected["ladder"],
        "horizons": horizons,
        "phase0_path": str(phase0_path),
        "phase0_sha256": sha256(phase0_path),
        "observer_record_sha256": sha256(innovation_root / "per_subject" / f"{subject}.json"),
        "human_test_outcomes_read": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }


def cohort_handoff(rows: list[dict[str, Any]], primary_horizon: int) -> dict[str, Any]:
    eligible = [row for row in rows if row.get("eligible")]
    primary = [row["horizons"][str(int(primary_horizon))] for row in eligible]
    propagation = np.asarray(
        [row["observable"]["propagation_gain_standardized"] for row in primary]
    )
    matched = np.asarray([row["true_minus_state_matched_null_gain"] for row in primary])
    direction = np.asarray([row["future_minus_past_state_gain"] for row in primary])
    route_open = bool(
        len(primary) > 0
        and np.median(propagation) > 0
        and np.median(matched) > 0
        and np.median(direction) > 0
    )
    return {
        "route": "goal2_local_response",
        "status": "OPEN" if route_open else "NOT_OPEN",
        "n_eligible": len(primary),
        "median_propagation_gain": float(np.median(propagation)) if len(primary) else float("nan"),
        "median_true_minus_state_matched_null_gain": float(np.median(matched)) if len(primary) else float("nan"),
        "median_future_minus_past_state_gain": float(np.median(direction)) if len(primary) else float("nan"),
        "favorable_propagation": int(np.sum(propagation > 0)),
        "favorable_matched": int(np.sum(matched > 0)),
        "favorable_future_past": int(np.sum(direction > 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    pin_thread_environment(1, disable_cuda=True)
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    phase0_root = ROOT / str(config["output_root"])
    innovation_root = ROOT / str(config["innovation_output_root"])
    innovation_state_path = innovation_root / "innovation_validity.json"
    innovation_state = json.loads(innovation_state_path.read_text(encoding="utf-8"))
    if (
        innovation_state.get("status") != "INNOVATION_VALIDITY_COMPLETE"
        or innovation_state.get("n_pass") != 34
        or innovation_state.get("human_test_outcomes_read") is not False
    ):
        raise SystemExit("34-patient validation-only innovation state is not complete")
    cohort = [row["subject"] for row in innovation_state["patients"]]
    subjects = cohort if not args.subjects else list(map(str, args.subjects))
    if sorted(set(subjects) - set(cohort)):
        raise SystemExit("requested subject is outside innovation-validity cohort")
    output = (
        args.output_dir
        if args.output_dir is not None and args.output_dir.is_absolute()
        else ROOT / (args.output_dir or Path(str(config.get(
            "local_response_output_root",
            "results/topic5_event_innovation_impulse_response/v3_0/local_response_validation_only",
        ))))
    )
    rows = []
    failures = []
    for subject in subjects:
        try:
            row = run_subject(subject, config, phase0_root, innovation_root)
            rows.append(row)
            atomic_write_json(output / "per_subject" / f"{subject}.json", _jsonable(row))
            print(subject, row["status"], flush=True)
        except Exception as exc:
            failures.append({"subject": subject, "error": f"{type(exc).__name__}: {exc}"})
            print(subject, "FAIL", exc, flush=True)
    primary = int(config["primary_horizon"])
    handoff = cohort_handoff(rows, primary)
    summary_rows = []
    for row in rows:
        primary_row = row.get("horizons", {}).get(str(primary), {})
        summary_rows.append({
            "subject": row["subject"],
            "status": row["status"],
            "eligible": row.get("eligible", False),
            "dimension": row.get("dimension", np.nan),
            "propagation_gain": primary_row.get("observable", {}).get("propagation_gain_standardized", np.nan),
            "rank_gain": primary_row.get("observable", {}).get("rank_gain", np.nan),
            "precedence_gain": primary_row.get("observable", {}).get("precedence_gain", np.nan),
            "true_minus_matched": primary_row.get("true_minus_state_matched_null_gain", np.nan),
            "future_minus_past": primary_row.get("future_minus_past_state_gain", np.nan),
            "n_validation_anchors": primary_row.get("n_validation_anchors", 0),
        })
    if summary_rows:
        _atomic_csv(output / "patient_local_effects.csv", pd.DataFrame(summary_rows))
    if failures:
        _atomic_csv(output / "failures.csv", pd.DataFrame(failures))
    state = {
        "contract": str(config["contract"]),
        "status": "LOCAL_RESPONSE_VALIDATION_COMPLETE" if not failures else "LOCAL_RESPONSE_VALIDATION_FAIL_CLOSED",
        "cohort_scope": "full_34_validation_only" if subjects == cohort else "explicit_partial_audit",
        "n_requested": len(subjects),
        "n_completed": len(rows),
        "n_failed": len(failures),
        "n_innovation_valid": int(sum(row.get("eligible", False) for row in rows)),
        "goal2_handoff": handoff,
        "patients": rows,
        "failures": failures,
        "innovation_state_sha256": sha256(innovation_state_path),
        "config_sha256": sha256(config_path),
        "source_sha256": sha256(Path(__file__)),
        "response_module_sha256": sha256(ROOT / "src/topic5_event_innovation_response_v3_0.py"),
        "human_test_outcomes_read": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
    }
    atomic_write_json(output / "local_projection_state.json", _jsonable(state))
    atomic_write_json(output / "GOAL2_HANDOFF_STATE.json", _jsonable(handoff))
    print(json.dumps({"status": state["status"], "handoff": handoff}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
