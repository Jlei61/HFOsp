"""Full-grid, patient-internal prequential hazard analyses for H2b v0.3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .v03_assay import (
    ALPHA_GRID,
    _fit_with_inner_alpha,
    _logistic_predict,
    _logloss,
    _standardise_train_test,
)


@dataclass(frozen=True)
class HazardDesign:
    source_index: np.ndarray
    time_epoch: np.ndarray
    segment: np.ndarray
    history: np.ndarray
    current_observation: np.ndarray
    persistent_state: np.ndarray
    memoryless_state: np.ndarray
    onset_time: np.ndarray
    onset_segment: np.ndarray

    def validate(self) -> None:
        n = len(self.time_epoch)
        if any(len(value) != n for value in (
            self.source_index,
            self.segment, self.history, self.current_observation,
            self.persistent_state, self.memoryless_state,
        )):
            raise ValueError("hazard design arrays disagree")
        if len(self.onset_time) != len(self.onset_segment):
            raise ValueError("hazard onset arrays disagree")
        if not np.all(np.diff(self.time_epoch) >= 0):
            raise ValueError("hazard rows are not chronological")
        for value in (
            self.history, self.current_observation, self.persistent_state,
            self.memoryless_state,
        ):
            if not np.isfinite(value).all():
                raise ValueError("hazard design contains non-finite values")


def downsample_recorded_grid(
    time_epoch: np.ndarray,
    segment: np.ndarray,
    *,
    spacing_seconds: float = 300.0,
) -> np.ndarray:
    """Select one causal row per 5 min without ever crossing a segment."""
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(segment, dtype=np.int64)
    selected: list[int] = []
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        rows = rows[np.argsort(time[rows], kind="stable")]
        cursor = -np.inf
        for row in rows:
            if float(time[row]) >= cursor - 1e-9:
                selected.append(int(row))
                cursor = float(time[row]) + float(spacing_seconds)
    return np.asarray(sorted(selected, key=lambda row: time[row]), dtype=np.int64)


def build_hazard_design(
    *,
    time_epoch: np.ndarray,
    segment: np.ndarray,
    history: np.ndarray,
    current_observation: np.ndarray,
    persistent_state: np.ndarray,
    memoryless_state: np.ndarray,
    observation_available: np.ndarray,
    onset_time: Sequence[float],
    onset_segment: Sequence[int],
    spacing_seconds: float = 300.0,
) -> HazardDesign:
    available = np.asarray(observation_available, dtype=bool)
    source = np.flatnonzero(available)
    take_local = downsample_recorded_grid(
        np.asarray(time_epoch)[source], np.asarray(segment)[source],
        spacing_seconds=float(spacing_seconds),
    )
    take = source[take_local]
    value = HazardDesign(
        source_index=take.astype(np.int64, copy=False),
        time_epoch=np.asarray(time_epoch, dtype=np.float64)[take],
        segment=np.asarray(segment, dtype=np.int64)[take],
        history=np.asarray(history, dtype=np.float64)[take],
        current_observation=np.asarray(current_observation, dtype=np.float64)[take],
        persistent_state=np.asarray(persistent_state, dtype=np.float64)[take],
        memoryless_state=np.asarray(memoryless_state, dtype=np.float64)[take],
        onset_time=np.asarray(onset_time, dtype=np.float64),
        onset_segment=np.asarray(onset_segment, dtype=np.int64),
    )
    value.validate()
    return value


def horizon_outcome(
    design: HazardDesign, horizon_minutes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Outcome plus eligibility; negative windows require complete coverage."""
    horizon = float(horizon_minutes) * 60.0
    outcome = np.zeros(len(design.time_epoch), dtype=np.int8)
    eligible = np.zeros(len(design.time_epoch), dtype=bool)
    for label in np.unique(design.segment):
        rows = np.flatnonzero(design.segment == label)
        if not len(rows):
            continue
        local_time = design.time_epoch[rows]
        onsets = np.sort(design.onset_time[design.onset_segment == label])
        next_position = np.searchsorted(onsets, local_time, side="right")
        has_next = next_position < len(onsets)
        next_onset = np.full(len(rows), np.inf, dtype=np.float64)
        next_onset[has_next] = onsets[next_position[has_next]]
        positive = has_next & (next_onset <= local_time + horizon + 1e-9)
        outcome[rows[positive]] = 1
        # Positive windows terminate at an observed onset.  For negatives the
        # entire future horizon must stay in this recorded coverage segment.
        complete_negative = local_time + horizon <= float(np.max(local_time)) + 1e-9
        eligible[rows] = positive | complete_negative
    return outcome, eligible


def preictal_mask(design: HazardDesign, horizon_minutes: float = 120.0) -> np.ndarray:
    return horizon_outcome(design, float(horizon_minutes))[0].astype(bool)


def _aggregate_fold_losses(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    if not rows:
        return {name: None for name in ("M0", "M1", "M2", "M3", "M4")}
    weight = np.asarray([row["n_test_rows"] for row in rows], dtype=np.float64)
    return {
        name: float(np.average([row[f"logloss_{name}"] for row in rows], weights=weight))
        for name in ("M0", "M1", "M2", "M3", "M4")
    }


def _fit_score(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
    alpha_grid: Sequence[float],
) -> tuple[float, float]:
    fit_x, score_x = _standardise_train_test(train_x, test_x)
    beta, alpha = _fit_with_inner_alpha(fit_x, train_y, alpha_grid)
    return _logloss(test_y, _logistic_predict(score_x, beta)), float(alpha)


def prequential_nested_hazard(
    design: HazardDesign,
    *,
    initial_k: int,
    horizon_minutes: float = 30.0,
    alpha_grid: Sequence[float] = ALPHA_GRID,
    persistent_override: np.ndarray | None = None,
) -> dict[str, Any]:
    """Nested M0--M4 risk on strictly later held-out seizures."""
    outcome, eligible = horizon_outcome(design, float(horizon_minutes))
    persistent = (
        np.asarray(persistent_override, dtype=np.float64)
        if persistent_override is not None else design.persistent_state
    )
    if persistent.shape != design.persistent_state.shape:
        raise ValueError("persistent override shape mismatch")
    ch = np.asarray(design.history[:, :min(11, design.history.shape[1])], dtype=np.float64)
    observation = np.column_stack([
        np.asarray(design.current_observation, dtype=np.float64),
        np.asarray(design.memoryless_state, dtype=np.float64),
    ])
    base = np.column_stack([ch, observation])
    matrices = {
        "M0": ch,
        "M1": base,
        "M2": np.column_stack([base, persistent]),
        "M3": base,
        "M4": np.column_stack([base, persistent - design.memoryless_state]),
    }
    onset_order = np.argsort(design.onset_time, kind="stable")
    onset_time = design.onset_time[onset_order]
    onset_segment = design.onset_segment[onset_order]
    supported = []
    for time, segment in zip(onset_time, onset_segment):
        if np.any(
            eligible & (design.segment == segment)
            & (design.time_epoch < time)
            & (design.time_epoch >= time - float(horizon_minutes) * 60.0)
        ):
            supported.append((float(time), int(segment)))
    fold_rows: list[dict[str, Any]] = []
    for position in range(int(initial_k), len(supported)):
        cutoff = float(supported[position - 1][0])
        heldout_time, heldout_segment = supported[position]
        train = np.flatnonzero(eligible & (design.time_epoch <= cutoff + 1e-9))
        test = np.flatnonzero(
            eligible & (design.time_epoch > cutoff + 1e-9)
            & (design.time_epoch <= heldout_time + 1e-9)
        )
        if len(train) < 30 or len(test) < 1 or len(np.unique(outcome[train])) < 2:
            continue
        # A training positive must be attributable to an onset already known
        # at the fold cutoff.  This assertion catches future-label leakage.
        for row in train[outcome[train] == 1]:
            known = (
                (design.onset_segment == design.segment[row])
                & (design.onset_time > design.time_epoch[row])
                & (design.onset_time <= design.time_epoch[row]
                   + float(horizon_minutes) * 60.0 + 1e-9)
                & (design.onset_time <= cutoff + 1e-9)
            )
            if not bool(np.any(known)):
                raise ValueError("prequential training label uses a future seizure")
        losses: dict[str, float] = {}
        alphas: dict[str, float] = {}
        for name, matrix in matrices.items():
            loss, alpha = _fit_score(
                matrix[train], matrix[test], outcome[train], outcome[test],
                alpha_grid,
            )
            losses[name], alphas[name] = loss, alpha
        fold_rows.append({
            "heldout_seizure_rank": int(position + 1),
            "heldout_onset_epoch": heldout_time,
            "heldout_segment": int(heldout_segment),
            "train_cutoff_epoch": cutoff,
            "n_train_rows": int(len(train)), "n_test_rows": int(len(test)),
            "n_train_positive_rows": int(np.sum(outcome[train])),
            "n_test_positive_rows": int(np.sum(outcome[test])),
            **{f"logloss_{name}": value for name, value in losses.items()},
            **{f"alpha_{name}": value for name, value in alphas.items()},
        })
    aggregate = _aggregate_fold_losses(fold_rows)
    if not fold_rows:
        return {
            "status": "NOT_ESTIMABLE", "initial_k": int(initial_k),
            "horizon_minutes": float(horizon_minutes), "folds": [],
            "n_supported_seizures": len(supported),
        }
    m0, m1, m2, m3, m4 = (aggregate[name] for name in ("M0", "M1", "M2", "M3", "M4"))
    assert all(value is not None for value in (m0, m1, m2, m3, m4))
    return {
        "status": "COMPLETE_EXPLORATORY",
        "initial_k": int(initial_k), "horizon_minutes": float(horizon_minutes),
        "n_grid_rows": int(len(design.time_epoch)),
        "n_eligible_grid_rows": int(np.sum(eligible)),
        "n_supported_seizures": len(supported),
        "n_oof_seizures": len(fold_rows), "folds": fold_rows,
        "logloss": aggregate,
        "O_relative_improvement": float((m0 - m1) / m0) if m0 > 0 else None,
        "T_M2_minus_M1": float(m2 - m1),
        "T_relative_improvement": float((m1 - m2) / m1) if m1 > 0 else None,
        "M_M4_minus_M3": float(m4 - m3),
        "M_relative_improvement": float((m3 - m4) / m3) if m3 > 0 else None,
        "T_direction_favourable": bool(m2 < m1),
        "M_direction_favourable": bool(m4 < m3),
        "patient_is_inference_unit": True,
        "seed_is_not_patient_replicate": True,
        "claim_status": "EXPLORATORY_ASSAY_NOT_SENSITIVE",
    }


def lagged_persistent_state(
    design: HazardDesign, lag_minutes: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Past-only same-segment state donor at or before t-lag."""
    lag = float(lag_minutes) * 60.0
    donor = np.full(len(design.time_epoch), -1, dtype=np.int64)
    for label in np.unique(design.segment):
        rows = np.flatnonzero(design.segment == label)
        rows = rows[np.argsort(design.time_epoch[rows], kind="stable")]
        position = np.searchsorted(
            design.time_epoch[rows], design.time_epoch[rows] - lag,
            side="right",
        ) - 1
        valid = position >= 0
        donor[rows[valid]] = rows[position[valid]]
    state = np.zeros_like(design.persistent_state)
    valid = donor >= 0
    state[valid] = design.persistent_state[donor[valid]]
    return state, valid


def patient_seed_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if str(row.get("status", "")).startswith("COMPLETE")]
    if not complete:
        return {"status": "NOT_ESTIMABLE", "n_seeds": len(rows)}
    return {
        "status": "COMPLETE_EXPLORATORY",
        "n_seeds": len(rows), "n_complete_seeds": len(complete),
        "median_T_relative_improvement": float(np.median([
            row["T_relative_improvement"] for row in complete
        ])),
        "median_M_relative_improvement": float(np.median([
            row["M_relative_improvement"] for row in complete
        ])),
        "n_T_direction_favourable": int(sum(row["T_direction_favourable"]
                                               for row in complete)),
        "n_M_direction_favourable": int(sum(row["M_direction_favourable"]
                                               for row in complete)),
        "median_n_oof_seizures": float(np.median([
            row["n_oof_seizures"] for row in complete
        ])),
    }
