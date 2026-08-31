"""Outcome-blind Q6 diagnostics for the H2b v0.3 state instrument.

The response in this module is the *next interictal event*, never seizure
risk.  Every row is evaluated strictly after the rows used to fit it.  The
base design already contains deterministic event history, the current-window
memoryless decoder and validated clock/recording nuisances; the only added
columns are the persistent-minus-memoryless decoder residual.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from src.topic5_continuous_marked_state_r1.r1_2 import FullAnchorDesign

if TYPE_CHECKING:
    from .v03_instrument import InterictalStateTrace


RIDGE_GRID = (0.1, 1.0, 10.0, 100.0)


def local_clock_features(epoch: np.ndarray, timezone_name: str) -> np.ndarray:
    """Exact local clock phase and recording day; DST is handled per row."""
    time = np.asarray(epoch, dtype=np.float64)
    zone = ZoneInfo(str(timezone_name))
    local = [datetime.fromtimestamp(float(value), tz=zone) for value in time]
    seconds = np.asarray([
        value.hour * 3600.0 + value.minute * 60.0 + value.second
        + value.microsecond / 1e6 for value in local
    ])
    phase = 2.0 * np.pi * seconds / 86400.0
    ordinal = np.asarray([value.date().toordinal() for value in local], dtype=np.float64)
    ordinal -= np.min(ordinal) if len(ordinal) else 0.0
    return np.column_stack([np.sin(phase), np.cos(phase), ordinal])


def prior_seizure_features(
    epoch: np.ndarray, seizure_onsets: Sequence[float],
) -> np.ndarray:
    """Use only seizures strictly earlier than each query time."""
    time = np.asarray(epoch, dtype=np.float64)
    onset = np.sort(np.asarray(seizure_onsets, dtype=np.float64))
    if not len(onset):
        return np.zeros((len(time), 2), dtype=np.float64)
    position = np.searchsorted(onset, time, side="left") - 1
    has_previous = position >= 0
    safe = np.clip(position, 0, len(onset) - 1)
    age_minutes = np.zeros(len(time), dtype=np.float64)
    age_minutes[has_previous] = np.maximum(
        time[has_previous] - onset[safe[has_previous]], 0.0,
    ) / 60.0
    return np.column_stack([
        has_previous.astype(np.float64), np.log1p(age_minutes),
    ])


def _session_one_hot(session: np.ndarray) -> np.ndarray:
    group = np.asarray(session, dtype=np.int64)
    labels = np.unique(group)
    if len(labels) <= 1:
        return np.empty((len(group), 0), dtype=np.float64)
    return np.column_stack([(group == value).astype(np.float64) for value in labels[1:]])


def _standardise_fit_apply(
    train: np.ndarray, test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    active = np.isfinite(scale) & (scale > 1e-8)
    if not bool(active.any()):
        return (
            np.empty((len(train), 0), dtype=np.float64),
            np.empty((len(test), 0), dtype=np.float64), active,
        )
    return (
        (train[:, active] - centre[active]) / scale[active],
        (test[:, active] - centre[active]) / scale[active],
        active,
    )


def _ridge_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float,
) -> np.ndarray:
    """Stable multi-response ridge via the thin SVD."""
    if train_x.shape[1] == 0:
        return np.repeat(np.mean(train_y, axis=0, keepdims=True), len(test_x), axis=0)
    u, singular, vt = np.linalg.svd(train_x, full_matrices=False)
    shrink = singular / (singular ** 2 + float(alpha))
    coefficient = (vt.T * shrink) @ (u.T @ train_y)
    return test_x @ coefficient


def _scaled_error(
    train_y: np.ndarray, test_y: np.ndarray, prediction: np.ndarray,
) -> tuple[float, np.ndarray]:
    centre = np.mean(train_y, axis=0, dtype=np.float64)
    scale = np.std(train_y, axis=0, dtype=np.float64)
    active = np.isfinite(scale) & (scale > 1e-8)
    if not bool(active.any()):
        return float("nan"), np.empty(0, dtype=np.float64)
    residual = (test_y[:, active] - prediction[:, active]) / scale[active]
    active_error = np.mean(residual ** 2, axis=0)
    per_dimension = np.full(train_y.shape[1], np.nan, dtype=np.float64)
    per_dimension[active] = active_error
    return float(np.mean(active_error)), per_dimension


def _select_alpha(
    x: np.ndarray, y: np.ndarray, *, ridge_grid: Sequence[float],
) -> float:
    n = len(x)
    split = max(20, int(np.floor(0.8 * n)))
    if split >= n - 10:
        return float(ridge_grid[len(ridge_grid) // 2])
    train_x, inner_x, _ = _standardise_fit_apply(x[:split], x[split:])
    train_y = y[:split]
    centre = np.mean(train_y, axis=0, dtype=np.float64)
    prediction_rows = []
    for alpha in ridge_grid:
        prediction = _ridge_predict(
            train_x, train_y - centre, inner_x, float(alpha),
        ) + centre
        error, _ = _scaled_error(train_y, y[split:], prediction)
        prediction_rows.append((float(error), float(alpha)))
    finite = [row for row in prediction_rows if np.isfinite(row[0])]
    return min(finite)[1] if finite else float(ridge_grid[len(ridge_grid) // 2])


def nested_prequential_increment(
    base: np.ndarray,
    increment: np.ndarray,
    target: np.ndarray,
    time_epoch: np.ndarray,
    *,
    family_slices: dict[str, slice] | None = None,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_outer_folds: int = 3,
) -> dict[str, Any]:
    """Compare nested ridge designs on strictly later chronological blocks."""
    x0 = np.asarray(base, dtype=np.float64)
    delta = np.asarray(increment, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    time = np.asarray(time_epoch, dtype=np.float64)
    if not (len(x0) == len(delta) == len(y) == len(time)):
        raise ValueError("Q6 arrays disagree")
    finite = (
        np.isfinite(x0).all(axis=1) & np.isfinite(delta).all(axis=1)
        & np.isfinite(y).all(axis=1) & np.isfinite(time)
    )
    order = np.flatnonzero(finite)
    order = order[np.argsort(time[order], kind="stable")]
    x0, delta, y, time = x0[order], delta[order], y[order], time[order]
    # Ridge permits p >= n mathematically, but that regime made the earliest
    # fold a high-variance capacity test instead of a nuisance adjustment.
    # Require at least 20 more chronological rows than the full raw design.
    minimum_train = max(
        60,
        int(np.floor(0.40 * len(order))),
        int(x0.shape[1] + delta.shape[1] + 20),
    )
    remaining = len(order) - minimum_train
    if len(order) < 100 or remaining < 30:
        return {
            "status": "NOT_ESTIMABLE_TOO_FEW_INTERICTAL_EVENTS",
            "n_rows": int(len(order)), "minimum_train_rows": int(minimum_train),
            "folds": [], "pass": False,
        }
    boundaries = np.linspace(
        minimum_train, len(order), min(int(n_outer_folds), max(1, remaining // 20)) + 1,
    ).round().astype(int)
    rows: list[dict[str, Any]] = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop - start < 10:
            continue
        outer_train = np.arange(start)
        outer_test = np.arange(start, stop)
        base_train, base_test, base_active = _standardise_fit_apply(
            x0[outer_train], x0[outer_test],
        )
        full_raw = np.column_stack([x0, delta])
        full_train, full_test, full_active = _standardise_fit_apply(
            full_raw[outer_train], full_raw[outer_test],
        )
        alpha_base = _select_alpha(
            x0[outer_train], y[outer_train], ridge_grid=ridge_grid,
        )
        alpha_full = _select_alpha(
            full_raw[outer_train], y[outer_train], ridge_grid=ridge_grid,
        )
        target_centre = np.mean(y[outer_train], axis=0, dtype=np.float64)
        pred_base = _ridge_predict(
            base_train, y[outer_train] - target_centre, base_test, alpha_base,
        ) + target_centre
        pred_full = _ridge_predict(
            full_train, y[outer_train] - target_centre, full_test, alpha_full,
        ) + target_centre
        base_error, base_by_dim = _scaled_error(y[outer_train], y[outer_test], pred_base)
        full_error, full_by_dim = _scaled_error(y[outer_train], y[outer_test], pred_full)
        relative = (
            (base_error - full_error) / base_error
            if np.isfinite(base_error) and base_error > 1e-12 else None
        )
        family = {}
        for name, selection in (family_slices or {}).items():
            selected = np.arange(y.shape[1])[selection]
            selected = selected[
                (selected < len(base_by_dim))
                & np.isfinite(base_by_dim[selected]) & np.isfinite(full_by_dim[selected])
            ]
            if len(selected):
                b = float(np.mean(base_by_dim[selected]))
                f = float(np.mean(full_by_dim[selected]))
                family[name] = {
                    "base_mse": b, "full_mse": f,
                    "relative_improvement": (b - f) / b if b > 1e-12 else None,
                }
        rows.append({
            "train_rows": int(len(outer_train)), "test_rows": int(len(outer_test)),
            "train_stop_epoch": float(time[start - 1]),
            "test_start_epoch": float(time[start]),
            "test_stop_epoch": float(time[stop - 1]),
            "base_active_dimensions": int(np.sum(base_active)),
            "full_active_dimensions": int(np.sum(full_active)),
            "base_alpha": alpha_base, "full_alpha": alpha_full,
            "base_mse": base_error, "full_mse": full_error,
            "relative_improvement": relative,
            "direction_favourable": bool(relative is not None and relative > 0),
            "families": family,
        })
    relative = [float(row["relative_improvement"]) for row in rows
                if row["relative_improvement"] is not None]
    favourable = sum(row["direction_favourable"] for row in rows)
    passed = bool(
        len(relative) >= 2 and favourable >= int(np.ceil(len(relative) / 2))
        and float(np.median(relative)) > 0.0
    )
    return {
        "status": "COMPLETE" if rows else "NOT_ESTIMABLE_NO_VALID_FOLDS",
        "n_rows": int(len(order)), "minimum_train_rows": int(minimum_train),
        "n_folds": len(rows), "folds": rows,
        "median_relative_improvement": (
            float(np.median(relative)) if relative else None
        ),
        "n_direction_favourable": int(favourable),
        "pass": passed,
        "interpretation": (
            "strictly-later next-IED prediction after current-window, event-history, "
            "clock, recording-day, segment and available prior-seizure adjustment"
        ),
    }


def build_nonoverlap_future_targets(
    design: FullAnchorDesign,
    validation_anchor: np.ndarray,
    *,
    horizon_seconds: float = 300.0,
) -> dict[str, np.ndarray]:
    """One causal, non-overlapping future-IED target per covered time window."""
    valid = np.asarray(validation_anchor, dtype=bool)
    cuts: list[int] = []
    count: list[float] = []
    first_delay: list[float] = []
    first_event: list[int] = []
    for label in np.unique(design.anchor_session[valid]):
        rows = np.flatnonzero(valid & (design.anchor_session == label))
        rows = rows[np.argsort(design.anchor_time[rows], kind="stable")]
        if len(rows) < 2:
            continue
        cursor = float(design.anchor_time[rows[0]])
        while cursor + float(horizon_seconds) <= float(design.anchor_time[rows[-1]]) + 1e-9:
            position = int(np.searchsorted(design.anchor_time[rows], cursor, side="left"))
            if position >= len(rows):
                break
            cut = int(rows[position])
            start = float(design.anchor_time[cut])
            stop = start + float(horizon_seconds)
            covered = rows[
                (design.anchor_time[rows] >= start)
                & (design.anchor_time[rows] <= stop)
            ]
            # The frozen anchor grid is 30 s.  Do not bridge an unrecorded gap
            # merely because the upstream continuity label happens to match.
            complete = bool(
                len(covered) >= 2
                and float(design.anchor_time[covered[-1]]) >= stop - 30.0 - 1e-9
                and np.max(np.diff(design.anchor_time[covered])) <= 90.0
            )
            if complete:
                event = np.flatnonzero(
                    (design.event_split == 1)
                    & (design.event_session == label)
                    & (design.event_time > start)
                    & (design.event_time <= stop)
                )
                cuts.append(cut)
                count.append(float(len(event)))
                if len(event):
                    chosen = int(event[np.argmin(design.event_time[event])])
                    first_event.append(chosen)
                    first_delay.append(float(design.event_time[chosen] - start))
                else:
                    first_event.append(-1)
                    first_delay.append(float(horizon_seconds))
            cursor = start + float(horizon_seconds)
    cut_array = np.asarray(cuts, dtype=np.int64)
    event_array = np.asarray(first_event, dtype=np.int64)
    return {
        "anchor": cut_array,
        "time": np.asarray(design.anchor_time[cut_array], dtype=np.float64),
        "session": np.asarray(design.anchor_session[cut_array], dtype=np.int64),
        "event_count": np.asarray(count, dtype=np.float64),
        "first_event_delay_seconds": np.asarray(first_delay, dtype=np.float64),
        "first_event": event_array,
        "has_event": event_array >= 0,
    }


def interictal_q6_diagnostic(
    design: FullAnchorDesign,
    trace: InterictalStateTrace,
    validation_anchor: np.ndarray,
    *,
    timezone_name: str,
    past_seizure_onsets: Sequence[float] = (),
) -> dict[str, Any]:
    """Build and score Q6 on non-overlapping future interictal windows."""
    targets = build_nonoverlap_future_targets(design, validation_anchor)
    anchor = targets["anchor"]
    if not len(anchor):
        return {
            "status": "NOT_ESTIMABLE_NO_COMPLETE_D_STATE_WINDOWS", "pass": False,
            "future_seizure_risk_outcome_read": False,
        }
    anchor_time = trace.anchor_time[anchor]
    session = trace.anchor_session[anchor]
    clock = local_clock_features(anchor_time, timezone_name)
    prior = prior_seizure_features(anchor_time, past_seizure_onsets)
    segment = _session_one_hot(session)
    base = np.column_stack([
        np.asarray(design.anchor_history[anchor], dtype=np.float64),
        np.asarray(trace.memoryless_decoder[anchor], dtype=np.float64),
        clock, prior, segment,
    ])
    increment = np.asarray(
        trace.persistent_decoder[anchor] - trace.memoryless_decoder[anchor],
        dtype=np.float64,
    )
    timing_target = np.column_stack([
        np.log1p(targets["event_count"]),
        np.log1p(targets["first_event_delay_seconds"]),
    ])
    timing = nested_prequential_increment(
        base, increment, timing_target, targets["time"],
        family_slices={"count": slice(0, 1), "next_event_delay": slice(1, 2)},
    )
    has_event = targets["has_event"]
    first_event = targets["first_event"][has_event]
    group_id = np.asarray(design.event_group_ids[first_event], dtype=np.int64)
    recruited = group_id >= 0
    if len(group_id):
        n_contacts = group_id.shape[1]
        extent = np.mean(recruited, axis=1, keepdims=True)
        group_count = (
            np.asarray(design.event_group_count[first_event], dtype=np.float64)[:, None]
            / max(n_contacts, 1)
        )
        order = np.where(
            recruited,
            group_id / np.maximum(
                np.asarray(
                    design.event_group_count[first_event], dtype=np.float64,
                )[:, None] - 1.0,
                1.0,
            ),
            0.0,
        )
        mark_target = np.column_stack([
            extent, group_count, recruited.astype(np.float64), order,
        ])
        mark = nested_prequential_increment(
            base[has_event], increment[has_event], mark_target,
            targets["time"][has_event],
            family_slices={
                "extent_and_group_count": slice(0, 2),
                "contact_subset": slice(2, 2 + n_contacts),
                "contact_order": slice(2 + n_contacts, 2 + 2 * n_contacts),
            },
        )
    else:
        mark = {
            "status": "NOT_ESTIMABLE_NO_FUTURE_EVENTS", "pass": False,
            "n_rows": 0, "folds": [],
        }
    result = {
        "status": (
            "COMPLETE" if "COMPLETE" in {timing["status"], mark["status"]}
            else "NOT_ESTIMABLE"
        ),
        "pass": bool(timing.get("pass") or mark.get("pass")),
        "timing": timing,
        "mark": mark,
        "n_nonoverlap_windows": int(len(anchor)),
        "n_windows_with_event": int(np.sum(has_event)),
        "window_minutes": 5.0,
        "timezone": str(timezone_name),
        "n_available_prior_seizures": int(len(past_seizure_onsets)),
        "past_seizure_nuisance_read": bool(len(past_seizure_onsets)),
        "future_seizure_risk_outcome_read": False,
        "validated_sleep_wake_available": False,
        "validated_medication_or_stimulation_available": False,
        "outcome": (
            "future 5-min IED count and time-to-next-event (including no-event windows); "
            "first-event tied-group contact expression"
        ),
        "design": "C+H+memoryless decoder versus same plus persistent residual",
        "windows_overlap": False,
        "cross_gap_windows_allowed": False,
    }
    return result
