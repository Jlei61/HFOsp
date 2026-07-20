"""Cheap timing/leverage race for persistence-gated additive M recovery.

This module is deliberately algebraic.  It asks whether a locked persistence
history and bounded first-order effector can overtake a precomputed fast-system
exit-current curve between the third and fifth fast cycles.  Passing this race
is necessary, not sufficient, for a full slow-fast lifecycle.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def causal_sustained_onset_ms(
    rate_hz: Sequence[float],
    *,
    dt_ms: float,
    envelope_ms: float,
    threshold_hz: float,
    minimum_duration_ms: float,
) -> tuple[float, np.ndarray]:
    """First sustained threshold crossing of a strictly trailing rate mean.

    The returned onset is the first sample of the qualifying component, not the
    later time at which the minimum-duration condition becomes certifiable.
    """

    rate = np.asarray(rate_hz, dtype=float)
    values = (dt_ms, envelope_ms, threshold_hz, minimum_duration_ms)
    if rate.ndim != 1 or rate.size == 0 or not np.all(np.isfinite(rate)):
        raise ValueError("rate_hz must be a non-empty finite 1D array")
    if not all(np.isfinite(values)) or dt_ms <= 0.0 or envelope_ms <= 0.0 or minimum_duration_ms <= 0.0:
        raise ValueError("invalid causal onset schedule")
    n_window = max(1, int(round(float(envelope_ms) / float(dt_ms))))
    index = np.arange(rate.size, dtype=int)
    start = np.maximum(0, index - n_window + 1)
    cumulative = np.r_[0.0, np.cumsum(rate, dtype=float)]
    envelope = (cumulative[index + 1] - cumulative[start]) / (index - start + 1)
    above = envelope >= float(threshold_hz)
    edges = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    minimum_samples = max(1, int(round(float(minimum_duration_ms) / float(dt_ms))))
    eligible = [int(left) for left, right in zip(starts, stops) if right - left >= minimum_samples]
    if not eligible:
        raise RuntimeError("no qualifying causal sustained state")
    return float(eligible[0] * float(dt_ms)), envelope


def compact_smoothstep(value: np.ndarray | float, low: float, high: float) -> np.ndarray:
    """C1 gate with an exact zero region and exact saturation region."""

    x = np.asarray(value, dtype=float)
    if not np.isfinite(low + high) or high < low:
        raise ValueError("compact gate requires finite high>=low")
    if np.isclose(high, low):
        return (x >= float(high)).astype(float)
    u = np.clip((x - float(low)) / (float(high) - float(low)), 0.0, 1.0)
    return np.where(x <= low, 0.0, np.where(x >= high, 1.0, 3.0 * u**2 - 2.0 * u**3))


def integrate_lowpass(
    drive: Sequence[float],
    *,
    dt_ms: float,
    tau_ms: float,
    initial: float = 0.0,
) -> np.ndarray:
    """Forward-Euler causal low-pass with inherited initial state."""

    drive = np.asarray(drive, dtype=float)
    if drive.ndim != 1 or not np.all(np.isfinite(drive)) or np.any((drive < 0.0) | (drive > 1.0)):
        raise ValueError("drive must be a finite 1D array in [0,1]")
    if dt_ms <= 0.0 or tau_ms <= dt_ms or not 0.0 <= initial <= 1.0:
        raise ValueError("invalid low-pass schedule or initial state")
    state = np.empty_like(drive)
    value = float(initial)
    for index, target in enumerate(drive):
        value += float(dt_ms) * (float(target) - value) / float(tau_ms)
        state[index] = value
    return state


def integrate_bounded_effector(
    persistence: Sequence[float],
    *,
    dt_ms: float,
    gate_low: float,
    gate_high: float,
    tau_up_ms: float,
    tau_down_ms: float,
    unsafe_decay_fraction: float,
    initial: float = 0.0,
    latch_after_first_activation: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate bounded M with exact-zero persistence gate and unsafe decay."""

    persistence = np.asarray(persistence, dtype=float)
    if persistence.ndim != 1 or not np.all(np.isfinite(persistence)):
        raise ValueError("persistence must be a finite 1D array")
    if (
        dt_ms <= 0.0
        or tau_up_ms <= dt_ms
        or tau_down_ms <= dt_ms
        or not 0.0 <= unsafe_decay_fraction <= 1.0
        or not 0.0 <= initial <= 1.0
    ):
        raise ValueError("invalid effector parameters")
    gate = compact_smoothstep(persistence, gate_low, gate_high)
    if latch_after_first_activation:
        # Diagnostic upper bound: once established-state evidence opens the
        # channel, retain that activation until a future z-safe release rule.
        # This is not the primary memoryless gate.
        gate = np.maximum.accumulate(gate)
    state = np.empty_like(persistence)
    value = float(initial)
    for index, active in enumerate(gate):
        derivative = (
            float(active) * (1.0 - value) / float(tau_up_ms)
            - (1.0 - float(active)) * float(unsafe_decay_fraction) * value / float(tau_down_ms)
        )
        value = float(np.clip(value + float(dt_ms) * derivative, 0.0, 1.0))
        state[index] = value
    return state, gate


def unopposed_z(
    time_ms: Sequence[float],
    *,
    z_start: float,
    depletion_occupancy: float,
    tau_z_ms: float,
) -> np.ndarray:
    """Cross-model timing oracle used in the previous entry/exit audit."""

    time = np.asarray(time_ms, dtype=float)
    if np.any(time < 0.0) or np.any(np.diff(time) < 0.0):
        raise ValueError("time must be non-negative and ordered")
    if not 0.0 <= depletion_occupancy <= 1.0 or tau_z_ms <= 0.0:
        raise ValueError("invalid Z oracle parameters")
    equilibrium = 1.0 - float(depletion_occupancy)
    return equilibrium + (float(z_start) - equilibrium) * np.exp(-time / float(tau_z_ms))


def required_additive_from_fold(
    z: Sequence[float],
    *,
    fold_z: Sequence[float],
    fold_additive_mv: Sequence[float],
) -> np.ndarray:
    """Monotone interpolation of the frozen fixed-point exit-current oracle."""

    z = np.asarray(z, dtype=float)
    surface_z = np.asarray(fold_z, dtype=float)
    surface_a = np.asarray(fold_additive_mv, dtype=float)
    if surface_z.ndim != 1 or surface_a.shape != surface_z.shape or surface_z.size < 2:
        raise ValueError("fold surface arrays must be aligned 1D arrays")
    order = np.argsort(surface_z)
    sorted_z = surface_z[order]
    sorted_a = surface_a[order]
    if np.any(np.diff(sorted_z) <= 0.0):
        raise ValueError("fold Z coordinates must be unique")
    if np.any(z < sorted_z[0]) or np.any(z > sorted_z[-1]):
        raise ValueError("requested Z lies outside the locked fold surface")
    return np.interp(z, sorted_z, sorted_a)


def classify_leverage_race(
    time_ms: Sequence[float],
    additive_available_mv: Sequence[float],
    additive_required_mv: Sequence[float],
    *,
    minimum_cycles: float,
    maximum_cycles: float,
    cycle_period_ms: float,
) -> dict:
    """Classify the first available-current crossing against the 3--5 cycle gate."""

    time = np.asarray(time_ms, dtype=float)
    available = np.asarray(additive_available_mv, dtype=float)
    required = np.asarray(additive_required_mv, dtype=float)
    if (
        time.ndim != 1
        or time.size == 0
        or available.shape != time.shape
        or required.shape != time.shape
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(available))
        or not np.all(np.isfinite(required))
        or np.any(np.diff(time) < 0.0)
        or minimum_cycles < 0.0
        or maximum_cycles < minimum_cycles
        or cycle_period_ms <= 0.0
    ):
        raise ValueError("race arrays must be aligned")
    margin = available - required
    previous_margin = np.r_[-np.inf, margin[:-1]]
    crossing = np.flatnonzero((margin >= 0.0) & (previous_margin < 0.0))
    crossing_index = int(crossing[0]) if crossing.size else None
    crossing_ms = float(time[crossing_index]) if crossing_index is not None else None
    if crossing_index is None:
        crossing_slope = None
    elif crossing_index == 0:
        crossing_slope = None
    else:
        delta_t = float(time[crossing_index] - time[crossing_index - 1])
        crossing_slope = (
            float((margin[crossing_index] - margin[crossing_index - 1]) / delta_t)
            if delta_t > 0.0 else None
        )
    earliest = float(minimum_cycles) * float(cycle_period_ms)
    latest = float(maximum_cycles) * float(cycle_period_ms)
    if crossing_ms is None:
        status = "insufficient_leverage"
    elif crossing_ms < earliest:
        status = "too_early_or_prevention_risk"
    elif crossing_ms <= latest:
        status = "timing_leverage_feasible"
    else:
        status = "too_late_for_registered_window"
    return {
        "status": status,
        "first_crossing_ms": crossing_ms,
        "first_crossing_cycles": (
            crossing_ms / float(cycle_period_ms) if crossing_ms is not None else None
        ),
        "crossing_margin_slope_mv_per_ms": crossing_slope,
        "registered_window_ms": [earliest, latest],
        "minimum_margin_mv": float(np.min(margin)),
        "maximum_margin_mv": float(np.max(margin)),
        "final_margin_mv": float(margin[-1]),
    }
