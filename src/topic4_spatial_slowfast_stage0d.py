"""Prospective local-basin replication helpers for the Stage-0C fast system.

This module adds no model term.  It constructs the locked phase/local-state
battery and applies fail-closed numerical and same-object gates to simulations
performed by :mod:`topic4_spatial_slowfast_stage0c_transfer`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from src.topic4_spatial_slowfast_stage0c import (
    E_CEILING_KHZ,
    I_CEILING_KHZ,
    PoolParameters,
    S_MAX,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (
    ExtendedSiegertTransfer,
    _rhs_and_moments,
    prepare_pool_parameters,
    resolution_pair_status,
    temporal_refinement_status,
)


PHASES: tuple[float, ...] = (0.0, 0.25, 0.50, 0.75)
HISTORIES: tuple[str, ...] = (
    "phase_anchor",
    "fast_plus",
    "fast_minus",
    "pool_plus",
    "pool_minus",
)
CENTRE: tuple[float, float] = (0.85, 16.0)
MANHATTAN_NEIGHBOURS: frozenset[tuple[float, float]] = frozenset(
    {(0.84, 16.0), (0.86, 16.0), (0.85, 15.0), (0.85, 17.0)}
)


def integrate_full_state_trace(
    initial_state: Sequence[float] | np.ndarray,
    params: PoolParameters,
    transfer: ExtendedSiegertTransfer,
    *,
    dt_ms: float,
    duration_ms: float,
    save_stride: int,
) -> dict[str, np.ndarray]:
    """Trace all nine states with the exact Stage-0C transfer RHS.

    This is used only to select orbit phases.  The runner independently compares
    all shared coordinates against ``simulate_extended_forks`` before accepting
    any phase state.
    """

    state = np.asarray(initial_state, dtype=float).copy()
    if state.shape != (9,) or not np.all(np.isfinite(state)):
        raise ValueError("initial_state must be finite with shape (9,)")
    params = params.validate()
    if dt_ms <= 0.0 or duration_ms <= dt_ms or save_stride < 1:
        raise ValueError("invalid integration contract")
    n_steps = int(round(duration_ms / dt_ms))
    if not np.isclose(n_steps * dt_ms, duration_ms):
        raise ValueError("duration must be an integer multiple of dt")
    sample_steps = np.arange(0, n_steps + 1, save_stride, dtype=int)
    if sample_steps[-1] != n_steps:
        sample_steps = np.r_[sample_steps, n_steps]
    trace = np.full((sample_steps.size, 9), np.nan, dtype=float)
    prepared = prepare_pool_parameters([params])
    sample_index = 0
    for step in range(n_steps + 1):
        if sample_index < sample_steps.size and step == sample_steps[sample_index]:
            trace[sample_index] = state
            sample_index += 1
        if step == n_steps:
            break
        rhs, _ = _rhs_and_moments(
            state[None, :],
            prepared,
            transfer,
            mechanism="dynamic",
            clamp_s=None,
            subtractive_beta_mv=None,
        )
        state += dt_ms * rhs[0]
        if not np.all(np.isfinite(state)):
            state[:] = np.nan
    return {
        "time_ms": sample_steps.astype(float) * dt_ms,
        "state": trace,
        "final_state": state,
    }


def select_phase_states(
    time_ms: Sequence[float] | np.ndarray,
    state_trace: np.ndarray,
    *,
    tail_start_ms: float,
    peak_height_hz: float,
    peak_prominence_hz: float,
    peak_min_distance_ms: float,
    phases: Sequence[float] = PHASES,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Select four locked phases from the last complete E-rate peak cycle."""

    time = np.asarray(time_ms, dtype=float)
    state = np.asarray(state_trace, dtype=float)
    phases_array = np.asarray(phases, dtype=float)
    if time.ndim != 1 or state.shape != (time.size, 9):
        raise ValueError("time/state trace mismatch")
    if tuple(float(value) for value in phases_array) != PHASES:
        raise ValueError("phase set drifted")
    if np.any(np.diff(time) <= 0.0) or not np.all(np.isfinite(state)):
        raise ValueError("phase trace must be finite and time increasing")
    dt = float(np.median(np.diff(time)))
    tail_indices = np.flatnonzero(time >= tail_start_ms)
    if tail_indices.size < 4:
        raise ValueError("phase-source tail is too short")
    tail_start = int(tail_indices[0])
    peaks, properties = find_peaks(
        1000.0 * state[tail_start:, 0],
        height=peak_height_hz,
        prominence=peak_prominence_hz,
        distance=max(1, int(np.ceil(peak_min_distance_ms / dt))),
    )
    peaks = peaks + tail_start
    if peaks.size < 2:
        raise ValueError("fewer than two qualifying phase-source peaks")
    start_index, stop_index = int(peaks[-2]), int(peaks[-1])
    if stop_index <= start_index + 4:
        raise ValueError("last phase-source cycle is too short")
    selected_indices = []
    rows: list[dict[str, Any]] = []
    for phase in phases_array:
        target = time[start_index] + float(phase) * (time[stop_index] - time[start_index])
        index = int(np.argmin(np.abs(time - target)))
        selected_indices.append(index)
        rows.append(
            {
                "phase_id": f"phase_{int(round(100 * phase)):03d}",
                "phase_fraction": float(phase),
                "state_index": index,
                "time_ms": float(time[index]),
                "rE_hz": float(1000.0 * state[index, 0]),
                "cycle_start_ms": float(time[start_index]),
                "cycle_stop_ms": float(time[stop_index]),
                "cycle_period_ms": float(time[stop_index] - time[start_index]),
                "n_qualifying_peaks": int(peaks.size),
                "last_peak_height_hz": float(properties["peak_heights"][-1]),
                "last_peak_prominence_hz": float(properties["prominences"][-1]),
            }
        )
    if len(set(selected_indices)) != len(PHASES):
        raise ValueError("phase selections are not distinct")
    return state[selected_indices].copy(), rows


def _validate_natural_state_bounds(state: np.ndarray) -> None:
    if state.shape != (9,) or not np.all(np.isfinite(state)) or np.any(state < 0.0):
        raise ValueError("battery state must be finite and nonnegative")
    if state[0] > E_CEILING_KHZ or state[1] > I_CEILING_KHZ:
        raise ValueError("battery state exceeds refractory rate ceiling")
    if np.any(state[[2, 4, 6]] > E_CEILING_KHZ) or np.any(state[[3, 5]] > I_CEILING_KHZ):
        raise ValueError("battery synaptic/history state exceeds rate ceiling")
    if state[7] > 1.0 or state[8] > S_MAX:
        raise ValueError("battery pool state exceeds natural bounds")


def build_local_battery(
    phase_states: np.ndarray,
    points: Sequence[tuple[float, float]],
    *,
    perturbation_fraction: float = 0.03,
) -> tuple[list[dict[str, Any]], np.ndarray, list[PoolParameters]]:
    """Build the locked 4-phase x 5-history battery at each parameter point."""

    phase_states = np.asarray(phase_states, dtype=float)
    if phase_states.shape != (4, 9):
        raise ValueError("phase_states must have shape (4,9)")
    if not np.isclose(perturbation_fraction, 0.03):
        raise ValueError("perturbation fraction drifted")
    metadata: list[dict[str, Any]] = []
    states: list[np.ndarray] = []
    params: list[PoolParameters] = []
    for z, alpha in points:
        point = PoolParameters(float(z), float(alpha), 1.1, 1.0).validate()
        for phase_index, phase in enumerate(PHASES):
            anchor = phase_states[phase_index]
            variants: dict[str, np.ndarray] = {"phase_anchor": anchor.copy()}
            for family, indices in (("fast", np.arange(0, 6)), ("pool", np.arange(6, 9))):
                for sign, factor in (("plus", 1.0 + perturbation_fraction), ("minus", 1.0 - perturbation_fraction)):
                    changed = anchor.copy()
                    changed[indices] *= factor
                    variants[f"{family}_{sign}"] = changed
            if tuple(variants) != HISTORIES:
                raise RuntimeError("history ordering drifted")
            for history, state in variants.items():
                _validate_natural_state_bounds(state)
                family = "anchor" if history == "phase_anchor" else history.split("_", 1)[0]
                metadata.append(
                    {
                        "z": point.z,
                        "alpha_G": point.alpha_g,
                        "w_ee_mult": point.w_ee_mult,
                        "ratio": point.ratio,
                        "phase_id": f"phase_{int(round(100 * phase)):03d}",
                        "phase_fraction": phase,
                        "history": history,
                        "perturbation_family": family,
                        "off_orbit": history != "phase_anchor",
                    }
                )
                states.append(state)
                params.append(point)
    return metadata, np.asarray(states, dtype=float), params


def audited_single_resolution_status(row: Mapping[str, Any], *, exact_error_pass: bool) -> str:
    """Fail-closed screen/confirm status at one extra-fine transfer resolution."""

    return resolution_pair_status(row, row, exact_error_pass=exact_error_pass)


def temporal_amplitude_status(
    confirm_row: Mapping[str, Any],
    refined_row: Mapping[str, Any],
    *,
    exact_error_pass: bool,
    amplitude_abs_hz: float = 5.0,
    amplitude_relative: float = 0.10,
) -> str:
    """Existing dt/2 gate plus the prospective amplitude-replication gate."""

    status = temporal_refinement_status(confirm_row, refined_row, exact_error_pass=exact_error_pass)
    if status != "candidate_survives":
        return status
    amplitudes = np.asarray(
        [
            float(confirm_row["tail_peak_hz"]) - float(confirm_row["tail_trough_hz"]),
            float(refined_row["tail_peak_hz"]) - float(refined_row["tail_trough_hz"]),
        ]
    )
    if not np.all(np.isfinite(amplitudes)) or float(np.ptp(amplitudes)) > max(
        amplitude_abs_hz, amplitude_relative * float(np.mean(amplitudes))
    ):
        return "numerical_unresolved"
    return "candidate_survives"


def metric_agreement(rows: Sequence[Mapping[str, Any]]) -> tuple[bool, dict[str, float | bool | None]]:
    """Apply locked within-object rate, frequency, and amplitude ranges."""

    if not rows:
        return False, {
            "mean_rate_hz": None,
            "mean_frequency_hz": None,
            "mean_amplitude_hz": None,
            "rate_agreement": False,
            "frequency_agreement": False,
            "amplitude_agreement": False,
        }
    rates = np.asarray([float(row["dt_half_tail_mean_hz"]) for row in rows])
    frequencies = np.asarray([float(row["dt_half_frequency_hz"]) for row in rows])
    amplitudes = np.asarray([float(row["dt_half_amplitude_hz"]) for row in rows])
    rate_ok = bool(float(np.ptp(rates)) <= max(1.0, 0.10 * float(np.mean(rates))))
    frequency_ok = bool(float(np.ptp(frequencies)) <= max(0.25, 0.10 * float(np.mean(frequencies))))
    amplitude_ok = bool(float(np.ptp(amplitudes)) <= max(5.0, 0.10 * float(np.mean(amplitudes))))
    metrics: dict[str, float | bool | None] = {
        "mean_rate_hz": float(np.mean(rates)),
        "mean_frequency_hz": float(np.mean(frequencies)),
        "mean_amplitude_hz": float(np.mean(amplitudes)),
        "range_rate_hz": float(np.ptp(rates)),
        "range_frequency_hz": float(np.ptp(frequencies)),
        "range_amplitude_hz": float(np.ptp(amplitudes)),
        "rate_agreement": rate_ok,
        "frequency_agreement": frequency_ok,
        "amplitude_agreement": amplitude_ok,
    }
    return bool(rate_ok and frequency_ok and amplitude_ok), metrics


def summarize_parameter_point(rows: Sequence[Mapping[str, Any]], z: float, alpha_g: float) -> dict[str, Any]:
    """Determine open-basin support without counting phase anchors."""

    members = [row for row in rows if np.isclose(float(row["z"]), z) and np.isclose(float(row["alpha_G"]), alpha_g)]
    survivors = [row for row in members if row["final_status"] == "candidate_survives"]
    off_orbit = [row for row in survivors if bool(row["off_orbit"])]
    families = sorted({str(row["perturbation_family"]) for row in off_orbit})
    phases = sorted({str(row["phase_id"]) for row in off_orbit})
    agreement, metrics = metric_agreement(off_orbit)
    open_basin = bool(len(off_orbit) >= 2 and len(families) >= 2 and len(phases) >= 2 and agreement)
    status_counts: dict[str, int] = {}
    for row in members:
        key = str(row["final_status"])
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "z": float(z),
        "alpha_G": float(alpha_g),
        "n_histories": len(members),
        "status_counts": status_counts,
        "n_survivors": len(survivors),
        "n_off_orbit_survivors": len(off_orbit),
        "surviving_families": families,
        "surviving_phase_ids": phases,
        "same_object_agreement": agreement,
        **metrics,
        "open_local_basin_support": open_basin,
        "is_centre": bool(np.isclose(z, CENTRE[0]) and np.isclose(alpha_g, CENTRE[1])),
        "is_manhattan_neighbour": (round(float(z), 2), float(alpha_g)) in MANHATTAN_NEIGHBOURS,
    }


def point_metric_compatibility(centre: Mapping[str, Any], neighbour: Mapping[str, Any]) -> bool:
    """Require neighbouring basin centroid to match the centre object."""

    checks = []
    for key, absolute, relative in (
        ("mean_rate_hz", 1.0, 0.10),
        ("mean_frequency_hz", 0.25, 0.10),
        ("mean_amplitude_hz", 5.0, 0.10),
    ):
        pair = np.asarray([centre[key], neighbour[key]], dtype=float)
        checks.append(bool(np.all(np.isfinite(pair)) and float(np.ptp(pair)) <= max(absolute, relative * float(np.mean(pair)))))
    return bool(all(checks))
