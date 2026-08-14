"""Pure observation and spatial-readout helpers for FCXR-LC6A trajectories."""

from __future__ import annotations

import math

import numpy as np


def observation_decision(
    *, total_ms: float, onset_ms: float | None, n_returning_ied: int,
    c0_ied_to_onset: int | None, saturated_contiguous_1s: bool,
    base_end_ms: float = 50000.0, post_onset_ms: float = 12000.0,
    hard_cap_ms: float = 65000.0, ied_multiplier: float = 1.5,
) -> dict:
    """Event-aligned continuation rule from the locked LC6A protocol."""

    total_ms = float(total_ms)
    if saturated_contiguous_1s:
        return {"continue": False, "reason": "REGISTERED_SATURATION_1S", "right_censored": False}
    if onset_ms is not None:
        target = float(onset_ms) + float(post_onset_ms)
        if total_ms >= target:
            return {"continue": False, "reason": "ONSET_PLUS_12S_OBSERVED", "right_censored": False}
        if total_ms >= float(hard_cap_ms):
            return {"continue": False, "reason": "HARD_CAP_AFTER_LATE_ONSET", "right_censored": True}
        return {"continue": True, "reason": "POST_ONSET_OBSERVATION_INCOMPLETE", "right_censored": False}
    if total_ms < float(base_end_ms):
        return {"continue": True, "reason": "NO_ONSET_BEFORE_MINIMUM", "right_censored": False}
    required = None if c0_ied_to_onset is None else int(math.ceil(
        float(ied_multiplier) * int(c0_ied_to_onset)
    ))
    if required is not None and int(n_returning_ied) >= required:
        return {
            "continue": False, "reason": "NO_ONSET_SUFFICIENT_IED_EXPOSURE",
            "right_censored": False, "required_ied_exposure": required,
        }
    if total_ms >= float(hard_cap_ms):
        return {
            "continue": False, "reason": "NO_ONSET_HARD_CAP_INSUFFICIENT_IED_EXPOSURE",
            "right_censored": True, "required_ied_exposure": required,
        }
    return {
        "continue": True, "reason": "EXTEND_FOR_IED_EXPOSURE",
        "right_censored": False, "required_ied_exposure": required,
    }


def cell_spatial_bins(positions, *, sheet_size_mm: float, n_bins_axis: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (cells, 2)")
    ij = np.floor(positions / float(sheet_size_mm) * int(n_bins_axis)).astype(int)
    ij = np.clip(ij, 0, int(n_bins_axis) - 1)
    ids = ij[:, 0] * int(n_bins_axis) + ij[:, 1]
    occupancy = np.bincount(ids, minlength=int(n_bins_axis) ** 2)
    return ids.astype(np.int32), occupancy.astype(np.int32)


def spatial_rate_maps(
    spike_steps, spike_cells, cell_bins, occupancy, *,
    n_steps: int, dt_ms: float, window_ms: float,
) -> np.ndarray:
    """Window x coarse-bin E-cell firing rates in Hz."""

    steps_per_window = int(round(float(window_ms) / float(dt_ms)))
    if steps_per_window <= 0 or n_steps % steps_per_window:
        raise ValueError("trajectory must contain an integer number of spatial windows")
    n_windows = int(n_steps // steps_per_window)
    n_bins = int(len(occupancy))
    spike_steps = np.asarray(spike_steps, np.int64)
    spike_cells = np.asarray(spike_cells, np.int64)
    time_bin = spike_steps // steps_per_window
    space_bin = np.asarray(cell_bins, np.int64)[spike_cells]
    flat = time_bin * n_bins + space_bin
    counts = np.bincount(flat, minlength=n_windows * n_bins).reshape(n_windows, n_bins)
    denominator = np.asarray(occupancy, float)[None, :] * (float(window_ms) / 1000.0)
    return np.divide(
        counts, denominator, out=np.full_like(counts, np.nan, dtype=float),
        where=denominator > 0,
    )


def per_second_cell_rates(
    spike_steps, spike_cells, *, n_steps: int, n_cells: int, dt_ms: float,
) -> np.ndarray:
    steps_per_second = int(round(1000.0 / float(dt_ms)))
    if n_steps % steps_per_second:
        raise ValueError("trajectory must contain an integer number of seconds")
    second = np.asarray(spike_steps, np.int64) // steps_per_second
    cell = np.asarray(spike_cells, np.int64)
    flat = second * int(n_cells) + cell
    return np.bincount(
        flat, minlength=(n_steps // steps_per_second) * int(n_cells),
    ).reshape(n_steps // steps_per_second, int(n_cells)).astype(float)


def local_saturation_readout(cell_rates_hz, *, refractory_ceiling_hz: float, fraction_gate: float = .05) -> dict:
    rates = np.asarray(cell_rates_hz, float)
    near = rates >= .9 * float(refractory_ceiling_hz)
    fraction = near.mean(axis=1)
    return {
        "max_near_refractory_fraction": float(fraction.max(initial=0.0)),
        "time_fraction_above_fraction_gate": float(np.mean(fraction > float(fraction_gate))),
        "per_second_near_refractory_fraction": fraction.tolist(),
        "near_refractory_rate_hz": .9 * float(refractory_ceiling_hz),
    }


def coarse_field_mean(field, cell_bins, occupancy) -> np.ndarray:
    field = np.asarray(field, float)
    total = np.bincount(cell_bins, weights=field, minlength=len(occupancy))
    return np.divide(
        total, occupancy, out=np.full_like(total, np.nan, dtype=float), where=occupancy > 0,
    )


def linear_slope(values, *, dt_s: float) -> float:
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    x = np.arange(values.size, dtype=float)[finite] * float(dt_s)
    return float(np.polyfit(x, values[finite], 1)[0])


def spatial_map_persistence(rate_maps) -> dict:
    maps = np.asarray(rate_maps, float)
    correlations = []
    for first, second in zip(maps[:-1], maps[1:]):
        finite = np.isfinite(first) & np.isfinite(second)
        if np.count_nonzero(finite) < 3 or np.std(first[finite]) == 0 or np.std(second[finite]) == 0:
            continue
        correlations.append(float(np.corrcoef(first[finite], second[finite])[0, 1]))
    return {
        "median_consecutive_correlation": (
            float(np.median(correlations)) if correlations else float("nan")
        ),
        "n_pairs": len(correlations),
    }
