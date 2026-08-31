"""Raw, event-resolved carrier readout for the frozen dual-core SNN."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from scipy.signal import find_peaks, welch


def arrays_equal_with_nan(left: np.ndarray, right: np.ndarray) -> bool:
    """Exact array equality while treating aligned missing values as equal."""
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if np.issubdtype(a.dtype, np.inexact):
        return bool(
            np.array_equal(np.isnan(a), np.isnan(b))
            and np.array_equal(a[~np.isnan(a)], b[~np.isnan(b)])
        )
    return bool(np.array_equal(a, b))


def dual_core_region_masks(
    positions: np.ndarray,
    centers_mm: np.ndarray,
    *,
    core_radius_mm: float,
    annulus_outer_radius_mm: float | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Assign neurons to two disjoint cores, two annuli, or background."""
    xy = np.asarray(positions, float)
    centers = np.asarray(centers_mm, float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("positions must have shape (neuron, 2)")
    if centers.shape != (2, 2):
        raise ValueError("centers_mm must have shape (2, 2)")
    if not np.isfinite(xy).all() or not np.isfinite(centers).all():
        raise ValueError("positions and centers must be finite")
    radius = float(core_radius_mm)
    outer = float(annulus_outer_radius_mm or 2.0 * radius)
    if not 0.0 < radius < outer:
        raise ValueError("region radii must satisfy 0 < core < annulus outer")

    distance = np.linalg.norm(
        xy[:, None, :] - centers[None, :, :], axis=2,
    )
    nearest = np.argmin(distance, axis=1)
    minimum = distance[np.arange(len(xy)), nearest]
    masks = np.column_stack([
        (nearest == 0) & (minimum <= radius),
        (nearest == 1) & (minimum <= radius),
        (nearest == 0) & (minimum > radius) & (minimum <= outer),
        (nearest == 1) & (minimum > radius) & (minimum <= outer),
        minimum > outer,
    ])
    if not np.all(masks.sum(axis=1) == 1):
        raise RuntimeError("dual-core regions must form an exact partition")
    return masks, ["core_1", "core_2", "annulus_1", "annulus_2", "background"]


def binned_group_rates_hz(
    spikes: np.ndarray,
    masks: np.ndarray,
    *,
    dt_ms: float,
    bin_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate unsmoothed boolean spikes into per-neuron group rates."""
    spike_array = np.asarray(spikes, bool)
    group_masks = np.asarray(masks, bool)
    if spike_array.ndim != 2 or group_masks.ndim != 2:
        raise ValueError("spikes and masks must both be two-dimensional")
    if spike_array.shape[1] != group_masks.shape[0]:
        raise ValueError("spike neurons and mask neurons do not align")
    if np.any(group_masks.sum(axis=0) == 0):
        raise ValueError("every carrier region must contain neurons")
    steps = int(round(float(bin_ms) / float(dt_ms)))
    if steps < 1 or not np.isclose(steps * dt_ms, bin_ms, atol=1e-12):
        raise ValueError("bin_ms must lie on the simulation time grid")
    n_bins = spike_array.shape[0] // steps
    truncated = spike_array[: n_bins * steps]
    counts = truncated.reshape(n_bins, steps, spike_array.shape[1]).sum(axis=1)
    group_counts = counts @ group_masks.astype(np.int64)
    group_sizes = group_masks.sum(axis=0).astype(np.int64)
    rates = group_counts / group_sizes[None, :] / (bin_ms * 1e-3)
    time_ms = (np.arange(n_bins, dtype=float) + 0.5) * bin_ms
    return rates.astype(np.float32), time_ms.astype(np.float32), group_sizes


def bin_continuous_trace(
    trace: np.ndarray, *, dt_ms: float, bin_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Average a continuous per-step trace without additional smoothing."""
    values = np.asarray(trace, float)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError("continuous trace must be one- or two-dimensional")
    steps = int(round(float(bin_ms) / float(dt_ms)))
    if steps < 1 or not np.isclose(steps * dt_ms, bin_ms, atol=1e-12):
        raise ValueError("bin_ms must lie on the simulation time grid")
    n_bins = len(values) // steps
    binned = values[: n_bins * steps].reshape(
        n_bins, steps, values.shape[1],
    ).mean(axis=1)
    time_ms = (np.arange(n_bins, dtype=float) + 0.5) * bin_ms
    return binned.astype(np.float32), time_ms.astype(np.float32)


def event_window_indices(
    event_onsets_ms: Sequence[float], *, trace_length: int, bin_ms: float,
    before_ms: float = 64.0, after_ms: float = 192.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed event windows and a validity mask without edge padding."""
    onsets = np.asarray(event_onsets_ms, float)
    offsets = np.arange(
        -int(round(before_ms / bin_ms)),
        int(round(after_ms / bin_ms)),
        dtype=np.int64,
    )
    centers = np.rint(onsets / bin_ms - 0.5).astype(np.int64)
    indices = centers[:, None] + offsets[None, :]
    valid = np.all((indices >= 0) & (indices < int(trace_length)), axis=1)
    return indices, valid


def _spectral_summary(signal: np.ndarray, *, fs_hz: float) -> dict:
    values = np.asarray(signal, float)
    values = values - np.mean(values)
    frequency, power = welch(
        values, fs=fs_hz, nperseg=len(values), noverlap=0, detrend=False,
    )
    carrier = (frequency >= 20.0) & (frequency <= 150.0)
    if not np.any(carrier) or float(power[carrier].sum()) <= 0.0:
        return {
            "peak_hz": None,
            "centroid_20_150_hz": None,
            "power_30_80_over_5_30": None,
            "power_80_150_over_5_30": None,
        }
    low = float(power[(frequency >= 5.0) & (frequency < 30.0)].sum())
    selected_frequency = frequency[carrier]
    selected_power = power[carrier]
    return {
        "peak_hz": float(selected_frequency[np.argmax(selected_power)]),
        "centroid_20_150_hz": float(
            np.sum(selected_frequency * selected_power) / selected_power.sum()
        ),
        "power_30_80_over_5_30": float(
            power[(frequency >= 30.0) & (frequency <= 80.0)].sum()
            / max(low, np.finfo(float).tiny)
        ),
        "power_80_150_over_5_30": float(
            power[(frequency > 80.0) & (frequency <= 150.0)].sum()
            / max(low, np.finfo(float).tiny)
        ),
    }


def raw_population_burst_summary(
    signal: np.ndarray,
    *,
    bin_ms: float,
    baseline_values: np.ndarray,
    minimum_peak_distance_ms: float = 6.0,
) -> dict:
    """Describe raw population peaks before any band-pass filtering.

    The cycle count is intentionally computed from the unsmoothed 1-ms rate.
    It is a conservative diagnostic against mistaking filter ringing for a
    native oscillation.
    """
    values = np.asarray(signal, float)
    baseline = np.asarray(baseline_values, float)
    if values.ndim != 1 or baseline.ndim != 1:
        raise ValueError("signal and baseline_values must be one-dimensional")
    baseline_center = float(np.median(baseline)) if len(baseline) else 0.0
    floor = (
        float(np.quantile(np.abs(baseline - baseline_center), 0.99))
        if len(baseline) else 0.0
    )
    event_peak = float(np.max(values, initial=0.0))
    prominence = max(floor, 0.15 * event_peak, np.finfo(float).eps)
    distance = max(1, int(round(minimum_peak_distance_ms / bin_ms)))
    peaks, properties = find_peaks(
        values, distance=distance, prominence=prominence,
    )
    intervals = np.diff(peaks) * bin_ms
    interval_frequency_hz = (
        float(1000.0 / np.median(intervals)) if len(intervals) else None
    )
    regular_cycles = False
    if len(peaks) >= 3 and len(intervals):
        mean_interval = float(np.mean(intervals))
        regular_cycles = bool(
            1000.0 / 150.0 <= mean_interval <= 1000.0 / 30.0
            and float(np.std(intervals) / mean_interval) <= 0.35
        )
    return {
        "raw_peak_count": int(len(peaks)),
        "raw_peak_indices": peaks.astype(int).tolist(),
        "raw_peak_prominences": np.asarray(
            properties.get("prominences", []), float,
        ).tolist(),
        "raw_peak_interval_frequency_hz": interval_frequency_hz,
        "regular_three_cycle_burst": regular_cycles,
        "baseline_variation_q99": floor,
        "event_peak_value": event_peak,
        **_spectral_summary(values, fs_hz=1000.0 / bin_ms),
    }


def baseline_mask_from_events(
    time_ms: np.ndarray,
    events: Sequence[Mapping],
    *,
    guard_before_ms: float = 100.0,
    guard_after_ms: float = 200.0,
) -> np.ndarray:
    """Select non-event samples for within-run carrier baselines."""
    times = np.asarray(time_ms, float)
    keep = np.ones(len(times), dtype=bool)
    for event in events:
        left = float(event["t_on_ms"]) - guard_before_ms
        right = float(event["t_off_ms"]) + guard_after_ms
        keep &= ~((times >= left) & (times <= right))
    return keep
