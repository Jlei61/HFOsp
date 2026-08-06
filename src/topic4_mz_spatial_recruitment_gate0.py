"""Artifact-level spatial recruitment audit for the MZ additive lifecycle line.

The existing SNN capture stores only the spatial mean and maximum of the
recruitment sensor, not the full 32 x 32 sensor field.  For a non-negative
field, their ratio is a bounded effective-extent coordinate::

    rho_eff = mean(Psi(r_E_fast)) / max(Psi(r_E_fast)).

This is an amplitude-normalised soft extent, not a geometric area fraction.
The independent movie and axial summaries are therefore audited with a
participation ratio before ``rho_eff`` is used as evidence for a missing
spatial coordinate.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def effective_extent(
    mean_sensor: Sequence[float],
    peak_sensor: Sequence[float],
    *,
    zero_tolerance: float = 1.0e-12,
    bound_tolerance: float = 1.0e-6,
) -> np.ndarray:
    """Return ``mean/peak`` with explicit non-negative-field invariants.

    A positive mean with a numerically zero peak, or a mean larger than its
    peak beyond rounding tolerance, indicates an invalid or misaligned input
    contract and is rejected rather than silently clipped.
    """

    mean = np.asarray(mean_sensor, dtype=float)
    peak = np.asarray(peak_sensor, dtype=float)
    if mean.ndim != 1 or peak.shape != mean.shape or mean.size == 0:
        raise ValueError("mean_sensor and peak_sensor must be aligned non-empty 1D arrays")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(peak)):
        raise ValueError("sensor traces must be finite")
    if zero_tolerance < 0.0 or bound_tolerance < 0.0:
        raise ValueError("tolerances must be non-negative")
    if np.any(mean < -bound_tolerance) or np.any(peak < -bound_tolerance):
        raise ValueError("sensor traces must be non-negative")
    if np.any(mean > peak + bound_tolerance):
        raise ValueError("a spatial sensor mean cannot exceed its peak")
    zero_peak = peak <= float(zero_tolerance)
    if np.any(zero_peak & (mean > float(bound_tolerance))):
        raise ValueError("positive mean is incompatible with a zero sensor peak")
    extent = np.divide(mean, peak, out=np.zeros_like(mean), where=~zero_peak)
    return np.clip(extent, 0.0, 1.0)


def participation_ratio(
    frames: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Amplitude-normalised spatial participation for a stack of frames.

    For each non-negative frame ``a`` over ``N`` valid locations, this returns
    ``(sum(a)^2) / (N*sum(a^2))``.  A one-bin frame has value ``1/N`` and a
    spatially uniform non-zero frame has value one.  All-zero frames map to
    zero.  The result measures effective support, not physical area.
    """

    values = np.asarray(frames, dtype=float)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError("frames must have shape (time, spatial...)")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("frames must be finite and non-negative")
    flat = values.reshape(values.shape[0], -1)
    if valid_mask is None:
        mask = np.ones(flat.shape[1], dtype=bool)
    else:
        mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if mask.size != flat.shape[1]:
            raise ValueError("valid_mask must match the spatial frame shape")
    if not np.any(mask):
        raise ValueError("valid_mask contains no valid spatial locations")
    selected = flat[:, mask]
    numerator = np.sum(selected, axis=1) ** 2
    denominator = selected.shape[1] * np.sum(selected**2, axis=1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(selected.shape[0], dtype=float),
        where=denominator > 0.0,
    )


def frame_average_trace(
    trace: Sequence[float],
    *,
    dt_ms: float,
    frame_starts_ms: Sequence[float],
    frame_duration_ms: float,
    alignment_tolerance_samples: float = 1.0e-4,
) -> np.ndarray:
    """Average a causal native trace over exact saved frame intervals."""

    values = np.asarray(trace, dtype=float)
    starts_ms = np.asarray(frame_starts_ms, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("trace must be a non-empty finite 1D array")
    if starts_ms.ndim != 1 or starts_ms.size == 0 or not np.all(np.isfinite(starts_ms)):
        raise ValueError("frame starts must be a non-empty finite 1D array")
    if dt_ms <= 0.0 or frame_duration_ms <= 0.0 or alignment_tolerance_samples < 0.0:
        raise ValueError("invalid frame schedule")
    starts_float = starts_ms / float(dt_ms)
    duration_float = float(frame_duration_ms) / float(dt_ms)
    starts = np.rint(starts_float).astype(int)
    duration = int(round(duration_float))
    if (
        duration < 1
        or np.max(np.abs(starts_float - starts)) > alignment_tolerance_samples
        or abs(duration_float - duration) > alignment_tolerance_samples
    ):
        raise ValueError("frame schedule is not aligned to the native sampling grid")
    stops = starts + duration
    if np.any(starts < 0) or np.any(np.diff(starts) < 0) or np.any(stops > values.size):
        raise ValueError("frame schedule lies outside the native trace")
    cumulative = np.r_[0.0, np.cumsum(values, dtype=float)]
    return (cumulative[stops] - cumulative[starts]) / float(duration)


def causal_frame_end_times(
    frame_starts_ms: Sequence[float],
    *,
    frame_duration_ms: float,
) -> np.ndarray:
    """Convert saved frame starts to the first causally available times."""

    starts = np.asarray(frame_starts_ms, dtype=float)
    if starts.ndim != 1 or starts.size == 0 or not np.all(np.isfinite(starts)):
        raise ValueError("frame starts must be a non-empty finite 1D array")
    if frame_duration_ms <= 0.0 or np.any(np.diff(starts) < 0.0):
        raise ValueError("invalid frame schedule")
    return starts + float(frame_duration_ms)


def first_crossing_ms(
    values: Sequence[float],
    *,
    threshold: float,
    dt_ms: float,
) -> float | None:
    """Return the first causal sample at or above a threshold."""

    trace = np.asarray(values, dtype=float)
    if trace.ndim != 1 or trace.size == 0 or not np.all(np.isfinite(trace)):
        raise ValueError("values must be a non-empty finite 1D array")
    if not np.isfinite(threshold) or dt_ms <= 0.0:
        raise ValueError("invalid crossing contract")
    indices = np.flatnonzero(trace >= float(threshold))
    return float(indices[0] * float(dt_ms)) if indices.size else None
