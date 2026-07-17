"""Pure helpers for baseline-referenced peri-onset energy timing.

The timing detector is deliberately scaffold-label blind.  It consumes a
contact-by-time baseline-normalized band-power matrix, subtracts a distal
per-contact baseline, summarizes the fixed contact set with a spatial
quantile, and detects the first sustained excursion above an empirical distal
baseline threshold.

This module contains no repository I/O so the pilot runner and its future
cohort extension share one tested timing contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TimingResult:
    detected: bool
    rise_sec: float
    peak_sec: float
    peak_value: float
    threshold: float
    baseline_median: float
    baseline_q: float
    total_above_sec: float
    longest_above_sec: float
    n_baseline_frames: int
    n_search_frames: int


@dataclass(frozen=True)
class TransitionResult:
    detected: bool
    transition_sec: float
    step_delta: float
    threshold: float
    baseline_q: float
    flank_sec: float
    n_baseline_centers: int
    n_search_centers: int


@dataclass(frozen=True)
class RecruitmentOnsetResult:
    detected: bool
    onset_sec: float
    step_delta: float
    step_threshold: float
    consensus_post_sustained: bool
    consensus_post_peak: float
    consensus_level_threshold: float
    n_band_post_sustained: int
    n_bands: int
    majority_required: int


def distal_baseline_delta(
    z_contact_time: np.ndarray,
    rel_t: np.ndarray,
    *,
    baseline: tuple[float, float] = (-120.0, -90.0),
) -> np.ndarray:
    """Subtract each contact's distal-baseline median from its full trace."""
    z = np.asarray(z_contact_time, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if z.ndim != 2:
        raise ValueError("z_contact_time must be [contact, time]")
    if t.ndim != 1 or z.shape[1] != t.size:
        raise ValueError("rel_t must be 1D and match the time dimension")
    mask = (t >= float(baseline[0])) & (t < float(baseline[1]))
    if not np.any(mask):
        raise ValueError(f"no frames in distal baseline {baseline}")
    with np.errstate(invalid="ignore"):
        center = np.nanmedian(z[:, mask], axis=1, keepdims=True)
    return z - center


def spatial_quantile_trace(z_contact_time: np.ndarray, *, q: float = 0.75) -> np.ndarray:
    """Return the per-time spatial quantile over a fixed contact set."""
    z = np.asarray(z_contact_time, dtype=float)
    if z.ndim != 2:
        raise ValueError("z_contact_time must be [contact, time]")
    if not 0.0 <= float(q) <= 1.0:
        raise ValueError("q must lie in [0, 1]")
    with np.errstate(invalid="ignore"):
        return np.nanquantile(z, float(q), axis=0)


def smooth_trace(trace: np.ndarray, rel_t: np.ndarray, *, smooth_sec: float = 2.0) -> np.ndarray:
    """NaN-aware centered moving average on the native time grid."""
    y = np.asarray(trace, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.size != t.size:
        raise ValueError("trace and rel_t must be matching 1D arrays")
    if y.size < 2 or float(smooth_sec) <= 0.0:
        return y.copy()
    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("rel_t must be strictly increasing")
    width = max(1, int(round(float(smooth_sec) / dt)))
    if width <= 1:
        return y.copy()
    kernel = np.ones(width, dtype=float)
    finite = np.isfinite(y)
    num = np.convolve(np.where(finite, y, 0.0), kernel, mode="same")
    den = np.convolve(finite.astype(float), kernel, mode="same")
    return np.divide(num, den, out=np.full_like(y, np.nan), where=den > 0)


def _run_lengths(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive-exclusive runs of True values."""
    x = np.asarray(mask, dtype=bool)
    padded = np.r_[False, x, False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(a), int(b)) for a, b in edges.reshape(-1, 2)]


def detect_sustained_enhancement(
    trace: np.ndarray,
    rel_t: np.ndarray,
    *,
    baseline: tuple[float, float] = (-120.0, -90.0),
    search: tuple[float, float] = (-60.0, 20.0),
    baseline_quantile: float = 0.99,
    sustain_sec: float = 2.0,
) -> TimingResult:
    """Detect the first sustained search-window excursion above baseline.

    ``trace`` is expected to be the already-smoothed, distal-baseline-referenced
    spatial summary.  The threshold is the empirical quantile of the same
    trace in the distal baseline.  A crossing is accepted only when it remains
    above threshold for at least ``sustain_sec`` on the native grid.
    """
    y = np.asarray(trace, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.size != t.size:
        raise ValueError("trace and rel_t must be matching 1D arrays")
    if y.size < 2:
        raise ValueError("at least two time frames are required")
    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("rel_t must be strictly increasing")
    bmask = (t >= float(baseline[0])) & (t < float(baseline[1])) & np.isfinite(y)
    smask = (t >= float(search[0])) & (t <= float(search[1])) & np.isfinite(y)
    if not np.any(bmask):
        raise ValueError(f"no finite frames in distal baseline {baseline}")
    if not np.any(smask):
        raise ValueError(f"no finite frames in search window {search}")

    bvals = y[bmask]
    threshold = float(np.nanquantile(bvals, float(baseline_quantile)))
    baseline_median = float(np.nanmedian(bvals))
    search_idx = np.flatnonzero(smask)
    above_local = y[search_idx] > threshold
    min_frames = max(1, int(np.ceil(float(sustain_sec) / dt)))
    runs = _run_lengths(above_local)
    accepted = [(a, b) for a, b in runs if (b - a) >= min_frames]
    detected = bool(accepted)
    rise_sec = float(t[search_idx[accepted[0][0]]]) if detected else float("nan")

    search_y = y[search_idx]
    peak_local = int(np.nanargmax(search_y))
    peak_sec = float(t[search_idx[peak_local]])
    peak_value = float(search_y[peak_local])
    total_above_sec = float(sum(b - a for a, b in runs) * dt)
    longest_above_sec = float(max((b - a for a, b in runs), default=0) * dt)
    return TimingResult(
        detected=detected,
        rise_sec=rise_sec,
        peak_sec=peak_sec,
        peak_value=peak_value,
        threshold=threshold,
        baseline_median=baseline_median,
        baseline_q=float(baseline_quantile),
        total_above_sec=total_above_sec,
        longest_above_sec=longest_above_sec,
        n_baseline_frames=int(np.sum(bmask)),
        n_search_frames=int(np.sum(smask)),
    )


def detect_centered_window_enhancement(
    trace: np.ndarray,
    rel_t: np.ndarray,
    *,
    center_sec: float,
    half_width_sec: float = 5.0,
    baseline: tuple[float, float] = (-120.0, -90.0),
    baseline_quantile: float = 0.99,
    sustain_sec: float = 2.0,
) -> TimingResult:
    """Test a pre-specified window for sustained baseline-extreme energy.

    This wrapper makes the onset-alignment question explicit: the window is
    fixed before inspecting the trace and the threshold is still learned only
    from the distal baseline.  It is therefore suitable for comparing the
    true EEG-onset window with identically sized clinical- or pseudo-onset
    windows.
    """
    half = float(half_width_sec)
    if not np.isfinite(half) or half <= 0.0:
        raise ValueError("half_width_sec must be finite and positive")
    center = float(center_sec)
    if not np.isfinite(center):
        raise ValueError("center_sec must be finite")
    return detect_sustained_enhancement(
        trace,
        rel_t,
        baseline=baseline,
        search=(center - half, center + half),
        baseline_quantile=baseline_quantile,
        sustain_sec=sustain_sec,
    )


def centered_window_hit_profile(
    trace: np.ndarray,
    rel_t: np.ndarray,
    centers_sec: np.ndarray,
    *,
    half_width_sec: float = 5.0,
    baseline: tuple[float, float] = (-120.0, -90.0),
    baseline_quantile: float = 0.99,
    sustain_sec: float = 2.0,
) -> np.ndarray:
    """Return one sustained-extreme hit flag for every supplied center."""
    centers = np.asarray(centers_sec, dtype=float)
    if centers.ndim != 1:
        raise ValueError("centers_sec must be one-dimensional")
    return np.asarray(
        [
            detect_centered_window_enhancement(
                trace,
                rel_t,
                center_sec=float(center),
                half_width_sec=half_width_sec,
                baseline=baseline,
                baseline_quantile=baseline_quantile,
                sustain_sec=sustain_sec,
            ).detected
            for center in centers
        ],
        dtype=bool,
    )


def max_upward_transition(
    trace: np.ndarray,
    rel_t: np.ndarray,
    *,
    baseline: tuple[float, float] = (-120.0, -90.0),
    search: tuple[float, float] = (-60.0, 20.0),
    flank_sec: float = 2.0,
    baseline_quantile: float = 0.99,
) -> tuple[np.ndarray, TransitionResult]:
    """Find the largest label-blind upward step in a broad search window.

    At every valid center ``t`` the step statistic is ``mean([t,t+w)) -
    mean([t-w,t))``.  Its empirical threshold is computed from centers whose
    two flanks both lie inside the distal baseline.  The returned transition
    time is selected without consulting EEG or clinical onset annotations.
    """
    y = np.asarray(trace, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if y.ndim != 1 or t.ndim != 1 or y.size != t.size:
        raise ValueError("trace and rel_t must be matching 1D arrays")
    if y.size < 3:
        raise ValueError("at least three time frames are required")
    dt = float(np.nanmedian(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("rel_t must be strictly increasing")
    n = max(1, int(round(float(flank_sec) / dt)))
    step = np.full(y.shape, np.nan, dtype=float)
    for center in range(n, y.size - n + 1):
        before = y[center - n:center]
        after = y[center:center + n]
        if np.isfinite(before).any() and np.isfinite(after).any():
            step[center] = float(np.nanmean(after) - np.nanmean(before))

    bmask = (
        (t - float(flank_sec) >= float(baseline[0]))
        & (t + float(flank_sec) <= float(baseline[1]))
        & np.isfinite(step)
    )
    smask = (
        (t >= float(search[0]))
        & (t + float(flank_sec) <= float(search[1]))
        & np.isfinite(step)
    )
    if not np.any(bmask):
        raise ValueError("no valid transition centers in distal baseline")
    if not np.any(smask):
        raise ValueError("no valid transition centers in search window")
    threshold = float(np.nanquantile(step[bmask], float(baseline_quantile)))
    search_idx = np.flatnonzero(smask)
    best = int(search_idx[int(np.nanargmax(step[search_idx]))])
    delta = float(step[best])
    return step, TransitionResult(
        detected=bool(delta > threshold),
        transition_sec=float(t[best]),
        step_delta=delta,
        threshold=threshold,
        baseline_q=float(baseline_quantile),
        flank_sec=float(flank_sec),
        n_baseline_centers=int(np.sum(bmask)),
        n_search_centers=int(np.sum(smask)),
    )


def detect_multiband_recruitment_onset(
    band_traces: np.ndarray,
    rel_t: np.ndarray,
    *,
    baseline: tuple[float, float] = (-120.0, -90.0),
    search: tuple[float, float] = (-80.0, 5.0),
    majority_required: int | None = None,
    post_sec: float = 5.0,
    flank_sec: float = 2.0,
    baseline_quantile: float = 0.99,
    sustain_sec: float = 2.0,
) -> RecruitmentOnsetResult:
    """Detect a multiband energy-recruitment change point in a broad window.

    The candidate time is the largest upward step of the pointwise median band
    trace.  It is confirmed only when the step exceeds its distal-baseline Q99,
    the consensus stays above its level Q99 after the candidate, and a strict
    majority (or caller-specified number) of individual bands do the same.
    No EEG/clinical onset annotation is consulted inside this function.
    """
    traces = np.asarray(band_traces, dtype=float)
    t = np.asarray(rel_t, dtype=float)
    if traces.ndim != 2 or traces.shape[1] != t.size:
        raise ValueError("band_traces must be [band, time] and match rel_t")
    if traces.shape[0] < 1:
        raise ValueError("at least one band trace is required")
    required = int(majority_required or (traces.shape[0] // 2 + 1))
    if not 1 <= required <= traces.shape[0]:
        raise ValueError("majority_required must lie in [1, n_bands]")
    consensus = np.nanmedian(traces, axis=0)
    _, transition = max_upward_transition(
        consensus,
        t,
        baseline=baseline,
        search=search,
        flank_sec=flank_sec,
        baseline_quantile=baseline_quantile,
    )
    post_hi = min(float(search[1]), float(transition.transition_sec) + float(post_sec))
    if post_hi <= float(transition.transition_sec):
        raise ValueError("candidate has no post-change search support")
    consensus_post = detect_sustained_enhancement(
        consensus,
        t,
        baseline=baseline,
        search=(float(transition.transition_sec), post_hi),
        baseline_quantile=baseline_quantile,
        sustain_sec=sustain_sec,
    )
    band_hits = [
        detect_sustained_enhancement(
            trace,
            t,
            baseline=baseline,
            search=(float(transition.transition_sec), post_hi),
            baseline_quantile=baseline_quantile,
            sustain_sec=sustain_sec,
        ).detected
        for trace in traces
    ]
    n_hits = int(np.sum(band_hits))
    detected = bool(transition.detected and consensus_post.detected and n_hits >= required)
    return RecruitmentOnsetResult(
        detected=detected,
        onset_sec=float(transition.transition_sec),
        step_delta=float(transition.step_delta),
        step_threshold=float(transition.threshold),
        consensus_post_sustained=bool(consensus_post.detected),
        consensus_post_peak=float(consensus_post.peak_value),
        consensus_level_threshold=float(consensus_post.threshold),
        n_band_post_sustained=n_hits,
        n_bands=int(traces.shape[0]),
        majority_required=required,
    )


def band_energy_timing(
    z_contact_time: np.ndarray,
    rel_t: np.ndarray,
    *,
    baseline: tuple[float, float] = (-120.0, -90.0),
    search: tuple[float, float] = (-60.0, 20.0),
    spatial_q: float = 0.75,
    smooth_sec: float = 2.0,
    baseline_quantile: float = 0.99,
    sustain_sec: float = 2.0,
) -> tuple[np.ndarray, TimingResult]:
    """Full pure pipeline for one seizure and frequency band."""
    delta = distal_baseline_delta(z_contact_time, rel_t, baseline=baseline)
    trace = spatial_quantile_trace(delta, q=spatial_q)
    smoothed = smooth_trace(trace, rel_t, smooth_sec=smooth_sec)
    timing = detect_sustained_enhancement(
        smoothed,
        rel_t,
        baseline=baseline,
        search=search,
        baseline_quantile=baseline_quantile,
        sustain_sec=sustain_sec,
    )
    return smoothed, timing
