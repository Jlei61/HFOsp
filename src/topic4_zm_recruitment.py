"""Bin-wise local recruitment, replacing first-spike onset density.

A "first spike inside the 100 ms before detection" statistic is not a
recruitment measure in this network: the unperturbed population sits above the
common detector 41 % of the time, so essentially every E neuron fires at least
once in any 100 ms window and the statistic is close to uniform noise.

What replaces it thresholds each 1 mm bin against ITS OWN interictal rate and
demands persistence, which is what lets the 10-90 % spread duration distinguish
sequential local spread from near-simultaneous whole-field ignition -- the
distinction the earlier q_I / g_K line failed to make.
"""
from __future__ import annotations

import numpy as np


def spatial_bins(positions_e, *, bin_mm, sheet_l_mm):
    positions = np.asarray(positions_e, float)
    n_side = int(np.ceil(float(sheet_l_mm) / float(bin_mm)))
    ix = np.clip((positions[:, 0] / bin_mm).astype(int), 0, n_side - 1)
    iy = np.clip((positions[:, 1] / bin_mm).astype(int), 0, n_side - 1)
    flat = ix * n_side + iy
    occupied, bin_index = np.unique(flat, return_inverse=True)
    counts = np.bincount(bin_index, minlength=occupied.size)
    centres = np.stack([(occupied // n_side + 0.5) * bin_mm,
                        (occupied % n_side + 0.5) * bin_mm], axis=-1)
    return {"bin_index": bin_index.astype(int), "bin_xy_mm": centres,
            "bin_counts": counts.astype(int), "n_bins": int(occupied.size),
            "bin_mm": float(bin_mm)}


def bin_rate_traces(E_spk_bool, bin_index, n_bins, *, dt_ms, kernel_ms):
    """Per-neuron firing rate in Hz within each bin, Gaussian-smoothed."""
    spikes = np.asarray(E_spk_bool, bool)
    bin_index = np.asarray(bin_index, int)
    counts = np.bincount(bin_index, minlength=n_bins).astype(float)
    counts[counts == 0] = np.nan
    summed = np.zeros((n_bins, spikes.shape[0]), float)
    for b in range(n_bins):
        members = bin_index == b
        if members.any():
            summed[b] = spikes[:, members].sum(axis=1)
    rate = summed / counts[:, None] / (dt_ms * 1e-3)
    sigma = float(kernel_ms) / float(dt_ms)
    if sigma > 0:
        half = int(np.ceil(3.0 * sigma))
        offsets = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
        rate = np.apply_along_axis(
            lambda row: np.convolve(row, kernel, mode="same"), 1, rate)
    return rate.astype(np.float32)


def bin_baseline(rate_traces, *, dt_ms, window_ms, quantile):
    """Per-bin threshold from that bin's OWN interictal rate.

    Never a single global threshold: bin occupancy and background rate vary
    several-fold across the sheet, so a global cut would label the busy bins
    recruited before anything happened and never label the quiet ones at all.
    """
    traces = np.asarray(rate_traces, float)
    lo = int(round(window_ms[0] / dt_ms))
    hi = int(round(window_ms[1] / dt_ms))
    segment = traces[:, lo:hi]
    if segment.shape[1] < 2:
        raise ValueError("baseline window is too short")
    return np.quantile(segment, float(quantile), axis=1)


def local_recruitment(rate_traces, thresholds, *, dt_ms, search_window_steps,
                      minimum_persistence_ms, search_start_step=0):
    traces = np.asarray(rate_traces, float)
    thresholds = np.asarray(thresholds, float)
    persist = max(1, int(round(float(minimum_persistence_ms) / float(dt_ms))))
    stop = min(traces.shape[1], search_start_step + int(search_window_steps))
    segment = traces[:, search_start_step:stop]
    above = segment > thresholds[:, None]
    recruitment = np.full(traces.shape[0], np.nan)
    for b in range(traces.shape[0]):
        row = above[b]
        if not row.any():
            continue
        # first index whose following `persist` samples are all above threshold
        padded = np.concatenate([row, np.zeros(persist, bool)])
        window = np.convolve(padded.astype(int), np.ones(persist, int), mode="valid")
        hits = np.flatnonzero(window[:len(row)] >= persist)
        if hits.size:
            recruitment[b] = float(search_start_step + hits[0])
    finite = recruitment[np.isfinite(recruitment)]
    if finite.size:
        lo, hi = np.percentile(finite, [10, 90])
        spread = float((hi - lo) * dt_ms)
        first = int(np.nanargmin(np.where(np.isfinite(recruitment), recruitment, np.inf)))
    else:
        spread, first = float("nan"), -1
    return {"recruitment_step": recruitment,
            "recruited_fraction": float(finite.size / traces.shape[0]),
            "spread_10_90_ms": spread,
            "first_recruited_bin": first,
            "n_recruited_bins": int(finite.size)}


def axial_lag(recruitment_step, bin_xy_mm, *, dt_ms, axis_unit, origin_xy):
    """Joint fit of recruitment time on signed along-axis and ABSOLUTE
    perpendicular distance.

    |d_perp| is required: with a signed perpendicular coordinate, spread that is
    symmetric about the axis cancels to a slope near zero and would be misread
    as "no off-axis propagation". Fitting both columns jointly is what stops an
    axial gradient from leaking into the off-axis slope.
    """
    recruitment = np.asarray(recruitment_step, float)
    xy = np.asarray(bin_xy_mm, float)
    axis_unit = np.asarray(axis_unit, float)
    axis_unit = axis_unit / np.linalg.norm(axis_unit)
    normal = np.array([-axis_unit[1], axis_unit[0]])
    finite = np.isfinite(recruitment)
    if finite.sum() < 3:
        return {"axial_slope_ms_per_mm": float("nan"),
                "offaxial_slope_ms_per_mm": float("nan"),
                "axial_r": float("nan"), "offaxial_r": float("nan"),
                "n_bins": int(finite.sum())}
    delta = xy[finite] - np.asarray(origin_xy, float)
    along = delta @ axis_unit
    perp = np.abs(delta @ normal)
    time_ms = recruitment[finite] * dt_ms
    design = np.column_stack([np.ones(finite.sum()), along, perp])
    coefficients, *_ = np.linalg.lstsq(design, time_ms, rcond=None)

    def _partial(index, other):
        residual_t = time_ms - np.column_stack(
            [np.ones(finite.sum()), other]) @ np.linalg.lstsq(
            np.column_stack([np.ones(finite.sum()), other]), time_ms, rcond=None)[0]
        predictor = design[:, index]
        residual_x = predictor - np.column_stack(
            [np.ones(finite.sum()), other]) @ np.linalg.lstsq(
            np.column_stack([np.ones(finite.sum()), other]), predictor, rcond=None)[0]
        if np.std(residual_x) < 1e-12 or np.std(residual_t) < 1e-12:
            return float("nan")
        return float(np.corrcoef(residual_x, residual_t)[0, 1])

    return {"axial_slope_ms_per_mm": float(coefficients[1]),
            "offaxial_slope_ms_per_mm": float(coefficients[2]),
            "axial_r": _partial(1, perp), "offaxial_r": _partial(2, along),
            "n_bins": int(finite.sum())}
