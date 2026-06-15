"""Topic 5 T0 — pure early-ictal activation features (no I/O).

Consumed by `scripts/build_topic5_t0_feature_cache.py`. The A-line primary activation is
`broadband_auc_0_10s` = mean of the baseline-robust-z broadband (1-45 Hz) power over the
[0, 10 s] post-onset window, per contact. (Mean over the window, not a literal trapezoid
area — A's statistic is a mirror-invariant CORRELATION which is scale-invariant, so the
robust per-channel magnitude is what matters, and nanmean tolerates dropped frames.)

The baseline robust-z itself reuses `src.topic5_ictal_recruitment.baseline_robust_z`; this
module only adds the windowing (which frames are the [0,10 s] activation window) and the
per-channel window summaries (mean, ramp slope).
"""
from __future__ import annotations

import numpy as np


def onset_window_indices(n_frames, *, pre_sec, hop_sec, t0_sec=0.0, t1_sec=10.0):
    """Frame indices of the [t0, t1]-seconds-post-onset window.

    The loaded window's frame f has time (f * hop_sec - pre_sec) relative to onset
    (onset = clinical onset, signal t=0; see extract_seizure_window). So onset is at
    frame pre_sec/hop_sec, and [t0, t1] post-onset is [ (pre+t0)/hop , (pre+t1)/hop ].
    Clipped to [0, n_frames-1]."""
    f0 = int(round((pre_sec + t0_sec) / hop_sec))
    f1 = int(round((pre_sec + t1_sec) / hop_sec))
    f0 = max(0, min(f0, n_frames))
    f1 = max(0, min(f1, n_frames))
    if f1 <= f0:
        return np.empty(0, dtype=int)
    return np.arange(f0, f1, dtype=int)


def activation_mean(z_trace, win_idx):
    """Per-channel nanmean of a robust-z trace over win_idx. z_trace = [n_ch, n_frames].
    Returns [n_ch] (NaN for a channel with no finite frame in the window)."""
    z = np.asarray(z_trace, float)
    if win_idx.size == 0 or z.shape[1] == 0:
        return np.full(z.shape[0], np.nan)
    sub = z[:, win_idx]
    with np.errstate(invalid="ignore"):
        out = np.where(np.isfinite(sub).any(axis=1), np.nanmean(sub, axis=1), np.nan)
    return out


# --- multi-window slicing (v2 cache: 6 alignment windows from one stored trace) ---
# The v2 cache stores the FULL-window robust-z trace [-pre,+post] plus its per-bin
# `rel_t` (= band_power_trace times - pre_sec). Every alignment window is then a slice
# of that single trace — no re-extraction. Inclusive bounds (>= a AND <= b).
AXIS_WINDOWS = {
    "post_0_5": (0, 5),
    "post_5_10": (5, 10),
    "post_0_10": (0, 10),
    "post_0_20": (0, 20),
    "pre_prox_m10_0": (-10, 0),       # proximal pre: inside [-60,0] guard, sensitivity ONLY
    "pre_distal_m120_m90": (-120, -90),  # distal pre: OUTSIDE guard, load-bearing neg control
}


def window_indices(rel_t, a, b):
    """Bin indices whose rel-to-onset time is in [a, b] (inclusive). rel_t = 1D [n_bins]."""
    rel = np.asarray(rel_t, float)
    return np.where((rel >= a) & (rel <= b))[0]


def window_activation(z_trace, rel_t, a, b):
    """Per-channel mean activation over the rel-to-onset window [a, b]. Reuses
    `activation_mean` (same NaN semantics). z_trace=[n_ch,n_bins], rel_t=[n_bins]."""
    return activation_mean(z_trace, window_indices(rel_t, a, b))


def ramp_slope(z_trace, win_idx, *, hop_sec):
    """Per-channel least-squares slope (z-units per second) of the trace over win_idx.
    NaN frames within the window are dropped per channel; <2 finite points -> NaN slope."""
    z = np.asarray(z_trace, float)
    n_ch = z.shape[0]
    out = np.full(n_ch, np.nan)
    if win_idx.size < 2:
        return out
    t = win_idx.astype(float) * hop_sec
    for c in range(n_ch):
        y = z[c, win_idx]
        m = np.isfinite(y)
        if m.sum() >= 2:
            out[c] = np.polyfit(t[m], y[m], 1)[0]
    return out
