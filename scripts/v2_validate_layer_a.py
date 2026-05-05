"""HFO detector v2 — Layer A validation extractor (single-channel quality).

Computes per-subject metrics from a v2 detection output (*_gpu.npz):
  - dur_in_band_frac
  - peak_side_ratio (p25, p50, p99)
  - threshold_margin (p50)
  - timestamp_jitter_p99 (requires twice-run inputs)
  - strong_chn_count_match (requires twice-run inputs)

Output: results/hfo_detector_v2/validation/layer_a_<subject>.json
"""
from __future__ import annotations

import numpy as np


CHUNK_SEC = 200.0
N_WINDOWS_PER_RECORD = 3  # first / middle / last 200s — non-stationarity coverage


def compute_dur_in_band_frac(events, min_ms, max_ms):
    if len(events) == 0:
        return 1.0
    durs_ms = (events[:, 1] - events[:, 0]) * 1000.0
    return float(np.mean((durs_ms >= min_ms) & (durs_ms < max_ms)))


def compute_peak_side_ratio(env, events, fs):
    """For each event, compute pick_mean / side_mean (using legacy convention)."""
    out = []
    n = len(env)
    for t0, t1 in events:
        dur = t1 - t0
        i_pre_s = max(0, int((t0 - dur) * fs))
        i_pre_e = int(t0 * fs)
        i_post_s = int(t1 * fs)
        i_post_e = min(n, int((t1 + dur) * fs))
        side = np.concatenate([env[i_pre_s:i_pre_e], env[i_post_s:i_post_e]])
        pick = env[int(t0 * fs):int(t1 * fs)]
        if len(side) == 0 or len(pick) == 0:
            out.append(np.nan)
            continue
        s_mean = float(np.mean(side))
        if s_mean <= 0:
            out.append(np.inf)
            continue
        out.append(float(np.mean(pick) / s_mean))
    return np.array(out)


def compute_threshold_margin(env, events, fs, threshold):
    """For each event, (max_env_in_event - threshold) / threshold."""
    out = []
    for t0, t1 in events:
        i0, i1 = int(t0 * fs), int(t1 * fs)
        if i1 <= i0:
            out.append(np.nan)
            continue
        env_max = float(np.max(env[i0:i1]))
        out.append((env_max - threshold) / threshold)
    return np.array(out)


def _window_starts_for_record(rec_duration_sec: float) -> list[float]:
    """Return start_times of N_WINDOWS_PER_RECORD evenly-spaced 200s windows.

    Recording shorter than 600s falls back to [0.0] (first chunk only).
    """
    if rec_duration_sec < CHUNK_SEC * 3:
        return [0.0]
    last_start = max(0.0, rec_duration_sec - CHUNK_SEC)
    if N_WINDOWS_PER_RECORD == 1:
        return [0.0]
    starts = []
    for i in range(N_WINDOWS_PER_RECORD):
        frac = i / (N_WINDOWS_PER_RECORD - 1)
        starts.append(frac * last_start)
    return starts
