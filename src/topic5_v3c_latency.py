"""Topic 5 V3c — recruitment-latency assay, censoring, AUC (PURE, no I/O).

first_crossing_latency distinguishes finite / t0 (left-censored, already hot at
onset) / censored (never sustained-crosses in window) — the t0/censored split
IS a spec-§5.4 result, not just QC, so we keep it (detect_contact_onset_zcross
only returns detected/unreached).
"""
from __future__ import annotations

import numpy as np


def first_crossing_latency(z_trace_1d, relt, onset, *, z_cross, window_sec, sustain_frames):
    z = np.asarray(z_trace_1d, dtype=float)
    relt = np.asarray(relt, dtype=float)
    m = (relt >= onset) & (relt <= onset + window_sec)
    idx = np.nonzero(m)[0]
    if idx.size < sustain_frames:
        return ("censored", float("nan"))
    zt = z[idx]
    zt = np.where(np.isfinite(zt), zt, -np.inf)
    if zt[0] >= z_cross:
        return ("t0", 0.0)
    for i in range(zt.size - sustain_frames + 1):
        if np.all(zt[i:i + sustain_frames] >= z_cross):
            return ("finite", float(relt[idx[i]] - onset))
    return ("censored", float("nan"))


def latency_seconds(kind: str, sec: float) -> float:
    """Seconds for Δt (finite→sec, t0→0.0, censored→nan)."""
    if kind == "finite":
        return float(sec)
    if kind == "t0":
        return 0.0
    return float("nan")


def encode_latency_for_rank(kind: str, sec: float, *, window_sec: float) -> float:
    """Sortable value for AUC (finite→sec, t0→earliest 0.0, censored→last window+1)."""
    if kind == "finite":
        return float(sec)
    if kind == "t0":
        return 0.0
    return float(window_sec) + 1.0
