"""Topic 5 V3c — recruitment-latency assay, censoring, AUC (PURE, no I/O).

first_crossing_latency distinguishes finite / t0 (left-censored, already hot at
onset) / censored (never sustained-crosses in window) — the t0/censored split
IS a spec-§5.4 result, not just QC, so we keep it (detect_contact_onset_zcross
only returns detected/unreached).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


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


def censoring_tallies(kinds: list) -> dict:
    n = len(kinds)
    if n == 0:
        return {"finite_frac": float("nan"), "t0_frac": float("nan"), "cens_frac": float("nan")}
    return {
        "finite_frac": sum(k == "finite" for k in kinds) / n,
        "t0_frac": sum(k == "t0" for k in kinds) / n,
        "cens_frac": sum(k == "censored" for k in kinds) / n,
    }


def rank_diagnostics(secs) -> dict:
    finite = np.asarray(secs, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"uniq_ranks": 0, "max_tie_block": 0}
    vals, counts = np.unique(np.round(finite, 3), return_counts=True)
    return {"uniq_ranks": int(vals.size), "max_tie_block": int(counts.max())}


def threshold_stability(secs_primary, secs_alt) -> float:
    a = np.asarray(secs_primary, dtype=float); b = np.asarray(secs_alt, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 4 or np.std(a[mask]) == 0 or np.std(b[mask]) == 0:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).correlation)


def assay_valid(qc: dict, cfg: dict) -> bool:
    """Label-blind assay gate (spec §5.2). Takes NO SOZ labels by contract."""
    g = cfg["v3c"]["assay_qc"]
    lat = cfg["v3c"]["latency"]
    return bool(
        qc["finite_frac"] >= g["finite_frac_min"]
        and qc["t0_frac"] <= g["t0_frac_max"]
        and qc["uniq_ranks_med"] >= g["uniq_ranks_min"]
        and (np.isfinite(qc["thr_spearman"]) and qc["thr_spearman"] >= g["thr_spearman_min"])
        and qc["n_informative"] >= lat["min_informative_seizures"]
    )
