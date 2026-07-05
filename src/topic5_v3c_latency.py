"""Topic 5 V3c — recruitment-latency assay, censoring, AUC (PURE, no I/O).

first_crossing_latency distinguishes finite / t0 (left-censored, already hot at
onset) / censored (never sustained-crosses in window) — the t0/censored split
IS a spec-§5.4 result, not just QC, so we keep it (detect_contact_onset_zcross
only returns detected/unreached).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

from src.topic5_v3_mode_transition import _coerce_rng, label_permute


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


def auc_late(surplus_vals, soz_vals) -> float:
    s = np.asarray(surplus_vals, dtype=float); z = np.asarray(soz_vals, dtype=float)
    if s.size == 0 or z.size == 0:
        return float("nan")
    gt = np.sum(s[:, None] > z[None, :])
    eq = np.sum(s[:, None] == z[None, :])
    return float((gt + 0.5 * eq) / (s.size * z.size))


def delta_t(surplus_secs, soz_secs) -> float:
    s = np.asarray(surplus_secs, dtype=float); z = np.asarray(soz_secs, dtype=float)
    return float(np.nanmedian(s) - np.nanmedian(z))


def auc_null_distribution(surplus_vals, soz_vals, shaft_by_name, surplus_names, soz_names,
                          *, n_perm, rng) -> np.ndarray:
    """Within-shaft relabel of surplus/soz-core over A∩S ∪ A∖S, preserving per-shaft
    surplus count; recompute auc_late (spec §5.5 primary label null)."""
    rng = _coerce_rng(rng)
    val_by_name = {**{n: float(v) for n, v in zip(surplus_names, surplus_vals)},
                   **{n: float(v) for n, v in zip(soz_names, soz_vals)}}
    out = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        new_surplus, new_soz = label_permute(surplus_names, soz_names, shaft_by_name, rng)
        out[i] = auc_late(np.array([val_by_name[n] for n in new_surplus]),
                          np.array([val_by_name[n] for n in new_soz]))
    return out
