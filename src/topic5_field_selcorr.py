"""Topic 5 — selection-corrected field-concordance null (board upgrade, 2026-06-26).

The best-only field-concordance board picks, per subject, the best of N candidate analyses
(substrate × band, e.g. bb/HFA × maxAB/broad) and asks if that best beats its OWN channel-null.
That per-candidate null does NOT pay the best-of-N selection cost, so the board is only a SCREEN.

This module makes the selection FORMAL: the null must, in every draw, repeat the "take the best
candidate" step. Implementation = the max-statistic family-wise null:
  observed   = max over candidates of the real field-alignment stat
  null_max[b]= max over candidates of the candidates' b-th null draw
  p_selcorr  = P(null_max >= observed)
Independent channel-shuffles per candidate → the max-of-independent null is (mildly) CONSERVATIVE
when candidates are positively correlated (overstates the chance max), which is the safe direction.

Field smoothing is matmul-accelerated (the kernel weights are fixed per plane; only the per-draw
activation values change), so B null draws are cheap. The smoother is verified to match
src.propagation_contact_plane_readout.smooth_field (R_smooth_rank) exactly.
"""
from __future__ import annotations

from typing import Dict, Sequence, List

import numpy as np

from src.propagation_contact_plane_readout import make_plane_grid, S_THRESH


def precompute_smoother(chans: Sequence[dict], X: np.ndarray, Y: np.ndarray, sigma: float):
    """Fixed kernel weights for a contact plane. chans = [{x_norm,y_norm,support}], support>0,
    finite coords. Returns dict with W (n_grid, n_ch), S (n_grid,), shape — so a field for any
    value vector v (aligned to chans) is T = (W @ v) / S (matches smooth_field's support-weighted
    kernel regression). Caller must pass v with the SAME channel order as chans."""
    use = [(float(c["x_norm"]), float(c["y_norm"]), float(c.get("support", 0.0))) for c in chans
           if np.isfinite(c.get("x_norm", np.nan)) and np.isfinite(c.get("y_norm", np.nan))
           and c.get("support", 0.0) > 0]
    if not use:
        return None
    pts = np.array([[u[0], u[1]] for u in use], float)
    sup = np.array([u[2] for u in use], float)
    gx = X.ravel(); gy = Y.ravel()
    sig2 = 2.0 * float(sigma) ** 2
    d2 = (gx[:, None] - pts[None, :, 0]) ** 2 + (gy[:, None] - pts[None, :, 1]) ** 2  # (n_grid, n_ch)
    W = sup[None, :] * np.exp(-d2 / sig2)
    S = W.sum(axis=1)
    return {"W": W, "S": S, "shape": X.shape, "n_ch": len(use)}


def field_from_values(sm: dict, v: np.ndarray) -> dict:
    """Smoothed field {T, S} for value vector v (aligned to the smoother's channel order)."""
    v = np.asarray(v, float)
    S = sm["S"]
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(S > 1e-12, (sm["W"] @ v) / S, np.nan)
    return {"T": T.reshape(sm["shape"]), "S": S.reshape(sm["shape"])}


def _pearson_vec(a: np.ndarray, Bcols: np.ndarray) -> np.ndarray:
    """Pearson of vector a (n,) with each column of Bcols (n, B). Non-finite/constant -> nan."""
    ac = a - a.mean()
    Bc = Bcols - Bcols.mean(axis=0, keepdims=True)
    na = np.linalg.norm(ac); nb = np.linalg.norm(Bc, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (ac @ Bc) / (na * nb)
    r[(nb < 1e-12) | (~np.isfinite(r))] = np.nan
    return r


def null_aligns_vectorized(F_inter: dict, sm: dict, V: np.ndarray,
                           s_thresh: float = S_THRESH, overlap_min: int = 25) -> np.ndarray:
    """B sign-free mirror-invariant |corr| draws for permuted value matrix V (B, n_ch), against a
    FIXED interictal field F_inter, on smoother sm. Vectorized: the support overlap mask is fixed
    across draws (S_inter, S_ict fixed), so only the per-draw ictal T changes — reproduces
    corr_pair_mirror_invariant(...) then abs, per draw. Returns (B,) with nan where insufficient.
    """
    shape = sm["shape"]
    S_ict = sm["S"].reshape(shape)
    Ti, Si = F_inter["T"], F_inter["S"]
    with np.errstate(invalid="ignore", divide="ignore"):
        Tn = (sm["W"] @ V.T) / np.where(sm["S"] > 1e-12, sm["S"], np.nan)[:, None]   # (n_grid, B)
    Tn = Tn.reshape(shape + (V.shape[0],))                                           # (ny, nx, B)
    # identity orientation
    m_id = (Si >= s_thresh) & (S_ict >= s_thresh) & np.isfinite(Ti)
    n_id = int(m_id.sum())
    c_id = _pearson_vec(Ti[m_id], Tn[m_id]) if n_id >= overlap_min else None
    # mirror orientation: flip the ICTAL field (F2), not the interictal (matches corr_pair_mirror_invariant)
    Tn_f = np.flip(Tn, axis=0)
    S_ict_f = np.flip(S_ict, axis=0)
    m_mir = (Si >= s_thresh) & (S_ict_f >= s_thresh) & np.isfinite(Ti)
    n_mir = int(m_mir.sum())
    c_mir = _pearson_vec(Ti[m_mir], Tn_f[m_mir]) if n_mir >= overlap_min else None
    B = V.shape[0]
    stack = [c for c in (c_id, c_mir) if c is not None]
    if not stack:
        return np.full(B, np.nan)
    signed_max = np.nanmax(np.vstack(stack), axis=0)     # max of available signed corrs (matches corr_pair)
    return np.abs(signed_max)


def selection_corrected_pvalue(real_by_cand: Dict[str, float],
                               nulldist_by_cand: Dict[str, Sequence[float]]) -> dict:
    """Family-wise (max-statistic) selection-corrected p-value over a subject's candidates.

    real_by_cand: candidate -> real field-alignment stat (median over seizures).
    nulldist_by_cand: candidate -> per-draw null stats (median over seizures), length B.
    observed = max real; null_max[b] = max over candidates of their b-th null; p = P(null_max>=obs).
    """
    cands = [c for c in real_by_cand if c in nulldist_by_cand
             and len(nulldist_by_cand[c]) > 0 and np.isfinite(real_by_cand[c])]
    if not cands:
        return {"status": "no_candidates"}
    B = min(len(nulldist_by_cand[c]) for c in cands)
    M = np.array([np.asarray(nulldist_by_cand[c][:B], float) for c in cands])   # (n_cand, B)
    null_max = np.nanmax(M, axis=0)                                             # (B,)
    observed = max(real_by_cand[c] for c in cands)
    best = max(cands, key=lambda c: real_by_cand[c])
    p = float((np.sum(null_max >= observed) + 1) / (B + 1))
    return {"status": "ok", "observed_max": float(observed), "best_candidate": best,
            "p_selcorr": p, "pass_selcorr": bool(p < 0.05),
            "null_max_p95": float(np.nanpercentile(null_max, 95)),
            "B": int(B), "n_candidates": len(cands),
            "per_candidate_real": {c: float(real_by_cand[c]) for c in cands},
            "candidates": cands}
