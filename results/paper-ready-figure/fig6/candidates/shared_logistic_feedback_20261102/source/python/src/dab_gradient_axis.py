"""Common pathological propagation axis = 3D least-squares gradient of D_AB.

Spec: docs/paper-draft/methods_axis_gradient_rewrite.md (contract) + methods_open_questions.md §E.
The axis is the gradient of the per-contact A/B relative-earliness contrast D_AB over 3D contact
coordinates, fitted on ALL joint-valid + coord-mapped contacts. It uses no source/sink endpoint,
no decision-k, no fixed k. The extreme +/-tercile centroids are DISPLAY ONLY and never enter the
estimate. This is the formal producer-grade function extracted from the plotting prototype
(scripts/plot_topic5_dab_axis_subject.py::compute).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from src.topic5_scaffold_ab_contrast import build_D_AB

AXIS_DEFINITION = "dab_gradient_v1"
# Relative singular-value floor for the least-squares gradient. Spec (methods_axis_gradient
# _rewrite.md numerical-QC): for rank-deficient / ill-conditioned X_c, u is the min-norm gradient
# in the SAMPLED subspace. Directions whose contact spread is < RCOND * the largest are dropped so
# a near-collinear single shaft yields the shaft-direction gradient instead of overfitting a
# poorly-sampled perpendicular direction. effective_rank reports how many directions survived.
RCOND = 0.05


def dab_from_ranks(rank_a, rank_b) -> np.ndarray:
    """Per-contact D_AB = e_A - e_B via the canonical builder (reuse, don't re-invent)."""
    return build_D_AB(np.asarray(rank_a, float), np.asarray(rank_b, float))["D_AB"]


def _fit_axis(Xc: np.ndarray, yc: np.ndarray):
    """Least-squares gradient beta = Xc^+ yc, its unit direction, and raw beta;
    unit direction is None if |beta|~0 (degenerate)."""
    beta, *_ = np.linalg.lstsq(Xc, yc, rcond=RCOND)
    bn = float(np.linalg.norm(beta))
    if not np.isfinite(bn) or bn < 1e-9:
        return None, bn, beta
    return beta / bn, bn, beta


def _degenerate(status: str, n: int) -> Dict[str, object]:
    return {"status": status, "u": None, "n": int(n), "axis_definition": AXIS_DEFINITION}


def compute_dab_gradient_axis(
    coords: np.ndarray,
    dab: np.ndarray,
    shafts: Optional[Sequence] = None,
    *,
    min_contacts: int = 6,
    n_boot: int = 200,
    seed: int = 0,
) -> Dict[str, object]:
    """Fit the D_AB 3D gradient axis and its numerical / sampling quality controls.

    Contract (methods_axis_gradient_rewrite.md), each clause honored by the marked block:
      C1 u = lstsq gradient beta/|beta|, oriented so axial projection correlates + with D_AB.
      C3 poles (+/-tercile centroids) are DISPLAY-ONLY; the arrow p_B->p_A is parallel to u.
      C4 fail-closed: <min_contacts valid, zero D_AB variance, or |beta|~0 -> not defined.
      C5 QC: n, sd_dab, beta_norm, R2, matrix_rank, condition_number, within_shaft_frac,
         leave-one-shaft-out cosine, contact-bootstrap cosine.
      C6 axis_definition = 'dab_gradient_v1' on the record.
      C7 rank-deficient X_c is FLAGGED (full_rank False); u is the min-norm gradient.
    """
    coords = np.asarray(coords, float)
    dab = np.asarray(dab, float)
    shafts = None if shafts is None else np.asarray(shafts, dtype=object)

    valid = np.isfinite(dab) & np.isfinite(coords).all(axis=1)
    n = int(valid.sum())
    if n < min_contacts:                                          # C4
        return _degenerate("insufficient_contacts", n)

    X = coords[valid]
    y = dab[valid]
    sh = None if shafts is None else shafts[valid]
    sd_dab = float(y.std(ddof=0))
    if sd_dab < 1e-12:                                            # C4
        return _degenerate("degenerate_no_variance", n)

    xbar = X.mean(0)
    Xc = X - xbar
    yc = y - y.mean()
    u, beta_norm, beta = _fit_axis(Xc, yc)
    if u is None:                                                 # C4
        return _degenerate("degenerate_low_beta", n)
    # R2 of the linear gradient fit: from the RAW lstsq beta (orientation-independent).
    ss_tot = float((yc ** 2).sum())
    ss_res = float(((yc - Xc @ beta) ** 2).sum())
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # C1: orient so axial projection increases with D_AB (B-lead pole -> A-lead pole).
    along = Xc @ u
    if np.corrcoef(along, y)[0, 1] < 0:
        u, along = -u, -along
    resid = Xc - np.outer(along, u)
    w = np.linalg.svd(resid, full_matrices=False)[2][0]
    signed_transverse = resid @ w

    # C5/C7: numerical QC of the coordinate design matrix.
    rank = int(np.linalg.matrix_rank(Xc))
    cond = float(np.linalg.cond(Xc))
    full_rank = bool(rank >= 3)
    sv = np.linalg.svd(Xc, compute_uv=False)
    effective_rank = int((sv >= RCOND * sv.max()).sum()) if sv.size and sv.max() > 0 else 0

    # C3: extreme +/-tercile pole centroids -- DISPLAY ONLY, not in beta/u.
    order = np.argsort(y)
    k = max(2, n // 3)
    idx_B, idx_A = order[:k], order[-k:]
    mu_B, mu_A = X[idx_B].mean(0), X[idx_A].mean(0)
    L_poles = float(np.linalg.norm(mu_A - mu_B))
    p_A = xbar + float((mu_A - xbar) @ u) * u                     # projected onto the fitted axis
    p_B = xbar + float((mu_B - xbar) @ u) * u
    # arrow p_B -> p_A is parallel to u by construction (both projected onto u).

    # C5: spatial-organization / sampling QC.
    within_frac, n_shafts, loso_cos = _shaft_qc(X, y, sh, u, min_contacts)
    boot_cos = _bootstrap_cosine(Xc, yc, u, n_boot, seed)
    moran = _morans_i(X, y)

    return {
        "status": "ok",
        "axis_definition": AXIS_DEFINITION,                       # C6
        "n": n,
        "valid_mask": valid.tolist(),
        "u": u,
        "w": w,
        "xbar": xbar,
        "beta": (u * beta_norm),
        "beta_norm": beta_norm,
        "sd_dab": sd_dab,
        "R2": r2,
        "along": along,
        "signed_transverse": signed_transverse,
        "matrix_rank": rank,
        "effective_rank": effective_rank,
        "condition_number": cond,
        "full_rank": full_rank,
        "moran_i": moran,
        "within_shaft_frac": within_frac,
        "n_shafts": n_shafts,
        "loso_cosine": loso_cos,
        "bootstrap_cosine": boot_cos,
        "mu_A": mu_A, "mu_B": mu_B, "p_A": p_A, "p_B": p_B, "L_poles": L_poles,
        "pole_A_idx": idx_A.tolist(), "pole_B_idx": idx_B.tolist(),
    }


def _shaft_qc(X, y, sh, u, min_contacts):
    """within-shaft D_AB variance fraction, n_shafts, and leave-one-shaft-out axis cosine."""
    if sh is None:
        return np.nan, None, np.nan
    uniq = sorted(set(sh.tolist()))
    yc = y - y.mean()
    sst = float((yc ** 2).sum())
    smean = np.array([y[sh == s].mean() for s in sh])
    between = float(((smean - y.mean()) ** 2).sum())
    within_frac = float(1 - between / sst) if sst > 0 else np.nan
    if len(uniq) < 2:
        return within_frac, len(uniq), np.nan
    cos = []
    for s in uniq:
        keep = sh != s
        if int(keep.sum()) < min_contacts:
            continue
        Xk = X[keep] - X[keep].mean(0)
        uk, bn, _ = _fit_axis(Xk, y[keep] - y[keep].mean())
        if uk is not None:
            cos.append(abs(float(uk @ u)))
    return within_frac, len(uniq), (float(np.median(cos)) if cos else np.nan)


def _bootstrap_cosine(Xc, yc, u, n_boot, seed):
    """Median |cos| between contact-bootstrap axes and the full-data axis."""
    n = Xc.shape[0]
    rng = np.random.default_rng(seed)
    cos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        uk, bn, _ = _fit_axis(Xc[idx] - Xc[idx].mean(0), yc[idx] - yc[idx].mean())
        if uk is not None:
            cos.append(abs(float(uk @ u)))
    return float(np.median(cos)) if cos else np.nan


def _morans_i(coords, y):
    """Inverse-distance Moran's I of D_AB (collinearity-free spatial-structure statistic)."""
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    with np.errstate(divide="ignore"):
        W = 1.0 / d
    W[~np.isfinite(W)] = 0.0
    s0 = W.sum()
    z = y - y.mean()
    denom = float((z ** 2).sum())
    if s0 <= 0 or denom <= 0:
        return np.nan
    return float((len(y) / s0) * float(z @ W @ z) / denom)
