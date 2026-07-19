"""Gradient-axis R3 dense-grid field similarity (Figure 3 recompute).

Primary estimand for the Figure 3 ictal recompute
(docs/archive/topic5/fig3_ictal_gradient_r3_full_recompute_handoff_2026-07-18.md):
a sign-free, transverse-mirror-invariant, A/B ``maxAB`` spatial-field
concordance evaluated on an adaptive dense grid (``R3``), with the
contact-evaluated smoothed-field similarity (``R2``) as a paired sensitivity.

The module is a pure-math layer with **no filesystem I/O**. It reuses the
frozen interictal gradient geometry (points / support / earliness / sigma) and
the historical mirror-invariant correlation primitive
``corr_pair_mirror_invariant_signed``; it only adds per-plane adaptive grids and
the ictal common-finite-mask field construction.

Design invariants (handoff §3):
* one ``sigma_common`` per subject is applied to TA and TB, R2 and R3, observed
  and every null draw (the caller supplies it);
* the interictal field uses the *full* frozen support; the ictal field uses the
  event's locked common finite-contact mask, so ``S_inter >= S_ictal``;
* the y axis is strictly symmetric about 0 so ``np.flip(axis=0)`` is exactly the
  transverse mirror;
* the adaptive bounds are derived from geometry / support / sigma only, never
  from the ictal outcome; the ``S >= 0.15`` region must not touch a grid edge.
"""
from __future__ import annotations

import hashlib
import math
from typing import Dict, Optional, Sequence

import numpy as np

from src.propagation_contact_plane_readout import corr_pair_mirror_invariant_signed

S_THRESH = 0.15
OVERLAP_MIN_N81 = 25
GRID_N = 81
R3_FORMULA_VERSION = "gradient_grid_field_maxab_v1"


# --------------------------------------------------------------------------
# resolution / overlap
# --------------------------------------------------------------------------
def overlap_min_for_n(n: int) -> int:
    """Minimum eligible-overlap pixels at grid resolution ``n`` (handoff §3.7).

    ``overlap_min(N) = ceil((25 / 81**2) * N**2)`` -> 25 at N=81, 99 at N=161.
    """
    return int(math.ceil(OVERLAP_MIN_N81 * (int(n) ** 2) / (GRID_N ** 2)))


# --------------------------------------------------------------------------
# adaptive grid (handoff §3.3)
# --------------------------------------------------------------------------
def support_radius(sigma: float, support_budget: float, s_thresh: float = S_THRESH) -> float:
    """Radius past which the smoothed support drops below ``s_thresh``.

    ``r = sigma * (sqrt(2 log(max(S_budget / s_thresh, 1))) + 1)``.  At distance
    ``r`` from every contributing contact the summed Gaussian support is strictly
    below ``s_thresh``, so extending the grid by ``r`` past the contact bounding
    box keeps the ``S >= s_thresh`` region off the boundary.
    """
    budget = max(float(support_budget) / float(s_thresh), 1.0)
    return float(sigma) * (math.sqrt(2.0 * math.log(budget)) + 1.0)


def _grid_sha256(model: str, x_lo: float, x_hi: float, y_ext: float, n: int,
                 sigma: float, support_budget: float) -> str:
    digest = hashlib.sha256()
    for token in (model, f"{x_lo:.12e}", f"{x_hi:.12e}", f"{y_ext:.12e}",
                  str(int(n)), f"{sigma:.12e}", f"{support_budget:.12e}"):
        digest.update(token.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def make_adaptive_grid(points: np.ndarray, sigma: float, support_budget: float,
                       n: int = GRID_N, s_thresh: float = S_THRESH,
                       model: str = "") -> Dict[str, object]:
    """Adaptive, y-symmetric dense grid for one plane (handoff §3.3).

    Bounds are derived from the frozen plane geometry, template support and
    ``sigma`` only.  ``Y`` is strictly symmetric about 0 (odd ``n`` contains the
    ``y=0`` row) so ``np.flip(F, axis=0)`` is an exact transverse mirror.
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (m, 2)")
    r = support_radius(sigma, support_budget, s_thresh)
    x_lo = float(pts[:, 0].min() - r)
    x_hi = float(pts[:, 0].max() + r)
    y_ext = float(np.abs(pts[:, 1]).max() + r)
    x = np.linspace(x_lo, x_hi, int(n))
    y = np.linspace(-y_ext, y_ext, int(n))
    # row index = y, col index = x; flip(axis=0) reflects y -> -y
    Y, X = np.meshgrid(y, x, indexing="ij")
    # exact-symmetry cleanup so flip(axis=0) is bit-exact
    Y = 0.5 * (Y - np.flip(Y, axis=0))
    return {
        "X": X, "Y": Y, "n": int(n),
        "x_lo": x_lo, "x_hi": x_hi, "y_ext": y_ext,
        "spacing_x": float(x[1] - x[0]) if n > 1 else 0.0,
        "spacing_y": float(y[1] - y[0]) if n > 1 else 0.0,
        "sigma": float(sigma), "support_budget": float(support_budget),
        "support_radius": r, "s_thresh": float(s_thresh),
        "model": str(model),
        "sha256": _grid_sha256(str(model), x_lo, x_hi, y_ext, n, sigma, support_budget),
    }


def support_region_touches_boundary(S: np.ndarray, s_thresh: float = S_THRESH) -> bool:
    """True if any ``S >= s_thresh`` pixel sits on the grid border."""
    S = np.asarray(S, float)
    border = np.zeros(S.shape, bool)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return bool(np.any((S >= s_thresh) & border))


# --------------------------------------------------------------------------
# grid field construction (handoff §3.4)
# --------------------------------------------------------------------------
def build_grid_field(X: np.ndarray, Y: np.ndarray, points: np.ndarray,
                     weight: np.ndarray, values: np.ndarray, sigma: float):
    """Support-weighted Gaussian kernel field on a grid.

    ``F[g] = sum_i K[g,i] w_i v_i / sum_i K[g,i] w_i`` and ``S[g] = sum_i K[g,i] w_i``
    with ``K[g,i] = exp(-||g - p_i||^2 / (2 sigma^2))``.  Returns ``(F, S)`` each
    shaped like ``X``.  Non-finite ``values`` are excluded from the numerator
    (their weight must already be 0 for the ictal case).
    """
    gx = np.asarray(X, float).ravel()
    gy = np.asarray(Y, float).ravel()
    pts = np.asarray(points, float)
    w = np.asarray(weight, float)
    v = np.asarray(values, float)
    sig2 = 2.0 * float(sigma) ** 2
    d2 = (gx[:, None] - pts[None, :, 0]) ** 2 + (gy[:, None] - pts[None, :, 1]) ** 2
    K = np.exp(-d2 / sig2)
    W = K * w[None, :]
    S = W.sum(axis=1)
    v_num = np.where(np.isfinite(v), v, 0.0)
    num = (W * v_num[None, :]).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        F = np.where(S > 1e-12, num / S, np.nan)
    return F.reshape(X.shape), S.reshape(X.shape)


def _kernel_matrix(grid: Dict[str, object], points: np.ndarray, sigma: float) -> np.ndarray:
    gx = grid["X"].ravel()
    gy = grid["Y"].ravel()
    pts = np.asarray(points, float)
    sig2 = 2.0 * float(sigma) ** 2
    d2 = (gx[:, None] - pts[None, :, 0]) ** 2 + (gy[:, None] - pts[None, :, 1]) ** 2
    return np.exp(-d2 / sig2)


def _flip_index(n: int) -> np.ndarray:
    """Ravel-index permutation implementing ``flip(axis=0)`` on an (n,n) grid."""
    rows = np.arange(n)[::-1]
    return (rows[:, None] * n + np.arange(n)[None, :]).ravel()


# --------------------------------------------------------------------------
# per-template R3 score (single activation; reference path via legacy primitive)
# --------------------------------------------------------------------------
def score_template_r3(grid: Dict[str, object], F_inter: np.ndarray, S_inter: np.ndarray,
                      points: np.ndarray, support: np.ndarray, activation: np.ndarray,
                      finite: np.ndarray) -> Dict[str, object]:
    """One-template abs-max identity/mirror grid-field correlation for one event.

    Uses the historical ``corr_pair_mirror_invariant_signed`` primitive so the
    single-activation path is definitionally the reference the batch path must
    reproduce (handoff §3.5, §8 test 8/12).
    """
    n = grid["n"]
    shape = grid["X"].shape
    finite_f = np.asarray(finite, float)
    support = np.asarray(support, float)
    act = np.asarray(activation, float)
    w_ict = support * finite_f
    v_ict = np.where(np.asarray(finite, bool), act, 0.0)
    F_ict, S_ict = build_grid_field(grid["X"], grid["Y"], points, w_ict, v_ict, grid["sigma"])
    res = corr_pair_mirror_invariant_signed(
        np.asarray(F_inter, float).reshape(shape), np.asarray(S_inter, float).reshape(shape),
        F_ict, S_ict, s_thresh=S_THRESH, overlap_min=overlap_min_for_n(n))
    return {
        "signed_r": res["signed_corr"],
        "abs_r": res["abs_corr"],
        "mirror_choice": res["mirror_choice"],
        "r_identity": res["corr_id"],
        "r_mirror": res["corr_mirror"],
        "n_overlap": res.get("n_overlap"),
        "insufficient_overlap": res.get("insufficient_overlap", True),
    }


# --------------------------------------------------------------------------
# event scorer (A/B maxAB; single reference + vectorized batch)
# --------------------------------------------------------------------------
class _TemplatePrecomp:
    __slots__ = ("grid", "points", "support", "F_inter", "S_inter", "Wict",
                 "Ksup", "S_ict", "S_ict_mir", "flip_idx", "M_id", "M_mir",
                 "overlap_id", "overlap_mir", "t_id", "t_mir", "overlap_min")

    def __init__(self, grid, points, support, earliness, finite, sigma):
        n = grid["n"]
        self.grid = grid
        self.points = np.asarray(points, float)
        self.support = np.asarray(support, float)
        finite_f = np.asarray(finite, float)
        K = _kernel_matrix(grid, self.points, sigma)              # (P, m)
        Wsup = K * self.support[None, :]
        self.Ksup = Wsup                                          # (P, m) reused by varmask
        self.S_inter = Wsup.sum(axis=1)                           # (P,)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.F_inter = np.where(self.S_inter > 1e-12,
                                    (Wsup * np.asarray(earliness, float)[None, :]).sum(axis=1)
                                    / self.S_inter, np.nan)
        self.Wict = K * (self.support * finite_f)[None, :]         # (P, m)
        self.S_ict = self.Wict.sum(axis=1)                         # (P,)  fixed across draws
        self.flip_idx = _flip_index(n)
        self.S_ict_mir = self.S_ict[self.flip_idx]
        self.overlap_min = overlap_min_for_n(n)
        t = float(S_THRESH)
        finite_inter = np.isfinite(self.F_inter)
        self.M_id = (self.S_inter >= t) & (self.S_ict >= t) & finite_inter
        self.M_mir = (self.S_inter >= t) & (self.S_ict_mir >= t) & finite_inter
        self.overlap_id = int(self.M_id.sum())
        self.overlap_mir = int(self.M_mir.sum())
        # template vectors on each candidate mask (constant across draws)
        self.t_id = self.F_inter[self.M_id]
        self.t_mir = self.F_inter[self.M_mir]

    def ictal_fields(self, acts: np.ndarray) -> np.ndarray:
        """(P, n_row) ictal grid fields for a (n_row, m) activation matrix."""
        v = np.where(np.isfinite(acts), acts, 0.0)              # (n_row, m)
        num = self.Wict @ v.T                                    # (P, n_row)
        with np.errstate(invalid="ignore", divide="ignore"):
            F = np.where(self.S_ict[:, None] > 1e-12, num / self.S_ict[:, None], np.nan)
        return F

    def abs_r_batch(self, acts: np.ndarray) -> np.ndarray:
        """Abs-max identity/mirror |r| per row (n_row,) reselecting mirror each row."""
        n_row = acts.shape[0]
        F_ict = self.ictal_fields(acts)                          # (P, n_row)
        r_id = np.full(n_row, np.nan)
        r_mir = np.full(n_row, np.nan)
        if self.overlap_id >= self.overlap_min and self.t_id.size >= 3:
            r_id = _pearson_cols(self.t_id, F_ict[self.M_id, :])
        if self.overlap_mir >= self.overlap_min and self.t_mir.size >= 3:
            F_ict_mir = F_ict[self.flip_idx, :]
            r_mir = _pearson_cols(self.t_mir, F_ict_mir[self.M_mir, :])
        abs_id = np.where(np.isfinite(r_id), np.abs(r_id), -np.inf)
        abs_mir = np.where(np.isfinite(r_mir), np.abs(r_mir), -np.inf)
        best = np.maximum(abs_id, abs_mir)
        return np.where(best > -np.inf, best, np.nan)

    def _masked_corr_cols(self, x_vec, Y, base, S_y):
        """Per-column Pearson with a per-column support-overlap mask (variable mask)."""
        t = float(S_THRESH)
        M = base[:, None] & (S_y >= t)                # (P, R)
        W = M.astype(float)
        n_r = W.sum(axis=0)                            # (R,)
        x = np.where(np.isfinite(x_vec), x_vec, 0.0)
        Yz = np.where(np.isfinite(Y), Y, 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_x = (W * x[:, None]).sum(0) / n_r
            mean_y = (W * Yz).sum(0) / n_r
            dx = x[:, None] - mean_x[None, :]
            dy = Yz - mean_y[None, :]
            cov = (W * dx * dy).sum(0)
            varx = (W * dx * dx).sum(0)
            vary = (W * dy * dy).sum(0)
            den = np.sqrt(varx * vary)
        out = np.full(Y.shape[1], np.nan)
        ok = (n_r >= self.overlap_min) & (den > 1e-12)
        out[ok] = cov[ok] / den[ok]
        return out

    def abs_r_batch_varmask(self, acts: np.ndarray) -> np.ndarray:
        """Abs-max |r| per row where each row carries its OWN missing-contact mask.

        Unlike :meth:`abs_r_batch` (fixed ictal support), this recomputes the ictal
        support ``S_ict`` per row, so a contact missing only in some rows (time
        windows) is excluded from both numerator and denominator for those rows.
        """
        acts = np.atleast_2d(np.asarray(acts, float))
        finite = np.isfinite(acts).astype(float)          # (R, m)
        v = np.where(np.isfinite(acts), acts, 0.0)         # (R, m)
        S_ict = self.Ksup @ finite.T                       # (P, R)
        num = self.Ksup @ v.T                              # (P, R)
        with np.errstate(invalid="ignore", divide="ignore"):
            F_ict = np.where(S_ict > 1e-12, num / S_ict, np.nan)
        base = (self.S_inter >= float(S_THRESH)) & np.isfinite(self.F_inter)
        r_id = self._masked_corr_cols(self.F_inter, F_ict, base, S_ict)
        r_mir = self._masked_corr_cols(self.F_inter, F_ict[self.flip_idx, :], base,
                                       S_ict[self.flip_idx, :])
        abs_id = np.where(np.isfinite(r_id), np.abs(r_id), -np.inf)
        abs_mir = np.where(np.isfinite(r_mir), np.abs(r_mir), -np.inf)
        best = np.maximum(abs_id, abs_mir)
        return np.where(best > -np.inf, best, np.nan)


def _pearson_cols(t: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r between fixed vector ``t`` (k,) and every column of ``Y`` (k, n)."""
    t = np.asarray(t, float)
    Y = np.asarray(Y, float)
    tc = t - t.mean()
    tss = float(tc @ tc)
    out = np.full(Y.shape[1], np.nan)
    if tss < 1e-24:
        return out
    Yc = Y - Y.mean(axis=0, keepdims=True)
    num = tc @ Yc
    den = np.sqrt(tss * (Yc * Yc).sum(axis=0))
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def build_event_scorer(*, pts_a, support_a, earliness_a, pts_b, support_b, earliness_b,
                       sigma: Optional[float] = None, sigma_a: Optional[float] = None,
                       sigma_b: Optional[float] = None, finite, n: int = GRID_N,
                       shared_grid: Optional[bool] = None,
                       model_a: str = "A", model_b: str = "B") -> Dict[str, object]:
    """Precompute both template grids/fields for one event (handoff §3.2-§3.5).

    Sigma may be given as one ``sigma`` applied to both templates (subject-fixed
    policy, backward compatible) or as per-template ``sigma_a`` / ``sigma_b``
    (frozen-per-model policy). Template A uses ``sigma_a`` for its grid, kernel
    and support radius; template B uses ``sigma_b``. A shared grid requires
    ``sigma_a == sigma_b`` (fail closed). When ``shared_grid`` (auto-detected from
    equal points) the two templates share one grid with
    ``S_budget = max(sum(support_a), sum(support_b))``; otherwise each
    own-fallback template builds its own grid from its own support.
    """
    if sigma is not None:
        if sigma_a is None:
            sigma_a = sigma
        if sigma_b is None:
            sigma_b = sigma
    if sigma_a is None or sigma_b is None:
        raise ValueError("provide sigma= or both sigma_a= and sigma_b=")
    sigma_a = float(sigma_a)
    sigma_b = float(sigma_b)
    pts_a = np.asarray(pts_a, float)
    pts_b = np.asarray(pts_b, float)
    support_a = np.asarray(support_a, float)
    support_b = np.asarray(support_b, float)
    finite = np.asarray(finite, bool)
    if shared_grid is None:
        shared_grid = pts_a.shape == pts_b.shape and np.array_equal(pts_a, pts_b)
    if shared_grid:
        if not np.isclose(sigma_a, sigma_b):
            raise ValueError("shared grid requires sigma_a == sigma_b")
        budget = max(float(support_a.sum()), float(support_b.sum()))
        grid_a = make_adaptive_grid(pts_a, sigma_a, budget, n=n, model=model_a)
        grid_b = grid_a
    else:
        grid_a = make_adaptive_grid(pts_a, sigma_a, float(support_a.sum()), n=n, model=model_a)
        grid_b = make_adaptive_grid(pts_b, sigma_b, float(support_b.sum()), n=n, model=model_b)
    pre_a = _TemplatePrecomp(grid_a, pts_a, support_a, earliness_a, finite, sigma_a)
    pre_b = _TemplatePrecomp(grid_b, pts_b, support_b, earliness_b, finite, sigma_b)
    return {"A": pre_a, "B": pre_b,
            "sigma": float(sigma_a) if np.isclose(sigma_a, sigma_b) else None,
            "sigma_a": sigma_a, "sigma_b": sigma_b, "n": int(n),
            "shared_grid": bool(shared_grid), "finite": finite,
            "grid_a": grid_a, "grid_b": grid_b}


def score_event_maxab_batch(ev: Dict[str, object], acts: np.ndarray,
                            chunk: int = 256) -> np.ndarray:
    """maxAB = max(|r_A|, |r_B|) per activation row (handoff §3.5)."""
    acts = np.atleast_2d(np.asarray(acts, float))
    n_row = acts.shape[0]
    out = np.full(n_row, np.nan)
    for lo in range(0, n_row, chunk):
        hi = min(lo + chunk, n_row)
        block = acts[lo:hi]
        abs_a = ev["A"].abs_r_batch(block)
        abs_b = ev["B"].abs_r_batch(block)
        stacked = np.vstack([abs_a, abs_b])
        with np.errstate(invalid="ignore"):
            out[lo:hi] = np.nanmax(stacked, axis=0)
        # a column that is all-NaN yields nan (nanmax of all-nan -> nan with warning)
        all_nan = ~np.isfinite(abs_a) & ~np.isfinite(abs_b)
        out[lo:hi][all_nan] = np.nan
    return out


def score_event_maxab_batch_varmask(ev: Dict[str, object], acts: np.ndarray,
                                    chunk: int = 256) -> np.ndarray:
    """maxAB per row where each row carries its own missing-contact mask (Fig3-C)."""
    acts = np.atleast_2d(np.asarray(acts, float))
    n_row = acts.shape[0]
    out = np.full(n_row, np.nan)
    for lo in range(0, n_row, chunk):
        hi = min(lo + chunk, n_row)
        block = acts[lo:hi]
        abs_a = ev["A"].abs_r_batch_varmask(block)
        abs_b = ev["B"].abs_r_batch_varmask(block)
        stacked = np.vstack([abs_a, abs_b])
        with np.errstate(invalid="ignore"):
            out[lo:hi] = np.nanmax(stacked, axis=0)
        out[lo:hi][~np.isfinite(abs_a) & ~np.isfinite(abs_b)] = np.nan
    return out


def score_event_detail_single(ev: Dict[str, object], activation: np.ndarray) -> Dict[str, object]:
    """Full per-template A/B detail for one activation (reference path).

    Uses the event's locked common finite mask (stored at build time), so it is
    definitionally the per-draw reference the batch path reproduces.
    """
    act = np.asarray(activation, float)
    finite_mask = np.asarray(ev["finite"], bool)
    a = score_template_r3(ev["grid_a"], ev["A"].F_inter, ev["A"].S_inter,
                          ev["A"].points, ev["A"].support, act, finite_mask)
    b = score_template_r3(ev["grid_b"], ev["B"].F_inter, ev["B"].S_inter,
                          ev["B"].points, ev["B"].support, act, finite_mask)
    abs_a = a["abs_r"]
    abs_b = b["abs_r"]
    cands = [(t, v) for t, v in (("A", abs_a), ("B", abs_b))
             if v is not None and np.isfinite(v)]
    if not cands:
        maxab = np.nan
        best = None
    else:
        best, maxab = max(cands, key=lambda z: z[1])
    return {
        "abs_a": abs_a, "abs_b": abs_b,
        "signed_a": a["signed_r"], "signed_b": b["signed_r"],
        "mirror_a": a["mirror_choice"], "mirror_b": b["mirror_choice"],
        "overlap_a": a["n_overlap"], "overlap_b": b["n_overlap"],
        "maxab": float(maxab) if np.isfinite(maxab) else np.nan,
        "best_template": best,
    }


def score_event_maxab_single(ev: Dict[str, object], activation: np.ndarray) -> float:
    """maxAB for one activation via the per-template reference primitive."""
    return score_event_detail_single(ev, activation)["maxab"]


# --------------------------------------------------------------------------
# cohort statistics (handoff §4.3, §5.2-§5.4)
# --------------------------------------------------------------------------
def seven_band_maxt_pfwer(D: np.ndarray, N: np.ndarray) -> Dict[str, np.ndarray]:
    """Coherent cross-band maxT family-wise error (handoff §5.3).

    ``D`` is the (n_subject, n_band) observed subject-level score; ``N`` is the
    (n_subject, n_band, n_draw) subject-level null score using the SAME draws
    across bands.  Returns per-band ``Cobs``, ``Zobs``, ``pFWER`` plus the
    per-subject Δ = D − median_k N and its cohort median (the figure bar).
    """
    D = np.asarray(D, float)
    N = np.asarray(N, float)
    if D.ndim != 2 or N.ndim != 3 or N.shape[:2] != D.shape:
        raise ValueError("D must be (subject, band); N must be (subject, band, draw)")
    n_draw = N.shape[2]
    Cobs = np.median(D, axis=0)                       # (band,)
    Cnull = np.median(N, axis=0)                      # (band, draw)
    Cnull_med = np.median(Cnull, axis=1)              # (band,)
    Zobs = Cobs - Cnull_med
    Znull = Cnull - Cnull_med[:, None]                # (band, draw)
    M = Znull.max(axis=0)                             # (draw,)  synchronized across bands
    pfwer = np.array([(1 + int(np.sum(M >= Zobs[b]))) / (n_draw + 1)
                      for b in range(D.shape[1])])
    nmed = np.median(N, axis=2)                       # (subject, band)
    per_subject_delta = D - nmed
    return {
        "Cobs": Cobs, "Cnull_median": Cnull_med, "Zobs": Zobs, "pFWER": pfwer,
        "M_null": M, "per_subject_delta": per_subject_delta,
        "cohort_delta_median": np.median(per_subject_delta, axis=0),
        "n_positive": (per_subject_delta > 0).sum(axis=0),
        "n_draw": int(n_draw),
    }


def paired_one_sided_wilcoxon_greater(data: Sequence[float], null: Sequence[float]) -> float:
    """Paired one-sided Wilcoxon signed-rank p for ``data > null``."""
    from scipy.stats import wilcoxon
    a = np.asarray(data, float)
    b = np.asarray(null, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size < 1 or np.allclose(a, b):
        return float("nan")
    try:
        return float(wilcoxon(a, b, alternative="greater").pvalue)
    except ValueError:
        return float("nan")


def _friedman_statistic(delta: np.ndarray) -> float:
    """Friedman chi-square statistic on a (n_subject, n_band) matrix (ties averaged)."""
    from scipy.stats import rankdata
    n, k = delta.shape
    ranks = np.vstack([rankdata(row) for row in delta])   # rank within each subject
    Rj = ranks.sum(axis=0)
    stat = (12.0 / (n * k * (k + 1))) * float((Rj ** 2).sum()) - 3.0 * n * (k + 1)
    return stat


def direct_band_omnibus(delta: np.ndarray, *, n_perm: int = 100000, seed: int = 20260718) -> Dict[str, object]:
    """Direct band-specificity omnibus on the (n_subject, n_band) Δ matrix (§5.4).

    Friedman rank statistic calibrated by within-subject band-label permutation
    (each subject's band values shuffled independently), plus Kendall's W.
    """
    delta = np.asarray(delta, float)
    n, k = delta.shape
    stat_obs = _friedman_statistic(delta)
    kendall_w = stat_obs / (n * (k - 1)) if n * (k - 1) > 0 else float("nan")
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(int(n_perm)):
        permuted = np.array([row[rng.permutation(k)] for row in delta])
        if _friedman_statistic(permuted) >= stat_obs - 1e-12:
            ge += 1
    calibrated_p = (1 + ge) / (int(n_perm) + 1)
    return {
        "n_subjects": int(n), "n_bands": int(k),
        "friedman_statistic": float(stat_obs),
        "kendall_w": float(kendall_w),
        "calibrated_p": float(calibrated_p),
        "n_permutations": int(n_perm),
    }


def _holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def direct_band_contrasts(delta: np.ndarray, band_labels: Sequence[str]) -> list:
    """All pairwise two-sided Wilcoxon band contrasts with Holm correction (§5.4)."""
    from scipy.stats import wilcoxon
    delta = np.asarray(delta, float)
    k = delta.shape[1]
    pairs, raw_p, meds, iqrlo, iqrhi = [], [], [], [], []
    for i in range(k):
        for j in range(i + 1, k):
            diff = delta[:, i] - delta[:, j]
            ok = np.isfinite(diff)
            d = diff[ok]
            if d.size < 1 or np.allclose(d, 0.0):
                p = float("nan")
            else:
                try:
                    p = float(wilcoxon(d, alternative="two-sided").pvalue)
                except ValueError:
                    p = float("nan")
            pairs.append((str(band_labels[i]), str(band_labels[j])))
            raw_p.append(p)
            meds.append(float(np.median(d)) if d.size else float("nan"))
            q = np.percentile(d, [25, 75]) if d.size else (float("nan"), float("nan"))
            iqrlo.append(float(q[0]))
            iqrhi.append(float(q[1]))
    finite_mask = np.isfinite(raw_p)
    holm = np.full(len(raw_p), np.nan)
    if finite_mask.any():
        holm[finite_mask] = _holm(np.asarray(raw_p)[finite_mask])
    return [
        {"band_i": pairs[t][0], "band_j": pairs[t][1],
         "median_difference": meds[t], "iqr_low": iqrlo[t], "iqr_high": iqrhi[t],
         "wilcoxon_p": raw_p[t], "holm_p": float(holm[t])}
        for t in range(len(pairs))
    ]


def within_shaft_permutations(names: Sequence[str], finite: Sequence[bool], *,
                              n_perm: int, seed: int, min_group: int = 4) -> Dict[str, object]:
    """Pure within-shaft permutations with ``min_group`` and NO fallback (§4.2).

    An event is eligible only if every finite contact belongs to a shaft that has
    at least ``min_group`` finite contacts; otherwise it is ``unavailable`` (no
    distance-bin / subject-wide fallback).  Missing contacts keep their identity.
    """
    from src.propagation_skeleton_geometry import parse_shaft
    names = [str(x) for x in names]
    finite = np.asarray(finite, bool)
    finite_idx = np.where(finite)[0]
    by_shaft: Dict[str, list] = {}
    for idx in finite_idx:
        by_shaft.setdefault(str(parse_shaft(names[idx])[0]), []).append(int(idx))
    small = {sh: len(v) for sh, v in by_shaft.items() if len(v) < int(min_group)}
    if small:
        return {"eligible": False, "permutations": None,
                "reason": "shaft_below_min_group",
                "small_shafts": small, "min_group": int(min_group)}
    groups = [np.asarray(v, int) for _, v in sorted(by_shaft.items())]
    base = np.arange(len(names))
    out = np.tile(base, (int(n_perm), 1))
    rng = np.random.default_rng(int(seed))
    for draw in range(int(n_perm)):
        for idx in groups:
            if idx.size > 1:
                out[draw, idx] = rng.permutation(idx)
    return {"eligible": True, "permutations": out, "reason": None,
            "n_shafts": len(groups), "min_group": int(min_group)}


def loo_contact_reconstruction(points: np.ndarray, support: np.ndarray,
                               values: np.ndarray, sigma: float) -> np.ndarray:
    """Leave-one-out kernel-regression reconstruction of each contact's value.

    ``recon[i] = sum_{j!=i} K(p_i,p_j) sup_j v_j / sum_{j!=i} K(p_i,p_j) sup_j`` with
    ``K = exp(-||p_i-p_j||^2 / (2 sigma^2))``. Used by the outcome-blind interictal
    sigma-policy adjudication (§五): a smoothing scale that better reconstructs a
    held-out contact's earliness from its neighbours is more geometrically
    supported, independent of any ictal outcome. A contact with no supported
    neighbour reconstructs to nan.
    """
    pts = np.asarray(points, float)
    sup = np.asarray(support, float)
    v = np.asarray(values, float)
    n = len(pts)
    sig2 = 2.0 * float(sigma) ** 2
    d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)
    K = np.exp(-d2 / sig2)
    np.fill_diagonal(K, 0.0)                      # leave self out
    W = K * sup[None, :]
    num = (W * v[None, :]).sum(axis=1)
    den = W.sum(axis=1)
    out = np.full(n, np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    return out


def permutation_mapping_hash(perms: np.ndarray) -> str:
    """SHA-256 of a permutation-index matrix for cross-use reuse audits (§4.1)."""
    arr = np.ascontiguousarray(np.asarray(perms, dtype="<i8"))
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()
