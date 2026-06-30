"""Topic5 contact-level similarity ladder (R1 raw / R2 same-plane kernel),
grid-free counterparts of the field maxAB. See
docs/superpowers/specs/2026-06-30-topic5-contact-similarity-ladder-design.md."""
import numpy as np
from scipy.stats import pearsonr


def kernel_smooth_at_contacts(values, source_pts, eval_pts, support, sigma):
    """Nadaraya-Watson Gaussian smoothing identical to smooth_field, but
    evaluated at arbitrary eval_pts instead of a grid (no grid -> no pixel
    density reweighting). Mirror = pass y-flipped eval_pts.

    values, support: (n_src,); source_pts: (n_src,2); eval_pts: (n_eval,2).
    Returns (n_eval,) with NaN where total support <= 1e-12.

    Weight per source k at eval point i:
        w_k = support_k * exp(-((x_i - x_k)^2 + (y_i - y_k)^2) / sig2)
    where sig2 = 2 * sigma^2  (byte-identical to smooth_field's sig2).
    T_i = sum(w_k * value_k for finite k) / sum(w_k for finite k).
    Gate: if sum(w_k for all k) <= 1e-12, output is NaN (matches smooth_field S gate).
    """
    v = np.asarray(values, float)
    sup = np.asarray(support, float)
    src = np.asarray(source_pts, float)
    ev = np.asarray(eval_pts, float)
    # MUST match smooth_field's sig2 exactly: sig2 = 2.0 * sigma_xy ** 2
    sig2 = 2.0 * float(sigma) ** 2
    out = np.full(ev.shape[0], np.nan)
    fin = np.isfinite(v)
    for i in range(ev.shape[0]):
        d2 = (src[:, 0] - ev[i, 0]) ** 2 + (src[:, 1] - ev[i, 1]) ** 2
        w = sup * np.exp(-d2 / sig2)
        wsum_all = w.sum()                      # support gate uses all sources (as smooth_field S)
        if wsum_all <= 1e-12:
            continue
        wf = w[fin]
        if wf.sum() > 1e-12:
            out[i] = float((wf * v[fin]).sum() / wf.sum())
    return out


# ---------------------------------------------------------------------------
# Task 2: polarity-free maxAB similarity (raw + same-plane kernel)
# ---------------------------------------------------------------------------

def _pearson_over_contacts(a, b):
    """Pearson r over finite, non-degenerate contacts; NaN if < 3 or zero std."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def contact_corr(rank, value, *, mode, source_pts, support, sigma, mirror=False):
    """Single-template signed Pearson between rank and value fields.

    mode="raw":    plain Pearson(rank, value), no geometry.
    mode="kernel": both fields Gaussian-smoothed at contacts before correlation.
                   mirror=True flips EVAL points' y (activation field only);
                   rank field always uses identity eval (source_pts -> source_pts).
    """
    rank = np.asarray(rank, float)
    value = np.asarray(value, float)
    if mode == "raw":
        return _pearson_over_contacts(rank, value)
    pts = np.asarray(source_pts, float)
    f_rank = kernel_smooth_at_contacts(rank, pts, pts, support, sigma)   # identity eval
    eval_pts = pts.copy()
    if mirror:
        eval_pts[:, 1] = -eval_pts[:, 1]        # flip EVAL points y only (source unchanged)
    f_val = kernel_smooth_at_contacts(value, pts, eval_pts, support, sigma)
    return _pearson_over_contacts(f_rank, f_val)


def _abs_mirror(rank, value, *, mode, source_pts, support, sigma):
    """Per-template abs-mirror score: abs(max(c_identity, c_mirror)) for kernel;
    abs(pearson) for raw. Replicates _abs_corr + corr_pair_mirror_invariant."""
    if mode == "raw":
        c = contact_corr(rank, value, mode="raw", source_pts=source_pts,
                         support=support, sigma=sigma)
        return abs(c) if np.isfinite(c) else np.nan
    c_id = contact_corr(rank, value, mode="kernel", source_pts=source_pts,
                        support=support, sigma=sigma, mirror=False)
    c_mr = contact_corr(rank, value, mode="kernel", source_pts=source_pts,
                        support=support, sigma=sigma, mirror=True)
    cand = [c for c in (c_id, c_mr) if np.isfinite(c)]
    return abs(max(cand)) if cand else np.nan   # max-by-value then abs (== _abs_corr)


def polarity_free_maxab(rank_a, rank_b, value, *, mode, source_pts, support, sigma):
    """maxAB polarity-free similarity: max(_abs_mirror(A), _abs_mirror(B)).

    If rank_b is None (single-template subject), returns _abs_mirror(A).
    Replicates window_maxab from run_topic5_axis_alignment.py.
    """
    r_a = _abs_mirror(rank_a, value, mode=mode, source_pts=source_pts,
                      support=support, sigma=sigma)
    if rank_b is None:
        return r_a
    r_b = _abs_mirror(rank_b, value, mode=mode, source_pts=source_pts,
                      support=support, sigma=sigma)
    vals = [v for v in (r_a, r_b) if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan
