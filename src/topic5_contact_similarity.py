"""Topic5 contact-level similarity ladder (R1 raw / R2 same-plane kernel),
grid-free counterparts of the field maxAB. See
docs/superpowers/specs/2026-06-30-topic5-contact-similarity-ladder-design.md."""
import numpy as np


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
