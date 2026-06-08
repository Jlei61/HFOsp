"""Source-space (oracle) event metrics for the SNN heterogeneity grid (spec
2026-06-08 §4). Operate on per-E-neuron spike booleans + coords; electrode-
independent so every grid cell is comparable."""
from __future__ import annotations

import numpy as np


def onset_times(E_spk_bool, dt, t_kick):
    """First-spike time (ms) after the kick per E neuron; NaN if it never fires."""
    i_kick = int(round(t_kick / dt))
    post = np.asarray(E_spk_bool)[i_kick:]
    ever = post.any(axis=0)
    return np.where(ever, (post.argmax(axis=0) + i_kick) * dt, np.nan)


def onset_axis(posE, onset, min_n=20):
    """Propagation axis from the onset-time spatial gradient (lstsq t ~ a + g·x).
    Returns a unit 2-vector (direction of increasing onset) or None if too few
    onsets / no gradient."""
    onset = np.asarray(onset, float)
    fin = np.isfinite(onset)
    if fin.sum() < min_n:
        return None
    X = np.asarray(posE, float)[fin]
    t = onset[fin]
    Xc = X - X.mean(0)
    g, *_ = np.linalg.lstsq(Xc, t - t.mean(), rcond=None)
    nrm = float(np.linalg.norm(g))
    return None if nrm < 1e-9 else g / nrm


def peak_active_fraction(E_spk_bool, dt, t_lo, t_hi, bin_ms=2.0):
    """Max over bin_ms windows of (distinct E neurons that fired in the bin)/NE,
    within [t_lo,t_hi) ms. Mirrors engine kick_probe.peak_active_fraction but lives
    in tracked src so the grid runner needn't import the gitignored engine for it."""
    spk = np.asarray(E_spk_bool)
    nsteps, NE = spk.shape
    bs = int(round(bin_ms / dt))
    i_lo = int(round(t_lo / dt))
    i_hi = int(round(t_hi / dt))
    best = 0.0
    for b0 in range(i_lo, i_hi, bs):
        b1 = min(b0 + bs, i_hi)
        if b1 > b0:
            best = max(best, spk[b0:b1].any(axis=0).sum() / NE)
    return float(best)
