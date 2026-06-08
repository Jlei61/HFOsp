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


def event_peak_time(rate, dt, t_lo, t_hi):
    """argmax time (ms) of the population rate within [t_lo, t_hi)."""
    i_lo = int(round(t_lo / dt)); i_hi = int(round(t_hi / dt))
    seg = np.asarray(rate, float)[i_lo:i_hi]
    return float((i_lo + int(seg.argmax())) * dt) if seg.size else float("nan")


def pre_kick_ignition(rate, dt, t_kick, rest_lo=20.0, thresh_hz=10.0):
    """Did the network ignite BEFORE the kick? (review fix 1B: unmatched/mean_only
    cores self-ignite ~80ms pre-kick, so their post-kick 'event' is not evoked.)
    Scans [rest_lo, t_kick); ignition = population rate crosses thresh_hz.
    Returns (ignited: bool, latency_ms or nan). rest_lo skips the t=0 startup
    transient; thresh_hz sits well above the ~2Hz quiescent rest."""
    i_lo = int(round(rest_lo / dt)); i_k = int(round(t_kick / dt))
    seg = np.asarray(rate, float)[i_lo:i_k]
    cross = np.flatnonzero(seg > thresh_hz)
    if cross.size:
        return True, float((i_lo + int(cross[0])) * dt)
    return False, float("nan")


def self_limit(rate, dt, t_kick, rest_win=(20.0, 50.0), decay_after=120.0,
               decay_dur=80.0, return_factor=1.5, burst_frac=0.5):
    """Self-limitation from the FULL rate trace (review fix 1A: the engine's
    [50,150] 'returned' reference can contain a pre-kick ignition; this uses a
    clean early quiescent window + a decay-ratio + a burst-duration instead of a
    single binary).

      rest_rate     : mean over rest_win (clean pre-event/pre-ignition ref)
      peak, peak_t  : max post-kick + its time
      decay_ratio   : mean rate over [peak_t+decay_after, +decay_dur] / rest_rate
      returned      : decay_ratio <= return_factor (activity falls back near rest)
      burst_duration_ms : total time the rate stays above rest + burst_frac*(peak-rest)
    """
    rate = np.asarray(rate, float)
    t = np.arange(len(rate)) * dt
    rl, rh = rest_win
    rest = float(rate[(t >= rl) & (t < rh)].mean())
    post = t >= t_kick
    peak = float(rate[post].max()); peak_t = float(t[post][rate[post].argmax()])
    dlo, dhi = peak_t + decay_after, peak_t + decay_after + decay_dur
    dmask = (t >= dlo) & (t < dhi)
    decay = float(rate[dmask].mean()) if dmask.any() else float(rate[-1])
    ratio = decay / max(rest, 1e-6)
    thr = rest + burst_frac * max(peak - rest, 1e-9)
    dur = float((rate > thr).sum() * dt)
    return dict(rest_rate=rest, peak=peak, peak_t=peak_t, decay_rate=decay,
                decay_ratio=ratio, returned=bool(ratio <= return_factor),
                burst_duration_ms=dur)
