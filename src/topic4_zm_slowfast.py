"""Section-8 slow-fast analysis (task §8). Treat z/m/S_G as slow parameters; characterize the fast E/I
burst subsystem of the Z/M+S_G run by its cycle-to-cycle behaviour.

Honest ceiling: this analyses the NATURAL trajectory only. Proving a limit cycle needs a frozen-slow
repeated trajectory + Poincare-section / perturbation-return, which the natural run does not provide.
So `slowfast_verdict` emits only:
  candidate_inner_cycle   -- bursts recur with stationary period + amplitude (relaxation-oscillation-like)
  transient_burst_train   -- period / amplitude drift or escalate (not a stationary cycle)
  not_oscillatory         -- too few bursts to say anything
It NEVER emits 'limit_cycle'.
"""
from __future__ import annotations

import numpy as np

NAN = float("nan")


def _merge_episodes(above, gap_bins):
    idx = np.flatnonzero(np.asarray(above, bool))
    if idx.size == 0:
        return []
    eps, start, prev = [], int(idx[0]), int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - prev > gap_bins:
            eps.append((start, prev + 1))
            start = i
        prev = i
    eps.append((start, prev + 1))
    return eps


def detect_bursts(rate, dt_ms, min_frac=0.3, min_sep_ms=30.0):
    """Burst peaks of a fine rate trace via threshold-crossing episodes (robust to flat-topped bursts,
    unlike a strict local-maxima finder). Returns (peak_idx, peak_amp)."""
    r = np.asarray(rate, float)
    base = float(np.median(r))
    peak = float(r.max()) if r.size else 0.0
    amp = peak - base
    if amp <= 0:
        return np.array([], int), np.array([], float)
    level = base + min_frac * amp
    gap = max(1, int(round(min_sep_ms / dt_ms)))
    eps = _merge_episodes(r >= level, gap)
    pk = np.array([i0 + int(np.argmax(r[i0:i1])) for i0, i1 in eps], int)
    return pk, r[pk]


def cycle_stats(peak_idx, peak_amp, dt_ms):
    """Inter-burst intervals + amplitudes and their drift/stationarity (last-half CV, first-third vs
    last-third drift fraction, linear slope). All scale-free where it matters."""
    pk = np.asarray(peak_idx, int)
    amp = np.asarray(peak_amp, float)
    n = int(pk.size)
    out = dict(n_bursts=n, ibi_ms=np.array([]), amp=amp,
               ibi_cv_tail=NAN, amp_cv_tail=NAN, ibi_drift_frac=NAN, amp_drift_frac=NAN,
               ibi_slope=NAN, amp_slope=NAN)
    if n < 2:
        return out
    ibi = np.diff(pk) * dt_ms

    def cv(x):
        x = np.asarray(x, float)
        return float(np.std(x) / (abs(np.mean(x)) + 1e-12)) if x.size else NAN

    def drift(x):
        x = np.asarray(x, float)
        if x.size < 2:
            return NAN
        k = max(1, x.size // 3)
        return float(abs(np.mean(x[-k:]) - np.mean(x[:k])) / (abs(np.mean(x)) + 1e-12))

    def slope(x):
        x = np.asarray(x, float)
        return float(np.polyfit(np.arange(x.size), x, 1)[0]) if x.size >= 2 else NAN

    out.update(ibi_ms=ibi,
               ibi_cv_tail=cv(ibi[len(ibi) // 2:]), amp_cv_tail=cv(amp[amp.size // 2:]),
               ibi_drift_frac=drift(ibi), amp_drift_frac=drift(amp),
               ibi_slope=slope(ibi), amp_slope=slope(amp))
    return out


def slowfast_verdict(cs, min_cycles=4, drift_tol=0.30, cv_tol=0.20):
    """candidate_inner_cycle iff >= min_cycles bursts AND period+amplitude are stationary (tail CV and
    first-vs-last drift both under tolerance); transient_burst_train if they drift; else not_oscillatory."""
    if int(cs["n_bursts"]) < min_cycles:
        return "not_oscillatory"
    stationary = (cs["ibi_cv_tail"] < cv_tol and cs["amp_cv_tail"] < cv_tol
                  and cs["ibi_drift_frac"] < drift_tol and cs["amp_drift_frac"] < drift_tol)
    return "candidate_inner_cycle" if stationary else "transient_burst_train"
