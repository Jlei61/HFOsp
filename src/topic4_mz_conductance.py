"""Pure metrics for the Topic-4 MZ conductance/global-inhibition screen."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import spearmanr


def staircase_metrics(events, z_mean, dt, *, transition_ms=None, transition_guard_ms=100.0,
                      pre_ms=20.0, post_ms=20.0):
    """Quantify event-locked increments of D=1-z_mean using a frozen event detector.

    ``events`` must contain ``t_on``/``t_off`` in ms.  The function does not
    re-detect or relabel events and is therefore safe to use with the same-seed
    slow-off event bar.
    """
    z = np.asarray(z_mean, float)
    if z.ndim != 1 or z.size == 0 or dt <= 0:
        raise ValueError("z_mean must be a non-empty 1-D trace and dt must be positive")
    D = 1.0 - z
    rows = []
    event_mask = np.zeros(D.size, bool)
    npre = max(1, int(round(pre_ms / dt)))
    npost = max(1, int(round(post_ms / dt)))
    excluded_nonreturning = 0
    excluded_posttransition = 0
    for ev in events:
        if not bool(ev.get("returned", False)):
            excluded_nonreturning += 1
            continue
        if transition_ms is not None and float(ev["t_off"]) >= float(transition_ms) - transition_guard_ms:
            excluded_posttransition += 1
            continue
        on = int(round(float(ev["t_on"]) / dt))
        off = int(round(float(ev["t_off"]) / dt))
        on = min(max(on, 0), D.size - 1)
        off = min(max(off, on), D.size - 1)
        pre = D[max(0, on - npre):on]
        post = D[off:min(D.size, off + npost)]
        if pre.size == 0 or post.size == 0:
            continue
        d_pre = float(np.median(pre))
        d_post = float(np.median(post))
        rows.append(dict(t_on=float(ev["t_on"]), t_off=float(ev["t_off"]),
                         D_pre=d_pre, D_post=d_post, delta_D=d_post - d_pre,
                         returned=bool(ev.get("returned", False))))
        event_mask[on:min(D.size, off + npost)] = True

    deltas = np.asarray([r["delta_D"] for r in rows], float)
    dpre = np.asarray([r["D_pre"] for r in rows], float)
    rho = float(spearmanr(np.arange(len(dpre)), dpre).statistic) if len(dpre) >= 3 else float("nan")
    increments = np.maximum(np.diff(D, prepend=D[0]), 0.0)
    total_inc = float(increments.sum())
    locked_frac = float(increments[event_mask].sum() / total_inc) if total_inc > 0 else float("nan")
    return dict(
        n_events=len(rows),
        n_returning=int(sum(r["returned"] for r in rows)),
        median_delta_D=float(np.median(deltas)) if deltas.size else float("nan"),
        positive_delta_fraction=float(np.mean(deltas > 0.0)) if deltas.size else float("nan"),
        event_locked_positive_increment_fraction=locked_frac,
        event_index_Dpre_spearman=rho,
        D_max=float(np.max(D)),
        excluded_nonreturning=excluded_nonreturning,
        excluded_posttransition=excluded_posttransition,
        rows=rows,
    )


def oscillation_metrics(rate_hz, dt, *, analysis_start_ms, baseline_rate, baseline_sigma,
                        active_fraction=None, af_bin_ms=None, baseline_af_q95=None,
                        runaway=False, min_peak_distance_ms=50.0):
    """Distinguish burst modulation from a flat elevated plateau and audit recovery."""
    rate = np.asarray(rate_hz, float)
    if rate.ndim != 1 or rate.size == 0 or dt <= 0:
        raise ValueError("rate_hz must be a non-empty 1-D trace and dt must be positive")
    start = min(rate.size, max(0, int(round(analysis_start_ms / dt))))
    seg = rate[start:]
    if seg.size == 0:
        return dict(n_bursts=0, n_ibi=0, dominant_hz=float("nan"), modulation=float("nan"),
                    tail_rate_band=False, tail_mean_hz=float("nan"), oscillatory_candidate=False)
    win = max(1, int(round(10.0 / dt)))
    smooth = np.convolve(seg, np.ones(win) / win, mode="same")
    spread = float(np.percentile(smooth, 90) - np.percentile(smooth, 10))
    prominence = max(5.0, 0.2 * spread)
    distance = max(1, int(round(min_peak_distance_ms / dt)))
    peaks, _ = find_peaks(smooth, prominence=prominence, distance=distance)
    ibis = np.diff(peaks) * dt
    p90 = float(np.percentile(smooth, 90))
    p10 = float(np.percentile(smooth, 10))
    modulation = (p90 - p10) / max(p90, 1e-12)
    if smooth.size >= max(16, int(round(500.0 / dt))):
        fs = 1000.0 / dt
        freq, power = welch(smooth - smooth.mean(), fs=fs,
                            nperseg=min(smooth.size, int(round(2000.0 / dt))))
        keep = (freq >= 0.5) & (freq <= 20.0)
        if np.any(keep):
            band_power = power[keep]
            peak_i = int(np.argmax(band_power))
            dominant = float(freq[keep][peak_i])
            spectral_peak_ratio = float(band_power[peak_i] / max(np.median(band_power), 1e-12))
        else:
            dominant = float("nan")
            spectral_peak_ratio = float("nan")
    else:
        dominant = float("nan")
        spectral_peak_ratio = float("nan")
    high_duration_ms = 0.0
    recruitment_pass = False
    if active_fraction is not None and af_bin_ms is not None and baseline_af_q95 is not None:
        af = np.asarray(active_fraction, float)
        af_start = min(af.size, max(0, int(round(analysis_start_ms / float(af_bin_ms)))))
        high = af[af_start:] > float(baseline_af_q95)
        if high.size:
            edges = np.flatnonzero(np.diff(np.r_[False, high, False]))
            longest = int(np.max(edges[1::2] - edges[::2])) if edges.size else 0
            high_duration_ms = float(longest * float(af_bin_ms))
            recruitment_pass = bool(np.max(af[af_start:]) > float(baseline_af_q95))
    tail_n = int(round(2000.0 / dt))
    tail = rate[-tail_n:] if rate.size >= tail_n else np.asarray([], float)
    band_hi = float(baseline_rate + 1.5 * max(baseline_sigma, 1e-9))
    band_lo = max(0.0, float(baseline_rate - 1.5 * max(baseline_sigma, 1e-9)))
    tail_rate_band = bool(tail.size == tail_n and band_lo <= float(tail.mean()) <= band_hi)
    spectral_pass = bool(np.isfinite(spectral_peak_ratio) and spectral_peak_ratio >= 5.0)
    oscillatory = bool(
        not runaway and recruitment_pass and high_duration_ms >= 1000.0
        and peaks.size >= 4 and ibis.size >= 3 and modulation >= 0.3 and spectral_pass
    )
    return dict(
        n_bursts=int(peaks.size), n_ibi=int(ibis.size),
        median_ibi_ms=float(np.median(ibis)) if ibis.size else float("nan"),
        dominant_hz=dominant, spectral_peak_ratio=spectral_peak_ratio, spectral_pass=spectral_pass,
        modulation=float(modulation), high_duration_ms=high_duration_ms,
        recruitment_pass=recruitment_pass, oscillatory_candidate=oscillatory,
        tail_rate_band=tail_rate_band,
        tail_mean_hz=float(tail.mean()) if tail.size else float("nan"),
        baseline_band_lo=band_lo, baseline_band_hi=band_hi,
    )
