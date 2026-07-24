"""FCXR-HEO2 broadband-spiky diagnosis — Phase-0 estimators + 4-class state map, Phase-1 arm classifier.

Fixes the HEO1 oscillation_probe 3.906 Hz resolution-floor artifact (2 s Welch + event-IPI agreement)
and scores each state against the REAL E1146 seizure six-band ΔdB vector instead of the (mis-specified)
HEO1 binary gate. Band order everywhere: [1-4, 4-8, 8-13, 13-30, 30-80, 80-150] Hz.
Spec: docs/superpowers/specs/2026-07-24-topic4-heo2-broadband-diagnostic-design.md.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks, welch
from scipy.stats import kurtosis

# real E1146 seizure territory-median six-band ΔdB (measured, real_e1146_seizure_gate.json)
REAL_E1146_DDB = (12.0, 10.4, 8.6, 8.3, 5.0, -1.2)


# ----------------------------------------------------------------- Phase-0 estimators
def dominant_2s(sig, fs, lo=1.0, hi=200.0):
    """Dominant frequency via a 2 s Welch (~0.5 Hz) — resolves ~3.5 Hz instead of the 3.906 Hz floor."""
    sig = np.asarray(sig, float)
    nper = min(len(sig), int(round(2 * fs)))
    f, p = welch(sig - sig.mean(), fs=fs, nperseg=nper)
    m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(p[m])]) if np.any(m) else float("nan")


def event_ipi_hz(rate, fs):
    """Event repetition frequency = 1 / median inter-peak interval (a rhythm only if this agrees with
    dominant_2s). Peaks = rate excursions above median + 0.3*(max-median), >=40 ms apart."""
    r = np.asarray(rate, float)
    thr = np.median(r) + 0.3 * (r.max() - np.median(r))
    peaks, _ = find_peaks(r, height=thr, distance=max(1, int(0.04 * fs)))
    if len(peaks) < 3:
        return float("nan")
    return float(1.0 / np.median(np.diff(peaks) / fs))


def spikiness(rate):
    """Excess kurtosis of the population rate: spike-wave (sharp, sparse) >> 0; pure sinusoid ~ -1.5."""
    return float(kurtosis(np.asarray(rate, float), fisher=True))


def spectral_entropy(sig, fs):
    """Shannon entropy of the normalized PSD: low = narrowband, high = broadband."""
    sig = np.asarray(sig, float)
    f, p = welch(sig - sig.mean(), fs=fs, nperseg=min(len(sig), 1024))
    p = p / (p.sum() + 1e-300)
    return float(-np.sum(p * np.log(p + 1e-300)))


def bw90(sig, fs):
    """90 %-power bandwidth (5th..95th cumulative-power frequency): narrow for a tone, broad for noise."""
    sig = np.asarray(sig, float)
    f, p = welch(sig - sig.mean(), fs=fs, nperseg=min(len(sig), 1024))
    c = np.cumsum(p) / (p.sum() + 1e-300)
    lo = f[min(np.searchsorted(c, 0.05), len(f) - 1)]
    hi = f[min(np.searchsorted(c, 0.95), len(f) - 1)]
    return float(hi - lo)


def spectral_distance_to_real(six_band_ddb):
    """(L2 dB distance, cosine similarity) of a six-band ΔdB vector vs the real E1146 seizure vector."""
    m = np.asarray(six_band_ddb, float)
    r = np.asarray(REAL_E1146_DDB, float)
    l2 = float(np.linalg.norm(m - r))
    cos = float(np.dot(m, r) / (np.linalg.norm(m) * np.linalg.norm(r) + 1e-300))
    return l2, cos


def duty_cycle(rate, fs, frac=0.3, win_ms=30.0):
    """Ictal-active fraction: fraction of time the (lightly 30 ms-smoothed) rate exceeds 0.3*q95.
    Sparse event train -> low (~0.1-0.2); sustained high state -> high (~0.9)."""
    r = np.asarray(rate, float)
    w = max(1, int(win_ms / 1000.0 * fs))
    roll = np.convolve(r, np.ones(w) / w, mode="same") if w > 1 else r
    thr = frac * np.percentile(roll, 95)
    return float(np.mean(roll > thr))


def max_silence_gap_ms(rate, fs, frac=0.3, win_ms=30.0):
    """Longest continuous below-threshold (silent) run, in ms."""
    r = np.asarray(rate, float)
    w = max(1, int(win_ms / 1000.0 * fs))
    roll = np.convolve(r, np.ones(w) / w, mode="same") if w > 1 else r
    active = roll > frac * np.percentile(roll, 95)
    gaps, run = [], 0
    for a in active:
        if not a:
            run += 1
        elif run:
            gaps.append(run); run = 0
    if run:
        gaps.append(run)
    return float((max(gaps) if gaps else 0) / fs * 1000.0)


def _mean_crossings(x):
    x = np.asarray(x, float)
    x = x - x.mean()
    return int(np.sum(np.diff(np.sign(x)) != 0))


def phase1_verdict(pre, post, m_mean_post, m_off_dist, m_off_coverage, safety):
    """Phase-1 arm verdict (spec §3, 8 conjunctive criteria). pre/post are metric dicts; m_mean_post is
    the adaptation trace in the post-enable window; m_off_* are the reference (no-adaptation) arm's values.
    verdict ∈ {transformed_broadband_spiky, unchanged_16Hz, collapsed_sparse, silenced, stalled,
    no_high_state, unsafe}. 'transformed' requires all six transform criteria (given established+safe)."""
    ddb = post["six_band_ddb"]
    crit = dict(
        established=bool(pre["mean_rate"] > 60.0 and pre["coherence"] >= 0.9),      # 1: 16Hz high state pre-enable
        spiky_freq=bool(3.0 <= post["dominant_hz"] <= 8.0 and abs(post["dominant_hz"] - post["event_ipi_hz"]) <= 2.0),  # 2
        broadband=bool(ddb[0] > 0 and ddb[1] > 0 and ddb[3] > 0),                   # 3: low + beta up
        closer_to_real=bool(post["dist_to_real"] < m_off_dist),                     # 4
        not_collapsed=bool(post["mean_rate"] > 20.0),                               # 5: not back to interictal
        coverage_kept=bool(post["coverage"] >= max(m_off_coverage, 4)),             # 6: not degraded to focal
        bursting=bool(_mean_crossings(m_mean_post) >= 3),                           # 7: m lags-and-recovers
        safe=bool(not safety.get("numerical_unsafe", False) and safety.get("runaway_early_stop_ms") is None),  # 8
    )
    if not crit["safe"]:
        v = "unsafe"
    elif not crit["established"]:
        v = "no_high_state"
    elif all(crit[k] for k in ("spiky_freq", "broadband", "closer_to_real", "not_collapsed", "coverage_kept", "bursting")):
        v = "transformed_broadband_spiky"
    elif post["mean_rate"] < 5.0:
        v = "silenced"
    elif post["dominant_hz"] > 13.0:
        v = "unchanged_16Hz"
    elif post["dominant_hz"] < 8.0:
        v = "collapsed_sparse"
    else:
        v = "stalled"
    return dict(verdict=v, criteria=crit)


def classify_state(m):
    """4-class label (spec §2). Precedence: target_like_spiky > tonic_16Hz_cycle > sparse_event_train
    > transitional. m keys: dominant_hz, duty_cycle, coverage, six_band_ddb, coherent."""
    dom, duty, cov = m["dominant_hz"], m["duty_cycle"], m["coverage"]
    ddb, coh = m["six_band_ddb"], m.get("coherent", False)
    if 3.0 <= dom <= 8.0 and all(x > 0 for x in ddb) and duty >= 0.6 and cov >= 8:
        return "target_like_spiky"
    if dom > 13.0 and duty > 0.7 and coh:
        return "tonic_16Hz_cycle"
    if dom < 8.0 and duty < 0.4:
        return "sparse_event_train"
    return "transitional"
