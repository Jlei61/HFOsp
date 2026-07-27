"""Offline temporal/spatial morphology audit for native carrier confirmations.

The primary source-space continuation uses 25 ms bins.  Those bins are
appropriate for survival and slow-drift tests, but they can average over a
30--80 Hz rhythm and must never be used alone to label a carrier as a fixed
point.  This module keeps the two statements separate:

* ``coarse_rate_label`` describes only the 25 ms population-rate envelope;
* ``readout_temporal_class`` describes native-sampled virtual-electrode
  periodicity and remains a candidate label, not an empirical ictal gate.

The functions are pure and operate on saved arrays.  They neither alter the
pre-registered carrier verdict nor substitute for the blocked real-data
observation reference lock.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as ss

from src import topic4_zm_empirical_carrier as EC


MORPHOLOGY_VERSION = "zm_carrier_morphology_v1_2026-07-27"


def _dominant_frequencies(lfp: np.ndarray, fs: float) -> np.ndarray:
    x = ss.detrend(np.asarray(lfp, float), axis=0)
    nperseg = min(x.shape[0], max(32, int(round(fs))))
    f, power = ss.welch(x, fs=fs, nperseg=nperseg, axis=0)
    band = (f >= 3.0) & (f <= min(150.0, 0.45 * fs))
    if band.sum() < 4:
        return np.full(x.shape[1], np.nan)
    return f[band][np.argmax(power[band], axis=0)]


def _safe_cv(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    mean = float(np.mean(x)) if x.size else 0.0
    return float(np.std(x) / mean) if mean > 1e-12 else float("nan")


def characterize_confirmation(
    rate_25ms,
    lfp,
    fs,
    *,
    burn_in_ms,
    kymo_axial,
    bin_ms,
):
    """Return a descriptive carrier-type audit from one saved confirmation.

    ``narrowband_readout_candidate`` requires a reproducible dominant
    frequency across at least half the contacts and low median spectral
    entropy.  It is deliberately named a candidate: this result alone does not
    establish a periodic source-space orbit, propagation, or ictal likeness.
    """
    rate = np.asarray(rate_25ms, float)
    x = np.asarray(lfp, float)
    kymo = np.asarray(kymo_axial, float)
    fs = float(fs)
    bin_ms = float(bin_ms)
    if rate.ndim != 1:
        raise ValueError("rate_25ms must be one-dimensional")
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or x.shape[1] < 1:
        raise ValueError("lfp must be time x contact")
    if kymo.ndim != 2 or kymo.shape[1] != rate.size:
        raise ValueError("kymo_axial time axis must match rate_25ms")
    if fs <= 0 or bin_ms <= 0:
        raise ValueError("fs and bin_ms must be positive")

    burn_rate = min(rate.size, max(0, int(round(float(burn_in_ms) / bin_ms))))
    burn_lfp = min(x.shape[0], max(0, int(round(float(burn_in_ms) * 1e-3 * fs))))
    rate_post = rate[burn_rate:]
    x_post = x[burn_lfp:]
    kymo_post = kymo[:, burn_rate:]
    if rate_post.size < 8 or x_post.shape[0] < max(32, int(fs)):
        raise ValueError("confirmation trace is too short after burn-in")

    rate_cv = _safe_cv(rate_post)
    coarse_label = (
        f"tonic_at_{bin_ms:g}ms"
        if np.isfinite(rate_cv) and rate_cv < 0.05
        else f"modulated_at_{bin_ms:g}ms"
    )

    dom = _dominant_frequencies(x_post, fs)
    finite_dom = dom[np.isfinite(dom)]
    dom_median = float(np.median(finite_dom)) if finite_dom.size else float("nan")
    dom_iqr = (
        float(np.percentile(finite_dom, 75) - np.percentile(finite_dom, 25))
        if finite_dom.size else float("nan")
    )
    tol = max(2.0, 0.05 * dom_median) if np.isfinite(dom_median) else float("nan")
    agreement = (
        float(np.mean(np.abs(finite_dom - dom_median) <= tol))
        if finite_dom.size else 0.0
    )

    entropy = np.asarray([
        EC.spectral_entropy(x_post[:, i], fs) for i in range(x_post.shape[1])
    ])
    drift = np.asarray([
        EC.inst_freq_drift(x_post[:, i], fs) for i in range(x_post.shape[1])
    ])
    comb = np.asarray([
        EC.harmonic_comb_concentration(x_post[:, i], fs)
        for i in range(x_post.shape[1])
    ])
    entropy_median = float(np.nanmedian(entropy))
    narrowband = (
        np.isfinite(dom_median)
        and dom_median >= 5.0
        and entropy_median < 0.55
        and agreement >= 0.50
    )

    time_mean = np.mean(kymo_post, axis=1)
    valid_space = time_mean > 1e-12
    temporal_cv = np.std(kymo_post, axis=1)[valid_space] / time_mean[valid_space]
    per_time_mean = np.mean(kymo_post, axis=0)
    valid_time = per_time_mean > 1e-12
    spatial_cv = (
        np.std(kymo_post, axis=0)[valid_time] / per_time_mean[valid_time]
    )

    return {
        "morphology_version": MORPHOLOGY_VERSION,
        "coarse_rate_label": coarse_label,
        "coarse_rate_cv": rate_cv,
        "readout_temporal_class": (
            "narrowband_readout_candidate"
            if narrowband else "broadband_or_asynchronous_readout"
        ),
        "dominant_frequency_median_hz": dom_median,
        "dominant_frequency_iqr_hz": dom_iqr,
        "dominant_frequency_agreement": agreement,
        "dominant_frequency_per_contact_hz": dom.tolist(),
        "spectral_entropy_median": entropy_median,
        "instantaneous_frequency_drift_median_hz": float(np.nanmedian(drift)),
        "harmonic_comb_median": float(np.nanmedian(comb)),
        "phase_coherence": float(EC.phase_coherence(x_post, fs)),
        "kymograph_temporal_cv_median": (
            float(np.median(temporal_cv)) if temporal_cv.size else float("nan")
        ),
        "kymograph_spatial_cv_mean": (
            float(np.mean(spatial_cv)) if spatial_cv.size else float("nan")
        ),
        "claim_boundary": (
            "descriptive native-readout morphology only; not a real-data "
            "observation gate and not sufficient to select fixed-point versus "
            "Floquet source operator without a fine source-space audit"
        ),
    }
