"""Criterion 10: separate an oscillatory high state from a tonic fixed point.

The nine LFP-based clauses in :mod:`src.topic4_global_recruited_oscillation`
describe the *shape* of the detrended 20-100 Hz contact spectrum.  That shape is
satisfied just as well by a small ripple riding on a near-saturated tonic
plateau as by a genuine population oscillation, because linear detrending
removes the plateau and leaves only the ripple.  Measured on this substrate:

* the interictal low state (q=1) oscillates at 22 Hz and moves the population
  firing rate by 54% of its own mean -- cells genuinely stop and restart;
* every 30-80 Hz high state found so far moves it by 2-4.5%, with essentially
  every excitatory cell firing in every 20 ms window.

So the population-rate modulation depth, not the LFP spectrum, is what
distinguishes the two.  This module measures it and applies the exclusion.

**Threshold provenance (stated because it matters):** the 20% clause was
declared after the frozen-q locator scan surfaced the issue, and before any
dynamic Stage B trajectory was scored.  It is deliberately lenient -- this same
substrate reaches 54% in its own baseline regime -- and its rationale is
structural: below 20%, more than four fifths of the population's firing is
unmodulated tonic activity, so the state's identity is the fixed point rather
than the rhythm.  The raw depth is always reported so a reader can apply a
different bar.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt

MODULATION_DEPTH_FLOOR = 0.20
N_PHASE_BINS = 24


def population_rate_modulation(rate_hz, *, dt_ms, band_hz=(20.0, 100.0),
                               n_phase_bins=N_PHASE_BINS):
    """Cycle-averaged peak-to-trough swing of the population firing rate.

    The dominant in-band frequency is taken from the periodogram, the rate is
    band-passed in a +-10 Hz window around it, and the Hilbert phase of that
    component is used to build a cycle-averaged rate profile.  Peak-to-trough of
    that profile, divided by the mean rate, is the modulation depth: 1.0 means
    the population is fully silenced once per cycle, 0.02 means the rhythm is a
    2% ripple on an otherwise constant plateau.
    """
    rate = np.asarray(rate_hz, float)
    if rate.ndim != 1 or len(rate) < 64:
        raise ValueError("rate_hz must be a one-dimensional segment")
    fs_hz = 1000.0 / float(dt_ms)
    mean_rate = float(rate.mean())
    if mean_rate <= 0.0:
        raise ValueError("population rate segment has no spikes")
    frequency, power = periodogram(rate - mean_rate, fs=fs_hz, window="hann",
                                   detrend="linear")
    keep = (frequency >= float(band_hz[0])) & (frequency <= float(band_hz[1]))
    if not np.any(keep):
        raise ValueError("band has no Fourier bins for this segment")
    dominant_hz = float(frequency[keep][int(np.argmax(power[keep]))])
    low = max(1.0, dominant_hz - 10.0)
    high = min(0.45 * fs_hz, dominant_hz + 10.0)
    if not low < high:
        raise ValueError("dominant frequency leaves no filter band")
    sos = butter(4, (low, high), btype="bandpass", fs=fs_hz, output="sos")
    component = sosfiltfilt(sos, rate - mean_rate)
    phase = np.angle(hilbert(component))
    edges = np.linspace(-np.pi, np.pi, int(n_phase_bins) + 1)
    index = np.digitize(phase, edges) - 1
    profile = np.array([
        rate[index == k].mean() if np.any(index == k) else np.nan
        for k in range(int(n_phase_bins))])
    if np.all(np.isnan(profile)):
        raise ValueError("cycle-averaged profile is empty")
    peak_to_trough = float(np.nanmax(profile) - np.nanmin(profile))
    return {
        "dominant_hz": dominant_hz,
        "mean_rate_hz": mean_rate,
        "cycle_peak_to_trough_hz": peak_to_trough,
        "modulation_depth": float(peak_to_trough / mean_rate),
        "band_limited_rms_hz": float(component.std()),
        "cycle_profile_hz": profile.tolist(),
    }


def classify_tonic_fixed_point(rate_hz, *, dt_ms, onset_ms, settle_ms=300.0,
                               post_ms=1000.0, pre_ms=500.0,
                               active_fraction_20ms=None,
                               depth_floor=MODULATION_DEPTH_FLOOR):
    """Apply criterion 10 to one continuous trajectory.

    The pre-transition window is measured with the same estimator and reported
    beside the high state, so the comparison "is the high state more or less
    rhythmically organised than this substrate's own baseline?" is available
    without any extra assumption.
    """
    rate = np.asarray(rate_hz, float)
    time = np.arange(len(rate), dtype=float) * float(dt_ms)
    post_start = float(onset_ms) + float(settle_ms)
    post = (time >= post_start) & (time < post_start + float(post_ms))
    pre = (time >= float(onset_ms) - float(pre_ms)) & (time < float(onset_ms))
    if int(np.sum(post)) < 64:
        raise ValueError("post-transition window is too short to score")
    high = population_rate_modulation(rate[post], dt_ms=dt_ms)
    low = None
    if int(np.sum(pre)) >= 64 and float(rate[pre].mean()) > 0.0:
        low = population_rate_modulation(rate[pre], dt_ms=dt_ms)
    detail = {
        "high_state": high,
        "pre_transition_state": low,
        "depth_floor": float(depth_floor),
        "threshold_provenance": (
            "declared after the frozen-q locator scan exposed the tonic-ripple "
            "failure mode and before any dynamic trajectory was scored; "
            "lenient relative to this substrate's own 54% baseline regime"),
    }
    if active_fraction_20ms is not None:
        active = np.asarray(active_fraction_20ms, float)
        detail["median_active_E_fraction_20ms_post"] = float(np.median(active))
        detail["fraction_of_20ms_windows_with_every_E_active"] = float(
            np.mean(active >= 0.999))
    checks = {
        "population_rate_is_appreciably_modulated": bool(
            high["modulation_depth"] >= float(depth_floor)),
    }
    return {
        "status": ("OSCILLATORY_HIGH_STATE" if all(checks.values())
                   else "TONIC_HIGH_RATE_FIXED_POINT_WITH_RIPPLE"),
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "detail": detail,
        "boundary": (
            "this clause scores the population firing rate, not the contact "
            "LFP; a state can pass every LFP spectral clause and fail here"),
    }
