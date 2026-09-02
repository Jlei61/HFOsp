"""Acceptance readout for a sustained globally recruited oscillatory SNN state.

The earlier runaway contract deliberately rewarded 95% occupancy and therefore
favoured a flat tonic plateau.  This contract keeps a high recruitment *duty*
but permits troughs, and requires a narrow-band rhythm to recur across time
windows and across virtual contacts.  It is a model-state screen, not a seizure
classifier.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import periodogram


def detect_sustained_high_state_onset(
    rate_hz,
    *,
    dt_ms,
    threshold_hz=120.0,
    block_ms=20.0,
    forward_window_ms=300.0,
):
    """Locate a high-state transition without requiring tonic occupancy.

    The legacy early-stop detector requires every 20-ms EMA sample to remain
    above threshold for 100 ms and is therefore an operational runaway tool.
    Here the scientific onset is the first block whose *forward* 300-ms window
    has a median block rate above threshold.  Oscillatory troughs are allowed;
    isolated bursts shorter than the window are not.
    """
    rate = np.asarray(rate_hz, float)
    if rate.ndim != 1:
        raise ValueError("rate_hz must be one-dimensional")
    block_steps = int(round(float(block_ms) / float(dt_ms)))
    if block_steps < 1 or not np.isclose(
            block_steps * float(dt_ms), float(block_ms), atol=1e-9):
        raise ValueError("block_ms must lie on the rate time grid")
    window_blocks = int(round(float(forward_window_ms) / float(block_ms)))
    if window_blocks < 2 or not np.isclose(
            window_blocks * float(block_ms), float(forward_window_ms), atol=1e-9):
        raise ValueError("forward_window_ms must contain whole blocks")
    n_blocks = len(rate) // block_steps
    if n_blocks < window_blocks:
        return None
    blocks = np.median(
        rate[:n_blocks * block_steps].reshape(n_blocks, block_steps), axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(blocks, window_blocks)
    high = np.flatnonzero(np.median(windows, axis=1) >= float(threshold_hz))
    if high.size == 0:
        return None
    window_start = int(high[0])
    local_high = np.flatnonzero(
        blocks[window_start:window_start + window_blocks] >= float(threshold_hz))
    if local_high.size == 0:  # Defensive: impossible when the window median passes.
        return None
    return float((window_start + int(local_high[0])) * float(block_ms))


def _window_spectrum(values, fs_hz, band_hz):
    frequency, power = periodogram(
        np.asarray(values, float), fs=fs_hz, window="hann", detrend="linear")
    keep = (frequency >= band_hz[0]) & (frequency <= band_hz[1])
    if not np.any(keep):
        raise ValueError("oscillation band has no Fourier bins")
    f, p = frequency[keep], power[keep]
    # Integrate the periodogram density.  The pre window is 500 ms and each
    # post subwindow is 250 ms, so comparing raw sums would inject a factor of
    # two solely from their different Fourier-bin widths.
    density_sum = float(np.sum(p))
    df_hz = float(np.median(np.diff(f))) if len(f) > 1 else 1.0
    total = density_sum * df_hz
    index = int(np.argmax(p))
    return (float(f[index]),
            float(p[index] / max(density_sum, 1e-20)),
            total)


def contact_rhythm_metrics(
    lfp_trace,
    *,
    dt_ms,
    onset_ms,
    pre_ms=500.0,
    settle_ms=300.0,
    post_ms=1000.0,
    window_ms=250.0,
    band_hz=(20.0, 100.0),
    target_hz=(30.0, 80.0),
):
    """Require a recurring spectral peak, not merely broadband high amplitude."""
    trace = np.asarray(lfp_trace, float)
    if trace.ndim != 2:
        raise ValueError("lfp_trace must be time x contact")
    fs_hz = 1000.0 / float(dt_ms)
    time = np.arange(len(trace), dtype=float) * float(dt_ms)
    pre = (time >= float(onset_ms) - float(pre_ms)) & (time < float(onset_ms))
    if int(np.sum(pre)) < int(round(float(pre_ms) / float(dt_ms))) - 1:
        raise ValueError("pre-rhythm window is incomplete")
    post_start = float(onset_ms) + float(settle_ms)
    post_stop = post_start + float(post_ms)
    if time[-1] + float(dt_ms) < post_stop - 1e-9:
        raise ValueError("post-rhythm window is incomplete")
    n_windows = int(round(float(post_ms) / float(window_ms)))
    if n_windows < 2 or not np.isclose(
            n_windows * float(window_ms), float(post_ms), atol=1e-9):
        raise ValueError("post_ms must contain an integer number of rhythm windows")

    n_contacts = trace.shape[1]
    peak_hz = np.empty((n_windows, n_contacts), float)
    peak_fraction = np.empty_like(peak_hz)
    power_ratio = np.empty_like(peak_hz)
    pre_power = np.empty(n_contacts, float)
    for contact in range(n_contacts):
        _, _, pre_power[contact] = _window_spectrum(
            trace[pre, contact], fs_hz, band_hz)
    for window in range(n_windows):
        lo = post_start + window * float(window_ms)
        hi = lo + float(window_ms)
        selected = (time >= lo) & (time < hi)
        expected = int(round(float(window_ms) / float(dt_ms)))
        if int(np.sum(selected)) < expected - 1:
            raise ValueError("one post-rhythm window is incomplete")
        for contact in range(n_contacts):
            peak, fraction, power = _window_spectrum(
                trace[selected, contact], fs_hz, band_hz)
            peak_hz[window, contact] = peak
            peak_fraction[window, contact] = fraction
            power_ratio[window, contact] = power / max(pre_power[contact], 1e-20)

    rhythmic_window = (
        (peak_hz >= float(target_hz[0]))
        & (peak_hz <= float(target_hz[1]))
        & (peak_fraction >= 0.20)
        & (power_ratio >= 2.0)
    )
    required_windows = max(1, n_windows - 1)
    rhythmic_contact = np.sum(rhythmic_window, axis=0) >= required_windows
    contact_peak = np.median(peak_hz, axis=0)
    global_peak = float(np.median(contact_peak))
    global_mad = float(np.median(np.abs(contact_peak - global_peak)))
    return {
        "band_hz": [float(band_hz[0]), float(band_hz[1])],
        "target_hz": [float(target_hz[0]), float(target_hz[1])],
        "pre_window_ms": [float(onset_ms) - float(pre_ms), float(onset_ms)],
        "post_window_ms": [post_start, post_stop],
        "subwindow_ms": float(window_ms),
        "n_post_windows": int(n_windows),
        "required_passing_windows_per_contact": int(required_windows),
        "contact_fraction_consistently_rhythmic": float(np.mean(rhythmic_contact)),
        "median_contact_peak_hz": global_peak,
        "contact_peak_mad_hz": global_mad,
        "median_peak_power_fraction": float(np.median(peak_fraction)),
        "median_band_power_ratio_post_over_pre": float(np.median(power_ratio)),
        "per_window_contact_peak_hz": peak_hz,
        "per_window_contact_peak_power_fraction": peak_fraction,
        "per_window_contact_band_power_ratio": power_ratio,
        "per_contact_consistently_rhythmic": rhythmic_contact,
    }


def fixed_state_contact_rhythm_metrics(
    lfp_trace,
    reference_lfp_trace,
    *,
    dt_ms,
    start_ms,
    reference_start_ms=None,
    window_ms=250.0,
    n_windows=4,
    band_hz=(20.0, 100.0),
    target_hz=(30.0, 80.0),
):
    """Screen the frozen fast subsystem for a stationary rhythmic attractor.

    Candidate and q=1 reference traces are measured over the same late
    interval.  This is a fast-subsystem diagnostic, not a transition gate.
    """
    trace = np.asarray(lfp_trace, float)
    reference = np.asarray(reference_lfp_trace, float)
    if trace.ndim != 2 or reference.ndim != 2:
        raise ValueError("lfp traces must be time x contact")
    if trace.shape[1] != reference.shape[1]:
        raise ValueError("candidate and reference contacts must align")
    fs_hz = 1000.0 / float(dt_ms)
    time = np.arange(len(trace), dtype=float) * float(dt_ms)
    reference_time = np.arange(len(reference), dtype=float) * float(dt_ms)
    if reference_start_ms is None:
        reference_start_ms = float(start_ms)
    peak_hz = np.empty((int(n_windows), trace.shape[1]), float)
    peak_fraction = np.empty_like(peak_hz)
    power_ratio = np.empty_like(peak_hz)
    for window in range(int(n_windows)):
        lo = float(start_ms) + window * float(window_ms)
        hi = lo + float(window_ms)
        selected = (time >= lo) & (time < hi)
        reference_lo = float(reference_start_ms) + window * float(window_ms)
        reference_hi = reference_lo + float(window_ms)
        reference_selected = (
            (reference_time >= reference_lo) & (reference_time < reference_hi))
        expected = int(round(float(window_ms) / float(dt_ms)))
        if (int(np.sum(selected)) < expected - 1
                or int(np.sum(reference_selected)) < expected - 1):
            raise ValueError("fixed-state rhythm window is incomplete")
        for contact in range(trace.shape[1]):
            peak, fraction, power = _window_spectrum(
                trace[selected, contact], fs_hz, band_hz)
            _, _, reference_power = _window_spectrum(
                reference[reference_selected, contact], fs_hz, band_hz)
            peak_hz[window, contact] = peak
            peak_fraction[window, contact] = fraction
            power_ratio[window, contact] = (
                power / max(reference_power, 1e-20))
    rhythmic_window = (
        (peak_hz >= float(target_hz[0]))
        & (peak_hz <= float(target_hz[1]))
        & (peak_fraction >= 0.20)
        & (power_ratio >= 2.0)
    )
    required_windows = max(1, int(n_windows) - 1)
    rhythmic_contact = np.sum(rhythmic_window, axis=0) >= required_windows
    contact_peak = np.median(peak_hz, axis=0)
    global_peak = float(np.median(contact_peak))
    return {
        "analysis_window_ms": [
            float(start_ms),
            float(start_ms) + int(n_windows) * float(window_ms),
        ],
        "reference_analysis_window_ms": [
            float(reference_start_ms),
            float(reference_start_ms) + int(n_windows) * float(window_ms),
        ],
        "subwindow_ms": float(window_ms),
        "n_windows": int(n_windows),
        "required_passing_windows_per_contact": int(required_windows),
        "contact_fraction_consistently_rhythmic": float(
            np.mean(rhythmic_contact)),
        "median_contact_peak_hz": global_peak,
        "contact_peak_mad_hz": float(
            np.median(np.abs(contact_peak - global_peak))),
        "median_peak_power_fraction": float(np.median(peak_fraction)),
        "median_band_power_ratio_over_q1_reference": float(
            np.median(power_ratio)),
        "per_window_contact_peak_hz": peak_hz,
        "per_window_contact_peak_power_fraction": peak_fraction,
        "per_window_contact_band_power_ratio_over_q1": power_ratio,
        "per_contact_consistently_rhythmic": rhythmic_contact,
    }


def state_rate_metrics(rate_hz, *, dt_ms, onset_ms, pre_ms=500.0,
                       settle_ms=300.0, post_ms=1000.0):
    rate = np.asarray(rate_hz, float)
    window = max(1, int(round(20.0 / float(dt_ms))))
    smooth = uniform_filter1d(rate, size=window, mode="nearest")
    time = np.arange(len(rate), dtype=float) * float(dt_ms)
    pre = (time >= float(onset_ms) - float(pre_ms)) & (time < float(onset_ms))
    post = ((time >= float(onset_ms) + float(settle_ms))
            & (time < float(onset_ms) + float(settle_ms) + float(post_ms)))
    if not np.any(pre) or not np.any(post):
        raise ValueError("state-rate windows are incomplete")
    return {
        "median_pre_hz": float(np.median(smooth[pre])),
        "q95_pre_hz": float(np.quantile(smooth[pre], 0.95)),
        "median_post_hz": float(np.median(smooth[post])),
        "q05_post_hz": float(np.quantile(smooth[post], 0.05)),
        "median_ratio_post_over_pre": float(
            np.median(smooth[post]) / max(float(np.median(smooth[pre])), 1e-12)),
    }


def recruitment_duty_metrics(recruitment, *, onset_ms, settle_ms=300.0,
                             post_ms=1000.0):
    time = np.asarray(recruitment["time_ms"], float)
    selected = ((time >= float(onset_ms) + float(settle_ms))
                & (time < float(onset_ms) + float(settle_ms) + float(post_ms)))
    if not np.any(selected):
        raise ValueError("recruitment window is incomplete")
    neurons = np.asarray(recruitment["active_neuron_fraction"], float)[selected]
    sheet = np.asarray(recruitment["recruited_spatial_fraction"], float)[selected]
    joint = (neurons >= 0.5) & (sheet >= 0.5)
    return {
        "median_active_neuron_fraction_20ms": float(np.median(neurons)),
        "median_recruited_spatial_fraction_1mm": float(np.median(sheet)),
        "fraction_windows_majority_E_active": float(np.mean(neurons >= 0.5)),
        "fraction_windows_majority_sheet_recruited": float(np.mean(sheet >= 0.5)),
        "joint_global_recruitment_duty": float(np.mean(joint)),
    }


def classify_global_recruited_oscillation(*, onset_ms, rates, recruitment,
                                          rhythm):
    """Fixed development gate designed to reject tonic and local-only states."""
    checks = {
        "interictal_dwell_at_least_2000ms": float(onset_ms) >= 2000.0,
        "pre_state_not_already_high": (
            float(rates["median_pre_hz"]) <= 60.0
            and float(rates["q95_pre_hz"]) < 120.0),
        "post_state_high": (
            float(rates["median_post_hz"]) >= 120.0
            and float(rates["median_ratio_post_over_pre"]) >= 2.0),
        "global_recruitment_duty": (
            float(recruitment["joint_global_recruitment_duty"]) >= 0.75),
        "rhythm_is_global": (
            float(rhythm["contact_fraction_consistently_rhythmic"]) >= 0.80),
        "rhythm_is_frequency_locked": (
            30.0 <= float(rhythm["median_contact_peak_hz"]) <= 80.0
            and float(rhythm["contact_peak_mad_hz"]) <= 8.0),
        "rhythm_is_not_broadband_tonic": (
            float(rhythm["median_peak_power_fraction"]) >= 0.20
            and float(rhythm["median_band_power_ratio_post_over_pre"]) >= 2.0),
    }
    return {
        "status": ("SUSTAINED_GLOBAL_RECRUITED_OSCILLATION"
                   if all(checks.values())
                   else "NOT_SUSTAINED_GLOBAL_RECRUITED_OSCILLATION"),
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "boundary": (
            "model-state morphology screen only; not a clinical seizure "
            "classifier, patient waveform fit, or mechanism identification"),
    }


TONIC_GLOBAL_RUNAWAY_THRESHOLDS = {
    "minimum_pre_transition_dwell_ms": 300.0,
    "maximum_pre_median_rate_hz": 80.0,
    "maximum_pre_q95_rate_hz": 120.0,
    "minimum_post_median_rate_hz": 300.0,
    "minimum_post_over_pre_rate_ratio": 4.0,
    "minimum_median_active_E_fraction_20ms": 0.85,
    "minimum_median_recruited_sheet_fraction_1mm": 0.85,
    "minimum_joint_global_recruitment_duty": 0.80,
    "minimum_observed_post_transition_ms": 1500.0,
}


def classify_global_tonic_runaway(
    *,
    onset_ms,
    observed_post_transition_ms,
    rates,
    recruitment,
    thresholds=None,
):
    """Accept the near-saturated tonic runaway requested for Fig. 5A.

    This is deliberately a different endpoint from
    :func:`classify_global_recruited_oscillation`.  It rewards a persistent,
    globally recruited high-rate plateau and therefore does *not* inspect a
    contact spectrum, a target frequency, or population-rate modulation depth.
    The two classifiers are kept side by side so a tonic-positive result cannot
    silently overwrite the later oscillation-negative audit.

    The default thresholds describe the requested B0 morphology: a readable
    300-ms low state, at least 1.5 s of recorded high state, a
    refractory-ceiling-adjacent population rate, and near-complete recruitment
    of both neurons and the spatial sheet.  Raw values are returned beside every
    threshold so the gate is auditable rather than encoded only in a label.
    """
    locked = dict(TONIC_GLOBAL_RUNAWAY_THRESHOLDS)
    if thresholds is not None:
        unknown = sorted(set(thresholds).difference(locked))
        if unknown:
            raise ValueError(f"unknown tonic-runaway thresholds: {unknown}")
        locked.update({key: float(value) for key, value in thresholds.items()})

    observed = {
        "pre_transition_dwell_ms": float(onset_ms),
        "pre_median_rate_hz": float(rates["median_pre_hz"]),
        "pre_q95_rate_hz": float(rates["q95_pre_hz"]),
        "post_median_rate_hz": float(rates["median_post_hz"]),
        "post_q05_rate_hz": float(rates["q05_post_hz"]),
        "post_over_pre_rate_ratio": float(rates["median_ratio_post_over_pre"]),
        "median_active_E_fraction_20ms": float(
            recruitment["median_active_neuron_fraction_20ms"]),
        "median_recruited_sheet_fraction_1mm": float(
            recruitment["median_recruited_spatial_fraction_1mm"]),
        "joint_global_recruitment_duty": float(
            recruitment["joint_global_recruitment_duty"]),
        "observed_post_transition_ms": float(observed_post_transition_ms),
    }
    checks = {
        "readable_low_state_dwell": (
            observed["pre_transition_dwell_ms"]
            >= locked["minimum_pre_transition_dwell_ms"]),
        "pre_state_below_runaway_threshold": (
            observed["pre_median_rate_hz"]
            <= locked["maximum_pre_median_rate_hz"]
            and observed["pre_q95_rate_hz"]
            < locked["maximum_pre_q95_rate_hz"]),
        "post_rate_is_near_saturated_plateau": (
            observed["post_median_rate_hz"]
            >= locked["minimum_post_median_rate_hz"]
            and observed["post_over_pre_rate_ratio"]
            >= locked["minimum_post_over_pre_rate_ratio"]),
        "near_complete_E_recruitment": (
            observed["median_active_E_fraction_20ms"]
            >= locked["minimum_median_active_E_fraction_20ms"]),
        "near_complete_sheet_recruitment": (
            observed["median_recruited_sheet_fraction_1mm"]
            >= locked["minimum_median_recruited_sheet_fraction_1mm"]),
        "global_plateau_is_sustained": (
            observed["joint_global_recruitment_duty"]
            >= locked["minimum_joint_global_recruitment_duty"]
            and observed["observed_post_transition_ms"]
            >= locked["minimum_observed_post_transition_ms"]),
    }
    passed = bool(all(checks.values()))
    return {
        "status": ("TONIC_GLOBAL_RUNAWAY" if passed
                   else "NOT_TONIC_GLOBAL_RUNAWAY"),
        "all_checks_pass": passed,
        "checks": checks,
        "observed": observed,
        "thresholds": locked,
        "explicitly_not_required": [
            "30-80 Hz contact peak",
            "deep population-rate modulation",
            "periodic silencing and reactivation",
        ],
        "boundary": (
            "model-state morphology screen for a near-saturated tonic runaway; "
            "not a clinical seizure classifier, waveform fit, or patient-mechanism "
            "identification"),
    }
