"""Readouts for sustained, spatially broad, high-amplitude oscillatory states."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt


def rolling_full_field_recruitment(
    spikes,
    positions,
    *,
    dt_ms,
    sheet_l_mm,
    window_ms=20.0,
    stride_ms=5.0,
    spatial_bin_mm=1.0,
    recruited_bin_fraction=0.5,
):
    """Measure how much of the E population and sheet is active per window."""
    spikes = np.asarray(spikes, bool)
    positions = np.asarray(positions, float)
    if spikes.ndim != 2 or positions.shape != (spikes.shape[1], 2):
        raise ValueError("spikes and E-neuron positions must align")
    window = max(1, int(round(float(window_ms) / float(dt_ms))))
    stride = max(1, int(round(float(stride_ms) / float(dt_ms))))
    if spikes.shape[0] < window:
        raise ValueError("trajectory is shorter than the recruitment window")
    n_bins = max(1, int(round(float(sheet_l_mm) / float(spatial_bin_mm))))
    ix = np.clip((positions[:, 0] / float(sheet_l_mm) * n_bins).astype(int),
                 0, n_bins - 1)
    iy = np.clip((positions[:, 1] / float(sheet_l_mm) * n_bins).astype(int),
                 0, n_bins - 1)
    flat = ix * n_bins + iy
    occupancy = np.bincount(flat, minlength=n_bins * n_bins).astype(float)
    occupied = occupancy > 0
    ends = np.arange(window, spikes.shape[0] + 1, stride, dtype=int)
    neuron_fraction = np.empty(len(ends), float)
    spatial_coverage = np.empty(len(ends), float)
    for row, end in enumerate(ends):
        active = np.any(spikes[end - window:end], axis=0)
        neuron_fraction[row] = float(np.mean(active))
        active_count = np.bincount(flat[active], minlength=n_bins * n_bins)
        local_fraction = np.divide(
            active_count, occupancy,
            out=np.zeros_like(occupancy), where=occupied)
        spatial_coverage[row] = float(np.mean(
            local_fraction[occupied] >= float(recruited_bin_fraction)))
    return {
        "time_ms": ends * float(dt_ms),
        "active_neuron_fraction": neuron_fraction,
        "recruited_spatial_fraction": spatial_coverage,
        "window_ms": float(window_ms),
        "stride_ms": float(stride_ms),
        "spatial_bin_mm": float(spatial_bin_mm),
        "recruited_bin_fraction": float(recruited_bin_fraction),
    }


def _longest_false_run(mask, dt_ms):
    mask = np.asarray(mask, bool)
    low = ~mask
    padded = np.r_[False, low, False]
    edges = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    return float(max((stop - start for start, stop in edges), default=0)
                 * float(dt_ms))


def contact_oscillation_metrics(
    lfp_trace,
    *,
    dt_ms,
    onset_ms,
    pre_ms=500.0,
    post_ms=500.0,
    band_hz=(20.0, 250.0),
    frequency_band_hz=(10.0, 250.0),
):
    """Compare sustained post-onset contact oscillation with its own pre-state."""
    trace = np.asarray(lfp_trace, float)
    if trace.ndim != 2:
        raise ValueError("lfp_trace must be time x contact")
    fs_hz = 1000.0 / float(dt_ms)
    if fs_hz <= 2.0 * float(band_hz[1]):
        raise ValueError("sampling rate is below the requested band Nyquist")
    time = np.arange(len(trace)) * float(dt_ms)
    pre = (time >= float(onset_ms) - float(pre_ms)) & (time < float(onset_ms))
    post = (time >= float(onset_ms)) & (time < float(onset_ms) + float(post_ms))
    expected_pre = int(round(float(pre_ms) / float(dt_ms)))
    expected_post = int(round(float(post_ms) / float(dt_ms)))
    if int(pre.sum()) < expected_pre or int(post.sum()) < expected_post:
        raise ValueError("pre/post morphology windows are incomplete")
    sos = butter(4, band_hz, btype="bandpass", fs=fs_hz, output="sos")
    filtered = sosfiltfilt(sos, trace, axis=0)
    envelope = np.abs(hilbert(filtered, axis=0))
    threshold = np.percentile(envelope[pre], 95.0, axis=0)
    high = envelope[post] > threshold[None, :]
    duty = np.mean(high, axis=0)
    pre_rms = np.sqrt(np.mean(np.square(filtered[pre]), axis=0))
    post_rms = np.sqrt(np.mean(np.square(filtered[post]), axis=0))
    ratio = post_rms / np.maximum(pre_rms, 1e-12)
    longest_low = np.asarray([
        _longest_false_run(high[:, contact], dt_ms)
        for contact in range(high.shape[1])
    ])
    peak_pre, peak_post, centroid_pre, centroid_post = [], [], [], []
    for contact in range(trace.shape[1]):
        frequency, power_pre = periodogram(
            trace[pre, contact], fs=fs_hz, window="hann", detrend="linear")
        _, power_post = periodogram(
            trace[post, contact], fs=fs_hz, window="hann", detrend="linear")
        selected = ((frequency >= float(frequency_band_hz[0]))
                    & (frequency <= float(frequency_band_hz[1])))
        if not np.any(selected):
            raise ValueError("frequency analysis band has no Fourier bins")
        f = frequency[selected]
        p_pre, p_post = power_pre[selected], power_post[selected]
        peak_pre.append(float(f[int(np.argmax(p_pre))]))
        peak_post.append(float(f[int(np.argmax(p_post))]))
        centroid_pre.append(float(np.dot(f, p_pre) / max(float(p_pre.sum()), 1e-20)))
        centroid_post.append(float(np.dot(f, p_post) / max(float(p_post.sum()), 1e-20)))
    peak_pre = np.asarray(peak_pre)
    peak_post = np.asarray(peak_post)
    centroid_pre = np.asarray(centroid_pre)
    centroid_post = np.asarray(centroid_post)
    return {
        "band_hz": [float(band_hz[0]), float(band_hz[1])],
        "pre_window_ms": [float(onset_ms) - float(pre_ms), float(onset_ms)],
        "post_window_ms": [float(onset_ms), float(onset_ms) + float(post_ms)],
        "median_post_high_envelope_duty": float(np.median(duty)),
        "contact_fraction_high_for_half_post_window": float(np.mean(duty >= 0.5)),
        "median_band_rms_ratio_post_over_pre": float(np.median(ratio)),
        "median_longest_low_envelope_gap_ms": float(np.median(longest_low)),
        "frequency_analysis_band_hz": [float(frequency_band_hz[0]),
                                        float(frequency_band_hz[1])],
        "frequency_resolution_hz": float(1000.0 / min(float(pre_ms), float(post_ms))),
        "median_peak_frequency_pre_hz": float(np.median(peak_pre)),
        "median_peak_frequency_post_hz": float(np.median(peak_post)),
        "median_peak_frequency_shift_hz": float(np.median(peak_post - peak_pre)),
        "median_spectral_centroid_pre_hz": float(np.median(centroid_pre)),
        "median_spectral_centroid_post_hz": float(np.median(centroid_post)),
        "median_spectral_centroid_shift_hz": float(
            np.median(centroid_post - centroid_pre)),
        "per_contact_post_high_envelope_duty": duty,
        "per_contact_band_rms_ratio": ratio,
        "per_contact_longest_low_envelope_gap_ms": longest_low,
        "per_contact_peak_frequency_pre_hz": peak_pre,
        "per_contact_peak_frequency_post_hz": peak_post,
        "per_contact_spectral_centroid_pre_hz": centroid_pre,
        "per_contact_spectral_centroid_post_hz": centroid_post,
    }


def population_rate_frequency_metrics(
    rate_hz,
    *,
    dt_ms,
    onset_ms,
    pre_ms=500.0,
    post_ms=500.0,
    frequency_band_hz=(10.0, 250.0),
):
    """Compare the population-rate spectrum before and after state onset."""
    rate = np.asarray(rate_hz, float)
    if rate.ndim != 1:
        raise ValueError("rate_hz must be one-dimensional")
    fs_hz = 1000.0 / float(dt_ms)
    time = np.arange(len(rate)) * float(dt_ms)
    pre = (time >= float(onset_ms) - float(pre_ms)) & (time < float(onset_ms))
    post = (time >= float(onset_ms)) & (time < float(onset_ms) + float(post_ms))
    expected_pre = int(round(float(pre_ms) / float(dt_ms)))
    expected_post = int(round(float(post_ms) / float(dt_ms)))
    if int(pre.sum()) < expected_pre or int(post.sum()) < expected_post:
        raise ValueError("pre/post population-rate windows are incomplete")

    def _spectrum(values):
        frequency, power = periodogram(
            values, fs=fs_hz, window="hann", detrend="linear")
        selected = ((frequency >= float(frequency_band_hz[0]))
                    & (frequency <= float(frequency_band_hz[1])))
        if not np.any(selected):
            raise ValueError("frequency analysis band has no Fourier bins")
        frequency, power = frequency[selected], power[selected]
        peak = float(frequency[int(np.argmax(power))])
        centroid = float(np.dot(frequency, power) / max(float(power.sum()), 1e-20))
        return peak, centroid

    peak_pre, centroid_pre = _spectrum(rate[pre])
    peak_post, centroid_post = _spectrum(rate[post])
    return {
        "frequency_analysis_band_hz": [float(frequency_band_hz[0]),
                                        float(frequency_band_hz[1])],
        "frequency_resolution_hz": float(1000.0 / min(float(pre_ms),
                                                        float(post_ms))),
        "peak_frequency_pre_hz": peak_pre,
        "peak_frequency_post_hz": peak_post,
        "peak_frequency_shift_hz": peak_post - peak_pre,
        "spectral_centroid_pre_hz": centroid_pre,
        "spectral_centroid_post_hz": centroid_post,
        "spectral_centroid_shift_hz": centroid_post - centroid_pre,
        "median_rate_pre_hz": float(np.median(rate[pre])),
        "median_rate_post_hz": float(np.median(rate[post])),
        "median_rate_ratio_post_over_pre": float(
            np.median(rate[post]) / max(float(np.median(rate[pre])), 1e-12)),
    }


def summarize_runaway_morphology(recruitment, oscillation, *, onset_ms,
                                 post_ms=500.0, population_frequency=None):
    time = np.asarray(recruitment["time_ms"], float)
    selected = (time >= float(onset_ms)) & (time < float(onset_ms) + float(post_ms))
    if not np.any(selected):
        raise ValueError("recruitment trace has no post-onset samples")
    neurons = np.asarray(recruitment["active_neuron_fraction"], float)[selected]
    space = np.asarray(recruitment["recruited_spatial_fraction"], float)[selected]
    summary = {
        "onset_ms": float(onset_ms),
        "post_window_ms": float(post_ms),
        "full_field_recruitment": {
            "median_active_neuron_fraction_20ms": float(np.median(neurons)),
            "q05_active_neuron_fraction_20ms": float(np.quantile(neurons, 0.05)),
            "fraction_windows_majority_E_active": float(np.mean(neurons >= 0.5)),
            "median_recruited_spatial_fraction_1mm": float(np.median(space)),
            "q05_recruited_spatial_fraction_1mm": float(np.quantile(space, 0.05)),
            "fraction_windows_majority_sheet_recruited": float(np.mean(space >= 0.5)),
        },
        "contact_oscillation": {
            key: value for key, value in oscillation.items()
            if not isinstance(value, np.ndarray)
        },
        "definition": (
            "sustained high-intensity oscillation requires both broad 20-ms E-neuron/"
            "spatial recruitment, persistent contact-envelope elevation, and a broadband "
            "frequency increase relative to the same trajectory's interictal state"
        ),
    }
    if population_frequency is not None:
        summary["population_rate_frequency"] = dict(population_frequency)
    return summary


def classify_sustained_runaway(summary):
    """Apply the fixed morphology contract used to accept a Fig. 5 runaway."""
    recruitment = summary["full_field_recruitment"]
    contact = summary["contact_oscillation"]
    population = summary["population_rate_frequency"]
    resolution = max(float(population["frequency_resolution_hz"]), 2.0)
    checks = {
        "majority_E_active_for_95pct_windows": (
            float(recruitment["q05_active_neuron_fraction_20ms"]) >= 0.5),
        "majority_sheet_recruited_for_95pct_windows": (
            float(recruitment["q05_recruited_spatial_fraction_1mm"]) >= 0.5),
        "contact_oscillation_sustained": (
            float(contact["median_post_high_envelope_duty"]) >= 0.8
            and float(contact["contact_fraction_high_for_half_post_window"]) >= 0.8),
        "contact_amplitude_increased": (
            float(contact["median_band_rms_ratio_post_over_pre"]) >= 2.0),
        "population_frequency_increased": (
            float(population["spectral_centroid_shift_hz"]) >= resolution),
        "population_rate_increased": (
            float(population["median_rate_ratio_post_over_pre"]) >= 2.0),
    }
    return {
        "status": ("SUSTAINED_HIGH_INTENSITY_OSCILLATION"
                   if all(checks.values()) else "NOT_SUSTAINED_ICTAL_MORPHOLOGY"),
        "all_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "thresholds": {
            "q05_active_neuron_fraction_20ms": 0.5,
            "q05_recruited_spatial_fraction_1mm": 0.5,
            "median_post_high_envelope_duty": 0.8,
            "contact_fraction_high_for_half_post_window": 0.8,
            "median_band_rms_ratio_post_over_pre": 2.0,
            "population_spectral_centroid_shift_hz": resolution,
            "population_rate_ratio_post_over_pre": 2.0,
        },
        "boundary": (
            "This is a model-state morphology contract, not a clinical seizure "
            "classifier and not evidence of patient ictal waveform reproduction."
        ),
    }
