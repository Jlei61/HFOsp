"""Nonlinear phase-map helpers for the dual-core spatial-Z reduction.

The existing continuation atlas describes fixed-point geometry.  It does not
establish a sustained oscillatory phase.  This module adds the missing pieces:

* a reproducible coarsening of the realized delay operators for discovery;
* a cycle-averaged population modulation estimator; and
* an explicit classifier that separates low, intermediate, tonic-recruited and
  oscillatory-recruited trajectories.

All rates are reported in Hz and time in ms.  The classifier is a model-state
assay, not a clinical seizure definition.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.signal import butter, hilbert, periodogram, sosfiltfilt

from src.topic4_dual_core_spatial_z_delay import CoarseDelayOperators


STATE_CODES = {
    "low": 0,
    "intermediate": 1,
    "tonic_recruited": 2,
    "oscillatory_recruited": 3,
}


def coarsen_delay_operators(
    operators: CoarseDelayOperators,
    *,
    factor: int,
) -> CoarseDelayOperators:
    """Aggregate native delay bins onto a larger integration step.

    Delay ``d * dt`` is assigned to the nearest strictly positive coarse bin.
    Every pathway weight is conserved exactly.  This approximation is only for
    discovery; selected phase boundaries must be checked again at native dt.
    """
    factor = int(factor)
    if factor < 1:
        raise ValueError("factor must be a positive integer")
    if factor == 1:
        return operators
    native_bins = int(operators.max_delay_steps)
    n = int(operators.w_ee_history.shape[0])
    assignments = np.maximum(
        1,
        np.floor(np.arange(1, native_bins + 1, dtype=float) / factor + 0.5)
        .astype(int),
    )
    coarse_bins = int(assignments.max())

    def aggregate(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        blocks = []
        for coarse in range(1, coarse_bins + 1):
            selected = np.flatnonzero(assignments == coarse)
            total = sparse.csr_matrix((n, n), dtype=float)
            for native in selected:
                total = total + matrix[:, native * n:(native + 1) * n]
            blocks.append(total)
        result = sparse.hstack(blocks, format="csr")
        before = np.asarray(matrix.sum(axis=1)).ravel()
        after = np.asarray(result.sum(axis=1)).ravel()
        if not np.allclose(before, after, rtol=1e-12, atol=1e-13):
            raise RuntimeError("delay coarsening did not conserve pathway weight")
        return result

    return CoarseDelayOperators(
        dt_ms=float(operators.dt_ms) * factor,
        max_delay_steps=coarse_bins,
        w_ee_history=aggregate(operators.w_ee_history),
        w_ei_history=aggregate(operators.w_ei_history),
        w_ie_history=aggregate(operators.w_ie_history),
        w_ii_history=aggregate(operators.w_ii_history),
    )


def population_cycle_modulation(
    rate_hz,
    *,
    dt_ms: float,
    band_hz=(20.0, 100.0),
    target_hz=(30.0, 80.0),
    n_phase_bins: int = 24,
) -> dict:
    """Measure the cycle-averaged peak-to-trough population-rate swing."""
    rate = np.asarray(rate_hz, float)
    if rate.ndim != 1 or rate.size < 64:
        raise ValueError("rate_hz must contain at least 64 one-dimensional samples")
    if float(dt_ms) <= 0.0:
        raise ValueError("dt_ms must be positive")
    mean_rate = float(np.mean(rate))
    if not mean_rate > 0.0:
        raise ValueError("rate segment has no activity")
    fs_hz = 1000.0 / float(dt_ms)
    frequency, power = periodogram(
        rate - mean_rate, fs=fs_hz, window="hann", detrend="linear")
    keep = ((frequency >= float(band_hz[0]))
            & (frequency <= float(band_hz[1])))
    if not np.any(keep):
        raise ValueError("analysis band has no Fourier bins")
    kept_frequency = frequency[keep]
    kept_power = power[keep]
    dominant = float(kept_frequency[int(np.argmax(kept_power))])
    low = max(1.0, dominant - 10.0)
    high = min(0.45 * fs_hz, dominant + 10.0)
    if not low < high:
        raise ValueError("dominant frequency leaves no filter band")
    component = sosfiltfilt(
        butter(4, (low, high), btype="bandpass", fs=fs_hz, output="sos"),
        rate - mean_rate,
    )
    phase = np.angle(hilbert(component))
    edges = np.linspace(-np.pi, np.pi, int(n_phase_bins) + 1)
    index = np.clip(np.digitize(phase, edges) - 1, 0, int(n_phase_bins) - 1)
    profile = np.asarray([
        np.mean(rate[index == phase_bin]) if np.any(index == phase_bin)
        else np.nan
        for phase_bin in range(int(n_phase_bins))
    ])
    # A nearly constant trace can occupy only one numerical phase bin.  Empty
    # bins carry no swing information and are therefore assigned the segment
    # mean rather than serialized as NaN.
    profile = np.where(np.isfinite(profile), profile, mean_rate)
    swing = float(np.nanmax(profile) - np.nanmin(profile))
    total_band_power = float(np.sum(kept_power))
    target = ((kept_frequency >= float(target_hz[0]))
              & (kept_frequency <= float(target_hz[1])))
    return {
        "dominant_hz": dominant,
        "mean_rate_hz": mean_rate,
        "cycle_peak_to_trough_hz": swing,
        "modulation_depth": swing / mean_rate,
        "band_limited_rms_hz": float(np.std(component)),
        "target_band_power_fraction": (
            float(np.sum(kept_power[target])) / max(total_band_power, 1e-20)),
        "cycle_profile_hz": profile,
    }


def contact_oscillation_assay(
    baseline_signals,
    late_signals,
    *,
    dt_ms: float,
    target_hz=(30.0, 80.0),
    broad_hz=(10.0, 100.0),
    minimum_band_rms_ratio: float = 1.0,
    n_persistence_windows: int = 4,
    minimum_passing_windows: int = 3,
    n_fine_persistence_windows: int = 10,
    minimum_fine_passing_windows: int = 7,
) -> dict:
    """Quantify whether a contact-wide high-frequency readout is present.

    Inputs have shape ``(time, contact)``.  A contact passes only when its
    dominant late peak over the broader diagnostic band lies in ``target_hz``
    *and* its target-band RMS exceeds the baseline RMS.  Persistence is then
    required at both 250-ms-like and 100-ms-like non-overlapping scales, so a
    few short bursts cannot fill a coarse window and masquerade as sustained
    activity.  PLV is descriptive:
    it is measured relative to the late target-band contact with the largest
    RMS and is not part of the pass gate.
    """
    baseline = np.asarray(baseline_signals, float)
    late = np.asarray(late_signals, float)
    if baseline.ndim != 2 or late.ndim != 2:
        raise ValueError("contact signals must have shape (time, contact)")
    if baseline.shape[1] != late.shape[1] or baseline.shape[1] < 1:
        raise ValueError("baseline and late signals must share contacts")
    if min(baseline.shape[0], late.shape[0]) < 64:
        raise ValueError("each contact window must contain at least 64 samples")
    if float(dt_ms) <= 0.0:
        raise ValueError("dt_ms must be positive")
    fs_hz = 1000.0 / float(dt_ms)
    if not (0.0 < float(target_hz[0]) < float(target_hz[1]) < 0.5 * fs_hz):
        raise ValueError("target_hz must lie below Nyquist")
    if not (0.0 < float(broad_hz[0]) < float(broad_hz[1]) < 0.5 * fs_hz):
        raise ValueError("broad_hz must lie below Nyquist")
    if float(minimum_band_rms_ratio) <= 0.0:
        raise ValueError("minimum_band_rms_ratio must be positive")
    if not 1 <= int(minimum_passing_windows) <= int(n_persistence_windows):
        raise ValueError("minimum_passing_windows must lie within the window count")
    if not 1 <= int(minimum_fine_passing_windows) <= int(n_fine_persistence_windows):
        raise ValueError(
            "minimum_fine_passing_windows must lie within the fine window count")

    sos = butter(4, target_hz, btype="bandpass", fs=fs_hz, output="sos")
    filtered_baseline = sosfiltfilt(sos, baseline, axis=0)
    filtered_late = sosfiltfilt(sos, late, axis=0)
    baseline_rms = np.sqrt(np.mean(filtered_baseline ** 2, axis=0))
    late_rms = np.sqrt(np.mean(filtered_late ** 2, axis=0))
    rms_ratio = late_rms / np.maximum(baseline_rms, 1e-12)

    frequency, power = periodogram(
        late, fs=fs_hz, window="hann", detrend="linear", axis=0)
    broad = ((frequency >= float(broad_hz[0]))
             & (frequency <= float(broad_hz[1])))
    if not np.any(broad):
        raise ValueError("broad diagnostic band has no Fourier bins")
    broad_frequency = frequency[broad]
    dominant = broad_frequency[np.argmax(power[broad], axis=0)]
    in_target = ((dominant >= float(target_hz[0]))
                 & (dominant <= float(target_hz[1])))
    passes = in_target & (rms_ratio >= float(minimum_band_rms_ratio))

    def window_assay(n_windows: int):
        window_dominant = []
        window_rms_ratio = []
        window_pass = []
        for window in np.array_split(np.arange(late.shape[0]), int(n_windows)):
            if window.size < 64:
                raise ValueError(
                    "persistence windows must contain at least 64 samples")
            segment = late[window]
            filtered_segment = sosfiltfilt(sos, segment, axis=0)
            segment_ratio = (
                np.sqrt(np.mean(filtered_segment ** 2, axis=0))
                / np.maximum(baseline_rms, 1e-12))
            segment_frequency, segment_power = periodogram(
                segment, fs=fs_hz, window="hann", detrend="linear", axis=0)
            segment_broad = (
                (segment_frequency >= float(broad_hz[0]))
                & (segment_frequency <= float(broad_hz[1])))
            segment_axis = segment_frequency[segment_broad]
            segment_dominant = segment_axis[
                np.argmax(segment_power[segment_broad], axis=0)]
            segment_pass = (
                (segment_dominant >= float(target_hz[0]))
                & (segment_dominant <= float(target_hz[1]))
                & (segment_ratio >= float(minimum_band_rms_ratio)))
            window_dominant.append(segment_dominant)
            window_rms_ratio.append(segment_ratio)
            window_pass.append(segment_pass)
        return (
            np.asarray(window_dominant).T,
            np.asarray(window_rms_ratio).T,
            np.asarray(window_pass).T,
        )

    window_dominant, window_rms_ratio, window_pass = window_assay(
        int(n_persistence_windows))
    fine_dominant, fine_rms_ratio, fine_pass = window_assay(
        int(n_fine_persistence_windows))
    passing_window_count = np.sum(window_pass, axis=1)
    fine_passing_window_count = np.sum(fine_pass, axis=1)
    coarse_persistent = passing_window_count >= int(minimum_passing_windows)
    fine_persistent = (
        fine_passing_window_count >= int(minimum_fine_passing_windows))
    persistent = coarse_persistent & fine_persistent

    phase = np.angle(hilbert(filtered_late, axis=0))
    reference = int(np.argmax(late_rms))
    plv = np.abs(np.mean(
        np.exp(1j * (phase - phase[:, reference, None])), axis=0))
    return {
        "n_contacts": int(late.shape[1]),
        "n_dominant_in_target_band": int(np.sum(in_target)),
        "n_band_rms_increased": int(np.sum(
            rms_ratio >= float(minimum_band_rms_ratio))),
        "n_contacts_passing_both": int(np.sum(passes)),
        "n_persistent_contacts": int(np.sum(persistent)),
        "dominant_frequency_hz": dominant.tolist(),
        "target_band_rms_ratio": rms_ratio.tolist(),
        "median_target_band_rms_ratio": float(np.median(rms_ratio)),
        "plv_to_max_rms_contact": plv.tolist(),
        "median_plv_to_max_rms_contact": float(np.median(plv)),
        "reference_contact_index": reference,
        "passing_contact_mask": passes.tolist(),
        "per_window_dominant_frequency_hz": window_dominant.tolist(),
        "per_window_target_band_rms_ratio": window_rms_ratio.tolist(),
        "passing_windows_per_contact": passing_window_count.tolist(),
        "coarse_persistent_contact_mask": coarse_persistent.tolist(),
        "fine_window_dominant_frequency_hz": fine_dominant.tolist(),
        "fine_window_target_band_rms_ratio": fine_rms_ratio.tolist(),
        "fine_passing_windows_per_contact": fine_passing_window_count.tolist(),
        "fine_persistent_contact_mask": fine_persistent.tolist(),
        "persistent_contact_mask": persistent.tolist(),
        "thresholds": {
            "target_frequency_hz": list(map(float, target_hz)),
            "broad_peak_search_hz": list(map(float, broad_hz)),
            "minimum_target_band_rms_ratio": float(minimum_band_rms_ratio),
            "n_persistence_windows": int(n_persistence_windows),
            "minimum_passing_windows": int(minimum_passing_windows),
            "n_fine_persistence_windows": int(n_fine_persistence_windows),
            "minimum_fine_passing_windows": int(minimum_fine_passing_windows),
        },
        "boundary": (
            "virtual-contact oscillation coverage with joint coarse- and fine-"
            "window persistence gates; this prevents sparse short bursts from "
            "being promoted to a sustained rhythm and remains distinct from "
            "population-rate modulation depth and regional recruitment"
        ),
    }


def classify_coarse_trajectory(
    rate_hz,
    *,
    dt_ms: float,
    regional_tail_rate_hz,
    tail_ms: float = 1000.0,
    window_ms: float = 250.0,
    minimum_high_rate_hz: float = 120.0,
    minimum_regional_rate_hz: float = 30.0,
    minimum_modulation_depth: float = 0.20,
) -> dict:
    """Classify a nonlinear delayed coarse trajectory by its late attractor."""
    rate = np.asarray(rate_hz, float)
    tail_steps = int(round(float(tail_ms) / float(dt_ms)))
    window_steps = int(round(float(window_ms) / float(dt_ms)))
    if tail_steps < 4 * 64 or rate.size < tail_steps:
        raise ValueError("trajectory does not contain the requested tail")
    if window_steps < 64 or tail_steps % window_steps:
        raise ValueError("tail must contain an integer number of valid windows")
    tail = rate[-tail_steps:]
    regional = np.asarray(regional_tail_rate_hz, float)
    if regional.ndim != 1 or regional.size < 3 or np.any(~np.isfinite(regional)):
        raise ValueError("regional_tail_rate_hz must contain finite core/surround rates")
    whole = population_cycle_modulation(tail, dt_ms=dt_ms)
    windows = [
        population_cycle_modulation(
            tail[start:start + window_steps], dt_ms=dt_ms)
        for start in range(0, tail_steps, window_steps)
    ]
    persistent_window = np.asarray([
        30.0 <= item["dominant_hz"] <= 80.0
        and item["modulation_depth"] >= 0.15
        for item in windows
    ], bool)
    globally_recruited = bool(
        whole["mean_rate_hz"] >= float(minimum_high_rate_hz)
        and float(np.min(regional)) >= float(minimum_regional_rate_hz)
    )
    deeply_oscillatory = bool(
        30.0 <= whole["dominant_hz"] <= 80.0
        and whole["modulation_depth"] >= float(minimum_modulation_depth)
        and int(np.sum(persistent_window)) >= max(1, len(windows) - 1)
    )
    if globally_recruited and deeply_oscillatory:
        label = "oscillatory_recruited"
    elif globally_recruited:
        label = "tonic_recruited"
    elif (whole["mean_rate_hz"] <= 60.0
          and float(np.max(regional)) < float(minimum_regional_rate_hz)):
        label = "low"
    else:
        label = "intermediate"
    return {
        "state": label,
        "state_code": STATE_CODES[label],
        "globally_recruited": globally_recruited,
        "deeply_oscillatory": deeply_oscillatory,
        "whole_tail": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in whole.items()
        },
        "window_dominant_hz": [item["dominant_hz"] for item in windows],
        "window_modulation_depth": [item["modulation_depth"] for item in windows],
        "passing_windows": int(np.sum(persistent_window)),
        "required_passing_windows": max(1, len(windows) - 1),
        "regional_tail_rate_hz": regional.tolist(),
        "thresholds": {
            "minimum_high_rate_hz": float(minimum_high_rate_hz),
            "minimum_regional_rate_hz": float(minimum_regional_rate_hz),
            "target_frequency_hz": [30.0, 80.0],
            "minimum_whole_tail_modulation_depth": float(
                minimum_modulation_depth),
            "minimum_window_modulation_depth": 0.15,
        },
        "boundary": (
            "nonlinear delayed 2-mm coarse-model state classification; "
            "not a full-SNN or clinical seizure label"
        ),
    }
