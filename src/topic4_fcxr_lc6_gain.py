"""Pure paired-response readouts for the FCXR-LC6A exact-state gain forks.

The functions here classify neither a carrier nor a lifecycle.  They only
compare a weak-patch arm with its exact-input sham and report how large and how
long the macroscopic response is.
"""
from __future__ import annotations

import numpy as np


def binned_global_rate(spike_steps, *, n_steps: int, n_cells: int, dt_ms: float,
                       bin_ms: float) -> np.ndarray:
    """Population E-cell rate in fixed, complete bins."""

    width = int(round(float(bin_ms) / float(dt_ms)))
    if width <= 0 or int(n_steps) % width:
        raise ValueError("gain trajectory must contain complete rate bins")
    bins = np.asarray(spike_steps, np.int64) // width
    counts = np.bincount(bins, minlength=int(n_steps) // width)
    return counts.astype(float) / int(n_cells) / (float(bin_ms) / 1000.0)


def active_area_mm2(rate_maps, occupancy, *, rate_threshold_hz: float,
                    sheet_size_mm: float) -> np.ndarray:
    """Total occupied coarse-bin area above the prelocked local-rate threshold."""

    maps = np.asarray(rate_maps, float)
    occupancy = np.asarray(occupancy)
    if maps.ndim != 2 or occupancy.shape != (maps.shape[1],):
        raise ValueError("rate maps and occupancy are not aligned")
    side = int(round(np.sqrt(maps.shape[1])))
    if side * side != maps.shape[1]:
        raise ValueError("gain area map must be square")
    bin_area = (float(sheet_size_mm) / side) ** 2
    return np.sum((maps >= float(rate_threshold_hz)) & (occupancy[None, :] > 0), axis=1) * bin_area


def relaxation_readout(delta_rate_10ms, delta_area_100ms, *, pulse_ms: float,
                       rate_bin_ms: float = 10.0, area_bin_ms: float = 100.0,
                       fraction: float = .1, hold_ms: float = 200.0) -> dict:
    """Find sustained return of both response envelopes below a fraction of peak."""

    rate = np.abs(np.asarray(delta_rate_10ms, float))
    area = np.abs(np.asarray(delta_area_100ms, float))
    if rate.ndim != 1 or area.ndim != 1 or rate.size == 0 or area.size == 0:
        raise ValueError("gain relaxation series must be non-empty 1-D arrays")
    if not (0 < float(fraction) < 1 and hold_ms > 0):
        raise ValueError("invalid relaxation contract")
    # Compare both readouts on the coarser registered 100-ms grid.
    ratio = int(round(float(area_bin_ms) / float(rate_bin_ms)))
    if ratio <= 0 or not np.isclose(ratio * float(rate_bin_ms), float(area_bin_ms)):
        raise ValueError("area_bin_ms must be an integer multiple of rate_bin_ms")
    usable = min(area.size, rate.size // ratio)
    coarse_rate = np.max(rate[:usable * ratio].reshape(usable, ratio), axis=1)
    area = area[:usable]
    peak_rate = float(np.max(coarse_rate, initial=0.0))
    peak_area = float(np.max(area, initial=0.0))
    rate_norm = coarse_rate / peak_rate if peak_rate > 0 else np.zeros_like(coarse_rate)
    area_norm = area / peak_area if peak_area > 0 else np.zeros_like(area)
    envelope = np.maximum(rate_norm, area_norm)
    first = int(np.ceil(float(pulse_ms) / float(area_bin_ms)))
    hold = max(1, int(np.ceil(float(hold_ms) / float(area_bin_ms))))
    relaxation_ms = None
    for index in range(first, max(first, envelope.size - hold + 1)):
        if np.all(envelope[index:index + hold] <= float(fraction)):
            relaxation_ms = float(index * float(area_bin_ms) - float(pulse_ms))
            break
    return {
        "peak_abs_delta_rate_hz": peak_rate,
        "peak_abs_delta_area_mm2": peak_area,
        "relaxation_ms_after_pulse": relaxation_ms,
        "right_censored": relaxation_ms is None,
        "terminal_normalized_envelope": float(envelope[-1]),
    }


def paired_gain_readout(rate_sham, rate_probe, area_sham, area_probe, *,
                        pulse_l2_current: float, pulse_ms: float,
                        susceptibility_window_ms: float,
                        rate_bin_ms: float, area_bin_ms: float,
                        relaxation_fraction: float, relaxation_hold_ms: float) -> dict:
    """Return dose-normalized susceptibility and independent relaxation fields."""

    sham = np.asarray(rate_sham, float)
    probe = np.asarray(rate_probe, float)
    area_sham = np.asarray(area_sham, float)
    area_probe = np.asarray(area_probe, float)
    if sham.shape != probe.shape or area_sham.shape != area_probe.shape:
        raise ValueError("paired gain series are not aligned")
    delta_rate = probe - sham
    delta_area = area_probe - area_sham
    n = min(delta_rate.size, int(round(float(susceptibility_window_ms) / float(rate_bin_ms))))
    input_charge = float(pulse_l2_current) * float(pulse_ms) / 1000.0
    response_area = float(np.sum(np.abs(delta_rate[:n])) * float(rate_bin_ms) / 1000.0)
    susceptibility = response_area / input_charge if input_charge > 0 else None
    relaxation = relaxation_readout(
        delta_rate, delta_area, pulse_ms=pulse_ms, rate_bin_ms=rate_bin_ms,
        area_bin_ms=area_bin_ms, fraction=relaxation_fraction,
        hold_ms=relaxation_hold_ms,
    )
    return {
        "susceptibility_hz_s_per_l2_current_s": susceptibility,
        "global_rate_l1_response_hz_s": response_area,
        "global_rate_rms_deviation_hz": float(np.sqrt(np.mean(delta_rate * delta_rate))),
        "active_area_l1_deviation_mm2_s": float(
            np.sum(np.abs(delta_area)) * float(area_bin_ms) / 1000.0
        ),
        "relaxation": relaxation,
        "delta_rate_hz": delta_rate,
        "delta_area_mm2": delta_area,
    }
