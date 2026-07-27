"""Fine source-space rhythm audit for the Z/M carrier branch.

Twenty-five millisecond survival bins cannot resolve a 30--80 Hz E/I rhythm.
This module bins exact E/I spike rasters on a 1--2 ms grid and distinguishes
three descriptive candidates:

* globally aligned periodic activity;
* locally periodic but phase-staggered activity whose population mean cancels;
* asynchronous or irregular activity.

These labels route the later operator analysis.  They do not by themselves
establish an ictal carrier or pass the observation-space gate.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as ss


SOURCE_RHYTHM_VERSION = "zm_source_rhythm_v1_2026-07-27"
_CARRIER_TYPE_BY_CLASS = {
    "stationary_rate_candidate": "fixed",
    "global_periodic_candidate": "periodic",
    "phase_staggered_periodic_candidate": "periodic",
    "asynchronous_or_irregular_candidate": "stochastic",
}


def source_rhythm_authorized(verdict):
    """Fail closed unless the two-seed native carrier confirmation passed."""
    confirmation = (verdict or {}).get("confirmation") or {}
    layers = (verdict or {}).get("layers") or {}
    return bool(
        (verdict or {}).get("verdict") == "carrier_at_visited_states"
        and confirmation.get("status") == "passed"
        and layers.get("source_space_carrier") == "carrier_window"
    )


def adjudicate_source_rhythm(rows, min_seeds=2):
    """Require a replicated source class before routing the operator audit."""

    by_seed = {}
    conflicts = []
    for row in rows:
        seed = int(row["seed"])
        klass = row.get("source_temporal_class")
        if seed in by_seed and by_seed[seed] != klass:
            conflicts.append(seed)
        by_seed[seed] = klass
    if conflicts:
        return {
            "status": "within_seed_conflict",
            "carrier_type": None,
            "conflicting_seeds": sorted(set(conflicts)),
            "source_rhythm_version": SOURCE_RHYTHM_VERSION,
        }
    if len(by_seed) < int(min_seeds):
        return {
            "status": "insufficient_seeds",
            "carrier_type": None,
            "n_seeds": len(by_seed),
            "min_seeds": int(min_seeds),
            "source_rhythm_version": SOURCE_RHYTHM_VERSION,
        }
    unresolved = {
        seed: klass for seed, klass in by_seed.items()
        if klass not in _CARRIER_TYPE_BY_CLASS
    }
    if unresolved:
        return {
            "status": "unresolved_source_class",
            "carrier_type": None,
            "unresolved": unresolved,
            "source_rhythm_version": SOURCE_RHYTHM_VERSION,
        }
    routed = {
        seed: _CARRIER_TYPE_BY_CLASS[klass] for seed, klass in by_seed.items()
    }
    types = sorted(set(routed.values()))
    if len(types) != 1:
        return {
            "status": "class_disagreement",
            "carrier_type": None,
            "seed_classes": by_seed,
            "seed_carrier_types": routed,
            "source_rhythm_version": SOURCE_RHYTHM_VERSION,
        }
    return {
        "status": "replicated",
        "carrier_type": types[0],
        "seed_classes": by_seed,
        "seed_carrier_types": routed,
        "source_rhythm_version": SOURCE_RHYTHM_VERSION,
        "claim_boundary": (
            "operator-tool routing only; not an ictal or lifecycle verdict"
        ),
    }


def _cell_index(pos, L, n_grid):
    p = np.asarray(pos, float)
    ix = np.clip((p[:, 0] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((p[:, 1] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    return iy * n_grid + ix


def _one_population_grid(spikes, pos, L, dt_ms, bin_ms, n_grid):
    x = np.asarray(spikes, bool)
    if x.ndim != 2 or x.shape[1] != len(pos):
        raise ValueError("spike raster must be time x neuron and align with positions")
    steps_per_bin = max(1, int(round(float(bin_ms) / float(dt_ms))))
    n_bins = x.shape[0] // steps_per_bin
    x = x[:n_bins * steps_per_bin]
    counts_neuron = x.reshape(n_bins, steps_per_bin, x.shape[1]).sum(axis=1)
    cell = _cell_index(pos, L, n_grid)
    n_cells = n_grid * n_grid
    neurons_per_cell = np.bincount(cell, minlength=n_cells).astype(float)
    counts_grid = np.zeros((n_bins, n_cells), float)
    for b in range(n_bins):
        counts_grid[b] = np.bincount(
            cell, weights=counts_neuron[b], minlength=n_cells
        )
    duration_s = steps_per_bin * float(dt_ms) * 1e-3
    denom = neurons_per_cell * duration_s
    rates = np.divide(
        counts_grid, denom[None, :],
        out=np.zeros_like(counts_grid), where=denom[None, :] > 0,
    )
    global_rate = counts_neuron.sum(axis=1) / max(1, x.shape[1]) / duration_s
    return rates.reshape(n_bins, n_grid, n_grid), global_rate, neurons_per_cell


def bin_spikes_to_grid(
    E_spk_bool,
    I_spk_bool,
    posE,
    posI,
    *,
    L,
    dt_ms,
    bin_ms=2.0,
    n_grid=16,
):
    """Coarse-grain exact E/I spike rasters without changing spike totals."""
    E_grid, E_global, nE = _one_population_grid(
        E_spk_bool, posE, L, dt_ms, bin_ms, n_grid
    )
    I_grid, I_global, nI = _one_population_grid(
        I_spk_bool, posI, L, dt_ms, bin_ms, n_grid
    )
    if E_grid.shape[0] != I_grid.shape[0]:
        raise ValueError("E and I rasters produce different time-bin counts")
    return {
        "source_rhythm_version": SOURCE_RHYTHM_VERSION,
        "bin_ms": float(bin_ms),
        "E_rate_grid": E_grid.astype(np.float32),
        "I_rate_grid": I_grid.astype(np.float32),
        "global_E_rate_hz": E_global.astype(np.float32),
        "global_I_rate_hz": I_global.astype(np.float32),
        "nE_per_cell": nE.reshape(n_grid, n_grid).astype(np.int32),
        "nI_per_cell": nI.reshape(n_grid, n_grid).astype(np.int32),
    }


def _spectrum_features(x, fs):
    x = ss.detrend(np.asarray(x, float), axis=0)
    nperseg = min(x.shape[0], max(64, int(round(fs))))
    f, power = ss.welch(x, fs=fs, nperseg=nperseg, axis=0)
    band = (f >= 5.0) & (f <= min(150.0, 0.45 * fs))
    if band.sum() < 8:
        raise ValueError("time series is too short for the source rhythm band")
    fb = f[band]
    pb = power[band]
    peak_idx = np.argmax(pb, axis=0)
    dom = fb[peak_idx]
    peak_fraction = np.zeros(pb.shape[1], float)
    for j, f0 in enumerate(dom):
        peak = np.abs(fb - f0) <= 2.0
        denom = float(pb[:, j].sum())
        peak_fraction[j] = float(pb[peak, j].sum() / denom) if denom > 0 else 0.0
    return dom, peak_fraction


def _pairwise_phase_locking(x, fs, f0):
    lo = max(3.0, f0 - 4.0)
    hi = min(0.45 * fs, f0 + 4.0)
    if hi <= lo:
        return float("nan")
    sos = ss.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
    filt = ss.sosfiltfilt(sos, np.asarray(x, float), axis=0)
    phase = np.angle(ss.hilbert(filt, axis=0))
    n = phase.shape[1]
    vals = []
    for a in range(n):
        for b in range(a + 1, n):
            vals.append(abs(np.mean(np.exp(1j * (phase[:, a] - phase[:, b])))))
    return float(np.mean(vals)) if vals else 1.0


def _peak_fraction_1d(x, fs, f0):
    x = ss.detrend(np.asarray(x, float))
    nperseg = min(x.size, max(64, int(round(fs))))
    f, power = ss.welch(x, fs=fs, nperseg=nperseg)
    band = (f >= 5.0) & (f <= min(150.0, 0.45 * fs))
    peak = band & (np.abs(f - f0) <= 2.0)
    denom = float(power[band].sum())
    return float(power[peak].sum() / denom) if denom > 0 else 0.0


def characterize_source_rhythm(
    E_rate_grid,
    I_rate_grid,
    *,
    bin_ms,
    active_floor_hz=5.0,
):
    """Describe fine E/I source dynamics; no lifecycle or ictal verdict is made."""
    E = np.asarray(E_rate_grid, float)
    I = np.asarray(I_rate_grid, float)
    if E.ndim != 3 or I.shape != E.shape:
        raise ValueError("E/I grids must share shape time x y x x")
    if E.shape[0] < 128:
        raise ValueError("source rhythm trace is too short")
    fs = 1000.0 / float(bin_ms)
    flat = E.reshape(E.shape[0], -1)
    active = np.mean(flat, axis=0) >= float(active_floor_hz)
    if active.sum() < 2:
        return {
            "source_rhythm_version": SOURCE_RHYTHM_VERSION,
            "source_temporal_class": "insufficient_active_cells",
            "n_active_cells": int(active.sum()),
        }
    local = flat[:, active]
    dom, peak_fraction = _spectrum_features(local, fs)
    f0 = float(np.median(dom))
    tol = max(2.0, 0.05 * f0)
    agreeing = np.abs(dom - f0) <= tol
    agreement = float(np.mean(agreeing))
    local_selected = local[:, agreeing]
    if local_selected.shape[1] > 32:
        pick = np.linspace(0, local_selected.shape[1] - 1, 32).astype(int)
        local_selected = local_selected[:, pick]
    phase_lock = (
        _pairwise_phase_locking(local_selected, fs, f0)
        if local_selected.shape[1] >= 2 else float("nan")
    )

    global_E = np.mean(flat[:, active], axis=1)
    mean_global = float(np.mean(global_E))
    modulation = (
        float((np.percentile(global_E, 95) - np.percentile(global_E, 5)) / mean_global)
        if mean_global > 1e-12 else float("nan")
    )
    global_peak_fraction = _peak_fraction_1d(global_E, fs, f0)
    local_peak_median = float(np.median(peak_fraction))
    periodic_local = (
        agreement >= 0.50
        and local_peak_median >= 0.25
        and np.isfinite(phase_lock)
        and phase_lock >= 0.50
    )
    global_periodic = (
        periodic_local and modulation >= 0.10 and global_peak_fraction >= 0.20
    )
    if global_periodic:
        klass = "global_periodic_candidate"
    elif periodic_local:
        klass = "phase_staggered_periodic_candidate"
    else:
        klass = "asynchronous_or_irregular_candidate"

    # E/I phase lag is descriptive and computed only on the population means.
    global_I = np.mean(I.reshape(I.shape[0], -1), axis=1)
    lo, hi = max(3.0, f0 - 4.0), min(0.45 * fs, f0 + 4.0)
    ei_phase_lag = float("nan")
    if hi > lo:
        sos = ss.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        e_phase = np.angle(ss.hilbert(ss.sosfiltfilt(sos, global_E)))
        i_phase = np.angle(ss.hilbert(ss.sosfiltfilt(sos, global_I)))
        ei_phase_lag = float(np.angle(np.mean(np.exp(1j * (i_phase - e_phase)))))

    return {
        "source_rhythm_version": SOURCE_RHYTHM_VERSION,
        "source_temporal_class": klass,
        "n_active_cells": int(active.sum()),
        "active_cell_fraction": float(np.mean(active)),
        "dominant_frequency_median_hz": f0,
        "local_frequency_agreement": agreement,
        "local_peak_fraction_median": local_peak_median,
        "local_phase_locking": phase_lock,
        "global_modulation_fraction": modulation,
        "global_peak_fraction": global_peak_fraction,
        "E_I_phase_lag_rad": ei_phase_lag,
        "claim_boundary": (
            "descriptive fine source-space routing audit; periodic candidate "
            "still requires state-return/Floquet validation"
        ),
    }
