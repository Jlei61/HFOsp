"""Fail-closed run-level phenotype labels for the Z/M Phase-C neighbourhood.

This module deliberately answers a narrower question than the lifecycle gate:
given an already simulated, frozen-slow-state run, did it contain a bounded
tonic state, a non-tonic oscillation, a clonic/bursting macro-rhythm, or merely
an HFO-like train of separated events?  A spatial-relay label is an independent
modifier and cannot rescue an invalid temporal phenotype.

All thresholds below are run-level descriptive thresholds.  They do not turn a
frozen-state phenotype into evidence for entry, exit, or recovery.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as ss
from scipy import stats


PHASEC_PHENOTYPE_VERSION = "zm_phasec_run_phenotype_v1_2026-07-28"


DEFAULTS = {
    "active_floor_hz": 5.0,
    "active_relative_to_p95": 0.15,
    "rest_relative_to_p95": 0.10,
    "rest_dwell_ms": 100.0,
    "minimum_active_occupancy": 0.80,
    "runaway_rate_hz": 250.0,
    "saturation_fraction": 0.50,
    "refractory_fraction": 0.80,
    "two_zone_occupancy": 0.80,
    "minimum_cycles": 10,
    "minimum_bursts": 6,
    "minimum_modulation": 0.20,
    "maximum_period_cv": 0.20,
    "maximum_periodic_rest_reset_fraction": 0.20,
    "maximum_ibi_cv": 0.50,
    "clonic_min_period_ms": 150.0,
    "clonic_max_period_ms": 2000.0,
    "minimum_hfo_train_events": 4,
    "maximum_abs_relative_drift": 0.50,
    "maximum_variance_ratio": 4.0,
}


def _longest_true_run(mask):
    mask = np.asarray(mask, bool).ravel()
    if not mask.size or not np.any(mask):
        return 0
    padded = np.r_[False, mask, False].astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return int(np.max(edges[1::2] - edges[::2]))


def _episode_count(mask):
    mask = np.asarray(mask, bool).ravel()
    if not mask.size:
        return 0
    return int(np.sum(np.diff(np.r_[False, mask].astype(np.int8)) == 1))


def _coefficient_of_variation(values):
    values = np.asarray(values, float)
    if values.size < 2:
        return float("nan")
    mean = float(np.mean(values))
    return float(np.std(values, ddof=1) / mean) if mean > 0 else float("nan")


def _modulation_fraction(x):
    x = np.asarray(x, float)
    mean = float(np.mean(x))
    if mean <= 0:
        return float("nan")
    return float((np.percentile(x, 95) - np.percentile(x, 5)) / mean)


def _peak_train(
    x, *, bin_ms, min_period_ms, max_period_ms, lowpass_hz=None,
    return_cycle_bounds=False,
):
    """Return peak-count/period statistics for a specified temporal scale."""
    x = np.asarray(x, float)
    fs = 1000.0 / float(bin_ms)
    # A small scale-relative smoothing prevents 2-ms bin noise from becoming a
    # fake cycle without erasing the fastest accepted (5-ms) oscillation.
    smooth_bins = max(1, int(round(min(10.0, 0.15 * min_period_ms) / bin_ms)))
    kernel = np.ones(smooth_bins, float) / smooth_bins
    smooth = np.convolve(x, kernel, mode="same")
    original_range = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    if lowpass_hz is not None:
        sos = ss.butter(4, float(lowpass_hz), btype="lowpass", fs=fs, output="sos")
        smooth = ss.sosfiltfilt(sos, smooth)
    dynamic_range = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    # A high-frequency carrier can otherwise be sub-sampled by the large
    # ``distance`` and masquerade as a clonic macro-rhythm.  When low-passing,
    # retain the prominence floor of the original signal.
    prominence = max(
        1e-9,
        0.20 * (
            max(dynamic_range, original_range)
            if lowpass_hz is not None else dynamic_range
        ),
    )
    distance = max(1, int(round(min_period_ms / bin_ms)))
    peaks, _ = ss.find_peaks(smooth, prominence=prominence, distance=distance)
    if peaks.size >= 2:
        all_periods_ms = np.diff(peaks) * float(bin_ms)
        keep = (
            (all_periods_ms >= float(min_period_ms))
            & (all_periods_ms <= float(max_period_ms))
        )
        periods_ms = all_periods_ms[keep]
        cycle_bounds = np.column_stack((peaks[:-1][keep], peaks[1:][keep]))
    else:
        periods_ms = np.empty(0, float)
        cycle_bounds = np.empty((0, 2), int)
    # Count cycles supported by accepted consecutive intervals, rather than all
    # local maxima returned by the peak finder.
    n_cycles = int(periods_ms.size)
    result = {
        "n_peaks": int(peaks.size),
        "n_cycles": n_cycles,
        "median_period_ms": (
            float(np.median(periods_ms)) if periods_ms.size else None
        ),
        "period_cv": (
            _coefficient_of_variation(periods_ms) if periods_ms.size >= 2
            else None
        ),
    }
    if return_cycle_bounds:
        result["_accepted_cycle_bounds_bins"] = cycle_bounds.astype(int)
    return result


def _periodic_source_phase_signature(
    kymograph, cycle_bounds, *, n_phase_bins=16
):
    """Compact phase-by-axis source signature averaged over accepted cycles.

    Each axial bin is centred across carrier phase before the complete matrix
    is unit-normalised.  This removes static spatial gain while retaining the
    phase ordering and relative source pattern.  Cross-run comparison is
    deliberately circular-shift invariant and is performed by the C1
    adjudicator, not here.
    """
    K = np.asarray(kymograph, float)
    bounds = np.asarray(cycle_bounds, int)
    n_phase = int(n_phase_bins)
    if (
        K.ndim != 2
        or K.shape[0] < 3
        or K.shape[1] < 2
        or not np.all(np.isfinite(K))
        or bounds.ndim != 2
        or bounds.shape[1:] != (2,)
        or bounds.shape[0] < 1
        or n_phase < 4
    ):
        return {
            "status": "unavailable",
            "n_cycles": 0,
            "n_phase_bins": n_phase,
            "n_axis_bins": int(K.shape[1]) if K.ndim == 2 else 0,
            "profile": None,
        }
    phase = np.linspace(0.0, 1.0, n_phase, endpoint=False)
    profiles = []
    sample_axis = np.arange(K.shape[0], dtype=float)
    for start, stop in bounds:
        if start < 0 or stop >= K.shape[0] or stop - start < 2:
            continue
        sample_at = float(start) + phase * float(stop - start)
        profiles.append(np.column_stack([
            np.interp(sample_at, sample_axis, K[:, axis])
            for axis in range(K.shape[1])
        ]))
    if not profiles:
        return {
            "status": "unavailable",
            "n_cycles": 0,
            "n_phase_bins": n_phase,
            "n_axis_bins": int(K.shape[1]),
            "profile": None,
        }
    profile = np.mean(np.stack(profiles, axis=0), axis=0)
    profile = profile - np.mean(profile, axis=0, keepdims=True)
    norm = float(np.linalg.norm(profile))
    if not np.isfinite(norm) or norm <= 1e-12:
        return {
            "status": "unavailable",
            "n_cycles": len(profiles),
            "n_phase_bins": n_phase,
            "n_axis_bins": int(K.shape[1]),
            "profile": None,
        }
    profile = profile / norm
    return {
        "status": "ok",
        "n_cycles": len(profiles),
        "n_phase_bins": n_phase,
        "n_axis_bins": int(K.shape[1]),
        "normalization": "axis_dc_removed_then_global_l2",
        "profile": profile.astype(float).tolist(),
    }


def _derive_active_area(E, active_floor_hz):
    return np.mean(E >= float(active_floor_hz), axis=(1, 2))


def _late_half_drift(values):
    """Match the upstream carrier stationarity audit on the latter half."""
    x = np.asarray(values, float).ravel()
    x = x[x.size // 2:]
    if x.size < 8 or not np.all(np.isfinite(x)):
        return {
            "status": "insufficient",
            "relative_drift": None,
            "variance_ratio": None,
        }
    t = np.arange(x.size, dtype=float)
    slope = float(np.polyfit(t, x, 1)[0])
    mean = float(np.mean(x))
    half = x.size // 2
    v0 = float(np.var(x[:half]))
    v1 = float(np.var(x[half:]))
    return {
        "status": "ok",
        "relative_drift": (
            float(slope * x.size / mean) if abs(mean) > 1e-12 else None
        ),
        "variance_ratio": float(v1 / max(v0, 1e-12)),
    }


def _spatial_entropy_series(E):
    flat = np.asarray(E, float).reshape(len(E), -1)
    total = flat.sum(axis=1)
    out = np.zeros(len(flat), float)
    valid = total > 0
    if np.any(valid):
        p = flat[valid] / total[valid, None]
        out[valid] = -np.sum(
            np.where(p > 0, p * np.log(p + 1e-30), 0.0), axis=1
        ) / np.log(flat.shape[1])
    return out


def common_bounded_gate(
    source_rate_hz,
    *,
    bin_ms,
    active_area_fraction,
    rest_mask=None,
    runaway_early_stop_ms=None,
    saturation_fraction=None,
    refractory_fraction=None,
    thresholds=None,
):
    """Evaluate the shared gate before assigning any bounded phenotype.

    The ordering is intentional: explicit/trending runaway and a sustained
    refractory-scale plateau are not allowed to fall through to ``tonic``;
    separated events with long rest dwells are labelled ``hfo_like_train``
    rather than treated as a low-duty-cycle carrier.
    """
    th = dict(DEFAULTS)
    if thresholds:
        th.update(thresholds)
    x = np.asarray(source_rate_hz, float).ravel()
    area = np.asarray(active_area_fraction, float).ravel()
    if x.size < 16 or area.shape != x.shape:
        raise ValueError("source rate and active-area series must align and contain >=16 bins")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(area)):
        raise ValueError("source rate and active-area series must be finite")
    if np.any((area < 0) | (area > 1)):
        raise ValueError("active-area fraction must lie in [0,1]")

    p95 = float(np.percentile(x, 95))
    activity_threshold = max(
        float(th["active_floor_hz"]),
        float(th["active_relative_to_p95"]) * p95,
    )
    rest_threshold = max(
        float(th["active_floor_hz"]),
        float(th["rest_relative_to_p95"]) * p95,
    )
    active = x >= activity_threshold
    if rest_mask is None:
        rest = x < rest_threshold
    else:
        rest = np.asarray(rest_mask, bool).ravel()
        if rest.shape != x.shape:
            raise ValueError("rest_mask must align with source_rate_hz")

    n_quarter = max(4, x.size // 4)
    early = float(np.median(x[:n_quarter]))
    tail = float(np.median(x[-n_quarter:]))
    tail_escalating = bool(
        tail - early >= 25.0 and tail >= 1.5 * max(early, th["active_floor_hz"])
    )
    explicit_runaway = bool(runaway_early_stop_ms is not None)
    runaway = bool(
        tail >= float(th["runaway_rate_hz"])
        or (tail_escalating and p95 >= float(th["runaway_rate_hz"]))
    )
    if saturation_fraction is None:
        sat_fraction = 0.0
    else:
        sat_fraction = float(saturation_fraction)
        if not np.isfinite(sat_fraction) or not 0 <= sat_fraction <= 1:
            raise ValueError("saturation_fraction must be a finite fraction")
    if refractory_fraction is None:
        ref_fraction = 0.0
    else:
        ref_fraction = float(refractory_fraction)
        if not np.isfinite(ref_fraction) or not 0 <= ref_fraction <= 1:
            raise ValueError("refractory_fraction must be a finite fraction")
    saturated = bool(
        sat_fraction >= float(th["saturation_fraction"])
        and ref_fraction >= float(th["refractory_fraction"])
    )

    occupancy = float(np.mean(active))
    longest_rest_ms = float(_longest_true_run(rest) * float(bin_ms))
    n_events = _episode_count(active)
    if explicit_runaway:
        status = "runaway"
    elif saturated:
        status = "saturation"
    elif runaway:
        status = "runaway"
    elif occupancy <= 0.10 and float(np.mean(x)) < activity_threshold:
        status = "rest"
    elif (
        occupancy < float(th["minimum_active_occupancy"])
        and n_events >= int(th["minimum_hfo_train_events"])
        and longest_rest_ms >= float(th["rest_dwell_ms"])
    ):
        status = "hfo_like_train"
    elif occupancy < float(th["minimum_active_occupancy"]):
        status = "indeterminate"
    else:
        status = "bounded"

    return {
        "status": status,
        "activity_threshold_hz": activity_threshold,
        "rest_threshold_hz": rest_threshold,
        "active_occupancy": occupancy,
        "longest_rest_dwell_ms": longest_rest_ms,
        "n_active_episodes": n_events,
        "tail_escalating": tail_escalating,
        "early_median_hz": early,
        "tail_median_hz": tail,
        "source_mean_hz": float(np.mean(x)),
        "source_p95_hz": p95,
        "median_active_area_fraction": float(np.median(area)),
        "saturation_fraction": sat_fraction,
        "refractory_fraction": ref_fraction,
    }


def two_zone_activity_gate(
    kymograph,
    axis_positions,
    *,
    readout_kernel_width_mm,
    active_floor_hz=5.0,
    occupancy_min=0.80,
):
    """Require persistent activity in two spatially independent axial zones.

    This is the common carrier-extent gate, not the ordered-propagation test.
    A spatial relay may fail while a persistent two-zone tonic carrier passes.
    Zone selection is deterministic and uses only the preregistered rate,
    occupancy, and physical readout-footprint thresholds.
    """
    K = np.asarray(kymograph, float)
    pos = np.asarray(axis_positions, float).ravel()
    width = float(readout_kernel_width_mm)
    if (
        K.ndim != 2
        or K.shape[1] != pos.size
        or K.shape[0] < 8
        or pos.size < 2
        or not np.all(np.isfinite(K))
        or not np.all(np.isfinite(pos))
        or not np.isfinite(width)
        or width <= 0
    ):
        return {
            "status": "indeterminate",
            "pass": False,
            "reason": "invalid_or_missing_two_zone_inputs",
        }
    occupancy = np.mean(K >= float(active_floor_hz), axis=0)
    active = np.flatnonzero(occupancy >= float(occupancy_min))
    ordered = active[np.argsort(pos[active])]
    kept = []
    for index in ordered:
        if all(abs(float(pos[index] - pos[prior])) > width for prior in kept):
            kept.append(int(index))
    passed = len(kept) >= 2
    return {
        "status": "pass" if passed else "fail",
        "pass": passed,
        "reason": (
            "two_persistent_zones_beyond_one_readout_kernel"
            if passed else "fewer_than_two_independent_persistent_zones"
        ),
        "active_zone_indices": active.astype(int).tolist(),
        "independent_zone_indices": kept,
        "n_independent_active_zones": len(kept),
        "active_zone_occupancy": occupancy[active].astype(float).tolist(),
        "occupancy_min": float(occupancy_min),
        "active_floor_hz": float(active_floor_hz),
        "readout_kernel_width_mm": width,
    }


def spatial_relay_modifier(
    kymograph,
    axis_positions,
    *,
    bin_ms,
    n_perm=999,
    rng_seed=0,
    min_axis_span_fraction=0.25,
    min_first_passage_bins=2,
):
    """Test ordered axial first passage against a location-permutation null.

    At least two spatial zones, a non-trivial first-passage spread, and a
    permutation-significant position/latency association are all required.
    A simultaneous whole-field flash therefore cannot be labelled a relay.
    """
    K = np.asarray(kymograph, float)
    pos = np.asarray(axis_positions, float).ravel()
    if K.ndim != 2 or K.shape[1] != pos.size or K.shape[0] < 8 or pos.size < 4:
        return {
            "status": "indeterminate",
            "is_spatial_relay": False,
            "reason": "kymograph must be time x axis with >=4 positions and >=8 bins",
        }
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(pos)):
        return {
            "status": "indeterminate",
            "is_spatial_relay": False,
            "reason": "non-finite kymograph or positions",
        }

    lo = np.percentile(K, 10, axis=0)
    hi = np.percentile(K, 95, axis=0)
    dynamic = hi - lo
    usable = dynamic > max(1e-9, 0.05 * float(np.max(dynamic)))
    threshold = lo + 0.35 * dynamic
    crossed = K >= threshold[None, :]
    active = usable & np.any(crossed, axis=0)
    idx = np.flatnonzero(active)
    if idx.size < 4:
        return {
            "status": "no_relay",
            "is_spatial_relay": False,
            "reason": "fewer than four active axial bins",
        }
    first = np.array([np.flatnonzero(crossed[:, j])[0] for j in idx], float)
    p = pos[idx]
    full_span = float(np.ptp(pos))
    observed_span = float(np.ptp(p))
    temporal_span_bins = float(np.ptp(first))
    if full_span <= 0 or observed_span < float(min_axis_span_fraction) * full_span:
        return {
            "status": "no_relay",
            "is_spatial_relay": False,
            "reason": "active positions do not span two separated zones",
            "n_active_positions": int(idx.size),
        }
    if temporal_span_bins < int(min_first_passage_bins):
        return {
            "status": "no_relay",
            "is_spatial_relay": False,
            "reason": "near-simultaneous first passage",
            "n_active_positions": int(idx.size),
            "first_passage_span_ms": temporal_span_bins * float(bin_ms),
        }

    earliest = int(np.min(first))
    flash_fraction = float(np.mean(crossed[earliest, idx]))
    if flash_fraction >= 0.80:
        return {
            "status": "no_relay",
            "is_spatial_relay": False,
            "reason": "whole-field flash at first passage",
            "flash_fraction": flash_fraction,
        }

    order = np.argsort(p)
    p = p[order]
    first = first[order]
    split = p.size // 2
    zone_gap_bins = abs(float(np.median(first[:split]) - np.median(first[split:])))
    if split < 2 or p.size - split < 2 or zone_gap_bins < min_first_passage_bins:
        return {
            "status": "no_relay",
            "is_spatial_relay": False,
            "reason": "two separated zones lack a first-passage delay",
            "zone_gap_ms": zone_gap_bins * float(bin_ms),
        }

    rho = float(stats.spearmanr(p, first).statistic)
    if not np.isfinite(rho):
        return {
            "status": "no_relay",
            "is_spatial_relay": False,
            "reason": "undefined axial first-passage ordering",
        }
    rng = np.random.default_rng(int(rng_seed))
    null = np.empty(int(n_perm), float)
    for i in range(int(n_perm)):
        null[i] = abs(float(stats.spearmanr(p, rng.permutation(first)).statistic))
    observed_ordered_spread = abs(rho) * temporal_span_bins
    null_ordered_spread = null * temporal_span_bins
    null_q975 = float(np.percentile(null_ordered_spread, 97.5))
    p_perm = float(
        (1 + np.sum(null_ordered_spread >= observed_ordered_spread))
        / (int(n_perm) + 1)
    )
    passed = bool(
        p_perm <= 0.025
        and observed_ordered_spread > null_q975
        and abs(rho) >= 0.50
    )
    return {
        "status": "relay" if passed else "no_relay",
        "is_spatial_relay": passed,
        "reason": (
            "ordered first passage exceeds the location-permutation null"
            if passed else "axial ordering does not exceed the permutation null"
        ),
        "n_active_positions": int(idx.size),
        "first_passage_span_ms": temporal_span_bins * float(bin_ms),
        "zone_gap_ms": zone_gap_bins * float(bin_ms),
        "axial_first_passage_rho": rho,
        "direction_sign": int(np.sign(rho)),
        "ordered_first_passage_spread": float(observed_ordered_spread),
        "permutation_null_q975": null_q975,
        "permutation_p": p_perm,
        "n_perm": int(n_perm),
        "flash_fraction": flash_fraction,
    }


def classify_phasec_run(
    E_rate_grid,
    I_rate_grid,
    *,
    bin_ms,
    source_rate_hz,
    rest_mask=None,
    active_area_fraction=None,
    kymograph=None,
    axis_positions=None,
    readout_kernel_width_mm=None,
    runaway_early_stop_ms=None,
    saturation_fraction=None,
    refractory_fraction=None,
    thresholds=None,
    relay_n_perm=999,
    relay_rng_seed=0,
):
    """Assign one fail-closed Phase-C temporal phenotype plus relay modifier."""
    th = dict(DEFAULTS)
    if thresholds:
        th.update(thresholds)
    E = np.asarray(E_rate_grid, float)
    I = np.asarray(I_rate_grid, float)
    if E.ndim != 3 or I.shape != E.shape:
        raise ValueError("E/I rate grids must align as time x y x x")
    if E.shape[0] != np.asarray(source_rate_hz).size:
        raise ValueError("source rate must align with E/I grid time")
    if not np.all(np.isfinite(E)) or not np.all(np.isfinite(I)):
        raise ValueError("E/I grids must be finite")
    if float(bin_ms) <= 0:
        raise ValueError("bin_ms must be positive")

    if active_area_fraction is None:
        area = _derive_active_area(E, th["active_floor_hz"])
    else:
        area = np.asarray(active_area_fraction, float).ravel()
    gate = common_bounded_gate(
        source_rate_hz,
        bin_ms=bin_ms,
        active_area_fraction=area,
        rest_mask=rest_mask,
        runaway_early_stop_ms=runaway_early_stop_ms,
        saturation_fraction=saturation_fraction,
        refractory_fraction=refractory_fraction,
        thresholds=th,
    )

    spatial_extent = (
        two_zone_activity_gate(
            kymograph,
            axis_positions,
            readout_kernel_width_mm=readout_kernel_width_mm,
            active_floor_hz=th["active_floor_hz"],
            occupancy_min=th["two_zone_occupancy"],
        )
        if (
            kymograph is not None
            and axis_positions is not None
            and readout_kernel_width_mm is not None
        )
        else {
            "status": "indeterminate",
            "pass": False,
            "reason": "two_zone_inputs_not_supplied",
        }
    )
    relay = {
        "status": "not_tested",
        "is_spatial_relay": False,
        "reason": "kymograph/axis positions not supplied",
    }
    if kymograph is not None and axis_positions is not None:
        relay = spatial_relay_modifier(
            kymograph,
            axis_positions,
            bin_ms=bin_ms,
            n_perm=relay_n_perm,
            rng_seed=relay_rng_seed,
        )

    diagnostics = {
        "global_modulation_fraction": _modulation_fraction(source_rate_hz),
        "E_I_mean_rate_ratio": float(
            np.mean(E) / np.mean(I) if np.mean(I) > 0 else np.nan
        ),
        "spatial_extent": spatial_extent,
    }
    if gate["status"] != "bounded":
        return {
            "phasec_phenotype_version": PHASEC_PHENOTYPE_VERSION,
            "phenotype": {
                "saturation": "refractory_saturated",
                "rest": "rest_or_silence",
                "hfo_like_train": "hfo_like_relaxation_train",
                "indeterminate": "probabilistically_indeterminate",
            }.get(gate["status"], gate["status"]),
            "bounded_gate_pass": False,
            "bounded_gate": gate,
            "temporal_diagnostics": diagnostics,
            "spatial_relay": relay,
            "claim_boundary": "run-level phenotype only; no entry, exit, or lifecycle claim",
        }
    if not spatial_extent["pass"]:
        return {
            "phasec_phenotype_version": PHASEC_PHENOTYPE_VERSION,
            "phenotype": "probabilistically_indeterminate",
            "bounded_gate_pass": False,
            "bounded_gate": gate,
            "temporal_diagnostics": diagnostics,
            "spatial_relay": relay,
            "claim_boundary": "run-level phenotype only; no entry, exit, or lifecycle claim",
        }

    x = np.asarray(source_rate_hz, float).ravel()
    stationarity = {
        "source_rate": _late_half_drift(x),
        "active_area": _late_half_drift(area),
        "source_energy": _late_half_drift(np.mean(E ** 2, axis=(1, 2))),
        "spatial_entropy": _late_half_drift(_spatial_entropy_series(E)),
    }
    stationarity_ok = True
    for row in stationarity.values():
        rel = row["relative_drift"]
        ratio = row["variance_ratio"]
        if (
            row["status"] != "ok"
            or rel is None
            or abs(rel) > float(th["maximum_abs_relative_drift"])
            or ratio is None
            or ratio > float(th["maximum_variance_ratio"])
        ):
            stationarity_ok = False
    diagnostics["stationarity"] = stationarity
    diagnostics["stationarity_ok"] = stationarity_ok
    if not stationarity_ok:
        return {
            "phasec_phenotype_version": PHASEC_PHENOTYPE_VERSION,
            "phenotype": "probabilistically_indeterminate",
            "bounded_gate_pass": False,
            "bounded_gate": gate,
            "temporal_diagnostics": diagnostics,
            "spatial_relay": relay,
            "claim_boundary": "run-level phenotype only; no entry, exit, or lifecycle claim",
        }
    modulation = diagnostics["global_modulation_fraction"]
    clonic = _peak_train(
        x,
        bin_ms=bin_ms,
        min_period_ms=th["clonic_min_period_ms"],
        max_period_ms=th["clonic_max_period_ms"],
        lowpass_hz=min(5.0, 0.8 * 1000.0 / th["clonic_min_period_ms"]),
    )
    periodic = _peak_train(
        x,
        bin_ms=bin_ms,
        min_period_ms=max(2.0 * float(bin_ms), 1000.0 / 150.0),
        max_period_ms=200.0,
        return_cycle_bounds=True,
    )
    periodic_bounds = periodic.pop("_accepted_cycle_bounds_bins")
    periodic["source_phase_signature"] = (
        _periodic_source_phase_signature(kymograph, periodic_bounds)
        if kymograph is not None
        else _periodic_source_phase_signature(
            np.empty((0, 0)), periodic_bounds
        )
    )
    if periodic_bounds.size:
        reset_cycles = [
            bool(np.any(np.asarray(rest_mask, bool)[start:stop + 1]))
            if rest_mask is not None else False
            for start, stop in periodic_bounds
        ]
        periodic["rest_reset_fraction"] = float(np.mean(reset_cycles))
    else:
        periodic["rest_reset_fraction"] = None
    diagnostics.update({"clonic": clonic, "periodic": periodic})

    rest_dwell_ok = gate["longest_rest_dwell_ms"] >= float(th["rest_dwell_ms"])
    clonic_pass = bool(
        clonic["n_cycles"] >= int(th["minimum_bursts"]) - 1
        and modulation >= float(th["minimum_modulation"])
        and gate["active_occupancy"] >= float(th["minimum_active_occupancy"])
        and not rest_dwell_ok
        and clonic["period_cv"] is not None
        and clonic["period_cv"] <= float(th["maximum_ibi_cv"])
    )
    periodic_pass = bool(
        periodic["n_cycles"] >= int(th["minimum_cycles"])
        and modulation >= float(th["minimum_modulation"])
        and not rest_dwell_ok
        and periodic["rest_reset_fraction"] is not None
        and periodic["rest_reset_fraction"]
        <= float(th["maximum_periodic_rest_reset_fraction"])
        and periodic["source_phase_signature"]["status"] == "ok"
        and periodic["period_cv"] is not None
        and periodic["period_cv"] <= float(th["maximum_period_cv"])
    )
    if clonic_pass:
        phenotype = "clonic_or_bursting_carrier"
    elif periodic_pass:
        phenotype = "periodic_non_tonic_carrier"
    elif np.isfinite(modulation) and modulation < float(th["minimum_modulation"]):
        # C0 metrics, evaluated by the caller, are needed to upgrade this to
        # AI_tonic_window.  Envelope stationarity alone only supports tonic.
        phenotype = "tonic_non_AI"
    else:
        phenotype = "probabilistically_indeterminate"

    return {
        "phasec_phenotype_version": PHASEC_PHENOTYPE_VERSION,
        "phenotype": phenotype,
        "bounded_gate_pass": True,
        "bounded_gate": gate,
        "temporal_diagnostics": diagnostics,
        "spatial_relay": relay,
        "claim_boundary": "run-level phenotype only; no entry, exit, or lifecycle claim",
    }
