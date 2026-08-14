"""Pure post-processing for the fixed five-arm FCXR-LC6A phenotype map."""

from __future__ import annotations

import numpy as np
from scipy.stats import theilslopes


def normalized_theil_sen(values, *, dt_s: float, tail_s: float = 2.0, floor: float = 1e-6) -> dict:
    values = np.asarray(values, float)
    n_tail = max(2, int(round(float(tail_s) / float(dt_s))))
    use = values[-n_tail:]
    finite = np.isfinite(use)
    if np.count_nonzero(finite) < 2:
        return {
            "slope_per_s": float("nan"), "ci_low_per_s": float("nan"),
            "ci_high_per_s": float("nan"), "normalized_ci_high_per_s": float("nan"),
            "n": int(np.count_nonzero(finite)),
        }
    x = np.arange(use.size, dtype=float) * float(dt_s)
    slope, _intercept, low, high = theilslopes(use[finite], x[finite], alpha=.95)
    scale = max(abs(float(np.median(use[finite]))), float(floor))
    return {
        "slope_per_s": float(slope), "ci_low_per_s": float(low),
        "ci_high_per_s": float(high), "normalized_ci_high_per_s": float(high / scale),
        "normalization": float(scale), "n": int(np.count_nonzero(finite)),
    }


def weighted_quantile(values, weights, quantile):
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(finite):
        return float("nan")
    order = np.argsort(values[finite])
    x = values[finite][order]
    w = weights[finite][order]
    cdf = (np.cumsum(w) - .5 * w) / np.sum(w)
    return float(np.interp(float(quantile), cdf, x, left=x[0], right=x[-1]))


def spatial_slow_flow_readout(
    rate_maps_1s, d_maps_1s, positions, cell_bins, occupancy, *, axis_unit,
    source_xy, sheet_size_mm: float, local_rate_threshold_hz: float,
    onset_ms: float | None,
) -> dict:
    rates = np.asarray(rate_maps_1s, float)
    d_maps = np.asarray(d_maps_1s, float)
    positions = np.asarray(positions, float)
    cell_bins = np.asarray(cell_bins, np.int64)
    occupancy = np.asarray(occupancy, float)
    axis = np.asarray(axis_unit, float)
    axis /= np.linalg.norm(axis)
    source = np.asarray(source_xy, float)
    if rates.shape != d_maps.shape or rates.shape[1] != occupancy.size:
        raise ValueError("spatial rate/D maps are incompatible")
    n_bins = occupancy.size
    bin_x = np.bincount(cell_bins, weights=positions[:, 0], minlength=n_bins)
    bin_y = np.bincount(cell_bins, weights=positions[:, 1], minlength=n_bins)
    centers = np.column_stack([
        np.divide(bin_x, occupancy, out=np.full(n_bins, np.nan), where=occupancy > 0),
        np.divide(bin_y, occupancy, out=np.full(n_bins, np.nan), where=occupancy > 0),
    ])
    axial = (centers - source) @ axis
    bin_area = (float(sheet_size_mm) / round(np.sqrt(n_bins))) ** 2
    d0 = d_maps[0]
    rate_front, d_front, d_width, d_lead, area, centroids = [], [], [], [], [], []
    for rate, d_field in zip(rates, d_maps):
        rate_weight = np.maximum(rate - float(local_rate_threshold_hz), 0.0) * occupancy
        d_weight = np.maximum(d_field - d0, 0.0) * occupancy
        r95 = weighted_quantile(axial, rate_weight, .95)
        d95 = weighted_quantile(axial, d_weight, .95)
        d05 = weighted_quantile(axial, d_weight, .05)
        rate_front.append(r95); d_front.append(d95)
        d_width.append(d95 - d05 if np.isfinite(d95) and np.isfinite(d05) else np.nan)
        d_lead.append(d95 - r95 if np.isfinite(d95) and np.isfinite(r95) else np.nan)
        area.append(float(np.count_nonzero((rate >= local_rate_threshold_hz) & (occupancy > 0))) * bin_area)
        total = np.sum(rate_weight)
        centroids.append(
            np.sum(centers * rate_weight[:, None], axis=0) / total
            if total > 0 else np.array([np.nan, np.nan])
        )
    rate_front = np.asarray(rate_front); d_front = np.asarray(d_front)
    d_width = np.asarray(d_width); d_lead = np.asarray(d_lead)
    area = np.asarray(area); centroids = np.asarray(centroids)
    onset_index = None if onset_ms is None else int(np.floor(float(onset_ms) / 1000.0))
    speed = float("nan")
    if onset_index is not None:
        stop = min(rate_front.size, onset_index + 6)
        use = rate_front[onset_index:stop]
        finite = np.isfinite(use)
        if np.count_nonzero(finite) >= 2:
            speed = float(np.polyfit(np.arange(use.size)[finite], use[finite], 1)[0])
    centroid_rms = float("nan")
    if onset_index is not None and onset_index < len(centroids):
        post = centroids[onset_index:]
        finite = np.all(np.isfinite(post), axis=1)
        if np.count_nonzero(finite) >= 2:
            mean = np.mean(post[finite], axis=0)
            centroid_rms = float(np.sqrt(np.mean(np.sum((post[finite] - mean) ** 2, axis=1))))
    return {
        "rate_front_q95_axis_mm": rate_front.tolist(),
        "D_front_q95_axis_mm": d_front.tolist(),
        "D_halo_width_q05_q95_mm": d_width.tolist(),
        "D_halo_lead_mm": d_lead.tolist(),
        "max_D_halo_lead_mm": float(np.nanmax(d_lead)) if np.any(np.isfinite(d_lead)) else None,
        "max_D_halo_width_mm": float(np.nanmax(d_width)) if np.any(np.isfinite(d_width)) else None,
        "active_area_mm2": area.tolist(),
        "max_active_area_mm2": float(np.max(area, initial=0.0)),
        "recruitment_front_speed_mm_per_s": speed,
        "centroid_rms_mm": centroid_rms,
    }


def event_metrics(events, *, end_ms: float) -> dict:
    use = [event for event in events if float(event["t_on"]) < float(end_ms)]
    onsets = np.asarray([event["t_on"] for event in use], float)
    durations = np.asarray([event["dur_ms"] for event in use], float)
    participation = np.asarray([event["peak_ext"] for event in use], float)
    iei = np.diff(onsets)
    return {
        "window_ms": [0.0, float(end_ms)], "n_events": int(len(use)),
        "event_rate_hz": float(len(use) / (float(end_ms) / 1000.0)),
        "iei_median_ms": float(np.median(iei)) if iei.size else None,
        "duration_median_ms": float(np.median(durations)) if durations.size else None,
        "participation_median": float(np.median(participation)) if participation.size else None,
    }


def baseline_tradeoff(metrics, reference, *, relative_tolerance: float = .25) -> dict:
    comparisons = {}
    for key in ("event_rate_hz", "iei_median_ms", "duration_median_ms", "participation_median"):
        value, target = metrics.get(key), reference.get(key)
        if value is None or target is None or float(target) == 0.0:
            comparisons[key] = None
        else:
            comparisons[key] = float((float(value) - float(target)) / abs(float(target)))
    finite = [abs(value) for value in comparisons.values() if value is not None]
    return {
        "relative_differences": comparisons,
        "max_absolute_relative_difference": max(finite) if finite else None,
        "tradeoff": bool(any(value > float(relative_tolerance) for value in finite)),
        "relative_tolerance": float(relative_tolerance),
    }


def classify_high_state(
    *, global_onset_ms, local_onset_ms, offset_ms, total_ms,
    global_rate_100ms, d_trace, h_trace, trace_dt_ms,
    max_near_refractory_fraction, right_censored=False,
) -> dict:
    onsets = [value for value in (global_onset_ms, local_onset_ms) if value is not None]
    onset = min(onsets) if onsets else None
    if onset is None:
        return {
            "headline": "NO_ONSET", "bounded_candidate": False,
            "onset_ms": None, "responsiveness": "NOT_TESTED",
        }
    if offset_ms is not None:
        return {
            "headline": "AUTONOMOUS_OFFSET_OBSERVED", "bounded_candidate": False,
            "onset_ms": float(onset), "responsiveness": "NOT_TESTED",
        }
    rate = np.asarray(global_rate_100ms, float)
    onset_bin = int(np.floor(float(onset) / 100.0))
    post = rate[onset_bin:]
    dwell_s = (float(total_ms) - float(onset)) / 1000.0
    rate_drift = normalized_theil_sen(post, dt_s=.1, tail_s=2.0)
    d_drift = normalized_theil_sen(d_trace, dt_s=float(trace_dt_ms) / 1000.0, tail_s=2.0)
    h_drift = normalized_theil_sen(h_trace, dt_s=float(trace_dt_ms) / 1000.0, tail_s=2.0)
    complete = post[: (post.size // 10) * 10].reshape(-1, 10).mean(axis=1) if post.size >= 10 else np.array([])
    global_saturated = bool(np.nanmax(complete) >= 250.0) if complete.size else False
    local_saturated = float(max_near_refractory_fraction) >= .05
    drift_ok = all(
        np.isfinite(row["normalized_ci_high_per_s"])
        and row["normalized_ci_high_per_s"] <= .05
        for row in (rate_drift, d_drift, h_drift)
    )
    bounded = bool(
        dwell_s >= 5.0 and not right_censored and not global_saturated
        and not local_saturated and drift_ok
    )
    if bounded:
        headline = "BOUNDED_CARRIER_CANDIDATE"
    elif global_saturated or local_saturated:
        headline = "SATURATED_HIGH_STATE"
    elif right_censored or dwell_s < 5.0:
        headline = "RIGHT_CENSORED_OR_SHORT_HIGH_STATE"
    else:
        headline = "NON_SATURATED_HIGH_WITH_POSITIVE_DRIFT"
    margin = min(
        (250.0 - float(np.nanmax(complete))) / 250.0 if complete.size else -np.inf,
        (.05 - float(max_near_refractory_fraction)) / .05,
        *(.05 - row["normalized_ci_high_per_s"] for row in (rate_drift, d_drift, h_drift)),
    )
    return {
        "headline": headline, "bounded_candidate": bounded,
        "onset_ms": float(onset), "high_state_dwell_s": float(dwell_s),
        "global_saturated": global_saturated, "local_saturated": local_saturated,
        "rate_drift": rate_drift, "D_drift": d_drift, "H_drift": h_drift,
        "boundedness_margin": float(margin), "responsiveness": "NOT_TESTED",
    }
