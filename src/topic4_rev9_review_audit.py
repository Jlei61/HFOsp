"""Pure helpers for the rev9 post-execution scientific audit.

These functions do not change the frozen rev9 experiment.  They make the
review semantics explicit and support zero-simulation reanalysis of existing
response and spontaneous-run artifacts.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from src.sef_hfo_events import detect_events


def finite_interval(values, *, seed, repeats=2000):
    """Mean and network bootstrap interval over finite values."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(estimate=None, interval_95=[None, None], n=0)
    estimate = float(values.mean())
    if len(values) == 1:
        interval = [estimate, estimate]
    else:
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(values, size=(int(repeats), len(values)), replace=True)
        interval = np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()
    return dict(estimate=estimate, interval_95=interval, n=int(len(values)))


def binary_js_divergence(left, right):
    """Jensen-Shannon divergence in bits for two binary proportions."""
    left = float(left)
    right = float(right)
    if not 0.0 <= left <= 1.0 or not 0.0 <= right <= 1.0:
        raise ValueError("binary proportions must lie in [0, 1]")
    p = np.asarray([1.0 - left, left], float)
    q = np.asarray([1.0 - right, right], float)
    return float(jensenshannon(p, q, base=2.0) ** 2)


def mode_evaluability(counts, ood_fraction, *, min_per_mode=10,
                      max_ood_fraction=0.5):
    """Adjudicate whether a patient-mode matrix has enough in-support data."""
    counts = np.asarray(counts, int)
    if counts.shape != (2,):
        raise ValueError("mode counts must have shape (2,)")
    reasons = []
    if int(counts.min()) < int(min_per_mode):
        reasons.append("fewer_than_minimum_in_distribution_events_per_mode")
    if not np.isfinite(ood_fraction) or float(ood_fraction) > max_ood_fraction:
        reasons.append("ood_fraction_above_limit")
    return dict(
        status="EVALUABLE" if not reasons else "NOT_EVALUABLE",
        in_distribution_counts=counts.tolist(),
        ood_fraction=None if not np.isfinite(ood_fraction) else float(ood_fraction),
        minimum_events_per_mode=int(min_per_mode),
        maximum_ood_fraction=float(max_ood_fraction),
        reasons=reasons,
    )


def network_mode_summary(labels, ood, seed_ids, seeds, *, duration_s,
                         patient_mode_b_fraction, bootstrap_seed,
                         bootstrap_repeats=2000):
    """Summarize frozen-mode occupancy with one network as the sampling unit."""
    labels = np.asarray(labels, int)
    ood = np.asarray(ood, bool)
    seed_ids = np.asarray(seed_ids, int)
    seeds = np.asarray(seeds, int)
    if labels.shape != ood.shape or labels.shape != seed_ids.shape:
        raise ValueError("labels, OOD flags and seed ids must align")
    rows = []
    for seed in seeds:
        selected = seed_ids == seed
        in_distribution = selected & ~ood
        counts_all = np.bincount(labels[selected], minlength=2)
        counts_id = np.bincount(labels[in_distribution], minlength=2)
        n_id = int(counts_id.sum())
        rows.append(dict(
            seed=int(seed),
            counts_all=counts_all.tolist(),
            counts_in_distribution=counts_id.tolist(),
            rate_a_hz=float(counts_id[0] / duration_s),
            rate_b_hz=float(counts_id[1] / duration_s),
            has_both_in_distribution=bool(np.all(counts_id > 0)),
            mode_b_fraction_in_distribution=(
                None if n_id == 0 else float(counts_id[1] / n_id)),
            ood_fraction=(None if not selected.any() else float(np.mean(ood[selected]))),
        ))
    b_fraction = np.asarray([
        np.nan if row["mode_b_fraction_in_distribution"] is None
        else row["mode_b_fraction_in_distribution"] for row in rows
    ], float)
    rate_a = np.asarray([row["rate_a_hz"] for row in rows], float)
    rate_b = np.asarray([row["rate_b_hz"] for row in rows], float)
    both = np.asarray([row["has_both_in_distribution"] for row in rows], float)
    pooled_counts = np.bincount(labels[~ood], minlength=2)
    pooled_b = (None if pooled_counts.sum() == 0 else
                float(pooled_counts[1] / pooled_counts.sum()))
    equal_network = finite_interval(
        b_fraction, seed=bootstrap_seed, repeats=bootstrap_repeats)
    return dict(
        per_network=rows,
        n_networks=int(len(seeds)),
        n_networks_with_both_in_distribution=int(both.sum()),
        fraction_networks_with_both_in_distribution=finite_interval(
            both, seed=int(bootstrap_seed) + 1, repeats=bootstrap_repeats),
        mode_a_rate_hz=finite_interval(
            rate_a, seed=int(bootstrap_seed) + 2, repeats=bootstrap_repeats),
        mode_b_rate_hz=finite_interval(
            rate_b, seed=int(bootstrap_seed) + 3, repeats=bootstrap_repeats),
        pooled_in_distribution_counts=pooled_counts.tolist(),
        pooled_mode_b_fraction=pooled_b,
        equal_network_weighted_mode_b_fraction=equal_network,
        patient_mode_b_fraction=float(patient_mode_b_fraction),
        pooled_mode_proportion_js_bits=(
            None if pooled_b is None else binary_js_divergence(
                pooled_b, patient_mode_b_fraction)),
        equal_network_mode_proportion_js_bits=(
            None if equal_network["estimate"] is None else binary_js_divergence(
                equal_network["estimate"], patient_mode_b_fraction)),
    )


def common_detector_metrics(active_fraction, bin_width_ms, threshold):
    """Redetect scalar event burden using one absolute activity threshold."""
    trace = np.asarray(active_fraction, float)
    if trace.ndim != 1 or not np.isfinite(trace).all():
        raise ValueError("active_fraction must be one finite trajectory")
    threshold = float(threshold)
    events = detect_events(trace, float(bin_width_ms), event_on_frac=threshold)
    above = np.clip(trace - threshold, 0.0, np.inf)
    return dict(
        n_events=int(len(events)),
        event_rate_hz=float(len(events) / (len(trace) * bin_width_ms / 1000.0)),
        time_above_fraction=float(np.mean(trace > threshold)),
        integrated_excess_fraction_ms=float(above.sum() * bin_width_ms),
        peak_active_fraction=float(trace.max(initial=0.0)),
        mean_active_fraction=float(trace.mean()),
        p95_active_fraction=float(np.quantile(trace, 0.95)),
    )


def response_map_spearman(left, right):
    """Spearman map agreement over the union of non-zero response cells."""
    left = np.asarray(left, float).ravel()
    right = np.asarray(right, float).ravel()
    valid = np.isfinite(left) & np.isfinite(right) & ((left > 0.0) | (right > 0.0))
    if valid.sum() < 3 or np.ptp(left[valid]) <= 0.0 or np.ptp(right[valid]) <= 0.0:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return None if not np.isfinite(value) else value


def response_site_adjudication(rows, *, minimum_valid_pairs,
                                gain_bounds=(0.8, 1.25),
                                maximum_abs_r90_delta_mm=1.0,
                                minimum_map_rho=0.8):
    """Post-hoc site table for the response-equivalence questions.

    A formal PASS is intentionally not returned: these positive-response gain
    definitions were not frozen before alpha selection.  The diagnostic label
    identifies the observed failure pattern while preserving that distinction.
    """
    rows = list(rows)
    valid = [row for row in rows if row.get("paired_eligible")]
    source_ratio = np.asarray([
        row.get("source_gain_ratio", np.nan) for row in valid], float)
    downstream_ratio = np.asarray([
        row.get("downstream_gain_ratio", np.nan) for row in valid], float)
    r90_delta = np.asarray([row.get("r90_delta_mm", np.nan) for row in valid], float)
    map_rho = np.asarray([row.get("map_rho", np.nan) for row in valid], float)

    def median(values):
        values = values[np.isfinite(values)]
        return None if not len(values) else float(np.median(values))

    source_med = median(source_ratio)
    downstream_med = median(downstream_ratio)
    r90_med = median(r90_delta)
    map_med = median(map_rho)
    low, high = map(float, gain_bounds)
    coverage_ok = len(valid) >= int(minimum_valid_pairs)
    source_ok = source_med is not None and low <= source_med <= high
    downstream_ok = downstream_med is not None and low <= downstream_med <= high
    r90_ok = r90_med is not None and abs(r90_med) <= maximum_abs_r90_delta_mm
    map_ok = map_med is not None and map_med >= minimum_map_rho
    if not coverage_ok:
        pattern = "COVERAGE_FAIL"
    elif source_ok and downstream_ok and r90_ok and map_ok:
        pattern = "POSTHOC_DIAGNOSTIC_MATCH"
    elif downstream_ok and not source_ok:
        pattern = "SOURCE_NUCLEATION_FAIL_DOWNSTREAM_PARTIAL_MATCH"
    else:
        pattern = "LOCAL_RESPONSE_MISMATCH"
    return dict(
        formal_status="UNRESOLVED_METRIC_NOT_FROZEN_BEFORE_SELECTION",
        diagnostic_pattern=pattern,
        n_valid_pairs=int(len(valid)),
        minimum_valid_pairs=int(minimum_valid_pairs),
        source_gain_ratio_median=source_med,
        downstream_gain_ratio_median=downstream_med,
        r90_edge_minus_node_median_mm=r90_med,
        positive_map_spearman_median=map_med,
        posthoc_checks=dict(
            coverage=coverage_ok, source_gain=source_ok,
            downstream_gain=downstream_ok, r90=r90_ok, map_rho=map_ok),
    )


def pareto_minimize_maximize(cost, benefit):
    """Return the non-dominated mask for lower cost and higher benefit."""
    cost = np.asarray(cost, float)
    benefit = np.asarray(benefit, float)
    if cost.shape != benefit.shape or cost.ndim != 1:
        raise ValueError("cost and benefit must be aligned one-dimensional arrays")
    finite = np.isfinite(cost) & np.isfinite(benefit)
    output = np.zeros(len(cost), bool)
    for index in np.flatnonzero(finite):
        dominated = finite & (cost <= cost[index]) & (benefit >= benefit[index])
        strictly = (cost < cost[index]) | (benefit > benefit[index])
        output[index] = not np.any(dominated & strictly)
    return output


__all__ = [
    "binary_js_divergence", "common_detector_metrics", "finite_interval",
    "mode_evaluability", "network_mode_summary", "response_map_spearman",
    "pareto_minimize_maximize", "response_site_adjudication",
]
