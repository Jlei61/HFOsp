"""Readouts and fail-closed adjudication for the spatial Z/M phase diagram.

The SNN is finite and stochastic.  This module therefore calls its outputs
branch or bistability *candidates*; mathematical bifurcation labels are left to
a later deterministic continuation/Jacobian layer.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from src.topic4_runaway_morphology import rolling_full_field_recruitment
from src.topic4_tonic_fixed_point import population_rate_modulation


@dataclass(frozen=True)
class BranchThresholds:
    low_median_rate_hz: float = 80.0
    low_active_fraction: float = 0.50
    high_rate_threshold_hz: float = 120.0
    low_max_sustained_high_run_ms: float = 100.0
    high_median_rate_hz: float = 300.0
    high_active_fraction: float = 0.85
    high_sheet_fraction: float = 0.85
    high_global_duty: float = 0.80


def _longest_true_run_ms(mask, dt_ms):
    values = np.asarray(mask, bool)
    padded = np.r_[False, values, False]
    intervals = np.flatnonzero(padded[1:] != padded[:-1]).reshape(-1, 2)
    return float(max((stop - start for start, stop in intervals), default=0)
                 * float(dt_ms))


def stationary_rate_metrics(rate_e_hz, *, dt_ms, burn_in_ms,
                            high_rate_threshold_hz=120.0):
    """Rate-only endpoint metrics, including event-tolerant persistence."""
    rate = np.asarray(rate_e_hz, float)
    if rate.ndim != 1:
        raise ValueError("rate must have one time dimension")
    burn_steps = int(round(float(burn_in_ms) / float(dt_ms)))
    if burn_steps < 0 or burn_steps >= len(rate):
        raise ValueError("burn-in leaves no scoring window")
    selected_rate = rate[burn_steps:]
    smoothed = uniform_filter1d(
        selected_rate,
        size=max(1, int(round(20.0 / float(dt_ms)))),
        mode="nearest",
    )
    high = smoothed >= float(high_rate_threshold_hz)
    midpoint = len(smoothed) // 2
    return {
        "scoring_duration_ms": float(len(selected_rate) * float(dt_ms)),
        "median_rate_hz": float(np.median(smoothed)),
        "q05_rate_hz": float(np.quantile(smoothed, 0.05)),
        "q95_rate_hz": float(np.quantile(smoothed, 0.95)),
        "first_half_median_rate_hz": float(np.median(smoothed[:midpoint])),
        "second_half_median_rate_hz": float(np.median(smoothed[midpoint:])),
        "fraction_rate_ge_120_hz": float(np.mean(high)),
        "longest_rate_ge_120_hz_ms": _longest_true_run_ms(high, dt_ms),
    }


def stationary_metrics(rate_e_hz, spikes_e, positions_e, *, dt_ms,
                       sheet_l_mm, burn_in_ms):
    """Measure one frozen-q endpoint after a predeclared burn-in."""
    rate = np.asarray(rate_e_hz, float)
    spikes = np.asarray(spikes_e, bool)
    if rate.ndim != 1 or spikes.ndim != 2 or len(rate) != len(spikes):
        raise ValueError("rate and spikes must share one time dimension")
    rate_metrics = stationary_rate_metrics(
        rate, dt_ms=dt_ms, burn_in_ms=burn_in_ms)
    burn_steps = int(round(float(burn_in_ms) / float(dt_ms)))
    selected_rate = rate[burn_steps:]
    selected_spikes = spikes[burn_steps:]
    recruitment = rolling_full_field_recruitment(
        selected_spikes, positions_e, dt_ms=dt_ms,
        sheet_l_mm=sheet_l_mm,
    )
    active = np.asarray(recruitment["active_neuron_fraction"], float)
    sheet = np.asarray(recruitment["recruited_spatial_fraction"], float)
    try:
        rhythm = population_rate_modulation(selected_rate, dt_ms=dt_ms)
    except ValueError:
        rhythm = None
    return {
        **rate_metrics,
        "median_active_E_fraction_20ms": float(np.median(active)),
        "median_recruited_sheet_fraction_1mm": float(np.median(sheet)),
        "joint_global_recruitment_duty": float(
            np.mean((active >= 0.5) & (sheet >= 0.5))),
        "population_rate_modulation": rhythm,
    }


def classify_stationary_branch(metrics, *, numerically_stable=True,
                               thresholds=None):
    """Classify one initial-condition arm without inferring bifurcation type."""
    limits = thresholds or BranchThresholds()
    if not numerically_stable:
        return {
            "label": "UNSTABLE",
            "checks": {"numerically_stable": False},
            "thresholds": asdict(limits),
        }
    low_checks = {
        "median_rate_le_low_limit": (
            float(metrics["median_rate_hz"]) <= limits.low_median_rate_hz),
        "active_fraction_lt_low_limit": (
            float(metrics["median_active_E_fraction_20ms"])
            < limits.low_active_fraction),
        "no_sustained_high_run": (
            float(metrics["longest_rate_ge_120_hz_ms"])
            < limits.low_max_sustained_high_run_ms),
    }
    high_checks = {
        "median_rate_ge_high_limit": (
            float(metrics["median_rate_hz"]) >= limits.high_median_rate_hz),
        "active_fraction_ge_high_limit": (
            float(metrics["median_active_E_fraction_20ms"])
            >= limits.high_active_fraction),
        "sheet_fraction_ge_high_limit": (
            float(metrics["median_recruited_sheet_fraction_1mm"])
            >= limits.high_sheet_fraction),
        "global_duty_ge_high_limit": (
            float(metrics["joint_global_recruitment_duty"]
                  ) >= limits.high_global_duty),
    }
    if all(low_checks.values()):
        label = "LOW"
    elif all(high_checks.values()):
        label = "TONIC_HIGH"
    else:
        label = "INTERMEDIATE"
    return {
        "label": label,
        "contract_version": "event_tolerant_low_v2",
        "checks": {
            "numerically_stable": True,
            "low": low_checks,
            "tonic_high": high_checks,
        },
        "thresholds": asdict(limits),
        "boundary": (
            "LOW permits isolated interictal-like events but rejects a rate "
            "at or above 120 Hz sustained for 100 ms; q95 is diagnostic only."),
    }


def classify_paired_initial_states(low_start_label, high_start_label):
    """Adjudicate the two arms at one q, eta_m and matched noise seed."""
    pair = (str(low_start_label), str(high_start_label))
    mapping = {
        ("LOW", "LOW"): "LOW_MONOSTABLE_CANDIDATE",
        ("TONIC_HIGH", "TONIC_HIGH"): "HIGH_MONOSTABLE_CANDIDATE",
        ("LOW", "TONIC_HIGH"): "BISTABLE_CANDIDATE",
        ("TONIC_HIGH", "LOW"): "REVERSE_SPLIT",
    }
    return mapping.get(pair, "MIXED_OR_UNRESOLVED")


def adjudicate_seed_family(pair_labels, *, minimum_seeds=3):
    """Require a complete prospective denominator before a robust label."""
    labels = [str(value) for value in pair_labels]
    counts = {label: labels.count(label) for label in sorted(set(labels))}
    if len(labels) < int(minimum_seeds):
        verdict = "INCOMPLETE_SEED_DENOMINATOR"
    elif counts.get("BISTABLE_CANDIDATE", 0) == len(labels):
        verdict = "ROBUST_SNN_BISTABILITY_CANDIDATE"
    elif counts.get("BISTABLE_CANDIDATE", 0) >= 2:
        verdict = "STOCHASTIC_OR_METASTABLE_BISTABILITY_CANDIDATE"
    elif counts.get("LOW_MONOSTABLE_CANDIDATE", 0) == len(labels):
        verdict = "LOW_MONOSTABLE_CANDIDATE"
    elif counts.get("HIGH_MONOSTABLE_CANDIDATE", 0) == len(labels):
        verdict = "HIGH_MONOSTABLE_CANDIDATE"
    else:
        verdict = "MIXED_OR_UNRESOLVED"
    return {
        "verdict": verdict,
        "n_seeds": len(labels),
        "minimum_seeds": int(minimum_seeds),
        "counts": counts,
        "boundary": (
            "finite stochastic SNN evidence; never a mathematical "
            "bifurcation label by itself"),
    }


def scientific_contract_digest(record):
    """Identity invariant to coordinates and additive stage amendments."""
    hybrid = dict(record["hybrid_config"])
    for coordinate in ("q_init", "q_min", "eta_m"):
        hybrid.pop(coordinate, None)
    source_hashes = {
        name: audit["observed_sha256"]
        for name, audit in record["source_audit"].items()
        if isinstance(audit, dict) and "observed_sha256" in audit
    }
    payload = {
        "schema_version": record["schema_version"],
        "source_hashes": source_hashes,
        "substrate_seed": record["source_trajectory"]["substrate_seed"],
        "candidate_id": record["source_trajectory"]["candidate_id"],
        "full_edge_contract": record["full_edge_contract"],
        "hybrid_noncoordinate_config": hybrid,
        "applied_spatial_ou": record["applied_spatial_ou"],
        "duration_ms": record["simulation"]["duration_ms"],
        "burn_in_ms": record["simulation"]["burn_in_ms"],
        "dt_ms": record["simulation"]["dt_ms"],
        "classification_contract_version": record["classification"].get(
            "contract_version", "q95_low_v1"),
        "classification_thresholds": record["classification"]["thresholds"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest(), payload
