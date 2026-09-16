#!/usr/bin/env python3
"""Build the isolated D_off payload and the two prespecified offline scans."""
from __future__ import annotations

import itertools
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_xy_research as base
from src.topic4_interictal_repaired_evaluation import rank_features
from src.topic4_multievent_distribution_objective_v2_1 import (
    OBJECTIVE_VERSION,
    MultieventDistributionObjectiveV21,
    a_b_from_matched_and_off_diagonal,
    explicit_off_diagonal_mean_distance,
    off_diagonal_mean_distance,
)
from src.topic4_multievent_distribution_objective import matched_mean_distance


CONFIG = ROOT / "config/topic4_multievent_distribution_search_v2_1.json"
OUT = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2_1"
EVALUATOR_ROOT = ROOT / "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2"
OLD = ROOT / "results/topic4_sef_hfo/multievent_distribution_search_v2"
NORMALIZERS = {
    "global": 0.0425770294722167,
    "balanced_modes": 0.21306191302831015,
}


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q025": float(np.quantile(values, 0.025)),
        "q975": float(np.quantile(values, 0.975)),
    }


def _scores(obj, phi, labels):
    balanced = obj.balanced_embedding(phi, labels)
    raw_off = {
        "global": off_diagonal_mean_distance(phi, obj.target_global),
        "balanced_modes": off_diagonal_mean_distance(
            balanced, obj.target_modes),
    }
    raw_d16 = {
        "global": matched_mean_distance(phi, obj.target_global, 16),
        "balanced_modes": matched_mean_distance(
            balanced, obj.target_modes, 16),
    }
    def total(raw):
        return float(
            0.5 * raw["global"] / NORMALIZERS["global"]
            + 0.5 * raw["balanced_modes"] / NORMALIZERS["balanced_modes"]
        )
    return {"D_off": total(raw_off), "D16": total(raw_d16),
            "raw_D_off": raw_off, "raw_D16": raw_d16}


def constant_binomial_scan(patient_p):
    rows = []
    probabilities = sorted(set([0.02, 0.1, 0.2, float(patient_p)]))
    for p in probabilities:
        # Include a dense fixed grid, the patient q, and the analytic D16
        # constant-coordinate optimum obtained by direct grid minimization.
        q_grid = np.unique(np.r_[np.linspace(0, 1, 201), p])
        for n in (16, 32, 64):
            curves = []
            for q in q_grid:
                off_values, d16_values, weights = [], [], []
                target = np.asarray([1.0, 1.0]) / np.sqrt(2.0)
                for count0 in range(n + 1):
                    labels = np.r_[np.zeros(count0, int), np.ones(n - count0, int)]
                    x = np.column_stack([
                        (labels == 0) / p,
                        (labels == 1) / (1.0 - p),
                    ]) / np.sqrt(2.0)
                    off_values.append(off_diagonal_mean_distance(x, target))
                    d16_values.append(matched_mean_distance(x, target, 16))
                    weights.append(binom.pmf(count0, n, q))
                weights = np.asarray(weights)
                curves.append({
                    "q": float(q),
                    "expected_D_off": float(np.dot(weights, off_values)),
                    "expected_D16": float(np.dot(weights, d16_values)),
                })
            min_d16 = min(curves, key=lambda row: (row["expected_D16"], row["q"]))
            rows.append({
                "patient_mode0_probability_p": p,
                "N": n,
                "q_equal_p_included": bool(np.any(np.isclose(q_grid, p))),
                "D16_grid_minimum_q": min_d16["q"],
                "curve": curves,
            })
    return rows


def full_feature_mixture_scan(obj, ev, rng):
    phi = obj.embedding(ev.fit)
    labels = np.asarray(ev.fit_labels, int)
    by_mode = [np.flatnonzero(labels == mode) for mode in range(obj.k)]
    p = float(obj.proportions[0])
    q_grid = np.unique(np.r_[np.linspace(0.02, 0.98, 49), p])
    rows = []
    for n in (16, 32, 64):
        for q in q_grid:
            off, d16, counts = [], [], []
            for _ in range(128):
                n0 = int(rng.binomial(n, q))
                selected = np.r_[
                    rng.choice(by_mode[0], n0, replace=True),
                    rng.choice(by_mode[1], n - n0, replace=True),
                ]
                draw_labels = labels[selected]
                score = _scores(obj, phi[selected], draw_labels)
                off.append(score["D_off"])
                d16.append(score["D16"])
                counts.append(n0)
            rows.append({
                "N": n, "q_mode0": float(q),
                "n_mode0": _summary(counts),
                "L_off": _summary(off), "L_D16": _summary(d16),
                "paired_same_draws": True,
            })
    return rows


def _temporal_scale(table, labels, scale):
    relative = table - np.nanmin(table, axis=1)[:, None]
    output = relative.copy()
    centers = []
    unsupported = []
    for mode in np.unique(labels):
        selected = labels == mode
        center = np.nanmedian(relative[selected], axis=0)
        centers.append(center.tolist())
        unsupported.append(np.flatnonzero(~np.isfinite(center)).astype(int).tolist())
        output[selected] = center + scale * (relative[selected] - center)
    output[~np.isfinite(table)] = np.nan
    return output, centers, unsupported


def temporal_spread_scan(obj, ev, rng):
    source_labels = np.asarray(ev.fit_labels, int)
    rows = []
    for scale in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5):
        transformed, centers, unsupported = _temporal_scale(
            ev.fit, source_labels, scale,
        )
        predicted = obj.km.predict(rank_features(transformed))
        phi = obj.embedding(transformed)
        changed = predicted != source_labels
        for n in (16, 32, 64):
            off, d16, changes = [], [], []
            for _ in range(128):
                selected = rng.choice(len(transformed), n, replace=True)
                score = _scores(obj, phi[selected], predicted[selected])
                off.append(score["D_off"])
                d16.append(score["D16"])
                changes.append(float(np.mean(changed[selected])))
            rows.append({
                "scale": scale, "N": n,
                "L_off": _summary(off), "L_D16": _summary(d16),
                "label_change_fraction_resampled": _summary(changes),
                "whole_FIT_label_change_fraction": float(np.mean(changed)),
                "participation_mask_preserved": bool(np.array_equal(
                    np.isfinite(transformed), np.isfinite(ev.fit))),
                "mode_channel_relative_time_centers_ms": centers,
                "unsupported_center_contacts_by_mode": unsupported,
            })
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = base.read(CONFIG)
    qualification = base.read(EVALUATOR_ROOT / "qualification.json")
    evaluator_path = EVALUATOR_ROOT / "evaluator.pkl"
    if base.sha(evaluator_path) != qualification["evaluator_sha256"]:
        raise RuntimeError("frozen repaired evaluator changed")
    if base.sha(OLD / "initial_candidate_manifest.json") != config["inheritance"]["initial_candidate_manifest"]["sha256"]:
        raise RuntimeError("inherited 48-candidate manifest changed")
    old_loss = base.read(OLD / "loss_qualification.json")
    if old_loss["patient_only_normalizers"] != NORMALIZERS:
        raise RuntimeError("frozen positive normalizers changed")
    with open(evaluator_path, "rb") as stream:
        ev = pickle.load(stream)
    obj = MultieventDistributionObjectiveV21(
        ev, matched_count=config["training"]["minimum_events_for_ranking_per_unit"],
        normalizers=NORMALIZERS,
    )

    rng = np.random.default_rng(202609070211)
    toy = rng.normal(size=(19, 7))
    target = rng.normal(size=7)
    direct = off_diagonal_mean_distance(toy, target)
    explicit = explicit_off_diagonal_mean_distance(toy, target)
    d16 = matched_mean_distance(toy, target, 16)
    a_recovered, b_recovered = a_b_from_matched_and_off_diagonal(
        d16, direct, len(toy), 16,
    )
    toy_mean = toy.mean(axis=0)
    a_direct = float(np.sum((toy_mean - target) ** 2))
    b_direct = float(
        np.mean(np.sum((toy - toy_mean) ** 2, axis=1)) / (len(toy) - 1)
    )
    negative = off_diagonal_mean_distance(
        np.asarray([[-1.0], [1.0]]), np.asarray([0.0]),
    )
    serialized_negative = json.loads(json.dumps({"value": negative}))["value"]
    one_good = ev.fit[:16]
    missing = obj.score_candidate({2511: one_good, 2512: ev.fit[:15]})
    checks = {
        "D_off_matches_explicit_i_ne_j": bool(np.isclose(direct, explicit, atol=1e-12)),
        "A_B_recovers_mean_distance_and_finite_event_subtraction_per_run": bool(
            np.isclose(a_recovered, a_direct, atol=1e-12)
            and np.isclose(b_recovered, b_direct, atol=1e-12)
            and np.isclose(direct, a_recovered - b_recovered, atol=1e-12)
        ),
        "negative_value_retained_after_json": serialized_negative == -1.0,
        "negative_value_sorts_before_positive": sorted([0.25, serialized_negative])[0] == -1.0,
        "positive_normalizers_exactly_inherited": obj.normalizers == NORMALIZERS,
        "missing_unit_forbids_partial_average": (
            missing["loss_off"] is None and missing["status"] == "INSUFFICIENT_EVENTS"
        ),
        "payload_has_no_patient_event_table": not any(
            key in vars(obj) for key in ("patient", "fit", "cal", "index", "partition", "cache_probe")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"v2.1 objective qualification failed: {checks}")
    payload_path = OUT / "training_objective_v2_1.pkl"
    with open(payload_path, "wb") as stream:
        pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
    base.write(OUT / "objective_qualification.json", {
        "status": "D_OFF_IMPLEMENTATION_VERIFIED",
        "objective_version": OBJECTIVE_VERSION,
        "checks": checks,
        "normalizers": NORMALIZERS,
        "mathematical_minimum_events": 2,
        "ranking_minimum_events_per_unit": 16,
        "payload_path": str(payload_path),
        "payload_sha256": base.sha(payload_path),
        "old_payload_unchanged_sha256": base.sha(OLD / "training_objective.pkl"),
    })

    diagnostics = {
        "status": "OFFLINE_SCANS_COMPLETE",
        "rng_seed": 202609070211,
        "resamples_per_point": 128,
        "normalizers_reused_not_refit": NORMALIZERS,
        "constant_binomial_enumeration": constant_binomial_scan(obj.proportions[0]),
        "full_feature_mixture_scan": full_feature_mixture_scan(obj, ev, rng),
        "within_mode_temporal_spread_scan": temporal_spread_scan(obj, ev, rng),
        "interpretation_limits": {
            "D_off": "off-diagonal empirical training statistic; no iid claim for continuous SNN events",
            "temporal_spread": "fixed-mask timing and rank variation only; not all biological variation",
            "curve_use": "diagnostic only; no weight or normalizer selected from these curves",
        },
    }
    base.write(OUT / "offline_diagnostics.json", diagnostics)
    g0_path = OUT / "g0_canary_audit.json"
    g0_pass = g0_path.exists() and base.read(g0_path).get("status") == "G0_CANARY_PASS"
    base.write(OUT / "preparation_v2_1.json", {
        "status": (
            "D_OFF_AND_OFFLINE_SCANS_COMPLETE_G0_ALREADY_PASS"
            if g0_pass else "D_OFF_AND_OFFLINE_SCANS_COMPLETE_PHYSICAL_CANARY_PENDING"
        ),
        "objective_qualification_sha256": base.sha(OUT / "objective_qualification.json"),
        "offline_diagnostics_sha256": base.sha(OUT / "offline_diagnostics.json"),
        "training_objective_sha256": base.sha(payload_path),
        "g0_canary_audit_sha256": base.sha(g0_path) if g0_pass else None,
        "simulations_dispatched": 0,
    })
    print(json.dumps({
        "status": "V2_1_OBJECTIVE_AND_SCANS_PREPARED",
        "checks": checks,
        "diagnostic_rows": {
            "constant": len(diagnostics["constant_binomial_enumeration"]),
            "mixture": len(diagnostics["full_feature_mixture_scan"]),
            "temporal": len(diagnostics["within_mode_temporal_spread_scan"]),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
