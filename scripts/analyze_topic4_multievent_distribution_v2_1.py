#!/usr/bin/env python3
"""Score G1 with D_off and freeze the first of two deferred DE batches."""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_xy_research as base
from scripts.run_topic4_multievent_distribution_v2_1 import (
    OUT, _prepare_execution, repaired_observation,
)
from src.topic4_interictal_repaired_evaluation import rank_features
from src.topic4_multievent_distribution_objective_v2_1 import OBJECTIVE_VERSION
from src.topic4_xy_search import audit_geometry, field_descriptor, geometry_allowed


PAYLOAD = OUT / "training_objective_v2_1.pkl"
EVALUATOR = ROOT / "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/evaluator.pkl"
NORMALIZERS = {"global": 0.0425770294722167,
               "balanced_modes": 0.21306191302831015}
BOUNDS = np.asarray([
    [0.75, 19.25], [0.75, 19.25], [0.75, 19.25], [0.75, 19.25],
    [0.5, 1.4], [0.8, 1.2], [0.8, 1.3], [0.8, 1.2],
    [12.0, 28.0], [-45.0, 45.0], [1.0, 3.0],
], dtype=float)


def _load_frozen():
    qualification = base.read(OUT / "objective_qualification.json")
    if (qualification["objective_version"] != OBJECTIVE_VERSION
            or qualification["normalizers"] != NORMALIZERS
            or base.sha(PAYLOAD) != qualification["payload_sha256"]):
        raise RuntimeError("D_off ranking payload or positive scale changed")
    with open(PAYLOAD, "rb") as stream:
        objective = pickle.load(stream)
    repaired = base.read(EVALUATOR.parent / "qualification.json")
    if base.sha(EVALUATOR) != repaired["evaluator_sha256"]:
        raise RuntimeError("frozen patient evaluator changed")
    with open(EVALUATOR, "rb") as stream:
        evaluator = pickle.load(stream)
    return objective, evaluator, qualification


def _unit(folder, candidate_id, topology_seed, dynamics_seed, objective, evaluator):
    stem = f"{candidate_id}_topo_{topology_seed}_dyn_{dynamics_seed}"
    worker_path = folder / "workers" / f"{stem}.json"
    if not worker_path.exists():
        return {"execution_status": "INCOMPLETE", "score": None}
    worker = base.read(worker_path)
    observation_json, observation_npz, observation = repaired_observation(worker_path)
    with np.load(observation_npz) as arrays:
        primary = np.asarray(arrays["primary_event_indices"], int)
        centroids = np.asarray(arrays["centroid_ms"], float)
        table = centroids[primary]
    score = objective.score_network(table)
    if worker["physical_status"] == "RUNAWAY":
        score = {**score, "status": "PHYSICAL_RUNAWAY_NO_INTERICTAL_RANK",
                 "loss_off": None, "loss_D16": None,
                 "loss_off_A_component": None,
                 "loss_off_B_subtraction": None}
    labels, support, distance = evaluator.classify(table)
    return {
        "execution_status": worker["execution_status"],
        "physical_status": worker["physical_status"],
        "actual_duration_ms": worker["simulation"]["actual_duration_ms"],
        "n_detected_windows": observation["n_detected_windows"],
        "n_primary_events": len(table),
        "mode_counts": np.bincount(labels[labels >= 0], minlength=evaluator.k).astype(int).tolist(),
        "mode_supported_counts": [int(np.sum((labels == mode) & (support == 1)))
                                  for mode in range(evaluator.k)],
        "score": score,
        "worker_path": str(worker_path), "worker_sha256": base.sha(worker_path),
        "arrays_sha256": worker["arrays"]["sha256"],
        "repaired_observation_path": str(observation_json),
        "repaired_observation_sha256": base.sha(observation_json),
        "static_array_identity": worker["static_array_identity"],
        "mean_patient_neighborhood_distance": (
            float(np.nanmean(distance)) if len(distance) else None
        ),
        "training_table": table,
    }


def score_candidates(folder, candidates, seed_pairs, *, output_path):
    objective, evaluator, qualification = _load_frozen()
    results = []
    for candidate in candidates:
        units = {}
        tables = {}
        for topology_seed, dynamics_seed in seed_pairs:
            key = f"topo_{topology_seed}_dyn_{dynamics_seed}"
            unit = _unit(folder, candidate["candidate_id"], topology_seed,
                         dynamics_seed, objective, evaluator)
            table = unit.pop("training_table", None)
            units[key] = unit
            if table is not None:
                tables[key] = table
        complete = all(unit["execution_status"] == "COMPLETE"
                       for unit in units.values())
        no_runaway = all(unit.get("physical_status") != "RUNAWAY"
                         for unit in units.values())
        candidate_score = objective.score_candidate(tables) if complete else {
            "status": "INCOMPLETE_EXECUTION", "loss_off": None,
            "loss_D16": None, "loss_off_A_component": None,
            "loss_off_B_subtraction": None, "per_network": {},
        }
        if not no_runaway:
            candidate_score.update({
                "status": "PHYSICAL_RUNAWAY_NO_INTERICTAL_RANK",
                "loss_off": None, "loss_D16": None,
                "loss_off_A_component": None,
                "loss_off_B_subtraction": None,
            })
        pooled = np.concatenate(list(tables.values())) if tables else np.empty((0, 15))
        pooled_units = np.concatenate([
            np.full(len(table), index, int)
            for index, table in enumerate(tables.values())
        ]) if tables else np.empty(0, int)
        diagnostic = evaluator.metrics(pooled, pooled_units, detail=False)
        results.append({
            "candidate_id": candidate["candidate_id"],
            "population": candidate["population"],
            "proposal": candidate["proposal"],
            "node_field": candidate["node_field"],
            "parameters": vector_from_candidate(candidate).tolist(),
            "units": units,
            "score": candidate_score,
            "round1_pooled_joint_diagnostic": diagnostic["joint_distance"],
            "round1_pooled_joint_aggregation": (
                "pooled repaired events across units; diagnostic only; unlike equal-unit L_off"
            ),
            "ranking_eligible": bool(
                complete and no_runaway and candidate_score["loss_off"] is not None
            ),
        })
    ranking = {
        str(population): [row["candidate_id"] for row in sorted(
            [item for item in results
             if item["population"] == population and item["ranking_eligible"]],
            key=lambda item: (item["score"]["loss_off"], item["candidate_id"]),
        )]
        for population in (0, 1)
    }
    report = {
        "status": "SCORED_WITH_FROZEN_D_OFF",
        "objective_version": OBJECTIVE_VERSION,
        "objective_qualification_sha256": base.sha(OUT / "objective_qualification.json"),
        "normalizers": NORMALIZERS,
        "unit_aggregation": "equal weight no event pooling",
        "minimum_events_per_unit": 16,
        "partial_unit_average": False,
        "candidates": results,
        "ranking_by_population": ranking,
    }
    base.write(output_path, report)
    return report


def review_offline_scans():
    path = OUT / "offline_diagnostics.json"
    diagnostics = base.read(path)
    constant_checks = []
    for row in diagnostics["constant_binomial_enumeration"]:
        best = min(row["curve"], key=lambda item: (
            item["expected_D_off"], abs(item["q"] - row["patient_mode0_probability_p"])
        ))
        constant_checks.append(bool(np.isclose(
            best["q"], row["patient_mode0_probability_p"], atol=1e-12,
        )))
    temporal = diagnostics["within_mode_temporal_spread_scan"]
    temporal_checks = []
    for n in (16, 32, 64):
        rows = {item["scale"]: item for item in temporal if item["N"] == n}
        center = rows[1.0]["L_off"]["median"]
        temporal_checks.append(
            center < rows[0.0]["L_off"]["median"]
            and center < rows[1.5]["L_off"]["median"]
        )
    result = {
        "status": "OFFLINE_SCANS_SUPPORT_G2" if all(constant_checks + temporal_checks)
        else "OFFLINE_SCAN_SYSTEMATIC_DEFECT_G2_PAUSED",
        "constant_D_off_minimum_at_q_equal_p": all(constant_checks),
        "temporal_patient_spread_beats_collapsed_and_1p5": all(temporal_checks),
        "full_feature_minima_not_used_as_hard_per_draw_gate": True,
        "diagnostics_sha256": base.sha(path),
        "weights_or_scales_changed_from_scans": False,
    }
    result["G2_allowed"] = result["status"] == "OFFLINE_SCANS_SUPPORT_G2"
    base.write(OUT / "offline_diagnostics_review.json", result)
    return result


def vector_from_candidate(candidate):
    centers = np.asarray(candidate["node_field"]["centers_mm"], float).reshape(-1)
    dynamic = candidate["dynamic_parameters"]
    return np.r_[
        centers,
        float(candidate["node_mapping"]["node_gain"]),
        float(dynamic["E_to_E_weight_scale"]),
        float(dynamic["E_to_I_weight_scale"]),
        float(dynamic["I_to_E_weight_scale"]),
        float(dynamic["tau_d_GABA_ms"]),
        float(candidate["mechanisms"]["ellipse_angle_deg"] - base.THETA),
        float(candidate["mechanisms"]["ellipse_aspect_ratio"]),
    ]


def _reflect_unit(values):
    phase = np.mod(values, 2.0)
    return np.where(phase <= 1.0, phase, 2.0 - phase)


def _offspring_row(target, vector, candidate_id, plan, positions):
    row = copy.deepcopy(target)
    centers = vector[:4].reshape(2, 2)
    row.update({
        "candidate_id": candidate_id,
        "proposal": "deferred_DE_rand_1_bin_v2_1",
        "parent_target_candidate_id": target["candidate_id"],
        "node_field": field_descriptor(centers, 1499),
        "geometry": audit_geometry(positions, centers, 1499),
        "de_plan_record": plan,
    })
    row["node_mapping"]["node_gain"] = float(vector[4])
    row["dynamic_parameters"].update({
        "E_to_E_weight_scale": float(vector[5]),
        "E_to_I_weight_scale": float(vector[6]),
        "I_to_E_weight_scale": float(vector[7]),
        "I_to_I_weight_scale": 1.0,
        "tau_d_GABA_ms": float(vector[8]),
    })
    row["mechanisms"].update({
        "g_EE": 0.0, "g_EtoI": 0.0, "Z_M": "off",
        "ellipse_angle_deg": float(base.THETA + vector[9]),
        "ellipse_aspect_ratio": float(vector[10]),
        "ellipse_reference_angle_deg": float(base.THETA),
        "ellipse_reference_aspect_ratio": 2.0,
    })
    return row


def freeze_de_batch(score_report, candidates, *, batch="A",
                    parent_score_path=None, parent_population_path=None):
    """Freeze one four-child-per-population deferred DE batch.

    Batch A uses the G1 population.  Batch B is called only after batch A has
    been scored and its delayed replacements have produced a new population.
    """
    review = review_offline_scans()
    if not review["G2_allowed"]:
        raise RuntimeError("offline scan found a systematic target defect; G2 paused")
    batch = str(batch).upper()
    if batch not in {"A", "B"}:
        raise ValueError("G2 batch must be A or B")
    if parent_score_path is None:
        parent_score_path = OUT / ("g1_scores.json" if batch == "A"
                                   else "g2a_pool_scores.json")
    if parent_population_path is None:
        parent_population_path = (
            OUT / "execution/training_24s/candidate_manifest.json"
            if batch == "A" else OUT / "g2a_updated_population.json"
        )
    path = OUT / f"g2{batch.lower()}_deferred_batch_plan.json"
    rows_path = OUT / f"g2{batch.lower()}_offspring_candidates.json"
    if path.exists():
        plan = base.read(path)
        if (plan["parent_score_sha256"] != base.sha(parent_score_path)
                or plan["parent_population_sha256"] != base.sha(parent_population_path)
                or plan["objective_qualification_sha256"]
                != base.sha(OUT / "objective_qualification.json")):
            raise RuntimeError("persisted G2 batch no longer matches frozen ranking")
        rows_record = base.read(rows_path)
        if rows_record.get("plan_sha256") != base.sha(path):
            raise RuntimeError("persisted G2 offspring no longer match frozen plan")
        expected_ids = [row["candidate_id"] for row in plan["plans"]]
        observed_ids = [row["candidate_id"] for row in rows_record["candidates"]]
        if observed_ids != expected_ids:
            raise RuntimeError("persisted G2 offspring order/identity changed")
        return plan, rows_record["candidates"]
    positions = base.positions()
    by_population = {
        population: [row for row in candidates if row["population"] == population]
        for population in (0, 1)
    }
    rng_seed = 3854897931 ^ (0xD0FF21 if batch == "A" else 0xD0FF22)
    rng = np.random.default_rng(rng_seed)
    all_plans, offspring = [], []
    for population in (0, 1):
        parents = by_population[population]
        vectors = np.asarray([vector_from_candidate(row) for row in parents])
        normalized_vectors = (
            (vectors - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0])
        )
        targets = rng.choice(len(parents), 4, replace=False)
        for local_child, target_index in enumerate(targets):
            attempts = []
            for attempt in range(10000):
                donor_indices = rng.choice(
                    np.delete(np.arange(len(parents)), target_index),
                    3, replace=False,
                )
                crossover_uniform = rng.random(vectors.shape[1])
                forced_index = int(rng.integers(vectors.shape[1]))
                mask = crossover_uniform < 0.8
                mask[forced_index] = True
                mutant_normalized = (
                    normalized_vectors[donor_indices[0]]
                    + 0.6 * (
                        normalized_vectors[donor_indices[1]]
                        - normalized_vectors[donor_indices[2]]
                    )
                )
                raw_normalized = np.where(
                    mask, mutant_normalized, normalized_vectors[target_index]
                )
                reflected_normalized = _reflect_unit(raw_normalized)
                reflected = (
                    BOUNDS[:, 0]
                    + reflected_normalized * (BOUNDS[:, 1] - BOUNDS[:, 0])
                )
                raw = BOUNDS[:, 0] + raw_normalized * (
                    BOUNDS[:, 1] - BOUNDS[:, 0]
                )
                valid = geometry_allowed(
                    reflected[:4].reshape(2, 2), positions,
                    domain="whole_sheet", target_count=1499,
                )
                attempt_record = {
                    "attempt": attempt,
                    "donor_indices": donor_indices.astype(int).tolist(),
                    "donor_candidate_ids": [parents[index]["candidate_id"]
                                            for index in donor_indices],
                    "crossover_uniform": crossover_uniform.tolist(),
                    "forced_mutant_dimension": forced_index,
                    "crossover_mask": mask.astype(bool).tolist(),
                    "mutant_normalized": mutant_normalized.tolist(),
                    "raw_normalized": raw_normalized.tolist(),
                    "reflected_normalized": reflected_normalized.tolist(),
                    "raw_vector": raw.tolist(),
                    "reflected_vector": reflected.tolist(),
                    "geometry_valid": bool(valid),
                }
                attempts.append(attempt_record)
                if valid:
                    break
            else:
                raise RuntimeError("DE geometric rejection exhausted")
            candidate_id = (
                f"v2_1_pop{population}_de_{batch.lower()}_{local_child:03d}"
            )
            child_plan = {
                "candidate_id": candidate_id, "population": population,
                "target_index": int(target_index),
                "target_candidate_id": parents[target_index]["candidate_id"],
                "F": 0.6, "CR": 0.8,
                "accepted_attempt_index": len(attempts) - 1,
                "proposal_attempts": attempts,
            }
            all_plans.append(child_plan)
            offspring.append(_offspring_row(
                parents[target_index], reflected, candidate_id,
                child_plan, positions,
            ))
    payload = {
        "status": f"G2{batch}_DEFERRED_BATCH_FROZEN_BEFORE_DISPATCH",
        "batch": batch,
        "objective_version": OBJECTIVE_VERSION,
        "objective_qualification_sha256": base.sha(OUT / "objective_qualification.json"),
        "parent_score_path": str(parent_score_path),
        "parent_score_sha256": base.sha(parent_score_path),
        "parent_population_path": str(parent_population_path),
        "parent_population_sha256": base.sha(parent_population_path),
        "parent_snapshot": [{
            "candidate_id": row["candidate_id"], "population": row["population"],
            "vector": vector_from_candidate(row).tolist(),
            "L_off": next(item for item in score_report["candidates"]
                          if item["candidate_id"] == row["candidate_id"])["score"]["loss_off"],
        } for row in candidates],
        "rng_seed": int(rng_seed),
        "rng_seed_expression": (
            f"3854897931 XOR {'0xD0FF21' if batch == 'A' else '0xD0FF22'}"
        ),
        "method": "DE/rand/1/bin normalized physical coordinates with reflected bounds",
        "whole_batch_deferred_update": True,
        "completion_order_cannot_change_proposals": True,
        "offspring_per_population": 4,
        "second_batch_uses_post_G2A_population": batch == "B",
        "feedback_path_guarantees_replacement_or_improvement": False,
        "scipy_polish_or_extra_local_evaluations": False,
        "plans": all_plans,
    }
    base.write(path, payload)
    base.write(rows_path, {
        "status": f"G2{batch}_OFFSPRING_FROZEN_BEFORE_DISPATCH",
        "plan_sha256": base.sha(path), "candidates": offspring,
    })
    return payload, offspring


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    folder, _, manifest_path, _ = _prepare_execution()
    candidates = base.read(manifest_path)["candidates"]
    expected = len(candidates) * 2
    existing = list((folder / "workers").glob("*.json"))
    if len(existing) != expected:
        raise RuntimeError(f"G1 incomplete: {len(existing)}/{expected} worker JSONs")
    from scripts.audit_topic4_multievent_execution_parameters_v2_1 import (
        audit as audit_execution_parameters,
    )
    parameter_audit = audit_execution_parameters(require_complete=True)
    if parameter_audit["status"] != "PARAMETER_APPLICATION_AUDIT_PASS":
        raise RuntimeError("actual executor-value audit failed; G2 must not start")
    g1 = score_candidates(
        folder, candidates, [(2511, 2511), (2512, 2512)],
        output_path=OUT / "g1_scores.json",
    )
    review_offline_scans()
    if not args.score_only:
        base.write(OUT / "g2_ranking_freeze.json", {
            "status": "G2_RANKING_AND_TWO_BATCH_SEARCH_FROZEN_BEFORE_G2A",
            "objective_version": OBJECTIVE_VERSION,
            "objective_qualification_sha256": base.sha(
                OUT / "objective_qualification.json"
            ),
            "parameter_application_audit_sha256": base.sha(
                OUT / "parameter_application_audit.json"
            ),
            "G1_scores_sha256": base.sha(OUT / "g1_scores.json"),
            "normalizers": NORMALIZERS,
            "unit_aggregation": "equal weight no event pooling",
            "minimum_events_per_unit": 16,
            "missing_score_semantics": "null; no partial unit average",
            "batches": [
                {"name": "G2-A", "offspring_per_population": 4,
                 "parent_population": "G1"},
                {"name": "G2-B", "offspring_per_population": 4,
                 "parent_population": "post-G2-A delayed replacement"},
            ],
            "total_offspring_conditions": 16,
            "validation_products_used_for_ranking": False,
            "A_B_decomposition_is_diagnostic_only": True,
        })
        freeze_de_batch(g1, candidates, batch="A")
    base.write(OUT / "g1_scientific_checkpoint.json", {
        "status": "G1_SCORED_G2A_BATCH_FROZEN" if not args.score_only
        else "G1_SCORED",
        "completed_candidates": len(g1["candidates"]),
        "estimable_candidates": int(sum(
            row["ranking_eligible"] for row in g1["candidates"]
        )),
        "G1_scores_sha256": base.sha(OUT / "g1_scores.json"),
        "offline_review_sha256": base.sha(OUT / "offline_diagnostics_review.json"),
        "parameter_application_audit_sha256": base.sha(
            OUT / "parameter_application_audit.json"
        ),
        "validation_products_used_for_ranking": False,
    })
    print(json.dumps(base.read(OUT / "g1_scientific_checkpoint.json"), indent=2))


if __name__ == "__main__":
    main()
