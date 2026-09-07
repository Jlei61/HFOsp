#!/usr/bin/env python3
"""Post-run executor-value audit for Topic 4 multievent v2.1.

The worker's requested/effective dictionaries have one origin and are retained
only as a declaration.  This audit instead reconstructs node arrays from the
candidate and topology, measures sparse-weight dose ratios, and checks the
frozen observer output.  GABA tau and absence of a kick are explicitly marked
as code-traced because they cannot be uniquely inverted from saved trajectories.
"""
from __future__ import annotations

import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_topic4_xy_research as base
from scripts.run_topic4_multievent_distribution_v2_1 import OUT, repaired_observation
from src.topic4_core_field_rev9 import reconstruct_node_from_h
from src.topic4_manual_dual_core import budget_matched_dual_core_h
from src.topic4_multievent_condition_identity_v2_1 import condition_key


EXECUTION = OUT / "execution/training_24s"
OBSERVER = ROOT / (
    "results/topic4_sef_hfo/multidimensional_interictal_observation_repair_v2/"
    "observation_contract.json"
)
WORKER_SOURCE = ROOT / "scripts/run_topic4_multidimensional_worker.py"
TRANSITION = ROOT / "config/topic4_rev22_dci_transition_execution.json"


def _close(left, right):
    return bool(np.isclose(float(left), float(right), rtol=1e-10, atol=1e-10))


def _ratio(pair, index):
    before = float(pair["before_sum"][index])
    after = float(pair["after_sum"][index])
    return after / before


@lru_cache(maxsize=None)
def _positions_for_topology(config_path, topology_seed):
    execution_config = base.read(Path(config_path))
    network = execution_config["corrected_networks"][str(topology_seed)]
    if base.sha(network["path"]) != network["sha256"]:
        raise RuntimeError("corrected topology cache changed")
    with open(network["path"], "rb") as stream:
        payload = pickle.load(stream)
    n_e, n_i = int(payload["NE"]), int(payload["NI"])
    positions = np.asarray(payload["net"]["pos"][:n_e], float).copy()
    return positions, n_e, n_i


@lru_cache(maxsize=1)
def _node_stage():
    transition = json.loads(TRANSITION.read_text())
    stage_record = transition["inputs"]["stage_config"]
    stage_path = ROOT / stage_record["path"]
    if base.sha(stage_path) != stage_record["sha256"]:
        raise RuntimeError("node stage config changed")
    return json.loads(stage_path.read_text())


def _node_expected(candidate, topology_seed, config_path):
    positions, n_e, n_i = _positions_for_topology(
        str(Path(config_path).resolve()), int(topology_seed)
    )
    field = candidate["node_field"]
    h, geometry = budget_matched_dual_core_h(
        positions, np.asarray(field["centers_mm"], float),
        target_count=int(field["target_count"]),
    )
    stage = _node_stage()
    engine = stage["engine"]
    mapping = candidate["node_mapping"]
    node = reconstruct_node_from_h(
        h, n_total=n_e + n_i, quantile_seed=stage["quantile_seed"],
        core_mean=engine["core_mean"], core_std=engine["core_std"],
        v_base=engine["v_base"],
        depth_shrinkage=float(mapping.get("signed_depth_shrinkage", 1.0)),
        node_gain=float(mapping.get("node_gain", 1.0)),
    )
    return node, geometry


def _parameter_rows(candidate, worker, arrays, node, geometry):
    expected_dynamic = candidate["dynamic_parameters"]
    sparse = worker["multidimensional_parameter_audit"]["sparse_pathways"]
    observed_scales = {
        "E_to_E_weight_scale": _ratio(sparse["ampa_by_delay"], 0),
        "E_to_I_weight_scale": _ratio(sparse["ampa_by_delay"], 1),
        "I_to_E_weight_scale": _ratio(sparse["gaba_by_delay"], 0),
        "I_to_I_weight_scale": _ratio(sparse["gaba_by_delay"], 1),
    }
    actual_h = np.asarray(arrays["h"], np.float32)
    actual_delta = np.asarray(arrays["delta_vtheta"], np.float32)
    actual_vtheta = np.asarray(arrays["vtheta"], np.float32)
    expected_h = np.asarray(node["h"], np.float32)
    expected_delta = np.asarray(node["delta_vtheta"], np.float32)
    expected_vtheta = np.asarray(node["vtheta"], np.float32)
    node_exact = bool(
        np.array_equal(actual_h, expected_h)
        and np.array_equal(actual_delta, expected_delta)
        and np.array_equal(actual_vtheta, expected_vtheta)
    )
    centers = np.asarray(candidate["node_field"]["centers_mm"], float).reshape(-1)
    rows = []
    for index, name in enumerate(("core_x1_mm", "core_y1_mm", "core_x2_mm", "core_y2_mm")):
        rows.append({
            "parameter": name, "expected": float(centers[index]),
            "observed": "deterministic expected-center field reproduces saved h/delta_vtheta/vtheta",
            "evidence": "independent_array_reconstruction", "pass": node_exact,
        })
    gain = float(candidate["node_mapping"]["node_gain"])
    rows.append({
        "parameter": "node_gain", "expected": gain,
        "observed": float(worker["node_mapping"]["node_gain"]),
        "evidence": "independent_array_reconstruction_and_mapping_budget",
        "pass": node_exact and _close(worker["node_mapping"]["node_gain"], gain)
        and _close(worker["node_mapping"]["gain_application_error"], 0.0),
    })
    for name in ("E_to_E_weight_scale", "E_to_I_weight_scale", "I_to_E_weight_scale"):
        rows.append({
            "parameter": name, "expected": float(expected_dynamic[name]),
            "observed": float(observed_scales[name]),
            "evidence": "measured_sparse_weight_sum_ratio_after_application",
            "pass": _close(observed_scales[name], expected_dynamic[name]),
        })
    tau = float(expected_dynamic["tau_d_GABA_ms"])
    rows.append({
        "parameter": "tau_d_GABA_ms", "expected": tau,
        "observed": float(worker["multidimensional_parameter_audit"]["effective"][
            "tau_d_GABA_ms"
        ]),
        "evidence": (
            "code_traced_assignment_to_substrate.params.tau_d_GABA_plus_worker_record; "
            "not_independently_inverted_from_trajectory"
        ),
        "pass": _close(
            worker["multidimensional_parameter_audit"]["effective"]["tau_d_GABA_ms"], tau
        ),
    })
    mechanisms = candidate["mechanisms"]
    ellipse = worker["mechanism_freeze"]["ellipse_audit"]
    for name, key in (("ellipse_angle_deg", "angle_deg"),
                      ("ellipse_aspect_ratio", "aspect_ratio")):
        rows.append({
            "parameter": name, "expected": float(mechanisms[name]),
            "observed": float(ellipse[key]),
            "evidence": "post_transform_sparse_weight_geometry_audit",
            "pass": _close(ellipse[key], mechanisms[name])
            and bool(ellipse["topology_unchanged"])
            and bool(ellipse["delay_assignment_unchanged"]),
        })
    extras = {
        "node_arrays_exact": node_exact,
        "geometry_selected_count_expected": int(geometry["selected_count"]),
        "geometry_selected_count_observed": int(worker["xy_geometry_audit"][
            "selected_count"
        ]),
        "I_to_I_expected": float(expected_dynamic["I_to_I_weight_scale"]),
        "I_to_I_observed": float(observed_scales["I_to_I_weight_scale"]),
    }
    return rows, extras


def audit(*, require_complete=False, execution=EXECUTION, seed_pairs=None,
          output_path=None):
    execution = Path(execution)
    manifest_path = execution / "candidate_manifest.json"
    config_path = execution / "execution_config.json"
    snapshot_path = execution / "runtime_snapshot.json"
    manifest = base.read(manifest_path)
    snapshot = base.read(snapshot_path)
    observer = base.read(OBSERVER)
    worker_text = WORKER_SOURCE.read_text()
    engine_text = (ROOT / "src/snn_engine/model.py").read_text()
    worker_hash_frozen = snapshot["source_hashes"].get(
        "scripts/run_topic4_multidimensional_worker.py"
    ) == base.sha(WORKER_SOURCE)
    code_checks = {
        "worker_source_matches_runtime_snapshot": worker_hash_frozen,
        "tau_assigned_after_parameter_application": (
            "substrate.params.tau_d_GABA = float(p['tau_d_GABA_ms'])"
            in (ROOT / "src/topic4_multidimensional_parameters.py").read_text()
        ),
        "engine_consumes_tau_in_GABA_exponential_decay": (
            "decay_II = np.exp(-dt / p.tau_d_GABA)" in engine_text
        ),
        "no_kick_or_slow_input": (
            "KICK_BOOST=0.0, t_kick=1e9" in worker_text and "slow=None" in worker_text
        ),
        "dynamics_rng_explicit": (
            "substrate.net[\"rng\"] = np.random.default_rng(dynamics_seed)" in worker_text
        ),
        "observer_burnin_is_500ms": _close(observer["burnin_ms"], 500.0),
        "model_event_specific_inputs_disabled": not bool(
            base.read(ROOT / "config/topic4_multievent_distribution_search_v2_1.json")[
                "fixed_model"
            ]["model_event_specific_inputs"]
        ),
    }
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    records = []
    failures = []
    for path in sorted((execution / "workers").glob("*.json")):
        worker = base.read(path)
        candidate = candidates[worker["candidate_id"]]
        node, geometry = _node_expected(
            candidate, int(worker["topology_seed"]), config_path,
        )
        with np.load(worker["arrays"]["path"]) as arrays:
            parameter_rows, extras = _parameter_rows(
                candidate, worker, arrays, node, geometry,
            )
            seed_check = bool(
                int(arrays["topology_seed"]) == int(worker["topology_seed"])
                and int(arrays["dynamics_seed"]) == int(worker["dynamics_seed"])
            )
            array_durations_ms = {
                "active_fraction": float(
                    len(arrays["active_fraction"])
                    * float(arrays["active_fraction_bin_ms"])
                ),
                "contact_envelope": float(
                    arrays["contact_envelope"].shape[1]
                    * float(arrays["contact_envelope_dt_ms"])
                ),
                "sheet_activity": float(
                    len(arrays["sheet_activity_counts"])
                    * float(arrays["sheet_activity_frame_ms"])
                ),
            }
            extras["array_durations_ms"] = array_durations_ms
            actual_duration_ms = float(worker["simulation"]["actual_duration_ms"])
            axis_checks = {
                "active_fraction_floor_bins": (
                    0.0 <= actual_duration_ms - array_durations_ms["active_fraction"]
                    < float(arrays["active_fraction_bin_ms"]) + 1e-9
                ),
                "contact_envelope_floor_bins": (
                    0.0 <= actual_duration_ms - array_durations_ms["contact_envelope"]
                    < float(arrays["contact_envelope_dt_ms"]) + 1e-9
                ),
                "sheet_activity_ceil_bins": (
                    -1e-9 <= array_durations_ms["sheet_activity"] - actual_duration_ms
                    <= float(arrays["sheet_activity_frame_ms"]) + 1e-9
                ),
            }
            extras["array_time_axis_contract_checks"] = axis_checks
        observation_json, _, observation = repaired_observation(path)
        burnin_check = all(
            float(event["window_ms"][0]) >= float(observer["burnin_ms"])
            for event in observation["events"]
        )
        fixed_checks = {
            "eleven_parameters_pass": all(row["pass"] for row in parameter_rows),
            "I_to_I_fixed_and_measured": _close(
                extras["I_to_I_observed"], extras["I_to_I_expected"]
            ) and _close(extras["I_to_I_expected"], 1.0),
            "actual_duration_or_explicit_runaway": (
                _close(worker["simulation"]["actual_duration_ms"], 24000.0)
                or worker["physical_status"] == "RUNAWAY"
            ),
            "actual_duration_matches_saved_floor_or_ceil_bin_contracts": all(
                axis_checks.values()
            ),
            "explicit_seed_arrays_match": seed_check,
            "Z_M_off": worker["mechanism_freeze"]["Z_M"] == "off",
            "learned_coefficients_zero": bool(
                worker["mechanism_freeze"]["edge_coefficients_all_zero"]
            ),
            "observer_excludes_pre_burnin_windows": burnin_check,
            "training_uses_frozen_repaired_observer": (
                observation["training_table_source"]
                == "centroid_ms at frozen primary_event_indices"
                and not observation["worker_lineage_onsets_used_for_training"]
            ),
            "worker_and_arrays_hashes_match": (
                base.sha(worker["arrays"]["path"]) == worker["arrays"]["sha256"]
            ),
        }
        passed = all(fixed_checks.values()) and all(code_checks.values())
        if not passed:
            failures.append(path.stem)
        records.append({
            "candidate_id": worker["candidate_id"],
            "topology_seed": worker["topology_seed"],
            "dynamics_seed": worker["dynamics_seed"],
            "physical_status": worker["physical_status"],
            "worker_path": str(path), "worker_sha256": base.sha(path),
            "observation_path": str(observation_json),
            "parameters_11d": parameter_rows,
            "fixed_contract_checks": fixed_checks,
            "requested_equals_effective_declaration_only": (
                worker["multidimensional_parameter_audit"]["requested"]
                == worker["multidimensional_parameter_audit"]["effective"]
            ),
            "array_and_geometry_details": extras,
            "pass": passed,
        })
    if seed_pairs is None:
        seed_pairs = [(2511, 2511), (2512, 2512)]
    seed_pairs = [tuple(map(int, pair)) for pair in seed_pairs]
    expected = len(candidates) * len(seed_pairs)
    observed_units = {
        (row["candidate_id"], int(row["topology_seed"]), int(row["dynamics_seed"]))
        for row in records
    }
    expected_units = {
        (candidate_id, topology, dynamics)
        for candidate_id in candidates
        for topology, dynamics in seed_pairs
    }
    missing_units = sorted(expected_units - observed_units)
    unexpected_units = sorted(observed_units - expected_units)
    complete = len(records) == expected
    complete = complete and not missing_units and not unexpected_units
    if require_complete and not complete:
        raise RuntimeError(f"parameter audit requires complete G1: {len(records)}/{expected}")
    baseline_id = "v2_anchor_historical__baseline"
    tau_id = "v2_anchor_historical__tau_d_GABA_ms_high"
    tau_regression_applicable = baseline_id in candidates and tau_id in candidates
    tau_condition_distinct = (
        condition_key(candidates[baseline_id]) != condition_key(candidates[tau_id])
        if tau_regression_applicable else None
    )
    static_collision = []
    record_lookup = {
        (row["candidate_id"], row["topology_seed"]): row for row in records
    }
    for topology in sorted({pair[0] for pair in seed_pairs}):
        left = record_lookup.get((baseline_id, topology))
        right = record_lookup.get((tau_id, topology))
        if left is None or right is None:
            continue
        left_worker = base.read(left["worker_path"])
        right_worker = base.read(right["worker_path"])
        static_collision.append({
            "topology_seed": topology,
            "static_array_identity_equal": (
                left_worker["static_array_identity"]
                == right_worker["static_array_identity"]
            ),
            "tau_values_differ": (
                candidates[baseline_id]["dynamic_parameters"]["tau_d_GABA_ms"]
                != candidates[tau_id]["dynamic_parameters"]["tau_d_GABA_ms"]
            ),
        })
    identity_groups = {}
    for candidate_id, candidate in candidates.items():
        identity_groups.setdefault(condition_key(candidate), []).append(candidate_id)
    exact_duplicates = [
        group for group in identity_groups.values() if len(group) > 1
    ]
    identity_audit = {
        "historical_baseline_vs_tau24_regression_applicable": tau_regression_applicable,
        "historical_baseline_vs_tau24_condition_keys_distinct": tau_condition_distinct,
        "training_topology_static_hash_collision_examples": static_collision,
        "static_collision_does_not_imply_condition_equivalence": True,
        "condition_identity_includes_time_constants": True,
        "core_swap_equivalence_applied": False,
        "exact_full_parameter_duplicate_groups_in_initial_48": exact_duplicates,
    }
    global_failure = tau_regression_applicable and not tau_condition_distinct
    status = (
        "PARAMETER_APPLICATION_AUDIT_PASS" if complete and not failures and not global_failure
        else "PARAMETER_APPLICATION_AUDIT_PARTIAL_PASS" if not failures and not global_failure
        else "PARAMETER_OR_CONDITION_IDENTITY_AUDIT_FAIL_REVIEW_REQUIRED"
    )
    report = {
        "status": status,
        "execution_path": str(execution),
        "candidate_manifest_sha256": base.sha(manifest_path),
        "execution_config_sha256": base.sha(config_path),
        "runtime_snapshot_sha256": base.sha(snapshot_path),
        "expected_seed_pairs": [list(pair) for pair in seed_pairs],
        "completed_units": len(records), "expected_units": expected,
        "all_expected_units_present": complete,
        "missing_units": [list(unit) for unit in missing_units],
        "unexpected_units": [list(unit) for unit in unexpected_units],
        "code_traced_checks": code_checks,
        "condition_identity_audit": identity_audit,
        "records": records,
        "failed_units": failures,
        "rerun_required": bool(failures),
        "downstream_logic_blocked": global_failure,
        "rerun_rule": (
            "only an actual physical value, seed, duration, or repaired-observer mismatch "
            "requires rerunning affected units; weak self-declared requested/effective evidence alone does not"
        ),
        "tau_evidence_limit": (
            "exact GABA decay is code-traced and recorded but not uniquely recoverable from output arrays"
        ),
    }
    path = Path(output_path) if output_path is not None else OUT / (
        "parameter_application_audit.json" if complete
        else "parameter_application_audit_partial.json"
    )
    base.write(path, report)
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
