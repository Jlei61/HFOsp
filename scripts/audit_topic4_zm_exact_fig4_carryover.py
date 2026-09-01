#!/usr/bin/env python3
"""Separate the exact Fig.4 carry-over arm from every calibrated candidate.

Two claims are audited here and nowhere else.

1. Dose semantics. ``dose_local_connectivity_coefficients`` must scale the two
   learned coefficient rows and nothing else: same edges, same delay bins, same
   GABA, and the same target-wise incoming budget the frozen mapper conserves.
   If a dose silently changed the graph, "5% of the learned E-to-I pathway"
   would not be a weaker version of Fig.4's substrate -- it would be a different
   substrate, and the whole arm distinction would be meaningless.

2. Arm identity. Exactly one parameter point is "Fig.4 substrate plus Z/M":
   candidate ``joint_04_control`` at both doses 1.0 with the reference Z/M
   protocol. Everything else -- a scaled ``I_th_EI``, a retuned ``tau_adp``, a
   pathway dose below 1.0, or a zeroed coefficient row -- is a different arm and
   is labelled as such, with its parameter delta recorded.

No simulation runs. No trajectory is regenerated.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src" / "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from scripts.freeze_topic4_zm_discovery_boundary import (  # noqa: E402
    guard_forbidden, load_audit_config, sha256_file)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402

EXACT_LABEL = "exact_fig4_carryover"
CALIBRATED_LABEL = "calibrated_transition"
DECOMPOSITION_LABEL = "pathway_decomposition"
REREGISTRATION_LABEL = "spatial_reregistration_control"
FORBIDDEN_LABEL_TEXT = "same Fig.4 substrate + only Z/M"

# The frozen library's three non-control arms are the pre-registered pathway
# factorial, not calibration attempts. They are dose-equivalent to the control
# because all four entries share one node field and differ only in which learned
# coefficient rows are zeroed -- verified below against the manifest.
ARM_DOSE_EQUIVALENT = {
    "joint_04_control": (1.0, 1.0),
    "joint_04_ee_only": (1.0, 0.0),
    "joint_04_etoi_only": (0.0, 1.0),
    "node_baseline": (0.0, 0.0),
}


def _pathway_totals(net, n_e, n_i):
    from src.topic4_local_connectivity import _incoming_by_pathway
    return {pathway: _incoming_by_pathway(net["ampa_by_delay"], n_e, n_i, pathway)
            for pathway in ("E_to_E", "E_to_I")}


def _structure_fingerprint(substrate):
    """Everything that must be invariant across doses, as hashes and values."""
    from src.topic4_core_connectivity import _hash_sparse_bins
    from src.topic4_graph_edge_flow import array_sha256
    net = substrate.net
    totals = _pathway_totals(net, substrate.n_e, substrate.n_i)
    return {
        "topology_sha256": _hash_sparse_bins(net["ampa_by_delay"],
                                             include_data=False),
        "gaba_sha256": _hash_sparse_bins(net["gaba_by_delay"]),
        "n_delay_bins": len(net["ampa_by_delay"]),
        "n_ampa_edges": int(sum(matrix.nnz for matrix in net["ampa_by_delay"])),
        "h_e_sha256": array_sha256(np.ascontiguousarray(substrate.h_e)),
        "h_i_sha256": array_sha256(np.ascontiguousarray(substrate.h_i)),
        "vtheta_sha256": array_sha256(np.ascontiguousarray(substrate.vtheta)),
        "delta_vtheta_sha256": array_sha256(
            np.ascontiguousarray(substrate.delta_vtheta)),
        "montage_sha256": array_sha256(np.ascontiguousarray(substrate.contact_xy)),
        "contact_names": list(substrate.contact_names),
        "effective_coefficients": substrate.edge_coefficients.tolist(),
        "incoming_budget_sha256": {
            pathway: array_sha256(np.ascontiguousarray(values))
            for pathway, values in totals.items()},
        "incoming_budget_sum": {pathway: float(values.sum())
                                for pathway, values in totals.items()},
        "edge_audit": {
            key: substrate.edge_audit[key]
            for key in ("topology_unchanged", "delay_assignment_unchanged",
                        "gaba_unchanged", "coefficients_sha256")
            if key in substrate.edge_audit},
        "pathway_audit": {
            pathway: {
                "n_edges": int(row["n_edges"]),
                "max_abs_incoming_error": float(row["max_abs_incoming_error"]),
            }
            for pathway, row in substrate.edge_audit.get(
                "pathway_audit", {}).items()},
        "_totals": totals,
    }


def _build(config, round_config, candidate_id, seed, ee_dose, etoi_dose):
    from src.topic4_zm_ictal_transition import build_substrate
    cache = ROOT / round_config["output_root"] / "network_cache"
    guard_forbidden(config, cache)
    substrate = build_substrate(round_config, candidate_id, seed,
                                cache_dir=str(cache), ee_dose=ee_dose,
                                etoi_dose=etoi_dose)
    return _structure_fingerprint(substrate)


def dose_semantics_audit(config, round_config, seed, probe_doses):
    """Clause C1.4: dose scales coefficient rows only and conserves budgets."""
    from src.topic4_zm_ictal_transition import dose_local_connectivity_coefficients

    manifest = json.loads(
        (ROOT / config["immutable_inputs"]["frozen_substrate_manifest"]["path"]
         ).read_text())
    library = {row["candidate_id"]: row
               for row in manifest["candidate_set"]["candidates"]}
    node_fields = {name: row["node_field"]["field_sha256"]
                   for name, row in library.items()}
    one_node_field = len(set(node_fields.values())) == 1

    learned = np.asarray(library["joint_04_control"]["coefficients"], float)
    algebra = []
    for ee_dose, etoi_dose in probe_doses:
        scaled = dose_local_connectivity_coefficients(
            learned, ee_dose=ee_dose, etoi_dose=etoi_dose)
        expected = learned * np.asarray([ee_dose, etoi_dose], float)[:, None]
        algebra.append({
            "E_to_E_dose": ee_dose, "E_to_I_dose": etoi_dose,
            "row_scaling_exact": bool(np.array_equal(scaled, expected)),
            "max_abs_row_error": float(np.max(np.abs(scaled - expected))),
            "shape_preserved": list(scaled.shape) == list(learned.shape),
            "learned_rows_not_mutated": bool(np.array_equal(
                learned, np.asarray(
                    library["joint_04_control"]["coefficients"], float))),
        })

    built = {}
    for ee_dose, etoi_dose in probe_doses:
        key = f"joint_04_control_ee{ee_dose:g}_etoi{etoi_dose:g}"
        built[key] = _build(config, round_config, "joint_04_control", seed,
                            ee_dose, etoi_dose)
    zeroed_arm = _build(config, round_config, "joint_04_ee_only", seed, 1.0, 1.0)

    reference = built[f"joint_04_control_ee1_etoi1"]
    invariants = {}
    for key, row in built.items():
        invariants[key] = {
            "topology_identical": row["topology_sha256"] == reference["topology_sha256"],
            "gaba_identical": row["gaba_sha256"] == reference["gaba_sha256"],
            "n_delay_bins_identical": row["n_delay_bins"] == reference["n_delay_bins"],
            "n_ampa_edges_identical": row["n_ampa_edges"] == reference["n_ampa_edges"],
            "node_field_identical": row["h_e_sha256"] == reference["h_e_sha256"],
            "threshold_identical": row["vtheta_sha256"] == reference["vtheta_sha256"],
            "montage_identical": row["montage_sha256"] == reference["montage_sha256"],
            "max_incoming_budget_delta": {
                pathway: float(np.max(np.abs(
                    row["_totals"][pathway] - reference["_totals"][pathway])))
                for pathway in ("E_to_E", "E_to_I")},
            "mapper_max_abs_incoming_error": {
                pathway: entry["max_abs_incoming_error"]
                for pathway, entry in row["pathway_audit"].items()},
        }

    dose_zero_key = "joint_04_control_ee1_etoi0"
    cross_check = None
    if dose_zero_key in built:
        cross_check = {
            "claim": ("E-to-I dose 0 on the control arm must realise the same "
                      "graph as the frozen library's joint_04_ee_only entry"),
            "incoming_budget_identical": {
                pathway: bool(np.allclose(
                    built[dose_zero_key]["_totals"][pathway],
                    zeroed_arm["_totals"][pathway], rtol=0.0, atol=1e-9))
                for pathway in ("E_to_E", "E_to_I")},
            "topology_identical": (built[dose_zero_key]["topology_sha256"]
                                   == zeroed_arm["topology_sha256"]),
            "effective_coefficients_identical": bool(np.allclose(
                np.asarray(built[dose_zero_key]["effective_coefficients"], float),
                np.asarray(zeroed_arm["effective_coefficients"], float),
                rtol=0.0, atol=0.0)),
        }

    for row in built.values():
        row.pop("_totals", None)
    zeroed_arm.pop("_totals", None)
    return {
        "seed": int(seed),
        "single_node_field_across_library": one_node_field,
        "library_node_field_sha256": node_fields,
        "coefficient_algebra": algebra,
        "built_fingerprints": built,
        "frozen_zeroed_arm_fingerprint": zeroed_arm,
        "dose_invariants_vs_exact_carryover": invariants,
        "dose_zero_vs_frozen_zeroed_arm": cross_check,
    }


def _log_parameter_distance(candidate, exact):
    """Distance in log space over the six work-point coordinates.

    Doses can be exactly zero, so a plain log is undefined; the config's
    ``zero_dose_epsilon`` floors the dose coordinates instead of dropping them,
    which would make a zeroed pathway look identical to the carry-over point.
    """
    epsilon = float(exact["zero_dose_epsilon"])
    total, per_axis = 0.0, {}
    for name in exact["log_parameter_distance_coordinates"]:
        observed = float(candidate[name])
        reference = float(exact[name])
        if name.endswith("_dose"):
            observed = max(observed, epsilon)
            reference = max(reference, epsilon)
        delta = float(np.log(observed) - np.log(reference))
        per_axis[name] = delta
        total += delta * delta
    return {"log_distance": float(np.sqrt(total)), "per_axis": per_axis}


def classify_candidate(exact, parameters, *, field_transform="none"):
    """Clauses C1.2, C1.3, C1.5, C1.6: arm identity and explicit delta.

    ``field_transform`` is part of the substrate, not a display option. A run
    under r90/r180/mx queries the node field at inverse-transformed positions and
    rotates the two directed flow coefficients, so its realised ``h`` and
    per-neuron thresholds are NOT Fig.4's. Its Z/M parameters are nevertheless
    identical, so a parameter-only comparison silently promotes 18 matched
    spatial re-registration controls into the exact carry-over arm.
    """
    changed = {}
    for name in exact["log_parameter_distance_coordinates"]:
        observed, reference = float(parameters[name]), float(exact[name])
        if observed != reference:
            changed[name] = {"exact_fig4": reference, "candidate": observed,
                             "ratio": (observed / reference
                                       if reference else None)}
    candidate_id = parameters.get("candidate_id", exact["candidate_id"])
    transform = str(field_transform or "none")
    if transform != "none":
        arm = REREGISTRATION_LABEL
    elif candidate_id != exact["candidate_id"]:
        arm = DECOMPOSITION_LABEL
    elif changed:
        arm = CALIBRATED_LABEL
    else:
        arm = EXACT_LABEL
    return {
        "arm": arm,
        "field_transform": transform,
        "changed_parameters": changed,
        "is_exact_fig4_carryover": arm == EXACT_LABEL,
        "may_be_described_as_fig4_substrate_plus_zm_only": arm == EXACT_LABEL,
        "distance_from_exact_carryover": _log_parameter_distance(
            {**parameters, "candidate_id": candidate_id}, exact),
    }


def _calibration_parameters(payload):
    """Doses were added to the runner after the first two calibration rounds.

    Absence of the dose keys means the run predates the flag, so both doses were
    the build default 1.0. The journalled command lines for those four runs carry
    no dose argument, which is the independent confirmation.
    """
    parameters = dict(payload["parameters"])
    parameters.setdefault("E_to_E_dose", 1.0)
    parameters.setdefault("E_to_I_dose", 1.0)
    parameters["dose_source"] = (
        "explicit" if "E_to_E_dose" in payload["parameters"]
        else "runner_default_1.0_predates_dose_flag")
    parameters["candidate_id"] = payload.get("candidate_id", "joint_04_control")
    return parameters


def collect_candidates(config, round_config):
    exact = config["exact_fig4_carryover"]
    rows = []
    for directory in config["candidate_inventory_roots"][
            "calibrated_transition_candidates"]:
        base = ROOT / directory
        if not base.exists():
            rows.append({"source": directory, "status": "DIRECTORY_MISSING"})
            continue
        payloads = sorted(p for p in base.glob("*.json")
                          if p.name != "calibration_summary.json")
        if not payloads:
            rows.append({"source": directory, "candidate": None,
                         "status": "NO_COMPLETED_RUN"})
            continue
        for path in payloads:
            guard_forbidden(config, path)
            payload = json.loads(path.read_text())
            parameters = _calibration_parameters(payload)
            rows.append({
                "source": str(path.relative_to(ROOT)),
                "candidate": path.stem,
                "family": "calibration",
                "seed": payload.get("seed"),
                "status": payload.get("status"),
                "parameters": parameters,
                **classify_candidate(exact, parameters)})

    zm = round_config["zm"]
    for path in sorted((ROOT / config["candidate_inventory_roots"][
            "exact_carryover_and_pathway_arms"]).glob("*.json")):
        guard_forbidden(config, path)
        payload = json.loads(path.read_text())
        if payload.get("zm_mode") != "z_plus_m":
            continue
        candidate_id = payload["candidate_id"]
        ee_dose, etoi_dose = ARM_DOSE_EQUIVALENT[candidate_id]
        parameters = {
            "I_th_EI": float(zm["I_th_EI"]), "tau_z": float(zm["tau_z"]),
            "tau_adp": float(zm["tau_adp"]), "eta_m": float(zm["eta_m"]),
            "E_to_E_dose": ee_dose, "E_to_I_dose": etoi_dose,
            "dose_source": "frozen_library_coefficient_rows",
            "candidate_id": candidate_id}
        rows.append({
            "source": str(path.relative_to(ROOT)),
            "candidate": path.stem,
            "family": "trajectory_worker",
            "seed": payload.get("seed"),
            "arm_name": payload.get("arm"),
            "field_transform": payload.get("field_transform"),
            "status": payload.get("status"),
            "parameters": parameters,
            **classify_candidate(
                exact, parameters,
                field_transform=payload.get("field_transform", "none"))})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_discovery_audit_v1.json")
    parser.add_argument("--seed", type=int, default=1801)
    parser.add_argument("--skip-substrate-build", action="store_true",
                        help="classify candidates only; skip the dose-semantics "
                             "rebuild (about 45 s per probe dose)")
    args = parser.parse_args()

    config = load_audit_config(args.config)
    round_path = config["immutable_inputs"]["round_config"]["path"]
    round_config = json.loads((ROOT / round_path).read_text())

    candidates = collect_candidates(config, round_config)
    exact_rows = [row for row in candidates if row.get("arm") == EXACT_LABEL]
    calibrated = [row for row in candidates if row.get("arm") == CALIBRATED_LABEL]
    decomposition = [row for row in candidates
                     if row.get("arm") == DECOMPOSITION_LABEL]
    reregistration = [row for row in candidates
                      if row.get("arm") == REREGISTRATION_LABEL]

    audit = {
        "status": config["status"],
        "exact_fig4_carryover_definition": config["exact_fig4_carryover"],
        "label_guard": {
            "forbidden_label": FORBIDDEN_LABEL_TEXT,
            "rule": ("only rows with arm == exact_fig4_carryover may carry this "
                     "label; every other row records its parameter delta"),
            "violations": [row["candidate"] for row in candidates
                           if row.get("may_be_described_as_fig4_substrate_plus_zm_only")
                           and row.get("arm") != EXACT_LABEL],
        },
        "counts": {
            "exact_fig4_carryover": len(exact_rows),
            "calibrated_transition": len(calibrated),
            "pathway_decomposition": len(decomposition),
            "spatial_reregistration_control": len(reregistration),
            "not_evaluable_sources": [row["source"] for row in candidates
                                      if row.get("status") in
                                      ("NO_COMPLETED_RUN", "DIRECTORY_MISSING")],
        },
        "exact_carryover_seeds": sorted(
            {row["seed"] for row in exact_rows if row.get("seed") is not None}),
        "candidates": candidates,
        "trajectories_regenerated": False,
        "simulation_launched": False,
    }
    if not args.skip_substrate_build:
        audit["dose_semantics"] = dose_semantics_audit(
            config, round_config, args.seed,
            probe_doses=[(1.0, 1.0), (1.0, 0.05), (1.0, 0.0)])

    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    audit["audit_config_sha256"] = sha256_file(ROOT / args.config)
    atomic_write_json(audit, str(output_root / "exact_carryover_audit.json"))
    print(json.dumps({"counts": audit["counts"],
                      "label_violations": audit["label_guard"]["violations"],
                      "exact_carryover_seeds": audit["exact_carryover_seeds"]},
                     indent=1))


if __name__ == "__main__":
    main()
