#!/usr/bin/env python3
"""Freeze the rev13 model-internal zero-sum Node recovery canary."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
EXPECTED_INPUTS = {"stage_ak_config", "stage_ak_manifest", "stage_al_manifest"}
FORBIDDEN_INPUT_TERMS = ("patient", "prototype", "classifier", "target", "heldout")
EXPECTED_ARM_IDS = (
    "exact_off",
    "zero_sum_c010",
    "zero_sum_c020",
    "zero_sum_c040",
    "raise_only_c020",
    "stratified_shuffle_c020",
)
EXPECTED_CONTROLLERS = {
    "zero_sum_c010": ("zero_sum", 0.1),
    "zero_sum_c020": ("zero_sum", 0.2),
    "zero_sum_c040": ("zero_sum", 0.4),
    "raise_only_c020": ("raise_only", 0.2),
    "stratified_shuffle_c020": ("stratified_shuffle", 0.2),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _unique_candidate(manifest: dict, candidate_id: str) -> dict:
    matches = [
        row for row in manifest.get("candidates", [])
        if row.get("candidate_id") == candidate_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"candidate is absent or duplicated: {candidate_id}")
    return matches[0]


def _validate_inputs_contract(config: dict) -> None:
    inputs = config.get("inputs", {})
    if set(inputs) != EXPECTED_INPUTS:
        raise RuntimeError("rev13 input set changed")
    for name, record in inputs.items():
        path = str(record.get("path", "")).lower()
        if any(term in name.lower() or term in path for term in FORBIDDEN_INPUT_TERMS):
            raise RuntimeError("patient labels/prototypes cannot enter the rev13 freezer")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"input hash is not frozen: {name}")


def _validate_static_contract(config: dict) -> None:
    contract = config["node_accessibility_contract"]
    if float(contract["tau_ms"]) != 250.0:
        raise RuntimeError("rev13 primary tau changed")
    if float(contract["reference_rate_hz"]) != 50.0:
        raise RuntimeError("rev13 reference rate changed")
    if float(contract["a_max_multiplier"]) != 2.0:
        raise RuntimeError("rev13 trace bound changed")
    if contract.get("uses_patient_labels") is not False:
        raise RuntimeError("patient labels cannot update rev13 state")
    if contract.get("uses_patient_prototypes") is not False:
        raise RuntimeError("patient prototypes cannot update rev13 state")
    if contract.get("draws_random_numbers_at_runtime") is not False:
        raise RuntimeError("rev13 controller must not draw random numbers")
    if config.get("pathways") != {
        "learned_E_to_E_redistribution": "off",
        "learned_E_to_I_redistribution": "off",
        "Z_M": "off",
    }:
        raise RuntimeError("EE, E-to-I and Z/M must remain off")
    search = config["search"]
    if search.get("canary_network_seeds") != [2311]:
        raise RuntimeError("rev13 canary seed changed")
    if search.get("fit_network_seeds") != [2312, 2313]:
        raise RuntimeError("rev13 fit seeds changed")
    if float(search["simulation"]["duration_ms"]) != 10000.0:
        raise RuntimeError("rev13 duration changed")


def _validate_arms(arms: list[dict]) -> None:
    arm_ids = [str(row.get("arm_id")) for row in arms]
    if tuple(arm_ids) != EXPECTED_ARM_IDS or len(set(arm_ids)) != len(arm_ids):
        raise RuntimeError("rev13 arm set or order changed")
    off_rows = [row for row in arms if row.get("controller") is None]
    if len(off_rows) != 1 or off_rows[0].get("arm_id") != "exact_off":
        raise RuntimeError("rev13 requires one literal exact-off arm")
    for row in arms:
        arm_id = row["arm_id"]
        if arm_id == "exact_off":
            continue
        controller = row.get("controller")
        expected_mode, expected_c = EXPECTED_CONTROLLERS[arm_id]
        if not isinstance(controller, dict):
            raise RuntimeError(f"controller is missing: {arm_id}")
        if controller.get("mode") != expected_mode:
            raise RuntimeError(f"controller mode changed: {arm_id}")
        if float(controller.get("c", -1.0)) != expected_c:
            raise RuntimeError(f"controller amplitude changed: {arm_id}")


def _validate_primary(primary: dict, contract: dict) -> None:
    if primary.get("candidate_id") != contract["candidate_id"]:
        raise RuntimeError("primary Stage-AK substrate changed")
    if primary.get("source_candidate_ids") != contract["source_candidate_ids"]:
        raise RuntimeError("primary Stage-AK source fields changed")
    mapping = primary.get("node_mapping", {})
    if mapping.get("mapping_type") != contract["mapping_type"]:
        raise RuntimeError("primary Node mapping type changed")
    if mapping.get("mapping_sha256") != contract["mapping_sha256"]:
        raise RuntimeError("primary Node mapping hash changed")
    mean_field = primary.get("node_field", {})
    dispersion_field = primary.get("node_dispersion_field", {})
    if mean_field.get("field_sha256") != contract["mean_field_sha256"]:
        raise RuntimeError("primary mean-support field changed")
    if dispersion_field.get("field_sha256") != contract["dispersion_field_sha256"]:
        raise RuntimeError("primary dispersion-support field changed")
    for field in (mean_field, dispersion_field):
        if field.get("field_type") != "spline_continuous":
            raise RuntimeError("rev13 support must remain a continuous spline")
        for key in ("coefficients", "n_basis", "degree", "field_sha256"):
            if key not in field:
                raise RuntimeError(f"rev13 support field is incomplete: {key}")
    if (mean_field["n_basis"], mean_field["degree"]) != (
            dispersion_field["n_basis"], dispersion_field["degree"]):
        raise RuntimeError("rev13 support fields are not shape-compatible")
    support = contract.get("support", {})
    if support.get("field_names") != ["node_field", "node_dispersion_field"]:
        raise RuntimeError("rev13 support fields changed")
    if support.get("normalization") != "maximum_over_E_neurons":
        raise RuntimeError("rev13 support normalization changed")
    if float(support.get("minimum_denominator", 0.0)) <= 0.0:
        raise RuntimeError("rev13 support denominator guard is absent")


def _validate_stage_al_alias(primary: dict, stage_al_manifest: dict,
                             contract: dict) -> dict:
    alias_contract = contract["stage_al_alias"]
    if alias_contract.get("count_as_additional_substrate") is not False:
        raise RuntimeError("Stage-AL alias cannot be a second substrate")
    alias = _unique_candidate(stage_al_manifest, alias_contract["candidate_id"])
    coordinates = alias.get("local_bridge_coordinates", {})
    if float(coordinates.get("mean_affine_weight", -1.0)) != float(
            alias_contract["mean_affine_weight"]):
        raise RuntimeError("Stage-AL alias mean coordinate changed")
    if float(coordinates.get("dispersion_interpolation_weight", -1.0)) != float(
            alias_contract["dispersion_interpolation_weight"]):
        raise RuntimeError("Stage-AL alias dispersion coordinate changed")
    if alias.get("source_candidate_ids") != primary.get("source_candidate_ids"):
        raise RuntimeError("Stage-AL alias source fields changed")
    tolerance = float(alias_contract["coefficient_absolute_tolerance"])
    maximum_error = 0.0
    for key in ("node_field", "node_dispersion_field"):
        canonical = np.asarray(primary[key]["coefficients"], dtype=np.float64)
        duplicate = np.asarray(alias[key]["coefficients"], dtype=np.float64)
        if canonical.shape != duplicate.shape:
            raise RuntimeError("Stage-AL alias support shape changed")
        error = float(np.max(np.abs(canonical - duplicate)))
        maximum_error = max(maximum_error, error)
        if not np.allclose(canonical, duplicate, rtol=0.0, atol=tolerance):
            raise RuntimeError("Stage-AL alias is not numerically equivalent")
    return {
        "candidate_id": alias["candidate_id"],
        "canonical_candidate_id": primary["candidate_id"],
        "counted_as_additional_substrate": False,
        "coefficient_absolute_tolerance": tolerance,
        "maximum_coefficient_absolute_error": maximum_error,
    }


def build_candidates(stage_ak_manifest: dict, stage_al_manifest: dict,
                     stage_ak_config: dict, config: dict) -> tuple[list[dict], dict]:
    """Build six controller arms around one canonical Stage-AK substrate."""
    _validate_inputs_contract(config)
    _validate_static_contract(config)
    _validate_arms(config["arms"])
    if stage_ak_manifest.get("event_unit") != stage_ak_config.get("event_unit"):
        raise RuntimeError("Stage-AK event unit is internally inconsistent")
    if config.get("event_unit") != stage_ak_config.get("event_unit"):
        raise RuntimeError("rev13 event unit drifted from Stage-AK")
    if config.get("source_topology") != stage_ak_config.get("source_topology"):
        raise RuntimeError("rev13 source topology drifted from Stage-AK")

    substrate_contract = config["primary_substrate"]
    primary = _unique_candidate(stage_ak_manifest, substrate_contract["candidate_id"])
    _validate_primary(primary, substrate_contract)
    alias_audit = _validate_stage_al_alias(primary, stage_al_manifest, substrate_contract)

    substrate = {
        "source_candidate_ids": copy.deepcopy(primary["source_candidate_ids"]),
        "node_field": copy.deepcopy(primary["node_field"]),
        "node_dispersion_field": copy.deepcopy(primary["node_dispersion_field"]),
        "node_mapping": copy.deepcopy(primary["node_mapping"]),
        "node_support": copy.deepcopy(substrate_contract["support"]),
    }
    controller_contract = config["node_accessibility_contract"]
    candidates = []
    for arm in config["arms"]:
        controller = arm["controller"]
        node_accessibility = None
        if controller is not None:
            node_accessibility = {
                "kind": "field_gated_bounded_node_recovery",
                "mode": controller["mode"],
                "c": float(controller["c"]),
                "tau_ms": float(controller_contract["tau_ms"]),
                "reference_rate_hz": float(controller_contract["reference_rate_hz"]),
                "a_max_multiplier": float(controller_contract["a_max_multiplier"]),
                "support_formula": substrate_contract["support"]["formula"],
                "increment_formula": controller_contract["increment_formula"],
                "state_update": controller_contract["state_update"],
                "draws_random_numbers_at_runtime": False,
            }
            if controller["mode"] == "stratified_shuffle":
                node_accessibility["stratified_shuffle"] = copy.deepcopy(
                    controller_contract["stratified_shuffle"]
                )
        candidates.append({
            "candidate_id": arm["arm_id"],
            "role": "rev13_model_internal_node_recovery_canary_not_selectable",
            "selection_eligible": False,
            "primary_substrate_id": primary["candidate_id"],
            **copy.deepcopy(substrate),
            "node_accessibility": node_accessibility,
            "pathways": copy.deepcopy(config["pathways"]),
        })
    if len(candidates) != 6:
        raise RuntimeError("rev13 candidate count changed")
    if any(row["candidate_id"].startswith("stage_al_") for row in candidates):
        raise RuntimeError("Stage-AL alias was counted as a separate substrate")
    return candidates, {
        "primary_candidate_id": primary["candidate_id"],
        "primary_mapping_sha256": primary["node_mapping"]["mapping_sha256"],
        "mean_field_sha256": primary["node_field"]["field_sha256"],
        "dispersion_field_sha256": primary["node_dispersion_field"]["field_sha256"],
        "support": copy.deepcopy(substrate_contract["support"]),
        "stage_al_alias": alias_audit,
        "n_unique_substrates": 1,
        "patient_labels_read": False,
        "patient_prototypes_read": False,
    }


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_hashed_inputs(config: dict, artifact_root: Path) -> tuple[dict, dict]:
    _validate_inputs_contract(config)
    loaded, audit = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"rev13 input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        audit[name] = {"path": str(path), "sha256": observed}
    return loaded, audit


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if current != expected:
        raise RuntimeError("rev13 freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("rev13 freezer paths are dirty")
    for relative in tracked:
        committed = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(committed).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"rev13 freezer path drifted: {relative}")
    return {"git_commit": expected, "tracked_modules": tracked, "dirty": False}


def build_manifest(config: dict, loaded: dict, input_audit: dict,
                   *, config_path: Path, provenance: dict) -> dict:
    candidates, substrate_audit = build_candidates(
        loaded["stage_ak_manifest"], loaded["stage_al_manifest"],
        loaded["stage_ak_config"], config,
    )
    return {
        "schema_id": "topic4_rev13_node_zero_sum_recovery_manifest_v1",
        "status": "REV13_NODE_ZERO_SUM_RECOVERY_CANARY_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "substrate_audit": substrate_audit,
        "event_unit": copy.deepcopy(config["event_unit"]),
        "source_topology": copy.deepcopy(config["source_topology"]),
        "search": copy.deepcopy(config["search"]),
        "pathways": copy.deepcopy(config["pathways"]),
        "inputs": input_audit,
        "provenance": provenance,
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    loaded, input_audit = _load_hashed_inputs(config, artifact_root)
    payload = build_manifest(
        config, loaded, input_audit, config_path=config_path,
        provenance=_provenance(config_path, args.expected_commit),
    )
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(payload["candidates"]),
        "n_unique_substrates": payload["substrate_audit"]["n_unique_substrates"],
        "n_canary_networks": len(config["search"]["canary_network_seeds"]),
        "n_fit_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
