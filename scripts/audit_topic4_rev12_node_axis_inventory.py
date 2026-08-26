#!/usr/bin/env python3
"""Inventory completed rev12 Node axes before another SNN run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _nested_values(value: Any, keys: set[str]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in keys:
                found.append({"key": key, "value": child})
            found.extend(_nested_values(child, keys))
    elif isinstance(value, list):
        for child in value:
            found.extend(_nested_values(child, keys))
    return found


def _surface_amplitudes(candidates: list[dict[str, Any]]) -> list[float]:
    values: set[float] = set()
    keys = {
        "signed_log_surface_rms", "nominal_l2_surface_rms",
        "observed_surface_rms", "stage_ag_residual_rms", "radius",
    }
    for candidate in candidates:
        residual = candidate.get("node_field", {}).get("residual_coordinates", {})
        for record in _nested_values(residual, keys):
            value = record["value"]
            if isinstance(value, (int, float)):
                values.add(round(abs(float(value)), 12))
    return sorted(values)


def inspect_manifest(path: Path, current_event_unit: str,
                     scalar_gain_keys: set[str]) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    candidates = list(manifest.get("candidates", []))
    event_unit = (manifest.get("event_unit") or {}).get("name")
    gain_records = []
    shrinkage_values: set[float] = set()
    field_hashes: set[str] = set()
    for candidate in candidates:
        gain_records.extend(_nested_values(candidate, scalar_gain_keys))
        mapping = candidate.get("node_mapping") or {}
        if "signed_depth_shrinkage" in mapping:
            shrinkage_values.add(float(mapping["signed_depth_shrinkage"]))
        field_hash = (candidate.get("node_field") or {}).get("field_sha256")
        if field_hash:
            field_hashes.add(str(field_hash))
    mapping_audit = manifest.get("mapping_audit") or {}
    if "reference_rho" in mapping_audit:
        shrinkage_values.add(float(mapping_audit["reference_rho"]))
    return {
        "stage": path.parent.name,
        "manifest": str(path),
        "manifest_sha256": _sha256(path),
        "status": manifest.get("status"),
        "n_candidates": len(candidates),
        "event_unit": event_unit,
        "current_causal_family_compatible": event_unit == current_event_unit,
        "n_unique_fields": len(field_hashes),
        "surface_geometry_amplitudes": _surface_amplitudes(candidates),
        "signed_depth_shrinkage_values": sorted(shrinkage_values),
        "scalar_node_gain_records": gain_records,
        "has_scalar_node_gain": bool(gain_records),
    }


def _provenance(config_path: Path, expected_commit: str) -> dict[str, Any]:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    observed = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if observed != expected:
        raise RuntimeError("inventory is not running at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_core_field_rev9.py",
        "src/topic4_zm_ictal_transition.py",
        "scripts/run_topic4_rev12_node_worker.py",
    ]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("inventory runtime paths are dirty")
    return {"git_commit": expected, "tracked_modules": tracked, "dirty": False}


def build_inventory(config: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    result_root = _resolve(artifact_root, config["result_root"])
    manifests = sorted(result_root.glob("node_stage_*/candidate_manifest.json"))
    if not manifests:
        raise RuntimeError("no rev12 Node candidate manifests found")
    gain_keys = set(config["scalar_gain_keys"])
    rows = [
        inspect_manifest(path, config["current_event_unit"], gain_keys)
        for path in manifests
    ]
    required_inputs = []
    for relative in config["required_result_inputs"]:
        path = _resolve(artifact_root, relative)
        if not path.exists():
            raise RuntimeError(f"required result input missing: {relative}")
        required_inputs.append({
            "path": str(path), "sha256": _sha256(path),
            "status": json.loads(path.read_text()).get("status"),
        })
    unexecuted = []
    for record in config["known_unexecuted_proposals"]:
        config_path = _resolve(artifact_root, record["config"])
        analysis_path = _resolve(artifact_root, record["analysis"])
        analysis = json.loads(analysis_path.read_text())
        decision = analysis.get("outer_amplitude_decision", {})
        observed = decision.get("status")
        if observed != record["expected_status"]:
            raise RuntimeError("known outer-amplitude decision changed")
        unexecuted.append({
            **record,
            "config_sha256": _sha256(config_path),
            "analysis_sha256": _sha256(analysis_path),
            "observed_status": observed,
        })
    gain_rows = [row for row in rows if row["has_scalar_node_gain"]]
    compatible = [row for row in rows if row["current_causal_family_compatible"]]
    geometry_amplitudes = sorted({
        value for row in compatible for value in row["surface_geometry_amplitudes"]
    })
    shrinkages = sorted({
        value for row in compatible for value in row["signed_depth_shrinkage_values"]
    })
    status = (
        "SCALAR_NODE_GAIN_ALREADY_TESTED"
        if gain_rows else
        "SCALAR_NODE_GAIN_NOT_PREVIOUSLY_TESTED_CURRENT_CAUSAL_EVENT_UNIT"
    )
    return {
        "schema_id": "topic4_rev12_nd_node_axis_inventory_result_v1",
        "status": status,
        "scientific_role": config["scientific_role"],
        "summary": {
            "n_manifests": len(rows),
            "n_current_causal_family_manifests": len(compatible),
            "current_geometry_amplitudes": geometry_amplitudes,
            "maximum_current_geometry_amplitude": (
                max(geometry_amplitudes) if geometry_amplitudes else None
            ),
            "signed_depth_shrinkage_values": shrinkages,
            "scalar_node_gain_values": [
                record for row in gain_rows
                for record in row["scalar_node_gain_records"]
            ],
            "stage_u_outer_followup_was_executed": False,
            "current_event_unit_rescoring_needed": False,
            "reason": (
                "Current causal-family stages already score broad continuous-field "
                "geometry through RMS 2.0 and signed-depth shrinkage through zero, "
                "but no completed candidate changes the global expression gain of "
                "delta_vtheta = -h*d. Older exact-neuron stages change observation "
                "or field coefficients and are superseded by causal-family refits."
            ),
        },
        "manifests": rows,
        "known_unexecuted_proposals": unexecuted,
        "required_result_inputs": required_inputs,
        "next_decision": (
            "A bounded paired scalar Node-gain canary is non-duplicative. Keep h, d, "
            "topology, delays, detector and random seeds fixed; do not open EE, "
            "E-to-I, Z/M, patient held-out or a new field geometry search."
        ),
        "claim_boundary": config["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    payload = build_inventory(config, args.artifact_root.resolve())
    payload["config"] = str(config_path.relative_to(ROOT))
    payload["config_sha256"] = _sha256(config_path)
    payload["provenance"] = _provenance(config_path, args.expected_commit)
    output = _resolve(args.artifact_root.resolve(), config["output"])
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_manifests": payload["summary"]["n_manifests"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
