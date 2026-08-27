#!/usr/bin/env python3
"""Freeze the final local bridge in dual continuous Node-channel space."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_node_field_search import array_sha256, spline_roughness


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _field(candidate: dict, candidate_id: str) -> dict:
    rows = [
        row for row in candidate["candidates"]
        if row["candidate_id"] == candidate_id
    ]
    if len(rows) != 1:
        raise RuntimeError(f"source field {candidate_id} is not unique")
    return copy.deepcopy(rows[0]["node_field"])


def affine_field(left: dict, right: dict, *, weight: float,
                 candidate_id: str, role: str) -> dict:
    """Move along one observation-invariant spline direction, allowing extrapolation."""
    for key in ("field_type", "n_basis", "degree"):
        if left[key] != right[key]:
            raise ValueError(f"spline fields differ in {key}")
    if left["field_type"] != "spline_continuous":
        raise ValueError("local bridge requires spline_continuous fields")
    left_values = np.asarray(left["coefficients"], float)
    right_values = np.asarray(right["coefficients"], float)
    if left_values.shape != right_values.shape:
        raise ValueError("spline coefficient tensors do not align")
    values = left_values + float(weight) * (right_values - left_values)
    return {
        "candidate_id": candidate_id,
        "field_type": "spline_continuous",
        "n_basis": int(left["n_basis"]),
        "degree": int(left["degree"]),
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "roughness": spline_roughness(values),
        "component_count": None,
        "peak_count_constraint": None,
        "role": role,
        "source_field_sha256": [left["field_sha256"], right["field_sha256"]],
        "residual_coordinates": {
            "affine_weight_toward_right": float(weight),
            "observation_coordinates_used": False,
            "manual_geometry_used": False,
        },
    }


def _mapping_hash(mean_hash: str, dispersion_hash: str, formula: str) -> str:
    encoded = json.dumps({
        "mean_field_sha256": mean_hash,
        "dispersion_field_sha256": dispersion_hash,
        "formula": formula,
    }, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_candidates(stage_ak_manifest: dict, stage_ak_result: dict,
                     contract: dict) -> tuple[list[dict], dict]:
    if stage_ak_result.get("status") != (
        "DUAL_CONTINUOUS_NODE_CHANNEL_DOES_NOT_RESOLVE_TWO_MODE_TRADEOFF"
    ):
        raise RuntimeError("Stage-AK does not justify a local bridge")
    source = {row["candidate_id"]: row for row in stage_ak_manifest["candidates"]}
    historical_id = contract["historical_anchor_candidate_id"]
    if historical_id not in source:
        raise RuntimeError("historical dual-channel anchor is absent")

    by_source = {
        (row["source_candidate_ids"]["mean"],
         row["source_candidate_ids"]["dispersion"]): row
        for row in stage_ak_manifest["candidates"]
    }
    mean_left = _field(stage_ak_manifest, next(
        row["candidate_id"] for row in stage_ak_manifest["candidates"]
        if row["source_candidate_ids"]["mean"] == contract["mean_left_candidate_id"]
        and row["source_candidate_ids"]["dispersion"] == contract["mean_left_candidate_id"]
    ))
    mean_right = _field(stage_ak_manifest, next(
        row["candidate_id"] for row in stage_ak_manifest["candidates"]
        if row["source_candidate_ids"]["mean"] == contract["mean_right_candidate_id"]
        and row["source_candidate_ids"]["dispersion"] == contract["mean_right_candidate_id"]
    ))
    dispersion_left = copy.deepcopy(by_source[
        (contract["dispersion_left_candidate_id"],
         contract["dispersion_left_candidate_id"])
    ]["node_dispersion_field"])
    dispersion_right = copy.deepcopy(by_source[
        (contract["dispersion_right_candidate_id"],
         contract["dispersion_right_candidate_id"])
    ]["node_dispersion_field"])

    candidates = []
    historical = copy.deepcopy(source[historical_id])
    historical.update({
        "candidate_id": "stage_al_historical_anchor",
        "role": "dual_node_local_bridge_anchor_not_selectable",
        "selection_eligible": False,
        "local_bridge_coordinates": None,
    })
    candidates.append(historical)

    coordinates = [
        (float(mean_weight), float(dispersion_weight), False)
        for mean_weight in contract["mean_affine_weights"]
        for dispersion_weight in contract["dispersion_interpolation_weights"]
    ] + [
        (float(mean_weight), float(dispersion_weight), True)
        for mean_weight, dispersion_weight in contract["required_sentinels"]
    ]
    if len(set(coordinates)) != len(coordinates):
        raise RuntimeError("local bridge coordinates are not unique")
    template = copy.deepcopy(by_source[
        (contract["mean_right_candidate_id"],
         contract["dispersion_left_candidate_id"])
    ])
    audits = []
    for mean_weight, dispersion_weight, sentinel in coordinates:
        tag_m = f"{round(mean_weight * 100):03d}"
        tag_d = f"{round(dispersion_weight * 100):03d}"
        candidate_id = f"stage_al_m{tag_m}_d{tag_d}"
        mean_field = affine_field(
            mean_left, mean_right, weight=mean_weight,
            candidate_id=f"{candidate_id}_mean",
            role="dual_node_local_mean_affine_field",
        )
        dispersion_field = affine_field(
            dispersion_left, dispersion_right, weight=dispersion_weight,
            candidate_id=f"{candidate_id}_dispersion",
            role="dual_node_local_dispersion_interpolation",
        )
        mapping_hash = _mapping_hash(
            mean_field["field_sha256"], dispersion_field["field_sha256"],
            contract["formula"],
        )
        candidate = copy.deepcopy(template)
        candidate.update({
            "candidate_id": candidate_id,
            "role": "dual_node_local_bridge_sentinel" if sentinel else
                    "dual_node_local_bridge_candidate",
            "selection_eligible": not sentinel,
            "source_candidate_ids": {
                "mean": contract["mean_right_candidate_id"],
                "dispersion": contract["dispersion_right_candidate_id"],
            },
            "node_field": mean_field,
            "node_dispersion_field": dispersion_field,
            "node_mapping": {
                "mapping_type": "dual_continuous_mean_dispersion",
                "signed_depth_shrinkage": 1.0,
                "node_gain": 1.0,
                "mapping_sha256": mapping_hash,
            },
            "local_bridge_coordinates": {
                "mean_affine_weight": mean_weight,
                "dispersion_interpolation_weight": dispersion_weight,
                "sentinel": sentinel,
            },
        })
        candidates.append(candidate)
        audits.append({
            "candidate_id": candidate_id,
            **candidate["local_bridge_coordinates"],
            "mean_field_sha256": mean_field["field_sha256"],
            "dispersion_field_sha256": dispersion_field["field_sha256"],
            "mapping_sha256": mapping_hash,
        })
    if len(candidates) != int(contract["expected_candidate_count"]):
        raise RuntimeError("local bridge candidate count changed")
    if len({row["mapping_sha256"] for row in audits}) != len(audits):
        raise RuntimeError("local bridge mappings are not unique")
    return candidates, {
        "historical_anchor_candidate_id": "stage_al_historical_anchor",
        "formula": contract["formula"],
        "coordinates": audits,
        "patient_heldout_used": False,
        "natural_kmeans_used_for_selection": False,
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


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("local-bridge freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_core_field_rev9.py",
        "src/topic4_zm_ictal_transition.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("local-bridge runtime paths are dirty")
    return {"git_commit": expected, "tracked_modules": tracked, "dirty": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"local-bridge input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_ak_manifest"], loaded["stage_ak_result"],
        config["dual_node_local_bridge"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_dual_node_local_bridge_manifest_v1",
        "status": "REV12ND_DUAL_NODE_LOCAL_BRIDGE_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "mapping_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "soft_objective": config["soft_objective"],
        "pareto_selection": config["pareto_selection"],
        "inputs": inputs,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "n_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
