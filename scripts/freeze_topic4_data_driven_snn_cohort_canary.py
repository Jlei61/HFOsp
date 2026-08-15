#!/usr/bin/env python3
"""Freeze the multisubject pretrained-field transfer canary library."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import continuous_candidate_hash  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_canary_v1.json"
EXPECTED_ROLE = (
    "development_only_multisubject_pretrained_continuous_substrate_canary"
)


def _transformed_node(source: dict, rotation_deg: int) -> dict:
    if source.get("field_type") != "spline_continuous":
        raise ValueError("cohort canary requires a continuous spline source field")
    n_basis = int(source["n_basis"])
    coefficients = np.asarray(source["coefficients"], float).reshape(n_basis, n_basis)
    if int(rotation_deg) % 90:
        raise ValueError("canary rotations must be multiples of 90 degrees")
    transformed = np.rot90(coefficients, k=(int(rotation_deg) // 90) % 4)
    node = deepcopy(source)
    node.update({
        "candidate_id": f"pretrained_rot{int(rotation_deg):03d}",
        "coefficients": transformed.tolist(),
        "source_field_sha256": source["field_sha256"],
        "transform": {
            "rotation_deg": int(rotation_deg),
            "reflection": False,
            "uses_patient_target": False,
            "uses_contact_geometry": False,
        },
        "role": "pretrained_E1146_continuous_morphology_transfer",
    })
    node["field_sha256"] = continuous_candidate_hash({
        key: node[key] for key in (
            "field_type", "n_basis", "degree", "coefficients",
        )
    })
    return node


def candidate_library(config: dict, source_manifest: dict) -> list[dict]:
    source_id = config["pretrained_source"]["candidate_id"]
    matches = [
        row for row in source_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == source_id
    ]
    if len(matches) != 1:
        raise RuntimeError("pretrained NLC source candidate is unavailable")
    source = matches[0]
    source_edge = np.asarray(source["coefficients"], float)
    rows = []
    for rotation in config["pretrained_source"]["field_rotations_deg"]:
        node = _transformed_node(source["node_field"], int(rotation))
        for arm in config["pretrained_source"]["arms"]:
            if arm == "Node":
                edge = np.zeros_like(source_edge)
                slug = "node"
            elif arm == "Node+EE+EtoI":
                edge = source_edge.copy()
                slug = "joint"
            else:
                raise ValueError(f"unsupported cohort canary arm: {arm}")
            candidate_id = f"rot{int(rotation):03d}_{slug}"
            rows.append({
                "candidate_id": candidate_id,
                "arm": arm,
                "node_field": node,
                "coefficients": edge.tolist(),
                "coefficients_sha256": array_sha256(edge),
                "raw_logit_clip": float(source.get("raw_logit_clip", 0.75)),
                "spatial_ou": deepcopy(source.get("spatial_ou", {"mode": "off"})),
                "pretrained_source_candidate_id": source_id,
                "patient_conditioning": "candidate_selection_only",
            })
    identifiers = [row["candidate_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("cohort canary candidate identifiers are not unique")
    return rows


def build_manifest(config_path: Path, expected_commit: str) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("cohort canary scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    source_manifest = json.loads(
        (ROOT / config["inputs"]["nlc_selection_manifest"]["path"]).read_text()
    )
    if source_manifest.get("status") != (
        "REV11NLC_JOINT_NODE_CONNECTIVITY_SELECTION_LIBRARY_FROZEN"
    ):
        raise RuntimeError("source NLC selection manifest is not frozen")
    target_audit = json.loads(
        (ROOT / config["inputs"]["cohort_target_audit"]["path"]).read_text()
    )
    eligible = {
        row["subject_id"] for row in target_audit["subjects"]
        if row["snn_eligible"]
    }
    missing = [subject for subject in config["subjects"] if subject not in eligible]
    if missing:
        raise RuntimeError(f"canary subjects are not SNN eligible: {missing}")
    candidates = candidate_library(config, source_manifest)

    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    config_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (config_dirty or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("cohort canary freezer runtime or config is not frozen")
    return {
        "status": "TOPIC4_DATA_DRIVEN_SNN_COHORT_CANARY_LIBRARY_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "candidates": candidates,
        },
        "subjects": list(config["subjects"]),
        "fixed_contract": {
            "network_axis_deg": 0.0,
            "geometry_projection": "geometry_only_pca_3d_to_2d",
            "fit_network_seeds": config["search"]["fit_network_seeds"],
            "selection_network_seeds": config["search"]["selection_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "detector": config["search"]["detector"],
            "topology": "frozen",
            "delays": "frozen",
            "beta": "closed",
            "Z_M": "off",
        },
        "claim_boundary": config["canary_boundary"],
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    payload = build_manifest(args.config, args.expected_commit)
    config = json.loads(args.config.read_text())
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": payload["candidate_set"]["n_candidates"],
        "subjects": payload["subjects"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
