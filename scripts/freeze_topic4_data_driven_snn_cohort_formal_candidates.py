#!/usr/bin/env python3
"""Freeze the shared formal cohort candidate library before any patient score."""
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

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
EXPECTED_ROLE = (
    "formal_34_subject_canonical_layout_cohort_with_28_subject_real_geometry_sensitivity"
)
NLC_SELECTION_MANIFEST = (
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "joint_node_connectivity_selection/candidate_manifest.json"
)
SOURCE_CANDIDATE_ID = "joint_04_control"


def _transformed_node(source: dict, rotation_deg: int, reflected: bool) -> dict:
    if source.get("field_type") != "spline_continuous":
        raise ValueError("the formal cohort requires a continuous spline source field")
    if int(rotation_deg) % 90:
        raise ValueError("field rotations must be multiples of 90 degrees")
    n_basis = int(source["n_basis"])
    coefficients = np.asarray(source["coefficients"], float).reshape(n_basis, n_basis)
    if reflected:
        coefficients = np.fliplr(coefficients)
    coefficients = np.rot90(coefficients, k=(int(rotation_deg) // 90) % 4)
    node = deepcopy(source)
    node.update({
        "candidate_id": (
            f"field_rot{int(rotation_deg):03d}_{'ref' if reflected else 'idn'}"
        ),
        "coefficients": coefficients.tolist(),
        "source_field_sha256": source["field_sha256"],
        "transform": {
            "rotation_deg": int(rotation_deg),
            "reflection": bool(reflected),
            "uses_patient_target": False,
            "uses_contact_geometry": False,
        },
        "role": "shared_pretrained_continuous_morphology",
    })
    node["field_sha256"] = continuous_candidate_hash({
        key: node[key] for key in ("field_type", "n_basis", "degree", "coefficients")
    })
    return node


def _arm_coefficients(source_edge: np.ndarray, arm: str) -> np.ndarray:
    """E->E is row 0 and E->I is row 1 of the pathway coefficient matrix."""
    if source_edge.ndim != 2 or source_edge.shape[0] != 2:
        raise ValueError("edge coefficients must be (pathway=2, feature)")
    if arm == "Node":
        return np.zeros_like(source_edge)
    if arm == "Node+EE":
        edge = np.zeros_like(source_edge)
        edge[0] = source_edge[0]
        return edge
    if arm == "Node+EE+EtoI":
        return source_edge.copy()
    raise ValueError(f"unsupported formal cohort arm: {arm}")


def candidate_library(config: dict, source_manifest: dict) -> list[dict]:
    matches = [
        row for row in source_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == SOURCE_CANDIDATE_ID
    ]
    if len(matches) != 1:
        raise RuntimeError("the pretrained NLC source candidate is unavailable")
    source = matches[0]
    source_edge = np.asarray(source["coefficients"], float)
    library = config["candidate_library"]
    rows = []
    for rotation in library["field_transforms"]["rotations_deg"]:
        for reflected in library["field_transforms"]["reflections"]:
            node = _transformed_node(source["node_field"], int(rotation), bool(reflected))
            for arm in library["arms"]:
                edge = _arm_coefficients(source_edge, arm)
                slug = {"Node": "node", "Node+EE": "ee", "Node+EE+EtoI": "joint"}[arm]
                rows.append({
                    "candidate_id": (
                        f"rot{int(rotation):03d}"
                        f"{'ref' if reflected else 'idn'}_{slug}"
                    ),
                    "arm": arm,
                    "node_field": node,
                    "coefficients": edge.tolist(),
                    "coefficients_sha256": array_sha256(edge),
                    "raw_logit_clip": float(source.get("raw_logit_clip", 0.75)),
                    "spatial_ou": deepcopy(source.get("spatial_ou", {"mode": "off"})),
                    "pretrained_source_candidate_id": SOURCE_CANDIDATE_ID,
                    "patient_conditioning": "candidate_selection_only",
                })
    identifiers = [row["candidate_id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("formal candidate identifiers are not unique")
    if len(rows) != int(library["n_candidates"]):
        raise RuntimeError(
            f"library holds {len(rows)} candidates, config declares "
            f"{library['n_candidates']}"
        )
    return rows


def build_manifest(config_path: Path, expected_commit: str) -> dict:
    config_path = config_path.resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("formal cohort scientific role changed")
    for name, record in config["inputs"].items():
        if "sha256" in record and _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"formal input hash changed for {name}")
    source_manifest = json.loads((ROOT / NLC_SELECTION_MANIFEST).read_text())
    if source_manifest.get("status") != (
        "REV11NLC_JOINT_NODE_CONNECTIVITY_SELECTION_LIBRARY_FROZEN"
    ):
        raise RuntimeError("source NLC selection manifest is not frozen")
    layout_audit_path = (
        ROOT / config["output_root"] / "cohort_layout_audit.json"
    )
    layout_audit = json.loads(layout_audit_path.read_text())
    if layout_audit.get("status") != "FORMAL_LAYOUTS_FROZEN_SNN_NOT_RUN":
        raise RuntimeError("formal observation layouts are not frozen")
    if layout_audit["denominators"]["primary_canonical_layout"] != int(
        config["observation"]["primary_denominator"]
    ):
        raise RuntimeError("frozen layouts do not hold the primary denominator")
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
        raise RuntimeError("formal candidate freezer runtime or config is not frozen")
    return {
        "status": "TOPIC4_DATA_DRIVEN_SNN_COHORT_FORMAL_LIBRARY_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "candidate_set": {
            "n_candidates": len(candidates),
            "candidates": candidates,
        },
        "layout_audit": {
            "path": str(layout_audit_path.relative_to(ROOT)),
            "sha256": _sha256(layout_audit_path),
            "denominators": layout_audit["denominators"],
        },
        "fixed_contract": {
            "network_axis_deg": 0.0,
            "primary_layout": config["observation"]["primary_layout"],
            "sensitivity_layout": config["observation"]["sensitivity_layout"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "detector": config["search"]["detector"],
            "readout_contact_chunk": config["search"]["readout_contact_chunk"],
            "selection_protocol": config["selection_protocol"],
            "topology": "frozen",
            "delays": "frozen",
            "incoming_weight_budget": "conserved_per_target_and_pathway",
            "beta": "closed",
            "Z_M": "off",
        },
        "claim_boundary": config["claim_boundary"],
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
        "arms": sorted({row["arm"] for row in payload["candidate_set"]["candidates"]}),
        "output": str(output.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
