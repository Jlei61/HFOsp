#!/usr/bin/env python3
"""Freeze local 6x6 whole-sheet residuals around the Stage-A selected field."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(ROOT))

from src.topic4_node_field_search import (  # noqa: E402
    coarse_residual_to_coefficients,
    residual_candidate,
    sobol_coarse_residuals,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative_path: str) -> Path:
    local = ROOT / relative_path
    return local if local.exists() else artifact_root / relative_path


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
    paths = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_field_search.py",
    ]
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("Stage-B freezer is not at the expected commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("Stage-B freezer paths are dirty")
    hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT,
        )
        hashes[path] = hashlib.sha256(content).hexdigest()
        if hashes[path] != _sha256(ROOT / path):
            raise RuntimeError(f"Stage-B freezer path drifted: {path}")
    return {"git_commit": expected, "path_sha256": hashes}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Stage-B input changed: {record['path']}")
    provenance = _provenance(config_path, args.expected_commit)
    decision = json.loads(_resolve(
        artifact_root, config["inputs"]["selection_decision"]["path"],
    ).read_text())
    summary = json.loads(_resolve(
        artifact_root, config["inputs"]["selection_summary"]["path"],
    ).read_text())
    source_manifest = json.loads(_resolve(
        artifact_root, config["inputs"]["selection_manifest"]["path"],
    ).read_text())
    selected_id = decision["selected_candidate_id"]
    rows = {row["candidate_id"]: row for row in summary["rows"]}
    fields = {row["candidate_id"]: row for row in source_manifest["candidates"]}
    selected = rows[selected_id]
    reference = rows["stage_a_base"]
    design = config["field_search"]
    selected_modes = [
        selected["score"]["mean_mode_0_loss"],
        selected["score"]["mean_mode_1_loss"],
    ]
    reference_modes = [
        reference["score"]["mean_mode_0_loss"],
        reference["score"]["mean_mode_1_loss"],
    ]
    if bool(design["parent_must_improve_both_modes"]) and not all(
            new < old for new, old in zip(selected_modes, reference_modes)):
        raise RuntimeError("selected Stage-A parent did not improve both modes")
    selected_topology = selected["source_topology"]
    reference_topology = reference["source_topology"]
    relative_separation = (
        selected_topology["equal_network_between_mode_distance"]
        / reference_topology["equal_network_between_mode_distance"]
    )
    reproducibility_drop = (
        reference_topology["mean_across_network_template_cosine"]
        - selected_topology["mean_across_network_template_cosine"]
    )
    if relative_separation < float(design["minimum_relative_topology_separation"]):
        raise RuntimeError("selected Stage-A parent collapsed source topology")
    if reproducibility_drop > float(design["maximum_reproducibility_drop"]):
        raise RuntimeError("selected Stage-A parent lost topology reproducibility")

    anchor = dict(fields[selected_id]["node_field"])
    candidates = [{
        "candidate_id": "stage_b_base",
        "role": "stage_a_selected_reference",
        "node_field": anchor,
        "source_selection_candidate_id": selected_id,
    }]
    n_signed = int(design["stage_b_residual_count"])
    coarse = sobol_coarse_residuals(
        n_residuals=n_signed // 2, n_basis=6, seed=int(design["sobol_seed"]),
    )
    projection_audit = []
    amplitude = float(design["stage_b_signed_log_surface_rms"])
    for index, control in enumerate(coarse):
        projected = coarse_residual_to_coefficients(
            control, target_n_basis=int(anchor["n_basis"]),
            degree=int(anchor["degree"]), sheet_mm=20.0,
        )
        projection_audit.append({
            "residual_index": index,
            "projection_rmse": projected["projection_rmse"],
            "surface_rms": projected["surface_rms"],
        })
        for sign, token in ((-1.0, "m"), (1.0, "p")):
            node = residual_candidate(
                anchor, projected["coefficients"], amplitude=sign * amplitude,
                candidate_id=f"stage_b_r{index:02d}_{token}",
                residual_index=index, coarse_n_basis=6,
            )
            candidates.append({
                "candidate_id": node["candidate_id"],
                "role": "whole_sheet_continuous_stage_b_residual_fit",
                "node_field": node,
            })
    if len(candidates) != 1 + n_signed:
        raise RuntimeError("Stage-B candidate count changed")
    payload = {
        "schema_id": "topic4_rev12_nd_node_stage_b_manifest_v1",
        "status": "REV12ND_NODE_STAGE_B_FIELDS_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "selected_stage_a_parent": {
            "candidate_id": selected_id,
            "mode_losses": selected_modes,
            "reference_mode_losses": reference_modes,
            "relative_topology_separation": relative_separation,
            "topology_reproducibility_drop": reproducibility_drop,
        },
        "projection_audit": projection_audit,
        "representation": {
            "coarse_control_grid": [6, 6],
            "stored_spline_grid": [18, 18],
            "contact_or_shaft_coordinates_used": False,
            "component_or_peak_count": None,
        },
        "provenance": provenance,
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "selected_parent": selected_id, "n_candidates": len(candidates),
    }, indent=2))


if __name__ == "__main__":
    main()
