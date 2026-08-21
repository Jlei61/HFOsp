#!/usr/bin/env python3
"""Freeze a small whole-sheet residual library around the canary-selected Node field."""
from __future__ import annotations

import argparse
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
        raise RuntimeError("Stage-A freezer is not at the expected commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("Stage-A freezer paths are dirty")
    hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT,
        )
        hashes[path] = hashlib.sha256(content).hexdigest()
        if hashes[path] != _sha256(ROOT / path):
            raise RuntimeError(f"Stage-A freezer path drifted: {path}")
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
    if config["scientific_role"] != "development_only_node_dualmode_refit":
        raise RuntimeError("Stage-A scientific role changed")
    for record in config["inputs"].values():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Stage-A input changed: {record['path']}")
    provenance = _provenance(config_path, args.expected_commit)
    canary_summary_path = _resolve(
        artifact_root, config["inputs"]["canary_summary"]["path"],
    )
    canary_manifest_path = _resolve(
        artifact_root, config["inputs"]["canary_manifest"]["path"],
    )
    summary = json.loads(canary_summary_path.read_text())
    manifest = json.loads(canary_manifest_path.read_text())
    expected_networks = len(summary["requested_seeds"])
    eligible = [
        row for row in summary["rows"]
        if int(row["n_networks"]) == expected_networks
    ]
    if len(eligible) != len(manifest["candidates"]):
        raise RuntimeError("Stage-A requires complete canary coverage for every field")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["score"]["mean_patient_objective"]),
            float(row["score"]["mean_weakest_mode_lse"]),
            row["candidate_id"],
        ),
    )
    source = next(
        row for row in manifest["candidates"]
        if row["candidate_id"] == selected["candidate_id"]
    )
    anchor = dict(source["node_field"])
    base = {
        "candidate_id": "stage_a_base",
        "role": "canary_selected_whole_sheet_reference",
        "node_field": anchor,
        "source_canary_candidate_id": selected["candidate_id"],
    }
    design = config["field_search"]
    n_residuals = int(design["stage_a_residual_count"]) // 2
    amplitude = float(design["stage_a_signed_log_surface_rms"])
    coarse = sobol_coarse_residuals(
        n_residuals=n_residuals, n_basis=4, seed=int(design["sobol_seed"]),
    )
    candidates = [base]
    projection_audit = []
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
                candidate_id=f"stage_a_r{index:02d}_{token}",
                residual_index=index,
            )
            candidates.append({
                "candidate_id": node["candidate_id"],
                "role": "whole_sheet_continuous_residual_fit",
                "node_field": node,
            })
    expected = 1 + int(design["stage_a_residual_count"])
    if len(candidates) != expected:
        raise RuntimeError("Stage-A candidate count changed")
    hashes = [row["node_field"]["field_sha256"] for row in candidates]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Stage-A generated duplicate fields")
    payload = {
        "schema_id": "topic4_rev12_nd_node_stage_a_manifest_v1",
        "status": "REV12ND_NODE_STAGE_A_FIELDS_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "selected_canary_base": {
            "candidate_id": selected["candidate_id"],
            "patient_objective": selected["score"]["mean_patient_objective"],
            "weakest_mode_lse": selected["score"]["mean_weakest_mode_lse"],
            "selection_uses_patient_training_and_model_internal_metrics_only": True,
        },
        "projection_audit": projection_audit,
        "representation": {
            "coarse_control_grid": [4, 4],
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
        "selected_base": selected["candidate_id"],
        "n_candidates": len(candidates),
    }, indent=2))


if __name__ == "__main__":
    main()
