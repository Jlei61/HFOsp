#!/usr/bin/env python3
"""Freeze Stage-C settled-episode continuous Node candidates."""
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
    interpolate_spline_candidates,
    residual_candidate,
    sobol_coarse_residuals,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


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
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if head != expected:
        raise RuntimeError("Stage-C freezer is not at the expected commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("Stage-C freezer paths are dirty")
    hashes = {}
    for path in paths:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT,
        )
        hashes[path] = hashlib.sha256(content).hexdigest()
        if hashes[path] != _sha256(ROOT / path):
            raise RuntimeError(f"Stage-C freezer path drifted: {path}")
    return {"git_commit": expected, "path_sha256": hashes}


def build_stage_c_candidates(source_manifest: dict, design: dict) -> list[dict]:
    fields = {
        str(row["candidate_id"]): row["node_field"]
        for row in source_manifest["candidates"]
    }
    required = list(design["source_candidate_ids"])
    if any(candidate_id not in fields for candidate_id in required):
        raise RuntimeError("Stage-C source candidate is absent")
    r02, r01, r03 = (fields[candidate_id] for candidate_id in required)
    candidates = []
    for candidate_id, field, role in (
        ("stage_c_r02_anchor", r02, "best_settled_episode_patient_fit"),
        ("stage_c_r01_anchor", r01, "best_settled_episode_direction_alignment"),
        ("stage_c_r03_anchor", r03, "third_development_reference"),
    ):
        node = interpolate_spline_candidates(
            field, field, weight=0.0, candidate_id=candidate_id,
        )
        candidates.append({"candidate_id": candidate_id, "role": role, "node_field": node})
    for index, weight in enumerate(design["interpolation_weights_toward_r01"]):
        candidate_id = f"stage_c_blend_{index:02d}"
        node = interpolate_spline_candidates(
            r02, r01, weight=float(weight), candidate_id=candidate_id,
        )
        candidates.append({
            "candidate_id": candidate_id,
            "role": "patient_fit_to_direction_alignment_interpolation",
            "node_field": node,
        })
    midpoint = interpolate_spline_candidates(
        r02, r01,
        weight=float(design["residual_anchor_weight_toward_r01"]),
        candidate_id="stage_c_residual_anchor",
    )
    signed_count = int(design["signed_residual_count"])
    if signed_count % 2:
        raise ValueError("signed residual count must be even")
    controls = sobol_coarse_residuals(
        n_residuals=signed_count // 2,
        n_basis=int(design["residual_control_grid"][0]),
        seed=int(design["sobol_seed"]),
    )
    amplitude = float(design["signed_log_surface_rms"])
    for index, control in enumerate(controls):
        projected = coarse_residual_to_coefficients(
            control, target_n_basis=int(midpoint["n_basis"]),
            degree=int(midpoint["degree"]), sheet_mm=20.0,
        )
        for sign, token in ((-1.0, "m"), (1.0, "p")):
            candidate_id = f"stage_c_r{index:02d}_{token}"
            node = residual_candidate(
                midpoint, projected["coefficients"],
                amplitude=sign * amplitude, candidate_id=candidate_id,
                residual_index=index,
                coarse_n_basis=int(design["residual_control_grid"][0]),
            )
            candidates.append({
                "candidate_id": candidate_id,
                "role": "settled_episode_whole_sheet_local_residual",
                "node_field": node,
            })
    hashes = [row["node_field"]["field_sha256"] for row in candidates]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("Stage-C generated duplicate fields")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"Stage-C input changed: {record['path']}")
        inputs[key] = path
    audit = json.loads(inputs["settled_episode_audit"].read_text())
    if (audit["status"] != "REV12ND_EVENT_FRAGMENTATION_AUDIT_COMPLETE"
            or float(audit["settle_consistent_primary_sensitivity_ms"]) != 50.0):
        raise RuntimeError("Stage-C event-unit audit is not frozen at 50 ms")
    confirmation_root = "node_confirmation"
    if confirmation_root in str(inputs["source_manifest"]):
        raise RuntimeError("Stage-C cannot inherit the opened confirmation pool")
    source_manifest = json.loads(inputs["source_manifest"].read_text())
    candidates = build_stage_c_candidates(source_manifest, config["field_search"])
    payload = {
        "schema_id": "topic4_rev12_nd_node_stage_c_manifest_v1",
        "status": "REV12ND_NODE_STAGE_C_FIELDS_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "event_unit": config["event_unit"],
        "representation": {
            "stored_spline_grid": [18, 18],
            "observation_coordinates_used": False,
            "component_or_peak_count": None,
        },
        "inputs": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in inputs.items()
        },
        "provenance": _provenance(config_path, args.expected_commit),
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "n_candidates": len(candidates),
    }, indent=2))


if __name__ == "__main__":
    main()
