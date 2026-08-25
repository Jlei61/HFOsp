#!/usr/bin/env python3
"""Freeze an observation-invariant orthogonal free-field Node screen."""
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
sys.path.insert(0, str(ROOT))

from src.topic4_continuous_field import continuous_surface  # noqa: E402
from src.topic4_node_field_search import (  # noqa: E402
    cosine_sheet_residuals,
    residual_candidate,
    uniform_sheet_grid,
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


def _field_by_id(manifest: dict) -> dict[str, dict]:
    return {
        str(candidate["candidate_id"]): candidate["node_field"]
        for candidate in manifest["candidates"]
    }


def capacity_span_audit(anchor: dict, manual: dict, basis_rows: list[dict], *,
                        grid_per_axis: int) -> dict:
    grid = uniform_sheet_grid(int(grid_per_axis), sheet_mm=20.0)
    anchor_surface = continuous_surface(
        anchor["coefficients"], grid, n_basis=18, degree=3, L=20.0,
    )
    manual_surface = continuous_surface(
        manual["coefficients"], grid, n_basis=18, degree=3, L=20.0,
    )
    anchor_surface = anchor_surface - float(np.mean(anchor_surface))
    manual_surface = manual_surface - float(np.mean(manual_surface))
    mode_surfaces = []
    for row in basis_rows:
        surface = continuous_surface(
            row["coefficients"], grid, n_basis=18, degree=3, L=20.0,
        )
        mode_surfaces.append(surface - float(np.mean(surface)))
    modes = np.column_stack(mode_surfaces)
    delta = manual_surface - anchor_surface
    coordinates, *_ = np.linalg.lstsq(modes, delta, rcond=None)
    fitted = modes @ coordinates
    denominator = float(np.linalg.norm(delta))
    return {
        "role": "capacity_span_diagnostic_only",
        "used_to_orient_modes": False,
        "used_for_candidate_generation": False,
        "used_for_selection": False,
        "projected_norm_fraction": float(np.linalg.norm(fitted) / denominator),
        "projected_energy_fraction": float(np.dot(fitted, fitted) / np.dot(delta, delta)),
        "relative_residual_norm": float(np.linalg.norm(delta - fitted) / denominator),
    }


def build_candidates(source_manifest: dict, source_summary: dict,
                     capacity_manifest: dict, design: dict) -> tuple[list[dict], dict]:
    source_fields = _field_by_id(source_manifest)
    rows = [row for row in source_summary["rows"] if row.get("selection_eligible", True)]
    if not rows:
        raise RuntimeError("corrected Stage-S summary has no selectable field")
    anchor_id = str(design["anchor_candidate_id"])
    if rows[0]["candidate_id"] != anchor_id:
        raise RuntimeError("orthogonal screen anchor is not the corrected Stage-S leader")
    anchor = copy.deepcopy(source_fields[anchor_id])
    anchor["candidate_id"] = "stage_u_anchor"
    anchor["role"] = "corrected_data_driven_anchor"
    anchor["residual_coordinates"] = {
        "observation_coordinates_used": False,
        "source_candidate_id": anchor_id,
        "signed_log_surface_rms": 0.0,
    }
    basis = cosine_sheet_residuals(
        maximum_frequency=int(design["maximum_cosine_frequency"]),
        target_n_basis=int(design["stored_n_basis"]),
        degree=int(design["degree"]),
        projection_grid_per_axis=int(design["projection_grid_per_axis"]),
    )
    if basis["n_modes"] != int(design["expected_mode_count"]):
        raise RuntimeError("orthogonal screen mode count drifted")
    candidates = [{
        "candidate_id": "stage_u_anchor",
        "role": "corrected_data_driven_anchor",
        "source_candidate_ids": [anchor_id],
        "selection_eligible": True,
        "node_field": anchor,
    }]
    amplitude = float(design["amplitude"])
    for row in basis["rows"]:
        for sign, signed_amplitude in (("m", -amplitude), ("p", amplitude)):
            candidate_id = f"stage_u_f{row['mode_index']:02d}_{sign}"
            field = residual_candidate(
                anchor, np.asarray(row["coefficients"], float),
                amplitude=signed_amplitude, candidate_id=candidate_id,
                residual_index=int(row["mode_index"]),
                coarse_n_basis=int(design["maximum_cosine_frequency"]) + 1,
            )
            field["role"] = "orthogonal_cosine_free_field_residual"
            field["residual_coordinates"].update({
                "basis_family": "uniform_sheet_cosine",
                "kx": int(row["kx"]), "ky": int(row["ky"]),
                "observation_coordinates_used": False,
            })
            candidates.append({
                "candidate_id": candidate_id,
                "role": "orthogonal_cosine_free_field_residual",
                "source_candidate_ids": [anchor_id],
                "selection_eligible": True,
                "node_field": field,
            })
    if len(candidates) != int(design["expected_candidate_count"]):
        raise RuntimeError("orthogonal screen candidate count drifted")
    if len({row["node_field"]["field_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("orthogonal screen contains duplicate fields")
    manual = _field_by_id(capacity_manifest)["stage_t_manual_smooth_capacity"]
    audit = {
        "basis": {
            key: value for key, value in basis.items() if key != "rows"
        },
        "modes": [{
            key: value for key, value in row.items() if key != "coefficients"
        } for row in basis["rows"]],
        "capacity_span": capacity_span_audit(
            anchor, manual, basis["rows"],
            grid_per_axis=int(design["projection_grid_per_axis"]),
        ),
    }
    return candidates, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    commit = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != commit:
        raise RuntimeError("orthogonal screen freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_field_search.py", "src/topic4_continuous_field.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_cascade_fit.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("orthogonal screen runtime paths are dirty")
    artifact_root = args.artifact_root.resolve()
    loaded, input_audit = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"orthogonal screen input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["source_manifest"], loaded["source_summary"],
        loaded["capacity_manifest"], config["orthogonal_screen"],
    )
    payload = {
        "schema_id": "topic4_rev12_orthogonal_free_field_screen_manifest_v1",
        "status": "REV12ND_ORTHOGONAL_FREE_FIELD_SCREEN_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "basis_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "cascade_objective": config["cascade_objective"],
        "inputs": input_audit,
        "provenance": {
            "git_commit": commit,
            "tracked_modules": tracked,
            "dirty": False,
        },
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(candidates),
        "basis_modes": audit["basis"]["n_modes"],
        "capacity_energy_fraction": audit["capacity_span"]["projected_energy_fraction"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
