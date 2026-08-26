#!/usr/bin/env python3
"""Freeze an orthogonal paired calibration around the data-driven g04 field."""
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
    uniform_sheet_grid,
)
from src.topic4_observation_invariant_spline import (  # noqa: E402
    array_sha256,
    spline_roughness,
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


def build_candidates(stage_aa: dict, stage_ae: dict,
                     design: dict) -> tuple[list[dict], dict]:
    if stage_ae.get("status") != design["required_stage_ae_status"]:
        raise RuntimeError("Stage-AE does not justify orthogonal calibration")
    rows = {row["candidate_id"]: row for row in stage_aa["rows"]}
    center_id = str(design["center_candidate_id"])
    if center_id not in rows or not rows[center_id].get("fit_valid", False):
        raise RuntimeError("orthogonal calibration center is absent or invalid")
    center_record = rows[center_id]["candidate"]
    if not center_record.get("selection_eligible", False):
        raise RuntimeError("orthogonal calibration center is not data-driven selectable")
    center = copy.deepcopy(center_record["node_field"])
    if center.get("field_type") != "spline_continuous":
        raise RuntimeError("orthogonal calibration center is not a spline")
    basis = cosine_sheet_residuals(
        maximum_frequency=int(design["maximum_cosine_frequency"]),
        target_n_basis=int(design["stored_n_basis"]),
        degree=int(design["degree"]),
        projection_grid_per_axis=int(design["projection_grid_per_axis"]),
    )
    if int(basis["n_modes"]) != int(design["expected_basis_modes"]):
        raise RuntimeError("orthogonal basis count changed")
    center_values = np.asarray(center["coefficients"], float)
    radius = float(design["surface_rms_radius"])
    grid = uniform_sheet_grid(int(design["projection_grid_per_axis"]))
    candidates, pair_audit = [], []
    for row in basis["rows"]:
        mode_index = int(row["mode_index"])
        direction = np.asarray(row["coefficients"], float)
        pair = []
        for sign, suffix in ((-1, "m"), (1, "p")):
            residual = sign * radius * direction
            values = center_values + residual
            candidate_id = f"stage_af_c{mode_index:02d}_{suffix}"
            surface = continuous_surface(
                residual, grid, n_basis=int(center["n_basis"]),
                degree=int(center["degree"]), L=20.0,
            )
            surface -= float(np.mean(surface))
            observed_rms = float(np.sqrt(np.mean(surface ** 2)))
            if not np.isclose(observed_rms, radius, rtol=2e-3, atol=1e-6):
                raise RuntimeError("orthogonal residual RMS changed")
            field = {
                "candidate_id": candidate_id,
                "field_type": "spline_continuous",
                "n_basis": int(center["n_basis"]),
                "degree": int(center["degree"]),
                "coefficients": values.tolist(),
                "field_sha256": array_sha256(values),
                "roughness": spline_roughness(values),
                "component_count": None,
                "peak_count_constraint": None,
                "role": "orthogonal_response_calibration_not_selectable",
                "source_field_sha256": center["field_sha256"],
                "residual_coordinates": {
                    "basis_family": "uniform_sheet_cosine_canonical",
                    "mode_index": mode_index,
                    "kx": int(row["kx"]),
                    "ky": int(row["ky"]),
                    "sign": int(sign),
                    "radius": radius,
                    "observed_surface_rms": observed_rms,
                    "observation_coordinates_used": False,
                },
            }
            candidates.append({
                "candidate_id": candidate_id,
                "role": "orthogonal_response_calibration_not_selectable",
                "selection_eligible": False,
                "source_candidate_ids": [center_id],
                "node_field": field,
            })
            pair.append(values)
        midpoint_error = float(np.max(np.abs((pair[0] + pair[1]) / 2.0 - center_values)))
        if midpoint_error > 1e-12:
            raise RuntimeError("orthogonal pair does not reconstruct its center")
        pair_audit.append({
            "mode_index": mode_index,
            "kx": int(row["kx"]), "ky": int(row["ky"]),
            "radius": radius,
            "midpoint_max_abs_error": midpoint_error,
        })
    if len(candidates) != int(design["expected_candidate_count"]):
        raise RuntimeError("orthogonal candidate count changed")
    if len({row["node_field"]["field_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("orthogonal calibration fields are not unique")
    return candidates, {
        "center_candidate_id": center_id,
        "center_field_sha256": center["field_sha256"],
        "basis": {key: value for key, value in basis.items() if key != "rows"},
        "pairs": pair_audit,
        "selection_eligible": False,
        "manual_field_used": False,
        "patient_or_contact_coordinates_used": False,
        "patient_heldout_used": False,
        "natural_kmeans_used": False,
    }


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("orthogonal freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_field_search.py", "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
        "scripts/analyze_topic4_rev12_orthogonal_response_calibration.py",
        "scripts/finish_topic4_rev12_orthogonal_response_calibration.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("orthogonal calibration runtime paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"orthogonal runtime path drifted: {relative}")
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
            raise RuntimeError(f"orthogonal input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_aa_summary"], loaded["stage_ae_audit"],
        config["orthogonal_calibration"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_orthogonal_response_manifest_v1",
        "status": "REV12ND_ORTHOGONAL_RESPONSE_CALIBRATION_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "orthogonal_calibration_audit": audit,
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
        "status": payload["status"],
        "n_candidates": len(candidates),
        "n_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
