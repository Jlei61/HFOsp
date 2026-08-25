#!/usr/bin/env python3
"""Freeze the final fit-only local-curvature canary after Stage-U."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
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


def _candidate_rows(manifest: dict) -> dict[str, dict]:
    rows = {row["candidate_id"]: row for row in manifest["candidates"]}
    if len(rows) != len(manifest["candidates"]):
        raise RuntimeError("source manifest contains duplicate candidates")
    return rows


def _composition_id(composition: list[dict]) -> str:
    return "_".join(
        f"f{int(item['mode_index']):02d}{'p' if float(item['amplitude']) > 0 else 'm'}"
        f"{int(round(abs(float(item['amplitude'])) * 100)):02d}"
        for item in composition
    )


def _combined_field(anchor: dict, basis_by_mode: dict[int, dict], *,
                    composition: list[dict], candidate_id: str,
                    grid_per_axis: int) -> tuple[dict, dict]:
    if not composition:
        raise RuntimeError("local-curvature candidate has an empty composition")
    mode_ids = [int(item["mode_index"]) for item in composition]
    if len(set(mode_ids)) != len(mode_ids):
        raise RuntimeError("local-curvature candidate repeats a basis mode")
    values = np.asarray(anchor["coefficients"], float).copy()
    audit_components = []
    for item in composition:
        mode = int(item["mode_index"])
        amplitude = float(item["amplitude"])
        if mode not in basis_by_mode or amplitude == 0.0:
            raise RuntimeError("local-curvature composition is invalid")
        basis = basis_by_mode[mode]
        values += amplitude * np.asarray(basis["coefficients"], float)
        audit_components.append({
            "mode_index": mode, "kx": int(basis["kx"]),
            "ky": int(basis["ky"]), "amplitude": amplitude,
        })
    grid = uniform_sheet_grid(int(grid_per_axis), sheet_mm=20.0)
    anchor_surface = continuous_surface(
        anchor["coefficients"], grid, n_basis=int(anchor["n_basis"]),
        degree=int(anchor["degree"]), L=20.0,
    )
    candidate_surface = continuous_surface(
        values, grid, n_basis=int(anchor["n_basis"]),
        degree=int(anchor["degree"]), L=20.0,
    )
    residual = candidate_surface - anchor_surface
    residual -= float(np.mean(residual))
    observed_rms = float(np.sqrt(np.mean(residual ** 2)))
    nominal_l2 = float(math.sqrt(sum(
        float(item["amplitude"]) ** 2 for item in composition
    )))
    if not np.isclose(observed_rms, nominal_l2, rtol=2e-3, atol=1e-6):
        raise RuntimeError("local-curvature residual RMS drifted")
    field = {
        "candidate_id": candidate_id,
        "field_type": "spline_continuous",
        "n_basis": int(anchor["n_basis"]),
        "degree": int(anchor["degree"]),
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "roughness": spline_roughness(values),
        "component_count": None,
        "peak_count_constraint": None,
        "role": "orthogonal_local_curvature_canary",
        "source_field_sha256": anchor["field_sha256"],
        "residual_coordinates": {
            "basis_family": "uniform_sheet_cosine",
            "composition": audit_components,
            "nominal_l2_surface_rms": nominal_l2,
            "observed_surface_rms": observed_rms,
            "observation_coordinates_used": False,
        },
    }
    return field, {
        "candidate_id": candidate_id,
        "composition": audit_components,
        "nominal_l2_surface_rms": nominal_l2,
        "observed_surface_rms": observed_rms,
    }


def build_candidates(stage_manifest: dict, analysis: dict,
                     design: dict) -> tuple[list[dict], list[dict]]:
    if analysis.get("status") != "REV12ND_ANCHOR_RELATIVE_LOCAL_AUDIT_COMPLETE":
        raise RuntimeError("anchor-relative audit is incomplete")
    if analysis["outer_amplitude_decision"]["status"] != (
            "NOT_OPENED_NO_ANCHOR_OBJECTIVE_IMPROVEMENT"):
        raise RuntimeError("outer-amplitude decision is not closed")
    stage_rows = _candidate_rows(stage_manifest)
    anchor = copy.deepcopy(stage_rows["stage_u_anchor"]["node_field"])
    basis = cosine_sheet_residuals(
        maximum_frequency=int(design["maximum_cosine_frequency"]),
        target_n_basis=int(design["stored_n_basis"]),
        degree=int(design["degree"]),
        projection_grid_per_axis=int(design["projection_grid_per_axis"]),
    )
    basis_by_mode = {int(row["mode_index"]): row for row in basis["rows"]}
    requested = []
    for row in design["single_mode_canaries"]:
        requested.append({
            "source": "observed_single_mode_attribution",
            "composition": [{
                "mode_index": int(row["mode_index"]),
                "amplitude": float(row["amplitude"]),
            }],
        })
    selected = analysis["local_canary_proposals"]["selected"]
    count = int(design["maximum_surrogate_combinations"])
    if len(selected) < count:
        raise RuntimeError("anchor-relative audit nominated too few combinations")
    for row in selected[:count]:
        requested.append({
            "source": "diagonal_quadratic_surrogate",
            "composition": copy.deepcopy(row["composition"]),
            "predicted_delta": copy.deepcopy(row["predicted_delta"]),
            "surrogate_rank_score": float(row["surrogate_rank_score"]),
        })

    anchor_field = {**anchor, "candidate_id": "stage_w_anchor"}
    anchor_field["role"] = "local_curvature_reference_anchor"
    candidates = [{
        "candidate_id": "stage_w_anchor",
        "role": "local_curvature_reference_anchor",
        "selection_eligible": False,
        "source_candidate_ids": ["stage_u_anchor"],
        "node_field": anchor_field,
    }]
    audit_rows = []
    for request in requested:
        composition = request["composition"]
        candidate_id = f"stage_w_{_composition_id(composition)}"
        field, audit = _combined_field(
            anchor, basis_by_mode, composition=composition,
            candidate_id=candidate_id,
            grid_per_axis=int(design["projection_grid_per_axis"]),
        )
        if audit["nominal_l2_surface_rms"] > (
                float(design["maximum_candidate_l2_amplitude"]) + 1e-12):
            raise RuntimeError("local-curvature candidate exceeds trust radius")
        audit.update({key: value for key, value in request.items()
                      if key != "composition"})
        candidates.append({
            "candidate_id": candidate_id,
            "role": "orthogonal_local_curvature_canary",
            "selection_eligible": False,
            "source_candidate_ids": ["stage_u_anchor"],
            "node_field": field,
        })
        audit_rows.append(audit)
    if len(candidates) != int(design["expected_candidate_count"]):
        raise RuntimeError("local-curvature candidate count drifted")
    field_hashes = [row["node_field"]["field_sha256"] for row in candidates]
    if len(field_hashes) != len(set(field_hashes)):
        raise RuntimeError("local-curvature manifest contains duplicate fields")
    return candidates, audit_rows


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("local-curvature freezer is not at expected commit")
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
        raise RuntimeError("local-curvature runtime paths are dirty")
    hashes = {}
    for relative in tracked:
        expected_content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        observed = _sha256(ROOT / relative)
        if hashlib.sha256(expected_content).hexdigest() != observed:
            raise RuntimeError(f"local-curvature runtime path drifted: {relative}")
        hashes[relative] = observed
    return {"git_commit": expected, "tracked_modules": tracked,
            "path_sha256": hashes, "dirty": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    artifact_root = args.artifact_root.resolve()
    loaded, input_audit = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"local-curvature input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_manifest"], loaded["anchor_relative_analysis"],
        config["local_curvature_screen"],
    )
    payload = {
        "schema_id": "topic4_rev12_local_curvature_canary_manifest_v1",
        "status": "REV12ND_LOCAL_CURVATURE_CANARY_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "local_curvature_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "cascade_objective": config["cascade_objective"],
        "inputs": input_audit,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
