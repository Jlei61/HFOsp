#!/usr/bin/env python3
"""Freeze the fit-only outer-amplitude check selected by Stage-U."""
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

from src.topic4_node_field_search import (  # noqa: E402
    cosine_sheet_residuals,
    residual_candidate,
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


def _candidate_by_id(manifest: dict) -> dict[str, dict]:
    rows = {
        str(row["candidate_id"]): row for row in manifest["candidates"]
    }
    if len(rows) != len(manifest["candidates"]):
        raise RuntimeError("Stage-U manifest contains duplicate candidate ids")
    return rows


def build_outer_candidates(stage_manifest: dict, analysis: dict,
                           design: dict) -> tuple[list[dict], list[dict]]:
    """Rebuild only the preselected single-mode outer-amplitude fields."""
    if analysis.get("status") != (
            "REV12ND_ORTHOGONAL_MODE_RESPONSE_ANALYSIS_COMPLETE"):
        raise RuntimeError("Stage-U mode-response analysis is incomplete")
    stage_rows = _candidate_by_id(stage_manifest)
    if "stage_u_anchor" not in stage_rows:
        raise RuntimeError("Stage-U anchor is absent")
    anchor = copy.deepcopy(stage_rows["stage_u_anchor"]["node_field"])
    followup = analysis["outer_followup"]
    selected = list(followup["selected_for_outer_amplitude"])
    maximum = int(design["maximum_outer_followup_modes"])
    if not selected or len(selected) > maximum:
        raise RuntimeError("Stage-U selected an invalid number of outer modes")
    outer_amplitude = float(followup["outer_amplitude"])
    if not np.isclose(outer_amplitude, float(design["outer_amplitude"]),
                      rtol=0.0, atol=1e-12):
        raise RuntimeError("outer amplitude differs from the frozen design")

    basis = cosine_sheet_residuals(
        maximum_frequency=int(design["maximum_cosine_frequency"]),
        target_n_basis=int(design["stored_n_basis"]),
        degree=int(design["degree"]),
        projection_grid_per_axis=int(design["projection_grid_per_axis"]),
    )
    by_mode = {int(row["mode_index"]): row for row in basis["rows"]}
    candidates = [{
        "candidate_id": "stage_v_anchor",
        "role": "outer_amplitude_reference_anchor",
        "selection_eligible": False,
        "source_candidate_ids": ["stage_u_anchor"],
        "node_field": {**anchor, "candidate_id": "stage_v_anchor"},
    }]
    audit_rows = []
    seen = set()
    for record in selected:
        mode = int(record["mode_index"])
        orientation = int(record["orientation"])
        if orientation not in {-1, 1} or mode not in by_mode:
            raise RuntimeError("Stage-U selected an invalid mode orientation")
        key = (mode, orientation)
        if key in seen:
            raise RuntimeError("Stage-U selected a duplicate mode orientation")
        seen.add(key)
        basis_row = by_mode[mode]
        if (int(record["kx"]), int(record["ky"])) != (
                int(basis_row["kx"]), int(basis_row["ky"])):
            raise RuntimeError("Stage-U mode coordinates drifted")
        token = "p" if orientation > 0 else "m"
        candidate_id = f"stage_v_f{mode:02d}_{token}_a{int(round(100 * outer_amplitude)):02d}"
        field = residual_candidate(
            anchor, np.asarray(basis_row["coefficients"], float),
            amplitude=float(orientation) * outer_amplitude,
            candidate_id=candidate_id, residual_index=mode,
            coarse_n_basis=int(design["maximum_cosine_frequency"]) + 1,
        )
        field["role"] = "orthogonal_outer_amplitude_scale_check"
        field["residual_coordinates"].update({
            "basis_family": "uniform_sheet_cosine",
            "kx": int(basis_row["kx"]), "ky": int(basis_row["ky"]),
            "orientation": orientation,
            "observation_coordinates_used": False,
        })
        candidates.append({
            "candidate_id": candidate_id,
            "role": "orthogonal_outer_amplitude_scale_check",
            "selection_eligible": False,
            "source_candidate_ids": ["stage_u_anchor"],
            "node_field": field,
        })
        audit_rows.append({
            "candidate_id": candidate_id,
            "mode_index": mode, "kx": int(basis_row["kx"]),
            "ky": int(basis_row["ky"]), "orientation": orientation,
            "selection_reasons": list(record["selection_reasons"]),
            "predicted_normalized_effects": record["normalized_effects"],
            "predicted_balanced_score": float(record["balanced_score"]),
        })
    if len({row["node_field"]["field_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("outer-amplitude manifest contains duplicate fields")
    return candidates, audit_rows


def _provenance(config_path: Path, expected_commit: str) -> dict:
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_field_search.py", "src/topic4_continuous_field.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("outer-amplitude freezer paths are dirty")
    hashes = {}
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        hashes[relative] = hashlib.sha256(content).hexdigest()
        if hashes[relative] != _sha256(ROOT / relative):
            raise RuntimeError(f"outer-amplitude freezer path drifted: {relative}")
    return {"git_commit": expected, "path_sha256": hashes, "dirty": False}


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
            raise RuntimeError(f"outer-amplitude input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit_rows = build_outer_candidates(
        loaded["stage_manifest"], loaded["mode_response_analysis"],
        config["outer_amplitude_check"],
    )
    payload = {
        "schema_id": "topic4_rev12_orthogonal_outer_amplitude_manifest_v1",
        "status": "REV12ND_ORTHOGONAL_OUTER_AMPLITUDE_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "outer_amplitude_audit": audit_rows,
        "inputs": input_audit,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
        "n_candidates": len(candidates),
    }, indent=2))


if __name__ == "__main__":
    main()
