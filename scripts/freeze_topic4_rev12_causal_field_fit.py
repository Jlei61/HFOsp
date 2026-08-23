#!/usr/bin/env python3
"""Freeze observation-invariant continuous fields under the causal event contract."""
from __future__ import annotations

import argparse
import hashlib
import itertools
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


def build_causal_field_candidates(source: dict, design: dict) -> list[dict]:
    rows = {
        str(row["candidate_id"]): row["node_field"]
        for row in source["candidates"]
    }
    source_ids = list(design["source_candidate_ids"])
    if any(candidate_id not in rows for candidate_id in source_ids):
        raise RuntimeError("causal fit anchor is absent from the source manifest")
    anchors = [rows[candidate_id] for candidate_id in source_ids]
    candidates = []
    for index, (source_id, anchor) in enumerate(zip(source_ids, anchors)):
        candidate_id = f"stage_i_anchor_{index:02d}"
        field = interpolate_spline_candidates(
            anchor, anchor, weight=0.0, candidate_id=candidate_id,
        )
        candidates.append({
            "candidate_id": candidate_id,
            "role": "causal_historical_anchor",
            "source_candidate_ids": [source_id],
            "node_field": field,
        })
    if bool(design["pairwise_midpoints"]):
        for midpoint_index, (left_index, right_index) in enumerate(
                itertools.combinations(range(len(anchors)), 2)):
            candidate_id = f"stage_i_midpoint_{midpoint_index:02d}"
            field = interpolate_spline_candidates(
                anchors[left_index], anchors[right_index], weight=0.5,
                candidate_id=candidate_id,
            )
            candidates.append({
                "candidate_id": candidate_id,
                "role": "causal_anchor_midpoint",
                "source_candidate_ids": [
                    source_ids[left_index], source_ids[right_index],
                ],
                "node_field": field,
            })
    grid = int(design["residual_control_grid"][0])
    if design["residual_control_grid"] != [grid, grid]:
        raise ValueError("causal residual grid must be square")
    directions = sobol_coarse_residuals(
        n_residuals=int(design["residual_directions_per_anchor"]),
        n_basis=grid, seed=int(design["sobol_seed"]),
    )
    for anchor_index, (source_id, anchor) in enumerate(zip(source_ids, anchors)):
        for direction_index, direction in enumerate(directions):
            projected = coarse_residual_to_coefficients(
                direction, target_n_basis=int(anchor["n_basis"]),
                degree=int(anchor["degree"]), sheet_mm=20.0,
            )
            for amplitude_index, magnitude in enumerate(
                    design["signed_log_surface_rms"]):
                for sign, sign_token in ((-1.0, "m"), (1.0, "p")):
                    candidate_id = (
                        f"stage_i_a{anchor_index:02d}_d{direction_index:02d}_"
                        f"s{amplitude_index:02d}_{sign_token}"
                    )
                    field = residual_candidate(
                        anchor, projected["coefficients"],
                        amplitude=sign * float(magnitude),
                        candidate_id=candidate_id,
                        residual_index=direction_index,
                        coarse_n_basis=grid,
                    )
                    candidates.append({
                        "candidate_id": candidate_id,
                        "role": "causal_whole_sheet_smooth_residual",
                        "source_candidate_ids": [source_id],
                        "node_field": field,
                    })
    expected = int(design["expected_candidate_count"])
    identifiers = [row["candidate_id"] for row in candidates]
    hashes = [row["node_field"]["field_sha256"] for row in candidates]
    if len(candidates) != expected or len(set(identifiers)) != expected:
        raise RuntimeError("causal field candidate count or identifiers drifted")
    if len(set(hashes)) != expected:
        raise RuntimeError("causal field freezer generated duplicate fields")
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
    expected = subprocess.check_output(
        ["git", "rev-parse", args.expected_commit], cwd=ROOT, text=True,
    ).strip()
    if subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip() != expected:
        raise RuntimeError("causal field freezer is not at the expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/aggregate_topic4_rev12_cascade_fit.py",
        "src/topic4_node_field_search.py",
        "src/topic4_node_dualmode.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("causal field runtime paths are dirty")
    inputs = {}
    for key, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"causal field input changed: {record['path']}")
        inputs[key] = {"path": str(path), "sha256": digest}
    canary = json.loads(Path(
        inputs["native_lineage_canary_audit"]["path"]
    ).read_text())
    if canary["status"] != "REV12ND_NATIVE_LINEAGE_CANARY_PARITY_COMPLETE":
        raise RuntimeError("native causal-lineage parity has not passed")
    source = json.loads(Path(inputs["source_manifest"]["path"]).read_text())
    candidates = build_causal_field_candidates(source, config["field_search"])
    payload = {
        "schema_id": "topic4_rev12_nd_causal_field_fit_manifest_v1",
        "status": "REV12ND_CAUSAL_FIELD_FIT_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "cascade_objective": config["cascade_objective"],
        "representation": {
            "stored_spline_grid": [18, 18],
            "residual_control_grid": config["field_search"][
                "residual_control_grid"
            ],
            "observation_coordinates_used": False,
            "component_or_peak_count": None,
        },
        "inputs": inputs,
        "provenance": {"git_commit": expected},
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
