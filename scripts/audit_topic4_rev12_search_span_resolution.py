#!/usr/bin/env python3
"""Audit whether prior Node searches reached the scale and span of capacity."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _field_surface(field: dict, grid: np.ndarray) -> np.ndarray:
    values = continuous_surface(
        np.asarray(field["coefficients"], float), grid,
        n_basis=int(field["n_basis"]), degree=int(field["degree"]), L=20.0,
    )
    return values - float(np.mean(values))


def project_difference(anchor: dict, capacity: dict, *, maximum_frequency: int,
                       grid_per_axis: int, stored_n_basis: int,
                       degree: int) -> dict:
    """Project one field contrast onto an observation-invariant cosine span."""
    grid = uniform_sheet_grid(int(grid_per_axis), sheet_mm=20.0)
    difference = _field_surface(capacity, grid) - _field_surface(anchor, grid)
    basis = cosine_sheet_residuals(
        maximum_frequency=int(maximum_frequency),
        target_n_basis=int(stored_n_basis), degree=int(degree),
        projection_grid_per_axis=int(grid_per_axis),
    )
    columns = []
    for row in basis["rows"]:
        surface = continuous_surface(
            np.asarray(row["coefficients"], float), grid,
            n_basis=int(stored_n_basis), degree=int(degree), L=20.0,
        )
        columns.append(surface - float(np.mean(surface)))
    matrix = np.column_stack(columns)
    coefficients = matrix.T @ difference / len(difference)
    predicted = matrix @ coefficients
    residual = difference - predicted
    total = float(np.sum(difference ** 2))
    return {
        "maximum_frequency": int(maximum_frequency),
        "n_modes": int(matrix.shape[1]),
        "difference_surface_rms": float(np.sqrt(np.mean(difference ** 2))),
        "projected_surface_rms": float(np.sqrt(np.mean(predicted ** 2))),
        "residual_surface_rms": float(np.sqrt(np.mean(residual ** 2))),
        "explained_fraction": float(1.0 - np.sum(residual ** 2) / total),
        "projection_coefficient_l2": float(np.linalg.norm(coefficients)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text())
    artifact_root = args.artifact_root.resolve()
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"span-audit input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    if loaded["stage_af_crossvalidation"].get("status") != (
            "CROSSVALIDATED_COMMON_DIRECTION_NOT_SUPPORTED"):
        raise RuntimeError("Stage-AF did not close the local linear proposal")
    rows = {row["candidate_id"]: row for row in loaded["stage_aa_summary"]["rows"]}
    anchor = rows[str(config["anchor_candidate_id"])]["candidate"]["node_field"]
    capacity_candidates = loaded["manual_capacity_manifest"]["candidates"]
    if len(capacity_candidates) != 1 or capacity_candidates[0].get(
            "selection_eligible", True):
        raise RuntimeError("manual capacity ruler is absent or selectable")
    capacity = capacity_candidates[0]["node_field"]
    projections = [
        project_difference(
            anchor, capacity, maximum_frequency=frequency,
            grid_per_axis=int(config["projection_grid_per_axis"]),
            stored_n_basis=int(config["stored_n_basis"]),
            degree=int(config["degree"]),
        )
        for frequency in config["maximum_frequencies"]
    ]
    broad = projections[-1]
    previous_radius = float(config["previous_maximum_search_rms"])
    status = (
        "PREVIOUS_SEARCH_DID_NOT_REACH_CAPACITY_SCALE_BROAD_SPAN_ADEQUATE"
        if broad["explained_fraction"] >= float(
            config["minimum_explained_fraction_for_broad_span"]
        ) and previous_radius < broad["difference_surface_rms"]
        else "SEARCH_SPAN_REVISION_NOT_JUSTIFIED"
    )
    payload = {
        "schema_id": config["schema_id"],
        "status": status,
        "anchor_candidate_id": config["anchor_candidate_id"],
        "projections": projections,
        "previous_maximum_search_rms": previous_radius,
        "stage_af_radius": float(config["stage_af_radius"]),
        "previous_radius_to_capacity_difference_ratio": (
            previous_radius / broad["difference_surface_rms"]
        ),
        "manual_capacity_selection_eligible": False,
        "manual_coefficients_used_for_candidate_generation": False,
        "patient_heldout_used": False,
        "simulation_run": False,
        "inputs": inputs,
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": status, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
