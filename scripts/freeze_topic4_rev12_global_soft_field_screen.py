#!/usr/bin/env python3
"""Freeze a global observation-invariant continuous Node-field screen."""
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
    sobol_cosine_combinations,
    uniform_sheet_grid,
)
from src.topic4_observation_invariant_spline import (  # noqa: E402
    array_sha256,
    spline_roughness,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def build_candidates(stage_u: dict, soft_audit: dict,
                     design: dict) -> tuple[list[dict], dict]:
    if soft_audit.get("status") != (
            "SOFT_MODE_TARGET_CONTROLS_PASS_ZERO_SIM_RESCORE_COMPLETE"):
        raise RuntimeError("continuous patient-mode target controls did not pass")
    source = {row["candidate_id"]: row for row in stage_u["candidates"]}
    anchor_id = str(design["anchor_candidate_id"])
    benchmark_id = str(design["benchmark_candidate_id"])
    if anchor_id not in source or benchmark_id not in source:
        raise RuntimeError("global screen source fields are absent")
    anchor = copy.deepcopy(source[anchor_id]["node_field"])
    benchmark = copy.deepcopy(source[benchmark_id]["node_field"])
    anchor["candidate_id"] = "stage_z_anchor"
    anchor["role"] = "global_screen_anchor"
    benchmark["candidate_id"] = "stage_z_soft_benchmark"
    benchmark["role"] = "historical_soft_score_benchmark"
    candidates = [
        {
            "candidate_id": "stage_z_anchor", "role": "global_screen_anchor",
            "selection_eligible": True, "source_candidate_ids": [anchor_id],
            "node_field": anchor,
        },
        {
            "candidate_id": "stage_z_soft_benchmark",
            "role": "historical_soft_score_benchmark",
            "selection_eligible": True, "source_candidate_ids": [benchmark_id],
            "node_field": benchmark,
        },
    ]
    basis = cosine_sheet_residuals(
        maximum_frequency=int(design["maximum_cosine_frequency"]),
        target_n_basis=int(design["stored_n_basis"]),
        degree=int(design["degree"]),
        projection_grid_per_axis=int(design["projection_grid_per_axis"]),
    )
    combinations = sobol_cosine_combinations(
        n_pairs=int(design["n_antithetic_pairs"]),
        n_modes=int(basis["n_modes"]),
        radii=tuple(design["radii"]), seed=int(design["sobol_seed"]),
    )
    anchor_values = np.asarray(anchor["coefficients"], float)
    grid = uniform_sheet_grid(
        int(design["projection_grid_per_axis"]), sheet_mm=20.0,
    )
    basis_values = [np.asarray(row["coefficients"], float) for row in basis["rows"]]
    audits = []
    for row in combinations:
        residual = np.sum(
            np.asarray(row["coefficients"])[:, None, None]
            * np.asarray(basis_values),
            axis=0,
        )
        values = anchor_values + residual
        sign = "m" if int(row["sign"]) < 0 else "p"
        candidate_id = f"stage_z_g{int(row['pair_index']):02d}_{sign}"
        surface = continuous_surface(
            residual, grid, n_basis=int(anchor["n_basis"]),
            degree=int(anchor["degree"]), L=20.0,
        )
        surface -= float(np.mean(surface))
        observed_rms = float(np.sqrt(np.mean(surface ** 2)))
        if not np.isclose(observed_rms, float(row["radius"]), rtol=2e-3, atol=1e-6):
            raise RuntimeError("global residual RMS drifted from its frozen radius")
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
            "role": "global_observation_invariant_cosine_combination",
            "source_field_sha256": anchor["field_sha256"],
            "residual_coordinates": {
                "basis_family": "uniform_sheet_cosine",
                "pair_index": int(row["pair_index"]),
                "sign": int(row["sign"]),
                "radius": float(row["radius"]),
                "coefficients": np.asarray(row["coefficients"]).tolist(),
                "observed_surface_rms": observed_rms,
                "observation_coordinates_used": False,
            },
        }
        candidates.append({
            "candidate_id": candidate_id,
            "role": "global_observation_invariant_cosine_combination",
            "selection_eligible": True,
            "source_candidate_ids": [anchor_id],
            "node_field": field,
        })
        audits.append({
            "candidate_id": candidate_id,
            "pair_index": int(row["pair_index"]),
            "sign": int(row["sign"]),
            "radius": float(row["radius"]),
            "observed_surface_rms": observed_rms,
        })
    if len(candidates) != int(design["expected_candidate_count"]):
        raise RuntimeError("global screen candidate count drifted")
    if len({row["node_field"]["field_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("global screen contains duplicate fields")
    return candidates, {
        "basis": {key: value for key, value in basis.items() if key != "rows"},
        "combinations": audits,
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
        raise RuntimeError("global screen freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_node_field_search.py", "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
        "scripts/finish_topic4_rev12_global_soft_field_screen.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("global screen runtime paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"global screen runtime path drifted: {relative}")
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
    loaded, input_audit = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"global screen input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_u_manifest"], loaded["soft_target_audit"],
        config["global_field_screen"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_global_soft_field_screen_manifest_v1",
        "status": "REV12ND_GLOBAL_SOFT_FIELD_SCREEN_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "global_field_audit": audit,
        "event_unit": config["event_unit"],
        "contact_readout": config["search"]["contact_readout"],
        "soft_objective": config["soft_objective"],
        "pareto_selection": config["pareto_selection"],
        "inputs": input_audit,
        "provenance": _provenance(config_path, args.expected_commit),
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "n_candidates": len(candidates),
        "n_fit_networks": len(config["search"]["fit_network_seeds"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
