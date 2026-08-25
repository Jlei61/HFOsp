#!/usr/bin/env python3
"""Freeze a small continuous-field interpolation around the balanced donor."""
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
from src.topic4_node_field_search import uniform_sheet_grid  # noqa: E402
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


def _field_map(manifest: dict) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in manifest["candidates"]}


def _summary_map(summary: dict) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in summary["rows"]}


def _validate_donor_evidence(stage_aa_summary: dict, stage_aa_audit: dict,
                             stage_ab_summary: dict, stage_ab_audit: dict,
                             design: dict) -> None:
    if stage_aa_audit.get("status") != (
            "FRESH_FIT_BALANCED_MEAN_CANDIDATE_MODE_STABILITY_UNRESOLVED"):
        raise RuntimeError("Stage-AA donor evidence changed")
    if stage_ab_audit.get("status") != (
            "OMITTED_PARETO_MODE1_ONLY_NO_BALANCED_STABILITY"):
        raise RuntimeError("Stage-AB does not justify a mode-1 donor")
    balanced = str(design["balanced_donor_id"])
    direction = str(design["direction_donor_id"])
    mode_1 = str(design["mode_1_donor_id"])
    if balanced not in stage_aa_audit["balanced_mean_candidate_ids"]:
        raise RuntimeError("balanced donor lost its Stage-AA evidence")
    direction_record = stage_aa_audit[
        "fresh_nine_networks_primary"
    ][direction]["endpoints"]["causal_direction"]
    if (direction_record["mean"] <= 0.0
            or direction_record["positive_networks"] < 6):
        raise RuntimeError("direction donor lost its fresh-network evidence")
    if stage_ab_audit["recovery_candidate_id"] != mode_1:
        raise RuntimeError("mode-1 donor identity changed")
    aa_rows = _summary_map(stage_aa_summary)
    ab_rows = _summary_map(stage_ab_summary)
    for candidate_id in (balanced, direction):
        if candidate_id not in aa_rows or not aa_rows[candidate_id].get("fit_valid"):
            raise RuntimeError(f"invalid Stage-AA donor: {candidate_id}")
    if mode_1 not in ab_rows or not ab_rows[mode_1].get("fit_valid"):
        raise RuntimeError("invalid Stage-AB donor")


def build_candidates(stage_z_manifest: dict, stage_aa_summary: dict,
                     stage_aa_audit: dict, stage_ab_summary: dict,
                     stage_ab_audit: dict, design: dict) -> tuple[list[dict], dict]:
    _validate_donor_evidence(
        stage_aa_summary, stage_aa_audit, stage_ab_summary, stage_ab_audit, design,
    )
    fields = _field_map(stage_z_manifest)
    ids = {
        name: str(design[key]) for name, key in (
            ("base", "base_field_id"),
            ("balanced", "balanced_donor_id"),
            ("mode_1", "mode_1_donor_id"),
            ("direction", "direction_donor_id"),
        )
    }
    if any(candidate_id not in fields for candidate_id in ids.values()):
        raise RuntimeError("an interpolation donor is absent from Stage-Z")
    node_fields = {name: fields[candidate_id]["node_field"] for name, candidate_id in ids.items()}
    shapes = {np.asarray(field["coefficients"], float).shape for field in node_fields.values()}
    if len(shapes) != 1:
        raise RuntimeError("interpolation donor coefficient shapes differ")
    base = np.asarray(node_fields["base"]["coefficients"], float)
    balanced = np.asarray(node_fields["balanced"]["coefficients"], float)
    residual_mode_1 = np.asarray(node_fields["mode_1"]["coefficients"], float) - base
    residual_direction = np.asarray(node_fields["direction"]["coefficients"], float) - base
    grid = uniform_sheet_grid(
        int(design["projection_grid_per_axis"]), sheet_mm=20.0,
    )
    candidates, audits = [], []
    for mode_1_dose in map(float, design["mode_1_doses"]):
        for direction_dose in map(float, design["direction_doses"]):
            if bool(design["exclude_zero_zero"]) and np.isclose(
                    mode_1_dose, 0.0) and np.isclose(direction_dose, 0.0):
                continue
            values = (
                balanced + mode_1_dose * residual_mode_1
                + direction_dose * residual_direction
            )
            if not np.isclose(np.sum(values), np.sum(base), atol=1e-10, rtol=0.0):
                raise RuntimeError("interpolation changed the frozen coefficient budget")
            residual = values - base
            surface = continuous_surface(
                residual, grid, n_basis=int(node_fields["base"]["n_basis"]),
                degree=int(node_fields["base"]["degree"]), L=20.0,
            )
            surface -= float(np.mean(surface))
            observed_rms = float(np.sqrt(np.mean(surface ** 2)))
            if observed_rms > float(design["maximum_residual_surface_rms"]) + 1e-12:
                raise RuntimeError("interpolated residual exceeds the frozen RMS envelope")
            candidate_id = (
                f"stage_ac_m{int(round(100 * mode_1_dose)):03d}_"
                f"d{int(round(100 * direction_dose)):03d}"
            )
            field = {
                "candidate_id": candidate_id,
                "field_type": "spline_continuous",
                "n_basis": int(node_fields["base"]["n_basis"]),
                "degree": int(node_fields["base"]["degree"]),
                "coefficients": values.tolist(),
                "field_sha256": array_sha256(values),
                "roughness": spline_roughness(values),
                "component_count": None,
                "peak_count_constraint": None,
                "role": "local_continuous_donor_interpolation",
                "source_field_sha256": node_fields["base"]["field_sha256"],
                "residual_coordinates": {
                    "basis_family": "stage_z_empirical_donor_directions",
                    "balanced_donor_coefficient": 1.0,
                    "mode_1_dose": mode_1_dose,
                    "direction_dose": direction_dose,
                    "base_field_id": ids["base"],
                    "balanced_donor_id": ids["balanced"],
                    "mode_1_donor_id": ids["mode_1"],
                    "direction_donor_id": ids["direction"],
                    "formula": design["formula"],
                    "observed_surface_rms": observed_rms,
                    "observation_coordinates_used": False,
                },
            }
            candidates.append({
                "candidate_id": candidate_id,
                "role": "local_continuous_donor_interpolation",
                "selection_eligible": True,
                "source_candidate_ids": list(ids.values()),
                "node_field": field,
            })
            audits.append({
                "candidate_id": candidate_id,
                "mode_1_dose": mode_1_dose,
                "direction_dose": direction_dose,
                "observed_surface_rms": observed_rms,
                "field_sha256": field["field_sha256"],
            })
    if len(candidates) != int(design["expected_candidate_count"]):
        raise RuntimeError("interpolation candidate count changed")
    if len({row["node_field"]["field_sha256"] for row in candidates}) != len(candidates):
        raise RuntimeError("interpolation contains duplicate fields")
    return candidates, {
        "donor_ids": ids,
        "formula": design["formula"],
        "candidates": audits,
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
        raise RuntimeError("interpolation freezer is not at expected commit")
    tracked = [
        str(config_path.relative_to(ROOT)),
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/topic4_continuous_field.py",
        "src/topic4_node_field_search.py",
        "src/topic4_observation_invariant_spline.py",
        "src/topic4_node_dualmode.py",
        "scripts/run_topic4_rev12_node_worker.py",
        "scripts/launch_topic4_rev12_node_workers.py",
        "scripts/aggregate_topic4_rev12_soft_global_fit.py",
        "scripts/finish_topic4_rev12_local_donor_interpolation.py",
    ]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *tracked], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError("interpolation runtime paths are dirty")
    for relative in tracked:
        content = subprocess.check_output(
            ["git", "show", f"{expected}:{relative}"], cwd=ROOT,
        )
        if hashlib.sha256(content).hexdigest() != _sha256(ROOT / relative):
            raise RuntimeError(f"interpolation runtime path drifted: {relative}")
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
            raise RuntimeError(f"interpolation input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        input_audit[name] = {"path": str(path), "sha256": observed}
    candidates, audit = build_candidates(
        loaded["stage_z_manifest"], loaded["stage_aa_summary"],
        loaded["stage_aa_paired_audit"], loaded["stage_ab_summary"],
        loaded["stage_ab_paired_audit"], config["interpolation"],
    )
    payload = {
        "schema_id": "topic4_rev12_nd_local_donor_interpolation_manifest_v1",
        "status": "REV12ND_LOCAL_DONOR_INTERPOLATION_FROZEN",
        "config": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "candidates": candidates,
        "interpolation_audit": audit,
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
