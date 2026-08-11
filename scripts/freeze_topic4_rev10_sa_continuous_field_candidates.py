"""Freeze non-component continuous-field candidates and source geometry."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9_node_kick_canary import _candidate  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import (  # noqa: E402
    build_continuous_field_candidates,
    continuous_field_h,
    continuous_surface,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_core_field_stage3 import params_to_h, params_to_q  # noqa: E402
from src.topic4_rev10_sa_canary import (  # noqa: E402
    equal_mode_earliest_shaft_centroid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_continuous_field_canary.json"


def _candidate_field(candidate, positions, target_count, L):
    if candidate["field_type"] == "gaussian_k3_benchmark":
        h = params_to_h(
            candidate["theta"], positions, K=3, L=L,
            target_count=target_count,
        )
        q = params_to_q(candidate["theta"], positions, K=3, L=L)
        return h, np.log(np.maximum(q, 1e-12))
    h, _ = continuous_field_h(
        candidate["coefficients"], positions,
        n_basis=candidate["n_basis"], degree=candidate["degree"],
        L=L, target_count=target_count,
    )
    surface = continuous_surface(
        candidate["coefficients"], positions,
        n_basis=candidate["n_basis"], degree=candidate["degree"], L=L,
    )
    return h, surface


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    assay = config["sa6f_continuous_field"]
    if assay["component_count"] is not None or assay["peak_count_constraint"] is not None:
        raise RuntimeError("continuous field cannot carry a component or peak count")
    inputs = config["inputs"]
    base = _load_json_input(inputs["rev9_base_config"])
    stage = _load_json_input(inputs["stage_config"])
    confirmation = _load_json_input(inputs["frozen_confirmation"])
    contract = _load_json_input(inputs["contact_contract"])
    sa5 = _load_json_input(inputs["sa5_summary"])
    detector_audit = _load_json_input(inputs["rev9_common_detector_audit"])
    if sa5["status"] != "SCL_READOUT_NOT_PRIMARY_LIMIT":
        raise RuntimeError("SA5 has not cleared the observation branch")
    threshold = float(assay["detector"]["population_active_fraction_threshold"])
    if threshold != float(detector_audit["common_detector"]["central_threshold"]):
        raise RuntimeError("common detector changed")

    frozen = _candidate(base)
    confirmed = [row for row in confirmation["candidates"]
                 if row["candidate_id"] == frozen["candidate_id"]]
    if len(confirmed) != 1 or confirmed[0]["theta_sha256"] != frozen["theta_sha256"]:
        raise RuntimeError("frozen K3 benchmark no longer reproduces")

    with np.load(inputs["shaft_aware_target_npz"]["path"], allow_pickle=False) as loaded:
        names = np.asarray(loaded["contact_names"]).astype(str)
        contact_xy = np.asarray(loaded["sheet_xy_mm"], float)
        shaft_ids = np.asarray(loaded["shaft_ids"]).astype(str)
        onsets = np.asarray(loaded["patient_train_onsets"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
    contract_names = np.asarray([
        row["contact_name"] for row in contract["contacts"]
    ]).astype(str)
    contract_xy = np.asarray([
        row["sheet_xy_mm"] for row in contract["contacts"]
    ], float)
    if not np.array_equal(names, contract_names) or not np.array_equal(
            contact_xy, contract_xy):
        raise RuntimeError("SA0 and patient target contact geometry differ")

    design = assay["candidate_design"]
    background = design["background_anchor_contract"]
    for row in design["designs"]:
        if int(row["effective_dof_after_mass_projection"]) != int(row["n_basis"]) ** 2 - 1:
            raise RuntimeError("continuous-field effective DoF contract is inconsistent")
    built = build_continuous_field_candidates(
        contact_xy, shaft_ids, onsets, labels,
        designs=design["designs"], degree=int(design["degree"]),
        L=float(stage["engine"]["L"]),
        fit_options={
            "background_weight": float(background["total_fit_weight"]),
            "background_probability": float(background["baseline_probability"]),
            "background_spacing_mm": float(background["spacing_mm"]),
            "background_exclusion_radius_mm": float(
                background["minimum_distance_from_any_contact_mm"]
            ),
        },
    )
    if list(built["targets"]) != list(design["target_ids"]):
        raise RuntimeError("patient contact target set differs from frozen config")
    candidates = [{
        "candidate_id": "frozen_K3_benchmark",
        "field_type": "gaussian_k3_benchmark",
        "role": "historical_K3_benchmark_not_continuous_family",
        "theta": frozen["theta"],
        "field_sha256": frozen["theta_sha256"],
        "component_count": 3,
    }]
    candidates.extend({**row, "role": (
        "continuous_matched_dof_primary" if row["n_basis"] == 4
        else "continuous_resolution_sensitivity"
    ), "component_count": None} for row in built["candidates"])
    if not (int(design["minimum_unique_candidate_count"]) <= len(candidates)
            <= int(design["maximum_candidate_count"])):
        raise RuntimeError("unique continuous-field candidate count is implausible")

    L = float(stage["engine"]["L"])
    axis = np.linspace(0.0, L, 100)
    xx, yy = np.meshgrid(axis, axis)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    expected_n_e = (float(stage["engine"]["density"]) * L ** 2 * 0.8)
    grid_budget = float(stage["N_core_manual"]) * len(grid) / expected_n_e
    preflight = []
    for row in candidates:
        h, surface = _candidate_field(row, grid, grid_budget, L)
        _, contact_surface = _candidate_field(
            row, contact_xy, max(1.0, 0.25 * len(contact_xy)), L,
        )
        normalized = np.exp(np.clip(
            contact_surface - np.max(contact_surface), -30.0, 0.0,
        ))
        preflight.append({
            "candidate_id": row["candidate_id"],
            "field_sha256": row["field_sha256"],
            "grid_budget": float(h.sum()),
            "grid_max_h": float(h.max()),
            "grid_surface_min": float(surface.min()),
            "grid_surface_max": float(surface.max()),
            "contact_relative_q": normalized.tolist(),
            "mean_relative_q_ICL": float(np.mean(normalized[shaft_ids == "ICL"])),
            "mean_relative_q_SCL": float(np.mean(normalized[shaft_ids == "SCL"])),
        })

    icl = np.flatnonzero(shaft_ids == "ICL")
    scl = np.flatnonzero(shaft_ids == "SCL")
    icl_earliest = equal_mode_earliest_shaft_centroid(onsets, labels, icl, contact_xy)
    scl_earliest = equal_mode_earliest_shaft_centroid(onsets, labels, scl, contact_xy)
    sources = [
        {"id": "ICL_mode_A", "patient_mode": 0,
         "xy_mm": icl_earliest["mode_centroids"][0].tolist()},
        {"id": "ICL_mode_B", "patient_mode": 1,
         "xy_mm": icl_earliest["mode_centroids"][1].tolist()},
        {"id": "SCL", "patient_mode": None,
         "xy_mm": scl_earliest["centroid"].tolist()},
    ]
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "SA6F_CONTINUOUS_FIELD_MANIFEST_FROZEN",
        "scientific_role": (
            "non-component continuous-field capacity initialization; spline "
            "coefficients are not cores"
        ),
        "candidate_set": {"candidates": candidates},
        "candidate_preflight": preflight,
        "patient_contact_targets": {
            key: np.asarray(value, float).tolist()
            for key, value in built["targets"].items()
        },
        "forced_sources": sources,
        "source_geometry": {
            "ICL_per_mode_centroids": icl_earliest["mode_centroids"].tolist(),
            "SCL_equal_mode_centroid": scl_earliest["centroid"].tolist(),
        },
        "fixed_contract": {
            "component_count": None,
            "peak_count_constraint": None,
            "N_core_manual": float(stage["N_core_manual"]),
            "network_seeds": assay["network_seeds"],
            "common_detector": threshold,
            "edge": "off",
            "beta": "closed",
        },
        "inputs": {
            key: {"path": value["path"], "sha256": value["sha256"]}
            for key, value in inputs.items()
        },
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit")
    parser.add_argument("--out")
    args = parser.parse_args()
    payload = build_manifest(args.config, args.expected_commit)
    output = Path(args.out or ROOT / payload["config"]["path"]).resolve()
    if args.out is None:
        raise RuntimeError("--out is required for the continuous-field manifest")
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(payload["candidate_set"]["candidates"]),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
