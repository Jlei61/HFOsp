#!/usr/bin/env python3
"""Freeze the rev11-NLC2 joint continuous-Node/connectivity fit library."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.freeze_topic4_rev10_d6_continuous_field_kmeans_screen import (  # noqa: E402
    _node_field,
    projected_uniform_residuals,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_joint_node_connectivity_fit.json"
EXPECTED_ROLE = "development_only_data_driven_node_local_connectivity_joint_fit"


def _candidate(candidate_id, node_field, edge, spatial_ou, *, arm, **metadata):
    edge = np.asarray(edge, float).reshape(2, 6)
    return {
        "candidate_id": candidate_id,
        "arm": arm,
        "node_field": node_field,
        "coefficients": edge.tolist(),
        "coefficients_sha256": array_sha256(edge),
        "raw_logit_clip": 0.75,
        "spatial_ou": deepcopy(spatial_ou),
        "adaptation": {"mode": "off"},
        "inhibitory_resource": {"mode": "off"},
        "ee_std": {"mode": "off"},
        "mz": {"mode": "off"},
        **metadata,
    }


def candidate_library(config, anchor, nlc1_manifest, rescore):
    parents = rescore["nlc2_joint_search_parent_ids"]
    if parents != config["candidate_library"]["parent_candidate_ids"]:
        raise RuntimeError("NLC2 parent IDs differ from the frozen NLC1 rescore")
    nlc1 = {
        row["candidate_id"]: row
        for row in nlc1_manifest["candidate_set"]["candidates"]
    }
    residuals = projected_uniform_residuals(config)
    residual_coeff = np.asarray(residuals["coefficients"], float)
    residual_surface = np.asarray(residuals["surfaces"], float)
    expected_directions = int(config["field_search"]["residual_direction_count"])
    if residual_coeff.shape[0] != expected_directions:
        raise RuntimeError("whole-sheet residual direction count changed")
    base_coeff = np.asarray(anchor["coefficients"], float)
    spatial_ou = deepcopy(config["fixed_spatial_ou"])
    spatial_ou.pop("role", None)
    edge_zero = np.zeros((2, 6), float)
    rows = [_candidate(
        "node_baseline", _node_field(
            "node_baseline", base_coeff, anchor,
            role="continuous_Node_control", residual_coordinates=None,
            source_field_sha256=anchor["field_sha256"],
        ), edge_zero, spatial_ou, arm="Node",
        search_coordinates={"control": True, "node_amplitude": 0.0,
                            "edge_parent": None, "edge_delta": edge_zero.tolist()},
    )]
    for parent_id in parents:
        parent_edge = np.asarray(nlc1[parent_id]["coefficients"], float)
        rows.append(_candidate(
            f"{parent_id}_control", _node_field(
                f"{parent_id}_control", base_coeff, anchor,
                role="NLC1_edge_parent_control", residual_coordinates=None,
                source_field_sha256=anchor["field_sha256"],
            ), parent_edge, spatial_ou, arm="Node+EE+EtoI",
            search_coordinates={"control": True, "node_amplitude": 0.0,
                                "edge_parent": parent_id,
                                "edge_delta": edge_zero.tolist()},
        ))
    samples = int(config["candidate_library"]["samples_per_parent"])
    if samples <= 0 or samples & (samples - 1):
        raise RuntimeError("samples_per_parent must be a positive power of two")
    amplitude_low, amplitude_high = map(
        float, config["field_search"]["residual_log_rms_amplitude_range"],
    )
    edge_bounds = np.asarray(
        config["local_connectivity_basis"]["coefficient_abs_bounds"], float,
    )[None, :]
    delta_width = np.asarray(
        config["local_connectivity_basis"]["local_delta_half_width"], float,
    )[None, :]
    for parent_index, parent_id in enumerate(parents):
        parent_edge = np.asarray(nlc1[parent_id]["coefficients"], float)
        sampler = qmc.Sobol(
            d=expected_directions + 1 + parent_edge.size,
            scramble=True,
            seed=int(config["candidate_library"]["seed"]) + parent_index,
        )
        draws = sampler.random_base2(int(np.log2(samples)))
        for index, draw in enumerate(draws):
            direction_weights = 2.0 * draw[:expected_directions] - 1.0
            surface = direction_weights @ residual_surface
            rms = float(np.sqrt(np.mean(surface ** 2)))
            if not rms > 1e-8:
                raise RuntimeError("degenerate whole-sheet residual combination")
            direction_coeff = direction_weights @ residual_coeff / rms
            amplitude = amplitude_low + draw[expected_directions] * (
                amplitude_high - amplitude_low
            )
            node_coeff = base_coeff + amplitude * direction_coeff.reshape(
                base_coeff.shape
            )
            edge_draw = draw[expected_directions + 1:].reshape(2, 6)
            edge_delta = (2.0 * edge_draw - 1.0) * delta_width
            edge = np.clip(
                parent_edge + edge_delta, -edge_bounds, edge_bounds,
            )
            actual_delta = edge - parent_edge
            candidate_id = f"nlc2_{parent_id}_{index:02d}"
            rows.append(_candidate(
                candidate_id, _node_field(
                    candidate_id, node_coeff, anchor,
                    role="normalized_whole_sheet_low_frequency_joint_fit",
                    residual_coordinates={
                        "basis": "all_real_Fourier_modes_through_harmonic_two",
                        "direction_weights": direction_weights.tolist(),
                        "signed_log_rms_amplitude": float(amplitude),
                    },
                    source_field_sha256=anchor["field_sha256"],
                ), edge, spatial_ou, arm="Node+EE+EtoI",
                search_coordinates={
                    "control": False,
                    "node_amplitude": float(amplitude),
                    "edge_parent": parent_id,
                    "edge_delta": actual_delta.tolist(),
                },
            ))
    expected = int(config["candidate_library"]["candidate_count"])
    if len(rows) != expected or len({row["candidate_id"] for row in rows}) != expected:
        raise RuntimeError("NLC2 candidate count or IDs changed")
    parameter_hashes = {
        (row["node_field"]["field_sha256"], row["coefficients_sha256"])
        for row in rows
    }
    if len(parameter_hashes) != expected:
        raise RuntimeError("NLC2 contains duplicate Node/edge parameter pairs")
    return rows, residuals


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("NLC2 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    anchor_manifest = json.loads(
        (ROOT / config["inputs"]["node_anchor_manifest"]["path"]).read_text()
    )
    anchor = next(
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    )
    if anchor["field_sha256"] != config["node_anchor"]["field_sha256"]:
        raise RuntimeError("NLC2 Node anchor changed")
    nlc1 = json.loads((ROOT / config["inputs"]["nlc1_manifest"]["path"]).read_text())
    rescore = json.loads(
        (ROOT / config["inputs"]["nlc1_crossfit_rescore"]["path"]).read_text()
    )
    if nlc1.get("status") != "REV11NLC_LOCAL_CONNECTIVITY_LIBRARY_FROZEN":
        raise RuntimeError("NLC1 parent library is not frozen")
    if rescore.get("status") != "REV11NLC_CONTACT_SPLIT_CROSS_FIT_RESCORING_COMPLETE":
        raise RuntimeError("NLC1 cross-fit rescore is incomplete")
    candidates, residuals = candidate_library(config, anchor, nlc1, rescore)
    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    provenance = _runtime_provenance(commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (provenance["config_dirty"] or provenance["runtime_modules_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("NLC2 freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("NLC2 workers exist before manifest freeze")
    direction_source = json.loads(
        (ROOT / config["inputs"]["direction_classifier_manifest"]["path"]).read_text()
    )
    return {
        "status": "REV11NLC_JOINT_NODE_CONNECTIVITY_FIT_LIBRARY_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {"path": str(config_path.relative_to(ROOT)),
                   "sha256": _sha256(config_path)},
        "candidate_set": {"n_candidates": len(candidates),
                          "candidates": candidates},
        "selection_freeze": {
            "paired_control_candidate_id": "node_baseline",
            "nlc1_parent_candidate_ids": rescore["nlc2_joint_search_parent_ids"],
            "fit_is_exploratory": True,
        },
        "representation_preflight": {
            "node_field": "uniform tensor cubic B-spline",
            "node_residual": "normalized combination of all whole-sheet real Fourier modes through harmonic two",
            "n_residual_directions": int(len(residuals["coefficients"])),
            "maximum_relative_projection_rmse": float(max(residuals["projection_rmse"])),
            "observation_geometry_used_by_candidate_builder": False,
            "component_count": None,
            "peak_count_constraint": None,
            "edge_pathways": ["E_to_E", "E_to_I"],
        },
        "direction_classifier": direction_source["direction_classifier"],
        "direction_classifier_source": {
            "path": config["inputs"]["direction_classifier_manifest"]["path"],
            "sha256": config["inputs"]["direction_classifier_manifest"]["sha256"],
            "copied_without_refit": True,
        },
        "fixed_contract": {
            "network_seeds": config["search"]["fit_network_seeds"],
            "duration_ms": config["search"]["simulation"]["duration_ms"],
            "spatial_ou": config["fixed_spatial_ou"],
            "topology": "frozen", "delays": "frozen",
            "GABA": "frozen", "beta": "closed", "Z_M": "off",
        },
        "forbidden_builder_inputs": config["field_search"]["forbidden_builder_inputs"],
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    payload = build_manifest(args.config, args.expected_commit)
    config = json.loads(Path(args.config).read_text())
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": payload["candidate_set"]["n_candidates"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
