"""Freeze observation-invariant continuous-field sensitivity candidates."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import tensor_basis  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_graph_edge_flow import array_sha256 as edge_array_sha256  # noqa: E402
from src.topic4_observation_invariant_spline import (  # noqa: E402
    array_sha256,
    spline_roughness,
)
from src.topic4_spectral_field import (  # noqa: E402
    fourier_basis_2d,
    fourier_wavevectors,
    uniform_sheet_grid,
)


DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d6_continuous_field_kmeans_screen.json"
EXPECTED_ROLE = "development_only_observation_invariant_continuous_field_kmeans_screen"


def projected_uniform_residuals(config):
    """Return unit-RMS whole-sheet Fourier directions in spline coordinates."""
    field = config["field_search"]
    grid = uniform_sheet_grid(field["projection_grid_per_axis"], L=20.0)
    spline = tensor_basis(
        grid, int(field["n_basis_per_axis"]),
        degree=int(field["degree"]), L=20.0,
    )
    raw = fourier_basis_2d(
        grid, int(field["residual_max_harmonic"]), L=20.0,
    )
    raw = raw - raw.mean(axis=0, keepdims=True)
    raw /= np.sqrt(np.mean(raw ** 2, axis=0, keepdims=True))
    coefficients, *_ = np.linalg.lstsq(spline, raw, rcond=None)
    reconstructed = spline @ coefficients
    rmse = np.sqrt(np.mean((reconstructed - raw) ** 2, axis=0))
    rms = np.sqrt(np.mean(reconstructed ** 2, axis=0))
    coefficients /= rms[None, :]
    reconstructed /= rms[None, :]
    return {
        "grid": grid,
        "coefficients": coefficients.T,
        "surfaces": reconstructed.T,
        "projection_rmse": (rmse / rms).tolist(),
        "wavevectors": fourier_wavevectors(
            int(field["residual_max_harmonic"]), L=20.0,
        ),
    }


def _node_field(candidate_id, coefficients, anchor, **metadata):
    coeff = np.asarray(coefficients, float)
    return {
        "candidate_id": str(candidate_id),
        "field_type": "spline_continuous",
        "n_basis": int(anchor["n_basis"]),
        "degree": int(anchor["degree"]),
        "coefficients": coeff.tolist(),
        "field_sha256": array_sha256(coeff),
        "roughness": spline_roughness(coeff),
        "component_count": None,
        "peak_count_constraint": None,
        **metadata,
    }


def candidate_library(config, anchor):
    residuals = projected_uniform_residuals(config)
    base_coefficients = np.asarray(anchor["coefficients"], float)
    edge_count = int(config["spatial_edge_basis"]["coefficient_count"])
    edge_zero = np.zeros(edge_count, float)
    spatial_ou = deepcopy(config["fixed_spatial_ou"])
    spatial_ou.pop("selection_role", None)
    rows = [{
        "candidate_id": "edge_noop",
        "coefficients": edge_zero.tolist(),
        "coefficients_sha256": edge_array_sha256(edge_zero),
        "node_field": _node_field(
            "d6_warm", base_coefficients, anchor,
            role="warm_field_baseline", residual_coordinates=None,
            source_field_sha256=anchor["field_sha256"],
        ),
        "spatial_ou": spatial_ou,
    }]
    wavevectors = residuals["wavevectors"]
    phases = ("cos", "sin")
    for direction_index, direction in enumerate(residuals["coefficients"]):
        direction = np.asarray(direction, float).reshape(base_coefficients.shape)
        wavevector = wavevectors[direction_index // 2]
        phase = phases[direction_index % 2]
        for amplitude in config["field_search"]["residual_log_rms_amplitudes"]:
            for sign in (-1, 1):
                signed = float(sign) * float(amplitude)
                token = f"{abs(signed):.1f}".replace(".", "p")
                sign_token = "m" if sign < 0 else "p"
                candidate_id = (
                    f"d6_f{direction_index:02d}_{phase}_{sign_token}{token}"
                )
                rows.append({
                    "candidate_id": candidate_id,
                    "coefficients": edge_zero.tolist(),
                    "coefficients_sha256": edge_array_sha256(edge_zero),
                    "node_field": _node_field(
                        candidate_id, base_coefficients + signed * direction,
                        anchor, role="uniform_sheet_low_frequency_sensitivity",
                        residual_coordinates={
                            "direction_index": int(direction_index),
                            "wavevector_per_mm": wavevector.tolist(),
                            "phase": phase,
                            "signed_log_rms_amplitude": signed,
                        },
                        source_field_sha256=anchor["field_sha256"],
                    ),
                    "spatial_ou": deepcopy(spatial_ou),
                })
    expected = int(config["field_search"]["candidate_count"])
    if len(rows) != expected:
        raise RuntimeError(f"D6 candidate count changed: {len(rows)} != {expected}")
    if len({row["node_field"]["field_sha256"] for row in rows}) != len(rows):
        raise RuntimeError("D6 continuous fields are not unique")
    return rows, residuals


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config.get("scientific_role") != EXPECTED_ROLE:
        raise RuntimeError("D6 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    anchor_manifest = json.loads(
        (ROOT / config["inputs"]["node_anchor_manifest"]["path"]).read_text()
    )
    matches = [
        row for row in anchor_manifest["candidate_set"]["candidates"]
        if row["candidate_id"] == config["node_anchor"]["candidate_id"]
    ]
    if (len(matches) != 1 or matches[0]["field_sha256"]
            != config["node_anchor"]["field_sha256"]):
        raise RuntimeError("D6 warm field does not match the frozen anchor")
    anchor = matches[0]
    candidates, residuals = candidate_library(config, anchor)
    source = json.loads(
        (ROOT / config["inputs"]["frozen_direction_classifier_manifest"]["path"])
        .read_text()
    )
    d54 = json.loads(
        (ROOT / config["inputs"]["d5_4_verdict"]["path"]).read_text()
    )
    if d54.get("status") != "REV10D5_4_FRESH_SELECTION_KMEANS_NOT_REPLICATED":
        raise RuntimeError("D6 requires the frozen D5.4 non-replication result")
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
        raise RuntimeError("D6 freezer runtime or config is not frozen")
    output_root = ROOT / config["output_root"]
    if any((output_root / "workers").glob("*.json")):
        raise RuntimeError("D6 workers exist before manifest freeze")
    return {
        "status": "REV10D6_CONTINUOUS_FIELD_SENSITIVITY_LIBRARY_FROZEN",
        "scientific_role": EXPECTED_ROLE,
        "config": {"path": str(config_path.relative_to(ROOT)),
                   "sha256": _sha256(config_path)},
        "candidate_set": {"n_candidates": len(candidates),
                          "candidates": candidates},
        "representation_preflight": {
            "field_type": "uniform_tensor_cubic_bspline_log_field",
            "residual_type": "whole_sheet_real_fourier",
            "n_residual_directions": int(len(residuals["coefficients"])),
            "maximum_relative_projection_rmse": float(max(
                residuals["projection_rmse"]
            )),
            "observation_geometry_used_by_field_builder": False,
            "component_count": None,
            "peak_count_constraint": None,
        },
        "direction_classifier": source["direction_classifier"],
        "direction_classifier_source": {
            "path": config["inputs"]["frozen_direction_classifier_manifest"]["path"],
            "sha256": config["inputs"]["frozen_direction_classifier_manifest"]["sha256"],
            "copied_without_refit": True,
        },
        "fixed_contract": {
            "network_seeds": config["search"]["fit_network_seeds"],
            "spatial_ou": config["fixed_spatial_ou"],
            "edge": "exact no-op", "beta": "closed",
        },
        "forbidden_builder_inputs": config["field_search"][
            "forbidden_builder_inputs"
        ],
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    payload = build_manifest(args.config, args.expected_commit)
    output = ROOT / config["output_root"] / "candidate_manifest.json"
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": payload["candidate_set"]["n_candidates"],
        "max_projection_rmse": payload["representation_preflight"][
            "maximum_relative_projection_rmse"
        ],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
