"""Freeze V3 uniform whole-sheet allocation candidates without observations."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_spectral_field import (  # noqa: E402
    array_sha256,
    fourier_basis_2d,
    fourier_wavevectors,
    spectral_roughness,
    uniform_sheet_grid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v3.json"


def uniform_allocation_centers(n_per_axis, *, margin_mm, L=20.0):
    """Uniform centers independent of any electrode or patient geometry."""
    axis = np.linspace(float(margin_mm), float(L) - float(margin_mm), int(n_per_axis))
    return np.asarray([(x, y) for y in axis for x in axis], dtype=float)


def _candidate(candidate_id, role, coefficients, config, **extra):
    coeff = np.asarray(coefficients, dtype=float)
    return {
        "candidate_id": str(candidate_id),
        "field_type": "spectral_continuous",
        "version": "V3",
        "role": str(role),
        "coefficients": coeff.tolist(),
        "field_sha256": array_sha256(coeff),
        "roughness": spectral_roughness(
            coeff, max_harmonic=config["field"]["max_harmonic"],
        ),
        "component_count": None,
        "peak_count_constraint": None,
        **extra,
    }


def build_candidates(config, initial_manifest, initial_summary):
    """Generate V3 candidates with no observation geometry argument."""
    initial = {
        row["candidate_id"]: row
        for row in initial_manifest["candidate_set"]["candidates"]
    }
    warm = np.asarray(
        initial["v0_stage3_spectral_projection"]["coefficients"], float,
    )
    selected_id = initial_summary["selected_candidate_id"]
    if selected_id not in initial or initial[selected_id]["field_type"] != "spectral_continuous":
        raise RuntimeError("initial selected candidate is not a spectral field")
    selected = np.asarray(initial[selected_id]["coefficients"], float)
    design = config["candidate_library"]
    candidates = []
    for scale in design["warm_scale_controls"]:
        slug = f"{float(scale):.2f}".replace(".", "p")
        candidates.append(_candidate(
            f"v3_warm_scale_{slug}", "warm_attenuation_control",
            float(scale) * warm, config, warm_scale=float(scale),
        ))
    if design["include_initial_selected"]:
        candidates.append(_candidate(
            "v3_initial_selected", "initial_search_winner_benchmark",
            selected, config, source_candidate_id=selected_id,
        ))

    grid = uniform_sheet_grid(
        config["field"]["projection_grid_per_axis"], L=20.0,
    )
    centers = uniform_allocation_centers(
        design["uniform_centers_per_axis"],
        margin_mm=design["uniform_center_margin_mm"], L=20.0,
    )
    width = float(design["allocation_width_mm"])
    amplitude = float(design["allocation_log_amplitude"])
    warm_scale = float(design["allocation_warm_scale"])
    surfaces = np.column_stack([
        amplitude * np.exp(
            -0.5 * np.sum((grid - center[None, :]) ** 2, axis=1) / width ** 2
        )
        for center in centers
    ])
    surfaces -= surfaces.mean(axis=0, keepdims=True)
    basis = fourier_basis_2d(
        grid, config["field"]["max_harmonic"], L=20.0,
    )
    fitted, *_ = np.linalg.lstsq(basis, surfaces, rcond=None)
    coefficient_shape = (
        len(fourier_wavevectors(config["field"]["max_harmonic"], L=20.0)), 2
    )
    for index, center in enumerate(centers):
        allocation = fitted[:, index].reshape(coefficient_shape)
        candidates.append(_candidate(
            f"v3_uniform_{index:02d}", "uniform_sheet_allocation_direction",
            warm_scale * warm + allocation, config,
            uniform_center_index=int(index),
            uniform_center_xy_mm=center.tolist(),
            allocation_width_mm=width,
            allocation_log_amplitude=amplitude,
            allocation_warm_scale=warm_scale,
        ))
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("V3 candidate count differs from the frozen design")
    return candidates, centers


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_observation_invariant_uniform_allocation_refinement"):
        raise RuntimeError("V3 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    initial_manifest = _load_json_input(config["inputs"]["initial_candidate_manifest"])
    initial_summary = _load_json_input(config["inputs"]["initial_search_summary"])
    candidates, centers = build_candidates(config, initial_manifest, initial_summary)
    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("V3 candidate ids are not unique")
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "REV10SA_SPECTRAL_V3_UNIFORM_ALLOCATION_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "max_harmonic": int(config["field"]["max_harmonic"]),
            "uniform_centers_per_axis": int(
                config["candidate_library"]["uniform_centers_per_axis"]
            ),
            "uniform_centers_xy_mm": centers.tolist(),
            "all_center_spacings_equal": True,
            "observation_geometry_used": False,
            "initial_selected_candidate_id": initial_summary["selected_candidate_id"],
            "initial_selected_score": initial_summary["selected_selection_score"],
        },
        "observation_boundary": config["observation_boundary"],
        "fixed_contract": {
            "network_seeds": config["search"]["network_seeds"],
            "common_detector": config["search"]["detector"][
                "population_active_fraction_threshold"
            ],
            "edge": "off", "beta": "closed",
        },
        "inputs": config["inputs"],
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
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = build_manifest(args.config, args.expected_commit)
    atomic_write_json(payload, Path(args.out))
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(payload["candidate_set"]["candidates"]),
        "observation_geometry_used": False,
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
