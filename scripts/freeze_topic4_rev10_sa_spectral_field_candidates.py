"""Freeze observation-invariant spectral-field candidates for rev10-SA."""
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
from src.topic4_core_field_stage3 import params_to_h, params_to_q  # noqa: E402
from src.topic4_spectral_field import (  # noqa: E402
    array_sha256,
    project_surface_to_spectral,
    sample_stationary_residual_pairs,
    spectral_roughness,
    spectral_field_h,
    spectral_surface,
    uniform_sheet_grid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field.json"


def _spectral_candidate(candidate_id, version, role, coefficients, *,
                        max_harmonic, pair=None, sign=None, amplitude=None):
    coeff = np.asarray(coefficients, dtype=float)
    row = {
        "candidate_id": str(candidate_id),
        "field_type": "spectral_continuous",
        "version": str(version),
        "role": str(role),
        "coefficients": coeff.tolist(),
        "field_sha256": array_sha256(coeff),
        "roughness": spectral_roughness(
            coeff, max_harmonic=int(max_harmonic),
        ),
        "component_count": None,
        "peak_count_constraint": None,
    }
    if pair is not None:
        row.update({
            "pair_index": int(pair), "antithetic_sign": int(sign),
            "residual_rms_amplitude": float(amplitude),
        })
    return row


def build_candidates(config, theta):
    """Build candidates without accepting or loading observation geometry."""
    field = config["field"]
    library = config["candidate_library"]
    max_harmonic = int(field["max_harmonic"])
    L = 20.0
    grid = uniform_sheet_grid(field["projection_grid_per_axis"], L=L)
    old_log_q = np.log(params_to_q(np.asarray(theta, float), grid, K=3, L=L))
    warm = project_surface_to_spectral(
        old_log_q, grid, max_harmonic=max_harmonic, L=L,
    )
    candidates = [{
        "candidate_id": "v0_exact_stage3_k3",
        "field_type": "gaussian_k3_benchmark",
        "version": "V0",
        "role": "historical_exact_benchmark",
        "theta": np.asarray(theta, float).tolist(),
        "field_sha256": array_sha256(theta),
        "component_count": 3,
        "peak_count_constraint": 3,
    }]
    candidates.append(_spectral_candidate(
        "v0_uniform_sheet", "V0", "uniform_negative_control",
        np.zeros_like(warm), max_harmonic=max_harmonic,
    ))
    candidates.append(_spectral_candidate(
        "v0_stage3_spectral_projection", "V0",
        "observation_invariant_stage3_warm_start", warm,
        max_harmonic=max_harmonic,
    ))

    prior = field["stationary_prior"]
    seed = int(library["seed"])
    for version, key, seed_offset in (
        ("V1", "v1_low_frequency_pairs", 0),
        ("V2", "v2_stationary_multiscale_pairs", 1000),
    ):
        design = library[key]
        pairs = sample_stationary_residual_pairs(
            n_pairs=design["n_pairs"], max_harmonic=max_harmonic,
            seed=seed + seed_offset,
            rms_amplitudes=design["rms_amplitudes"],
            n_grid=96, L=L,
            correlation_harmonics=prior["correlation_harmonics"],
            smoothness=prior["smoothness"],
            active_max_harmonic=design["max_mode"],
        )
        for pair in pairs:
            for sign_name, sign_value, residual in (
                ("plus", 1, pair["positive"]),
                ("minus", -1, pair["negative"]),
            ):
                candidate_id = (
                    f"{version.lower()}_pair{pair['pair_index']:02d}_{sign_name}"
                )
                candidates.append(_spectral_candidate(
                    candidate_id, version,
                    "low_frequency_stationary_residual" if version == "V1"
                    else "multiscale_stationary_residual",
                    warm + residual, pair=pair["pair_index"], sign=sign_value,
                    amplitude=pair["rms_amplitude"],
                    max_harmonic=max_harmonic,
                ))
    return candidates, warm, grid, old_log_q


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_observation_invariant_continuous_node_field_search"):
        raise RuntimeError("spectral-field scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    stage = _load_json_input(config["inputs"]["stage_config"])
    selection = _load_json_input(config["inputs"]["stage3_selection"])
    theta = np.asarray(selection["selected_candidate"]["theta"], float)
    if selection["selected_theta_sha256"] != selection["selected_candidate"]["theta_sha256"]:
        raise RuntimeError("Stage 3 selection hash is internally inconsistent")
    if array_sha256(theta) != selection["selected_theta_sha256"]:
        raise RuntimeError("Stage 3 selected theta bytes do not match the frozen hash")
    candidates, warm, grid, old_log_q = build_candidates(config, theta)
    ids = [row["candidate_id"] for row in candidates]
    if len(candidates) != 21 or len(ids) != len(set(ids)):
        raise RuntimeError("initial spectral library must contain 21 unique candidates")

    warm_surface = spectral_surface(
        warm, grid, max_harmonic=config["field"]["max_harmonic"],
        L=stage["engine"]["L"],
    )
    target = old_log_q - old_log_q.mean()
    warm_surface -= warm_surface.mean()
    residual = warm_surface - target
    expected_n_e = float(stage["engine"]["density"]) * float(
        stage["engine"]["L"]
    ) ** 2 * 0.8
    grid_budget = float(stage["N_core_manual"]) * len(grid) / expected_n_e
    exact_h = params_to_h(
        theta, grid, K=3, L=stage["engine"]["L"], target_count=grid_budget,
    )
    warm_h, _ = spectral_field_h(
        warm, grid, max_harmonic=config["field"]["max_harmonic"],
        L=stage["engine"]["L"], target_count=grid_budget,
    )
    top_count = max(1, int(np.ceil(0.05 * len(grid))))
    exact_top = set(np.argpartition(exact_h, -top_count)[-top_count:].tolist())
    warm_top = set(np.argpartition(warm_h, -top_count)[-top_count:].tolist())
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "REV10SA_OBSERVATION_INVARIANT_SPECTRAL_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "uniform_grid_per_axis": int(config["field"]["projection_grid_per_axis"]),
            "max_harmonic": int(config["field"]["max_harmonic"]),
            "effective_coefficients": int(warm.size),
            "stage3_projection_logq_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "stage3_projection_logq_max_abs_error": float(np.max(np.abs(residual))),
            "stage3_projection_h_rmse": float(np.sqrt(np.mean(
                (warm_h - exact_h) ** 2
            ))),
            "stage3_projection_top5_jaccard": float(
                len(exact_top & warm_top) / len(exact_top | warm_top)
            ),
            "warm_coefficients_sha256": array_sha256(warm),
            "observation_geometry_used": False,
        },
        "observation_boundary": config["observation_boundary"],
        "fixed_contract": {
            "N_core_manual": float(stage["N_core_manual"]),
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
        "projection_rmse": payload["representation_preflight"][
            "stage3_projection_logq_rmse"
        ],
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
