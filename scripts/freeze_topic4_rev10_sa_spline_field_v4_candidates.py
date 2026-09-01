"""Freeze V4 stable continuous random fields before patient scoring."""
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
from src.topic4_continuous_field import (  # noqa: E402
    continuous_field_h,
    continuous_surface,
    tensor_basis,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_core_field_stage3 import params_to_h, params_to_q  # noqa: E402
from src.topic4_observation_invariant_spline import (  # noqa: E402
    allocation_direction,
    array_sha256,
    fit_uniform_surface,
    sample_smooth_residual_pairs,
    spline_roughness,
    uniform_allocation_centers,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402
from src.topic4_shaft_aware_direction import fit_direction_classifier  # noqa: E402
from src.topic4_spectral_field import uniform_sheet_grid  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v4.json"


def _candidate(candidate_id, role, coefficients, config, **extra):
    values = np.asarray(coefficients, dtype=float)
    return {
        "candidate_id": str(candidate_id),
        "field_type": "spline_continuous",
        "version": "V4",
        "role": str(role),
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "n_basis": int(config["field"]["n_basis_per_axis"]),
        "degree": int(config["field"]["degree"]),
        "roughness": spline_roughness(values),
        "component_count": None,
        "peak_count_constraint": None,
        **extra,
    }


def build_candidates(config, theta):
    """Build fields without accepting any observation geometry argument."""
    field, design = config["field"], config["candidate_library"]
    n_basis, degree = int(field["n_basis_per_axis"]), int(field["degree"])
    grid = uniform_sheet_grid(field["projection_grid_per_axis"], L=20.0)
    old_log_q = np.log(np.maximum(
        params_to_q(np.asarray(theta, float), grid, K=3, L=20.0), 1e-300,
    ))
    warm = fit_uniform_surface(
        old_log_q, grid, n_basis=n_basis, degree=degree, L=20.0,
    )
    candidates = [
        _candidate("v4_stage3_spline_warm", "stage3_uniform_spline_warm", warm, config),
        _candidate("v4_uniform_sheet", "uniform_negative_control", np.zeros_like(warm), config),
    ]
    centers = uniform_allocation_centers(
        design["uniform_centers_per_axis"],
        margin_mm=design["uniform_center_margin_mm"], L=20.0,
    )
    for center_index, center in enumerate(centers):
        for amplitude in design["allocation_log_amplitudes"]:
            direction = allocation_direction(
                center, grid, n_basis=n_basis,
                width_mm=design["allocation_width_mm"],
                log_amplitude=amplitude, degree=degree, L=20.0,
            )
            slug = str(float(amplitude)).replace(".", "p")
            candidates.append(_candidate(
                f"v4_alloc_{center_index:02d}_a{slug}",
                "uniform_sheet_allocation_refinement", warm + direction, config,
                uniform_center_index=int(center_index),
                uniform_center_xy_mm=center.tolist(),
                allocation_width_mm=float(design["allocation_width_mm"]),
                allocation_log_amplitude=float(amplitude),
            ))
    pairs = sample_smooth_residual_pairs(
        n_pairs=design["random_pair_count"], n_basis=n_basis,
        seed=design["random_seed"],
        rms_amplitudes=design["random_rms_amplitudes"],
        positions=grid,
        smoothing_controls=design["random_smoothing_controls"],
        degree=degree, L=20.0,
    )
    for pair in pairs:
        for sign_name, sign, residual in (
            ("plus", 1, pair["positive"]),
            ("minus", -1, pair["negative"]),
        ):
            candidates.append(_candidate(
                f"v4_random_{pair['pair_index']:02d}_{sign_name}",
                "observation_free_smooth_random_residual",
                warm + residual, config,
                pair_index=pair["pair_index"], antithetic_sign=sign,
                residual_rms_amplitude=pair["rms_amplitude"],
            ))
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("V4 candidate count differs from the frozen design")
    return candidates, warm, grid, old_log_q, centers


def _patient_classifier(config, contract):
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    with np.load(target_path, allow_pickle=False) as loaded:
        onsets = np.asarray(loaded["patient_train_onsets"], float)
        labels = np.asarray(loaded["patient_train_old_labels"], int)
        blocks = np.asarray(loaded["patient_train_block_ids"])
        names = np.asarray(loaded["contact_names"]).astype(str)
        embedding = {
            "center": np.asarray(loaded["feature_center"], float),
            "scale": np.asarray(loaded["feature_scale"], float),
            "components": np.asarray(loaded["pca_components"], float),
        }
    expected = np.asarray([row["contact_name"] for row in contract["contacts"]])
    if not np.array_equal(names, expected):
        raise RuntimeError("patient target and contact contract order differ")
    design = config["direction_classifier"]
    classifier = fit_direction_classifier(
        onsets, labels, blocks, groups=contract_groups(contract),
        embedding=embedding, n_splits=design["recording_block_cv_folds"],
        regularization_c=design["regularization_c"],
        ood_quantile=design["ood_quantile"],
    )
    if classifier["heldout_balanced_accuracy"] < float(
            design["minimum_cv_balanced_accuracy"]):
        raise RuntimeError("shaft-aware direction classifier failed block CV")
    return classifier


def _json_classifier(classifier):
    return {
        key: (value.tolist() if isinstance(value, np.ndarray) else value)
        for key, value in classifier.items()
    }


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_only_stable_spline_random_field_screen":
        raise RuntimeError("V4 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    stage = _load_json_input(config["inputs"]["stage_config"])
    selection = _load_json_input(config["inputs"]["stage3_selection"])
    contract = _load_json_input(config["inputs"]["contact_contract"])
    theta = np.asarray(selection["selected_candidate"]["theta"], float)
    candidates, warm, grid, old_log_q, centers = build_candidates(config, theta)
    n_basis, degree = (
        int(config["field"]["n_basis_per_axis"]),
        int(config["field"]["degree"]),
    )
    basis = tensor_basis(grid, n_basis, degree=degree, L=20.0)
    warm_surface = continuous_surface(
        warm, grid, n_basis=n_basis, degree=degree, L=20.0,
    )
    target_surface = old_log_q - old_log_q.mean()
    expected_n_e = float(stage["engine"]["density"]) * 20.0 ** 2 * 0.8
    grid_budget = float(stage["N_core_manual"]) * len(grid) / expected_n_e
    exact_h = params_to_h(theta, grid, K=3, L=20.0, target_count=grid_budget)
    warm_h, _ = continuous_field_h(
        warm, grid, n_basis=n_basis, degree=degree, L=20.0,
        target_count=grid_budget,
    )
    top_count = max(1, int(np.ceil(0.05 * len(grid))))
    exact_top = set(np.argpartition(exact_h, -top_count)[-top_count:].tolist())
    warm_top = set(np.argpartition(warm_h, -top_count)[-top_count:].tolist())
    classifier = _patient_classifier(config, contract)
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "REV10SA_V4_STABLE_SPLINE_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "n_basis_per_axis": n_basis,
            "effective_coefficients": int(n_basis ** 2 - 1),
            "uniform_design_condition_number": float(np.linalg.cond(basis)),
            "stage3_projection_logq_rmse": float(np.sqrt(np.mean(
                (warm_surface - target_surface) ** 2
            ))),
            "stage3_projection_h_rmse": float(np.sqrt(np.mean(
                (warm_h - exact_h) ** 2
            ))),
            "stage3_projection_top5_jaccard": float(
                len(exact_top & warm_top) / len(exact_top | warm_top)
            ),
            "uniform_centers_xy_mm": centers.tolist(),
            "observation_geometry_used_by_field_builder": False,
        },
        "direction_classifier": _json_classifier(classifier),
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
        "condition_number": payload["representation_preflight"][
            "uniform_design_condition_number"
        ],
        "classifier_cv_balanced_accuracy": payload["direction_classifier"][
            "heldout_balanced_accuracy"
        ],
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
