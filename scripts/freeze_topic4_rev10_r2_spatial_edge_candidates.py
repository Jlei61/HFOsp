"""Freeze whitened, amplitude-bounded rev10-R2 spatial edge candidates."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.getcwd())
from scripts.freeze_topic4_rev10_sa_spline_field_v4_candidates import (  # noqa: E402
    _json_classifier,
    _patient_classifier,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_spatial_edge_flow import FEATURE_NAMES, array_sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r2_spatial_edge_flow.json"


def _audit_records(config, config_sha):
    records, maxima, covariance, second_moment = [], [], [], []
    root = ROOT / config["output_root"] / "feature_audit"
    for seed in map(int, config["search"]["fit_network_seeds"]):
        json_path, npz_path = root / f"seed_{seed}.json", root / f"seed_{seed}.npz"
        payload = json.loads(json_path.read_text())
        provenance = payload.get("provenance", {})
        source_commit = provenance.get("expected_git_commit")
        if not (
            payload.get("worker_status") == "REV10R2_SPATIAL_EDGE_AUDIT_COMPLETE"
            and payload.get("status") == "REV10R2_SPATIAL_FEATURE_CAPACITY_PASS"
            and payload.get("config", {}).get("sha256") == config_sha
            and source_commit == provenance.get("git_commit")
            and provenance.get("runtime_modules_match_expected_commit") is True
            and not provenance.get("runtime_modules_dirty")
            and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
        ):
            raise RuntimeError(f"stale or failed spatial feature audit: seed {seed}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            maxima.append(np.asarray(loaded["feature_abs_max"], float))
            covariance.append(np.asarray(loaded["covariance"], float))
            count = int(loaded["n_ee_delay_entries"])
            second_moment.append(
                np.asarray(loaded["feature_gram"], float) / count
            )
            names = np.asarray(loaded["feature_names"]).astype(str)
        if not np.array_equal(names, np.asarray(FEATURE_NAMES)):
            raise RuntimeError(f"spatial feature order changed: seed {seed}")
        records.append({
            "seed": seed,
            "json": str(json_path.relative_to(ROOT)),
            "json_sha256": _sha256(json_path),
            "npz": str(npz_path.relative_to(ROOT)),
            "npz_sha256": _sha256(npz_path),
            "effective_rank": payload["feature_audit"]["effective_rank"],
            "covariance_condition_number": payload["feature_audit"][
                "covariance_condition_number"
            ],
            "producer_commit": source_commit,
        })
    return (
        records, np.asarray(maxima), np.asarray(covariance),
        np.asarray(second_moment),
    )


def whitened_directions(config, covariance):
    count = int(config["candidate_library"]["base_direction_count"])
    dimension = int(config["spatial_edge_basis"]["coefficient_count"])
    if count < 1 or count & (count - 1):
        raise ValueError("Sobol base direction count must be a power of two")
    mean_covariance = np.mean(np.asarray(covariance, float), axis=0)
    mean_covariance = 0.5 * (mean_covariance + mean_covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(mean_covariance)
    if np.any(eigenvalues <= 0.0):
        raise RuntimeError("pooled spatial feature covariance is not positive definite")
    inverse_root = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))[None, :]
    ) @ eigenvectors.T
    raw = qmc.Sobol(
        d=dimension, scramble=True,
        seed=int(config["candidate_library"]["sobol_seed"]),
    ).random_base2(int(np.log2(count)))
    latent = 2.0 * raw - 1.0
    latent /= np.linalg.norm(latent, axis=1)[:, None]
    physical = latent @ inverse_root.T
    metric_norm = np.sqrt(np.einsum(
        "ni,ij,nj->n", physical, mean_covariance, physical,
    ))
    physical /= metric_norm[:, None]
    return physical, latent, mean_covariance, eigenvalues


def build_candidates(config, feature_abs_max, covariance, second_moment=None):
    directions, latent, mean_covariance, eigenvalues = whitened_directions(
        config, covariance,
    )
    dimension = directions.shape[1]
    library = config["candidate_library"]
    bound = float(library["raw_logit_abs_bound"])
    target_rms = library.get("target_unclipped_logit_rms")
    if target_rms is not None:
        target_rms = float(target_rms)
        if not np.isfinite(target_rms) or target_rms <= 0.0:
            raise ValueError("target_unclipped_logit_rms must be positive")
    mean_second_moment = np.mean(
        np.asarray(second_moment if second_moment is not None else covariance),
        axis=0,
    )
    candidates = [{
        "candidate_id": "edge_noop", "version": "rev10-R2",
        "role": "reused_Node_baseline_exact_noop",
        "mechanism": "continuous_quadratic_midpoint_vector_flow_v1",
        "coefficients": np.zeros(dimension).tolist(),
        "coefficients_sha256": array_sha256(np.zeros(dimension)),
        "raw_logit_abs_bound": 0.0,
        "raw_logit_clip": bound if target_rms is not None else None,
        "target_unclipped_logit_rms": 0.0,
        "edge_ratio_guarantee": [1.0, 1.0],
        "antithetic_pair": None,
    }]
    for index, direction in enumerate(directions):
        if target_rms is None:
            worst = float(np.max(feature_abs_max @ np.abs(direction)))
            scale = bound / worst
        else:
            direction_rms = float(np.sqrt(
                direction @ mean_second_moment @ direction
            ))
            scale = target_rms / direction_rms
        for suffix, sign in (("pos", 1.0), ("neg", -1.0)):
            coefficients = sign * scale * direction
            candidates.append({
                "candidate_id": f"edge_spatial_{index:02d}_{suffix}",
                "version": "rev10-R2",
                "role": "symmetric_continuous_spatial_route_probe",
                "mechanism": "continuous_quadratic_midpoint_vector_flow_v1",
                "coefficients": coefficients.tolist(),
                "coefficients_sha256": array_sha256(coefficients),
                "direction_index": index,
                "latent_whitened_direction": (sign * latent[index]).tolist(),
                "coefficient_l2": float(np.linalg.norm(coefficients)),
                "raw_logit_abs_bound": bound,
                "raw_logit_clip": bound if target_rms is not None else None,
                "target_unclipped_logit_rms": (
                    target_rms if target_rms is not None else None
                ),
                "edge_ratio_guarantee": [
                    float(np.exp(-2.0 * bound)),
                    float(np.exp(2.0 * bound)),
                ],
                "antithetic_pair": (
                    f"edge_spatial_{index:02d}_{'neg' if suffix == 'pos' else 'pos'}"
                ),
            })
    return candidates, directions, latent, mean_covariance, eigenvalues


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_observation_invariant_spatial_route_capacity"):
        raise RuntimeError("rev10-R2 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    config_sha = _sha256(config_path)
    audits, maxima, covariance, second_moment = _audit_records(config, config_sha)
    candidates, directions, latent, mean_covariance, eigenvalues = build_candidates(
        config, maxima, covariance, second_moment,
    )
    contract = _load_json_input(config["inputs"]["contact_contract"])
    classifier = _patient_classifier(config, contract)
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    if (provenance["runtime_modules_dirty"] or provenance["config_dirty"]
            or not provenance["runtime_modules_match_expected_commit"]):
        raise RuntimeError("candidate freezer runtime is not frozen")
    return {
        "status": "REV10R2_SPATIAL_EDGE_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {
            "candidates": candidates,
            "n_nonzero": int(sum(np.any(row["coefficients"]) for row in candidates)),
            "n_exact_noop": int(sum(not np.any(row["coefficients"]) for row in candidates)),
        },
        "capacity_preflight": {
            "all_fit_networks_pass": True,
            "feature_names": FEATURE_NAMES,
            "feature_abs_max_by_fit_graph": maxima.tolist(),
            "equal_network_mean_covariance": mean_covariance.tolist(),
            "equal_network_mean_second_moment": np.mean(
                second_moment, axis=0,
            ).tolist(),
            "equal_network_mean_covariance_eigenvalues": eigenvalues.tolist(),
            "physical_directions_sha256": array_sha256(directions),
            "latent_sobol_directions_sha256": array_sha256(latent),
            "all_fit_graph_edges_used_for_covariance_and_feature_max": True,
            "contact_or_patient_values_used_for_directions_or_amplitude": False,
            "guaranteed_edge_ratio_interval": config["candidate_library"][
                "guaranteed_pre_simulation_edge_ratio_interval"
            ],
            "dose_parameterization": config["candidate_library"].get(
                "amplitude_parameterization",
                "exact_full_edge_maximum_without_clipping",
            ),
        },
        "feature_audits": audits,
        "direction_classifier": _json_classifier(classifier),
        "fixed_contract": {
            "node_anchor": config["node_anchor"],
            "fit_network_seeds": config["search"]["fit_network_seeds"],
            "common_detector": config["search"]["detector"][
                "population_active_fraction_threshold"
            ],
            "mechanism": "continuous_quadratic_midpoint_vector_flow_v1",
            "beta": "closed",
        },
        "inputs": config["inputs"],
        "config": {"path": str(config_path.relative_to(ROOT)), "sha256": config_sha},
        "provenance": provenance,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.out or ROOT / config["output_root"] / "candidate_manifest.json")
    payload = build_manifest(args.config, args.expected_commit)
    atomic_write_json(payload, output)
    print(json.dumps({
        "status": payload["status"],
        "n_candidates": len(payload["candidate_set"]["candidates"]),
        "n_nonzero": payload["candidate_set"]["n_nonzero"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
