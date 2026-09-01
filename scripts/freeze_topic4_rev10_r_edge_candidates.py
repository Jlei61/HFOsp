"""Freeze a graph-only amplitude-bounded rev10-R Sobol edge library."""
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
from src.topic4_graph_edge_flow import array_sha256  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r_graph_edge_flow.json"


def _basis_records(config, config_sha, expected_commit):
    records, maxima = [], []
    root = ROOT / config["output_root"] / "graph_basis"
    for seed in map(int, config["search"]["fit_network_seeds"]):
        json_path, npz_path = root / f"seed_{seed}.json", root / f"seed_{seed}.npz"
        payload = json.loads(json_path.read_text())
        provenance = payload.get("provenance", {})
        if not (
            payload.get("status") == "REV10R_GRAPH_BASIS_COMPLETE"
            and payload.get("config", {}).get("sha256") == config_sha
            and provenance.get("expected_git_commit") == expected_commit
            and provenance.get("runtime_modules_match_expected_commit") is True
            and not provenance.get("runtime_modules_dirty")
            and payload.get("arrays", {}).get("sha256") == _sha256(npz_path)
        ):
            raise RuntimeError(f"stale or incomplete graph basis: seed {seed}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            maxima.append(np.asarray(loaded["feature_abs_max"], float))
        records.append({
            "seed": seed,
            "json": str(json_path.relative_to(ROOT)),
            "json_sha256": _sha256(json_path),
            "npz": str(npz_path.relative_to(ROOT)),
            "npz_sha256": _sha256(npz_path),
            "graph_weight_sha256": payload["basis"]["graph_weight_sha256"],
            "singular_values": payload["basis"]["singular_values"],
            "truncation_boundary_relative_gap": payload["basis"][
                "truncation_boundary_relative_gap"
            ],
        })
    return records, np.asarray(maxima, float)


def build_candidates(config, feature_abs_max):
    library = config["candidate_library"]
    rank = int(config["graph_basis"]["rank"])
    count = int(library["base_direction_count"])
    if count < 1 or count & (count - 1):
        raise ValueError("Sobol base direction count must be a positive power of two")
    raw = qmc.Sobol(
        d=rank, scramble=True, seed=int(library["sobol_seed"]),
    ).random_base2(int(np.log2(count)))
    directions = 2.0 * raw - 1.0
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 1e-12):
        raise RuntimeError("Sobol library contains a zero direction")
    directions /= norms[:, None]
    raw_bound = float(library["raw_logit_abs_bound"])
    candidates = [{
        "candidate_id": "edge_noop",
        "version": "rev10-R1",
        "role": "reused_Node_baseline_exact_noop",
        "coefficients": np.zeros(rank).tolist(),
        "coefficients_sha256": array_sha256(np.zeros(rank)),
        "raw_logit_abs_bound": 0.0,
        "edge_ratio_guarantee": [1.0, 1.0],
        "antithetic_pair": None,
    }]
    for index, direction in enumerate(directions):
        worst_feature_bound = float(np.max(
            feature_abs_max @ np.abs(direction)
        ))
        scale = raw_bound / worst_feature_bound
        for suffix, sign in (("pos", 1.0), ("neg", -1.0)):
            coefficients = sign * scale * direction
            candidates.append({
                "candidate_id": f"edge_sobol_{index:02d}_{suffix}",
                "version": "rev10-R1",
                "role": "symmetric_graph_spectral_route_probe",
                "coefficients": coefficients.tolist(),
                "coefficients_sha256": array_sha256(coefficients),
                "direction_index": int(index),
                "direction": (sign * direction).tolist(),
                "coefficient_l2": float(np.linalg.norm(coefficients)),
                "raw_logit_abs_bound": raw_bound,
                "edge_ratio_guarantee": [
                    float(np.exp(-2.0 * raw_bound)),
                    float(np.exp(2.0 * raw_bound)),
                ],
                "antithetic_pair": f"edge_sobol_{index:02d}_{'neg' if suffix == 'pos' else 'pos'}",
            })
    return candidates, directions


def build_manifest(config_path, expected_commit):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_contact_density_invariant_route_capacity"):
        raise RuntimeError("rev10-R scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    commit = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True,
    ).strip()
    config_sha = _sha256(config_path)
    bases, maxima = _basis_records(config, config_sha, commit)
    candidates, directions = build_candidates(config, maxima)
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
        "status": "REV10R_GRAPH_SPECTRAL_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {
            "candidates": candidates,
            "n_nonzero": int(sum(np.any(row["coefficients"]) for row in candidates)),
            "n_exact_noop": int(sum(not np.any(row["coefficients"]) for row in candidates)),
        },
        "amplitude_preflight": {
            "rule": config["candidate_library"]["bound_derivation"],
            "feature_abs_max_by_fit_graph": maxima.tolist(),
            "sobol_unit_directions_sha256": array_sha256(directions),
            "all_fit_graph_edges_used_for_feature_max": True,
            "contact_or_patient_values_used_for_amplitude": False,
            "guaranteed_edge_ratio_interval": config["candidate_library"][
                "guaranteed_pre_simulation_edge_ratio_interval"
            ],
        },
        "graph_bases": bases,
        "direction_classifier": _json_classifier(classifier),
        "fixed_contract": {
            "node_anchor": config["node_anchor"],
            "fit_network_seeds": config["search"]["fit_network_seeds"],
            "common_detector": config["search"]["detector"][
                "population_active_fraction_threshold"
            ],
            "rank": config["graph_basis"]["rank"],
            "beta": "closed",
        },
        "inputs": config["inputs"],
        "config": {
            "path": str(config_path.relative_to(ROOT)), "sha256": config_sha,
        },
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
