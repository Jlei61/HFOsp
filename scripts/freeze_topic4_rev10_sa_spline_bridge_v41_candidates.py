"""Project every frozen V3 field onto a stable uniform spline basis."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

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
from src.topic4_continuous_field import (  # noqa: E402
    continuous_field_h,
    tensor_basis,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_observation_invariant_spline import (  # noqa: E402
    array_sha256,
    fit_uniform_surface,
    spline_roughness,
)
from src.topic4_spectral_field import (  # noqa: E402
    spectral_field_h,
    spectral_surface,
    uniform_sheet_grid,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v4_1.json"


def _bridge_candidate(source, coefficients, config, metrics):
    values = np.asarray(coefficients, float)
    source_id = str(source["candidate_id"])
    return {
        "candidate_id": f"v41_bridge_{source_id}",
        "field_type": "spline_continuous",
        "version": "V4.1",
        "role": "v3_spectral_to_spline_bridge",
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "n_basis": int(config["field"]["n_basis_per_axis"]),
        "degree": int(config["field"]["degree"]),
        "roughness": spline_roughness(values),
        "component_count": None,
        "peak_count_constraint": None,
        "source_candidate_id": source_id,
        "source_field_sha256": source["field_sha256"],
        "source_role": source["role"],
        "projection_metrics": metrics,
    }


def build_candidates(config, source_manifest, stage):
    """Build the complete bridge without any observation geometry argument."""
    field = config["field"]
    n_basis = int(field["n_basis_per_axis"])
    degree = int(field["degree"])
    grid = uniform_sheet_grid(field["projection_grid_per_axis"], L=20.0)
    source_config = _load_json_input(config["inputs"]["v3_config"])
    max_harmonic = int(source_config["field"]["max_harmonic"])
    expected_n_e = float(stage["engine"]["density"]) * 20.0 ** 2 * 0.8
    grid_budget = float(stage["N_core_manual"]) * len(grid) / expected_n_e
    candidates = []
    for source in source_manifest["candidate_set"]["candidates"]:
        if source["field_type"] != "spectral_continuous":
            raise RuntimeError("V4.1 bridge accepts only V3 spectral fields")
        source_coefficients = np.asarray(source["coefficients"], float)
        source_latent = spectral_surface(
            source_coefficients, grid, max_harmonic=max_harmonic, L=20.0,
        )
        coefficients = fit_uniform_surface(
            source_latent, grid, n_basis=n_basis, degree=degree, L=20.0,
        )
        source_h, _ = spectral_field_h(
            source_coefficients, grid, max_harmonic=max_harmonic,
            target_count=grid_budget, L=20.0,
        )
        bridge_h, _ = continuous_field_h(
            coefficients, grid, n_basis=n_basis, degree=degree,
            target_count=grid_budget, L=20.0,
        )
        top_count = max(1, int(np.ceil(0.05 * len(grid))))
        source_top = set(np.argpartition(source_h, -top_count)[-top_count:].tolist())
        bridge_top = set(np.argpartition(bridge_h, -top_count)[-top_count:].tolist())
        metrics = {
            "h_rmse": float(np.sqrt(np.mean((source_h - bridge_h) ** 2))),
            "h_correlation": float(np.corrcoef(source_h, bridge_h)[0, 1]),
            "top5_jaccard": float(
                len(source_top & bridge_top) / len(source_top | bridge_top)
            ),
        }
        candidates.append(_bridge_candidate(source, coefficients, config, metrics))
    expected = int(config["candidate_library"]["candidate_count"])
    if len(candidates) != expected:
        raise RuntimeError(f"expected {expected} V3 bridge candidates")
    return candidates, grid


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_only_v3_to_stable_spline_bridge":
        raise RuntimeError("V4.1 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    stage = _load_json_input(config["inputs"]["stage_config"])
    contract = _load_json_input(config["inputs"]["contact_contract"])
    source_manifest = _load_json_input(config["inputs"]["v3_candidate_manifest"])
    candidates, grid = build_candidates(config, source_manifest, stage)
    n_basis = int(config["field"]["n_basis_per_axis"])
    degree = int(config["field"]["degree"])
    projection = [row["projection_metrics"] for row in candidates]
    classifier = _patient_classifier(config, contract)
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "REV10SA_V41_ALL_V3_FIELDS_STABLE_SPLINE_BRIDGE_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "n_basis_per_axis": n_basis,
            "effective_coefficients": n_basis ** 2 - 1,
            "uniform_design_condition_number": float(np.linalg.cond(
                tensor_basis(grid, n_basis, degree=degree, L=20.0)
            )),
            "maximum_h_rmse": float(max(row["h_rmse"] for row in projection)),
            "minimum_h_correlation": float(min(
                row["h_correlation"] for row in projection
            )),
            "minimum_top5_jaccard": float(min(
                row["top5_jaccard"] for row in projection
            )),
            "source_candidate_count": len(candidates),
            "source_candidate_selection_used": False,
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
            "edge": "off",
            "beta": "closed",
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
        "representation_preflight": payload["representation_preflight"],
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
