"""Freeze a fine continuous path across the observed A/B support boundary."""
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
from scripts.freeze_topic4_rev10_sa_spline_interpolation_v5_candidates import (  # noqa: E402
    _candidate,
    density_mixture,
)
from scripts.run_topic4_rev9l_forced_source_worker import (  # noqa: E402
    _load_json_input,
    _runtime_provenance,
    _sha256,
)
from src.topic4_continuous_field import tensor_basis  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_spectral_field import uniform_sheet_grid  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v6.json"


def build_candidates(config, source_manifest, audit):
    design, field = config["candidate_library"], config["field"]
    if audit["status"] != "MODE_CONDITIONED_JOINT_SUPPORT_NOT_FOUND":
        raise RuntimeError("V6 boundary refinement assumes no existing eligible field")
    sources = {
        row["candidate_id"]: row
        for row in source_manifest["candidate_set"]["candidates"]
    }
    left_id, right_id = (
        design["left_source_candidate_id"], design["right_source_candidate_id"],
    )
    left = np.asarray(sources[left_id]["coefficients"], float)
    right = np.asarray(sources[right_id]["coefficients"], float)
    grid = uniform_sheet_grid(field["projection_grid_per_axis"], L=20.0)
    candidates = []
    for fraction in design["fractions"]:
        t = float(fraction)
        if t == 0.0:
            coefficients = left
        else:
            coefficients = density_mixture(
                left, right, t, grid,
                n_basis=int(field["n_basis_per_axis"]),
                degree=int(field["degree"]),
            )
        slug = f"{int(round(1000 * t)):03d}"
        candidate = _candidate(
            f"v6_density_t{slug}", "mode_conditioned_density_boundary",
            coefficients, config,
            left_source_candidate_id=left_id,
            right_source_candidate_id=right_id,
            interpolation_fraction=t,
        )
        candidate["version"] = "V6"
        candidates.append(candidate)
    if len(candidates) != int(design["candidate_count"]):
        raise RuntimeError("V6 candidate count changed")
    known = sources[design["known_boundary_candidate_id"]]["field_sha256"]
    if candidates[-1]["field_sha256"] != known:
        raise RuntimeError("V6 t=0.25 does not reproduce the V5 boundary field")
    return candidates, grid


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_mode_conditioned_boundary_refinement"):
        raise RuntimeError("V6 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    stage = _load_json_input(config["inputs"]["stage_config"])
    contract = _load_json_input(config["inputs"]["contact_contract"])
    source_manifest = _load_json_input(config["inputs"]["v5_candidate_manifest"])
    audit = _load_json_input(config["inputs"]["v5_mode_conditioned_audit"])
    candidates, grid = build_candidates(config, source_manifest, audit)
    n_basis, degree = (
        int(config["field"]["n_basis_per_axis"]),
        int(config["field"]["degree"]),
    )
    classifier = _patient_classifier(config, contract)
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "REV10SA_V6_MODE_BOUNDARY_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "n_basis_per_axis": n_basis,
            "effective_coefficients": n_basis ** 2 - 1,
            "uniform_design_condition_number": float(np.linalg.cond(
                tensor_basis(grid, n_basis, degree=degree, L=20.0)
            )),
            "left_source_candidate_id": config["candidate_library"][
                "left_source_candidate_id"
            ],
            "right_source_candidate_id": config["candidate_library"][
                "right_source_candidate_id"
            ],
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
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
