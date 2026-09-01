"""Freeze a diverse V5 subset before reading selection-network outcomes."""
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
from src.topic4_continuous_field import tensor_basis  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_observation_invariant_spline import (  # noqa: E402
    array_sha256,
    spline_roughness,
)
from src.topic4_spectral_field import uniform_sheet_grid  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v5_1.json"


def select_confirmation_ids(source_manifest, fit_summary):
    """Apply a frozen diversity rule using fit outcomes only."""
    rows = list(fit_summary["candidate_rows"])
    candidates = {
        row["candidate_id"]: row
        for row in source_manifest["candidate_set"]["candidates"]
    }
    selected = []

    def add(candidate_id):
        if candidate_id not in candidates:
            raise RuntimeError(f"unknown V5 candidate: {candidate_id}")
        if candidate_id not in selected:
            selected.append(candidate_id)

    winner = fit_summary["selected_candidate_id"]
    if winner is None:
        raise RuntimeError("V5 fit has no joint-eligible candidate")
    add(winner)
    for row in sorted(
        rows,
        key=lambda value: (
            -float(value["joint_fraction"]), float(value["selection_score"]),
            value["candidate_id"],
        ),
    ):
        if row["role"] == "adaptive_training_anchor" and row["n_joint"] > 0:
            add(row["candidate_id"])
    for row in rows:
        if row["pareto_nondominated"]:
            add(row["candidate_id"])
    pair_index = candidates[winner].get("anchor_pair_index")
    if pair_index is None:
        raise RuntimeError("V5 fit winner is not an interpolation candidate")
    for candidate in source_manifest["candidate_set"]["candidates"]:
        if (
            candidate["role"] == "adaptive_density_mixture_interpolation"
            and candidate.get("anchor_pair_index") == pair_index
        ):
            add(candidate["candidate_id"])
    return selected


def build_candidates(config, source_manifest, fit_summary):
    identifiers = select_confirmation_ids(source_manifest, fit_summary)
    source = {
        row["candidate_id"]: row
        for row in source_manifest["candidate_set"]["candidates"]
    }
    candidates = []
    for source_id in identifiers:
        row = source[source_id]
        coefficients = np.asarray(row["coefficients"], float)
        candidates.append({
            **row,
            "candidate_id": f"v51_{source_id}",
            "version": "V5.1",
            "role": "selection_confirmation_" + row["role"],
            "coefficients": coefficients.tolist(),
            "field_sha256": array_sha256(coefficients),
            "roughness": spline_roughness(coefficients),
            "source_candidate_id": source_id,
            "source_field_sha256": row["field_sha256"],
        })
    expected = int(config["candidate_library"]["candidate_count"])
    if len(candidates) != expected:
        raise RuntimeError(f"expected {expected} V5.1 candidates")
    return candidates, identifiers


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_stable_spline_selection_confirmation"):
        raise RuntimeError("V5.1 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    stage = _load_json_input(config["inputs"]["stage_config"])
    contract = _load_json_input(config["inputs"]["contact_contract"])
    source_manifest = _load_json_input(config["inputs"]["v5_candidate_manifest"])
    fit_summary = _load_json_input(config["inputs"]["v5_fit_summary"])
    candidates, source_ids = build_candidates(config, source_manifest, fit_summary)
    n_basis, degree = (
        int(config["field"]["n_basis_per_axis"]),
        int(config["field"]["degree"]),
    )
    grid = uniform_sheet_grid(config["field"]["projection_grid_per_axis"], L=20.0)
    classifier = _patient_classifier(config, contract)
    provenance = _runtime_provenance(expected_commit)
    provenance["config_dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(config_path.relative_to(ROOT))],
        cwd=ROOT, text=True,
    ).strip())
    return {
        "status": "REV10SA_V51_DIVERSE_SELECTION_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "n_basis_per_axis": n_basis,
            "effective_coefficients": n_basis ** 2 - 1,
            "uniform_design_condition_number": float(np.linalg.cond(
                tensor_basis(grid, n_basis, degree=degree, L=20.0)
            )),
            "source_candidate_ids": source_ids,
            "selection_network_outcomes_used": False,
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
        "source_candidate_ids": payload["representation_preflight"][
            "source_candidate_ids"
        ],
        "output": str(Path(args.out).resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
