"""Freeze complementary continuous fields before fresh-network confirmation."""
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
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v5_2.json"


def select_final_sources(v51_summary, reference_source_id):
    """Select complementary fields using only frozen fit/selection outcomes."""
    selected = v51_summary["selected_candidate_id"]
    if selected is None:
        raise RuntimeError("V5.1 has no cross-network candidate")
    minimum_seeds = int(v51_summary["minimum_seeds_with_joint_for_selection"])
    eligible = [
        row for row in v51_summary["candidate_rows"]
        if row["n_runaway_networks"] == 0
        and row["n_joint"] > 0
        and row["n_seeds_with_joint"] >= minimum_seeds
    ]
    if not eligible:
        raise RuntimeError("V5.1 has no eligible joint-support anchor")
    support = min(
        eligible,
        key=lambda row: (
            -int(row["n_joint"]), -float(row["joint_fraction"]),
            float(row["selection_score"]), row["candidate_id"],
        ),
    )["candidate_id"]
    if support == selected:
        raise RuntimeError("final contrast needs distinct score and support fields")
    return selected, support, reference_source_id


def build_candidates(config, v5_manifest, v51_manifest, v51_summary):
    reference_id = config["candidate_library"]["reference_source_candidate_id"]
    selected_id, support_id, reference_id = select_final_sources(
        v51_summary, reference_id,
    )
    v51_source = {
        row["candidate_id"]: row
        for row in v51_manifest["candidate_set"]["candidates"]
    }
    v5_source = {
        row["candidate_id"]: row
        for row in v5_manifest["candidate_set"]["candidates"]
    }
    definitions = [
        ("v52_score_winner", "final_confirmation_score_winner",
         v51_source[selected_id], selected_id),
        ("v52_joint_support", "final_confirmation_joint_support",
         v51_source[support_id], support_id),
        ("v52_stage3_reference", "final_confirmation_stage3_reference",
         v5_source[reference_id], reference_id),
    ]
    candidates = []
    for candidate_id, role, source, source_id in definitions:
        coefficients = np.asarray(source["coefficients"], float)
        candidates.append({
            **source,
            "candidate_id": candidate_id,
            "version": "V5.2",
            "role": role,
            "coefficients": coefficients.tolist(),
            "field_sha256": array_sha256(coefficients),
            "roughness": spline_roughness(coefficients),
            "source_candidate_id": source_id,
            "source_field_sha256": source["field_sha256"],
        })
    expected = int(config["candidate_library"]["candidate_count"])
    if len(candidates) != expected:
        raise RuntimeError(f"expected {expected} V5.2 candidates")
    return candidates, [selected_id, support_id, reference_id]


def build_manifest(config_path, expected_commit=None):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != (
            "development_only_stable_spline_final_confirmation"):
        raise RuntimeError("V5.2 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    stage = _load_json_input(config["inputs"]["stage_config"])
    contract = _load_json_input(config["inputs"]["contact_contract"])
    v5_manifest = _load_json_input(config["inputs"]["v5_candidate_manifest"])
    v51_manifest = _load_json_input(config["inputs"]["v51_candidate_manifest"])
    v51_summary = _load_json_input(config["inputs"]["v51_selection_summary"])
    candidates, source_ids = build_candidates(
        config, v5_manifest, v51_manifest, v51_summary,
    )
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
        "status": "REV10SA_V52_FINAL_LIBRARY_FROZEN",
        "scientific_role": config["scientific_role"],
        "candidate_set": {"candidates": candidates},
        "representation_preflight": {
            "n_basis_per_axis": n_basis,
            "effective_coefficients": n_basis ** 2 - 1,
            "uniform_design_condition_number": float(np.linalg.cond(
                tensor_basis(grid, n_basis, degree=degree, L=20.0)
            )),
            "source_candidate_ids": source_ids,
            "selection_network_outcomes_used": True,
            "final_network_outcomes_used": False,
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
