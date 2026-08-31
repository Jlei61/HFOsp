#!/usr/bin/env python3
"""Prepare fresh-network rev17 dual-field candidates from the response atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev17_dual_field_candidates import nomination_blueprint  # noqa: E402


DEFAULT_ATLAS_CONFIG = ROOT / "config/topic4_rev17_dual_field_residual_atlas.json"
DEFAULT_ANALYSIS_CONFIG = ROOT / "config/topic4_rev17_dual_field_response_analysis.json"
DEFAULT_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "dual_field_residual_atlas/analysis/dual_field_response_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev17_dual_field_selection.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(path: Path, primary: Path, fallback: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(primary.resolve()))
    except ValueError:
        return str(path.relative_to(fallback.resolve()))


def build_config(
    *, aggregate_path: Path, atlas_config_path: Path,
    analysis_config_path: Path, artifact_root: Path,
) -> dict[str, Any]:
    aggregate_path = aggregate_path.resolve()
    atlas_config_path = atlas_config_path.resolve()
    analysis_config_path = analysis_config_path.resolve()
    artifact_root = artifact_root.resolve()
    for path in (aggregate_path, atlas_config_path, analysis_config_path):
        if not path.is_file():
            raise RuntimeError(f"rev17 selection input is missing: {path}")
    aggregate = json.loads(aggregate_path.read_text())
    analysis = json.loads(analysis_config_path.read_text())
    atlas = json.loads(atlas_config_path.read_text())
    blueprints, nomination = nomination_blueprint(aggregate, analysis)
    if not blueprints:
        raise RuntimeError("rev17 response atlas nominated no bounded local direction")
    atlas_manifest = artifact_root / atlas["candidate_manifest"]
    if not atlas_manifest.is_file():
        raise RuntimeError("rev17 source atlas manifest is missing")
    inputs = {
        "transition_config": atlas["inputs"]["transition_config"],
        "source_atlas_config": {
            "path": _relative(atlas_config_path, ROOT, artifact_root),
            "sha256": _sha256(atlas_config_path),
        },
        "source_atlas_manifest": {
            "path": _relative(atlas_manifest, ROOT, artifact_root),
            "sha256": _sha256(atlas_manifest),
        },
        "response_analysis_config": {
            "path": _relative(analysis_config_path, ROOT, artifact_root),
            "sha256": _sha256(analysis_config_path),
        },
        "response_aggregate": {
            "path": _relative(aggregate_path, ROOT, artifact_root),
            "sha256": _sha256(aggregate_path),
        },
        "j14_config": atlas["inputs"]["j14_config"],
        "patient_support_config": atlas["inputs"]["patient_support_config"],
    }
    selection_seeds = [
        int(seed) for seed in
        analysis["direction_construction"]["fresh_selection_network_seeds"]
    ]
    return {
        "schema_id": "topic4_rev17_dual_field_selection_v1",
        "scientific_role": "development_only_dual_continuous_node_residual_selection",
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
            "dual_field_selection"
        ),
        "candidate_manifest": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
            "dual_field_selection/candidate_manifest.json"
        ),
        "network_cache": atlas["network_cache"],
        "inputs": inputs,
        "dual_field_selection": {
            "source_formula": (
                "delta_vtheta_i=-h_mean_i*mu_mean-"
                "h_dispersion_i*(d_i-mu_dispersion)"
            ),
            "maximum_frequency": atlas["dual_field_residual"]["maximum_frequency"],
            "target_n_basis": atlas["dual_field_residual"]["target_n_basis"],
            "degree": atlas["dual_field_residual"]["degree"],
            "sheet_mm": atlas["dual_field_residual"]["sheet_mm"],
            "projection_grid_per_axis": atlas["dual_field_residual"][
                "projection_grid_per_axis"
            ],
            "candidate_blueprint": blueprints,
            "nomination_audit": nomination,
            "selection_candidate_count": len(blueprints),
            "candidate_count_including_anchor": 1 + len(blueprints),
        },
        "search": {
            "canary_network_seeds": [], "fit_network_seeds": [],
            "selection_network_seeds": selection_seeds,
            "confirmation_network_seeds": [],
            "common_random_numbers_across_candidates": True,
            "edge": "off", "beta": "closed",
            "simulation": atlas["search"]["simulation"],
            "contact_readout": atlas["search"]["contact_readout"],
        },
        "selection": {
            "J14_improvement_required_networks": 3,
            "A_improvement_required_networks": 3,
            "B_protection_required_networks": 3,
            "B_protection_ratio": 1.1,
            "minimum_effective_support_per_mode_per_network": 6.0,
            "natural_kmeans_used": False,
            "patient_heldout_used": False,
            "ictal_data_used": False,
            "figure_used": False,
        },
        "event_unit": atlas["event_unit"],
        "source_topology": atlas["source_topology"],
        "resources": atlas["resources"],
        "pathways": {
            "learned_E_to_E_redistribution": "off",
            "learned_E_to_I_redistribution": "off", "Z_M": "off",
        },
        "claim_boundary": (
            "Response-nominated dual continuous Node fields on three disjoint "
            "development networks. Only complete training-target J14, weak-mode "
            "loss, strong-mode protection and per-network support may select a "
            "candidate. Natural KMeans, held-out, figures, EE, E-to-I and Z/M "
            "remain closed."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--atlas-config", type=Path, default=DEFAULT_ATLAS_CONFIG)
    parser.add_argument(
        "--analysis-config", type=Path, default=DEFAULT_ANALYSIS_CONFIG,
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_config(
        aggregate_path=args.aggregate,
        atlas_config_path=args.atlas_config,
        analysis_config_path=args.analysis_config,
        artifact_root=args.artifact_root,
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV17_DUAL_FIELD_SELECTION_CONFIG_PREPARED",
        "output": str(args.output.resolve()),
        "candidate_count": payload["dual_field_selection"][
            "candidate_count_including_anchor"
        ],
        "selection_jobs": payload["dual_field_selection"][
            "candidate_count_including_anchor"
        ] * len(payload["search"]["selection_network_seeds"]),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
