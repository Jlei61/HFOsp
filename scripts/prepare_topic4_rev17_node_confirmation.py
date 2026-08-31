#!/usr/bin/env python3
"""Prepare unseen-network confirmation of one rev17 dual-field Node candidate."""
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


DEFAULT_SELECTION_CONFIG = ROOT / "config/topic4_rev17_dual_field_selection.json"
DEFAULT_SELECTION_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "dual_field_selection/analysis/fresh_selection_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev17_node_confirmation.json"
CONFIRMATION_SEEDS = [2381, 2382, 2383]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"rev17 confirmation input is missing: {path}")
    return {"path": str(path.relative_to(root.resolve())), "sha256": _sha256(path)}


def _resolve(root: Path, repository: Path, relative: str) -> Path:
    for path in (repository / relative, root / relative):
        if path.is_file():
            return path
    raise RuntimeError(f"rev17 inherited input is missing: {relative}")


def build_config(
    *, selection_config_path: Path, selection_aggregate_path: Path,
    artifact_root: Path = ARTIFACT_ROOT, repository_root: Path = ROOT,
) -> dict[str, Any]:
    selection_config = json.loads(selection_config_path.read_text())
    aggregate = json.loads(selection_aggregate_path.read_text())
    if selection_config.get("schema_id") != "topic4_rev17_dual_field_selection_v1":
        raise RuntimeError("rev17 selection config schema changed")
    if aggregate.get("schema_id") != (
        "topic4_rev17_dual_field_fresh_selection_aggregate_v1"
    ) or aggregate.get("status") != (
        "REV17_DUAL_FIELD_FRESH_SELECTION_AGGREGATE_COMPLETE"
    ):
        raise RuntimeError("rev17 fresh selection aggregate is incomplete")
    if not aggregate.get("inventory", {}).get("complete_cartesian_product"):
        raise RuntimeError("rev17 fresh selection Cartesian product is incomplete")
    boundaries = aggregate.get("boundaries", {})
    if (
        any(boundaries.get(key) is not False for key in (
            "natural_kmeans_used", "patient_heldout_used", "ictal_data_used",
            "figure_used",
        )) or boundaries.get("EE_EtoI_ZM") != "off"
    ):
        raise RuntimeError("rev17 confirmation source crossed a boundary")
    candidate_id = aggregate.get("selected_candidate_id")
    eligible = {
        row["candidate_id"]: row for row in aggregate.get("eligible_candidates", [])
    }
    if not isinstance(candidate_id, str) or candidate_id not in eligible:
        raise RuntimeError("rev17 selection produced no eligible Node candidate")
    manifest_path = artifact_root / selection_config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "REV17_DUAL_FIELD_SELECTION_CANDIDATES_FROZEN"
        or manifest.get("config_sha256") != _sha256(selection_config_path)
    ):
        raise RuntimeError("rev17 selection manifest is not frozen")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_id not in candidates or "exact_dual_anchor" not in candidates:
        raise RuntimeError("rev17 confirmation candidates are absent")
    selected = candidates[candidate_id]
    if selected.get("selection_eligible") is not True:
        raise RuntimeError("rev17 confirmation candidate is not selectable")
    inputs = {
        "selection_config": _record(selection_config_path, repository_root),
        "selection_manifest": _record(manifest_path, artifact_root),
        "selection_aggregate": _record(selection_aggregate_path, artifact_root),
    }
    for name in ("transition_config", "j14_config", "patient_support_config"):
        record = selection_config["inputs"][name]
        path = _resolve(artifact_root, repository_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev17 inherited input changed: {name}")
        inputs[name] = dict(record)
    return {
        "schema_id": "topic4_rev17_node_confirmation_v1",
        "scientific_role": "development_only_dual_continuous_node_confirmation",
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
            "node_confirmation"
        ),
        "candidate_manifest": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
            "node_confirmation/candidate_manifest.json"
        ),
        "network_cache": selection_config["network_cache"],
        "inputs": inputs,
        "selected_candidate": {
            "candidate_id": candidate_id,
            "mapping_sha256": selected["node_mapping"]["mapping_sha256"],
            "selection_network_seeds": selection_config["search"][
                "selection_network_seeds"
            ],
            "selection_summary": eligible[candidate_id],
        },
        "search": {
            "canary_network_seeds": [], "fit_network_seeds": [],
            "selection_network_seeds": [],
            "confirmation_network_seeds": CONFIRMATION_SEEDS,
            "common_random_numbers_across_candidates": True,
            "edge": "off", "beta": "closed",
            "simulation": selection_config["search"]["simulation"],
            "contact_readout": selection_config["search"]["contact_readout"],
        },
        "confirmation_acceptance": {
            "J14_improvement_required_networks": 3,
            "A_improvement_required_networks": 3,
            "B_protection_required_networks": 3,
            "B_protection_ratio": 1.1,
            "minimum_effective_support_per_mode_per_network": 6.0,
        },
        "event_unit": selection_config["event_unit"],
        "source_topology": selection_config["source_topology"],
        "pathways": selection_config["pathways"],
        "resources": selection_config["resources"],
        "boundaries": {
            "field_reranking_allowed": False,
            "patient_heldout_used": False,
            "natural_kmeans_used": False,
            "ictal_data_used": False, "figure_used": False,
            "EE_EtoI_ZM": "off",
        },
        "claim_boundary": (
            "The single training-selected dual field and exact dual anchor are "
            "copied to three unseen networks. Failure cannot promote another "
            "field. Natural KMeans, held-out, figures, EE, E-to-I and Z/M remain "
            "closed until this confirmation passes."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-config", type=Path, default=DEFAULT_SELECTION_CONFIG)
    parser.add_argument(
        "--selection-aggregate", type=Path, default=DEFAULT_SELECTION_AGGREGATE,
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_config(
        selection_config_path=args.selection_config.resolve(),
        selection_aggregate_path=args.selection_aggregate.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV17_NODE_CONFIRMATION_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "n_jobs": 6, "output": str(args.output.resolve()),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
