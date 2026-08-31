#!/usr/bin/env python3
"""Validate and score the rev17 dual-field unseen-network confirmation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = ROOT / "config/topic4_rev17_node_confirmation.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import aggregate_topic4_rev14_m3_canary as canary  # noqa: E402
from scripts import aggregate_topic4_rev15_m3_coordinate_atlas as flat  # noqa: E402
from scripts import aggregate_topic4_rev17_dual_field_residual_atlas as atlas  # noqa: E402
from scripts.aggregate_topic4_rev17_dual_field_selection import (  # noqa: E402
    evaluate_candidates,
)
from scripts import rescore_topic4_rev14_static_node_historical_libraries as historical  # noqa: E402


STATUS = "REV17_NODE_CONFIRMATION_SCORED_COMPLETE"
MANIFEST_STATUS = "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN"


def _load_contract(config_path: Path, root: Path) -> tuple[dict, dict]:
    config = json.loads(config_path.read_text())
    if config.get("schema_id") != "topic4_rev17_node_confirmation_v1":
        raise RuntimeError("rev17 confirmation schema changed")
    if config.get("scientific_role") != (
        "development_only_dual_continuous_node_confirmation"
    ):
        raise RuntimeError("rev17 confirmation role changed")
    if any(config["boundaries"].get(key) is not False for key in (
        "field_reranking_allowed", "patient_heldout_used", "natural_kmeans_used",
        "ictal_data_used", "figure_used",
    )) or config["boundaries"].get("EE_EtoI_ZM") != "off":
        raise RuntimeError("rev17 confirmation crossed a boundary")
    for name, record in config["inputs"].items():
        path = atlas._resolve(root, record["path"])
        if not path.is_file() or atlas._sha256(path) != record["sha256"]:
            raise RuntimeError(f"rev17 confirmation input changed: {name}")
    manifest_path = root / config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != MANIFEST_STATUS
        or manifest.get("config_sha256") != atlas._sha256(config_path)
        or not manifest.get("provenance", {}).get("formal_ready")
    ):
        raise RuntimeError("rev17 confirmation manifest is not formally frozen")
    if [row["candidate_id"] for row in manifest["candidates"]] != [
        "exact_dual_anchor", config["selected_candidate"]["candidate_id"],
    ]:
        raise RuntimeError("rev17 confirmation candidate identity changed")
    return config, manifest


def audit(config_path: Path = DEFAULT_CONFIG,
          root: Path = ARTIFACT_ROOT) -> dict[str, Any]:
    config_path = config_path.resolve(); root = root.resolve()
    config, manifest = _load_contract(config_path, root)
    provenance = atlas._analysis_provenance(manifest)
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    seeds = [int(seed) for seed in config["search"]["confirmation_network_seeds"]]
    worker_root = root / config["output_root"] / "workers"
    expected = 2 * len(seeds)
    missing, invalid, validated = [], [], []
    positions, stage = ({}, None)
    for candidate_id in candidates:
        for seed in seeds:
            path = worker_root / f"{candidate_id}_seed_{seed}.json"
            if not path.is_file():
                missing.append(f"{candidate_id}:{seed}")
    if not missing:
        positions, stage = atlas._reference_geometry(config, manifest, root)
        for candidate_id, candidate in candidates.items():
            for seed in seeds:
                path = (worker_root / f"{candidate_id}_seed_{seed}.json").resolve()
                try:
                    validated.append(atlas._validate_worker(
                        path, candidate, seed, config, manifest,
                        positions[seed], stage, root,
                    ))
                except Exception as error:
                    invalid.append(f"{candidate_id}:{seed}:{error}")
    status, error, scored, evaluations, eligible = "INCOMPLETE", None, [], [], []
    if not provenance["analysis_worktree_clean"]:
        status, error = "INVALID_PROVENANCE", "analysis worktree is dirty"
    elif len(validated) == expected and not missing and not invalid:
        try:
            j14 = json.loads(atlas._resolve(
                root, config["inputs"]["j14_config"]["path"]
            ).read_text())
            context = historical._patient_context(j14, root)
            support = canary._load_support_context(
                atlas._resolve(root, config["inputs"]["patient_support_config"]["path"]),
                root, j14,
            )
            for record in validated:
                candidate = candidates[record["candidate_id"]]
                row = flat._flat_row(
                    canary._score_worker(record, context, support), candidate,
                )
                coordinates = candidate.get("residual_coordinates") or {}
                row.update({
                    "family": coordinates.get("family"),
                    "radius": coordinates.get("joint_dual_field_radius"),
                })
                scored.append(row)
            acceptance = config["confirmation_acceptance"]
            evaluations, eligible = evaluate_candidates(
                scored, candidates, seeds, {
                    "B_protection_ratio": acceptance["B_protection_ratio"],
                    "minimum_effective_support_per_mode_per_network": acceptance[
                        "minimum_effective_support_per_mode_per_network"
                    ],
                },
            )
            status = STATUS
        except Exception as caught:
            status, error = "INVALID_INPUT", str(caught)
    candidate_id = config["selected_candidate"]["candidate_id"]
    accepted = bool(
        len(eligible) == 1 and eligible[0]["candidate_id"] == candidate_id
    )
    output_path = root / config["output_root"] / "analysis/confirmation_audit.json"
    payload = {
        "schema_id": "topic4_rev17_node_confirmation_audit_v1",
        "status": status, "input_error": error,
        "candidate_id": candidate_id, "network_seeds": seeds,
        "inventory": {
            "expected_runs": expected, "present_validated": len(validated),
            "missing": missing, "invalid_artifact": invalid,
            "complete_cartesian_product": len(validated) == expected and not missing and not invalid,
        },
        "per_confirmation_run": scored,
        "candidate_evaluations": evaluations,
        "scientific_confirmation": {
            "accepted": accepted,
            "candidate_id": candidate_id,
            "evaluation": next(
                (row for row in evaluations if row["candidate_id"] == candidate_id),
                None,
            ),
            "failure_does_not_trigger_reranking": True,
        },
        "provenance": provenance,
        "boundaries": {**config["boundaries"], "SNN_simulation_run": False},
        "claim_boundary": config["claim_boundary"],
        "outputs": {"json": str(output_path)},
    }
    atlas._atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    payload = audit(args.config, args.artifact_root)
    print(json.dumps({
        "status": payload["status"],
        "present_validated": payload["inventory"]["present_validated"],
        "scientifically_confirmed": payload["scientific_confirmation"]["accepted"],
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
