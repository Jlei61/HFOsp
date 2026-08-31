#!/usr/bin/env python3
"""Freeze one-time held-out and source-topology audit for rev17 Node."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIRMATION_CONFIG = ROOT / "config/topic4_rev17_node_confirmation.json"
DEFAULT_CONFIRMATION_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_confirmation/analysis/confirmation_audit.json"
)
DEFAULT_POSTSELECTION_CONFIG = ROOT / "config/topic4_rev17_node_postselection.json"
DEFAULT_POSTSELECTION_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_postselection/analysis/node_postselection_audit.json"
)
DEFAULT_REV12_CONFIG = ROOT / "config/topic4_rev12_nd_node_dualmode_refit.json"
DEFAULT_OUTPUT = ROOT / "config/topic4_rev17_node_final_science.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, *, repository_root: Path,
            artifact_root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"rev17 final-science input is missing: {path}")
    for root in (repository_root.resolve(), artifact_root.resolve()):
        try:
            return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
        except ValueError:
            continue
    raise RuntimeError("rev17 final-science input is outside frozen roots")


def _resolve(record: Mapping[str, Any], *, repository_root: Path,
             artifact_root: Path) -> Path:
    for root in (artifact_root, repository_root):
        path = root / str(record["path"])
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"rev17 final-science input changed: {record['path']}")


def build_config(
    *, confirmation_config_path: Path, confirmation_audit_path: Path,
    postselection_config_path: Path, postselection_audit_path: Path,
    rev12_config_path: Path = DEFAULT_REV12_CONFIG,
    repository_root: Path = ROOT, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    confirmation = json.loads(confirmation_config_path.read_text())
    confirmation_audit = json.loads(confirmation_audit_path.read_text())
    postselection_config = json.loads(postselection_config_path.read_text())
    postselection = json.loads(postselection_audit_path.read_text())
    if confirmation.get("schema_id") != "topic4_rev17_node_confirmation_v1":
        raise RuntimeError("rev17 confirmation config changed")
    if confirmation_audit.get("scientific_confirmation", {}).get("accepted") is not True:
        raise RuntimeError("rev17 unseen-network confirmation is not accepted")
    if (
        postselection.get("status") != "REV17_NODE_POSTSELECTION_ACCEPTED"
        or postselection.get("acceptance", {}).get("accepted") is not True
    ):
        raise RuntimeError("rev17 natural-KMeans postselection is not accepted")
    candidate_id = postselection["candidate_id"]
    if candidate_id != confirmation["selected_candidate"]["candidate_id"] or (
        candidate_id != postselection_config["selected_candidate"]["candidate_id"]
    ):
        raise RuntimeError("rev17 final-science candidate identity changed")
    manifest_path = artifact_root / confirmation["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN":
        raise RuntimeError("rev17 confirmation manifest is not frozen")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_id not in candidates or "exact_dual_anchor" not in candidates:
        raise RuntimeError("rev17 final-science candidates are absent")
    rev12 = json.loads(rev12_config_path.read_text())
    inherited = {
        name: _resolve(
            rev12["inputs"][name], repository_root=repository_root,
            artifact_root=artifact_root,
        )
        for name in ("cohort_config", "classifier_config")
    }
    return {
        "schema_id": "topic4_rev17_node_final_science_v1",
        "scientific_role": (
            "development_only_rev17_one_time_heldout_topology_and_intervention_entry"
        ),
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
            "node_final_science"
        ),
        "inputs": {
            "confirmation_config": _record(
                confirmation_config_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "confirmation_manifest": _record(
                manifest_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "confirmation_audit": _record(
                confirmation_audit_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "postselection_config": _record(
                postselection_config_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "postselection_audit": _record(
                postselection_audit_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "rev12_config": _record(
                rev12_config_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            **{
                name: _record(
                    path, repository_root=repository_root,
                    artifact_root=artifact_root,
                ) for name, path in inherited.items()
            },
        },
        "selected_candidate": {
            "candidate_id": candidate_id,
            "mapping_sha256": candidates[candidate_id]["node_mapping"][
                "mapping_sha256"
            ],
            "paired_reference_candidate_id": "exact_dual_anchor",
            "reference_mapping_sha256": candidates["exact_dual_anchor"][
                "node_mapping"
            ]["mapping_sha256"],
        },
        "network_seeds": confirmation["search"]["confirmation_network_seeds"],
        "patient_endpoint": {
            "role": "one_time_development_heldout_evaluation_after_node_freeze",
            "primary_event_set": "complete_returned_causal_families",
            "eventwise_prototype_r2_must_be_positive": True,
            "eventwise_prototype_r2_must_improve_paired_exact_anchor": True,
            "mode_conditioned_event_cloud_metric": (
                "shaft_balanced_sliced_wasserstein_over_all_heldout_events"
            ),
            "both_mode_cloud_losses_and_weakest_cloud_must_improve": True,
            "both_mode_losses_and_weakest_mode_must_improve": True,
        },
        "source_topology": {
            "event_set": "complete_returned_source_evaluable_causal_families",
            "features": "early_10pct_mask_plus_normalized_local_onset",
            "permutation_draws": 4096,
            "permutation_seed": 20260831,
            "null": "within_network_labels_preserving_mode_occupancy",
            "continuous_quality": (
                "weakest_mode_cross_network_cosine_times_between_mode_distance"
            ),
            "must_improve_paired_exact_anchor": True,
        },
        "boundaries": {
            "field_reranking_allowed": False,
            "patient_heldout_used_for_field_selection": False,
            "patient_heldout_opened_by_final_audit": True,
            "natural_kmeans_must_already_be_accepted": True,
            "SNN_simulation_run": False, "EE_EtoI_ZM": "off",
        },
        "claim_boundary": (
            "One-time development held-out and model-internal source-topology "
            "audit of one already frozen rev17 dual Node mapping. It cannot "
            "rerank fields, establish patient-blind generalization, or activate "
            "EE, E-to-I or Z/M."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation-config", type=Path, default=DEFAULT_CONFIRMATION_CONFIG,
    )
    parser.add_argument(
        "--confirmation-audit", type=Path, default=DEFAULT_CONFIRMATION_AUDIT,
    )
    parser.add_argument(
        "--postselection-config", type=Path, default=DEFAULT_POSTSELECTION_CONFIG,
    )
    parser.add_argument(
        "--postselection-audit", type=Path, default=DEFAULT_POSTSELECTION_AUDIT,
    )
    parser.add_argument("--rev12-config", type=Path, default=DEFAULT_REV12_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_config(
        confirmation_config_path=args.confirmation_config.resolve(),
        confirmation_audit_path=args.confirmation_audit.resolve(),
        postselection_config_path=args.postselection_config.resolve(),
        postselection_audit_path=args.postselection_audit.resolve(),
        rev12_config_path=args.rev12_config.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV17_NODE_FINAL_SCIENCE_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "output": str(args.output.resolve()), "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
