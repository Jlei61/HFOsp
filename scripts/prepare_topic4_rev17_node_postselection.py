#!/usr/bin/env python3
"""Prepare frozen-candidate natural-KMeans audit for rev17 Node."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_CONFIRMATION_CONFIG = ROOT / "config/topic4_rev17_node_confirmation.json"
DEFAULT_CONFIRMATION_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_confirmation/analysis/confirmation_audit.json"
)
DEFAULT_FIGURE2_FIELD = ARTIFACT_ROOT / (
    "results/interictal_propagation_masked/template_gradient_fields/"
    "per_subject/epilepsiae_1146.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev17_node_postselection.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(path: Path, root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"rev17 postselection input is missing: {path}")
    return {"path": str(path.relative_to(root.resolve())), "sha256": _sha256(path)}


def _resolve_verified(
    record: Mapping[str, Any], *, repository_root: Path, artifact_root: Path,
) -> Path:
    for path in (
        artifact_root / str(record["path"]),
        repository_root / str(record["path"]),
    ):
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path
    raise RuntimeError(f"rev17 postselection input changed: {record['path']}")


def build_config(
    *, confirmation_config_path: Path, confirmation_audit_path: Path,
    artifact_root: Path, repository_root: Path = ROOT,
    figure2_field_path: Path = DEFAULT_FIGURE2_FIELD,
) -> dict[str, Any]:
    confirmation = json.loads(confirmation_config_path.read_text())
    audit = json.loads(confirmation_audit_path.read_text())
    if confirmation.get("schema_id") != "topic4_rev17_node_confirmation_v1":
        raise RuntimeError("rev17 confirmation config schema changed")
    if (
        audit.get("schema_id") != "topic4_rev17_node_confirmation_audit_v1"
        or audit.get("status") != "REV17_NODE_CONFIRMATION_SCORED_COMPLETE"
        or audit.get("scientific_confirmation", {}).get("accepted") is not True
        or not audit.get("inventory", {}).get("complete_cartesian_product")
    ):
        raise RuntimeError("rev17 confirmation is incomplete or failed")
    candidate_id = confirmation["selected_candidate"]["candidate_id"]
    if audit.get("candidate_id") != candidate_id:
        raise RuntimeError("rev17 confirmed candidate identity changed")
    manifest_path = artifact_root / confirmation["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("status") != "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN"
        or manifest.get("config_sha256") != _sha256(confirmation_config_path)
    ):
        raise RuntimeError("rev17 confirmation manifest is not frozen")
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    selected = candidates.get(candidate_id)
    if selected is None or selected["node_mapping"]["mapping_sha256"] != confirmation[
        "selected_candidate"
    ]["mapping_sha256"]:
        raise RuntimeError("rev17 selected mapping changed")
    j14_record = confirmation["inputs"]["j14_config"]
    j14_path = _resolve_verified(
        j14_record, repository_root=repository_root, artifact_root=artifact_root,
    )
    j14 = json.loads(j14_path.read_text())
    inputs = {
        "confirmation_config": _record(confirmation_config_path, repository_root),
        "confirmation_manifest": _record(manifest_path, artifact_root),
        "confirmation_audit": _record(confirmation_audit_path, artifact_root),
        "j14_config": _record(j14_path, repository_root),
        "figure2_template_field": _record(figure2_field_path, artifact_root),
    }
    for name in (
        "patient_training_target", "frozen_direction_classifier_manifest",
        "contact_contract",
    ):
        record = j14["inputs"][name]
        _resolve_verified(
            record, repository_root=repository_root, artifact_root=artifact_root,
        )
        inputs[name] = dict(record)
    seeds = [int(seed) for seed in confirmation["search"]["confirmation_network_seeds"]]
    return {
        "schema_id": "topic4_rev17_node_postselection_v1",
        "scientific_role": "development_only_rev17_frozen_node_natural_kmeans",
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
            "node_postselection"
        ),
        "inputs": inputs,
        "selected_candidate": {
            "candidate_id": candidate_id,
            "mapping_sha256": selected["node_mapping"]["mapping_sha256"],
            "selection_source": (
                "training-only selection plus unseen-network confirmation before "
                "natural KMeans or held-out access"
            ),
        },
        "network_seeds": seeds,
        "event_contract": {
            "unit": "nonoverlapping_returned_complete_causal_family",
            "temporal_overlap_components_excluded_whole": True,
            "minimum_readable_contacts": 3,
            "both_shafts_required": True,
            "frozen_classifier_ood_excluded": True,
            "contact_order": "patient_training_target_exact",
            "frozen_direction_classifier_input": "full_contact_onset_timing",
        },
        "natural_kmeans": {
            "k": 2, "feature": "masked_normalized_event_ranks",
            "use_masked_features": True, "minimum_shared_contacts": 3,
            "n_sample": 100, "n_tau_seeds": 5,
            "patient_labels_used_to_fit_kmeans": False,
            "cluster_mapping": "posthoc maximize contingency with frozen A/B labels",
        },
        "acceptance": {
            "minimum_supervised_events_per_mode_per_network": 3,
            "minimum_kmeans_events_per_cluster_per_network": 3,
            "same_networks_with_both_modes_required": 3,
            "minimum_per_network_kmeans_ami_with_supervised_direction": 0.8,
            "networks_meeting_kmeans_ami_required": 3,
            "pooled_patient_matrix_rule": "positive_diagonal_and_negative_crossed_cells",
        },
        "boundaries": {
            "patient_training_used": True, "patient_heldout_used": False,
            "natural_kmeans_used_for_field_selection": False,
            "natural_kmeans_used_for_node_acceptance": True,
            "figure_used_for_field_selection": False,
            "EE_EtoI_ZM": "off", "SNN_simulation_run": False,
        },
        "claim_boundary": (
            "This audit may accept or reject the already confirmed rev17 Node "
            "mapping. It cannot rerank fields, access patient held-out events, "
            "activate EE/E-to-I/Z/M, or establish patient-blind generalization."
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
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--figure2-field", type=Path, default=DEFAULT_FIGURE2_FIELD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_config(
        confirmation_config_path=args.confirmation_config.resolve(),
        confirmation_audit_path=args.confirmation_audit.resolve(),
        artifact_root=args.artifact_root.resolve(),
        figure2_field_path=args.figure2_field.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV17_NODE_POSTSELECTION_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "output": str(args.output.resolve()), "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
