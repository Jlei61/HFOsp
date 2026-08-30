#!/usr/bin/env python3
"""Prepare the post-selection natural-KMeans contract for one robust Node field."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_ROBUST_CONFIG = ROOT / "config/topic4_rev15_m3_robust_candidates.json"
DEFAULT_ROBUST_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_robust_candidates/analysis/m3_robust_candidates_aggregate.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev15_node_postselection.json"
DEFAULT_FIGURE2_FIELD = ARTIFACT_ROOT / (
    "results/interictal_propagation_masked/template_gradient_fields/"
    "per_subject/epilepsiae_1146.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, root: Path) -> dict[str, str]:
    if not path.is_file():
        raise RuntimeError(f"post-selection input is missing: {path}")
    return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}


def _resolve_verified(
    record: Mapping[str, Any], *, repository_root: Path, artifact_root: Path,
) -> Path:
    candidates = (
        artifact_root / str(record["path"]),
        repository_root / str(record["path"]),
    )
    for path in candidates:
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path
    raise RuntimeError(f"post-selection input changed: {record['path']}")


def build_config(
    *, robust_config_path: Path, robust_aggregate_path: Path,
    artifact_root: Path, repository_root: Path = ROOT,
    figure2_field_path: Path | None = None,
) -> dict[str, Any]:
    robust_config = json.loads(robust_config_path.read_text())
    aggregate = json.loads(robust_aggregate_path.read_text())
    if aggregate.get("status") != "COMPLETE":
        raise RuntimeError("robust-candidate aggregate is incomplete")
    inventory = aggregate.get("inventory", {})
    if not inventory.get("complete_cartesian_product"):
        raise RuntimeError("robust-candidate Cartesian product is incomplete")
    ranking = aggregate.get("ranking_contract", {})
    forbidden = (
        "natural_kmeans_used", "patient_heldout_used", "ictal_data_used",
        "figure_used",
    )
    if any(ranking.get(key) is not False for key in forbidden):
        raise RuntimeError("Node construction crossed a forbidden boundary")
    if ranking.get("EE_EtoI_ZM") != "off":
        raise RuntimeError("Node construction activated another mechanism")
    candidate_id = aggregate.get("best_usable_anchor")
    if not isinstance(candidate_id, str) or candidate_id not in aggregate.get(
        "usable_two_mode_anchor_ids", []
    ):
        raise RuntimeError("no training-qualified robust Node anchor exists")
    manifest_path = artifact_root / robust_config["candidate_manifest"]
    manifest = json.loads(manifest_path.read_text())
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if candidate_id not in candidates or not candidates[candidate_id].get(
        "selection_eligible"
    ):
        raise RuntimeError("selected robust Node anchor is not frozen/selectable")
    j14_record = robust_config["inputs"]["j14_config"]
    j14_path = _resolve_verified(
        j14_record, repository_root=repository_root,
        artifact_root=artifact_root,
    )
    j14_config = json.loads(j14_path.read_text())
    inputs = {
        "robust_config": _record(robust_config_path, repository_root),
        "robust_manifest": _record(manifest_path, artifact_root),
        "robust_aggregate": _record(robust_aggregate_path, artifact_root),
        "j14_config": _record(j14_path, repository_root),
        "figure2_template_field": _record(
            (figure2_field_path or DEFAULT_FIGURE2_FIELD), artifact_root,
        ),
    }
    for name in (
        "patient_training_target", "frozen_direction_classifier_manifest",
        "contact_contract",
    ):
        record = j14_config["inputs"][name]
        path = _resolve_verified(
            record, repository_root=repository_root,
            artifact_root=artifact_root,
        )
        inputs[name] = {"path": record["path"], "sha256": record["sha256"]}
    return {
        "schema_id": "topic4_rev15_node_postselection_v1",
        "scientific_role": "development_only_postselection_natural_kmeans_and_fig4",
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
            "m3_robust_candidates/postselection"
        ),
        "inputs": inputs,
        "selected_candidate": {
            "candidate_id": candidate_id,
            "field_coefficients_sha256": candidates[candidate_id][
                "fourier_coordinate"
            ]["coefficients_sha256"],
            "selection_source": (
                "training-only robust aggregate before natural KMeans, figure, "
                "or patient held-out access"
            ),
        },
        "network_seeds": [2341, 2342, 2343],
        "event_contract": {
            "unit": "nonoverlapping_returned_complete_causal_family",
            "temporal_overlap_components_excluded_whole": True,
            "minimum_readable_contacts": 3,
            "both_shafts_required": True,
            "frozen_classifier_ood_excluded": True,
            "contact_order": "patient_training_target_exact",
        },
        "natural_kmeans": {
            "k": 2,
            "feature": "masked_normalized_event_ranks",
            "use_masked_features": True,
            "minimum_shared_contacts": 3,
            "n_sample": 100,
            "n_tau_seeds": 5,
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
            "patient_matched_direction_purity_q05": 0.8842105263157894,
            "purity_benchmark_role": "reported_context_not_extra_gate",
        },
        "boundaries": {
            "patient_training_used": True,
            "patient_heldout_used": False,
            "natural_kmeans_used_for_field_selection": False,
            "natural_kmeans_used_for_node_acceptance": True,
            "figure_used_for_field_selection": False,
            "EE_EtoI_ZM": "off",
            "SNN_simulation_run": False,
        },
        "semantic_mapping": {
            "method": "runtime audit against frozen patient-training labels",
            "expected_numeric_label_to_mode": {"0": "MTB", "1": "MTA"},
        },
        "claim_boundary": (
            "This post-selection audit may accept or reject the already selected "
            "Node field on patient-training semantics. It cannot re-rank fields, "
            "access patient held-out events, activate EE/E-to-I/Z/M, or support "
            "patient-blind generalization."
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robust-config", type=Path, default=DEFAULT_ROBUST_CONFIG)
    parser.add_argument("--robust-aggregate", type=Path, default=DEFAULT_ROBUST_AGGREGATE)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure2-field", type=Path, default=DEFAULT_FIGURE2_FIELD)
    args = parser.parse_args(argv)
    payload = build_config(
        robust_config_path=args.robust_config.resolve(),
        robust_aggregate_path=args.robust_aggregate.resolve(),
        artifact_root=args.artifact_root.resolve(),
        figure2_field_path=args.figure2_field.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV15_NODE_POSTSELECTION_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "output": str(args.output.resolve()),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
