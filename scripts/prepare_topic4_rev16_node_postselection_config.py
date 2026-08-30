#!/usr/bin/env python3
"""Prepare rev16 training-semantic KMeans audit after fresh Node selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_topic4_rev15_node_postselection_config as base  # noqa: E402
from scripts import audit_topic4_rev16_node_confirmation as confirmation_audit  # noqa: E402
from scripts import prepare_topic4_rev16_node_confirmation_config as confirmation  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CANDIDATE_CONFIG = ROOT / "config/topic4_rev16_node_confirmation.json"
DEFAULT_CANDIDATE_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "joint_m3_m4_candidates/analysis/joint_candidates_aggregate.json"
)
DEFAULT_CONFIRMATION_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "node_confirmation/analysis/confirmation_audit.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_node_postselection.json"
EXPECTED_CANDIDATE_SCHEMA = confirmation.OUTPUT_SCHEMA
EXPECTED_AGGREGATE_SCHEMA = "topic4_rev16_joint_m3_m4_candidates_aggregate_v1"
OUTPUT_SCHEMA = "topic4_rev16_node_postselection_v1"
NETWORK_SEEDS = confirmation.CONFIRMATION_SEEDS


def build_config(
    *, candidate_config_path: Path, candidate_aggregate_path: Path,
    confirmation_audit_path: Path,
    artifact_root: Path, repository_root: Path = ROOT,
    figure2_field_path: Path | None = None,
) -> dict[str, Any]:
    candidate_config = json.loads(candidate_config_path.read_text())
    aggregate = json.loads(candidate_aggregate_path.read_text())
    if candidate_config.get("schema_id") != EXPECTED_CANDIDATE_SCHEMA:
        raise RuntimeError("rev16 joint-candidate config schema changed")
    if aggregate.get("schema_id") != EXPECTED_AGGREGATE_SCHEMA:
        raise RuntimeError("rev16 joint-candidate aggregate schema changed")
    if candidate_config.get("search", {}).get("active_network_seeds") != NETWORK_SEEDS:
        raise RuntimeError("rev16 post-selection network pool changed")
    if aggregate.get("status") != "COMPLETE":
        raise RuntimeError("rev16 joint-candidate aggregate is incomplete")
    if aggregate.get("best_usable_anchor") is None:
        raise RuntimeError("no training-qualified rev16 Node anchor exists")
    if aggregate.get("ranking_contract", {}).get("J14_improvement") != (
        "3/3 fresh networks"
    ):
        raise RuntimeError("rev16 Node selection predates the full-J14 contract")
    confirmation_result = json.loads(confirmation_audit_path.read_text())
    if (
        confirmation_result.get("schema_id") != confirmation_audit.OUTPUT_SCHEMA
        or confirmation_result.get("status") != confirmation_audit.COMPLETE_STATUS
        or confirmation_result.get("candidate_id") != aggregate["best_usable_anchor"]
        or confirmation_result.get("network_seeds") != NETWORK_SEEDS
        or not confirmation_result.get("inventory", {}).get(
            "complete_cartesian_product"
        )
    ):
        raise RuntimeError("rev16 unseen-network confirmation is incomplete")

    payload = base.build_config(
        robust_config_path=candidate_config_path,
        robust_aggregate_path=candidate_aggregate_path,
        artifact_root=artifact_root,
        repository_root=repository_root,
        figure2_field_path=figure2_field_path,
    )
    payload.update({
        "schema_id": OUTPUT_SCHEMA,
        "scientific_role": (
            "development_only_rev16_postselection_natural_kmeans_and_fig4"
        ),
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
            "joint_m3_m4_candidates/postselection"
        ),
        "network_seeds": NETWORK_SEEDS,
    })
    payload["inputs"]["confirmation_audit"] = base._record(
        confirmation_audit_path, artifact_root,
    )
    payload["selected_candidate"]["selection_source"] = (
        "training-only rev16 joint M3+M4 selection on seeds 2351-2353; "
        "postselection uses frozen unseen-network confirmation seeds 2361-2363"
    )
    payload["boundaries"]["patient_heldout_used"] = False
    payload["boundaries"]["natural_kmeans_used_for_field_selection"] = False
    payload["boundaries"]["figure_used_for_field_selection"] = False
    payload["boundaries"]["EE_EtoI_ZM"] = "off"
    payload["claim_boundary"] = (
        "This rev16 post-selection audit may accept or reject one field already "
        "selected on three fresh networks. It cannot re-rank fields, access patient "
        "held-out events, activate EE/E-to-I/Z/M, or support patient-blind "
        "generalization."
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", type=Path, default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument(
        "--candidate-aggregate", type=Path, default=DEFAULT_CANDIDATE_AGGREGATE,
    )
    parser.add_argument(
        "--confirmation-audit", type=Path, default=DEFAULT_CONFIRMATION_AUDIT,
    )
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure2-field", type=Path, default=base.DEFAULT_FIGURE2_FIELD)
    args = parser.parse_args(argv)
    payload = build_config(
        candidate_config_path=args.candidate_config.resolve(),
        candidate_aggregate_path=args.candidate_aggregate.resolve(),
        confirmation_audit_path=args.confirmation_audit.resolve(),
        artifact_root=args.artifact_root.resolve(),
        figure2_field_path=args.figure2_field.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV16_NODE_POSTSELECTION_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "output": str(args.output.resolve()),
        "snn_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
