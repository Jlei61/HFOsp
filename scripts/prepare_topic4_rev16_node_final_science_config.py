#!/usr/bin/env python3
"""Freeze rev16 one-time held-out audit after KMeans post-selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_topic4_rev15_node_final_science_config as base  # noqa: E402
from scripts import prepare_topic4_rev16_node_postselection_config as post  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CANDIDATE_CONFIG = ROOT / "config/topic4_rev16_joint_m3_m4_candidates.json"
DEFAULT_CANDIDATE_AGGREGATE = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "joint_m3_m4_candidates/analysis/joint_candidates_aggregate.json"
)
DEFAULT_POSTSELECTION_CONFIG = ROOT / "config/topic4_rev16_node_postselection.json"
DEFAULT_POSTSELECTION_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "joint_m3_m4_candidates/postselection/analysis/node_postselection_audit.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_node_final_science.json"
OUTPUT_SCHEMA = "topic4_rev16_node_final_science_v1"


def build_config(
    *, candidate_config_path: Path, candidate_aggregate_path: Path,
    postselection_config_path: Path, postselection_audit_path: Path,
    rev12_config_path: Path = base.DEFAULT_REV12_CONFIG,
    repository_root: Path = ROOT, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    candidate_config = json.loads(candidate_config_path.read_text())
    aggregate = json.loads(candidate_aggregate_path.read_text())
    postselection_config = json.loads(postselection_config_path.read_text())
    if candidate_config.get("schema_id") != post.EXPECTED_CANDIDATE_SCHEMA:
        raise RuntimeError("rev16 joint-candidate config schema changed")
    if aggregate.get("schema_id") != post.EXPECTED_AGGREGATE_SCHEMA:
        raise RuntimeError("rev16 joint-candidate aggregate schema changed")
    if postselection_config.get("schema_id") != post.OUTPUT_SCHEMA:
        raise RuntimeError("rev16 post-selection config schema changed")
    payload = base.build_config(
        robust_config_path=candidate_config_path,
        robust_aggregate_path=candidate_aggregate_path,
        postselection_config_path=postselection_config_path,
        postselection_audit_path=postselection_audit_path,
        rev12_config_path=rev12_config_path,
        repository_root=repository_root, artifact_root=artifact_root,
    )
    payload.update({
        "schema_id": OUTPUT_SCHEMA,
        "scientific_role": (
            "development_only_rev16_one_time_heldout_topology_and_intervention_entry"
        ),
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
            "joint_m3_m4_candidates/final_science"
        ),
        "network_seeds": post.NETWORK_SEEDS,
    })
    payload["claim_boundary"] = (
        "This is a one-time development held-out and model-internal topology audit "
        "of one rev16 candidate already frozen before held-out access. It cannot "
        "rerank fields, establish patient-blind generalization, or activate EE, "
        "E-to-I, or Z/M."
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", type=Path, default=DEFAULT_CANDIDATE_CONFIG)
    parser.add_argument(
        "--candidate-aggregate", type=Path, default=DEFAULT_CANDIDATE_AGGREGATE,
    )
    parser.add_argument("--postselection-config", type=Path, default=DEFAULT_POSTSELECTION_CONFIG)
    parser.add_argument("--postselection-audit", type=Path, default=DEFAULT_POSTSELECTION_AUDIT)
    parser.add_argument("--rev12-config", type=Path, default=base.DEFAULT_REV12_CONFIG)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_config(
        candidate_config_path=args.candidate_config.resolve(),
        candidate_aggregate_path=args.candidate_aggregate.resolve(),
        postselection_config_path=args.postselection_config.resolve(),
        postselection_audit_path=args.postselection_audit.resolve(),
        rev12_config_path=args.rev12_config.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV16_NODE_FINAL_SCIENCE_CONFIG_PREPARED",
        "candidate_id": payload["selected_candidate"]["candidate_id"],
        "output": str(args.output.resolve()), "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
