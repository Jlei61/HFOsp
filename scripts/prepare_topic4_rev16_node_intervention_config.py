#!/usr/bin/env python3
"""Freeze rev16 crossed same-checkpoint hotspot intervention."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_topic4_rev15_node_intervention_config as base  # noqa: E402
from scripts import prepare_topic4_rev16_node_postselection_config as post  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_FINAL_CONFIG = ROOT / "config/topic4_rev16_node_final_science.json"
DEFAULT_FINAL_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "joint_m3_m4_candidates/final_science/analysis/node_final_science_audit.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_node_intervention.json"
OUTPUT_SCHEMA = "topic4_rev16_node_crossed_intervention_v1"


def build_config(
    *, final_config_path: Path, final_audit_path: Path,
    repository_root: Path = ROOT, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    previous_seeds = base.EXPECTED_NETWORK_SEEDS
    base.EXPECTED_NETWORK_SEEDS = post.NETWORK_SEEDS
    try:
        payload = base.build_config(
            final_config_path=final_config_path,
            final_audit_path=final_audit_path,
            repository_root=repository_root, artifact_root=artifact_root,
        )
    finally:
        base.EXPECTED_NETWORK_SEEDS = previous_seeds
    payload.update({
        "schema_id": OUTPUT_SCHEMA,
        "scientific_role": (
            "development_only_rev16_model_internal_crossed_hotspot_necessity"
        ),
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
            "joint_m3_m4_candidates/final_science/intervention"
        ),
    })
    payload["claim_boundary"] = (
        "A strong local E-threshold pulse tests rev16 model-internal regional "
        "necessity and mode selectivity from the same checkpoint. It is not a "
        "patient intervention, cellular-core identification, or therapeutic "
        "simulation; EE, E-to-I and Z/M remain off."
    )
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-config", type=Path, default=DEFAULT_FINAL_CONFIG)
    parser.add_argument("--final-audit", type=Path, default=DEFAULT_FINAL_AUDIT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_config(
        final_config_path=args.final_config.resolve(),
        final_audit_path=args.final_audit.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV16_NODE_CROSSED_INTERVENTION_CONFIG_PREPARED",
        "candidate_id": payload["candidate_id"],
        "network_seeds": payload["network_seeds"],
        "output": str(args.output.resolve()), "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
