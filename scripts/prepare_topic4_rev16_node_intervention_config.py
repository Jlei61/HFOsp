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
from scripts import audit_topic4_rev16_node_final_science as final_audit_contract  # noqa: E402
from scripts import prepare_topic4_rev16_node_final_science_config as final_config_contract  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_FINAL_CONFIG = ROOT / "config/topic4_rev16_node_final_science.json"
DEFAULT_FINAL_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev16/"
    "joint_m3_m4_candidates/final_science/analysis/node_final_science_audit.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev16_node_intervention.json"
OUTPUT_SCHEMA = "topic4_rev16_node_crossed_intervention_v2"


def build_config(
    *, final_config_path: Path, final_audit_path: Path,
    repository_root: Path = ROOT, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    final_config = json.loads(final_config_path.read_text())
    final_audit = json.loads(final_audit_path.read_text())
    if final_config.get("schema_id") != final_config_contract.OUTPUT_SCHEMA:
        raise RuntimeError("rev16 final-science config schema changed")
    if final_audit.get("schema_id") != final_audit_contract.OUTPUT_SCHEMA:
        raise RuntimeError("rev16 final-science audit schema changed")
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
        "resources": {
            "maximum_workers": 3,
            "numerical_threads_per_worker": 1,
            "safe_peak_rss_gib_per_worker": 16.0,
            "stop_launching_below_available_memory_gib": 80.0,
            "minimum_free_disk_gib": 40.0,
            "monitor_interval_seconds": 600,
            "long_run_launcher": "systemd-run --user plus nohup",
        },
    })
    payload["hotspot_construction"].update({
        "source": (
            "leave_one_network_out_equal_network_mode_early_onset_probability"
        ),
        "leave_one_network_out": True,
        "mode_discriminative_contrast": True,
        "covariate_footprint": "pulse_target_disk",
        "matching_covariates": [
            "mean_node_h", "targeted_E_neuron_count", "baseline_E_rate_hz",
        ],
        "maximum_control_standardized_l1": 2.0,
        "maximum_control_standardized_component": 1.0,
    })
    payload["claim_boundary"] = (
        "A strong local E-threshold pulse tests rev16 model-internal regional "
        "necessity and mode selectivity from the same checkpoint. Targets are "
        "leave-one-network-out mode-contrast hotspots and controls are matched "
        "over the actual pulse footprint. It is not a "
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
