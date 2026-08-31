#!/usr/bin/env python3
"""Freeze the rev17 same-checkpoint mode-selective hotspot intervention."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_FINAL_CONFIG = ROOT / "config/topic4_rev17_node_final_science.json"
DEFAULT_FINAL_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
    "node_final_science/analysis/node_final_science_audit.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev17_node_intervention.json"
NETWORK_SEEDS = [2381, 2382, 2383]
OUTPUT_SCHEMA = "topic4_rev17_node_crossed_intervention_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    path: Path, *, repository_root: Path, artifact_root: Path,
) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"rev17 intervention input is missing: {path}")
    for root in (repository_root.resolve(), artifact_root.resolve()):
        try:
            return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
        except ValueError:
            continue
    raise RuntimeError("rev17 intervention input is outside frozen roots")


def _resolve(
    record: Mapping[str, Any], *, repository_root: Path, artifact_root: Path,
) -> Path:
    for root in (repository_root, artifact_root):
        path = root / str(record["path"])
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"rev17 intervention input changed: {record['path']}")


def build_config(
    *, final_config_path: Path, final_audit_path: Path,
    repository_root: Path = ROOT, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    final_config = json.loads(final_config_path.read_text())
    final_audit = json.loads(final_audit_path.read_text())
    if final_config.get("schema_id") != "topic4_rev17_node_final_science_v1":
        raise RuntimeError("rev17 final-science config schema changed")
    if (
        final_audit.get("schema_id") != "topic4_rev17_node_final_science_audit_v1"
        or final_audit.get("status")
        != "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION"
        or final_audit.get("decision", {}).get(
            "accepted_for_same_checkpoint_intervention"
        ) is not True
        or final_audit.get("decision", {}).get("node_freeze_permitted") is not False
    ):
        raise RuntimeError("rev17 final Node audit does not permit intervention")
    candidate_id = str(final_config["selected_candidate"]["candidate_id"])
    mapping_sha256 = str(final_config["selected_candidate"]["mapping_sha256"])
    if (
        final_audit.get("candidate_id") != candidate_id
        or final_audit.get("inputs", {}).get("config", {}).get("sha256")
        != _sha256(final_config_path)
    ):
        raise RuntimeError("rev17 intervention candidate/final config changed")
    seeds = [int(value) for value in final_config["network_seeds"]]
    if seeds != NETWORK_SEEDS:
        raise RuntimeError("rev17 intervention network pool changed")
    confirmation_path = _resolve(
        final_config["inputs"]["confirmation_config"],
        repository_root=repository_root, artifact_root=artifact_root,
    )
    manifest_path = _resolve(
        final_config["inputs"]["confirmation_manifest"],
        repository_root=repository_root, artifact_root=artifact_root,
    )
    confirmation = json.loads(confirmation_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    candidates = {row["candidate_id"]: row for row in manifest["candidates"]}
    if (
        confirmation.get("schema_id") != "topic4_rev17_node_confirmation_v1"
        or confirmation.get("selected_candidate", {}).get("candidate_id")
        != candidate_id
        or confirmation.get("selected_candidate", {}).get("mapping_sha256")
        != mapping_sha256
        or manifest.get("status") != "REV17_NODE_CONFIRMATION_CANDIDATES_FROZEN"
        or manifest.get("config_sha256") != _sha256(confirmation_path)
        or candidates.get(candidate_id, {}).get("node_mapping", {}).get(
            "mapping_sha256"
        ) != mapping_sha256
    ):
        raise RuntimeError("rev17 confirmation mapping is not frozen")
    cohort_path = _resolve(
        final_config["inputs"]["cohort_config"],
        repository_root=repository_root, artifact_root=artifact_root,
    )
    classifier_path = _resolve(
        final_config["inputs"]["classifier_config"],
        repository_root=repository_root, artifact_root=artifact_root,
    )
    return {
        "schema_id": OUTPUT_SCHEMA,
        "scientific_role": (
            "development_only_rev17_model_internal_crossed_hotspot_necessity"
        ),
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev17/"
            "node_final_science/intervention"
        ),
        "inputs": {
            "final_science_config": _record(
                final_config_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "final_science_audit": _record(
                final_audit_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            # Established intervention worker names retained as explicit aliases.
            "robust_config": _record(
                confirmation_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "robust_manifest": _record(
                manifest_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "cohort_config": _record(
                cohort_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
            "classifier_config": _record(
                classifier_path, repository_root=repository_root,
                artifact_root=artifact_root,
            ),
        },
        "candidate_id": candidate_id,
        "candidate_mapping_sha256": mapping_sha256,
        "network_seeds": seeds,
        "event_selection": {
            "one_event_per_native_mode_per_network": True,
            "event_set": "complete_returned_source_evaluable_causal_families",
            "representative_rule": (
                "within_network_joint_source_topology_and_contact_rank_medoid"
            ),
            "visual_selection": False,
        },
        "hotspot_construction": {
            "source": (
                "leave_one_network_out_equal_network_mode_early_onset_probability"
            ),
            "bin_mm": 1.0,
            "early_support_fraction": 0.10,
            "leave_one_network_out": True,
            "mode_discriminative_contrast": True,
            "minimum_hotspot_separation_mm": 3.0,
            "matched_control": (
                "off_template_matched_over_actual_pulse_footprint"
            ),
            "covariate_footprint": "pulse_target_disk",
            "matching_covariates": [
                "mean_node_h", "mean_delta_vtheta",
                "targeted_E_neuron_count", "baseline_E_rate_hz",
            ],
            "matching_covariate_keys": [
                "h_mean", "delta_vtheta_mean", "e_density", "baseline_rate_hz",
            ],
            "maximum_control_standardized_l1": 2.5,
            "maximum_control_standardized_component": 1.0,
            "uses_patient_heldout": False,
        },
        "intervention": {
            "checkpoint_lead_ms": 40.0,
            "continuation_ms": 300.0,
            "maximum_event_shift_ms": 200.0,
            "pulse_delay_from_checkpoint_ms": 5.0,
            "pulse_duration_ms": 70.0,
            "pulse_delta_vtheta_mv": 20.0,
            "target_radius_mm": 1.2,
            "baseline_activity_window_ms": [500.0, 1500.0],
            "arms_per_native_mode": [
                "sham", "MTA_hotspot", "MTA_matched_off_template",
                "MTB_hotspot", "MTB_matched_off_template",
            ],
            "crossed_design": True,
            "common_checkpoint_state_and_random_stream": True,
        },
        "decision": {
            "independent_unit": "network_seed",
            "required_selective_networks": 2,
            "primary_ordered_endpoint": ["event_survival", "onset_latency"],
            "continuous_explanatory_endpoints": [
                "mode_identity", "rank_displacement",
                "source_topology_displacement",
            ],
            "mode_selective_contrast": (
                "predicted_mode_effect_gt_opposite_mode_effect_and_matched_control"
            ),
            "node_freeze_requires_intervention": True,
        },
        "resources": {
            "maximum_workers": 3,
            "numerical_threads_per_worker": 1,
            "safe_peak_rss_gib_per_worker": 16.0,
            "stop_launching_below_available_memory_gib": 80.0,
            "minimum_free_disk_gib": 40.0,
            "monitor_interval_seconds": 600,
            "long_run_launcher": "systemd-run --user plus nohup",
        },
        "mechanism_freeze": {"EE": "off", "E_to_I": "off", "Z_M": "off"},
        "claim_boundary": (
            "A strong local threshold-raising pulse tests rev17 model-internal "
            "regional necessity and mode selectivity from the same checkpoint. "
            "Targets are leave-one-network-out mode-contrast hotspots and controls "
            "are matched over the pulse footprint. This is not a patient, cellular "
            "core or therapeutic intervention; EE, E-to-I and Z/M remain off."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-config", type=Path, default=DEFAULT_FINAL_CONFIG)
    parser.add_argument("--final-audit", type=Path, default=DEFAULT_FINAL_AUDIT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = build_config(
        final_config_path=args.final_config.resolve(),
        final_audit_path=args.final_audit.resolve(),
        artifact_root=args.artifact_root.resolve(),
    )
    args.output.resolve().write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": "REV17_NODE_CROSSED_INTERVENTION_CONFIG_PREPARED",
        "candidate_id": payload["candidate_id"],
        "network_seeds": payload["network_seeds"],
        "output": str(args.output.resolve()), "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
