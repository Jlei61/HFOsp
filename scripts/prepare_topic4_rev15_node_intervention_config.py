#!/usr/bin/env python3
"""Freeze the crossed same-checkpoint intervention after the final audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_FINAL_CONFIG = ROOT / "config/topic4_rev15_node_final_science.json"
DEFAULT_FINAL_AUDIT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
    "m3_robust_candidates/final_science/analysis/node_final_science_audit.json"
)
DEFAULT_OUTPUT = ROOT / "config/topic4_rev15_node_intervention.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(path: Path, *, repository_root: Path,
            artifact_root: Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"intervention input is missing: {path}")
    for root in (repository_root.resolve(), artifact_root.resolve()):
        try:
            return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}
        except ValueError:
            continue
    raise RuntimeError(f"intervention input lies outside frozen roots: {path}")


def _resolve_record(record: dict[str, str], *, repository_root: Path,
                    artifact_root: Path) -> Path:
    for root in (repository_root, artifact_root):
        path = root / str(record["path"])
        if path.is_file() and _sha256(path) == str(record["sha256"]):
            return path.resolve()
    raise RuntimeError(f"intervention input changed: {record['path']}")


def build_config(
    *, final_config_path: Path, final_audit_path: Path,
    repository_root: Path = ROOT, artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    final_config = json.loads(final_config_path.read_text())
    final_audit = json.loads(final_audit_path.read_text())
    if final_audit.get("status") != "NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION":
        raise RuntimeError("final Node audit has not advanced to intervention")
    decision = final_audit.get("decision", {})
    if decision.get("accepted_for_same_checkpoint_intervention") is not True:
        raise RuntimeError("final Node audit clauses are incomplete")
    if decision.get("node_freeze_permitted") is not False:
        raise RuntimeError("Node was frozen before the intervention")
    candidate_id = str(final_audit["candidate_id"])
    if candidate_id != final_config["selected_candidate"]["candidate_id"]:
        raise RuntimeError("intervention candidate differs from final audit")
    seeds = [int(value) for value in final_config["network_seeds"]]
    if seeds != [2341, 2342, 2343]:
        raise RuntimeError("intervention network pool changed")
    robust_record = final_config["inputs"]["robust_config"]
    robust_config_path = repository_root / str(robust_record["path"])
    if not robust_config_path.is_file():
        robust_config_path = artifact_root / str(robust_record["path"])
    if not robust_config_path.is_file() or _sha256(robust_config_path) != robust_record["sha256"]:
        raise RuntimeError("intervention robust config changed")
    robust_config = json.loads(robust_config_path.read_text())
    manifest_path = artifact_root / str(robust_config["candidate_manifest"])
    cohort_path = _resolve_record(
        final_config["inputs"]["cohort_config"],
        repository_root=repository_root, artifact_root=artifact_root,
    )
    classifier_path = _resolve_record(
        final_config["inputs"]["classifier_config"],
        repository_root=repository_root, artifact_root=artifact_root,
    )
    return {
        "schema_id": "topic4_rev15_node_crossed_intervention_v1",
        "scientific_role": "development_only_model_internal_crossed_hotspot_necessity",
        "output_root": (
            "results/topic4_sef_hfo/data_driven_node_dualmode_rev15/"
            "m3_robust_candidates/final_science/intervention"
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
            "robust_config": _record(
                robust_config_path, repository_root=repository_root,
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
        "network_seeds": seeds,
        "event_selection": {
            "one_event_per_native_mode_per_network": True,
            "event_set": "complete_returned_source_evaluable_causal_families",
            "representative_rule": (
                "closest_to_equal_network_mode_source_template_then_mode_rank_template"
            ),
        },
        "hotspot_construction": {
            "source": "equal_network_training_mode_early_onset_probability",
            "bin_mm": 1.0,
            "minimum_hotspot_separation_mm": 3.0,
            "matched_control": (
                "same_local_h_density_and_baseline_activity_outside_mode_template"
            ),
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
                "mode_identity", "rank_displacement", "source_topology_displacement",
            ],
            "mode_selective_contrast": (
                "predicted_mode_effect_gt_opposite_mode_effect_and_matched_control"
            ),
            "node_freeze_requires_intervention": True,
        },
        "mechanism_freeze": {"EE": "off", "E_to_I": "off", "Z_M": "off"},
        "claim_boundary": (
            "A strong local E-threshold pulse tests model-internal regional necessity "
            "and mode selectivity. It is not a patient intervention, cellular-core "
            "identification or therapeutic simulation."
        ),
    }


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
        "status": "REV15_NODE_CROSSED_INTERVENTION_CONFIG_PREPARED",
        "candidate_id": payload["candidate_id"],
        "network_seeds": payload["network_seeds"],
        "output": str(args.output.resolve()),
        "SNN_simulation_run": False,
    }, indent=2))


if __name__ == "__main__":
    main()
