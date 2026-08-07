#!/usr/bin/env python3
"""Machine-check and freeze the pre-outcome V3.0 human-test release."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_release(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = ROOT / "results/topic5_event_innovation_impulse_response/v3_0"
    paths = {
        "phase0": root / "phase0/anchor_contract.json",
        "measurement": root / "phase1_measurement_validation_only/state_reliability.json",
        "innovation": ROOT / str(config["innovation_output_root"]) / "innovation_validity.json",
        "goal2": ROOT / str(config["local_response_output_root"]) / "local_projection_state.json",
        "goal3": ROOT / str(config["cumulative_output_root"]) / "cumulative_response_state.json",
        "synthetic": root / "synthetic_calibration/synthetic_identifiability_state.json",
        "transition_synthetic": ROOT / "results/topic5_event_innovation_state_space/v3_1/synthetic_calibration/synthetic_transition_acceptance_state.json",
        "handoff": root / "V3_1_HANDOFF_STATE.json",
    }
    expected = {
        "phase0": "PHASE0_COMPLETE",
        "measurement": "STATE_MEASUREMENT_COMPLETE_INNOVATION_PENDING",
        "innovation": "INNOVATION_VALIDITY_COMPLETE",
        "goal2": "LOCAL_RESPONSE_VALIDATION_COMPLETE",
        "goal3": "CUMULATIVE_RESPONSE_VALIDATION_COMPLETE",
        "synthetic": "SYNTHETIC_IDENTIFIABILITY_COMPLETE",
        "transition_synthetic": "SYNTHETIC_TRANSITION_IDENTIFICATION_COMPLETE",
    }
    records = {}
    failures = []
    for name, path in paths.items():
        if not path.exists():
            failures.append(f"missing:{name}")
            continue
        value = load(path)
        records[name] = value
        if name in expected and value.get("status") != expected[name]:
            failures.append(f"status:{name}:{value.get('status')}")
    for name in ("measurement", "innovation", "goal2", "goal3"):
        if name in records and records[name].get("human_test_outcomes_read") is not False:
            failures.append(f"test_read:{name}")
    if "handoff" in records and records["handoff"].get("status") not in {
        "OPEN", "NOT_TRIGGERED"
    }:
        failures.append("handoff_not_frozen")
    human_runner = ROOT / "scripts/run_topic5_event_innovation_v3_0_human_test.py"
    human_source = human_runner.read_text(encoding="utf-8")
    if "topic5_event_innovation_transition_v3_1" in human_source:
        failures.append("human_runner_imports_v3_1_transition")
    test_root = root / "human_exploratory"
    existing_test = list(test_root.glob("**/per_subject/*.json")) if test_root.exists() else []
    if existing_test:
        failures.append("human_test_artifacts_already_exist_before_release")
    phase0_hashes = {}
    phase0_subject_root = root / "phase0/per_subject"
    for path in sorted(phase0_subject_root.glob("*.json")):
        value = load(path)
        phase0_hashes[path.stem] = {
            "artifact_sha256": sha256(path),
            "split_index_sha256": value.get("split_index_sha256"),
            "continuity_contract": value.get("continuity_contract"),
            "anchor_contract": value.get("anchor_contract"),
        }
    if len(phase0_hashes) != 34:
        failures.append(f"phase0_patient_count:{len(phase0_hashes)}")
    checklist = {
        "source_continuity_manifest_and_reset_rule": "phase0" in records,
        "family_specific_state_score_innovation_schema": "measurement" in records and "innovation" in records,
        "blocked_crossfit_and_innovation_validity": "innovation" in records,
        "dense_train_and_nonoverlap_anchor_indices": len(phase0_hashes) == 34,
        "dimension_diagnostics_and_selection": "measurement" in records and "innovation" in records,
        "local_cumulative_iei_grid_frozen": "goal2" in records and "goal3" in records,
        "nulls_statistics_and_evidence_rules_frozen": "handoff" in records,
        "no_v3_1_transition_selection_in_human_runner": "human_runner_imports_v3_1_transition" not in failures,
        "config_code_source_input_hashes_recorded": True,
    }
    if not all(checklist.values()):
        failures.append("release_checklist_incomplete")
    specs = {
        "spec": ROOT / "docs/superpowers/specs/2026-08-03-topic5-event-innovation-low-rank-state-space-v3_0.md",
        "plan": ROOT / "docs/superpowers/plans/2026-08-03-topic5-event-innovation-low-rank-state-space-v3_0.md",
        "human_runner": human_runner,
        "test_helpers": ROOT / "src/topic5_event_innovation_test_v3_0.py",
    }
    return {
        "contract": str(config["contract"]),
        "status": "HUMAN_TEST_RELEASED" if not failures else "BLOCKED",
        "failures": failures,
        "checklist": checklist,
        "phase0_patient_indices": phase0_hashes,
        "inputs_sha256": {name: sha256(path) for name, path in paths.items() if path.exists()},
        "implementation_sha256": {name: sha256(path) for name, path in specs.items()},
        "config_sha256": sha256(config_path),
        "human_test_outcomes_read": False,
        "v2_7_completion_is_release_condition": False,
        "v3_1_transition_selected_in_human_run": False,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
        "runner_sha256": sha256(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path,
        default=ROOT / "config/topic5_event_innovation_v3_0.yaml",
    )
    args = parser.parse_args()
    state = build_release(args.config.resolve())
    destination = ROOT / "results/topic5_event_innovation_impulse_response/v3_0/HUMAN_TEST_RELEASE_STATE.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps(state, indent=2, sort_keys=True))
    if state["status"] != "HUMAN_TEST_RELEASED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
