#!/usr/bin/env python3
"""Fail-closed release decision for the six-patient v2.4 development rerun."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SPEC = ROOT / "docs/superpowers/specs/2026-08-02-topic5-stable-repertoire-event-history-v2_4.md"
MODULE = ROOT / "src/topic5_stable_repertoire_event_history_v2_4.py"
RUNNER = ROOT / "scripts/run_topic5_stable_repertoire_event_history_v2_4.py"
CONFIGS = {
    20: ROOT / "config/topic5_stable_repertoire_event_history_v2_4.yaml",
    40: ROOT / "config/topic5_stable_repertoire_event_history_v2_4_h40.yaml",
}
ROOTS = {
    20: ROOT / "results/topic5_stable_repertoire_event_history/v2_4/development",
    40: ROOT / "results/topic5_stable_repertoire_event_history/v2_4_h40/development",
}
OUTPUT = ROOT / "results/topic5_stable_repertoire_event_history/v2_4/development_acceptance"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.load(path.open())


def all_nested_true(value) -> bool:
    if isinstance(value, dict):
        return all(all_nested_true(item) for item in value.values())
    if isinstance(value, bool):
        return value
    return True


def audit_circular_npz(path: Path) -> dict[str, int | bool]:
    values = np.load(path, allow_pickle=False)
    prefixes = sorted({key.rsplit("_", 1)[0] for key in values.files if key.endswith("_history")})
    checked = 0
    overlap = 0
    same_donor = 0
    for prefix in prefixes:
        histories = values[f"{prefix}_history"]
        targets = values[f"{prefix}_target"]
        origin = values[f"{prefix}_origin_row"]
        donor = values[f"{prefix}_donor_row"]
        checked += len(histories)
        overlap += sum(
            np.intersect1d(history, target).size > 0
            for history, target in zip(histories, targets)
        )
        same_donor += int(np.sum(origin == donor))
    return {
        "n_rows_checked": int(checked),
        "history_target_overlap_rows": int(overlap),
        "same_origin_donor_rows": int(same_donor),
        "pass": bool(checked > 0 and overlap == 0 and same_donor == 0),
    }


def main():
    primary_config = yaml.safe_load(CONFIGS[20].open())
    subjects = primary_config["development_subjects"]
    rows = []
    artifact_audit = {}
    for horizon in (20, 40):
        state = load_json(ROOTS[horizon] / "STATE.json")
        if state["n_completed"] != 6 or state["n_requested"] != 6:
            raise RuntimeError(f"H={horizon}: development run incomplete")
        expected = {
            "config_sha256": sha256(CONFIGS[horizon]),
            "spec_sha256": sha256(SPEC),
            "module_sha256": sha256(MODULE),
            "runner_sha256": sha256(RUNNER),
        }
        for key, value in expected.items():
            if state[key] != value:
                raise RuntimeError(f"H={horizon}: stale {key}")
        for subject in subjects:
            result = load_json(ROOTS[horizon] / "per_subject" / f"{subject}.json")
            if not all_nested_true(result["contract_checks"]):
                raise RuntimeError(f"H={horizon} {subject}: contract check failed")
            raw = np.load(
                ROOT / primary_config["dataset_root"] / f"{subject}.npz",
                allow_pickle=False,
            )
            predictions = np.load(
                ROOTS[horizon] / "per_subject" / f"{subject}_predictions.npz",
                allow_pickle=False,
            )
            indices = np.concatenate(
                [predictions["history_event_indices"].ravel(), predictions["target_event_indices"].ravel()]
            )
            heldout_count = int(np.sum(np.asarray(raw["event_split"], int)[indices] != 0))
            circular = audit_circular_npz(
                ROOTS[horizon] / "per_subject" / f"{subject}_safe_circular_indices.npz"
            )
            if heldout_count or not circular["pass"]:
                raise RuntimeError(f"H={horizon} {subject}: artifact index audit failed")
            artifact_audit[f"H{horizon}:{subject}"] = {
                "old_heldout20_indices": heldout_count,
                "safe_circular": circular,
            }
            true = result["true_chronology"]
            rows.append(
                {
                    "horizon": horizon,
                    "subject": subject,
                    "dataset": result["dataset"],
                    "n_test_windows": result["n_prediction_windows"]["test"],
                    "selected_matched_baseline": true["validation_selected_matched_baseline"],
                    "state_minus_matched_propagation": true["state_minus_matched"]["propagation"],
                    "state_minus_matched_recruitment": true["state_minus_matched"]["recruitment"],
                    "true_minus_block_null_gain": result["chronology_gain"]["true_minus_block_null_gain"],
                    "true_minus_circular_null_gain": result["chronology_gain"]["true_minus_circular_null_gain"],
                    "recent_h_propagation": true["b1_last_h"]["test_score"]["propagation"],
                    "unordered_l_propagation": true["b2_unordered_l"]["test_score"]["propagation"],
                    "first_h_propagation": true["b3_first_h"]["test_score"]["propagation"],
                    "random_h_propagation": true["b3_random_h"]["median_test_score"]["propagation"],
                    "dynamic_occupancy_reliability": result["validation_future_window_reliability"]["occupancy"]["train_mean_residualized"]["variance_reliability_median"],
                    "dynamic_rank_reliability": result["validation_future_window_reliability"]["rank"]["train_mean_residualized"]["variance_reliability_median"],
                    "dynamic_participation_reliability": result["validation_future_window_reliability"]["participation"]["train_mean_residualized"]["variance_reliability_median"],
                    "validation_template_grade": result["train_to_partition_template_stability"]["validation"]["grade"],
                    "test_template_grade": result["train_to_partition_template_stability"]["test"]["grade"],
                }
            )
    frame = pd.DataFrame(rows)
    primary = frame[frame.horizon == 20]
    sensitivity = frame[frame.horizon == 40]
    gates = {
        "p0_engineering_and_provenance": True,
        "primary_median_state_beats_matched_recency": bool(
            primary.state_minus_matched_propagation.median() < 0
        ),
        "primary_median_true_gain_beats_block_null": bool(
            primary.true_minus_block_null_gain.median() > 0
        ),
        "primary_median_true_gain_beats_circular_null": bool(
            primary.true_minus_circular_null_gain.median() > 0
        ),
        "propagation_not_hidden_by_participation": bool(
            primary.state_minus_matched_propagation.median() <= 0
        ),
        "sensitivity_direction_state": bool(
            sensitivity.state_minus_matched_propagation.median() < 0
        ),
        "sensitivity_direction_block_null": bool(
            sensitivity.true_minus_block_null_gain.median() > 0
        ),
        "sensitivity_direction_circular_null": bool(
            sensitivity.true_minus_circular_null_gain.median() > 0
        ),
    }
    release = all(gates.values())
    locked_hashes = {
        "spec_sha256": sha256(SPEC),
        "module_sha256": sha256(MODULE),
        "runner_sha256": sha256(RUNNER),
        "config_h20_sha256": sha256(CONFIGS[20]),
        "config_h40_sha256": sha256(CONFIGS[40]),
    }
    state = {
        "contract": "topic5_stable_repertoire_event_history_v2_4_development_release",
        "status": "START_LOCKED_28_PATIENT_EXTENSION" if release else "STOP_AFTER_DEVELOPMENT",
        "n_development_patients": 6,
        "development_patients_excluded_from_primary_inference": True,
        "locked_extension_subjects": primary_config["locked_extension_subjects"],
        "gates": gates,
        "primary_summary": {
            "median_state_minus_matched_propagation": float(
                primary.state_minus_matched_propagation.median()
            ),
            "n_state_beats_matched": int(
                np.sum(primary.state_minus_matched_propagation < 0)
            ),
            "median_true_minus_block_null_gain": float(
                primary.true_minus_block_null_gain.median()
            ),
            "median_true_minus_circular_null_gain": float(
                primary.true_minus_circular_null_gain.median()
            ),
            "n_unordered_l_beats_random_h": int(
                np.sum(primary.unordered_l_propagation < primary.random_h_propagation)
            ),
            "n_recent_h_beats_random_h": int(
                np.sum(primary.recent_h_propagation < primary.random_h_propagation)
            ),
        },
        "frozen_hashes": locked_hashes,
        "extension_run_must_match_all_frozen_hashes": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "development_patient_horizon_summary.csv", index=False)
    with (OUTPUT / "ARTIFACT_AUDIT.json").open("w") as stream:
        json.dump(artifact_audit, stream, indent=2, sort_keys=True)
    with (OUTPUT / "LOCKED_EXTENSION_RELEASE.json").open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

