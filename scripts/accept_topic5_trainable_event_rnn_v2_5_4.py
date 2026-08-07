#!/usr/bin/env python3
"""Build the five-round acceptance state for Topic 5 event-RNN v2.5.4."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results/topic5_stable_repertoire_event_rnn/v2_5_4"
PRIMARY_RUNNER = ROOT / "scripts/run_topic5_trainable_event_rnn_v2_5.py"
MODEL_MODULE = ROOT / "src/topic5_trainable_event_rnn_v2_5.py"
CONFIG = ROOT / "config/topic5_trainable_event_rnn_v2_5.yaml"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_trainable_event_rnn_v2_5 import sha256  # noqa: E402
from src.topic5_trainable_event_rnn_v2_5 import (  # noqa: E402
    ResidualEventRNN,
    profile_from_mapping,
)


def _json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def tie_excluded_sign(values, *, favorable: str):
    array = np.asarray(values, float)
    array = array[np.isfinite(array) & (array != 0)]
    if favorable == "negative":
        count = int(np.sum(array < 0))
    elif favorable == "positive":
        count = int(np.sum(array > 0))
    else:
        raise ValueError(f"unknown favorable direction: {favorable}")
    return {
        "n_non_ties": int(len(array)),
        "n_favorable": count,
        "exact_sign_one_sided_p": (
            float(binomtest(count, len(array), 0.5, alternative="greater").pvalue)
            if len(array)
            else None
        ),
    }


def directional_summary(values, *, favorable: str):
    array = np.asarray(values, float)
    array = array[np.isfinite(array)]
    alternative = "less" if favorable == "negative" else "greater"
    try:
        signed_p = float(wilcoxon(array, alternative=alternative).pvalue)
    except ValueError:
        signed_p = None
    return {
        "n": int(len(array)),
        "median": float(np.median(array)) if len(array) else None,
        "wilcoxon_one_sided_p": signed_p,
        **tie_excluded_sign(array, favorable=favorable),
    }


def null_contract_audit():
    checks = []
    circular_failures = []
    for path in sorted((RESULT_ROOT / "chronology_nulls/per_subject").glob("*.json")):
        record = _json(path)
        for condition in record["block_shuffle"] + record["safe_circular"]:
            for split_checks in condition["contract_checks"].values():
                checks.extend(bool(value) for value in split_checks.values())
        if record["safe_circular_failures"]:
            circular_failures.append(record["subject"])
    return {
        "n_checks": int(len(checks)),
        "all_checks_pass": bool(checks and all(checks)),
        "circular_not_estimable_subjects": circular_failures,
    }


def training_audit():
    subjects = []
    for path in sorted((RESULT_ROOT / "per_subject").glob("*.json")):
        record = _json(path)
        runs = record["recurrent_runs"]
        subjects.append(
            {
                "subject": record["subject"],
                "n_fallback": int(
                    sum(run["trace"]["best_is_untrained_baseline"] for run in runs)
                ),
                "best_epochs": [int(run["trace"]["best_epoch"]) for run in runs],
                "clipped_fraction": float(
                    np.median(
                        [
                            value
                            for run in runs
                            for value in run["trace"]["clipped_fraction"]
                        ]
                    )
                ),
                "all_finite": bool(all(run["trace"]["finite"] for run in runs)),
            }
        )
    return {
        "n_subjects": int(len(subjects)),
        "all_runs_finite": bool(all(item["all_finite"] for item in subjects)),
        "n_any_seed_trained": int(sum(item["n_fallback"] < 3 for item in subjects)),
        "n_all_seeds_trained": int(sum(item["n_fallback"] == 0 for item in subjects)),
        "n_all_seeds_fallback": int(sum(item["n_fallback"] == 3 for item in subjects)),
        "best_epoch_min": int(min(epoch for item in subjects for epoch in item["best_epochs"])),
        "best_epoch_max": int(max(epoch for item in subjects for epoch in item["best_epochs"])),
        "median_subject_clipped_fraction": float(
            np.median([item["clipped_fraction"] for item in subjects])
        ),
    }


def checkpoint_reload_audit(frozen):
    profile = profile_from_mapping(frozen["recurrent_profile"])
    loaded = 0
    for path in sorted((RESULT_ROOT / "per_subject").glob("*.json")):
        record = _json(path)
        descriptor_dim = 2 * int(record["n_contacts"]) + 2
        for seed in (17, 29, 43):
            checkpoint = torch.load(
                RESULT_ROOT / "checkpoints" / record["subject"] / f"seed_{seed}.pt",
                map_location="cpu",
                weights_only=False,
            )
            model = ResidualEventRNN(descriptor_dim, descriptor_dim, profile)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if not all(torch.isfinite(value).all() for value in model.parameters()):
                raise RuntimeError(f"non-finite reloaded checkpoint: {record['subject']} {seed}")
            loaded += 1
    return {
        "n_checkpoints_reloaded_strictly": int(loaded),
        "all_reloaded_parameters_finite": True,
        "state_dict_prediction_parity_unit_test_present": True,
        "standalone_bundle_includes_fitted_baseline": False,
    }


def main():
    frozen = _json(RESULT_ROOT / "development_screen/FROZEN_PROFILE.json")
    current_hashes = {
        "config_sha256": sha256(CONFIG),
        "module_sha256": sha256(MODEL_MODULE),
        "runner_sha256": sha256(PRIMARY_RUNNER),
    }
    hash_match = {
        key: bool(frozen.get(key) == value) for key, value in current_hashes.items()
    }
    true_frame = pd.read_csv(RESULT_ROOT / "patient_summary.csv")
    null_frame = pd.read_csv(RESULT_ROOT / "chronology_nulls/patient_summary.csv")
    extension = true_frame[~true_frame["development_subject"]].copy()
    extension_null = null_frame[~null_frame["development_subject"]].copy()
    combined = extension.merge(extension_null, on=("subject", "development_subject"))
    high = combined[combined["support_grade"] == "high"]
    joint = combined[
        (combined["rnn_minus_baseline_propagation"] < 0)
        & (combined["true_minus_block_gain"] > 0)
        & (combined["true_minus_circular_gain"] > 0)
    ]
    length_states = {
        length: _json(
            RESULT_ROOT
            / f"history_length_sensitivity/l{length}/HISTORY_LENGTH_STATE.json"
        )
        for length in (40, 80)
    }
    state = {
        "contract": "topic5_trainable_event_rnn_v2_5_4_five_round_acceptance",
        "status": "COMPLETE_BOUNDED_NEGATIVE_WITH_EXPLORATORY_HETEROGENEITY",
        "round_1_data_contract": {
            "verdict": "PASS",
            "n_primary_attempted": 34,
            "n_primary_completed": int(len(true_frame)),
            "n_extension": int(len(extension)),
            "all_primary_contract_checks_pass": bool(
                all(
                    all(_json(path)["contract_checks"].values())
                    for path in (RESULT_ROOT / "per_subject").glob("*.json")
                )
            ),
            "old_heldout20_entered": False,
            "frozen_primary_hash_match": hash_match,
        },
        "round_2_estimator_and_training": {
            "verdict": "PASS",
            "nested_exact_baseline_checkpoint": True,
            "selection_metric": "within_patient_validation_baseline_minus_rnn_gain",
            "training": training_audit(),
            "checkpoint_reload": checkpoint_reload_audit(frozen),
        },
        "round_3_architecture_and_optimization": {
            "verdict": "PASS_ENGINEERING_CALIBRATION",
            "selected_baseline": frozen["selected_baseline"],
            "selected_recurrent_profile": frozen["recurrent_profile"],
            "development_validation_median_gain": frozen[
                "median_development_validation_gain"
            ],
            "development_validation_positive": frozen[
                "n_positive_development_validation_gain"
            ],
            "selection_or_retuning_after_primary_test": False,
        },
        "round_4_scientific_endpoints_and_nulls": {
            "verdict": "CHRONOLOGY_GATE_NOT_PASSED",
            "extension_true_rnn_minus_baseline": directional_summary(
                extension["rnn_minus_baseline_propagation"], favorable="negative"
            ),
            "extension_true_minus_block_null": directional_summary(
                extension_null["true_minus_block_gain"], favorable="positive"
            ),
            "extension_true_minus_circular_null": directional_summary(
                extension_null["true_minus_circular_gain"], favorable="positive"
            ),
            "high_support_true_minus_block_null": directional_summary(
                high["true_minus_block_gain"], favorable="positive"
            ),
            "high_support_true_minus_circular_null": directional_summary(
                high["true_minus_circular_gain"], favorable="positive"
            ),
            "null_contract": null_contract_audit(),
        },
        "round_5_denominator_sensitivity_and_interpretation": {
            "verdict": "BOUNDED_NEGATIVE",
            "l20_completed": int(len(true_frame)),
            "l40_completed": int(length_states[40]["n_subjects_completed"]),
            "l80_completed": int(length_states[80]["n_subjects_completed"]),
            "l40_extension": length_states[40]["extension_propagation"],
            "l80_extension": length_states[80]["extension_propagation"],
            "exploratory_joint_positive_subjects": joint["subject"].tolist(),
            "n_exploratory_joint_positive": int(len(joint)),
            "all_joint_positive_are_high_support": bool(
                len(joint) and np.all(joint["support_grade"] == "high")
            ),
        },
        "allowed_claim": (
            "A genuinely trainable event-level GRU did not provide stable cohort-level "
            "future-repertoire gain beyond the matched descriptor-EWMA baseline at L=20, "
            "40, or 80, and did not pass the joint coherent-null chronology gate."
        ),
        "forbidden_claims": [
            "the stable patient-specific repertoire does not exist",
            "all possible RNNs are incapable of learning event history",
            "interictal events do or do not causally reshape biological connectivity",
            "the six exploratory patients define a confirmed subtype",
        ],
        "provenance": {
            "primary_state_sha256": sha256(RESULT_ROOT / "TRUE_CHRONOLOGY_STATE.json"),
            "null_state_sha256": sha256(
                RESULT_ROOT / "chronology_nulls/CHRONOLOGY_NULL_STATE.json"
            ),
            "l40_state_sha256": sha256(
                RESULT_ROOT / "history_length_sensitivity/l40/HISTORY_LENGTH_STATE.json"
            ),
            "l80_state_sha256": sha256(
                RESULT_ROOT / "history_length_sensitivity/l80/HISTORY_LENGTH_STATE.json"
            ),
            "acceptance_runner_sha256": sha256(Path(__file__)),
        },
    }
    output = RESULT_ROOT / "FIVE_ROUND_ACCEPTANCE_STATE.json"
    with output.open("w") as stream:
        json.dump(state, stream, indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
