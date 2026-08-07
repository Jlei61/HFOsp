#!/usr/bin/env python3
"""Patient-first acceptance of the locked v2.4 extension cohort."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_stable_repertoire_event_history"
ROOTS = {
    20: BASE / "v2_4/extension",
    40: BASE / "v2_4_h40/extension",
}
CONFIGS = {
    20: ROOT / "config/topic5_stable_repertoire_event_history_v2_4.yaml",
    40: ROOT / "config/topic5_stable_repertoire_event_history_v2_4_h40.yaml",
}
RELEASE = BASE / "v2_4/development_acceptance/LOCKED_EXTENSION_RELEASE.json"
OUTPUT = BASE / "acceptance_v2_4"
SPEC = ROOT / "docs/superpowers/specs/2026-08-02-topic5-stable-repertoire-event-history-v2_4.md"
MODULE = ROOT / "src/topic5_stable_repertoire_event_history_v2_4.py"
RUNNER = ROOT / "scripts/run_topic5_stable_repertoire_event_history_v2_4.py"


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


def one_sided(values, alternative: str) -> dict[str, float | int | None]:
    array = np.asarray(values, float)
    nonzero = array[np.abs(array) > 1e-15]
    if len(nonzero):
        statistic = wilcoxon(nonzero, alternative=alternative, zero_method="wilcox")
        p_wilcoxon = float(statistic.pvalue)
    else:
        p_wilcoxon = 1.0
    if alternative == "less":
        positive = int(np.sum(array < 0))
    elif alternative == "greater":
        positive = int(np.sum(array > 0))
    else:
        raise ValueError(alternative)
    return {
        "n": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "n_directional": positive,
        "wilcoxon_one_sided_p": p_wilcoxon,
        "sign_test_one_sided_p": float(
            binomtest(positive, len(array), 0.5, alternative="greater").pvalue
        ),
    }


def bootstrap_median_ci(values, seed: int, repeats: int = 20_000):
    array = np.asarray(values, float)
    rng = np.random.default_rng(int(seed))
    draws = np.median(
        array[rng.integers(0, len(array), size=(int(repeats), len(array)))], axis=1
    )
    return {
        "median": float(np.median(array)),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "repeats": int(repeats),
    }


def audit_circular_npz(path: Path):
    values = np.load(path, allow_pickle=False)
    prefixes = sorted({key.rsplit("_", 1)[0] for key in values.files if key.endswith("_history")})
    count = overlap = same = 0
    for prefix in prefixes:
        histories = values[f"{prefix}_history"]
        targets = values[f"{prefix}_target"]
        origin = values[f"{prefix}_origin_row"]
        donor = values[f"{prefix}_donor_row"]
        count += len(histories)
        overlap += sum(
            np.intersect1d(history, target).size > 0
            for history, target in zip(histories, targets)
        )
        same += int(np.sum(origin == donor))
    return {
        "n_rows": int(count),
        "overlap_rows": int(overlap),
        "same_origin_donor_rows": int(same),
        "pass": bool(count > 0 and overlap == 0 and same == 0),
    }


def all_contract_checks_pass(value) -> bool:
    if isinstance(value, dict):
        return all(all_contract_checks_pass(item) for item in value.values())
    if isinstance(value, bool):
        return value
    return True


def result_row(result, horizon: int):
    true = result["true_chronology"]
    matched = true["selected_matched_test_score"]
    state = true["low_dimensional_state"]["median_test_score"]
    selected_name = true["validation_selected_matched_baseline"]
    selected_info = true["matched_recency_candidates"][selected_name]
    true_recruitment_gain = -true["state_minus_matched"]["recruitment"]
    block_recruitment_gain = float(
        np.median(
            [
                item["matched_minus_state_gain"]["recruitment"]
                for item in result["coherent_block_shuffle"]
            ]
        )
    )
    circular_recruitment_gain = float(
        np.median(
            [
                item["matched_minus_state_gain"]["recruitment"]
                for item in result["safe_circular_pairing"]
                if "matched_minus_state_gain" in item
            ]
        )
    )
    return {
        "horizon": horizon,
        "subject": result["subject"],
        "dataset": result["dataset"],
        "n_test_windows": result["n_prediction_windows"]["test"],
        "selected_matched_baseline": selected_name,
        "selected_matched_decay": selected_info["decay"],
        "static_propagation": true["b0_static"]["test_score"]["propagation"],
        "static_recruitment": true["b0_static"]["test_score"]["recruitment"],
        "recent_h_propagation": true["b1_last_h"]["test_score"]["propagation"],
        "unordered_l_propagation": true["b2_unordered_l"]["test_score"]["propagation"],
        "first_h_propagation": true["b3_first_h"]["test_score"]["propagation"],
        "random_h_propagation": true["b3_random_h"]["median_test_score"]["propagation"],
        "time_nuisance_propagation": true["b6_time_iei_nuisance"]["test_score"]["propagation"],
        "matched_propagation": matched["propagation"],
        "matched_recruitment": matched["recruitment"],
        "state_propagation": state["propagation"],
        "state_minus_matched_propagation": true["state_minus_matched"]["propagation"],
        "state_minus_matched_recruitment": true["state_minus_matched"]["recruitment"],
        "state_minus_matched_repertoire": true["state_minus_matched"]["repertoire"],
        "true_minus_block_null_gain": result["chronology_gain"]["true_minus_block_null_gain"],
        "true_minus_circular_null_gain": result["chronology_gain"]["true_minus_circular_null_gain"],
        "true_minus_block_null_recruitment_gain": true_recruitment_gain - block_recruitment_gain,
        "true_minus_circular_null_recruitment_gain": true_recruitment_gain - circular_recruitment_gain,
        "dynamic_occupancy_reliability": result["validation_future_window_reliability"]["occupancy"]["train_mean_residualized"]["variance_reliability_median"],
        "dynamic_rank_reliability": result["validation_future_window_reliability"]["rank"]["train_mean_residualized"]["variance_reliability_median"],
        "dynamic_participation_reliability": result["validation_future_window_reliability"]["participation"]["train_mean_residualized"]["variance_reliability_median"],
        "history_duration_q50_seconds": result["time_scale_audit"]["test"]["history_duration_seconds"]["q50"],
        "target_duration_q50_seconds": result["time_scale_audit"]["test"]["target_duration_seconds"]["q50"],
        "validation_template_grade": result["train_to_partition_template_stability"]["validation"]["grade"],
        "test_template_grade": result["train_to_partition_template_stability"]["test"]["grade"],
    }


def main():
    release = load_json(RELEASE)
    if release["status"] != "START_LOCKED_28_PATIENT_EXTENSION":
        raise RuntimeError("extension was not released")
    expected_hashes = {
        "spec_sha256": sha256(SPEC),
        "module_sha256": sha256(MODULE),
        "runner_sha256": sha256(RUNNER),
        "config_h20_sha256": sha256(CONFIGS[20]),
        "config_h40_sha256": sha256(CONFIGS[40]),
    }
    if release["frozen_hashes"] != expected_hashes:
        raise RuntimeError("frozen implementation changed after release")
    subjects = release["locked_extension_subjects"]
    rows = []
    denominator = []
    artifact_audit = {}
    for horizon in (20, 40):
        state = load_json(ROOTS[horizon] / "STATE.json")
        for key in ("spec_sha256", "module_sha256", "runner_sha256"):
            if state[key] != expected_hashes[key]:
                raise RuntimeError(f"H={horizon}: stale {key}")
        config_key = f"config_h{horizon}_sha256"
        if state["config_sha256"] != expected_hashes[config_key]:
            raise RuntimeError(f"H={horizon}: stale config")
        failures = {item["subject"]: item["error"] for item in load_json(ROOTS[horizon] / "failures.json")}
        completed = set()
        config = yaml.safe_load(CONFIGS[horizon].open())
        for subject in subjects:
            path = ROOTS[horizon] / "per_subject" / f"{subject}.json"
            if not path.exists():
                denominator.append(
                    {
                        "horizon": horizon,
                        "subject": subject,
                        "dataset": subject.split("_", 1)[0],
                        "analysis_eligible": False,
                        "reason": failures.get(subject, "MISSING_WITHOUT_FAILURE_RECORD"),
                    }
                )
                continue
            completed.add(subject)
            result = load_json(path)
            if not all_contract_checks_pass(result["contract_checks"]):
                raise RuntimeError(f"H={horizon} {subject}: contract failed")
            raw = np.load(
                ROOT / config["dataset_root"] / f"{subject}.npz", allow_pickle=False
            )
            predictions = np.load(
                ROOTS[horizon] / "per_subject" / f"{subject}_predictions.npz",
                allow_pickle=False,
            )
            indices = np.concatenate(
                [predictions["history_event_indices"].ravel(), predictions["target_event_indices"].ravel()]
            )
            heldout = int(np.sum(np.asarray(raw["event_split"], int)[indices] != 0))
            circular = audit_circular_npz(
                ROOTS[horizon] / "per_subject" / f"{subject}_safe_circular_indices.npz"
            )
            if heldout or not circular["pass"]:
                raise RuntimeError(f"H={horizon} {subject}: artifact audit failed")
            artifact_audit[f"H{horizon}:{subject}"] = {
                "old_heldout20_indices": heldout,
                "safe_circular": circular,
            }
            rows.append(result_row(result, horizon))
            denominator.append(
                {
                    "horizon": horizon,
                    "subject": subject,
                    "dataset": result["dataset"],
                    "analysis_eligible": True,
                    "reason": "PASS_FROZEN_DATA_CONTRACT",
                }
            )
        if len(completed) != state["n_completed"]:
            raise RuntimeError(f"H={horizon}: completed artifact count mismatch")
    frame = pd.DataFrame(rows)
    denominator_frame = pd.DataFrame(denominator)
    primary = frame[frame.horizon == 20].copy()
    sensitivity = frame[frame.horizon == 40].copy()
    inference = {
        "primary_h20": {
            "state_minus_matched_propagation": one_sided(
                primary.state_minus_matched_propagation, "less"
            ),
            "true_minus_block_null_gain": one_sided(
                primary.true_minus_block_null_gain, "greater"
            ),
            "true_minus_circular_null_gain": one_sided(
                primary.true_minus_circular_null_gain, "greater"
            ),
            "matched_minus_static_propagation": one_sided(
                primary.matched_propagation - primary.static_propagation, "less"
            ),
            "matched_minus_unordered_l_propagation": one_sided(
                primary.matched_propagation - primary.unordered_l_propagation, "less"
            ),
            "unordered_l_minus_static_propagation": one_sided(
                primary.unordered_l_propagation - primary.static_propagation, "less"
            ),
            "unordered_l_minus_random_h": one_sided(
                primary.unordered_l_propagation - primary.random_h_propagation,
                "less",
            ),
            "recent_h_minus_random_h": one_sided(
                primary.recent_h_propagation - primary.random_h_propagation,
                "less",
            ),
            "state_minus_matched_recruitment": one_sided(
                primary.state_minus_matched_recruitment, "less"
            ),
            "true_minus_block_null_recruitment_gain": one_sided(
                primary.true_minus_block_null_recruitment_gain, "greater"
            ),
            "true_minus_circular_null_recruitment_gain": one_sided(
                primary.true_minus_circular_null_recruitment_gain, "greater"
            ),
            "matched_minus_static_recruitment": one_sided(
                primary.matched_recruitment - primary.static_recruitment, "less"
            ),
            "bootstrap_state_minus_matched_propagation": bootstrap_median_ci(
                primary.state_minus_matched_propagation, seed=2401
            ),
        },
        "sensitivity_h40": {
            "state_minus_matched_propagation": one_sided(
                sensitivity.state_minus_matched_propagation, "less"
            ),
            "true_minus_block_null_gain": one_sided(
                sensitivity.true_minus_block_null_gain, "greater"
            ),
            "true_minus_circular_null_gain": one_sided(
                sensitivity.true_minus_circular_null_gain, "greater"
            ),
            "matched_minus_static_propagation": one_sided(
                sensitivity.matched_propagation - sensitivity.static_propagation,
                "less",
            ),
        },
        "primary_dataset_strata": {
            dataset: {
                "n": int(len(group)),
                "median_state_minus_matched_propagation": float(
                    group.state_minus_matched_propagation.median()
                ),
                "n_state_beats_matched": int(
                    np.sum(group.state_minus_matched_propagation < 0)
                ),
            }
            for dataset, group in primary.groupby("dataset")
        },
    }
    primary_state = inference["primary_h20"]["state_minus_matched_propagation"]
    primary_block = inference["primary_h20"]["true_minus_block_null_gain"]
    primary_circular = inference["primary_h20"]["true_minus_circular_null_gain"]
    primary_pass = bool(
        primary_state["median"] < 0
        and primary_state["wilcoxon_one_sided_p"] < 0.05
        and primary_block["median"] > 0
        and primary_block["wilcoxon_one_sided_p"] < 0.05
        and primary_circular["median"] > 0
        and primary_circular["wilcoxon_one_sided_p"] < 0.05
    )
    recruitment_state = inference["primary_h20"]["state_minus_matched_recruitment"]
    recruitment_block = inference["primary_h20"]["true_minus_block_null_recruitment_gain"]
    recruitment_circular = inference["primary_h20"]["true_minus_circular_null_recruitment_gain"]
    recruitment_joint_pass = bool(
        recruitment_state["median"] < 0
        and recruitment_state["wilcoxon_one_sided_p"] < 0.05
        and recruitment_block["median"] > 0
        and recruitment_block["wilcoxon_one_sided_p"] < 0.05
        and recruitment_circular["median"] > 0
        and recruitment_circular["wilcoxon_one_sided_p"] < 0.05
    )
    final = {
        "contract": "topic5_stable_repertoire_event_history_v2_4_locked_extension_acceptance",
        "status": "CHRONOLOGY_SENSITIVE_LOW_DIMENSIONAL_STATE_SUPPORTED"
        if primary_pass
        else "UNORDERED_LONG_HISTORY_SUFFICIENT_NO_COHORT_LOW_DIMENSIONAL_STATE",
        "locked_extension_n_requested": 28,
        "primary_h20_n_eligible": int(len(primary)),
        "sensitivity_h40_n_eligible": int(len(sensitivity)),
        "development_patients_in_primary_inference": False,
        "primary_joint_gate_pass": primary_pass,
        "secondary_recruitment_joint_gate_pass": recruitment_joint_pass,
        "inference": inference,
        "measurement": {
            "primary_dynamic_reliability_medians": {
                "occupancy": float(primary.dynamic_occupancy_reliability.median()),
                "rank": float(primary.dynamic_rank_reliability.median()),
                "participation": float(primary.dynamic_participation_reliability.median()),
            },
            "primary_history_duration_q50_across_patients_seconds": float(
                primary.history_duration_q50_seconds.median()
            ),
            "primary_target_duration_q50_across_patients_seconds": float(
                primary.target_duration_q50_seconds.median()
            ),
            "validation_template_grade_counts": primary.validation_template_grade.value_counts().to_dict(),
            "test_template_grade_counts": primary.test_template_grade.value_counts().to_dict(),
        },
        "allowed_claim": (
            "Using more past events improves estimation of the future stable repertoire, but the locked extension does not support an additional cohort-level low-dimensional chronology-sensitive propagation state."
            if not recruitment_joint_pass
            else "The locked extension does not support a chronology-sensitive propagation state, but supports a distinct secondary chronology-sensitive recruitment-topography signal."
        ),
        "forbidden_claims": [
            "activity-dependent network shaping",
            "pathological graph evolution",
            "recovered biological connectivity",
            "general recurrent computation across patients",
        ],
        "frozen_hashes": expected_hashes,
        "artifact_audit_pass": True,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "extension_patient_horizon_summary.csv", index=False)
    denominator_frame.to_csv(OUTPUT / "denominator_audit.csv", index=False)
    with (OUTPUT / "ARTIFACT_AUDIT.json").open("w") as stream:
        json.dump(artifact_audit, stream, indent=2, sort_keys=True)
    with (OUTPUT / "LOCKED_EXTENSION_ACCEPTANCE.json").open("w") as stream:
        json.dump(final, stream, indent=2, sort_keys=True)
    print(json.dumps(final, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
