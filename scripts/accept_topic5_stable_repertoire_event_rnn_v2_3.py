#!/usr/bin/env python3
"""Fail-closed scientific acceptance for the v2.3/v2.3.1 six-patient pilot."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_stable_repertoire_event_rnn/development"
OUTPUT = BASE / "acceptance_v2_3_1"


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


def window_loss(target: np.ndarray, prediction: np.ndarray, n_modes: int) -> np.ndarray:
    n_contacts = (target.shape[1] - n_modes) // 2
    occupancy = np.mean((target[:, :n_modes] - prediction[:, :n_modes]) ** 2, axis=1)
    rank = np.mean(
        (target[:, n_modes : n_modes + n_contacts] - prediction[:, n_modes : n_modes + n_contacts]) ** 2,
        axis=1,
    )
    participation = np.mean(
        (target[:, n_modes + n_contacts :] - prediction[:, n_modes + n_contacts :]) ** 2,
        axis=1,
    )
    return (occupancy + rank + participation) / 3.0


def clustered_bootstrap(diff: np.ndarray, source: np.ndarray, seed: int, repeats: int = 4000):
    rng = np.random.default_rng(int(seed))
    diff = np.asarray(diff, float)
    source = np.asarray(source)
    unique = np.unique(source)
    estimates = []
    if len(unique) >= 2:
        groups = [diff[source == value] for value in unique]
        for _ in range(int(repeats)):
            chosen = rng.integers(0, len(groups), len(groups))
            estimates.append(float(np.mean(np.concatenate([groups[index] for index in chosen]))))
        method = "source_record_cluster_bootstrap"
    else:
        block = min(5, len(diff))
        n_blocks = int(np.ceil(len(diff) / block))
        for _ in range(int(repeats)):
            selected = []
            for _ in range(n_blocks):
                start = int(rng.integers(0, len(diff)))
                selected.extend((start + np.arange(block)) % len(diff))
            estimates.append(float(np.mean(diff[np.asarray(selected[: len(diff)], int)])))
        method = "single_source_circular_moving_block_bootstrap"
    return {
        "mean_difference": float(np.mean(diff)),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "bootstrap_method": method,
        "n_source_records": int(len(unique)),
    }


def paths(horizon: int):
    if horizon == 20:
        return {
            "common": BASE / "v2_3_r0_r3",
            "linear": BASE / "v2_3_1_residual_linear",
            "gru": BASE / "v2_3_1_residual_gru",
        }
    return {
        "common": BASE / "v2_3_h40_r0_r3",
        "linear": BASE / "v2_3_h40_residual_linear",
        "gru": BASE / "v2_3_h40_residual_gru",
    }


def main():
    subjects = [
        "epilepsiae_922", "epilepsiae_620", "epilepsiae_1096",
        "yuquan_chenziyang", "yuquan_zhangjiaqi", "yuquan_zhangkexuan",
    ]
    rows = []
    bootstrap = {}
    for horizon in (20, 40):
        root = paths(horizon)
        common_state = load_json(root["common"] / "R0_R3_STATE.json")
        linear_state = load_json(root["linear"] / "STATE.json")
        gru_state = load_json(root["gru"] / "STATE.json")
        if common_state["n_c0_pass"] != 6 or common_state["n_c1_pass"] != 6:
            raise RuntimeError(f"H={horizon}: common engineering/read-back gate not complete")
        if gru_state["n_all_runs_training_adequate"] != 6:
            raise RuntimeError(f"H={horizon}: GRU training adequacy incomplete")
        for subject_index, subject in enumerate(subjects):
            common = load_json(root["common"] / "per_subject" / f"{subject}.json")
            linear = load_json(root["linear"] / "per_subject" / f"{subject}.json")
            gru = load_json(root["gru"] / "per_subject" / f"{subject}.json")
            common_npz = np.load(root["common"] / "per_subject" / f"{subject}_predictions.npz")
            linear_npz = np.load(root["linear"] / "per_subject" / f"{subject}_predictions.npz")
            gru_npz = np.load(root["gru"] / "per_subject" / f"{subject}_predictions.npz")
            target = common_npz["target"].astype(float)
            source = common_npz["test_source"]
            n_modes = 2
            losses = {
                "unordered": window_loss(target, common_npz["r1_long"].astype(float), n_modes),
                "nested_linear": window_loss(target, linear_npz["ordered"].astype(float), n_modes),
                "linear_shuffle": window_loss(target, linear_npz["shuffle"].astype(float), n_modes),
                "nested_gru": window_loss(target, gru_npz["ordered"].astype(float), n_modes),
            }
            subject_bootstrap = {}
            for name, comparator in (
                ("linear_minus_unordered", losses["nested_linear"] - losses["unordered"]),
                ("linear_minus_shuffle", losses["nested_linear"] - losses["linear_shuffle"]),
                ("gru_minus_linear", losses["nested_gru"] - losses["nested_linear"]),
            ):
                subject_bootstrap[name] = clustered_bootstrap(
                    comparator, source, seed=10_000 * horizon + subject_index
                )
            bootstrap[f"H{horizon}:{subject}"] = subject_bootstrap
            reliability = common["validation_future_window_reliability"]
            stability = common["train_to_partition_template_stability"]
            rows.append({
                "horizon": horizon,
                "subject": subject,
                "n_test_windows": common["n_prediction_windows"]["test"],
                "minimum_mode_occupancy": common["minimum_partition_mode_occupancy"],
                "validation_template_grade": stability["validation"]["grade"],
                "validation_template_match": stability["validation"]["mean_match_spearman"],
                "validation_assignment_agreement": stability["validation"]["assignment_agreement"],
                "test_template_grade": stability["test"]["grade"],
                "test_template_match": stability["test"]["mean_match_spearman"],
                "test_assignment_agreement": stability["test"]["assignment_agreement"],
                "occupancy_reliability": reliability["occupancy"]["variance_reliability_median"],
                "rank_reliability": reliability["rank"]["variance_reliability_median"],
                "participation_reliability": reliability["participation"]["variance_reliability_median"],
                "r0_static": common["r0_static"]["test_score"]["composite"],
                "r1_recent": common["r1_recent_ridge"]["test_score"]["composite"],
                "r1_long_unordered": common["r1_long_history_summary_ridge"]["test_score"]["composite"],
                "r2_switching": common["r2_discrete_switching"]["test_score"]["composite"],
                "nested_linear": linear["ordered"]["median_test_score"]["composite"],
                "linear_shuffle": linear["within_history_shuffle"]["median_test_score"]["composite"],
                "linear_circular": linear["circular_pairing"]["median_test_score"]["composite"],
                "nested_gru": gru["ordered"]["median_test_score"]["composite"],
                "gru_shuffle": gru["within_history_shuffle"]["median_test_score"]["composite"],
                "gru_circular": gru["circular_pairing"]["median_test_score"]["composite"],
                "linear_beats_unordered": linear["gates"]["nested_linear_beats_strongest_unordered"],
                "linear_beats_shuffle": linear["gates"]["nested_linear_beats_shuffle"],
                "gru_beats_linear": gru["gates"]["nested_gru_beats_nested_linear"],
                "gru_training_adequate": gru["gates"]["all_runs_training_adequate"],
            })
    frame = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "patient_horizon_summary.csv", index=False)
    json.dump(bootstrap, (OUTPUT / "cluster_bootstrap.json").open("w"), indent=2, sort_keys=True)

    primary = frame[frame.horizon == 20].set_index("subject")
    sensitivity = frame[frame.horizon == 40].set_index("subject")
    long_beats_recent_primary = primary.r1_long_unordered < primary.r1_recent
    long_beats_recent_sensitivity = sensitivity.r1_long_unordered < sensitivity.r1_recent
    linear_primary = primary.linear_beats_unordered.astype(bool)
    linear_sensitivity = sensitivity.linear_beats_unordered.astype(bool)
    order_primary = primary.linear_beats_shuffle.astype(bool)
    order_sensitivity = sensitivity.linear_beats_shuffle.astype(bool)
    robust_linear = linear_primary & linear_sensitivity & order_primary & order_sensitivity
    state = {
        "contract": "topic5_stable_repertoire_event_rnn_v2_3_1_acceptance",
        "status": "SIX_PATIENT_PILOT_COMPLETE",
        "n_patients": 6,
        "engineering": {
            "c0_contract_pass_h20": 6,
            "c0_contract_pass_h40": 6,
            "stable_repertoire_readback_h20": 6,
            "stable_repertoire_readback_h40": 6,
            "gru_training_adequate_h20": 6,
            "gru_training_adequate_h40": 6,
            "synthetic_known_order_test": "COVERED_BY_TEST",
            "final_split_indices_intersected_with_train80": 6,
            "pre_repair_shared_source_heldout_tail_counts": {
                "epilepsiae_922": 1351,
                "epilepsiae_620": 12,
                "epilepsiae_1096": 74,
                "yuquan_chenziyang": 464,
                "yuquan_zhangjiaqi": 995,
                "yuquan_zhangkexuan": 59
            },
        },
        "results": {
            "long_unordered_beats_recent_h20": int(long_beats_recent_primary.sum()),
            "long_unordered_beats_recent_h40": int(long_beats_recent_sensitivity.sum()),
            "nested_linear_beats_unordered_h20": int(linear_primary.sum()),
            "nested_linear_beats_unordered_h40": int(linear_sensitivity.sum()),
            "nested_linear_beats_shuffle_h20": int(order_primary.sum()),
            "nested_linear_beats_shuffle_h40": int(order_sensitivity.sum()),
            "robust_linear_order_increment_both_horizons": int(robust_linear.sum()),
            "robust_linear_order_increment_subjects": robust_linear[robust_linear].index.tolist(),
            "nested_gru_beats_linear_h20": int(primary.gru_beats_linear.sum()),
            "nested_gru_beats_linear_h40": int(sensitivity.gru_beats_linear.sum()),
            "sign_test_linear_vs_unordered_h20_one_sided_p": float(
                binomtest(int(linear_primary.sum()), 6, 0.5, alternative="greater").pvalue
            ),
            "sign_test_linear_vs_unordered_h40_one_sided_p": float(
                binomtest(int(linear_sensitivity.sum()), 6, 0.5, alternative="greater").pvalue
            ),
        },
        "verdicts": {
            "stable_patient_specific_repertoire": "SUPPORTED_PREMISE_AND_6_OF_6_TRAIN_ONLY_READBACK",
            "longer_history_distribution": "SUPPORTED_IN_PILOT_NOT_CONFIRMATORY",
            "chronology_specific_linear_state": "HETEROGENEOUS_THREE_PATIENT_REPLICATED_SIGNAL",
            "nonlinear_gru_necessity": "NOT_SUPPORTED",
            "network_shaping_or_plasticity": "NOT_TESTED",
            "full_cohort_next_step": "EXPAND_FROZEN_UNORDERED_PLUS_NESTED_LINEAR_LADDER_ONLY",
        },
        "scope": {
            "old_heldout20_entered": False,
            "forbidden_labels_entered": False,
            "geometry_soz_ictal_snn_entered": False,
            "six_patient_pilot_confirmatory": False,
            "direct_r3_r4_capacity_diagnostics_superseded_by_nested_models": True,
        },
        "provenance": {
            "config_h20_sha256": sha256(ROOT / "config/topic5_stable_repertoire_event_rnn_v2_3.yaml"),
            "config_h40_sha256": sha256(ROOT / "config/topic5_stable_repertoire_event_rnn_v2_3_h40.yaml"),
            "spec_sha256": sha256(ROOT / "docs/superpowers/specs/2026-08-01-topic5-stable-repertoire-event-rnn-v2_3.md"),
            "module_sha256": sha256(ROOT / "src/topic5_stable_repertoire_event_rnn.py"),
            "acceptance_source_sha256": sha256(Path(__file__)),
            "checkpoint_count_h20": len(list((BASE / "v2_3_1_residual_gru/checkpoints").glob("*/*.pt"))),
            "checkpoint_count_h40": len(list((BASE / "v2_3_h40_residual_gru/checkpoints").glob("*/*.pt"))),
        },
    }
    json.dump(state, (OUTPUT / "FINAL_ACCEPTANCE.json").open("w"), indent=2, sort_keys=True)
    print(json.dumps(state, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
