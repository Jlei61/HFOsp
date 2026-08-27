from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.topic5_continuous_marked_state_h2b.risk_probe import (
    build_risk_sets,
    chronological_split_map,
    make_positive_synthetic_risk_table,
    risk_set_hash,
    run_probe_table,
    time_label_permutation_audit,
    validate_risk_table,
)


def _anchor_table(n_seizures: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    seizure_rows = []
    leads = (5, 15, 30, 60, 120)
    for index in range(n_seizures):
        onset = 100_000.0 + 20_000.0 * index
        seizure_id = f"sz{index:02d}"
        seizure_rows.append({
            "patient_id": "p1",
            "seizure_id": seizure_id,
            "onset_time": np.float64(onset),
            "segment_id": "segment0",
        })
        times = [(f"case_{index}_{lead}", onset - 60.0 * lead) for lead in leads]
        times += [(f"control_{index}_{control}", onset - 15_000.0 - 100.0 * control)
                  for control in range(8)]
        for anchor_id, anchor_time in times:
            rows.append({
                "patient_id": "p1",
                "seed": 0,
                "anchor_id": anchor_id,
                "anchor_time": np.float64(anchor_time),
                "segment_id": "segment0",
                "segment_start": np.float64(0.0),
                "segment_end": np.float64(400_000.0),
                "observation_available": True,
                "observation_signature": "complete_10m",
                "in_ictal_or_postictal": False,
                "wrong_time_donor_valid": True,
                "wrong_time_same_segment": True,
                "wrong_time_exclusion_clear": True,
                "history__recent_count": float(index),
                "observation__spectral": float(anchor_time % 17),
                "state__persistent_0": float(anchor_time % 23),
                "memoryless__code_0": float(anchor_time % 19),
                "wrong_time__state_0": float(anchor_time % 29),
            })
    return pd.DataFrame(rows), pd.DataFrame(seizure_rows)


def test_chronological_split_is_60_20_20_and_tiered():
    seizures = pd.DataFrame({
        "seizure_id": [f"s{i}" for i in range(10)],
        "onset_time": np.arange(10, dtype=float),
    })
    mapping, tier = chronological_split_map(seizures)
    assert tier == "primary_chronological"
    assert list(mapping.values()).count("TRAIN") == 6
    assert list(mapping.values()).count("SELECT") == 2
    assert list(mapping.values()).count("TEST") == 2


def test_build_risk_sets_enforces_same_segment_horizon_and_split_contract():
    anchors, seizures = _anchor_table()
    frame, audit = build_risk_sets(
        anchors, seizures, controls_per_case=3, random_seed=7,
    )
    assert frame["risk_set_id"].nunique() == 10 * 5
    assert set(frame.groupby("risk_set_id").size()) == {4}
    assert frame.loc[~frame["is_case"], "horizon_seizure_free"].all()
    assert frame.groupby(["patient_id", "seizure_id"])["split"].nunique().max() == 1
    assert audit["identical_risk_sets_across_arms"] is True
    assert len(audit["risk_set_hash"]) == 64


def test_lead_sensitivity_keeps_nonprimary_seizure_without_promoting_tier():
    anchors, seizures = _anchor_table(n_seizures=4)
    seizures["primary_30min_supported"] = True
    seizures.loc[seizures["seizure_id"] == "sz03", "primary_30min_supported"] = False
    anchors = anchors[anchors["anchor_id"] != "case_3_30"].copy()
    frame, audit = build_risk_sets(
        anchors, seizures, controls_per_case=3, random_seed=17,
    )
    per_lead = frame.groupby("lead_minutes")["seizure_id"].nunique().to_dict()
    assert per_lead[5] == 4
    assert per_lead[15] == 4
    assert per_lead[30] == 3
    assert audit["split_by_patient"]["p1"]["evaluation_tier"] == (
        "descriptive_case_series"
    )
    assert set(frame["split"]) == {"DESCRIPTIVE"}


def test_validator_rejects_exact_anchor_cross_split_leakage():
    frame = make_positive_synthetic_risk_table(n_seizures=20, random_seed=1)
    train_index = frame.index[(frame["split"] == "TRAIN") & ~frame["is_case"]][0]
    test_index = frame.index[(frame["split"] == "TEST") & ~frame["is_case"]][0]
    frame.loc[test_index, "anchor_id"] = frame.loc[train_index, "anchor_id"]
    with pytest.raises(ValueError, match="same anchor"):
        validate_risk_table(frame)


def test_validator_rejects_same_time_with_different_ids_across_splits():
    frame = make_positive_synthetic_risk_table(n_seizures=20, random_seed=11)
    train_index = frame.index[(frame["split"] == "TRAIN") & ~frame["is_case"]][0]
    test_index = frame.index[(frame["split"] == "TEST") & ~frame["is_case"]][0]
    frame.loc[test_index, "anchor_time"] = frame.loc[train_index, "anchor_time"]
    with pytest.raises(ValueError, match="same time point"):
        validate_risk_table(frame)


def test_validator_rejects_wrong_time_donor_contract_failure():
    frame = make_positive_synthetic_risk_table(n_seizures=12, random_seed=12)
    frame.loc[0, "wrong_time_exclusion_clear"] = False
    with pytest.raises(ValueError, match="wrong-time donor contract"):
        validate_risk_table(frame)


def test_primary_arms_do_not_lose_seizures_to_secondary_wrong_time_support():
    frame = make_positive_synthetic_risk_table(n_seizures=12, random_seed=121)
    frame.loc[0, "wrong_time_donor_valid"] = False
    frame.loc[0, "wrong_time_same_segment"] = False
    frame.loc[0, "wrong_time_exclusion_clear"] = False
    frame.loc[0, "wrong_time__state_0"] = np.nan
    arms = ("B_history", "B_observation", "B_state", "memoryless")
    audit = validate_risk_table(
        frame, arms=arms, require_wrong_time=False,
    )
    result = run_probe_table(frame, arms=arms)
    assert audit["wrong_time_required_for_entry"] is False
    assert result.audit["arms"] == list(arms)
    assert "correct_minus_wrong_time_conditional_log_loss" not in result.per_seed


def test_validator_rejects_lead_dependent_seizure_split():
    frame = make_positive_synthetic_risk_table(n_seizures=20, random_seed=2)
    first = frame[frame["seizure_id"] == "sz000"].copy()
    first["risk_set_id"] = first["risk_set_id"] + "__lead15m"
    first["lead_minutes"] = 15
    first["anchor_id"] = first["anchor_id"] + "__lead15m"
    first.loc[~first["is_case"], "anchor_time"] += 123.456
    first.loc[first["is_case"], "anchor_time"] = (
        first.loc[first["is_case"], "seizure_onset"] - 15 * 60.0
    )
    first["split"] = "SELECT"
    mixed = pd.concat([frame, first], ignore_index=True)
    with pytest.raises(ValueError, match="different splits across lead"):
        validate_risk_table(mixed)


def test_positive_synthetic_recovers_frozen_state_increment_and_direct_controls():
    frame = make_positive_synthetic_risk_table(
        n_seizures=60, n_seeds=2, state_strength=4.0, random_seed=3,
    )
    result = run_probe_table(frame)
    assert (result.per_seed["state_minus_observation_conditional_log_loss"] < -0.25).all()
    assert (result.per_seed["persistent_minus_memoryless_conditional_log_loss"] < -0.20).all()
    assert (result.per_seed["correct_minus_wrong_time_conditional_log_loss"] < -0.20).all()
    assert result.patient_medians["n_optimizer_seeds"].iloc[0] == 2
    assert result.audit["seed_is_patient_replicate"] is False
    assert "negative" in result.audit["primary_metric"]


def test_time_label_permutation_returns_state_increment_to_near_zero():
    frame = make_positive_synthetic_risk_table(
        n_seizures=40, state_strength=4.0, random_seed=4,
    )
    audit = time_label_permutation_audit(
        frame, n_permutations=16, random_seed=5,
    )
    assert audit["observed_state_minus_observation"] < -0.25
    assert abs(audit["null_median"]) < 0.15
    assert audit["null_q025"] < 0 < audit["null_q975"]


@pytest.mark.parametrize(
    ("n_seizures", "tier", "split"),
    [
        (7, "sensitivity_loso", "LOSO"),
        (3, "descriptive_case_series", "DESCRIPTIVE"),
        (1, "not_estimable", "NOT_ESTIMABLE"),
    ],
)
def test_support_tiers_run_without_promoting_evidence(n_seizures, tier, split):
    frame = make_positive_synthetic_risk_table(n_seizures=12, random_seed=6)
    keep_ids = [f"sz{i:03d}" for i in range(n_seizures)]
    frame = frame[frame["seizure_id"].isin(keep_ids)].copy()
    frame["evaluation_tier"] = tier
    frame["split"] = split
    result = run_probe_table(frame)
    assert result.per_seed["evaluation_tier"].iloc[0] == tier
    if n_seizures >= 2:
        assert result.per_seed["B_state__status"].iloc[0] == "ok"
        assert np.isfinite(result.per_seed["B_state__conditional_log_loss"].iloc[0])
    else:
        assert result.per_seed["B_state__status"].iloc[0] == "NOT_ESTIMABLE"
        assert "state_minus_observation_conditional_log_loss" not in result.per_seed


def test_risk_set_hash_does_not_depend_on_comparison_arm_features():
    frame = make_positive_synthetic_risk_table(n_seizures=12, random_seed=9)
    before = risk_set_hash(frame)
    frame["state__persistent_0"] *= -7.0
    frame["wrong_time__state_0"] += 123.0
    assert risk_set_hash(frame) == before
