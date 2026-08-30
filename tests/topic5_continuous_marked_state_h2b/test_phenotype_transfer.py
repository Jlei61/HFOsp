from __future__ import annotations

import numpy as np
import pytest

from src.topic5_continuous_marked_state_h2b.phenotype_transfer import (
    make_synthetic_phenotype_table,
    run_phenotype_table,
    target_table_hash,
    validate_phenotype_table,
)


def test_positive_synthetic_recovers_continuous_and_multiclass_transfer():
    frame = make_synthetic_phenotype_table(n_seizures=60, n_seeds=2, random_seed=1)
    result = run_phenotype_table(frame)
    effects = result.patient_medians.set_index("target_name")[
        "state_minus_observation_loss"
    ]
    assert effects["early_recruitment_extent"] < -1.0
    assert effects["frozen_subtype"] < -0.01
    assert result.patient_medians["n_optimizer_seeds"].eq(2).all()
    assert result.audit["target_reclustered"] is False
    assert result.audit["regularization_selected_only_on_train_select"] is True


def test_target_hash_is_feature_independent():
    frame = make_synthetic_phenotype_table(random_seed=2)
    before = target_table_hash(frame)
    frame["state__persistent_0"] *= -99
    frame["wrong_time__state_0"] += 50
    assert target_table_hash(frame) == before


def test_target_must_be_frozen_and_have_provenance():
    frame = make_synthetic_phenotype_table(random_seed=3)
    frame.loc[0, "target_frozen"] = False
    with pytest.raises(ValueError, match="not frozen"):
        validate_phenotype_table(frame)
    frame = make_synthetic_phenotype_table(random_seed=3)
    frame.loc[0, "target_source_sha256"] = "not-a-hash"
    with pytest.raises(ValueError, match="provenance"):
        validate_phenotype_table(frame)


def test_missing_target_is_machine_readable_not_estimable():
    frame = make_synthetic_phenotype_table(missing_target=True)
    result = run_phenotype_table(frame)
    assert set(result.per_seed["status"]) == {"NOT_ESTIMABLE_MISSING_TARGET"}
    assert result.audit["status"] == "NOT_ESTIMABLE_NO_USABLE_FROZEN_TARGET"
    assert "state_minus_observation_loss" not in result.per_seed


def test_missing_target_columns_are_machine_readable_not_estimable():
    frame = make_synthetic_phenotype_table().drop(columns=["target_value"])
    result = run_phenotype_table(frame)
    assert result.per_seed.empty
    assert result.audit["status"] == "NOT_ESTIMABLE_MISSING_TARGET_COLUMNS"
    assert result.audit["missing_columns"] == ["target_value"]


def test_missing_wrong_time_donor_does_not_gate_frozen_phenotype_transfer():
    """Secondary phenotype uses primary risk-table support, not donor support."""
    frame = make_synthetic_phenotype_table(n_seizures=30, random_seed=9)
    frame["wrong_time__state_0"] = np.nan

    result = run_phenotype_table(frame)

    assert result.audit["status"] == "COMPLETE"
    assert result.audit["matched_wrong_time_is_not_a_phenotype_gate"] is True
    assert result.patient_medians["state_minus_observation_loss"].notna().all()
    assert "correct_minus_wrong_time_loss" not in result.patient_medians


@pytest.mark.parametrize(
    ("count", "tier", "split", "expected_status"),
    [
        (7, "sensitivity_loso", "LOSO", "ok"),
        (3, "descriptive_case_series", "DESCRIPTIVE", "ok"),
        (1, "not_estimable", "NOT_ESTIMABLE",
         "NOT_ESTIMABLE_FEWER_THAN_TWO_SEIZURES"),
    ],
)
def test_support_tiers_do_not_promote_low_support(
    count, tier, split, expected_status,
):
    frame = make_synthetic_phenotype_table(n_seizures=30, n_seeds=1, random_seed=4)
    keep = [f"sz{i:03d}" for i in range(count)]
    frame = frame[frame["seizure_id"].isin(keep)].copy()
    frame["evaluation_tier"] = tier
    frame["split"] = split
    result = run_phenotype_table(frame)
    assert set(result.per_seed["evaluation_tier"]) == {tier}
    if count >= 2:
        # Tiny binary folds may be unestimable if a class disappears; continuous
        # recruitment remains estimable and never gets promoted above its tier.
        continuous = result.per_seed[
            result.per_seed["target_name"] == "early_recruitment_extent"
        ]
        assert continuous["status"].iloc[0] == expected_status
        assert np.isfinite(continuous["state__loss"].iloc[0])
    else:
        assert set(result.per_seed["status"]) == {expected_status}


def test_frozen_target_cannot_change_across_optimizer_seeds():
    frame = make_synthetic_phenotype_table(n_seeds=2, random_seed=5)
    selected = (
        (frame["seed"] == 1)
        & (frame["seizure_id"] == "sz000")
        & (frame["target_name"] == "early_recruitment_extent")
    )
    frame.loc[selected, "target_value"] += 1
    with pytest.raises(ValueError, match="changes across optimizer seeds"):
        validate_phenotype_table(frame)
