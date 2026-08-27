from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.topic4_rev14_patient_support import (
    PRIMARY_LABEL_KEY,
    PatientSupportSchemaError,
    build_patient_support_calibration,
    evaluate_patient_support,
    model_contact_primary_from_mapping,
    patient_training_from_mapping,
    sample_cross_block_pairs,
    score_patient_support,
)


def _patient_mapping(*, diagnostic_flip: bool = False) -> dict:
    names = np.asarray(["ICL1", "ICL2", "SCL1", "SCL2"])
    shafts = np.asarray(["ICL", "ICL", "SCL", "SCL"])
    onsets, labels, blocks, ood = [], [], [], []
    for block in range(24):
        jitter = 0.02 * (block % 4)
        mode_a = np.asarray([0.0, 1.0 + jitter, 2.0 - jitter, 3.0])
        mode_b = np.asarray([2.0 + jitter, 3.0, 0.0, 1.0 - jitter])
        if block % 5 == 0:
            mode_a[3] = np.nan
        if block % 7 == 0:
            mode_b[1] = np.nan
        if block % 6 == 0:
            mode_a[2:] -= 2.5
        if block % 8 == 0:
            mode_b[:2] -= 2.5
        onsets.extend((mode_a, mode_b))
        labels.extend((0, 1))
        blocks.extend((block, block))
        ood.extend((block % 11 == 0, block % 13 == 0))
    old = np.asarray(labels, dtype=np.int8)
    diagnostic = 1 - old if diagnostic_flip else old.copy()
    return {
        "contact_names": names,
        "shaft_ids": shafts,
        "patient_train_onsets": np.asarray(onsets),
        "patient_train_old_labels": old,
        "patient_train_classifier_labels": old.copy(),
        "patient_train_shaft_aware_k2_labels": diagnostic,
        "patient_train_block_ids": np.asarray(blocks),
        "patient_train_ood": np.asarray(ood),
        "primary_label_key": PRIMARY_LABEL_KEY,
        "source_sha256": "a" * 64,
    }


def _model_mapping(patient, indices=None) -> dict:
    use = np.arange(len(patient.onsets)) if indices is None else np.asarray(indices)
    return {
        "contact_names": np.asarray(patient.contact_names),
        "onsets": patient.onsets[use].copy(),
        "classifier_labels": patient.old_labels[use].copy(),
        "ood": patient.ood[use].copy(),
    }


@pytest.fixture(scope="module")
def calibrated():
    patient = patient_training_from_mapping(_patient_mapping())
    calibration = build_patient_support_calibration(
        patient, floor_draws=256, joint_draws=128,
        joint_inner_draws=64, seed=17,
    )
    return patient, calibration


def test_floor_and_score_are_deterministic_and_use_old_labels(calibrated):
    patient, calibration = calibrated
    repeated = build_patient_support_calibration(
        patient, floor_draws=256, joint_draws=128,
        joint_inner_draws=64, seed=17,
    )
    assert calibration.primary_label_key == PRIMARY_LABEL_KEY
    assert set(calibration.floor_q95) == {
        "mode_0.recruitment.ICL", "mode_0.recruitment.SCL",
        "mode_0.precedence.ICL-SCL", "mode_0.joint_shaft_fraction",
        "mode_1.recruitment.ICL", "mode_1.recruitment.SCL",
        "mode_1.precedence.ICL-SCL", "mode_1.joint_shaft_fraction", "OOD",
    }
    for key in calibration.floor_q95:
        np.testing.assert_array_equal(
            calibration.floor_distributions[key], repeated.floor_distributions[key]
        )

    model = model_contact_primary_from_mapping(_model_mapping(patient))
    left = score_patient_support(
        patient, calibration, model, score_draws=64, seed=29,
    )
    right = score_patient_support(
        patient, calibration, model, score_draws=64, seed=29,
    )
    assert left == right
    assert left.status == (
        "PASS" if left.score <= calibration.joint_threshold_q95 else "FAIL"
    )
    assert left.support["n_contact_primary"] == len(patient.onsets)

    flipped = patient_training_from_mapping(_patient_mapping(diagnostic_flip=True))
    flipped_calibration = build_patient_support_calibration(
        flipped, floor_draws=256, joint_draws=128,
        joint_inner_draws=64, seed=17,
    )
    assert flipped.data_sha256 == patient.data_sha256
    assert flipped_calibration.floor_q95 == calibration.floor_q95
    np.testing.assert_array_equal(
        flipped_calibration.joint_distribution, calibration.joint_distribution,
    )


def test_floor_draws_are_block_disjoint_and_one_event_per_block():
    patient = patient_training_from_mapping(_patient_mapping())
    left, right, left_blocks, right_blocks = sample_cross_block_pairs(
        patient.old_labels, patient.block_ids, mode=0,
        sample_size=6, draws=64, rng=np.random.default_rng(8),
    )
    for draw in range(len(left)):
        assert len(np.unique(left_blocks[draw])) == 6
        assert len(np.unique(right_blocks[draw])) == 6
        assert set(left_blocks[draw]).isdisjoint(right_blocks[draw])
        np.testing.assert_array_equal(patient.block_ids[left[draw]], left_blocks[draw])
        np.testing.assert_array_equal(patient.block_ids[right[draw]], right_blocks[draw])


def test_joint_null_is_synchronized_block_disjoint_and_has_q95_coverage(calibrated):
    patient, calibration = calibrated
    endpoint_matrix = np.stack([
        calibration.joint_endpoint_distributions[key]
        for key in (
            "mode_0.recruitment.ICL", "mode_0.recruitment.SCL",
            "mode_0.precedence.ICL-SCL", "mode_0.joint_shaft_fraction",
            "mode_1.recruitment.ICL", "mode_1.recruitment.SCL",
            "mode_1.precedence.ICL-SCL", "mode_1.joint_shaft_fraction", "OOD",
        )
    ], axis=1)
    np.testing.assert_array_equal(
        endpoint_matrix.max(axis=1), calibration.joint_distribution,
    )
    coverage = np.mean(
        calibration.joint_distribution <= calibration.joint_threshold_q95
    )
    assert 0.93 <= coverage <= 0.97
    assert len(calibration.joint_distribution_sha256) == 64
    assert len(calibration.joint_block_audit_sha256) == 64

    for pseudo, reference in zip(
            calibration.joint_pseudo_blocks,
            calibration.joint_reference_blocks, strict=True):
        assert set(pseudo).isdisjoint(reference)
        for side in (pseudo, reference):
            mask = np.isin(patient.block_ids, side)
            for mode in (0, 1):
                assert len(np.unique(patient.block_ids[
                    mask & (patient.old_labels == mode)
                ])) >= calibration.sample_size


def test_score_draw_count_is_frozen_by_joint_calibration(calibrated):
    patient, calibration = calibrated
    model = model_contact_primary_from_mapping(_model_mapping(patient))
    invalid = evaluate_patient_support(
        patient, calibration, model, score_draws=63, seed=23,
    )
    assert invalid.status == "INVALID"
    assert "joint calibration inner draw count" in invalid.invalid_reason


def test_scl_censoring_is_visible_and_cannot_pass_silently(calibrated):
    patient, calibration = calibrated
    baseline = model_contact_primary_from_mapping(_model_mapping(patient))
    censored_mapping = _model_mapping(patient)
    censored_mapping["onsets"][:, 2:] = np.nan
    censored = model_contact_primary_from_mapping(censored_mapping)
    reference = score_patient_support(
        patient, calibration, baseline, score_draws=64, seed=31,
    )
    result = score_patient_support(
        patient, calibration, censored, score_draws=64, seed=31,
    )
    assert result.status == "FAIL"
    assert result.score > calibration.joint_threshold_q95
    assert (result.endpoints["mode_0.recruitment.SCL"]["ratio"]
            > reference.endpoints["mode_0.recruitment.SCL"]["ratio"])
    assert (result.endpoints["mode_1.recruitment.SCL"]["ratio"]
            > reference.endpoints["mode_1.recruitment.SCL"]["ratio"])


def test_cross_shaft_timing_swap_changes_precedence_not_recruitment(calibrated):
    patient, calibration = calibrated
    original_mapping = _model_mapping(patient)
    swapped_mapping = _model_mapping(patient)
    values = swapped_mapping["onsets"]
    for row in values:
        finite_i = np.isfinite(row[:2])
        finite_s = np.isfinite(row[2:])
        if finite_i.any() and finite_s.any():
            row[2:][finite_s] -= 6.0
    original = score_patient_support(
        patient, calibration,
        model_contact_primary_from_mapping(original_mapping),
        score_draws=64, seed=37,
    )
    swapped = score_patient_support(
        patient, calibration,
        model_contact_primary_from_mapping(swapped_mapping),
        score_draws=64, seed=37,
    )
    for mode in (0, 1):
        for shaft in ("ICL", "SCL"):
            key = f"mode_{mode}.recruitment.{shaft}"
            assert swapped.endpoints[key]["raw_distance"] == pytest.approx(
                original.endpoints[key]["raw_distance"], abs=1e-15,
            )
    assert any(
        swapped.endpoints[f"mode_{mode}.precedence.ICL-SCL"]["ratio"]
        > original.endpoints[f"mode_{mode}.precedence.ICL-SCL"]["ratio"]
        for mode in (0, 1)
    )


def test_ood_contact_row_remains_in_mode_metrics(calibrated):
    patient, calibration = calibrated
    selected = np.concatenate([
        np.flatnonzero(patient.old_labels == 0)[:6],
        np.flatnonzero(patient.old_labels == 1)[:6],
    ])
    base = _model_mapping(patient, selected)
    added = {
        "contact_names": base["contact_names"],
        "onsets": np.vstack([base["onsets"], [0.0, 0.2, np.nan, np.nan]]),
        "classifier_labels": np.append(base["classifier_labels"], 0),
        "ood": np.append(base["ood"], True),
    }
    unflagged = {**added, "ood": added["ood"].copy()}
    unflagged["ood"][-1] = False

    baseline = score_patient_support(
        patient, calibration, model_contact_primary_from_mapping(base), seed=39,
    )
    flagged = score_patient_support(
        patient, calibration, model_contact_primary_from_mapping(added), seed=39,
    )
    same_row_unflagged = score_patient_support(
        patient, calibration, model_contact_primary_from_mapping(unflagged), seed=39,
    )
    mode_keys = [key for key in flagged.endpoints if key.startswith("mode_")]
    assert any(
        flagged.endpoints[key]["raw_distance"]
        != baseline.endpoints[key]["raw_distance"]
        for key in mode_keys
    )
    for key in mode_keys:
        assert flagged.endpoints[key] == same_row_unflagged.endpoints[key]
    assert flagged.endpoints["OOD"] != same_row_unflagged.endpoints["OOD"]


def test_single_mode_is_padded_not_copied_and_fails(calibrated):
    patient, calibration = calibrated
    only_a = np.flatnonzero(patient.old_labels == 0)
    model = model_contact_primary_from_mapping(_model_mapping(patient, only_a))
    result = score_patient_support(
        patient, calibration, model, score_draws=64, seed=41,
    )
    assert result.status == "FAIL"
    assert result.score > calibration.joint_threshold_q95
    assert result.support["n_in_support_classifier_B"] == 0
    assert result.support["padding_rows_across_draws"]["mode_1"] == 64 * 6
    assert result.support["n_contact_primary"] == len(only_a)


def test_all_missing_events_remain_in_ood_and_fail(calibrated):
    patient, calibration = calibrated
    model = model_contact_primary_from_mapping({
        "contact_names": np.asarray(patient.contact_names),
        "onsets": np.full((9, len(patient.contact_names)), np.nan),
        "classifier_labels": np.asarray([0, 1] * 4 + [0]),
        "ood": np.zeros(9, dtype=bool),
    })
    result = score_patient_support(
        patient, calibration, model, score_draws=64, seed=43,
    )
    assert result.status == "FAIL"
    assert result.support["n_contact_primary"] == 9
    assert result.support["n_all_missing"] == 9
    assert result.support["n_ood_including_all_missing"] == 9
    assert result.support["all_contact_primary_rows_retained"] is True
    assert result.endpoints["OOD"]["raw_distance"] > 0.5


def test_contact_permutation_is_invariant(calibrated):
    patient, calibration = calibrated
    mapping = _model_mapping(patient)
    baseline = score_patient_support(
        patient, calibration, model_contact_primary_from_mapping(mapping),
        score_draws=64, seed=47,
    )
    permutation = np.asarray([2, 0, 3, 1])
    permuted = {
        **mapping,
        "contact_names": mapping["contact_names"][permutation],
        "onsets": mapping["onsets"][:, permutation],
    }
    observed = score_patient_support(
        patient, calibration, model_contact_primary_from_mapping(permuted),
        score_draws=64, seed=47,
    )
    assert observed.status == baseline.status
    assert observed.score == baseline.score
    assert observed.endpoints == baseline.endpoints

    patient_mapping = _patient_mapping()
    patient_mapping["contact_names"] = patient_mapping["contact_names"][permutation]
    patient_mapping["shaft_ids"] = patient_mapping["shaft_ids"][permutation]
    patient_mapping["patient_train_onsets"] = (
        patient_mapping["patient_train_onsets"][:, permutation]
    )
    permuted_patient = patient_training_from_mapping(patient_mapping)
    permuted_calibration = build_patient_support_calibration(
        permuted_patient, floor_draws=256, joint_draws=128,
        joint_inner_draws=64, seed=17,
    )
    assert permuted_patient.data_sha256 == patient.data_sha256
    assert permuted_calibration.floor_q95 == calibration.floor_q95


def test_heldout_poison_is_invalid_and_never_loaded(calibrated):
    poisoned = _patient_mapping()
    poisoned["patient_heldout_onsets"] = poisoned["patient_train_onsets"][:1]
    with pytest.raises(PatientSupportSchemaError, match="held-out"):
        patient_training_from_mapping(poisoned)

    patient, calibration = calibrated
    poisoned_model = _model_mapping(patient)
    poisoned_model["metadata"] = {"heldout_hint": np.asarray([1])}
    with pytest.raises(PatientSupportSchemaError, match="held-out"):
        model_contact_primary_from_mapping(poisoned_model)

    legacy_named = _model_mapping(patient)
    legacy_named["old_labels"] = legacy_named.pop("classifier_labels")
    with pytest.raises(PatientSupportSchemaError, match="classifier_labels"):
        model_contact_primary_from_mapping(legacy_named)


def test_hash_mismatch_is_invalid_but_low_support_is_fail(calibrated):
    patient, calibration = calibrated
    model = model_contact_primary_from_mapping(_model_mapping(patient))
    bad_calibration = replace(calibration, patient_data_sha256="0" * 64)
    invalid = evaluate_patient_support(
        patient, bad_calibration, model, score_draws=64, seed=53,
    )
    assert invalid.status == "INVALID"
    assert invalid.score is None
    assert "SHA-256 changed" in invalid.invalid_reason

    changed_floors = dict(calibration.floor_q95)
    changed_floors["OOD"] += 0.01
    changed_calibration = replace(calibration, floor_q95=changed_floors)
    invalid_floor = evaluate_patient_support(
        patient, changed_calibration, model, score_draws=64, seed=53,
    )
    assert invalid_floor.status == "INVALID"
    assert "q95 changed" in invalid_floor.invalid_reason

    changed_joint = calibration.joint_distribution.copy()
    changed_joint[0] += 0.01
    invalid_joint = evaluate_patient_support(
        patient, replace(calibration, joint_distribution=changed_joint), model,
        score_draws=64, seed=53,
    )
    assert invalid_joint.status == "INVALID"
    assert "joint null" in invalid_joint.invalid_reason

    all_missing = model_contact_primary_from_mapping({
        "contact_names": np.asarray(patient.contact_names),
        "onsets": np.full((1, len(patient.contact_names)), np.nan),
        "classifier_labels": np.asarray([0]),
        "ood": np.asarray([False]),
    })
    ordinary = evaluate_patient_support(
        patient, calibration, all_missing, score_draws=64, seed=53,
    )
    assert ordinary.status == "FAIL"
    assert ordinary.invalid_reason is None
