from dataclasses import replace

import numpy as np
import pytest

from src.topic5_event_innovation_data import (
    AnchorSplits,
    ContinuitySequence,
    SingleEventAnchors,
    SourceSegment,
    assign_continuity_units,
    audit_crossfit_folds,
    audit_cumulative_anchors,
    audit_phase0_contract,
    audit_single_event_anchors,
    build_blocked_chronological_crossfit_folds,
    build_continuity_sequences,
    build_cumulative_anchor_splits,
    build_single_event_anchor_splits,
    resolve_crossfit_fold,
    resolve_cumulative_anchor,
    resolve_single_event_anchor,
)


def _segment(
    source_id,
    start,
    stop,
    *,
    group="session_a",
    montage="montage_a",
    verified=True,
):
    return SourceSegment(
        source_id=source_id,
        start_time=float(start),
        stop_time=float(stop),
        continuity_group=group,
        montage_hash=montage,
        continuity_verified=verified,
    )


def _sequence(unit, start, length):
    indices = np.arange(start, start + length, dtype=np.int64)
    return ContinuitySequence(
        continuity_unit_id=unit,
        event_indices=indices,
        event_times=indices.astype(float),
        source_ids=np.repeat(f"source_{unit}", length),
    )


def _split_sequences(length=30):
    return {
        "train": (_sequence("train", 0, length),),
        "validation": (_sequence("validation", 100, length),),
        "test": (_sequence("test", 200, length),),
    }


def test_continuity_requires_verified_metadata_montage_and_gap():
    decisions = assign_continuity_units(
        [
            _segment("s0", 0, 10),
            _segment("s1", 10.5, 20),
            _segment("s2", 20.5, 30, montage="montage_b"),
            _segment("s3", 30.5, 40, group="session_b", montage="montage_b"),
            _segment(
                "s4", 40.5, 50, group="session_b", montage="montage_b", verified=False
            ),
        ],
        maximum_gap_seconds=1.0,
    )
    assert [item.decision for item in decisions] == [
        "reset",
        "join_previous",
        "reset",
        "reset",
        "reset",
    ]
    assert decisions[1].continuity_unit_id == decisions[0].continuity_unit_id
    assert decisions[2].reason == "montage_mismatch"
    assert decisions[3].reason == "independent_continuity_group"
    assert decisions[4].reason == "continuity_relationship_unverified"


def test_continuity_does_not_guess_from_event_density_or_missing_metadata():
    decisions = assign_continuity_units(
        [
            _segment("s0", 0, 10, group=None),
            _segment("s1", 10, 20, group=None),
            _segment("s2", 20, 30, group="x", montage=None),
            _segment("s3", 30, 40, group="x", montage=None),
        ],
        maximum_gap_seconds=100.0,
    )
    assert all(item.decision == "reset" for item in decisions)
    assert decisions[1].reason == "continuity_group_missing"
    assert decisions[3].reason == "montage_hash_missing"


def test_build_continuity_sequences_is_index_only_and_fail_closed():
    decisions = assign_continuity_units(
        [_segment("s0", 0, 10), _segment("s1", 10, 20)],
        maximum_gap_seconds=0.0,
    )
    sequences = build_continuity_sequences(
        np.arange(8, dtype=float),
        np.asarray(["s0"] * 4 + ["s1"] * 4),
        decisions,
        eligible_indices=np.arange(8),
    )
    assert len(sequences) == 1
    assert np.array_equal(sequences[0].event_indices, np.arange(8))
    assert not hasattr(sequences[0], "features")

    with pytest.raises(ValueError, match="preserve canonical event order"):
        build_continuity_sequences(
            np.arange(8, dtype=float),
            np.asarray(["s0"] * 4 + ["s1"] * 4),
            decisions,
            eligible_indices=[0, 2, 1, 3],
        )
    with pytest.raises(ValueError, match="lack continuity decisions"):
        build_continuity_sequences(
            np.arange(2, dtype=float),
            np.asarray(["s0", "unknown"]),
            decisions,
        )


def test_single_event_anchors_are_dense_then_formal_and_small_memory():
    split_sequences = _split_sequences()
    anchors = build_single_event_anchor_splits(
        split_sequences, pre_events=4, horizon=5
    )
    assert len(anchors.train) == 21
    assert len(anchors.validation) == 5
    assert len(anchors.test) == 5
    assert anchors.train.sequence_index.dtype == np.int32
    assert not hasattr(anchors.train, "pre_indices")
    pre, innovation, post = resolve_single_event_anchor(
        anchors.train, 0, split_sequences["train"]
    )
    assert np.array_equal(pre, np.arange(4))
    assert innovation == 4
    assert np.array_equal(post, np.arange(5, 10))
    assert audit_single_event_anchors(
        anchors.train,
        split_sequences["train"],
        require_nonoverlap_post=False,
    )["internally_disjoint"]
    formal = audit_single_event_anchors(
        anchors.validation,
        split_sequences["validation"],
        require_nonoverlap_post=True,
    )
    assert formal["strict_event_order"]
    assert formal["formal_post_windows_nonoverlap"]


def test_cumulative_pre_exposure_and_post_are_disjoint():
    split_sequences = _split_sequences()
    anchors = build_cumulative_anchor_splits(
        split_sequences,
        pre_events=4,
        exposure_events=3,
        horizon=5,
    )
    assert len(anchors.train) == 19
    assert len(anchors.validation) == 4
    pre, exposure, post = resolve_cumulative_anchor(
        anchors.train, 0, split_sequences["train"]
    )
    assert np.array_equal(pre, np.arange(4))
    assert np.array_equal(exposure, np.arange(4, 7))
    assert np.array_equal(post, np.arange(7, 12))
    assert audit_cumulative_anchors(
        anchors.train,
        split_sequences["train"],
        require_nonoverlap_post=False,
    )["internally_disjoint"]
    formal = audit_cumulative_anchors(
        anchors.validation,
        split_sequences["validation"],
        require_nonoverlap_post=True,
    )
    assert formal["strict_event_order"]
    assert formal["formal_post_windows_nonoverlap"]


def test_crossfit_is_forward_blocked_and_has_exact_embargo():
    sequences = (_sequence("u0", 0, 30),)
    folds = build_blocked_chronological_crossfit_folds(
        sequences,
        n_splits=2,
        embargo_events=2,
        minimum_train_events=5,
        minimum_validation_events=5,
    )
    assert len(folds) == 2
    first = resolve_crossfit_fold(folds[0], sequences)
    second = resolve_crossfit_fold(folds[1], sequences)
    assert np.array_equal(first[0], np.arange(8))
    assert np.array_equal(first[1], np.arange(8, 10))
    assert np.array_equal(first[2], np.arange(10, 20))
    assert np.array_equal(second[0], np.arange(18))
    assert np.array_equal(second[1], np.arange(18, 20))
    assert np.array_equal(second[2], np.arange(20, 30))
    audit = audit_crossfit_folds(folds, sequences)
    assert all(value for key, value in audit.items() if key != "n_folds")


def test_phase0_audit_passes_only_event_level_index_contract():
    split_sequences = {
        "train": (_sequence("train", 0, 30),),
        "validation": (_sequence("validation", 30, 30),),
        "test": (_sequence("test", 60, 30),),
    }
    single = build_single_event_anchor_splits(
        split_sequences, pre_events=4, horizon=5
    )
    cumulative = build_cumulative_anchor_splits(
        split_sequences, pre_events=4, exposure_events=3, horizon=5
    )
    folds = build_blocked_chronological_crossfit_folds(
        split_sequences["train"],
        n_splits=2,
        embargo_events=4,
        minimum_train_events=5,
    )
    audit = audit_phase0_contract(split_sequences, single, cumulative, folds)
    assert audit["status"] == "PASS"
    assert audit["one_step_is_one_complete_event"] is True
    assert audit["feature_windows_materialized"] is False
    assert audit["anchor_storage"] == "columnar_scalar_position_bounds"
    assert audit["split_event_indices_disjoint"] is True
    assert audit["anchors_belong_to_declared_split_and_unit"] is True
    assert audit["crossfit_indices_belong_to_train_unit"] is True


def test_formal_anchor_audit_detects_overlapping_target_windows():
    sequences = (_sequence("validation", 0, 30),)
    anchors = build_single_event_anchor_splits(
        {"train": sequences, "validation": sequences, "test": sequences},
        pre_events=4,
        horizon=5,
    ).validation
    broken = replace(
        anchors,
        sequence_index=np.insert(anchors.sequence_index, 1, anchors.sequence_index[0]),
        pre_start=np.insert(anchors.pre_start, 1, anchors.pre_start[0]),
        pre_stop=np.insert(anchors.pre_stop, 1, anchors.pre_stop[0]),
        innovation_position=np.insert(
            anchors.innovation_position, 1, anchors.innovation_position[0]
        ),
        post_start=np.insert(anchors.post_start, 1, anchors.post_start[0]),
        post_stop=np.insert(anchors.post_stop, 1, anchors.post_stop[0]),
    )
    audit = audit_single_event_anchors(
        broken, sequences, require_nonoverlap_post=True
    )
    assert audit["internally_disjoint"] is True
    assert audit["formal_post_windows_nonoverlap"] is False


def test_phase0_audit_fails_closed_on_false_anchor_provenance():
    split_sequences = {
        "train": (_sequence("train", 0, 30),),
        "validation": (_sequence("validation", 30, 30),),
        "test": (_sequence("test", 60, 30),),
    }
    single = build_single_event_anchor_splits(
        split_sequences, pre_events=4, horizon=5
    )
    cumulative = build_cumulative_anchor_splits(
        split_sequences, pre_events=4, exposure_events=3, horizon=5
    )
    folds = build_blocked_chronological_crossfit_folds(
        split_sequences["train"],
        n_splits=2,
        embargo_events=4,
        minimum_train_events=5,
    )
    forged_validation = replace(
        single.validation,
        sequence_index=np.full_like(single.validation.sequence_index, 9),
    )
    broken_single = AnchorSplits(
        train=single.train,
        validation=forged_validation,
        test=single.test,
    )
    audit = audit_phase0_contract(
        split_sequences, broken_single, cumulative, folds
    )
    assert audit["status"] == "FAIL"
    assert audit["anchors_belong_to_declared_split_and_unit"] is False


def test_dense_anchor_storage_scales_with_rows_not_history_length():
    sequence = (_sequence("large", 0, 100_000),)
    short = build_single_event_anchor_splits(
        {"train": sequence, "validation": sequence, "test": sequence},
        pre_events=5,
        horizon=20,
    ).train
    long = build_single_event_anchor_splits(
        {"train": sequence, "validation": sequence, "test": sequence},
        pre_events=80,
        horizon=20,
    ).train
    short_bytes = sum(value.nbytes for value in short.__dict__.values() if hasattr(value, "nbytes"))
    long_bytes = sum(value.nbytes for value in long.__dict__.values() if hasattr(value, "nbytes"))
    assert long_bytes < short_bytes
    assert long_bytes < 3_000_000
    assert all(
        value.dtype == np.int32
        for value in long.__dict__.values()
        if hasattr(value, "dtype")
    )


def test_empty_formal_anchor_split_fails_closed():
    sequences = (_sequence("small", 0, 8),)
    anchors = build_single_event_anchor_splits(
        {"train": sequences, "validation": sequences, "test": sequences},
        pre_events=4,
        horizon=5,
    ).validation
    assert len(anchors) == 0
    audit = audit_single_event_anchors(
        anchors, sequences, require_nonoverlap_post=True
    )
    assert audit["nonempty"] is False
