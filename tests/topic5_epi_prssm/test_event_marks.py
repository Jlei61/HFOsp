"""Event representation contract: ties, masks, phantom ranks, splits, node marks."""
import numpy as np
import pytest

from src.topic5_epi_prssm.event_marks import (
    ADMITTED_CONTACT_FEATURES, REJECTED_CONTACT_FEATURES, SPLIT_TEST, SPLIT_TRAIN,
    SPLIT_VALIDATION, available_subjects, load_patient, recruitment_groups,
    _three_way_split, _validate_event_encoding,
)

SUBJECT = "epilepsiae_1084"


def test_cohort_is_the_frozen_34():
    assert len(available_subjects()) == 34


def test_non_participants_carry_no_rank_and_no_group():
    events = load_patient(SUBJECT)
    assert np.all(events.group_ids[~events.participation] == -1)
    assert np.all(np.isnan(events.normalized_rank[~events.participation]))


def test_phantom_rank_fails_closed():
    events = load_patient(SUBJECT)
    rank = events.normalized_rank.copy()
    rank[~events.participation] = 0.0          # the classic phantom-rank pattern
    with pytest.raises(ValueError, match="phantom rank"):
        _validate_event_encoding(SUBJECT, events.participation, events.group_ids,
                                 events.group_count, rank, events.event_time)


def test_phantom_group_id_fails_closed():
    events = load_patient(SUBJECT)
    gid = events.group_ids.copy()
    gid[~events.participation] = 0
    with pytest.raises(ValueError, match="phantom rank"):
        _validate_event_encoding(SUBJECT, events.participation, gid, events.group_count,
                                 events.normalized_rank, events.event_time)


def test_ties_come_from_explicit_group_identity_not_equal_ranks():
    events = load_patient(SUBJECT)
    tied = np.flatnonzero(events.group_count < events.participation.sum(axis=1))
    if len(tied) == 0:
        pytest.skip("no exact ties in this subject")
    index = int(tied[0])
    groups = recruitment_groups(events, index)
    assert len(groups) == int(events.group_count[index])
    assert sum(len(g) for g in groups) == int(events.participation[index].sum())
    assert max(len(g) for g in groups) > 1


def test_recruitment_groups_are_ordered_and_complete():
    events = load_patient(SUBJECT)
    for index in (0, 1, 100, events.n_events - 1):
        groups = recruitment_groups(events, index)
        members = np.concatenate(groups)
        assert set(members.tolist()) == set(np.flatnonzero(events.participation[index]).tolist())
        ranks = [events.normalized_rank[index, g].mean() for g in groups]
        assert all(a <= b + 1e-6 for a, b in zip(ranks, ranks[1:]))


def test_node_marks_zero_the_rank_channel_where_masked():
    events = load_patient(SUBJECT)
    marks = events.node_marks()
    assert marks.shape == (events.n_events, events.n_contacts, 3)
    assert np.all(marks[..., 1][~events.participation] == 0.0)
    assert np.all(marks[..., 0] == events.participation)
    assert np.isfinite(marks).all()


def test_splits_are_chronological_and_test_is_the_sealed_partition():
    events = load_patient(SUBJECT)
    train = np.flatnonzero(events.split == SPLIT_TRAIN)
    validation = np.flatnonzero(events.split == SPLIT_VALIDATION)
    test = np.flatnonzero(events.split == SPLIT_TEST)
    assert train.max() < validation.min() < validation.max() < test.min()
    assert len(test) / events.n_events == pytest.approx(0.2, abs=0.005)


def test_split_fractions_match_the_frozen_policy():
    legacy = np.array([0] * 800 + [1] * 200, dtype=np.int8)
    split = _three_way_split(legacy)
    assert (split == SPLIT_TRAIN).sum() == 600
    assert (split == SPLIT_VALIDATION).sum() == 200
    assert (split == SPLIT_TEST).sum() == 200


def test_unverifiable_prefix_support_features_are_rejected():
    events = load_patient(SUBJECT)
    assert set(events.contact_feature_names) <= set(ADMITTED_CONTACT_FEATURES)
    assert not set(events.contact_feature_names) & set(REJECTED_CONTACT_FEATURES)


def test_load_is_the_participating_contact_fraction():
    events = load_patient(SUBJECT)
    expected = events.participation.sum(axis=1) / events.n_contacts
    assert np.allclose(events.load, expected)
