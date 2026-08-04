import numpy as np
import pytest

from src.topic5_slow_state_repertoire import (
    family_agreement,
    local_repertoire,
    resolved_families,
)


def _block(order, groups, n_events, participation=None):
    rank = np.tile(np.asarray(order, float), (n_events, 1))
    gids = np.tile(np.asarray(groups, np.int16), (n_events, 1))
    part = (
        np.ones_like(rank, dtype=np.uint8)
        if participation is None
        else np.tile(np.asarray(participation, np.uint8), (n_events, 1))
    )
    return rank, part, gids


def test_ties_come_from_group_ids_not_from_equal_rank_values():
    # contacts 0 and 1 carry different normalised ranks but share a recruitment group,
    # so they are tied and precedence must be exactly one half
    rank, part, gids = _block([0.1, 0.2, 0.9], [0, 0, 1], 20)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["precedence"][out["pair_index"].index((0, 1))] == pytest.approx(0.5)


def test_equal_rank_values_in_different_groups_are_not_treated_as_tied():
    rank, part, gids = _block([0.5, 0.5, 0.9], [0, 1, 2], 20)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["precedence"][out["pair_index"].index((0, 1))] != pytest.approx(0.5)


def test_precedence_is_one_when_the_earlier_group_always_comes_first():
    rank, part, gids = _block([0.1, 0.5, 0.9], [0, 1, 2], 20)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["precedence"][out["pair_index"].index((0, 2))] == pytest.approx(1.0)


def test_a_contact_below_the_participation_floor_is_excluded_not_averaged():
    rank = np.tile(np.array([0.1, 0.5, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2], np.int16), (20, 1))
    part = np.ones((20, 3), np.uint8)
    part[:-2, 2] = 0  # contact 2 participates in only 2 of 20 events
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert np.isnan(out["masked_mean_rank"][2])
    assert out["n_supported_contacts"] == 2


def test_a_pair_below_the_co_participation_floor_is_excluded():
    rank = np.tile(np.array([0.1, 0.5, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2], np.int16), (20, 1))
    part = np.ones((20, 3), np.uint8)
    part[:10, 1] = 0
    part[10:, 2] = 0  # contacts 1 and 2 never co-participate
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert np.isnan(out["precedence"][out["pair_index"].index((1, 2))])


def test_a_contact_that_never_participates_gets_nan_not_a_phantom_number():
    rank = np.tile(np.array([0.1, 5.0, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2], np.int16), (20, 1))
    part = np.ones((20, 3), np.uint8)
    part[:, 1] = 0
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert np.isnan(out["masked_mean_rank"][1])
    assert out["participation_rate"][1] == 0.0


def test_agreement_reports_each_family_separately_and_has_no_combined_key():
    rank, part, gids = _block([0.1, 0.4, 0.7, 0.9], [0, 1, 2, 3], 20)
    left = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    agreement = family_agreement(left, left)
    assert set(agreement) == {"participation", "mean_rank", "precedence"}
    assert agreement["mean_rank"] == pytest.approx(1.0)


def test_reversed_orderings_disagree_maximally_on_rank():
    rank_a, part, gids = _block([0.1, 0.4, 0.7, 0.9], [0, 1, 2, 3], 20)
    rank_b, _, gids_b = _block([0.9, 0.7, 0.4, 0.1], [3, 2, 1, 0], 20)
    left = local_repertoire(rank_a, part, gids, min_participation_count=5, min_pair_count=5)
    right = local_repertoire(rank_b, part, gids_b, min_participation_count=5, min_pair_count=5)
    assert family_agreement(left, right)["mean_rank"] == pytest.approx(-1.0)


def test_a_single_resolved_family_cannot_stand_in_for_the_repertoire():
    assert resolved_families({"participation": 0.9, "mean_rank": None, "precedence": None}) == 1
    assert resolved_families({"participation": 0.9, "mean_rank": 0.8, "precedence": None}) == 2


def test_status_is_resolved_when_contacts_and_pairs_clear_the_floors():
    rank, part, gids = _block([0.1, 0.5, 0.9], [0, 1, 2], 20)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["status"] == "RESOLVED"
    assert out["n_supported_contacts"] >= 3
    assert out["n_supported_pairs"] >= 3


def test_status_reports_too_few_contacts_when_the_contact_floor_bites():
    rank = np.tile(np.array([0.1, 0.5]), (20, 1))
    gids = np.tile(np.array([0, 1], np.int16), (20, 1))
    part = np.ones((20, 2), np.uint8)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)
    assert out["status"] == "UNRESOLVED_TOO_FEW_CONTACTS"
    assert out["n_supported_contacts"] < 3


def test_status_reports_too_few_pairs_when_the_pair_floor_bites():
    # 4 contacts: 0, 1 always participate; 2 in events 0-9; 3 in events 10-19
    # All meet contact floor (5)
    # Only pair (0,1) meets pair floor (11+): has 20 co-participations
    # Others have 10 co-participations, below the floor of 11
    rank = np.tile(np.array([0.1, 0.5, 0.9, 0.7]), (20, 1))
    gids = np.tile(np.array([0, 1, 2, 3], np.int16), (20, 1))
    part = np.ones((20, 4), np.uint8)
    part[10:20, 2] = 0
    part[0:10, 3] = 0
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=11)
    assert out["status"] == "UNRESOLVED_TOO_FEW_PAIRS"
    assert out["n_supported_contacts"] >= 3
    assert out["n_supported_pairs"] < 3


def test_a_pair_with_no_co_participation_is_nan_even_with_a_zero_floor():
    rank = np.tile(np.array([0.1, 0.5, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2], np.int16), (20, 1))
    part = np.ones((20, 3), np.uint8)
    part[0:10, 1] = 0
    part[10:20, 2] = 0
    # Pair (1, 2): never co-participate (support = 0)
    out = local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=0)
    assert np.isnan(out["precedence"][out["pair_index"].index((1, 2))])
