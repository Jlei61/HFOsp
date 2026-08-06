import numpy as np
import pytest

from src.topic5_slow_state_repertoire import (
    estimate_backbone,
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


# ---------------------------------------------------------------------------
# estimate_backbone — rev3 R3-C. The patient's global per-contact / per-pair main
# effects, fitted on TRAIN windows only, so the scale curve can be recomputed on the
# deviation from the backbone instead of on a descriptor the backbone dominates.
# ---------------------------------------------------------------------------

_MU = np.array([0.10, 0.26, 0.42, 0.58, 0.74, 0.90])


def _noisy_window(mu, *, seed, n_events=40, noise=0.01):
    """One window whose per-event rank is `mu` plus i.i.d. noise, all contacts present.

    Recruitment groups are the fixed contact order, so this window's mean rank is `mu`
    to within `noise / sqrt(n_events)` and nothing else about it varies.
    """
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    n_contacts = mu.size
    rank = mu[None, :] + rng.normal(0.0, noise, size=(n_events, n_contacts))
    gids = np.tile(np.arange(n_contacts, dtype=np.int16), (n_events, 1))
    part = np.ones((n_events, n_contacts), np.uint8)
    return local_repertoire(rank, part, gids, min_participation_count=5, min_pair_count=5)


def test_the_backbone_is_estimated_on_train_windows_only():
    # rev3 R3-C, third test. 40 windows, chronological 32 train / 8 held out. The
    # held-out windows carry the REVERSED rank backbone, so including them would move
    # every contact's main effect by a large, easily visible amount -- that is what the
    # second assertion pins, and without it this fixture would be vacuous (a held-out
    # set drawn from the same distribution as train could not move the estimate whether
    # or not it was used, and the test would pass against a fit-on-everything
    # implementation).
    train = [_noisy_window(_MU, seed=s) for s in range(32)]
    held_out_v1 = [_noisy_window(_MU[::-1], seed=100 + s) for s in range(8)]
    held_out_v2 = [_noisy_window(_MU[::-1] * 0.5 + 0.05, seed=200 + s) for s in range(8)]

    before = estimate_backbone(train)
    # change ONLY the held-out windows
    after = estimate_backbone(train)

    for family in ("participation_rate", "masked_mean_rank", "precedence"):
        np.testing.assert_array_equal(before[family], after[family])
    assert before["pair_index"] == after["pair_index"]
    assert before["n_train_windows"] == after["n_train_windows"] == 32

    # non-vacuity, asserted as a MAGNITUDE and not as `not allclose`: sampling noise
    # alone (~0.01/sqrt(40*32) per contact) already defeats `allclose`, so `not
    # allclose` would hold even for a held-out set drawn from the train distribution and
    # would prove nothing. The 0.05 floor cannot be reached by that noise. Arithmetic:
    # train-only gives contact 0 = 0.10; a fit on all 40 gives 0.8*0.10 + 0.2*0.90 =
    # 0.26 under held_out_v1 (diff 0.16) and 0.8*0.10 + 0.2*0.50 = 0.18 under
    # held_out_v2 (v1 - v2 = 0.08).
    fit_on_everything_v1 = estimate_backbone(train + held_out_v1)
    fit_on_everything_v2 = estimate_backbone(train + held_out_v2)
    assert (
        np.max(np.abs(before["masked_mean_rank"] - fit_on_everything_v1["masked_mean_rank"]))
        > 0.05
    )
    assert (
        np.max(
            np.abs(
                fit_on_everything_v1["masked_mean_rank"]
                - fit_on_everything_v2["masked_mean_rank"]
            )
        )
        > 0.05
    )


def test_a_descriptor_no_train_window_could_estimate_stays_nan():
    # contact 1 is under the participation floor in window A and above it in window B:
    # the backbone must be window B's value alone, not nan (a plain mean would poison
    # the column) and not half of it (a nan -> 0 substitution would halve it).
    # contact 3 never participates in either window: no train window estimated it, so
    # its main effect stays nan and `_residualise_descriptors` will remove nothing.
    rank = np.tile(np.array([0.1, 0.4, 0.7, 0.9]), (20, 1))
    gids = np.tile(np.array([0, 1, 2, 3], np.int16), (20, 1))

    part_a = np.ones((20, 4), np.uint8)
    part_a[2:, 1] = 0  # 2 of 20 events -- under the floor of 5
    part_a[:, 3] = 0
    part_b = np.ones((20, 4), np.uint8)
    part_b[:, 3] = 0

    window_a = local_repertoire(
        rank, part_a, gids, min_participation_count=5, min_pair_count=5
    )
    window_b = local_repertoire(
        rank, part_b, gids, min_participation_count=5, min_pair_count=5
    )
    backbone = estimate_backbone([window_a, window_b])

    assert np.isnan(window_a["masked_mean_rank"][1])
    assert backbone["masked_mean_rank"][1] == pytest.approx(
        window_b["masked_mean_rank"][1]
    )
    assert np.isnan(backbone["masked_mean_rank"][3])
    assert backbone["masked_mean_rank"][0] == pytest.approx(0.1)
    # participation rate is finite in both windows, so contact 1's main effect is the
    # ordinary two-window mean of 0.1 and 1.0
    assert backbone["participation_rate"][1] == pytest.approx(0.55)


def test_estimate_backbone_refuses_windows_whose_layout_does_not_line_up():
    four = _noisy_window(_MU[:4], seed=0)
    six = _noisy_window(_MU, seed=1)
    with pytest.raises(ValueError, match="contact count"):
        estimate_backbone([four, six])


def test_estimate_backbone_refuses_an_empty_train_set():
    with pytest.raises(ValueError, match="at least one train window"):
        estimate_backbone([])
