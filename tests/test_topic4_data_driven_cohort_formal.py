from __future__ import annotations

import numpy as np
import pytest

from src.topic4_data_driven_cohort import TargetConfig, build_crossfit_patient_target
from src.topic4_data_driven_cohort_formal import (
    _shaft_balanced_profile_error,
    score_model_ranks_shaft_balanced,
    within_shaft_null_contract,
    within_shaft_permutations,
)


def _patient_target(names=("A1", "A2", "A3", "A4", "B1", "B2")):
    names = [str(name) for name in names]
    rng = np.random.default_rng(4)
    n_blocks, per_block = 10, 16
    ranks = np.empty((len(names), n_blocks * per_block), float)
    bools = np.ones_like(ranks, bool)
    for event in range(ranks.shape[1]):
        order = np.arange(len(names))
        if event % 2:
            order = order[::-1]
        noisy = order + rng.normal(0.0, 0.01, len(names))
        ranks[:, event] = np.argsort(np.argsort(noisy, kind="stable"), kind="stable")
    data = {
        "channel_names": names,
        "ranks": ranks,
        "bools": bools,
        "block_ids": np.repeat(np.arange(n_blocks), per_block),
    }
    pair = {
        "contact_order": names,
        "rank_a": np.arange(len(names), dtype=float),
        "rank_b": np.arange(len(names), dtype=float)[::-1],
        "cluster_id_a": 0,
        "cluster_id_b": 1,
    }
    target = build_crossfit_patient_target(
        data, pair,
        config=TargetConfig(
            minimum_participating_contacts=3,
            heldout_block_fraction=0.3,
            split_seed=8,
            kmeans_fit_max_events=1000,
            kmeans_n_init=10,
            kmeans_seed=9,
            stability_seeds=(9, 10, 11),
            stored_events_per_mode_per_split=100,
        ),
    )
    return names, target


def _descriptor_target(target, split, names):
    descriptors = target[f"{split}_descriptors"]
    return {
        "contact_order": list(names),
        "profiles": target[f"{split}_profiles"],
        "recruitment": np.asarray([
            descriptors["TA"]["recruitment"], descriptors["TB"]["recruitment"],
        ]),
        "precedence": np.asarray([
            descriptors["TA"]["precedence"], descriptors["TB"]["precedence"],
        ]),
    }


def _heldout_ranks(target, per_mode=20):
    return np.vstack([
        target["heldout_samples"]["TA"][:per_mode],
        target["heldout_samples"]["TB"][:per_mode],
    ])


# --- scoring behaviour -----------------------------------------------------

def test_formal_scorer_recovers_two_modes_without_shaft_count_weighting():
    names, target = _patient_target()
    score = score_model_ranks_shaft_balanced(
        _heldout_ranks(target),
        patient_centers=target["kmeans_centers"],
        target=_descriptor_target(target, "heldout", names),
        contact_names=names,
        patient_ood_threshold=target["train_distance_q95"],
    )
    assert score["status"] == "EVALUABLE"
    assert score["weakest_mode_loss"] < 0.05
    assert score["natural_kmeans"]["weakest_mode_loss"] < 0.05
    assert score["natural_kmeans"]["seed_ami_median"] == 1.0


def test_formal_scorer_penalizes_missing_mode_support():
    names, target = _patient_target()
    ranks = np.repeat(target["heldout_samples"]["TA"][:1], 20, axis=0)
    score = score_model_ranks_shaft_balanced(
        ranks,
        patient_centers=target["kmeans_centers"],
        target=_descriptor_target(target, "train", names),
        contact_names=names,
        patient_ood_threshold=target["train_distance_q95"],
    )
    assert score["status"] == "INSUFFICIENT_IN_DISTRIBUTION_MODE_SUPPORT"
    assert score["selection_score"] >= 1.5


def test_formal_scorer_reports_insufficient_events_before_clustering():
    names, target = _patient_target()
    score = score_model_ranks_shaft_balanced(
        _heldout_ranks(target, per_mode=2),
        patient_centers=target["kmeans_centers"],
        target=_descriptor_target(target, "heldout", names),
        contact_names=names,
        patient_ood_threshold=target["train_distance_q95"],
    )
    assert score["status"] == "INSUFFICIENT_EVENTS"
    assert score["selection_score"] == 2.0


def test_single_shaft_subject_is_scorable_without_cross_shaft_pairs():
    names, target = _patient_target(("A1", "A2", "A3", "A4", "A5", "A6"))
    score = score_model_ranks_shaft_balanced(
        _heldout_ranks(target),
        patient_centers=target["kmeans_centers"],
        target=_descriptor_target(target, "heldout", names),
        contact_names=names,
        patient_ood_threshold=target["train_distance_q95"],
    )
    assert score["status"] == "EVALUABLE"
    assert score["weakest_mode_loss"] < 0.05


def test_profile_loss_gives_each_shaft_equal_weight():
    shafts = ["A", "A", "A", "A", "B"]
    patient = np.zeros(5)
    model = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0])
    assert _shaft_balanced_profile_error(model, patient, shafts) == 0.5


# --- provenance guards -----------------------------------------------------

def test_formal_scorer_rejects_a_reordered_patient_target():
    names, target = _patient_target()
    reordered = list(names[::-1])
    with pytest.raises(ValueError, match="contact order"):
        score_model_ranks_shaft_balanced(
            _heldout_ranks(target),
            patient_centers=target["kmeans_centers"],
            target=_descriptor_target(target, "heldout", reordered),
            contact_names=names,
            patient_ood_threshold=target["train_distance_q95"],
        )


def test_formal_scorer_rejects_a_target_without_a_contact_order():
    names, target = _patient_target()
    payload = _descriptor_target(target, "heldout", names)
    payload.pop("contact_order")
    with pytest.raises(ValueError, match="contact_order"):
        score_model_ranks_shaft_balanced(
            _heldout_ranks(target),
            patient_centers=target["kmeans_centers"],
            target=payload,
            contact_names=names,
            patient_ood_threshold=target["train_distance_q95"],
        )


def test_formal_scorer_rejects_a_target_spanning_other_contacts():
    names, target = _patient_target()
    payload = _descriptor_target(target, "heldout", names)
    payload["profiles"] = np.asarray(payload["profiles"])[:, :-1]
    payload["recruitment"] = np.asarray(payload["recruitment"])[:, :-1]
    with pytest.raises(ValueError, match="profile target"):
        score_model_ranks_shaft_balanced(
            _heldout_ranks(target),
            patient_centers=target["kmeans_centers"],
            target=payload,
            contact_names=names,
            patient_ood_threshold=target["train_distance_q95"],
        )


# --- within-shaft null contract -------------------------------------------

def test_within_shaft_null_enumerates_exactly_when_the_group_is_small():
    names = ["A1", "A2", "A3", "B1", "B2"]
    contract = within_shaft_null_contract(names, n_permutations=64, seed=12)
    rows = contract["permutations"]
    assert contract["within_shaft_group_size"] == 12
    assert contract["exhaustive"] is True
    assert contract["effective_null_size"] == 11
    assert rows.shape == (11, 5)
    assert len({row.tobytes() for row in rows}) == 11
    assert contract["minimum_reachable_p"] == pytest.approx(1.0 / 12.0)
    shafts = np.asarray(["A", "A", "A", "B", "B"])
    for row in rows:
        assert not np.array_equal(row, np.arange(5))
        np.testing.assert_array_equal(shafts[row], shafts)


def test_within_shaft_null_draws_distinct_rows_when_the_group_is_large():
    names = [f"{shaft}{index}" for shaft in "ABC" for index in range(1, 5)]
    contract = within_shaft_null_contract(names, n_permutations=64, seed=12)
    rows = contract["permutations"]
    assert contract["within_shaft_group_size"] == 24 ** 3
    assert contract["exhaustive"] is False
    assert contract["effective_null_size"] == 64
    assert len({row.tobytes() for row in rows}) == 64
    shafts = np.asarray([name[0] for name in names])
    for row in rows:
        np.testing.assert_array_equal(shafts[row], shafts)


def test_within_shaft_null_is_frozen_by_its_seed():
    names = [f"{shaft}{index}" for shaft in "ABC" for index in range(1, 5)]
    first = within_shaft_permutations(names, n_permutations=64, seed=12)
    repeat = within_shaft_permutations(names, n_permutations=64, seed=12)
    other = within_shaft_permutations(names, n_permutations=64, seed=13)
    np.testing.assert_array_equal(first, repeat)
    assert not np.array_equal(first, other)


def test_within_shaft_null_rejects_a_montage_without_a_multi_contact_shaft():
    with pytest.raises(ValueError, match="multi-contact shaft"):
        within_shaft_null_contract(["A1", "B1", "C1"], n_permutations=64, seed=12)
