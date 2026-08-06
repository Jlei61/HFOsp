"""Regression tests for the leave-contact-out condition.

Both bugs this file locks down produced a publishable-looking number rather than
an error. The first deleted the withheld contact from the test targets as well as
the training ones, so the evaluation asked "can you predict that these contacts
never participate" -- answered perfectly and for free, at 0.0006 loss. The second
left the withheld contact its own free bias, trained as a permanent negative
example, so the model was handed the answer we claimed to withhold.

The invariant underneath both: after the withholding, the ONLY thing still
specific to a withheld contact is where it sits.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_topic5_spo_unit import (  # noqa: E402
    densify, neutralise_holdout_bias, partition,
)
from src.topic5_spatial_propagation_operator import OperatorConfig, SPOModel  # noqa: E402

N_CONTACTS = 8
HOLDOUT = np.array([2, 5])


@pytest.fixture
def events():
    rng = np.random.default_rng(0)
    rows = np.full((60, N_CONTACTS), -1, dtype=np.int64)
    for e in range(60):
        members = rng.choice(N_CONTACTS, size=rng.integers(3, N_CONTACTS), replace=False)
        rows[e, members] = np.arange(len(members))
    return {"group_ids": rows, "split": np.repeat([0, 1, 2], 20)}


@pytest.fixture
def parts(events):
    return partition(events, HOLDOUT)


def test_withheld_contacts_keep_their_truth_in_the_test_target(parts):
    """The point of the test set is what the withheld contacts actually did."""
    index = torch.as_tensor(HOLDOUT)
    assert float(parts["test"].target[:, :, index].sum()) > 0


def test_withheld_contacts_are_absent_from_the_training_target(parts):
    index = torch.as_tensor(HOLDOUT)
    for name in ("train", "validation"):
        assert float(parts[name].target[:, :, index].sum()) == 0


def test_withheld_contacts_never_enter_the_input(parts):
    """The strong condition: unseen in training and unseen at test."""
    index = torch.as_tensor(HOLDOUT)
    for name in ("train", "validation", "test"):
        assert float(parts[name].x[:, :, index].sum()) == 0
    assert float(parts["test"].recruited[:, :, index].sum()) == 0


def test_withheld_contacts_are_not_scored_during_training(parts):
    """Otherwise they are permanent negatives and their bias learns the answer."""
    index = torch.as_tensor(HOLDOUT)
    for name in ("train", "validation"):
        assert not bool(parts[name].available[:, :, index].any())


def test_withheld_contacts_are_candidates_at_test(parts):
    """They must be able to win, or top-1 is zero by construction."""
    index = torch.as_tensor(HOLDOUT)
    available = parts["test"].available[:, :, index]
    valid = parts["test"].valid[:, :, None].expand(-1, -1, len(HOLDOUT))
    assert torch.equal(available, valid)


def test_densify_closes_the_gap_the_withholding_leaves():
    rows = np.array([[0, 1, 2, 3, 4, 5, -1, -1]])
    trimmed = rows.copy()
    trimmed[:, HOLDOUT] = -1
    assert sorted(trimmed[0][trimmed[0] >= 0]) == [0, 1, 3, 4]  # holes at 2 and 5
    kept = densify(trimmed)[0]
    assert sorted(kept[kept >= 0]) == [0, 1, 2, 3]


def test_neutral_bias_leaves_position_as_the_only_contact_specific_input():
    config = OperatorConfig(
        variant="STATIC", n_contacts=N_CONTACTS, grid_shape=(4, 4), microsteps=1, seed=0)
    model = SPOModel(config)
    with torch.no_grad():
        model.contact_bias.copy_(torch.tensor(
            [1.0, 2.0, -9.0, 4.0, 5.0, -9.0, 7.0, 8.0]))
    neutral = neutralise_holdout_bias(model, HOLDOUT)
    retained = np.setdiff1d(np.arange(N_CONTACTS), HOLDOUT)
    assert neutral == pytest.approx(float(np.mean([1, 2, 4, 5, 7, 8])))
    for c in HOLDOUT:
        assert float(model.contact_bias[c]) == pytest.approx(neutral)
    for c in retained:
        assert float(model.contact_bias[c]) != pytest.approx(neutral)
