from dataclasses import dataclass

import numpy as np

from src.topic5_patient_specific_rnn_bridge import (
    chronological_60_20_20,
    distribution_fields,
    permutation_indices,
    record_with_split,
    train_only_contact_features,
)
from scripts.summarize_topic5_patient_specific_rnn_bridge_v0_1 import (
    score_one_target,
)


@dataclass(frozen=True)
class Record:
    subject: str
    group_ids: np.ndarray
    group_count: np.ndarray
    event_split: np.ndarray
    contact_features: np.ndarray

    @property
    def train_indices(self):
        return np.flatnonzero(self.event_split == 0)

    @property
    def eval_indices(self):
        return np.flatnonzero(self.event_split == 1)


def record():
    groups = np.full((100, 6), -1, dtype=np.int16)
    groups[:, 0] = 0
    groups[::2, 1] = 1
    groups[1::2, 2] = 1
    return Record(
        subject="p1",
        group_ids=groups,
        group_count=np.max(groups, axis=1) + 1,
        event_split=np.r_[np.zeros(80, np.uint8), np.ones(20, np.uint8)],
        contact_features=np.zeros((6, 1), np.float32),
    )


def test_chronological_split_is_disjoint_and_60_20_20():
    fit, validation, test = chronological_60_20_20(record())
    assert (len(fit), len(validation), len(test)) == (60, 20, 20)
    assert not np.intersect1d(fit, validation).size
    assert not np.intersect1d(np.r_[fit, validation], test).size


def test_train_features_use_only_requested_events():
    item = record()
    fit, validation, _ = chronological_60_20_20(item)
    base = train_only_contact_features(item.group_ids, fit)
    changed = item.group_ids.copy()
    changed[validation, 5] = 0
    np.testing.assert_array_equal(base, train_only_contact_features(changed, fit))


def test_record_copy_exposes_only_requested_train_and_eval():
    item = record()
    fit, validation, _ = chronological_60_20_20(item)
    features = train_only_contact_features(item.group_ids, fit)
    copied = record_with_split(item, fit, validation, features)
    np.testing.assert_array_equal(copied.train_indices, fit)
    np.testing.assert_array_equal(copied.eval_indices, validation)


def test_distribution_fields_are_contact_aligned():
    item = record()
    fields = distribution_fields(item.group_ids, item.group_count)
    assert set(fields) == {
        "participation", "early_joint_mass", "late_joint_mass",
        "endpoint_joint_mass", "weighted_earliness",
    }
    assert all(value.shape == (6,) for value in fields.values())
    np.testing.assert_allclose(fields["participation"][:3], [1.0, 0.5, 0.5])


def test_permutation_draws_preserve_shaft_membership():
    names = np.asarray(["A1", "A2", "B1", "B2", "B3", "C1"])
    draw = permutation_indices(names, n_draws=20, seed=4, within_shaft=True)
    assert draw.shape == (20, 6)
    assert np.all(np.isin(draw[:, :2], [0, 1]))
    assert np.all(np.isin(draw[:, 2:5], [2, 3, 4]))
    assert np.all(draw[:, 5] == 5)


def test_max_field_selection_is_repeated_inside_the_null():
    fields = {
        "a": np.arange(6, dtype=float),
        "b": np.asarray([0, 2, 1, 4, 3, 5], dtype=float),
    }
    target = np.arange(6, dtype=float)
    permutations = np.asarray([
        [5, 4, 3, 2, 1, 0],
        [0, 2, 1, 4, 3, 5],
    ])
    observed, null, selected = score_one_target(
        fields, target, ["a", "b"], permutations
    )
    assert observed == 1.0
    assert selected == "a"
    assert null.shape == (2,)
    # Draw one is best explained by b, proving candidate selection was rerun.
    assert null[1] == 1.0
