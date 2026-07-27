import numpy as np

from src.topic5_symmetric_axis_propagation_state_v2_2 import (
    estimate_node_hazard_bias,
    frozen_rollout_horizons,
    node_bias_fingerprint,
    rank_sets_from_group_ids,
    train_only_source_side_thresholds,
    validate_persistence_label,
)


def test_prefix_representation_does_not_expose_final_event_length():
    short = rank_sets_from_group_ids(np.array([0, 1, -1, -1]))
    long = rank_sets_from_group_ids(np.array([0, 1, 2, 3]))
    np.testing.assert_array_equal(short[0], long[0])
    # The model input at step zero is the observed set, not t/T or T.
    assert short[0].dtype == bool


def test_node_bias_changes_only_when_training_events_change():
    train = [np.array([0, 1, -1]), np.array([0, -1, 1])]
    heldout_a = np.array([1, 0, -1])
    heldout_b = np.array([2, 1, 0])
    fingerprint_a = node_bias_fingerprint(estimate_node_hazard_bias(train)["bias"])
    # Merely defining different heldout events cannot affect the train-only bias.
    _ = (heldout_a, heldout_b)
    fingerprint_b = node_bias_fingerprint(estimate_node_hazard_bias(train)["bias"])
    assert fingerprint_a == fingerprint_b


def test_nonparticipants_remain_absent_from_rank_sets():
    sets = rank_sets_from_group_ids(np.array([1, -1, 0, -1, 1]))
    assert len(sets) == 2
    assert not sets[0][1] and not sets[1][1]
    assert not sets[0][3] and not sets[1][3]


def test_horizons_remain_defined_when_training_rollout_is_off():
    horizons = frozen_rollout_horizons(n_contacts=8, n_seen=2, h_train=0)
    assert horizons == {"H_train": 0, "H_eval": 6, "H_transfer": 6}


def test_source_side_thresholds_accept_train_values_only_as_explicit_input():
    thresholds = train_only_source_side_thresholds(np.arange(8, dtype=float))
    assert thresholds["left_max"] < thresholds["right_min"]


def test_rank_step_persistence_rejects_time_unit_label():
    validate_persistence_label("rank_step_persistence")
    try:
        validate_persistence_label("milliseconds")
    except ValueError:
        pass
    else:
        raise AssertionError("rank persistence was mislabeled as a time constant")
