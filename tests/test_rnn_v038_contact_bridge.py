import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location('bridge_metrics', Path(__file__).parents[1] / 'scripts/contact_bridge_metrics.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_true_set_probability_normalizes_over_subsets():
    targets = m.subsets(5, 2)
    z = np.tile([2, -1, 0, 0.4, 1], (len(targets), 1))
    scores = m.exact_set_scores(z, targets, np.ones_like(z, dtype=bool))
    assert np.exp(-scores['set_nll']).sum() == pytest.approx(1)


def test_uniform_multicontact_is_log_choose_not_average_log_softmax():
    score = m.exact_set_scores([[0, 0, 0, 0]], [[1, 1, 0, 0]], [[1, 1, 1, 1]])
    assert score['set_nll'][0] == pytest.approx(np.log(6))
    assert score['set_nll'][0] != pytest.approx(np.log(4))


def test_k_one_reduces_to_softmax_and_additive_constant_cancels():
    z = np.array([[2., 1., 0.]])
    target = [[1, 0, 0]]
    available = [[1, 1, 1]]
    a = m.exact_set_scores(z, target, available)['set_nll']
    b = m.exact_set_scores(z + 100, target, available)['set_nll']
    assert a[0] == pytest.approx(np.log(np.exp(z).sum()) - 2)
    np.testing.assert_allclose(a, b)


def test_forced_and_stop_rows_are_excluded_from_identity():
    a = m.exact_set_scores(np.zeros((2, 3)), [[1, 1, 0], [0, 0, 0]], [[1, 1, 0], [1, 1, 0]])
    assert not a['informative'].any()
    assert np.isnan(a['set_nll']).all()


def test_target_must_be_available():
    with pytest.raises(ValueError):
        m.exact_set_scores([[0, 0]], [[1, 0]], [[0, 1]])


def test_fit_dictionary_does_not_learn_branching_from_selection():
    ranks = np.array([[0, 1, 2, -1]] * 5 + [[0, 1, -1, 2]] * 5)
    assert m.fit_branch_dictionary(ranks, np.arange(5)) == {}
    assert len(m.fit_branch_dictionary(ranks, np.arange(10))) == 1


def test_wrong_time_donors_preserve_prefix_cardinality_and_segment():
    ranks = np.array([[0, 1, 2, -1], [0, 1, -1, 2]] * 6)
    times = np.arange(12) * 10000
    segments = np.repeat([0, 1], 6)
    donors = m.matched_donors(ranks, times, segments, np.arange(12))
    assert (donors >= 0).all()
    for i, d in enumerate(donors):
        assert segments[i] == segments[d]
        assert m.prefix_key(ranks[i], include_k=True) == m.prefix_key(ranks[d], include_k=True)
        assert abs(times[i] - times[d]) >= 7200


def test_known_signal_instrument_and_null():
    target = np.eye(4)[np.arange(40) % 4]
    state_logits = target * 4
    base = m.exact_set_scores(np.zeros_like(target), target, np.ones_like(target))
    state = m.exact_set_scores(state_logits, target, np.ones_like(target))
    shifted = m.exact_set_scores(np.roll(state_logits, 1, axis=0), target, np.ones_like(target))
    assert state['set_nll'].mean() < base['set_nll'].mean() < shifted['set_nll'].mean()
    assert state['set_accuracy'].mean() == 1
    assert shifted['set_accuracy'].mean() == 0
    null = m.paired_summary(base['set_nll'], base['set_nll'], np.arange(40) * 1000, np.zeros(40, dtype=int))
    assert null['event_weighted_gain'] == 0
    assert null['positive_blocks'] == 0


def test_patient_support_not_multiplied_by_event_density():
    a = m.paired_summary(np.zeros(11), np.r_[np.ones(10), -2], np.r_[np.arange(10), 8000], np.zeros(11, dtype=int))
    assert a['event_weighted_gain'] > 0
    assert a['block_equal_gain'] < 0
    assert a['n_blocks'] == 2


def test_suffix_empty_events_not_assigned_zero_loss():
    values = np.array([[np.nan, np.nan], [1, np.nan]])
    result = m.event_mean(values, np.ones_like(values, dtype=bool))
    assert np.isnan(result[0])
    assert result[1] == 1
