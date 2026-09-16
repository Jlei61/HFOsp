import numpy as np
import pytest

from src.sef_hfo_snn_adapter import _bin_and_smooth
from src.topic4_node_dualmode import lineage_restricted_neuron_contact_readout
from src.topic4_xy_readout_audit import (
    window_onsets, support_summary, support_gate_probability,
)

GROUPS = {'ICL': [0, 1], 'SCL': [2, 3]}
PAIRS = {'ICL-ICL': [[0, 1]], 'SCL-SCL': [[2, 3]],
         'ICL-SCL': [[0, 2], [0, 3], [1, 2], [1, 3]]}


def test_full_readout_matches_original_when_every_neuron_belongs_to_root():
    rng = np.random.default_rng(17)
    spikes = rng.random((1600, 8)) < .06
    weights = rng.random((4, 8)); weights /= weights.sum(axis=1, keepdims=True)
    rate, dt = _bin_and_smooth(spikes, .1, 2., 5.)
    env = (rate @ weights.T).T
    binned = spikes.reshape(80, 20, 8).sum(axis=1)
    windows = [(0., 30.), (46., 100.), (140., 160.)]
    events = [dict(t_on=lo, t_off=hi, cascade_id=1) for lo, hi in windows]
    original = lineage_restricted_neuron_contact_readout(
        binned, np.arange(8), np.ones((80, 2, 4), int), events, weights,
        frame_ms=dt, smooth_ms=5., participation_margin_fraction=.1,
        timing_fraction=.5)['onsets']
    np.testing.assert_allclose(window_onsets(env, windows, dt), original, equal_nan=True)


def test_silence_and_empty_windows_stay_unreadable_without_dropping_rows():
    got = window_onsets(np.zeros((4, 30)), [(0, 10), (20, 20), (100, 110)], 2.)
    assert got.shape == (3, 4)
    assert np.isnan(got).all()


def test_same_event_local_threshold_is_not_controlled_by_remote_large_event():
    env = np.zeros((2, 30)); env[:, 2:4] = [[1.], [.2]]; env[0, 20] = 100.
    local = window_onsets(env, [(0, 10)], 1.)
    whole = window_onsets(env, [(0, 10)], 1., bar_scope='run')
    np.testing.assert_array_equal(local, [[2., 2.]])
    assert np.isnan(whole).all()


def test_both_shaft_events_do_not_imply_sufficient_pair_coverage():
    # Every event bridges shafts, but only one of four cross pairs is observed.
    x = np.tile([0., np.nan, 1., np.nan], (100, 1))
    s = support_summary(x, GROUPS, PAIRS)
    assert s['both_shafts_fraction'] == 1.
    assert s['eligible_cross_pairs'] == 1
    assert not s['conditional_gate']


def test_bootstrap_cannot_invent_unobserved_joint_recruitment():
    x = np.array([[0., 1., np.nan, np.nan], [np.nan, np.nan, 0., 1.]])
    assert support_gate_probability([x, x], GROUPS, PAIRS, 320, 64, 12) == 0.


def test_bootstrap_positive_control_and_determinism():
    x = np.zeros((8, 4))
    assert support_gate_probability([x, x], GROUPS, PAIRS, 10, 64, 12) == 1.
    x[3:, 3] = np.nan
    a = support_gate_probability([x, x], GROUPS, PAIRS, 10, 64, 12)
    assert a == support_gate_probability([x, x], GROUPS, PAIRS, 10, 64, 12)


@pytest.mark.parametrize('env,dt', [(np.ones(4), 2.), (np.array([[np.inf]]), 2.),
                                  (np.ones((4, 3)), 0.)])
def test_invalid_envelope_rejected(env, dt):
    with pytest.raises(ValueError): window_onsets(env, [(0., 2.)], dt)
