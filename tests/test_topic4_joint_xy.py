import numpy as np
import pytest
from src.topic4_joint_xy import (joint_features, projections, projected_quantiles,
    joint_distance, proposal_rng, round_action, observable_groups)

XY = np.array([[2, 4], [5, 4], [8, 4], [11, 4], [3, 12], [9, 12]], float)
GROUPS = {'ICL': [0, 1, 2, 3], 'SCL': [4, 5]}


def test_rank_only_participants_and_unreadable_rows_are_explicit():
    a = np.array([[0, np.nan, 2, np.nan, np.nan, np.nan], [np.nan]*6])
    f = joint_features(a, XY, GROUPS)
    assert f.shape == (2, 26)
    assert np.isfinite(f).all()
    assert np.count_nonzero(f[1]) == 0
    # First / last participating ranks encode 0 and 1, independent of missing slots.
    assert f[0, 6] == pytest.approx(.5/np.sqrt(6)/2)
    assert f[0, 8] == pytest.approx(1./np.sqrt(6)/2)


def test_time_origin_shift_does_not_change_joint_representation():
    a = np.array([[10, 20, np.nan, 40, 50, 70], [1, 2, 3, 4, 5, 6]])
    np.testing.assert_allclose(joint_features(a, XY, GROUPS), joint_features(a+1000, XY, GROUPS))


def test_joint_loss_distinguishes_reversal_mode_mass_and_spatial_correspondence():
    a = np.array([0, 20, 40, 60, 80, 100.]); b = 100-a
    source = np.array([a]*24+[b]*8)
    f = joint_features(source, XY, GROUPS); axes = projections(f.shape[1]); ref = projected_quantiles(f, axes)
    assert joint_distance(f, axes, ref) == 0
    assert joint_distance(joint_features(np.array([a]*8+[b]*24), XY, GROUPS), axes, ref) > .005
    assert joint_distance(joint_features(source, XY[::-1], GROUPS), axes, ref) > .005


def test_sparse_and_empty_are_never_a_perfect_match():
    source = np.array([[0., 20, 40, 60, 80, 100.]])
    f = joint_features(source, XY, GROUPS); axes = projections(f.shape[1]); ref = projected_quantiles(f, axes)
    sparse = source.copy(); sparse[:, 1:] = np.nan
    assert joint_distance(joint_features(sparse, XY, GROUPS), axes, ref) > .01
    assert joint_distance(joint_features(np.empty((0, 6)), XY, GROUPS), axes, ref) is None


def test_different_rounds_different_starts_same_round_reproducible():
    np.testing.assert_array_equal(proposal_rng(2, 1).normal(size=100), proposal_rng(2, 1).normal(size=100))
    assert not np.array_equal(proposal_rng(2, 1).normal(size=100), proposal_rng(2, 2).normal(size=100))


def test_plateau_restarts_but_does_not_change_loss_or_claim_capacity_failure():
    result = round_action([.1, .099, .098], .098, 2., 30)
    assert result['action'] == 'increase_random_restart_fraction'
    assert 'UNRESOLVED' in result['diagnosis']
    assert round_action([], .1, 2., 3)['action'] == 'longer_paired_screen'


def test_group_centroids_capture_known_contact_delays_and_preserve_mask():
    time = np.arange(1000)*2.
    centers = np.array([950, 970, 990, 1010, 1030, 1050.])
    env = np.exp(-.5*((time[None]-centers[:, None])/12.)**2)
    x, meta = observable_groups(env, 2.)
    assert len(x) == 1
    np.testing.assert_allclose(x[0], centers, atol=1.)
    assert meta['windows_ms'][0][1]-meta['windows_ms'][0][0] == pytest.approx(250.)


def test_silence_and_single_contact_noise_do_not_make_group_events():
    env = np.zeros((6, 1000))
    assert observable_groups(env, 2.)[0].shape == (0, 6)
    env[0, 490:510] = 1.
    assert observable_groups(env, 2.)[0].shape == (0, 6)


def test_two_overlapping_group_windows_are_not_counted_as_independent():
    time = np.arange(1000)*2.
    # Two distinct bursts, separated 200 ms, yield overlapping 250-ms windows.
    env = np.tile(np.exp(-.5*((time-900)/5.)**2)+np.exp(-.5*((time-1100)/5.)**2), (6, 1))
    assert len(observable_groups(env, 2.)[0]) == 0


def test_infinite_values_are_rejected():
    with pytest.raises(ValueError): joint_features(np.full((1, 6), np.inf), XY, GROUPS)


def test_confirmation_cannot_pass_on_direction_alone_or_one_lucky_network():
    import copy
    from scripts.run_topic4_joint_xy_adaptive import is_qualified
    keys = ('joint_distance', 'D_support', 'D_order', 'D_lag', 'direction_distance')
    calibration = {'samples': {str(n): {'q95': {k: .1 for k in keys}} for n in (16, 64, 128)}}
    plan = {'confirmation': {'require_disk_clearance_mm': 0., 'minimum_pooled_events': 64,
                            'n_seeds': 6, 'minimum_events_per_seed': 5, 'minimum_good_joint_loss_seeds': 5}}
    row = {**{k: .05 for k in keys}, 'n_events': 80, 'support': {'conditional_gate': True},
           'units': [{'seed': i, 'runaway': False, 'geometry': {'minimum_clearance_mm': 1.},
                      'metrics': {'n_events': 13, 'joint_distance': .05}} for i in range(6)]}
    assert is_qualified(row, calibration, plan, confirmation=True)['pass']
    bad = copy.deepcopy(row); bad['joint_distance'] = .2; bad['direction_distance'] = 0.
    assert not is_qualified(bad, calibration, plan, confirmation=True)['pass']
    bad = copy.deepcopy(row)
    for unit in bad['units'][1:]: unit['metrics']['joint_distance'] = .5
    assert not is_qualified(bad, calibration, plan, confirmation=True)['pass']


def test_threshold_does_not_relax_at_sample_count_boundary():
    from scripts.run_topic4_joint_xy_adaptive import threshold_table
    c = {'16': {'q95': {'joint_distance': .2}}, '32': {'q95': {'joint_distance': .1}}}
    assert threshold_table(c, 17)['joint_distance'] == .1


def test_low_conditional_support_still_generates_reproducible_xy_proposals():
    from scripts.run_topic4_joint_xy_adaptive import new_proposals
    from src.topic4_xy_search import field_descriptor
    grid = np.linspace(.05, 19.95, 180); x, y = np.meshgrid(grid, grid)
    pos = np.column_stack([x.ravel(), y.ravel()])
    pool = [{'candidate': {'node_field': field_descriptor([[4., 9.], [16., 4.]])},
             'explorable': True, 'exploration_score': .1, 'conditional_gate': False}]
    plan = {'search': {'proposals_per_round': 4, 'random_fraction': .5, 'restart_random_fraction': .85,
                       'interior_fraction': .75, 'local_scales_mm': [.6, 1.5, 3.]}}
    action = {'action': 'multi_anchor_local_plus_random'}
    a = new_proposals(pool, pos, 10, 1, plan, action)
    b = new_proposals(pool, pos, 10, 1, plan, action)
    c = new_proposals(pool, pos, 10, 2, plan, action)
    assert len(a) == 4
    assert [r['node_field']['field_sha256'] for r in a] == [r['node_field']['field_sha256'] for r in b]
    assert [r['node_field']['field_sha256'] for r in a] != [r['node_field']['field_sha256'] for r in c]
