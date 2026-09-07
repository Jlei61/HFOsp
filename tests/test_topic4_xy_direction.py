import numpy as np
import pytest
from src.topic4_xy_direction import (onset_directions, direction_histogram,
    direction_distance, direction_summary, core_alignment_penalty, admission_slots)

XY = np.array([[0,0],[1,0],[0,1],[1,1],[2,0],[0,2]], float)


def test_signed_early_to_late_and_translation_invariance():
    onsets = (XY @ np.array([2., 1.]))[None]
    v = onset_directions(onsets, XY)
    assert v['angle_rad'][0] == pytest.approx(np.arctan2(1, 2))
    assert v['coherence'][0] == pytest.approx(1)
    shifted = onset_directions(onsets + 900, XY + [30, -40])
    np.testing.assert_allclose(v['angle_rad'], shifted['angle_rad'])


def test_distribution_detects_reversal_and_missing_mode():
    a = XY[:, 0]; b = -a
    hp = direction_histogram(onset_directions(np.array([a]*7+[b]*3), XY))
    hr = direction_histogram(onset_directions(np.array([b]*7+[a]*3), XY))
    hm = direction_histogram(onset_directions(np.array([a]*10), XY))
    assert direction_distance(hp, hp) == pytest.approx(0)
    assert direction_distance(hp, hr) > .1
    assert direction_distance(hp, hm) > .1
    # An exactly balanced forward/reverse distribution is reversal-invariant.
    balanced = np.array([a,b])
    assert direction_distance(direction_histogram(onset_directions(balanced,XY)),
        direction_histogram(onset_directions(-balanced,XY))) < 1e-7


def test_unreadable_and_degenerate_events_keep_denominator():
    rows = np.array([XY[:,0], np.full(6,np.nan), np.ones(6)])
    hist = direction_histogram(onset_directions(rows, XY))
    assert hist.sum() == pytest.approx(1)
    assert hist[-1] == pytest.approx(2/3)
    line = XY.copy(); line[:,1] = 0
    hist = direction_histogram(onset_directions(rows, line))
    assert hist[-1] == 1


def test_sparse_near_collinear_is_unresolved_and_shape_checked():
    line = np.column_stack([np.arange(6), np.arange(6)**2 * 1e-5])
    assert direction_histogram(onset_directions(XY[:,0][None], line))[-1] == 1
    with pytest.raises(ValueError): onset_directions(np.ones((2,5)), XY)
    with pytest.raises(ValueError): onset_directions(np.full((1,6),np.inf), XY)


def test_core_prior_invariant_to_core_labels():
    centers = [[2,3],[12,3]]
    assert core_alignment_penalty(centers,0) == pytest.approx(0)
    assert core_alignment_penalty(centers,90) == pytest.approx(1)
    assert core_alignment_penalty(centers,20) == pytest.approx(core_alignment_penalty(centers[::-1],20))


def test_admission_accounts_for_unmaterialized_memory_and_reserve():
    assert admission_slots(200,[2]*10,10,24,40) == 8
    assert admission_slots(100,[2]*10,10,24,40) == 0
    assert admission_slots(500,[2]*24,10,24,40) == 0
    assert admission_slots(39,[],10,24,40) == 0


def test_rotating_geometry_rotates_measured_behavior():
    angle=.4; rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    v=onset_directions(XY[:,0][None],XY@rot.T)
    assert v['angle_rad'][0] == pytest.approx(angle)


def test_empty_event_set_not_perfect_match():
    h=direction_histogram(onset_directions(np.empty((0,6)),XY))
    assert h is None
    assert direction_distance(h,np.ones(25)/25) is None
