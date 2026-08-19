import numpy as np

from scripts.paper_figures.plot_fig5_data_driven_zm_main import (
    _contact_order,
    _registered_xy,
    _signed_bandpass,
)
from scripts.replay_topic4_zm_fig5_frames import _select_display_event


def test_display_event_rule_uses_latest_complete_observed_event():
    t_on = np.array([10.0, 30.0, 50.0, 70.0])
    t_off = np.array([20.0, 40.0, 60.0, 80.0])
    returned = np.array([True, True, False, True])
    before = np.array([True, True, True, False])
    onsets = np.full((4, 10), np.nan)
    onsets[0, :8] = 12.0
    onsets[1, :9] = 32.0
    onsets[2, :] = 52.0
    onsets[3, :] = 72.0
    assert _select_display_event(
        t_on, t_off, returned, before, onsets, onset_ms=100.0) == 1


def test_registered_coordinates_align_axis_with_positive_x():
    xy = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]])
    got = _registered_xy(xy, axis_unit=np.array([1.0, 0.0]),
                         origin=np.array([1.0, 1.0]))
    assert np.allclose(got, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def test_contact_order_places_scl_below_icl_and_preserves_numeric_order():
    names = np.array(["ICL11", "SCL9", "ICL1", "SCL6", "ICL2"])
    order = _contact_order(names)
    assert names[order].tolist() == ["SCL6", "SCL9", "ICL1", "ICL2", "ICL11"]


def test_signed_bandpass_retains_50hz_and_rejects_10hz():
    dt_ms = 1.0
    t = np.arange(0.0, 4.0, dt_ms * 1e-3)
    raw = (np.sin(2 * np.pi * 50.0 * t)
           + np.sin(2 * np.pi * 10.0 * t))[:, None]
    filtered = _signed_bandpass(raw, dt_ms)[:, 0]
    reference_50 = np.sin(2 * np.pi * 50.0 * t)
    reference_10 = np.sin(2 * np.pi * 10.0 * t)
    centre = slice(500, -500)
    amp_50 = abs(np.dot(filtered[centre], reference_50[centre]))
    amp_10 = abs(np.dot(filtered[centre], reference_10[centre]))
    assert amp_50 > 20.0 * amp_10
    assert np.min(filtered[centre]) < 0.0 < np.max(filtered[centre])
