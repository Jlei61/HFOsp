import numpy as np
import pytest

from src.topic5_axis_direction import (gradient_angle, event_angles_by_template,
                                       circular_mean, resultant_length, rotate_to_reference,
                                       axial_mean, axial_resultant_length)

# a small 5-point cloud spanning x and y
X = np.array([0.0, 1.0, 2.0, 1.0, 1.0])
Y = np.array([1.0, 1.0, 1.0, 0.0, 2.0])


def test_gradient_angle_increasing_in_x_is_zero():
    ang = gradient_angle(X, Y, X)          # values increase with x -> angle ~ 0
    assert abs(ang) < 1e-6 or abs(ang - 2 * np.pi) < 1e-6


def test_gradient_angle_increasing_in_y_is_half_pi():
    assert abs(gradient_angle(X, Y, Y) - np.pi / 2) < 1e-6


def test_gradient_angle_reversed_values_flip_by_pi():
    a = gradient_angle(X, Y, X)
    b = gradient_angle(X, Y, -X)
    assert abs(np.mod(b - a, 2 * np.pi) - np.pi) < 1e-6


def test_gradient_angle_degenerate_returns_nan():
    assert np.isnan(gradient_angle(X, Y, np.ones_like(X)))      # no gradient
    assert np.isnan(gradient_angle([0, 1], [0, 1], [0, 1]))     # <3 points


def test_event_angles_by_template_groups_and_drops_unassigned():
    # event 0 increases +x (angle 0), event 1 increases -x (angle pi), event 2 unassigned
    vals = np.column_stack([X, -X, X])
    labels = np.array([0, 1, -1])
    grp = event_angles_by_template(vals, X, Y, labels)
    assert grp[0].size == 1 and abs(np.mod(grp[0][0], 2 * np.pi)) < 1e-6
    assert grp[1].size == 1 and abs(grp[1][0] - np.pi) < 1e-6


def test_event_angles_by_template_label_length_mismatch_raises():
    with pytest.raises(ValueError):
        event_angles_by_template(np.zeros((5, 3)), X, Y, np.array([0, 1]))


def test_circular_mean_wraps_around_zero():
    # -10 deg and +10 deg -> mean ~ 0
    m = circular_mean([np.deg2rad(350), np.deg2rad(10)])
    assert min(abs(m), abs(m - 2 * np.pi)) < 1e-6


def test_resultant_length_extremes():
    assert resultant_length([0.3, 0.3, 0.3]) == pytest.approx(1.0)
    assert resultant_length(np.linspace(0, 2 * np.pi, 360, endpoint=False)) < 1e-6


def test_rotate_to_reference_puts_ref_at_zero():
    out = rotate_to_reference(np.array([0.0, np.pi / 2]), np.pi / 2)
    assert out[0] == pytest.approx(3 * np.pi / 2)
    assert out[1] == pytest.approx(0.0)


def test_axial_mean_does_not_cancel_on_bidirectional_set():
    # {10deg, 190deg} are the SAME axis -> plain circular mean ~cancels (R~0), axial mean ~10deg
    bi = [np.deg2rad(10), np.deg2rad(190)]
    assert resultant_length(bi) < 1e-6                      # plain mean cancels
    assert axial_mean(bi) == pytest.approx(np.deg2rad(10), abs=1e-6)
    assert axial_resultant_length(bi) == pytest.approx(1.0)  # perfect axis


def test_axial_mean_orthogonal_axes_degenerate_nan():
    # 0deg and 90deg are orthogonal axes -> no net axis
    assert np.isnan(axial_mean([0.0, np.pi / 2]))


def test_axial_mean_in_zero_to_pi():
    m = axial_mean([np.deg2rad(200), np.deg2rad(210)])      # ~205deg axis == ~25deg
    assert 0 <= m < np.pi
    assert m == pytest.approx(np.deg2rad(25), abs=1e-6)
