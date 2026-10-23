import numpy as np

from src.interictal_nearest_direction_holdout import (
    nearest_direction_errors, shuffled_template_axes, shuffled_direction_errors,
)
from src.interictal_two_target_holdout import direction_error_suite
from src.topic5_interictal_direction_rose import fit_event_directions_3d


def test_direction_selects_nearest_axis_instead_of_rank_selected_axis():
    vectors = np.array([[1., 0, 0], [-1., 0, 0], [0., 0, 1]])
    axes = np.array([[[1., 0, 0], [-1., 0, 0]]])
    new = nearest_direction_errors(vectors, axes)
    old = direction_error_suite(vectors, axes, np.array([[1, 1, 1]]))
    np.testing.assert_allclose(new['angular_mae_deg'], [[0, 0, 90]])
    np.testing.assert_allclose(old['angular_mae_deg'], [[180, 0, 90]])
    assert np.all(new['angular_mae_deg'] <= old['angular_mae_deg'])


def test_out_of_plane_angle_and_signed_direction_are_not_discarded():
    vectors = np.array([[.5, 0, np.sqrt(.75)], [-1, 0, 0]])
    axes = np.array([[[1., 0, 0], [0, 1., 0]]])
    np.testing.assert_allclose(nearest_direction_errors(vectors, axes)['angular_mae_deg'], [[60, 90]])


def test_event_permutation_cannot_be_a_null_for_nearest_direction_loss():
    vectors = np.eye(3)
    axes = np.array([[[1., 0, 0], [0, 1., 0]]])
    a = nearest_direction_errors(vectors, axes)['angular_mae_deg'].mean()
    b = nearest_direction_errors(vectors[[2, 0, 1]], axes)['angular_mae_deg'].mean()
    np.testing.assert_allclose(a, b)


def test_template_null_preserves_support_and_shared_shuffle_and_refits_gradient():
    coords = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [9, 9, 9]])
    row = np.array([0., 1, 2, 3, np.nan])
    templates = np.array([[row, row], [row, row]])
    axes, info = shuffled_template_axes(templates, coords, 4, 7)
    keys = np.random.default_rng(7).random((4, 5))
    for i in range(4):
        shuffled = row.copy(); shuffled[:4] = row[:4][np.argsort(keys[i, :4])]
        expected = fit_event_directions_3d(shuffled[:, None], coords)['directions'][0]
        np.testing.assert_allclose(axes[i], np.broadcast_to(expected, (2, 2, 3)), atol=1e-12)
    assert info['support_counts'] == [4]*4 and info['rejected_degenerate_draws'] == 0


def test_null_averages_event_angles_not_cosines():
    vectors = np.array([[1., 0, 0], [0, 1., 0]])
    axes = np.array([[[[1., 0, 0], [1., 0, 0]]]])
    np.testing.assert_allclose(shuffled_direction_errors(vectors, axes, chunk_size=1), [[45]])
