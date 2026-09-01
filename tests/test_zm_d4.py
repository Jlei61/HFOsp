"""Covariant square-symmetry transform of the data-driven substrate."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_d4 import (  # noqa: E402
    D4_ELEMENTS, d4_matrix, inverse_query_positions, transform_flow_coefficients,
    transform_report)
from src.topic4_local_connectivity import local_pair_features  # noqa: E402


def test_elements_are_orthogonal_with_unit_determinant_magnitude():
    for element in D4_ELEMENTS:
        R = d4_matrix(element)
        assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)
        assert np.isclose(abs(np.linalg.det(R)), 1.0)


def test_r90_has_order_four():
    R = d4_matrix("r90")
    assert np.allclose(np.linalg.matrix_power(R, 4), np.eye(2), atol=1e-12)


def test_inverse_query_round_trips_about_the_sheet_centre():
    rng = np.random.default_rng(0)
    xy = rng.random((50, 2)) * 20.0
    for element in D4_ELEMENTS:
        once = inverse_query_positions(xy, element, L=20.0)
        R = d4_matrix(element)
        back = (R @ (once - 10.0).T).T + 10.0
        assert np.allclose(back, xy, atol=1e-10)


def test_flow_coefficients_only_swap_and_negate_so_bounds_are_preserved():
    coefficients = np.array([[0.5, -0.5, 0.15, -0.15, 0.15, -0.15],
                             [-0.4, 0.3, 0.1, 0.05, -0.12, 0.09]])
    bounds = np.array([0.5, 0.5, 0.15, 0.15, 0.15, 0.15])
    for element in D4_ELEMENTS:
        out = transform_flow_coefficients(coefficients, element)
        assert out.shape == coefficients.shape
        assert np.all(np.abs(out) <= bounds + 1e-12)
        # the group permutes and negates the two components, so the multiset
        # of magnitudes -- and hence the flow vector's norm -- is preserved
        assert np.array_equal(np.sort(np.abs(out[:, 4:]), axis=1),
                              np.sort(np.abs(coefficients[:, 4:]), axis=1))
        assert np.allclose(np.linalg.norm(out[:, 4:], axis=1),
                           np.linalg.norm(coefficients[:, 4:], axis=1))
        assert np.array_equal(out[:, :4], coefficients[:, :4])


def test_flow_rule_is_covariant_when_the_edge_is_rotated_with_it():
    """Proves the field-and-flow RULE is a rigid image under the transform.
    Note the edge endpoints are rotated here; the actual control does not rotate
    the graph, which is why it is a re-registration control and not an isometry."""
    rng = np.random.default_rng(3)
    target = rng.random((200, 2)) * 20.0
    source = rng.random((200, 2)) * 20.0
    h_t, h_s = rng.random(200), rng.random(200)
    coefficients = np.array([[0.3, -0.2, 0.1, -0.05, 0.12, -0.07],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    base = local_pair_features(target, source, h_t, h_s, length_scale=0.38)
    contribution = base @ coefficients[0]

    for element in D4_ELEMENTS:
        R = d4_matrix(element)
        rot_t = (R @ (target - 10.0).T).T + 10.0
        rot_s = (R @ (source - 10.0).T).T + 10.0
        rot_c = transform_flow_coefficients(coefficients, element)
        rotated = local_pair_features(rot_t, rot_s, h_t, h_s, length_scale=0.38)
        assert np.allclose(rotated @ rot_c[0], contribution, atol=1e-10)


def test_field_only_rotation_would_reverse_the_flow_term():
    """Why the covariant coefficient transform is required at all: the last two
    features are signed and linear in displacement, so rotating the field alone
    breaks the field-to-flow correspondence."""
    rng = np.random.default_rng(5)
    target = rng.random((200, 2)) * 20.0
    source = rng.random((200, 2)) * 20.0
    h_t, h_s = rng.random(200), rng.random(200)
    coefficients = np.array([[0.0, 0.0, 0.0, 0.0, 0.12, -0.07],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    base = local_pair_features(target, source, h_t, h_s, length_scale=0.38)
    R = d4_matrix("r180")
    rot_t = (R @ (target - 10.0).T).T + 10.0
    rot_s = (R @ (source - 10.0).T).T + 10.0
    rotated = local_pair_features(rot_t, rot_s, h_t, h_s, length_scale=0.38)
    # field rotated, coefficients NOT -> the flow contribution flips sign
    assert np.allclose(rotated @ coefficients[0], -(base @ coefficients[0]), atol=1e-10)


def test_transform_report_separates_undirected_axis_from_directed_sense():
    axis = np.array([0.92182673, -0.38760221])
    coefficients = np.zeros((2, 6))
    r180 = transform_report("r180", coefficients, axis_unit=axis)
    assert r180["preserves_undirected_axis"] is True
    assert r180["preserves_directed_axis"] is False
    r90 = transform_report("r90", coefficients, axis_unit=axis)
    assert r90["preserves_undirected_axis"] is False
    assert "isometric" not in r180["name"].lower()
