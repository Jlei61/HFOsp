import numpy as np
import pytest
from src.topic4_core_field import axis_coords, axial_basis_centers, partition_of_unity


def test_axis_coords_projects_onto_axis_and_perpendicular():
    pos = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]])
    s, r = axis_coords(pos, np.array([0.0, 0.0]), np.array([1.0, 0.0]))
    assert np.allclose(s, [1.0, 0.0, 2.0])
    assert np.allclose(np.abs(r), [0.0, 1.0, 2.0])


def test_axis_coords_axis_flip_negates_s_and_preserves_abs_r():
    rng = np.random.default_rng(0)
    pos = rng.uniform(-5, 5, size=(50, 2))
    center, u = np.array([0.3, -0.2]), np.array([0.6, 0.8])
    s1, r1 = axis_coords(pos, center, u)
    s2, r2 = axis_coords(pos, center, -u)
    assert np.allclose(s2, -s1)
    assert np.allclose(np.abs(r2), np.abs(r1))


def test_partition_of_unity_rows_sum_to_one():
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    Phi = partition_of_unity(s, kappa, 1.2 * (kappa[1] - kappa[0]))
    assert Phi.shape == (200, 9)
    assert np.allclose(Phi.sum(axis=1), 1.0, atol=1e-12)


def test_uniform_weights_give_a_flat_axial_profile():
    """Why partition-of-unity is required: unnormalised Gaussians sag where fewer
    bases overlap, which would make `uniform_axial` a broad peak, not a corridor."""
    kappa = axial_basis_centers((-8.0, 8.0), M=9)
    s = np.linspace(-8.0, 8.0, 200)
    profile = partition_of_unity(s, kappa, 1.2 * (kappa[1] - kappa[0])) @ np.full(9, 1 / 9)
    assert (profile.max() - profile.min()) / profile.mean() < 1e-6
