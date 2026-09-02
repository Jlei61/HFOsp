import numpy as np

from src.topic5_latent_perturbation_v0_2 import (
    local_residual_normal_directions,
    residual_covariance_direction_sd,
)


def test_directional_sd_matches_full_covariance_with_diagonal_correction():
    components = np.eye(4)[:, :2]
    values = np.array([3.0, 2.0])
    diagonal = np.array([3.0, 2.0, 0.5, 0.25])
    direction = np.array([1.0, 1.0, 1.0, 1.0])
    covariance = components @ np.diag(values) @ components.T
    covariance += np.diag(diagonal - np.diag(covariance))
    unit = direction / np.linalg.norm(direction)
    expected = np.sqrt(unit @ covariance @ unit)
    actual = residual_covariance_direction_sd(direction, values, components, diagonal)
    assert np.isclose(actual, expected)


def test_local_normals_are_orthonormal_and_axis_normal():
    components = np.eye(12)
    progress = np.eye(12)[0]
    field = np.eye(12)[1]
    normals = local_residual_normal_directions(components, progress, field, 8)
    assert np.allclose(normals @ normals.T, np.eye(8))
    assert np.allclose(normals @ progress, 0.0)
    assert np.allclose(normals @ field, 0.0)
