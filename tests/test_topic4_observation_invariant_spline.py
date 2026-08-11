import inspect

import numpy as np

from src.topic4_continuous_field import continuous_surface, tensor_basis
from src.topic4_observation_invariant_spline import (
    allocation_direction,
    fit_uniform_surface,
    sample_smooth_residual_pairs,
    uniform_allocation_centers,
)
from src.topic4_spectral_field import uniform_sheet_grid


def test_uniform_spline_coordinates_are_stable_and_observation_free():
    from scripts.freeze_topic4_rev10_sa_spline_field_v4_candidates import (
        build_candidates,
    )

    grid = uniform_sheet_grid(32, L=20.0)
    basis = tensor_basis(grid, 10, degree=3, L=20.0)
    condition = np.linalg.cond(basis)
    forbidden = {"contacts", "contact_xy", "shaft_ids", "onsets", "labels"}

    assert condition < 100.0
    for function in (
        uniform_allocation_centers, fit_uniform_surface,
        allocation_direction, sample_smooth_residual_pairs, build_candidates,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


def test_smooth_random_pairs_are_antithetic_at_frozen_rms():
    grid = uniform_sheet_grid(32, L=20.0)
    pairs = sample_smooth_residual_pairs(
        n_pairs=2, n_basis=10, seed=3, rms_amplitudes=[2.0, 3.0],
        positions=grid, smoothing_controls=1.0, degree=3, L=20.0,
    )
    for pair, expected in zip(pairs, (2.0, 3.0)):
        positive = continuous_surface(
            pair["positive"], grid, n_basis=10, degree=3, L=20.0,
        )
        negative = continuous_surface(
            pair["negative"], grid, n_basis=10, degree=3, L=20.0,
        )
        assert np.allclose(positive, -negative)
        assert np.isclose(
            np.sqrt(np.mean((positive - positive.mean()) ** 2)), expected,
        )
