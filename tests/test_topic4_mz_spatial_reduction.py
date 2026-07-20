from __future__ import annotations

import numpy as np
import pytest

from src.sef_hfo_field import convolve_periodic
from src.topic4_mz_spatial_reduction import (
    binary_block_average,
    canonical_m3b_core_surround,
)


def test_binary_block_average_preserves_constant_and_area_mass():
    reduction = canonical_m3b_core_surround()
    weights = reduction.kernels.weights()
    for matrix in (reduction.kernels.K_EE, reduction.kernels.K_I):
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(weights @ matrix, weights, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(matrix @ np.ones(2), np.ones(2), rtol=0.0, atol=1e-14)


def test_canonical_m3b_single_core_reduction_is_numerically_locked():
    reduction = canonical_m3b_core_surround()
    assert reduction.grid_n == 48
    assert reduction.grid_L_mm == 12.0
    assert reduction.grid_spacing_mm == 0.25
    assert reduction.core_cells == 113
    assert reduction.surround_cells == 2191
    assert reduction.ell_parallel_mm == 0.54
    assert reduction.ell_perpendicular_mm == 0.27
    assert reduction.inhibitory_width_mm == 0.25
    np.testing.assert_allclose(
        reduction.kernels.weights(),
        np.asarray([113.0 / 2304.0, 2191.0 / 2304.0]),
        rtol=0.0,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        reduction.kernels.K_EE,
        np.asarray(
            [
                [0.779680238653688, 0.220319761346312],
                [0.011362908732147, 0.988637091267853],
            ]
        ),
        rtol=0.0,
        atol=5e-15,
    )
    np.testing.assert_allclose(
        reduction.kernels.K_I,
        np.asarray(
            [
                [0.867570011856607, 0.132429988143393],
                [0.006830026773256, 0.993169973226744],
            ]
        ),
        rtol=0.0,
        atol=5e-15,
    )


def test_binary_projection_does_not_multiply_source_area_twice():
    kernel = np.full((2, 2), 0.25)
    core = np.asarray([[True, False], [False, False]])
    matrix = binary_block_average(kernel, core)
    lifted = np.where(core, 3.0, 7.0)
    convolved = convolve_periodic(lifted, kernel)
    projected = np.asarray([convolved[core].mean(), convolved[~core].mean()])
    np.testing.assert_allclose(matrix @ np.asarray([3.0, 7.0]), projected)


def test_binary_projection_rejects_empty_partition():
    with pytest.raises(ValueError, match="non-empty"):
        binary_block_average(np.full((2, 2), 0.25), np.ones((2, 2), dtype=bool))
