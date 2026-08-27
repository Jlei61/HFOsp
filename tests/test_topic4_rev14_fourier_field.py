import inspect

import numpy as np
import pytest

from src.topic4_rev14_fourier_field import (
    analytic_roughness,
    array_sha256,
    canonical_mode,
    evaluate,
    fourier_basis,
    mode_inventory,
    mode_shell,
    normalize_shell_rms,
    spectral_roughness_surrogate,
    uniform_quadrature,
    wavevectors,
    weighted_centered_surface_rms,
    weighted_surface_rms,
)


def test_frozen_m3_m4_inventory_counts_and_dimensions():
    m3 = mode_inventory(3)
    m4 = mode_inventory(4)
    shell = mode_shell(3, 4)

    assert len(m3) == 14
    assert len(m4) == 24
    assert len(shell) == 10
    assert 2 * len(m3) == 28
    assert 2 * len(m4) == 48
    assert 2 * len(shell) == 20
    assert tuple(mode for mode in m4 if mode not in set(m3)) == shell


def test_inventory_has_exactly_one_representative_of_each_plus_minus_pair():
    modes = mode_inventory(4)
    mode_set = set(modes)
    assert (0, 0) not in mode_set
    assert all(nx > 0 or (nx == 0 and ny > 0) for nx, ny in modes)
    assert len(mode_set) == len(modes)
    assert all((-nx, -ny) not in mode_set for nx, ny in modes)
    assert all(nx * nx + ny * ny <= 16 for nx, ny in modes)


def test_inventory_is_closed_under_square_rotations_and_reflections_after_pairing():
    modes = mode_inventory(4)
    mode_set = set(modes)
    transforms = (
        lambda x, y: (x, y),
        lambda x, y: (-y, x),
        lambda x, y: (-x, -y),
        lambda x, y: (y, -x),
        lambda x, y: (-x, y),
        lambda x, y: (x, -y),
        lambda x, y: (y, x),
        lambda x, y: (-y, -x),
    )
    for transform in transforms:
        image = {canonical_mode(*transform(*mode)) for mode in modes}
        assert image == mode_set


def test_basis_has_two_L_period_but_does_not_force_opposite_boundaries_equal():
    modes = ((1, 0),)
    coefficients = np.array([[1.0, 0.0]])
    points = np.array([[0.0, 7.0], [20.0, 7.0], [40.0, 7.0]])
    values = evaluate(coefficients, points, modes, L=20.0)

    assert values[0] == pytest.approx(values[2], abs=1e-14)
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(-1.0)
    assert values[0] != pytest.approx(values[1])


def test_basis_interleaves_cosine_and_sine_for_each_mode():
    modes = ((0, 1), (1, -1))
    points = np.array([[0.0, 0.0], [10.0, 5.0]])
    basis = fourier_basis(points, modes, L=20.0)
    k = wavevectors(modes, L=20.0)
    phase = points @ k.T

    assert basis.shape == (2, 4)
    assert np.allclose(basis[:, 0::2], np.cos(phase))
    assert np.allclose(basis[:, 1::2], np.sin(phase))


@pytest.mark.parametrize("modes", [mode_inventory(3), mode_shell(3, 4)])
def test_shell_normalization_reaches_requested_centered_surface_rms(modes):
    rng = np.random.default_rng(811)
    coefficients = rng.normal(size=(len(modes), 2))
    normalized = normalize_shell_rms(
        coefficients, modes, target_rms=1.7, n_per_axis=192, L=20.0,
    )
    points, weights = uniform_quadrature(192, L=20.0)
    rms = weighted_centered_surface_rms(
        evaluate(normalized, points, modes), weights,
    )

    assert rms == pytest.approx(1.7, rel=2e-13, abs=2e-13)
    ratios = normalized / coefficients
    assert np.allclose(ratios, ratios.flat[0])


def test_cosine_and_sine_phase_receive_equal_centered_dose():
    modes = ((1, 0),)
    cosine = normalize_shell_rms(
        np.asarray([[1.0, 0.0]]), modes, target_rms=1.0,
        n_per_axis=256, L=20.0,
    )
    sine = normalize_shell_rms(
        np.asarray([[0.0, 1.0]]), modes, target_rms=1.0,
        n_per_axis=256, L=20.0,
    )
    points, weights = uniform_quadrature(256, L=20.0)
    for coefficients in (cosine, sine):
        values = evaluate(coefficients, points, modes, L=20.0)
        assert weighted_centered_surface_rms(values, weights) == pytest.approx(
            1.0, rel=2e-13, abs=2e-13,
        )


def test_centered_rms_is_invariant_to_additive_gauge():
    values = np.asarray([-2.0, 0.0, 3.0, 5.0])
    weights = np.asarray([1.0, 2.0, 1.0, 4.0])
    assert weighted_centered_surface_rms(values, weights) == pytest.approx(
        weighted_centered_surface_rms(values + 17.0, weights),
    )


def test_analytic_roughness_matches_frozen_formula_and_phase_symmetry():
    modes = ((1, 0), (1, 2))
    coefficients = np.array([[3.0, 4.0], [5.0, 12.0]])
    k = wavevectors(modes, L=20.0)
    expected = np.sum(
        np.sum(k * k, axis=1) ** 2 * np.sum(coefficients ** 2, axis=1)
    )

    assert analytic_roughness(coefficients, modes, L=20.0) == pytest.approx(expected)
    assert spectral_roughness_surrogate(
        coefficients, modes, L=20.0,
    ) == pytest.approx(expected)
    assert analytic_roughness(coefficients[:, ::-1], modes, L=20.0) == pytest.approx(expected)


def test_uniform_quadrature_covers_sheet_with_equal_area_weights():
    points, weights = uniform_quadrature(8, L=20.0)
    assert points.shape == (64, 2)
    assert weights.shape == (64,)
    assert np.all(points > 0.0)
    assert np.all(points < 20.0)
    assert np.unique(weights).size == 1
    assert weights.sum() == pytest.approx(400.0)


def test_inventory_basis_and_hash_are_deterministic():
    left_modes = np.asarray(mode_inventory(4), dtype=np.int64)
    right_modes = np.asarray(mode_inventory(4), dtype=np.int64)
    points, _ = uniform_quadrature(17)
    left_basis = fourier_basis(points, left_modes)
    right_basis = fourier_basis(points.copy(), right_modes.copy())

    assert np.array_equal(left_modes, right_modes)
    assert array_sha256(left_modes) == array_sha256(right_modes)
    assert array_sha256(left_basis) == array_sha256(right_basis)
    assert array_sha256(np.zeros((2, 3))) != array_sha256(np.zeros((3, 2)))


def test_public_field_api_cannot_receive_observation_geometry():
    forbidden = {"contacts", "contact_xy", "shaft_ids", "electrodes", "onsets"}
    for function in (
        mode_inventory, mode_shell, wavevectors, fourier_basis, evaluate,
        uniform_quadrature, normalize_shell_rms, analytic_roughness,
        spectral_roughness_surrogate,
    ):
        assert forbidden.isdisjoint(inspect.signature(function).parameters)


@pytest.mark.parametrize(
    ("call", "error"),
    [
        (lambda: mode_inventory(0), ValueError),
        (lambda: mode_inventory(3.0), TypeError),
        (lambda: mode_shell(4, 3), ValueError),
        (lambda: canonical_mode(0, 0), ValueError),
        (lambda: canonical_mode(1.5, 0), TypeError),
        (lambda: wavevectors(((1, 0), (-1, 0))), ValueError),
        (lambda: wavevectors(((1, 0), (1, 0))), ValueError),
        (lambda: wavevectors(((1, 0),), L=0.0), ValueError),
        (lambda: fourier_basis(np.zeros(3), ((1, 0),)), ValueError),
        (lambda: fourier_basis(np.array([[np.nan, 0.0]]), ((1, 0),)), ValueError),
        (lambda: evaluate(np.zeros((2, 2)), np.zeros((1, 2)), ((1, 0),)), ValueError),
        (lambda: evaluate(np.array([[np.nan, 0.0]]), np.zeros((1, 2)), ((1, 0),)), ValueError),
        (lambda: uniform_quadrature(1), ValueError),
        (lambda: uniform_quadrature(8, L=np.inf), ValueError),
        (lambda: weighted_surface_rms(np.ones(2), np.ones(3)), ValueError),
        (lambda: weighted_surface_rms(np.ones(2), np.array([1.0, 0.0])), ValueError),
        (lambda: weighted_centered_surface_rms(np.ones(2), np.ones(3)), ValueError),
        (lambda: normalize_shell_rms(np.zeros((1, 2)), ((1, 0),)), ValueError),
        (lambda: normalize_shell_rms(np.ones((1, 2)), ((1, 0),), target_rms=0.0), ValueError),
        (lambda: analytic_roughness(np.zeros((2, 2)), ((1, 0),)), ValueError),
        (lambda: array_sha256(np.array([1.0 + 2.0j])), TypeError),
        (lambda: array_sha256(np.array(["not numeric"])), TypeError),
    ],
)
def test_invalid_inputs_fail_closed(call, error):
    with pytest.raises(error):
        call()
