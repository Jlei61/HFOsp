import numpy as np
import pytest

from src.topic4_core_field import (
    SIGMA_S_FACTOR, axial_basis_centers, project_to_budget, uniform_axial_q)
from src.topic4_core_field_stage2 import (
    N_PARAMS, params_to_h, shape_of, uniform_theta)

M = 9


def _sheet(n=32_000, L=24.0, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, L, size=(n, 2))
    return pos[:, 0] - L / 2.0, pos[:, 1] - L / 2.0


def _geom():
    return dict(sep=13.32, s_support=(-10.7, 7.4), M=M, sigma_perp=1.5, shift_mm=3.0)


def test_parameter_vector_is_M_plus_one():
    assert N_PARAMS(M) == M + 1
    assert uniform_theta(M).shape == (M + 1,)


def test_uniform_theta_reproduces_the_uniform_axial_arm_exactly():
    """alpha = 0 and log rho = 0 must give back the Stage 1 uniform corridor, so
    the optimiser starts from a field whose behaviour is already measured."""
    s, r = _sheet()
    g = _geom()
    target = 1129.0
    h_learned = params_to_h(uniform_theta(M), s, r, g, target)
    kappa = axial_basis_centers(g["s_support"], M)
    q = uniform_axial_q(s, r, kappa, SIGMA_S_FACTOR * (kappa[1] - kappa[0]), g["sigma_perp"])
    h_arm, _ = project_to_budget(q, target)
    assert np.allclose(h_learned, h_arm, atol=1e-12)


def test_softmax_shift_invariance_leaves_the_field_untouched():
    """Adding a constant to every alpha is a no-op on the field; the search must
    not be able to wander along that redundant direction."""
    s, r = _sheet()
    rng = np.random.default_rng(3)
    theta = np.concatenate([rng.normal(size=M), [0.2]])
    shifted = theta.copy(); shifted[:M] += 7.5
    assert np.allclose(params_to_h(theta, s, r, _geom(), 1129.0),
                       params_to_h(shifted, s, r, _geom(), 1129.0))


def test_every_candidate_hits_the_budget():
    s, r = _sheet()
    rng = np.random.default_rng(5)
    for _ in range(5):
        theta = np.concatenate([rng.normal(scale=2.0, size=M), rng.normal(scale=0.5, size=1)])
        h = params_to_h(theta, s, r, _geom(), 1129.0)
        assert abs(h.sum() - 1129.0) / 1129.0 < 1e-6
        assert (h >= 0).all() and (h <= 1).all()


def test_rho_trades_axial_extent_against_transverse_extent():
    """rho is a FIXED-AREA aspect ratio: larger rho elongates along the axis and
    narrows across it (spec 4.1 / 4.4)."""
    s, r = _sheet()
    concentrated = np.zeros(M + 1); concentrated[M // 2] = 6.0
    wide = shape_of(np.append(concentrated[:M], np.log(0.5)), s, r, _geom(), 1129.0)
    narrow = shape_of(np.append(concentrated[:M], np.log(2.0)), s, r, _geom(), 1129.0)
    assert wide["rms_transverse"] > 2.0 * narrow["rms_transverse"]
    assert narrow["rms_axial"] > wide["rms_axial"]


def test_extreme_parameters_stay_finite():
    s, r = _sheet()
    for theta in (np.concatenate([np.full(M, 50.0), [3.0]]),
                  np.concatenate([np.full(M, -50.0), [-3.0]]),
                  np.concatenate([np.linspace(-40, 40, M), [0.0]])):
        h = params_to_h(theta, s, r, _geom(), 1129.0)
        assert np.isfinite(h).all()
        assert abs(h.sum() - 1129.0) / 1129.0 < 1e-6


def test_log_rho_is_clipped_so_the_field_cannot_degenerate():
    """An unbounded log rho would collapse the corridor to a line or a disc and
    make the budget projection meaningless."""
    s, r = _sheet()
    huge = params_to_h(np.concatenate([np.zeros(M), [50.0]]), s, r, _geom(), 1129.0)
    clipped = params_to_h(np.concatenate([np.zeros(M), [np.log(4.0)]]), s, r, _geom(), 1129.0)
    assert np.allclose(huge, clipped)
