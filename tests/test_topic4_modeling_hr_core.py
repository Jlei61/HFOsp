"""Tests for src/topic4_modeling/hr_core.py (HRParams + hr_rhs + RK4)."""

from __future__ import annotations

import numpy as np
import pytest


# ── HRParams ─────────────────────────────────────────────────────────────

def test_hr_params_defaults_match_spec():
    """Default HRParams match spec §5.1 baseline."""
    from src.topic4_modeling.hr_core import HRParams
    p = HRParams()
    assert p.a == 1.0 and p.b == 3.0 and p.c == 1.0 and p.d == 5.0
    assert p.r == 0.006 and p.s == 4.0 and p.x_R == -1.6


def test_hr_params_is_frozen():
    """HRParams is frozen (hashable for caching, immutable for safety)."""
    from src.topic4_modeling.hr_core import HRParams
    p = HRParams()
    with pytest.raises((AttributeError, Exception)):
        p.a = 2.0  # type: ignore[misc]


# ── hr_rhs algebraic invariants ──────────────────────────────────────────

def test_hr_rhs_returns_finite_3vec():
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    dx, dy, dz = hr_rhs(0.0, 0.0, 0.0, p, I=-1.6, eta=0.0)
    assert np.isfinite(dx) and np.isfinite(dy) and np.isfinite(dz)


def test_hr_rhs_y_eq_at_known_point():
    """At arbitrary x, y: dy/dt = c - d*x² - y (algebraic identity)."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    x_test, y_test = 0.5, -3.0
    _, dy, _ = hr_rhs(x_test, y_test, 0.0, p, I=0.0, eta=0.0)
    expected = p.c - p.d * x_test**2 - y_test
    assert dy == pytest.approx(expected)


def test_hr_rhs_z_eq_at_x_R_zero_z_yields_zero():
    """At x = x_R and z = 0: dz/dt = r * (s * 0 - 0) = 0."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    _, _, dz = hr_rhs(p.x_R, 0.0, 0.0, p, I=0.0, eta=0.0)
    assert dz == pytest.approx(0.0)


def test_hr_rhs_eta_linear_only_in_dx():
    """eta enters dx/dt linearly, doesn't enter dy or dz."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs
    p = HRParams()
    dx0, dy0, dz0 = hr_rhs(0.0, 0.0, 0.0, p, I=0.0, eta=0.0)
    dx1, dy1, dz1 = hr_rhs(0.0, 0.0, 0.0, p, I=0.0, eta=0.5)
    assert dx1 - dx0 == pytest.approx(0.5)
    assert dy1 == pytest.approx(dy0)
    assert dz1 == pytest.approx(dz0)


# ── RK4 step ─────────────────────────────────────────────────────────────

def test_rk4_step_deterministic_no_noise():
    """RK4 step with eta=0 is fully deterministic."""
    from src.topic4_modeling.hr_core import HRParams, rk4_step
    p = HRParams()
    s0 = (0.0, 0.0, 0.0)
    out1 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    out2 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    assert out1 == out2


def test_rk4_step_returns_finite_3tuple():
    from src.topic4_modeling.hr_core import HRParams, rk4_step
    p = HRParams()
    x, y, z = rk4_step((0.0, 0.0, 0.0), p, I=-1.6, eta=0.0, dt=0.05)
    assert np.isfinite(x) and np.isfinite(y) and np.isfinite(z)


def test_rk4_step_matches_euler_at_tiny_dt():
    """As dt → 0, RK4 step ≈ Euler step within O(dt²)."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs, rk4_step
    p = HRParams()
    s0 = (-1.0, -5.0, 0.5)
    dt = 1e-4
    x_rk4, y_rk4, z_rk4 = rk4_step(s0, p, I=-1.6, eta=0.0, dt=dt)
    dx, dy, dz = hr_rhs(*s0, p, I=-1.6, eta=0.0)
    np.testing.assert_allclose(
        [x_rk4, y_rk4, z_rk4],
        [s0[0] + dt * dx, s0[1] + dt * dy, s0[2] + dt * dz],
        atol=1e-6,
    )


# ── numba JIT smoke ──────────────────────────────────────────────────────

def test_hr_rhs_jit_matches_python_version():
    """Numba-JIT-compiled hr_rhs gives same answer as Python ref (if exposed)."""
    from src.topic4_modeling.hr_core import HRParams, hr_rhs, hr_rhs_jit
    p = HRParams()
    for x in [-1.5, 0.0, 1.0]:
        for y in [-5.0, 0.0, 2.0]:
            ref = hr_rhs(x, y, 0.5, p, I=-1.0, eta=0.1)
            # JIT version takes tuple of params (frozen dataclass → tuple unpack)
            jit_out = hr_rhs_jit(x, y, 0.5,
                                 p.a, p.b, p.c, p.d, p.r, p.s, p.x_R,
                                 I=-1.0, eta=0.1)
            np.testing.assert_allclose(ref, jit_out, rtol=1e-10)


def test_rk4_step_jit_matches_python_version():
    """JIT RK4 matches Python RK4 within 1e-10."""
    from src.topic4_modeling.hr_core import HRParams, rk4_step, rk4_step_jit
    p = HRParams()
    s0 = (-1.0, -5.0, 0.5)
    ref = rk4_step(s0, p, I=-1.6, eta=0.0, dt=0.05)
    jit_out = rk4_step_jit(s0[0], s0[1], s0[2],
                            p.a, p.b, p.c, p.d, p.r, p.s, p.x_R,
                            I=-1.6, eta=0.0, dt=0.05)
    np.testing.assert_allclose(ref, jit_out, rtol=1e-10)
