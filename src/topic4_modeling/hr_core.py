"""Hindmarsh-Rose ODE core: params + rhs + RK4 step.

Provides both pure-Python reference and numba @njit JIT versions.
Hot loops (sweep / trajectory) call the _jit variants; tests verify
JIT and reference agree to numerical tolerance.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §5.1
"""

from __future__ import annotations

from dataclasses import dataclass

from numba import njit


# ── Parameters (immutable, hashable) ──────────────────────────────────────

@dataclass(frozen=True)
class HRParams:
    """Hindmarsh-Rose parameters. Defaults from spec §5.1 baseline."""
    a: float = 1.0
    b: float = 3.0
    c: float = 1.0
    d: float = 5.0
    r: float = 0.006
    s: float = 4.0
    x_R: float = -1.6


# ── Pure-Python reference (used by tests for ground-truth check) ─────────

def hr_rhs(
    x: float, y: float, z: float,
    params: HRParams,
    I: float, eta: float,
) -> tuple[float, float, float]:
    """HR ODE right-hand side (reference)."""
    p = params
    dx = y - p.a * x**3 + p.b * x**2 - z + I + eta
    dy = p.c - p.d * x**2 - y
    dz = p.r * (p.s * (x - p.x_R) - z)
    return dx, dy, dz


def rk4_step(
    state: tuple[float, float, float],
    params: HRParams,
    I: float, eta: float, dt: float,
) -> tuple[float, float, float]:
    """Classical RK4 step (reference). eta held constant over the step."""
    x, y, z = state
    k1x, k1y, k1z = hr_rhs(x, y, z, params, I, eta)
    k2x, k2y, k2z = hr_rhs(
        x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z,
        params, I, eta,
    )
    k3x, k3y, k3z = hr_rhs(
        x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z,
        params, I, eta,
    )
    k4x, k4y, k4z = hr_rhs(
        x + dt * k3x, y + dt * k3y, z + dt * k3z,
        params, I, eta,
    )
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    z_new = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return x_new, y_new, z_new


# ── numba JIT variants (called from hot loops) ───────────────────────────
#
# numba does not accept dataclass directly; params expanded to scalars.
# Cache=True bakes JIT to disk so first-call cost amortized across runs.

@njit(cache=True, fastmath=True)
def hr_rhs_jit(
    x, y, z,
    a, b, c, d, r, s, x_R,
    I, eta,
):
    """HR ODE rhs (numba JIT). Same math as hr_rhs."""
    dx = y - a * x**3 + b * x**2 - z + I + eta
    dy = c - d * x**2 - y
    dz = r * (s * (x - x_R) - z)
    return dx, dy, dz


@njit(cache=True, fastmath=True)
def rk4_step_jit(
    x, y, z,
    a, b, c, d, r, s, x_R,
    I, eta, dt,
):
    """RK4 step (numba JIT)."""
    k1x, k1y, k1z = hr_rhs_jit(x, y, z, a, b, c, d, r, s, x_R, I, eta)
    k2x, k2y, k2z = hr_rhs_jit(
        x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, z + 0.5 * dt * k1z,
        a, b, c, d, r, s, x_R, I, eta,
    )
    k3x, k3y, k3z = hr_rhs_jit(
        x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, z + 0.5 * dt * k2z,
        a, b, c, d, r, s, x_R, I, eta,
    )
    k4x, k4y, k4z = hr_rhs_jit(
        x + dt * k3x, y + dt * k3y, z + dt * k3z,
        a, b, c, d, r, s, x_R, I, eta,
    )
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y_new = y + (dt / 6.0) * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    z_new = z + (dt / 6.0) * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)
    return x_new, y_new, z_new
