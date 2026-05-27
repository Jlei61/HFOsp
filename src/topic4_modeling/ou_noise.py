"""Ornstein-Uhlenbeck noise generator (numba JIT).

Exact discrete OU update:
    η[t+dt] = η[t] · exp(-dt/τ) + sigma · sqrt(1 - exp(-2 dt/τ)) · N(0,1)

so that stationary variance = sigma² regardless of dt.

Spec: docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md §5.3
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _ou_loop_jit(n_steps: int, decay: float, noise_scale: float,
                  randn: np.ndarray) -> np.ndarray:
    """numba hot loop: applies discrete OU update n_steps times."""
    eta = np.zeros(n_steps)
    for i in range(1, n_steps):
        eta[i] = eta[i - 1] * decay + noise_scale * randn[i]
    return eta


def generate_ou_noise(
    n_steps: int,
    dt: float,
    tau: float,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Generate OU noise trace with stationary variance sigma².

    Returns trace of shape (n_steps,) starting from eta_0 = 0.
    """
    if sigma == 0.0:
        return np.zeros(n_steps)
    rng = np.random.default_rng(seed)
    decay = float(np.exp(-dt / tau))
    noise_scale = float(sigma * np.sqrt(1.0 - decay**2))
    randn = rng.standard_normal(n_steps)
    return _ou_loop_jit(n_steps, decay, noise_scale, randn)
