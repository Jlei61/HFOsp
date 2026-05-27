"""Tests for src/topic4_modeling/ou_noise.py."""

from __future__ import annotations

import numpy as np
import pytest


def test_ou_noise_seed_reproducibility():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    t1 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    t2 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    np.testing.assert_array_equal(t1, t2)


def test_ou_noise_different_seeds_diverge():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    t1 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=42)
    t2 = generate_ou_noise(n_steps=1000, dt=0.05, tau=10.0, sigma=0.1, seed=43)
    assert not np.array_equal(t1, t2)


def test_ou_noise_stationary_variance():
    """Long sample variance ≈ sigma²."""
    from src.topic4_modeling.ou_noise import generate_ou_noise
    trace = generate_ou_noise(
        n_steps=200_000, dt=0.05, tau=10.0, sigma=0.1, seed=0
    )
    burn_in = 5000
    assert np.var(trace[burn_in:]) == pytest.approx(0.1**2, rel=0.15)


def test_ou_noise_autocorrelation_at_lag_tau_is_inv_e():
    """Autocorrelation at lag = tau ≈ 1/e."""
    from src.topic4_modeling.ou_noise import generate_ou_noise
    tau, dt = 10.0, 0.05
    trace = generate_ou_noise(n_steps=400_000, dt=dt, tau=tau, sigma=0.2, seed=1)
    burn_in = 5000
    x = trace[burn_in:] - trace[burn_in:].mean()
    var0 = (x * x).mean()
    lag = int(tau / dt)
    autocorr = (x[:-lag] * x[lag:]).mean() / var0
    assert autocorr == pytest.approx(np.exp(-1.0), abs=0.05)


def test_ou_noise_length_exact():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    trace = generate_ou_noise(n_steps=137, dt=0.05, tau=10.0, sigma=0.1, seed=0)
    assert trace.shape == (137,)


def test_ou_noise_zero_sigma_returns_zero_trace():
    from src.topic4_modeling.ou_noise import generate_ou_noise
    trace = generate_ou_noise(n_steps=100, dt=0.05, tau=10.0, sigma=0.0, seed=0)
    np.testing.assert_array_equal(trace, np.zeros(100))
