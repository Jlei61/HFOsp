"""Tests for src/topic4_modeling/hr_dynamics.py + hr_config.py."""

from __future__ import annotations

import numpy as np
import pytest


# ── hr_config ────────────────────────────────────────────────────────────

def test_burst_thresholds_defaults():
    """BurstConfig defaults = spike-level (recalibrated 2026-05-28 to HR's
    measured fast-spike timescale; old 1.0/5.0/2.0 made detect_bursts
    return 0 across the whole grid — see hr_config.py docstring)."""
    from src.topic4_modeling.hr_config import BurstConfig
    c = BurstConfig()
    assert c.x_threshold == 0.0
    assert c.min_burst_duration == 0.3
    assert c.bridge_gap == 1.0


def test_regime_thresholds_defaults():
    """RegimeConfig has defaults matching spec §3 stage 1."""
    from src.topic4_modeling.hr_config import RegimeConfig
    c = RegimeConfig()
    assert c.max_burst_duration == 100.0
    assert c.excitable_max_burst == 50.0
    assert c.excitable_min_ibi == 30.0


# ── simulate_trajectory ──────────────────────────────────────────────────

def test_simulate_trajectory_shapes():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    t, traj = simulate_trajectory(p, I=-1.6, T=10.0, dt=0.05,
                                   sigma_ou=0.0, tau_ou=10.0, seed=0)
    n = int(10.0 / 0.05)
    assert t.shape == (n,)
    assert traj.shape == (n, 3)


def test_simulate_trajectory_default_burn_in_is_zero():
    """Without explicit burn_in, behavior is unchanged (back-compat)."""
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    t_ref, traj_ref = simulate_trajectory(p, I=-1.6, T=10.0, dt=0.05,
                                            sigma_ou=0.0, tau_ou=10.0, seed=0)
    t_burn0, traj_burn0 = simulate_trajectory(p, I=-1.6, T=10.0, dt=0.05,
                                                sigma_ou=0.0, tau_ou=10.0, seed=0,
                                                burn_in=0.0)
    np.testing.assert_array_equal(t_ref, t_burn0)
    np.testing.assert_array_equal(traj_ref, traj_burn0)


def test_simulate_trajectory_burn_in_discards_initial_steps():
    """burn_in=B slices off first B time units; returned shape stays int(T/dt).

    Verified by: sim(T=10, burn_in=5, seed=S) returns the same trajectory
    as sim(T=15, burn_in=0, seed=S) starting from index 5/dt = 100.
    """
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    t_burn, traj_burn = simulate_trajectory(
        p, I=2.0, T=10.0, dt=0.05,
        sigma_ou=0.1, tau_ou=10.0, seed=0, burn_in=5.0,
    )
    t_full, traj_full = simulate_trajectory(
        p, I=2.0, T=15.0, dt=0.05,
        sigma_ou=0.1, tau_ou=10.0, seed=0, burn_in=0.0,
    )
    # Returned shape: still 10/0.05 = 200 steps
    assert traj_burn.shape == (200, 3)
    # Returned trajectory = tail of the full trajectory after burn_in
    np.testing.assert_array_equal(traj_burn, traj_full[100:])
    # Returned t starts at 0 (re-zeroed after burn-in)
    assert t_burn[0] == 0.0


def test_simulate_trajectory_reproducibility():
    from src.topic4_modeling.hr_core import HRParams
    from src.topic4_modeling.hr_dynamics import simulate_trajectory
    p = HRParams()
    _, a = simulate_trajectory(p, I=-1.6, T=50.0, dt=0.05,
                                sigma_ou=0.1, tau_ou=10.0, seed=42)
    _, b = simulate_trajectory(p, I=-1.6, T=50.0, dt=0.05,
                                sigma_ou=0.1, tau_ou=10.0, seed=42)
    np.testing.assert_array_equal(a, b)


# (Empirical regime-behavior tests "silent at I=-3.0" / "≥3 bursts at I=2.0"
#  moved to tests/test_topic4_modeling_hr_dynamics_integration.py per
#  v3 user-return critique — not algebraic, would invite model tuning.)


# ── detect_bursts ────────────────────────────────────────────────────────
#
# These are ALGORITHM tests (bridging + min-duration filtering on synthetic
# pulses). They pin an EXPLICIT BurstConfig matching the synthetic-pulse
# scale (x_threshold=1.0, min_burst_duration=5.0, bridge_gap=2.0) so they
# stay valid regardless of the production default — which was recalibrated
# 2026-05-28 to HR's real spike timescale (x_threshold=0.0, min=0.3,
# bridge=1.0). Decoupling algorithm correctness from the production tuning
# constant is the right test hygiene.

_ALG_CFG_KWARGS = dict(x_threshold=1.0, min_burst_duration=5.0, bridge_gap=2.0)


def test_detect_bursts_zero_on_silent_trace():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    assert detect_bursts(x, t, BurstConfig(**_ALG_CFG_KWARGS)) == []


def test_detect_bursts_one_pulse():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    bursts = detect_bursts(x, t, BurstConfig(**_ALG_CFG_KWARGS))
    assert len(bursts) == 1
    assert bursts[0][0] == pytest.approx(10.0, abs=0.1)


def test_detect_bursts_hysteresis_bridges_short_dip():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 30.0)] = 1.5
    x[(t >= 19.5) & (t <= 20.5)] = 0.5  # 1-unit dip < bridge_gap=2.0
    assert len(detect_bursts(x, t, BurstConfig(**_ALG_CFG_KWARGS))) == 1


def test_detect_bursts_three_separated():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 200, 0.05)
    x = np.full_like(t, -1.6)
    for t0 in [10.0, 60.0, 120.0]:
        x[(t >= t0) & (t <= t0 + 10.0)] = 1.5
    assert len(detect_bursts(x, t, BurstConfig(**_ALG_CFG_KWARGS))) == 3


def test_detect_bursts_rejects_short_noise_spike():
    from src.topic4_modeling.hr_dynamics import detect_bursts
    from src.topic4_modeling.hr_config import BurstConfig
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 11.0)] = 1.5  # 1-unit < min_burst_duration=5.0
    assert detect_bursts(x, t, BurstConfig(**_ALG_CFG_KWARGS)) == []


# ── classify_regime ─────────────────────────────────────────────────────

def test_classify_regime_silent():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    assert classify_regime([], T=1000.0, cfg=RegimeConfig()) == "silent"


def test_classify_regime_excitable_sparse_short():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    bursts = [(100.0, 110.0), (300.0, 310.0), (700.0, 710.0)]
    assert classify_regime(bursts, T=1000.0, cfg=RegimeConfig()) == "excitable"


def test_classify_regime_repetitive_burst():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    bursts = [(t, t + 5.0) for t in range(50, 500, 20)]  # IBI=15<30
    assert classify_regime(bursts, T=1000.0, cfg=RegimeConfig()) == "repetitive-burst"


def test_classify_regime_unstable_long_burst():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    assert classify_regime([(50.0, 200.0)], T=1000.0, cfg=RegimeConfig()) == "unstable"


def test_classify_regime_unstable_takes_precedence():
    from src.topic4_modeling.hr_dynamics import classify_regime
    from src.topic4_modeling.hr_config import RegimeConfig
    bursts = [(50.0, 55.0), (60.0, 65.0), (70.0, 200.0)]
    assert classify_regime(bursts, T=1000.0, cfg=RegimeConfig()) == "unstable"
