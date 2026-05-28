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
    assert c.envelope_gap == 30.0  # Stage 1b burst-envelope grouping valley


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


# ── detect_burst_envelopes (Stage 1b burst-envelope unit) ────────────────
#
# Algorithm tests: spike atoms come from detect_bursts, then merge by
# envelope_gap. Use an EXPLICIT BurstConfig at the synthetic-pulse scale
# (same _ALG_CFG_KWARGS decoupling as detect_bursts tests above) + an
# explicit envelope_gap, so these pin the merge ALGORITHM independent of the
# production tuning constants. Contract clauses (Stage 1b plan §contract):
#   #1 reuse detect_bursts for spike atoms   #2 merge by gap < envelope_gap
#   #3 onset = first spike START             #4 secondary fields
#   #5 envelope_gap configurable
from dataclasses import replace as _replace


def _env_cfg(envelope_gap: float):
    from src.topic4_modeling.hr_config import BurstConfig
    return BurstConfig(envelope_gap=envelope_gap, **_ALG_CFG_KWARGS)


def test_burst_envelopes_empty_on_silent():
    """Clause: silent trace → no envelopes."""
    from src.topic4_modeling.hr_dynamics import detect_burst_envelopes
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    assert detect_burst_envelopes(x, t, _env_cfg(10.0)) == []


def test_burst_envelopes_single_spike_is_one_envelope_n1():
    """Clause #1+#4: one isolated spike → 1 envelope, n_spikes == 1,
    onset = spike start."""
    from src.topic4_modeling.hr_dynamics import detect_burst_envelopes
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    envs = detect_burst_envelopes(x, t, _env_cfg(10.0))
    assert len(envs) == 1
    assert envs[0].n_spikes == 1
    assert envs[0].onset == pytest.approx(10.0, abs=0.1)


def test_burst_envelopes_merges_close_spikes():
    """Clause #2+#3: two spikes with gap < envelope_gap → 1 envelope spanning
    first-spike-start to last-spike-end, n_spikes == 2."""
    from src.topic4_modeling.hr_dynamics import detect_burst_envelopes
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5   # spike 1
    x[(t >= 25.0) & (t <= 35.0)] = 1.5   # spike 2; inter-spike gap = 5
    # gap 5: > bridge_gap(2) so detect_bursts sees 2 spikes; < envelope_gap(10)
    envs = detect_burst_envelopes(x, t, _env_cfg(10.0))
    assert len(envs) == 1
    assert envs[0].n_spikes == 2
    assert envs[0].onset == pytest.approx(10.0, abs=0.1)
    assert envs[0].offset == pytest.approx(35.0, abs=0.1)


def test_burst_envelopes_separates_far_spikes():
    """Clause #2: two spikes with gap > envelope_gap → 2 envelopes."""
    from src.topic4_modeling.hr_dynamics import detect_burst_envelopes
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    x[(t >= 40.0) & (t <= 50.0)] = 1.5   # inter-spike gap = 20 > envelope_gap 10
    envs = detect_burst_envelopes(x, t, _env_cfg(10.0))
    assert len(envs) == 2
    assert all(e.n_spikes == 1 for e in envs)


def test_burst_envelopes_onset_is_first_spike_start_not_peak():
    """Clause #3 (load-bearing): onset = first spike START, NOT the time of the
    peak. Build a spike whose peak sample is late in the window."""
    from src.topic4_modeling.hr_dynamics import detect_burst_envelopes
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    x[(t >= 18.0) & (t <= 20.0)] = 3.0   # peak occurs at t≈18-20, well after onset
    envs = detect_burst_envelopes(x, t, _env_cfg(10.0))
    assert len(envs) == 1
    assert envs[0].onset == pytest.approx(10.0, abs=0.1)   # start, not peak time
    assert envs[0].peak_x == pytest.approx(3.0, abs=1e-9)


def test_burst_envelopes_peak_x_and_duration():
    """Clause #4: peak_x = max x over the envelope window; duration =
    offset - onset (across a merged 2-spike envelope)."""
    from src.topic4_modeling.hr_dynamics import detect_burst_envelopes
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    x[(t >= 25.0) & (t <= 35.0)] = 2.5   # higher peak in 2nd spike; gap 5 < 10
    envs = detect_burst_envelopes(x, t, _env_cfg(10.0))
    assert len(envs) == 1
    assert envs[0].peak_x == pytest.approx(2.5, abs=1e-9)
    assert envs[0].duration == pytest.approx(envs[0].offset - envs[0].onset)
    assert envs[0].duration == pytest.approx(25.0, abs=0.2)  # 35 - 10


def test_burst_envelopes_gap_equal_to_envelope_gap_not_merged():
    """Clause #2 boundary: merge uses strict <, so gap == envelope_gap splits."""
    from src.topic4_modeling.hr_dynamics import detect_burst_envelopes
    t = np.arange(0, 100, 0.05)
    x = np.full_like(t, -1.6)
    x[(t >= 10.0) & (t <= 20.0)] = 1.5
    x[(t >= 30.0) & (t <= 40.0)] = 1.5   # gap = 30 - 20 = 10 == envelope_gap
    envs = detect_burst_envelopes(x, t, _env_cfg(10.0))
    assert len(envs) == 2


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
