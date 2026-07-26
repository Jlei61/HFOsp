"""FCXR pump-lifecycle Task 1 TDD — dimensionless activity-dependent load + electrogenic pump.

Contract source: docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md §2
(+ plan §3 test list). u_i is an ACTIVITY-DEPENDENT INTRACELLULAR LOAD (Na/pump-inspired), NOT a
sodium concentration, ATP model or ionic-homeostasis model. Each test = one contract clause; the
clauses that would silently corrupt the science if violated are:

  * spike jump is per-spike (NOT multiplied by dt) -- a dt-scaled jump makes a_load dt-dependent;
  * clearance IS multiplied by dt/tau_N -- the two sides of the mass balance have different dt laws;
  * the SAME phi(u) drives clearance and the membrane current -- two phis = free extra parameter;
  * excess current is Imax*(phi-p0) with NO positive part -- rectification injects a positive mean
    bias from baseline noise alone and destroys local smoothness for the response operator.
"""
import inspect
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.topic4_mz_fcxr_pump as PUMP  # noqa: E402


# ============================== clause 1/2: phi(u) = u^h/(1+u^h) ==============================
def test_pump_activation_is_hill_h3_exact():
    u = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
    got = PUMP.pump_activation(u)
    assert np.allclose(got, u ** 3 / (1.0 + u ** 3))


def test_pump_activation_zero_load_gives_zero_activation():
    assert PUMP.pump_activation(np.array([0.0]))[0] == 0.0
    assert PUMP.pump_activation(0.0) == 0.0


def test_pump_activation_is_monotone_bounded_finite_and_smooth():
    u = np.linspace(0.0, 50.0, 5001)
    phi = PUMP.pump_activation(u)
    assert np.all(np.isfinite(phi))
    assert np.all(phi >= 0.0) and np.all(phi <= 1.0)
    assert np.all(np.diff(phi) >= 0.0)                      # monotone non-decreasing
    # smooth: no kink -- second difference stays bounded and phi'(0)=0 for h=3 (unlike a rectifier)
    d2 = np.diff(phi, 2)
    assert np.max(np.abs(d2)) < 1e-3
    assert phi[1] / (u[1] ** 3) == pytest.approx(1.0, rel=1e-6)   # phi ~ u^3 near 0


def test_pump_activation_saturates_toward_one_but_never_reaches_it():
    assert PUMP.pump_activation(1.0) == pytest.approx(0.5)
    assert 0.99 < PUMP.pump_activation(100.0) < 1.0


# ============================== clause 3/4: dt laws differ per term ==============================
def test_spike_jump_is_per_spike_and_not_scaled_by_dt():
    """a_load is a per-spike jump. If it were multiplied by dt, halving dt would halve the jump."""
    u0 = np.zeros(3)
    spk = np.array([True, False, True])
    kw = dict(a_load=0.3, tau_N=1000.0, dt=0.05)
    u1 = PUMP.step_spike_load(u0, spk, **kw)
    kw2 = dict(kw, dt=0.025)
    u1_half = PUMP.step_spike_load(u0, spk, **kw2)
    assert u1[0] == pytest.approx(0.3) and u1[2] == pytest.approx(0.3)
    assert u1[1] == 0.0
    assert u1_half[0] == pytest.approx(u1[0])               # dt does NOT enter the jump


def test_clearance_is_scaled_by_dt_over_tau_N_using_pre_step_phi():
    u0 = np.array([2.0])
    no_spk = np.array([False])
    dt, tau = 0.05, 1000.0
    u1 = PUMP.step_spike_load(u0, no_spk, a_load=0.3, tau_N=tau, dt=dt)
    expected = 2.0 - (dt / tau) * PUMP.pump_activation(2.0)
    assert u1[0] == pytest.approx(expected, rel=1e-12)
    # halving dt halves the clearance increment
    u1h = PUMP.step_spike_load(u0, no_spk, a_load=0.3, tau_N=tau, dt=dt / 2)
    assert (u0[0] - u1h[0]) == pytest.approx((u0[0] - u1[0]) / 2, rel=1e-12)


def test_clearance_uses_pre_step_u_not_post_jump_u():
    """Causal order (spec §2.2): clearance is evaluated at u(t^-), the jump is added on top."""
    u0 = np.array([2.0])
    spk = np.array([True])
    dt, tau, a = 0.05, 1000.0, 0.4
    got = PUMP.step_spike_load(u0, spk, a_load=a, tau_N=tau, dt=dt)[0]
    assert got == pytest.approx(2.0 + a - (dt / tau) * PUMP.pump_activation(2.0), rel=1e-12)
    wrong = 2.0 + a - (dt / tau) * PUMP.pump_activation(2.0 + a)     # post-jump clearance
    assert abs(got - wrong) > 0.0


# ============================== clause 5/6: non-negativity + monotone clearing ==============================
def test_update_never_returns_negative_load():
    u0 = np.array([1e-9, 0.0])
    u1 = PUMP.step_spike_load(u0, np.array([False, False]), a_load=0.0, tau_N=1e-6, dt=1.0)
    assert np.all(u1 >= 0.0)


def test_zero_spike_state_clears_monotonically_toward_zero():
    u = np.array([3.0])
    prev = np.inf
    for _ in range(4000):
        u = PUMP.step_spike_load(u, np.array([False]), a_load=0.5, tau_N=50.0, dt=0.05)
        assert u[0] <= prev + 1e-15
        prev = u[0]
    assert u[0] < 3.0


def test_spike_counts_accumulate_linearly_in_the_jump():
    """N_i^spike may exceed 1 (integer count input): the jump must scale with the count."""
    u0 = np.zeros(2)
    counts = np.array([0, 3])
    u1 = PUMP.step_spike_load(u0, counts, a_load=0.2, tau_N=1000.0, dt=0.05)
    assert u1[1] == pytest.approx(0.6)
    assert u1[0] == 0.0


def test_non_finite_load_fails_fast():
    with pytest.raises(FloatingPointError):
        PUMP.step_spike_load(np.array([np.inf]), np.array([False]),
                             a_load=0.1, tau_N=1000.0, dt=0.05)


# ============================== clause 7: one phi for clearance AND membrane ==============================
def test_same_phi_drives_clearance_and_membrane_current():
    """The clearance term removed from u and the membrane pump activation must be the same phi(u).
    A second, differently-shaped activation would be a free unconstrained parameter."""
    u = np.array([0.3, 1.7, 4.0])
    dt, tau = 0.05, 800.0
    cleared = u - PUMP.step_spike_load(u, np.zeros(3, bool), a_load=0.0, tau_N=tau, dt=dt)
    phi_from_clearance = cleared * tau / dt
    phi_from_membrane = (PUMP.excess_pump_current(u, np.zeros(3), Imax=1.0)) / 1.0
    assert np.allclose(phi_from_clearance, phi_from_membrane, rtol=1e-12)


# ============================== clause 8/9/10: baseline-centered, NO positive part ==============================
def test_excess_current_is_imax_times_phi_minus_p0():
    u = np.array([0.5, 2.0])
    p0 = np.array([0.1, 0.2])
    got = PUMP.excess_pump_current(u, p0, Imax=3.0)
    assert np.allclose(got, 3.0 * (PUMP.pump_activation(u) - p0))


def test_excess_current_is_zero_when_phi_equals_p0():
    u = np.array([1.0])
    p0 = PUMP.pump_activation(u)
    assert PUMP.excess_pump_current(u, p0, Imax=5.0)[0] == 0.0


def test_excess_current_allows_negative_compensation_no_rectification():
    """REGRESSION for the rectified draft: phi<p0 MUST give a negative excess. A positive part would
    clamp it to 0, giving baseline noise a positive-only mean bias (spec §2.3)."""
    u = np.array([0.05])                                     # phi(u) ~ 1.2e-4
    p0 = np.array([0.4])
    got = PUMP.excess_pump_current(u, p0, Imax=2.0)[0]
    assert got < 0.0
    assert got == pytest.approx(2.0 * (PUMP.pump_activation(0.05) - 0.4))


def test_excess_current_mean_is_zero_on_a_symmetric_baseline_distribution():
    """The point of the p0 compensation: with p0 = E_baseline[phi(u)], the membrane sees zero MEAN
    extra current on baseline. Rectification would leave a strictly positive mean."""
    rng = np.random.default_rng(0)
    u = np.abs(rng.normal(1.0, 0.3, size=20000))
    p0 = float(PUMP.pump_activation(u).mean())
    ex = PUMP.excess_pump_current(u, np.full(u.shape, p0), Imax=1.0)
    assert abs(float(ex.mean())) < 1e-12
    rectified_mean = float(np.maximum(ex, 0.0).mean())
    assert rectified_mean > 1e-3                             # the forbidden variant is biased


# ============================== clause 11: h is fixed at 3 for the primary tier ==============================
def test_primary_tier_requires_h_equals_3():
    PUMP.require_primary_h(3)                                # no raise
    for bad in (2, 4, 1):
        with pytest.raises(ValueError):
            PUMP.require_primary_h(bad)


def test_h_is_still_a_parameter_for_the_deferred_sensitivity_tier():
    u = np.array([2.0])
    assert PUMP.pump_activation(u, h=2)[0] == pytest.approx(4.0 / 5.0)
    assert PUMP.pump_activation(u, h=3)[0] == pytest.approx(8.0 / 9.0)


# ============================== clause 12: primary load is spike-only ==============================
def test_primary_load_function_has_no_synaptic_conductance_input():
    """Spec §2.5: g_rec_raw has neither a driving force nor the applied tanh saturation, so it must
    not enter the PRIMARY load. This is a structural guard: no conductance/charge argument exists."""
    params = set(inspect.signature(PUMP.step_spike_load).parameters)
    assert params == {"u", "spikes", "a_load", "tau_N", "dt", "h"}
    banned = ("g_rec", "gerec", "conduct", "charge", "q_e", "influx", "i_e")
    src = inspect.getsource(PUMP.step_spike_load).lower()
    assert not any(b in src for b in banned)
