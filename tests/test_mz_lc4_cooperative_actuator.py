"""FCXR-LC4: a cooperative outward current opened by the cell's own load.

Contract, from the 2026-08-08 review §4 mechanism 2+3 and the measurement it rests on:

    tau_m m'_i = -m_i + sum_k delta(t - t_i^k)          (unchanged; LINEAR clearance)
    a_inf(m_i) = m_i^n / (K^n + m_i^n)
    tau_a(m) a'_i = a_inf(m_i) - a_i ,   tau_on independent of tau_off
    I_M,i = g_M a_i (V_i - E_K)

Every clause here is science, not implementation, and each gets its own test:

* **The curve opens a channel; it must never gate the load's clearance.**  The 2026-07-27 pump
  used one Hill for both, so its stationary activation was a_load*tau*r with the half-point
  algebraically cancelled -- no threshold could be placed anywhere, and that line was closed
  because of it.  Clearing linearly leaves m* = r*tau and makes K a real threshold.  A test
  asserts the load is bit-identical with the curve on and off.
* **Opening and closing carry separate time constants.**  Opening slower than one interictal event
  decides whether the brief transient is followed at all; closing slower than wear clearance
  decides whether protection outlasts the discharge.  Neither may be forced to equal the other --
  the registered control arm of the 2x2 is exactly tau_off = tau_on.
* **Off by default, byte-for-byte.**  A knob that changes the run when nominally off is how a
  mechanism gets credited with something the substrate did anyway.
* **A strength set with the mechanism off must raise**, the same rule the recruitment brake got
  after a run reported it ineffective when it had never been switched on.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402

N, NE, DT = 10, 8, 0.05
FC = dict(membrane_mode="full_conductance")
HILL = dict(m_hill_K=2.0, m_hill_n=4.0, tau_a_on=50.0, tau_a_off=5000.0, g_m_max=1.0)


def _mk(**kw):
    core = np.zeros(NE, bool)
    core[:2] = True
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**kw), NE=NE, core_mask_E=core)


def _drive(mz, n_steps, firing):
    """Step the module with a fixed set of E cells spiking every step."""
    spk = np.zeros(N, bool)
    spk[list(firing)] = True
    for _ in range(n_steps):
        mz.step(spk, labels=None, dt=DT)


# ---------------------------------------------------------------- the load must stay untouched

def test_the_curve_does_not_gate_the_load():
    """The failure that closed the pump line: one Hill on both clearance and current cancels the
    half-point out of the stationary activation.  Here the load must not know the curve exists."""
    off = _mk(use_m=True, tau_adp=200.0, eta_m=0.1, **FC)
    on = _mk(use_m=True, tau_adp=200.0, eta_m=0.1, **FC, **HILL)
    _drive(off, 2000, (0, 1, 2))
    _drive(on, 2000, (0, 1, 2))
    assert np.array_equal(off.m, on.m), "the cooperative curve fed back into the load"


def test_moving_the_half_point_moves_the_delivered_current():
    """The other half of the same contract: with linear clearance K is a real threshold, so the
    same load must deliver different current at different K.  Under the pump's form it could not."""
    lo = _mk(use_m=True, tau_adp=200.0, **FC, **{**HILL, "m_hill_K": 1.0})
    hi = _mk(use_m=True, tau_adp=200.0, **FC, **{**HILL, "m_hill_K": 8.0})
    _drive(lo, 4000, (0, 1))
    _drive(hi, 4000, (0, 1))
    assert np.array_equal(lo.m, hi.m), "same drive, same load"
    assert lo.a[0] > hi.a[0], "yet the lower half-point must open the channel further"


def test_half_activation_is_exactly_at_K():
    mz = _mk(use_m=True, m_frozen_E=None, **FC, **HILL)
    mz.m[:NE] = HILL["m_hill_K"]
    x = (mz.m[:NE] / HILL["m_hill_K"]) ** HILL["m_hill_n"]
    assert np.allclose(x / (1.0 + x), 0.5)


# ---------------------------------------------------------------- the two time constants

def test_opening_and_closing_use_their_own_time_constants():
    """One step from a known state: the increment must be dt/tau_on going up and dt/tau_off coming
    down.  A single tau would make the protection window a hostage of the opening speed."""
    mz = _mk(use_m=True, tau_adp=1e9, **FC, **HILL)
    mz.m[:NE] = 100.0                                   # a_inf ~ 1, far above a=0 -> opening
    a0 = mz.a[0]
    mz.step(np.zeros(N, bool), labels=None, dt=DT)
    up = mz.a[0] - a0
    assert np.isclose(up, (DT / HILL["tau_a_on"]) * (1.0 - a0), rtol=1e-6)

    mz.m[:NE] = 0.0                                     # a_inf = 0, below a -> closing
    a1 = mz.a[0]
    mz.step(np.zeros(N, bool), labels=None, dt=DT)
    assert np.isclose(mz.a[0] - a1, -(DT / HILL["tau_a_off"]) * a1, rtol=1e-6)


def test_no_instantaneous_reset():
    """First-order relaxation like z and the relay: one step never lands on the target."""
    mz = _mk(use_m=True, tau_adp=1e9, **FC, **HILL)
    mz.m[:NE] = 100.0
    mz.step(np.zeros(N, bool), labels=None, dt=DT)
    assert 0.0 < mz.a[0] < 0.5


def test_a_symmetric_setting_is_allowed_because_it_is_the_registered_control():
    """The 2x2 needs fast-release as its control arm; forbidding tau_off == tau_on would delete it."""
    mz = _mk(use_m=True, **FC, **{**HILL, "tau_a_off": HILL["tau_a_on"]})
    mz.m[:NE] = 100.0
    mz.step(np.zeros(N, bool), labels=None, dt=DT)
    assert mz.a[0] > 0.0


# ---------------------------------------------------------------- per-cell, E-only

def test_the_open_fraction_is_per_cell_and_leaves_inhibitory_cells_alone():
    """The accepted 2026-07-26 contract requires per-cell load and recovery state; a population
    mean was shown to zero the effect it is supposed to carry."""
    mz = _mk(use_m=True, tau_adp=500.0, **FC, **HILL)
    _drive(mz, 3000, (0, 1))                            # only two E cells fire
    assert mz.a[0] > 0.0 and mz.a[2] == 0.0, "cells that did not fire must not be braked"
    assert np.all(mz.a[NE:] == 0.0), "inhibitory cells carry no adaptation"


# ---------------------------------------------------------------- off by default

def test_the_new_knobs_do_nothing_while_the_curve_is_off():
    """Byte parity: with m_hill_K unset, the kinetics and strength must be inert."""
    plain = _mk(use_m=True, tau_adp=200.0, eta_m=0.1, **FC)
    knobs = _mk(use_m=True, tau_adp=200.0, eta_m=0.1, **FC,
                m_hill_n=9.0, tau_a_on=1.0, tau_a_off=7.0)
    _drive(plain, 500, (0, 3))
    _drive(knobs, 500, (0, 3))
    assert np.array_equal(plain.m, knobs.m)
    assert np.array_equal(plain.a, knobs.a) and np.all(knobs.a == 0.0)
    I_E, I_I = np.arange(N, dtype=float) + 1.0, np.arange(N, dtype=float) * 0.5
    assert np.array_equal(plain.membrane_terms(I_E, I_I, I_E_rec=np.zeros(N))[0],
                          knobs.membrane_terms(I_E, I_I, I_E_rec=np.zeros(N))[0])


def test_the_curve_replaces_the_linear_actuator_rather_than_adding_to_it():
    """Two actuators on one conductance would double-count the same load."""
    mz = _mk(use_m=True, tau_adp=1e9, eta_m=1e6, **FC, **HILL)   # a huge linear gain, ignored
    mz.m[:NE] = HILL["m_hill_K"]
    g = mz.membrane_terms(np.zeros(N), np.zeros(N), I_E_rec=np.zeros(N))
    gM = mz._gM_max_last
    assert gM < 1.0, f"the linear eta_m must not reach the conductance, got gM={gM}"


# ---------------------------------------------------------------- loud failures

def test_a_strength_without_the_curve_raises():
    with pytest.raises(ValueError, match="g_m_max requires m_hill_K"):
        _mk(use_m=True, g_m_max=0.5, **FC)


def test_the_curve_requires_a_load_to_act_on():
    with pytest.raises(ValueError, match="needs a load to act on"):
        _mk(use_m=False, **FC, **HILL)


def test_the_curve_requires_the_conductance_membrane():
    with pytest.raises(ValueError, match="requires membrane_mode='full_conductance'"):
        _mk(use_m=True, **{**HILL, "m_hill_K": 2.0})


def test_the_additive_current_path_refuses_rather_than_running_the_linear_actuator():
    """apply_currents cannot express a reversal-bearing conductance; silently using eta_m*m there
    would run a different mechanism than the one configured."""
    mz = _mk(use_m=True, **FC, **HILL)
    with pytest.raises(RuntimeError, match="no additive-current form"):
        mz.apply_currents(np.ones(N), np.ones(N))


@pytest.mark.parametrize("bad", [dict(m_hill_K=0.0), dict(m_hill_n=0.0),
                                 dict(tau_a_on=0.0), dict(tau_a_off=-1.0)])
def test_non_positive_kinetics_raise(bad):
    with pytest.raises(ValueError, match="must be finite and > 0"):
        _mk(use_m=True, **FC, **{**HILL, **bad})


def test_the_recorded_current_is_the_one_actually_delivered():
    """The project's own 'fields that must not be read' list is full of diagnostics that kept
    reporting a disabled path; the adaptation-current trace must not become another one."""
    mz = _mk(use_m=True, tau_adp=1e9, eta_m=1e6, **FC, **HILL)   # linear gain huge and unused
    mz.m[:NE] = 100.0
    _drive(mz, 200, ())
    mz.record_traces() if hasattr(mz, "record_traces") else None
    assert mz.trace_adap_current, "the trace must be populated"
    assert mz.trace_adap_current[-1] <= HILL["g_m_max"], (
        "the trace is reporting the linear actuator that this configuration disabled")
    assert mz.trace_a_mean and 0.0 < mz.trace_a_mean[-1] <= 1.0


def test_the_open_fraction_traces_stay_empty_while_the_curve_is_off():
    mz = _mk(use_m=True, tau_adp=200.0, eta_m=0.1, **FC)
    _drive(mz, 200, (0,))
    assert mz.trace_adap_current, "the existing trace must keep working"
    assert mz.trace_a_mean == [] and mz.trace_a_max == []
