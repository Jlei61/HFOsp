"""TDD for M4-2 engine instrumentation (spec 2026-07-07 rev2, Task 1).

Two NON-behavioral hooks added to simulate_kick, OFF by default -> byte-identical to today:
  1. dump_ee_std_trace  -> x_dep depression summary trace (mean/min, + optional axis-mask mean).
                           Arm 0 (ee_std_u=0) emits CONSTANT 1.0 (availability un-depleted).
                           Recording point is AFTER the spike depletion (:371), NOT after recovery (:259).
  2. t_kick2/KICK_BOOST2 -> a second kick window (post-offset retrigger). t_kick2=None -> parity.
                           Pre-probe identity: for t < t_kick2 the trajectory is byte-identical to a
                           run without the second kick (makes retrigger_probe interpretable).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params                          # noqa: E402
from connectivity import place_neurons             # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick               # noqa: E402

DT = 0.1


def _net(L=6.0, T=200.0, seed=1, density=100.0, nu=0.8):
    p = Params(L=L, density=density, T=T, dt=DT, nu_ext_ratio=nu, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net


def _fresh(net, seed=1):
    net["rng"] = np.random.default_rng(seed)
    return net


# -------------------------------------------------- byte-parity of the new (default) params
def test_new_params_default_byte_identical():
    """simulate_kick with all new params at their defaults == today (no alloc/RNG/float change)."""
    p, net = _net()
    base = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0)
    new = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                        dump_ee_std_trace=False, ee_std_trace_maskE=None,
                        t_kick2=None, KICK_BOOST2=0.0)
    assert np.array_equal(base["E_spk_bool"], new["E_spk_bool"])
    assert np.array_equal(base["rate_E"], new["rate_E"])


def test_trace_does_not_perturb_dynamics():
    """dump_ee_std_trace is read-only: with ee_std_u>0 it must not change spikes; it only adds outputs."""
    p, net = _net()
    a = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=0.2, ee_std_tau_ms=500.0)
    b = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=0.2, ee_std_tau_ms=500.0, dump_ee_std_trace=True)
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert "xdep_min" in b and "xdep_min" not in a


# -------------------------------------------------- Arm 0 constant-ones schema (P1-b)
def test_arm0_xdep_trace_constant_ones():
    """ee_std_u=0 with dump_ee_std_trace=True -> constant 1.0 trace (schema aligned with Arm 1)."""
    p, net = _net()
    r = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=0.0, dump_ee_std_trace=True)
    assert np.allclose(r["xdep_mean"], 1.0)
    assert np.allclose(r["xdep_min"], 1.0)


# -------------------------------------------------- pre-probe identity (P1-a)
def test_second_kick_prewindow_identity():
    """A run with a 2nd kick at t2 is byte-identical to a run without it for all t < t2
    (the t_kick2 branch is skipped for t<t2); t>=t2 must differ (the 2nd kick does something)."""
    p, net = _net(T=300.0)
    t2 = 150.0
    i2 = int(round(t2 / DT))
    a = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0)                 # no 2nd kick
    b = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      t_kick2=t2, KICK_BOOST2=3.0)                                              # 2nd kick at t2
    assert np.array_equal(a["E_spk_bool"][:i2], b["E_spk_bool"][:i2])          # pre-probe identity (承重)
    assert not np.array_equal(a["E_spk_bool"][i2:], b["E_spk_bool"][i2:])      # 2nd kick perturbs t>=t2 (not a no-op)


# -------------------------------------------------- trace recorded AFTER depletion (P1-c phase)
def test_xdep_trace_phase_post_depletion():
    """x_dep trace must be recorded AFTER the spike depletion (:371), not after recovery (:259):
    at the FIRST E-spike step the min availability already shows the depletion (== 1-u), not 1.0."""
    p, net = _net()
    u = 0.2
    r = simulate_kick(p, _fresh(net), 3.0, slow=None, t_kick=50.0, r_kick=2.0,
                      ee_std_u=u, ee_std_tau_ms=500.0, dump_ee_std_trace=True)
    espk = r["E_spk_bool"]
    xmin = r["xdep_min"]
    fired = np.where(espk.any(axis=1))[0]
    assert fired.size > 0
    t_spk = int(fired[0])                        # first step an E neuron fires
    assert xmin[t_spk] < 1.0                      # depletion reflected SAME step -> recorded post-:371
    assert np.isclose(xmin[t_spk], 1.0 - u)       # first spike: firers deplete from 1 -> (1-u)
