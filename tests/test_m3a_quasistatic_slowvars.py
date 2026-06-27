"""TDD for the M3A-A1 quasi-static (frozen) slow-variable layer.

A1 freezes ONE slow variable at a constant value and runs a no-kick spontaneous sim, asking
whether the spontaneous-event phenotype shifts. These tests lock the contract:

  - slow=None is bit-identical to the pre-slow engine (regression anchor BASELINE_SHA).
  - FrozenSlowVars(z=1.0) is BYTE-IDENTICAL to that baseline (frozen z=1 = full inhibition = OFF).
  - step() is a no-op (the state is clamped — that IS quasi-static).
  - sign semantics: lower z -> more excitable; higher phi -> more inhibitory; higher gK -> more
    inhibitory.
  - build_frozen_slowvars enforces EXACTLY ONE active variable (A1 single-variable rule).
  - e_GABA is ONLY effective on the membrane shunt path (slow=None, shunt_gaba=True); on the
    current-based path it is inert. And the engine BYPASSES shunt_gaba whenever slow is not None
    (the "do not combine z and e_GABA" trap the runner must guard).

Spec: docs/superpowers/plans/2026-06-24-sef-hfo-m3a-quasistatic-slowstate-plan.md Task 1/2.
"""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick

from src.sef_hfo_slowvars_quasistatic import FrozenSlowVars, build_frozen_slowvars

# Same small-net config as tests/test_snn_shunting.py — its slow=None spike SHA is the bit-parity anchor.
BASELINE_SHA = "da5fc18c27d5340a"


def _build(L=6.0, density=100.0, T=300.0, nu=0.6, seed=1):
    p = Params(L=L, density=density, T=T, dt=0.1, nu_ext_ratio=nu, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net, NE, NI


def _run(p, net, NE, NI, *, slow=None, V_th_per_neuron=None, shunt_gaba=False,
         e_gaba=None, g_gaba_scale=0.0):
    net["rng"] = np.random.default_rng(1)
    return simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, slow=slow,
                         V_th_per_neuron=V_th_per_neuron, shunt_gaba=shunt_gaba,
                         e_gaba=e_gaba, g_gaba_scale=g_gaba_scale)


def _sha(res):
    return hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16]


def _espikes(res):
    return int(res["E_spk_bool"].sum())


# ---- bit-parity (item 4: "slow=None bit-parity") ----
def test_slow_none_matches_baseline_sha():
    p, net, NE, NI = _build()
    res = _run(p, net, NE, NI, slow=None, V_th_per_neuron=np.full(NE + NI, 18.0))
    assert _sha(res) == BASELINE_SHA


def test_frozen_z1_is_byte_identical_to_baseline():
    # Frozen z=1.0 (full inhibition) on the slow= path must reproduce the no-slow baseline exactly:
    # I_net = I_E - 1*I_I, threshold = p.V_th=18 — identical arithmetic, step() frozen.
    p, net, NE, NI = _build()
    sv = build_frozen_slowvars(NE + NI, p.V_th, z=1.0)
    res = _run(p, net, NE, NI, slow=sv)
    assert _sha(res) == BASELINE_SHA


# ---- freeze (item 3: quasi-static = clamped, no dynamics) ----
def test_step_is_noop_keeps_state_frozen():
    sv = build_frozen_slowvars(10, 18.0, z=0.7)
    z0, phi0, gK0 = sv.z.copy(), sv.phi.copy(), sv.gK.copy()
    spk = np.zeros(10, dtype=bool); spk[3] = True
    for _ in range(50):
        sv.step(spk, np.zeros(10, dtype=int), dt=0.1)
    np.testing.assert_array_equal(sv.z, z0)
    np.testing.assert_array_equal(sv.phi, phi0)
    np.testing.assert_array_equal(sv.gK, gK0)


# ---- sign semantics (item 4) ----
def test_lower_z_is_more_excitable():
    p, net, NE, NI = _build(T=500.0)
    hi = _espikes(_run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, z=1.0)))
    lo = _espikes(_run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, z=0.7)))
    assert lo > hi


def test_higher_phi_is_more_inhibitory():
    p, net, NE, NI = _build(T=500.0)
    base = _espikes(_run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, phi_offset=0.0)))
    raised = _espikes(_run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, phi_offset=2.0)))
    assert raised < base


def test_higher_gK_is_more_inhibitory():
    p, net, NE, NI = _build(T=500.0)
    base = _espikes(_run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, gK=0.0)))
    suppressed = _espikes(_run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, gK=2.0)))
    assert suppressed < base


# ---- single-variable rule (item 3: one variable at a time) ----
def test_build_phi_with_vth_field_rides_core():
    # phi rides the Stage-3 core: per-neuron phi = core threshold field + offset (so the adaptive
    # threshold is anchored to the heterogeneous core, not a uniform V_th0).
    core = np.array([16.0, 18.0, 17.0])
    sv = build_frozen_slowvars(3, 18.0, phi_offset=1.0, vth_field=core)
    np.testing.assert_array_equal(sv.phi, core + 1.0)


def test_build_requires_exactly_one_variable():
    import pytest
    with pytest.raises(ValueError):
        build_frozen_slowvars(10, 18.0)                       # none
    with pytest.raises(ValueError):
        build_frozen_slowvars(10, 18.0, z=0.8, gK=1.0)        # two


# ---- e_GABA path semantics (item 4: only effective in shunt path) ----
def test_e_gaba_inert_on_current_path():
    # shunt_gaba=False: e_gaba is unused by membrane_step -> changing it cannot change spikes.
    p, net, NE, NI = _build()
    vth = np.full(NE + NI, 18.0)
    a = _run(p, net, NE, NI, V_th_per_neuron=vth, shunt_gaba=False, e_gaba=11.0)
    b = _run(p, net, NE, NI, V_th_per_neuron=vth, shunt_gaba=False, e_gaba=16.0)
    assert _sha(a) == _sha(b)


def test_e_gaba_active_on_shunt_path():
    # shunt_gaba=True, g_gaba_scale=1.0 (calibrated, see test_snn_shunting): depolarizing e_gaba
    # (11 -> 16, toward V_th=18) makes shunting LESS protective -> spikes change.
    p, net, NE, NI = _build()
    vth = np.full(NE + NI, 18.0)
    a = _run(p, net, NE, NI, V_th_per_neuron=vth, shunt_gaba=True, e_gaba=11.0, g_gaba_scale=1.0)
    b = _run(p, net, NE, NI, V_th_per_neuron=vth, shunt_gaba=True, e_gaba=16.0, g_gaba_scale=1.0)
    assert _sha(a) != _sha(b)


def test_slow_path_respects_V_th_per_neuron_substrate():
    # Off-by-default HOOK: under slow!=None with use_phi=False (z/gK), a per-neuron threshold
    # substrate (the Stage-3 heterogeneous core) must lower thresholds where the core is -> more
    # spikes than the uniform p.V_th. This lets z/gK ride the Stage-3 excitable core.
    p, net, NE, NI = _build()
    N = NE + NI
    vth_core = np.full(N, 18.0); vth_core[:NE // 2] = 16.0     # half the E cells = excitable core
    uniform = _run(p, net, NE, NI, slow=build_frozen_slowvars(N, p.V_th, z=1.0))            # ->p.V_th=18
    cored = _run(p, net, NE, NI, slow=build_frozen_slowvars(N, p.V_th, z=1.0), V_th_per_neuron=vth_core)
    assert _espikes(cored) > _espikes(uniform)


def test_slow_path_uniform_V_th_per_neuron_equals_none():
    # Off-by-default: passing V_th_per_neuron=full(p.V_th) under slow == passing None (both -> p.V_th).
    p, net, NE, NI = _build()
    a = _run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, z=0.8))
    b = _run(p, net, NE, NI, slow=build_frozen_slowvars(NE + NI, p.V_th, z=0.8),
             V_th_per_neuron=np.full(NE + NI, p.V_th))
    assert _sha(a) == _sha(b)


def test_slow_path_bypasses_shunt_gaba():
    # The trap the runner guards: with slow!=None the engine bypasses membrane_step, so shunt_gaba
    # is silently ignored. Lock it so a future combination of z + e_GABA fails loudly, not silently.
    p, net, NE, NI = _build()
    sv1 = build_frozen_slowvars(NE + NI, p.V_th, z=0.8)
    sv2 = build_frozen_slowvars(NE + NI, p.V_th, z=0.8)
    no_shunt = _run(p, net, NE, NI, slow=sv1, shunt_gaba=False)
    with_shunt = _run(p, net, NE, NI, slow=sv2, shunt_gaba=True, e_gaba=16.0, g_gaba_scale=1.0)
    assert _sha(no_shunt) == _sha(with_shunt)   # shunt args had NO effect under slow!=None
