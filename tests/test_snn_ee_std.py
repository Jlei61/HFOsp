"""M1 mechanism unit tests: presynaptic E->E short-term depression + slow recovery.

The four contract invariants (06-17 / 06-18 M1 plan Stage 1 engine gate):
  - U=0 is bit-identical to M0 (no new RNG draws / float touches on the default path);
  - x_j drops by factor (1-U) on a spike;
  - x_j recovers toward 1 between spikes;
  - ONLY E->E edges are scaled (E->I untouched).
"""
import sys, os, hashlib
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from params import Params
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
import kick_probe
from kick_probe import (
    ee_std_apply,
    ee_std_recover_factor,
    ee_std_source_availability,
    simulate_kick,
)

# captured from the pre-M1-edit engine (a51e0875c3ec) on the L=6/d100/T300/seed1 fixture
M0_BASELINE_SHA = "da5fc18c27d5340a"


def _net(seed=1, L=6.0):
    p = Params(L=L, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net, NE, NI


def test_recover_factor():
    # exact solution of dx/dt=(1-x)/tau over dt: 1 - exp(-dt/tau)
    assert ee_std_recover_factor(0.1, 100.0) == pytest.approx(1 - np.exp(-0.1 / 100.0))


def test_apply_depresses_only_EE():
    # 4 edges from one source: 2 to E targets (dst<NE), 2 to I targets (dst>=NE); NE=10
    a_w = np.array([1.0, 1.0, 1.0, 1.0]); a_dst = np.array([3, 12, 5, 11]); NE = 10
    x = np.full(4, 0.25)                       # presynaptic availability per edge
    w = ee_std_apply(a_w, a_dst, x, NE)
    assert np.allclose(w, [0.25, 1.0, 0.25, 1.0])   # E targets scaled, I targets untouched


def test_global_std_applies_exact_latent_mean_without_source_identity():
    state = np.asarray([0.2, 0.6, 1.0, 0.8])
    sources = np.asarray([0, 2])
    local = ee_std_source_availability(state, sources, "local")
    global_ = ee_std_source_availability(state, sources, "global")
    np.testing.assert_allclose(local, [0.2, 1.0])
    np.testing.assert_allclose(global_, [state.mean(), state.mean()])


def test_std_mode_is_validated_only_when_std_is_active():
    p, net, NE, NI = _net()
    net["rng"] = np.random.default_rng(1)
    with pytest.raises(ValueError, match="ee_std_mode"):
        simulate_kick(
            p, net, KICK_BOOST=0.0, t_kick=1e9,
            V_th_per_neuron=np.full(NE + NI, 18.0),
            ee_std_u=0.2, ee_std_tau_ms=200.0, ee_std_mode="bad",
        )


def test_u0_is_bit_identical_to_M0():
    p, net, NE, NI = _net()
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        ee_std_u=0.0, ee_std_tau_ms=0.0)
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == M0_BASELINE_SHA


def test_u_positive_changes_spikes():
    p, net, NE, NI = _net()
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        ee_std_u=0.4, ee_std_tau_ms=200.0)
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] != M0_BASELINE_SHA


def test_requires_tau_when_u_positive():
    p, net, NE, NI = _net()
    net["rng"] = np.random.default_rng(1)
    with pytest.raises(AssertionError):
        simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                      ee_std_u=0.2, ee_std_tau_ms=0.0)
