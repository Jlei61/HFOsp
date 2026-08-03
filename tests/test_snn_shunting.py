"""M2 conductance shunting — TDD for the membrane_step pure helper + shunting + simulate_kick wiring.

Default path (shunt_gaba=False) MUST stay spike-identical to the pre-edit engine: small-net
spike SHA da5fc18c27d5340a is the single bit-parity anchor.
"""
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from params import Params
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick, membrane_step, som_shunt_membrane_step

BASELINE_SHA = "da5fc18c27d5340a"


def test_membrane_step_current_path_matches_formula():
    V = np.array([12.0, 15.0])
    I_E = np.array([20.0, 5.0])
    I_I = np.array([4.0, 1.0])
    decay = np.array([0.99, 0.99])
    I_net = I_E - I_I
    expected = I_net + (V - I_net) * decay
    np.testing.assert_allclose(membrane_step(V, I_E, I_I, decay), expected)


def _sha():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0))
    return hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16]


def test_extraction_preserves_bit_parity():
    assert _sha() == BASELINE_SHA


# --- Task 2: conductance shunting (membrane_step already supports it; these LOCK the behavior).
# Discrimination check (confirms NOT vacuous): set g_gaba_scale=0 in
# test_shunting_gates_high_drive_below_threshold and Vs settles at I_E-I_I=20 > 18 -> the
# `Vs < 18` assertion FAILS. Verified manually, then restored to g_gaba_scale=1.0.
def test_shunt_g0_reduces_to_leak_toward_drive():
    # g_gaba_scale=0 under shunting: V relaxes toward I_E with decay_V (no inhibition) -> V_inf=I_E
    V = np.array([12.0])
    I_E = np.array([20.0])
    I_I = np.array([5.0])
    decay = np.array([0.9])
    out = membrane_step(V, I_E, I_I, decay, shunt_gaba=True, e_gaba=11.0, g_gaba_scale=0.0)
    np.testing.assert_allclose(out, I_E + (V - I_E) * decay)


def test_shunting_gates_high_drive_below_threshold():
    # KEY: a strongly-driven cell (I_E=30 >> V_th=18). Current-subtraction with I_I=10 settles at
    # I_E-I_I=20 > V_th (drive wins -> fires). Shunting with g_I=10 settles at
    # (30+10*11)/(1+10) = 12.7 < V_th (clamped toward e_gaba -> spike-initiation gated).
    decay = np.array([np.exp(-0.1 / 20.0)])
    Vc = np.array([11.0])
    Vs = np.array([11.0])
    for _ in range(3000):
        Vc = membrane_step(Vc, np.array([30.0]), np.array([10.0]), decay, shunt_gaba=False)
        Vs = membrane_step(Vs, np.array([30.0]), np.array([10.0]), decay,
                           shunt_gaba=True, e_gaba=11.0, g_gaba_scale=1.0)
    assert Vc[0] > 18.0    # current-subtraction: drive overwhelms inhibition
    assert Vs[0] < 18.0    # shunting: clamped below threshold regardless of drive


def test_shunting_changes_engine_spikes():
    # helper diverges from the current path once shunting is on
    V = np.array([15.0])
    I_E = np.array([22.0])
    I_I = np.array([6.0])
    decay = np.array([0.9])
    cur = membrane_step(V, I_E, I_I, decay, shunt_gaba=False)
    sh = membrane_step(V, I_E, I_I, decay, shunt_gaba=True, e_gaba=11.0, g_gaba_scale=0.5)
    assert not np.allclose(cur, sh)


def test_som_shunt_is_e_only_z_scaled_and_zero_scale_exact():
    V = np.array([15.0, 15.0, 15.0])
    I_net = np.array([24.0, 24.0, 6.0])
    I_slow = np.array([4.0, 4.0, 9.0])
    decay = np.full(3, 0.9)
    is_E = np.array([True, True, False])
    native = I_net + (V - I_net) * decay
    np.testing.assert_array_equal(
        som_shunt_membrane_step(
            V, I_net, I_slow, decay, is_E,
            g_scale=0.0, e_gaba=11.0, z_e=np.array([1.0, 0.5]),
        ),
        native,
    )
    out = som_shunt_membrane_step(
        V, I_net, I_slow, decay, is_E,
        g_scale=1.0, e_gaba=11.0, z_e=np.array([1.0, 0.5]),
    )
    assert out[0] < out[1] < native[1]
    assert out[2] == native[2]


# --- Task 3: shunting wired through simulate_kick (default-off parity is the safety gate).
def test_simulate_kick_shunt_off_is_bit_identical():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        shunt_gaba=False)
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == BASELINE_SHA


def test_simulate_kick_shunt_on_changes_spikes():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        shunt_gaba=True, e_gaba=11.0, g_gaba_scale=0.5)
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] != BASELINE_SHA


def test_simulate_kick_shunt_plus_recovery_coexist():
    # P0 coexistence guard: shunting AND E->E recovery (M1) both active in ONE simulate_kick call
    # -> the "+recovery" leg can't be silently dropped when shunting is added.
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        shunt_gaba=True, g_gaba_scale=0.5, ee_std_u=0.2, ee_std_tau_ms=200.0)
    assert "E_spk_bool" in res   # both legs on, ran without error


# --- Task 4.5: I-spike + peak-drive recording (READOUT-only -> bit-parity preserved).
def test_dump_i_spikes_and_drive_parity_and_presence():
    p = Params(L=6.0, density=100.0, T=300.0, dt=0.1, nu_ext_ratio=0.6, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        dump_i_spikes=True, dump_drive=True)
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == BASELINE_SHA   # recording-only
    assert "I_spk_bool" in res and res["I_spk_bool"].shape == (res["E_spk_bool"].shape[0], NI)
    assert "I_E_peak" in res and res["I_E_peak"].shape == (NE + NI,)
    assert "I_I_peak" in res and res["I_I_peak"].shape == (NE + NI,)
