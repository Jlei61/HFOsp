"""TDD for A1c dynamic global feedback RESTRAINT (spec 2026-06-25, adversarially vetted + P1 review).

I_global(t) = feedback_gain * EMA_Hz(global E rate), injected on E cells only:
  I_net = I_E - (I_I + I_global)   for E cells.
EMA: r_ema += (1 - exp(-dt/tau)) * (rate_E[t]/NE/(dt*1e-3) - r_ema), consumed at the TOP of the next
step (one-step causal delay). Off-by-default (feedback_gain=0 => byte-identical => re-bless gate).
NAME DISCIPLINE: this is a global-feedback-restraint screen, NOT inhibitory-exhaustion validation.
"""
import hashlib
import math
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
from params import Params
from connectivity import place_neurons
from connectivity_rot import build_connectivity_rot
from kick_probe import simulate_kick
from lfp import LFPRecorder
from src.sef_hfo_slowvars_quasistatic import build_frozen_slowvars

FIXTURE = pickle.load(open(os.path.join(ROOT, "tests", "fixtures", "a1c_parity_baseline.pkl"), "rb"))
ENGINE_VERSIONS = os.path.join(ROOT, "results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
DT = 0.1


def _net(L=6.0, T=300.0, seed=1, density=100.0, nu=0.6):
    p = Params(L=L, density=density, T=T, dt=DT, nu_ext_ratio=nu, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net, NE, NI


def _assert_parity(res, base, net):
    assert np.array_equal(res["lfp_trace"], base["lfp_trace"])        # continuous (sub-threshold) recorder
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == base["spk_sha"]
    assert np.array_equal(res["rate_E"], base["rate_E"])
    assert net["rng"].bit_generator.state == base["rng_state"]        # zero added RNG draws


# ---- T1: gain=0 bit-parity vs the FROZEN pre-edit baseline (the re-bless gate) ----
def test_T1_gain0_byte_identical_plain():
    p, net, NE, NI = _net()
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=np.array([[2., 2.], [3., 3.], [4., 4.]]))
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        lfp_recorder=rec, feedback_gain=0.0)
    _assert_parity(res, FIXTURE["plain"], net)


# ---- T2: gain=0 bit-parity across a representative caller (M1 E->E depression) ----
def test_T2_gain0_byte_identical_M1_caller():
    p, net, NE, NI = _net()
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=np.array([[2., 2.], [3., 3.], [4., 4.]]))
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        lfp_recorder=rec, ee_std_u=0.2, ee_std_tau_ms=200.0, feedback_gain=0.0)
    _assert_parity(res, FIXTURE["m1"], net)


# ---- T3: default kwargs (gain=0, tau=0) must NOT divide-by-zero ----
def test_T3_default_kwargs_no_nan():
    p, net, NE, NI = _net()
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0))
    assert np.isfinite(res["rate_E"]).all()


# ---- T4: EMA coefficient = 1 - exp(-dt/tau) (NOT dt/tau) reaches ~63% at one tau ----
def test_T4_ema_alpha_exact():
    tau = 50.0
    alpha = 1.0 - math.exp(-DT / tau)
    r_star, r_ema = 7.0, 0.0
    for _ in range(int(round(tau / DT))):
        r_ema += alpha * (r_star - r_ema)
    assert abs(r_ema - (1.0 - 1.0 / math.e) * r_star) < 0.05 * r_star


# ---- T5: DETERMINISTIC causal delay + EMA — recompute I_global from rate_E, match the dumped trace ----
def test_T5_dumped_I_global_equals_recomputed_causal_EMA():
    gain, tau = 1.0, 50.0
    p, net, NE, NI = _net(T=300.0)
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 16.5),
                        feedback_gain=gain, feedback_tau_ms=tau, dump_fb=True)
    # res["rate_E"] is ALREADY Hz (count/NE/dt*1e3) == the engine's internal rate_E[t]/NE*inv_dt_ms.
    rate_hz = np.asarray(res["rate_E"], float); ig = np.asarray(res["I_global_trace"], float)
    assert rate_hz.sum() > 0                                         # non-vacuous: it actually spiked
    assert ig.shape == rate_hz.shape
    alpha = 1.0 - np.exp(-DT / tau); r_ema = 0.0                     # match the engine's np.exp alpha
    expected = np.empty_like(rate_hz)
    for t in range(len(rate_hz)):
        expected[t] = gain * r_ema                                  # I_global at step t uses rate_E[<t] only
        r_ema += alpha * (rate_hz[t] - r_ema)
    assert np.allclose(ig, expected, rtol=0, atol=1e-9)             # causal one-step delay + exact EMA


# ---- T6: mutual-exclusion / required-tau asserts fire (incl. slow != None) ----
def test_T6_guards_raise():
    p, net, NE, NI = _net(T=100.0); vth = np.full(NE + NI, 18.0)
    with pytest.raises(Exception):                                  # gain>0 requires tau>0
        net["rng"] = np.random.default_rng(1)
        simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=vth, feedback_gain=0.5, feedback_tau_ms=0.0)
    with pytest.raises(Exception):                                  # gain>0 incompatible with shunt_gaba
        net["rng"] = np.random.default_rng(1)
        simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=vth, feedback_gain=0.5,
                      feedback_tau_ms=50.0, shunt_gaba=True, g_gaba_scale=1.0)
    with pytest.raises(Exception):                                  # gain>0 incompatible with slow!=None
        net["rng"] = np.random.default_rng(1)
        simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, feedback_gain=0.5, feedback_tau_ms=50.0,
                      slow=build_frozen_slowvars(NE + NI, p.V_th, z=0.8))


# ---- T7: monotone braking — higher gain => non-increasing total E spikes ----
def test_T7_monotone_braking():
    def total_spk(gain):
        p, net, NE, NI = _net(T=400.0)
        net["rng"] = np.random.default_rng(1)
        kw = dict(feedback_gain=gain, feedback_tau_ms=50.0) if gain > 0 else {}
        res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 16.5), **kw)
        return float(res["E_spk_bool"].sum())
    r0, r_small, r_large = total_spk(0.0), total_spk(1.0), total_spk(8.0)
    assert r_large <= r_small <= r0


# ---- T8: engine is BLESSED — engine_versions.json sha256 matches current kick_probe.py (re-bless gate) ----
def test_T8_engine_blessed():
    import json
    rec = json.load(open(ENGINE_VERSIONS))
    kp = os.path.join(ROOT, "src", "snn_engine", "kick_probe.py")
    cur = hashlib.sha256(open(kp, "rb").read()).hexdigest()
    assert rec["src/snn_engine/kick_probe.py"] == cur               # FAILS until re-blessed after the edit


# ---- T9 (P1-3 regression): adding fb_override_trace did NOT change the DYNAMIC path (override=None). ----
# FIXTURE["dyn16"] was captured from the PRE-fb_override engine commit 065e54a (kick_probe.py sha256
# e8e9524903..., which has NO fb_override_trace param) — a genuine pre-edit reference, regenerable via
# `git show 065e54a:src/snn_engine/kick_probe.py`. If the refactor diverged the gain>0 path this FAILS.
def test_T9_dynamic_path_unchanged_by_override_param():
    p, net, NE, NI = _net(T=300.0)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=np.array([[2., 2.], [3., 3.], [4., 4.]]))
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 16.5),
                        lfp_recorder=rec, feedback_gain=16.0, feedback_tau_ms=50.0, dump_fb=True)  # override=None
    b = FIXTURE["dyn16"]
    assert np.array_equal(res["lfp_trace"], b["lfp_trace"])
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == b["spk_sha"]
    assert np.array_equal(res["rate_E"], b["rate_E"])
    assert np.array_equal(np.asarray(res["I_global_trace"], float), b["I_global_trace"])
    assert net["rng"].bit_generator.state == b["rng_state"]


# ---- T10 (P1-3): a ZERO prescribed brake is byte-identical to the no-feedback baseline (I_I + 0 == I_I). ----
def test_T10_zero_override_byte_identical_baseline():
    p, net, NE, NI = _net()
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=np.array([[2., 2.], [3., 3.], [4., 4.]]))
    nsteps = int(round(p.T / DT))
    net["rng"] = np.random.default_rng(1)
    res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                        lfp_recorder=rec, fb_override_trace=np.zeros(nsteps))
    _assert_parity(res, FIXTURE["plain"], net)                     # zero brake must not perturb anything


# ---- T11 (P1-3): a prescribed CONSTANT brake (a) is returned verbatim in the dump, (b) actually brakes. ----
def test_T11_constant_override_brakes_and_dumps_verbatim():
    def total_and_dump(level):
        p, net, NE, NI = _net(T=400.0)
        nsteps = int(round(p.T / DT))
        net["rng"] = np.random.default_rng(1)
        ov = np.full(nsteps, level, float)
        res = simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 16.5),
                            fb_override_trace=ov, dump_fb=True)
        return float(res["E_spk_bool"].sum()), np.asarray(res["I_global_trace"], float), ov
    n0, dump0, ov0 = total_and_dump(0.0)
    n50, dump50, ov50 = total_and_dump(50.0)
    assert np.array_equal(dump50, ov50)                            # dump returns the PRESCRIBED brake verbatim
    assert n50 < n0                                                # a real DC brake suppresses E firing


# ---- T12 (P1-3): fb_static control also rides the default membrane_step -> mutex with slow / shunt. ----
def test_T12_override_guards_raise():
    p, net, NE, NI = _net(T=100.0); nsteps = int(round(p.T / DT))
    with pytest.raises(Exception):
        net["rng"] = np.random.default_rng(1)
        simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, fb_override_trace=np.zeros(nsteps),
                      slow=build_frozen_slowvars(NE + NI, p.V_th, z=0.8))
    with pytest.raises(Exception):
        net["rng"] = np.random.default_rng(1)
        simulate_kick(p, net, KICK_BOOST=0.0, t_kick=1e9, V_th_per_neuron=np.full(NE + NI, 18.0),
                      fb_override_trace=np.zeros(nsteps), shunt_gaba=True, g_gaba_scale=1.0)
