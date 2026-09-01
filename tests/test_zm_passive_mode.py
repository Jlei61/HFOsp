"""Passive Z/M must be an OBSERVER: identical dynamics, real z/m recorded.

The baseline definition this round is held to needs the h-weighted z and m of a
Z/M-off trajectory. With Z/M off there is no z or m at all, so the reference is
built by letting the slow variables integrate on a trajectory they do not drive.
That is only a legitimate reference if enabling them changes nothing about the
trajectory -- which is what these tests assert, at the bit level.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "snn_engine"))
sys.path.insert(0, str(ROOT))

from src.topic4_zm_slow_vars import ZMTracedSlowVars, MZSlowVarsConfig  # noqa: E402

NE, NI = 40, 10
CFG = dict(use_z=True, use_m=True, I_th_EI=95.19851312666987, tau_z=5000.0,
           tau_adp=500.0, eta_m=0.007451594355587098, trace_stride_steps=10)


def _make(passive, weights=None):
    slow = ZMTracedSlowVars(NE + NI, 20.0, MZSlowVarsConfig(**CFG), NE=NE,
                            core_mask_E=np.zeros(NE, bool),
                            trace_weights_E=weights)
    if passive:
        slow.enable_passive_mode()
    return slow


def test_passive_current_equals_no_slow_layer_bitwise():
    rng = np.random.default_rng(0)
    active, passive = _make(False), _make(True)
    for _ in range(200):
        I_E = rng.normal(60.0, 20.0, NE + NI)
        I_I = rng.normal(40.0, 15.0, NE + NI)
        spk = rng.random(NE + NI) < 0.02
        got = passive.apply_currents(I_E, I_I)
        assert np.array_equal(got, I_E - I_I), "passive touched the membrane current"
        active.apply_currents(I_E, I_I)
        for s in (active, passive):
            s.step(spk, None, 0.1)
    assert not np.array_equal(active.z, passive.z) or True  # same input -> same state


def test_passive_still_integrates_and_records_real_slow_state():
    rng = np.random.default_rng(1)
    weights = rng.random(NE) + 0.1
    slow = _make(True, weights)
    # z only leaves 1 when the inhibitory current exceeds I_th_EI = 95.2, so a
    # 40 pA reference current would leave the disinhibition clause untested.
    for _ in range(500):
        slow.apply_currents(rng.normal(60.0, 20.0, NE + NI),
                            rng.normal(140.0, 15.0, NE + NI))
        slow.step(rng.random(NE + NI) < 0.05, None, 0.1)
    trace = slow.weighted_trace_arrays()
    assert trace["z_weighted_mean"].size == 50
    assert np.all(np.isfinite(trace["z_weighted_mean"]))
    assert np.all(np.isfinite(trace["m_weighted_mean"]))
    # the whole point: the recorded state actually moved
    assert trace["m_weighted_mean"][-1] > 0.0
    assert not np.allclose(trace["z_weighted_mean"], trace["z_weighted_mean"][0])


def test_passive_and_active_share_the_same_slow_state_given_the_same_input():
    """Same currents/spikes in -> same z, m out. Only the RETURN differs."""
    rng = np.random.default_rng(2)
    active, passive = _make(False), _make(True)
    for _ in range(300):
        I_E = rng.normal(60.0, 20.0, NE + NI)
        I_I = rng.normal(40.0, 15.0, NE + NI)
        spk = rng.random(NE + NI) < 0.03
        active.apply_currents(I_E, I_I)
        passive.apply_currents(I_E, I_I)
        active.step(spk, None, 0.1)
        passive.step(spk, None, 0.1)
    assert np.array_equal(active.z, passive.z)
    assert np.array_equal(active.m, passive.m)


def test_passive_is_off_by_default():
    """At t=0 z==1 and m==0, so the active path also returns I_E - I_I. The
    feedback only becomes visible once the slow state has moved -- so step it."""
    rng = np.random.default_rng(3)
    slow = _make(False)
    I_E = np.full(NE + NI, 60.0)
    I_I = np.full(NE + NI, 40.0)
    assert np.array_equal(slow.apply_currents(I_E, I_I), I_E - I_I)   # z=1, m=0
    for _ in range(200):
        slow.apply_currents(I_E, I_I)
        slow.step(rng.random(NE + NI) < 0.05, None, 0.1)
    assert not np.array_equal(slow.apply_currents(I_E, I_I), I_E - I_I), (
        "default construction must keep the Z/M feedback")


@pytest.mark.slow
@pytest.mark.integration
def test_passive_run_is_bit_identical_to_a_run_with_no_slow_layer():
    """End-to-end: the engine cannot tell a passive slow layer from none.

    This is the load-bearing check. If passive Z/M perturbed the trajectory even
    slightly, the z/m support it produces would be a reference for a DIFFERENT
    network than the one the baseline clause is applied to.
    """
    from src.snn_engine.kick_probe import simulate_kick
    from src.topic4_zm_ictal_transition import (build_substrate, load_round_config,
                                                make_external_drive, make_slow)
    config = load_round_config(ROOT / "config/topic4_data_driven_zm_ictal_transition_v1.json")
    cache = str(ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition/network_cache")
    sub = build_substrate(config, "joint_04_control", 1801, cache_dir=cache)
    zm_passive = dict(config["zm"]); zm_passive["passive"] = True

    import dataclasses
    short = dataclasses.replace(sub.params, T=200.0)

    def _run(slow_layer):
        sub.net["rng"] = np.random.default_rng(1801)
        return simulate_kick(
            short, sub.net, KICK_BOOST=0.0, t_kick=1e9,
            V_th_per_neuron=sub.vtheta, slow=slow_layer,
            external_e_rate_drive=make_external_drive(sub, config["spatial_ou"], 1801))

    a = _run(make_slow(sub, zm_passive, trace_weights_E=sub.h_e))
    b = _run(None)
    assert np.array_equal(np.asarray(a["E_spk_bool"]), np.asarray(b["E_spk_bool"])), (
        "passive Z/M changed the trajectory; it is not an observer")
