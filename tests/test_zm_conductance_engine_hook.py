"""Guarded-engine tests for the Phase-D conductance Z/M membrane."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src/snn_engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from params import Params  # noqa: E402
from slow_field import (  # noqa: E402
    SpatialSlowField,
    SpatialSlowFieldConfig,
)
from zm_conductance import conductance_membrane_step  # noqa: E402
from src.topic4_zm_checkpoint import ZMCheckpoint  # noqa: E402


def _build(seed=1, T=80.0):
    p = Params(
        L=1.0,
        density=400.0,
        T=T,
        dt=0.1,
        seed=seed,
        nu_ext_ratio=1.0,
    )
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(
        p, pos, labels, NE, NI, rng, theta_EE=0.0, AR=2.0
    )
    net["rng"] = np.random.default_rng(19)
    return p, net


def _slow(p, net, *, gamma=1 / 6, use_phi=False):
    cfg = SpatialSlowFieldConfig(
        n_grid=8,
        use_qI=False,
        use_gK=False,
        use_z=True,
        use_m=True,
        tau_z=5000.0,
        I_th_EI=1.28,
        tau_adp=500.0,
        eta_m=0.001,
        use_phi=use_phi,
        tau_phi=100.0,
        delta_phi=1.0 if use_phi else 0.0,
        use_zm_conductance=True,
        cond_kappa_E=0.1,
        cond_kappa_I=0.25,
        cond_g_M=0.001 / 15.0,
        cond_gamma=gamma,
        cond_g_L=1.0,
        cond_E_L=0.0,
        cond_E_E=25.0,
        cond_E_I=11.0,
        cond_E_K=0.0,
        cond_tau_m_E=20.0,
    )
    pos = net["pos"]
    nE = net["NE"]
    return SpatialSlowField(
        nE + net["NI"], p.V_th, pos[:nE], pos[nE:], p.L, cfg=cfg
    )


def test_conductance_config_requires_the_clean_zm_substrate():
    with pytest.raises(ValueError, match="use_z"):
        SpatialSlowFieldConfig(
            use_qI=False, use_gK=False, use_zm_conductance=True
        ).validate()
    with pytest.raises(ValueError, match="use_SG"):
        SpatialSlowFieldConfig(
            use_qI=False,
            use_gK=False,
            use_z=True,
            use_m=True,
            use_SG=True,
            use_zm_conductance=True,
        ).validate()


def test_slow_delegation_matches_pure_math_and_stashes_raw_gaba():
    p, net = _build(T=1.0)
    slow = _slow(p, net)
    n = net["NE"] + net["NI"]
    V = np.full(n, 15.0)
    I_E = np.linspace(2.0, 6.0, n)
    I_I = np.linspace(1.0, 3.0, n)
    decay = np.exp(
        -p.dt / np.where(net["labels"] == 0, p.tau_m_E, p.tau_m_I)
    )
    got = slow.zm_conductance_step(V, I_E, I_I, decay)
    expected = conductance_membrane_step(
        V,
        I_E,
        I_I,
        slow.z,
        slow.m,
        decay,
        slow.is_E,
        slow.zm_conductance_config(),
    )
    np.testing.assert_array_equal(got["V_next"], expected["V_next"])
    np.testing.assert_array_equal(slow._I_I_last, I_I[: net["NE"]])
    assert slow.trace_cond_gI_global == [got["g_I_global"]]


def test_engine_bypasses_additive_apply_currents_and_emits_both_lfp_proxies():
    p, net = _build()
    slow = _slow(p, net)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("conductance path called additive apply_currents")

    slow.apply_currents = forbidden
    recorder = LFPRecorder(
        p,
        net["pos"],
        net["labels"],
        sites=np.array([[0.5, 0.5]]),
    )
    result = simulate_kick(
        p,
        net,
        4.0,
        slow=slow,
        kick_center=np.array([0.5, 0.5]),
        r_kick=0.3,
        t_kick=20.0,
        V_th_per_neuron=np.full(net["NE"] + net["NI"], 16.5),
        lfp_recorder=recorder,
    )
    assert result["E_spk_bool"].sum() > 0
    assert result["lfp_trace"].shape == result["lfp_current_proxy_trace"].shape
    assert not np.array_equal(
        result["lfp_trace"], result["lfp_current_proxy_trace"]
    )
    assert len(slow.trace_cond_vinf_mean) == result["E_spk_bool"].shape[0]


def test_old_SG_value_does_not_divide_recurrent_E_in_conductance_mode():
    def run(S_G):
        p, net = _build()
        slow = _slow(p, net)
        slow.S_G = S_G
        result = simulate_kick(
            p,
            net,
            4.0,
            slow=slow,
            kick_center=np.array([0.5, 0.5]),
            r_kick=0.3,
            t_kick=20.0,
            V_th_per_neuron=np.full(net["NE"] + net["NI"], 16.5),
        )
        return result

    zero = run(0.0)
    huge = run(1.0)
    np.testing.assert_array_equal(zero["E_spk_bool"], huge["E_spk_bool"])
    np.testing.assert_array_equal(zero["rate_E"], huge["rate_E"])


def test_dynamic_threshold_is_live_on_the_conductance_path():
    def run(use_phi):
        p, net = _build(T=120.0)
        slow = _slow(p, net, use_phi=use_phi)
        result = simulate_kick(
            p,
            net,
            5.0,
            slow=slow,
            kick_center=np.array([0.5, 0.5]),
            r_kick=0.35,
            t_kick=20.0,
            V_th_per_neuron=np.full(net["NE"] + net["NI"], 16.0),
        )
        return result, slow

    off, slow_off = run(False)
    on, slow_on = run(True)
    assert slow_on.trace_phi_max and max(slow_on.trace_phi_max) > 0.0
    assert on["E_spk_bool"].sum() < off["E_spk_bool"].sum()
    assert slow_off.trace_phi_max == []


def test_conductance_checkpoint_resume_is_bit_exact():
    fork_step = 400
    p, net = _build(T=80.0)
    slow = _slow(p, net, use_phi=True)
    recorder = LFPRecorder(
        p, net["pos"], net["labels"], sites=np.array([[0.5, 0.5]])
    )
    full_ck = ZMCheckpoint(snapshot_steps=[fork_step], dump_ext=True)
    full = simulate_kick(
        p,
        net,
        4.0,
        slow=slow,
        kick_center=np.array([0.5, 0.5]),
        r_kick=0.3,
        t_kick=20.0,
        V_th_per_neuron=np.full(net["NE"] + net["NI"], 16.5),
        lfp_recorder=recorder,
        zm_ckpt=full_ck,
    )

    p2, net2 = _build(T=40.0)
    slow2 = _slow(p2, net2, use_phi=True)
    recorder2 = LFPRecorder(
        p2, net2["pos"], net2["labels"], sites=np.array([[0.5, 0.5]])
    )
    resume_ck = ZMCheckpoint(
        initial_state=full_ck.snapshots[fork_step], dump_ext=True
    )
    resumed = simulate_kick(
        p2,
        net2,
        4.0,
        slow=slow2,
        kick_center=np.array([0.5, 0.5]),
        r_kick=0.3,
        t_kick=20.0,
        V_th_per_neuron=np.full(net2["NE"] + net2["NI"], 16.5),
        lfp_recorder=recorder2,
        zm_ckpt=resume_ck,
    )
    for key in (
        "E_spk_bool",
        "rate_E",
        "rate_I",
        "lfp_trace",
        "lfp_current_proxy_trace",
        "zm_ext_nu",
        "zm_ext_sum",
    ):
        np.testing.assert_array_equal(resumed[key], full[key][fork_step:])
