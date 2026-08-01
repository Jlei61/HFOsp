"""FCXR-LC2 local-H vertical-slice contracts.

These tests intentionally stay below the scientific parameter-selection layer.  They pin the
only data flow that the Core sprint is allowed to test: post-X recurrent AMPA -> causal H state ->
pre-saturation recurrent conductance.  H disabled, or enabled as a sensor with rho=0, must not
change the membrane trajectory.
"""
import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from mz_slow_vars import MZSlowVars, MZSlowVarsConfig, lc2_h_gate  # noqa: E402
from params import Params  # noqa: E402
from connectivity import place_neurons, build_connectivity  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402


def _cfg(**kw):
    base = dict(
        membrane_mode="full_conductance",
        E_E=58.0,
        c_E=1.0,
        ff_conductance=False,
        rec_conductance=True,
        rec_sat_g=10.0,
        v_match=18.0,
        e_gaba=0.0,
        e_k=0.0,
        max_total_conductance=99.0,
        use_h_lc2=True,
        tau_h_lc2=100.0,
        theta_h_lc2=0.4,
        k_h_lc2=0.1,
        rho_h_lc2=2.0,
    )
    base.update(kw)
    return MZSlowVarsConfig(**base)


def _slow(cfg=None, *, snapshot_steps=None):
    return MZSlowVars(6, 18.0, cfg or _cfg(), NE=4,
                      core_mask_E=np.zeros(4, bool), snapshot_steps=snapshot_steps)


def _inputs(raw=(1.0, 0.5, 0.0, 2.0)):
    """At c_E=1 and E_E-v_match=40, recurrent currents 40*raw map exactly to gA_raw."""
    rec = np.zeros(6, float)
    rec[:4] = 40.0 * np.asarray(raw, float)
    total = rec.copy()
    inh = np.zeros(6, float)
    return total, inh, rec


def test_lc2_h_gate_is_exactly_zero_at_zero_and_is_smooth_bounded_monotone():
    h = np.array([0.0, 0.1, 0.4, 1.0, 10.0])
    y = lc2_h_gate(h, theta=0.4, k=0.1)
    assert y[0] == 0.0
    assert np.all(np.isfinite(y))
    assert np.all((y >= 0.0) & (y <= 1.0))
    assert np.all(np.diff(y) > 0.0)


def test_lc2_h_uses_post_relay_raw_recurrent_conductance_as_its_only_source():
    mz = _slow(_cfg(rho_h_lc2=0.0))
    I_E, I_I, I_rec = _inputs(raw=(1.0, 0.5, 0.0, 2.0))
    mz.membrane_terms(I_E, I_I, I_E_rec=I_rec)
    np.testing.assert_array_equal(mz._h_source_lc2_E, np.array([1.0, 0.5, 0.0, 2.0]))

    # I_E_rec is already post-X in the blessed scatter/delay path.  Halving that incoming array must
    # halve the H source; neither total I_E nor the post-tanh recurrent conductance may leak into it.
    mz.membrane_terms(I_E, I_I, I_E_rec=0.5 * I_rec)
    np.testing.assert_array_equal(mz._h_source_lc2_E, np.array([0.5, 0.25, 0.0, 1.0]))


def test_lc2_h_is_one_step_causal_and_exact_exponential():
    mz = _slow()
    I_E, I_I, I_rec = _inputs(raw=(1.0, 1.0, 1.0, 1.0))

    _, g0, _ = mz.membrane_terms(I_E, I_I, I_E_rec=I_rec)
    expected_without_h = 10.0 * np.tanh(1.0 / 10.0)
    np.testing.assert_allclose(g0[:4], expected_without_h, atol=1e-14)
    np.testing.assert_array_equal(mz.h_lc2_E, np.zeros(4))

    dt = 10.0
    mz.step(np.zeros(6, bool), None, dt)
    expected_h = 1.0 - np.exp(-dt / 100.0)
    np.testing.assert_allclose(mz.h_lc2_E, expected_h, atol=1e-14)

    # Only the NEXT membrane evaluation may see the H just accumulated above.
    _, g1, _ = mz.membrane_terms(I_E, I_I, I_E_rec=I_rec)
    assert np.all(g1[:4] > g0[:4])


def test_lc2_h_sensor_only_is_membrane_identical_to_h_off():
    off_cfg = _cfg(use_h_lc2=False, rho_h_lc2=0.0)
    on_cfg = _cfg(use_h_lc2=True, rho_h_lc2=0.0)
    off = _slow(off_cfg)
    on = _slow(on_cfg)
    spk = np.zeros(6, bool)

    for raw in ((1.0, 0.5, 0.0, 2.0), (0.0, 3.0, 1.5, 0.1), (0.2, 0.2, 0.2, 0.2)):
        args = _inputs(raw)
        a = off.membrane_terms(*args[:2], I_E_rec=args[2])
        b = on.membrane_terms(*args[:2], I_E_rec=args[2])
        for xa, xb in zip(a, b):
            np.testing.assert_array_equal(xa, xb)
        off.step(spk, None, 0.1)
        on.step(spk, None, 0.1)

    assert np.any(on.h_lc2_E > 0.0)


def test_lc2_h_is_added_before_rc1_tanh_saturation():
    h0 = np.full(4, 0.8)
    mz = _slow(_cfg(h_lc2_init_E=h0, rho_h_lc2=2.0))
    I_E, I_I, I_rec = _inputs(raw=(1.0, 1.0, 1.0, 1.0))
    _, g_rel, _ = mz.membrane_terms(I_E, I_I, I_E_rec=I_rec)
    u = 1.0 + 2.0 * lc2_h_gate(h0, theta=0.4, k=0.1)
    expected = 10.0 * np.tanh(u / 10.0)
    np.testing.assert_allclose(g_rel[:4], expected, atol=1e-14)


def test_lc2_h_snapshot_can_initialize_deterministic_continuation():
    first = _slow(_cfg(rho_h_lc2=0.0), snapshot_steps={0: "fork"})
    args = _inputs(raw=(1.0, 0.5, 0.25, 2.0))
    first.membrane_terms(*args[:2], I_E_rec=args[2])
    first.step(np.zeros(6, bool), None, 7.0)
    snap = first.snapshots["fork"]
    np.testing.assert_array_equal(snap["h_E"], first.h_lc2_E)

    resumed = _slow(_cfg(rho_h_lc2=0.0, h_lc2_init_E=snap["h_E"]))
    for obj in (first, resumed):
        obj.membrane_terms(*args[:2], I_E_rec=args[2])
        obj.step(np.zeros(6, bool), None, 3.0)
    np.testing.assert_array_equal(first.h_lc2_E, resumed.h_lc2_E)


@pytest.mark.parametrize("updates,match", [
    ({"tau_h_lc2": 0.0}, "tau_h_lc2"),
    ({"k_h_lc2": 0.0}, "k_h_lc2"),
    ({"theta_h_lc2": -1.0}, "theta_h_lc2"),
    ({"rho_h_lc2": -1.0}, "rho_h_lc2"),
    ({"coop_A": 1.0, "coop_uc": 1.0, "coop_Kc": 1.0}, "mutually exclusive"),
])
def test_lc2_h_rejects_invalid_or_confounded_configs(updates, match):
    with pytest.raises(ValueError, match=match):
        _slow(_cfg(**updates))


def test_lc2_h_init_field_requires_correct_finite_nonnegative_shape():
    for bad in (np.ones(3), np.array([0.0, 1.0, np.nan, 2.0]), np.array([0.0, -1.0, 1.0, 2.0])):
        with pytest.raises(ValueError, match="h_lc2_init_E"):
            _slow(_cfg(h_lc2_init_E=bad))


def _run_engine(*, use_h, rho, T, seed=17):
    """Small engine-level vertical slice; the scientific 40k substrate is deliberately not used in R0."""
    p = Params(L=6.0, density=100.0, T=T, dt=0.1, nu_ext_ratio=0.9, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    net["rng"] = np.random.default_rng(seed)
    vth = np.full(NE + NI, 18.0)
    vth[:5] = 16.0
    cfg = MZSlowVarsConfig(
        membrane_mode="full_conductance", E_E=58.0, c_E=1.0,
        ff_conductance=False, rec_conductance=True, rec_sat_g=21.6,
        v_match=18.0, e_gaba=0.0, e_k=0.0,
        max_total_conductance=99.0, fail_on_clip=False,
        use_h_lc2=use_h, tau_h_lc2=80.0, theta_h_lc2=0.02,
        k_h_lc2=0.01, rho_h_lc2=rho,
    )
    slow = MZSlowVars(NE + NI, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    res = simulate_kick(
        p, net, KICK_BOOST=4.0, slow=slow, kick_center=np.array([3.0, 3.0]),
        r_kick=0.5, t_kick=50.0, V_th_per_neuron=vth)
    return res, slow, net["rng"].bit_generator.state


def test_lc2_h_sensor_only_engine_raster_and_rng_are_byte_identical():
    off, _, rng_off = _run_engine(use_h=False, rho=0.0, T=100.0)
    sensor, slow, rng_sensor = _run_engine(use_h=True, rho=0.0, T=100.0)
    np.testing.assert_array_equal(off["E_spk_bool"], sensor["E_spk_bool"])
    np.testing.assert_array_equal(off["rate_E"], sensor["rate_E"])
    np.testing.assert_array_equal(off["rate_I"], sensor["rate_I"])
    assert rng_off == rng_sensor
    assert np.any(np.asarray(slow.trace_h_lc2_max) > 0.0)


def test_lc2_h_active_500ms_smoke_is_deterministic_finite_and_bounded():
    a, sa, _ = _run_engine(use_h=True, rho=0.2, T=500.0)
    b, sb, _ = _run_engine(use_h=True, rho=0.2, T=500.0)
    np.testing.assert_array_equal(a["E_spk_bool"], b["E_spk_bool"])
    np.testing.assert_array_equal(a["rate_E"], b["rate_E"])
    assert np.all(np.isfinite(a["rate_E"]))
    assert max(sa.trace_gH_lc2_max) > 0.0
    assert max(sa.trace_conductance_clip_frac) == 0.0
    np.testing.assert_array_equal(sa.h_lc2_E, sb.h_lc2_E)
