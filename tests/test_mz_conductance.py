"""Contract tests for the off-by-default MZ conductance membrane.

The current MZ path remains the reference.  The conductance path converts the
received GABA current proxy and the M adaptation current into leak-relative
conductances by matching their instantaneous force at ``V_match``.
"""
import os
import sys

import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402


def _mk(N=6, NE=4, **kwargs):
    cfg = MZSlowVarsConfig(**kwargs)
    return MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))


def test_conductance_mode_is_off_by_default():
    mz = _mk()
    assert mz.uses_conductance_membrane() is False


def test_membrane_terms_match_old_force_at_reference_voltage():
    mz = _mk(
        membrane_mode="conductance",
        use_z=True,
        use_m=True,
        eta_m=0.4,
        gaba_gain=1.0,
        m_conductance_gain=1.0,
        global_gaba_fraction=0.0,
        v_match=18.0,
        e_gaba=11.0,
        e_k=0.0,
    )
    mz.z[:mz.NE] = np.array([0.25, 0.5, 0.75, 1.0])
    mz.m[:mz.NE] = np.array([1.0, 2.0, 3.0, 4.0])
    I_E = np.array([25.0, 24.0, 23.0, 22.0, 7.0, 8.0])
    I_I = np.array([8.0, 7.0, 6.0, 5.0, 2.0, 3.0])

    drive, g_rel, g_rev = mz.membrane_terms(I_E, I_I, labels=None)
    v = mz.cfg.v_match
    new_rhs = drive[:mz.NE] + g_rev[:mz.NE] - (1.0 + g_rel[:mz.NE]) * v
    old_rhs = (
        I_E[:mz.NE]
        - mz.z[:mz.NE] * I_I[:mz.NE]
        - mz.cfg.eta_m * mz.m[:mz.NE]
        - v
    )
    np.testing.assert_allclose(new_rhs, old_rhs, atol=1e-12)


def test_inhibitory_cells_remain_on_literal_current_path():
    mz = _mk(membrane_mode="conductance", use_z=True, use_m=True, eta_m=0.3)
    I_E = np.arange(6, dtype=float) + 2.0
    I_I = np.arange(6, dtype=float) * 0.5
    drive, g_rel, g_rev = mz.membrane_terms(I_E, I_I, labels=None)
    np.testing.assert_array_equal(drive[mz.NE:], I_E[mz.NE:] - I_I[mz.NE:])
    np.testing.assert_array_equal(g_rel[mz.NE:], 0.0)
    np.testing.assert_array_equal(g_rev[mz.NE:], 0.0)


def test_global_fraction_is_mean_preserving_when_z_off():
    I_E = np.zeros(6)
    I_I = np.array([1.0, 2.0, 5.0, 8.0, 0.0, 0.0])
    means = []
    for gamma in (0.0, 0.25, 1.0):
        mz = _mk(
            membrane_mode="conductance",
            use_z=False,
            global_gaba_fraction=gamma,
            gaba_gain=1.0,
            v_match=18.0,
            e_gaba=11.0,
        )
        _, g_rel, _ = mz.membrane_terms(I_E, I_I, labels=None)
        means.append(float(g_rel[:mz.NE].mean()))
    np.testing.assert_allclose(means, means[0], atol=1e-12)


def test_additive_global_mode_retains_local_and_adds_population_mean():
    I_E = np.zeros(6)
    I_I = np.array([1.0, 2.0, 5.0, 8.0, 0.0, 0.0])
    local = _mk(membrane_mode="conductance", use_z=False, global_gaba_fraction=0.0,
                global_gaba_mode="additive", gaba_gain=1.0, v_match=18.0, e_gaba=11.0)
    added = _mk(membrane_mode="conductance", use_z=False, global_gaba_fraction=0.25,
                global_gaba_mode="additive", gaba_gain=1.0, v_match=18.0, e_gaba=11.0)
    _, g0, _ = local.membrane_terms(I_E, I_I, labels=None)
    _, g1, _ = added.membrane_terms(I_E, I_I, labels=None)
    expected_increment = 0.25 * np.mean(I_I[:4]) / (18.0 - 11.0)
    np.testing.assert_allclose(g1[:4] - g0[:4], expected_increment)


def test_z_scope_total_and_local_only_are_distinct():
    I_E = np.zeros(6)
    I_I = np.array([1.0, 2.0, 5.0, 8.0, 0.0, 0.0])
    kwargs = dict(
        membrane_mode="conductance",
        use_z=True,
        global_gaba_fraction=0.5,
        gaba_gain=1.0,
        v_match=18.0,
        e_gaba=11.0,
    )
    total = _mk(z_scope="total", **kwargs)
    local = _mk(z_scope="local_only", **kwargs)
    total.z[:total.NE] = 0.25
    local.z[:local.NE] = 0.25
    _, gt, _ = total.membrane_terms(I_E, I_I, labels=None)
    _, gl, _ = local.membrane_terms(I_E, I_I, labels=None)
    assert np.all(gl[:local.NE] > gt[:total.NE])


def test_total_conductance_clip_is_finite_and_audited():
    mz = _mk(
        membrane_mode="conductance",
        use_z=False,
        gaba_gain=10.0,
        max_total_conductance=9.0,
        v_match=18.0,
        e_gaba=11.0,
    )
    I_E = np.zeros(6)
    I_I = np.array([1e5, 1e5, 1e5, 1e5, 0.0, 0.0])
    _, g_rel, g_rev = mz.membrane_terms(I_E, I_I, labels=None)
    assert np.all(np.isfinite(g_rel)) and np.all(np.isfinite(g_rev))
    assert np.all(g_rel[:mz.NE] <= 9.0)
    assert mz._clip_frac_last == 1.0


def test_scientific_mode_fails_instead_of_using_clipped_conductance():
    mz = _mk(
        membrane_mode="conductance",
        use_z=False,
        gaba_gain=10.0,
        max_total_conductance=9.0,
        fail_on_clip=True,
        v_match=18.0,
        e_gaba=11.0,
    )
    with pytest.raises(FloatingPointError, match="exceeded cap"):
        mz.membrane_terms(np.zeros(6), np.array([1e5, 1e5, 1e5, 1e5, 0.0, 0.0]))


def test_total_scope_z_sensor_uses_same_global_mixture_as_membrane():
    mz = _mk(
        membrane_mode="conductance",
        use_z=True,
        tau_z=10.0,
        I_th_EI=5.0,
        global_gaba_fraction=1.0,
        z_scope="total",
        v_match=18.0,
        e_gaba=11.0,
    )
    mz.membrane_terms(np.zeros(6), np.array([0.0, 10.0, 0.0, 10.0, 0.0, 0.0]))
    np.testing.assert_allclose(mz._z_sensor_last_E, 5.0)
    mz.step(np.zeros(6, bool), None, 1.0)
    np.testing.assert_allclose(mz.z[:mz.NE], 0.9)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(gaba_gain=float("nan")),
        dict(use_z=True, tau_z=0.0),
        dict(use_m=True, tau_adp=0.0),
        dict(use_phi=True, tau_phi=0.0),
        dict(max_total_conductance=float("nan")),
        dict(global_gaba_mode="unknown"),
    ],
)
def test_invalid_numeric_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        _mk(**kwargs)


def test_dynamic_threshold_is_optional_and_e_only():
    mz = _mk(use_phi=True, tau_phi=100.0, delta_phi=2.5)
    base = np.full(mz.N, 18.0)
    np.testing.assert_array_equal(mz.threshold(base), base)
    spk = np.zeros(mz.N, bool)
    spk[0] = True
    spk[-1] = True  # I spike must not load phi
    mz.step(spk, None, 0.1)
    out = mz.threshold(base)
    assert out[0] == 20.5
    assert np.all(out[1:] == 18.0)


def test_engine_conductance_path_runs_and_is_deterministic():
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick

    seed = 2
    p = Params(L=1.0, density=400.0, T=120.0, dt=0.1, seed=seed, nu_ext_ratio=1.0)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    n = NE + NI
    vth = np.full(n, 18.0)

    def run():
        net["rng"] = np.random.default_rng(seed)
        slow = MZSlowVars(
            n,
            18.0,
            MZSlowVarsConfig(
                membrane_mode="conductance",
                use_z=True,
                use_m=True,
                I_th_EI=5.0,
                eta_m=0.01,
                gaba_gain=0.5,
                global_gaba_fraction=1.0 / 6.0,
                max_total_conductance=99.0,
            ),
            NE=NE,
            core_mask_E=np.zeros(NE, bool),
        )
        return simulate_kick(
            p,
            net,
            2.0,
            slow=slow,
            kick_center=np.array([0.5, 0.5]),
            r_kick=0.3,
            t_kick=30.0,
            V_th_per_neuron=vth,
        )

    a = run()
    b = run()
    np.testing.assert_array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert np.all(np.isfinite(a["rate_E"]))
