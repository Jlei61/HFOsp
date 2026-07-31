"""Unit and sign contracts for the Phase-D Z/M conductance membrane."""
from __future__ import annotations

import numpy as np
import pytest

from src.snn_engine.zm_conductance import (
    ZMConductanceConfig,
    analytic_anchor,
    conductance_currents,
    conductance_membrane_step,
    decompose_conductances,
    distribution_magnitude_anchor,
)


def _cfg(*, gamma=0.0, z_spares_global=False):
    return analytic_anchor(
        V_ref=15.0,
        V_th_median=18.0,
        V_reset=11.0,
        eta_m=0.001,
        gamma=gamma,
        z_spares_global=z_spares_global,
    )


def test_analytic_anchor_matches_current_rhs_at_reference_voltage():
    cfg = _cfg()
    is_E = np.array([True, True, False])
    V = np.full(3, 15.0)
    I_E = np.array([7.0, 5.0, 4.0])
    I_I = np.array([3.0, 2.0, 1.5])
    z = np.array([0.8, 0.6, 1.0])
    m = np.array([12.0, 20.0, 0.0])
    dec = decompose_conductances(I_E, I_I, z, m, cfg, is_E)
    cur = conductance_currents(
        V,
        dec["g_E"],
        dec["g_I_eff"],
        dec["g_Mm"],
        cfg,
        is_E,
    )
    rhs = (
        cfg.g_L * (cfg.E_L - V)
        + cur["I_exc"]
        + cur["I_inh"]
        + cur["I_sahp"]
    )
    expected = -V + I_E - z * I_I - 0.001 * m
    np.testing.assert_allclose(rhs[is_E], expected[is_E], atol=1e-12)


def test_conductance_step_is_exact_and_preserves_current_based_I_cells():
    cfg = _cfg(gamma=1.0 / 6.0)
    is_E = np.array([True, True, False])
    V = np.array([14.0, 16.0, 13.0])
    I_E = np.array([6.0, 8.0, 5.0])
    I_I = np.array([2.0, 4.0, 1.0])
    z = np.array([0.9, 0.7, 1.0])
    m = np.array([10.0, 30.0, 0.0])
    decay = np.exp(-0.1 / np.array([20.0, 20.0, 10.0]))
    out = conductance_membrane_step(
        V, I_E, I_I, z, m, decay, is_E, cfg
    )
    expected_E = out["V_inf"][is_E] + (
        V[is_E] - out["V_inf"][is_E]
    ) * np.exp(-0.1 / out["tau_eff_ms"][is_E])
    np.testing.assert_allclose(out["V_next"][is_E], expected_E, atol=1e-13)
    i = 2
    current_target = I_E[i] - I_I[i]
    expected_I = current_target + (V[i] - current_target) * decay[i]
    assert out["V_next"][i] == expected_I


def test_more_gaba_lowers_vinf_above_reversal_and_shortens_tau():
    cfg = _cfg()
    is_E = np.array([True])
    V = np.array([15.0])
    decay = np.exp(-0.1 / np.array([20.0]))
    base = conductance_membrane_step(
        V, np.array([10.0]), np.array([1.0]), np.array([1.0]),
        np.array([0.0]), decay, is_E, cfg,
    )
    gaba = conductance_membrane_step(
        V, np.array([10.0]), np.array([3.0]), np.array([1.0]),
        np.array([0.0]), decay, is_E, cfg,
    )
    adap = conductance_membrane_step(
        V, np.array([10.0]), np.array([1.0]), np.array([1.0]),
        np.array([50.0]), decay, is_E, cfg,
    )
    assert base["V_inf"][0] > cfg.E_I
    assert gaba["V_inf"][0] < base["V_inf"][0]
    assert gaba["tau_eff_ms"][0] < base["tau_eff_ms"][0]
    assert adap["V_inf"][0] < base["V_inf"][0]
    assert adap["tau_eff_ms"][0] < base["tau_eff_ms"][0]


def test_z_depletion_disinhibits_and_global_budget_matches_uniform_state():
    is_E = np.array([True, True, True])
    I_E = np.full(3, 6.0)
    I_I = np.full(3, 3.0)
    m = np.zeros(3)
    z_full = np.ones(3)
    z_low = np.full(3, 0.4)
    local = decompose_conductances(
        I_E, I_I, z_full, m, _cfg(gamma=0.0), is_E
    )
    mixed = decompose_conductances(
        I_E, I_I, z_full, m, _cfg(gamma=1.0 / 6.0), is_E
    )
    np.testing.assert_allclose(local["g_I_eff"], mixed["g_I_eff"])
    depleted = decompose_conductances(
        I_E, I_I, z_low, m, _cfg(gamma=1.0 / 6.0), is_E
    )
    assert np.all(depleted["g_I_eff"] < mixed["g_I_eff"])


def test_local_and_global_differ_only_transversally_for_heterogeneous_input():
    is_E = np.ones(4, dtype=bool)
    I_E = np.full(4, 6.0)
    I_I = np.array([1.0, 2.0, 4.0, 5.0])
    z = np.ones(4)
    m = np.zeros(4)
    local = decompose_conductances(
        I_E, I_I, z, m, _cfg(gamma=0.0), is_E
    )
    global_only = decompose_conductances(
        I_E, I_I, z, m, _cfg(gamma=1.0), is_E
    )
    assert np.std(local["g_I_eff"]) > 0.0
    assert np.std(global_only["g_I_eff"]) == pytest.approx(0.0)
    assert np.mean(local["g_I_eff"]) == pytest.approx(
        np.mean(global_only["g_I_eff"])
    )


def test_local_only_z_sensitivity_spares_global_component():
    is_E = np.array([True, True])
    I_E = np.full(2, 6.0)
    I_I = np.array([2.0, 4.0])
    z = np.full(2, 0.5)
    m = np.zeros(2)
    primary = decompose_conductances(
        I_E, I_I, z, m,
        _cfg(gamma=0.5, z_spares_global=False), is_E,
    )
    sensitivity = decompose_conductances(
        I_E, I_I, z, m,
        _cfg(gamma=0.5, z_spares_global=True), is_E,
    )
    assert np.all(sensitivity["g_I_eff"] > primary["g_I_eff"])


def test_mV_drives_are_converted_and_global_is_an_instantaneous_mean():
    cfg = _cfg(gamma=0.25)
    is_E = np.ones(3, dtype=bool)
    I_E = np.array([5.0, 7.0, 9.0])
    I_I = np.array([2.0, 3.0, 5.0])
    z = np.array([0.5, 0.7, 0.9])
    out = decompose_conductances(I_E, I_I, z, np.zeros(3), cfg, is_E)
    np.testing.assert_allclose(out["g_E"], cfg.kappa_E * I_E)
    np.testing.assert_allclose(out["g_I_local"], cfg.kappa_I * I_I)
    assert out["g_I_global"] == pytest.approx(
        np.mean(cfg.kappa_I * I_I)
    )
    expected = z * (
        0.75 * cfg.kappa_I * I_I + 0.25 * out["g_I_global"]
    )
    np.testing.assert_allclose(out["g_I_eff"], expected)
    assert not hasattr(cfg, "tau_G")


def test_distribution_anchor_stays_positive_across_gaba_reversal():
    voltage = np.array([-2.0, 8.0, 10.0, 12.0, 14.0])
    cfg, diag = distribution_magnitude_anchor(
        V_free=voltage,
        V_th_median=18.0,
        V_reset=11.0,
        eta_m=0.001,
    )
    assert cfg.kappa_E > 0.0
    assert cfg.kappa_I > 0.0
    assert cfg.g_M > 0.0
    assert diag["signed_point_tangent_feasible_at_median"] is False
    assert diag["pointwise_sign_equivalence_claimed"] is False
    assert diag["fraction_V_above_EI"] == pytest.approx(0.4)
    assert cfg.kappa_I == pytest.approx(1.0 / np.median(np.abs(voltage - 11.0)))


def test_distribution_anchor_scale_panel_is_bounded_and_literal():
    voltage = np.array([8.0, 10.0, 12.0, 14.0])
    base, _ = distribution_magnitude_anchor(
        V_free=voltage,
        V_th_median=18.0,
        V_reset=11.0,
        eta_m=0.001,
    )
    scaled, _ = distribution_magnitude_anchor(
        V_free=voltage,
        V_th_median=18.0,
        V_reset=11.0,
        eta_m=0.001,
        scale_E=0.8,
        scale_I=1.2,
        scale_M=0.8,
    )
    assert scaled.kappa_E == pytest.approx(0.8 * base.kappa_E)
    assert scaled.kappa_I == pytest.approx(1.2 * base.kappa_I)
    assert scaled.g_M == pytest.approx(0.8 * base.g_M)


def test_distribution_anchor_rejects_invalid_baseline_samples():
    with pytest.raises(ValueError, match="one-dimensional"):
        distribution_magnitude_anchor(
            V_free=np.ones((2, 2)),
            V_th_median=18.0,
            V_reset=11.0,
            eta_m=0.001,
        )
    with pytest.raises(ValueError, match="reaches/exceeds"):
        distribution_magnitude_anchor(
            V_free=np.array([10.0, 25.0]),
            V_th_median=18.0,
            V_reset=11.0,
            eta_m=0.001,
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"V_ref": 11.0}, "V_ref"),
        ({"V_ref": 25.0}, "V_ref"),
        ({"gamma": -0.1}, "gamma"),
        ({"scale_E": 1.21}, "scale_E"),
    ],
)
def test_invalid_anchor_fails_closed(kwargs, match):
    args = dict(
        V_ref=15.0,
        V_th_median=18.0,
        V_reset=11.0,
        eta_m=0.001,
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        analytic_anchor(**args)


def test_config_rejects_nonfinite_or_negative_conductance_scale():
    with pytest.raises(ValueError, match="kappa_E"):
        ZMConductanceConfig(kappa_E=np.nan).validate()
    with pytest.raises(ValueError, match="g_M"):
        ZMConductanceConfig(g_M=-1.0).validate()
