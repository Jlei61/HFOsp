"""TDD contracts for M3A-v2.2 global inhibitory recovery h_G.

Encodes the §B6 / design-spec contracts: h_G off-by-default byte-parity, M/B/Pi sensors +
chi_G smooth-AND, h_G ODE (k_G=0 still decays) + hG_script clamp/surrogate, E-only coupling,
proxy Y=P_global-beta_G*h_G (X invariant), sustained ramp+HOLD drive, and the pilot's
(time,neuron) readout + fail-closed segmentation + paired/order-invariant RNG.

Spec:  docs/snn_core_model_equations.md §B6
Plan:  docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2.2-global-recovery-plan.md

Run the fast set with `pytest tests/test_m3a_v2_2_global_recovery.py -m "not slow"`.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.topic4_m3a_v2_2_sensors import (  # noqa: E402
    hill, global_M, global_B, global_participation, chi_G)
from src.topic4_m3a_v2_phenotype import proxy_phase_point, region_masks  # noqa: E402
from src.topic4_m3a_v2_2_protocol import ramp_hold_drive, ramp_release_drive  # noqa: E402


def _tiny_field(use_hG=False, **cfgkw):
    """Small deterministic SpatialSlowField for the field-level h_G tests."""
    rng = np.random.default_rng(0)
    L, nE, nI = 10.0, 12, 6
    posE = rng.uniform(0, L, (nE, 2)); posI = rng.uniform(0, L, (nI, 2))
    cfg = SpatialSlowFieldConfig(n_grid=8, use_hG=use_hG, **cfgkw)
    return SpatialSlowField(nE + nI, 18.0, posE, posI, L, cfg=cfg), nE, nI


# ===========================================================================
# Task 2 -- config h_G fields + validate (off-by-default)
# ===========================================================================
def test_config_hG_off_by_default():
    c = SpatialSlowFieldConfig()
    assert c.use_hG is False and c.hG_init == 0.0 and c.k_G == 0.0
    c.validate()  # locked defaults are valid


def test_config_rejects_tau_s_nonpositive():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(tau_s=0.0).validate()


def test_config_rejects_negative_kG_and_lambdaG():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(k_G=-1.0).validate()
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(lambda_G=-0.1).validate()


def test_config_rejects_nonpositive_hill_thresholds():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(M50=0.0).validate()


# ===========================================================================
# Task 3 -- sensors (M/B/Pi + chi_G smooth-AND)
# ===========================================================================
def test_sensor_hill_half_at_z50():
    assert abs(hill(2.0, 2.0, 4.0) - 0.5) < 1e-9
    assert hill(0.0, 1.0, 4.0) == 0.0
    assert hill(1e6, 1.0, 4.0) > 0.999


def test_sensor_participation_single_hot_low_uniform_high():
    n = 8
    hot = np.zeros((n, n)); hot[0, 0] = 1.0
    uni = np.ones((n, n))
    assert global_participation(hot) < 0.05           # ~1/N_x
    assert global_participation(uni) > 0.99           # ~1


def test_sensor_B_soft_between_0_and_1():
    field = np.linspace(-2, 2, 16).reshape(4, 4)
    b = global_B(field, r_A=0.0, Delta_A=1.0)
    assert 0.0 < b < 1.0


def test_chi_low_for_local_axial_high_for_global():
    n = 16
    local = np.zeros((n, n)); local[7:9, :] = 1.0     # thin axial stripe
    Ml, Bl, Pl = global_M(local), global_B(local, 0.0, 0.3), global_participation(local)
    glob = np.full((n, n), 1.0)                        # broad high recruitment
    Mg, Bg, Pg = global_M(glob), global_B(glob, 0.0, 0.3), global_participation(glob)
    chi_local = chi_G(Ml, Bl, Pl, 0.5, 0.5, 0.45, 4, 4, 4)
    chi_glob = chi_G(Mg, Bg, Pg, 0.5, 0.5, 0.45, 4, 4, 4)
    assert chi_local < 0.1 < chi_glob


# ===========================================================================
# Task 4 -- h_G state + apply_currents E-only coupling (hard-gated)
# ===========================================================================
def test_apply_currents_hG_subtracts_on_E_only():
    fld, nE, nI = _tiny_field(use_hG=True, eta_G=0.5)
    fld.h_G = 2.0                                       # external set
    I_E = np.ones(nE + nI) * 3.0; I_I = np.ones(nE + nI) * 1.0
    out = fld.apply_currents(I_E, I_I)
    # E cells: 3 - q_I(=1)*1 - eta_K*g_K(=0) - eta_G*h_G(=0.5*2) = 3-1-0-1 = 1
    assert np.allclose(out[:nE], 1.0)
    # I cells: 3 - 1 = 2 (h_G does NOT touch I)
    assert np.allclose(out[nE:], 2.0)


def test_apply_currents_etaG_zero_is_no_op():
    fld, nE, nI = _tiny_field(use_hG=True, eta_G=0.0)
    fld.h_G = 5.0
    I_E = np.ones(nE + nI) * 3.0; I_I = np.ones(nE + nI) * 1.0
    out = fld.apply_currents(I_E, I_I)
    assert np.allclose(out[:nE], 2.0)                  # 3 - 1 - 0 - 0


def test_apply_currents_hG_off_ignores_etaG_and_hG_init():
    # HARD gate: use_hG=False must zero h_G even with eta_G>0, hG_init>0, and an external h_G set.
    fld, nE, nI = _tiny_field(use_hG=False, eta_G=9.0, hG_init=1.0)
    fld.h_G = 2.0
    I_E = np.ones(nE + nI) * 3.0; I_I = np.ones(nE + nI) * 1.0
    out = fld.apply_currents(I_E, I_I)
    assert np.allclose(out[:nE], 2.0)                  # 3 - 1 - 0 - 0 (h_G HARD-gated off)
    assert np.allclose(out[nE:], 2.0)


# ===========================================================================
# Task 5 -- step h_G ODE + hG_script override + byte parity
# ===========================================================================
def test_step_hG_builds_then_decays_bounded():
    fld, nE, nI = _tiny_field(use_hG=True, k_G=5.0, tau_G=200.0, hG_max=1.0,
                              M50=1e-6, B50=1e-6, Pi50=1e-6)   # easy trigger
    spk = np.zeros(nE + nI, bool); spk[:nE] = True
    for _ in range(200):
        fld.step(spk, None, dt=0.1)
    assert 0.0 < fld.h_G <= 1.0                       # built, bounded
    quiet = np.zeros(nE + nI, bool)
    for _ in range(8000):                              # 800 ms quiet >> tau_G=200 -> decays well below 0.05
        fld.step(quiet, None, dt=0.1)
    assert fld.h_G < 0.05


def test_step_hG_larger_kG_builds_more():
    def peak(kG):
        fld, nE, nI = _tiny_field(use_hG=True, k_G=kG, tau_G=500.0,
                                  M50=1e-6, B50=1e-6, Pi50=1e-6)
        spk = np.zeros(nE + nI, bool); spk[:nE] = True
        for _ in range(100):
            fld.step(spk, None, 0.1)
        return fld.h_G
    assert peak(5.0) > peak(1.0)


def test_step_hG_kG_zero_still_decays():
    # k_G=0 means NO BUILD, not NO DECAY: a pre-set h_G must still relax via -h_G/tau_G.
    fld, nE, nI = _tiny_field(use_hG=True, k_G=0.0, hG_init=0.8, tau_G=100.0)
    quiet = np.zeros(nE + nI, bool)
    for _ in range(1000):
        fld.step(quiet, None, 0.1)
    assert fld.h_G < 0.8


def test_step_hG_script_overrides_ode():
    fld, nE, nI = _tiny_field(use_hG=True, k_G=9.0)
    fld.hG_script = lambda t: 0.7 if t >= 5.0 else 0.0   # onset-gated constant
    spk = np.zeros(nE + nI, bool); spk[:nE] = True
    fld.step(spk, None, 0.1)                              # t=0 -> script 0
    assert fld.h_G == 0.0
    for _ in range(100):
        fld.step(spk, None, 0.1)                          # t>5 -> script 0.7
    assert fld.h_G == 0.7                                 # exact, ODE skipped


def test_step_hG_no_nan_long_random():
    fld, nE, nI = _tiny_field(use_hG=True, k_G=3.0)
    rng = np.random.default_rng(1)
    for _ in range(3000):
        fld.step(rng.random(nE + nI) < 0.2, None, 0.1)
    assert np.isfinite(fld.h_G)


@pytest.mark.slow
def test_hG_offparity_byte_identical_to_slow_none():
    # Real parity harness from the v2.1 test module (_build/_run/_sha). use_hG=False must HARD-gate
    # h_G even with nonzero eta_G/k_G/hG_init.
    from tests.test_m3a_v2_spatial_slowvars import _build, _run, _sha
    p, net, NE, NI = _build()
    pos = net["pos"]; N = NE + NI
    vth = np.full(N, 18.0)
    res_none = _run(p, net, NE, NI, slow=None, V_th_per_neuron=vth)
    fld = SpatialSlowField(N, 18.0, pos[:NE], pos[NE:], p.L,
                           cfg=SpatialSlowFieldConfig(k_q=0.0, k_K=0.0, q_init=1.0,
                                                      use_hG=False, eta_G=9.0, k_G=9.0, hG_init=1.0))
    res_off = _run(p, net, NE, NI, slow=fld, V_th_per_neuron=vth)
    assert _sha(res_off) == _sha(res_none)


# ===========================================================================
# Task 6 -- lambda_G q-replenish (arm F isolation; primary arm E lambda_G=0)
# ===========================================================================
def test_lambdaG_zero_no_replenish_positive_refills_faster():
    def q_after(lamG):
        fld, nE, nI = _tiny_field(use_hG=True, k_q=0.0, lambda_G=lamG, k_G=0.0)
        fld.q_I[:] = 0.4                              # pre-depressed
        fld.hG_script = lambda t: 1.0                # h_G held high (overrides ODE)
        quiet = np.zeros(nE + nI, bool)
        for _ in range(500):
            fld.step(quiet, None, 0.1)
        return float(fld.q_I.mean())
    q_armE = q_after(0.0)                             # primary: no replenish term
    q_armF = q_after(1.0 / 250.0)                     # arm F: replenish on
    assert q_armF > q_armE                            # F refills q_I faster
    assert abs(q_armE - 0.4) < 1e-6                   # arm E: k_q=0 -> q_I unchanged


# ===========================================================================
# Task 7 -- proxy phase plane Y = P_global - beta_G*h_G (X invariant)
# ===========================================================================
def test_proxy_hG_lowers_Y_not_X():
    fld, nE, nI = _tiny_field(use_hG=True)
    masks = region_masks(fld.L, fld.cfg.n_grid, center=(5.0, 5.0),
                         u_axis=(1.0, 1.0), corridor_halfwidth=1.5)
    fld.h_G = 0.0
    X0, Y0 = proxy_phase_point(fld, masks, lgr=2.0, beta_K=1.0, beta_G=0.8)
    fld.h_G = 1.5
    X1, Y1 = proxy_phase_point(fld, masks, lgr=2.0, beta_K=1.0, beta_G=0.8)
    assert abs(X1 - X0) < 1e-9                        # X invariant under uniform h_G
    assert abs(Y1 - (Y0 - 0.8 * 1.5)) < 1e-9          # Y drops by beta_G*h_G


def test_proxy_betaG_default_zero_backcompat():
    fld, nE, nI = _tiny_field(use_hG=True); fld.h_G = 3.0
    masks = region_masks(fld.L, fld.cfg.n_grid, (5.0, 5.0), (1.0, 1.0), 1.5)
    X, Y = proxy_phase_point(fld, masks, 2.0, 1.0)    # beta_G defaults 0 -> old behavior
    Xb, Yb = proxy_phase_point(fld, masks, 2.0, 1.0, beta_G=0.0)
    assert (X, Y) == (Xb, Yb)


# ===========================================================================
# Task 8 -- sustained ramp+HOLD / release drive builders (nu_signal_fn)
# ===========================================================================
def test_drive_ramp_hold_shape():
    f = ramp_hold_drive(nu_theta=1.0, r0=0.2, r_hold=0.6, t0=100.0, t_ramp=200.0)
    assert abs(f(0.0) - 0.2) < 1e-9          # before ramp: r0
    assert abs(f(100.0) - 0.2) < 1e-9        # ramp start
    assert abs(f(200.0) - 0.4) < 1e-9        # mid-ramp: halfway
    assert abs(f(300.0) - 0.6) < 1e-9        # ramp end: r_hold
    assert abs(f(9999.0) - 0.6) < 1e-9       # HOLD: stays r_hold (never released)


def test_drive_release_drops_back():
    f = ramp_release_drive(1.0, 0.2, 0.6, t0=100.0, t_ramp=100.0, t_release=500.0)
    assert abs(f(300.0) - 0.6) < 1e-9        # holding
    assert abs(f(600.0) - 0.2) < 1e-9        # after release: back to r0
