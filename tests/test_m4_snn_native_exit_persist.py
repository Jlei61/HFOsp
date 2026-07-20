"""TDD for the persistence-gated recovery field p(x,t) — SNN-native M4 exit line (spec 2026-07-21 §5).

Contract clauses (each -> one test, deep-contract-verify ritual):
  1. off-by-default byte-parity  : use_persist=False OR eta_r=0 -> E_spk_bool identical (BASELINE_SHA gate)
  2. duration selectivity        : tau_p >> event -> short burst charges p far less than sustained drive
  3. E-only local recovery current: apply_currents subtracts eta_r*Phi(p_E) on E cells only
  4. clamp_persist               : freezes p (open-loop probe / E4 ablation)
  5. validate                    : tau_p>0, a50_p>0, sigma_p>0, eta_r>=0, p_init/clamp in [0,1]
Plus a positive control (clamped high p + strong eta_r DOES change output) so parity tests are not vacuous.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params  # noqa: E402
from connectivity import place_neurons  # noqa: E402
from connectivity_rot import build_connectivity_rot  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402

DT = 0.1


def _net(L=6.0, T=200.0, seed=1, density=100.0, nu=0.6):
    p = Params(L=L, density=density, T=T, dt=DT, nu_ext_ratio=nu, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=np.radians(45), AR=2.0)
    return p, net, NE, NI


def _slow(p, net, **cfgkw):
    posE = net["pos"][net["labels"] == 0]
    posI = net["pos"][net["labels"] == 1]
    cfg = SpatialSlowFieldConfig(n_grid=8, **cfgkw)
    return SpatialSlowField(net["NE"] + net["NI"], p.V_th, posE, posI, p.L, cfg=cfg)


def _run(p, net, slow, kick=6.0, seed=7):
    """Reset net rng to a fixed seed so paired runs share the noise realization, then kick-sim."""
    net["rng"] = np.random.default_rng(seed)
    return simulate_kick(p, net, kick, slow=slow)


# ---- Clause 1: off-by-default byte-parity -------------------------------------------------
def test_persist_off_byte_parity():
    """use_persist=False with persist params set == a plain config (E_spk_bool identical)."""
    p, net, NE, NI = _net()
    res_a = _run(p, net, _slow(p, net, use_qI=True, k_q=0.1, use_SG=True, alpha_G=16.0))
    res_b = _run(p, net, _slow(p, net, use_qI=True, k_q=0.1, use_SG=True, alpha_G=16.0,
                               use_persist=False, tau_p=3000.0, theta_p=0.05, eta_r=5.0))
    assert np.array_equal(res_a["E_spk_bool"], res_b["E_spk_bool"])


def test_persist_eta0_byte_parity():
    """use_persist=True but eta_r=0 (p evolves, no membrane coupling) == use_persist=False."""
    p, net, NE, NI = _net()
    res_a = _run(p, net, _slow(p, net, use_qI=True, k_q=0.1))
    res_b = _run(p, net, _slow(p, net, use_qI=True, k_q=0.1,
                               use_persist=True, tau_p=100.0, theta_p=0.0, eta_r=0.0))
    assert np.array_equal(res_a["E_spk_bool"], res_b["E_spk_bool"])


# ---- Positive control: coupling is live (parity tests not vacuous) -------------------------
def test_persist_on_changes_output():
    """Frozen high p + strong eta_r -> strong recovery current -> strictly fewer E spikes."""
    p, net, NE, NI = _net()
    res_off = _run(p, net, _slow(p, net))
    res_on = _run(p, net, _slow(p, net, use_persist=True, clamp_persist=0.9, eta_r=25.0))
    assert not np.array_equal(res_off["E_spk_bool"], res_on["E_spk_bool"])
    assert int(np.asarray(res_on["E_spk_bool"]).sum()) < int(np.asarray(res_off["E_spk_bool"]).sum())


# ---- Clause 3: E-only local recovery current ----------------------------------------------
def test_recovery_current_E_only():
    p, net, NE, NI = _net()
    slow = _slow(p, net, use_persist=True, eta_r=2.0, theta_p=0.0)   # linear Phi (p50_r default 0)
    slow.p[:] = 0.5
    I_E = np.full(NE + NI, 3.0)
    I_I = np.full(NE + NI, 1.0)
    out = slow.apply_currents(I_E, I_I)
    assert np.allclose(out[:NE], 3.0 - 1.0 - 2.0 * 0.5)    # E: I_E - qI(=1)*I_I - eta_r*Phi(0.5)
    assert np.allclose(out[NE:], 3.0 - 1.0)                # I cells untouched


# ---- Clause 2: duration selectivity -------------------------------------------------------
def test_duration_selectivity():
    p, net, NE, NI = _net()
    slow = _slow(p, net, use_persist=True, tau_p=50.0, theta_p=0.0, a50_p=0.5, eta_r=0.0)
    spk = np.zeros(NE + NI, bool)
    spk[:NE] = True                                        # sustained supra-theta drive
    for _ in range(50):                                    # 5 ms << tau_p=50 ms
        slow.step(spk, net["labels"], DT)
    p_short = float(slow.p.mean())
    for _ in range(5000):                                  # +500 ms >> tau_p -> saturate
        slow.step(spk, net["labels"], DT)
    p_long = float(slow.p.mean())
    assert p_short > 0.0
    assert p_long > 5.0 * p_short                          # sustained accumulates far more than a brief burst


# ---- Clause 4: clamp ----------------------------------------------------------------------
def test_clamp_persist_frozen():
    p, net, NE, NI = _net()
    slow = _slow(p, net, use_persist=True, clamp_persist=0.3, eta_r=1.0)
    spk = np.zeros(NE + NI, bool)
    spk[:NE] = True
    for _ in range(100):
        slow.step(spk, net["labels"], DT)
    assert np.allclose(slow.p, 0.3)                        # frozen despite activity


# ---- Clause 5: validate -------------------------------------------------------------------
def test_validate_persist():
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(use_persist=True, tau_p=0.0).validate()
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(use_persist=True, sigma_p=0.0).validate()
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(use_persist=True, a50_p=0.0).validate()
    with pytest.raises(ValueError):
        SpatialSlowFieldConfig(use_persist=True, clamp_persist=1.5).validate()
    SpatialSlowFieldConfig(use_persist=True, tau_p=5000.0, sigma_p=1.5, a50_p=1.0).validate()  # ok
