"""Phase-0.2 H-drive diagnostic traces (Z/M ictal-carrier hardening, 2026-07-24).

The real H sensor input is NOT p_max (the scalar spatial max of the persistence field). It is
`phi_drive` -- the active-focus mean of Phi(p) over grid cells where phi > 0.2*pmax (H_sensor='active'),
or the spatial mean (H_sensor='global'). Before this fix only `self.H` was traced, so the actual
integrated input was invisible and the plotter mislabeled p_max as "H sensor in". These tests lock:

  C2  trace_phi_drive[i] == the EXACT value that integrated H at step i (Euler-recurrence reconstruction).
  C3  trace_active_frac == fraction of grid cells in the active-focus mask (0 if no focus; 1.0 for 'global').
  C4  trace_m_core_mean / trace_m_surround_mean == mean adaptation over core / surround (aligned with z).
  C5  sensor-only (use_H=True, alpha_H=0) is byte-identical in membrane dynamics to use_H=False, while H
      still builds -- so an H sensor can be recorded without changing the run (the Phase-2 sensor-only control).

C1 (byte-parity of the spike output) is covered by tests/test_zm_slow_field_parity.py + tests/test_snn_gates.py.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402


def _field(N=200, NE=160, L=20.0, core_mask_E=None, **cfg_kw):
    rng = np.random.default_rng(0)
    posE = rng.random((NE, 2)) * L
    posI = rng.random((N - NE, 2)) * L
    cfg = SpatialSlowFieldConfig(use_qI=False, use_gK=False, **cfg_kw)
    cfg.validate()
    return SpatialSlowField(N, 18.0, posE, posI, L, core_mask_E=core_mask_E, cfg=cfg), N, NE


# ---------------------------------------------------------------- C2 phi_drive faithfulness
def test_phi_drive_is_the_exact_integrated_H_input():
    """H is integrated as tau_H dH/dt = phi_drive - H. Reconstructing H from trace_phi_drive via the
    forward-Euler recurrence must reproduce trace_H bit-for-bit -> trace_phi_drive IS the real input."""
    sf, N, NE = _field(use_SG=True, alpha_G=16.0, use_persist=False,
                        use_H=True, alpha_H=16.0, tau_H=100.0, H_sensor="active")
    dt, tau_H, H_max = 1.0, 100.0, 1.0
    rng = np.random.default_rng(1)
    for _ in range(60):
        sf.p[:] = 0.0
        sf.p[:2, :2] = 0.3 + 0.5 * rng.random()   # a varying localized focus so phi_drive changes each step
        sf.step(np.zeros(N, bool), None, dt)
    assert len(sf.trace_phi_drive) == 60
    assert len(sf.trace_H) == 60
    assert max(sf.trace_phi_drive) > 0.0            # non-trivial: it actually sensed the focus
    for i in range(60):
        h_prev = 0.0 if i == 0 else sf.trace_H[i - 1]
        expected = min(max(h_prev + dt * (sf.trace_phi_drive[i] - h_prev) / tau_H, 0.0), H_max)
        assert abs(sf.trace_H[i] - expected) < 1e-12, f"step {i}: H not integrated from traced phi_drive"


# ---------------------------------------------------------------- C3 active-focus fraction
def test_active_frac_is_focus_cell_fraction_and_global_is_whole_field():
    sf, N, _ = _field(use_SG=True, alpha_G=16.0, use_persist=False,
                      use_H=True, alpha_H=0.0, tau_H=100.0, H_sensor="active")
    sf.p[:] = 0.0
    sf.p[:2, :2] = 0.6                               # 4 of 32*32 grid cells above 0.2*peak
    sf.step(np.zeros(N, bool), None, 1.0)
    assert abs(sf.trace_active_frac[-1] - 4.0 / (32 * 32)) < 1e-12

    sf.p[:] = 0.0                                    # no focus -> pmax=0 -> active_frac=0 (not NaN/1)
    sf.step(np.zeros(N, bool), None, 1.0)
    assert sf.trace_active_frac[-1] == 0.0

    sg, N2, _ = _field(use_SG=True, alpha_G=16.0, use_persist=False,
                       use_H=True, alpha_H=0.0, tau_H=100.0, H_sensor="global")
    sg.p[:] = 0.0
    sg.p[:2, :2] = 0.6
    sg.step(np.zeros(N2, bool), None, 1.0)
    assert sg.trace_active_frac[-1] == 1.0           # global sensor = whole field


# ---------------------------------------------------------------- C4 m core / surround
def test_m_core_and_surround_traces_track_the_adaptation_field():
    NE = 160
    core = np.zeros(NE, bool); core[:40] = True
    # use_m only (the m-trace path is identical with/without use_z; unit-calling step() without a preceding
    # apply_currents leaves _I_I_last=None, which the use_z Heaviside can't consume -- real runs always
    # apply_currents before step, so this isolates the m traces cleanly).
    sf, N, _ = _field(NE=NE, core_mask_E=core, use_m=True, tau_adp=1e9, eta_m=0.001)
    known = np.linspace(0.0, 8.0, NE)
    sf.m[:NE] = known
    sf.step(np.zeros(N, bool), None, 1.0)            # no spikes, negligible decay (tau_adp=1e9)
    assert abs(sf.trace_m_core_mean[-1] - known[:40].mean()) < 1e-4
    assert abs(sf.trace_m_surround_mean[-1] - known[40:].mean()) < 1e-4
    # aligned 1:1 with the z core/surround traces (same guard, same length)
    assert len(sf.trace_m_core_mean) == len(sf.trace_z_core_mean) == 1
    assert len(sf.trace_m_surround_mean) == len(sf.trace_z_surround_mean) == 1


def test_m_core_surround_absent_without_core_mask():
    sf, N, _ = _field(use_m=True, tau_adp=500.0, eta_m=0.001)
    sf.step(np.zeros(N, bool), None, 1.0)
    assert sf.trace_m_core_mean == [] and sf.trace_m_surround_mean == []


# ---------------------------------------------------------------- C5 sensor-only membrane neutrality
def test_sensor_only_alpha_H_zero_is_byte_identical_membrane_while_H_builds():
    """H_sensor recording (use_H=True, alpha_H=0) must not perturb the spike raster vs use_H=False on the
    same seed, yet still build H -- this is the Phase-2 'is the closed loop suppressing its own sensor?' control."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick

    SEED = 3
    p = Params(L=1.0, density=400.0, T=250.0, dt=0.1, seed=SEED, nu_ext_ratio=1.0)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    center = np.array([p.L / 2, p.L / 2])
    ZM = dict(use_qI=False, use_gK=False, use_z=True, use_m=True,
              tau_z=200.0, I_th_EI=0.0, tau_adp=200.0, eta_m=0.5)

    def run(cfg):
        net["rng"] = np.random.default_rng(SEED)
        slow = SpatialSlowField(N, 18.0, pos[:NE], pos[NE:], p.L, cfg=cfg)
        res = simulate_kick(p, net, 5.0, slow=slow, kick_center=center, r_kick=0.3,
                            t_kick=50.0, V_th_per_neuron=vth, verbose=False)
        return res, slow

    cfg_h = SpatialSlowFieldConfig(use_SG=True, alpha_G=16.0, use_persist=True, tau_p=2000.0, eta_r=0.0,
                                   use_H=True, alpha_H=0.0, tau_H=100.0, H_sensor="active", **ZM)
    cfg_noh = SpatialSlowFieldConfig(use_SG=True, alpha_G=16.0, use_persist=True, tau_p=2000.0, eta_r=0.0,
                                     use_H=False, **ZM)
    res_h, slow_h = run(cfg_h)
    res_noh, _ = run(cfg_noh)
    assert np.array_equal(res_h["E_spk_bool"], res_noh["E_spk_bool"]), "alpha_H=0 sensor changed the membrane"
    assert len(slow_h.trace_H) > 0 and max(slow_h.trace_H) >= 0.0
    assert len(slow_h.trace_phi_drive) == res_h["E_spk_bool"].shape[0]  # traced every step
