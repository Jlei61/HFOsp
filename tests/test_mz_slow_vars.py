"""P0 TDD for M4-MZ per-neuron slow vars (src/snn_engine/mz_slow_vars.py).

Contract source: docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md §2/§3/§5.
Each test = one contract clause (deep-contract-verify ritual). z_i (inhibitory efficacy) and
m_i (spike-frequency adaptation) act ON E CELLS ONLY; both OFF -> byte-parity with slow=None.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402


def _mk(N=10, NE=8, core=(0, 1), **kw):
    """Small module with E cells [:NE], I cells [NE:], core = first E indices in `core`."""
    core_mask_E = np.zeros(NE, bool)
    for i in core:
        core_mask_E[i] = True
    cfg = MZSlowVarsConfig(**kw)
    return MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=core_mask_E)


# ---- clause 2 (user test 1, unit level): both-off apply_currents is exact I_E - I_I ----
def test_both_off_apply_currents_is_exact_parity():
    mz = _mk(use_z=False, use_m=False)
    I_E = np.arange(10, dtype=float) + 1.0
    I_I = np.arange(10, dtype=float) * 0.5
    out = mz.apply_currents(I_E, I_I, labels=None)
    assert np.array_equal(out, I_E - I_I)


# ---- clause 1 (user test 1, engine level): full simulate_kick both-off == slow=None ----
def test_engine_byte_parity_both_off_equals_slow_none():
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick

    SEED = 1
    p = Params(L=1.0, density=400.0, T=200.0, dt=0.1, seed=SEED, nu_ext_ratio=1.0)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    N = NE + NI
    vth = np.full(N, 18.0)
    vth[:5] = 16.0  # a few easy-firing E cells -> exercise the per-neuron threshold path
    center = np.array([p.L / 2, p.L / 2])

    def run(slow):
        net["rng"] = np.random.default_rng(SEED)  # identical noise realization
        return simulate_kick(p, net, 5.0, slow=slow, kick_center=center, r_kick=0.3,
                             t_kick=50.0, V_th_per_neuron=vth, verbose=False)

    core_mask_E = np.zeros(NE, bool)
    mz = MZSlowVars(N, 18.0, MZSlowVarsConfig(use_z=False, use_m=False), NE=NE, core_mask_E=core_mask_E)
    res_none = run(None)
    res_mz = run(mz)
    assert np.array_equal(res_none["rate_E"], res_mz["rate_E"])
    assert np.array_equal(res_none["rate_I"], res_mz["rate_I"])
    assert np.array_equal(res_none["E_spk_bool"], res_mz["E_spk_bool"])   # full E raster bit-identical
    assert np.array_equal(res_none["spk_inside"], res_mz["spk_inside"])
    assert np.array_equal(res_none["spk_outside"], res_mz["spk_outside"])
    assert res_mz["E_spk_bool"].sum() > 0  # non-trivial: there WAS activity to disagree on


# ---- clause 3 (user test 2): z and m modify ONLY E cells ----
def test_z_and_m_modify_only_E_cells():
    mz = _mk(use_z=True, use_m=True, eta_m=0.3)
    mz.z[:mz.NE] = 0.5          # deplete E inhibition efficacy
    mz.m[:mz.NE] = 2.0          # E adaptation loaded
    I_E = np.ones(10); I_I = np.ones(10)
    out = mz.apply_currents(I_E, I_I, None)
    # E cells: 1 - 0.5*1 - 0.3*2 = -0.1 ; I cells: 1 - 1 = 0 (z==1, m==0 on I)
    assert np.allclose(out[:8], 1.0 - 0.5 * 1.0 - 0.3 * 2.0)
    assert np.allclose(out[8:], 1.0 - 1.0)


# ---- clause 4 (user test 3): I_I >= I_th_EI depletes z ----
def test_z_depletes_above_threshold():
    mz = _mk(use_z=True, I_th_EI=5.0, tau_z=5000.0)
    I_E = np.zeros(10); I_I = np.full(10, 10.0)  # I_I=10 >= 5 -> z_inf=0
    mz.apply_currents(I_E, I_I, None)            # store _I_I_last
    z_before = mz.z[:mz.NE].copy()
    mz.step(np.zeros(10, bool), None, 0.1)
    assert np.all(mz.z[:mz.NE] < z_before)       # z decreased on E cells


# ---- clause 5 (user test 4): I_I < I_th_EI recovers z toward 1 ----
def test_z_recovers_below_threshold():
    mz = _mk(use_z=True, I_th_EI=5.0, tau_z=1000.0)
    mz.z[:mz.NE] = 0.5
    I_E = np.zeros(10); I_I = np.full(10, 1.0)   # I_I=1 < 5 -> z_inf=1
    mz.apply_currents(I_E, I_I, None)
    mz.step(np.zeros(10, bool), None, 0.1)
    assert np.all(mz.z[:mz.NE] > 0.5)            # z increased toward 1


# ---- clause 5 (user test 5): z stays in [0,1] under extreme drive ----
def test_z_bounded_unit_interval():
    mz = _mk(use_z=True, I_th_EI=5.0, tau_z=50.0)
    for _ in range(3000):
        mz.apply_currents(np.zeros(10), np.full(10, 100.0), None)  # hammer depletion
        mz.step(np.zeros(10, bool), None, 0.1)
        assert np.all((mz.z >= 0.0) & (mz.z <= 1.0))
    for _ in range(3000):
        mz.apply_currents(np.zeros(10), np.zeros(10), None)        # full recovery
        mz.step(np.zeros(10, bool), None, 0.1)
        assert np.all((mz.z >= 0.0) & (mz.z <= 1.0))


# ---- clause: Heaviside is STRICT (I_I == I_th -> z_inf = 0, depletes) ----
def test_heaviside_strict_at_equality():
    mz = _mk(use_z=True, I_th_EI=5.0, tau_z=1000.0)
    mz.apply_currents(np.zeros(10), np.full(10, 5.0), None)  # I_I exactly == I_th
    mz.step(np.zeros(10, bool), None, 0.1)
    assert np.all(mz.z[:mz.NE] < 1.0)                        # equality -> deplete (>= branch)


# ---- clause 6 (user test 6): E spike increments m by 1; decays by tau_adp otherwise ----
def test_m_increments_on_E_spike_and_decays():
    mz = _mk(use_m=True, tau_adp=2000.0, eta_m=1.0)
    spk = np.zeros(10, bool); spk[2] = True          # E cell 2 spikes
    mz.step(spk, None, 0.1)
    assert abs(mz.m[2] - 1.0) < 1e-9                  # +1 (decay of 0 is 0)
    assert mz.m[3] == 0.0                             # quiet E cell unchanged
    m2 = mz.m[2]
    for _ in range(500):
        mz.step(np.zeros(10, bool), None, 0.1)       # quiet -> decay
    assert mz.m[2] < m2


# ---- clause 7 (user test 7): I spike does NOT increment m ----
def test_I_spike_does_not_increment_m():
    mz = _mk(use_m=True, tau_adp=2000.0, eta_m=1.0)
    spk = np.zeros(10, bool); spk[9] = True          # I cell (index 9 >= NE=8) spikes
    mz.step(spk, None, 0.1)
    assert np.all(mz.m == 0.0)                        # no E spike -> m stays 0 everywhere


# ---- clause 8 (user test 8): m-only adds ONLY a subtractive adaptation current ----
def test_m_only_subtractive_no_inhibition_scaling():
    mz = _mk(use_z=False, use_m=True, eta_m=0.5)
    mz.m[:mz.NE] = 2.0
    I_E = np.ones(10); I_I = np.ones(10)
    out = mz.apply_currents(I_E, I_I, None)
    # z OFF -> inhibition unscaled (full I_I); E cells lose eta_m*m; I cells untouched
    assert np.allclose(out[:8], 1.0 - 1.0 - 0.5 * 2.0)
    assert np.allclose(out[8:], 1.0 - 1.0)


# ---- clause 9 (user test 9): z-only scales E inhibition only, no adaptation ----
def test_z_only_scales_inhibition_no_adaptation():
    mz = _mk(use_z=True, use_m=False, I_th_EI=5.0)
    mz.z[:mz.NE] = 0.25
    mz.m[:mz.NE] = 9.0                                # should be IGNORED (use_m False)
    I_E = np.ones(10); I_I = np.ones(10)
    out = mz.apply_currents(I_E, I_I, None)
    assert np.allclose(out[:8], 1.0 - 0.25 * 1.0)    # scaled inhibition, no m term
    assert np.allclose(out[8:], 1.0 - 1.0)


# ---- clause 10 (user test 10): threshold() returns per-neuron V_th unchanged ----
def test_threshold_passthrough_per_neuron():
    mz = _mk(use_z=True, use_m=True)
    vth = np.linspace(16.0, 18.0, 10)                # heterogeneous double-core proxy
    out = mz.threshold(vth)
    assert np.array_equal(out, vth)                  # identity -> core preserved
    assert mz.threshold(18.0) == 18.0                # scalar passthrough too


# ---- clause 11 (user test 11): identical config + inputs -> identical state ----
def test_reproducible_same_inputs():
    def run():
        mz = _mk(use_z=True, use_m=True, I_th_EI=5.0, tau_z=3000.0, tau_adp=1500.0, eta_m=0.2)
        rng = np.random.default_rng(0)
        for _ in range(400):
            I_I = rng.random(10) * 8.0
            mz.apply_currents(rng.random(10), I_I, None)
            mz.step(rng.random(10) < 0.15, None, 0.1)
        return mz.z.copy(), mz.m.copy(), np.array(mz.trace_z_mean)
    z1, m1, t1 = run()
    z2, m2, t2 = run()
    assert np.array_equal(z1, z2) and np.array_equal(m1, m2) and np.array_equal(t1, t2)


# ---- clause: m never negative ----
def test_m_nonnegative():
    mz = _mk(use_m=True, tau_adp=100.0, eta_m=1.0)
    for _ in range(2000):
        mz.step(np.zeros(10, bool), None, 0.1)       # only decay from 0
        assert np.all(mz.m >= 0.0)


# ---- clause: all required audit traces present with correct length ----
def test_audit_traces_present_and_lengths():
    mz = _mk(use_z=True, use_m=True, I_th_EI=5.0, eta_m=0.2)
    n = 50
    for _ in range(n):
        mz.apply_currents(np.ones(10), np.full(10, 6.0), None)
        mz.step(np.zeros(10, bool), None, 0.1)
    for name in ("trace_z_mean", "trace_z_min", "trace_z_core_mean", "trace_z_surround_mean",
                 "trace_m_mean", "trace_m_max", "trace_m_core_mean", "trace_m_surround_mean",
                 "trace_adap_current", "trace_rate_E", "trace_rate_I"):
        tr = getattr(mz, name)
        assert len(tr) == n, f"{name} length {len(tr)} != {n}"


# ---- clause: record_calib is a pure side-effect (does NOT break both-off parity) ----
def test_record_calib_side_effect_preserves_parity():
    edges = np.linspace(0.0, 20.0, 129)
    mz = _mk(use_z=False, use_m=False, record_calib=True, calib_hist_edges=edges)
    I_E = np.arange(10, dtype=float) + 1.0
    I_I = np.arange(10, dtype=float) * 0.5
    out = mz.apply_currents(I_E, I_I, None)
    assert np.array_equal(out, I_E - I_I)            # recording did not change the return
    mz.step(np.zeros(10, bool), None, 0.1)
    assert len(mz.calib_hist_I_EI) == 1             # E-cell inhibitory-current histogram captured
    assert len(mz.calib_hist_I_EE) == 1


# ==================== slow-state snapshot observer (design §4.3 / plan Task 2 / Gate B) ====================
# The observer copies z_E/m_E at registered INTEGER steps only, AFTER the slow update, storing an
# n_snapshots x NE payload (never n_steps x NE). Off by default -> exact simulation parity.

def _mk_snap(snapshot_steps=None, N=10, NE=8, core=(0, 1), **kw):
    core_mask_E = np.zeros(NE, bool)
    for i in core:
        core_mask_E[i] = True
    cfg = MZSlowVarsConfig(**kw)
    return MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=core_mask_E, snapshot_steps=snapshot_steps)


def test_snapshot_off_by_default_no_capture_counter_advances():
    mz = _mk_snap(snapshot_steps=None, use_z=True, I_th_EI=5.0)
    for _ in range(5):
        mz.apply_currents(np.zeros(10), np.full(10, 10.0), None)
        mz.step(np.zeros(10, bool), None, 0.1)
    assert mz.snapshots == {}
    assert mz.n_steps_run == 5                       # counter advances harmlessly even with no capture


def test_snapshot_captures_requested_steps_once_correct_shape():
    mz = _mk_snap(snapshot_steps={0: "a", 3: "b"}, use_z=True, use_m=True, I_th_EI=5.0, eta_m=0.1)
    for _ in range(5):
        mz.apply_currents(np.ones(10), np.full(10, 6.0), None)
        mz.step(np.zeros(10, bool), None, 0.1)
    assert set(mz.snapshots) == {"a", "b"}
    for lab, want_step in (("a", 0), ("b", 3)):
        snap = mz.snapshots[lab]
        assert snap["z_E"].shape == (mz.NE,) and snap["m_E"].shape == (mz.NE,)
        assert snap["step"] == want_step
        assert snap["captured_after_update"] is True


def test_snapshot_memory_is_n_snapshots_not_n_steps():
    mz = _mk_snap(snapshot_steps={10: "x", 100: "y"}, use_z=True, I_th_EI=5.0)
    for _ in range(200):
        mz.apply_currents(np.zeros(10), np.full(10, 10.0), None)
        mz.step(np.zeros(10, bool), None, 0.1)
    assert len(mz.snapshots) == 2                     # 200 steps but only 2 stored arrays


def test_snapshot_mean_matches_trace_at_step():      # pins step<->trace index => time = step*dt
    mz = _mk_snap(snapshot_steps={0: "a", 7: "b", 19: "c"}, use_z=True, I_th_EI=5.0, tau_z=200.0)
    rng = np.random.default_rng(0)
    for _ in range(25):
        mz.apply_currents(np.zeros(10), rng.random(10) * 20.0, None)
        mz.step(np.zeros(10, bool), None, 0.1)
    for lab, step in (("a", 0), ("b", 7), ("c", 19)):
        assert mz.snapshots[lab]["z_E"].mean() == mz.trace_z_mean[step]


def test_snapshot_z_bounds_and_m_nonneg():
    mz = _mk_snap(snapshot_steps={50: "s"}, use_z=True, use_m=True, I_th_EI=5.0, tau_z=50.0,
                  eta_m=0.1, tau_adp=100.0)
    spk = np.arange(10) < 3                            # a few E spikes -> load m
    for _ in range(100):
        mz.apply_currents(np.zeros(10), np.full(10, 100.0), None)
        mz.step(spk, None, 0.1)
    z = mz.snapshots["s"]["z_E"]; m = mz.snapshots["s"]["m_E"]
    assert np.all((z >= 0.0) & (z <= 1.0)) and np.all(m >= 0.0)


def test_snapshot_primary_zonly_has_m_zero():
    mz = _mk_snap(snapshot_steps={5: "s"}, use_z=True, use_m=False, I_th_EI=5.0)
    for _ in range(10):
        mz.apply_currents(np.zeros(10), np.full(10, 10.0), None)
        mz.step(np.ones(10, bool), None, 0.1)         # spikes present but use_m False -> m stays 0
    assert np.all(mz.snapshots["s"]["m_E"] == 0.0)


def test_snapshot_invalid_steps_raise():
    import pytest
    with pytest.raises(ValueError):
        _mk_snap(snapshot_steps={-1: "a"})            # negative step
    with pytest.raises(ValueError):
        _mk_snap(snapshot_steps={5.5: "a"})           # non-integer-valued step
    with pytest.raises(ValueError):
        _mk_snap(snapshot_steps={1: "dup", 2: "dup"})  # duplicate label


def test_snapshot_observer_does_not_perturb_engine_output():
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick

    SEED = 1
    p = Params(L=1.0, density=400.0, T=200.0, dt=0.1, seed=SEED, nu_ext_ratio=1.0)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    center = np.array([p.L / 2, p.L / 2])
    core_mask_E = np.zeros(NE, bool)

    def run(snapshot_steps):
        net["rng"] = np.random.default_rng(SEED)
        mz = MZSlowVars(N, 18.0,
                        MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=5.0, tau_z=500.0,
                                         tau_adp=500.0, eta_m=0.05),
                        NE=NE, core_mask_E=core_mask_E, snapshot_steps=snapshot_steps)
        res = simulate_kick(p, net, 5.0, slow=mz, kick_center=center, r_kick=0.3,
                            t_kick=50.0, V_th_per_neuron=vth, verbose=False)
        return res, mz

    res_off, _ = run(None)
    res_on, mz_on = run({0: "t0", 500: "t500", 1999: "tend"})   # 200ms/0.1 = 2000 steps -> last idx 1999
    assert np.array_equal(res_off["E_spk_bool"], res_on["E_spk_bool"])   # capture is a pure read
    assert np.array_equal(res_off["rate_E"], res_on["rate_E"])
    assert np.array_equal(res_off["rate_I"], res_on["rate_I"])
    assert set(mz_on.snapshots) == {"t0", "t500", "tend"}
    assert np.all(mz_on.z[NE:] == 1.0) and np.all(mz_on.m[NE:] == 0.0)   # I cells never modulated
    assert mz_on.snapshots["t500"]["z_E"].mean() == mz_on.trace_z_mean[500]  # index/time pin


# =====================================================================================
# FCXR pump-lifecycle Task 2 — off-by-default per-cell activity-dependent load / pump plugin.
# Contract: docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md §2 + plan §4.
# `u_pump_E` is an ACTIVITY-DEPENDENT INTRACELLULAR LOAD (Na/pump-inspired), never a Na concentration.
# =====================================================================================
import src.topic4_mz_fcxr_pump as PUMP  # noqa: E402


def _fc_pump(NE=4, N=6, **kw):
    """Minimal full_conductance module (the locked FCXR substrate shape) + pump overrides."""
    base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0,
                e_gaba=0.0, e_k=0.0, ff_conductance=False, rec_conductance=True, rec_sat_g=21.6,
                gaba_gain=1.125, max_total_conductance=99.0)
    base.update(kw)
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**base), NE=NE, core_mask_E=np.zeros(NE, bool))


def _fc_inputs(N=6, NE=4):
    I_E = np.linspace(1.0, 3.0, N)
    I_I = np.linspace(0.5, 1.5, N)
    I_E_rec = 0.6 * I_E
    return I_E, I_I, I_E_rec


# ---- clause 1: use_pump=False -> full-engine byte parity (no allocation, no float touched) ----
def test_pump_off_engine_byte_parity_equals_slow_none():
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick

    SEED = 1
    p = Params(L=1.0, density=400.0, T=200.0, dt=0.1, seed=SEED, nu_ext_ratio=1.0)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    center = np.array([p.L / 2, p.L / 2])

    def run(slow):
        net["rng"] = np.random.default_rng(SEED)
        return simulate_kick(p, net, 5.0, slow=slow, kick_center=center, r_kick=0.3,
                             t_kick=50.0, V_th_per_neuron=vth, verbose=False)

    mz = MZSlowVars(N, 18.0, MZSlowVarsConfig(use_pump=False), NE=NE, core_mask_E=np.zeros(NE, bool))
    res_none, res_off = run(None), run(mz)
    assert np.array_equal(res_none["E_spk_bool"], res_off["E_spk_bool"])
    assert np.array_equal(res_none["rate_E"], res_off["rate_E"])
    assert res_off["E_spk_bool"].sum() > 0
    assert mz.u_pump_E is None                                  # no state allocated when pump is off


# ---- clause 2: the pump acts on E cells only ----
def test_pump_acts_on_E_cells_only():
    NE, N = 4, 6
    mz = _fc_pump(NE, N, use_pump=True, pump_a_load=0.5, pump_tau_ms=500.0, pump_Imax=2.0,
                  pump_p0_E=np.zeros(NE))
    mz.u_pump_E[:] = 1.0                                        # phi=0.5 -> excess = 2*0.5 = 1.0
    I_E, I_I, I_E_rec = _fc_inputs(N, NE)
    drive_on, g_rel, g_rev = mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    mz0 = _fc_pump(NE, N, use_pump=False)
    drive_off, g_rel0, g_rev0 = mz0.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    assert np.allclose(drive_on[:NE] - drive_off[:NE], -1.0)
    assert np.array_equal(drive_on[NE:], drive_off[NE:])        # I cells untouched
    assert np.array_equal(g_rel, g_rel0) and np.array_equal(g_rev, g_rev0)   # pump is a CURRENT


# ---- clause 3: sensor-only evolves u but leaves the membrane byte-identical ----
def test_sensor_only_updates_load_but_membrane_output_unchanged():
    NE, N = 4, 6
    I_E, I_I, I_E_rec = _fc_inputs(N, NE)
    mz = _fc_pump(NE, N, use_pump=True, pump_sensor_only=True, pump_a_load=0.5, pump_tau_ms=500.0)
    mz0 = _fc_pump(NE, N, use_pump=False)
    d1, r1, v1 = mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    d0, r0, v0 = mz0.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    assert np.array_equal(d1, d0) and np.array_equal(r1, r0) and np.array_equal(v1, v0)
    spk = np.zeros(N, bool); spk[1] = True
    mz.step(spk, None, 0.05)
    assert mz.u_pump_E[1] == pytest.approx(0.5) and mz.u_pump_E[0] == 0.0   # u still evolves


# ---- clause 4: Imax>0 requires a finite per-E p0 field ----
def test_imax_positive_requires_finite_p0_field_of_length_NE():
    with pytest.raises(ValueError):
        _fc_pump(4, 6, use_pump=True, pump_a_load=0.5, pump_tau_ms=500.0, pump_Imax=2.0)
    with pytest.raises(ValueError):
        _fc_pump(4, 6, use_pump=True, pump_a_load=0.5, pump_tau_ms=500.0, pump_Imax=2.0,
                 pump_p0_E=np.zeros(3))
    with pytest.raises(ValueError):
        _fc_pump(4, 6, use_pump=True, pump_a_load=0.5, pump_tau_ms=500.0, pump_Imax=2.0,
                 pump_p0_E=np.array([0.1, np.nan, 0.1, 0.1]))


# ---- clause 5/6: membrane sees u(t-); the load update happens in step() AFTER the membrane ----
def test_membrane_uses_pre_step_load_and_step_applies_the_jump_after():
    NE, N = 4, 6
    p0 = np.zeros(NE)
    mz = _fc_pump(NE, N, use_pump=True, pump_a_load=0.7, pump_tau_ms=500.0, pump_Imax=3.0,
                  pump_p0_E=p0)
    I_E, I_I, I_E_rec = _fc_inputs(N, NE)
    u_before = mz.u_pump_E.copy()
    drive_a, _, _ = mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    assert np.array_equal(mz.u_pump_E, u_before)                # membrane_terms never mutates u
    assert np.allclose(drive_a[:NE] - 0.0, drive_a[:NE])        # u=0 -> phi=0 -> excess = -Imax*p0 = 0
    spk = np.ones(N, bool)
    mz.step(spk, None, 0.05)
    assert np.allclose(mz.u_pump_E, 0.7)                        # E spikes jumped the load
    # the NEXT membrane call sees the post-jump load -> excess = 3*phi(0.7)
    drive_b, _, _ = mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    assert np.allclose(drive_a[:NE] - drive_b[:NE], 3.0 * PUMP.pump_activation(0.7))


def test_load_update_uses_the_locked_discrete_form():
    NE, N = 4, 6
    mz = _fc_pump(NE, N, use_pump=True, pump_sensor_only=True, pump_a_load=0.4, pump_tau_ms=250.0)
    mz.u_pump_E[:] = 1.5
    spk = np.zeros(N, bool); spk[0] = True
    mz.step(spk, None, 0.05)
    expect = PUMP.step_spike_load(np.full(NE, 1.5), np.array([1, 0, 0, 0]),
                                  a_load=0.4, tau_N=250.0, dt=0.05)
    assert np.allclose(mz.u_pump_E, expect)


# ---- clause 7/8: compensation is -Imax*phi + Imax*p0, no rectification at the crossing ----
def test_p0_compensation_is_applied_without_rectification():
    NE, N = 4, 6
    p0 = np.full(NE, 0.5)
    mz = _fc_pump(NE, N, use_pump=True, pump_a_load=0.4, pump_tau_ms=500.0, pump_Imax=2.0,
                  pump_p0_E=p0)
    mz0 = _fc_pump(NE, N, use_pump=False)
    I_E, I_I, I_E_rec = _fc_inputs(N, NE)
    d0, _, _ = mz0.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    mz.u_pump_E[:] = 1.0                                        # phi = p0 = 0.5 -> excess 0
    d_eq, _, _ = mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    assert np.allclose(d_eq[:NE], d0[:NE])
    mz.u_pump_E[:] = 0.0                                        # phi=0 < p0 -> NEGATIVE excess
    d_lo, _, _ = mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    assert np.all(d_lo[:NE] > d0[:NE])                          # a rectifier would give equality
    assert np.allclose(d_lo[:NE] - d0[:NE], 2.0 * 0.5)


# ---- clause 9: the calibration observer is a pure read ----
def test_calibration_observer_does_not_change_spikes_and_accumulates_phi():
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick

    SEED = 1
    p = Params(L=1.0, density=400.0, T=100.0, dt=0.1, seed=SEED, nu_ext_ratio=1.0)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0

    def run(record):
        net["rng"] = np.random.default_rng(SEED)
        mz = MZSlowVars(N, 18.0,
                        MZSlowVarsConfig(use_pump=True, pump_sensor_only=True, pump_a_load=0.3,
                                         pump_tau_ms=500.0, pump_record_calibration=record),
                        NE=NE, core_mask_E=np.zeros(NE, bool))
        res = simulate_kick(p, net, 5.0, slow=mz, kick_center=np.array([p.L / 2, p.L / 2]),
                            r_kick=0.3, t_kick=50.0, V_th_per_neuron=vth, verbose=False)
        return res, mz

    res_a, mz_a = run(False)
    res_b, mz_b = run(True)
    assert np.array_equal(res_a["E_spk_bool"], res_b["E_spk_bool"])
    assert np.array_equal(mz_a.u_pump_E, mz_b.u_pump_E)
    assert mz_b.pump_phi_count == len(res_b["rate_E"])
    assert mz_b.pump_spike_count_E.sum() == res_b["E_spk_bool"].sum()
    assert mz_a.pump_phi_sum_E is None                          # not allocated when not calibrating


# ---- clause 10/11/12/13: scheduled interventions ----
def test_no_intervention_path_is_byte_identical():
    NE, N = 4, 6
    I_E, I_I, I_E_rec = _fc_inputs(N, NE)
    a = _fc_pump(NE, N, use_pump=True, pump_a_load=0.4, pump_tau_ms=500.0, pump_Imax=2.0,
                 pump_p0_E=np.zeros(NE))
    b = _fc_pump(NE, N, use_pump=True, pump_a_load=0.4, pump_tau_ms=500.0, pump_Imax=2.0,
                 pump_p0_E=np.zeros(NE), pump_interventions=[])
    for _ in range(5):
        da, _, _ = a.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
        db, _, _ = b.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
        assert np.array_equal(da, db)
        spk = np.ones(N, bool)
        a.step(spk, None, 0.05); b.step(spk, None, 0.05)
    assert np.array_equal(a.u_pump_E, b.u_pump_E)


def test_scheduled_current_knockout_zeroes_membrane_pump_but_keeps_load_dynamics():
    NE, N = 4, 6
    I_E, I_I, I_E_rec = _fc_inputs(N, NE)
    mz = _fc_pump(NE, N, use_pump=True, pump_a_load=0.4, pump_tau_ms=500.0, pump_Imax=2.0,
                  pump_p0_E=np.zeros(NE),
                  pump_interventions=[dict(step=3, kind="pump_current_knockout")])
    ref = _fc_pump(NE, N, use_pump=False)
    d_ref, _, _ = ref.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
    seen = []
    for _ in range(6):
        d, _, _ = mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
        seen.append(d[:NE].copy())
        mz.step(np.ones(N, bool), None, 0.05)
    assert not np.allclose(seen[2], d_ref[:NE])                 # pump still on at step 2
    assert np.allclose(seen[3], d_ref[:NE])                     # knocked out from step 3 onward
    assert np.allclose(seen[5], d_ref[:NE])
    assert mz.u_pump_E.max() > 0.4 * 5                          # load kept accumulating


def test_scheduled_load_reset_sets_u_to_the_supplied_field():
    NE, N = 4, 6
    base = np.array([0.05, 0.06, 0.07, 0.08])
    mz = _fc_pump(NE, N, use_pump=True, pump_sensor_only=True, pump_a_load=0.4, pump_tau_ms=500.0,
                  pump_interventions=[dict(step=2, kind="set_load", field=base)])
    for _ in range(2):
        mz.step(np.ones(N, bool), None, 0.05)
    assert mz.u_pump_E[0] > 0.7                                 # accumulated before the reset
    mz.step(np.ones(N, bool), None, 0.05)                       # step index 2 -> reset applies at end
    assert np.array_equal(mz.u_pump_E, base)


def test_scheduled_load_injection_changes_u_only_at_the_registered_step():
    NE, N = 4, 6
    inj = np.array([2.0, 2.0, 2.0, 2.0])
    mz = _fc_pump(NE, N, use_pump=True, pump_sensor_only=True, pump_a_load=0.0, pump_tau_ms=1e9,
                  pump_interventions=[dict(step=4, kind="set_load", field=inj)])
    for s in range(4):
        mz.step(np.zeros(N, bool), None, 0.05)
        assert np.all(mz.u_pump_E == 0.0), f"u changed before the registered step at {s}"
    mz.step(np.zeros(N, bool), None, 0.05)                      # step index 4
    assert np.array_equal(mz.u_pump_E, inj)
    mz.step(np.zeros(N, bool), None, 0.05)
    assert np.allclose(mz.u_pump_E, inj, atol=1e-9)             # not re-applied, just decaying slowly


def test_intervention_steps_must_be_integers_not_float_times():
    with pytest.raises(ValueError):
        _fc_pump(4, 6, use_pump=True, pump_sensor_only=True, pump_a_load=0.4, pump_tau_ms=500.0,
                 pump_interventions=[dict(step=2.5, kind="set_load", field=np.zeros(4))])
    with pytest.raises(ValueError):
        _fc_pump(4, 6, use_pump=True, pump_sensor_only=True, pump_a_load=0.4, pump_tau_ms=500.0,
                 pump_interventions=[dict(step=2, kind="not_a_kind")])


# ---- clause 14/15: existing Z/M/X behaviour and update ORDER are unchanged ----
def test_default_ZMX_behaviour_unchanged_by_the_pump_edit():
    mz = _mk(use_z=True, use_m=True, eta_m=0.3, I_th_EI=1.0, tau_z=100.0, tau_adp=100.0)
    assert mz.u_pump_E is None and mz.cfg.use_pump is False
    I_E = np.arange(10, dtype=float) + 1.0
    I_I = np.full(10, 2.0)
    out = mz.apply_currents(I_E, I_I, labels=None)
    assert np.array_equal(out, I_E - mz.z * I_I - mz._eta_full * mz.m)


def test_existing_ZMX_update_order_unchanged():
    """The engine order is membrane_terms(t) -> spikes(t) -> step(t). Z senses the SAME pre-z GABA
    the membrane used this step; m decays THEN takes the spike jump; x is snapshotted BEFORE its own
    update (so a spike never weakens its own send). This test recomputes all three independently and
    pins them step by step -- any reordering caused by the pump edit shows up as a value mismatch."""
    NE, N, dt = 4, 6, 0.05
    cfg = dict(use_z=True, I_th_EI=0.8, tau_z=200.0, use_m=True, tau_adp=150.0, eta_m=0.2,
               use_x=True, tau_y=120.0, tau_x=1000.0, x_min=0.1, y_gate=1.0, K_y=5.0, hill_n=4,
               z_scope="local_only", global_gaba_fraction=0.0)
    mz = _fc_pump(NE, N, **cfg)
    I_E, I_I, I_E_rec = _fc_inputs(N, NE)
    z = np.ones(NE); m = np.zeros(NE); y = np.zeros(NE); x = np.ones(NE)
    rng = np.random.default_rng(3)
    for t in range(25):
        spk = np.zeros(N, bool)
        spk[:NE] = rng.random(NE) < 0.4
        mz.membrane_terms(I_E, I_I, None, I_E_rec=I_E_rec)
        send_expected = x.copy()                                # x(t-) snapshot at scatter time
        mz.step(spk, None, dt)
        # --- reference update, in the locked order ---
        local = np.maximum(I_I[:NE], 0.0)                       # z_scope=local_only, gamma=0
        y = y * np.exp(-dt / 120.0)
        y[spk[:NE]] += 1000.0 / 120.0
        uu = np.maximum(y - 1.0, 0.0) ** 4
        x_inf = 1.0 - (1.0 - 0.1) * (uu / (5.0 ** 4 + uu))
        x = x + (x_inf - x) * (1.0 - np.exp(-dt / 1000.0))
        z_inf = (local < 0.8).astype(float)
        z = np.clip(z + (dt / 200.0) * (z_inf - z), 0.0, 1.0)
        m = np.maximum(m - (m / 150.0) * dt, 0.0)
        m[spk[:NE]] += 1.0
        assert np.allclose(mz.ee_relay_send, send_expected), f"relay send order broke at t={t}"
        assert np.allclose(mz.z[:NE], z), f"z order/value broke at t={t}"
        assert np.allclose(mz.m[:NE], m), f"m order/value broke at t={t}"
        assert np.allclose(mz.x_relay, x), f"x order/value broke at t={t}"
        assert np.allclose(mz.y, y), f"y order/value broke at t={t}"


# ---- clause 16: snapshots store landmark u_E vectors only (never N_cell x T) ----
def test_pump_snapshot_stores_only_landmark_load_vectors():
    NE, N = 4, 6
    mz = MZSlowVars(N, 18.0,
                    MZSlowVarsConfig(use_pump=True, pump_sensor_only=True, pump_a_load=0.5,
                                     pump_tau_ms=500.0, pump_record_calibration=True),
                    NE=NE, core_mask_E=np.zeros(NE, bool), snapshot_steps={1: "a", 4: "b"})
    for _ in range(6):
        mz.step(np.ones(N, bool), None, 0.05)
    assert set(mz.snapshots) == {"a", "b"}
    for lab in ("a", "b"):
        snap = mz.snapshots[lab]
        assert snap["u_E"].shape == (NE,)
        assert snap["pump_phi_sum_E"].shape == (NE,)
        assert isinstance(snap["pump_phi_count"], int)
        assert snap["pump_spike_count_E"].shape == (NE,)
    assert mz.snapshots["b"]["u_E"][0] > mz.snapshots["a"]["u_E"][0]
    # per-block mean phi is recoverable by differencing the cumulative sums (no per-step matrix)
    d_sum = mz.snapshots["b"]["pump_phi_sum_E"] - mz.snapshots["a"]["pump_phi_sum_E"]
    d_cnt = mz.snapshots["b"]["pump_phi_count"] - mz.snapshots["a"]["pump_phi_count"]
    assert d_cnt == 3 and np.all(d_sum > 0)
    assert len(mz.trace_u_mean) == 6 and len(mz.trace_pump_excess_mean) == 0   # sensor-only: no excess
