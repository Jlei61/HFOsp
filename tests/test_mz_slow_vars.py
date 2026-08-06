"""P0 TDD for M4-MZ per-neuron slow vars (src/snn_engine/mz_slow_vars.py).

Contract source: docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md §2/§3/§5.
Each test = one contract clause (deep-contract-verify ritual). z_i (inhibitory efficacy) and
m_i (spike-frequency adaptation) act ON E CELLS ONLY; both OFF -> byte-parity with slow=None.
"""
import os
import sys

import numpy as np

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
