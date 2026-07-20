"""FCXR (full-conductance + persistence-gated E->E relay) contract + re-bless gate.

Design: docs/superpowers/specs/2026-07-20-topic4-mz-full-conductance-spatial-relay-design.md.
Two families:
  (A) PARITY — the new paths are OFF unless requested; slow=None / membrane_mode='conductance' / M1
      ee_std must reproduce the PRE-EDIT baseline (tests/fixtures/fcxr_parity_baseline.pkl) byte-for-byte.
  (B) CAUSAL/CONTRACT — full-conductance force matching, the x_j(t-) causal send, x scales only E->E,
      relay<->M1 mutual exclusion, determinism/finiteness, relay-off identity.
Plus the engine-bless gate (kick_probe.py sha256 matches engine_versions.json).
"""
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from params import Params  # noqa: E402
from connectivity import place_neurons, build_connectivity  # noqa: E402
from kick_probe import simulate_kick, ee_std_apply  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402

FIXTURE = pickle.load(open(os.path.join(ROOT, "tests", "fixtures", "fcxr_parity_baseline.pkl"), "rb"))
ENGINE_VERSIONS = os.path.join(ROOT, "results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
SEED = 1
DT = 0.1


def _net():
    """Small recurrently-active substrate identical to the fixture-capture script."""
    p = Params(L=6.0, density=100.0, T=250.0, dt=DT, nu_ext_ratio=0.9, seed=SEED)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    return p, net, NE, NI


def _assert_parity(res, base, net):
    assert hashlib.sha1(res["E_spk_bool"].tobytes()).hexdigest()[:16] == base["spk_sha"]
    assert int(res["E_spk_bool"].sum()) == base["n_spk"]
    assert np.array_equal(res["rate_E"], base["rate_E"])
    assert np.array_equal(res["rate_I"], base["rate_I"])
    assert np.array_equal(res["spk_inside"], base["spk_inside"])
    assert np.array_equal(res["spk_outside"], base["spk_outside"])
    if base["lfp_trace"] is not None:
        assert np.array_equal(res["lfp_trace"], base["lfp_trace"])
    assert net["rng"].bit_generator.state == base["rng_state"]   # zero added RNG draws


# ============================== (A) PARITY vs PRE-EDIT baseline ==============================
def test_parity_slow_none():
    p, net, NE, NI = _net()
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=np.array([[2., 2.], [3., 3.], [4., 4.]]))
    vth = np.full(NE + NI, 18.0); vth[:5] = 16.0
    net["rng"] = np.random.default_rng(SEED)
    res = simulate_kick(p, net, KICK_BOOST=4.0, slow=None, kick_center=np.array([3., 3.]),
                        r_kick=0.5, t_kick=50.0, V_th_per_neuron=vth, lfp_recorder=rec)
    _assert_parity(res, FIXTURE["slow_none"], net)


def test_parity_partial_conductance():
    p, net, NE, NI = _net(); N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    cfg = MZSlowVarsConfig(
        membrane_mode="conductance", use_z=True, use_m=False,
        I_th_EI=6.0, tau_z=2500.0, gaba_gain=1.125,
        global_gaba_fraction=1.0 / 12.0, global_gaba_mode="additive", z_scope="local_only",
        v_match=18.0, e_gaba=0.0, e_k=0.0, max_total_conductance=99.0, fail_on_clip=True,
    )
    slow = MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    net["rng"] = np.random.default_rng(SEED)
    res = simulate_kick(p, net, KICK_BOOST=4.0, slow=slow, kick_center=np.array([3., 3.]),
                        r_kick=0.5, t_kick=50.0, V_th_per_neuron=vth)
    _assert_parity(res, FIXTURE["conductance"], net)


def test_parity_m1_ee_std():
    p, net, NE, NI = _net()
    vth = np.full(NE + NI, 18.0); vth[:5] = 16.0
    net["rng"] = np.random.default_rng(SEED)
    res = simulate_kick(p, net, KICK_BOOST=4.0, slow=None, kick_center=np.array([3., 3.]),
                        r_kick=0.5, t_kick=50.0, V_th_per_neuron=vth,
                        ee_std_u=0.2, ee_std_tau_ms=200.0)
    _assert_parity(res, FIXTURE["m1"], net)


# ============================== (B) full-conductance membrane ==============================
def _mk_fc(N=6, NE=4, **kw):
    base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0,
                e_gaba=0.0, e_k=0.0, max_total_conductance=99.0)
    base.update(kw)
    cfg = MZSlowVarsConfig(**base)
    return MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))


def test_full_conductance_force_match_at_v_match():
    """At V=v_match the exact-conductance RHS equals c_E*I_E - gaba_gain*I_inh_eff - v_match (m off)."""
    mz = _mk_fc(use_z=True, use_m=False, c_E=1.15, gaba_gain=1.125,
                global_gaba_fraction=0.0, z_scope="local_only")
    mz.z[:mz.NE] = np.array([0.25, 0.5, 0.75, 1.0])
    I_E = np.array([25.0, 24.0, 23.0, 22.0, 7.0, 8.0])
    I_E_rec = np.array([10.0, 8.0, 6.0, 4.0, 3.0, 3.0])          # 0 <= I_E_rec <= I_E
    I_I = np.array([8.0, 7.0, 6.0, 5.0, 2.0, 3.0])
    drive, g_rel, g_rev = mz.membrane_terms(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    v = mz.cfg.v_match
    new_rhs = drive[:mz.NE] + g_rev[:mz.NE] - (1.0 + g_rel[:mz.NE]) * v
    I_inh_eff = mz.z[:mz.NE] * I_I[:mz.NE]                        # local_only, global 0
    expected = mz.cfg.c_E * I_E[:mz.NE] - mz.cfg.gaba_gain * I_inh_eff - v
    np.testing.assert_allclose(new_rhs, expected, atol=1e-11)


def test_full_conductance_e_cells_have_no_additive_drive():
    """E cells: drive==0 (all excitation is conductance); I cells keep literal I_E-I_I."""
    mz = _mk_fc(use_z=False)
    I_E = np.array([20.0, 19.0, 18.0, 17.0, 6.0, 5.0])
    I_E_rec = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0])
    I_I = np.array([3.0, 2.0, 1.0, 0.5, 2.0, 1.0])
    drive, g_rel, g_rev = mz.membrane_terms(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    assert np.all(drive[:mz.NE] == 0.0)
    np.testing.assert_array_equal(drive[mz.NE:], I_E[mz.NE:] - I_I[mz.NE:])
    np.testing.assert_array_equal(g_rel[mz.NE:], 0.0)
    np.testing.assert_array_equal(g_rev[mz.NE:], 0.0)
    # AMPA conductance sum == c_E*I_E/(E_E-v_match): recover gE from g_rev (gI reversal e_gaba=0 -> gI term 0)
    gE = g_rev[:mz.NE] / mz.cfg.E_E
    np.testing.assert_allclose(gE, mz.cfg.c_E * I_E[:mz.NE] / (mz.cfg.E_E - mz.cfg.v_match), atol=1e-11)


@pytest.mark.parametrize("ff_cond,rec_cond", [(True, True), (True, False), (False, True), (False, False)])
def test_pathway_arms_share_v_match_force_anchor(ff_cond, rec_cond):
    """All ff/rec conductance-vs-additive arms force-match to the SAME thing at V_match
    (ampa_drive + gE*(E_E-V_match) == c_E*I_E); they differ only OFF V_match."""
    mz = _mk_fc(use_z=True, use_m=False, c_E=1.15, gaba_gain=1.125, global_gaba_fraction=0.0,
                z_scope="local_only", ff_conductance=ff_cond, rec_conductance=rec_cond)
    mz.z[:mz.NE] = np.array([0.25, 0.5, 0.75, 1.0])
    I_E = np.array([25.0, 24.0, 23.0, 22.0, 7.0, 8.0])
    I_E_rec = np.array([10.0, 8.0, 6.0, 4.0, 3.0, 3.0])
    I_I = np.array([8.0, 7.0, 6.0, 5.0, 2.0, 3.0])
    drive, g_rel, g_rev = mz.membrane_terms(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    v = mz.cfg.v_match
    new_rhs = drive[:mz.NE] + g_rev[:mz.NE] - (1.0 + g_rel[:mz.NE]) * v
    expected = mz.cfg.c_E * I_E[:mz.NE] - mz.cfg.gaba_gain * (mz.z[:mz.NE] * I_I[:mz.NE]) - v
    np.testing.assert_allclose(new_rhs, expected, atol=1e-11)


def test_arm_D_default_is_all_conductance_no_additive_drive():
    mz = _mk_fc(use_z=False)                                   # default ff=rec=True (arm D)
    I_E = np.array([12.0, 12.0, 12.0, 12.0, 4.0, 4.0]); I_E_rec = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0])
    drive, _, _ = mz.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_E_rec)
    assert np.all(drive[:mz.NE] == 0.0)                        # arm D: all AMPA is conductance


def test_arm_B_and_C_route_additive_to_drive():
    """Arm B (ff cond / rec additive) -> drive == c_E*I_rec; arm C (ff additive / rec cond) -> drive == c_E*I_ff."""
    I_E = np.array([12.0, 12.0, 12.0, 12.0, 4.0, 4.0]); I_E_rec = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 1.0])
    I_ff = I_E[:4] - I_E_rec[:4]
    b = _mk_fc(use_z=False, c_E=1.15, ff_conductance=True, rec_conductance=False)
    db, _, _ = b.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_E_rec)
    np.testing.assert_allclose(db[:4], 1.15 * I_E_rec[:4], atol=1e-11)
    c = _mk_fc(use_z=False, c_E=1.15, ff_conductance=False, rec_conductance=True)
    dc, _, _ = c.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_E_rec)
    np.testing.assert_allclose(dc[:4], 1.15 * I_ff, atol=1e-11)


def test_full_conductance_requires_and_gates_I_E_rec():
    mz = _mk_fc(use_z=False)
    with pytest.raises(ValueError, match="requires the recurrent AMPA"):
        mz.membrane_terms(np.zeros(6), np.zeros(6), labels=None)          # missing I_E_rec
    partial = MZSlowVars(6, 18.0, MZSlowVarsConfig(membrane_mode="conductance", v_match=18.0, e_gaba=0.0),
                         NE=4, core_mask_E=np.zeros(4, bool))
    with pytest.raises(ValueError, match="does not accept I_E_rec"):
        partial.membrane_terms(np.zeros(6), np.zeros(6), labels=None, I_E_rec=np.zeros(6))


def test_full_conductance_split_recurrent_only_reflects_I_E_rec():
    """g_E_rec diagnostic tracks I_E_rec; g_E_ff tracks the feedforward remainder."""
    mz = _mk_fc(use_z=False, c_E=1.0)
    I_E = np.array([12.0, 12.0, 12.0, 12.0, 4.0, 4.0])
    I_E_rec = np.array([12.0, 0.0, 6.0, 3.0, 2.0, 2.0])          # all-rec, all-ff, half, quarter
    I_I = np.zeros(6)
    mz.membrane_terms(I_E, I_I, labels=None, I_E_rec=I_E_rec)
    denom = mz.cfg.E_E - mz.cfg.v_match
    assert abs(mz._gErec_mean_last - float(np.mean(I_E_rec[:4]) / denom)) < 1e-11
    assert abs(mz._gEff_mean_last - float(np.mean((I_E[:4] - I_E_rec[:4])) / denom)) < 1e-11


def test_clip_identity_observer_is_pure_side_effect():
    """FCXR-RC1: record_clip_identity records WHICH cells clip without changing membrane_terms output."""
    def mk(rec):
        return _mk_fc(use_z=False, c_E=10.0, max_total_conductance=9.0, fail_on_clip=False,
                      record_clip_identity=rec)
    I_E = np.array([100.0, 5.0, 5.0, 5.0, 4.0, 4.0])
    I_E_rec = np.array([100.0, 0.0, 0.0, 0.0, 1.0, 1.0])            # E cell 0 has huge recurrent -> clips
    off = mk(False); on = mk(True)
    do = off.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_E_rec)
    dn = on.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_E_rec)
    for a, b in zip(do, dn):
        np.testing.assert_array_equal(a, b)                        # identical output -> pure side-effect
    assert not hasattr(off, "clip_count")                         # off-by-default: no allocation
    assert on.clip_count[0] == 1 and int(on.clip_count[1:].sum()) == 0   # only cell 0 clipped
    assert on.max_raw_gErec[0] > 9.0                              # recorded the raw pre-clip recurrent conductance
    assert on.first_clip_step[0] == 0 and on.last_clip_step[0] == 0


def test_full_conductance_cap_and_fail_on_clip():
    mz = _mk_fc(use_z=False, c_E=10.0, max_total_conductance=9.0, fail_on_clip=True)
    with pytest.raises(FloatingPointError, match="exceeded cap"):
        mz.membrane_terms(np.full(6, 1e5), np.zeros(6), labels=None, I_E_rec=np.full(6, 1e4))


# ============================== (B) persistence sensor + relay causality ==============================
def test_relay_snapshot_is_x_t_minus():
    """slow.step snapshots ee_relay_send = x_j(t-) BEFORE updating y/x; the current spike sends with it."""
    mz = _mk_fc(use_x=True, tau_y=120.0, tau_x=1000.0, x_min=0.0, y_gate=0.0, K_y=5.0, hill_n=4)
    mz.x_relay[:] = np.array([0.9, 0.8, 0.7, 0.6])
    pre = mz.x_relay.copy()
    spk = np.zeros(mz.N, bool); spk[0] = spk[1] = True           # E cells 0,1 fire this frame
    mz.step(spk, None, DT)
    np.testing.assert_array_equal(mz.ee_relay_send, pre)         # send scale is the PRE-update value
    assert not np.array_equal(mz.x_relay, pre)                   # x itself advanced (post-update)


def test_relay_spike_does_not_weaken_own_send():
    """From full availability, a firing cell's send is 1.0 (unweakened), while its NEXT send drops."""
    mz = _mk_fc(use_x=True, tau_y=120.0, tau_x=1000.0, x_min=0.0, y_gate=0.0, K_y=5.0, hill_n=4)
    spk = np.zeros(mz.N, bool); spk[0] = True
    mz.step(spk, None, DT)
    assert mz.ee_relay_send[0] == 1.0                            # the spike sent at full relay availability
    assert mz.x_relay[0] < 1.0                                   # but its subsequent availability is reduced
    assert mz.x_relay[2] == 1.0 and mz.x_relay[3] == 1.0         # quiet cells stay full


def test_relay_y_is_hz_lowpass_and_e_only():
    """y decays exp(-dt/tau_y) and jumps 1000/tau_y per E spike; I cells never own y."""
    mz = _mk_fc(N=6, NE=4, use_x=True, tau_y=120.0, tau_x=1e9, x_min=0.0, y_gate=1e9, K_y=5.0)
    spk = np.zeros(6, bool); spk[0] = True; spk[5] = True        # E cell 0 + I cell 5
    mz.step(spk, None, DT)
    assert abs(mz.y[0] - 1000.0 / 120.0) < 1e-9                  # E spike jump
    assert mz.y.shape == (4,)                                    # E-only sensor (length NE)
    y0 = mz.y[0]
    for _ in range(50):
        mz.step(np.zeros(6, bool), None, DT)                     # quiet -> exact decay
    assert mz.y[0] < y0 and abs(mz.y[0] - y0 * np.exp(-50 * DT / 120.0)) < 1e-9


def test_relay_x_hill_gate_stays_unit_interval():
    mz = _mk_fc(use_x=True, tau_y=120.0, tau_x=50.0, x_min=0.2, y_gate=3.0, K_y=5.0, hill_n=4)
    spk = np.arange(mz.N) < 3
    for _ in range(400):
        mz.step(spk, None, DT)
        assert np.all((mz.x_relay >= 0.0) & (mz.x_relay <= 1.0))
    assert np.all(mz.x_relay[:3] < 1.0)                          # sustained firing depleted the relay


def test_ee_std_apply_scales_only_E_targets():
    """The relay reuses ee_std_apply, which scales AMPA edges to E targets (dst<NE) only."""
    a_w = np.array([1.0, 1.0, 1.0, 1.0])
    a_dst = np.array([0, 1, 4, 5])                               # 2 E targets, 2 I targets (NE=4)
    x = np.array([0.5, 0.25, 0.5, 0.25])
    out = ee_std_apply(a_w, a_dst, x, NE=4)
    np.testing.assert_array_equal(out, np.array([0.5, 0.25, 1.0, 1.0]))


# ============================== (B) engine-level relay ==============================
def _run_fc(*, use_x, y_gate=5.0, seed=SEED):
    # Small aggressive test net (kick + L=6) drives a few cells past the tau_eff safety cap in full
    # conductance; these smoke tests probe mechanism (determinism/finiteness/relay), so allow the safe
    # clip-scaling path. fail_on_clip=True (the scientific gate) is exercised separately by the unit test.
    p, net, NE, NI = _net(); N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    cfg = MZSlowVarsConfig(
        membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0, e_gaba=0.0, e_k=0.0,
        use_z=False, use_m=False, gaba_gain=1.125, max_total_conductance=99.0, fail_on_clip=False,
        use_x=use_x, tau_y=120.0, tau_x=1000.0, x_min=0.0, y_gate=y_gate, K_y=5.0, hill_n=4,
    )
    slow = MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    net["rng"] = np.random.default_rng(seed)
    res = simulate_kick(p, net, KICK_BOOST=4.0, slow=slow, kick_center=np.array([3., 3.]),
                        r_kick=0.5, t_kick=50.0, V_th_per_neuron=vth)
    return res, slow


def test_full_conductance_runs_finite_and_deterministic():
    a, _ = _run_fc(use_x=False)
    b, _ = _run_fc(use_x=False)
    assert np.all(np.isfinite(a["rate_E"])) and a["E_spk_bool"].sum() > 0
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert np.array_equal(a["rate_E"], b["rate_E"])


def test_relay_pinned_at_one_is_identical_to_relay_off():
    """use_x with y_gate huge (x never leaves 1.0) is byte-identical to use_x=False -> relay scatter is a
    clean identity when x==1 (clause C7 off-by-default + x=1 no-op)."""
    off, _ = _run_fc(use_x=False)
    on, slow = _run_fc(use_x=True, y_gate=1e9)
    assert np.array_equal(off["E_spk_bool"], on["E_spk_bool"])
    assert np.array_equal(off["rate_E"], on["rate_E"])
    assert min(slow.trace_x_relay_min) == 1.0                    # x stayed pinned at full availability


def test_active_relay_is_deterministic_and_depletes():
    a, sa = _run_fc(use_x=True, y_gate=5.0)
    b, sb = _run_fc(use_x=True, y_gate=5.0)
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert np.all(np.isfinite(a["rate_E"]))
    assert min(sa.trace_x_relay_min) < 1.0                       # the active relay actually depleted somewhere


def test_relay_and_ee_std_mutually_exclusive():
    p, net, NE, NI = _net(); N = NE + NI
    vth = np.full(N, 18.0)
    cfg = MZSlowVarsConfig(membrane_mode="full_conductance", use_x=True, tau_y=120.0, tau_x=1000.0,
                           K_y=5.0, y_gate=5.0, e_gaba=0.0, v_match=18.0)
    slow = MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    net["rng"] = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="mutually exclusive"):
        simulate_kick(p, net, KICK_BOOST=0.0, slow=slow, t_kick=1e9, V_th_per_neuron=vth,
                      ee_std_u=0.2, ee_std_tau_ms=200.0)


def test_use_x_requires_full_conductance():
    with pytest.raises(ValueError, match="requires membrane_mode='full_conductance'"):
        MZSlowVarsConfig_and_build(membrane_mode="conductance", use_x=True)


def MZSlowVarsConfig_and_build(**kw):
    base = dict(v_match=18.0, e_gaba=0.0, tau_y=120.0, tau_x=1000.0, K_y=5.0, y_gate=5.0)
    base.update(kw)
    return MZSlowVars(6, 18.0, MZSlowVarsConfig(**base), NE=4, core_mask_E=np.zeros(4, bool))


# ============================== engine bless gate ==============================
def test_engine_blessed_fcxr():
    rec = json.load(open(ENGINE_VERSIONS))
    kp = os.path.join(ROOT, "src", "snn_engine", "kick_probe.py")
    cur = hashlib.sha256(open(kp, "rb").read()).hexdigest()
    assert rec["src/snn_engine/kick_probe.py"] == cur           # FAILS until re-blessed after the edit
