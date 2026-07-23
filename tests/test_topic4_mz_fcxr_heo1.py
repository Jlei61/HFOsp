"""FCXR-HEO1 TDD.

Two contract groups:
  T1/T2 — cooperative recurrent-conductance gate + gErec_raw streaming histogram in
          src/snn_engine/mz_slow_vars.py (each test == one design-lock §T1/§T2 clause).
  T3    — HEO spectral classifier in src/topic4_mz_fcxr_heo1.py (7 synthetic clauses).

Design lock: docs/superpowers/plans/2026-07-24-topic4-mz-fcxr-heo1.md.
The cooperative gate is OFF by default (coop_A=0) -> byte-identical to FCXR-RC1; the whole
existing mz_slow_vars + full-conductance suites stay green (the pre-edit parity guard).
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mz_slow_vars import (  # noqa: E402
    MZSlowVars, MZSlowVarsConfig, cooperative_u_tilde, gerec_baseline_quantiles)

DENOM = 58.0 - 18.0   # E_E - v_match
GSAT = 21.6


def _mk_fc(N=6, NE=4, **kw):
    """Full-conductance arm-C MZSlowVars; recurrent-only saturation on, feedforward additive."""
    base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0,
                e_gaba=0.0, e_k=0.0, ff_conductance=False, rec_conductance=True,
                rec_sat_g=GSAT, max_total_conductance=99.0)
    base.update(kw)
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**base), NE=NE, core_mask_E=np.zeros(NE, bool))


def _gErec_from_terms(mz, I_E_rec):
    """Isolate gErec: I_I=0 (gI=0), use_m=False (gM=0), I_E==I_E_rec (feedforward part 0).
    Then g_rel[:NE] == gErec and g_rev[:NE] == gErec * E_E."""
    N = mz.N
    I_E = np.zeros(N); I_E[:len(I_E_rec)] = I_E_rec
    rec = np.zeros(N); rec[:len(I_E_rec)] = I_E_rec
    _, g_rel, _ = mz.membrane_terms(I_E, np.zeros(N), labels=None, I_E_rec=rec)
    return g_rel[:mz.NE]


# ============================== §T1 pure cooperative transform ==============================
def test_cooperative_u_tilde_off_returns_u_unchanged():
    u = np.array([0.0, 0.5, 3.0, 100.0])
    out = cooperative_u_tilde(u, A_c=0.0, u_c=1.0, K_c=0.25, n=4)
    assert np.array_equal(out, u)                       # OFF -> exact identity (byte parity)


def test_cooperative_u_tilde_identity_below_uc():
    u = np.array([0.0, 0.25, 0.5, 1.0])                 # all <= u_c
    out = cooperative_u_tilde(u, A_c=8.0, u_c=1.0, K_c=0.25, n=4)
    assert np.array_equal(out, u)                       # relu=0 -> H=0 -> u*(1+0) == u exactly


def test_cooperative_u_tilde_monotone_nonneg_finite():
    u = np.linspace(0.0, 200.0, 4001)
    out = cooperative_u_tilde(u, A_c=6.0, u_c=2.0, K_c=0.5, n=4)
    assert np.all(np.isfinite(out)) and np.all(out >= 0.0)
    assert np.all(np.diff(out) >= -1e-12)               # monotone non-decreasing


def test_cooperative_u_tilde_superlinear_above_uc():
    # transform slope d(u_tilde)/du > 1 in the mid region (cooperative bump), == 1 below u_c
    u = np.array([0.5, 6.0])
    du = 1e-4
    for x, expect_super in ((0.5, False), (6.0, True)):
        s = (cooperative_u_tilde(np.array([x + du]), 6.0, 2.0, 0.5, 4)[0]
             - cooperative_u_tilde(np.array([x - du]), 6.0, 2.0, 0.5, 4)[0]) / (2 * du)
        if expect_super:
            assert s > 1.0 + 1e-3                        # steeper than identity above u_c
        else:
            assert abs(s - 1.0) < 1e-6                   # identity below u_c


def test_cooperative_u_tilde_bounded_by_1_plus_A():
    u = np.linspace(0.0, 500.0, 2001)
    A = 4.0
    out = cooperative_u_tilde(u, A_c=A, u_c=1.0, K_c=0.25, n=4)
    assert np.all(out <= u * (1.0 + A) + 1e-9)          # H < 1 -> boost bounded by (1+A_c)


# ============================== §T1 membrane_terms integration ==============================
def test_coop_off_matches_rc1_tanh_formula():
    """coop_A=0 -> gErec == g_sat*tanh(gErec_raw/g_sat) exactly (RC1 pin / pre-edit parity)."""
    I_rec = np.array([2.0, 40.0, 400.0, 4000.0])
    g = _gErec_from_terms(_mk_fc(coop_A=0.0), I_rec)
    raw = I_rec / DENOM
    np.testing.assert_allclose(g, GSAT * np.tanh(raw / GSAT), atol=1e-11)


def test_coop_below_uc_is_exact_rc1():
    """coop_A>0 but every gErec_raw <= u_c -> gErec identical to RC1 (u<=u_c exact at membrane)."""
    I_rec = np.array([2.0, 4.0, 8.0, 12.0])             # raw = I_rec/40 = 0.05..0.30
    u_c = 1.0                                           # above all raw
    g_on = _gErec_from_terms(_mk_fc(coop_A=8.0, coop_uc=u_c, coop_Kc=0.25 * u_c, coop_n=4), I_rec)
    g_off = _gErec_from_terms(_mk_fc(coop_A=0.0), I_rec)
    np.testing.assert_array_equal(g_on, g_off)


def test_coop_boosts_gErec_above_uc():
    I_rec = np.array([400.0, 400.0, 400.0, 400.0])      # raw=10 (> u_c, below hard saturation)
    u_c = 1.0
    g_on = _gErec_from_terms(_mk_fc(coop_A=4.0, coop_uc=u_c, coop_Kc=0.25 * u_c, coop_n=4), I_rec)
    g_off = _gErec_from_terms(_mk_fc(coop_A=0.0), I_rec)
    assert np.all(g_on > g_off + 1e-6)                  # cooperative gain raises effective conductance


def test_coop_saturates_at_gsat():
    I_rec = np.full(4, 1e5)                             # huge -> u_tilde huge -> tanh -> g_sat
    g_on = _gErec_from_terms(_mk_fc(coop_A=8.0, coop_uc=1.0, coop_Kc=0.25, coop_n=4), I_rec)
    assert np.all(g_on <= GSAT + 1e-9) and np.all(g_on > 0.99 * GSAT)


def test_coop_recurrent_only_ff_and_raw_audit_preserved():
    """Cooperative gate changes recurrent gErec ONLY: feedforward gEff mean and the RAW gErec_raw
    audit value (record_clip_identity) are untouched; gI/gM/I-cells unchanged."""
    N, NE = 6, 4
    I_E = np.array([30.0, 30.0, 30.0, 30.0, 4.0, 4.0])
    I_rec = np.array([12.0, 12.0, 12.0, 12.0, 1.0, 1.0])   # feedforward part = I_E - I_rec = 18
    I_I = np.array([2.0, 2.0, 2.0, 2.0, 0.0, 0.0])
    kw = dict(ff_conductance=True, record_clip_identity=True)   # both AMPA sides conductance
    off = MZSlowVars(N, 18.0, MZSlowVarsConfig(membrane_mode="full_conductance", E_E=58.0, c_E=1.0,
                     v_match=18.0, e_gaba=0.0, e_k=0.0, rec_conductance=True, rec_sat_g=GSAT,
                     max_total_conductance=99.0, **kw), NE=NE, core_mask_E=np.zeros(NE, bool))
    on = MZSlowVars(N, 18.0, MZSlowVarsConfig(membrane_mode="full_conductance", E_E=58.0, c_E=1.0,
                    v_match=18.0, e_gaba=0.0, e_k=0.0, rec_conductance=True, rec_sat_g=GSAT,
                    max_total_conductance=99.0, coop_A=6.0, coop_uc=0.05, coop_Kc=0.0125, coop_n=4,
                    **kw), NE=NE, core_mask_E=np.zeros(NE, bool))
    off.membrane_terms(I_E, I_I, labels=None, I_E_rec=I_rec)
    on.membrane_terms(I_E, I_I, labels=None, I_E_rec=I_rec)
    assert abs(off._gEff_mean_last - on._gEff_mean_last) < 1e-12     # feedforward conductance unchanged
    assert abs(off._gI_mean_last - on._gI_mean_last) < 1e-12         # inhibitory conductance unchanged
    np.testing.assert_array_equal(off.max_raw_gErec, on.max_raw_gErec)  # RAW audit uses pre-coop value
    assert on._gErec_mean_last > off._gErec_mean_last + 1e-9         # recurrent gErec DID change


def test_coop_validation_raises():
    def cfg(**kw):
        base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0,
                    e_gaba=0.0, e_k=0.0, rec_conductance=True, rec_sat_g=GSAT)
        base.update(kw)
        return MZSlowVars(6, 18.0, MZSlowVarsConfig(**base), NE=4, core_mask_E=np.zeros(4, bool))
    with pytest.raises(ValueError):
        cfg(coop_A=4.0, coop_uc=0.0, coop_Kc=0.25)                   # u_c must be > 0
    with pytest.raises(ValueError):
        cfg(coop_A=4.0, coop_uc=1.0, coop_Kc=0.0)                    # K_c must be > 0
    with pytest.raises(ValueError):
        cfg(coop_A=4.0, coop_uc=1.0, coop_Kc=0.25, coop_n=0)         # n >= 1
    with pytest.raises(ValueError):
        cfg(coop_A=-1.0)                                             # A_c >= 0
    with pytest.raises(ValueError):                                  # coop gain needs saturation on
        cfg(coop_A=4.0, coop_uc=1.0, coop_Kc=0.25, rec_sat_g=0.0)
    with pytest.raises(ValueError):                                  # coop needs full_conductance+rec
        MZSlowVars(6, 18.0, MZSlowVarsConfig(membrane_mode="conductance", v_match=18.0, e_gaba=0.0,
                   coop_A=4.0, coop_uc=1.0, coop_Kc=0.25), NE=4, core_mask_E=np.zeros(4, bool))


# ============================== §T2 gErec_raw histogram + engagement traces ==============================
def _mk_fc_hist(edges, core=(0, 1), N=6, NE=4, **kw):
    cm = np.zeros(NE, bool)
    for i in core:
        cm[i] = True
    base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0, e_gaba=0.0,
                e_k=0.0, ff_conductance=False, rec_conductance=True, rec_sat_g=GSAT,
                max_total_conductance=99.0, record_gerec_hist=True, gerec_hist_edges=edges)
    base.update(kw)
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**base), NE=NE, core_mask_E=cm)


def test_gerec_hist_requires_edges():
    with pytest.raises(ValueError):
        _mk_fc(record_gerec_hist=True)                  # edges=None -> raise


def test_gerec_hist_counts_and_partitions():
    edges = np.linspace(0.0, 20.0, 41)                  # width 0.5 bins over [0,20]
    mz = _mk_fc_hist(edges, core=(0, 1))                # core = E cells {0,1}, surround = {2,3}
    # gErec_raw = I_recE/40 ; feed 3 steps with I_I=0 -> gErec_raw known per E cell
    recs = [np.array([40.0, 80.0, 200.0, 400.0]),       # raw = 1, 2, 5, 10
            np.array([40.0, 40.0, 40.0, 40.0]),         # raw = 1,1,1,1
            np.array([400.0, 400.0, 80.0, 80.0])]       # raw = 10,10,2,2
    all_raw = []
    for rec in recs:
        I_E = np.zeros(6); I_E[:4] = rec
        I_rec = np.zeros(6); I_rec[:4] = rec
        mz.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_rec)
        all_raw.append(rec / 40.0)
    ref_overall = sum(np.histogram(r, bins=edges)[0] for r in all_raw)
    np.testing.assert_array_equal(mz.gerec_hist_overall, ref_overall)
    # core (cells 0,1) + surround (cells 2,3) partition the overall E histogram exactly
    np.testing.assert_array_equal(mz.gerec_hist_core + mz.gerec_hist_surround, mz.gerec_hist_overall)
    ref_core = sum(np.histogram(r[:2], bins=edges)[0] for r in all_raw)
    np.testing.assert_array_equal(mz.gerec_hist_core, ref_core)


def test_gerec_hist_engine_pure_side_effect():
    """record_gerec_hist=True must not change the raster (pure observer)."""
    edges = np.linspace(0.0, 30.0, 601)
    off = _engine_run(dict(coop_A=0.0))
    on = _engine_run(dict(coop_A=0.0, record_gerec_hist=True, gerec_hist_edges=edges))
    assert np.array_equal(off["E_spk_bool"], on["E_spk_bool"])
    assert np.array_equal(off["rate_E"], on["rate_E"])


def test_coop_engagement_traces_present_and_bounded():
    edges = np.linspace(0.0, 20.0, 41)
    mz = _mk_fc_hist(edges, coop_A=6.0, coop_uc=2.0, coop_Kc=0.5, coop_n=4)
    for _ in range(7):
        I_E = np.zeros(6); I_E[:4] = np.array([40.0, 400.0, 40.0, 400.0])   # raw 1,10,1,10 (u_c=2)
        I_rec = I_E.copy()
        mz.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_rec)
        mz.step(np.zeros(6, bool), None, 0.05)
    assert len(mz.trace_coop_engaged_frac) == 7 and len(mz.trace_coop_H_mean) == 7
    assert all(0.0 <= f <= 1.0 for f in mz.trace_coop_engaged_frac)
    assert abs(mz.trace_coop_engaged_frac[0] - 0.5) < 1e-9    # 2 of 4 E cells have raw(=10) > u_c(=2)


def test_gerec_baseline_quantiles_uniform():
    edges = np.arange(0.0, 11.0)                        # 10 unit bins over [0,10)
    counts = np.full(10, 100, dtype=np.int64)          # uniform
    q = gerec_baseline_quantiles(counts, edges, [0.5, 0.9, 0.99])
    assert abs(q[0.5] - 5.0) < 1e-9
    assert abs(q[0.9] - 9.0) < 1e-9
    assert abs(q[0.99] - 9.9) < 1e-9


def test_gerec_baseline_quantiles_overflow_is_inf():
    edges = np.array([0.0, 1.0, 2.0, np.inf])          # last bin = overflow
    counts = np.array([980, 15, 5], dtype=np.int64)    # 0.5% mass in the overflow bin
    q = gerec_baseline_quantiles(counts, edges, [0.999])
    assert not np.isfinite(q[0.999])                   # Q99.9 falls in overflow -> inf (runner widens)


# ============================== §T1 engine-level parity / non-triviality ==============================
def _engine_run(coop_kw):
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick
    SEED = 1
    p = Params(L=6.0, density=100.0, T=250.0, dt=0.1, nu_ext_ratio=0.9, seed=SEED)
    rng = np.random.default_rng(SEED)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    N = NE + NI
    vth = np.full(N, 18.0); vth[:5] = 16.0
    base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0, e_gaba=0.0,
                e_k=0.0, ff_conductance=False, rec_conductance=True, rec_sat_g=GSAT,
                max_total_conductance=99.0)
    base.update(coop_kw)
    slow = MZSlowVars(N, 18.0, MZSlowVarsConfig(**base), NE=NE, core_mask_E=np.zeros(NE, bool))
    net["rng"] = np.random.default_rng(SEED)
    res = simulate_kick(p, net, KICK_BOOST=4.0, slow=slow, kick_center=np.array([3., 3.]),
                        r_kick=0.5, t_kick=50.0, V_th_per_neuron=vth)
    return res


def test_engine_coop_high_uc_equals_rc1():
    """u_c above every gErec_raw -> gate never engages -> byte-identical raster to coop OFF."""
    off = _engine_run(dict(coop_A=0.0))
    inert = _engine_run(dict(coop_A=8.0, coop_uc=1e9, coop_Kc=2.5e8, coop_n=4))
    assert np.array_equal(off["E_spk_bool"], inert["E_spk_bool"])
    assert off["E_spk_bool"].sum() > 0                  # non-trivial: there WAS activity


def test_engine_coop_on_changes_dynamics():
    """coop_A>0 with a low u_c actually engages -> raster differs from RC1 (mechanism is live)."""
    off = _engine_run(dict(coop_A=0.0))
    on = _engine_run(dict(coop_A=8.0, coop_uc=1e-3, coop_Kc=2.5e-4, coop_n=4))
    assert not np.array_equal(off["E_spk_bool"], on["E_spk_bool"])


# ============================== §T3 HEO spectral classifier (7 synthetic cases) ==============================
import src.topic4_mz_fcxr_heo1 as HEO  # noqa: E402

FS_T = 1000.0
DT_T = 1000.0 / FS_T          # dt(ms) so fs_raw == FS_WORK -> decimation is a no-op (test the gate logic)
NC = 15
SCL_T = np.array([True] * 4 + [False] * 11)   # SCL6-9 are the first 4 contacts (matches real montage)


def _tt(n):
    return np.arange(n) / FS_T


def _white(rng, n, sigma, C=NC):
    return rng.standard_normal((n, C)) * sigma


def _baseline_ref(rng):
    lfp = _white(rng, 4000, 1.0)
    rate = 0.5 + 0.1 * rng.standard_normal(4000)
    return HEO.build_baseline_reference(lfp, rate, DT_T)


def _safe():
    return dict(numerical_unsafe=False, runaway_early_stop_ms=None)


def _hi_broadband(rng, n, sigma=5.0, C=NC):
    """High white noise -> flat PSD -> every band elevated (broadband, no oscillatory peak)."""
    return _white(rng, n, sigma, C)


def _oscillate(n, freq=50.0, amp=6.0, C=NC):
    s = amp * (np.sin(2 * np.pi * freq * _tt(n)) + 0.4 * np.sin(2 * np.pi * 2 * freq * _tt(n)))
    return np.tile(s[:, None], (1, C))


def test_heo_oscillatory_broadband_platform_passes():
    rng = np.random.default_rng(0)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _hi_broadband(rng, n, sigma=5.0) + _oscillate(n, 50.0, 6.0)
    rate = 150.0 + 45.0 * np.sin(2 * np.pi * 50.0 * _tt(n)) + 2.0 * rng.standard_normal(n)
    v = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, _safe())
    assert v["HEO_BRANCH"] is True
    assert v["gate_A_plateau"] and v["gate_C_platform"] and v["gate_D_oscillation"]
    assert abs(v["oscillation"]["center_hz"] - 50.0) <= HEO.OSC_CENTER_TOL_HZ


def test_heo_sparse_irregular_ied_fails():
    rng = np.random.default_rng(1)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _white(rng, n, 1.0)                       # quiet baseline
    for c in (500, 1500, 2500):                      # 3 isolated 30 ms broadband bursts
        lfp[c:c + 30] += rng.standard_normal((30, NC)) * 8.0
    rate = 0.6 + 0.1 * rng.standard_normal(n)
    v = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, _safe())
    assert v["HEO_BRANCH"] is False
    assert v["gate_A_plateau"] is False              # no sustained >=1000 ms platform


def test_heo_dense_event_train_fails():
    rng = np.random.default_rng(2)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _white(rng, n, 1.0)
    for start in range(300, 2700, 80):               # burst 40 ms every 80 ms -> returns to baseline between
        lfp[start:start + 40] += rng.standard_normal((40, NC)) * 6.0
    rate = 0.6 + 0.1 * rng.standard_normal(n)
    v = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, _safe())
    assert v["HEO_BRANCH"] is False                  # dips to baseline -> not a plateau (A) or no osc (D)


def test_heo_tonic_ceiling_fails():
    rng = np.random.default_rng(3)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _hi_broadband(rng, n, sigma=5.0)           # high broadband, NO oscillation
    rate = 450.0 + 1.0 * rng.standard_normal(n)      # pinned near the 500 Hz refractory ceiling, flat
    v = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, _safe())
    assert v["HEO_BRANCH"] is False
    assert (v["gate_D_oscillation"] is False) or (v["gate_E_numerical"] is False)


def test_heo_narrowband_local_only_fails():
    rng = np.random.default_rng(4)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _white(rng, n, 1.0)                        # quiet broadband floor
    lfp[:, :3] += _oscillate(n, 50.0, 8.0, C=3)      # sustained 50 Hz on only 3 contacts (narrow band)
    rate = 0.6 + 0.1 * rng.standard_normal(n)
    v = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, _safe())
    assert v["HEO_BRANCH"] is False                  # not broadband (few bands) and not a platform (few contacts)


def test_heo_broadband_nonoscillatory_fails():
    rng = np.random.default_rng(5)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _hi_broadband(rng, n, sigma=5.0)           # sustained high broadband, WHITE (no peak)
    rate = 150.0 + 30.0 * rng.standard_normal(n)     # high but noisy -> no coherent oscillation
    v = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, _safe())
    assert v["gate_A_plateau"] is True               # it IS a sustained broadband platform ...
    assert v["gate_D_oscillation"] is False          # ... but no oscillation -> not HEO
    assert v["HEO_BRANCH"] is False


def test_heo_silent_post_tail_localizes_plateau():
    rng = np.random.default_rng(6)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _white(rng, n, 1.0)
    rate = 0.6 + 0.1 * rng.standard_normal(n)
    lfp[:1500] = _hi_broadband(rng, 1500, sigma=5.0) + _oscillate(1500, 50.0, 6.0)   # plateau first half
    rate[:1500] = 150.0 + 45.0 * np.sin(2 * np.pi * 50.0 * _tt(1500)) + 2.0 * rng.standard_normal(1500)
    v = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, _safe())
    assert v["gate_A_plateau"] is True and v["HEO_BRANCH"] is True
    # plateau is localized to the high first half; the silent tail is NOT swept into it
    assert v["plateau"]["j"] * HEO.HOP_MS < 2000.0


def test_heo_runaway_and_unsafe_drop_gate_E():
    rng = np.random.default_rng(7)
    ref = _baseline_ref(rng)
    n = 3000
    lfp = _hi_broadband(rng, n, sigma=5.0) + _oscillate(n, 50.0, 6.0)
    rate = 150.0 + 45.0 * np.sin(2 * np.pi * 50.0 * _tt(n)) + 2.0 * rng.standard_normal(n)
    v_run = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, dict(numerical_unsafe=False, runaway_early_stop_ms=1234.0))
    v_uns = HEO.classify_heo(lfp, rate, DT_T, SCL_T, ref, dict(numerical_unsafe=True, runaway_early_stop_ms=None))
    assert v_run["gate_E_numerical"] is False and v_run["HEO_BRANCH"] is False
    assert v_uns["gate_E_numerical"] is False and v_uns["HEO_BRANCH"] is False
