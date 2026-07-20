"""FCXR Stage D — D1 frozen fast-branch map + D2 mode analysis unit tests.

Fast unit tests only (synthetic observable rows / tiny fields). The real 40k
`build_substrate` alignment check and SNN cell runs are exercised by the runner
scripts, not here (build_substrate ~137s / 6.8GB).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.topic4_mz_fcxr_dynamics import (
    load_onset_depletion_pi,
    assert_field_substrate_aligned,
    frozen_z_field,
)
from src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig

SNAP = "results/topic4_sef_hfo/state_conditioned_susceptibility/snapshots/zA_q75_tz5000/seed_1.npz"


# ---------------- D0.1: locked p_i loader ----------------

def test_pi_is_mean_one_and_nonneg():
    pk = load_onset_depletion_pi(SNAP)
    assert pk["p_i"].shape == (32000,)
    assert np.isclose(float(pk["p_i"].mean()), 1.0, atol=1e-6)   # mean-depletion normalization
    assert (pk["p_i"] >= 0).all()
    assert pk["pos_E"].shape == (32000, 2)
    assert pk["vth_E"].shape == (32000,)


# ---------------- D0.1: substrate-alignment gate (synthetic S, fast) ----------------

def _fake_pack(NE=50, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(NE, 2))
    vth = rng.normal(18.0, 1.0, size=NE)
    return dict(p_i=np.ones(NE), pos_E=pos, vth_E=vth,
                src_xy=np.zeros(2), snk_xy=np.ones(2), axis_unit=np.array([1.0, 0.0]), L=20.0)


def _fake_S(pk, NI=10):
    NE = pk["pos_E"].shape[0]
    posI = np.zeros((NI, 2))
    vth_full = np.concatenate([pk["vth_E"], np.full(NI, 18.0)])   # E-then-I, length N
    return dict(NE=NE, NI=NI, N=NE + NI, posE=pk["pos_E"].copy(),
                posI=posI, vth=vth_full)


def test_alignment_passes_when_field_matches_substrate():
    pk = _fake_pack()
    S = _fake_S(pk)
    assert_field_substrate_aligned(pk, S)   # must not raise


def test_alignment_rejects_shuffled_field():
    pk = _fake_pack()
    S = _fake_S(pk)
    bad = dict(pk); bad["pos_E"] = pk["pos_E"][::-1].copy()   # neuron order reversed
    with pytest.raises(ValueError):
        assert_field_substrate_aligned(bad, S)


def test_alignment_rejects_NE_mismatch():
    pk = _fake_pack(NE=50)
    S = _fake_S(_fake_pack(NE=40))   # substrate has 40 E cells, field has 50
    with pytest.raises(ValueError):
        assert_field_substrate_aligned(pk, S)


# ---------------- D1.3: frozen-Z field + injection ----------------

def test_frozen_z_field_clips():
    p = np.array([0.0, 1.0, 2.0, 10.0])
    z = frozen_z_field(p, 0.15)
    assert np.allclose(z, np.clip(1 - 0.15 * p, 0, 1))
    assert z[0] == 1.0 and z[-1] == 0.0          # p_i=0 -> no depletion; large p_i -> full


def _mzsv(z_frozen_E=None, use_z=False, N=10, NE=8):
    cfg = MZSlowVarsConfig(use_z=use_z, z_frozen_E=z_frozen_E)
    return MZSlowVars(N, 18.0, cfg=cfg, NE=NE)


def test_z_none_is_all_ones_byte_identical_init():
    s = _mzsv(z_frozen_E=None)
    assert np.array_equal(s.z, np.ones(10))       # unchanged default -> byte-parity path


def test_z_frozen_sets_E_block_only():
    field = np.linspace(0.2, 0.9, 8)
    s = _mzsv(z_frozen_E=field)
    assert np.allclose(s.z[:8], field)            # E block = frozen field
    assert np.allclose(s.z[8:], 1.0)              # I block pinned at 1 (E-only clause)


def test_z_frozen_wrong_length_raises():
    with pytest.raises(ValueError):
        _mzsv(z_frozen_E=np.ones(7))              # NE=8, field length 7


def test_z_frozen_out_of_range_raises():
    with pytest.raises(ValueError):
        _mzsv(z_frozen_E=np.full(8, 1.2))         # value > 1


def test_z_frozen_requires_use_z_false():
    with pytest.raises(ValueError):
        _mzsv(z_frozen_E=np.linspace(0.2, 0.9, 8), use_z=True)   # frozen field must not evolve


# ---------------- D1.5: two-layer branch classifier (synthetic rows, no sim) ----------------
from src.topic4_mz_fcxr_dynamics import (           # noqa: E402
    classify_run_provisional, resolve_high_ic, classify_branch_D, THRESHOLDS,
)


def _row(**kw):
    # "high" = sustained: long contiguous elevation (high_duration_ms) AND still elevated at end (tail_high_frac).
    base = dict(numerical_unsafe=False, high_duration_ms=0.0, tail_high_frac=0.02, af_tail=0.001,
                modulation=0.05, oscillatory_candidate=False, end_rate_hz=5.0)
    base.update(kw)
    return base


def test_run_numerical_unsafe_wins_first():
    # even a high-looking row is UNSAFE if the numerical flag is set (clause 1: checked first)
    assert classify_run_provisional(_row(numerical_unsafe=True, high_duration_ms=3000, tail_high_frac=0.95)) == "NUMERICAL_UNSAFE"


def test_run_finite_high_fixed():
    assert classify_run_provisional(_row(high_duration_ms=3000, tail_high_frac=0.95, af_tail=0.4,
                                         modulation=0.15)) == "FINITE_HIGH_FIXED"


def test_run_finite_high_orbit():
    assert classify_run_provisional(_row(high_duration_ms=3000, tail_high_frac=0.95, af_tail=0.4,
                                         modulation=0.4, oscillatory_candidate=True)) == "FINITE_HIGH_ORBIT"


def test_run_refractory_ceiling_beats_finite():
    # pinned: nearly all cells firing (af_tail>=0.90) with ~no modulation -> ceiling, not a finite attractor
    assert classify_run_provisional(_row(high_duration_ms=3000, tail_high_frac=1.0, af_tail=0.95,
                                         modulation=0.02)) == "REFRACTORY_CEILING"


def test_run_decays_to_low():
    # interictal-like: only brief events (~12ms contiguous), tail back at the quiet floor
    assert classify_run_provisional(_row(high_duration_ms=12, tail_high_frac=0.02)) == "DECAYS_TO_LOW"


def test_run_excursion_decayed():
    # long contiguous excursion (>=300ms) but decayed by the end (low tail occupancy) -> metastable candidate
    assert classify_run_provisional(_row(high_duration_ms=1500, tail_high_frac=0.05)) == "EXCURSION_DECAYED"


def test_resolve_high_ic_finite_needs_both_windows():
    assert resolve_high_ic("FINITE_HIGH_FIXED", "FINITE_HIGH_FIXED") == "FINITE_HIGH_FIXED"
    assert resolve_high_ic("FINITE_HIGH_ORBIT", "FINITE_HIGH_FIXED") == "FINITE_HIGH_ORBIT"
    # high at short window, gone by the longer window -> long transient (F1)
    assert resolve_high_ic("FINITE_HIGH_FIXED", "EXCURSION_DECAYED") == "METASTABLE_TRANSIENT"
    assert resolve_high_ic("NUMERICAL_UNSAFE", "FINITE_HIGH_FIXED") == "NUMERICAL_UNSAFE"
    assert resolve_high_ic("REFRACTORY_CEILING", "FINITE_HIGH_FIXED") == "REFRACTORY_CEILING"
    assert resolve_high_ic("DECAYS_TO_LOW", "DECAYS_TO_LOW") == "DECAYS_TO_LOW"


def test_D_bistable():
    d = classify_branch_D("DECAYS_TO_LOW", ["FINITE_HIGH_FIXED", "FINITE_HIGH_FIXED"], [60.0, 62.0])
    assert d["D_label"] == "BISTABLE"


def test_D_low_only():
    d = classify_branch_D("DECAYS_TO_LOW", ["DECAYS_TO_LOW", "DECAYS_TO_LOW"], [None, None])
    assert d["D_label"] == "LOW_ONLY"


def test_D_unresolved_when_plateaus_disagree():
    # both high ICs go high but to very different plateaus (spread > 0.20) -> not a single branch
    d = classify_branch_D("DECAYS_TO_LOW", ["FINITE_HIGH_FIXED", "FINITE_HIGH_FIXED"], [40.0, 90.0])
    assert d["D_label"] == "UNRESOLVED"


def test_D_finite_high_monostable():
    d = classify_branch_D("FINITE_HIGH_FIXED", ["FINITE_HIGH_FIXED", "FINITE_HIGH_FIXED"], [60.0, 61.0])
    assert d["D_label"] == "FINITE_HIGH"


def test_D_ceiling_and_metastable_and_unsafe():
    assert classify_branch_D("DECAYS_TO_LOW", ["REFRACTORY_CEILING", "REFRACTORY_CEILING"], [None, None])["D_label"] == "REFRACTORY_CEILING"
    assert classify_branch_D("DECAYS_TO_LOW", ["METASTABLE_TRANSIENT", "METASTABLE_TRANSIENT"], [None, None])["D_label"] == "METASTABLE_TRANSIENT"
    assert classify_branch_D("NUMERICAL_UNSAFE", ["FINITE_HIGH_FIXED", "FINITE_HIGH_FIXED"], [60.0, 61.0])["D_label"] == "NUMERICAL_UNSAFE"


def test_D_single_finite_ic_is_unresolved():
    # only one of two high ICs reached high -> not confirmed (needs >=2 concordant)
    d = classify_branch_D("DECAYS_TO_LOW", ["FINITE_HIGH_FIXED", "DECAYS_TO_LOW"], [60.0, None])
    assert d["D_label"] == "UNRESOLVED"


def test_z_frozen_is_applied_by_conductance_membrane():
    # THE load-bearing invariant the pilot exposed: a frozen field must CHANGE the membrane, not just be stored.
    def mt(z_frozen_E, N=6, NE=4):
        cfg = MZSlowVarsConfig(membrane_mode="full_conductance", E_E=58.0, v_match=18.0, e_gaba=0.0,
                               ff_conductance=False, rec_conductance=True, rec_sat_g=21.6, z_frozen_E=z_frozen_E)
        s = MZSlowVars(N, 18.0, cfg=cfg, NE=NE)
        rng = np.random.default_rng(0)
        I_E = np.abs(rng.normal(5, 2, N)); I_I = np.abs(rng.normal(3, 1, N)); I_E_rec = 0.5 * I_E
        return s.membrane_terms(I_E, I_I, I_E_rec=I_E_rec)
    grel_dep = np.asarray(mt(np.full(4, 0.3))[1])[:4]    # depleted -> 30% of inhibitory efficacy
    grel_full = np.asarray(mt(None)[1])[:4]              # z=1 (full inhibition, use_z False)
    assert not np.allclose(grel_dep, grel_full)          # frozen depletion actually changes the membrane
    assert grel_dep.sum() < grel_full.sum()              # less inhibition -> lower total relative conductance


# ---------------- D2.10: SNN sech^2 effective-operator lens ----------------
from src.topic4_mz_fcxr_dynamics import snn_landmark_sech2   # noqa: E402
import scipy.sparse as sp   # noqa: E402


def test_snn_landmark_sech2_iprs_in_range_and_sech2_bounded():
    N = 120
    W = sp.random(N, N, density=0.1, random_state=0, data_rvs=lambda n: np.abs(np.random.default_rng(1).normal(size=n))).tocsr()
    g_raw = np.abs(np.random.default_rng(2).normal(15, 8, N))
    out = snn_landmark_sech2(W, g_raw, g_sat=21.6, k=4)
    assert out["N"] == N
    assert 0.0 < out["sech2_mean"] <= 1.0 and 0.0 <= out["sech2_min"] <= 1.0   # sech^2 in [0,1]
    for key in ("raw_lead_ipr", "eff_lead_ipr"):
        v = out[key]
        assert np.isnan(v) or (1.0 / N - 1e-9 <= v <= 1.0 + 1e-9)              # IPR in [1/N, 1]
