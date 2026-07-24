"""FCXR-HEO2 TDD.

Task 1 — Phase-0 estimators + 4-class state map (fixes the HEO1 3.906 Hz resolution-floor artifact).
Task 2/3 — m_enable_ms (delayed adaptation) + m_frozen_E (static-K control) in mz_slow_vars (byte-parity when off).
Task 5 — Phase-1 arm classifier (8 conjunctive success criteria).
Spec: docs/superpowers/specs/2026-07-24-topic4-heo2-broadband-diagnostic-design.md.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "snn_engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.topic4_mz_fcxr_heo2 as H2  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402

FS = 1000.0


def _mk(N=10, NE=8, **kw):
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**kw), NE=NE, core_mask_E=np.zeros(NE, bool))


def _t(n):
    return np.arange(n) / FS


# ============================== Task 1: Phase-0 estimators ==============================
def test_dominant_2s_recovers_low_freq_not_floor():
    n = 4000
    s35 = np.sin(2 * np.pi * 3.5 * _t(n))
    s16 = np.sin(2 * np.pi * 16.0 * _t(n))
    assert abs(H2.dominant_2s(s35, FS) - 3.5) <= 0.6      # NOT the 3.906 coarse-probe floor
    assert abs(H2.dominant_2s(s16, FS) - 16.0) <= 0.6


def test_event_ipi_hz_matches_pulse_rate():
    n = 4000
    rate = np.zeros(n)
    rate[::250] = 100.0                                    # a pulse every 250 ms -> 4 Hz
    assert abs(H2.event_ipi_hz(rate, FS) - 4.0) <= 0.6


def test_spikiness_high_for_spikewave_low_for_sine():
    n = 4000
    sine = np.sin(2 * np.pi * 4.0 * _t(n))
    spikewave = np.zeros(n); spikewave[::250] = 10.0      # sparse sharp spikes
    assert H2.spikiness(spikewave) > 2.0
    assert H2.spikiness(sine) < 0.0                        # pure sinusoid is platykurtic


def test_spectral_entropy_and_bw90_narrow_vs_broad():
    rng = np.random.default_rng(0)
    n = 4000
    tone = np.sin(2 * np.pi * 20.0 * _t(n))
    noise = rng.standard_normal(n)
    assert H2.spectral_entropy(noise, FS) > H2.spectral_entropy(tone, FS)
    assert H2.bw90(noise, FS) > H2.bw90(tone, FS)


def test_spectral_distance_to_real():
    l2_self, cos_self = H2.spectral_distance_to_real(list(H2.REAL_E1146_DDB))
    assert l2_self < 1e-9 and cos_self > 0.999
    narrow = [-20.0, -14.0, -9.0, 16.0, 3.0, 0.0]          # model ~16Hz narrowband profile
    l2_n, cos_n = H2.spectral_distance_to_real(narrow)
    assert l2_n > 10.0 and cos_n < 0.5                     # far from the real broadband profile


def test_duty_cycle_low_sparse_high_sustained():
    n = 4000
    sparse = np.zeros(n)
    for c in range(0, n, 400):
        sparse[c:c + 40] = 120.0                           # ~10% active
    sustained = np.full(n, 90.0) + 5.0 * np.random.default_rng(1).standard_normal(n)
    assert H2.duty_cycle(sparse, FS) < 0.4
    assert H2.duty_cycle(sustained, FS) > 0.7


def test_classify_state_four_classes():
    tonic = dict(dominant_hz=16.0, duty_cycle=0.85, coverage=13, coherent=True,
                 six_band_ddb=[-20, -14, -9, 16, 3, 0])
    sparse = dict(dominant_hz=3.5, duty_cycle=0.18, coverage=3, coherent=False,
                  six_band_ddb=[4, 6.7, 5.8, 6, 4.1, 7.4])
    target = dict(dominant_hz=5.0, duty_cycle=0.7, coverage=11, coherent=True,
                  six_band_ddb=[8, 9, 7, 8, 5, 1])
    trans = dict(dominant_hz=10.0, duty_cycle=0.5, coverage=6, coherent=True,
                 six_band_ddb=[2, 3, 1, 5, 2, -1])
    assert H2.classify_state(tonic) == "tonic_16Hz_cycle"
    assert H2.classify_state(sparse) == "sparse_event_train"
    assert H2.classify_state(target) == "target_like_spiky"
    assert H2.classify_state(trans) == "transitional"


# ============================== Task 2: m_enable_ms (delayed adaptation) ==============================
def test_m_enable_ms_none_accumulates_from_step0():
    """Default None -> current behavior: m accumulates from the first step (byte-parity path)."""
    mz = _mk(use_m=True, eta_m=1.0, tau_adp=5000.0)          # m_enable_ms default None
    spk = np.zeros(10, bool); spk[:3] = True
    mz.step(spk, None, 0.1)
    assert mz.trace_m_mean[0] > 0.0                           # accumulated at step 0


def test_m_enable_ms_delays_accumulation():
    """m stays 0 while step*dt < m_enable_ms, then accumulates (delayed onset)."""
    mz = _mk(use_m=True, eta_m=1.0, tau_adp=5000.0, m_enable_ms=100.0)   # dt=0.1 -> enable at step 1000
    spk = np.zeros(10, bool); spk[:3] = True
    for _ in range(1100):
        mz.apply_currents(np.zeros(10), np.zeros(10), None)
        mz.step(spk, None, 0.1)
    assert mz.trace_m_mean[500] == 0.0 and mz.trace_m_mean[999] == 0.0   # 50/99.9 ms < 100 -> m=0
    assert mz.trace_m_mean[1000] > 0.0                                    # 100 ms -> accumulates


def test_m_enable_ms_no_adaptation_current_before_enable():
    """Pre-enable, m=0 -> apply_currents has NO eta_m*m subtraction (distinguishes from the None path,
    where continual spiking would have grown m)."""
    mz = _mk(use_m=True, eta_m=0.5, tau_adp=5000.0, m_enable_ms=100.0)
    spk = np.zeros(10, bool); spk[:3] = True
    for _ in range(500):                                     # 50 ms < 100 ms
        out = mz.apply_currents(np.ones(10), np.ones(10), None)
        assert np.allclose(out[:8], 0.0)                     # 1 - 1 - 0.5*0 (m gated off)
        mz.step(spk, None, 0.1)


def test_m_enable_ms_requires_use_m():
    with pytest.raises(ValueError):
        _mk(use_m=False, m_enable_ms=100.0)


# ============================== Task 3: m_frozen_E (static-K control) ==============================
def _mk_fc(NE=4, N=6, **kw):
    base = dict(membrane_mode="full_conductance", E_E=58.0, c_E=1.0, v_match=18.0, e_gaba=0.0, e_k=0.0,
                rec_conductance=True, rec_sat_g=21.6, eta_m=0.5, m_conductance_gain=1.0)
    base.update(kw)
    return MZSlowVars(N, 18.0, MZSlowVarsConfig(**base), NE=NE, core_mask_E=np.zeros(NE, bool))


def test_m_frozen_E_static_gM():
    """Static-K: m held at a preset field -> gM = m_cond_gain*eta_m*m/(v_match-e_k), unchanged over steps."""
    mz = _mk_fc(m_frozen_E=np.full(4, 2.0))
    I_E = np.array([10., 10., 10., 10., 4., 4.]); I_rec = np.array([2., 2., 2., 2., 1., 1.])
    mz.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_rec)
    expected = 1.0 * 0.5 * 2.0 / (18.0 - 0.0)                 # m_cond_gain * eta_m * const / (v_match - e_k)
    assert abs(mz._gM_mean_last - expected) < 1e-11
    mz.step(np.ones(6, bool), None, 0.05)                    # a step must NOT change the frozen field
    mz.membrane_terms(I_E, np.zeros(6), labels=None, I_E_rec=I_rec)
    assert abs(mz._gM_mean_last - expected) < 1e-11


def test_m_frozen_E_validation():
    with pytest.raises(ValueError):
        _mk_fc(m_frozen_E=np.full(4, 2.0), use_m=True)                        # must not evolve
    with pytest.raises(ValueError):
        _mk_fc(m_frozen_E=np.full(4, -1.0))                                   # negative
    with pytest.raises(ValueError):
        _mk_fc(m_frozen_E=np.full(3, 2.0))                                    # wrong length (NE=4)
    with pytest.raises(ValueError):
        MZSlowVars(6, 18.0, MZSlowVarsConfig(membrane_mode="current", m_frozen_E=np.full(4, 2.0)),
                   NE=4, core_mask_E=np.zeros(4, bool))                       # requires full_conductance


# ============================== Task 5: Phase-1 arm verdict ==============================
_PRE_HI = dict(mean_rate=90.0, coherence=0.97)                                # established 16Hz high state
_BURST = np.tile([0.0, 1.0, 2.0, 1.0], 200)                                   # oscillating m -> bursting
_MONO = np.linspace(0, 5, 800)                                               # monotone m -> not bursting
_MOFF_DIST, _MOFF_COV = 60.0, 0


def test_phase1_transformed():
    post = dict(dominant_hz=5.0, event_ipi_hz=5.2, mean_rate=45.0, coverage=10,
                dist_to_real=12.0, six_band_ddb=[8, 9, 7, 8, 5, 1])
    v = H2.phase1_verdict(_PRE_HI, post, _BURST, _MOFF_DIST, _MOFF_COV,
                          dict(numerical_unsafe=False, runaway_early_stop_ms=None))
    assert v["verdict"] == "transformed_broadband_spiky" and all(v["criteria"].values())


def test_phase1_unchanged_16hz():
    post = dict(dominant_hz=16.0, event_ipi_hz=16.0, mean_rate=90.0, coverage=0,
                dist_to_real=60.0, six_band_ddb=[-20, -14, -9, 16, 3, 0])
    assert H2.phase1_verdict(_PRE_HI, post, _MONO, _MOFF_DIST, _MOFF_COV,
                             dict(numerical_unsafe=False, runaway_early_stop_ms=None))["verdict"] == "unchanged_16Hz"


def test_phase1_collapsed_sparse():
    post = dict(dominant_hz=3.5, event_ipi_hz=3.6, mean_rate=9.0, coverage=2,
                dist_to_real=13.0, six_band_ddb=[5, 6, 5, 6, 4, 5])
    assert H2.phase1_verdict(_PRE_HI, post, _BURST, _MOFF_DIST, _MOFF_COV,
                             dict(numerical_unsafe=False, runaway_early_stop_ms=None))["verdict"] == "collapsed_sparse"


def test_phase1_silenced_stalled_unsafe():
    silent = dict(dominant_hz=2.0, event_ipi_hz=2.0, mean_rate=2.0, coverage=0, dist_to_real=40.0,
                  six_band_ddb=[-5, -5, -5, -5, -5, -5])
    assert H2.phase1_verdict(_PRE_HI, silent, _MONO, _MOFF_DIST, _MOFF_COV,
                             dict(numerical_unsafe=False, runaway_early_stop_ms=None))["verdict"] == "silenced"
    stalled = dict(dominant_hz=10.0, event_ipi_hz=10.0, mean_rate=55.0, coverage=5, dist_to_real=40.0,
                   six_band_ddb=[3, 4, 2, 6, 3, 0])
    assert H2.phase1_verdict(_PRE_HI, stalled, _MONO, _MOFF_DIST, _MOFF_COV,
                             dict(numerical_unsafe=False, runaway_early_stop_ms=None))["verdict"] == "stalled"
    ok = dict(dominant_hz=5.0, event_ipi_hz=5.0, mean_rate=45.0, coverage=10, dist_to_real=12.0,
              six_band_ddb=[8, 9, 7, 8, 5, 1])
    assert H2.phase1_verdict(_PRE_HI, ok, _BURST, _MOFF_DIST, _MOFF_COV,
                             dict(numerical_unsafe=True, runaway_early_stop_ms=None))["verdict"] == "unsafe"
