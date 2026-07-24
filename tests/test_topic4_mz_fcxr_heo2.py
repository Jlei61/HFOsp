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

FS = 1000.0


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
