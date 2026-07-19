"""Contract tests for Topic 4 MZ slow–fast dynamical transition (design §10).

Pure-function tests use no SNN; the tiny-network smoke tests build a small substrate so the
freeze / independent-replay invariants are exercised without the full E1146 substrate.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import src.topic4_mz_slow_fast_transition as MZSF  # noqa: E402


def test_module_imports_and_schema():
    assert MZSF.SCHEMA_VERSION == "mz-slow-fast-transition-1.0"


# ---------------------------------------------------------------- Task 2: branch_rng_state + wilson_ci
def test_branch_rng_state_deterministic_and_independent():
    a = MZSF.branch_rng_state(1, "mz_runaway", "pre_onset_100ms", 0)
    a2 = MZSF.branch_rng_state(1, "mz_runaway", "pre_onset_100ms", 0)
    b = MZSF.branch_rng_state(1, "mz_runaway", "pre_onset_100ms", 1)
    assert a == a2                                     # deterministic in inputs
    assert a != b                                      # distinct branch idx -> distinct stream
    assert a["bit_generator"] == "PCG64"               # swappable into a PCG64 LoopState.rng_state
    g = np.random.default_rng(0); g.bit_generator.state = a
    x = g.standard_normal(8)
    g2 = np.random.default_rng(0); g2.bit_generator.state = b
    assert not np.allclose(x, g2.standard_normal(8))   # independent future noise


def test_branch_rng_state_varies_with_every_key_field():
    base = MZSF.branch_rng_state(1, "mz_runaway", "pre_onset_100ms", 0)
    assert base != MZSF.branch_rng_state(3, "mz_runaway", "pre_onset_100ms", 0)   # seed
    assert base != MZSF.branch_rng_state(1, "mz_plateau", "pre_onset_100ms", 0)   # condition
    assert base != MZSF.branch_rng_state(1, "mz_runaway", "baseline_1000ms", 0)   # state


def test_wilson_ci_bounds_and_monotone():
    lo0, hi0 = MZSF.wilson_ci(0, 20)
    assert lo0 == 0.0 and 0.0 <= hi0 <= 1.0
    loN, hiN = MZSF.wilson_ci(20, 20)
    assert hiN == 1.0 and 0.0 <= loN <= 1.0
    lo1, hi1 = MZSF.wilson_ci(5, 20)
    lo2, hi2 = MZSF.wilson_ci(15, 20)
    assert lo1 < lo2 and hi1 < hi2                     # monotone in k
    lo, hi = MZSF.wilson_ci(10, 20)
    assert lo < 0.5 < hi                               # brackets the point estimate
    assert np.isnan(MZSF.wilson_ci(0, 0)[0])           # n=0 -> nan


# ---------------------------------------------------------------- Task 3: recovery_time
def test_recovery_time_returns_finite_for_decay():
    dt = 0.1
    t = np.arange(6000) * dt
    rate = 5.0 + 20.0 * np.exp(-t / 50.0)              # elevated, decays toward 5 Hz (in band)
    rt = MZSF.recovery_time(rate, dt, pulse_off_idx=0, band_lo=4.0, band_hi=6.0, min_hold_ms=50.0)
    assert rt is not None and 80.0 < rt < 400.0


def test_recovery_time_censored_when_never_returns():
    rate = np.full(3000, 40.0)                         # stays elevated -> never re-enters band
    assert MZSF.recovery_time(rate, 0.1, 0, band_lo=4.0, band_hi=6.0, min_hold_ms=50.0) is None


def test_recovery_time_already_in_band_is_near_zero():
    rate = np.full(3000, 5.0)                          # already inside [4,6]
    rt = MZSF.recovery_time(rate, 0.1, 0, band_lo=4.0, band_hi=6.0, min_hold_ms=50.0)
    assert rt is not None and rt < 25.0                # essentially immediate


# ---------------------------------------------------------------- Task 4: schedules + classifier
def test_state_step_schedule():
    sched = MZSF.state_step_schedule(9293.6, 0.1)
    assert list(sched) == ["baseline_1000ms", "mid_fraction", "pre_onset_2000ms", "pre_onset_1000ms",
                           "pre_onset_500ms", "pre_onset_200ms", "pre_onset_100ms"]
    assert sched["baseline_1000ms"] == 10000
    assert sched["mid_fraction"] == 46468
    assert sched["pre_onset_100ms"] == 91936
    assert list(sched.values()) == sorted(sched.values())     # chronological


def test_matched_d_times():
    t = np.arange(0, 1000, 10.0)
    D = np.linspace(0.0, 0.1, t.size)
    got = MZSF.matched_d_times(D, t, [0.02, 0.2])
    assert got[0.02] is not None and abs(got[0.02] - 200.0) < 20.0
    assert got[0.2] is None                            # never reached -> censored


def _ps(D, p, eps, tau):
    return [dict(D=D[i], a=0.0, p_runaway=p[i], p_runaway_ci=(max(0, p[i] - .1), min(1, p[i] + .1)),
                 epsilon_c=eps[i], tau_rec=tau[i]) for i in range(len(D))]


def test_classify_dynamical_tipping():
    ps = _ps([0.02, 0.05, 0.08, 0.10], [0.0, 0.05, 0.9, 1.0], [0.20, 0.10, 0.025, 0.0], [30, 50, 120, None])
    out = MZSF.classify_transition(ps, natural_crosses=True, plateau_outside=True)
    assert out["label"] == "dynamical_tipping" and out["features"]


def test_classify_finite_amplitude_escape():
    ps = _ps([0.02, 0.05, 0.08, 0.10], [0.0, 0.0, 0.05, 0.05], [0.20, 0.15, 0.10, 0.05], [30, 35, 40, 45])
    out = MZSF.classify_transition(ps, natural_crosses=False, plateau_outside=False)
    assert out["label"] == "finite_amplitude_escape"


def test_classify_noise_driven_escape():
    ps = _ps([0.02, 0.05, 0.08, 0.10], [0.0, 0.2, 0.5, 0.8], [None, None, None, None], [30, 32, 31, 33])
    out = MZSF.classify_transition(ps, natural_crosses=True, plateau_outside=False)
    assert out["label"] == "noise_driven_escape"


def test_classify_smooth_crossover():
    ps = _ps([0.02, 0.05, 0.08, 0.10], [0.0, 0.0, 0.02, 0.01], [0.20, 0.20, 0.20, 0.20], [30, 30, 32, 31])
    out = MZSF.classify_transition(ps, natural_crosses=False, plateau_outside=True)
    assert out["label"] == "smooth_crossover"


def test_classify_unresolved_when_too_few():
    ps = _ps([0.02, 0.05], [0.0, np.nan], [None, None], [None, None])
    out = MZSF.classify_transition(ps, natural_crosses=False, plateau_outside=False)
    assert out["label"] == "unresolved"
