"""Section-8 slow-fast analysis (task §8): treat z/m/S_G as slow parameters, characterize the fast
E/I burst subsystem's cycle-to-cycle behavior. Tested on synthetic burst trains so the drift / stationarity
discrimination is verified independently. NEVER emits 'limit_cycle' -- proving a limit cycle needs a
frozen-slow repeated trajectory + Poincare/perturbation-return, which the natural run does not provide;
the honest ceiling is 'candidate_inner_cycle' (stationary) vs 'transient_burst_train' (drifting).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_slowfast import detect_bursts, cycle_stats, slowfast_verdict  # noqa: E402


def _train(ibi_ms, amps, dt_ms=5.0, width_bins=3, pad=200):
    """Synthetic core-rate burst train: bursts at cumulative IBIs with given peak amps."""
    times = np.cumsum([pad] + list(ibi_ms))
    n = int(times[-1] + pad)
    r = np.zeros(n)
    for t, a in zip(times, amps):
        ti = int(t)
        r[max(0, ti - width_bins):ti + width_bins] = a
    return r, dt_ms


def test_detect_bursts_counts_and_amps():
    r, dt = _train([200] * 9, [100.0] * 10)
    idx, amp = detect_bursts(r, dt)
    assert 9 <= len(idx) <= 11
    assert abs(np.median(amp) - 100.0) < 1e-6


def test_stationary_train_is_candidate_inner_cycle():
    r, dt = _train([200] * 19, [120.0] * 20)          # constant IBI + amp over 20 cycles
    cs = cycle_stats(*detect_bursts(r, dt), dt)
    assert cs["n_bursts"] >= 10
    assert cs["ibi_cv_tail"] < 0.15 and cs["amp_cv_tail"] < 0.15
    assert slowfast_verdict(cs) == "candidate_inner_cycle"


def test_escalating_train_is_transient_burst_train():
    ibi = list(np.linspace(400, 120, 15))              # IBI shrinking = accelerating
    amp = list(np.linspace(60, 260, 16))               # amplitude growing = escalating
    cs = cycle_stats(*detect_bursts(*_train(ibi, amp)), 5.0)
    assert cs["n_bursts"] >= 10
    assert cs["ibi_drift_frac"] > 0.3 or cs["amp_drift_frac"] > 0.3
    assert slowfast_verdict(cs) == "transient_burst_train"


def test_too_few_bursts_is_not_oscillatory():
    r, dt = _train([200], [100.0, 100.0])              # 2 bursts only
    cs = cycle_stats(*detect_bursts(r, dt), dt)
    assert slowfast_verdict(cs) == "not_oscillatory"


def test_verdict_never_claims_limit_cycle():
    for ibi, amp in (([200] * 19, [120.0] * 20), (list(np.linspace(400, 120, 15)), list(np.linspace(60, 260, 16)))):
        cs = cycle_stats(*detect_bursts(*_train(ibi, amp)), 5.0)
        assert slowfast_verdict(cs) in ("candidate_inner_cycle", "transient_burst_train", "not_oscillatory")
