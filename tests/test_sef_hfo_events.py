"""TDD for src/sef_hfo_events — Step 1 noise-driven event detection + OU noise.

Detector tests run on SYNTHETIC active-fraction traces (fast, no field sim) to
lock the regime logic against the frozen contract §1. Noise tests lock the
amp=0 identity and seed reproducibility; one short field run verifies the OU
drive is wired through integrate_lif_field.
"""
import numpy as np

from src.sef_hfo_events import (
    EVENT_ON_FRAC,
    classify_run,
    detect_events,
    make_ou_noise,
)
from src.sef_hfo_lif import integrate_lif_field, mean_field

DT = 0.25


def _times(t_max):
    return np.arange(0, t_max, DT)


def _bump(t, center, width, peak):
    return peak * np.exp(-0.5 * ((t - center) / width) ** 2)


# ---------------------------------------------------------------------------
# Detector regime logic
# ---------------------------------------------------------------------------

def test_single_bump_is_one_discrete_event():
    t = _times(300.0)
    ext = _bump(t, 100.0, 10.0, 0.1)
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "discrete_events", res["label"]
    assert res["n_events"] == 1, res["n_events"]


def test_two_close_bumps_merge_to_one():
    t = _times(300.0)
    ext = _bump(t, 100.0, 2.0, 0.1) + _bump(t, 110.0, 2.0, 0.1)  # gap ~1.4ms < 12ms
    evs = detect_events(ext, DT)
    assert len(evs) == 1, [(e["t_on"], e["t_off"]) for e in evs]


def test_two_separated_bumps_are_two_events():
    t = _times(300.0)
    ext = _bump(t, 100.0, 2.0, 0.1) + _bump(t, 140.0, 2.0, 0.1)  # gap ~30ms > 12ms
    evs = detect_events(ext, DT)
    assert len(evs) == 2, [(e["t_on"], e["t_off"]) for e in evs]


def test_sustained_plateau():
    t = _times(520.0)
    ext = np.where((t >= 50.0), 0.1, 0.0)  # on from 50ms, never returns (>400ms)
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "sustained", res


def test_runaway():
    t = _times(250.0)
    ext = np.where(t >= 50.0, 0.7, 0.0)  # high and stays high to the end (not returned)
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "runaway", res


def test_captured_high_window_B_only():
    """Elevated, non-returning, moderate plateau + final field at the HIGH root.

    Window B (bistable, 2 roots) -> captured_high. Same trace under window A
    (no capture check) must NOT be labeled captured_high.
    """
    t = _times(300.0)
    ext = np.where((t >= 50.0), 0.1, 0.0)[: int(300.0 / DT)]
    # truncate the "never returns" to <400ms so it is not 'sustained'
    ext = np.where((t >= 50.0) & (t <= 250.0), 0.1, 0.0)
    opB = mean_field(1.0, w_ee_mult=1.4)
    assert len(opB["roots"]) >= 2
    hi = opB["roots"][-1]["nuE"]
    rE_final = np.full((16, 16), hi)  # field settled at the high root
    resB = classify_run(ext, DT, opB, rE_final=rE_final, window="B")
    assert resB["label"] == "captured_high", resB
    resA = classify_run(ext, DT, mean_field(1.0), rE_final=rE_final, window="A")
    assert resA["label"] != "captured_high", resA


def test_extinction_when_below_threshold():
    t = _times(300.0)
    ext = np.full_like(t, 0.5 * EVENT_ON_FRAC)  # always below ON threshold
    op = mean_field(1.0)
    res = classify_run(ext, DT, op, window="A")
    assert res["label"] == "extinction_only", res
    assert res["n_events"] == 0


# ---------------------------------------------------------------------------
# OU noise
# ---------------------------------------------------------------------------

def test_noise_amp_zero_is_identity():
    """sigma_noise=0 -> the field run is identical to the deterministic run."""
    op = mean_field(1.0)
    n, L = 32, 16.0
    zero_noise = make_ou_noise(n, L, DT, sigma_noise=0.0, seed=0)
    assert zero_noise(0.0) == 0.0 and zero_noise(10.0) == 0.0
    ext_noise, _ = integrate_lif_field(op, zero_noise, dt=DT, t_max=40.0, n=n, L=L)
    ext_det, _ = integrate_lif_field(op, lambda t: 0.0, dt=DT, t_max=40.0, n=n, L=L)
    assert np.array_equal(ext_noise, ext_det)


def test_noise_seed_reproducible_and_distinct():
    n, L = 32, 16.0
    f0a = make_ou_noise(n, L, DT, sigma_noise=2.0, seed=7)
    f0b = make_ou_noise(n, L, DT, sigma_noise=2.0, seed=7)
    f1 = make_ou_noise(n, L, DT, sigma_noise=2.0, seed=8)
    a = [f0a(i * DT) for i in range(5)]
    b = [f0b(i * DT) for i in range(5)]
    c = [f1(i * DT) for i in range(5)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))   # same seed -> identical
    assert not np.array_equal(a[-1], c[-1])                  # different seed -> differ


def test_noise_steady_state_std_matches_sigma():
    n, L = 48, 16.0
    sigma = 2.0
    f = make_ou_noise(n, L, DT, sigma_noise=sigma, tau_noise=5.0, ell_noise=0.5, seed=3)
    # advance well past tau_noise, then sample the per-pixel std over the field
    field = None
    for i in range(4000):
        field = f(i * DT)
    assert 0.5 * sigma < field.std() < 1.6 * sigma, field.std()
