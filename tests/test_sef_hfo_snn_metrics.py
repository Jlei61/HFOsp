import numpy as np

from src.sef_hfo_snn_metrics import onset_times, onset_axis, peak_active_fraction


def test_onset_axis_recovers_linear_wave_direction():
    rng = np.random.default_rng(0)
    NE = 400
    posE = rng.uniform(0, 3, size=(NE, 2))
    dt, t_kick = 0.1, 150.0
    nsteps = int(300 / dt)
    spk = np.zeros((nsteps, NE), bool)
    # onset increases along +x: neuron at x fires at t_kick + 20*x ms
    for i in range(NE):
        ti = int((t_kick + 20.0 * posE[i, 0]) / dt)
        if ti < nsteps:
            spk[ti, i] = True
    onset = onset_times(spk, dt, t_kick)
    axis = onset_axis(posE, onset, min_n=20)
    ang = np.degrees(np.arctan2(axis[1], axis[0])) % 180.0
    assert min(ang, 180 - ang) < 15.0          # ~along x (0 deg)


def test_peak_active_fraction_counts_distinct():
    dt = 0.1
    spk = np.zeros((100, 10), bool)
    spk[40:45, :6] = True                       # 6/10 distinct in one 5ms bin
    paf = peak_active_fraction(spk, dt, 0.0, 10.0, bin_ms=5.0)
    assert abs(paf - 0.6) < 1e-9


# ---- improved self-limit / ignition metrics (2026-06-08c, review fix 1A/1B) ----
from src.sef_hfo_snn_metrics import pre_kick_ignition, self_limit, event_peak_time


def _trace(dt, t_max, bumps):
    """rate trace (Hz) = sum of gaussian bumps (center_ms, amp_hz, width_ms) on ~2Hz rest."""
    t = np.arange(0, t_max, dt)
    r = np.full_like(t, 2.0)
    for c, a, w in bumps:
        r += a * np.exp(-((t - c) ** 2) / (2 * w ** 2))
    return r


def test_pre_kick_ignition_detects_prekick_burst():
    dt = 0.1
    # burst at 80ms (pre-kick), kick at 150ms
    r = _trace(dt, 450, [(80, 200, 6)])
    ig, lat = pre_kick_ignition(r, dt, t_kick=150.0, rest_lo=20.0, thresh_hz=10.0)
    assert ig is True
    assert 60 < lat < 90                         # crossing on the rising edge of the 80ms bump
    # a quiet trace (only a post-kick event) does NOT ignite pre-kick
    r2 = _trace(dt, 450, [(185, 150, 6)])
    ig2, lat2 = pre_kick_ignition(r2, dt, t_kick=150.0)
    assert ig2 is False and np.isnan(lat2)


def test_self_limit_returns_for_transient_not_for_sustained():
    dt = 0.1
    r_ret = _trace(dt, 450, [(185, 150, 6)])     # single transient -> returns to rest
    m = self_limit(r_ret, dt, t_kick=150.0)
    assert m["returned"] is True
    assert m["rest_rate"] < 5.0 and m["peak"] > 100.0
    assert m["burst_duration_ms"] < 120.0        # a ~6ms-sigma bump is short
    r_sus = np.full(int(450 / dt), 2.0); r_sus[int(150 / dt):] = 80.0   # ignites @kick, never returns
    m2 = self_limit(r_sus, dt, t_kick=150.0)
    assert m2["returned"] is False               # never decays back (rest 2 vs decay 80)


def test_event_peak_time_finds_post_kick_peak():
    dt = 0.1
    r = _trace(dt, 450, [(80, 200, 6), (300, 150, 6)])   # pre-kick + post-kick peaks
    assert abs(event_peak_time(r, dt, 150.0, 350.0) - 300.0) < 2.0   # picks the post-kick one
