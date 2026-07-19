"""TDD for M4-MZ phenotype classifier + calibration helpers (src/topic4_mz_slowvars.py).

Contract: docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md §6/§8.
Thresholds are exercised on SYNTHETIC fixtures (anti-circularity, mirrors sef_hfo_m4_termination):
never tune the gate on the same z+m traces you then classify. Runner --confirm-run gate tests
are added in the runner section (Task 4).
"""
import numpy as np

from src.topic4_mz_slowvars import (
    MZBaseline, MZPhenotypeGates, classify_mz_run,
    pooled_quantiles_from_hist, replay_adaptation_peak, eta_m_from_frac, select_by_targets,
)


def _baseline(**kw):
    d = dict(n_events=10, dur_med=60.0, dur_hi=90.0, part_lo=0.02, part_hi=0.08,
             act_lo=20.0, act_hi=50.0, floor_af=0.01, baseline_rate=2.0, sigma_rate=1.0)
    d.update(kw)
    return MZBaseline(**d)


def _run(**kw):
    d = dict(n_events=3, peak_dur=60.0, peak_participation=0.05, peak_rate=30.0,
             peak_returned=True, max_dur=60.0, peak_af=0.06)
    d.update(kw)
    return d


# ---------------- phenotype classifier (7 labels) ----------------
def test_runaway_wins_shape_independent():
    # runaway_ms injected -> "runaway" regardless of the (bounded-looking) run metrics
    assert classify_mz_run(_run(), _baseline(), runaway_ms=812.0) == "runaway"


def test_insufficient_when_baseline_underpowered():
    # baseline slow-off had < min_base_events returning events -> cannot define expansion
    assert classify_mz_run(_run(), _baseline(n_events=1), runaway_ms=None) == "insufficient"


def test_expanded_returned_all_three_up_and_returns():
    rm = _run(peak_dur=150.0, peak_participation=0.20, peak_rate=100.0, peak_returned=True)
    assert classify_mz_run(rm, _baseline(), runaway_ms=None) == "expanded_returned"


def test_frozen_event_bar_overrides_per_trajectory_max():
    """P0-2: a passed event_bar (frozen from the same-seed slow-off) is used verbatim instead of each
    trajectory's own af.max(). A bar above the trajectory's peak -> 0 events; a tiny bar -> >= default."""
    import os
    import sys
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import run_topic4_mz_slowvars as R

    rng = np.random.default_rng(0)
    n_cells, dt, n_steps = 200, 0.1, 6000              # 600 ms; baseline window (5-50ms) stays quiet
    spk = np.zeros((n_steps, n_cells), bool)
    spk[2000:2200, :] = rng.random((200, n_cells)) < 0.8      # big event (~all cells) -> high af
    spk[4000:4100, :50] = rng.random((100, 50)) < 0.8         # small event (25% of cells) -> low af
    res = dict(E_spk_bool=spk, rate_E=spk.mean(axis=1) * 1000.0)

    bar = R.slowoff_event_bar(res, dt)
    assert bar > 0.0
    ev_default, *_ = R._events_from_res(res, dt)                    # self-referential af.max() bar
    ev_tiny, *_ = R._events_from_res(res, dt, event_bar=1e-6)       # frozen tiny -> >= default
    ev_huge, *_ = R._events_from_res(res, dt, event_bar=2.0)        # above max possible af (<=1) -> none
    assert len(ev_tiny) >= len(ev_default) >= 1
    assert len(ev_huge) == 0


def test_expanded_bounded_all_three_up_but_no_return():
    rm = _run(peak_dur=150.0, peak_participation=0.20, peak_rate=100.0, peak_returned=False)
    assert classify_mz_run(rm, _baseline(), runaway_ms=None) == "expanded_bounded"


def test_not_expanded_if_only_one_metric_up():
    # duration up but participation + rate at baseline -> NOT expanded (AND of all three)
    rm = _run(peak_dur=150.0, peak_participation=0.05, peak_rate=30.0)
    assert classify_mz_run(rm, _baseline(), runaway_ms=None) == "interictal_like"


def test_suppress_when_activity_killed_below_baseline():
    rm = _run(n_events=1, peak_participation=0.005, peak_rate=5.0, peak_af=0.006)
    assert classify_mz_run(rm, _baseline(), runaway_ms=None) == "suppress"


def test_suppress_when_no_events():
    rm = _run(n_events=0, peak_participation=0.0, peak_rate=0.0, peak_af=0.0)
    assert classify_mz_run(rm, _baseline(), runaway_ms=None) == "suppress"


def test_fragment_many_short_events():
    rm = _run(n_events=25, peak_dur=20.0, max_dur=25.0, peak_participation=0.04, peak_rate=25.0)
    assert classify_mz_run(rm, _baseline(), runaway_ms=None) == "fragment"


def test_interictal_like_when_comparable_to_baseline():
    rm = _run(n_events=8, peak_dur=60.0, peak_participation=0.05, peak_rate=30.0, max_dur=70.0)
    assert classify_mz_run(rm, _baseline(), runaway_ms=None) == "interictal_like"


# ---------------- calibration helpers ----------------
def test_pooled_quantiles_from_uniform_hist():
    edges = np.linspace(0.0, 10.0, 101)          # 100 bins width 0.1
    hist = np.ones(100, dtype=np.int64) * 5      # uniform on [0,10]
    q = pooled_quantiles_from_hist(hist, edges, [0.5, 0.75, 0.9])
    assert abs(q[0.5] - 5.0) < 0.15
    assert abs(q[0.75] - 7.5) < 0.15
    assert abs(q[0.9] - 9.0) < 0.15


def test_replay_adaptation_single_spike_peak_is_one():
    E = np.zeros((100, 2), dtype=bool)
    E[0, 0] = True                               # cell 0 spikes once at t=0
    peak = replay_adaptation_peak(E, dt=0.1, tau_adp=2000.0)
    assert abs(peak[0] - 1.0) < 1e-9             # decay of 0 then +1
    assert peak[1] == 0.0                         # cell 1 never spikes


def test_replay_adaptation_more_spikes_higher_peak_and_event_mask():
    E = np.zeros((200, 2), dtype=bool)
    E[:50, 0] = True                             # cell 0 spikes every step for 50 steps
    E[0, 1] = True                               # cell 1 spikes once
    peak = replay_adaptation_peak(E, dt=0.1, tau_adp=2000.0)
    assert peak[0] > peak[1] > 0.0
    # event mask excluding the spiking window -> cell 0 peak only reflects the decayed tail
    mask = np.zeros(200, bool); mask[100:] = True
    peak_late = replay_adaptation_peak(E, dt=0.1, tau_adp=2000.0, event_step_mask=mask)
    assert peak_late[0] < peak[0]


def test_eta_m_from_frac():
    assert abs(eta_m_from_frac(0.10, I_EE_scale=8.0, peak_m=4.0) - 0.20) < 1e-12


def test_select_by_targets_picks_closest():
    vals = [0.95, 0.82, 0.55, 0.48, 0.30, 0.18]
    idx = select_by_targets(vals, targets=[0.8, 0.5, 0.2])
    assert idx == [1, 3, 5]                       # closest to 0.8, 0.5, 0.2


# ==================== runner (scripts/run_topic4_mz_slowvars.py) ====================
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUNNER = os.path.join(_ROOT, "scripts", "run_topic4_mz_slowvars.py")
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "src", "snn_engine"))


def _synth_res(burst_windows, nsteps=3000, NE=200, dt=0.1, frac=0.3, seed=0):
    """Synthetic simulate_kick-like result: quiet baseline + explicit burst windows (returning)."""
    rng = np.random.default_rng(seed)
    spk = np.zeros((nsteps, NE), bool)
    for (s, e) in burst_windows:
        for t in range(s, e):
            spk[t, rng.choice(NE, int(frac * NE), replace=False)] = True
    rate_E = spk.sum(axis=1) / NE / dt * 1e3
    return dict(E_spk_bool=spk, rate_E=rate_E, runaway_early_stop_ms=None)


def test_runner_import_is_side_effect_free():
    import importlib
    m = importlib.import_module("run_topic4_mz_slowvars")
    assert hasattr(m, "main") and hasattr(m, "run_mz_cell")
    assert hasattr(m, "extract_run_metrics") and hasattr(m, "compute_baseline_ref")


def test_runner_refuses_sim_without_confirm_run():
    # user test 12: a sim subcommand without --confirm-run must NOT start a simulation
    r = subprocess.run([sys.executable, _RUNNER, "calibrate"], capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "confirm-run" in (r.stdout + r.stderr).lower()


def test_compute_baseline_ref_counts_returning_events():
    import importlib
    m = importlib.import_module("run_topic4_mz_slowvars")
    res = _synth_res([(700, 1000), (1400, 1700), (2100, 2400)])   # 3 returning bursts, 30 ms each
    base = m.compute_baseline_ref(res, dt=0.1)
    assert base.n_events == 3
    assert base.dur_hi > 0 and base.part_hi > 0 and base.act_hi > 0


def test_extract_run_metrics_detects_event_and_return():
    import importlib
    m = importlib.import_module("run_topic4_mz_slowvars")
    base = MZBaseline(n_events=5, dur_med=30.0, dur_hi=35.0, part_lo=0.1, part_hi=0.5,
                      act_lo=100.0, act_hi=500.0, floor_af=0.0, baseline_rate=0.0, sigma_rate=1.0)
    res = _synth_res([(1000, 1300)])                              # one 30 ms returning burst
    rm, events, af, bin_w, runaway_ms = m.extract_run_metrics(res, 0.1, base)
    assert rm["n_events"] == 1
    assert rm["peak_dur"] >= 8.0
    assert rm["peak_returned"] is True                            # activity returns to baseline band
    assert runaway_ms is None                                     # 30 ms burst is not sustained runaway
