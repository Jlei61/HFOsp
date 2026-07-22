"""FCXR-LC1 lifecycle classifier — synthetic-window contract (design §3, plan clauses L1-L7 + anti-cheat).

Each test = one clause. The classifier is pure logic over an ordered list of analysis windows; these tests
build synthetic window sequences (no simulation) so the state machine is pinned independently of the SNN.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.topic4_mz_fcxr_lifecycle import (  # noqa: E402
    window_regime, classify_lifecycle, depletion_coordinate, build_windows, LC_THRESHOLDS,
)


def _band(win_ms=1000.0, **kw):
    b = dict(win_ms=win_ms, event_rate_lo=1.0, event_rate_hi=10.0, recruit_p90=0.30)
    b.update(kw)
    return b


def _w(occ, er, rec=0.0, unsafe=False):
    return dict(occ=occ, event_rate_hz=er, recruit_frac=rec, numerical_unsafe=unsafe)


# window archetypes
II = _w(0.02, 5.0, rec=0.10)     # interictal: low occ, in-band returning events
ICT = _w(0.80, 60.0, rec=0.60)   # ictal-like: high occ AND recruitment beyond baseline P90
DEN = _w(0.25, 20.0, rec=0.20)   # dense event train: above the occ floor but not ictal
SIL = _w(0.00, 0.0, rec=0.0)     # silent: no events, low occ  -> NOT interictal


# ---- window regime mapping (building block for L2/L3) ----
def test_window_regime_mapping():
    b = _band()
    assert window_regime(II, b) == "INTERICTAL"
    assert window_regime(ICT, b) == "ICTAL"
    assert window_regime(_w(0.80, 60.0, rec=0.20), b) == "DENSE"   # high occ but NOT recruited -> not ictal
    assert window_regime(DEN, b) == "DENSE"
    assert window_regime(SIL, b) == "SILENT"
    assert window_regime(_w(0.02, 0.2), b) == "OTHER"              # sub-band trickle is not a return
    assert window_regime(_w(0.02, 5.0, unsafe=True), b) == "UNSAFE"


# ---- L1: safety takes priority ----
def test_L1_runaway_and_unsafe_take_priority():
    b = _band()
    assert classify_lifecycle([II] * 3, b, runaway=True)["label"] == "RUNAWAY"
    ws = [II] * 3 + [_w(0.8, 60.0, 0.6, unsafe=True)] + [II] * 3
    assert classify_lifecycle(ws, b)["label"] == "NUMERICAL_UNSAFE"


# ---- L2: all interictal -> baseline ----
def test_L2_all_interictal_is_baseline():
    assert classify_lifecycle([II] * 12, _band())["label"] == "INTERICTAL_BASELINE"


# ---- L3: elevated but no >=ICTAL_MS bout -> dense event train ----
def test_L3_elevated_no_bout_is_dense_event_train():
    b = _band()
    assert classify_lifecycle([II] * 5 + [DEN] * 5, b)["label"] == "DENSE_EVENT_TRAIN"
    # a single sub-ICTAL_MS ictal window (win_ms=500 -> need 2 windows for a bout) is NOT a bout either
    bf = _band(win_ms=500.0)
    assert classify_lifecycle([II] * 5 + [ICT] + [II] * 5, bf)["label"] == "DENSE_EVENT_TRAIN"


# ---- real-baseline shape: warm-up silence + interictal + one end burst -> interictal baseline ----
def test_baseline_like_shape_is_interictal_baseline():
    # matches the observed seed1 slow-off run: 2 warm-up SILENT + 21 INTERICTAL + 1 isolated DENSE end burst
    ws = [SIL] * 2 + [II] * 21 + [DEN]
    assert classify_lifecycle(ws, _band())["label"] == "INTERICTAL_BASELINE"


# ---- isolated interictal burst does NOT shatter the pre-ictal interictal run ----
def test_isolated_pre_burst_still_recovers():
    b = _band()
    ws = [II] * 5 + [DEN] + [II] * 3 + [ICT] * 2 + [II] * 9   # a lone burst inside the pre-ictal interictal
    r = classify_lifecycle(ws, b)
    assert r["label"] == "RECOVERED_INTERICTAL"
    assert r["pre_ms"] >= LC_THRESHOLDS["PRE_MS"]             # smoothing keeps the 9 s pre-ictal run intact


# ---- L4 (load-bearing): full lifecycle -> recovered ----
def test_L4_full_lifecycle_recovered():
    b = _band()
    ws = [II] * 9 + [ICT] * 2 + [II] * 9
    r = classify_lifecycle(ws, b)
    assert r["label"] == "RECOVERED_INTERICTAL"
    assert r["bout"] == (9, 10)
    assert r["pre_ms"] >= LC_THRESHOLDS["PRE_MS"] and r["post_return_ms"] >= LC_THRESHOLDS["POST_MS"]


# ---- L5 (anti-cheat): a silent tail is PERMANENT_SILENCE, never RECOVERED ----
def test_L5_silent_tail_is_permanent_silence_not_recovered():
    b = _band()
    # post-ictal occupancy DROPS below the band (occ=0) but there are NO returning events -> must not recover
    ws = [II] * 9 + [ICT] * 2 + [SIL] * 9
    r = classify_lifecycle(ws, b)
    assert r["label"] == "PERMANENT_SILENCE"
    assert r["label"] != "RECOVERED_INTERICTAL"
    assert r["post_return_ms"] == 0.0


# ---- L5b: leading post-ictal silence THEN returning events is allowed to recover ----
def test_L5b_leading_silence_then_return_recovers():
    b = _band()
    ws = [II] * 9 + [ICT] * 2 + [SIL] * 2 + [II] * 9      # brief refractory silence then statistical return
    r = classify_lifecycle(ws, b)
    assert r["label"] == "RECOVERED_INTERICTAL"
    assert r["post_return_ms"] >= LC_THRESHOLDS["POST_MS"]


# ---- L6: ictal re-entry within the guard -> rapid relapse ----
def test_L6_reentry_within_guard_is_rapid_relapse():
    b = _band()
    ws = [II] * 9 + [ICT] * 2 + [DEN] + [ICT] * 2 + [II] * 3
    assert classify_lifecycle(ws, b)["label"] == "RAPID_RELAPSE"


# ---- L7: terminated but only a short return -> terminated refractory (no statistical return) ----
def test_L7_short_return_is_terminated_refractory():
    b = _band()
    ws = [II] * 9 + [ICT] * 2 + [II] * 3 + [DEN] * 5
    r = classify_lifecycle(ws, b)
    assert r["label"] == "TERMINATED_REFRACTORY"
    assert r["post_return_ms"] == 3000.0


# ---- bout to end -> ictal-like bounded (no autonomous termination observed) ----
def test_ictal_bout_to_end_is_ictal_like_bounded():
    b = _band()
    assert classify_lifecycle([II] * 9 + [ICT] * 4, b)["label"] == "ICTAL_LIKE_BOUNDED"


# ---- insufficient pre-ictal interictal -> unresolved (cannot call it a lifecycle) ----
def test_insufficient_pre_ictal_is_unresolved():
    b = _band()
    ws = [II] * 3 + [ICT] * 2 + [II] * 9
    assert classify_lifecycle(ws, b)["label"] == "UNRESOLVED"


# ---- runner CLI gates (--confirm-run required; dry-run runs no simulation) ----
RUNNER = os.path.join(ROOT, "scripts", "run_topic4_mz_fcxr_lifecycle.py")


def test_confirm_run_gate_required():
    import subprocess
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, RUNNER, "smoke", "--seed", "1", "--T", "100"],
                       capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)
    assert r.returncode != 0
    assert "REFUSING" in (r.stdout + r.stderr)


def test_dry_run_needs_no_confirm_and_runs_no_sim():
    import subprocess
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    r = subprocess.run([sys.executable, RUNNER, "dry-run", "--T", "24000", "--workers", "1"],
                       capture_output=True, text=True, cwd=ROOT, env=env, timeout=180)
    assert r.returncode == 0
    assert "dry-run" in r.stdout and "raster" in r.stdout


# ---- run -> windows reducer (pure over synthetic rate + events) ----
def test_build_windows_reducer():
    dt, win, roll_hi, n = 1.0, 1000.0, 5.0, 10000
    rate = np.ones(n)
    rate[4000:6000] = 20.0                                   # windows 4,5 are sustained above the band
    af_bin = 1.0
    af = np.full(n, 0.05)                                    # baseline spatial spread ~5% of E cells
    af[4000:6000] = 0.5                                      # ictal windows recruit ~50%
    events = [dict(t_on=wi * 1000 + 100 + k * 150) for wi in range(10) for k in range(5)]  # 5/window
    W = build_windows(rate, dt, af, af_bin, roll_hi, events, win, roll_ms=300.0, start_ms=0.0)
    assert len(W) == 10
    assert W[4]["occ"] > 0.8 and W[5]["occ"] > 0.8          # high segment occupies the window
    assert W[0]["occ"] < 0.2 and W[9]["occ"] < 0.2
    assert abs(W[0]["event_rate_hz"] - 5.0) < 1e-9          # 5 events in the trailing 1 s
    assert abs(W[4]["recruit_frac"] - 0.5) < 1e-9 and abs(W[0]["recruit_frac"] - 0.05) < 1e-9
    b = _band(win_ms=1000.0, recruit_p90=0.10)
    assert window_regime(W[0], b) == "INTERICTAL"
    assert window_regime(W[4], b) == "ICTAL"                # high occ + recruit 0.5 > baseline P90 0.10


# ---- slow-variable phase-portrait coordinate ----
def test_depletion_coordinate():
    p = np.array([1.0, 1.0, 2.0])
    assert depletion_coordinate(np.ones(3), p) == 0.0                       # fully available
    assert depletion_coordinate(np.zeros(3), p) == 1.0                      # fully depleted
    assert abs(depletion_coordinate(np.full(3, 0.5), p) - 0.5) < 1e-12
    assert abs(depletion_coordinate(np.array([1.0, 1.0, 0.0]), p) - 0.5) < 1e-12  # weight-2 neuron dominates
    with pytest.raises(ValueError):
        depletion_coordinate(np.zeros(3), np.zeros(3))                      # zero-sum weights
    with pytest.raises(ValueError):
        depletion_coordinate(np.zeros(2), np.zeros(3))                      # shape mismatch
