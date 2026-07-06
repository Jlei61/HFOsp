"""Stage-4 v2: spontaneous big-focus q_I/g_K working-point search + stim GIF.

Task 1 (build + single-focus safety + vth/early-abort on `_simulate_continuous`) and
Task 2 (working-point classifier) tests. The SNN-touching tests are marked `slow`
(they build a real L=20 sheet and run short sims); the classifier tests are pure.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(ROOT, "src", "snn_engine"), os.path.join(ROOT, "scripts"),
           os.path.join(ROOT, "scripts", "paper_figures"), ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import plot_fig_m3a_v2_2_hG_runaway_transition_gif as H  # noqa: E402
from run_stage4_v2_workpoint_search import classify_workpoint, is_working_point  # noqa: E402


# ---- Task 2: working-point classifier (pure, fast) ----

def test_classify_one_shot_burst():
    assert classify_workpoint([30.0], runaway_ms=32.0, aborted_ms=180.0, T=2500.0) == "one_shot_burst"


def test_classify_abort_counts_as_runaway():
    # abort fired at 250 ms with only 1 prior event -> still a burst, NOT train_no_runaway
    assert classify_workpoint([40.0], runaway_ms=None, aborted_ms=250.0, T=2500.0) == "one_shot_burst"


def test_classify_train_then_runaway():
    v = classify_workpoint([300.0, 700.0, 1100.0, 1500.0], runaway_ms=1800.0, aborted_ms=None, T=2500.0)
    assert v == "train_then_runaway" and is_working_point(v)


def test_classify_train_then_runaway_via_abort():
    # a real working point can also end via abort (sustained runaway detected online)
    v = classify_workpoint([300.0, 700.0, 1100.0, 1500.0], runaway_ms=None, aborted_ms=1800.0, T=2500.0)
    assert v == "train_then_runaway"


def test_classify_train_no_runaway():
    v = classify_workpoint([300.0, 800.0, 1400.0, 2100.0], runaway_ms=None, aborted_ms=None, T=2500.0)
    assert v == "train_no_runaway" and not is_working_point(v)


def test_classify_silent():
    assert classify_workpoint([], runaway_ms=None, aborted_ms=None, T=2500.0) == "silent"


@pytest.mark.slow
def test_build_stage4_patch_single_core():
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, eta_K=0.5,
                           tau_K=200.0, core_mean=17.0, core_std=1.5, core_radius=6.0,
                           T=150.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    assert S["layout"]["kind"] == "stage4_patch"
    assert len(S["layout"]["foci"]) == 1                         # ONE big focus
    assert S["core_mask"].shape[0] == S["N"]
    assert int(S["core_mask"][:S["NE"]].sum()) >= 200            # r=6 mm disk on L=20 is large
    assert S["patch_vth"].shape[0] == S["N"]


@pytest.mark.slow
def test_spontaneous_single_focus_never_touches_tempB():
    # stage4_patch has ONE focus; _source_xy must not index foci[1], and n_pulses=0 must build no masks
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, eta_K=0.0,
                           core_mean=16.5, core_std=1.5, core_radius=6.0,
                           T=120.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    assert np.allclose(H._source_xy(S, "tempA"), H._source_xy(S, "tempB"))   # both -> the one focus
    # the sim runs to completion (no IndexError from a missing second focus)
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"])
    assert res["E_spk_bool"].shape[0] == int(round(cfg.T / S["p"].dt))


@pytest.mark.slow
def test_early_abort_uses_shared_runaway_criterion():
    cfg = H.ProtocolConfig(layout="stage4_patch", top="qI", use_gK=True, eta_K=0.0,
                           core_mean=16.5, core_std=1.5, core_radius=6.0,
                           T=600.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    res = H._simulate_continuous(S, cfg, record_gif=False, vth=S["patch_vth"], abort_on_runaway=True)
    assert res["aborted_ms"] is not None and res["aborted_ms"] < 400.0     # hot core bursts, abort fires
    n = res["E_spk_bool"].shape[0]
    assert n <= int(round(res["aborted_ms"] / S["p"].dt)) + 1              # arrays truncated at abort
    # the SAME shared criterion, run post-hoc on the (truncated) rate, agrees an onset exists
    rate_hz = np.asarray(res["rate_E"], float)
    assert H._first_sustained(H._smooth_rate(rate_hz, S["p"].dt, 20.0), S["p"].dt, 120.0, 100.0) is not None


@pytest.mark.slow
def test_abort_off_and_vth_none_is_unchanged():
    cfg = H.ProtocolConfig(layout="subject1146", top="qI", use_gK=True, eta_K=0.0,
                           use_hG=False, T=150.0, n_pulses=0, seed=1)
    S = H._build(cfg)
    a = H._simulate_continuous(S, cfg, record_gif=False)
    b = H._simulate_continuous(S, cfg, record_gif=False, vth=None, abort_on_runaway=False)
    assert np.array_equal(a["E_spk_bool"], b["E_spk_bool"])
    assert b.get("aborted_ms") is None
