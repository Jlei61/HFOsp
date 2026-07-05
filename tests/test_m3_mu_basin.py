"""Contract tests for src/sef_hfo_mu_basin.py — R0–R4 classifier + R_event (static-μ pilot).

Spec: docs/archive/topic4/sef_hfo/m3_static_mu_pilot_2026-06-24.md §4-§5. The classifier maps
one (μ,K,seed) event's scalar metrics to a regime label; the R4a-vs-R4b split (sustained with a
propagation front vs uniform tonic runaway) is load-bearing — only R4a is a bridge candidate.
Pure functions on synthetic metric dicts; no SNN.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import sef_hfo_mu_basin as mb  # noqa: E402

CAPS = mb.DEFAULT_CAPS


def _m(**kw):
    base = dict(event_detected=True, returned=True, runaway=False, r95_ea=3.0, far_ea=0.05,
                active_peak=0.05, sustained_front_score=0.0)
    base.update(kw)
    return base


def test_R0_no_onset():
    assert mb.classify_event(_m(event_detected=False), CAPS) == "R0"


def test_R1_failed_ignition_negligible_spread():
    # onset but active_peak below floor -> failed ignition
    assert mb.classify_event(_m(active_peak=1e-4), CAPS) == "R1"


def test_R2_finite_local_returned():
    assert mb.classify_event(_m(returned=True, runaway=False, r95_ea=3.0, far_ea=0.05),
                             CAPS) == "R2"


def test_R3_large_returned_not_local():
    # returned but big extent / high far -> near-critical large returned
    assert mb.classify_event(_m(returned=True, runaway=False, r95_ea=9.0, far_ea=0.4),
                             CAPS) == "R3"


def test_R4a_sustained_with_front():
    # not returned (sustained) AND retains a propagation front -> W-aligned sustained
    assert mb.classify_event(_m(returned=False, runaway=True, sustained_front_score=0.8),
                             CAPS) == "R4a"


def test_R4b_tonic_runaway_uniform():
    # not returned AND spatially uniform/saturated (no front) -> tonic runaway
    assert mb.classify_event(_m(returned=False, runaway=True, sustained_front_score=0.1,
                                far_ea=0.9), CAPS) == "R4b"


def test_R4_split_is_only_difference_front_score():
    a = mb.classify_event(_m(returned=False, runaway=True, sustained_front_score=0.9), CAPS)
    b = mb.classify_event(_m(returned=False, runaway=True, sustained_front_score=0.05), CAPS)
    assert (a, b) == ("R4a", "R4b")    # ONLY the front score flips R4a<->R4b


def test_R_event_recruitment_gain():
    # front bins {1,2,3}; next-gen active {1,2,3,4,5,6} -> 3 new / 3 front = 1.0
    assert mb.r_event(front_bins={1, 2, 3}, next_active={1, 2, 3, 4, 5, 6}) == pytest.approx(1.0)
    assert mb.r_event(front_bins={1, 2}, next_active={1, 2}) == pytest.approx(0.0)   # no growth
    assert np.isnan(mb.r_event(front_bins=set(), next_active={1}))                   # empty front


# --- μ coupling: V_th_eff = vth_core - ΔVth·μ·h --------------------------------- #
VTH0, CORE_MEAN = 18.0, 17.6


def _vth_core(n_core=4, n_out=6):
    # core neurons at 17.6 (depressed), surround at 18.0
    return np.concatenate([np.full(n_core, 17.6), np.full(n_out, 18.0)])


def test_apply_mu_zero_is_bit_parity():
    vc = _vth_core()
    rng = np.random.default_rng(0)
    for mode in ("core_susceptibility", "uniform", "shuffled"):
        out = mb.apply_mu(vc, VTH0, CORE_MEAN, mu=0.0, dvth_at_mu1=1.333, h_mode=mode, rng=rng)
        assert np.array_equal(out, vc)          # EXACT identity at μ=0 (any h mode)


def test_apply_mu_core_susceptibility_lowers_only_core():
    vc = _vth_core()
    out = mb.apply_mu(vc, VTH0, CORE_MEAN, mu=0.5, dvth_at_mu1=1.2,
                      h_mode="core_susceptibility", rng=np.random.default_rng(0))
    # core (h≈1): lowered by ~1.2*0.5*1 = 0.6 -> 17.0; surround (h=0): unchanged 18.0
    assert np.allclose(out[:4], 17.6 - 0.6)
    assert np.allclose(out[4:], 18.0)


def test_apply_mu_uniform_lowers_everyone():
    vc = _vth_core()
    out = mb.apply_mu(vc, VTH0, CORE_MEAN, mu=0.5, dvth_at_mu1=1.2,
                      h_mode="uniform", rng=np.random.default_rng(0))
    assert np.allclose(out, vc - 1.2 * 0.5)     # global μ heating


def test_apply_mu_shuffled_preserves_value_set_not_location():
    vc = _vth_core()
    kw = dict(mu=0.5, dvth_at_mu1=1.2, rng=np.random.default_rng(1))
    cs = mb.apply_mu(vc, VTH0, CORE_MEAN, h_mode="core_susceptibility",
                     **dict(kw, rng=np.random.default_rng(1)))
    sh = mb.apply_mu(vc, VTH0, CORE_MEAN, h_mode="shuffled",
                     **dict(kw, rng=np.random.default_rng(1)))
    # same multiset of depressions, different spatial placement
    assert np.allclose(sorted(vc - cs), sorted(vc - sh))
    assert not np.array_equal(cs, sh)


# --- spontaneous-event detection on a long no-kick activity trace --------------- #
def test_detect_events_separates_two_bursts():
    tr = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0], dtype=float)
    ev = mb.detect_events(tr, thresh=0.5, min_gap_bins=1)
    assert ev == [(2, 3), (6, 6)]            # gap of 2 bins -> NOT merged


def test_detect_events_merges_close_bursts():
    tr = np.array([0, 0, 1, 1, 0, 0, 1, 0, 0], dtype=float)
    ev = mb.detect_events(tr, thresh=0.5, min_gap_bins=3)
    assert ev == [(2, 6)]                     # gap 2 < 3 -> merged


def test_detect_events_none_below_threshold():
    assert mb.detect_events(np.array([0.1, 0.2, 0.1]), thresh=0.5) == []


def test_event_props_returned_vs_sustained():
    tr = np.array([0, 0, 1, 1, 0, 0, 1, 1, 1], dtype=float)
    # event ending before the record end -> returned
    r = mb.event_props(tr, (2, 3), bin_ms=2.0, n_record_bins=9)
    assert r["duration_ms"] == 4.0 and r["returned"] is True and r["sustained"] is False
    # event extending to the last bin -> sustained (still active at record end)
    s = mb.event_props(tr, (6, 8), bin_ms=2.0, n_record_bins=9)
    assert s["returned"] is False and s["sustained"] is True


def test_aggregate_spontaneous_rate_and_fractions():
    # 2 events over a 1000 ms record -> 2 events/s; one R2 one R4a
    classes = ["R2", "R4a"]
    agg = mb.aggregate_spontaneous(n_events=2, record_ms=1000.0, classes=classes)
    assert agg["event_rate_hz"] == pytest.approx(2.0)
    assert agg["frac"]["R2"] == pytest.approx(0.5)
    assert agg["frac"]["R4a"] == pytest.approx(0.5)
    assert agg["frac"].get("R3", 0.0) == 0.0
