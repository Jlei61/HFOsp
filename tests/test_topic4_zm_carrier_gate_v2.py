"""carrier_gate_v2: faithful implementation of the pre-registered onset/baseline (spec §2) + the B2
temporal-overlap and A7 active-area clauses that v1 (src/topic4_zm_carrier_verdict.analyze_macroepisode)
silently deviated from. v1 is kept for history; v2 is what the recompute uses.

The v1 bug (caught in review): onset = start of longest FLOOR episode (not first sustained ON crossing),
baseline = fixed first 300 ms (not [0,onset)), MIN_ONSET_MS unused. For a burst train (bursts < 100 ms,
gaps > 250 ms) v1 reports a late arbitrary onset (sg: 8720 ms) whose [0,onset) window is burst-polluted.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.topic4_zm_carrier_gate_v2 import (  # noqa: E402
    first_sustained_crossing, analyze_macroepisode_v2, b2_highfreq_overlaps_window)
from src.topic4_zm_carrier_verdict import MIN_MACRO_MS, ON_FRAC  # noqa: E402

DT = 5.0


def test_first_sustained_crossing_needs_min_duration():
    # a 40 ms spike (8 bins) does NOT satisfy a 100 ms (20-bin) sustain requirement
    e = np.zeros(400); e[100:108] = 10.0
    assert first_sustained_crossing(e >= 3.0, min_bins=20) is None
    e[200:230] = 10.0                                   # a 150 ms plateau does
    assert first_sustained_crossing(e >= 3.0, min_bins=20) == 200


def test_brief_bursts_with_long_gaps_give_no_onset():
    # 40 ms bursts every 400 ms -> never 100 ms sustained above ON -> onset None
    e = np.full(2000, 0.1)
    for i in range(60, 2000, 80):
        e[i:i + 8] = 8.0
    m = analyze_macroepisode_v2(e, DT)
    assert m["onset_ms"] is None and not m["sustained"]


def test_sustained_high_gives_onset_and_macroepisode_with_preonset_baseline():
    e = np.concatenate([np.full(60, 0.1), np.full(700, 6.0)])   # 300 ms baseline + 3500 ms sustained
    m = analyze_macroepisode_v2(e, DT)
    assert m["onset_ms"] is not None and m["onset_ms"] < 400.0   # onset is EARLY (first sustained crossing)
    assert m["duration_ms"] >= MIN_MACRO_MS and m["occupancy"] > 0.95 and m["sustained"]
    assert abs(m["baseline"] - 0.1) < 0.05                      # baseline from [0,onset), not polluted


def test_v2_onset_is_earlier_than_v1_longest_episode_for_a_dense_train():
    # dense-ish train that eventually has a long merged stretch late: v1 would pick the late longest
    # episode; v2 anchors onset at the FIRST sustained crossing (early).
    e = np.full(3000, 0.2)
    e[100:140] = 8.0                                    # an early 200 ms sustained crossing
    for i in range(1500, 3000, 30):
        e[i:i + 20] = 8.0                               # a late dense stretch
    m = analyze_macroepisode_v2(e, DT)
    assert m["onset_ms"] is not None and m["onset_ms"] < 1000.0


def test_b2_requires_temporal_overlap_with_lowgamma_window():
    n = 200
    hf = np.zeros(n)
    win = (60, 100)                                     # the B1 low-gamma macroepisode frames
    hf[20:30] = 10.0                                    # high-freq peak OUTSIDE the window -> no overlap
    assert not b2_highfreq_overlaps_window(hf, win, enh_db=6.0)
    hf[70:80] = 10.0                                    # now inside the window -> overlap
    assert b2_highfreq_overlaps_window(hf, win, enh_db=6.0)
