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
    first_sustained_crossing, analyze_macroepisode_v2, b2_highfreq_overlaps_window,
    _axial_recruitment, compute_source_gate_v2, observed_sep_count_v2, FLASH_WINDOW_MS)
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


# ---------------------------------------------------------------- A8 axial gradient vs flash
def test_axial_recruitment_gradient_vs_flash():
    n_pos, n_t = 10, 300
    kt = np.arange(n_t) * DT
    grad = np.zeros((n_pos, n_t))                       # each axial bin ignites LATER -> spatial gradient
    for a in range(n_pos):
        grad[a, 40 + a * 12:] = 1.0
    r = _axial_recruitment(grad, kt, onset_ms=200.0, dt_ms=DT, window_ms=1000.0)
    assert r["spread_ms"] > FLASH_WINDOW_MS and not r["whole_field_flash"]
    flash = np.zeros((n_pos, n_t)); flash[:, 40:] = 1.0  # all bins ignite simultaneously -> flash
    assert _axial_recruitment(flash, kt, onset_ms=200.0, dt_ms=DT, window_ms=1000.0)["whole_field_flash"]


def test_axial_recruitment_uses_kymograph_time_axis_not_rate_bin():
    # kymograph columns are 10ms apart; the rate bin dt_ms=5 is DIFFERENT. A 9-column ignition spread is
    # 90ms (>FLASH_WINDOW 50) under the correct col_dt, but would be 45ms (<50, falsely "flash") if the
    # buggy rate-bin dt were used. Correct code must NOT call it a flash.
    n_pos, n_t = 10, 200
    kt = np.arange(n_t) * 10.0                          # 10 ms columns
    grad = np.zeros((n_pos, n_t))
    for a in range(n_pos):
        grad[a, 20 + a:] = 1.0                          # 9-column spread
    r = _axial_recruitment(grad, kt, onset_ms=200.0, dt_ms=5.0, window_ms=1000.0)
    assert abs(r["spread_ms"] - 90.0) < 1e-6 and not r["whole_field_flash"]


# ---------------------------------------------------------------- A7 active-area + tail + saturation
def test_source_gate_a7_active_area_and_tail_and_saturation():
    n = 2000
    core = np.full(n, 1.0); core[60:75] = 20.0; core[100:] = 50.0      # brief pre-onset event + sustained
    active = np.full(n, 0.02); active[60:75] = 0.05; active[100:] = 0.30
    kymo = np.zeros((10, n)); kymo[:, 100:] = 1.0
    kt = np.arange(n) * DT
    src = compute_source_gate_v2(core, np.full(n, 5.0), active, kymo, kt, DT, None)
    assert src["macro"]["sustained"]
    assert src["src_sep_count"] == 3                    # duration + peak + ACTIVE-AREA all separated
    assert src["tail_escalating"] is False and src["saturated_plateau"] is False
    # escalating all-E tail + whole-sheet active area -> both safety flags fire (were hardcoded False before)
    hot = np.full(n, 5.0); hot[100:] = 200.0
    sat = np.full(n, 0.02); sat[100:] = 0.6
    src2 = compute_source_gate_v2(core, hot, sat, kymo, kt, DT, None)
    assert src2["tail_escalating"] is True and src2["saturated_plateau"] is True


# ---------------------------------------------------------------- B6 real event-to-event 4-dim
def test_observed_sep_count_v2_reference_window_and_cross_contact_extent():
    n = 200
    lg = np.zeros((n, 3))
    lg[10:14, 0] = 10.0                                 # a brief 1-contact returning event in [0, onset)
    lg[40:, :3] = 12.0                                  # long macro on ALL 3 contacts (spatial extent 3 vs event 1)
    obs = dict(best_contact_idx=0, lg_db=lg, frame_dt_ms=25.0, pre_frames=20,   # onset=1000ms -> onset_f=40
               best_macro=dict(onset_ms=1000.0, duration_ms=3000.0, occupancy=0.95, peak=12.0))
    assert observed_sep_count_v2(obs) == 4              # duration + energy + duty + spatial-extent all separate
    weak = dict(obs, best_macro=dict(onset_ms=1000.0, duration_ms=100.0, occupancy=0.30, peak=12.0))
    assert observed_sep_count_v2(weak) < 3              # short low macro -> duration/energy/duty fail


def test_observed_sep_count_v2_fail_closed_without_reference_events():
    n = 200
    lg = np.zeros((n, 3)); lg[40:, :3] = 12.0           # NO pre-onset returning event in [0, onset=40)
    obs = dict(best_contact_idx=0, lg_db=lg, frame_dt_ms=25.0, pre_frames=20,
               best_macro=dict(onset_ms=1000.0, duration_ms=3000.0, occupancy=0.95, peak=12.0))
    assert observed_sep_count_v2(obs) == 0             # no reference -> not evaluable -> fail-closed


# ---------------------------------------------------------------- onset re-validation invariant
def test_onset_survives_the_final_baseline_threshold():
    e = np.concatenate([np.full(60, 0.1), np.full(700, 6.0)])
    m = analyze_macroepisode_v2(e, DT)
    oi = int(round(m["onset_ms"] / DT))
    on2 = m["baseline"] + ON_FRAC * (m["peak"] - m["baseline"])
    assert (e[oi:oi + 20] >= on2).all()                 # the reported onset holds ON for >=100ms under FINAL baseline


def test_source_gate_a7_fail_closed_without_reference_events():
    n = 2000
    core = np.full(n, 1.0); core[100:] = 50.0           # sustained, but NO pre-onset event in [0, onset)
    kymo = np.zeros((10, n)); kymo[:, 100:] = 1.0
    kt = np.arange(n) * DT
    src = compute_source_gate_v2(core, np.full(n, 5.0), np.full(n, 0.30), kymo, kt, DT, None)
    assert src["macro"]["sustained"] and src["src_sep_count"] == 0   # A7 fail-closed: no reference events


def test_tail_escalating_v2_flags_ramp_below_absolute_threshold():
    from src.topic4_zm_carrier_gate_v2 import _tail_escalating_v2
    assert _tail_escalating_v2(np.linspace(10.0, 100.0, 2000)) is True   # 10->100 Hz ramp, never hits 150
    assert _tail_escalating_v2(np.full(2000, 50.0)) is False             # stationary -> not escalating
