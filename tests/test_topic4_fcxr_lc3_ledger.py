"""Contracts for the FCXR-LC3 per-event ledger.

All synthetic; none of these touch a 40k substrate, so the bearing measurement is
verifiable in seconds instead of after an eight-hour trajectory.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from src.topic4_fcxr_lc3_ledger import (
    ACCUMULATION_BAR,
    REGION_KEYS,
    bracketing_snapshots,
    build_event_ledger,
    classify_entry,
    event_dose_af,
    event_dose_rate,
    regional_means,
    snapshot_table,
)


def _masks():
    m = {k: np.zeros(8, bool) for k in REGION_KEYS}
    m["core_A"][0:2] = True
    m["core_B"][2:4] = True
    m["axial"][4:6] = True
    m["off_axis"][6:8] = True
    return m


# --- Task 1: per-event dose -------------------------------------------------

def test_af_dose_subtracts_the_frozen_floor_and_clips_at_zero():
    af = np.array([0.0, 0.10, 0.20, 0.10, 0.0])          # 1 ms bins
    ev = dict(t_on=1.0, t_off=3.0)
    assert event_dose_af(af, 1.0, ev, 0.05) == pytest.approx(0.25)


def test_af_dose_is_zero_when_the_event_never_clears_the_floor():
    af = np.array([0.05, 0.04, 0.05])
    assert event_dose_af(af, 1.0, dict(t_on=0.0, t_off=2.0), 0.05) == 0.0


def test_rate_dose_integrates_at_the_full_step_not_the_decimated_one():
    # A 12 ms event is 241 samples at dt=0.05 ms; the 10 ms NPZ trace would see 2.
    rate = np.full(400, 30.0)
    rate[:100] = 2.0
    rate[341:] = 2.0
    got = event_dose_rate(rate, 0.05, dict(t_on=5.0, t_off=17.0), 2.0)
    assert got == pytest.approx(28.0 * 241 * 0.05)


def test_rate_dose_clips_below_baseline_to_zero():
    rate = np.full(100, 1.0)
    assert event_dose_rate(rate, 0.05, dict(t_on=0.0, t_off=1.0), 5.0) == 0.0


# --- Task 2: regional snapshot table ---------------------------------------

def test_regional_means_split_by_mask_and_keep_the_whole_array_too():
    got = regional_means(np.array([1., 1., 2., 2., 3., 3., 4., 4.]), _masks())
    assert got == pytest.approx(
        dict(core_A=1.0, core_B=2.0, axial=3.0, off_axis=4.0, all=2.5))


def test_regional_means_reject_a_mask_set_that_is_not_the_registered_regions():
    bad = _masks()
    bad.pop("axial")
    with pytest.raises(ValueError, match="must be exactly"):
        regional_means(np.zeros(8), bad)


def test_snapshot_table_is_sorted_by_time_and_converts_steps_to_ms():
    snaps = {
        "t500": dict(step=10000, z_E=np.zeros(8), h_E=np.ones(8),
                     x_E=np.full(8, 0.5), y_E=np.zeros(8)),
        "t250": dict(step=5000, z_E=np.ones(8), h_E=np.zeros(8),
                     x_E=np.ones(8), y_E=np.ones(8)),
    }
    table = snapshot_table(snaps, 0.05, _masks())
    assert [r["t_ms"] for r in table] == [250.0, 500.0]
    assert table[0]["label"] == "t250"
    assert table[0]["D"]["all"] == pytest.approx(0.0)     # D = 1 - z
    assert table[1]["D"]["all"] == pytest.approx(1.0)


# --- Task 3: bracketing and entry class ------------------------------------

def test_bracketing_picks_last_before_onset_and_first_after_offset():
    table = [dict(t_ms=t) for t in (0.0, 250.0, 500.0, 750.0, 1000.0)]
    pre, post = bracketing_snapshots(table, 510.0, 522.0)
    assert pre["t_ms"] == 500.0 and post["t_ms"] == 750.0


def test_bracketing_returns_none_when_no_snapshot_exists_on_a_side():
    table = [dict(t_ms=500.0)]
    pre, post = bracketing_snapshots(table, 100.0, 120.0)
    assert pre is None and post["t_ms"] == 500.0
    pre, post = bracketing_snapshots(table, 900.0, 920.0)
    assert pre["t_ms"] == 500.0 and post is None


@pytest.mark.parametrize("n,onset,expected", [
    (0, 1000.0, "ONE_SHOT"), (1, 1000.0, "ONE_SHOT"), (2, 1000.0, "AMBIGUOUS_2"),
    (3, 1000.0, "CUMULATIVE"), (9, 1000.0, "CUMULATIVE"),
    (0, None, "NO_ONSET"), (7, None, "NO_ONSET"),
])
def test_entry_class_never_collapses_the_accumulation_question(n, onset, expected):
    assert classify_entry(n, onset) == expected


def test_accumulation_bar_is_the_registered_three():
    assert ACCUMULATION_BAR == 3


# --- Task 4: the ledger -----------------------------------------------------

def _ledger_case():
    events = [dict(t_on=100.0 + 1000.0 * k, t_off=110.0 + 1000.0 * k,
                   dur_ms=10.0, peak_ext=0.05, returned=True) for k in range(4)]
    af = np.zeros(5000)
    rate = np.full(100000, 2.0)
    for k in range(4):
        af[100 + 1000 * k: 111 + 1000 * k] = 0.10
        rate[int((100 + 1000 * k) / 0.05): int((111 + 1000 * k) / 0.05)] = 30.0
    snaps = {}
    for i, t in enumerate(np.arange(0.0, 4500.0, 250.0)):
        snaps[f"t{int(t)}"] = dict(
            step=int(round(t / 0.05)), z_E=np.full(8, 1.0 - 0.01 * i),
            h_E=np.full(8, 0.02 * i), x_E=np.ones(8), y_E=np.zeros(8))
    return events, af, rate, snapshot_table(snaps, 0.05, _masks())


def _ledger(onset_ms, offset_ms=None):
    events, af, rate, table = _ledger_case()
    return build_event_ledger(
        events=events, af=af, af_bin_ms=1.0, floor_af=0.05, rate_hz=rate,
        dt_ms=0.05, r_base_hz=2.0, table=table, onset_ms=onset_ms,
        offset_ms=offset_ms, total_ms=5000.0)


def test_four_returning_events_before_onset_read_as_cumulative_entry():
    led = _ledger(4200.0)
    assert led["n_returning_before_onset"] == 4
    assert led["entry_class"] == "CUMULATIVE"
    assert led["first_non_returning_index"] is None
    assert [e["phase"] for e in led["events"]] == ["pre_onset"] * 4


def test_an_early_onset_after_one_event_reads_as_one_shot_not_accumulation():
    # The startup-transient case: the trajectory ignites before any load builds.
    led = _ledger(1500.0)
    assert led["n_returning_before_onset"] == 2
    assert led["entry_class"] == "AMBIGUOUS_2"
    led = _ledger(500.0)
    assert led["n_returning_before_onset"] == 1
    assert led["entry_class"] == "ONE_SHOT"


def test_cumulative_dose_is_monotone_and_both_scales_are_present():
    led = _ledger(4200.0)
    q_af = [e["Q_af"] for e in led["events"]]
    q_rate = [e["Q_rate"] for e in led["events"]]
    assert q_af == sorted(q_af) and q_rate == sorted(q_rate)
    assert all(e["dose_af"] > 0 and e["dose_rate"] > 0 for e in led["events"])
    assert led["Q_af_to_onset"] == pytest.approx(q_af[-1])
    assert led["Q_rate_to_onset"] == pytest.approx(q_rate[-1])


def test_no_onset_is_reported_as_such_not_as_one_shot():
    led = _ledger(None)
    assert led["entry_class"] == "NO_ONSET"
    assert led["Q_af_to_onset"] is None and led["Q_rate_to_onset"] is None
    assert all(e["phase"] == "pre_onset" for e in led["events"])


def test_events_are_phased_against_onset_and_offset():
    led = _ledger(1500.0, offset_ms=2500.0)
    assert [e["phase"] for e in led["events"]] == [
        "pre_onset", "pre_onset", "ictal", "post_offset"]
    assert led["post_offset"]["n_returning"] == 1


def test_per_event_slow_state_keeps_regions_whose_mean_would_invert_them():
    # Cores deplete X while the off-axis majority recovers; the mean rises.
    events = [dict(t_on=300.0, t_off=310.0, dur_ms=10.0, peak_ext=0.05, returned=True)]
    af = np.zeros(1000)
    af[300:311] = 0.10
    rate = np.full(20000, 2.0)
    rate[6000:6220] = 30.0

    def snap(step, core_x, off_x):
        x = np.empty(8)
        x[0:4] = core_x
        x[4:8] = off_x
        return dict(step=step, z_E=np.zeros(8), h_E=np.zeros(8), x_E=x, y_E=np.zeros(8))

    table = snapshot_table({"a": snap(5000, 0.90, 0.50), "b": snap(10000, 0.70, 0.95)},
                           0.05, _masks())
    led = build_event_ledger(
        events=events, af=af, af_bin_ms=1.0, floor_af=0.05, rate_hz=rate, dt_ms=0.05,
        r_base_hz=2.0, table=table, onset_ms=None, offset_ms=None, total_ms=1000.0)
    delta = led["events"][0]["delta"]["X"]
    assert delta["all"] > 0        # the whole-array mean says X recovered
    assert delta["core_A"] < 0     # both cores actually depleted
    assert delta["core_B"] < 0


def test_ledger_records_the_baseline_it_used_so_the_choice_is_auditable():
    led = _ledger(4200.0)
    cal = led["calibration"]
    assert cal["r_base_hz"] == 2.0
    assert cal["floor_af"] == 0.05
    assert cal["accumulation_bar"] == ACCUMULATION_BAR
    assert "quiet median" in cal["r_base_definition"]


def test_snapshot_lag_is_recorded_on_both_sides_of_every_event():
    led = _ledger(4200.0)
    for ev in led["events"]:
        assert ev["pre"]["lag_ms"] >= 0.0 and ev["pre"]["lag_ms"] <= 250.0
        assert ev["post"]["lag_ms"] >= 0.0 and ev["post"]["lag_ms"] <= 250.0


# --- Task 5: the ledger must reach disk ------------------------------------

def test_recon_runner_persists_the_ledger_and_the_full_snapshot_table():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "scripts", "run_topic4_fcxr_lc3_recon.py")) as f:
        body = ast.dump(ast.parse(f.read()))
    # Everything the ledger needs is in memory when a row ends; the failure mode
    # this pins is computing it and never writing it out.
    assert "build_event_ledger" in body
    assert "event_ledger" in body
    assert "snapshot_t_ms" in body
