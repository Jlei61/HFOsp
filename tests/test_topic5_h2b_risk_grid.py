"""B0.2 contract tests: fixed 5-min risk grid, seizure boundaries, censoring.

Clauses come from the v0.2 common contract §5.2/§7 and the H2b spec §1/§5:
grid anchors live on real physical time, never cross a seizure or a coverage
gap, and a row that simply ran out of monitoring is *censored*, never scored as
"no seizure for six hours".
"""

from __future__ import annotations

from src.topic5_h2b_transfer.risk_grid import (
    HORIZON_EDGES_SECONDS,
    build_risk_rows,
    merge_spans,
)

HOUR = 3600.0
MIN = 60.0


def _sz(onset, offset, sid="s"):
    return {"seizure_id": sid, "onset_epoch": onset, "offset_epoch": offset}


def _rows(state_spans, monitoring_spans, seizures, **kw):
    return build_risk_rows(
        subject="epilepsiae_958",
        state_spans=state_spans,
        monitoring_spans=monitoring_spans,
        seizures=seizures,
        **kw,
    )


# --- span merging -------------------------------------------------------------


def test_touching_spans_merge_and_a_real_gap_does_not():
    assert merge_spans([(0.0, 10.0), (10.0, 20.0)], tolerance_seconds=2.0) == ((0.0, 20.0),)
    assert merge_spans([(0.0, 10.0), (30.0, 40.0)], tolerance_seconds=2.0) == (
        (0.0, 10.0),
        (30.0, 40.0),
    )


# --- D1: anchors sit on a reproducible absolute grid ---------------------------


def test_anchors_are_multiples_of_the_grid_step_in_absolute_epoch():
    rows = _rows([(1000.0, 1000.0 + 3 * 300.0)], [(0.0, 1e9)], [])
    assert [r.anchor_epoch for r in rows] == [1200.0, 1500.0, 1800.0]


# --- D2: anchors require state coverage ---------------------------------------


def test_no_anchor_is_emitted_outside_a_state_coverage_span():
    rows = _rows([(0.0, 600.0), (10_000.0, 10_600.0)], [(0.0, 1e9)], [])
    assert all(r.anchor_epoch <= 600.0 or r.anchor_epoch >= 10_000.0 for r in rows)
    assert not any(600.0 < r.anchor_epoch < 10_000.0 for r in rows)


# --- D3: the trajectory terminates at onset -----------------------------------


def test_anchor_inside_the_ictal_interval_is_dropped():
    rows = _rows([(0.0, 4 * HOUR)], [(0.0, 1e9)], [_sz(3600.0, 3900.0)])
    assert not any(3600.0 <= r.anchor_epoch <= 3900.0 for r in rows)


# --- D4: postictal exclusion ---------------------------------------------------


def test_sixty_minutes_after_offset_is_excluded_by_default():
    rows = _rows([(0.0, 6 * HOUR)], [(0.0, 1e9)], [_sz(HOUR, HOUR + 300.0)])
    resume = HOUR + 300.0 + 3600.0
    assert not any(HOUR <= r.anchor_epoch < resume for r in rows)
    assert any(r.anchor_epoch >= resume for r in rows)


def test_postictal_exclusion_length_is_a_parameter_for_sensitivity():
    rows = _rows(
        [(0.0, 6 * HOUR)],
        [(0.0, 1e9)],
        [_sz(HOUR, HOUR + 300.0)],
        postictal_exclusion_seconds=1800.0,
    )
    resume = HOUR + 300.0 + 1800.0
    assert any(abs(r.anchor_epoch - resume) < 300.0 for r in rows)


# --- D5: horizon bins ----------------------------------------------------------


def test_horizon_edges_match_the_preregistered_bins():
    assert HORIZON_EDGES_SECONDS == (
        5 * MIN,
        15 * MIN,
        30 * MIN,
        60 * MIN,
        2 * HOUR,
        6 * HOUR,
    )


def test_time_to_next_seizure_selects_the_next_onset_strictly_after_the_anchor():
    rows = _rows([(0.0, 2 * HOUR)], [(0.0, 1e9)], [_sz(4200.0, 4300.0, "a")])
    row = next(r for r in rows if r.anchor_epoch == 3900.0)
    assert row.time_to_next_seizure_sec == 300.0
    assert row.next_seizure_id == "a"
    assert row.outcome_bin == 0  # 0-5 min


def test_a_row_two_hours_out_lands_in_the_two_to_six_hour_bin():
    rows = _rows([(0.0, 8 * HOUR)], [(0.0, 1e9)], [_sz(5 * HOUR, 5 * HOUR + 60.0, "a")])
    row = next(r for r in rows if r.anchor_epoch == 2 * HOUR)
    assert row.outcome_bin == 5
    assert not row.censored


# --- D6: censoring vs. genuinely-far ------------------------------------------


def test_monitoring_gap_before_the_next_seizure_censors_the_row():
    """Coverage stops one hour after the anchor: we do not know what happened."""
    rows = _rows(
        [(0.0, 2 * HOUR)],
        [(0.0, 3 * HOUR)],
        [_sz(10 * HOUR, 10 * HOUR + 60.0, "a")],
    )
    row = next(r for r in rows if r.anchor_epoch == 2 * HOUR)
    assert row.censored
    assert row.outcome_bin is None
    assert row.last_observed_bin == 3  # 60 min fully observed, 2 h is not


def test_full_six_hour_coverage_without_a_seizure_is_beyond_horizon_not_censored():
    rows = _rows([(0.0, HOUR)], [(0.0, 24 * HOUR)], [])
    row = next(r for r in rows if r.anchor_epoch == 600.0)
    assert not row.censored
    assert row.outcome_bin is None
    assert row.beyond_horizon
    assert row.last_observed_bin == len(HORIZON_EDGES_SECONDS) - 1


def test_an_unmonitored_hole_between_anchor_and_seizure_still_censors():
    """A documented seizure 3 h out does not rescue a coverage hole at 1 h."""
    rows = _rows(
        [(0.0, HOUR)],
        [(0.0, 2 * HOUR), (2.5 * HOUR, 24 * HOUR)],
        [_sz(3 * HOUR, 3 * HOUR + 60.0, "a")],
    )
    row = next(r for r in rows if r.anchor_epoch == HOUR)
    assert row.censored
    assert row.last_observed_bin == 3  # 60 min fully observed, 2 h is not


# --- bookkeeping the spec asks for --------------------------------------------


def test_time_since_previous_seizure_is_reported_for_the_baseline():
    rows = _rows([(0.0, 8 * HOUR)], [(0.0, 1e9)], [_sz(HOUR, HOUR + 300.0, "a")])
    row = next(r for r in rows if r.anchor_epoch == 5 * HOUR)
    assert row.time_since_prev_seizure_sec == 5 * HOUR - (HOUR + 300.0)
    assert row.prev_seizure_id == "a"


def test_rows_before_any_seizure_have_no_previous_seizure():
    rows = _rows([(0.0, HOUR)], [(0.0, 1e9)], [_sz(10 * HOUR, 10 * HOUR + 60.0)])
    assert all(r.time_since_prev_seizure_sec is None for r in rows)


# --- B2 lead anchors: exact onset - lead, not snapped to the grid -------------


def test_lead_anchor_is_available_when_state_covers_it():
    from src.topic5_h2b_transfer.risk_grid import lead_anchor_status

    status = lead_anchor_status(
        anchor_epoch=10 * HOUR - 2 * HOUR,
        state_spans=((0.0, 12 * HOUR),),
        seizures=[_sz(10 * HOUR, 10 * HOUR + 60.0)],
    )
    assert status == "ok"


def test_lead_anchor_without_state_coverage_is_not_backfilled():
    """'6 h 仅在连续 coverage 存在时计入，不用缺失 anchor 补零'."""
    from src.topic5_h2b_transfer.risk_grid import lead_anchor_status

    status = lead_anchor_status(
        anchor_epoch=10 * HOUR - 6 * HOUR,
        state_spans=((8 * HOUR, 12 * HOUR),),
        seizures=[_sz(10 * HOUR, 10 * HOUR + 60.0)],
    )
    assert status == "no_state_coverage"


def test_lead_anchor_landing_in_an_earlier_seizure_is_rejected():
    from src.topic5_h2b_transfer.risk_grid import lead_anchor_status

    status = lead_anchor_status(
        anchor_epoch=10 * HOUR - 6 * HOUR,
        state_spans=((0.0, 12 * HOUR),),
        seizures=[_sz(4 * HOUR, 4 * HOUR + 120.0), _sz(10 * HOUR, 10 * HOUR + 60.0)],
    )
    assert status == "in_ictal"


def test_lead_anchor_inside_the_postictal_window_is_rejected():
    from src.topic5_h2b_transfer.risk_grid import lead_anchor_status

    status = lead_anchor_status(
        anchor_epoch=10 * HOUR - 2 * HOUR,
        state_spans=((0.0, 12 * HOUR),),
        seizures=[_sz(7.5 * HOUR, 7.6 * HOUR), _sz(10 * HOUR, 10 * HOUR + 60.0)],
    )
    assert status == "in_postictal"


# --- D9: seizure clusters are one predictable episode, not N samples ---------


def test_seizures_inside_the_postictal_window_of_the_previous_one_share_an_episode():
    """A cluster cannot supply N independent held-out targets.

    Anchors inside [onset, offset + exclusion) are already dropped, so only the
    *lead* seizure of a cluster can ever be predicted from a valid anchor.
    Counting the followers as separate held-out seizures would inflate the
    denominator ("event rows 不冒充样本量").
    """
    from src.topic5_h2b_transfer.risk_grid import group_seizure_episodes

    seizures = [
        _sz(0.0, 0.0, "a"),
        _sz(120.0, 120.0, "b"),
        _sz(240.0, 240.0, "c"),
        _sz(50 * HOUR, 50 * HOUR + 60.0, "d"),
    ]
    episodes = group_seizure_episodes(seizures, gap_seconds=3600.0)
    assert [[s["seizure_id"] for s in ep] for ep in episodes] == [["a", "b", "c"], ["d"]]


def test_lead_seizure_of_each_episode_is_the_earliest_onset():
    from src.topic5_h2b_transfer.risk_grid import group_seizure_episodes

    episodes = group_seizure_episodes(
        [_sz(240.0, 240.0, "late"), _sz(0.0, 0.0, "lead")], gap_seconds=3600.0
    )
    assert [s["seizure_id"] for s in episodes[0]] == ["lead", "late"]


def test_widely_separated_seizures_stay_separate_episodes():
    from src.topic5_h2b_transfer.risk_grid import group_seizure_episodes

    episodes = group_seizure_episodes(
        [_sz(0.0, 60.0, "a"), _sz(3 * HOUR, 3 * HOUR + 60.0, "b")], gap_seconds=3600.0
    )
    assert len(episodes) == 2
