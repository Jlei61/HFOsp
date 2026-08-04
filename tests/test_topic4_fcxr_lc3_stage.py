"""Contracts for the five registered lifecycle stages."""
import pytest

from src.topic4_fcxr_lc3_stage import (
    ACCUMULATION_BAR,
    STAGE_ORDER,
    lifecycle_stage,
    reference_band,
    returned_to_reference,
    stage_index,
)

BAND = dict(event_rate_lo=0.086, event_rate_hi=3.15,
            dur_lo_ms=8.0, dur_hi_ms=22.0, part_lo=0.045, part_hi=0.080)


# --- the return test ---------------------------------------------------------

def test_silence_fails_on_count_before_any_rate_is_consulted():
    """The measured clamped arms: 0 events over 18 s at a plausible-looking rate."""
    out = returned_to_reference(n_returning_after_offset=0, event_rate_hz=0.093,
                                band=BAND)
    assert not out["returned"] and out["reason"] == "no returning events after offset"
    assert "checks" not in out


def test_events_inside_the_reference_distribution_return():
    out = returned_to_reference(n_returning_after_offset=12, event_rate_hz=1.2,
                                band=BAND, durations_ms=[9.0, 14.0, 20.0],
                                participation=[0.05, 0.06, 0.07])
    assert out["returned"] and all(out["checks"].values())


def test_a_smouldering_train_fails_on_rate_not_on_count():
    """The measured 32-40 Hz arms have plenty of events at the wrong rate."""
    out = returned_to_reference(n_returning_after_offset=200, event_rate_hz=36.0,
                                band=BAND)
    assert not out["returned"] and out["checks"]["event_rate"] is False
    assert "event_rate" in out["reason"]


def test_events_of_the_wrong_shape_fail_even_at_the_right_rate():
    out = returned_to_reference(n_returning_after_offset=8, event_rate_hz=1.0,
                                band=BAND, durations_ms=[9.0, 40.0])
    assert not out["returned"] and out["checks"]["duration"] is False


def test_shape_checks_are_skipped_when_not_supplied_rather_than_assumed_passing():
    out = returned_to_reference(n_returning_after_offset=8, event_rate_hz=1.0,
                                band=BAND)
    assert out["returned"] and set(out["checks"]) == {"event_rate"}


def test_a_band_missing_its_rate_edges_is_refused():
    with pytest.raises(ValueError, match="event_rate_hi"):
        returned_to_reference(n_returning_after_offset=1, event_rate_hz=1.0,
                              band=dict(event_rate_lo=0.0))


# --- the five stages ---------------------------------------------------------

def test_no_onset_is_a_train_without_entry():
    out = lifecycle_stage(onset_ms=None, offset_ms=None, n_returning_before_onset=40)
    assert out["stage"] == "IED_TRAIN_NO_ONSET"


def test_entry_without_a_preceding_train_is_one_shot():
    out = lifecycle_stage(onset_ms=500.0, offset_ms=None, n_returning_before_onset=1)
    assert out["stage"] == "ONE_SHOT" and str(ACCUMULATION_BAR) in out["reason"]


def test_entry_at_the_accumulation_bar_is_not_one_shot():
    out = lifecycle_stage(onset_ms=4000.0, offset_ms=None,
                          n_returning_before_onset=ACCUMULATION_BAR)
    assert out["stage"] == "ONSET_NO_OFFSET"


def test_the_measured_trajectories_are_onset_without_offset():
    """12, 15 and 7 returning events before onset; no termination in 45 s."""
    for n in (12, 15, 7):
        out = lifecycle_stage(onset_ms=5000.0, offset_ms=None,
                              n_returning_before_onset=n)
        assert out["stage"] == "ONSET_NO_OFFSET"


def test_an_unmeasured_return_is_recorded_as_unmeasured_not_failed():
    out = lifecycle_stage(onset_ms=4000.0, offset_ms=20000.0,
                          n_returning_before_onset=12, return_check=None)
    assert out["stage"] == "OFFSET_NO_RECOVERY"
    assert out["return_measured"] is False
    assert "never measured" in out["reason"]


def test_termination_without_a_return_stops_at_offset():
    check = returned_to_reference(n_returning_after_offset=0, event_rate_hz=0.0,
                                  band=BAND)
    out = lifecycle_stage(onset_ms=4000.0, offset_ms=20000.0,
                          n_returning_before_onset=12, return_check=check)
    assert out["stage"] == "OFFSET_NO_RECOVERY" and out["return_measured"] is True


def test_the_whole_loop_closes_only_with_all_three():
    check = returned_to_reference(n_returning_after_offset=10, event_rate_hz=1.0,
                                  band=BAND, durations_ms=[10.0],
                                  participation=[0.06])
    out = lifecycle_stage(onset_ms=4000.0, offset_ms=20000.0,
                          n_returning_before_onset=12, return_check=check)
    assert out["stage"] == "FULL_LIFECYCLE"


def test_stage_order_runs_around_the_loop():
    assert stage_index("IED_TRAIN_NO_ONSET") < stage_index("ONSET_NO_OFFSET")
    assert stage_index("ONSET_NO_OFFSET") < stage_index("OFFSET_NO_RECOVERY")
    assert stage_index("OFFSET_NO_RECOVERY") < stage_index("FULL_LIFECYCLE")
    assert len(STAGE_ORDER) == 5


def test_an_unknown_stage_is_refused_rather_than_ranked():
    with pytest.raises(ValueError, match="unknown stage"):
        stage_index("MOSTLY_THERE")


# --- the reference band carries event shape, not just rate -------------------

def test_the_band_on_disk_alone_would_skip_the_shape_check():
    """Rate edges live in `band`; event shape lives one level up."""
    baseline = dict(band=dict(event_rate_lo=0.086, event_rate_hi=3.15),
                    event_durations_ms=[8.0, 10.0, 22.0],
                    event_participation=[0.0445, 0.06, 0.0795])
    assert "dur_lo_ms" not in baseline["band"]
    band = reference_band(baseline)
    assert (band["dur_lo_ms"], band["dur_hi_ms"]) == (8.0, 22.0)
    assert (band["part_lo"], band["part_hi"]) == (0.0445, 0.0795)
    assert band["n_reference_events"] == 3


def test_a_baseline_without_reference_events_yields_rate_only():
    band = reference_band(dict(band=dict(event_rate_lo=0.0, event_rate_hi=3.0)))
    assert "dur_lo_ms" not in band and band["n_reference_events"] == 0


def test_the_folded_band_makes_a_wrong_shaped_event_fail():
    baseline = dict(band=dict(event_rate_lo=0.086, event_rate_hi=3.15),
                    event_durations_ms=[8.0, 22.0], event_participation=[0.0445, 0.0795])
    out = returned_to_reference(n_returning_after_offset=4, event_rate_hz=1.0,
                                band=reference_band(baseline),
                                durations_ms=[45.0], participation=[0.06])
    assert not out["returned"] and out["checks"]["duration"] is False
