from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from src.topic5_propagation_drift_diurnal import (
    as_phase_contrast_pairs,
    assign_block_phase,
    attach_phase,
    phase_exposure,
    timezone_for_dataset,
)
from src.topic5_propagation_drift_sensitivity import annotated_pairs


def _epoch(year, month, day, hour, tz):
    return datetime(year, month, day, hour, 0, 0, tzinfo=ZoneInfo(tz)).timestamp()


def test_timezone_contract_matches_the_repository_mounts():
    assert timezone_for_dataset("epilepsiae") == "Europe/Berlin"
    assert timezone_for_dataset("yuquan") == "Asia/Shanghai"


def test_timezone_lookup_fails_loudly_on_an_unknown_dataset():
    with pytest.raises(ValueError):
        timezone_for_dataset("mystery_cohort")


def test_assign_block_phase_uses_local_time_not_utc():
    # 07:00 Berlin in July is 05:00 UTC; classifying on UTC would call it night.
    tz = "Europe/Berlin"
    blocks = [
        {"t_mid": _epoch(2009, 7, 1, 7, tz)},
        {"t_mid": _epoch(2009, 7, 1, 9, tz)},
        {"t_mid": _epoch(2009, 7, 1, 19, tz)},
        {"t_mid": _epoch(2009, 7, 1, 21, tz)},
    ]
    assert assign_block_phase(blocks, tz) == ["night", "day", "day", "night"]


def test_assign_block_phase_boundaries_are_half_open_at_eight_and_twenty():
    tz = "Asia/Shanghai"
    blocks = [
        {"t_mid": _epoch(2020, 1, 3, 8, tz)},
        {"t_mid": _epoch(2020, 1, 3, 20, tz)},
    ]
    assert assign_block_phase(blocks, tz) == ["day", "night"]


def test_attach_phase_joins_by_block_index_not_by_position():
    pairs = [
        {"left_index": 2, "right_index": 0, "d_seconds": 10.0},
        {"left_index": 1, "right_index": 2, "d_seconds": 20.0},
    ]
    phases = ["day", "night", "day"]
    rows = attach_phase(pairs, phases)
    assert rows[0]["left_phase"] == "day" and rows[0]["right_phase"] == "day"
    assert rows[0]["same_phase"] is True
    assert rows[1]["left_phase"] == "night" and rows[1]["right_phase"] == "day"
    assert rows[1]["same_phase"] is False


def test_attach_phase_preserves_the_original_pair_fields():
    pairs = [{"left_index": 0, "right_index": 1, "similarity": 0.42, "d_events": 7.0}]
    rows = attach_phase(pairs, ["day", "day"])
    assert rows[0]["similarity"] == 0.42 and rows[0]["d_events"] == 7.0


def test_as_phase_contrast_pairs_swaps_the_grouping_key_without_touching_data():
    pairs = [
        {"same_phase": True, "same_source": False, "similarity": 0.7, "d_events": 5.0},
        {"same_phase": False, "same_source": True, "similarity": 0.3, "d_events": 6.0},
    ]
    rows = as_phase_contrast_pairs(pairs)
    assert [row["same_source"] for row in rows] == [True, False]
    assert [row["similarity"] for row in rows] == [0.7, 0.3]
    assert [row["d_events"] for row in rows] == [5.0, 6.0]
    # the original rows must not be mutated in place
    assert pairs[0]["same_source"] is False


def test_as_phase_contrast_pairs_feeds_the_frozen_matched_cell_contrast():
    from src.topic5_propagation_drift import matched_event_distance_contrast

    pairs = [
        {"same_phase": True, "d_events": 4.0, "d_seconds": 40.0, "similarity": 0.9},
        {"same_phase": True, "d_events": 6.0, "d_seconds": 60.0, "similarity": 0.8},
        {"same_phase": False, "d_events": 5.0, "d_seconds": 50.0, "similarity": 0.4},
        {"same_phase": False, "d_events": 7.0, "d_seconds": 70.0, "similarity": 0.5},
    ]
    cells = matched_event_distance_contrast(
        as_phase_contrast_pairs(pairs), bin_edges=[0.0, 20.0], min_pairs_per_cell=2
    )
    assert len(cells) == 1
    # "same_source" now means "same diurnal phase"
    assert cells[0]["median_same_source"] == pytest.approx(0.85)
    assert cells[0]["median_cross_source"] == pytest.approx(0.45)


def test_phase_exposure_reports_contamination_and_time_reach():
    pairs = [
        {"same_phase": True, "d_seconds": 100.0},
        {"same_phase": True, "d_seconds": 200.0},
        {"same_phase": False, "d_seconds": 50000.0},
        {"same_phase": False, "d_seconds": 60000.0},
    ]
    summary = phase_exposure(pairs)
    assert summary["n_pairs"] == 4
    assert summary["cross_phase_fraction"] == pytest.approx(0.5)
    assert summary["median_d_seconds"] == pytest.approx(25100.0)
    assert summary["max_d_seconds"] == pytest.approx(60000.0)


def test_phase_exposure_handles_a_patient_with_no_pairs():
    assert phase_exposure([]) == {"n_pairs": 0}


def _block(source, mid_index, t_start, t_end, ranks):
    ranks = np.asarray(ranks, dtype=float)
    return {
        "source_id": source,
        "event_mid_index": float(mid_index),
        "t_mid": 0.5 * (t_start + t_end),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "mean_rank": ranks,
        "support": np.ones_like(ranks),
    }


def test_annotated_pairs_carry_block_indices_so_phase_can_be_joined():
    blocks = [
        _block("recA", 0.0, 0.0, 10.0, [1.0, 2.0, 3.0, 4.0, 5.0]),
        _block("recA", 20.0, 20.0, 30.0, [2.0, 1.0, 3.0, 4.0, 5.0]),
        _block("recB", 40.0, 40.0, 50.0, [5.0, 4.0, 3.0, 2.0, 1.0]),
    ]
    pairs = annotated_pairs(blocks, max_pairs=99, seed=0, min_support=0.5, min_shared=3)
    assert {(row["left_index"], row["right_index"]) for row in pairs} == {
        (0, 1),
        (0, 2),
        (1, 2),
    }
    joined = attach_phase(pairs, ["day", "night", "night"])
    by_key = {(row["left_index"], row["right_index"]): row for row in joined}
    assert by_key[(0, 1)]["same_phase"] is False
    assert by_key[(1, 2)]["same_phase"] is True
