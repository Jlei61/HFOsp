import numpy as np
import pytest

from src.topic5_event_innovation_followup import (
    detectability_floor,
    source_spans,
    source_gap_census,
)


def test_source_spans_groups_by_source_and_keeps_record_name():
    times = np.array([0.0, 10.0, 20.0, 1000.0, 1010.0])
    source = np.array([0, 0, 0, 1, 1])
    record = np.array(["recA", "recA", "recA", "recB", "recB"])
    spans = source_spans(times, source, record)
    assert [row["source_index"] for row in spans] == ["0", "1"]
    assert [row["n_events"] for row in spans] == [3, 2]
    assert spans[0]["t_start"] == 0.0 and spans[0]["t_end"] == 20.0
    assert spans[0]["duration_seconds"] == 20.0
    assert spans[0]["record_name"] == "recA"
    assert spans[1]["record_name"] == "recB"


def test_source_spans_are_ordered_by_start_time_not_by_label():
    times = np.array([500.0, 510.0, 0.0, 10.0])
    source = np.array([7, 7, 3, 3])
    record = np.array(["late", "late", "early", "early"])
    spans = source_spans(times, source, record)
    assert [row["source_index"] for row in spans] == ["3", "7"]
    assert spans[0]["t_start"] == 0.0


def test_source_spans_reports_mixed_record_names_instead_of_silently_picking_one():
    times = np.array([0.0, 10.0])
    source = np.array([0, 0])
    record = np.array(["recA", "recB"])
    spans = source_spans(times, source, record)
    assert spans[0]["record_name"] == "MIXED(recA|recB)"


def test_source_gap_census_counts_only_gaps_above_threshold():
    spans = [
        {"source_index": "0", "n_events": 100, "t_start": 0.0, "t_end": 100.0},
        {"source_index": "1", "n_events": 100, "t_start": 200.0, "t_end": 300.0},
        {"source_index": "2", "n_events": 100, "t_start": 100000.0, "t_end": 100100.0},
    ]
    census = source_gap_census(spans, min_gap_seconds=3600.0, min_events_per_side=50)
    assert census["n_sources"] == 3
    assert census["n_consecutive_gaps"] == 2
    assert census["n_qualifying_consecutive_gaps"] == 1
    assert census["max_gap_seconds"] == pytest.approx(99700.0)
    assert census["total_span_seconds"] == pytest.approx(100100.0)


def test_source_gap_census_requires_both_sides_to_have_enough_events():
    spans = [
        {"source_index": "0", "n_events": 100, "t_start": 0.0, "t_end": 100.0},
        {"source_index": "1", "n_events": 3, "t_start": 100000.0, "t_end": 100100.0},
    ]
    census = source_gap_census(spans, min_gap_seconds=3600.0, min_events_per_side=50)
    assert census["n_consecutive_gaps"] == 1
    assert census["n_qualifying_consecutive_gaps"] == 0
    assert census["cross_gap_eligible"] is False


def test_source_gap_census_single_source_has_no_gap():
    spans = [{"source_index": "0", "n_events": 500, "t_start": 0.0, "t_end": 9000.0}]
    census = source_gap_census(spans, min_gap_seconds=3600.0, min_events_per_side=50)
    assert census["n_consecutive_gaps"] == 0
    assert census["n_qualifying_consecutive_gaps"] == 0
    assert census["max_gap_seconds"] is None
    assert census["cross_gap_eligible"] is False


def test_detectability_floor_has_near_nominal_power_at_zero_shift():
    rng_effects = np.random.default_rng(0).normal(scale=1.0, size=17)
    result = detectability_floor(
        rng_effects, deltas=[0.0], n_draws=2000, seed=11, alpha=0.05
    )
    # Level-2-style rule is one-sided in the median, so the false-positive rate
    # sits near alpha/2, not alpha.
    assert result["curve"][0]["power"] < 0.10


def test_detectability_floor_reaches_full_power_for_a_large_shift():
    effects = np.random.default_rng(1).normal(scale=0.01, size=17)
    result = detectability_floor(
        effects, deltas=[0.0, 1.0], n_draws=500, seed=12, alpha=0.05
    )
    assert result["curve"][1]["power"] > 0.99


def test_detectability_floor_power_is_monotone_and_reports_delta80():
    effects = np.random.default_rng(2).normal(scale=1.0, size=17)
    deltas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    result = detectability_floor(
        effects, deltas=deltas, n_draws=1500, seed=13, alpha=0.05
    )
    powers = [row["power"] for row in result["curve"]]
    assert powers == sorted(powers)
    assert result["delta80"] is not None
    assert result["delta80"] in deltas


def test_detectability_floor_returns_none_when_grid_never_reaches_80_percent():
    effects = np.random.default_rng(3).normal(scale=10.0, size=17)
    result = detectability_floor(
        effects, deltas=[0.0, 0.001], n_draws=400, seed=14, alpha=0.05
    )
    assert result["delta80"] is None


def test_detectability_floor_records_observed_median_and_tie_fraction():
    effects = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7, 0.8, -0.9, 1.0])
    result = detectability_floor(
        effects, deltas=[0.0], n_draws=300, seed=15, alpha=0.05
    )
    assert result["observed_median"] == pytest.approx(np.median(effects))
    assert result["n_patients"] == 10
    assert 0.0 <= result["curve"][0]["tie_fraction"] <= 1.0


def test_detectability_floor_smoothed_mode_removes_bootstrap_ties():
    effects = np.random.default_rng(4).normal(scale=1.0, size=17)
    smoothed = detectability_floor(
        effects, deltas=[0.5], n_draws=400, seed=16, alpha=0.05, smooth=True
    )
    plain = detectability_floor(
        effects, deltas=[0.5], n_draws=400, seed=16, alpha=0.05, smooth=False
    )
    assert smoothed["curve"][0]["tie_fraction"] == 0.0
    assert plain["curve"][0]["tie_fraction"] > 0.0


def test_detectability_floor_rejects_too_few_patients():
    with pytest.raises(ValueError):
        detectability_floor([0.1, 0.2], deltas=[0.0], n_draws=10, seed=1)
