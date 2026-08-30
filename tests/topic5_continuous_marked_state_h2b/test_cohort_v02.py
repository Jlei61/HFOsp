from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_h2b.cohort_v02 import (
    build_subject_query_bundle,
)
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable


def _coverage() -> CoverageTable:
    value = CoverageTable(
        subject="epilepsiae_fixture",
        start=np.asarray([0.0, 6000.0], dtype=np.float64),
        stop=np.asarray([5000.0, 12000.0], dtype=np.float64),
        session=np.asarray([0, 1], dtype=np.int64),
        train_end_epoch=8000.0,
        dev_end_epoch=11500.0,
        source_hashes={"fixture": "f" * 64},
    )
    value.validate()
    return value


def test_query_bundle_uses_exact_cutoff_and_never_crosses_gap():
    coverage = _coverage()
    first = np.arange(30.0, 5000.0, 30.0)
    second = np.arange(6030.0, 12000.0, 30.0)
    grid = np.concatenate([first, second]).astype(np.float64)
    segment = np.concatenate([
        np.zeros(len(first), dtype=np.int64),
        np.ones(len(second), dtype=np.int64),
    ])
    session = segment.copy()
    bundle = build_subject_query_bundle(
        subject="epilepsiae_fixture",
        seizure_rows=[
            {"seizure_id": "s1", "onset_epoch": 4005.0, "offset_epoch": 4040.0},
            {"seizure_id": "s2", "onset_epoch": 7005.0, "offset_epoch": 7050.0},
            {"seizure_id": "sealed", "onset_epoch": 11600.0, "offset_epoch": 11620.0},
        ],
        coverage=coverage,
        grid_time_epoch=grid,
        grid_segment=segment,
        grid_continuity_session=session,
    )
    support = {
        (row["seizure_id"], row["lead_minutes"]): row
        for row in bundle.support_rows
    }
    assert support[("s1", 30)]["eligible"] is True
    assert support[("s1", 30)]["current_observation_age_seconds"] == 15.0
    assert support[("s2", 30)]["eligible"] is False
    assert support[("s2", 30)]["exclusion_reason"] == (
        "lead_window_crosses_gap_or_excluded_interval"
    )
    assert support[("sealed", 5)]["eligible"] is False
    assert support[("sealed", 5)]["exclusion_reason"] == (
        "outside_development_partition"
    )
    exact = [
        row for row in bundle.query_rows
        if row["case_seizure_id"] == "s1" and row["case_lead_minutes"] == 30
    ]
    assert len(exact) == 1
    assert exact[0]["anchor_time_epoch"] == 2205.0
    assert exact[0]["coverage_segment_index"] == 0
    assert bundle.summary["formal_test_partition_opened"] is False
    assert bundle.summary["sealed_opened"] is False
    assert bundle.summary["post_development_seizure_identifiers_persisted"] is False
    assert {row["seizure_id"] for row in bundle.seizure_rows} == {"s1", "s2"}
    unsupported = next(row for row in bundle.seizure_rows if row["seizure_id"] == "s2")
    assert unsupported["primary_30min_supported"] is False


def test_case_cutoff_inside_prior_postictal_interval_is_not_eligible():
    coverage = _coverage()
    grid = np.arange(30.0, 5000.0, 30.0, dtype=np.float64)
    bundle = build_subject_query_bundle(
        subject="epilepsiae_fixture",
        seizure_rows=[
            {"seizure_id": "prior", "onset_epoch": 1000.0, "offset_epoch": 1100.0},
            {"seizure_id": "target", "onset_epoch": 4000.0, "offset_epoch": 4050.0},
        ],
        coverage=coverage,
        grid_time_epoch=grid,
        grid_segment=np.zeros(len(grid), dtype=np.int64),
        grid_continuity_session=np.zeros(len(grid), dtype=np.int64),
    )
    row = next(
        value for value in bundle.support_rows
        if value["seizure_id"] == "target" and value["lead_minutes"] == 30
    )
    assert row["eligible"] is False
    assert row["cutoff_ictal_postictal_exclusion_clear"] is False
    assert row["exclusion_reason"] == "cutoff_in_ictal_or_postictal_interval"


def test_no_fresh_background_observation_is_measured_not_imputed():
    coverage = _coverage()
    bundle = build_subject_query_bundle(
        subject="epilepsiae_fixture",
        seizure_rows=[
            {"seizure_id": "s1", "onset_epoch": 4005.0, "offset_epoch": 4040.0},
        ],
        coverage=coverage,
        grid_time_epoch=np.asarray([2100.0], dtype=np.float64),
        grid_segment=np.asarray([0], dtype=np.int64),
        grid_continuity_session=np.asarray([0], dtype=np.int64),
    )
    row = next(
        value for value in bundle.support_rows
        if value["seizure_id"] == "s1" and value["lead_minutes"] == 30
    )
    assert row["eligible"] is False
    assert row["exclusion_reason"] == "no_causal_observation_within_30_seconds"
    assert bundle.summary["n_primary_eligible_seizures"] == 0
    assert bundle.summary["support_tier"] == "not_estimable"


def test_coverage_stop_equal_to_onset_is_a_complete_preonset_window():
    coverage = CoverageTable(
        subject="epilepsiae_fixture",
        start=np.asarray([0.0], dtype=np.float64),
        stop=np.asarray([4005.0], dtype=np.float64),
        session=np.asarray([0], dtype=np.int64),
        train_end_epoch=3000.0, dev_end_epoch=4500.0,
        source_hashes={"fixture": "f" * 64},
    )
    coverage.validate()
    grid = np.arange(15.0, 4005.0, 30.0, dtype=np.float64)
    bundle = build_subject_query_bundle(
        subject="epilepsiae_fixture",
        seizure_rows=[
            {"seizure_id": "s1", "onset_epoch": 4005.0, "offset_epoch": 4020.0},
        ],
        coverage=coverage,
        grid_time_epoch=grid,
        grid_segment=np.zeros(len(grid), dtype=np.int64),
        grid_continuity_session=np.zeros(len(grid), dtype=np.int64),
    )
    row = next(value for value in bundle.support_rows
               if value["lead_minutes"] == 30)
    assert row["complete_recorded_lead_window"] is True
    assert row["eligible"] is True
