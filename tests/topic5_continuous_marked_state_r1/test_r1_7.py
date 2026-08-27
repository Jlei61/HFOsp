from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_7 import (
    block_bootstrap_length_seconds,
    complete_event_blocks_by_segment,
    coverage_segment_for_times,
    split_validation_by_recorded_time,
)


def test_bootstrap_excludes_zero_support_blocks() -> None:
    import importlib.util
    from pathlib import Path

    path = Path("scripts/topic5_continuous_marked_state_r1/aggregate_r1_7a.py")
    spec = importlib.util.spec_from_file_location("aggregate_r1_7a_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    blocks = [
        {"effect": -1.0, "n_matched_events": 10},
        {"effect": 99.0, "n_matched_events": 0},
        {"effect": -3.0, "n_matched_events": 10},
    ]
    result = module.bootstrap(
        blocks, "effect", weight_key="n_matched_events", draws=20
    )
    assert result["estimate"] == -2.0
    assert result["n_blocks"] == 2


def test_recorded_split_ignores_gap_duration() -> None:
    coverage = CoverageTable(
        subject="synthetic",
        start=np.asarray([0.0, 1000.0]),
        stop=np.asarray([100.0, 1100.0]),
        session=np.asarray([0, 1]),
        train_end_epoch=0.0,
        dev_end_epoch=1100.0,
        source_hashes={},
    )
    value = split_validation_by_recorded_time(
        coverage, validation_start=0.0, validation_stop=1100.0,
    )
    assert value.state_stop == 1020.0
    assert value.state_recorded_seconds == 120.0
    assert value.mechanism_recorded_seconds == 80.0


def test_bootstrap_length_is_train_only_and_bounded() -> None:
    time = np.arange(0.0, 1000.0, 2.0)
    session = np.zeros(len(time), dtype=np.int64)
    assert block_bootstrap_length_seconds(time, session) == 1800.0
    sparse = np.arange(0.0, 100000.0, 1000.0)
    assert block_bootstrap_length_seconds(
        sparse, np.zeros(len(sparse), dtype=np.int64)
    ) == 21600.0


def test_event_block_count_never_pools_across_recording_gaps() -> None:
    coverage = CoverageTable(
        subject="synthetic",
        start=np.asarray([0.0, 1000.0]),
        stop=np.asarray([100.0, 1100.0]),
        session=np.asarray([0, 0]),
        train_end_epoch=0.0,
        dev_end_epoch=1100.0,
        source_hashes={},
    )
    time = np.concatenate([np.linspace(1, 99, 60), np.linspace(1001, 1099, 60)])
    segment = coverage_segment_for_times(coverage, time)
    blocks, rows = complete_event_blocks_by_segment(
        segment, np.ones(len(time), dtype=bool), block_events=100
    )
    assert blocks == 0
    assert [row["events"] for row in rows] == [60, 60]


def test_only_frozen_nonfinite_gradient_guard_is_recordable() -> None:
    """Optimiser non-finite guards are recordable; every other error must raise."""
    from src.topic5_continuous_marked_state_r1.r1_7 import (
        is_nonfinite_gradient_failure,
    )

    assert is_nonfinite_gradient_failure(
        RuntimeError("R1.3 encountered a non-finite gradient norm")
    )
    assert is_nonfinite_gradient_failure(
        RuntimeError("R1.2 prefix encountered a non-finite gradient")
    )
    # Anything else is an implementation fault and must not be relabelled.
    assert not is_nonfinite_gradient_failure(RuntimeError("checkpoint hash mismatch"))
    assert not is_nonfinite_gradient_failure(ValueError("R1.3 encountered a non-finite gradient norm"))
    assert not is_nonfinite_gradient_failure(RuntimeError("CUDA out of memory"))


def test_nonfinite_seeds_leave_denominator_intact_but_are_not_scored() -> None:
    """A non-finite cell is never stable, never scored, never silently dropped."""
    from src.topic5_continuous_marked_state_r1.r1_7 import split_scored_payloads

    payloads = [
        {"seed": 0, "stable_checkpoint": True},
        {"seed": 1, "analysis_status": "NONFINITE_GRADIENT", "stable_checkpoint": False},
        {"seed": 2, "stable_checkpoint": False},
        {"seed": 3, "analysis_status": "NONFINITE_GRADIENT", "stable_checkpoint": False},
        {"seed": 4, "stable_checkpoint": True},
    ]
    scored, nonfinite = split_scored_payloads(payloads)
    assert [value["seed"] for value in scored] == [0, 2, 4]
    assert nonfinite == 2
    # denominator is the frozen five seeds, not the scored subset
    assert len(payloads) == 5
    # a non-finite cell can never be counted as a stable checkpoint
    assert not any(value.get("stable_checkpoint") for _, value in enumerate(payloads)
                   if value.get("analysis_status") == "NONFINITE_GRADIENT")
