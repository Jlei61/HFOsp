from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_7 import (
    block_bootstrap_length_seconds,
    split_validation_by_recorded_time,
)


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
