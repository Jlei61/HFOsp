from __future__ import annotations

import math

import pytest

from scripts.topic5_continuous_marked_state_r1.aggregate_r1_4 import (
    select_estimable_raw,
)
from src.topic5_continuous_marked_state_r1.r1_3 import (
    classify_raw_gradient_coverage,
)


def test_r1_4_raw_structural_zero_is_persisted_not_raised() -> None:
    assert classify_raw_gradient_coverage([0.0, 0.0, 0.0]) == (
        "NOT_ESTIMABLE", "ZERO_OR_PARTIAL_TARGET_GRADIENT",
    )
    assert classify_raw_gradient_coverage([1.0, 0.0, 2.0]) == (
        "NOT_ESTIMABLE", "ZERO_OR_PARTIAL_TARGET_GRADIENT",
    )
    assert classify_raw_gradient_coverage([1.0, 2.0, 3.0]) == (
        "ESTIMATED", None,
    )
    with pytest.raises(ValueError):
        classify_raw_gradient_coverage([1.0, math.nan, 2.0])


def test_r1_4_raw_aggregation_excludes_non_estimable_seeds() -> None:
    legacy = {"seed": 0}
    estimated = {"seed": 1, "raw_analysis_status": "ESTIMATED"}
    dead = {"seed": 2, "raw_analysis_status": "NOT_ESTIMABLE"}
    assert select_estimable_raw([legacy, estimated, dead]) == [legacy, estimated]
