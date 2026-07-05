"""Task 0 (M2 data-side audit) pure-metrics TDD.

Tests `event_extent` (per-event axial/lateral footprint on the accepted axis) and
`matched_null_extent` (same-subject same-n_part null under uniform / rate / shaft-matched
draws). These gate whether real interictal HFO group events axially self-limit (cover only
a segment of the propagable axis) or merely run laterally narrow along it.
"""

import numpy as np

from src.topic4_event_extent_audit import (
    cohort_verdict,
    event_extent,
    event_shaft_counts,
    matched_null_extent,
)


def test_event_extent_axially_full_laterally_narrow():
    # 20 channels spread 0..24mm along axis, +-1mm off -> fills axis, narrow lateral
    along = np.linspace(0, 24, 20)
    off = np.array([-1, 1] * 10)
    e = event_extent(along, off, axis_length=24.0)
    # p5->p95 spans 90% of a uniform full-coverage run, so the ceiling here is ~0.90,
    # cleanly above the axial-segment case (~0.225). p5-p95 is deliberately outlier-robust.
    assert e["axial_fraction"] > 0.85         # fills the axis
    assert e["lateral_ratio"] < 0.2           # narrow sideways


def test_event_extent_axial_segment():
    # channels only over 0..6mm of a 24mm axis -> covers a SEGMENT
    along = np.linspace(0, 6, 12)
    off = np.array([-1, 1] * 6)
    e = event_extent(along, off, axis_length=24.0)
    assert e["axial_fraction"] < 0.35         # only a segment of the axis


def test_matched_null_reports_three_modes_and_respects_shaft_counts():
    rng = np.random.default_rng(0)
    along_all = np.linspace(0, 24, 12)
    off_all = np.tile([-1.0, 1.0], 6)
    shaft = np.array(["A"] * 6 + ["B"] * 6, object)   # 6 on each shaft
    rate = np.r_[np.full(6, 3.0), np.full(6, 1.0)]     # shaft A participates more
    out = matched_null_extent(along_all, off_all, n_part=5, axis_length=24.0,
                              n_draw=50, rng=rng, shaft=shaft,
                              shaft_counts={"A": 4, "B": 1}, rate=rate)
    assert set(out) == {"uniform", "rate", "shaft_matched"}     # all three layers present
    # shaft_matched draws 4 from A + 1 from B every time -> n_part=5 honored, never borrows
    assert len(out["shaft_matched"]["axial_fraction"]) == 50
    # a shaft asking for more than its eligible pool yields NO valid draws (skipped, not borrowed)
    bad = matched_null_extent(along_all, off_all, n_part=9, axis_length=24.0, n_draw=20,
                              rng=rng, shaft=shaft, shaft_counts={"A": 9, "B": 0})
    assert len(bad["shaft_matched"]["axial_fraction"]) == 0


def test_event_shaft_counts_groups_by_shaft_and_drops_unparseable():
    counts = event_shaft_counts(["A1", "A3", "B2", "B5", "B7", "??"])
    assert counts == {"A": 2, "B": 3}     # 'A' x2, 'B' x3; unparseable dropped


# --- cohort_verdict: the pre-registered Step-9 gate. Each branch pinned to a number.
def _recs(n, axial_obs, axial_null, lateral_obs, lateral_null, jitter=0.01):
    rng = np.random.default_rng(1)
    return [dict(axial_obs=axial_obs + jitter * rng.standard_normal(),
                 axial_null=axial_null + jitter * rng.standard_normal(),
                 lateral_obs=lateral_obs + jitter * rng.standard_normal(),
                 lateral_null=lateral_null + jitter * rng.standard_normal())
            for _ in range(n)]


def test_verdict_axial_segment_when_low_AF_and_below_null():
    # AF ~0.25 (<=0.5) AND observed axial well below the shaft-matched null -> model it
    v = cohort_verdict(_recs(12, 0.25, 0.60, 0.30, 0.32), rng=np.random.default_rng(0))
    assert v["verdict"] == "AXIAL_SEGMENT"
    assert v["AF"] < 0.5 and v["axial_ci"][1] < 0     # delta CI excludes 0, confining


def test_verdict_axial_extended_lateral_narrow_when_high_AF_narrow_below_null():
    # AF ~0.88 (>=0.75) AND LR ~0.20 (<=0.5) AND lateral below null -> reframe to lateral
    v = cohort_verdict(_recs(12, 0.88, 0.86, 0.20, 0.55), rng=np.random.default_rng(0))
    assert v["verdict"] == "AXIAL_EXTENDED_LATERAL_NARROW"
    assert v["AF"] >= 0.75 and v["LR"] <= 0.5


def test_verdict_sampling_artifact_when_observed_equals_null():
    # AF low BUT observed ~= null (CI includes 0) -> implant artifact, do not over-model
    v = cohort_verdict(_recs(12, 0.25, 0.25, 0.30, 0.30), rng=np.random.default_rng(0))
    assert v["verdict"] == "SAMPLING_ARTIFACT"
    assert v["axial_ci"][0] <= 0 <= v["axial_ci"][1]


def test_verdict_inconclusive_when_few_subjects():
    v = cohort_verdict(_recs(6, 0.25, 0.60, 0.30, 0.32), rng=np.random.default_rng(0))
    assert v["verdict"] == "INCONCLUSIVE"


def test_verdict_inconclusive_when_AF_in_gray_band():
    # 0.5 < AF < 0.75 -> no hard branch
    v = cohort_verdict(_recs(12, 0.62, 0.80, 0.30, 0.40), rng=np.random.default_rng(0))
    assert v["verdict"] == "INCONCLUSIVE"
