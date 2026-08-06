"""Tests for locating an event's ignition site and attributing it to a source region."""
from __future__ import annotations

import numpy as np
import pytest

from src.topic4_fcxr_lc3_ignition import (
    assign_to_source,
    classify_events,
    event_ignition_xy,
    pick_representatives,
)

CORE_A = np.array([3.5, 8.6])
CORE_B = np.array([16.6, 8.6])


def _sheet(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, 20, n), rng.uniform(0, 20, n)])


def _wave(pos, origin, speed=1.0, radius=None):
    """First-spike times for a wave leaving ``origin``; NaN outside ``radius``."""
    d = np.linalg.norm(pos - np.asarray(origin, float), axis=1)
    t = d / speed
    if radius is not None:
        t = np.where(d <= radius, t, np.nan)
    return t


def test_ignition_recovers_the_origin_of_a_radial_wave():
    pos = _sheet()
    xy = event_ignition_xy(_wave(pos, CORE_A), pos)
    assert np.linalg.norm(xy - CORE_A) < 0.6


def test_ignition_ignores_cells_that_never_fired():
    pos = _sheet()
    onset = _wave(pos, CORE_B, radius=4.0)
    xy = event_ignition_xy(onset, pos)
    assert np.linalg.norm(xy - CORE_B) < 0.6


def test_ignition_returns_none_when_almost_nothing_fired():
    pos = _sheet()
    onset = np.full(len(pos), np.nan)
    onset[:5] = 0.0
    assert event_ignition_xy(onset, pos) is None


def test_ignition_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        event_ignition_xy(np.zeros(10), _sheet())


def test_a_patch_around_one_core_is_attributed_to_that_core():
    """The failure this guards: a patch confined to one core carries a direction sign,
    but its origin -- what the column claims -- is the core it sits on."""
    pos = _sheet()
    for core, want in ((CORE_A, "a"), (CORE_B, "b")):
        xy = event_ignition_xy(_wave(pos, core, radius=4.0), pos)
        assert assign_to_source(xy, CORE_A, CORE_B) == want


def test_a_transit_is_attributed_to_the_end_it_left_from_not_the_one_it_reached():
    pos = _sheet()
    xy = event_ignition_xy(_wave(pos, CORE_B), pos)          # covers the whole sheet
    assert assign_to_source(xy, CORE_A, CORE_B) == "b"


def test_a_midway_ignition_is_attributed_to_neither():
    mid = (CORE_A + CORE_B) / 2.0
    assert assign_to_source(mid, CORE_A, CORE_B) is None


def test_attribution_is_stable_across_the_range_of_ratios_the_data_spans():
    """The 2x rule is a convention, not a fit: the answer must not turn on its value."""
    pos = _sheet()
    xy_patch = event_ignition_xy(_wave(pos, CORE_A, radius=4.0), pos)
    xy_mid = (CORE_A + CORE_B) / 2.0 + np.array([0.9, 0.0])
    for factor in (1.4, 2.0, 3.0, 5.0):
        assert assign_to_source(xy_patch, CORE_A, CORE_B, closer_by=factor) == "a"
        assert assign_to_source(xy_mid, CORE_A, CORE_B, closer_by=factor) is None


def test_assign_passes_through_a_missing_ignition():
    assert assign_to_source(None, CORE_A, CORE_B) is None


def test_representatives_pick_the_least_ambiguous_event_per_region():
    pos = _sheet()
    near_a = _wave(pos, CORE_A, radius=4.0)
    off_a = _wave(pos, CORE_A + np.array([1.8, 0.0]), radius=4.0)
    near_b = _wave(pos, CORE_B, radius=4.0)
    cls = classify_events([off_a, near_b, near_a], pos, CORE_A, CORE_B)
    assert [c["source"] for c in cls] == ["a", "b", "a"]
    assert pick_representatives(cls) == (2, 1)


def test_representatives_report_none_for_a_region_that_ignited_nothing():
    pos = _sheet()
    cls = classify_events([_wave(pos, CORE_A, radius=4.0)], pos, CORE_A, CORE_B)
    assert pick_representatives(cls) == (0, None)


def test_classify_reports_both_distances_so_the_margin_is_auditable():
    pos = _sheet()
    cls = classify_events([_wave(pos, CORE_A, radius=4.0)], pos, CORE_A, CORE_B)[0]
    assert cls["dist_a_mm"] < cls["dist_b_mm"]
    assert cls["ignition_xy"] is not None
