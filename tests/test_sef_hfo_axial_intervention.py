import numpy as np
import pytest

from src.sef_hfo_axial_intervention import (
    band_mask, split_near_target_far, core_source_raw,
    participation_ratio, exclude_target_contacts,
)


# ---------------- Task 1: geometry + source helpers ----------------

def test_band_mask_perpendicular_to_axis():
    axis = np.array([1.0, 0.0]); center = np.array([5.0, 5.0])
    coords = np.array([[5.0, 5.0], [5.4, 9.0], [5.6, 0.0], [8.0, 5.0]])
    m = band_mask(coords, axis, center, thickness=1.0)   # |x-5| <= 0.5
    assert m.tolist() == [True, True, False, False]


def test_split_near_target_far_orients_by_source():
    axis = np.array([1.0, 0.0]); center = np.array([5.0, 5.0])
    coords = np.array([[9.0, 5.0], [1.0, 5.0], [5.0, 5.0]])
    s = split_near_target_far(coords, axis, center, np.array([9.0, 5.0]), target_thickness=1.0)
    assert s["near"].tolist() == [True, False, False]
    assert s["far"].tolist() == [False, True, False]
    assert s["target"].tolist() == [False, False, True]
    # source on the -axis side flips near/far
    s2 = split_near_target_far(coords, axis, center, np.array([1.0, 5.0]), target_thickness=1.0)
    assert s2["near"].tolist() == [False, True, False]
    assert s2["far"].tolist() == [True, False, False]


def test_core_source_raw_independent_of_readability():
    assert core_source_raw(10.0, 60.0, 30.0) == "neg"          # |10-60|=50 > 30
    assert core_source_raw(60.0, 10.0, 30.0) == "pos"
    assert core_source_raw(10.0, 25.0, 30.0) == "collision"    # |10-25|=15 <= 30
    assert core_source_raw(10.0, None, 30.0) == "neg"          # single core fires
    assert core_source_raw(None, 15.0, 30.0) == "pos"
    assert core_source_raw(None, None, 30.0) == "none"         # neither -> none (not ambiguous)


def test_participation_ratio_excludes_clamped_denominator():
    fired = np.array([True, True, False, False])
    region = np.array([True, True, True, True])
    assert participation_ratio(fired, region) == 0.5            # 2/4
    free = np.array([True, True, False, False])                 # idx 2,3 clamped -> out of denom
    assert participation_ratio(fired, region, valid=free) == 1.0   # 2/2 (fair)
    r = participation_ratio(fired, np.zeros(4, bool))
    assert r != r                                              # NaN on empty denom


def test_exclude_target_contacts():
    valid = np.array([True, True, True, False])
    target = np.array([False, True, False, True])
    out = exclude_target_contacts(valid, target)
    assert out.tolist() == [True, False, True, False]
    assert valid.tolist() == [True, True, True, False]         # input not mutated
