import numpy as np

from scripts.build_topic5_interictal_operator_dataset import (
    yuquan_interictal_block_mask,
)


def test_yuquan_two_hour_adjacency_is_not_treated_as_a_gap():
    starts = np.array([0.0, 7200.0, 14400.0])
    good = yuquan_interictal_block_mask(
        starts,
        ["r0", "r1", "r2"],
        [],
        post_guard_sec=7200.0,
        end_by_record={},
    )
    np.testing.assert_array_equal(good, [True, True, True])


def test_yuquan_seizure_and_post_guard_exclude_overlapping_records_only():
    starts = np.array([0.0, 7200.0, 14400.0, 21600.0])
    ends = {f"r{i}": float(start + 7200) for i, start in enumerate(starts)}
    # Seizure in r1; 2 h post guard also overlaps r2 but not r3.
    intervals = [(0, 8000.0, 8500.0)]
    good = yuquan_interictal_block_mask(
        starts,
        ["r0", "r1", "r2", "r3"],
        intervals,
        post_guard_sec=7200.0,
        end_by_record=ends,
    )
    np.testing.assert_array_equal(good, [True, False, False, True])
