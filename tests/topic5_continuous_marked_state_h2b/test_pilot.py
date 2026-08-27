from pathlib import Path

import numpy as np

from src.topic5_continuous_marked_state_h2b.pilot import (
    _mask_signature,
    _outside_intervals,
)


def test_mask_signature_is_deterministic_and_order_sensitive():
    left = _mask_signature(np.asarray([True, False, True]))
    right = _mask_signature(np.asarray([True, False, True]))
    changed = _mask_signature(np.asarray([True, True, False]))
    assert left == right
    assert left != changed


def test_wrong_time_interval_check_is_closed_interval():
    intervals = [(10.0, 20.0), (30.0, 40.0)]
    assert _outside_intervals(9.0, intervals)
    assert not _outside_intervals(10.0, intervals)
    assert not _outside_intervals(35.0, intervals)
    assert _outside_intervals(41.0, intervals)
