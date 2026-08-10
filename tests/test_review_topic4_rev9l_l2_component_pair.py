import numpy as np

from scripts.review_topic4_rev9l_l2_oracle import (
    _curve_signature,
    arrays_equal_nan,
)


def test_arrays_equal_nan_requires_shape_and_values():
    left = np.asarray([1.0, np.nan, 2.0])
    assert arrays_equal_nan(left, left.copy())
    assert not arrays_equal_nan(left, np.asarray([1.0, np.nan, 3.0]))
    assert not arrays_equal_nan(left, np.asarray([[1.0, np.nan, 2.0]]))


def test_curve_signature_preserves_nan_mask_and_order():
    first = [np.asarray([1.0, np.nan]), np.asarray([2.0, 3.0])]
    assert _curve_signature(first) == _curve_signature([row.copy() for row in first])
    assert _curve_signature(first) != _curve_signature(first[::-1])
    assert _curve_signature(first) != _curve_signature([
        np.asarray([1.0, 0.0]), np.asarray([2.0, 3.0])])
