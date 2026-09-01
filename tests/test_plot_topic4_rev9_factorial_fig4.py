import numpy as np

from scripts.plot_topic4_rev9_factorial_fig4 import (
    _event_order,
    _window_bounds,
)


def test_window_bounds_is_fixed_width_and_stays_inside_record():
    assert _window_bounds(20.0, 40.0, 0.0, 8000.0) == (0.0, 700.0)
    assert _window_bounds(7970.0, 7990.0, 0.0, 8000.0) == (7300.0, 8000.0)


def test_event_order_groups_kmeans_labels_before_slope():
    curves = np.asarray([
        [0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [2.0, 0.0]])
    labels = np.asarray([1, 0, 1, 0])
    order = _event_order(curves, labels)
    np.testing.assert_array_equal(labels[order], [0, 0, 1, 1])
