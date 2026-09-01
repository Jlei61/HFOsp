import numpy as np

from scripts.plot_topic4_core_field_stage3_kmeans_consistency import display_indices


def test_display_indices_groups_modes_without_dropping_small_inputs():
    labels = np.asarray([1, 0, 1, 0, 0])
    index = display_indices(labels, max_events=10)
    np.testing.assert_array_equal(labels[index], [0, 0, 0, 1, 1])
    assert sorted(index.tolist()) == list(range(len(labels)))


def test_display_indices_caps_large_patient_row_deterministically():
    labels = np.repeat([0, 1], [900, 100])
    first = display_indices(labels, max_events=200)
    second = display_indices(labels, max_events=200)
    np.testing.assert_array_equal(first, second)
    assert len(first) == 200
    assert np.sum(labels[first] == 0) == 180
    assert np.sum(labels[first] == 1) == 20
    assert np.all(np.diff(first[:180]) > 0)
    assert np.all(np.diff(first[180:]) > 0)
