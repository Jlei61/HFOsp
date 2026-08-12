import numpy as np

from scripts.paper_figures.plot_fig4_continuous_field_v62_validation import (
    formal_clean_mask,
    matrix_acceptance_status,
    normalize_event_ranks,
)


def test_normalize_event_ranks_preserves_missing_contacts():
    ranks = np.asarray([[2.0, np.nan, 4.0, 3.0], [7.0, 7.0, np.nan, np.nan]])
    normalized = normalize_event_ranks(ranks)
    np.testing.assert_allclose(normalized[0, [0, 2, 3]], [0.0, 1.0, 0.5])
    assert np.isnan(normalized[0, 1])
    np.testing.assert_allclose(normalized[1, :2], [0.0, 0.0])


def test_formal_clean_mask_keeps_filter_terms_coupled():
    onsets = np.asarray([
        [1.0, np.nan, 2.0, np.nan],
        [1.0, np.nan, np.nan, np.nan],
        [1.0, np.nan, 2.0, np.nan],
        [np.nan, np.nan, 2.0, np.nan],
    ])
    groups = {"ICL": np.asarray([0, 1]), "SCL": np.asarray([2, 3])}
    labels = np.asarray([0, 0, 1, 1])
    ood = np.asarray([False, False, True, False])
    np.testing.assert_array_equal(
        formal_clean_mask(onsets, labels, ood, groups),
        [True, False, False, False],
    )


def test_matrix_acceptance_fails_when_one_direction_is_under_supported():
    labels = np.asarray([0] + [1] * 8)
    clean = np.ones(len(labels), bool)
    valid, counts = matrix_acceptance_status(labels, clean, required_per_mode=6)
    assert not valid
    np.testing.assert_array_equal(counts, [1, 8])


def test_matrix_acceptance_passes_only_when_both_directions_reach_budget():
    labels = np.asarray([0] * 6 + [1] * 7)
    clean = np.ones(len(labels), bool)
    valid, counts = matrix_acceptance_status(labels, clean, required_per_mode=6)
    assert valid
    np.testing.assert_array_equal(counts, [6, 7])
