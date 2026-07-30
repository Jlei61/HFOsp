import numpy as np
import pytest

from src.topic5_scaffold_reliability import (
    event_count_saturation,
    field_comparison,
    participation_field,
)


def test_participation_field_is_event_first_and_masks_nonparticipants():
    groups = np.array([[0, -1, 1], [1, 0, -1], [-1, 0, 1]])
    np.testing.assert_allclose(
        participation_field(groups), np.array([2 / 3, 2 / 3, 2 / 3])
    )


def test_field_comparison_returns_exact_identity_metrics():
    metric = field_comparison(
        np.array([0.1, 0.8, 0.4, 0.2]),
        np.array([0.1, 0.8, 0.4, 0.2]),
    )
    assert metric["spearman_rho"] == pytest.approx(1.0)
    assert metric["top_quartile_jaccard"] == pytest.approx(1.0)
    assert metric["mean_absolute_error"] == pytest.approx(0.0)


def test_saturation_never_uses_more_events_than_available_and_is_deterministic():
    groups = np.array(
        [[0, -1, 1], [1, 0, -1], [-1, 0, 1], [0, 1, -1]]
    )
    reference = participation_field(groups)
    first = event_count_saturation(
        groups,
        reference,
        event_counts=[2, 4, 5],
        n_subsamples=3,
        seed=17,
    )
    second = event_count_saturation(
        groups,
        reference,
        event_counts=[2, 4, 5],
        n_subsamples=3,
        seed=17,
    )
    assert first == second
    assert {row["event_count"] for row in first} == {2, 4}
