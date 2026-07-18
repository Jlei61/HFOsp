import numpy as np

from scripts.run_topic5_axis_representativeness import _template_row
from src.topic5_axis_representativeness import (
    rank_shuffle_axis_null,
    summarize_direction_representativeness,
)


def test_representativeness_is_one_for_perfectly_aligned_events():
    events = np.tile([1.0, 0.0, 0.0], (20, 1))
    result = summarize_direction_representativeness(events, [1.0, 0.0, 0.0])
    assert result["n_events"] == 20
    assert np.isclose(result["mean_signed_cosine"], 1.0)
    assert np.isclose(result["resultant_length_3d"], 1.0)
    assert np.isclose(result["axis_to_main_direction_deg"], 0.0)
    assert np.isclose(result["fraction_within_45deg"], 1.0)


def test_representativeness_penalizes_bidirectional_cancellation():
    events = np.vstack([
        np.tile([1.0, 0.0, 0.0], (10, 1)),
        np.tile([-1.0, 0.0, 0.0], (10, 1)),
    ])
    result = summarize_direction_representativeness(events, [1.0, 0.0, 0.0])
    assert np.isclose(result["mean_signed_cosine"], 0.0)
    assert np.isclose(result["resultant_length_3d"], 0.0)
    assert np.isnan(result["axis_to_main_direction_deg"])


def test_rank_shuffle_null_is_reproducible_for_both_estimators():
    rng = np.random.default_rng(31)
    coords = rng.normal(size=(10, 3))
    rank = np.arange(10.0)
    events = np.tile([1.0, 0.0, 0.0], (30, 1))
    for method in ("gradient", "endpoint"):
        first = rank_shuffle_axis_null(
            rank, coords, events, method=method, n_perm=25, seed=8
        )
        second = rank_shuffle_axis_null(
            rank, coords, events, method=method, n_perm=25, seed=8
        )
        np.testing.assert_allclose(first["mean_signed_cosine"], second["mean_signed_cosine"])
        assert np.asarray(first["mean_signed_cosine"]).shape == (25,)


def test_nonstrict_axis_remains_primary_eligible_when_event_count_is_sufficient():
    rng = np.random.default_rng(81)
    coords = rng.normal(size=(8, 3))
    row = _template_row(
        subject_id="epilepsiae_test",
        dataset="epilepsiae",
        subject="test",
        template="TA",
        template_rank=np.arange(8.0),
        coords=coords,
        event_directions=np.tile([1.0, 0.0, 0.0], (20, 1)),
        axis=np.array([1.0, 0.0, 0.0]),
        axis_strict=False,
        n_perm=5,
        min_events=20,
        seed=2,
    )
    assert row["analysis_eligible"] is True
    assert row["axis_strict_stability"] is False
    assert row["null_n"] == 5
