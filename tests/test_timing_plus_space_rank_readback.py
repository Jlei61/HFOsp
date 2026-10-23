import numpy as np

from scripts.run_timing_plus_space_rank_readback import (
    _fit_missing_view_labels,
    _readback_fold,
)


def test_masked_spatial_view_retains_events_without_finite_directions():
    temporal = np.array([
        [0.0, 0.1, 0.2],
        [0.1, 0.0, 0.2],
        [0.0, 0.2, 0.1],
        [0.2, 0.1, 0.0],
        [0.9, 1.0, 0.8],
        [1.0, 0.9, 0.8],
        [0.8, 1.0, 0.9],
        [1.0, 0.8, 0.9],
    ])
    directions = np.array([
        [1.0, 0.0, 0.0],
        [np.nan, np.nan, np.nan],
        [0.9, 0.1, 0.0],
        [1.0, -0.1, 0.0],
        [-1.0, 0.0, 0.0],
        [-0.9, -0.1, 0.0],
        [np.nan, np.nan, np.nan],
        [-1.0, 0.1, 0.0],
    ])

    result = _fit_missing_view_labels(
        temporal, directions, random_state=3, n_init=4
    )

    assert result["labels"].shape == (8,)
    assert int(np.sum(result["counts"])) == 8
    assert int(np.sum(result["spatial_counts"])) == 6
    assert result["labels"][0] == result["labels"][1]
    assert result["labels"][4] == result["labels"][6]
    assert result["labels"][0] != result["labels"][4]


def test_rank_only_heldout_assignment_reads_back_both_template_axes():
    x = np.arange(6, dtype=float)
    coords = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)])
    forward = x / x.max()
    reverse = forward[::-1]
    masked_ranks = np.column_stack(
        [forward, reverse, forward, reverse, forward, reverse, forward, reverse]
    )
    bools = np.ones_like(masked_ranks, dtype=bool)
    model = {
        "templates": np.vstack([forward, reverse]),
        "axes": np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
    }

    result = _readback_fold(
        model,
        masked_ranks,
        bools,
        coords,
        np.arange(masked_ranks.shape[1]),
    )

    assert result["assignment_coverage"] == 1.0
    assert result["equal_template_rho"] == 1.0
    assert [row["n_events"] for row in result["cluster_rows"]] == [4, 4]
