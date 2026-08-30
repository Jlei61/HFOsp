import numpy as np

from src.interictal_spatial_information_gain import (
    blockwise_permutation,
    build_hybrid_training_features,
    compute_crossfit_spatial_information_gain,
    equal_view_spatial_scale,
    fit_evaluate_crossfit_fold,
    fit_full_spatial_template_model,
    fit_full_temporal_template_model,
)
from src.topic5_interictal_direction_rose import fit_event_directions_3d


def test_equal_view_scale_matches_total_training_variance():
    temporal = np.array([
        [0.0, 0.2, 0.8, 1.0],
        [0.1, 0.4, 0.7, 0.9],
        [0.9, 0.6, 0.3, 0.1],
        [1.0, 0.8, 0.2, 0.0],
    ])
    directions = np.array([
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [-0.8, -0.6, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    scale = equal_view_spatial_scale(temporal, directions)
    hybrid, observed_scale = build_hybrid_training_features(temporal, directions)
    temporal_variance = np.var(hybrid[:, : temporal.shape[1]], axis=0).sum()
    spatial_variance = np.var(hybrid[:, temporal.shape[1] :], axis=0).sum()
    assert observed_scale == scale
    assert np.isclose(temporal_variance, spatial_variance)


def test_blockwise_permutation_never_crosses_recording_blocks():
    blocks = np.repeat([2, 5, 9], [4, 3, 5])
    permutation = blockwise_permutation(blocks, np.random.default_rng(7))
    assert sorted(permutation.tolist()) == list(range(len(blocks)))
    assert np.array_equal(blocks[permutation], blocks)


def test_crossfit_uses_rank_only_heldout_assignment_and_returns_null():
    coords = np.array([
        [-2.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    base_forward = np.argsort(np.argsort(coords[:, 0])).astype(float)
    base_reverse = np.argsort(np.argsort(-coords[:, 0])).astype(float)
    rows = []
    blocks = []
    rng = np.random.default_rng(11)
    for block in range(4):
        for event in range(40):
            base = base_forward if event % 2 == 0 else base_reverse
            rows.append(base + rng.normal(0.0, 0.03, size=len(base)))
            blocks.append(block)
    ranks = np.asarray(rows, float).T
    bools = np.ones_like(ranks, dtype=bool)
    directions = fit_event_directions_3d(ranks, coords, min_contacts=3)[
        "directions"
    ]

    fold_zero = fit_evaluate_crossfit_fold(
        ranks,
        bools,
        directions,
        blocks,
        coords,
        fold_index=0,
        min_cluster_events=5,
    )
    assert set(fold_zero["train_blocks"].tolist()) == {0, 2}
    assert set(fold_zero["test_blocks"].tolist()) == {1, 3}
    assert fold_zero["train_label_overlap"] >= 0.5

    result = compute_crossfit_spatial_information_gain(
        ranks,
        bools,
        directions,
        blocks,
        coords,
        min_cluster_events=5,
        n_null=20,
        seed=3,
    )
    assert result["status"] == "ok"
    assert len(result["folds"]) == 2
    assert np.asarray(result["direction_shuffle_null_gain"]).shape == (20,)
    assert np.asarray(
        result["direction_shuffle_null_timing_only_score"]
    ).shape == (20,)
    assert np.asarray(
        result["direction_shuffle_null_timing_plus_space_score"]
    ).shape == (20,)
    assert np.allclose(
        result["direction_shuffle_null_gain"],
        np.asarray(result["direction_shuffle_null_timing_plus_space_score"])
        - np.asarray(result["direction_shuffle_null_timing_only_score"]),
    )
    assert np.isfinite(result["timing_only_score"])
    assert np.isfinite(result["timing_plus_space_score"])
    assert result["contract"]["heldout_assignment"].startswith("rank-template")

    # Fold 0 trains on even-indexed blocks and tests on odd-indexed blocks.
    # Changing only those held-out directions may change its outcome score, but
    # it must not change the frozen hybrid model's rank-only assignments.
    changed_directions = directions.copy()
    odd_test = np.isin(np.asarray(blocks), [1, 3])
    changed_directions[odd_test] = changed_directions[odd_test][::-1]
    changed = compute_crossfit_spatial_information_gain(
        ranks,
        bools,
        changed_directions,
        blocks,
        coords,
        min_cluster_events=5,
        n_null=5,
        seed=4,
    )
    assert np.array_equal(
        result["folds"][0]["timing_plus_space_test_cluster_counts"],
        changed["folds"][0]["timing_plus_space_test_cluster_counts"],
    )
    assert np.array_equal(
        result["folds"][0]["timing_plus_space_train_cluster_counts"],
        changed["folds"][0]["timing_plus_space_train_cluster_counts"],
    )


def test_full_spatial_fit_orders_template_a_by_event_prevalence():
    coords = np.array([
        [-2.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
    ])
    forward = np.argsort(np.argsort(coords[:, 0])).astype(float)
    reverse = np.argsort(np.argsort(-coords[:, 0])).astype(float)
    rng = np.random.default_rng(29)
    events = [forward + rng.normal(0, 0.02, 6) for _ in range(42)]
    events += [reverse + rng.normal(0, 0.02, 6) for _ in range(18)]
    ranks = np.asarray(events).T
    bools = np.ones_like(ranks, dtype=bool)
    directions = fit_event_directions_3d(ranks, coords, min_contacts=3)[
        "directions"
    ]

    fitted = fit_full_spatial_template_model(
        ranks,
        bools,
        directions,
        coords,
        min_cluster_events=5,
    )
    assert fitted["cluster_counts"].tolist() == [42, 18]
    assert np.sum(fitted["labels"] == 0) == 42
    assert np.sum(fitted["labels"] == 1) == 18
    assert fitted["supports"].shape == (2, 6)
    assert fitted["template_label_rule"].startswith("A=more events")

    timing = fit_full_temporal_template_model(
        ranks,
        bools,
        coords,
        min_cluster_events=5,
    )
    assert timing["cluster_counts"].tolist() == [42, 18]
    assert timing["templates"].shape == (2, 6)
