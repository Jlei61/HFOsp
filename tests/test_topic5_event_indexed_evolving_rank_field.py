import numpy as np

from src.topic5_event_indexed_evolving_rank_field import (
    block_reliability,
    dynamic_observability_audit,
    estimate_block_descriptor,
    global_rank_prior,
    make_equal_event_blocks,
    permute_block_events_within_source,
    stratified_uniform_subsample_blocks_with_adjacency,
    within_source_pairs,
    pca_reconstruction_gain,
)


def _synthetic(dynamic: bool, seed: int = 7):
    rng = np.random.default_rng(seed)
    n_events = 2400
    n_contacts = 8
    source = np.repeat(np.arange(3), n_events // 3)
    base = np.linspace(-1.0, 1.0, n_contacts)
    loading = np.asarray([-1.0, -0.7, -0.3, 0.0, 0.2, 0.5, 0.8, 1.0])
    participation = rng.random((n_events, n_contacts)) < 0.82
    local_rank = np.full((n_events, n_contacts), np.nan, dtype=float)
    groups = np.full((n_events, n_contacts), -1, dtype=np.int16)
    for event in range(n_events):
        phase = 1.3 * np.sin(2 * np.pi * event / 400.0) if dynamic else 0.0
        score = base + phase * loading + rng.normal(0.0, 0.35, n_contacts)
        present = np.flatnonzero(participation[event])
        if len(present) < 3:
            present = np.arange(3)
            participation[event, :3] = True
        order = present[np.argsort(score[present], kind="stable")]
        groups[event, order] = np.arange(len(order), dtype=np.int16)
        local_rank[event, order] = np.linspace(0.0, 1.0, len(order))
    return local_rank, participation, groups, source


def test_equal_event_blocks_do_not_cross_source_records():
    source = np.asarray([0] * 5 + [1] * 7)
    blocks = make_equal_event_blocks(np.arange(12), source, 3)
    assert [block.source_block_id for block in blocks] == [0, 1, 1]
    assert all(len(block.indices) == 3 for block in blocks)
    assert all(len(np.unique(source[block.indices])) == 1 for block in blocks)


def test_block_permutation_preserves_sampled_positions_and_membership():
    source = np.asarray([0] * 12 + [1] * 12)
    blocks = make_equal_event_blocks(np.arange(24), source, 3)
    sampled = [blocks[0], blocks[2], blocks[4], blocks[7]]
    shuffled = permute_block_events_within_source(
        sampled, np.random.default_rng(9)
    )
    assert [block.source_block_id for block in shuffled] == [0, 0, 1, 1]
    assert [block.within_source_order for block in shuffled] == [0, 2, 0, 3]
    for source_id in (0, 1):
        original = np.concatenate(
            [block.indices for block in sampled if block.source_block_id == source_id]
        )
        permuted = np.concatenate(
            [block.indices for block in shuffled if block.source_block_id == source_id]
        )
        assert set(map(int, original)) == set(map(int, permuted))


def test_adjacency_means_true_original_block_adjacency_after_subsampling():
    source = np.asarray([0] * 60)
    blocks = make_equal_event_blocks(np.arange(60), source, 3)
    sparse = [blocks[0], blocks[2], blocks[5]]
    assert within_source_pairs(sparse, adjacent_only=True) == []
    sampled = stratified_uniform_subsample_blocks_with_adjacency(blocks, 10)
    adjacent = within_source_pairs(sampled, adjacent_only=True)
    assert len(sampled) == 10
    assert len(adjacent) >= 4
    assert all(lag == 1 for _, _, lag in adjacent)


def test_descriptor_respects_mask_and_ties():
    rank = np.asarray([[0.0, 0.0, np.nan], [0.0, 1.0, 0.5]])
    participation = np.isfinite(rank)
    groups = np.asarray([[0, 0, -1], [0, 2, 1]])
    descriptor = estimate_block_descriptor(
        rank,
        participation,
        groups,
        [0, 1],
        rank_prior=np.asarray([0.0, 0.5, 0.5]),
        shrinkage_prior_events=0.0,
        beta_prior=0.5,
    )
    # One tie contributes 0.5 and one ordered observation contributes 1.0;
    # the Beta(0.5, 0.5) prior gives (1.5 + 0.5) / 3.
    assert np.isclose(descriptor.precedence[0], 2.0 / 3.0)
    assert descriptor.participation[2] < descriptor.participation[0]


def test_reliability_and_dynamic_null_separate_synthetic_processes():
    dynamic = _synthetic(True)
    static = _synthetic(False)
    indices = np.arange(1800)
    dynamic_prior = global_rank_prior(dynamic[0], dynamic[1], indices)
    blocks = make_equal_event_blocks(indices, dynamic[3], 40)
    reliability = block_reliability(
        dynamic[0],
        dynamic[1],
        dynamic[2],
        blocks,
        rank_prior=dynamic_prior,
        shrinkage_prior_events=4.0,
        beta_prior=0.5,
        repeats=3,
        seed=11,
    )
    assert reliability["rank_spearman_median"] > 0.6
    confirmation = np.arange(1800, 2400)
    dynamic_result = dynamic_observability_audit(
        dynamic[0], dynamic[1], dynamic[2], confirmation, dynamic[3], 40,
        rank_prior=dynamic_prior,
        shrinkage_prior_events=4.0,
        beta_prior=0.5,
        split_repeats=4,
        permutation_draws=39,
        max_between_pairs=1000,
        distance_ratio_threshold=1.10,
        p_threshold=0.05,
        seed=19,
    )
    static_prior = global_rank_prior(static[0], static[1], indices)
    static_result = dynamic_observability_audit(
        static[0], static[1], static[2], confirmation, static[3], 40,
        rank_prior=static_prior,
        shrinkage_prior_events=4.0,
        beta_prior=0.5,
        split_repeats=4,
        permutation_draws=39,
        max_between_pairs=1000,
        distance_ratio_threshold=1.10,
        p_threshold=0.05,
        seed=19,
    )
    assert dynamic_result["observed"]["field_ratio"] > static_result["observed"]["field_ratio"]
    assert dynamic_result["g0_pass"]
    assert not static_result["g0_pass"]


def test_pca_gain_detects_low_rank_signal():
    rng = np.random.default_rng(5)
    loading = rng.normal(size=(10, 2))
    train = rng.normal(size=(200, 2)) @ loading.T + rng.normal(0, 0.05, (200, 10))
    test = rng.normal(size=(100, 2)) @ loading.T + rng.normal(0, 0.05, (100, 10))
    assert pca_reconstruction_gain(train, test, 2) > 0.9
