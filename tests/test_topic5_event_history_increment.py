import numpy as np

from src.topic5_event_history_increment import (
    PairDataset,
    build_block_observations,
    evaluate_model_ladder,
    make_pair_dataset,
)
from src.topic5_event_indexed_evolving_rank_field import (
    global_rank_prior,
    make_equal_event_blocks,
)


def _pairs(event_driven: bool, seed: int):
    rng = np.random.default_rng(seed)
    n_train, n_test, dimension = 240, 100, 4
    loading = rng.normal(size=(dimension, 10))

    def sample(n):
        current_state = rng.normal(size=(n, dimension))
        delta_state = rng.normal(size=(n, dimension))
        future_state = 0.65 * current_state
        if event_driven:
            future_state += 0.85 * delta_state
        future_state += rng.normal(0.0, 0.15, size=future_state.shape)
        current = current_state @ loading
        delta = delta_state @ loading
        target = future_state @ loading
        return PairDataset(
            current=current,
            delta=delta,
            target=target,
            covariates=rng.normal(size=(n, 3)),
            source=np.zeros(n, dtype=int),
            current_block_index=np.arange(n),
        )

    return sample(n_train), sample(n_test)


def test_event_history_ladder_detects_increment_not_autonomous_noise():
    positive_train, positive_test = _pairs(True, 3)
    positive = evaluate_model_ladder(
        positive_train,
        positive_test,
        dimension=4,
        alpha_grid=[0.01, 0.1, 1.0, 10.0],
        switching_state_grid=[2, 3],
        validation_fraction=0.2,
        seed=7,
    )
    null_train, null_test = _pairs(False, 3)
    null = evaluate_model_ladder(
        null_train,
        null_test,
        dimension=4,
        alpha_grid=[0.01, 0.1, 1.0, 10.0],
        switching_state_grid=[2, 3],
        validation_fraction=0.2,
        seed=7,
    )
    assert positive["event_relative_gain_over_best"] > 0.5
    assert null["event_relative_gain_over_best"] < 0.05


def test_block_delta_uses_order_but_field_and_pairing_do_not():
    rng = np.random.default_rng(9)
    n_events, n_contacts = 60, 6
    source = np.zeros(n_events, dtype=int)
    participation = np.ones((n_events, n_contacts), dtype=bool)
    local_rank = np.empty((n_events, n_contacts), dtype=float)
    groups = np.empty((n_events, n_contacts), dtype=np.int16)
    for event in range(n_events):
        order = np.arange(n_contacts) if event % 10 < 5 else np.arange(n_contacts)[::-1]
        groups[event, order] = np.arange(n_contacts)
        local_rank[event, order] = np.linspace(0.0, 1.0, n_contacts)
    indices = np.arange(n_events)
    prior = global_rank_prior(local_rank, participation, indices)
    blocks = make_equal_event_blocks(indices, source, 10)
    ordered = build_block_observations(
        local_rank,
        participation,
        groups,
        np.arange(n_events, dtype=float),
        blocks,
        rank_prior=prior,
        shrinkage_prior_events=4.0,
        beta_prior=0.5,
    )
    shuffled = build_block_observations(
        local_rank,
        participation,
        groups,
        np.arange(n_events, dtype=float),
        blocks,
        rank_prior=prior,
        shrinkage_prior_events=4.0,
        beta_prior=0.5,
        rng=rng,
    )
    assert all(np.allclose(a.field, b.field) for a, b in zip(ordered, shuffled))
    assert any(not np.allclose(a.delta, b.delta) for a, b in zip(ordered, shuffled))
    pairs = make_pair_dataset(ordered)
    assert len(pairs.target) == len(blocks) - 1
