from itertools import combinations

import numpy as np
import torch

from src.topic5_shared_scaffold_rnn import (
    SharedScaffoldPropagationRNN,
    brute_force_conditional_subset_probabilities,
)
from src.topic5_shared_scaffold_rollout import (
    exact_conditional_k_subset_sample,
    rollout_from_source_pool,
)


def test_small_graph_exact_dp_sampling_matches_enumerated_distribution():
    logits = torch.tensor([[-0.7, 0.2, 1.1, -0.1]], dtype=torch.float64)
    eligible = torch.ones_like(logits, dtype=torch.bool)
    cardinality = torch.tensor([2])
    generator = torch.Generator().manual_seed(119)
    n_draws = 30_000
    observed = {subset: 0 for subset in combinations(range(4), 2)}
    for _ in range(n_draws // 500):
        selected = exact_conditional_k_subset_sample(
            node_logits=logits.expand(500, -1),
            eligible=eligible.expand(500, -1),
            cardinality=cardinality.expand(500),
            generator=generator,
        )
        for row in selected.numpy():
            observed[tuple(np.flatnonzero(row))] += 1
    expected = brute_force_conditional_subset_probabilities(
        logits[0], eligible[0], cardinality=2
    )
    for subset, probability in expected.items():
        assert abs(observed[subset] / n_draws - float(probability)) < 0.012


def test_exact_sampler_respects_row_specific_eligibility_and_cardinality():
    logits = torch.tensor([[0.2, -0.3, 0.7, 1.0], [0.5, -0.1, 0.1, 0.3]])
    eligible = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 1]], dtype=torch.bool)
    cardinality = torch.tensor([2, 1])
    selected = exact_conditional_k_subset_sample(
        node_logits=logits,
        eligible=eligible,
        cardinality=cardinality,
        generator=torch.Generator().manual_seed(7),
    )
    assert torch.equal(selected.sum(dim=1), cardinality)
    assert not torch.any(selected & ~eligible)


def test_opposite_source_pools_roll_out_in_opposite_axis_directions():
    """One frozen model must produce both directional fields, not one."""

    graph = np.zeros((7, 7), dtype=np.float32)
    for index in range(6):
        graph[index, index + 1] = graph[index + 1, index] = 1.0
    model = SharedScaffoldPropagationRNN(
        fixed_adjacency=graph,
        participation_bias=np.zeros(7, dtype=np.float32),
        low_rank=2,
    )
    with torch.no_grad():
        model.axis_coordinate_raw.copy_(torch.linspace(-2.0, 2.0, 7))
    axis = model.operator_components()["axis_coordinate"].detach().numpy()

    def earliness(source_index: int) -> np.ndarray:
        source = np.zeros(7, dtype=bool)
        source[source_index] = True
        result = rollout_from_source_pool(
            model,
            source_pool=source,
            horizon=6,
            n_rollouts=400,
            seed=17,
            batch_size=200,
        )
        horizon = result.first_arrival_mass.shape[0]
        weight = 1.0 - np.arange(1, horizon + 1) / horizon
        return weight @ result.first_arrival_mass

    from_minus_end = earliness(0)
    from_plus_end = earliness(6)
    # Earliness projected on the learned axis must point away from whichever
    # end started the event, so the two projections have opposite signs.
    assert float(axis @ from_minus_end) > 0.0
    assert float(axis @ from_plus_end) < 0.0


def test_source_rollout_saves_first_arrival_mass_and_is_deterministic():
    graph = np.zeros((5, 5), dtype=np.float32)
    for index in range(4):
        graph[index, index + 1] = graph[index + 1, index] = 1.0
    torch.manual_seed(3)
    model = SharedScaffoldPropagationRNN(
        fixed_adjacency=graph,
        participation_bias=np.linspace(-0.3, 0.3, 5, dtype=np.float32),
        low_rank=2,
    )
    kwargs = {
        "source_pool": np.asarray([1, 0, 0, 0, 0], dtype=bool),
        "horizon": 4,
        "n_rollouts": 200,
        "seed": 91,
        "batch_size": 64,
    }
    first = rollout_from_source_pool(model, **kwargs)
    second = rollout_from_source_pool(model, **kwargs)
    np.testing.assert_array_equal(first.event_group_ids, second.event_group_ids)
    np.testing.assert_array_equal(first.event_group_count, second.event_group_count)
    np.testing.assert_allclose(first.first_arrival_mass, second.first_arrival_mass)
    assert first.first_arrival_mass.shape == (4, 5)
    assert np.all(first.first_arrival_mass[:, 0] == 0.0)
    assert first.source_at_step_zero[0] == 1.0
    assert np.all(first.first_arrival_mass.sum(axis=0) <= 1.0 + 1e-12)
    np.testing.assert_allclose(
        first.cumulative_participation_post_source,
        first.first_arrival_mass.sum(axis=0),
    )
    assert first.stop_step_histogram.sum() == 200
