from itertools import combinations

import numpy as np
import pytest
import torch

from src.topic5_shared_scaffold_rnn import (
    OrdinaryDenseGRUBaseline,
    PropagationRestraintState,
    SharedScaffoldPropagationRNN,
    batched_exact_conditional_k_subset_log_probability,
    build_fixed_local_shaft_adjacency,
    cardinality_log_probability,
    decomposed_one_step_log_probability,
    exact_conditional_k_subset_log_probability,
    source_conditioned_shared_scaffold,
)


def _fixed_graph(n_contacts: int = 5) -> np.ndarray:
    graph = np.zeros((n_contacts, n_contacts), dtype=np.float32)
    for index in range(n_contacts - 1):
        graph[index, index + 1] = 1.0
        graph[index + 1, index] = 1.0
    return graph


def _monotone_axis_model(n_contacts: int = 5) -> SharedScaffoldPropagationRNN:
    """Structured model whose axis is exactly reversal-antisymmetric.

    The chain graph is invariant under contact reversal and the coordinate
    flips sign under it, so the two endpoint sources must produce exactly
    mirrored drives.
    """

    model = SharedScaffoldPropagationRNN(
        fixed_adjacency=_fixed_graph(n_contacts),
        participation_bias=np.zeros(n_contacts),
        low_rank=2,
    )
    with torch.no_grad():
        model.axis_coordinate_raw.copy_(torch.linspace(-2.0, 2.0, n_contacts))
    return model


def _one_hot(index: int, n_contacts: int = 5) -> torch.Tensor:
    vector = torch.zeros(n_contacts, dtype=torch.bool)
    vector[index] = True
    return vector


def _rank_sets() -> list[torch.Tensor]:
    return [
        torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool),
        torch.tensor([0, 0, 1, 0, 0], dtype=torch.bool),
        torch.tensor([0, 0, 0, 1, 1], dtype=torch.bool),
    ]


def test_fixed_local_shaft_graph_is_symmetric_and_connects_shaft_neighbors():
    graph = build_fixed_local_shaft_adjacency(
        channel_names=["A1", "A2", "A3", "B1", "B2"]
    )
    np.testing.assert_allclose(graph, graph.T)
    np.testing.assert_array_equal(np.diag(graph), 0.0)
    assert graph[0, 1] > 0 and graph[1, 2] > 0
    assert graph[3, 4] > 0
    assert graph[0, 3] == 0


def test_geometry_and_shaft_contributions_are_fixed_and_target_free():
    coords = np.asarray(
        [[0, 0, 0], [1, 0, 0], [4, 0, 0], [5, 0, 0]], dtype=float
    )
    graph = build_fixed_local_shaft_adjacency(
        coords=coords,
        channel_names=["A1", "A2", "B1", "B2"],
        distance_scale=1.0,
    )
    assert graph[0, 1] > graph[0, 2]
    assert graph[2, 3] > graph[0, 2]
    np.testing.assert_allclose(graph, graph.T)


def _scaffold(coordinate: torch.Tensor) -> dict[str, torch.Tensor]:
    return source_conditioned_shared_scaffold(
        torch.as_tensor(_fixed_graph(), dtype=torch.float64),
        coordinate,
        endpoint_temperature=1.0,
        gamma=0.6,
        gain=1.4,
    )


def test_symmetric_operator_is_symmetric_and_skew_operator_is_antisymmetric():
    output = _scaffold(torch.linspace(-1.5, 1.5, 5, dtype=torch.float64))
    torch.testing.assert_close(output["W"], output["W"].T)
    torch.testing.assert_close(output["W_skew"], -output["W_skew"].T)
    torch.testing.assert_close(output["A"], output["A"].T)
    # The seed-ensemble mean of W and the diffusion source pools both rely on
    # W staying a non-negative symmetric graph.
    assert torch.all(output["W"] >= 0)
    assert torch.all(output["K_axis_symmetric"] >= 0)
    # Only the skew term may be signed; that is what lets one scaffold push
    # in two directions instead of learning two operators.
    assert bool(torch.any(output["K_axis_skew"] < 0))


def test_both_axis_operators_are_analytically_rank_at_most_two():
    output = _scaffold(torch.tensor([-1.7, -0.4, 0.1, 0.9, 1.6], dtype=torch.float64))
    assert int(torch.linalg.matrix_rank(output["K_axis_symmetric"])) <= 2
    assert int(torch.linalg.matrix_rank(output["K_axis_skew"])) <= 2
    # Both come from the same two endpoint memberships, not from two
    # independently parameterized forward/reverse paths.
    minus, plus = output["endpoint_minus"], output["endpoint_plus"]
    outer = plus[:, None] * minus[None, :]
    torch.testing.assert_close(
        output["K_axis_symmetric"] * torch.linalg.matrix_norm(outer + outer.T),
        outer + outer.T,
    )
    torch.testing.assert_close(
        output["K_axis_skew"] * torch.linalg.matrix_norm(outer - outer.T),
        outer - outer.T,
    )


def test_participation_bias_is_fixed_and_node_readout_has_no_dense_bypass():
    bias = np.linspace(-1.0, 1.0, 5, dtype=np.float32)
    model = SharedScaffoldPropagationRNN(
        fixed_adjacency=_fixed_graph(), participation_bias=bias, low_rank=2
    )
    assert "participation_bias" in dict(model.named_buffers())
    assert "participation_bias" not in dict(model.named_parameters())
    # The only learned contact-indexed latent object is one signed axis.  No
    # N x N parameter and no dense contact decoder or mixer exists here.
    assert model.axis_coordinate_raw.shape == (5,)
    assert not any(parameter.shape == (5, 5) for parameter in model.parameters())
    assert not hasattr(model, "contact_decoder")
    # STOP and cardinality read three permutation-invariant summaries, so no
    # head can name a contact even though the cardinality support happens to
    # have the same width as the contact axis.
    assert model.stop_head.in_features == 3
    assert model.cardinality_head.in_features == 3

    state = PropagationRestraintState(
        propagation=torch.linspace(0.0, 0.4, 5),
        restraint=torch.linspace(0.4, 0.0, 5),
        direction=torch.tensor(0.3),
        source_initialized=torch.tensor(True),
    )
    seen = torch.tensor([1, 0, 0, 0, 0], dtype=torch.bool)
    decision = model.decision(state, seen)
    expected = (
        model.participation_bias
        + model.propagation_weight * state.propagation
        - model.restraint_weight * state.restraint
    )
    torch.testing.assert_close(decision["raw_node_logits"], expected)
    assert torch.isneginf(decision["node_logits"][0])


def test_propagation_carries_signed_flow_and_restraint_stays_symmetric():
    model = _monotone_axis_model()
    assert 0 < float(model.rho_p) < float(model.rho_r) < 1
    components = model.operator_components()
    symmetric, skew = components["W"], components["W_skew"]

    first_set = _one_hot(0)
    first = model.observe(model.reset_state(), first_set)
    symmetric_drive = symmetric @ first_set.float()
    skew_drive = skew @ first_set.float()
    torch.testing.assert_close(
        first.propagation,
        symmetric_drive + model.skew_gain * first.direction * skew_drive,
    )
    torch.testing.assert_close(first.restraint, symmetric_drive)

    second_set = _one_hot(1)
    second = model.observe(first, second_set)
    next_symmetric = symmetric @ second_set.float()
    next_skew = skew @ second_set.float()
    torch.testing.assert_close(
        second.propagation,
        model.rho_p * first.propagation
        + next_symmetric
        + model.skew_gain * first.direction * next_skew,
    )
    torch.testing.assert_close(
        second.restraint, model.rho_r * first.restraint + next_symmetric
    )


def test_opposite_endpoint_sources_give_opposite_direction_and_mirrored_flow():
    model = _monotone_axis_model()
    minus_end = model.observe(model.reset_state(), _one_hot(0))
    plus_end = model.observe(model.reset_state(), _one_hot(4))
    assert float(minus_end.direction) > 0.5
    torch.testing.assert_close(minus_end.direction, -plus_end.direction)
    # Chain graph and coordinate are both reversal-antisymmetric, so the two
    # endpoint drives must be exact mirror images of one another.
    torch.testing.assert_close(minus_end.propagation.flip(0), plus_end.propagation)
    torch.testing.assert_close(minus_end.restraint.flip(0), plus_end.restraint)
    # A direction-blind model would make these identical; the skew term is
    # what separates propagation from restraint.
    assert float((minus_end.propagation - minus_end.restraint).abs().max()) > 1e-3


def test_flipping_the_learned_axis_sign_leaves_the_event_likelihood_unchanged():
    """One scaffold expresses both directions, so its sign carries no label."""

    model = _monotone_axis_model()
    before = model.event_log_likelihood(_rank_sets())["total"]
    with torch.no_grad():
        model.axis_coordinate_raw.mul_(-1.0)
    after = model.event_log_likelihood(_rank_sets())["total"]
    torch.testing.assert_close(before, after)


def test_direction_is_written_by_the_first_rank_set_and_then_frozen():
    model = _monotone_axis_model()
    initial = model.reset_state()
    assert not bool(initial.source_initialized)
    torch.testing.assert_close(initial.direction, torch.tensor(0.0))

    first = model.observe(initial, _one_hot(0))
    assert bool(first.source_initialized)
    assert float(first.direction) > 0.5
    # Later rank steps recruit the opposite endpoint; the frozen direction
    # must not follow them.
    second = model.observe(first, _one_hot(4))
    third = model.observe(second, _one_hot(3))
    torch.testing.assert_close(second.direction, first.direction)
    torch.testing.assert_close(third.direction, first.direction)
    # A fresh event starting from that opposite endpoint must be free to take
    # the other sign, which proves the freeze is per event and not global.
    reversed_event = model.observe(model.reset_state(), _one_hot(4))
    torch.testing.assert_close(reversed_event.direction, -first.direction)


def test_reset_state_clears_every_state_field_in_scalar_and_batched_mode():
    model = _monotone_axis_model()
    used = model.observe(model.reset_state(), _one_hot(0))
    assert float(used.propagation.abs().max()) > 0
    scalar = model.reset_state()
    assert scalar.propagation.shape == (5,) and scalar.direction.shape == ()
    assert not bool(scalar.source_initialized)
    torch.testing.assert_close(scalar.propagation, torch.zeros(5))
    torch.testing.assert_close(scalar.restraint, torch.zeros(5))
    torch.testing.assert_close(scalar.direction, torch.tensor(0.0))

    batched = model.reset_state(batch_size=3)
    assert batched.propagation.shape == (3, 5) and batched.direction.shape == (3,)
    torch.testing.assert_close(batched.propagation, torch.zeros(3, 5))
    torch.testing.assert_close(batched.restraint, torch.zeros(3, 5))
    torch.testing.assert_close(batched.direction, torch.zeros(3))
    assert not bool(batched.source_initialized.any())


def test_terminated_batch_rows_keep_their_frozen_direction():
    model = _monotone_axis_model()
    state = model.reset_state(batch_size=2)
    first = torch.stack([_one_hot(0), _one_hot(4)])
    state = model.observe(state, first, active=torch.tensor([True, True]))
    frozen = state.direction.clone()
    assert float(frozen[0]) > 0.5 and float(frozen[1]) < -0.5
    # Row one has terminated; its state and direction must not move even
    # though the batched call still supplies a current set for it.
    state = model.observe(
        state,
        torch.stack([_one_hot(1), _one_hot(3)]),
        active=torch.tensor([True, False]),
    )
    torch.testing.assert_close(state.direction, frozen)
    assert bool(state.source_initialized.all())


def test_exact_conditional_k_subset_matches_enumeration_and_normalizes():
    logits = torch.tensor([-0.7, 0.2, 1.1, -0.1], dtype=torch.float64)
    eligible = torch.tensor([1, 1, 1, 1], dtype=torch.bool)
    observed = {}
    for subset in combinations(range(4), 2):
        target = torch.zeros(4, dtype=torch.bool)
        target[list(subset)] = True
        observed[subset] = torch.exp(
            exact_conditional_k_subset_log_probability(
                node_logits=logits,
                eligible=eligible,
                next_set=target,
            )
        )
    scores = torch.stack([logits[list(subset)].sum() for subset in observed])
    expected = torch.softmax(scores, dim=0)
    torch.testing.assert_close(torch.stack(list(observed.values())), expected)
    torch.testing.assert_close(
        torch.stack(list(observed.values())).sum(), torch.tensor(1.0, dtype=torch.float64)
    )


def test_exact_subset_likelihood_respects_eligibility_and_has_gradients():
    logits = torch.tensor([0.3, -0.4, 0.8, 1.5], requires_grad=True)
    eligible = torch.tensor([1, 1, 1, 0], dtype=torch.bool)
    target = torch.tensor([1, 0, 1, 0], dtype=torch.bool)
    value = exact_conditional_k_subset_log_probability(
        node_logits=logits, eligible=eligible, next_set=target
    )
    value.backward()
    assert torch.isfinite(value)
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[3] == 0
    with pytest.raises(ValueError, match="ineligible"):
        exact_conditional_k_subset_log_probability(
            node_logits=logits.detach(),
            eligible=eligible,
            next_set=torch.tensor([0, 0, 0, 1], dtype=torch.bool),
        )


def test_batched_exact_subset_likelihood_matches_scalar_rows():
    logits = torch.tensor(
        [[0.3, -0.4, 0.8, -1.2], [-0.1, 0.2, 0.9, 0.4]],
        dtype=torch.float64,
    )
    eligible = torch.tensor(
        [[1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool
    )
    target = torch.tensor(
        [[1, 0, 1, 0], [0, 0, 1, 0]], dtype=torch.bool
    )
    observed = batched_exact_conditional_k_subset_log_probability(
        node_logits=logits, eligible=eligible, next_set=target
    )
    expected = torch.stack(
        [
            exact_conditional_k_subset_log_probability(
                node_logits=logits[index],
                eligible=eligible[index],
                next_set=target[index],
            )
            for index in range(2)
        ]
    )
    torch.testing.assert_close(observed, expected)


def test_stop_cardinality_and_contact_identity_are_separate_scores():
    base = {
        "node_logits": torch.tensor([0.2, 0.7, -0.4, -torch.inf]),
        "eligible": torch.tensor([1, 1, 1, 0], dtype=torch.bool),
        "stop_logit": torch.tensor(-0.3),
        "cardinality_logits": torch.tensor([-0.2, 0.9, 0.1, -torch.inf]),
    }
    target = torch.tensor([1, 1, 0, 0], dtype=torch.bool)
    first = decomposed_one_step_log_probability(
        base, next_set=target, terminal=False
    )
    changed_stop = dict(base, stop_logit=torch.tensor(2.0))
    second = decomposed_one_step_log_probability(
        changed_stop, next_set=target, terminal=False
    )
    torch.testing.assert_close(first.cardinality, second.cardinality)
    torch.testing.assert_close(first.conditional_contacts, second.conditional_contacts)
    assert first.stop != second.stop
    torch.testing.assert_close(
        first.total,
        first.stop + first.cardinality + first.conditional_contacts,
    )

    terminal = decomposed_one_step_log_probability(
        base, next_set=torch.zeros(4, dtype=torch.bool), terminal=True
    )
    assert terminal.cardinality_target == 0
    assert terminal.cardinality == 0
    assert terminal.conditional_contacts == 0
    torch.testing.assert_close(terminal.total, terminal.stop)


def test_cardinality_head_is_normalized_only_over_feasible_sizes():
    logits = torch.tensor([0.0, 1.0, -torch.inf, -torch.inf])
    first = torch.exp(cardinality_log_probability(logits, 1))
    second = torch.exp(cardinality_log_probability(logits, 2))
    torch.testing.assert_close(first + second, torch.tensor(1.0))
    with pytest.raises(ValueError, match="ineligible"):
        cardinality_log_probability(logits, 3)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SharedScaffoldPropagationRNN(
            fixed_adjacency=_fixed_graph(),
            participation_bias=np.zeros(5),
            low_rank=2,
        ),
        lambda: OrdinaryDenseGRUBaseline(
            participation_bias=np.zeros(5), hidden_size=7
        ),
    ],
)
def test_structured_and_ordinary_models_share_event_scoring_interface(factory):
    model = factory()
    first = model.event_log_likelihood(_rank_sets())
    second = model.event_log_likelihood(_rank_sets())
    assert set(first) == {
        "total", "stop", "cardinality", "conditional_contacts"
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in first.values())
    for key in first:
        torch.testing.assert_close(first[key], second[key])
    torch.testing.assert_close(
        first["total"],
        first["stop"] + first["cardinality"] + first["conditional_contacts"],
    )


def test_multicontact_rank_sets_backpropagate_through_structured_model():
    model = SharedScaffoldPropagationRNN(
        fixed_adjacency=_fixed_graph(),
        participation_bias=np.zeros(5),
        low_rank=2,
    )
    nll = model.event_nll(_rank_sets())
    nll["total"].backward()
    for name in ("axis_coordinate_raw", "gamma_raw", "skew_gain_raw", "direction_gain_raw"):
        gradient = getattr(model, name).grad
        assert gradient is not None and torch.isfinite(gradient).all()


def test_initialization_does_not_cancel_propagation_against_restraint():
    """v0.2 started with equal weights, so the first rank step cancelled."""

    model = _monotone_axis_model()
    assert float(model.propagation_weight) == pytest.approx(1.0, abs=1e-4)
    assert float(model.restraint_weight) == pytest.approx(0.25, abs=1e-4)
    assert float(model.skew_gain) == pytest.approx(0.5, abs=1e-4)
    assert float(model.direction_gain) == pytest.approx(2.0, abs=1e-4)

    state = model.observe(model.reset_state(), _one_hot(0))
    drive = (
        model.propagation_weight * state.propagation
        - model.restraint_weight * state.restraint
    )
    # Both traces still carry the same symmetric drive at the first step, so
    # equal weights would leave only the skew residual here.
    assert float(drive.abs().max()) > 0.5 * float(state.restraint.abs().max())


def test_group_id_helper_preserves_tied_rank_sets():
    model = SharedScaffoldPropagationRNN(
        fixed_adjacency=_fixed_graph(),
        participation_bias=np.zeros(5),
        low_rank=2,
    )
    scores = model.score_group_ids(np.asarray([0, 0, 1, 2, 2]))
    assert all(torch.isfinite(value) for value in scores.values())


def _batched_groups():
    group_ids = torch.tensor(
        [
            [0, 0, 1, 2, 2],
            [0, 1, -1, 1, 2],
            [0, -1, -1, -1, -1],
        ],
        dtype=torch.long,
    )
    return group_ids, torch.tensor([3, 3, 1], dtype=torch.long)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SharedScaffoldPropagationRNN(
            fixed_adjacency=_fixed_graph(),
            participation_bias=np.linspace(-0.4, 0.4, 5),
            low_rank=2,
        ),
        lambda: OrdinaryDenseGRUBaseline(
            participation_bias=np.linspace(-0.4, 0.4, 5), hidden_size=7
        ),
    ],
)
def test_batched_event_scores_match_single_event_reference(factory):
    torch.manual_seed(12)
    model = factory()
    groups, counts = _batched_groups()
    batched = model.batched_event_log_likelihood(groups, counts)
    for event_index in range(groups.shape[0]):
        single = model.score_group_ids(groups[event_index].numpy())
        for field in ("total", "stop", "cardinality", "conditional_contacts"):
            torch.testing.assert_close(
                batched[field][event_index], single[field], atol=1e-6, rtol=1e-6
            )
    torch.testing.assert_close(
        batched["decision_count"], counts
    )
    torch.testing.assert_close(
        batched["nonterminal_decision_count"], torch.clamp(counts - 1, min=0)
    )


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SharedScaffoldPropagationRNN(
            fixed_adjacency=_fixed_graph(),
            participation_bias=np.zeros(5),
            low_rank=2,
        ),
        lambda: OrdinaryDenseGRUBaseline(
            participation_bias=np.zeros(5), hidden_size=7
        ),
    ],
)
def test_batched_event_first_loss_is_finite_and_differentiable(factory):
    model = factory()
    groups, counts = _batched_groups()
    loss = model.batched_event_nll(groups, counts, reduction="event_first")
    assert set(loss) == {
        "total", "stop", "cardinality", "conditional_contacts"
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in loss.values())
    loss["total"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        gradient is None or torch.isfinite(gradient).all() for gradient in gradients
    )


def test_batched_group_contract_rejects_noncontiguous_rank_ids():
    model = SharedScaffoldPropagationRNN(
        fixed_adjacency=_fixed_graph(),
        participation_bias=np.zeros(5),
        low_rank=2,
    )
    with pytest.raises(ValueError, match="contiguous"):
        model.batched_event_log_likelihood(
            torch.tensor([[0, 2, -1, -1, -1]]), torch.tensor([3])
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_batched_structured_cpu_gpu_scores_agree():
    torch.manual_seed(31)
    cpu = SharedScaffoldPropagationRNN(
        fixed_adjacency=_fixed_graph(),
        participation_bias=np.linspace(-0.2, 0.2, 5),
        low_rank=2,
    )
    gpu = SharedScaffoldPropagationRNN(
        fixed_adjacency=_fixed_graph(),
        participation_bias=np.linspace(-0.2, 0.2, 5),
        low_rank=2,
    ).cuda()
    gpu.load_state_dict(cpu.state_dict())
    groups, counts = _batched_groups()
    cpu_value = cpu.batched_event_nll(groups, counts)
    gpu_value = gpu.batched_event_nll(groups.cuda(), counts.cuda())
    for field in cpu_value:
        torch.testing.assert_close(
            cpu_value[field], gpu_value[field].cpu(), atol=2e-6, rtol=2e-6
        )
