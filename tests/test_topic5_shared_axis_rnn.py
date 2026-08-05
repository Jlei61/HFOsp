import numpy as np
import pytest
import torch

from src.topic5_shared_axis_rnn import (
    AxisPropagationState,
    SharedAxisPropagationRNN,
    axis_smoothness_penalty,
    build_fixed_local_shaft_adjacency,
    continuous_axis_operators,
)


def _chain(n_contacts: int = 6) -> np.ndarray:
    graph = np.zeros((n_contacts, n_contacts), dtype=np.float32)
    for index in range(n_contacts - 1):
        graph[index, index + 1] = graph[index + 1, index] = 1.0
    return graph


def _model(n_contacts: int = 6, monotone: bool = True) -> SharedAxisPropagationRNN:
    model = SharedAxisPropagationRNN(
        fixed_adjacency=_chain(n_contacts),
        participation_bias=np.zeros(n_contacts),
    )
    if monotone:
        with torch.no_grad():
            model.axis_coordinate_raw.copy_(torch.linspace(-2.0, 2.0, n_contacts))
    return model


def _one_hot(index: int, n_contacts: int = 6) -> torch.Tensor:
    vector = torch.zeros(n_contacts, dtype=torch.bool)
    vector[index] = True
    return vector


def _rank_sets(n_contacts: int = 6):
    first = torch.zeros(n_contacts, dtype=torch.bool)
    first[0] = True
    second = torch.zeros(n_contacts, dtype=torch.bool)
    second[1] = second[2] = True
    third = torch.zeros(n_contacts, dtype=torch.bool)
    third[3] = True
    return [first, second, third]


# --------------------------------------------------------------- operators
def test_symmetric_operator_is_symmetric_and_flow_is_exactly_antisymmetric():
    output = continuous_axis_operators(
        torch.as_tensor(_chain(), dtype=torch.float64),
        torch.tensor([-2.0, -0.9, -0.1, 0.4, 1.2, 1.9], dtype=torch.float64),
        gamma=0.5,
        gain=1.0,
    )
    torch.testing.assert_close(output["W"], output["W"].T)
    torch.testing.assert_close(output["W_skew"], -output["W_skew"].T)
    torch.testing.assert_close(
        torch.diagonal(output["W_skew"]),
        torch.zeros(6, dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.diagonal(output["A"]), torch.zeros(6, dtype=torch.float64)
    )
    assert torch.all(output["W"] >= 0)


def test_flow_is_the_scaffold_times_an_odd_function_of_the_axis_gap():
    """The flow is not a second free connectivity."""

    coordinate = torch.tensor([-1.6, -0.5, 0.2, 0.7, 1.1, 1.8], dtype=torch.float64)
    output = continuous_axis_operators(
        torch.as_tensor(_chain(), dtype=torch.float64), coordinate,
        gamma=0.4, gain=1.3,
    )
    normalized = output["axis_coordinate"]
    expected = output["W"] * torch.tanh(
        (normalized[:, None] - normalized[None, :]) / float(output["delta"])
    )
    torch.testing.assert_close(output["W_skew"], expected)


def test_axis_coordinate_is_centred_and_unit_rms_so_raw_scale_cannot_matter():
    base = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    small = continuous_axis_operators(
        torch.as_tensor(_chain(), dtype=torch.float64), base, gamma=0.5, gain=1.0
    )
    large = continuous_axis_operators(
        torch.as_tensor(_chain(), dtype=torch.float64), base * 7.0, gamma=0.5, gain=1.0
    )
    torch.testing.assert_close(small["W"], large["W"])
    torch.testing.assert_close(small["W_skew"], large["W_skew"])
    coordinate = small["axis_coordinate"]
    assert float(coordinate.mean().abs()) < 1e-12
    assert float(torch.sqrt(torch.mean(coordinate.square()))) == pytest.approx(1.0, abs=1e-6)


def test_gamma_mixes_two_terms_that_were_first_put_on_a_common_norm():
    output = continuous_axis_operators(
        torch.as_tensor(_chain(), dtype=torch.float64),
        torch.linspace(-2.0, 2.0, 6, dtype=torch.float64),
        gamma=0.5, gain=1.0,
    )
    for key in ("K_local", "K_axis"):
        assert float(torch.linalg.matrix_norm(output[key])) == pytest.approx(1.0, abs=1e-9)


def test_smoothness_penalty_prefers_a_coordinate_that_follows_the_shaft():
    graph = torch.as_tensor(_chain(), dtype=torch.float64)
    smooth = axis_smoothness_penalty(
        torch.linspace(-2.0, 2.0, 6, dtype=torch.float64), graph, weight=1.0
    )
    jagged = axis_smoothness_penalty(
        torch.tensor([-2.0, 2.0, -1.5, 1.5, -1.0, 1.0], dtype=torch.float64),
        graph, weight=1.0,
    )
    assert float(smooth) < float(jagged)


# --------------------------------------------------------------- direction
def test_opposite_ends_give_opposite_direction_and_mirrored_drive():
    model = _model()
    negative_end = model.observe(model.reset_state(), _one_hot(0))
    positive_end = model.observe(model.reset_state(), _one_hot(5))
    assert float(negative_end.direction) > 0.5
    torch.testing.assert_close(negative_end.direction, -positive_end.direction)
    # Chain and coordinate are both reversal-antisymmetric.
    torch.testing.assert_close(
        negative_end.propagation.flip(0), positive_end.propagation
    )
    torch.testing.assert_close(negative_end.restraint.flip(0), positive_end.restraint)
    assert float((negative_end.propagation - negative_end.restraint).abs().max()) > 1e-4


def test_direction_is_written_by_the_first_rank_set_and_then_frozen():
    model = _model()
    initial = model.reset_state()
    assert not bool(initial.source_initialized)
    first = model.observe(initial, _one_hot(0))
    assert bool(first.source_initialized)
    second = model.observe(first, _one_hot(5))
    third = model.observe(second, _one_hot(4))
    torch.testing.assert_close(second.direction, first.direction)
    torch.testing.assert_close(third.direction, first.direction)
    fresh = model.observe(model.reset_state(), _one_hot(5))
    torch.testing.assert_close(fresh.direction, -first.direction)


def test_reset_clears_every_state_field():
    model = _model()
    used = model.observe(model.reset_state(), _one_hot(0))
    assert float(used.propagation.abs().max()) > 0
    for batch in (None, 4):
        state = model.reset_state(batch_size=batch)
        torch.testing.assert_close(state.propagation, torch.zeros_like(state.propagation))
        torch.testing.assert_close(state.restraint, torch.zeros_like(state.restraint))
        torch.testing.assert_close(state.direction, torch.zeros_like(state.direction))
        assert not bool(state.source_initialized.any())


def test_terminated_batch_rows_keep_their_frozen_direction():
    model = _model()
    state = model.reset_state(batch_size=2)
    state = model.observe(
        state, torch.stack([_one_hot(0), _one_hot(5)]),
        active=torch.tensor([True, True]),
    )
    frozen = state.direction.clone()
    assert float(frozen[0]) > 0.5 and float(frozen[1]) < -0.5
    state = model.observe(
        state, torch.stack([_one_hot(1), _one_hot(4)]),
        active=torch.tensor([True, False]),
    )
    torch.testing.assert_close(state.direction, frozen)


def test_flipping_the_axis_sign_swaps_the_direction_labels_but_not_the_prediction():
    """The coordinate's sign is not identifiable, so it must carry no information."""

    model = _model()
    before = model.event_log_likelihood(_rank_sets())["total"]
    with torch.no_grad():
        model.axis_coordinate_raw.mul_(-1.0)
    after = model.event_log_likelihood(_rank_sets())["total"]
    torch.testing.assert_close(before, after)


def test_rank_set_input_is_normalized_by_how_many_contacts_fired():
    model = _model()
    single = model.observe(model.reset_state(), _one_hot(2))
    pair = torch.zeros(6, dtype=torch.bool)
    pair[2] = pair[3] = True
    both = model.observe(model.reset_state(), pair)
    assert float(single.restraint.sum()) == pytest.approx(
        float(both.restraint.sum()), rel=0.35
    )


# ------------------------------------------------------------------ scoring
def test_no_dense_contact_bypass_exists():
    model = _model()
    assert not hasattr(model, "contact_decoder")
    assert not any(parameter.shape == (6, 6) for parameter in model.parameters())
    assert model.axis_coordinate_raw.shape == (6,)
    assert model.stop_head.in_features == 3
    assert model.cardinality_head.in_features == 3
    state = AxisPropagationState(
        propagation=torch.linspace(0.0, 0.5, 6),
        restraint=torch.linspace(0.5, 0.0, 6),
        direction=torch.tensor(0.4),
        source_initialized=torch.tensor(True),
    )
    seen = torch.zeros(6, dtype=torch.bool)
    seen[0] = True
    decision = model.decision(state, seen)
    torch.testing.assert_close(
        decision["raw_node_logits"],
        model.participation_bias
        + model.propagation_weight * state.propagation
        - model.restraint_weight * state.restraint,
    )
    assert torch.isneginf(decision["node_logits"][0])


def test_masked_nonparticipants_never_enter_the_contact_likelihood():
    model = _model()
    groups = torch.tensor([[0, 1, 1, 2, -1, -1]], dtype=torch.long)
    counts = torch.tensor([3], dtype=torch.long)
    value = model.batched_event_log_likelihood(groups, counts)
    assert torch.isfinite(value["total"]).all()
    # Contacts 4 and 5 never participate; a finite rank for them would have
    # been consumed as a real observation and changed the likelihood.
    other = torch.tensor([[0, 1, 1, 2, -1, -1]], dtype=torch.long)
    torch.testing.assert_close(
        model.batched_event_log_likelihood(other, counts)["total"], value["total"]
    )


def test_batched_scores_match_single_event_reference():
    torch.manual_seed(5)
    model = _model()
    groups = torch.tensor(
        [[0, 0, 1, 2, 2, -1], [0, 1, -1, 1, 2, -1], [0, -1, -1, -1, -1, -1]],
        dtype=torch.long,
    )
    counts = torch.tensor([3, 3, 1], dtype=torch.long)
    batched = model.batched_event_log_likelihood(groups, counts)
    for index in range(groups.shape[0]):
        single = model.score_group_ids(groups[index].numpy())
        for field in ("total", "stop", "cardinality", "conditional_contacts"):
            torch.testing.assert_close(
                batched[field][index], single[field], atol=1e-6, rtol=1e-6
            )


def test_loss_is_finite_and_reaches_the_axis_and_flow_parameters():
    model = _model()
    groups = torch.tensor([[0, 0, 1, 2, 2, -1], [0, 1, 2, -1, -1, -1]], dtype=torch.long)
    counts = torch.tensor([3, 3], dtype=torch.long)
    loss = model.batched_event_nll(groups, counts, reduction="event_first")["total"]
    (loss + model.smoothness_penalty()).backward()
    for name in ("axis_coordinate_raw", "gamma_raw", "gain_raw", "flow_weight_raw"):
        gradient = getattr(model, name).grad
        assert gradient is not None and torch.isfinite(gradient).all()


def test_checkpoint_reload_reproduces_the_operators():
    torch.manual_seed(9)
    model = _model(monotone=False)
    reloaded = SharedAxisPropagationRNN(
        fixed_adjacency=_chain(), participation_bias=np.zeros(6)
    )
    reloaded.load_state_dict(model.state_dict())
    for key in ("W", "W_skew", "axis_coordinate"):
        torch.testing.assert_close(
            model.operator_components()[key], reloaded.operator_components()[key]
        )


def test_shaft_graph_builder_is_reused_unchanged():
    graph = build_fixed_local_shaft_adjacency(channel_names=["A1", "A2", "A3", "B1"])
    np.testing.assert_allclose(graph, graph.T)
    assert graph[0, 1] > 0 and graph[0, 3] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_and_gpu_agree():
    torch.manual_seed(17)
    cpu = _model(monotone=False)
    gpu = SharedAxisPropagationRNN(
        fixed_adjacency=_chain(), participation_bias=np.zeros(6)
    ).cuda()
    gpu.load_state_dict(cpu.state_dict())
    groups = torch.tensor([[0, 0, 1, 2, 2, -1], [0, 1, 2, -1, -1, -1]], dtype=torch.long)
    counts = torch.tensor([3, 3], dtype=torch.long)
    cpu_value = cpu.batched_event_nll(groups, counts)
    gpu_value = gpu.batched_event_nll(groups.cuda(), counts.cuda())
    for field in cpu_value:
        torch.testing.assert_close(
            cpu_value[field], gpu_value[field].cpu(), atol=2e-6, rtol=2e-6
        )
