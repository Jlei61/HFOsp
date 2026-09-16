from __future__ import annotations

import json

import pytest
import torch

from src.topic5_group_event_state.v037 import (
    CheckpointEntry,
    DiagonalEventCTSSM,
    DualStreamEventCTSSM,
    GridBackgroundCTSSM,
    dual_stream_features_at_queries,
    VisibilityStamp,
    affine_associative_scan,
    affine_sequential_scan,
    assert_causal_visibility,
    fixed_mark_ewma,
    mark_ewma_features_at_queries,
    update_checkpoint_registry,
    zoh_diagonal_step,
)


def test_background_ctssm_scan_parity_and_missing_frame_invariance() -> None:
    torch.manual_seed(9)
    model = GridBackgroundCTSSM(3, taus_seconds=(600.0, 7200.0), channels_per_tau=2)
    time = torch.tensor([0.0, 300.0, 600.0, 1200.0])
    value = torch.randn(4, 3, requires_grad=True)
    available = torch.tensor([1.0, 1.0, 0.0, 1.0])
    parallel = model(time, value, available, scan="associative")
    serial = model(time, value, available, scan="sequential")
    assert torch.allclose(parallel.features, serial.features, atol=2e-6, rtol=2e-6)
    # A missing frame's value must be irrelevant.
    changed = value.detach().clone(); changed[2] = 1e6
    missing_changed = model(time, changed, available)
    assert torch.allclose(parallel.features.detach(), missing_changed.features, atol=2e-6, rtol=2e-6)


def test_background_ctssm_density_does_not_change_composition() -> None:
    model = GridBackgroundCTSSM(1, taus_seconds=(3600.0,), channels_per_tau=1)
    with torch.no_grad():
        model.input.weight.fill_(2.0)
    sparse = model(torch.tensor([0.0, 3600.0]), torch.ones(2, 1), torch.ones(2))
    dense = model(torch.arange(0.0, 3600.1, 300.0), torch.ones(13, 1), torch.ones(13))
    assert torch.allclose(sparse.composition[-1], torch.tensor([[2.0]]), atol=1e-5)
    assert torch.allclose(dense.composition[-1], torch.tensor([[2.0]]), atol=1e-5)


def test_affine_associative_scan_matches_sequential_and_gradients() -> None:
    torch.manual_seed(7)
    phi_a = (0.7 + 0.29 * torch.rand(31, 3, 8)).requires_grad_()
    q_a = torch.randn(31, 3, 8, requires_grad=True)
    phi_b = phi_a.detach().clone().requires_grad_()
    q_b = q_a.detach().clone().requires_grad_()
    pa, qa = affine_associative_scan(phi_a, q_a)
    pb, qb = affine_sequential_scan(phi_b, q_b)
    assert torch.allclose(pa, pb, atol=2e-6, rtol=2e-6)
    assert torch.allclose(qa, qb, atol=3e-6, rtol=3e-6)
    (qa.square().mean() + pa.mean()).backward()
    (qb.square().mean() + pb.mean()).backward()
    assert torch.allclose(phi_a.grad, phi_b.grad, atol=2e-5, rtol=2e-5)
    assert torch.allclose(q_a.grad, q_b.grad, atol=2e-5, rtol=2e-5)


def test_null_event_does_not_erase_physical_time_history() -> None:
    model = DiagonalEventCTSSM(1, 1, taus_seconds=(600.0, 3600.0), channels_per_tau=1)
    with torch.no_grad():
        model.burden_input.weight.fill_(1.0)
        model.grammar_input.weight.zero_()
    base = model(
        torch.tensor([0.0, 1200.0]),
        torch.tensor([[2.0], [0.0]]),
        torch.zeros(2, 1),
        scan="associative",
    ).post_state[-1]
    inserted = model(
        torch.tensor([0.0, 300.0, 1200.0]),
        torch.tensor([[2.0], [0.0], [0.0]]),
        torch.zeros(3, 1),
        scan="associative",
    ).post_state[-1]
    assert torch.allclose(base, inserted, atol=1e-6, rtol=1e-6)


def test_physical_time_constant_is_invariant_to_zero_event_density() -> None:
    model = DiagonalEventCTSSM(1, 1, taus_seconds=(7200.0,), channels_per_tau=1)
    with torch.no_grad():
        model.burden_input.weight.fill_(1.0)
        model.grammar_input.weight.zero_()
    sparse = model(
        torch.tensor([0.0, 7200.0]),
        torch.tensor([[1.0], [0.0]]),
        torch.zeros(2, 1),
    ).post_state[-1]
    dense_times = torch.linspace(0.0, 7200.0, 102)
    dense_mark = torch.zeros(102, 1)
    dense_mark[0] = 1.0
    dense = model(dense_times, dense_mark, torch.zeros_like(dense_mark)).post_state[-1]
    assert torch.allclose(sparse, dense, atol=1e-6, rtol=1e-6)
    assert torch.allclose(sparse, torch.exp(torch.tensor(-1.0))[None], atol=1e-6)


def test_first_event_receives_gradient_with_nonzero_input_path() -> None:
    torch.manual_seed(1)
    model = DiagonalEventCTSSM(2, 3, taus_seconds=(600.0, 7200.0), channels_per_tau=2)
    burden = torch.randn(40, 2, requires_grad=True)
    grammar = torch.randn(40, 3, requires_grad=True)
    output = model(torch.linspace(0.0, 6.0 * 3600.0, 40), burden, grammar)
    readout = torch.randn(model.state_dim)
    loss = (output.post_state[-1] * readout).sum()
    loss.backward()
    assert burden.grad is not None and float(burden.grad[0].norm()) > 0.0
    assert grammar.grad is not None and float(grammar.grad[0].norm()) > 0.0


def test_pre_event_state_excludes_current_event() -> None:
    model = DiagonalEventCTSSM(1, 1, taus_seconds=(3600.0,), channels_per_tau=1)
    with torch.no_grad():
        model.burden_input.weight.fill_(1.0)
        model.grammar_input.weight.zero_()
    a = model(torch.tensor([0.0, 10.0]), torch.tensor([[1.0], [2.0]]), torch.zeros(2, 1))
    b = model(torch.tensor([0.0, 10.0]), torch.tensor([[1.0], [99.0]]), torch.zeros(2, 1))
    assert torch.allclose(a.pre_state[1], b.pre_state[1])
    assert not torch.allclose(a.post_state[1], b.post_state[1])


def test_fixed_mark_ewma_normalises_grammar_but_reports_mass() -> None:
    one = fixed_mark_ewma(
        torch.tensor([0.0]), torch.tensor([[1.0]]), torch.tensor([[0.25, 0.75]]), taus_seconds=(3600.0,)
    )
    two = fixed_mark_ewma(
        torch.tensor([0.0, 0.0]),
        torch.tensor([[1.0], [1.0]]),
        torch.tensor([[0.25, 0.75], [0.25, 0.75]]),
        taus_seconds=(3600.0,),
    )
    assert torch.allclose(one.grammar_composition[-1], two.grammar_composition[-1])
    assert float(two.grammar_effective_mass[-1]) == pytest.approx(2.0)
    assert float(two.burden_sum[-1]) == pytest.approx(2.0)


def test_dual_stream_grammar_composition_is_density_invariant() -> None:
    model = DualStreamEventCTSSM(
        1, 2, taus_seconds=(3600.0,), burden_channels_per_tau=1, grammar_channels_per_tau=2
    )
    with torch.no_grad():
        model.burden_input.weight.fill_(1.0)
        model.grammar_input.weight.copy_(torch.eye(2))
    one = model(
        torch.tensor([0.0]), torch.ones(1, 1), torch.tensor([[0.25, 0.75]])
    )
    many = model(
        torch.zeros(7), torch.ones(7, 1), torch.tensor([[0.25, 0.75]]).repeat(7, 1)
    )
    assert torch.allclose(
        one.post_grammar_composition[-1], many.post_grammar_composition[-1], atol=1e-6
    )
    assert float(many.post_grammar_mass[-1]) == pytest.approx(7.0)
    assert float(many.post_burden[-1]) == pytest.approx(7.0)


def test_dual_stream_pre_state_cannot_see_current_mark() -> None:
    torch.manual_seed(3)
    model = DualStreamEventCTSSM(2, 3, taus_seconds=(600.0, 7200.0))
    times = torch.tensor([0.0, 600.0])
    burden = torch.randn(2, 2)
    grammar = torch.randn(2, 3)
    changed_burden = burden.clone(); changed_burden[1] += 100.0
    changed_grammar = grammar.clone(); changed_grammar[1] -= 100.0
    left = model(times, burden, grammar)
    right = model(times, changed_burden, changed_grammar)
    assert torch.allclose(left.pre_features[1], right.pre_features[1])
    assert not torch.allclose(left.post_features[1], right.post_features[1])


def test_dual_stream_null_event_insertion_is_invariant() -> None:
    torch.manual_seed(4)
    model = DualStreamEventCTSSM(1, 2, taus_seconds=(600.0, 3600.0))
    base = model(
        torch.tensor([0.0, 1200.0]),
        torch.tensor([[2.0], [0.0]]),
        torch.tensor([[0.2, 0.8], [0.0, 0.0]]),
        event_weight=torch.tensor([1.0, 0.0]),
    )
    inserted = model(
        torch.tensor([0.0, 300.0, 1200.0]),
        torch.tensor([[2.0], [99.0], [0.0]]),
        torch.tensor([[0.2, 0.8], [99.0, -99.0], [0.0, 0.0]]),
        event_weight=torch.tensor([1.0, 0.0, 0.0]),
    )
    assert torch.allclose(base.post_features[-1], inserted.post_features[-1], atol=1e-6)


def test_dual_stream_query_is_causal_and_physically_decayed() -> None:
    model = DualStreamEventCTSSM(
        1, 1, taus_seconds=(100.0,), burden_channels_per_tau=1, grammar_channels_per_tau=1
    )
    with torch.no_grad():
        model.burden_input.weight.fill_(1.0)
        model.grammar_input.weight.fill_(1.0)
    times = torch.tensor([0.0, 100.0])
    output = model(times, torch.tensor([[1.0], [9.0]]), torch.tensor([[0.25], [7.0]]))
    query = dual_stream_features_at_queries(output, times, torch.tensor([100.0, 200.0]))
    # At t=100 the current event is excluded; only the t=0 event survives.
    assert float(query[0, 0]) == pytest.approx(torch.exp(torch.tensor(-1.0)).item(), rel=1e-6)
    assert float(query[0, 1]) == pytest.approx(0.25, rel=1e-6)
    # At t=200 both events have already been observed.
    assert float(query[1, 0]) > float(query[0, 0])


def test_mark_ewma_query_excludes_equal_time_event() -> None:
    times = torch.tensor([0.0, 100.0])
    output = fixed_mark_ewma(
        times,
        torch.tensor([[1.0], [9.0]]),
        torch.tensor([[0.25], [7.0]]),
        taus_seconds=(100.0,),
    )
    query = mark_ewma_features_at_queries(
        output, times, torch.tensor([100.0]), taus_seconds=(100.0,)
    )
    assert float(query[0, 0]) == pytest.approx(torch.exp(torch.tensor(-1.0)).item(), rel=1e-6)
    assert float(query[0, 1]) == pytest.approx(0.25, rel=1e-6)


def test_zoh_step_matches_constant_input_solution() -> None:
    state = torch.tensor([2.0, -1.0])
    value = torch.tensor([0.5, 0.25])
    tau = torch.tensor([10.0, 20.0])
    got = zoh_diagonal_step(state, value, 5.0, tau)
    phi = torch.exp(-torch.tensor(5.0) / tau)
    expected = phi * state + tau * (1.0 - phi) * value
    assert torch.allclose(got, expected, atol=1e-6)


def test_visibility_contract_rejects_current_or_future_event() -> None:
    assert_causal_visibility([VisibilityStamp("past", 9.9)], 10.0)
    with pytest.raises(ValueError, match="causal visibility failure"):
        assert_causal_visibility([VisibilityStamp("current", 10.0)], 10.0)
    assert_causal_visibility([VisibilityStamp("grid-left", 10.0)], 10.0, allow_equal=True)


def test_checkpoint_registry_is_versioned_immutable_and_interictal(tmp_path) -> None:
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"weights")
    entry = CheckpointEntry(
        key="S_event::p1::seed1",
        model_family="S_event",
        state_semantics="observer",
        subject="p1",
        seed=1,
        checkpoint_path=str(checkpoint),
        maximum_training_time=100.0,
        input_streams=("group_event",),
        objectives=("future_burden", "conditional_grammar"),
        code_commit="deadbeef",
        normalization_provenance="train-only",
    )
    path = tmp_path / "registry.json"
    update_checkpoint_registry(path, entry)
    update_checkpoint_registry(path, entry)
    payload = json.loads(path.read_text())
    assert payload["architecture_version"] == "0.3.7"
    assert len(payload["entries"]) == 1
    bad = CheckpointEntry(**{**entry.__dict__, "sealed_partition_opened": True})
    with pytest.raises(ValueError, match="downstream/sealed"):
        update_checkpoint_registry(path, bad)
