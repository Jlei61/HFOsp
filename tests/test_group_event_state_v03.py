from __future__ import annotations

import torch
import numpy as np

from src.topic5_continuous_marked_state_r1.mark_likelihood import tied_group_mark_log_prob
from src.topic5_group_event_state.v02.timeline import CoverageSegment
from src.topic5_group_event_state.v03.grammar import (
    build_train_only_grammar,
    factorized_size_log_prob,
)
from src.topic5_group_event_state.v03.partition import nested_time_partition
from src.topic5_group_event_state.v03.pilot import _physical_event_chunks
from src.topic5_group_event_state.v03.point_process import interval_point_process_terms
from src.topic5_group_event_state.v03.state import FixedTimescaleEventState, StateConfig
from src.topic5_rank_distribution import FullHistorySequenceGRU


def test_point_process_has_event_and_no_event_terms_and_masks_gaps():
    start = torch.tensor([0.2, 0.2, 0.2], requires_grad=True)
    event = torch.tensor([0.4, 0.4, 0.4], requires_grad=True)
    dt = torch.tensor([2.0, 5.0, 1000.0])
    terms = interval_point_process_terms(
        start, event, dt, torch.tensor([True, True, False])
    )
    assert torch.allclose(terms.survival_integral, torch.tensor([0.6, 1.5, 0.0]))
    assert terms.event_log_intensity[-1].item() == 0.0
    assert terms.observed_seconds.sum().item() == 7.0
    terms.event_nll.sum().backward()
    assert torch.isfinite(start.grad).all()
    assert torch.isfinite(event.grad).all()


def test_exact_tied_mark_supports_multicontact_group_and_stop():
    # Terminal step has zero eligible contacts; this used to create NaN
    # gradients through logsumexp(all -inf).
    group_ids = torch.tensor([[0, 0, 1, 1]])
    group_count = torch.tensor([2])
    size = torch.zeros(1, 3, 5, requires_grad=True)
    contact = torch.zeros(1, 3, 4, requires_grad=True)
    terms = tied_group_mark_log_prob(group_ids, group_count, size, contact)
    assert terms.active_step.tolist() == [[True, True, True]]
    assert terms.select_step.tolist() == [[True, True, False]]
    (-terms.event_log_prob.mean()).backward()
    assert torch.isfinite(size.grad).all()
    assert torch.isfinite(contact.grad).all()


def test_fixed_timescale_state_uses_real_seconds_and_event_updates_after_flow():
    cfg = StateConfig(taus_seconds=(10.0, 100.0), channels_per_tau=1, event_dim=3)
    model = FixedTimescaleEventState(cfg)
    with torch.no_grad():
        model.mean.zero_()
        model.initial_offset.fill_(1.0)
    initial = model.initial(1, "cpu")
    flowed = model.evolve(initial, torch.tensor([10.0]))
    assert torch.allclose(flowed[0, 0], torch.exp(torch.tensor(-1.0)), atol=1e-6)
    assert flowed[0, 1] > flowed[0, 0]
    post = model.update(flowed, torch.ones(1, 3))
    assert post.shape == flowed.shape
    assert torch.isfinite(model.intensity(flowed)).all()


def test_repeated_event_updates_remain_bounded_and_slower_modes_move_less():
    torch.manual_seed(3)
    cfg = StateConfig(taus_seconds=(10.0, 1000.0), channels_per_tau=1, event_dim=3)
    model = FixedTimescaleEventState(cfg)
    with torch.no_grad():
        model.update_net[-1].weight.zero_()
        model.update_net[-1].bias[:2].zero_()
        model.update_net[-1].bias[2:].fill_(1.0)
    state = model.initial(1, "cpu")
    event = torch.ones(1, 3)
    first = model.update(state, event)
    assert first[0, 0].abs() > first[0, 1].abs()
    state = first
    for _ in range(10_000):
        state = model.update(state, event)
    assert float(state.abs().max()) <= 1.01


def test_primary_grammar_reads_legacy_architecture_but_not_legacy_weights(tmp_path):
    old = FullHistorySequenceGRU(
        8, hidden_size=8, contact_embedding_dim=8,
        contact_encoder_hidden=8, local_offset_dim=2,
    )
    with torch.no_grad():
        for parameter in old.parameters():
            parameter.fill_(7.0)
    checkpoint = tmp_path / "legacy.pt"
    torch.save({
        "model_kwargs": {
            "hidden_size": 8,
            "contact_embedding_dim": 8,
            "contact_encoder_hidden": 8,
            "local_offset_dim": 2,
        },
        "model_state": old.state_dict(),
        "heldout_local_offset": torch.full((4, 2), 7.0),
    }, checkpoint)
    grammar = build_train_only_grammar(
        checkpoint, np.zeros((4, 8), np.float32), state_dim=4
    )
    assert all(not torch.allclose(p, torch.full_like(p, 7.0)) for p in grammar.base.parameters())
    assert torch.equal(grammar.local_offset, torch.zeros_like(grammar.local_offset))
    assert any(p is grammar.local_offset for p in grammar.calibration_parameters)


def test_intensity_equilibrium_is_exactly_the_train_marginal_rate():
    cfg = StateConfig(taus_seconds=(10.0, 100.0), channels_per_tau=1, event_dim=3)
    model = FixedTimescaleEventState(cfg)
    model.initialise_intensity_rate(events=50, observed_seconds=100.0)
    with torch.no_grad():
        model.mean.copy_(torch.tensor([3.0, -2.0]))
        model.intensity_head.weight.copy_(torch.tensor([[4.0, -7.0]]))
    equilibrium = model.mean.unsqueeze(0).repeat(4, 1)
    assert torch.allclose(model.intensity(equilibrium), torch.full((4,), 0.5), atol=1e-7)


def test_primary_state_bank_is_fixed_slow_physical_scales():
    cfg = StateConfig()
    assert cfg.taus_seconds == (300.0, 1800.0, 7200.0, 21600.0)
    assert cfg.state_dim == 16


def test_nested_partition_uses_recorded_time_not_gap_duration():
    segments = [
        CoverageSegment(0, 0, 0.0, 100.0),
        CoverageSegment(1, 1, 1000.0, 1100.0),
    ]
    partition = nested_time_partition(segments)
    assert np.allclose(partition.boundary_epochs, [40.0, 1040.0, 1060.0])
    assert partition.grammar_fit_stop_epoch == 32.0
    assert partition.labels_of(np.array([10.0, 50.0, 1050.0, 1080.0])).tolist() == [0, 1, 2, 3]


def test_tbptt_chunk_respects_event_and_physical_limits_without_reset_semantics():
    times = np.array([0.0, 1.0, 2.0, 100.0, 101.0, 102.0])
    positions = np.arange(times.size)
    chunks = list(
        _physical_event_chunks(positions, times, max_events=3, max_seconds=10.0)
    )
    assert [c.tolist() for c in chunks] == [[0, 1, 2], [3, 4, 5]]
    chunks = list(
        _physical_event_chunks(positions, times, max_events=10, max_seconds=1.1)
    )
    assert [c.tolist() for c in chunks] == [[0, 1], [2], [3, 4], [5]]


def test_single_size_categorical_factorises_without_double_counting_stop():
    torch.manual_seed(1)
    group_ids = torch.tensor([[0, 0, 1, 1]])
    group_count = torch.tensor([2])
    logits = torch.randn(1, 3, 5, requires_grad=True)
    contact = torch.randn(1, 3, 4)
    mark = tied_group_mark_log_prob(group_ids, group_count, logits, contact)
    factor = factorized_size_log_prob(
        group_ids, group_count, logits, torch.ones_like(group_ids, dtype=torch.bool)
    )
    rebuilt = (
        factor.continue_step_log_prob + factor.positive_size_step_log_prob
    ).sum(-1)
    assert torch.allclose(rebuilt, mark.group_size_log_prob, atol=1e-6)
    (-rebuilt.mean()).backward()
    assert torch.isfinite(logits.grad).all()


def test_state_free_grammar_really_disables_all_state_adapters(tmp_path):
    old = FullHistorySequenceGRU(
        8,
        hidden_size=8,
        contact_embedding_dim=8,
        contact_encoder_hidden=8,
        local_offset_dim=2,
    )
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model_kwargs": {
                "hidden_size": 8,
                "contact_embedding_dim": 8,
                "contact_encoder_hidden": 8,
                "local_offset_dim": 2,
            },
            "model_state": old.state_dict(),
            "heldout_local_offset": torch.zeros(4, 2),
        },
        checkpoint,
    )
    grammar = build_train_only_grammar(
        checkpoint, np.zeros((4, 8), np.float32), state_dim=4
    )
    group_ids = torch.tensor([[0, 0, 1, -1]])
    group_count = torch.tensor([2])
    _, a = grammar(
        group_ids, group_count, torch.zeros(1, 4), use_state_adapter=False
    )
    _, b = grammar(
        group_ids, group_count, torch.full((1, 4), 100.0), use_state_adapter=False
    )
    assert torch.equal(a["size_logits"], b["size_logits"])
    assert torch.equal(a["contact_logits"], b["contact_logits"])
