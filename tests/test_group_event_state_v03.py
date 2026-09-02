from __future__ import annotations

import torch

from src.topic5_continuous_marked_state_r1.mark_likelihood import tied_group_mark_log_prob
from src.topic5_group_event_state.v03.point_process import interval_point_process_terms
from src.topic5_group_event_state.v03.state import FixedTimescaleEventState, StateConfig


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
    group_ids = torch.tensor([[0, 0, 1, -1]])
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
