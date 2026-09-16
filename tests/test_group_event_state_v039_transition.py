import pytest
import torch
from src.topic5_group_event_state.v039.transition import EventTransition, FutureReadout, endpoint_loss


def test_event_content_cannot_change_own_pre_event_prediction():
    m = EventTransition(3, 'N', seed=5).double()
    x = torch.randn(2, 5, 3, dtype=torch.float64)
    dt = torch.full((2, 5), .03, dtype=torch.float64)
    _, pre = m.scan(x, dt, return_pre_event=True)
    changed = x.clone(); changed[:, 3] += 20
    _, other = m.scan(changed, dt, return_pre_event=True)
    torch.testing.assert_close(pre[:, :4], other[:, :4], rtol=0, atol=0)
    assert not torch.allclose(pre[:, 4], other[:, 4])


def test_checkpoint_preserves_eight_hour_event_and_transition_gradients():
    m = EventTransition(3, 'N', seed=2).double()
    x = torch.randn(2, 96, 3, dtype=torch.float64, requires_grad=True)
    dt = torch.full((2, 96), 1/12, dtype=torch.float64)
    exact = m.scan(x, dt, checkpoint_chunk=0).square().mean()
    grads = torch.autograd.grad(exact, (x, m.skew, m.u, m.v, m.write))
    recomputed = m.scan(x, dt, checkpoint_chunk=13).square().mean()
    other = torch.autograd.grad(recomputed, (x, m.skew, m.u, m.v, m.write))
    for a, b in zip(grads, other):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
        assert b.abs().max() > 0
    assert grads[0][:, :12].abs().max() > 0


def test_linear_transition_is_stable_and_integrator_matches_matrix_exponential():
    m = EventTransition(2, 'L', seed=4).double()
    with torch.no_grad(): m.skew.normal_(std=.2)
    a = m.generator_matrix()
    assert torch.linalg.eigvals(a).real.max() < 0
    s = torch.randn(4, 16, dtype=torch.float64)
    exact = s @ torch.matrix_exp(a * 6).T
    numerical = m.advance(s, 6)
    torch.testing.assert_close(exact, numerical, atol=3e-6, rtol=3e-5)
    finer = m.advance(s, 6, max_step_hours=1/24)
    assert (finer-exact).norm() < (numerical-exact).norm()


def test_zero_nonlinear_correction_recovers_linear_model_and_same_event_write():
    l = EventTransition(2, 'L', seed=4)
    n = EventTransition(2, 'N', seed=4)
    torch.testing.assert_close(l.write, n.write)
    with torch.no_grad(): n.u.zero_()
    x = torch.randn(3, 9, 2); dt = torch.rand(3, 9)/12
    torch.testing.assert_close(l.scan(x, dt), n.scan(x, dt), atol=0, rtol=0)


def test_full_history_bank_and_missing_time_steps_are_not_event_updates():
    m = EventTransition(2, 'F')
    x = torch.zeros(1, 2, 2); x[0, 0] = torch.tensor([2., 3.])
    dt = torch.tensor([[0., 1/12]])
    torch.testing.assert_close(m.scan(x, dt), x[:, 0].repeat(1, 7)*torch.exp(-m.decay/12))
    with pytest.raises(ValueError, match='Insert zero-input'):
        m.scan(x, torch.ones_like(dt))


def test_objective_separates_count_and_recruitment_training_views():
    h = FutureReadout(16, 3)
    state = torch.randn(4, 16)
    mu, logits = h(state, torch.empty(4, 0), 2)
    counts = torch.tensor([0., 1., 3., 2.]); targets = torch.zeros(4, 3)
    count, _, _ = endpoint_loss(mu, logits, counts, targets, h.log_dispersion, 'count')
    assert torch.autograd.grad(count.mean(), logits, allow_unused=True)[0] is None
    spatial, _, _ = endpoint_loss(mu, logits, counts, targets, h.log_dispersion, 'recruitment')
    assert torch.autograd.grad(spatial.mean(), mu, allow_unused=True)[0] is None
