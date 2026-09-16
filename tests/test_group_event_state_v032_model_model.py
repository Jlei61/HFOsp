

def test_m6_state_readout_uses_train_only_fixed_mean_and_scale():
    """Design §3.1/§4: no LayerNorm; the readout sees (S - mean_train) / scale_train."""

    cfg = ModelConfig()
    model = build_model(cfg, in_dim=7, log_r_init=0.0, seed=3)
    x, times, seg = _stream(n=60)
    train_events = torch.zeros(x.shape[0], dtype=torch.bool)
    train_events[:30] = True
    t_anchor = times[5:40] + 1.0
    last = torch.arange(5, 40)
    model.refresh_train_statistics(x, train_events, times, seg, t_anchor, last)
    _pre, post = model.trajectory(x, times, seg)
    s_train = model.anchor_states(post, times, t_anchor, last)
    z = model.standardize_state(s_train)
    assert torch.allclose(z.mean(0), torch.zeros(12), atol=1e-5)
    assert torch.allclose(z.std(0, unbiased=False), torch.ones(12), atol=1e-4)
    assert torch.allclose(model.train_mean_state, s_train.mean(0), atol=1e-6)
    assert not model.train_state_scale.requires_grad and not model.train_mean_state.requires_grad
    # the TRAIN-mean arm is a zero modulation by construction
    mean_arm = model.log_mu(torch.zeros(3), model.train_mean_state.expand(3, -1))
    assert torch.allclose(mean_arm, torch.zeros(3), atol=1e-6)
    # the readout is linear in the standardised state with slope alpha * w
    s = torch.randn(4, 12)
    expected = model.adapter.alpha * (model.standardize_state(s) @ model.adapter.w.weight.squeeze(0))
    assert torch.allclose(model.log_mu(torch.zeros(4), s), expected, atol=1e-6)
