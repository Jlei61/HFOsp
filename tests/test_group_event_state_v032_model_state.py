"""Task 1: leaky-bank state core contract (design §3.1, plan clauses C1-C7)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from src.topic5_group_event_state.v032_model.config import ModelConfig, load_config
from src.topic5_group_event_state.v032_model.state import (
    MarkedLeakyBank,
    RepairedRecurrentState,
    anchor_states,
    leaky_bank_trajectory,
)


def _brute_force(u, times, seg, taus):
    n, k = u.shape
    t = taus.numel()
    pre = torch.zeros(n, t * k, dtype=torch.float64)
    post = torch.zeros_like(pre)
    for e in range(n):
        for ti, tau in enumerate(taus.tolist()):
            acc = torch.zeros(k, dtype=torch.float64)
            for j in range(e):
                if int(seg[j]) == int(seg[e]):
                    acc += math.exp(-(float(times[e]) - float(times[j])) / tau) * u[j].double()
            pre[e, ti * k : (ti + 1) * k] = acc
            post[e, ti * k : (ti + 1) * k] = acc + u[e].double()
    return pre.float(), post.float()


def _case(seed=0, n=40, k=2):
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n, k, generator=g)
    gaps = torch.rand(n, generator=g).double() * 40.0 + 1.0
    times = torch.cumsum(gaps, 0)
    seg = torch.zeros(n, dtype=torch.long)
    seg[n // 2 :] = 1
    taus = torch.tensor([30.0, 300.0])
    return u, times, seg, taus


def test_c1_trajectory_matches_brute_force_across_chunks():
    u, times, seg, taus = _case()
    pre, post = leaky_bank_trajectory(u, times, seg, taus, chunk_seconds=50.0)
    ref_pre, ref_post = _brute_force(u, times, seg, taus)
    assert pre.dtype == torch.float32 and post.dtype == torch.float32
    assert torch.allclose(pre, ref_pre, atol=1e-5, rtol=1e-5)
    assert torch.allclose(post, ref_post, atol=1e-5, rtol=1e-5)


def test_c2_no_state_to_state_mixing_between_channels():
    u, times, seg, taus = _case(n=6, k=2)
    k = u.shape[1]

    def f(x):
        return leaky_bank_trajectory(x, times, seg, taus, chunk_seconds=1e9)[1]

    jac = torch.autograd.functional.jacobian(f, u)  # (N, D, N, K)
    d = jac.shape[1]
    for i in range(d):
        for kk in range(k):
            if i % k != kk:
                assert torch.count_nonzero(jac[:, i, :, kk]) == 0


def test_c3_dt_enters_only_through_exponential_decay():
    u, times, seg, taus = _case(n=10)
    _pre, post = leaky_bank_trajectory(u, times, seg, taus, chunk_seconds=1e9)
    pre, _ = leaky_bank_trajectory(u, times, seg, taus, chunk_seconds=1e9)
    taus_full = taus.repeat_interleave(u.shape[1])
    for e in range(1, 5):
        dt = float(times[e] - times[e - 1])
        expected = post[e - 1] * torch.exp(-dt / taus_full)
        assert torch.allclose(pre[e], expected, atol=1e-6)


def test_c4_segment_start_state_is_zero():
    u, times, seg, taus = _case()
    pre, _post = leaky_bank_trajectory(u, times, seg, taus, chunk_seconds=50.0)
    first = [0, int((seg == 1).nonzero()[0])]
    for e in first:
        assert torch.count_nonzero(pre[e]) == 0


def test_c5_full_credit_across_chunks_unless_detach_requested():
    u, times, seg, taus = _case(n=12)
    seg[:] = 0
    k = u.shape[1]
    for detach, expect_zero in ((False, False), (True, True)):
        x = u.clone().requires_grad_(True)
        _pre, post = leaky_bank_trajectory(
            x, times, seg, taus, chunk_seconds=30.0, detach_chunks=detach
        )
        post[-1, 0].backward()  # tau index 0, channel 0
        grad_first = float(x.grad[0, 0])
        expected = math.exp(-float(times[-1] - times[0]) / float(taus[0]))
        if expect_zero:
            assert grad_first == 0.0
        else:
            assert abs(grad_first - expected) < 1e-6
            assert float(x.grad[0, 1]) == 0.0  # other channel untouched


def test_c6_anchor_state_is_autonomous_decay_of_last_post_state():
    u, times, seg, taus = _case(n=8)
    _pre, post = leaky_bank_trajectory(u, times, seg, taus, chunk_seconds=1e9)
    taus_full = taus.repeat_interleave(u.shape[1])
    t_anchor = torch.tensor([float(times[3]) + 7.0, 0.5], dtype=torch.float64)
    last = torch.tensor([3, -1])
    out = anchor_states(post, times, t_anchor, last, taus_full)
    assert out.shape == (2, post.shape[1])
    assert torch.allclose(out[0], post[3] * torch.exp(-7.0 / taus_full), atol=1e-6)
    assert torch.count_nonzero(out[1]) == 0


def test_bank_module_layout_and_dims():
    bank = MarkedLeakyBank((300.0, 1800.0, 7200.0), 4, chunk_seconds=3600.0)
    assert bank.state_dim == 12
    assert bank.taus_full.tolist() == [300.0] * 4 + [1800.0] * 4 + [7200.0] * 4
    assert sum(p.numel() for p in bank.parameters()) == 0
    u = torch.randn(5, 4)
    times = torch.arange(5, dtype=torch.float64) * 10.0
    seg = torch.zeros(5, dtype=torch.long)
    pre, post = bank(u, times, seg)
    assert pre.shape == (5, 12) and post.shape == (5, 12)


def test_c7_repaired_rnn_has_no_layernorm_and_same_autonomous_decay():
    torch.manual_seed(0)
    rnn = RepairedRecurrentState((30.0, 300.0), 2, event_dim=3, hidden=8)
    assert not any(isinstance(m, nn.LayerNorm) for m in rnn.modules())
    e = torch.randn(9, 3)
    times = torch.cumsum(torch.rand(9).double() * 20 + 1, 0)
    seg = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
    pre, post = rnn(e, times, seg)
    assert pre.shape == (9, 4) and post.shape == (9, 4)
    assert torch.count_nonzero(pre[0]) == 0 and torch.count_nonzero(pre[4]) == 0
    taus_full = rnn.taus_full
    for i in (1, 2, 3, 5, 6):
        dt = float(times[i] - times[i - 1])
        assert torch.allclose(pre[i], post[i - 1] * torch.exp(-dt / taus_full), atol=1e-6)
    # the update is state dependent (this is the mixing the bank forbids)
    assert not torch.allclose(post[1] - pre[1], post[2] - pre[2])
    assert sum(p.numel() for p in rnn.parameters()) > 0


def test_config_state_dim_and_hash():
    cfg = ModelConfig()
    assert cfg.state_dim == 12
    assert cfg.taus_seconds == (300.0, 1800.0, 7200.0)
    assert 0.02 <= cfg.alpha_init <= 0.05
    assert cfg.alpha_freeze_steps > 0
    assert cfg.config_hash() == ModelConfig().config_hash()
    assert cfg.config_hash() != ModelConfig(alpha_init=0.04).config_hash()
    loaded = load_config(None, architecture="repaired_rnn")
    assert loaded.architecture == "repaired_rnn"
    with pytest.raises(ValueError):
        ModelConfig(architecture="free_rnn").validate()
