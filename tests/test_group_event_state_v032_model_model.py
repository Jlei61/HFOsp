"""Task 5: encoder + model assembly + optimizer groups + shift donors (M1-M5)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.topic5_group_event_state.v032_model.config import ModelConfig
from src.topic5_group_event_state.v032_model.model import ResidualStateModel, build_model
from src.topic5_group_event_state.v032_model.shift import block_circular_donor


def _stream(n=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 7, generator=g)
    times = torch.cumsum(torch.rand(n, generator=g).double() * 200 + 1, 0)
    seg = torch.zeros(n, dtype=torch.long)
    seg[n // 2 :] = 1
    return x, times, seg


def test_m1_m3_param_groups_cover_every_parameter_exactly_once_and_bank_state_is_empty():
    cfg = ModelConfig()
    model = build_model(cfg, in_dim=7, log_r_init=0.5, seed=1)
    groups = model.param_groups(cfg)
    names = [g["name"] for g in groups]
    assert set(names) == {
        "encoder_weights", "encoder_bias", "state_weights", "state_bias",
        "adapter_w", "adapter_gate_alpha", "adapter_dispersion",
    }
    seen: list[int] = []
    for g in groups:
        seen.extend(id(p) for p in g["params"])
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert sorted(seen) == sorted(id(p) for p in trainable)
    assert len(seen) == len(set(seen))
    buffers = {id(b) for b in model.buffers()}
    assert not buffers & set(seen)
    by_name = {g["name"]: g for g in groups}
    assert by_name["state_weights"]["params"] == [] and by_name["state_bias"]["params"] == []
    for name in ("encoder_bias", "state_bias", "adapter_gate_alpha", "adapter_dispersion"):
        assert by_name[name]["weight_decay"] == 0.0
    assert by_name["encoder_weights"]["weight_decay"] == cfg.weight_decay
    assert float(model.adapter.alpha) == pytest.approx(cfg.alpha_init)


def test_m2_rnn_state_groups_are_populated_and_split_bias():
    cfg = ModelConfig(architecture="repaired_rnn")
    model = build_model(cfg, in_dim=7, log_r_init=0.0, seed=1)
    by_name = {g["name"]: g for g in model.param_groups(cfg)}
    assert len(by_name["state_weights"]["params"]) == 2 and len(by_name["state_bias"]["params"]) == 2
    assert all(p.ndim == 1 for p in by_name["state_bias"]["params"])
    assert not any(isinstance(m, nn.LayerNorm) for m in model.modules())


def test_m4_train_mean_uses_only_train_events_and_writes_are_centered_tanh():
    cfg = ModelConfig()
    model = build_model(cfg, in_dim=7, log_r_init=0.0, seed=2)
    x, times, seg = _stream()
    train_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    train_mask[:10] = True
    model.refresh_train_mean(x, train_mask)
    phi = model.encoder(x)
    assert torch.allclose(model.phi_mean, phi[:10].mean(0), atol=1e-6)
    assert not model.phi_mean.requires_grad
    u = model.writes(x)
    assert torch.allclose(u, torch.tanh(phi - model.phi_mean), atol=1e-6)
    pre, post = model.trajectory(x, times, seg)
    assert pre.shape == (x.shape[0], 12)
    # phi_mean is a buffer -> lives in the state_dict, so replay is checkpoint-specific
    assert "phi_mean" in model.state_dict() and "train_mean_state" in model.state_dict()


def test_m5_block_circular_donor_is_segment_preserving_and_horizon_separated():
    # segment 0: 20 anchors over 5700 s -> half shift = 3000 s >= horizon, all admissible;
    # segment 1: 2 anchors (< 3 -> no donor); segment 2: a single anchor.
    t = np.concatenate([np.arange(0, 6000, 300.0), np.arange(10_000, 10_600, 300.0), [50_000.0]])
    seg = np.array([0] * 20 + [1] * 2 + [2])
    idx = np.arange(t.size)
    donor = block_circular_donor(t, seg, idx, horizon=1800.0, fraction=0.5)
    assert donor.shape == idx.shape
    ok = donor >= 0
    assert ok.sum() == 20
    assert np.all(seg[idx[ok]] == seg[idx[donor[ok]]])
    assert np.all(np.abs(t[idx[ok]] - t[idx[donor[ok]]]) >= 1800.0)
    assert np.all(donor[20:] == -1)          # segments with < 3 anchors have no donor
    assert np.all(donor[ok] != idx[ok])
    # a short segment cannot host donors one horizon away under a half shift
    short = block_circular_donor(np.arange(0, 3000, 300.0), np.zeros(10, int), np.arange(10),
                                 horizon=1800.0, fraction=0.5)
    assert np.all(short == -1)
