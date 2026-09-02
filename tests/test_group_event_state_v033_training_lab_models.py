"""Task 3: flexible residual state model + count-profile trainable (clauses M1-M9)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.topic5_group_event_state.v033_training_lab.data import build_view
from src.topic5_group_event_state.v033_training_lab.models import (
    ArchConfig,
    FlexibleResidualStateModel,
    GatedEventState,
    build_flexible_model,
)
from src.topic5_group_event_state.v033_training_lab.objective import (
    TRAINABLE_REGISTRY,
    ResidualCountTrainable,
)
from tests.test_group_event_state_v032_model_toyutil import make_toy_bundle

CPU = torch.device("cpu")
GROUPS = {"encoder_weights", "encoder_bias", "state_weights", "state_bias",
          "adapter_w", "adapter_gate_alpha", "adapter_dispersion"}


def _model(**arch):
    return build_flexible_model(ArchConfig(**arch), in_dim=7, n_bins=3, log_r_init=np.zeros(3), seed=1)


def _layernorms(module: nn.Module) -> list[str]:
    return [name for name, m in module.named_modules() if isinstance(m, nn.LayerNorm)]


def test_m1_layernorm_only_ever_lives_in_the_encoder():
    plain = _model()
    assert _layernorms(plain) == []
    normed = _model(hidden_norm="layernorm", depth=2)
    names = _layernorms(normed)
    assert names and all(n.startswith("encoder.") for n in names)
    assert _layernorms(normed.state) == [] and _layernorms(normed.adapter) == []
    gated = _model(state_family="gated_exploratory", hidden_norm="layernorm")
    assert all(n.startswith("encoder.") for n in _layernorms(gated))


def test_m2_gate_is_trainable_from_construction_with_no_freeze_api():
    model = _model(alpha_init=0.05)
    assert model.adapter.alpha.requires_grad is True
    assert float(model.adapter.alpha) == pytest.approx(0.05)
    assert not hasattr(model.adapter, "set_alpha_trainable")


def test_m3_param_groups_cover_every_parameter_once_and_no_decay_on_bias_gate_dispersion():
    for arch in (dict(), dict(hidden_norm="layernorm", depth=2), dict(state_family="gated_exploratory")):
        model = _model(**arch)
        lrs = {g: 1e-3 for g in GROUPS}
        groups = model.param_groups(lrs, weight_decay=1e-4)
        assert {g["name"] for g in groups} == GROUPS
        ids = [id(p) for g in groups for p in g["params"]]
        trainable = [id(p) for p in model.parameters() if p.requires_grad]
        assert sorted(ids) == sorted(trainable) and len(ids) == len(set(ids))
        assert not ({id(b) for b in model.buffers()} & set(ids))
        by = {g["name"]: g for g in groups}
        for name in ("encoder_bias", "state_bias", "adapter_gate_alpha", "adapter_dispersion"):
            assert by[name]["weight_decay"] == 0.0
        assert by["encoder_weights"]["weight_decay"] == 1e-4
        assert all(p.ndim <= 1 for p in by["encoder_bias"]["params"])
        assert all(p.ndim > 1 for p in by["encoder_weights"]["params"])
        if arch.get("state_family") == "gated_exploratory":
            assert by["state_weights"]["params"] and by["state_bias"]["params"]
        else:
            assert by["state_weights"]["params"] == [] and by["state_bias"]["params"] == []


def test_m4_write_scale_only_rescales_the_last_layer_init_and_orthogonal_init_is_orthogonal():
    one = build_flexible_model(ArchConfig(write_scale=1.0), in_dim=7, n_bins=3, log_r_init=np.zeros(3), seed=3)
    tenth = build_flexible_model(ArchConfig(write_scale=0.1), in_dim=7, n_bins=3, log_r_init=np.zeros(3), seed=3)
    first_one, first_tenth = one.encoder.layers[0].weight, tenth.encoder.layers[0].weight
    assert torch.allclose(first_one, first_tenth)
    last_one, last_tenth = one.encoder.output.weight, tenth.encoder.output.weight
    assert torch.allclose(last_tenth, 0.1 * last_one)
    ortho = build_flexible_model(ArchConfig(init="orthogonal", width=32), in_dim=32, n_bins=3,
                                 log_r_init=np.zeros(3), seed=3)
    w = ortho.encoder.layers[0].weight
    assert torch.allclose(w @ w.T, torch.eye(32), atol=1e-5)
    with pytest.raises(ValueError):
        ArchConfig(init="kaiming").validate()


def test_m5_state_dimension_and_time_bank_follow_the_config():
    for width in (2, 4, 8):
        model = _model(write_width=width)
        assert model.state_dim == 3 * width
        assert model.adapter.W.weight.shape == (3, 3 * width)
    slow = _model(taus_seconds=(600.0, 3600.0, 10800.0))
    assert torch.allclose(slow.state.taus, torch.tensor([600.0, 3600.0, 10800.0]))
    assert ArchConfig(write_width=4).state_dim == 12


def test_m6_gated_state_truncates_gradient_at_tbptt_chunks_without_resetting_the_state():
    torch.manual_seed(0)
    state = GatedEventState((300.0, 1800.0, 7200.0), 2, event_dim=4, hidden=8, tbptt_seconds=1000.0,
                            gate_bias_init=0.0)
    n = 12
    times = torch.linspace(0.0, 2750.0, n, dtype=torch.float64)      # chunks: [0,1000) [1000,2000) [2000,3000)
    seg = torch.zeros(n, dtype=torch.long)
    e = torch.randn(n, 4, requires_grad=True)
    pre, post = state(e, times, seg)
    grads = torch.autograd.grad(post[-1].sum(), e)[0].abs().sum(dim=1)
    chunk = torch.floor(times / 1000.0)
    same_chunk = chunk == chunk[-1]
    assert (grads[same_chunk] > 0).all()
    assert (grads[~same_chunk] == 0).all()
    # continuity across the chunk boundary: pre-state of the first event in a new chunk is the decayed
    # post-state of the previous event (values are carried, only the graph is cut)
    first_in_chunk = torch.nonzero(chunk[1:] != chunk[:-1]).flatten() + 1
    for j in first_in_chunk.tolist():
        dt = float(times[j] - times[j - 1])
        decay = torch.exp(-dt / state.taus_full.double()).float()
        assert torch.allclose(pre[j], post[j - 1] * decay, atol=1e-6)
        assert not torch.allclose(pre[j], torch.zeros_like(pre[j]))
    with_bias = GatedEventState((300.0, 1800.0, 7200.0), 2, event_dim=4, hidden=8, tbptt_seconds=1000.0,
                                gate_bias_init=3.0)
    assert with_bias.initial_gate_mean() > state.initial_gate_mean() + 0.3


def test_m7_residual_readout_has_no_free_intercept():
    model = _model()
    log_mu_h = torch.randn(5, 3)
    state = torch.randn(5, model.state_dim)
    out = model.log_mu(log_mu_h, state)
    expected = log_mu_h + model.adapter.alpha * (model.standardize_state(state) @ model.adapter.W.weight.T)
    assert torch.allclose(out, expected, atol=1e-6)
    with torch.no_grad():
        model.adapter.W.weight.zero_()
    assert torch.allclose(model.log_mu(log_mu_h, state), log_mu_h)


def test_m8_dropout_is_stochastic_in_train_and_deterministic_in_eval():
    model = _model(dropout=0.1, depth=2)
    x = torch.randn(64, 7)
    model.train()
    a, b = model.encoder(x), model.encoder(x)
    assert not torch.allclose(a, b)
    model.eval()
    assert torch.allclose(model.encoder(x), model.encoder(x))


def test_m9_count_profile_trainable_produces_per_anchor_terms_on_train_and_inner_val_only():
    bundle, _ = make_toy_bundle(seed=7, planted_beta=0.0)
    view = build_view(bundle)
    trainable = TRAINABLE_REGISTRY["count_profile"]()
    assert isinstance(trainable, ResidualCountTrainable) and trainable.name == "count_profile"
    model = trainable.build(ArchConfig(), view, seed=1).to(CPU)
    assert torch.allclose(model.adapter.log_r.detach(), torch.tensor(view.log_r_h, dtype=torch.float32))
    terms = trainable.loss_terms(model, view, "train", device=CPU, differentiable_statistics=True,
                                 sampling="event_balanced", lookback_seconds=7200.0)
    n = view.n("train")
    assert terms.nll.shape == (n,) and terms.per_bin_nll.shape == (n, 3) and terms.weights.shape == (n,)
    assert terms.modulation.shape == (n, 3) and terms.state_std.shape == (n, model.state_dim)
    assert torch.allclose(terms.nll, terms.per_bin_nll.sum(dim=1))
    assert terms.nll.requires_grad and terms.nll.dtype == torch.float32
    assert abs(float(terms.weights.mean()) - 1.0) < 1e-6
    val = trainable.loss_terms(model, view, "inner_val", device=CPU, differentiable_statistics=False,
                               sampling="anchor_balanced", lookback_seconds=7200.0)
    assert val.nll.shape == (view.n("inner_val"),)
    h = trainable.h_only_nll(view, "inner_val")
    assert h.shape == (view.n("inner_val"),) and np.isfinite(h).all()
    with pytest.raises(KeyError):
        trainable.loss_terms(model, view, "dev_test", device=CPU, differentiable_statistics=False,
                             sampling="anchor_balanced", lookback_seconds=7200.0)
    with pytest.raises(ValueError):
        trainable.loss_terms(model, view, "inner_val", device=CPU, differentiable_statistics=True,
                             sampling="anchor_balanced", lookback_seconds=7200.0)
