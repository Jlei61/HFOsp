"""Task 2: negative-binomial residual readout (design §4, clauses R1-R5)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy import stats

from src.topic5_group_event_state.v032_model.readout import (
    ResidualCountAdapter,
    fit_nb_log_dispersion,
    moment_log_dispersion,
    nb_log_prob,
)


def test_r1_nb_log_prob_matches_scipy_pointwise():
    y = torch.tensor([0.0, 1.0, 7.0, 250.0, 1300.0])
    mu = torch.tensor([0.5, 2.0, 5.0, 300.0, 900.0])
    log_r = torch.tensor(math.log(3.5))
    got = nb_log_prob(y, mu, log_r)
    r = 3.5
    y64, mu64 = y.numpy().astype(np.float64), mu.numpy().astype(np.float64)
    ref = stats.nbinom.logpmf(y64, r, r / (r + mu64))  # float64 reference (float32 gammaln is 1e-3 off)
    assert np.allclose(got.numpy(), ref, atol=1e-5)


def test_r2_dispersion_mle_recovers_simulated_r():
    rng = np.random.default_rng(0)
    mu = np.exp(rng.normal(4.0, 0.5, size=20000))
    r_true = 4.0
    y = rng.negative_binomial(r_true, r_true / (r_true + mu))
    log_r = fit_nb_log_dispersion(y, mu)
    assert abs(log_r - math.log(r_true)) < 0.1
    assert math.isfinite(moment_log_dispersion(y, mu))


def test_r3_adapter_has_no_free_intercept_and_is_residual_on_log_mu_h():
    torch.manual_seed(0)
    adapter = ResidualCountAdapter(state_dim=12, alpha_init=0.03, log_r_init=0.0)
    names = sorted(n for n, _ in adapter.named_parameters())
    assert names == ["alpha", "log_r", "w.weight"]
    s = torch.randn(7, 12)
    log_mu_h = torch.randn(7)
    out = adapter(log_mu_h, s)
    assert torch.allclose(out - log_mu_h, adapter.modulation(s), atol=1e-7)
    assert torch.allclose(adapter.modulation(s), adapter.alpha * adapter.w(s).squeeze(-1))
    # w is ordinary random init, not zero
    assert float(adapter.w.weight.abs().max()) > 0


def test_r4_alpha_init_and_freeze_switch():
    adapter = ResidualCountAdapter(state_dim=4, alpha_init=0.03, log_r_init=0.0)
    assert float(adapter.alpha) == pytest.approx(0.03)
    s = torch.randn(3, 4)
    adapter.set_alpha_trainable(False)
    adapter(torch.zeros(3), s).sum().backward()
    assert adapter.alpha.grad is None
    assert adapter.w.weight.grad is not None
    adapter.zero_grad(set_to_none=True)
    adapter.set_alpha_trainable(True)
    adapter(torch.zeros(3), s).sum().backward()
    assert adapter.alpha.grad is not None


def test_r5_nb_likelihood_is_computed_in_fp32_even_for_low_precision_inputs():
    y = torch.tensor([3.0, 40.0])
    mu = torch.tensor([2.5, 38.0])
    log_r = torch.tensor(1.0)
    low = nb_log_prob(y.to(torch.bfloat16), mu.to(torch.bfloat16), log_r.to(torch.bfloat16))
    assert low.dtype == torch.float32
    ref = nb_log_prob(y.to(torch.bfloat16).float(), mu.to(torch.bfloat16).float(), log_r.float())
    assert torch.allclose(low, ref)
