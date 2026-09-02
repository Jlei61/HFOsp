"""Negative-binomial residual readout on top of the explicit history baseline.

    log mu_{H+S} = log mu_H + alpha * w^T S

There is deliberately **no free intercept**: a constant offset would let
``H+S`` beat ``H`` by re-calibrating the level of ``H``.  That effect is measured
explicitly by the ``H + mean(S_train)`` arm instead (design §4/§5).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, special
import torch
from torch import Tensor, nn

LOG_R_MIN = math.log(0.05)
LOG_R_MAX = math.log(1e5)


def nb_log_prob(y: Tensor, mu: Tensor, log_r: Tensor) -> Tensor:
    """Elementwise NB(y | mu, r) log-probability.

    Evaluated in float64 and returned as float32: the contract forbids anything
    *below* FP32 here, and float32 ``lgamma`` differences lose ~1e-3 nats at the
    30-minute counts of the densest patient (~1e3 events), which would leak into
    paired contrasts between arms with different dispersions.
    """

    y64 = y.to(torch.float64)
    mu64 = mu.to(torch.float64).clamp_min(1e-8)
    log_r64 = log_r.to(torch.float64)
    r = torch.exp(log_r64)
    log_r_plus_mu = torch.log(r + mu64)
    value = (
        torch.lgamma(y64 + r)
        - torch.lgamma(r)
        - torch.lgamma(y64 + 1.0)
        + r * (log_r64 - log_r_plus_mu)
        + y64 * (torch.log(mu64) - log_r_plus_mu)
    )
    return value.to(torch.float32)


def _nb_log_prob_np(y: np.ndarray, mu: np.ndarray, log_r: float) -> np.ndarray:
    r = math.exp(log_r)
    mu = np.clip(np.asarray(mu, dtype=np.float64), 1e-8, None)
    y = np.asarray(y, dtype=np.float64)
    return (
        special.gammaln(y + r) - special.gammaln(r) - special.gammaln(y + 1.0)
        + r * (log_r - np.log(r + mu)) + y * (np.log(mu) - np.log(r + mu))
    )


def moment_log_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Method-of-moments ``log r`` from the excess variance around ``mu``."""

    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    excess = float(np.mean((y - mu) ** 2 - mu))
    m2 = float(np.mean(mu ** 2))
    r = m2 / max(excess, 1e-6 * max(m2, 1e-12))
    return float(np.clip(math.log(max(r, 1e-12)), LOG_R_MIN, LOG_R_MAX))


def fit_nb_log_dispersion(
    y: np.ndarray, mu: np.ndarray, *, lo: float = LOG_R_MIN, hi: float = LOG_R_MAX
) -> float:
    """One-dimensional MLE of ``log r`` for fixed means (used for the H-only arm)."""

    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    if y.size == 0:
        raise ValueError("cannot fit a dispersion on zero anchors")
    result = optimize.minimize_scalar(
        lambda v: -float(_nb_log_prob_np(y, mu, float(v)).sum()),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


class ResidualCountAdapter(nn.Module):
    """``log mu_H + alpha * w^T S`` with a learnable NB dispersion and no bias."""

    def __init__(self, state_dim: int, alpha_init: float, log_r_init: float) -> None:
        super().__init__()
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")
        self.w = nn.Linear(int(state_dim), 1, bias=False)  # ordinary random init
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.log_r = nn.Parameter(torch.tensor(float(log_r_init)))

    def set_alpha_trainable(self, flag: bool) -> None:
        self.alpha.requires_grad_(bool(flag))

    def modulation(self, state: Tensor) -> Tensor:
        return self.alpha * self.w(state.to(torch.float32)).squeeze(-1)

    def forward(self, log_mu_h: Tensor, state: Tensor) -> Tensor:
        return log_mu_h.to(torch.float32) + self.modulation(state)
