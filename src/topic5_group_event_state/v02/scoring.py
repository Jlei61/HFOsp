"""The one place a future block is scored (clause C9).

`B_multiscale`, `P_local`, `P_slow` and every null differ only in the columns of
``X``.  Anchors, windows, masks, standardisation and these likelihoods are shared
verbatim, so a reported gain cannot be an artefact of two arms having extracted
different windows and then subtracting.

Three families, each a proper score with its own unit:

    count          negative binomial, nats per anchor
    participation  Bernoulli, nats per (event x contact)
    continuous     Gaussian, nats per (valid event x dimension)

All three are computed from the window sufficient statistics only, so no dense
per-event target is ever built (clause C5).  Because eligibility requires the
whole window to lie inside one coverage segment, exposure is exactly the horizon
for every anchor -- there is no exposure offset that a fit could turn into a free
intercept, which is the failure mode this line has already been burned by.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import Tensor

FAMILIES = ("count", "participation", "continuous")


@dataclass(frozen=True)
class ScoreResult:
    """Mean NLL of one family plus the denominator it was averaged over."""

    nll_per_unit: float
    n_units: float
    n_anchors: int

    def as_dict(self) -> dict[str, float]:
        return {
            "nll_per_unit": float(self.nll_per_unit),
            "n_units": float(self.n_units),
            "n_anchors": int(self.n_anchors),
        }


def negative_binomial_nll(
    count: Tensor, log_mu: Tensor, log_alpha: Tensor
) -> Tensor:
    """NB2 NLL: mean ``mu``, variance ``mu + alpha * mu**2``.

    Poisson is not an option here: interictal event counts over 5-120 min are
    strongly overdispersed, and a Poisson score would reward any arm that merely
    shrinks its mean.
    """

    y = count.double()
    mu = torch.exp(log_mu.double().clamp(-30.0, 30.0))
    alpha = torch.exp(log_alpha.double().clamp(-10.0, 10.0))
    r = 1.0 / alpha
    return -(
        torch.lgamma(y + r)
        - torch.lgamma(r)
        - torch.lgamma(y + 1.0)
        + r * torch.log(r / (r + mu))
        + y * torch.log(mu / (r + mu))
    )


def participation_nll_from_counts(
    k: Tensor, n: Tensor, logit: Tensor
) -> Tensor:
    """Bernoulli NLL of ``n`` events of which ``k`` used each contact.

    ``sum_i BCE(p_c, y_ic)`` collapses exactly to ``-k log p - (n-k) log(1-p)``,
    which is why the participation field needs no dense (event x contact) target.
    """

    k = k.double()
    n = n.double().unsqueeze(-1)
    p = logit.double()
    log_p = torch.nn.functional.logsigmoid(p)
    log_q = torch.nn.functional.logsigmoid(-p)
    return -(k * log_p + (n - k) * log_q)


def gaussian_nll_from_moments(
    n: Tensor, sum_x: Tensor, sum_x2: Tensor, mu: Tensor, log_sigma: Tensor
) -> Tensor:
    """Exact Gaussian NLL of every event in the window, from its first two moments.

    ``sum_i 0.5*(log 2pi + 2 log s + (x_i - m)^2 / s^2)`` expands into
    ``n*(...) + (sum_x2 - 2 m sum_x + n m^2) / (2 s^2)`` -- no event is visited.
    """

    n = n.double().unsqueeze(-1)
    m = mu.double()
    ls = log_sigma.double().clamp(-8.0, 8.0)
    inv_var = torch.exp(-2.0 * ls)
    quad = sum_x2.double() - 2.0 * m * sum_x.double() + n * m * m
    return n * (0.5 * math.log(2 * math.pi) + ls) + 0.5 * quad * inv_var


def family_units(
    family: str, count: np.ndarray, n_valid: np.ndarray, n_contacts: int, n_dims: int
) -> float:
    if family == "count":
        return float(count.size)
    if family == "participation":
        return float(count.sum() * n_contacts)
    if family == "continuous":
        return float(n_valid.sum() * n_dims)
    raise ValueError(f"unknown family {family!r}")


def intercept_only_reference(
    stats_train, stats_eval, *, n_contacts: int, n_dims: int
) -> dict[str, ScoreResult]:
    """The floor every arm must clear: TRAIN marginals, no features at all.

    Reported next to every arm because a fit that is *worse* than this is not a
    weak effect, it is an unusable estimate (EI 2).
    """

    out: dict[str, ScoreResult] = {}

    n_tr = np.asarray(stats_train.count, dtype=np.float64)
    mu = max(float(n_tr.mean()), 1e-6)
    var = max(float(n_tr.var()), mu * 1.000001)
    alpha = max((var - mu) / (mu * mu), 1e-6)
    y = torch.as_tensor(np.asarray(stats_eval.count), dtype=torch.float64)
    nll = negative_binomial_nll(
        y,
        torch.full_like(y, math.log(mu)),
        torch.full_like(y, math.log(alpha)),
    )
    out["count"] = ScoreResult(
        float(nll.sum() / max(y.numel(), 1)), float(y.numel()), int(y.numel())
    )

    k_tr = np.asarray(stats_train.sum_participation, dtype=np.float64).sum(0)
    n_ev_tr = float(np.asarray(stats_train.count).sum())
    rate = np.clip(k_tr / max(n_ev_tr, 1.0), 1e-6, 1 - 1e-6)
    logit = torch.as_tensor(np.log(rate / (1 - rate)), dtype=torch.float64)
    k_ev = torch.as_tensor(np.asarray(stats_eval.sum_participation), dtype=torch.float64)
    n_ev = torch.as_tensor(np.asarray(stats_eval.count), dtype=torch.float64)
    nll_p = participation_nll_from_counts(k_ev, n_ev, logit.unsqueeze(0))
    units_p = family_units("participation", np.asarray(stats_eval.count),
                           np.asarray(stats_eval.n_valid_mark), n_contacts, n_dims)
    out["participation"] = ScoreResult(
        float(nll_p.sum() / max(units_p, 1.0)), units_p, int(n_ev.numel())
    )

    n_v_tr = max(float(np.asarray(stats_train.n_valid_mark).sum()), 1.0)
    m_tr = np.asarray(stats_train.sum_x, dtype=np.float64).sum(0) / n_v_tr
    v_tr = np.maximum(
        np.asarray(stats_train.sum_x2, dtype=np.float64).sum(0) / n_v_tr - m_tr ** 2, 1e-8
    )
    nll_c = gaussian_nll_from_moments(
        torch.as_tensor(np.asarray(stats_eval.n_valid_mark), dtype=torch.float64),
        torch.as_tensor(np.asarray(stats_eval.sum_x), dtype=torch.float64),
        torch.as_tensor(np.asarray(stats_eval.sum_x2), dtype=torch.float64),
        torch.as_tensor(m_tr, dtype=torch.float64).unsqueeze(0),
        torch.as_tensor(0.5 * np.log(v_tr), dtype=torch.float64).unsqueeze(0),
    )
    units_c = family_units("continuous", np.asarray(stats_eval.count),
                           np.asarray(stats_eval.n_valid_mark), n_contacts, n_dims)
    out["continuous"] = ScoreResult(
        float(nll_c.sum() / max(units_c, 1.0)), units_c,
        int(np.asarray(stats_eval.count).size),
    )
    return out


def block_scores(
    stats_eval, mu_count, log_alpha, logit_part, mu_cont, log_sigma_cont,
    *, block_slices: dict[str, slice], n_contacts: int, n_dims: int
) -> dict[str, ScoreResult]:
    """Per-family scores plus one entry per named continuous block.

    The per-block split is a first-pass requirement, not a diagnostic: a state
    that only sharpens ``size`` is an extent state and a state that only sharpens
    ``embedding`` is a repertoire state, and SP 8 needs those told apart.
    """

    count = np.asarray(stats_eval.count)
    n_valid = np.asarray(stats_eval.n_valid_mark)
    out: dict[str, ScoreResult] = {}

    y = torch.as_tensor(count, dtype=torch.float64)
    nll = negative_binomial_nll(y, mu_count, log_alpha)
    out["count"] = ScoreResult(float(nll.sum() / max(y.numel(), 1)), float(y.numel()),
                               int(y.numel()))

    k = torch.as_tensor(np.asarray(stats_eval.sum_participation), dtype=torch.float64)
    n_ev = torch.as_tensor(count, dtype=torch.float64)
    nll_p = participation_nll_from_counts(k, n_ev, logit_part)
    units_p = family_units("participation", count, n_valid, n_contacts, n_dims)
    out["participation"] = ScoreResult(float(nll_p.sum() / max(units_p, 1.0)), units_p,
                                       int(count.size))

    nv = torch.as_tensor(n_valid, dtype=torch.float64)
    sx = torch.as_tensor(np.asarray(stats_eval.sum_x), dtype=torch.float64)
    sx2 = torch.as_tensor(np.asarray(stats_eval.sum_x2), dtype=torch.float64)
    per_dim = gaussian_nll_from_moments(nv, sx, sx2, mu_cont, log_sigma_cont)
    units_c = family_units("continuous", count, n_valid, n_contacts, n_dims)
    out["continuous"] = ScoreResult(float(per_dim.sum() / max(units_c, 1.0)), units_c,
                                    int(count.size))
    for name, sl in block_slices.items():
        width = sl.stop - sl.start
        units = float(n_valid.sum() * width)
        out[f"continuous:{name}"] = ScoreResult(
            float(per_dim[:, sl].sum() / max(units, 1.0)), units, int(count.size)
        )
    return out
