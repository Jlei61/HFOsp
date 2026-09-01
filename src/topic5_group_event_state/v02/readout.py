"""Nested-increment readouts: ``B`` vs ``B + S`` on identical anchors (CC 6).

Why the load-bearing comparison is a re-fitted readout rather than each
producer's own head: ``P_local`` has no future-block head at all, so a direct
head-to-head with ``P_slow`` would compare a trained future predictor against a
model that was never asked the question.  Freezing each producer's state and
fitting the *same* readout family on top of it makes ``B``, ``B+S(P_local)``,
``B+S(P_slow)`` and the shifted null differ in exactly one thing: which columns
sit in ``X``.  ``P_slow``'s own trained heads are reported alongside as a
secondary number, not as the increment.

**Each endpoint family is fitted and regularised on its own.**  A count target
gives one observation per anchor, while the participation target gives
``events x contacts`` observations per anchor -- three orders of magnitude more.
A single shared ridge value cannot serve both: the first smoke run on
``yuquan_chengshuai`` (195 train anchors, 112 features) produced a count NLL of
27.5 against an intercept floor of 6.85, i.e. a fit so overfitted it was
unusable, while the participation fit on the same lambda was fine.

Two failure modes this file is written against, both already paid for in this
repository:

* a fit that saturates into a free intercept and then reports a huge gain --
  every arm is scored against the TRAIN-marginal intercept model, and an arm
  that lands far *worse* than that floor is marked ``not_estimable`` instead of
  being read as a weak effect;
* a ridge grid whose selected value sits on the edge for most fits -- the chosen
  lambda and whether it was an edge value are recorded per family, per fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .scoring import (
    ScoreResult,
    block_scores,
    family_units,
    gaussian_nll_from_moments,
    intercept_only_reference,
    negative_binomial_nll,
    participation_nll_from_counts,
)

# Ridge grid on standardised features with a per-unit-normalised data term.  The
# top reaches 1e5 because a 120 min horizon can leave a patient with ~100 training
# anchors against ~110 features: the first smoke run selected the old top value
# 1e3 for the count family, i.e. it sat on the grid edge, and an edge selection
# cannot say whether the honest answer was "even more shrinkage".  At the top of
# this grid the fit is numerically the intercept-only model, so the grid now
# spans the whole useful range and an edge hit is informative rather than a cap.
LAMBDA_GRID: tuple[float, ...] = (
    1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5,
)

# An arm whose held-out NLL is this much worse than the intercept-only floor is
# not a weak result, it is an unusable fit.
NOT_ESTIMABLE_MARGIN_NATS = 0.5

FAMILIES = ("count", "participation", "continuous")


def _subset_stats(stats, idx: np.ndarray):
    """Row subset of the window sufficient statistics (used by the lambda CV)."""

    from .targets import WindowStats

    return WindowStats(
        count=np.asarray(stats.count)[idx],
        n_valid_mark=np.asarray(stats.n_valid_mark)[idx],
        sum_participation=np.asarray(stats.sum_participation)[idx],
        sum_x=np.asarray(stats.sum_x)[idx],
        sum_x2=np.asarray(stats.sum_x2)[idx],
    )


@dataclass(frozen=True)
class ReadoutConfig:
    lambdas: tuple[float, ...] = LAMBDA_GRID
    max_iter: int = 200
    hidden: int = 0          # 0 = linear GLM (primary); >0 = small-MLP sensitivity
    seed: int = 0
    cv_folds: int = 3        # chronological blocks inside TRAIN for lambda selection


@dataclass
class FittedReadout:
    params: dict[str, Tensor]
    lam: dict[str, float]
    lam_at_grid_edge: dict[str, bool]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    n_features: int
    hidden: int
    val_objective: dict[str, float]
    lambda_path: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    def transform(self, x: np.ndarray) -> Tensor:
        z = (np.asarray(x, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return torch.as_tensor(z, dtype=torch.float64)


def _standardise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-mean unit-scale columns, so one ridge grid fits every patient (EI 2)."""

    mean = np.asarray(x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(x, dtype=np.float64).std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return mean, scale


def _marginals(stats, n_contacts: int, n_dims: int) -> dict[str, np.ndarray]:
    n = np.asarray(stats.count, dtype=np.float64)
    mu = max(float(n.mean()), 1e-6)
    var = max(float(n.var()), mu * 1.000001)
    alpha = max((var - mu) / (mu * mu), 1e-6)
    k = np.asarray(stats.sum_participation, dtype=np.float64).sum(0)
    rate = np.clip(k / max(float(n.sum()), 1.0), 1e-6, 1 - 1e-6)
    nv = max(float(np.asarray(stats.n_valid_mark).sum()), 1.0)
    m = np.asarray(stats.sum_x, dtype=np.float64).sum(0) / nv
    v = np.maximum(np.asarray(stats.sum_x2, dtype=np.float64).sum(0) / nv - m ** 2, 1e-8)
    return {
        "log_mu": np.array([math.log(mu)]),
        "log_alpha": np.array([math.log(alpha)]),
        "logit": np.log(rate / (1 - rate)),
        "cont_mean": m,
        "cont_log_sigma": 0.5 * np.log(v),
    }


def _family_params(
    family: str, stats, n_contacts: int, n_dims: int, n_features: int,
    hidden: int, seed: int,
) -> dict[str, Tensor]:
    """Start at the intercept-only solution, so 'features add nothing' is a fixed point."""

    g = torch.Generator().manual_seed(int(seed))
    marg = _marginals(stats, n_contacts, n_dims)
    n_in = hidden if hidden > 0 else n_features
    out_dim = {"count": 1, "participation": n_contacts, "continuous": n_dims}[family]

    params: dict[str, Tensor] = {
        "W": torch.zeros((n_in, out_dim), dtype=torch.float64, requires_grad=True),
    }
    if hidden > 0:
        # The capacity check needs a *skip connection*.  Without it the arm is
        # ``tanh(zH + b) @ W``, which is a restriction of the linear model rather
        # than an extension of it -- and the first real run showed exactly that:
        # the MLP arm collapsed onto the intercept while the linear GLM was far
        # better, so "extra capacity does not help" would have been meaningless.
        params["W_lin"] = torch.zeros((n_features, out_dim), dtype=torch.float64,
                                      requires_grad=True)
        params["H"] = (torch.randn((n_features, hidden), generator=g, dtype=torch.float64)
                       * 0.1).requires_grad_(True)
        params["hb"] = torch.zeros(hidden, dtype=torch.float64, requires_grad=True)
    if family == "count":
        params["b"] = torch.as_tensor(marg["log_mu"]).clone().requires_grad_(True)
        params["log_alpha"] = torch.as_tensor(marg["log_alpha"]).clone().requires_grad_(True)
    elif family == "participation":
        params["b"] = torch.as_tensor(marg["logit"]).clone().requires_grad_(True)
    else:
        params["b"] = torch.as_tensor(marg["cont_mean"]).clone().requires_grad_(True)
        params["log_sigma"] = (
            torch.as_tensor(marg["cont_log_sigma"]).clone().requires_grad_(True)
        )
    return params


def _family_linear(params: Mapping[str, Tensor], z: Tensor, hidden: int) -> Tensor:
    if hidden <= 0:
        return z @ params["W"] + params["b"]
    h = torch.tanh(z @ params["H"] + params["hb"])
    return z @ params["W_lin"] + h @ params["W"] + params["b"]


def _family_nll(
    family: str, params: Mapping[str, Tensor], z: Tensor, stats, *,
    n_contacts: int, n_dims: int, hidden: int,
) -> tuple[Tensor, float]:
    count = np.asarray(stats.count)
    n_valid = np.asarray(stats.n_valid_mark)
    out = _family_linear(params, z, hidden)
    units = max(family_units(family, count, n_valid, n_contacts, n_dims), 1.0)
    if family == "count":
        y = torch.as_tensor(count, dtype=torch.float64)
        return negative_binomial_nll(y, out[:, 0], params["log_alpha"]).sum(), units
    if family == "participation":
        k = torch.as_tensor(np.asarray(stats.sum_participation), dtype=torch.float64)
        n_ev = torch.as_tensor(count, dtype=torch.float64)
        return participation_nll_from_counts(k, n_ev, out).sum(), units
    nv = torch.as_tensor(n_valid, dtype=torch.float64)
    sx = torch.as_tensor(np.asarray(stats.sum_x), dtype=torch.float64)
    sx2 = torch.as_tensor(np.asarray(stats.sum_x2), dtype=torch.float64)
    return gaussian_nll_from_moments(nv, sx, sx2, out, params["log_sigma"]).sum(), units


def _fit_family_one_lambda(
    family: str, z_train: Tensor, stats_train, *, lam: float, n_contacts: int,
    n_dims: int, config: ReadoutConfig, init: Mapping[str, Tensor] | None = None,
) -> dict[str, Tensor]:
    """Fit one family at one ridge value, optionally warm-started.

    ``init`` is used only to walk the lambda path from strong to weak
    regularisation within a single arm: at the top of the grid the solution is
    the intercept model, and each step down moves a little.  It is applied
    identically to every arm and every fold, so it cannot favour one arm -- the
    asymmetry this repository was burned by came from inheriting an already
    converged read-out *between* arms, which is not what happens here.
    """

    params = _family_params(
        family, stats_train, n_contacts, n_dims, z_train.shape[1], config.hidden,
        config.seed,
    )
    if init is not None:
        with torch.no_grad():
            for key, value in init.items():
                if key in params and params[key].shape == value.shape:
                    params[key].copy_(value)
    opt = torch.optim.LBFGS(
        list(params.values()), max_iter=config.max_iter, history_size=20,
        line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12,
    )

    def closure() -> Tensor:
        opt.zero_grad(set_to_none=True)
        total, units = _family_nll(
            family, params, z_train, stats_train, n_contacts=n_contacts,
            n_dims=n_dims, hidden=config.hidden,
        )
        loss = total / units + float(lam) * (params["W"] ** 2).sum()
        for extra in ("H", "W_lin"):
            if extra in params:
                loss = loss + float(lam) * (params[extra] ** 2).sum()
        if torch.isfinite(loss):
            loss.backward()
        return loss

    opt.step(closure)
    return {k: v.detach() for k, v in params.items()}


def _chronological_folds(n: int, k: int) -> list[np.ndarray]:
    """Contiguous, time-ordered blocks of the TRAIN anchors."""

    edges = np.linspace(0, n, k + 1).astype(int)
    return [np.arange(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def fit_readout(
    x_train: np.ndarray,
    stats_train,
    x_val: np.ndarray,
    stats_val,
    *,
    n_contacts: int,
    n_dims: int,
    config: ReadoutConfig = ReadoutConfig(),
    family_lambdas: Mapping[str, Sequence[float]] | None = None,
) -> FittedReadout:
    """Fit on TRAIN; select each family's ridge by chronological CV inside TRAIN.

    Selecting on the 10% inner-validation slice looked natural but breaks at long
    horizons: at 2 h that slice can hold a single independent window, so the
    choice is a coin flip.  An earlier version tried to repair this by inheriting
    the 5 min horizon's lambda, which was worse -- a value tuned where there are
    200 training anchors and an easy target is far too weak for a 2 h count, and
    it drove one smoke fit to an NLL of 892 against an intercept floor of 8.2.

    Chronological K-fold blocks *inside* TRAIN give every horizon a selection set
    that scales with it, never touch the development test split, and are applied
    identically to every arm.  The inner-validation NLL of the final fit is still
    reported, as a diagnostic.
    """

    x_train = np.asarray(x_train, dtype=np.float64)
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2-D")
    mean, scale = _standardise(x_train)
    z_train = torch.as_tensor((x_train - mean) / scale, dtype=torch.float64)
    z_val = torch.as_tensor(
        (np.asarray(x_val, dtype=np.float64) - mean) / scale, dtype=torch.float64
    )

    params: dict[str, Tensor] = {}
    lam_by_family: dict[str, float] = {}
    edge_by_family: dict[str, bool] = {}
    val_by_family: dict[str, float] = {}
    path_by_family: dict[str, list[dict[str, float]]] = {}

    folds = _chronological_folds(z_train.shape[0], config.cv_folds)
    for family in FAMILIES:
        grid = tuple((family_lambdas or {}).get(family, config.lambdas))
        path: list[dict[str, float]] = []
        best_lam: float | None = None
        best_cv = math.inf
        # Walk from the strongest shrinkage downwards so each fit starts from the
        # previous, slightly-more-regularised solution.
        warm: dict[int, dict[str, Tensor]] = {}
        for lam in sorted(grid, reverse=True):
            if len(grid) == 1:
                path.append({"lambda": float(lam), "cv_nll_per_unit": float("nan")})
                best_lam, best_cv = float(lam), 0.0
                continue
            total_nll = 0.0
            total_units = 0.0
            ok = True
            for f_i, held in enumerate(folds):
                keep = np.setdiff1d(np.arange(z_train.shape[0]), held)
                if keep.size < 2 or held.size < 1:
                    continue
                fitted = _fit_family_one_lambda(
                    family, z_train[keep], _subset_stats(stats_train, keep), lam=lam,
                    n_contacts=n_contacts, n_dims=n_dims, config=config,
                    init=warm.get(f_i),
                )
                warm[f_i] = fitted
                with torch.no_grad():
                    nll, units = _family_nll(
                        family, fitted, z_train[held], _subset_stats(stats_train, held),
                        n_contacts=n_contacts, n_dims=n_dims, hidden=config.hidden,
                    )
                if not math.isfinite(float(nll)):
                    ok = False
                    break
                total_nll += float(nll)
                total_units += units
            cv = total_nll / total_units if (ok and total_units > 0) else math.inf
            path.append({"lambda": float(lam), "cv_nll_per_unit": cv})
            if cv < best_cv:
                best_cv, best_lam = cv, float(lam)
        if best_lam is None:
            raise RuntimeError(f"{family}: every ridge value gave a non-finite CV NLL")
        fitted = _fit_family_one_lambda(
            family, z_train, stats_train, lam=best_lam, n_contacts=n_contacts,
            n_dims=n_dims, config=config,
        )
        with torch.no_grad():
            nll, units = _family_nll(
                family, fitted, z_val, stats_val, n_contacts=n_contacts,
                n_dims=n_dims, hidden=config.hidden,
            )
        for key, value in fitted.items():
            params[f"{family}.{key}"] = value
        lam_by_family[family] = best_lam
        edge_by_family[family] = len(grid) > 1 and best_lam in (grid[0], grid[-1])
        val_by_family[family] = float(nll) / units
        path_by_family[family] = path

    return FittedReadout(
        params=params, lam=lam_by_family, lam_at_grid_edge=edge_by_family,
        feature_mean=mean, feature_scale=scale, n_features=int(x_train.shape[1]),
        hidden=config.hidden, val_objective=val_by_family, lambda_path=path_by_family,
    )


def _family_view(fit: FittedReadout, family: str) -> dict[str, Tensor]:
    prefix = f"{family}."
    return {k[len(prefix):]: v for k, v in fit.params.items() if k.startswith(prefix)}


def score_readout(
    fit: FittedReadout,
    x_eval: np.ndarray,
    stats_eval,
    *,
    block_slices: Mapping[str, slice],
    n_contacts: int,
    n_dims: int,
) -> dict[str, ScoreResult]:
    """Score held-out anchors -- the single scoring entry point (clause C9)."""

    with torch.no_grad():
        z = fit.transform(x_eval)
        p_count = _family_view(fit, "count")
        p_part = _family_view(fit, "participation")
        p_cont = _family_view(fit, "continuous")
        log_mu = _family_linear(p_count, z, fit.hidden)[:, 0]
        logit = _family_linear(p_part, z, fit.hidden)
        mu_cont = _family_linear(p_cont, z, fit.hidden)
        return block_scores(
            stats_eval, log_mu, p_count["log_alpha"], logit, mu_cont,
            p_cont["log_sigma"], block_slices=dict(block_slices),
            n_contacts=n_contacts, n_dims=n_dims,
        )


def estimability(
    arm_scores: Mapping[str, ScoreResult],
    intercept_scores: Mapping[str, ScoreResult],
    *,
    margin_nats: float = NOT_ESTIMABLE_MARGIN_NATS,
) -> dict[str, str]:
    """``ok`` / ``not_estimable`` per endpoint, against the intercept-only floor."""

    out: dict[str, str] = {}
    for key, score in arm_scores.items():
        base_key = key.split(":")[0] if ":" in key else key
        base = intercept_scores.get(base_key)
        if base is None or not math.isfinite(score.nll_per_unit):
            out[key] = "not_estimable"
        elif score.nll_per_unit > base.nll_per_unit + float(margin_nats):
            out[key] = "not_estimable"
        else:
            out[key] = "ok"
    return out


def block_circular_shift(
    values: np.ndarray,
    session_id: np.ndarray,
    t_anchor: np.ndarray,
    shift_steps: int,
) -> np.ndarray:
    """Roll the state within each recorded session (clause C10).

    Keeps every marginal, the state's own autocorrelation and the session /
    coarse clock structure; destroys only the correspondence between a state and
    the block that actually followed it.
    """

    values = np.asarray(values, dtype=np.float64)
    out = np.array(values, copy=True)
    sess = np.asarray(session_id)
    order = np.argsort(np.asarray(t_anchor, dtype=np.float64), kind="stable")
    for s in np.unique(sess):
        idx = order[sess[order] == s]
        if idx.size <= 1:
            continue
        k = int(shift_steps) % idx.size
        if k == 0:
            continue
        out[idx] = values[np.roll(idx, k)]
    return out


def validate_shift_exceeds_horizon(
    shift_steps: int, grid_seconds: float, horizon_seconds: float
) -> None:
    if shift_steps * float(grid_seconds) <= float(horizon_seconds):
        raise ValueError(
            f"block shift of {shift_steps} x {grid_seconds}s does not exceed the "
            f"{horizon_seconds}s horizon; the shifted state would still overlap "
            "the block it is being asked to predict"
        )


def shiftable_sessions(session_id: np.ndarray, shift_steps: int) -> tuple[int, int]:
    """How many anchors sit in a session long enough for this shift to be a shift."""

    sess = np.asarray(session_id)
    total = 0
    for s in np.unique(sess):
        n = int((sess == s).sum())
        if n > shift_steps:
            total += n
    return int(total), int(sess.size)


def reference_scores(
    stats_train, stats_eval, *, n_contacts: int, n_dims: int
) -> dict[str, ScoreResult]:
    return intercept_only_reference(
        stats_train, stats_eval, n_contacts=n_contacts, n_dims=n_dims
    )


def gain_table(
    arm_scores: Mapping[str, ScoreResult],
    baseline_scores: Mapping[str, ScoreResult],
) -> dict[str, float]:
    """Positive means the arm is better than ``B_multiscale`` on that endpoint."""

    out: dict[str, float] = {}
    for key, score in arm_scores.items():
        base = baseline_scores.get(key)
        if base is None:
            continue
        out[key] = float(base.nll_per_unit - score.nll_per_unit)
    return out
