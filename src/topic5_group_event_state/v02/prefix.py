"""H2a same-prefix continuation: does the state change how a started event goes on?

The question (SP A3): among events that *began* the same way -- same first
contact, same first two resolvable recruitment steps, same early waveform and
early band energy -- does knowing the slow state improve the prediction of what
happened next?

Two layers, reported separately and never collapsed:

* **matched-set layer (interpretable).**  Events are bucketed by a discrete,
  TRAIN-frozen prefix key, and the per-patient count of matchable events per
  bucket is reported.  This is what "same prefix" means in plain language.
* **nested-increment layer (primary).**  ``outcome ~ prefix`` versus
  ``outcome ~ prefix + state`` on held-out events, with the same ridge and
  chronological-CV discipline as the future-block readout.  It needs no bucket
  count and therefore no arbitrary K.

Outcomes are split, because "the state changed the continuation" is four
different claims: whether recruitment continued at all, which further contacts
it reached, how far it got before stopping, and what its later spectral
expression was.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .marks import BAND_FEATURE_ENERGY, BAND_FEATURE_PEAK
from .readout import LAMBDA_GRID, ReadoutConfig, _chronological_folds, _standardise

# The prefix is "the first two resolvable recruitment steps": one 10 ms centroid
# hop is the producer's resolution, so groups 0 and 1 are what can be ordered.
PREFIX_GROUPS = 2

# Early window used for the waveform / energy part of the prefix (SP A3).
EARLY_WINDOW_SECONDS = 0.100

# A discrete prefix bucket is kept only if TRAIN holds at least this many events
# of it; smaller buckets go to an explicit "rare" bucket rather than being
# dropped, so no event silently leaves the denominator.
MIN_BUCKET_TRAIN_EVENTS = 20

OUTCOMES = ("continues", "later_participation", "extent", "later_multiband")


@dataclass(frozen=True)
class PrefixData:
    features: np.ndarray             # (N, F) prefix descriptor
    bucket: np.ndarray               # (N,) discrete TRAIN-frozen bucket id
    bucket_labels: list[str]
    prefix_mask: np.ndarray          # (N, C) True where the contact is IN the prefix
    continues: np.ndarray            # (N,) any participation beyond the prefix
    later_participation: np.ndarray  # (N, C) participation of non-prefix contacts
    later_valid: np.ndarray          # (N, C) contact is scoreable (not in prefix)
    extent: np.ndarray               # (N, 2) size, span  (standardised on TRAIN)
    later_multiband: np.ndarray      # (N, 2B) later-participant energy / peak
    later_multiband_valid: np.ndarray  # (N,) the event has any later participant
    n_events: int


def _tied_groups(delay: np.ndarray, part: np.ndarray, tol: float) -> np.ndarray:
    """Group id per contact for one event; -1 for non-participants."""

    out = np.full(delay.shape, -1, dtype=np.int16)
    idx = np.flatnonzero(part & np.isfinite(delay))
    if idx.size == 0:
        return out
    order = idx[np.argsort(delay[idx], kind="stable")]
    g = 0
    out[order[0]] = 0
    for prev, cur in zip(order[:-1], order[1:]):
        if float(delay[cur] - delay[prev]) > tol:
            g += 1
        out[cur] = g
    return out


def build_prefix_data(
    participation: np.ndarray,
    relative_delay: np.ndarray,
    band_features: np.ndarray,
    waveform_early: np.ndarray,
    *,
    band_keep: np.ndarray,
    train_positions: np.ndarray,
    tie_tolerance_seconds: float,
) -> PrefixData:
    """Prefix descriptor, bucket and outcomes for every event.

    ``waveform_early`` is ``(N, C, V)`` summaries of the first
    ``EARLY_WINDOW_SECONDS`` of the event core -- the caller extracts them, since
    only it knows the montage layout and the sampling rate.
    """

    part = np.asarray(participation, dtype=bool)
    n, c = part.shape
    delay = np.asarray(relative_delay, dtype=np.float64)
    groups = np.stack([
        _tied_groups(delay[i], part[i], tie_tolerance_seconds) for i in range(n)
    ])

    in_prefix = (groups >= 0) & (groups < PREFIX_GROUPS)
    first = (groups == 0)
    second = (groups == 1)

    first_contact = np.full(n, -1, dtype=np.int64)
    has_first = first.any(1)
    first_contact[has_first] = np.argmax(first[has_first], axis=1)

    energy = np.asarray(band_features, dtype=np.float64)[:, :, band_keep, BAND_FEATURE_ENERGY]
    peak = np.asarray(band_features, dtype=np.float64)[:, :, band_keep, BAND_FEATURE_PEAK]

    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = mask[:, :, None] & np.isfinite(values)
        total = np.where(m, values, 0.0).sum(1)
        count = m.sum(1)
        return np.where(count > 0, total / np.maximum(count, 1), 0.0)

    early_energy = _masked_mean(energy, first)
    early_peak = _masked_mean(peak, first)
    wave = np.asarray(waveform_early, dtype=np.float64)
    early_wave = np.where(first[:, :, None], wave, 0.0).sum(1) / np.maximum(
        first.sum(1)[:, None], 1
    )

    step_delay = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if second[i].any() and first[i].any():
            step_delay[i] = float(delay[i, second[i]].min() - delay[i, first[i]].min())

    onehot = np.zeros((n, c), dtype=np.float64)
    ok = first_contact >= 0
    onehot[np.flatnonzero(ok), first_contact[ok]] = 1.0
    features = np.concatenate([
        onehot,
        first.astype(np.float64),
        second.astype(np.float64),
        first.sum(1, keepdims=True).astype(np.float64),
        second.sum(1, keepdims=True).astype(np.float64),
        step_delay[:, None],
        early_energy,
        early_peak,
        early_wave,
    ], axis=1)

    # Discrete bucket: which contact went first, and how many went with it.
    keys = np.array([
        f"c{int(fc)}|n{int(k)}" for fc, k in zip(first_contact, first.sum(1))
    ])
    train_keys, counts = np.unique(keys[np.asarray(train_positions, dtype=np.int64)],
                                   return_counts=True)
    kept = set(train_keys[counts >= MIN_BUCKET_TRAIN_EVENTS].tolist())
    labels = sorted(kept) + ["__rare__"]
    lookup = {name: i for i, name in enumerate(labels)}
    bucket = np.array([lookup.get(k, lookup["__rare__"]) for k in keys], dtype=np.int64)

    later = part & ~in_prefix
    later_valid = ~in_prefix
    continues = later.any(1)
    later_energy = _masked_mean(energy, later)
    later_peak = _masked_mean(peak, later)
    extent = np.stack([
        part.sum(1).astype(np.float64),
        np.nan_to_num(np.nanmax(np.where(part, delay, np.nan), axis=1)),
    ], axis=1)

    tr = np.asarray(train_positions, dtype=np.int64)
    def _std(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
        mu = x[rows].mean(0)
        sd = x[rows].std(0)
        return (x - mu) / np.where(sd > 1e-9, sd, 1.0)

    return PrefixData(
        features=features,
        bucket=bucket,
        bucket_labels=labels,
        prefix_mask=in_prefix,
        continues=continues,
        later_participation=later,
        later_valid=later_valid,
        extent=_std(extent, tr),
        later_multiband=_std(np.concatenate([later_energy, later_peak], 1), tr),
        later_multiband_valid=continues,
        n_events=n,
    )


# --------------------------------------------------------------------- fitting


def _bernoulli_nll(logit: Tensor, y: Tensor, mask: Tensor) -> tuple[Tensor, float]:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logit.double(), y.double(), reduction="none"
    )
    bce = torch.where(mask, bce, torch.zeros_like(bce))
    return bce.sum(), float(mask.double().sum())


def _gauss_nll(mu: Tensor, log_sigma: Tensor, y: Tensor, mask: Tensor) -> tuple[Tensor, float]:
    ls = log_sigma.double().clamp(-8.0, 8.0)
    nll = 0.5 * math.log(2 * math.pi) + ls + 0.5 * ((y.double() - mu.double()) ** 2) * torch.exp(-2 * ls)
    nll = torch.where(mask, nll, torch.zeros_like(nll))
    return nll.sum(), float(mask.double().sum())


@dataclass
class OutcomeFit:
    params: dict[str, Tensor]
    lam: float
    lam_at_grid_edge: bool
    mean: np.ndarray
    scale: np.ndarray
    kind: str


def _fit_outcome(
    x: np.ndarray, y: np.ndarray, mask: np.ndarray, kind: str, lam: float,
    max_iter: int,
) -> dict[str, Tensor]:
    z = torch.as_tensor(x, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    mt = torch.as_tensor(mask, dtype=torch.bool)
    out_dim = 1 if y.ndim == 1 else y.shape[1]
    yt = yt.reshape(yt.shape[0], out_dim)
    mt = mt.reshape(mt.shape[0], out_dim) if mt.ndim > 1 else mt.reshape(-1, 1).expand(-1, out_dim)

    if kind == "bernoulli":
        rate = np.clip(
            (np.asarray(y, dtype=float).reshape(-1, out_dim) *
             np.asarray(mask, dtype=float).reshape(yt.shape[0], -1)).sum(0)
            / np.maximum(np.asarray(mask, dtype=float).reshape(yt.shape[0], -1).sum(0), 1.0),
            1e-4, 1 - 1e-4,
        )
        b0 = np.log(rate / (1 - rate))
    else:
        b0 = np.asarray(y, dtype=float).reshape(-1, out_dim).mean(0)

    params = {
        "W": torch.zeros((z.shape[1], out_dim), dtype=torch.float64, requires_grad=True),
        "b": torch.as_tensor(np.asarray(b0, dtype=float).reshape(out_dim),
                             dtype=torch.float64).clone().requires_grad_(True),
    }
    if kind == "gaussian":
        sd = np.asarray(y, dtype=float).reshape(-1, out_dim).std(0)
        params["log_sigma"] = torch.as_tensor(
            np.log(np.maximum(sd, 1e-4)), dtype=torch.float64
        ).clone().requires_grad_(True)

    opt = torch.optim.LBFGS(list(params.values()), max_iter=max_iter, history_size=20,
                            line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        opt.zero_grad(set_to_none=True)
        pred = z @ params["W"] + params["b"]
        if kind == "bernoulli":
            total, units = _bernoulli_nll(pred, yt, mt)
        else:
            total, units = _gauss_nll(pred, params["log_sigma"], yt, mt)
        loss = total / max(units, 1.0) + float(lam) * (params["W"] ** 2).sum()
        if torch.isfinite(loss):
            loss.backward()
        return loss

    opt.step(closure)
    return {k: v.detach() for k, v in params.items()}


def _score_outcome(params: Mapping[str, Tensor], x: np.ndarray, y: np.ndarray,
                   mask: np.ndarray, kind: str) -> tuple[float, float]:
    z = torch.as_tensor(x, dtype=torch.float64)
    out_dim = 1 if y.ndim == 1 else y.shape[1]
    yt = torch.as_tensor(y, dtype=torch.float64).reshape(z.shape[0], out_dim)
    mt = torch.as_tensor(mask, dtype=torch.bool)
    mt = mt.reshape(z.shape[0], out_dim) if mt.ndim > 1 else mt.reshape(-1, 1).expand(-1, out_dim)
    with torch.no_grad():
        pred = z @ params["W"] + params["b"]
        if kind == "bernoulli":
            total, units = _bernoulli_nll(pred, yt, mt)
        else:
            total, units = _gauss_nll(pred, params["log_sigma"], yt, mt)
    return float(total) / max(units, 1.0), units


def fit_and_score_outcome(
    x_train: np.ndarray, y_train, m_train, x_test: np.ndarray, y_test, m_test,
    *, kind: str, config: ReadoutConfig = ReadoutConfig(),
) -> dict[str, Any]:
    """Ridge chosen by chronological CV inside TRAIN, then scored on held-out events."""

    mean, scale = _standardise(x_train)
    zt = (np.asarray(x_train, dtype=np.float64) - mean) / scale
    ze = (np.asarray(x_test, dtype=np.float64) - mean) / scale
    folds = _chronological_folds(zt.shape[0], config.cv_folds)
    best_lam, best_cv = None, math.inf
    path = []
    for lam in config.lambdas:
        total, units = 0.0, 0.0
        for held in folds:
            keep = np.setdiff1d(np.arange(zt.shape[0]), held)
            if keep.size < 2 or held.size < 1:
                continue
            p = _fit_outcome(zt[keep], np.asarray(y_train)[keep],
                             np.asarray(m_train)[keep], kind, lam, config.max_iter)
            nll, u = _score_outcome(p, zt[held], np.asarray(y_train)[held],
                                    np.asarray(m_train)[held], kind)
            total += nll * u
            units += u
        cv = total / units if units > 0 else math.inf
        path.append({"lambda": float(lam), "cv_nll_per_unit": cv})
        if cv < best_cv:
            best_cv, best_lam = cv, float(lam)
    params = _fit_outcome(zt, y_train, m_train, kind, best_lam, config.max_iter)
    nll, units = _score_outcome(params, ze, y_test, m_test, kind)
    return {
        "nll_per_unit": nll, "n_units": units, "lambda": best_lam,
        "lambda_at_grid_edge": best_lam in (config.lambdas[0], config.lambdas[-1]),
        "lambda_path": path,
    }
