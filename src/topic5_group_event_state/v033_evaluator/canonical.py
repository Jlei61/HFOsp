"""Canonical per-anchor proper score for Group-Event State v0.3.3 (plan Task 1).

The score is a *pure function* of ``(target, prediction, dispersion, mask,
weight)``.  Nothing is fitted here: no intercept, no dispersion, no ridge.  A
recalibration that a branch wants to compare against is a *declared arm*
(``extra_arms={"H_plus_intercept": ...}``), never a hidden correction (C1).

Both v0.3.2 branches used the same NB2 likelihood under two parameterisations
(model side ``log r``, evaluation side ``alpha = 1/r``); ``alpha_to_log_r``
pins the mapping (C3).  ``nb_nll`` (numpy) and ``nb_nll_torch`` (torch) are
the same formula in float64 (C2) so the training branch can call the torch
form and the evaluator the numpy form and still agree row by row.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.special import gammaln

SCHEMA_VERSION = "group_event_state_v0_3_3_canonical_per_anchor_1"
TOLERANCE_NATS = 1e-6
ARMS = ("H", "H_plus_state")
DISPERSION_RULES = ("shared", "per_arm")
REDUCTIONS = ("mean", "sum", "block_mean")
SCHEMA_COLUMNS = (
    "subject", "seed", "checkpoint_hash", "anchor_time", "split",
    "target", "prediction_H", "prediction_H_plus_state",
    "dispersion", "mask", "weight",
    "per_anchor_NLL_H", "per_anchor_NLL_H_plus_state",
    "eligibility", "evidence_label",
)
SIGN_CONVENTION = "gain = NLL(control) - NLL(treated); positive favours treated"


class EvaluatorDisagreement(RuntimeError):
    """Two branches scored the same object differently -- global hard stop #3."""


# --------------------------------------------------------------------------- likelihood
def alpha_to_log_r(alpha: float) -> float:
    """NB2 ``Var = mu + alpha mu^2`` (evaluation side) -> ``log r`` with ``r = 1/alpha``."""

    value = float(alpha)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("alpha must be a positive finite number")
    return float(-np.log(value))


def log_r_to_alpha(log_r: float) -> float:
    return float(np.exp(-float(log_r)))


def nb_nll(target: np.ndarray, log_mu: np.ndarray, log_r: float | np.ndarray) -> np.ndarray:
    """Per-row negative log-likelihood of ``y ~ NB(mu = exp(log_mu), r = exp(log_r))``.

    Float64 throughout; ``log(r + mu)`` is evaluated as ``logaddexp`` so large
    rates do not overflow.  ``log_r`` may be a scalar or one value per row.
    """

    y = np.asarray(target, dtype=np.float64)
    lm = np.asarray(log_mu, dtype=np.float64)
    if lm.shape != y.shape:
        raise ValueError("target and log_mu must have the same shape")
    lr = np.broadcast_to(np.asarray(log_r, dtype=np.float64), y.shape)
    r = np.exp(lr)
    lse = np.logaddexp(lr, lm)  # log(r + mu)
    ll = gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) + r * (lr - lse) + y * (lm - lse)
    return -ll


def nb_nll_torch(target, log_mu, log_r):
    """Torch twin of :func:`nb_nll` (same formula, float64) for the training branch."""

    import torch

    y = target.to(torch.float64)
    lm = log_mu.to(torch.float64)
    if lm.shape != y.shape:
        raise ValueError("target and log_mu must have the same shape")
    lr = torch.as_tensor(log_r, dtype=torch.float64, device=y.device)
    lr = lr.expand_as(y) if lr.ndim == 0 else lr.to(torch.float64)
    r = torch.exp(lr)
    lse = torch.logaddexp(lr, lm)
    ll = torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0) + r * (lr - lse) + y * (lm - lse)
    return -ll


def conditional_subset_nll(logits: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """``-log P(S | |S| = K)`` for independent Bernoulli contacts with the given logits (rows vectorised).

    The single grammar proper score: independent-Bernoulli participation
    conditioned on the observed size ``K`` (Poisson-binomial normaliser by a
    log-space DP over contacts).  Float64 throughout.
    """

    x = np.asarray(subset, dtype=bool)
    lg = np.asarray(logits, dtype=np.float64)
    if lg.ndim != 2 or lg.shape != x.shape:
        raise ValueError("logits and subset must both be (E, C)")
    e, c = lg.shape
    logp = -np.logaddexp(0.0, -lg)
    log1mp = -np.logaddexp(0.0, lg)
    joint = np.where(x, logp, log1mp).sum(axis=1)
    k = x.sum(axis=1)
    kmax = int(k.max()) if e else 0
    dp = np.full((e, kmax + 1), -np.inf)
    dp[:, 0] = 0.0
    for j in range(c):
        new = dp + log1mp[:, j:j + 1]
        if kmax:
            new[:, 1:] = np.logaddexp(new[:, 1:], dp[:, :-1] + logp[:, j:j + 1])
        dp = new
    return -(joint - dp[np.arange(e), k])


# --------------------------------------------------------------------------- table
def _column(values: Any, name: str, n: int, dtype=np.float64) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _broadcast_label(value: Any, name: str, n: int) -> np.ndarray:
    if isinstance(value, (str, bytes)) or np.ndim(value) == 0:
        return np.full(n, value, dtype=object)
    arr = np.asarray(value, dtype=object).reshape(-1)
    if arr.shape != (n,):
        raise ValueError(f"{name} must be a scalar or have shape ({n},)")
    return arr


def _resolve_dispersion(dispersion: Any, rule: str, arms: tuple[str, ...], n: int):
    if rule not in DISPERSION_RULES:
        raise ValueError(f"dispersion_rule must be one of {DISPERSION_RULES}, got {rule!r}")
    if rule == "shared":
        if isinstance(dispersion, Mapping):
            missing = [arm for arm in arms if arm not in dispersion]
            if missing:
                raise ValueError(f"shared dispersion rule is missing log_r for arms {missing}")
            values = [float(dispersion[arm]) for arm in arms]
            if not values or (max(values) - min(values)) > 1e-12:
                raise ValueError("shared dispersion rule requires one identical log_r for every arm")
            shared = values[0]
        else:
            shared = float(dispersion)
        if not np.isfinite(shared):
            raise ValueError("log_r must be finite")
        return {arm: shared for arm in arms}, np.full(n, shared, dtype=np.float64)
    if not isinstance(dispersion, Mapping):
        raise ValueError("per_arm dispersion rule requires a mapping arm -> log_r")
    missing = [arm for arm in arms if arm not in dispersion]
    if missing:
        raise ValueError(f"per_arm dispersion rule is missing log_r for arms {missing}")
    log_r = {arm: float(dispersion[arm]) for arm in arms}
    if not all(np.isfinite(v) for v in log_r.values()):
        raise ValueError("log_r must be finite for every arm")
    return log_r, np.column_stack([np.full(n, log_r[arm], dtype=np.float64) for arm in arms])


def build_per_anchor_table(
    *,
    subject: Any,
    seed: Any,
    checkpoint_hash: Any,
    split: Any,
    anchor_time: np.ndarray,
    target: np.ndarray,
    prediction_H: np.ndarray,
    prediction_H_plus_state: np.ndarray,
    dispersion: float | Mapping[str, float],
    mask: np.ndarray | None,
    weight: np.ndarray | None,
    eligibility: Any,
    evidence_label: Any,
    dispersion_rule: str = "shared",
    extra_arms: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """One row per anchor; ``per_anchor_NLL_<arm>`` for ``H``, ``H_plus_state`` and extra arms.

    ``dispersion`` is ``log r``: a scalar under ``"shared"``, a mapping
    ``arm -> log_r`` covering every arm under ``"per_arm"``.  Masked rows keep
    NaN scores and are excluded from every reduction (C5).
    """

    t = np.asarray(anchor_time, dtype=np.float64).reshape(-1)
    n = int(t.size)
    y = _column(target, "target", n)
    if not np.isfinite(y).all() or (y < 0).any() or not np.equal(y, np.floor(y)).all():
        raise ValueError("count target must be finite, non-negative and integer-valued")
    predictions: dict[str, np.ndarray] = {
        "H": _column(prediction_H, "prediction_H", n),
        "H_plus_state": _column(prediction_H_plus_state, "prediction_H_plus_state", n),
    }
    for name, values in (extra_arms or {}).items():
        if name in predictions:
            raise ValueError(f"extra arm {name!r} collides with a schema arm")
        predictions[str(name)] = _column(values, f"extra_arms[{name}]", n)
    arms = tuple(predictions)
    log_r, dispersion_column = _resolve_dispersion(dispersion, dispersion_rule, arms, n)
    m = np.ones(n, dtype=bool) if mask is None else _column(mask, "mask", n, dtype=bool)
    for arm, prediction in predictions.items():
        bad = np.flatnonzero(m & ~np.isfinite(prediction))
        if bad.size:
            raise ValueError(f"unmasked non-finite prediction for arm {arm} at row {int(bad[0])}")
    if weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = _column(weight, "weight", n)
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weight must be finite and non-negative")
    table: dict[str, Any] = {
        "subject": _broadcast_label(subject, "subject", n),
        "seed": _broadcast_label(seed, "seed", n),
        "checkpoint_hash": _broadcast_label(checkpoint_hash, "checkpoint_hash", n),
        "anchor_time": t,
        "split": _broadcast_label(split, "split", n),
        "target": y,
        "prediction_H": predictions["H"],
        "prediction_H_plus_state": predictions["H_plus_state"],
        "dispersion": dispersion_column,
        "mask": m,
        "weight": w,
        "eligibility": _broadcast_label(eligibility, "eligibility", n),
        "evidence_label": _broadcast_label(evidence_label, "evidence_label", n),
    }
    for arm in arms:
        if arm not in ARMS:
            table[f"prediction_{arm}"] = predictions[arm]
        nll = np.full(n, np.nan, dtype=np.float64)
        if m.any():
            nll[m] = nb_nll(y[m], predictions[arm][m], log_r[arm])
        table[f"per_anchor_NLL_{arm}"] = nll
    table["meta"] = {
        "schema_version": SCHEMA_VERSION,
        "score_family": "nb_count_nll",
        "dispersion_rule": dispersion_rule,
        "arms": list(arms),
        "log_r": dict(log_r),
        "weights_used": weight is not None,
        "n_rows": n,
        "n_masked": int((~m).sum()),
        "sign_convention": SIGN_CONVENTION,
    }
    return table


def build_per_anchor_table_from_scores(
    *,
    subject: Any,
    seed: Any,
    checkpoint_hash: Any,
    split: Any,
    anchor_time: np.ndarray,
    target: np.ndarray,
    per_anchor_nll: Mapping[str, np.ndarray],
    score_family: str,
    mask: np.ndarray | None,
    weight: np.ndarray | None,
    eligibility: Any,
    evidence_label: Any,
    extra_nll: Mapping[str, np.ndarray] | None = None,
    prediction_H: np.ndarray | None = None,
    prediction_H_plus_state: np.ndarray | None = None,
    dispersion: np.ndarray | None = None,
    dispersion_rule: str = "not_applicable",
) -> dict[str, Any]:
    """Same schema, mask, weight and reductions for a score family that is not a scalar NB prediction.

    ``per_anchor_nll`` must carry ``H`` and ``H_plus_state``. Prediction and
    dispersion payloads may be supplied for parity auditing; otherwise those
    columns are NaN. ``target`` is the family's per-anchor target summary.
    """

    t = np.asarray(anchor_time, dtype=np.float64).reshape(-1)
    n = int(t.size)
    missing = [arm for arm in ARMS if arm not in per_anchor_nll]
    if missing:
        raise ValueError(f"per_anchor_nll is missing arms {missing}")
    y = _column(target, "target", n)
    if not np.isfinite(y).all() or (y < 0).any() or not np.equal(y, np.floor(y)).all():
        raise ValueError("score-table target summary must be finite, non-negative and integer-valued")
    m = np.ones(n, dtype=bool) if mask is None else _column(mask, "mask", n, dtype=bool)
    if weight is None:
        w = np.ones(n, dtype=np.float64)
    else:
        w = _column(weight, "weight", n)
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weight must be finite and non-negative")
    table: dict[str, Any] = {
        "subject": _broadcast_label(subject, "subject", n),
        "seed": _broadcast_label(seed, "seed", n),
        "checkpoint_hash": _broadcast_label(checkpoint_hash, "checkpoint_hash", n),
        "anchor_time": t,
        "split": _broadcast_label(split, "split", n),
        "target": y,
        "prediction_H": np.full(n, np.nan) if prediction_H is None else np.asarray(prediction_H, dtype=np.float64),
        "prediction_H_plus_state": (np.full(n, np.nan) if prediction_H_plus_state is None
                                     else np.asarray(prediction_H_plus_state, dtype=np.float64)),
        "dispersion": np.full(n, np.nan) if dispersion is None else np.asarray(dispersion, dtype=np.float64),
        "mask": m,
        "weight": w,
        "eligibility": _broadcast_label(eligibility, "eligibility", n),
        "evidence_label": _broadcast_label(evidence_label, "evidence_label", n),
    }
    for key in ("prediction_H", "prediction_H_plus_state", "dispersion"):
        if np.asarray(table[key]).shape[0] != n:
            raise ValueError(f"{key} must have one leading row per anchor")
    arms = dict(per_anchor_nll)
    for name, values in (extra_nll or {}).items():
        if name in arms:
            raise ValueError(f"extra arm {name!r} collides with a schema arm")
        arms[str(name)] = values
    for arm, values in arms.items():
        nll = _column(values, f"per_anchor_nll[{arm}]", n).copy()
        bad = np.flatnonzero(m & ~np.isfinite(nll))
        if bad.size:
            raise ValueError(f"unmasked non-finite NLL for arm {arm} at row {int(bad[0])}")
        nll[~m] = np.nan
        table[f"per_anchor_NLL_{arm}"] = nll
    table["meta"] = {
        "schema_version": SCHEMA_VERSION,
        "score_family": str(score_family),
        "dispersion_rule": str(dispersion_rule),
        "arms": list(arms),
        "weights_used": weight is not None,
        "n_rows": n,
        "n_masked": int((~m).sum()),
        "sign_convention": SIGN_CONVENTION,
    }
    return table


def table_arms(table: Mapping[str, Any]) -> list[str]:
    prefix = "per_anchor_NLL_"
    return [key[len(prefix):] for key in table if key.startswith(prefix)]


# --------------------------------------------------------------------------- reduction
def paired_gain(
    table: Mapping[str, Any],
    *,
    control: str = "H",
    treated: str = "H_plus_state",
    reduction: str = "mean",
    block: np.ndarray | None = None,
) -> dict[str, Any]:
    """``gain = NLL(control) - NLL(treated)`` over unmasked rows; positive favours treated.

    An unmasked non-finite score is a contract failure, never silently dropped.

    Reductions: ``mean`` (weighted mean), ``sum`` (weighted sum), ``block_mean``
    (mean over blocks of the per-block weighted means; ``block`` ids required).
    """

    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction must be one of {REDUCTIONS}, got {reduction!r}")
    for arm in (control, treated):
        if f"per_anchor_NLL_{arm}" not in table:
            raise ValueError(f"arm {arm!r} is not in the table; available {table_arms(table)}")
    c = np.asarray(table[f"per_anchor_NLL_{control}"], dtype=np.float64)
    t = np.asarray(table[f"per_anchor_NLL_{treated}"], dtype=np.float64)
    m = np.asarray(table["mask"], dtype=bool)
    w = np.asarray(table["weight"], dtype=np.float64)
    n = int(c.size)
    finite = np.isfinite(c) & np.isfinite(t)
    bad = np.flatnonzero(m & ~finite)
    if bad.size:
        raise ValueError(f"unmasked non-finite paired score at row {int(bad[0])}")
    used = m
    out: dict[str, Any] = {
        "control": control,
        "treated": treated,
        "reduction": reduction,
        "sign_convention": SIGN_CONVENTION,
        "n_rows_total": n,
        "n_rows_used": int(used.sum()),
        "n_rows_masked": int((~m).sum()),
        "n_rows_nonfinite": 0,
        "weights_used": bool(table["meta"]["weights_used"]),
    }
    if reduction == "block_mean":
        if block is None:
            raise ValueError("block_mean reduction needs block ids")
        b = _column(block, "block", n, dtype=np.int64)
    if not used.any():
        out.update({"gain": None, "direction": "not_estimable", "mean_nll_control": None,
                    "mean_nll_treated": None})
        return out
    g = c[used] - t[used]
    ww = w[used]
    if ww.sum() <= 0:
        raise ValueError("weights of the used rows sum to zero")
    if reduction == "mean":
        value = float((ww * g).sum() / ww.sum())
    elif reduction == "sum":
        value = float((ww * g).sum())
    else:
        unique_blocks, inverse = np.unique(b[used], return_inverse=True)
        block_means = np.bincount(inverse, weights=ww * g) / np.bincount(inverse, weights=ww)
        value = float(block_means.mean())
        out["n_blocks"] = int(unique_blocks.size)
        out["block_means"] = block_means.tolist()
        out["fraction_blocks_positive"] = float(np.mean(block_means > 0.0))
    out["gain"] = value
    out["direction"] = "favours_treated" if value > 0 else ("favours_control" if value < 0 else "tie")
    out["mean_nll_control"] = float((ww * c[used]).sum() / ww.sum())
    out["mean_nll_treated"] = float((ww * t[used]).sum() / ww.sum())
    return out


# --------------------------------------------------------------------------- hard stop
def assert_tables_agree(a: Mapping[str, Any], b: Mapping[str, Any], *,
                        tolerance: float = TOLERANCE_NATS) -> None:
    """Raise :class:`EvaluatorDisagreement` unless both tables score the same object identically.

    Same object = same row labels, anchors, targets, predictions, dispersion,
    mask and weights. Per-anchor NLL of every arm must agree within
    ``tolerance`` nats; the first offending row is named.
    """

    n = len(a["anchor_time"])
    if len(b["anchor_time"]) != n:
        raise EvaluatorDisagreement(f"row counts differ: {n} vs {len(b['anchor_time'])}")
    for key in ("subject", "seed", "checkpoint_hash", "anchor_time", "split", "target", "mask",
                "eligibility", "evidence_label"):
        x = np.asarray(a[key])
        y = np.asarray(b[key])
        if not np.array_equal(x, y):
            row = int(np.flatnonzero(x != y)[0])
            raise EvaluatorDisagreement(
                f"{key} differs at row {row} ({x[row]!r} vs {y[row]!r}): not the same object")
    for key in ("prediction_H", "prediction_H_plus_state", "dispersion", "weight"):
        x = np.asarray(a[key], dtype=np.float64)
        y = np.asarray(b[key], dtype=np.float64)
        if x.shape != y.shape:
            raise EvaluatorDisagreement(f"{key} shapes differ: {x.shape} vs {y.shape}")
        ok = np.isclose(x, y, atol=tolerance, rtol=0.0, equal_nan=True)
        bad = np.flatnonzero(~ok.reshape(-1))
        if bad.size:
            i = int(bad[0])
            raise EvaluatorDisagreement(f"{key} differs at flattened row {i}: {x.reshape(-1)[i]!r} vs {y.reshape(-1)[i]!r}")
    for key in ("score_family", "dispersion_rule", "weights_used"):
        if a.get("meta", {}).get(key) != b.get("meta", {}).get(key):
            raise EvaluatorDisagreement(f"meta.{key} differs: {a.get('meta', {}).get(key)!r} vs {b.get('meta', {}).get(key)!r}")
    arms_a, arms_b = sorted(table_arms(a)), sorted(table_arms(b))
    if arms_a != arms_b:
        raise EvaluatorDisagreement(f"arm sets differ: {arms_a} vs {arms_b}")
    for arm in arms_a:
        x = np.asarray(a[f"per_anchor_NLL_{arm}"], dtype=np.float64)
        y = np.asarray(b[f"per_anchor_NLL_{arm}"], dtype=np.float64)
        both_nan = np.isnan(x) & np.isnan(y)
        diff = np.abs(x - y)
        bad = np.flatnonzero(~(both_nan | (diff <= tolerance)))
        if bad.size:
            i = int(bad[0])
            raise EvaluatorDisagreement(
                f"per-anchor NLL for arm {arm} differs by {diff[i]:.3e} nats (> {tolerance:g}) "
                f"at row {i} (anchor_time={a['anchor_time'][i]!r}): {x[i]!r} vs {y[i]!r}")
