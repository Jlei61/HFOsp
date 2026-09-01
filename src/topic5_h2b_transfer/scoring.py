"""Scoring for H2b: discrete-time seizure survival and early-field agreement.

The censoring arithmetic is the load-bearing part. A grid anchor whose
monitoring ended early has *not* told us "no seizure for six hours"; it has told
us only about the bins it actually survived. Every function here therefore takes
``last_observed_bin`` alongside ``outcome_bin`` and refuses to score past it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import spearmanr

#: Keeps a confidently wrong hazard finite instead of -inf.
HAZARD_EPS = 1e-6


def _prep(hazards, outcome_bin, last_observed_bin, censored):
    h = np.clip(np.asarray(hazards, float), HAZARD_EPS, 1.0 - HAZARD_EPS)
    ob = list(outcome_bin)
    lb = np.asarray(list(last_observed_bin), int)
    cs = np.asarray(list(censored), bool)
    if not (h.shape[0] == len(ob) == lb.size == cs.size):
        raise ValueError("hazards, outcome_bin, last_observed_bin, censored must align")
    for i, b in enumerate(ob):
        # An event may legitimately fall in the first *partially* observed bin:
        # every earlier bin was survived and the event itself was seen. Only an
        # event separated from the observed span by a whole unobserved bin is
        # inconsistent.
        if b is not None and b > lb[i] + 1:
            raise ValueError(
                f"row {i}: event in bin {b} lies beyond the last observed bin {lb[i]}"
            )
    return h, ob, lb, cs


def discrete_time_log_score(
    hazards: np.ndarray,
    outcome_bin: Sequence[int | None],
    last_observed_bin: Sequence[int],
    censored: Sequence[bool],
) -> np.ndarray:
    """Per-row discrete-time survival log-likelihood.

    Event in bin ``j``      -> ``log h_j + sum_{k<j} log(1 - h_k)``
    Censored after bin ``m``-> ``sum_{k<=m} log(1 - h_k)``  (nothing beyond ``m``)
    """

    h, ob, lb, _cs = _prep(hazards, outcome_bin, last_observed_bin, censored)
    out = np.zeros(h.shape[0], float)
    for i, b in enumerate(ob):
        m = lb[i]
        if m < 0:
            continue  # observed nothing: contributes no information
        if b is None:
            out[i] = float(np.sum(np.log1p(-h[i, : m + 1])))
        else:
            out[i] = float(np.sum(np.log1p(-h[i, :b])) + np.log(h[i, b]))
    return out


def brier_by_bin(
    hazards: np.ndarray,
    outcome_bin: Sequence[int | None],
    last_observed_bin: Sequence[int],
    censored: Sequence[bool],
) -> np.ndarray:
    """Mean Brier score per bin, over the rows genuinely at risk in that bin."""

    h, ob, lb, _cs = _prep(hazards, outcome_bin, last_observed_bin, censored)
    n_bins = h.shape[1]
    out = np.full(n_bins, np.nan)
    for k in range(n_bins):
        vals = []
        for i, b in enumerate(ob):
            if lb[i] < k:
                continue  # bin k was never observed for this row
            if b is not None and b < k:
                continue  # already had its event; no longer at risk
            y = 1.0 if b == k else 0.0
            vals.append((h[i, k] - y) ** 2)
        if vals:
            out[k] = float(np.mean(vals))
    return out


def nested_increment(baseline: np.ndarray, full: np.ndarray) -> dict:
    """Paired gain of ``full`` over ``baseline`` on identical rows."""

    b = np.asarray(baseline, float)
    f = np.asarray(full, float)
    if b.shape != f.shape:
        raise ValueError("nested arms must be scored on the same rows")
    d = f - b
    ok = np.isfinite(d)
    if not ok.any():
        return {"mean_gain": float("nan"), "median_gain": float("nan"), "n": 0,
                "n_positive": 0}
    return {
        "mean_gain": float(np.mean(d[ok])),
        "median_gain": float(np.median(d[ok])),
        "n": int(ok.sum()),
        "n_positive": int((d[ok] > 0).sum()),
    }


def field_score(predicted: np.ndarray, observed: np.ndarray, min_contacts: int = 4) -> float:
    """Rank agreement between a predicted and an observed early ictal field.

    Rank-based so the score reflects *which contacts* lead, not the overall
    amplitude of the seizure. Contacts missing on either side are dropped.
    """

    p = np.asarray(predicted, float)
    o = np.asarray(observed, float)
    ok = np.isfinite(p) & np.isfinite(o)
    if ok.sum() < min_contacts:
        return float("nan")
    r = spearmanr(p[ok], o[ok]).statistic
    return float(r) if np.isfinite(r) else float("nan")
