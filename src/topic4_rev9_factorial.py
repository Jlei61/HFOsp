"""Small pure helpers for the exploratory rev9 four-arm experiment."""
from __future__ import annotations

import numpy as np


ARM_ORDER = ("Null", "Node", "Edge", "Node+Edge")


def arm_contract(arm):
    """Return the two factorial switches for a named arm."""
    if arm not in ARM_ORDER:
        raise ValueError(f"unknown rev9 factorial arm: {arm}")
    return dict(
        arm=arm,
        node=arm in {"Node", "Node+Edge"},
        edge=arm in {"Edge", "Node+Edge"},
    )


def event_equal_density(histograms):
    """Average event-normalized onset histograms without size weighting."""
    values = np.asarray(histograms, float)
    if values.ndim != 3:
        raise ValueError("histograms must have shape (event, y, x)")
    totals = values.sum(axis=(1, 2))
    valid = np.isfinite(values).all(axis=(1, 2)) & (totals > 0.0)
    if not valid.any():
        return np.zeros(values.shape[1:], float), 0
    normalized = values[valid] / totals[valid, None, None]
    return normalized.mean(axis=0), int(valid.sum())


def normalized_event_ranks(ranks):
    """Map each event's participating contact ranks to [0, 1]."""
    ranks = np.asarray(ranks, float)
    if ranks.ndim != 2:
        raise ValueError("ranks must have shape (event, contact)")
    output = np.full_like(ranks, np.nan)
    for index, row in enumerate(ranks):
        finite = np.isfinite(row)
        if finite.sum() < 2:
            continue
        spread = float(np.ptp(row[finite]))
        if spread <= 0.0:
            continue
        output[index, finite] = (row[finite] - row[finite].min()) / spread
    return output


def pairwise_precedence(ranks):
    """Probability that row-contact i precedes column-contact j."""
    ranks = np.asarray(ranks, float)
    if ranks.ndim != 2:
        raise ValueError("ranks must have shape (event, contact)")
    n_contact = ranks.shape[1]
    probability = np.full((n_contact, n_contact), np.nan)
    support = np.zeros((n_contact, n_contact), int)
    for left in range(n_contact):
        for right in range(n_contact):
            valid = np.isfinite(ranks[:, left]) & np.isfinite(ranks[:, right])
            support[left, right] = int(valid.sum())
            if valid.any():
                probability[left, right] = float(np.mean(
                    ranks[valid, left] < ranks[valid, right]))
    return probability, support


def _bootstrap_mean(values, seed, repeats):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(estimate=None, interval_95=[None, None], n_paired=0)
    estimate = float(values.mean())
    if len(values) == 1:
        interval = [estimate, estimate]
    else:
        rng = np.random.default_rng(int(seed))
        draws = rng.choice(values, size=(int(repeats), len(values)), replace=True)
        interval = np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()
    return dict(estimate=estimate, interval_95=interval,
                n_paired=int(len(values)))


def factorial_effects(values_by_arm, *, seed, repeats=2000):
    """Paired mean main effects and interaction over common network seeds."""
    if set(values_by_arm) != set(ARM_ORDER):
        raise ValueError("factorial effects require exactly the four rev9 arms")
    arrays = {key: np.asarray(values_by_arm[key], float) for key in ARM_ORDER}
    lengths = {len(value) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError("all arms must use the same ordered seed vector")
    raw = {
        "delta_node": arrays["Node"] - arrays["Null"],
        "delta_edge": arrays["Edge"] - arrays["Null"],
        "delta_node_edge": arrays["Node+Edge"] - arrays["Null"],
        "interaction": (
            arrays["Node+Edge"] - arrays["Node"]
            - arrays["Edge"] + arrays["Null"]),
    }
    return {
        key: dict(**_bootstrap_mean(value, int(seed) + index, repeats),
                  per_seed=value.tolist())
        for index, (key, value) in enumerate(raw.items())
    }
