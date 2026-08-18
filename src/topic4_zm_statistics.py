"""Pre-registered statistics for the Z/M ictal-transition round."""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_bootstrap(a, b, *, draws, seed):
    """Resample NETWORK indices, not values independently -- the design is paired."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align")
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if not len(a):
        return {"status": "NOT_EVALUABLE", "n": 0}
    difference = a - b
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, len(difference), size=(int(draws), len(difference)))
    sampled = difference[index].mean(axis=1)
    return {"status": "OK", "n": int(len(difference)),
            "mean_difference": float(difference.mean()),
            "q05": float(np.quantile(sampled, 0.05)),
            "q50": float(np.quantile(sampled, 0.50)),
            "q95": float(np.quantile(sampled, 0.95)),
            "n_positive": int(np.sum(difference > 0))}


def phase2_decision(interval):
    """Three-way and directional. `q05 > 0` is the ONLY continue branch."""
    q05, q95 = float(interval["q05"]), float(interval["q95"])
    if q05 > 0.0:
        return {"action": "continue", "reason": "resolved_upward",
                "permitted_wording": "pre-ictal susceptibility is higher than baseline"}
    if q95 < 0.0:
        return {"action": "stop", "reason": "opposite_direction",
                "permitted_wording": ("pre-ictal susceptibility is LOWER than baseline "
                                      "-- the opposite of the hypothesised direction")}
    return {"action": "stop", "reason": "unresolved",
            "permitted_wording": ("the change was unresolved at this n; this is a "
                                  "statement about resolution, not about absence")}


def restricted_ictal_free_time(onset_ms, *, cap_ms):
    """The only latency number that may be compared across arms under censoring."""
    values, censored = [], 0
    for value in onset_ms:
        if value is None or not np.isfinite(value):
            values.append(float(cap_ms))
            censored += 1
        else:
            values.append(min(float(value), float(cap_ms)))
    values = np.asarray(values, float)
    return {"restricted_mean_ms": float(values.mean()), "n": int(len(values)),
            "n_censored": int(censored), "cap_ms": float(cap_ms),
            "entered_fraction": float((len(values) - censored) / len(values)),
            "note": "cap_ms is a censoring bound, never an onset time"}


def paired_onset_difference(onset_a, onset_b):
    a, b, dropped = [], [], 0
    for left, right in zip(onset_a, onset_b):
        if left is None or right is None or not np.isfinite(left) or not np.isfinite(right):
            dropped += 1
            continue
        a.append(float(left)); b.append(float(right))
    if not a:
        return {"status": "NOT_EVALUABLE", "n": 0, "n_dropped": int(dropped)}
    difference = np.asarray(a) - np.asarray(b)
    return {"status": "OK", "n": int(len(difference)), "n_dropped": int(dropped),
            "mean_difference_ms": float(difference.mean()),
            "median_difference_ms": float(np.median(difference))}


def exact_toroidal_shifts(grid_n):
    grid_n = int(grid_n)
    dx, dy = np.meshgrid(np.arange(grid_n), np.arange(grid_n), indexing="ij")
    return np.stack([dx.ravel(), dy.ravel()], axis=-1)


def spatial_correlation_exact_shift(values, covariate, *, grid_n):
    """Rigid toroidal shifts, ENUMERATED not sampled.

    Rigid shifts preserve the covariate's spatial autocorrelation, which a plain
    permutation would destroy, making it anticonservative. The group is small
    enough to enumerate, so there is no `draws`, no seed, and the p-value floor
    is exactly 1/grid_n**2 -- reported, so nobody reads more precision into the
    per-network p than the design has.
    """
    grid_n = int(grid_n)
    values = np.asarray(values, float)
    covariate = np.asarray(covariate, float)
    if values.shape != (grid_n * grid_n,) or covariate.shape != values.shape:
        raise ValueError(f"both fields must be flat length {grid_n * grid_n}")
    finite = np.isfinite(values) & np.isfinite(covariate)
    if finite.sum() < 4:
        return {"status": "NOT_EVALUABLE", "n_sites": int(finite.sum())}
    observed = float(stats.spearmanr(values[finite], covariate[finite]).statistic)
    field = covariate.reshape(grid_n, grid_n)
    null = []
    for dx, dy in exact_toroidal_shifts(grid_n):
        shifted = np.roll(np.roll(field, int(dx), axis=0), int(dy), axis=1).ravel()
        null.append(float(stats.spearmanr(values[finite], shifted[finite]).statistic))
    null = np.asarray(null, float)
    return {"status": "OK", "spearman_r": observed,
            "n_distinct_shifts": int(grid_n * grid_n),
            "p_floor": 1.0 / float(grid_n * grid_n),
            "p_value": float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
            "null_r": null.tolist(), "n_sites": int(finite.sum())}


def covariate_collinearity(covariates):
    """Reports, decides nothing.

    `h` and local recruitment time are the primary spatial covariates by design,
    fixed before any run; the pathway gains are descriptive companions. A
    data-dependent merge rule would be a degree of freedom, and the composite it
    implied was never defined -- so this returns the table and stops.
    """
    names = sorted(covariates)
    pairwise = {}
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            a = np.asarray(covariates[left], float)
            b = np.asarray(covariates[right], float)
            finite = np.isfinite(a) & np.isfinite(b)
            pairwise[(left, right)] = float(
                stats.spearmanr(a[finite], b[finite]).statistic)
    return {"pairwise_spearman": pairwise,
            "max_abs_r": float(max(abs(v) for v in pairwise.values())) if pairwise else 0.0,
            "note": ("descriptive only; h and local recruitment time are the primary "
                     "spatial covariates by design, not by these numbers")}
