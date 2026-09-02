"""Block-level inference: anchors overlap, blocks do not.

A 5-minute anchor grid with 30-minute windows shares 83% of its content with
its neighbour.  Every uncertainty statement therefore resamples *blocks* of
covered time inside one segment, never individual anchors.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def block_ids_for_times(
    times: np.ndarray,
    segment_of: np.ndarray,
    segment_start: Mapping[int, float],
    block_seconds: float,
) -> np.ndarray:
    """Dense integer block id: consecutive ``block_seconds`` bins inside a segment."""

    t = np.asarray(times, dtype=np.float64)
    seg = np.asarray(segment_of, dtype=np.int64)
    if t.shape != seg.shape:
        raise ValueError("times and segment_of must align")
    starts = np.asarray([float(segment_start[int(s)]) for s in seg], dtype=np.float64)
    bin_index = np.floor((t - starts) / float(block_seconds)).astype(np.int64)
    if (bin_index < 0).any():
        raise ValueError("a time precedes its own segment start")
    keys = np.stack([seg, bin_index], axis=1)
    _unique, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64)


def block_bootstrap_mean_ci(
    values: np.ndarray,
    blocks: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Moving-block bootstrap of the mean: resample whole blocks with replacement."""

    v = np.asarray(values, dtype=np.float64)
    b = np.asarray(blocks, dtype=np.int64)
    finite = np.isfinite(v)
    v = v[finite]
    b = b[finite]
    if v.size == 0:
        return {"mean": None, "lower": None, "upper": None, "n_blocks": 0,
                "n_values": 0, "n_boot": int(n_boot), "seed": int(seed)}
    unique_blocks, inverse = np.unique(b, return_inverse=True)
    n_blocks = unique_blocks.size
    block_sums = np.bincount(inverse, weights=v, minlength=n_blocks)
    block_sizes = np.bincount(inverse, minlength=n_blocks).astype(np.float64)
    mean = float(v.mean())
    if n_blocks < 2:
        return {"mean": mean, "lower": None, "upper": None, "n_blocks": int(n_blocks),
                "n_values": int(v.size), "n_boot": int(n_boot), "seed": int(seed)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_blocks, size=(int(n_boot), n_blocks))
    sums = block_sums[draws].sum(axis=1)
    sizes = block_sizes[draws].sum(axis=1)
    replicates = sums / np.maximum(sizes, 1.0)
    lo = float(np.quantile(replicates, (1.0 - ci) / 2.0))
    hi = float(np.quantile(replicates, 1.0 - (1.0 - ci) / 2.0))
    return {"mean": mean, "lower": lo, "upper": hi, "n_blocks": int(n_blocks),
            "n_values": int(v.size), "n_boot": int(n_boot), "seed": int(seed)}


def paired_gain_summary(
    control_nll: np.ndarray,
    treated_nll: np.ndarray,
    blocks: np.ndarray,
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """gain = control NLL - treated NLL on every finite pair; positive favours treated.

    Every finite pair is kept.  Missing pairs are reported, never imputed, and
    no audit quantity may remove a pair after the fact.
    """

    c = np.asarray(control_nll, dtype=np.float64)
    t = np.asarray(treated_nll, dtype=np.float64)
    b = np.asarray(blocks, dtype=np.int64)
    if c.shape != t.shape or c.shape != b.shape:
        raise ValueError("control, treated and blocks must align")
    finite = np.isfinite(c) & np.isfinite(t)
    gain = c[finite] - t[finite]
    if gain.size == 0:
        return {"n_pairs": 0, "n_missing_pairs": int((~finite).sum()), "mean_gain": None,
                "median_gain": None, "lower": None, "upper": None, "n_blocks": 0,
                "fraction_blocks_positive": None, "direction": "not_estimable",
                "ci_excludes_zero": None}
    boot = block_bootstrap_mean_ci(gain, b[finite], n_boot=n_boot, seed=seed)
    unique_blocks, inverse = np.unique(b[finite], return_inverse=True)
    block_means = np.bincount(inverse, weights=gain) / np.bincount(inverse)
    mean_gain = float(gain.mean())
    lower, upper = boot["lower"], boot["upper"]
    excludes = None if lower is None else bool(lower > 0.0 or upper < 0.0)
    return {
        "n_pairs": int(gain.size),
        "n_missing_pairs": int((~finite).sum()),
        "mean_gain": mean_gain,
        "median_gain": float(np.median(gain)),
        "lower": lower,
        "upper": upper,
        "n_blocks": int(unique_blocks.size),
        "fraction_blocks_positive": float(np.mean(block_means > 0.0)),
        "direction": "favours_treated" if mean_gain > 0 else ("favours_control" if mean_gain < 0 else "tie"),
        "ci_excludes_zero": excludes,
        "bootstrap": {"n_boot": int(n_boot), "seed": int(seed)},
    }


def cohort_patient_summary(values: Mapping[str, float]) -> dict[str, Any]:
    """Patient-level cohort statistics; patients (not seeds/anchors) are the units."""

    from scipy.stats import binomtest, wilcoxon

    finite = {k: float(v) for k, v in values.items() if v is not None and np.isfinite(float(v))}
    n = len(finite)
    out: dict[str, Any] = {"n_patients": n, "patients": sorted(finite)}
    if n == 0:
        out.update({"median": None, "n_positive": 0, "sign_test_p": None, "wilcoxon_p": None})
        return out
    arr = np.asarray(list(finite.values()), dtype=np.float64)
    n_pos = int((arr > 0).sum())
    n_nonzero = int((arr != 0).sum())
    out["median"] = float(np.median(arr))
    out["n_positive"] = n_pos
    out["sign_test_p"] = float(binomtest(n_pos, n_nonzero, 0.5).pvalue) if n_nonzero else None
    if n_nonzero >= 2:
        try:
            out["wilcoxon_p"] = float(wilcoxon(arr[arr != 0]).pvalue) if n_nonzero >= 2 else None
        except ValueError:
            out["wilcoxon_p"] = None
    else:
        out["wilcoxon_p"] = None
    return out
