"""PR-7: Template antagonistic temporal pairing.

Pure statistical layer. Tests whether forward/reverse template pairs
(PR-2.5 + PR-6 Step 4) are temporally coupled at short scales, or whether
they coexist as independent slow-modulated streams.

Contract: docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, wilcoxon


# ---------------------------------------------------------------------------
# Core lift estimator (block-aware to prevent cross-block spurious counts)
# ---------------------------------------------------------------------------
def _assign_blocks(
    times: np.ndarray, block_time_ranges: Sequence[Tuple[float, float]]
) -> np.ndarray:
    """Map each event to its block index, or -1 if outside every block."""
    out = np.full(times.size, -1, dtype=int)
    for b_idx, (b_start, b_end) in enumerate(block_time_ranges):
        in_block = (times >= b_start) & (times <= b_end)
        out[in_block] = b_idx
    return out


def compute_pairing_lift(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
    delta_t_seconds: float,
    block_time_ranges: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """Per-anchor opposite/same counts in (t_i, t_i + Δt] within the same block.

    Returns dict with p_opposite / p_same / n_used (all anchors used,
    including those whose window is empty after block clipping).
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    n = times.size
    if n == 0:
        return {"p_opposite": 0.0, "p_same": 0.0, "n_used": 0}

    sort_idx = np.argsort(times, kind="stable")
    times_s = times[sort_idx]
    labels_s = labels[sort_idx]
    block_s = _assign_blocks(times_s, block_time_ranges)

    block_ends = {b_idx: b_end for b_idx, (_, b_end) in enumerate(block_time_ranges)}

    n_opp = 0
    n_same = 0
    n_anchors = 0
    for i in range(n):
        if block_s[i] < 0:
            continue
        n_anchors += 1
        b_end = block_ends[int(block_s[i])]
        window_end = min(times_s[i] + delta_t_seconds, b_end)
        if window_end <= times_s[i]:
            continue
        hi = int(np.searchsorted(times_s, window_end, side="right"))
        for j in range(i + 1, hi):
            if block_s[j] != block_s[i]:
                continue
            if labels_s[j] == labels_s[i]:
                n_same += 1
            else:
                n_opp += 1

    if n_anchors == 0:
        return {"p_opposite": 0.0, "p_same": 0.0, "n_used": 0}

    return {
        "p_opposite": float(n_opp) / float(n_anchors),
        "p_same": float(n_same) / float(n_anchors),
        "n_used": int(n_anchors),
    }


# ---------------------------------------------------------------------------
# Surrogate label-shufflers (N0 / N1 / N2 / N3) and cluster-ISI resampler (N4)
# ---------------------------------------------------------------------------
def shuffle_labels_global(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """N0 — global Fisher-Yates shuffle (weak ceiling sanity)."""
    return rng.permutation(np.asarray(labels, dtype=int))


def shuffle_labels_block_aware(
    labels: np.ndarray,
    event_abs_times: np.ndarray,
    block_time_ranges: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """N1 — shuffle inside each block independently."""
    labels = np.asarray(labels, dtype=int)
    times = np.asarray(event_abs_times, dtype=float)
    out = labels.copy()
    for b_start, b_end in block_time_ranges:
        idx = np.where((times >= b_start) & (times <= b_end))[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def shuffle_labels_local_window(
    labels: np.ndarray,
    event_abs_times: np.ndarray,
    window_seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """N2 — non-overlapping local-window shuffle (main null).

    Each event is assigned to exactly one window of size `window_seconds`
    starting at ``times[0]``. Per archive §4.2 the runner exposes a
    `--null-window-min` flag; the pure shuffler keeps a single deterministic
    partition for testability.
    """
    labels = np.asarray(labels, dtype=int)
    times = np.asarray(event_abs_times, dtype=float)
    if times.size == 0:
        return labels.copy()
    out = labels.copy()
    t_min = float(times.min())
    t_max = float(times.max())
    span = max(t_max - t_min, 1e-9)
    n_windows = int(np.ceil(span / window_seconds)) + 1
    for w in range(n_windows):
        w_start = t_min + w * window_seconds
        w_end = w_start + window_seconds
        idx = np.where((times >= w_start) & (times < w_end))[0]
        if idx.size > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def shuffle_labels_circular(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """N3 — circular shift of label sequence (preserves burst time structure)."""
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    if n < 2:
        return labels.copy()
    lo = max(1, n // 10)
    hi = max(lo + 1, 9 * n // 10)
    if hi <= lo:
        hi = lo + 1
    shift = int(rng.integers(lo, hi))
    return np.roll(labels, shift)


def resample_isi_per_cluster(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    local_window_seconds: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """N4 — conditional follow-up null. Per-cluster ISI shuffle keeping per-cluster total count.

    NOT in PR-7 main TDD set; only triggered when N2 and N3 disagree (archive §4.1).
    The implementation here is a simple ISI permutation per cluster anchored
    at each cluster's first event time. local_window_seconds is unused in
    this minimal stub — reserved for the gamma-fit-per-window upgrade if
    follow-up is ever triggered.
    """
    del local_window_seconds  # placeholder for future per-window gamma fit
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    new_times = times.copy()
    for c in np.unique(labels):
        mask = labels == c
        cluster_times = times[mask]
        if cluster_times.size < 2:
            continue
        isis = np.diff(cluster_times)
        rng.shuffle(isis)
        new_cluster_times = np.empty_like(cluster_times)
        new_cluster_times[0] = cluster_times[0]
        for k in range(1, cluster_times.size):
            new_cluster_times[k] = new_cluster_times[k - 1] + isis[k - 1]
        new_times[mask] = new_cluster_times
    sort_idx = np.argsort(new_times, kind="stable")
    return new_times[sort_idx], labels[sort_idx]


# ---------------------------------------------------------------------------
# Per-subject driver: empirical + nulls + lift / excess
# ---------------------------------------------------------------------------
_DEFAULT_DELTA_T_GRID: Tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 1800.0, 3600.0)
_DEFAULT_NULLS: Tuple[str, ...] = ("N0", "N1", "N2", "N3")


def _generate_null_labels(
    null_name: str,
    labels: np.ndarray,
    times: np.ndarray,
    block_time_ranges: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
    n2_window_seconds: float,
) -> np.ndarray:
    if null_name == "N0":
        return shuffle_labels_global(labels, rng)
    if null_name == "N1":
        return shuffle_labels_block_aware(labels, times, block_time_ranges, rng)
    if null_name == "N2":
        return shuffle_labels_local_window(labels, times, n2_window_seconds, rng)
    if null_name == "N3":
        return shuffle_labels_circular(labels, rng)
    raise ValueError(f"Unknown null id: {null_name}")


def compute_pairing_with_nulls(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
    block_time_ranges: Sequence[Tuple[float, float]],
    delta_t_grid: Sequence[float] = _DEFAULT_DELTA_T_GRID,
    n_perm: int = 1000,
    nulls: Sequence[str] = _DEFAULT_NULLS,
    n2_window_seconds: float = 1800.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Per-subject driver. Returns nested dict matching archive §6.1.

    Keys: 'empirical' / 'null' / 'lift' (per null × per Δt).
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    rng = np.random.default_rng(seed)

    grid = tuple(float(d) for d in delta_t_grid)

    empirical: Dict[float, Dict[str, float]] = {}
    for dt in grid:
        empirical[dt] = compute_pairing_lift(times, labels, dt, block_time_ranges)

    null_results: Dict[str, Dict[float, Dict[str, list]]] = {
        n_id: {dt: {"p_opposite_dist": [], "p_same_dist": []} for dt in grid}
        for n_id in nulls
    }
    for null_name in nulls:
        for _ in range(n_perm):
            shuffled = _generate_null_labels(
                null_name, labels, times, block_time_ranges, rng, n2_window_seconds
            )
            for dt in grid:
                out = compute_pairing_lift(times, shuffled, dt, block_time_ranges)
                null_results[null_name][dt]["p_opposite_dist"].append(out["p_opposite"])
                null_results[null_name][dt]["p_same_dist"].append(out["p_same"])

    lift_results: Dict[str, Dict[float, Dict[str, float]]] = {}
    for null_name in nulls:
        lift_results[null_name] = {}
        for dt in grid:
            null_p_opp = float(np.mean(null_results[null_name][dt]["p_opposite_dist"]))
            null_p_same = float(np.mean(null_results[null_name][dt]["p_same_dist"]))
            opp_lift = empirical[dt]["p_opposite"] / max(null_p_opp, 1e-12)
            same_lift = empirical[dt]["p_same"] / max(null_p_same, 1e-12)
            lift_results[null_name][dt] = {
                "opposite_lift": float(opp_lift),
                "same_lift": float(same_lift),
                "excess": float(opp_lift - same_lift),
                "null_p_opposite_mean": null_p_opp,
                "null_p_same_mean": null_p_same,
            }

    return {"empirical": empirical, "null": null_results, "lift": lift_results}


# ---------------------------------------------------------------------------
# Secondary descriptive: next-event transition odds (archive §3.6)
# ---------------------------------------------------------------------------
def compute_transition_odds(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
) -> Dict[str, float]:
    """Next-event transition odds + time-to-next medians.

    Assumes events are sorted by time within their valid range. Returns
    transition_odds vs the i.i.d. baseline odds 2*p*(1-p)/(1-2p(1-p)).
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    n = times.size
    if n < 2:
        return {
            "p_next_opposite": float("nan"),
            "transition_odds": float("nan"),
            "baseline_odds": float("nan"),
            "time_to_next_opposite_median": float("nan"),
            "time_to_next_same_median": float("nan"),
            "n_pairs": 0,
        }

    sort_idx = np.argsort(times, kind="stable")
    times_s = times[sort_idx]
    labels_s = labels[sort_idx]

    next_diff = labels_s[:-1] != labels_s[1:]
    p_next_opposite = float(np.mean(next_diff))

    eps = 1e-12
    transition_odds = p_next_opposite / max(1.0 - p_next_opposite, eps)

    n_total = labels_s.size
    n_pos = int(np.sum(labels_s == 1))
    p1 = n_pos / n_total if n_total else 0.0
    baseline_p_opposite = 2.0 * p1 * (1.0 - p1)
    baseline_odds = baseline_p_opposite / max(1.0 - baseline_p_opposite, eps)

    isis = np.diff(times_s)
    t_opp = isis[next_diff]
    t_same = isis[~next_diff]

    return {
        "p_next_opposite": p_next_opposite,
        "transition_odds": float(transition_odds),
        "baseline_odds": float(baseline_odds),
        "time_to_next_opposite_median": float(np.median(t_opp)) if t_opp.size else float("nan"),
        "time_to_next_same_median": float(np.median(t_same)) if t_same.size else float("nan"),
        "n_pairs": int(next_diff.size),
    }


# ---------------------------------------------------------------------------
# Cohort-level paired test + triple-gate PASS judgment
# ---------------------------------------------------------------------------
def cohort_paired_test(
    excess_per_subject: Dict[str, float],
    alternative: str = "greater",
) -> Dict[str, float]:
    """Paired Wilcoxon + sign test on subject-level excess values."""
    excess = np.asarray(list(excess_per_subject.values()), dtype=float)
    n = int(excess.size)
    if n < 2:
        return {"wilcoxon_p": float("nan"), "sign_test_p": float("nan"), "median": float("nan"), "n": n}

    try:
        wilc = wilcoxon(excess, alternative=alternative)
        wilc_p = float(wilc.pvalue)
    except ValueError:
        wilc_p = float("nan")

    n_pos = int(np.sum(excess > 0))
    n_neg = int(np.sum(excess < 0))
    n_used = n_pos + n_neg
    if n_used == 0:
        sign_p = float("nan")
    elif alternative == "greater":
        sign_p = float(binomtest(n_pos, n_used, p=0.5, alternative="greater").pvalue)
    elif alternative == "less":
        sign_p = float(binomtest(n_neg, n_used, p=0.5, alternative="greater").pvalue)
    else:
        # two-sided: 2 * min tail
        side = max(n_pos, n_neg)
        sign_p = float(binomtest(side, n_used, p=0.5, alternative="greater").pvalue)
        sign_p = min(1.0, 2.0 * sign_p)

    return {
        "wilcoxon_p": wilc_p,
        "sign_test_p": sign_p,
        "median": float(np.median(excess)),
        "n": n,
    }


def evaluate_pass_criteria(
    cohort_excess_10s: Dict[str, float],
    cohort_excess_30s: Dict[str, float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Triple-gate PASS judgment per archive §3.2.

    PASS requires ALL THREE:
      (1) excess(10s) Wilcoxon (greater) p < alpha
      (2) excess(10s) sign test (greater) p < alpha
      (3) excess(30s) cohort median > 0 (sensitivity, not significance)

    Returns dict with per-gate values + 'pass' bool.
    """
    excess_10s = np.asarray(list(cohort_excess_10s.values()), dtype=float)
    excess_30s = np.asarray(list(cohort_excess_30s.values()), dtype=float)

    test_10s = cohort_paired_test(cohort_excess_10s, alternative="greater")
    median_30s = float(np.median(excess_30s)) if excess_30s.size else float("nan")
    median_30s_positive = bool(median_30s > 0.0)

    overall_pass = (
        (test_10s["wilcoxon_p"] < alpha)
        and (test_10s["sign_test_p"] < alpha)
        and median_30s_positive
    )

    return {
        "pass": bool(overall_pass),
        "wilcoxon_10s": test_10s["wilcoxon_p"],
        "sign_10s": test_10s["sign_test_p"],
        "median_10s": test_10s["median"],
        "median_30s": median_30s,
        "median_30s_positive": median_30s_positive,
        "n_subjects_10s": int(excess_10s.size),
        "n_subjects_30s": int(excess_30s.size),
        "alpha": float(alpha),
    }
