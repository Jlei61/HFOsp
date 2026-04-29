"""PR-7: Template antagonistic temporal pairing.

Pure statistical layer. Tests whether forward/reverse template pairs
(PR-2.5 + PR-6 Step 4) are temporally coupled at short scales, or whether
they coexist as independent slow-modulated streams.

Contract: docs/archive/topic1/pr7_template_antagonistic_pairing_plan_2026-04-28.md.
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, wilcoxon


# ---------------------------------------------------------------------------
# Block bookkeeping (every helper that walks events sequentially MUST use this)
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


# ---------------------------------------------------------------------------
# Core lift estimator (symmetric + directional fields)
# ---------------------------------------------------------------------------
def compute_pairing_lift(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
    delta_t_seconds: float,
    block_time_ranges: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """Per-anchor opposite/same counts in (t_i, t_i + Δt] within the same block.

    Returns:
        p_opposite / p_same — symmetric (per-anchor opposite/same averages)
        p_a_to_b / p_b_to_a — directional (avg opposite count per anchor of class)
        p_a_to_a / p_b_to_b — directional same
        n_a_anchors / n_b_anchors — anchor counts per class
        n_used — total in-block anchors

    Cluster labels MUST be in {0, 1} (`a` = 0, `b` = 1) per archive §3.1
    label normalization. Other values are ignored as anchors but still
    block-mapped.
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    n = times.size
    empty: Dict[str, float] = {
        "p_opposite": 0.0,
        "p_same": 0.0,
        "p_a_to_b": 0.0,
        "p_b_to_a": 0.0,
        "p_a_to_a": 0.0,
        "p_b_to_b": 0.0,
        "n_a_anchors": 0,
        "n_b_anchors": 0,
        "n_used": 0,
    }
    if n == 0:
        return empty

    sort_idx = np.argsort(times, kind="stable")
    times_s = times[sort_idx]
    labels_s = labels[sort_idx]
    block_s = _assign_blocks(times_s, block_time_ranges)

    n_opp = 0
    n_same = 0
    n_anchors = 0
    n_a_to_b = 0
    n_b_to_a = 0
    n_a_to_a = 0
    n_b_to_b = 0
    n_a_anchors = 0
    n_b_anchors = 0

    # Per-block vectorized accumulation (assumes blocks are non-overlapping;
    # any event with t in (t_i, t_i+Δt] outside this block is by definition
    # outside the block_end clamp, so the block_s[j] != block_s[i] check
    # from the scalar version is implicit here).
    for b_idx, (b_start, b_end) in enumerate(block_time_ranges):
        bl_idx = np.where(block_s == b_idx)[0]
        if bl_idx.size == 0:
            continue
        bl_times = times_s[bl_idx]
        bl_labels = labels_s[bl_idx]
        nb = bl_idx.size

        # Cumulative label counts, padded so cum[k] = count over indices [0..k-1].
        cum_0_pad = np.concatenate([[0], np.cumsum(bl_labels == 0)])
        cum_1_pad = np.concatenate([[0], np.cumsum(bl_labels == 1)])

        # Per-anchor window end clamped to block end.
        window_ends = np.minimum(bl_times + delta_t_seconds, b_end)
        # `hi[i]` = first index in bl_times that is > window_ends[i].
        hi = np.searchsorted(bl_times, window_ends, side="right")

        # Events strictly after anchor i but at or before window_end[i] live
        # at indices [i+1, hi[i]-1]. Count them via the padded cumsum.
        i_arr = np.arange(nb)
        n_label0_in_window = cum_0_pad[hi] - cum_0_pad[i_arr + 1]
        n_label1_in_window = cum_1_pad[hi] - cum_1_pad[i_arr + 1]

        is_a = bl_labels == 0
        is_b = bl_labels == 1

        n_opp += int(
            n_label1_in_window[is_a].sum() + n_label0_in_window[is_b].sum()
        )
        n_same += int(
            n_label0_in_window[is_a].sum() + n_label1_in_window[is_b].sum()
        )
        n_anchors += nb
        n_a_anchors += int(is_a.sum())
        n_b_anchors += int(is_b.sum())
        n_a_to_b += int(n_label1_in_window[is_a].sum())
        n_b_to_a += int(n_label0_in_window[is_b].sum())
        n_a_to_a += int(n_label0_in_window[is_a].sum())
        n_b_to_b += int(n_label1_in_window[is_b].sum())

    if n_anchors == 0:
        return empty

    return {
        "p_opposite": float(n_opp) / float(n_anchors),
        "p_same": float(n_same) / float(n_anchors),
        "p_a_to_b": float(n_a_to_b) / float(max(n_a_anchors, 1)),
        "p_b_to_a": float(n_b_to_a) / float(max(n_b_anchors, 1)),
        "p_a_to_a": float(n_a_to_a) / float(max(n_a_anchors, 1)),
        "p_b_to_b": float(n_b_to_b) / float(max(n_b_anchors, 1)),
        "n_a_anchors": int(n_a_anchors),
        "n_b_anchors": int(n_b_anchors),
        "n_used": int(n_anchors),
    }


# ---------------------------------------------------------------------------
# Surrogate label-shufflers (N0 / N1 / N2 / N3); N4 raises NotImplementedError
# ---------------------------------------------------------------------------
def shuffle_labels_global(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """N0 — global Fisher-Yates shuffle (sanity ceiling)."""
    return rng.permutation(np.asarray(labels, dtype=int))


def shuffle_labels_block_aware(
    labels: np.ndarray,
    event_abs_times: np.ndarray,
    block_time_ranges: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """N1 — shuffle inside each block independently (sanity / mid-strength)."""
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
    block_time_ranges: Sequence[Tuple[float, float]],
    rng: np.random.Generator,
) -> np.ndarray:
    """N2 main null — local-window shuffle inside each block, 50% overlap, first-covering rule.

    Per archive §4.1 / §4.2:
    - window_seconds default 30 min (caller passes seconds explicitly)
    - step = window_seconds / 2 (50% overlap of windows)
    - **first-covering rule**: each event is assigned to the EARLIEST window
      that covers it (smallest window index `w` such that
      `b_start + w*step <= t < b_start + w*step + window_seconds`)
    - **never crosses block boundaries**: each block is processed independently;
      events outside every block are not shuffled

    The 50% overlap + first-covering partitioning gives effective shuffle
    pools of: first window full-sized [b_start, b_start+window), then
    half-step pools [b_start+w*step, b_start+(w+1)*step) for w >= 2 i.e.
    starting at b_start+window. This is intentional and matches plan §4.2.
    """
    labels = np.asarray(labels, dtype=int)
    times = np.asarray(event_abs_times, dtype=float)
    out = labels.copy()
    if times.size == 0:
        return out
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    step = window_seconds / 2.0
    for b_start, b_end in block_time_ranges:
        in_block = (times >= b_start) & (times <= b_end)
        block_idx_arr = np.where(in_block)[0]
        if block_idx_arr.size < 2:
            continue
        block_times = times[block_idx_arr]
        # First-covering window index: smallest w >= 0 such that
        #   b_start + w*step <= t  AND  b_start + w*step + window_seconds > t
        # Algebra: w = max(0, floor((t - b_start - window_seconds) / step) + 1)
        first_w = np.maximum(
            0,
            np.floor((block_times - b_start - window_seconds) / step).astype(int) + 1,
        )
        for w_idx in np.unique(first_w):
            mask = first_w == w_idx
            sub = block_idx_arr[mask]
            if sub.size > 1:
                out[sub] = rng.permutation(out[sub])
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


def resample_isi_per_cluster(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, np.ndarray]:
    """N4 conditional follow-up null — NOT IMPLEMENTED.

    Plan §4.1 / §4.2 specifies a per-cluster rate-matched ISI null with a
    gamma fit per local window. That is intentionally deferred until the
    follow-up trigger condition fires (N2 positive but N3 inconsistent).

    Calling this function raises NotImplementedError on purpose: a silent
    stub would let unsuitable values into primary results. When the
    follow-up is actually triggered, implement the gamma-fit-per-window
    construction here and add a dedicated TDD test.
    """
    raise NotImplementedError(
        "resample_isi_per_cluster (N4) is a conditional follow-up null. "
        "Implement gamma-fit-per-window per archive §4.1 before use; the "
        "stub is intentionally absent so it cannot leak into primary "
        "judgments."
    )


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
        return shuffle_labels_local_window(
            labels, times, n2_window_seconds, block_time_ranges, rng
        )
    if null_name == "N3":
        return shuffle_labels_circular(labels, rng)
    if null_name == "N4":
        raise NotImplementedError(
            "N4 must not be invoked from compute_pairing_with_nulls; it is a "
            "conditional follow-up. See resample_isi_per_cluster docstring."
        )
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

    Keys:
      - 'empirical': {Δt: pairing_lift_dict (full directional fields)}
      - 'null':     {null_id: {Δt: {p_opposite_dist / p_same_dist /
                                    p_a_to_b_dist / p_b_to_a_dist}}}
      - 'lift':     {null_id: {Δt: {opposite_lift / same_lift / excess /
                                    a_to_b_lift / b_to_a_lift / asym}}}

    `n2_window_seconds` is only consumed by N2 and is independent of any
    Δt in `delta_t_grid`.
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    rng = np.random.default_rng(seed)

    grid = tuple(float(d) for d in delta_t_grid)

    empirical: Dict[float, Dict[str, float]] = {}
    for dt in grid:
        empirical[dt] = compute_pairing_lift(times, labels, dt, block_time_ranges)

    null_dist_keys = ("p_opposite_dist", "p_same_dist", "p_a_to_b_dist", "p_b_to_a_dist")
    null_results: Dict[str, Dict[float, Dict[str, list]]] = {
        n_id: {dt: {k: [] for k in null_dist_keys} for dt in grid}
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
                null_results[null_name][dt]["p_a_to_b_dist"].append(out["p_a_to_b"])
                null_results[null_name][dt]["p_b_to_a_dist"].append(out["p_b_to_a"])

    lift_results: Dict[str, Dict[float, Dict[str, float]]] = {}
    for null_name in nulls:
        lift_results[null_name] = {}
        for dt in grid:
            ed = empirical[dt]
            null_p_opp = float(np.mean(null_results[null_name][dt]["p_opposite_dist"]))
            null_p_same = float(np.mean(null_results[null_name][dt]["p_same_dist"]))
            null_p_atb = float(np.mean(null_results[null_name][dt]["p_a_to_b_dist"]))
            null_p_bta = float(np.mean(null_results[null_name][dt]["p_b_to_a_dist"]))
            opp_lift = ed["p_opposite"] / max(null_p_opp, 1e-12)
            same_lift = ed["p_same"] / max(null_p_same, 1e-12)
            atb_lift = ed["p_a_to_b"] / max(null_p_atb, 1e-12)
            bta_lift = ed["p_b_to_a"] / max(null_p_bta, 1e-12)
            lift_results[null_name][dt] = {
                "opposite_lift": float(opp_lift),
                "same_lift": float(same_lift),
                "excess": float(opp_lift - same_lift),
                "a_to_b_lift": float(atb_lift),
                "b_to_a_lift": float(bta_lift),
                "asym": float(atb_lift - bta_lift),
                "null_p_opposite_mean": null_p_opp,
                "null_p_same_mean": null_p_same,
                "null_p_a_to_b_mean": null_p_atb,
                "null_p_b_to_a_mean": null_p_bta,
            }

    return {"empirical": empirical, "null": null_results, "lift": lift_results}


# ---------------------------------------------------------------------------
# Secondary descriptive: next-event transition odds (block-aware, archive §3.6)
# ---------------------------------------------------------------------------
def compute_transition_odds(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
    block_time_ranges: Sequence[Tuple[float, float]],
) -> Dict[str, float]:
    """Next-event transition odds + time-to-next medians, BLOCK-AWARE.

    Per archive §3.6 with §10 risk: cross-block pairs (e.g. an event at the
    end of one recording block and the next event in a later block separated
    by a gap of hours) are NOT counted as transitions. They would otherwise
    pollute mechanistic interpretation.

    Returns transition_odds vs the i.i.d. baseline odds 2*p*(1-p)/(1-2p(1-p)).
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    n = times.size
    empty = {
        "p_next_opposite": float("nan"),
        "transition_odds": float("nan"),
        "baseline_odds": float("nan"),
        "time_to_next_opposite_median": float("nan"),
        "time_to_next_same_median": float("nan"),
        "n_pairs": 0,
    }
    if n < 2:
        return empty

    sort_idx = np.argsort(times, kind="stable")
    times_s = times[sort_idx]
    labels_s = labels[sort_idx]
    block_s = _assign_blocks(times_s, block_time_ranges)

    same_block = (block_s[:-1] == block_s[1:]) & (block_s[:-1] >= 0)
    n_pairs = int(np.sum(same_block))
    if n_pairs == 0:
        return empty

    next_diff = labels_s[:-1] != labels_s[1:]
    valid_diff = next_diff & same_block

    p_next_opposite = float(np.sum(valid_diff)) / float(n_pairs)
    eps = 1e-12
    transition_odds = p_next_opposite / max(1.0 - p_next_opposite, eps)

    # Baseline: same-block events only, treat as i.i.d. label draws
    block_eligible_event_mask = block_s >= 0
    valid_labels = labels_s[block_eligible_event_mask]
    n_eligible = valid_labels.size
    if n_eligible == 0:
        return empty
    p1 = float(np.sum(valid_labels == 1)) / float(n_eligible)
    baseline_p_opposite = 2.0 * p1 * (1.0 - p1)
    baseline_odds = baseline_p_opposite / max(1.0 - baseline_p_opposite, eps)

    isis = np.diff(times_s)
    t_opp = isis[same_block & next_diff]
    t_same = isis[same_block & ~next_diff]

    return {
        "p_next_opposite": p_next_opposite,
        "transition_odds": float(transition_odds),
        "baseline_odds": float(baseline_odds),
        "time_to_next_opposite_median": float(np.median(t_opp)) if t_opp.size else float("nan"),
        "time_to_next_same_median": float(np.median(t_same)) if t_same.size else float("nan"),
        "n_pairs": n_pairs,
    }


# ---------------------------------------------------------------------------
# Cohort-level paired test + triple-gate PASS judgment (key-matched)
# ---------------------------------------------------------------------------
def _aligned_excess_arrays(
    excess_a: Mapping[Hashable, float],
    excess_b: Mapping[Hashable, float],
) -> Tuple[Tuple[Hashable, ...], np.ndarray, np.ndarray]:
    """Return (sorted_keys, arr_a, arr_b) with subject-key alignment enforced.

    Raises ValueError if the two dicts do not share an identical subject key
    set. This is required by archive §3.2 (subject-level paired design):
    `excess(10s)` and `excess(30s)` MUST be the same subjects.
    """
    keys_a = set(excess_a.keys())
    keys_b = set(excess_b.keys())
    if keys_a != keys_b:
        only_a = sorted(keys_a - keys_b)
        only_b = sorted(keys_b - keys_a)
        raise ValueError(
            "Subject key mismatch between cohort dicts; paired design requires "
            f"identical subjects. Only in first dict: {only_a}; "
            f"only in second dict: {only_b}"
        )
    keys_sorted = tuple(sorted(keys_a))
    arr_a = np.asarray([excess_a[k] for k in keys_sorted], dtype=float)
    arr_b = np.asarray([excess_b[k] for k in keys_sorted], dtype=float)
    return keys_sorted, arr_a, arr_b


def cohort_paired_test(
    excess_per_subject: Mapping[Hashable, float],
    alternative: str = "greater",
) -> Dict[str, float]:
    """Wilcoxon + sign test on subject-level excess values."""
    keys_sorted = tuple(sorted(excess_per_subject.keys()))
    excess = np.asarray([excess_per_subject[k] for k in keys_sorted], dtype=float)
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
    cohort_excess_10s: Mapping[Hashable, float],
    cohort_excess_30s: Mapping[Hashable, float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Triple-gate PASS judgment per archive §3.2.

    Subject keys MUST match between the two dicts (raises ValueError on
    mismatch). This guards the subject-level paired design: 10s and 30s
    statistics must be computed on the same cohort.

    PASS requires ALL THREE:
      (1) excess(10s) Wilcoxon (greater) p < alpha
      (2) excess(10s) sign test (greater) p < alpha
      (3) excess(30s) cohort median > 0 (sensitivity, not significance)
    """
    keys_sorted, arr_10, arr_30 = _aligned_excess_arrays(
        cohort_excess_10s, cohort_excess_30s
    )

    aligned_10s = {k: cohort_excess_10s[k] for k in keys_sorted}
    test_10s = cohort_paired_test(aligned_10s, alternative="greater")

    median_30s = float(np.median(arr_30)) if arr_30.size else float("nan")
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
        "n_subjects": int(arr_10.size),
        "subjects": list(keys_sorted),
        "alpha": float(alpha),
    }


# ---------------------------------------------------------------------------
# Step 3.5 — Burst-level run diagnostic (post-hoc exploratory)
# Plan: docs/archive/topic1/pr7_step3p5_burst_diagnostic_plan_2026-04-29.md.
# Does NOT change H1 verdict; not a PASS/FAIL gate.
# ---------------------------------------------------------------------------
def compute_runs(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
    block_time_ranges: Sequence[Tuple[float, float]],
) -> Dict[str, np.ndarray]:
    """Maximal consecutive same-label runs, block-aware.

    Run boundary triggers: label change OR block change. No ISI threshold
    (no ad hoc parameter per archive §3).

    Returns dict with arrays keyed by run:
      'run_label'         (n_runs,)  cluster label of the run
      'run_length'        (n_runs,)  number of events in the run
      'run_block_id'      (n_runs,)  block index for the run
      'run_t_first'       (n_runs,)  absolute time of first event in run
      'run_t_last'        (n_runs,)  absolute time of last event in run
      'within_run_iei'    (n_events_in_runs - n_runs,)  pooled ISIs inside runs
      'between_run_gap'   (n_runs - n_runs_per_block_total,)  gaps between
                                     adjacent runs in the SAME block
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    n = times.size
    if n == 0:
        return {
            "run_label": np.zeros(0, dtype=int),
            "run_length": np.zeros(0, dtype=int),
            "run_block_id": np.zeros(0, dtype=int),
            "run_t_first": np.zeros(0, dtype=float),
            "run_t_last": np.zeros(0, dtype=float),
            "within_run_iei": np.zeros(0, dtype=float),
            "between_run_gap": np.zeros(0, dtype=float),
        }

    sort_idx = np.argsort(times, kind="stable")
    times_s = times[sort_idx]
    labels_s = labels[sort_idx]
    block_s = _assign_blocks(times_s, block_time_ranges)

    in_block_mask = block_s >= 0
    if not np.any(in_block_mask):
        return {
            "run_label": np.zeros(0, dtype=int),
            "run_length": np.zeros(0, dtype=int),
            "run_block_id": np.zeros(0, dtype=int),
            "run_t_first": np.zeros(0, dtype=float),
            "run_t_last": np.zeros(0, dtype=float),
            "within_run_iei": np.zeros(0, dtype=float),
            "between_run_gap": np.zeros(0, dtype=float),
        }

    # Run boundary at index i (>0) iff labels_s[i] != labels_s[i-1]
    # OR block_s[i] != block_s[i-1] OR either side is out-of-block.
    run_labels: List[int] = []
    run_lengths: List[int] = []
    run_block_ids: List[int] = []
    run_t_firsts: List[float] = []
    run_t_lasts: List[float] = []
    within_iei_chunks: List[np.ndarray] = []
    between_gap_chunks: List[float] = []

    cur_label: Optional[int] = None
    cur_block: int = -1
    cur_start: int = 0
    cur_count: int = 0
    last_run_end_idx_per_block: Dict[int, int] = {}

    def _close_run(end_idx_excl: int) -> None:
        if cur_count == 0:
            return
        run_labels.append(int(cur_label) if cur_label is not None else -1)
        run_lengths.append(int(cur_count))
        run_block_ids.append(int(cur_block))
        first_idx = cur_start
        last_idx = end_idx_excl - 1
        run_t_firsts.append(float(times_s[first_idx]))
        run_t_lasts.append(float(times_s[last_idx]))
        if cur_count >= 2:
            within_iei_chunks.append(np.diff(times_s[first_idx:end_idx_excl]))
        prev_last = last_run_end_idx_per_block.get(cur_block)
        if prev_last is not None:
            between_gap_chunks.append(
                float(times_s[first_idx] - times_s[prev_last])
            )
        last_run_end_idx_per_block[cur_block] = last_idx

    for i in range(n):
        if not in_block_mask[i]:
            # Out-of-block event: close current run (if any), reset.
            _close_run(i)
            cur_label = None
            cur_block = -1
            cur_count = 0
            continue
        l_i = int(labels_s[i])
        b_i = int(block_s[i])
        if cur_count == 0:
            cur_label = l_i
            cur_block = b_i
            cur_start = i
            cur_count = 1
        elif l_i == cur_label and b_i == cur_block:
            cur_count += 1
        else:
            _close_run(i)
            cur_label = l_i
            cur_block = b_i
            cur_start = i
            cur_count = 1
    _close_run(n)

    within_iei = (
        np.concatenate(within_iei_chunks)
        if within_iei_chunks
        else np.zeros(0, dtype=float)
    )
    between_gap = (
        np.asarray(between_gap_chunks, dtype=float)
        if between_gap_chunks
        else np.zeros(0, dtype=float)
    )

    return {
        "run_label": np.asarray(run_labels, dtype=int),
        "run_length": np.asarray(run_lengths, dtype=int),
        "run_block_id": np.asarray(run_block_ids, dtype=int),
        "run_t_first": np.asarray(run_t_firsts, dtype=float),
        "run_t_last": np.asarray(run_t_lasts, dtype=float),
        "within_run_iei": within_iei,
        "between_run_gap": between_gap,
    }


def compute_run_metrics(
    runs: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Per-subject summary statistics from a `compute_runs` result.

    Returns archive §4.2 A/B/C metrics: run-length stats, run-time
    structure, global burst fractions. lag1_same_excess is computed
    separately (needs raw events + null shuffles).
    """
    rl = runs["run_length"]
    rlab = runs["run_label"]
    n_runs = int(rl.size)
    if n_runs == 0:
        return {
            "n_runs_total": 0,
            "mean_run_length": float("nan"),
            "median_run_length": float("nan"),
            "p95_run_length": float("nan"),
            "mean_run_length_a": float("nan"),
            "mean_run_length_b": float("nan"),
            "run_duration_seconds_median": float("nan"),
            "within_run_iei_median": float("nan"),
            "between_run_gap_median": float("nan"),
            "gap_to_iei_ratio": float("nan"),
            "event_fraction_in_long_runs": 0.0,
            "event_fraction_in_singleton_runs": 0.0,
            "n_events_total": 0,
        }

    n_events_total = int(rl.sum())
    durations = runs["run_t_last"] - runs["run_t_first"]
    within_iei = runs["within_run_iei"]
    between_gap = runs["between_run_gap"]

    rl_a = rl[rlab == 0]
    rl_b = rl[rlab == 1]

    mean_within = float(np.median(within_iei)) if within_iei.size else float("nan")
    mean_between = float(np.median(between_gap)) if between_gap.size else float("nan")
    if np.isfinite(mean_within) and mean_within > 0 and np.isfinite(mean_between):
        gap_iei_ratio = mean_between / mean_within
    else:
        gap_iei_ratio = float("nan")

    return {
        "n_runs_total": n_runs,
        "mean_run_length": float(np.mean(rl)),
        "median_run_length": float(np.median(rl)),
        "p95_run_length": float(np.percentile(rl, 95)) if n_runs >= 2 else float(rl[0]),
        "mean_run_length_a": float(np.mean(rl_a)) if rl_a.size else float("nan"),
        "mean_run_length_b": float(np.mean(rl_b)) if rl_b.size else float("nan"),
        "run_duration_seconds_median": float(np.median(durations)),
        "within_run_iei_median": mean_within,
        "between_run_gap_median": mean_between,
        "gap_to_iei_ratio": float(gap_iei_ratio),
        "event_fraction_in_long_runs": float(np.sum(rl[rl >= 3])) / float(n_events_total),
        "event_fraction_in_singleton_runs": float(np.sum(rl[rl == 1])) / float(n_events_total),
        "n_events_total": n_events_total,
    }


def compute_lag1_same_fraction(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
    block_time_ranges: Sequence[Tuple[float, float]],
) -> float:
    """Fraction of consecutive same-block event pairs with same label.

    Returns nan if no eligible pairs.
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    if times.size < 2:
        return float("nan")
    sort_idx = np.argsort(times, kind="stable")
    times_s = times[sort_idx]
    labels_s = labels[sort_idx]
    block_s = _assign_blocks(times_s, block_time_ranges)
    same_block = (block_s[:-1] == block_s[1:]) & (block_s[:-1] >= 0)
    if not np.any(same_block):
        return float("nan")
    same_label = labels_s[:-1] == labels_s[1:]
    return float(np.mean(same_label[same_block]))


def compute_burst_diagnostic_with_nulls(
    event_abs_times: Sequence[float],
    cluster_labels: Sequence[int],
    block_time_ranges: Sequence[Tuple[float, float]],
    n_perm: int = 500,
    nulls: Sequence[str] = ("N1", "N2"),
    n2_window_seconds: float = 1800.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """Per-subject burst diagnostic driver. Returns nested dict:

    {
      'empirical': {... compute_run_metrics output ..., 'lag1_same': float},
      'null': {N1: {metric_name: {'mean': float, 'std': float}}, N2: {...}},
      'lift': {N1: {metric_name: float lift}, N2: {...}},
      'lag1_same_excess': {N1: float, N2: float},
    }
    """
    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    rng = np.random.default_rng(seed)

    runs = compute_runs(times, labels, block_time_ranges)
    emp_metrics = compute_run_metrics(runs)
    emp_lag1 = compute_lag1_same_fraction(times, labels, block_time_ranges)
    empirical: Dict[str, Any] = dict(emp_metrics)
    empirical["lag1_same"] = emp_lag1

    metric_keys = (
        "mean_run_length",
        "median_run_length",
        "gap_to_iei_ratio",
        "event_fraction_in_long_runs",
    )

    null_block: Dict[str, Dict[str, Dict[str, float]]] = {}
    lift_block: Dict[str, Dict[str, float]] = {}
    lag1_excess_block: Dict[str, float] = {}

    for null_name in nulls:
        per_perm: Dict[str, List[float]] = {k: [] for k in metric_keys}
        per_perm["lag1_same"] = []
        for _ in range(n_perm):
            shuffled = _generate_null_labels(
                null_name, labels, times, block_time_ranges, rng, n2_window_seconds
            )
            r = compute_runs(times, shuffled, block_time_ranges)
            m = compute_run_metrics(r)
            for k in metric_keys:
                per_perm[k].append(m[k])
            per_perm["lag1_same"].append(
                compute_lag1_same_fraction(times, shuffled, block_time_ranges)
            )

        per_null_summary: Dict[str, Dict[str, float]] = {}
        for k in metric_keys:
            arr = np.asarray(per_perm[k], dtype=float)
            arr = arr[np.isfinite(arr)]
            per_null_summary[k] = {
                "mean": float(np.mean(arr)) if arr.size else float("nan"),
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan"),
            }
        lag1_arr = np.asarray(per_perm["lag1_same"], dtype=float)
        lag1_arr = lag1_arr[np.isfinite(lag1_arr)]
        per_null_summary["lag1_same"] = {
            "mean": float(np.mean(lag1_arr)) if lag1_arr.size else float("nan"),
            "std": float(np.std(lag1_arr, ddof=1)) if lag1_arr.size > 1 else float("nan"),
        }
        null_block[null_name] = per_null_summary

        lift_per_null: Dict[str, float] = {}
        for k in metric_keys:
            null_mean = per_null_summary[k]["mean"]
            emp_val = empirical[k]
            if not np.isfinite(emp_val) or not np.isfinite(null_mean) or null_mean == 0:
                lift_per_null[k] = float("nan")
            else:
                lift_per_null[k] = float(emp_val / null_mean)
        # gap_to_iei_lift specifically named for clarity
        lift_per_null["run_length_lift"] = lift_per_null.get(
            "mean_run_length", float("nan")
        )
        lift_per_null["gap_to_iei_lift"] = lift_per_null.get(
            "gap_to_iei_ratio", float("nan")
        )
        lift_block[null_name] = lift_per_null

        # lag1_same_excess = empirical - null_mean (signed difference, NOT ratio)
        if np.isfinite(emp_lag1) and np.isfinite(per_null_summary["lag1_same"]["mean"]):
            lag1_excess_block[null_name] = float(
                emp_lag1 - per_null_summary["lag1_same"]["mean"]
            )
        else:
            lag1_excess_block[null_name] = float("nan")

    return {
        "empirical": empirical,
        "null": null_block,
        "lift": lift_block,
        "lag1_same_excess": lag1_excess_block,
        "n_perm": int(n_perm),
        "nulls_run": list(nulls),
    }
