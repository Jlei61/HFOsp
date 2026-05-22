"""SEF-ITP framework Phase 2 — H3 mark-independence + H4 normalized rate/geometry instability.

Phase 2 sits on top of Phase 1's n=23 cohort (Phase 1 = H1 sanity / H2 primary cohort claim /
H6 secondary). Phase 2 adds:

- **H3** = mark-independent template sampling at multiple time scales + endpoint geometric
  stability. Mostly ingest of PR-7 burst diagnostic (lag1_same + run_length_lift) + PR-7 pairing
  (window excess at {10, 30, 60, 1800}s) + PR-6 anchoring (split_half_robustness endpoint
  Jaccard). Cohort verdict via bootstrap TOST equivalence vs ±δ_excess=0.05 band.
- **H4** = normalized rate vs geometry instability. New pipeline: slice cohort 24h / multi-day
  data into 2h epochs (preserving time order), compute per-epoch rate(t) and endpoint(t),
  compare normalized instability `I_rate` vs `I_geom` via cohort Wilcoxon + Cohen's d.

**Verdict naming locked at framework time:**

- H3: SUPPORTED / NOT_SUPPORTED_MEMORY / NOT_SUPPORTED_GEOMETRY_UNSTABLE / NOT_SUPPORTED_BOTH /
       CONTRADICTED. NOT PASS/NULL/FAIL — guards against "PASS = proves independence".
- H4: PASS / NULL / FAIL / UNDERPOWERED (standard verdict family).

**Locked contracts:**

- δ_excess = 0.05 lock at framework time. Forbid post-hoc adjustment.
- H3 endpoint stability combinator = **OR** (project convention; mirror of
  AGENTS.md cross-PR `forward_reverse_reproduced` = split-half OR odd-even).
- H3 wording lock: "compatible with mark-independent sampling within tested precision."
- H4 Cohen's d ≥ 0.30 floor for PASS verdict.
- H4 I_rate matched null is a **Phase 2 v1.0.0 spec amendment proposal** (see
  `docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md`). Both the literal
  framework v1.0.5 §3.4 `epoch_order_shuffle` (degenerate by construction) and a non-degenerate
  `circular_shift_within_block` variant are implemented; user decides which enters
  framework on return.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

__version__ = "v1.0.0"


@dataclass
class SubjectPhase2Data:
    """One subject's Phase 2 inputs after ingest.

    H3 fields are scalar / dict, populated by `load_subject_for_phase2_h3` from PR-7 and PR-6
    per-subject JSONs (no recomputation here — pure ingest).

    H4 fields are raw arrays (events, labels, block ranges, template ranks, channel names)
    needed for per-epoch endpoint recomputation downstream.
    """

    dataset: str
    subject_id: str

    # H3 ingest (already-computed per-subject metrics from PR-7 / PR-6 JSONs)
    lag1_same_excess_n2: float
    window_excess_n2: Dict[float, float]  # {10.0, 30.0, 60.0, 1800.0} → excess
    run_length_lift_n2: float
    endpoint_jaccard_first_half: float
    endpoint_jaccard_odd_even: float

    # H4 raw (downstream computes per-epoch features from these)
    event_abs_times: np.ndarray
    cluster_labels: np.ndarray
    block_time_ranges: List[Tuple[float, float]]
    template_ranks: Dict[int, np.ndarray]  # {cluster_id: rank vector aligned to channel_names}
    channel_names: List[str]


# --------------------------------------------------------------------------- #
# H3 ingest extractors
#
# H3's three statistical layers all source from existing per-subject JSONs:
#   1. mark-transition lag1 + window excess: PR-7 burst + PR-7 pairing
#   2. burst run length: PR-7 burst
#   3. endpoint geometric stability: PR-6 anchoring (split_half_robustness)
#
# CLAUDE.md §6.1 re-use check (question-match):
#   - PR-7 burst's lag1_same_excess vs N2: same question as H3 layer 1 (lag-1 same-template
#     frequency vs marginal-preserving null). DIRECT INGEST.
#   - PR-7 pairing's window excess @ {10, 30, 60, 1800}s vs N2: same question as H3 multi-scale
#     mark-transition. DIRECT INGEST.
#   - PR-6's split_half_robustness.subject_mean_jaccard_endpoint: same question as H3
#     endpoint geometric stability (Jaccard recall of endpoint set across temporal split).
#     DIRECT INGEST.
# --------------------------------------------------------------------------- #


def extract_window_excess_from_pairing(
    pairing_json: dict,
    windows: Tuple[float, ...] = (10.0, 30.0, 60.0, 1800.0),
    null_key: str = "N2",
) -> Dict[float, float]:
    """Pull `excess` at the requested Δt windows from a PR-7 pairing per-subject JSON.

    PR-7 schema: `pairing_with_nulls.lift.<null>.<window_str>.excess`. Windows are stored as
    string keys (e.g. `"10.0"`).

    Raises KeyError if `pairing_with_nulls`, `lift`, `null_key`, or any requested window is
    missing. No silent default — H3 must not paper over a missing PR-7 window.
    """
    lift = pairing_json["pairing_with_nulls"]["lift"][null_key]
    return {w: float(lift[f"{w}"]["excess"]) for w in windows}


def extract_lag1_and_runlength_from_burst(
    burst_json: dict,
    null_key: str = "N2",
) -> Tuple[float, float]:
    """Pull (lag1_same_excess, run_length_lift) from a PR-7 burst per-subject JSON.

    PR-7 schema:
      `burst_diagnostic.lag1_same_excess.<null_key>` → float (target=0 in H3 TOST)
      `burst_diagnostic.lift.<null_key>.run_length_lift` → float (target=1 in H3 TOST)
    """
    bd = burst_json["burst_diagnostic"]
    lag1 = float(bd["lag1_same_excess"][null_key])
    run_length = float(bd["lift"][null_key]["run_length_lift"])
    return lag1, run_length


def extract_endpoint_jaccard_from_anchoring(
    anchoring_json: dict,
) -> Tuple[float, float]:
    """Pull (first_half_second_half, odd_even_block) endpoint Jaccard from a PR-6 anchoring
    per-subject JSON.

    PR-6 schema:
      `split_half_robustness.per_split.first_half_second_half.subject_mean_jaccard_endpoint`
      `split_half_robustness.per_split.odd_even_block.subject_mean_jaccard_endpoint`
    """
    per_split = anchoring_json["split_half_robustness"]["per_split"]
    fh = float(per_split["first_half_second_half"]["subject_mean_jaccard_endpoint"])
    oe = float(per_split["odd_even_block"]["subject_mean_jaccard_endpoint"])
    return fh, oe


# --------------------------------------------------------------------------- #
# TOST equivalence (cohort bootstrap CI vs ±δ band)
#
# Ported verbatim from scripts/pr7_addendum_p3_equivalence.py:123. Same statistical
# guarantee: equivalence iff p_lower < α AND p_upper < α AND CI ⊂ (target ± δ).
#
# CLAUDE.md §6.1 re-use check (question-match):
#   - PR-7 addendum question: "is cohort median of antagonistic-pair metric M
#     equivalent to target (0 for excess, 1 for run_length_lift) within ±δ_excess?"
#   - Phase 2 H3 question: identical, for different metrics (lag1_same_excess,
#     window_excess @ {10,30,60,1800}s, run_length_lift). Same δ_excess=0.05.
#   - DIRECT REUSE. PR-7 addendum's leave-one-out-548 is PR-7-specific; Phase 2's LOO
#     iterates over its own cohort (n=23) via `cohort_tost_with_loo` (Task 11.5).
# --------------------------------------------------------------------------- #


def tost_equivalence(
    values: np.ndarray,
    target: float,
    delta: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Two One-Sided Test (TOST) for equivalence of cohort median to target ± delta.

    Equivalence is declared iff bootstrap `p_lower < alpha` AND `p_upper < alpha` AND
    the 95% CI on the median is wholly inside `(target - delta, target + delta)`.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(values[idx], axis=1)
    obs_median = float(np.median(values))
    ci_lo = float(np.quantile(boot_medians, alpha / 2))
    ci_hi = float(np.quantile(boot_medians, 1 - alpha / 2))
    p_lower = float(np.mean(boot_medians <= target - delta))
    p_upper = float(np.mean(boot_medians >= target + delta))
    p_tost = max(p_lower, p_upper)
    inside_band = (target - delta) <= obs_median <= (target + delta)
    equivalence_pass = (
        (p_tost < alpha) and (ci_lo > target - delta) and (ci_hi < target + delta)
    )
    return {
        "median_obs": obs_median,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "tost_p_lower": p_lower,
        "tost_p_upper": p_upper,
        "tost_p": p_tost,
        "equivalence_pass": bool(equivalence_pass),
        "median_inside_band": bool(inside_band),
        "ci_inside_band": bool(ci_lo > target - delta and ci_hi < target + delta),
        "target": float(target),
        "delta": float(delta),
        "n": int(n),
    }


# --------------------------------------------------------------------------- #
# H3 integrated verdict (SUPPORTED / NOT_SUPPORTED_* / CONTRADICTED)
#
# Framework v1.0.5 §3.3 verdict naming locked at framework time. NOT PASS/NULL/FAIL —
# guards against "PASS = proves independence" misreading.
#
# Mark-transition layer: lag1_same_excess + window_excess @ {10, 30, 60, 1800}s +
#   run_length_lift; "compatible" = all 6 TOST tests equivalence_pass = True.
# Endpoint geometry layer: endpoint_jaccard_first_half_median ≥ 0.7 **OR**
#   endpoint_jaccard_odd_even_median ≥ 0.7 (OR combinator — project convention
#   from AGENTS.md forward_reverse_reproduced = split-half OR odd-even).
#
# Robustness check: when mark-transition layer fails, inspect each failing metric's
# leave_one_out_min_pass_rate. If any failing metric has min_pass_rate ≥
# `loo_robust_threshold`, the cohort failure is single-subject sensitive — verdict
# is NOT_SUPPORTED_MEMORY. If all failing metrics have min_pass_rate <
# loo_robust_threshold, the failure persists under LOO → robust failure →
# CONTRADICTED.
# --------------------------------------------------------------------------- #


_MARK_TRANSITION_KEYS = (
    "lag1_same_excess",
    "window_excess_10s",
    "window_excess_30s",
    "window_excess_60s",
    "window_excess_1800s",
    "run_length_lift",
)


def compute_h3_integrated_verdict(
    cohort_tost: Dict[str, dict],
    endpoint_jaccard_first_half_median: float,
    endpoint_jaccard_odd_even_median: float,
    *,
    jaccard_threshold: float = 0.70,
    loo_robust_threshold: float = 0.50,
) -> str:
    """Integrate H3 mark-transition + endpoint-stability into a framework v1.0.5 verdict.

    Returns one of:
      "SUPPORTED" — mark-transition layer compatible AND endpoint geometry stable
      "NOT_SUPPORTED_GEOMETRY_UNSTABLE" — mark-transition compatible BUT geometry unstable
      "NOT_SUPPORTED_MEMORY" — mark-transition not compatible (single-subject sensitive) +
                                geometry stable
      "NOT_SUPPORTED_BOTH" — mark-transition not compatible + geometry unstable
      "CONTRADICTED" — mark-transition robustly not compatible (LOO does not restore) +
                       geometry stable

    Args:
      cohort_tost: dict keyed by metric name (see `_MARK_TRANSITION_KEYS`). Each value carries
        `equivalence_pass` and optionally `leave_one_out_min_pass_rate` (fraction of LOO
        subsets that still pass equivalence; populated by `cohort_tost_with_loo` in Task 11.5;
        defaults to 1.0 if absent, which means "treat as not robust"). Per advisor catch C,
        Task 11.5 fills this field; without it the CONTRADICTED branch never fires.
      endpoint_jaccard_first_half_median: cohort median of subject-level mean Jaccard endpoint
        on first-half / second-half split.
      endpoint_jaccard_odd_even_median: cohort median of same on odd / even block split.
      jaccard_threshold: minimum Jaccard for "endpoint stable" call (default 0.70).
      loo_robust_threshold: a failing metric is "robust" iff LOO min_pass_rate <
        loo_robust_threshold (default 0.50; i.e., < half of LOO subsets restore equivalence).
    """
    mark_pass = all(cohort_tost[k]["equivalence_pass"] for k in _MARK_TRANSITION_KEYS)
    endpoint_stable = (
        endpoint_jaccard_first_half_median >= jaccard_threshold
        or endpoint_jaccard_odd_even_median >= jaccard_threshold
    )  # OR — project convention (advisor catch A, AGENTS.md forward_reverse_reproduced)

    if mark_pass and endpoint_stable:
        return "SUPPORTED"
    if mark_pass and not endpoint_stable:
        return "NOT_SUPPORTED_GEOMETRY_UNSTABLE"
    # mark not pass — check LOO robustness of failures
    failing = [k for k in _MARK_TRANSITION_KEYS if not cohort_tost[k]["equivalence_pass"]]
    robust = any(
        cohort_tost[k].get("leave_one_out_min_pass_rate", 1.0) < loo_robust_threshold
        for k in failing
    )
    if not endpoint_stable:
        return "NOT_SUPPORTED_BOTH"
    return "CONTRADICTED" if robust else "NOT_SUPPORTED_MEMORY"


# --------------------------------------------------------------------------- #
# H4 epoch slicer
#
# Block-aware: each recording block is sliced independently into contiguous
# `epoch_hours` windows starting at the block's t_start. Inter-block gaps are
# preserved as gaps — no phantom epoch covers a gap.
#
# Time order is preserved across the returned list.
# --------------------------------------------------------------------------- #


def slice_events_into_epochs(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    block_time_ranges: List[Tuple[float, float]],
    epoch_hours: float = 2.0,
    min_events: int = 10,
) -> List[Dict[str, np.ndarray]]:
    """Slice events into time-ordered, block-aware epochs of fixed wall-clock duration.

    Each block is independently sliced into `floor(block_duration / epoch_seconds)` contiguous
    epochs starting at the block's `t_start`. Events falling in
    `[t_start + k * Δ, t_start + (k+1) * Δ)` are assigned to epoch k. Epochs with fewer than
    `min_events` events are dropped.

    Returns a list of dicts, one per kept epoch:
      `{"t_start": float, "t_end": float, "event_indices": np.ndarray, "block_index": int}`
    """
    times = np.asarray(event_abs_times, dtype=float)
    epoch_seconds = epoch_hours * 3600.0
    epochs: List[Dict[str, np.ndarray]] = []
    for bi, (b_start, b_end) in enumerate(block_time_ranges):
        block_duration = b_end - b_start
        n_epochs = int(np.floor(block_duration / epoch_seconds))
        for k in range(n_epochs):
            t_s = b_start + k * epoch_seconds
            t_e = t_s + epoch_seconds
            mask = (times >= t_s) & (times < t_e)
            idx = np.where(mask)[0]
            if len(idx) < min_events:
                continue
            epochs.append({
                "t_start": float(t_s),
                "t_end": float(t_e),
                "event_indices": idx,
                "block_index": bi,
            })
    return epochs


# --------------------------------------------------------------------------- #
# H4 per-epoch local endpoint + Jaccard
#
# CLAUDE.md §6.1 re-use check (question-match):
#   PR-6 anchoring's full helper enforces minimum n_valid + audit gates designed for
#   cohort-level reproducibility. H4 epoch endpoint is a different question — "what's
#   the endpoint set of this epoch's events", gated only by `epoch had ≥ min_events
#   events`. Therefore we write a Phase 2-local thin helper that operates on the
#   epoch's events ONLY, not the PR-6 anchoring driver. The statistic computed
#   (top-k by mean participation per cluster) is the same as PR-2's `template_rank`
#   restricted to the epoch's events.
# --------------------------------------------------------------------------- #


def compute_local_endpoint(
    events_bool: np.ndarray,
    labels: np.ndarray,
    k: int = 3,
    valid_mask: np.ndarray = None,
) -> Dict[int, Dict[str, List[int]]]:
    """Compute per-cluster top-k source / bottom-k sink endpoint from a slice of events.

    For each cluster c present in `labels`, take `mean(events_bool[labels == c], axis=0)` →
    channel-wise participation rate. The top-k channels (highest participation) form the
    source; the bottom-k channels (lowest participation) form the sink.

    If `valid_mask` is provided, only channels where mask is True are eligible. When the
    eligible pool is smaller than 2k, k degrades gracefully to `eligible // 2`.

    Returns `{cluster_id: {"source": [int, ...], "sink": [int, ...]}}`.
    """
    n_ch = events_bool.shape[1]
    if valid_mask is None:
        valid_mask = np.ones(n_ch, dtype=bool)
    eligible_idx = np.where(valid_mask)[0]
    out: Dict[int, Dict[str, List[int]]] = {}
    for c in np.unique(labels):
        cluster_rows = events_bool[labels == c]
        if cluster_rows.shape[0] == 0:
            continue
        ch_mean = cluster_rows.mean(axis=0)
        elig_means = ch_mean[eligible_idx]
        k_eff = k if len(elig_means) >= 2 * k else max(1, len(elig_means) // 2)
        top_k_local = np.argsort(elig_means)[::-1][:k_eff]
        bot_k_local = np.argsort(elig_means)[:k_eff]
        out[int(c)] = {
            "source": [int(eligible_idx[i]) for i in top_k_local],
            "sink": [int(eligible_idx[i]) for i in bot_k_local],
        }
    return out


def endpoint_jaccard(
    local: Dict[int, Dict[str, List[int]]],
    global_: Dict[int, Dict[str, List[int]]],
    cluster_id: int,
) -> float:
    """Jaccard(source ∪ sink) between local and global per-cluster endpoint sets.

    Returns |L ∩ G| / |L ∪ G|, or 0.0 if both sets empty.
    """
    L = set(local[cluster_id]["source"]) | set(local[cluster_id]["sink"])
    G = set(global_[cluster_id]["source"]) | set(global_[cluster_id]["sink"])
    union = L | G
    if not union:
        return 0.0
    return len(L & G) / len(union)


# --------------------------------------------------------------------------- #
# H4 I_rate normalized rate instability
#
# Framework v1.0.5 §3.4 prose is mathematically degenerate (advisor catch B,
# 2026-05-23). Phase 2 v1.0.0 implements BOTH the literal `epoch_order_shuffle`
# null (degenerate by construction; reports `I_rate_undefined_under_shuffle_null`)
# AND the proposed `circular_shift_within_block` null (non-degenerate).
#
# User decides which null enters the framework on return. See
# docs/archive/topic4/sef_itp_phase2/spec_amendment_2026-05-23.md.
# --------------------------------------------------------------------------- #


def compute_I_rate_normalized(
    rates: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Literal framework v1.0.5 §3.4 null (degenerate by construction).

    I_rate = std(log(rate)) across epochs / sqrt(var of std(log(rate)) over epoch-order
             shuffles).

    Returns I_rate = +inf and `I_rate_undefined_under_shuffle_null=True` whenever the null
    is degenerate (always, since std is permutation-invariant; kept for spec-faithful audit).
    """
    rates = np.asarray(rates, dtype=float)
    log_rates = np.log(rates + 1e-12)
    obs_std = float(np.std(log_rates))
    rng = np.random.default_rng(seed)
    null_stds = np.array([
        np.std(log_rates[rng.permutation(len(rates))]) for _ in range(n_perm)
    ])
    null_var = float(np.var(null_stds))
    null_mean = float(np.mean(null_stds))
    undefined = bool(null_var < 1e-12)
    I_rate = float("inf") if undefined else obs_std / np.sqrt(null_var)
    return {
        "I_rate": I_rate,
        "log_rate_std_obs": obs_std,
        "null_std_mean": null_mean,
        "null_std_var": null_var,
        "I_rate_undefined_under_shuffle_null": undefined,
        "null_method": "epoch_order_shuffle",
        "n_epochs": int(len(rates)),
    }


def compute_I_rate_normalized_circular_shift(
    event_abs_times: np.ndarray,
    block_time_ranges: List[Tuple[float, float]],
    epoch_hours: float = 2.0,
    min_events: int = 10,
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Non-degenerate null variant (proposed Phase 2 v1.0.0 spec amendment).

    For each permutation: pick a random offset Δ ∈ [0, epoch_seconds) per block, shift event
    times within block by Δ with wrap-around, re-slice into epochs starting at block start,
    compute rate per epoch, take std(log(rate)). The null distribution is non-degenerate
    because epoch-membership of events is randomized while sub-epoch temporal structure is
    preserved.
    """
    times = np.asarray(event_abs_times, dtype=float)
    epoch_seconds = epoch_hours * 3600.0
    rng = np.random.default_rng(seed)

    def _rates_for(times_in: np.ndarray) -> np.ndarray:
        rates: List[float] = []
        for (b_start, b_end) in block_time_ranges:
            block_duration = b_end - b_start
            n_epochs = int(np.floor(block_duration / epoch_seconds))
            for k in range(n_epochs):
                t_s = b_start + k * epoch_seconds
                t_e = t_s + epoch_seconds
                cnt = int(np.sum((times_in >= t_s) & (times_in < t_e)))
                if cnt >= min_events:
                    rates.append(cnt / epoch_hours)
        return np.asarray(rates, dtype=float)

    obs_rates = _rates_for(times)
    obs_log_rates = np.log(obs_rates + 1e-12)
    obs_std = float(np.std(obs_log_rates))

    null_stds = []
    for _ in range(n_perm):
        shifted = times.copy()
        for (b_start, b_end) in block_time_ranges:
            mask = (shifted >= b_start) & (shifted < b_end)
            block_duration = b_end - b_start
            delta = float(rng.uniform(0.0, epoch_seconds))
            wrapped = ((shifted[mask] + delta - b_start) % block_duration) + b_start
            shifted[mask] = wrapped
        null_log_rates = np.log(_rates_for(shifted) + 1e-12)
        null_stds.append(float(np.std(null_log_rates)))

    null_stds_arr = np.asarray(null_stds)
    null_var = float(np.var(null_stds_arr))
    null_mean = float(np.mean(null_stds_arr))
    I_rate = obs_std / np.sqrt(null_var) if null_var > 1e-12 else float("inf")
    return {
        "I_rate": I_rate,
        "log_rate_std_obs": obs_std,
        "null_std_mean": null_mean,
        "null_std_var": null_var,
        "n_epochs": int(len(obs_rates)),
        "null_method": "circular_shift_within_block",
    }


# --------------------------------------------------------------------------- #
# H4 I_geom normalized endpoint-geometry instability
#
# Per framework v1.0.5 §3.4: null is "role-shuffle endpoint per epoch within
# valid_mask=True pool". For each permutation, replace each epoch's local endpoint
# with a random sample of `endpoint_size` channels from the valid pool, then compute
# the per-epoch Jaccard against global and the across-epoch std. This null is
# non-degenerate by construction (random samples produce variable Jaccard).
# --------------------------------------------------------------------------- #


def compute_I_geom_normalized(
    per_epoch_local: List[Dict[int, Dict[str, List[int]]]],
    global_endpoint: Dict[int, Dict[str, List[int]]],
    valid_mask: np.ndarray,
    endpoint_size: int,
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Normalized endpoint-geometry instability across epochs.

    obs: for each epoch e, compute mean-across-cluster Jaccard(local_e, global). Take
    std(1 - Jaccard_e) across epochs as `geom_dispersion_std_obs`.

    null: for each permutation, for each epoch, pick `endpoint_size` channels uniformly at
    random from `valid_mask=True` pool, split half-source half-sink, recompute Jaccard, take
    std across epochs. 1000 permutations give null distribution of std.

    I_geom = obs_std / sqrt(var(null_std)).
    """
    cluster_ids = list(global_endpoint.keys())
    eligible = np.where(valid_mask)[0]
    rng = np.random.default_rng(seed)

    def _epoch_jaccard(local: Dict[int, Dict[str, List[int]]]) -> float:
        js = [endpoint_jaccard(local, global_endpoint, c) for c in cluster_ids if c in local]
        return float(np.mean(js)) if js else 0.0

    obs_dispersion = np.array([1.0 - _epoch_jaccard(le) for le in per_epoch_local])
    obs_std = float(np.std(obs_dispersion))
    n_epochs = len(per_epoch_local)
    half = endpoint_size // 2

    null_stds: List[float] = []
    for _ in range(n_perm):
        epoch_jacs: List[float] = []
        for _epoch in range(n_epochs):
            chosen = rng.choice(eligible, size=endpoint_size, replace=False)
            random_local = {
                c: {"source": chosen[:half].tolist(), "sink": chosen[half:].tolist()}
                for c in cluster_ids
            }
            epoch_jacs.append(_epoch_jaccard(random_local))
        null_dispersion = 1.0 - np.array(epoch_jacs)
        null_stds.append(float(np.std(null_dispersion)))

    null_stds_arr = np.asarray(null_stds)
    null_var = float(np.var(null_stds_arr))
    null_mean = float(np.mean(null_stds_arr))
    I_geom = obs_std / np.sqrt(null_var) if null_var > 1e-12 else float("inf")
    return {
        "I_geom": I_geom,
        "geom_dispersion_std_obs": obs_std,
        "null_std_mean": null_mean,
        "null_std_var": null_var,
        "n_epochs": int(n_epochs),
    }
