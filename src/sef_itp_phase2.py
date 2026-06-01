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
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wilcoxon

# Reuse PR-2/PR-6 template estimator + endpoint extraction (CLAUDE.md §6.1
# question-match): per-epoch H4 v1.1 asks "top-k source / bottom-k sink by template
# rank in this epoch", same question as PR-6 anchoring asks over full data. Reusing
# the same primitives guarantees per-epoch B1(full) == PR-6 set-equality on calibration.
from src.interictal_propagation import _legacy_hist_mean_rank
from src.rank_displacement import compute_swap_score_sweep
from src.template_anatomical_anchoring import extract_endpoint_middle

__version__ = "v1.1.0"  # v1.1 — rank-based endpoint geometry drift (user-catch 2026-05-23)


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
    epoch_hours: float = 1.0,
    min_events: int = 10,
    epoch_tolerance: float = 0.0,
) -> List[Dict[str, np.ndarray]]:
    """Slice events into time-ordered, block-aware epochs of fixed wall-clock duration.

    Each block is independently sliced into `floor((block_duration / epoch_seconds) +
    epoch_tolerance)` contiguous epochs starting at the block's `t_start`. Events falling in
    `[t_start + k * Δ, t_start + (k+1) * Δ)` are assigned to epoch k. Epochs with fewer than
    `min_events` events are dropped.

    `epoch_tolerance` (default 0.0 = strict floor) lets short blocks count as 1 epoch when
    their duration is close-but-under `epoch_seconds`. For Epilepsiae blocks (~59 min 41 s
    natural duration vs 1h target), set `epoch_tolerance=0.1` to count each block as 1 epoch.

    Returns a list of dicts, one per kept epoch:
      `{"t_start": float, "t_end": float, "event_indices": np.ndarray, "block_index": int}`
    """
    times = np.asarray(event_abs_times, dtype=float)
    epoch_seconds = epoch_hours * 3600.0
    epochs: List[Dict[str, np.ndarray]] = []
    for bi, (b_start, b_end) in enumerate(block_time_ranges):
        block_duration = b_end - b_start
        n_epochs = int(np.floor(block_duration / epoch_seconds + epoch_tolerance))
        for k in range(n_epochs):
            t_s = b_start + k * epoch_seconds
            t_e = t_s + epoch_seconds
            # Last epoch may end early when block_duration < (k+1) * epoch_seconds
            # (short block + epoch_tolerance > 0); clip to block end.
            if t_e > b_end:
                t_e = b_end
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

    ⚠️ **v1.1 banner (2026-05-23 user catch)**: this helper measures **participation field
    top-k/bottom-k**, NOT propagation rank endpoint. v1.0 used this as the H4 main
    endpoint definition — user 2026-05-23 catch flagged that this is
    "participation field drift", not "propagation endpoint geometry drift". v1.1
    keeps this function as **supplementary** sensitivity (participation-field stability)
    and adds `compute_local_rank_endpoint` as the new H4 main-line endpoint extractor.

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
# v1.1 — rank-based endpoint (H4 main line; user catch 2026-05-23)
#
# Per-channel masked mean lag rank → top-k source / bottom-k sink. Replaces the
# v1.0 participation-field top-k as the H4 main endpoint extractor.
#
# Phantom-mask discipline (AGENTS.md cross-PR lagPatRank contract):
# every non-participating channel in *_lagPat*.npz carries a phantom int rank from
# the legacy producer (hfo_net.py:289 argsort(argsort) is unmasked). Inclusion of
# those phantom ranks in mean = silent contamination. We mask by bools (canonical
# participation flag) before summing. Reference: src.lagpat_rank_audit
# .build_masked_kmeans_features for the same pattern.
# --------------------------------------------------------------------------- #


def _per_cluster_template_rank(
    ranks: np.ndarray,
    bools: np.ndarray,
    cluster_evt_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-cluster template rank using PR-2's legacy estimator.

    Returns (template_rank, valid_mask) per channel:
      - template_rank: `argsort(argsort(_legacy_hist_mean_rank(masked_ranks)))` — int 0..n_ch-1.
        Same estimator PR-2 stores in `adaptive_cluster.clusters[k].template_rank` and PR-6
        anchoring consumes. For non-participating channels, _legacy_hist_mean_rank's
        `template[ci] = ci` fallback kicks in (CLAUDE.md cross-PR contract `template_rank`).
      - valid_mask: per-channel "any participation in cluster's epoch events" — mirrors
        PR-6 anchoring runner's `per_cluster_masks` construction (channels with ≥1
        participating event in the cluster slice).

    Phantom-mask discipline (AGENTS.md cross-PR `lagPatRank`): non-participating
    channels carry phantom int ranks in raw lagPat. `_legacy_hist_mean_rank` already
    filters by `bools[ci] > 0` per channel, so the phantom ranks never enter the
    template estimator. The valid_mask additionally enforces downstream exclusion.
    """
    n_ch = ranks.shape[0]
    if cluster_evt_idx.size == 0:
        return np.zeros(n_ch, dtype=int), np.zeros(n_ch, dtype=bool)
    cl_ranks = ranks[:, cluster_evt_idx]
    cl_bools = bools[:, cluster_evt_idx]
    template = _legacy_hist_mean_rank(cl_ranks, cl_bools)
    template_rank = np.argsort(np.argsort(template, kind="stable"), kind="stable")
    valid_mask = cl_bools.any(axis=1)
    return template_rank, valid_mask


def compute_local_rank_endpoint(
    ranks: np.ndarray,
    bools: np.ndarray,
    labels: np.ndarray,
    event_indices: np.ndarray,
    k: int = 3,
    valid_mask_per_cluster: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[int, Dict[str, List[int]]]:
    """Per-cluster top-k source / bottom-k sink endpoint by **template rank**.

    Calibration contract (C5 in plan): when called with `event_indices = all events`,
    must reproduce PR-6 anchoring's `per_template[].source/sink` per cluster, because
    we reuse the same primitives (`_legacy_hist_mean_rank` template + `extract_endpoint_middle`
    top-k selection + per-cluster valid_mask = "any participation in cluster events").

    For each cluster c present in `labels[event_indices]`:
      1. Slice events: `cluster_evt_idx = event_indices ∩ {labels == c}`.
      2. Compute `template_rank` (legacy hist mean rank → argsort(argsort)) per channel.
      3. valid_mask: **always** `derived_valid` (= channels that participated in cluster
         events within this epoch slice). If `valid_mask_per_cluster` is also provided,
         it acts as an upstream RESTRICTION (intersection): `mask = derived_valid &
         external_mask`. The external mask **never replaces** derived — this prevents
         full-data PR-6 mask from re-introducing zero-participation channels in per-epoch
         endpoint (user-review catch 2026-05-23).
      4. Source / sink via `extract_endpoint_middle` (PR-6 helper): lowest k = source,
         highest k = sink, both restricted to valid_mask.
      5. k degrades to `n_valid // 2` automatically via extract_endpoint_middle's exit
         (returns empty source/sink when n_valid < 2k).

    Args:
      ranks: (n_ch, n_events_total) per-event per-channel lag rank.
      bools: (n_ch, n_events_total) per-event per-channel participation flag.
      labels: (n_events_total,) cluster id per event.
      event_indices: indices into the global arrays for this epoch / slice.
      k: target endpoint size (source / sink each).
      valid_mask_per_cluster: optional per-cluster upstream restriction mask; intersected
        with epoch-derived participation mask (NEVER replaces derived; see step 3).
        Typical production usage: pass `None` (derived alone is correct). Calibration
        can pass PR-6 per-cluster mask, which equals derived on full data anyway.

    Returns:
      `{cluster_id: {"source": [int, ...], "sink": [int, ...]}}` — only clusters with
      at least one event in `event_indices` and at least 2k valid channels are included.
      Smaller-pool clusters degrade gracefully: when extract_endpoint_middle reports
      `n_ch<6`-style exit, we fall back to k = n_valid // 2.
    """
    n_ch = ranks.shape[0]
    out: Dict[int, Dict[str, List[int]]] = {}
    if event_indices.size == 0:
        return out
    slice_labels = labels[event_indices]
    channel_names_pos = list(range(n_ch))  # use channel indices as names → endpoint returns indices
    for c in np.unique(slice_labels):
        cluster_evt_idx = event_indices[slice_labels == c]
        if cluster_evt_idx.size == 0:
            continue
        template_rank, derived_valid = _per_cluster_template_rank(ranks, bools, cluster_evt_idx)
        # External mask (if any) is an upstream RESTRICTION only — intersect with derived.
        # Never replace derived: a full-data PR-6 mask would otherwise re-introduce channels
        # that didn't participate in this epoch (silent participation-field pollution
        # because _legacy_hist_mean_rank fallbacks template[ci] = ci for non-participating).
        if valid_mask_per_cluster is not None and int(c) in valid_mask_per_cluster:
            external = np.asarray(valid_mask_per_cluster[int(c)], dtype=bool)
            mask = derived_valid & external
        else:
            mask = derived_valid
        if not mask.any():
            continue
        n_valid = int(mask.sum())
        k_eff = k if n_valid >= 2 * k else max(1, n_valid // 2)
        rec = extract_endpoint_middle(
            channel_names=channel_names_pos,
            template_rank=template_rank.tolist(),
            n=k_eff,
            valid_mask=mask.tolist(),
        )
        if rec.get("exit_reason") is not None:
            continue
        out[int(c)] = {
            "source": [int(x) for x in rec["source"]],
            "sink": [int(x) for x in rec["sink"]],
        }
    return out


# --------------------------------------------------------------------------- #
# v1.1 — spatial radius drift (H4 secondary geometry metric; user catch 2026-05-23)
#
# Per-cluster per-epoch endpoint set radius — source / sink computed independently
# (don't混 source+sink into one radius, otherwise "轴变长" and "每端变散" are confounded
# — user 2026-05-23 catch). Source / sink centroid distance gives propagation axis length.
# Persistent homology (cloud-level) deferred to future analysis (per user's catch:
# "持续同调留给完整 participation field 分析").
# --------------------------------------------------------------------------- #


def _min_enclosing_ball_radius(pts: np.ndarray) -> float:
    """Smallest enclosing ball radius for k ≤ 5 points (brute force).

    Tries (a) 2-point antipodal spheres; (b) circumspheres of all triples (in 3D
    each triple defines a circle; the smallest enclosing sphere passing through a
    triple is the circumcircle's diameter sphere); returns smallest radius whose
    sphere contains all points within numeric tolerance. Welzl's algorithm is
    overkill for k ≤ 5.
    """
    k = pts.shape[0]
    if k <= 1:
        return 0.0
    if k == 2:
        return 0.5 * float(np.linalg.norm(pts[0] - pts[1]))
    candidates: List[float] = []
    eps = 1e-9
    # (a) 2-point antipodal
    for i in range(k):
        for j in range(i + 1, k):
            c = 0.5 * (pts[i] + pts[j])
            r = 0.5 * float(np.linalg.norm(pts[i] - pts[j]))
            if np.all(np.linalg.norm(pts - c, axis=1) <= r + eps):
                candidates.append(r)
    # (b) 3-point circumsphere (in 3D, the smallest sphere passing through 3 points
    # is the circumscribed circle of the triangle they form, lifted to 3D — its
    # diameter is the circumdiameter of the triangle).
    for i in range(k):
        for j in range(i + 1, k):
            for m in range(j + 1, k):
                try:
                    c, r = _triangle_circumsphere(pts[i], pts[j], pts[m])
                except (ValueError, np.linalg.LinAlgError):
                    continue
                if np.all(np.linalg.norm(pts - c, axis=1) <= r + eps):
                    candidates.append(r)
    if not candidates:
        # fallback (shouldn't happen for k ≤ 5 with reasonable input): use farthest-from-centroid
        c = pts.mean(axis=0)
        return float(np.max(np.linalg.norm(pts - c, axis=1)))
    return min(candidates)


def _triangle_circumsphere(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, float]:
    """Circumcenter + circumradius of triangle (a, b, c) in 3D space (sphere through all three).

    Center lies in the plane of the triangle; standard barycentric formula.
    """
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    norm_sq = float(np.dot(cross, cross))
    if norm_sq < 1e-18:
        raise ValueError("degenerate triangle (collinear)")
    alpha = float(np.dot(ac, ac)) * float(np.dot(ab, ab) - np.dot(ab, ac)) / (2.0 * norm_sq)
    beta = float(np.dot(ab, ab)) * float(np.dot(ac, ac) - np.dot(ab, ac)) / (2.0 * norm_sq)
    center = a + alpha * ab + beta * ac
    radius = float(np.linalg.norm(center - a))
    return center, radius


def compute_endpoint_spatial_radius(
    endpoint_indices: List[int],
    coords: np.ndarray,
) -> Dict[str, float]:
    """Spatial radius metrics for an endpoint channel set.

    Returns:
      centroid_rms: sqrt(mean(||pt - centroid||²)) over endpoint pts; 0 for k=1, NaN for empty.
      mean_pairwise: mean of pairwise Euclidean distances; NaN for k ≤ 1.
      min_enclosing_radius: radius of smallest enclosing ball (brute force for k ≤ 5).
      n_points: number of endpoint channels.

    coords: (n_ch, 3) — assumed mm Euclidean (Phase 1 contract; not re-asserted here
    since coords come pre-validated from `src.seeg_coord_loader.load_subject_coords`).
    """
    if not endpoint_indices:
        return {
            "centroid_rms": float("nan"),
            "mean_pairwise": float("nan"),
            "min_enclosing_radius": float("nan"),
            "n_points": 0,
        }
    pts = coords[endpoint_indices]
    k = int(pts.shape[0])
    if k == 1:
        return {
            "centroid_rms": 0.0,
            "mean_pairwise": float("nan"),
            "min_enclosing_radius": 0.0,
            "n_points": 1,
        }
    centroid = pts.mean(axis=0)
    centroid_rms = float(np.sqrt(((pts - centroid) ** 2).sum(axis=1).mean()))
    pdist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    iu = np.triu_indices(k, k=1)
    mean_pairwise = float(pdist[iu].mean())
    min_enclosing = _min_enclosing_ball_radius(pts)
    return {
        "centroid_rms": centroid_rms,
        "mean_pairwise": mean_pairwise,
        "min_enclosing_radius": min_enclosing,
        "n_points": k,
    }


def compute_source_sink_centroid_distance(
    source_indices: List[int],
    sink_indices: List[int],
    coords: np.ndarray,
) -> float:
    """Propagation axis length = ||centroid(source) − centroid(sink)|| in coord space.

    Returns NaN if either side is empty.
    """
    if not source_indices or not sink_indices:
        return float("nan")
    c_src = coords[source_indices].mean(axis=0)
    c_snk = coords[sink_indices].mean(axis=0)
    return float(np.linalg.norm(c_src - c_snk))


# --------------------------------------------------------------------------- #
# v1.1 — per-epoch decision-k drift (B3; only 9/23 swap-positive subjects)
#
# For each epoch with sufficient events in both clusters, recompute per-cluster
# template rank (via _per_cluster_template_rank) and run rank_displacement
# `compute_swap_score_sweep` to extract `decision_k`. Returns per-epoch decision_k
# list + summary stats. Epochs with insufficient events drop to None (do NOT
# carry forward — advisor catch 4).
#
# Perf budget (advisor catch 4): n_perm=500 default for swap_sweep within drift to
# keep cohort cost reasonable (9 subj × ~50 epoch × 500 perm).
# --------------------------------------------------------------------------- #


def compute_decision_k_drift(
    ranks: np.ndarray,
    bools: np.ndarray,
    labels: np.ndarray,
    epochs: List[Dict[str, np.ndarray]],
    cluster_a: int,
    cluster_b: int,
    min_events_per_cluster: int = 20,
    n_perm: int = 500,
    seed: int = 0,
) -> Dict[str, object]:
    """Per-epoch decision-k drift via rank_displacement swap_sweep.

    Args:
      ranks, bools, labels: global arrays (n_ch, n_events_total) + (n_events_total,)
      epochs: list of dicts with `event_indices` key (output of slice_events_into_epochs)
      cluster_a, cluster_b: cluster IDs from `primary_pair` JSON (NOT np.unique(labels);
        advisor catch 3 — caller passes explicitly to document which pair is swept)
      min_events_per_cluster: epoch event-count gate per cluster (default 20; advisor
        catch 4 — < 20 events makes template rank too noisy)
      n_perm: swap_sweep family-wise null permutation count (default 500 for perf)
      seed: rng seed (incremented per epoch internally)

    Returns:
      {
        "decision_k_per_epoch": [int|None, ...],   # one per input epoch
        "n_epochs_with_decision_k": int,
        "decision_k_std": float,                    # std of finite values; NaN if <2
        "decision_k_mean": float,                   # mean of finite values; NaN if 0
        "decision_k_range": [min, max] | None,
        "cluster_a": int, "cluster_b": int,
        "min_events_per_cluster": int,
        "n_perm": int,
      }
    """
    decision_k_per_epoch: List[Optional[int]] = []
    for ei, ep in enumerate(epochs):
        evt_idx = np.asarray(ep["event_indices"], dtype=int)
        idx_a = evt_idx[labels[evt_idx] == cluster_a]
        idx_b = evt_idx[labels[evt_idx] == cluster_b]
        if len(idx_a) < min_events_per_cluster or len(idx_b) < min_events_per_cluster:
            decision_k_per_epoch.append(None)
            continue
        rank_a, valid_a = _per_cluster_template_rank(ranks, bools, idx_a)
        rank_b, valid_b = _per_cluster_template_rank(ranks, bools, idx_b)
        sweep = compute_swap_score_sweep(
            rank_a.astype(float), rank_b.astype(float),
            valid_a, valid_b,
            n_perm=n_perm, seed=seed + ei,
        )
        dk = sweep.get("decision_k")
        decision_k_per_epoch.append(int(dk) if dk is not None else None)
    finite = [k for k in decision_k_per_epoch if k is not None]
    n_finite = len(finite)
    return {
        "decision_k_per_epoch": decision_k_per_epoch,
        "n_epochs_with_decision_k": n_finite,
        "decision_k_std": float(np.std(finite)) if n_finite >= 2 else float("nan"),
        "decision_k_mean": float(np.mean(finite)) if n_finite else float("nan"),
        "decision_k_range": [int(min(finite)), int(max(finite))] if finite else None,
        "cluster_a": int(cluster_a),
        "cluster_b": int(cluster_b),
        "min_events_per_cluster": int(min_events_per_cluster),
        "n_perm": int(n_perm),
    }


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
    epoch_hours: float = 1.0,
    min_events: int = 10,
    n_perm: int = 1000,
    seed: int = 0,
    epoch_tolerance: float = 0.1,
) -> Dict[str, float]:
    """Non-degenerate null variant (proposed Phase 2 v1.0.0 spec amendment).

    For each permutation: pick a random offset Δ ∈ [0, epoch_seconds) per block, shift event
    times within block by Δ with wrap-around, re-slice into epochs starting at block start,
    compute rate per epoch, take std(log(rate)). The null distribution is non-degenerate
    because epoch-membership of events is randomized while sub-epoch temporal structure is
    preserved.

    `epoch_tolerance` mirrors `slice_events_into_epochs` — needed so Epilepsiae's natural
    ~59min41sec blocks count as 1 epoch when `epoch_hours=1.0`.
    """
    times = np.asarray(event_abs_times, dtype=float)
    epoch_seconds = epoch_hours * 3600.0
    rng = np.random.default_rng(seed)

    def _rates_for(times_in: np.ndarray) -> np.ndarray:
        rates: List[float] = []
        for (b_start, b_end) in block_time_ranges:
            block_duration = b_end - b_start
            n_epochs = int(np.floor(block_duration / epoch_seconds + epoch_tolerance))
            for k in range(n_epochs):
                t_s = b_start + k * epoch_seconds
                t_e = min(t_s + epoch_seconds, b_end)
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


# --------------------------------------------------------------------------- #
# H4 cohort verdict: Wilcoxon signed-rank + Cohen's d
#
# Framework v1.0.5 §3.4 lock:
#   PASS: cohort Wilcoxon p < 0.05 AND Cohen's d ≥ 0.3 (rate more unstable than geom)
#   NULL: not significant or |d| < 0.3
#   FAIL: significant in REVERSE direction (geom more unstable than rate)
#   UNDERPOWERED: n < 6 → no verdict
# --------------------------------------------------------------------------- #


def compute_h4_cohort_verdict(
    I_rate_per_subject: np.ndarray,
    I_geom_per_subject: np.ndarray,
    *,
    p_threshold: float = 0.05,
    cohen_d_floor: float = 0.30,
) -> Dict[str, float]:
    """Cohort H4 verdict per framework v1.0.5 §3.4 lock.

    Returns dict: `verdict`, `wilcoxon_p`, `cohen_d`, `n_subjects`, `median_I_rate`,
    `median_I_geom`. Non-finite I_rate or I_geom rows are dropped before stats.
    """
    a = np.asarray(I_rate_per_subject, dtype=float)
    b = np.asarray(I_geom_per_subject, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    n = len(a)
    if n < 6:
        return {
            "verdict": "UNDERPOWERED",
            "wilcoxon_p": float("nan"),
            "cohen_d": float("nan"),
            "n_subjects": int(n),
            "median_I_rate": float(np.median(a)) if n else float("nan"),
            "median_I_geom": float(np.median(b)) if n else float("nan"),
        }
    diff = a - b  # positive → rate more unstable than geom (SEF-ITP direction)
    p_greater = float(wilcoxon(diff, alternative="greater", zero_method="wilcox").pvalue)
    cohen_d = float(np.mean(diff) / (np.std(diff, ddof=1) + 1e-12))

    if p_greater < p_threshold and cohen_d >= cohen_d_floor:
        verdict = "PASS"
    elif cohen_d < 0:
        p_less = float(wilcoxon(diff, alternative="less", zero_method="wilcox").pvalue)
        verdict = "FAIL" if p_less < p_threshold else "NULL"
    else:
        verdict = "NULL"

    return {
        "verdict": verdict,
        "wilcoxon_p": p_greater,
        "cohen_d": cohen_d,
        "n_subjects": int(n),
        "median_I_rate": float(np.median(a)),
        "median_I_geom": float(np.median(b)),
    }


# --------------------------------------------------------------------------- #
# Cohort TOST with leave-one-out (advisor catch C — required for H3 CONTRADICTED branch)
#
# `compute_h3_integrated_verdict` reads `cohort_tost[metric].leave_one_out_min_pass_rate` to
# distinguish:
#   - robust failure (no LOO subset restores equivalence; min_pass_rate < threshold) → CONTRADICTED
#   - single-subject-sensitive failure (≥ 1 LOO subset restores; min_pass_rate ≥ threshold) →
#     NOT_SUPPORTED_MEMORY
#
# Without populating this field, the verdict logic defaults `min_pass_rate=1.0` → CONTRADICTED
# branch silently never fires.
# --------------------------------------------------------------------------- #


def cohort_tost_with_loo(
    values: np.ndarray,
    target: float,
    delta: float,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Cohort TOST equivalence + leave-one-out robustness.

    Computes `tost_equivalence` on the full cohort, then again for each LOO drop.
    Returns dict with:
      cohort_main: full-cohort tost_equivalence result
      equivalence_pass: alias for cohort_main['equivalence_pass'] (verdict consumer compat)
      leave_one_out: {f"drop_{i}": tost_equivalence_result} per dropped index
      leave_one_out_min_pass_rate: fraction of LOO subsets that pass equivalence
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    main = tost_equivalence(values, target, delta, n_boot=n_boot, alpha=alpha, seed=seed)
    loo: Dict[str, dict] = {}
    n_pass = 0
    for i in range(n):
        sub = np.delete(values, i)
        loo[f"drop_{i}"] = tost_equivalence(
            sub, target, delta, n_boot=n_boot, alpha=alpha, seed=seed + 1 + i,
        )
        if loo[f"drop_{i}"]["equivalence_pass"]:
            n_pass += 1
    return {
        "cohort_main": main,
        "equivalence_pass": main["equivalence_pass"],
        "leave_one_out": loo,
        "leave_one_out_min_pass_rate": n_pass / n if n > 0 else 0.0,
    }
