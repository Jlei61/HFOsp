"""Data-driven ictal-onset SOZ audit (PR-T3-1).

Step 0 helpers:

- ``annotate_clinical_soz``: 3-state (SOZ/nonSOZ/unknown) annotation of an
  analysis channel set against a clinical SOZ list, using the canonical
  bipolar-to-any matcher from ``src.event_periodicity``.

Step 1 helpers (M1 — HFO-onset rate enrichment):

- ``compute_hfo_onset_metrics``: per-seizure / per-channel three M1
  variants (``M1_raw``, ``M1_log``, ``M1_pois``).
- ``rank_top_k_per_seizure``: deterministic top-k by score with NaN
  routed to the bottom and alphabetical tie-break.
- ``aggregate_consensus``: channel must appear in ≥ ``min_fraction`` of
  the per-seizure top-k lists.
- ``aggregate_median_rank``: median rank across seizures (channels
  missing in a seizure receive the worst rank), top-k smallest medians.

Step 2 (M2 ER-log-ratio + Nyquist / filter padding guards) lives in a
later commit.

See ``docs/archive/topic3/pr_t3_1_data_driven_soz_audit_plan_2026-04-30.md``.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set

import numpy as np

from src.event_periodicity import _normalize_channel_name, match_bipolar_soz


SOZ_LABEL = "soz"
NON_SOZ_LABEL = "non_soz"
UNKNOWN_LABEL = "unknown"


def annotate_clinical_soz(
    analysis_channels: Iterable[str],
    clinical_soz: Iterable[str],
) -> Dict[str, str]:
    """Annotate each analysis channel as SOZ / non_soz / unknown.

    Plan §3.2 contract:

    - Bipolar ``X-Y``: if X or Y is in ``clinical_soz`` → ``"soz"``;
      else → ``"non_soz"``.
    - If ``X`` or ``Y`` is empty / whitespace-only (malformed name)
      → ``"unknown"``.
    - CAR / monopolar ``X``: same logic with single contact.

    The matcher reuses ``src.event_periodicity.match_bipolar_soz`` for the
    SOZ vs nonSOZ branch but adds the ``"unknown"`` branch the plan
    requires.

    Returns ``{channel_name: label}`` preserving the input ordering of
    ``analysis_channels``.
    """
    soz_set = {_normalize_channel_name(s) for s in clinical_soz}
    out: Dict[str, str] = {}
    for ch in analysis_channels:
        normalized = _normalize_channel_name(ch)
        parts = [p.strip() for p in normalized.split("-")]
        if any(not p for p in parts):
            out[ch] = UNKNOWN_LABEL
            continue
        out[ch] = match_bipolar_soz(ch, soz_set)
    return out


# ---------------------------------------------------------------------------
# Step 1 — M1 (HFO-onset rate enrichment)
# ---------------------------------------------------------------------------


def compute_hfo_onset_metrics(
    hfo_event_times_per_channel: Mapping[str, np.ndarray],
    seizure_onset: float,
    w_pre: float = 30.0,
    w_post: float = 10.0,
) -> Dict[str, Dict[str, float]]:
    """Per-channel HFO-onset enrichment (plan §3.3).

    Parameters
    ----------
    hfo_event_times_per_channel
        ``{channel_name: ndarray of absolute event timestamps}``.
        Timestamps must be in the same time base as ``seizure_onset``.
    seizure_onset
        Seizure onset (absolute seconds).
    w_pre, w_post
        Pre / post window lengths in seconds (defaults 30 / 10 s).

    Returns
    -------
    dict
        ``{ch: {"n_pre": int, "n_post": int, "rate_pre": float,
        "rate_post": float, "M1_raw": float, "M1_log": float,
        "M1_pois": float}}``.

    Notes
    -----
    Per plan §3.3:

    - Pre window is half-open ``[t_s - w_pre, t_s)``.
    - Post window is half-open ``(t_s, t_s + w_post]``.
    - ``M1_raw   = rate_post - rate_pre``.
    - ``M1_log   = log(n_post + 1) - log(n_pre + 1) - log(W_post / W_pre)``.
    - ``M1_pois  = (n_post - μ_pre) / sqrt(μ_pre + 1)`` with
      ``μ_pre = rate_pre × w_post``.
    - When a channel has zero events in both windows, all three variants
      short-circuit to 0 (plan T4: "通道全无 events → 三 variant 全 0").
      Without the short-circuit, ``M1_log`` would equal
      ``-log(W_post / W_pre)`` and rank silent channels above zero-baseline
      channels with sparse post events, which is not what the plan wants.
    """
    out: Dict[str, Dict[str, float]] = {}
    if w_pre <= 0 or w_post <= 0:
        raise ValueError(f"w_pre ({w_pre}) and w_post ({w_post}) must be positive")
    log_window_correction = math.log(w_post / w_pre)
    for ch, times in hfo_event_times_per_channel.items():
        arr = np.asarray(times, dtype=float).reshape(-1)
        pre_mask = (arr >= seizure_onset - w_pre) & (arr < seizure_onset)
        post_mask = (arr > seizure_onset) & (arr <= seizure_onset + w_post)
        n_pre = int(pre_mask.sum())
        n_post = int(post_mask.sum())
        rate_pre = n_pre / w_pre
        rate_post = n_post / w_post
        if n_pre == 0 and n_post == 0:
            m1_raw = 0.0
            m1_log = 0.0
            m1_pois = 0.0
        else:
            m1_raw = rate_post - rate_pre
            m1_log = (
                math.log(n_post + 1.0)
                - math.log(n_pre + 1.0)
                - log_window_correction
            )
            mu_pre = rate_pre * w_post
            m1_pois = (n_post - mu_pre) / math.sqrt(mu_pre + 1.0)
        out[ch] = {
            "n_pre": n_pre,
            "n_post": n_post,
            "rate_pre": rate_pre,
            "rate_post": rate_post,
            "M1_raw": m1_raw,
            "M1_log": m1_log,
            "M1_pois": m1_pois,
        }
    return out


def rank_top_k_per_seizure(
    per_channel_score: Mapping[str, float],
    k: int,
    nan_handling: str = "rank_last",
) -> List[str]:
    """Top-k channels by ``per_channel_score`` (plan §3.5).

    Sort key is ``(-score, channel_name_ascending)`` so ties resolve
    deterministically by alphabetical channel name. NaN scores are routed
    to the bottom regardless of ``nan_handling`` (the parameter is kept
    for forward compatibility — only ``"rank_last"`` is implemented).
    """
    if k <= 0:
        return []
    if nan_handling != "rank_last":
        raise ValueError(
            f"unsupported nan_handling={nan_handling!r}; only 'rank_last' is supported"
        )
    finite_items: List[tuple] = []
    for ch, score in per_channel_score.items():
        s = float(score)
        if math.isnan(s):
            continue
        finite_items.append((ch, s))
    finite_items.sort(key=lambda x: (-x[1], x[0]))
    return [ch for ch, _ in finite_items[:k]]


def aggregate_consensus(
    per_seizure_topk: Sequence[Sequence[str]],
    min_seizure_fraction: float = 0.5,
) -> Set[str]:
    """Channels appearing in ≥ ``min_seizure_fraction`` of per-seizure top-k
    lists (plan §3.5).
    """
    n_seizures = len(per_seizure_topk)
    if n_seizures == 0:
        return set()
    threshold_count = min_seizure_fraction * n_seizures
    counts: Dict[str, int] = {}
    for topk in per_seizure_topk:
        for ch in set(topk):  # guard against duplicate names within one list
            counts[ch] = counts.get(ch, 0) + 1
    return {ch for ch, c in counts.items() if c >= threshold_count}


def aggregate_median_rank(
    per_seizure_ranks: Sequence[Mapping[str, int]],
    k: int,
) -> Set[str]:
    """Top-k channels by median rank across seizures (plan §3.5).

    Channels missing from a given seizure dict are assigned the worst
    available rank for that seizure (``n_channels``), so a channel that
    only ranks high in one seizure but is missing in others does not
    crowd out a channel that ranks moderately in every seizure. Ties on
    median are broken alphabetically.
    """
    if k <= 0 or not per_seizure_ranks:
        return set()
    all_channels: Set[str] = set()
    max_observed_rank = 0
    for ranks in per_seizure_ranks:
        all_channels.update(ranks.keys())
        if ranks:
            max_observed_rank = max(max_observed_rank, max(ranks.values()))
    worst_rank = max(max_observed_rank, len(all_channels))
    medians: Dict[str, float] = {}
    for ch in all_channels:
        ranks = [s.get(ch, worst_rank) for s in per_seizure_ranks]
        medians[ch] = float(np.median(ranks))
    sorted_channels = sorted(medians.items(), key=lambda x: (x[1], x[0]))
    return {ch for ch, _ in sorted_channels[:k]}


__all__ = [
    "annotate_clinical_soz",
    "SOZ_LABEL",
    "NON_SOZ_LABEL",
    "UNKNOWN_LABEL",
    "compute_hfo_onset_metrics",
    "rank_top_k_per_seizure",
    "aggregate_consensus",
    "aggregate_median_rank",
]
