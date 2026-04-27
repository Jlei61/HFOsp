"""PR-6: Stable Template Endpoint Anatomical Anchoring.

Pure statistical layer.

Contract: docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md.
Reuses match_bipolar_soz / match_bipolar_focus_rel from src.event_periodicity for
channel-name alignment (Yuquan bipolar endpoint match + Epilepsiae CAR direct match).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

from src.event_periodicity import (
    _normalize_channel_name,
    match_bipolar_focus_rel,
    match_bipolar_soz,
)


# ---------------------------------------------------------------------------
# Source / sink / endpoint / middle extraction
# ---------------------------------------------------------------------------
def extract_endpoint_middle(
    channel_names: Sequence[str],
    template_rank: Sequence[int],
    n: int = 3,
    valid_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Split channels into source / sink / endpoint / middle by centroid rank.

    template_rank[i] is the rank of channel_names[i] (smaller = earlier).

    valid_mask (optional): bool per channel; if False, that channel did not
    participate in this cluster's template and must be excluded from source/sink
    candidates.  When valid_mask is None, channels with template_rank == -1 are
    also treated as invalid (PR-6 split-half stores -1 as sentinel).

    Returns exit_reason='n_ch<6' when fewer than 2*n VALID channels are
    available to fill source ∪ sink disjointly.
    """
    channel_names = list(channel_names)
    n_ch_total = len(channel_names)
    rank_arr = np.asarray(template_rank, dtype=float)

    if valid_mask is None:
        # Default: -1 sentinel and non-finite both mark invalid
        valid = np.isfinite(rank_arr) & (rank_arr >= 0)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.size != n_ch_total:
            raise ValueError("valid_mask length must equal channel_names length")
        valid = valid & np.isfinite(rank_arr) & (rank_arr >= 0)

    valid_indices = np.where(valid)[0]
    n_valid = int(valid_indices.size)

    if n_valid < 2 * n:
        return {
            "source": [],
            "sink": [],
            "endpoint": [],
            "middle": [],
            "exit_reason": "n_ch<6",
            "n_valid": n_valid,
        }

    valid_ranks = rank_arr[valid_indices]
    if float(np.std(valid_ranks)) < 1e-12:
        return {
            "source": [],
            "sink": [],
            "endpoint": [],
            "middle": [],
            "exit_reason": "no_polarity",
            "n_valid": n_valid,
        }

    # Sort valid channels by rank (ascending)
    order_within_valid = np.argsort(valid_ranks, kind="mergesort")
    sorted_valid_idx = valid_indices[order_within_valid]

    source_idx = sorted_valid_idx[:n].tolist()
    sink_idx = sorted_valid_idx[-n:][::-1].tolist()  # largest rank first

    source = [channel_names[i] for i in source_idx]
    sink = [channel_names[i] for i in sink_idx]
    endpoint = source + sink
    endpoint_set = set(source_idx) | set(sink_idx)
    # middle is restricted to VALID channels (so non-participating channels are
    # not silently treated as middle-rank evidence)
    middle = [
        channel_names[i] for i in valid_indices.tolist() if i not in endpoint_set
    ]

    return {
        "source": source,
        "sink": sink,
        "endpoint": endpoint,
        "middle": middle,
        "exit_reason": None,
        "n_valid": n_valid,
    }


# ---------------------------------------------------------------------------
# Per-template SOZ anchoring
# ---------------------------------------------------------------------------
def _frac_in_set(channels: Sequence[str], soz_set: set) -> float:
    if not channels:
        return float("nan")
    matched = sum(1 for ch in channels if match_bipolar_soz(ch, soz_set) == "soz")
    return matched / len(channels)


def _frac_focus_rel(channels: Sequence[str], focus_rel: Dict[str, list], label: str) -> float:
    if not channels:
        return float("nan")
    matched = sum(
        1 for ch in channels if match_bipolar_focus_rel(ch, focus_rel) == label
    )
    return matched / len(channels)


def compute_template_anchoring(
    channel_names: Sequence[str],
    template_rank: Sequence[int],
    soz_channels: Sequence[str],
    focus_rel_dict: Optional[Dict[str, list]] = None,
    n: int = 3,
    valid_mask: Optional[Sequence[bool]] = None,
) -> Dict[str, Any]:
    """Compute endpoint/middle SOZ + (optional) focus_rel enrichment for ONE template.

    valid_mask: bool per channel (True if channel participates in this cluster's
    template); used by split-half consumers to restrict source/sink candidates
    to actually-participating channels.

    Output shape:
        {
            'source': [...], 'sink': [...], 'endpoint': [...], 'middle': [...],
            'frac_SOZ_source', 'frac_SOZ_sink', 'frac_SOZ_endpoint', 'frac_SOZ_middle',
            ['frac_<i|l|e>_<source|sink|endpoint|middle>'] if focus_rel_dict given,
            'matched_soz_channels', 'unmatched_soz_channels',
            'exit_reason', 'n_valid',
        }
    """
    parts = extract_endpoint_middle(
        channel_names, template_rank, n=n, valid_mask=valid_mask
    )
    rec: Dict[str, Any] = dict(parts)

    if rec["exit_reason"] is not None:
        return rec

    soz_set = {_normalize_channel_name(c) for c in soz_channels}

    rec["frac_SOZ_source"] = _frac_in_set(rec["source"], soz_set)
    rec["frac_SOZ_sink"] = _frac_in_set(rec["sink"], soz_set)
    rec["frac_SOZ_endpoint"] = _frac_in_set(rec["endpoint"], soz_set)
    rec["frac_SOZ_middle"] = _frac_in_set(rec["middle"], soz_set)

    # Audit unmatched SOZ
    channel_atoms: set = set()
    for ch in channel_names:
        normalized = _normalize_channel_name(ch)
        for part in normalized.split("-"):
            channel_atoms.add(part.strip())
    matched_soz = [c for c in soz_channels if _normalize_channel_name(c) in channel_atoms]
    unmatched_soz = [c for c in soz_channels if _normalize_channel_name(c) not in channel_atoms]
    rec["matched_soz_channels"] = matched_soz
    rec["unmatched_soz_channels"] = unmatched_soz

    if focus_rel_dict is not None:
        for label in ("i", "l", "e"):
            rec[f"frac_{label}_source"] = _frac_focus_rel(
                rec["source"], focus_rel_dict, label
            )
            rec[f"frac_{label}_sink"] = _frac_focus_rel(rec["sink"], focus_rel_dict, label)
            rec[f"frac_{label}_endpoint"] = _frac_focus_rel(
                rec["endpoint"], focus_rel_dict, label
            )
            rec[f"frac_{label}_middle"] = _frac_focus_rel(
                rec["middle"], focus_rel_dict, label
            )

    return rec


# ---------------------------------------------------------------------------
# Subject-level deltas (average over k templates)
# ---------------------------------------------------------------------------
def _safe_mean(values: List[float]) -> float:
    """Mean of finite values; NaN if none are finite."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _subject_delta_for_pair(
    per_template_records: List[Dict[str, Any]],
    a_key: str,
    b_key: str,
) -> Dict[str, Any]:
    diffs = []
    for rec in per_template_records:
        a_val = rec.get(a_key)
        b_val = rec.get(b_key)
        if a_val is None or b_val is None:
            continue
        if not (np.isfinite(a_val) and np.isfinite(b_val)):
            continue
        diffs.append(float(a_val) - float(b_val))
    if not diffs:
        return {"delta": float("nan"), "n_used": 0}
    return {"delta": float(np.mean(diffs)), "n_used": len(diffs)}


def compute_subject_delta(
    per_template_records: List[Dict[str, Any]],
    *,
    focus_rel: bool = False,
) -> Dict[str, Any]:
    """Subject-level deltas, averaging over the subject's k templates.

    Always returns delta_endpoint_vs_middle and delta_source_vs_sink. When
    focus_rel=True, also adds delta_<i|l|e>_endpoint_vs_middle.
    """
    out: Dict[str, Any] = {"n_templates_used": len(per_template_records)}

    em = _subject_delta_for_pair(
        per_template_records, "frac_SOZ_endpoint", "frac_SOZ_middle"
    )
    ss = _subject_delta_for_pair(
        per_template_records, "frac_SOZ_source", "frac_SOZ_sink"
    )
    out["delta_endpoint_vs_middle"] = em["delta"]
    out["n_templates_endpoint_middle_valid"] = em["n_used"]
    out["delta_source_vs_sink"] = ss["delta"]
    out["n_templates_source_sink_valid"] = ss["n_used"]

    if focus_rel:
        for label in ("i", "l", "e"):
            r = _subject_delta_for_pair(
                per_template_records,
                f"frac_{label}_endpoint",
                f"frac_{label}_middle",
            )
            out[f"delta_{label}_endpoint_vs_middle"] = r["delta"]
            out[f"n_templates_{label}_endpoint_middle_valid"] = r["n_used"]

    return out


# ---------------------------------------------------------------------------
# Cohort statistics
# ---------------------------------------------------------------------------
def cohort_wilcoxon(
    deltas: Sequence[float], alternative: str = "greater"
) -> Dict[str, Any]:
    """One-sample Wilcoxon signed-rank test against 0."""
    arr = np.asarray([d for d in deltas if np.isfinite(d)], dtype=float)
    n = arr.size
    if n < 3 or np.allclose(arr, 0.0):
        return {
            "n": int(n),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "median": float(np.median(arr)) if n else float("nan"),
        }
    try:
        stat, p = wilcoxon(arr, alternative=alternative, zero_method="wilcox")
    except ValueError:
        return {
            "n": int(n),
            "statistic": float("nan"),
            "p_value": float("nan"),
            "median": float(np.median(arr)),
        }
    return {
        "n": int(n),
        "statistic": float(stat),
        "p_value": float(p),
        "median": float(np.median(arr)),
    }


def cohort_sign_test(deltas: Sequence[float]) -> Dict[str, Any]:
    """Two-sided sign test (binomial) against null prob 0.5 of positive delta."""
    arr = np.asarray([d for d in deltas if np.isfinite(d)], dtype=float)
    n = arr.size
    n_pos = int(np.sum(arr > 0))
    n_neg = int(np.sum(arr < 0))
    n_eff = n_pos + n_neg  # ties dropped
    if n_eff == 0:
        return {
            "n": int(n),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_effective": 0,
            "p_value": float("nan"),
        }
    # two-sided binomial p-value
    from scipy.stats import binomtest

    res = binomtest(n_pos, n_eff, p=0.5, alternative="two-sided")
    return {
        "n": int(n),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "n_effective": int(n_eff),
        "p_value": float(res.pvalue),
    }


# ---------------------------------------------------------------------------
# Forward/reverse swap mechanism check (H2)
# ---------------------------------------------------------------------------
def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = {_normalize_channel_name(x) for x in a}
    set_b = {_normalize_channel_name(x) for x in b}
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def forward_reverse_swap_check(
    t0_source: Sequence[str],
    t0_sink: Sequence[str],
    t1_source: Sequence[str],
    t1_sink: Sequence[str],
    channel_names: Sequence[str],
    n_perm: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Test whether T0 source ↔ T1 sink (and T0 sink ↔ T1 source) overlap beyond chance.

    swap_score = mean( Jaccard(T0_source, T1_sink), Jaccard(T0_sink, T1_source) )
    Permutation null: randomly redraw 3-channel sets from channel_names.
    """
    n_set = max(len(t0_source), len(t0_sink), len(t1_source), len(t1_sink))
    j1 = _jaccard(t0_source, t1_sink)
    j2 = _jaccard(t0_sink, t1_source)
    swap_score = float((j1 + j2) / 2.0)

    rng = np.random.default_rng(seed)
    pool = list(channel_names)
    n_pool = len(pool)
    if n_pool < 2 * n_set:
        return {
            "swap_score": swap_score,
            "jaccard_t0src_t1snk": j1,
            "jaccard_t0snk_t1src": j2,
            "null_p": float("nan"),
            "null_95th": float("nan"),
            "null_median": float("nan"),
            "n_perm": 0,
            "exit_reason": "channel_pool_too_small",
        }

    null_scores = np.empty(n_perm, dtype=float)
    indices = np.arange(n_pool)
    for k in range(n_perm):
        idx_a = rng.choice(indices, size=n_set, replace=False)
        idx_b = rng.choice(indices, size=n_set, replace=False)
        idx_c = rng.choice(indices, size=n_set, replace=False)
        idx_d = rng.choice(indices, size=n_set, replace=False)
        a = [pool[i] for i in idx_a]
        b = [pool[i] for i in idx_b]
        c = [pool[i] for i in idx_c]
        d = [pool[i] for i in idx_d]
        null_scores[k] = (_jaccard(a, b) + _jaccard(c, d)) / 2.0

    null_p = float(np.mean(null_scores >= swap_score))

    return {
        "swap_score": swap_score,
        "jaccard_t0src_t1snk": j1,
        "jaccard_t0snk_t1src": j2,
        "null_p": null_p,
        "null_median": float(np.median(null_scores)),
        "null_95th": float(np.percentile(null_scores, 95)),
        "n_perm": n_perm,
        "exit_reason": None,
    }


# ---------------------------------------------------------------------------
# Cohort eligibility audit
# ---------------------------------------------------------------------------
def audit_subject_eligibility(
    candidates: List[Dict[str, Any]],
    *,
    n_endpoint: int = 3,
) -> List[Dict[str, Any]]:
    """Check each candidate against PR-6 cohort inclusion (§4 of plan).

    Each candidate dict must contain:
      subject_id, dataset, stable_k, soz_channels, channel_names, template_ranks

    Returns one row per candidate with TWO orthogonal eligibility flags:
      - 'endpoint_defined': source/sink can be extracted (n_ch >= 2*n_endpoint)
      - 'h1_primary_eligible': middle is non-empty so frac_SOZ_middle is defined
        (n_ch > 2*n_endpoint, i.e. n_ch >= 2*n_endpoint + 1)
      - 'pass' is an alias for h1_primary_eligible (the H1 main analysis gate)

    n_ch = 2*n_endpoint (default 6) → endpoint_defined=True but middle is empty,
    so subject can be visualized / case-series'd but cannot enter H1 Wilcoxon.
    Such subjects get exit_reason='middle_empty' and pass=False.

    Inclusion order (first failure wins for exit_reason):
      1. stable_k == 2                  -> 'k!=2'
      2. soz_channels non-empty         -> 'empty_soz'
      3. n_ch >= 2*n_endpoint           -> 'n_ch<6'
      4. matched SOZ >= 1               -> 'no_matched_soz'
      5. centroid has polarity          -> 'no_polarity'
      6. n_ch >= 2*n_endpoint + 1       -> 'middle_empty'  (H1 ineligible only)
    """
    rows = []
    min_n_ch_endpoint = 2 * n_endpoint
    min_n_ch_h1 = 2 * n_endpoint + 1
    for cand in candidates:
        subject_id = cand["subject_id"]
        dataset = cand.get("dataset", "?")
        stable_k = cand.get("stable_k")
        soz_channels = list(cand.get("soz_channels") or [])
        channel_names = list(cand.get("channel_names") or [])
        template_ranks = cand.get("template_ranks") or []
        n_ch = len(channel_names)

        row = {
            "subject_id": subject_id,
            "dataset": dataset,
            "stable_k": stable_k,
            "n_ch": n_ch,
            "n_soz_listed": len(soz_channels),
            "n_soz_matched": 0,
            "exit_reason": None,
            "endpoint_defined": False,
            "h1_primary_eligible": False,
            "pass": False,
        }

        if stable_k != 2:
            row["exit_reason"] = "k!=2"
            rows.append(row)
            continue

        if not soz_channels:
            row["exit_reason"] = "empty_soz"
            rows.append(row)
            continue

        if n_ch < min_n_ch_endpoint:
            row["exit_reason"] = "n_ch<6"
            rows.append(row)
            continue

        # Channel-atom set for matching
        atoms = set()
        for ch in channel_names:
            normalized = _normalize_channel_name(ch)
            for part in normalized.split("-"):
                atoms.add(part.strip())
        matched = [c for c in soz_channels if _normalize_channel_name(c) in atoms]
        row["n_soz_matched"] = len(matched)
        if not matched:
            row["exit_reason"] = "no_matched_soz"
            rows.append(row)
            continue

        # Polarity check on at least one template
        has_polarity = False
        for tr in template_ranks:
            arr = np.asarray(tr, dtype=float)
            if arr.size and np.isfinite(arr).all() and float(np.std(arr)) > 1e-12:
                has_polarity = True
                break
        if not has_polarity:
            row["exit_reason"] = "no_polarity"
            rows.append(row)
            continue

        # endpoint can be extracted at this point (n_ch >= 2*n_endpoint)
        row["endpoint_defined"] = True

        if n_ch < min_n_ch_h1:
            # n_ch == 2*n_endpoint: endpoint covers all, middle empty
            # subject is endpoint_defined but NOT H1 eligible
            row["exit_reason"] = "middle_empty"
            rows.append(row)
            continue

        row["h1_primary_eligible"] = True
        row["pass"] = True
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Step 5a — coreness composite sensitivity (plan §5.2)
# ---------------------------------------------------------------------------
def compute_template_coreness(
    ranks: np.ndarray,
    bools: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> List[Dict[str, Any]]:
    """Per-cluster coreness composite (plan §5.2):

        coreness_ch = (1 / (IQR(ranks_ch) + 1))
                    × |median_rank_ch − (n_ch − 1) / 2| / ((n_ch − 1) / 2)
                    × mean(bools_ch)

    where the IQR / median are taken over events of THIS cluster restricted to
    events where the channel actually participates (bools[ch, ev] == 1) and
    rank is finite.

    Returns a list of n_clusters dicts; each dict has ``coreness``,
    ``median_rank``, ``rank_iqr``, ``participation`` and ``valid_mask``
    (channel participated in ≥1 cluster event).
    """
    if ranks.shape != bools.shape:
        raise ValueError("ranks and bools shapes must match")
    n_ch, n_events = ranks.shape
    labels = np.asarray(labels, dtype=int)
    if labels.size != n_events:
        raise ValueError("labels length must equal n_events")

    out: List[Dict[str, Any]] = []
    for k in range(n_clusters):
        cluster_mask = labels == k
        coreness = np.zeros(n_ch, dtype=float)
        median_rank = np.full(n_ch, np.nan, dtype=float)
        rank_iqr = np.full(n_ch, np.nan, dtype=float)
        participation = np.zeros(n_ch, dtype=float)
        valid = np.zeros(n_ch, dtype=bool)

        if not np.any(cluster_mask):
            out.append(
                {
                    "coreness": coreness.tolist(),
                    "median_rank": median_rank.tolist(),
                    "rank_iqr": rank_iqr.tolist(),
                    "participation": participation.tolist(),
                    "valid_mask": valid.tolist(),
                }
            )
            continue

        c_ranks = ranks[:, cluster_mask]
        c_bools = bools[:, cluster_mask]
        n_cluster_events = int(cluster_mask.sum())

        for ci in range(n_ch):
            participating = c_bools[ci, :] > 0
            participation[ci] = float(participating.sum() / max(n_cluster_events, 1))
            if not np.any(participating):
                continue
            vals = c_ranks[ci, participating]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            valid[ci] = True
            med = float(np.median(vals))
            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            median_rank[ci] = med
            rank_iqr[ci] = iqr
            stability = 1.0 / (iqr + 1.0)
            polarity_denom = (n_ch - 1) / 2.0 if n_ch > 1 else 1.0
            polarity = abs(med - (n_ch - 1) / 2.0) / polarity_denom if polarity_denom > 0 else 0.0
            coreness[ci] = stability * polarity * participation[ci]

        out.append(
            {
                "coreness": coreness.tolist(),
                "median_rank": median_rank.tolist(),
                "rank_iqr": rank_iqr.tolist(),
                "participation": participation.tolist(),
                "valid_mask": valid.tolist(),
            }
        )
    return out


def extract_endpoint_middle_by_coreness(
    channel_names: Sequence[str],
    coreness_record: Dict[str, Any],
    n: int = 3,
) -> Dict[str, Any]:
    """Pick top-2n channels by coreness (within valid set), then split into
    source / sink half-half by median_rank: lower → source, upper → sink.

    Endpoint size is **identical to the main top-3/bottom-3 definition (2n)**
    so the only thing varying between this sensitivity and §5.1 main is the
    selection rule (rank-position vs coreness composite); endpoint cardinality
    is fixed for fair H1 direction comparison.
    """
    channel_names = list(channel_names)
    n_ch = len(channel_names)
    coreness = np.asarray(coreness_record.get("coreness", []), dtype=float)
    median_rank = np.asarray(coreness_record.get("median_rank", []), dtype=float)
    valid = np.asarray(coreness_record.get("valid_mask", []), dtype=bool)

    if coreness.size != n_ch or median_rank.size != n_ch or valid.size != n_ch:
        raise ValueError(
            "coreness_record arrays must align with channel_names length"
        )

    valid_indices = np.where(valid)[0]
    n_valid = int(valid_indices.size)
    n_endpoint = 2 * n
    if n_valid < n_endpoint:
        return {
            "source": [],
            "sink": [],
            "endpoint": [],
            "middle": [],
            "exit_reason": "n_valid<2n",
            "n_valid": n_valid,
        }

    # Sort valid channels by coreness DESCENDING (highest coreness first)
    valid_coreness = coreness[valid_indices]
    if float(np.std(valid_coreness)) < 1e-12:
        return {
            "source": [],
            "sink": [],
            "endpoint": [],
            "middle": [],
            "exit_reason": "no_coreness_polarity",
            "n_valid": n_valid,
        }
    order_desc = np.argsort(-valid_coreness, kind="mergesort")
    top_valid_idx = valid_indices[order_desc[:n_endpoint]]

    # Within top-2n, split by median_rank: lower n → source, upper n → sink
    top_median_rank = median_rank[top_valid_idx]
    inner_order = np.argsort(top_median_rank, kind="mergesort")
    source_idx = top_valid_idx[inner_order[:n]].tolist()
    sink_idx = top_valid_idx[inner_order[-n:][::-1]].tolist()  # largest first

    source = [channel_names[i] for i in source_idx]
    sink = [channel_names[i] for i in sink_idx]
    endpoint = source + sink
    endpoint_set = set(source_idx) | set(sink_idx)
    middle = [
        channel_names[i] for i in valid_indices.tolist() if i not in endpoint_set
    ]

    return {
        "source": source,
        "sink": sink,
        "endpoint": endpoint,
        "middle": middle,
        "exit_reason": None,
        "n_valid": n_valid,
        "n_endpoint": n_endpoint,
    }


def compute_template_anchoring_by_coreness(
    channel_names: Sequence[str],
    coreness_record: Dict[str, Any],
    soz_channels: Sequence[str],
    focus_rel_dict: Optional[Dict[str, list]] = None,
    n: int = 3,
) -> Dict[str, Any]:
    """Like compute_template_anchoring but using coreness-selected endpoints."""
    parts = extract_endpoint_middle_by_coreness(
        channel_names, coreness_record, n=n
    )
    rec: Dict[str, Any] = dict(parts)
    if rec["exit_reason"] is not None:
        return rec

    soz_set = {_normalize_channel_name(c) for c in soz_channels}
    rec["frac_SOZ_source"] = _frac_in_set(rec["source"], soz_set)
    rec["frac_SOZ_sink"] = _frac_in_set(rec["sink"], soz_set)
    rec["frac_SOZ_endpoint"] = _frac_in_set(rec["endpoint"], soz_set)
    rec["frac_SOZ_middle"] = _frac_in_set(rec["middle"], soz_set)

    if focus_rel_dict is not None:
        for label in ("i", "l", "e"):
            rec[f"frac_{label}_source"] = _frac_focus_rel(
                rec["source"], focus_rel_dict, label
            )
            rec[f"frac_{label}_sink"] = _frac_focus_rel(
                rec["sink"], focus_rel_dict, label
            )
            rec[f"frac_{label}_endpoint"] = _frac_focus_rel(
                rec["endpoint"], focus_rel_dict, label
            )
            rec[f"frac_{label}_middle"] = _frac_focus_rel(
                rec["middle"], focus_rel_dict, label
            )

    return rec


# ---------------------------------------------------------------------------
# Step 5b — split-half endpoint robustness via Jaccard
# ---------------------------------------------------------------------------
def compute_split_half_endpoint_jaccards(
    channel_names: Sequence[str],
    cluster_rank_a: Sequence[Sequence[int]],
    cluster_valid_mask_a: Sequence[Sequence[bool]],
    cluster_rank_b_matched_to_a: Sequence[Optional[Sequence[int]]],
    cluster_valid_mask_b_matched_to_a: Sequence[Optional[Sequence[bool]]],
    n: int = 3,
) -> List[Dict[str, Any]]:
    """For each split-A cluster (already aligned via mapping_a_to_b on the B
    side — see ``compute_time_split_reproducibility``), extract source / sink
    / endpoint sets in both halves and compute set Jaccards.

    Inputs use the per-cluster fields written by Step 1's split-half
    extension: rank vectors carry ``-1`` for non-participating channels and
    valid_mask gives the ground truth.  ``cluster_rank_b_matched_to_a[k]``
    can be ``None`` when no Hungarian match was found for split-A cluster k
    (e.g. degenerate split with only one cluster); in that case we record
    ``exit_reason='no_mapping'`` and skip the Jaccards.

    Returns one dict per split-A cluster, with
    ``jaccard_source / jaccard_sink / jaccard_endpoint`` (or exit_reason).
    """
    n_clusters = len(cluster_rank_a)
    out: List[Dict[str, Any]] = []
    for k in range(n_clusters):
        rank_a = cluster_rank_a[k]
        mask_a = cluster_valid_mask_a[k]
        rank_b = cluster_rank_b_matched_to_a[k] if k < len(cluster_rank_b_matched_to_a) else None
        mask_b = (
            cluster_valid_mask_b_matched_to_a[k]
            if k < len(cluster_valid_mask_b_matched_to_a)
            else None
        )
        if rank_b is None or mask_b is None:
            out.append({"cluster_id": k, "exit_reason": "no_mapping"})
            continue

        sa = extract_endpoint_middle(channel_names, rank_a, n=n, valid_mask=mask_a)
        sb = extract_endpoint_middle(channel_names, rank_b, n=n, valid_mask=mask_b)
        if sa["exit_reason"] is not None or sb["exit_reason"] is not None:
            out.append(
                {
                    "cluster_id": k,
                    "exit_reason": f"extract_failed:a={sa['exit_reason']},b={sb['exit_reason']}",
                }
            )
            continue

        out.append(
            {
                "cluster_id": k,
                "jaccard_source": _jaccard(sa["source"], sb["source"]),
                "jaccard_sink": _jaccard(sa["sink"], sb["sink"]),
                "jaccard_endpoint": _jaccard(sa["endpoint"], sb["endpoint"]),
                "n_valid_a": sa["n_valid"],
                "n_valid_b": sb["n_valid"],
                "exit_reason": None,
            }
        )
    return out


__all__ = [
    "extract_endpoint_middle",
    "compute_template_anchoring",
    "compute_subject_delta",
    "cohort_wilcoxon",
    "cohort_sign_test",
    "forward_reverse_swap_check",
    "audit_subject_eligibility",
    "compute_template_coreness",
    "extract_endpoint_middle_by_coreness",
    "compute_template_anchoring_by_coreness",
    "compute_split_half_endpoint_jaccards",
]
