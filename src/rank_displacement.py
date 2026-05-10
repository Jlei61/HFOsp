"""Per-channel signed rank displacement metrics for cluster template comparison.

Supplementary to PR-6 endpoint anchoring (forward/reverse template geometry).
Continuous version of PR-6 discrete swap_node count.

Plan: docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md

Sign anchor (per plan §3.0): T_a = cluster with smaller cluster_id (engineering
only). Δr = rank_T_b − rank_T_a is per-subject only; never aggregate signed
values across subjects.
"""
from __future__ import annotations

from itertools import combinations, product
from math import floor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


def compute_signed_rank_displacement(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    channel_names: Sequence[str],
) -> Dict[str, object]:
    """Compute per-channel signed rank displacement Δr(ch) = rank_b(ch) - rank_a(ch).

    All inputs MUST share the same channel ordering (length n_channels).
    Joint valid set = valid_mask_a AND valid_mask_b. Ranks are re-densified
    within the joint set before subtraction so -1 sentinels in the input
    template_rank cannot pollute the displacement.

    Aggregation: Spearman footrule, Diaconis-Graham normalized footrule,
    Kendall tau, Spearman rho. No NaN imputation; no default valid_mask.
    """
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)
    valid_mask_a = np.asarray(valid_mask_a, dtype=bool)
    valid_mask_b = np.asarray(valid_mask_b, dtype=bool)
    channel_names = list(channel_names)

    n_channels = len(channel_names)
    if not (
        rank_a.shape
        == rank_b.shape
        == valid_mask_a.shape
        == valid_mask_b.shape
        == (n_channels,)
    ):
        raise ValueError(
            f"Shape mismatch: rank_a={rank_a.shape}, rank_b={rank_b.shape}, "
            f"valid_mask_a={valid_mask_a.shape}, valid_mask_b={valid_mask_b.shape}, "
            f"n_channels={n_channels}"
        )

    joint_valid = valid_mask_a & valid_mask_b
    n_valid = int(joint_valid.sum())

    delta_full = np.full(n_channels, np.nan)
    rank_a_dense_full = np.full(n_channels, np.nan)
    rank_b_dense_full = np.full(n_channels, np.nan)
    out: Dict[str, object] = {
        "channel_names": channel_names,
        "joint_valid": joint_valid.tolist(),
        "n_valid": n_valid,
        "rank_a_full": rank_a.tolist(),
        "rank_b_full": rank_b.tolist(),
        "rank_a_dense_full": rank_a_dense_full.tolist(),
        "rank_b_dense_full": rank_b_dense_full.tolist(),
        "signed_displacement_full": delta_full.tolist(),
        "signed_displacement_dense": [],
        "footrule": float("nan"),
        "footrule_max": float("nan"),
        "footrule_normalized": float("nan"),
        "kendall_tau": float("nan"),
        "kendall_p": float("nan"),
        "spearman_rho": float("nan"),
        "spearman_p": float("nan"),
        "exit_reason": "ok",
    }

    if n_valid < 4:
        out["exit_reason"] = f"n_valid<4 (got {n_valid})"
        return out

    r_a_subset = rank_a[joint_valid]
    r_b_subset = rank_b[joint_valid]
    r_a_dense = stats.rankdata(r_a_subset, method="average") - 1.0
    r_b_dense = stats.rankdata(r_b_subset, method="average") - 1.0

    delta_subset = r_b_dense - r_a_dense
    abs_subset = np.abs(delta_subset)

    delta_full[joint_valid] = delta_subset
    rank_a_dense_full[joint_valid] = r_a_dense
    rank_b_dense_full[joint_valid] = r_b_dense

    footrule = float(abs_subset.sum())
    footrule_max = float(floor(n_valid * n_valid / 2))
    footrule_normalized = (
        footrule / footrule_max if footrule_max > 0 else float("nan")
    )

    tau_res = stats.kendalltau(r_a_dense, r_b_dense)
    rho_res = stats.spearmanr(r_a_dense, r_b_dense)

    def _stat(res, idx: int) -> float:
        return float(res.statistic) if hasattr(res, "statistic") else float(res[idx])

    def _pval(res, idx: int) -> float:
        return float(res.pvalue) if hasattr(res, "pvalue") else float(res[idx])

    out.update(
        {
            "rank_a_dense_full": rank_a_dense_full.tolist(),
            "rank_b_dense_full": rank_b_dense_full.tolist(),
            "signed_displacement_full": delta_full.tolist(),
            "signed_displacement_dense": delta_subset.tolist(),
            "footrule": footrule,
            "footrule_max": footrule_max,
            "footrule_normalized": footrule_normalized,
            "kendall_tau": _stat(tau_res, 0),
            "kendall_p": _pval(tau_res, 1),
            "spearman_rho": _stat(rho_res, 0),
            "spearman_p": _pval(rho_res, 1),
        }
    )
    return out


def compute_swap_score_at_k(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    k: int,
) -> float:
    """Variable-k generalization of PR-6 swap_score (top-3-fixed).

    swap_score(k) = mean( Jaccard(top_k_T_a, bottom_k_T_b),
                          Jaccard(bottom_k_T_a, top_k_T_b) )

    Top/bottom are taken on the joint-valid subset under dense ranks.
    Returns 0.0 if 2*k > n_valid (top and bottom would overlap inside
    one template).
    """
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)
    joint_valid = np.asarray(valid_mask_a, dtype=bool) & np.asarray(
        valid_mask_b, dtype=bool
    )
    n_valid = int(joint_valid.sum())
    if n_valid < 4 or 2 * k > n_valid or k < 1:
        return 0.0

    # Stable argsort for deterministic tie-breaking on the joint subset
    valid_idx = np.flatnonzero(joint_valid)
    r_a = rank_a[valid_idx]
    r_b = rank_b[valid_idx]
    order_a = np.argsort(r_a, kind="stable")
    order_b = np.argsort(r_b, kind="stable")
    bot_a = set(valid_idx[order_a[:k]].tolist())
    top_a = set(valid_idx[order_a[-k:]].tolist())
    bot_b = set(valid_idx[order_b[:k]].tolist())
    top_b = set(valid_idx[order_b[-k:]].tolist())

    def _jaccard(s1: set, s2: set) -> float:
        u = s1 | s2
        return (len(s1 & s2) / len(u)) if u else 0.0

    return 0.5 * (_jaccard(top_a, bot_b) + _jaccard(bot_a, top_b))


def compute_swap_score_sweep(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
    alpha_fw: float = 0.05,
    alpha_candidate: float = 0.20,
    score_floor: float = 0.5,
) -> Dict[str, object]:
    """Sweep k = 2 .. floor(n_valid/2), test max-k swap with family-wise null.

    Per-permutation statistic (mirrors observed):
        T_perm[i] = max_k swap_score_perm[i, k]
        T_obs    = max_k swap_score_obs(k)
        decision_k = argmin_k k subject to swap_score_obs(k) == T_obs

    Family-wise p-value (single test on max statistic):
        p_fw = (1 + sum(T_perm >= T_obs)) / (n_perm + 1)

    Dual-tier decision (user-locked 2026-05-07):
        swap_class = "strict"    if T_obs >= score_floor AND p_fw < alpha_fw
                   = "candidate" if T_obs >= score_floor AND alpha_fw <= p_fw < alpha_candidate
                   = "none"      otherwise
        has_swap   = (swap_class == "strict")  # backward compat alias

    Strict tier supports subject-level binary label and channel-level
    swap_endpoint_candidate label (with split-half validation if reused
    downstream). Candidate tier is descriptive / exploratory only - geometry
    suggests swap but FW evidence does not clear standard alpha.

    Per-k swap_score + per-k null_95th are still reported as descriptive
    inputs to the subject x k visualization, but the decision lives on
    the max statistic only - per-k thresholds are NOT used as a decision
    gate (would break FW control).

    Determinism: rng = numpy default_rng(seed); seed and n_perm are
    persisted in the output dict so reruns are bit-reproducible.
    """
    rank_a = np.asarray(rank_a, dtype=float)
    rank_b = np.asarray(rank_b, dtype=float)
    joint_valid = np.asarray(valid_mask_a, dtype=bool) & np.asarray(
        valid_mask_b, dtype=bool
    )
    n_valid = int(joint_valid.sum())
    out: Dict[str, object] = {
        "n_valid": n_valid,
        "k_max": 0,
        "swap_score_by_k": {},
        "null_95th_by_k": {},
        "T_obs": float("nan"),
        "p_fw": float("nan"),
        "decision_k": None,
        "decision_swap_score": float("nan"),
        "swap_class": "none",
        "has_swap": False,
        "decision_rule": (
            "T_obs=max_k swap_score(k); p_fw vs max-null; "
            "swap_class=strict iff T_obs>=score_floor AND p_fw<alpha_fw; "
            "swap_class=candidate iff T_obs>=score_floor AND alpha_fw<=p_fw<alpha_candidate"
        ),
        "score_floor": float(score_floor),
        "alpha_fw": float(alpha_fw),
        "alpha_candidate": float(alpha_candidate),
        "n_perm": int(n_perm),
        "seed": int(seed),
        "exit_reason": "ok",
    }
    if n_valid < 4:
        out["exit_reason"] = f"n_valid<4 (got {n_valid})"
        return out

    k_max = n_valid // 2
    out["k_max"] = k_max

    valid_idx = np.flatnonzero(joint_valid)
    r_a = rank_a[valid_idx]
    r_b = rank_b[valid_idx]

    def _swap_inner(ra: np.ndarray, rb: np.ndarray, k: int) -> float:
        order_a = np.argsort(ra, kind="stable")
        order_b = np.argsort(rb, kind="stable")
        bot_a = set(order_a[:k].tolist())
        top_a = set(order_a[-k:].tolist())
        bot_b = set(order_b[:k].tolist())
        top_b = set(order_b[-k:].tolist())

        def _jacc(s1: set, s2: set) -> float:
            u = s1 | s2
            return (len(s1 & s2) / len(u)) if u else 0.0

        return 0.5 * (_jacc(top_a, bot_b) + _jacc(bot_a, top_b))

    rng = np.random.default_rng(seed)
    n_k = k_max - 1
    null_table = np.empty((n_perm, n_k), dtype=float)
    for i in range(n_perm):
        rb_shuf = rng.permutation(r_b)
        for ki, k in enumerate(range(2, k_max + 1)):
            null_table[i, ki] = _swap_inner(r_a, rb_shuf, k)

    swap_by_k: Dict[int, float] = {}
    null_95: Dict[int, float] = {}
    for ki, k in enumerate(range(2, k_max + 1)):
        swap_by_k[k] = float(_swap_inner(r_a, r_b, k))
        null_95[k] = float(np.percentile(null_table[:, ki], 95))

    T_obs = float(max(swap_by_k.values()))
    decision_k = min(k for k, v in swap_by_k.items() if v == T_obs)
    T_perm = null_table.max(axis=1)
    p_fw = float((1 + np.sum(T_perm >= T_obs)) / (n_perm + 1))

    if T_obs >= score_floor and p_fw < alpha_fw:
        swap_class = "strict"
    elif T_obs >= score_floor and p_fw < alpha_candidate:
        swap_class = "candidate"
    else:
        swap_class = "none"

    out.update(
        {
            "swap_score_by_k": {str(k): v for k, v in swap_by_k.items()},
            "null_95th_by_k": {str(k): v for k, v in null_95.items()},
            "T_obs": T_obs,
            "p_fw": p_fw,
            "decision_k": int(decision_k),
            "decision_swap_score": float(T_obs),
            "swap_class": swap_class,
            "has_swap": bool(swap_class == "strict"),
        }
    )
    return out


def aggregate_pair_metrics(
    rank_a: np.ndarray,
    rank_b: np.ndarray,
    valid_mask_a: np.ndarray,
    valid_mask_b: np.ndarray,
    channel_names: Sequence[str],
    soz_channels: Optional[set] = None,
) -> Dict[str, object]:
    """Wrap compute_signed_rank_displacement with baseline-corrected SOZ split.

    Outputs (descriptive only, no PASS gate):
      - soz_channel_fraction       (chance baseline for contribution_fraction)
      - soz_contribution_fraction  (Σ|Δr|_SOZ / footrule)
      - nonsoz_contribution_fraction
      - soz_contribution_excess    (= contribution_fraction − channel_fraction)
      - soz_abs_mean, nonsoz_abs_mean    (per-channel |Δr| means; count-confound free)
      - soz_minus_nonsoz_abs_mean

    Does NOT export signed_displacement_mean_soz / nonsoz - those are anchor-
    dependent and per plan §3.0 cannot be aggregated across subjects.
    """
    base = compute_signed_rank_displacement(
        rank_a=rank_a,
        rank_b=rank_b,
        valid_mask_a=valid_mask_a,
        valid_mask_b=valid_mask_b,
        channel_names=channel_names,
    )
    if base["exit_reason"] != "ok":
        return base

    delta_full = np.asarray(base["signed_displacement_full"], dtype=float)
    joint_valid = np.asarray(base["joint_valid"], dtype=bool)
    soz_set = set(soz_channels or [])
    soz_mask = np.array([ch in soz_set for ch in channel_names], dtype=bool)
    soz_joint = soz_mask & joint_valid
    nonsoz_joint = (~soz_mask) & joint_valid

    n_soz_joint = int(soz_joint.sum())
    n_nonsoz_joint = int(nonsoz_joint.sum())
    n_valid = n_soz_joint + n_nonsoz_joint
    footrule = float(base["footrule"])
    abs_full = np.abs(delta_full)
    nan = float("nan")

    soz_channel_fraction = (n_soz_joint / n_valid) if n_valid > 0 else nan
    soz_contribution = (
        float(np.nansum(abs_full[soz_joint]) / footrule)
        if footrule > 0 and n_soz_joint > 0
        else nan
    )
    nonsoz_contribution = (
        float(np.nansum(abs_full[nonsoz_joint]) / footrule)
        if footrule > 0 and n_nonsoz_joint > 0
        else nan
    )
    soz_excess = (
        soz_contribution - soz_channel_fraction
        if not (np.isnan(soz_contribution) or np.isnan(soz_channel_fraction))
        else nan
    )
    soz_abs_mean = (
        float(np.nanmean(abs_full[soz_joint])) if n_soz_joint > 0 else nan
    )
    nonsoz_abs_mean = (
        float(np.nanmean(abs_full[nonsoz_joint])) if n_nonsoz_joint > 0 else nan
    )
    soz_minus_nonsoz_abs_mean = (
        soz_abs_mean - nonsoz_abs_mean
        if not (np.isnan(soz_abs_mean) or np.isnan(nonsoz_abs_mean))
        else nan
    )

    base.update(
        {
            "soz_mask": soz_mask.tolist(),
            "n_soz_joint": n_soz_joint,
            "n_nonsoz_joint": n_nonsoz_joint,
            "soz_channel_fraction": soz_channel_fraction,
            "soz_contribution_fraction": soz_contribution,
            "nonsoz_contribution_fraction": nonsoz_contribution,
            "soz_contribution_excess": soz_excess,
            "soz_abs_mean": soz_abs_mean,
            "nonsoz_abs_mean": nonsoz_abs_mean,
            "soz_minus_nonsoz_abs_mean": soz_minus_nonsoz_abs_mean,
        }
    )
    return base


# =============================================================================
# §9 — Swap × clinical SOZ set-relationship
# Plan: docs/archive/topic1/pr6_supplementary_swap_clinical_soz_plan_2026-05-08.md
# =============================================================================


def derive_swap_endpoint(
    channel_names: Sequence[str],
    rank_a_dense: np.ndarray,
    decision_k: int,
) -> list:
    """Pick top decision_k ∪ bottom decision_k channels by rank_a_dense ascending.

    "top" = lowest dense ranks = source side in T_a;
    "bottom" = highest dense ranks = sink side in T_a.

    Per §8 main figure swap-marker convention (column = decision_k - 1 left
    triangle, column = n_v - decision_k right triangle when columns sorted
    by rank_a_dense ascending).
    """
    rank_a_dense = np.asarray(rank_a_dense, dtype=float)
    if rank_a_dense.shape != (len(channel_names),):
        raise ValueError(
            f"rank_a_dense shape {rank_a_dense.shape} != n_channels "
            f"({len(channel_names)})"
        )
    if 2 * decision_k > len(channel_names):
        raise ValueError(
            f"2*decision_k ({2 * decision_k}) > n_channels "
            f"({len(channel_names)})"
        )
    order = np.argsort(rank_a_dense, kind="stable")
    top_idx = order[:decision_k]
    bottom_idx = order[-decision_k:]
    endpoint_idx = sorted(set(top_idx.tolist()) | set(bottom_idx.tolist()))
    return [channel_names[i] for i in endpoint_idx]


def compute_clinical_soz_set_relation(
    valid_chs: Sequence[str],
    endpoint_chs: Sequence[str],
    soz_chs: Sequence[str],
) -> Dict[str, object]:
    """Set-relationship readouts of swap_endpoint vs clinical SOZ within lagPat.

    Universe = valid_chs (lagPat valid set). SOZ is restricted to lagPat
    (n_S = |soz ∩ valid|). Endpoint is required ⊆ valid (caller's contract;
    derive_swap_endpoint guarantees this).

    Decision tree for typology (first match wins):
      degenerate    if n_S == 0 OR n_S == n_L OR n_E == n_L
      disjoint      if n_E_inter_S == 0
      E_subset_S    if n_E_inter_S == n_E AND n_E_inter_S < n_S
      S_subset_E    if n_E_inter_S == n_S AND n_E_inter_S < n_E
      partial       otherwise

    Returned schema is pre-registered in plan §2; do not silently add or drop
    fields. enrichment_over_lagPat = precision − lagpat_baseline; null when
    n_S == 0 or n_L == 0.
    """
    valid_set = set(valid_chs)
    endpoint_set = set(endpoint_chs)
    soz_in_lagpat = set(soz_chs) & valid_set

    if not endpoint_set <= valid_set:
        raise ValueError(
            f"endpoint_chs has {len(endpoint_set - valid_set)} channels "
            f"outside valid_chs; derive_swap_endpoint should have prevented this"
        )

    n_L = len(valid_set)
    n_E = len(endpoint_set)
    n_S = len(soz_in_lagpat)
    n_E_inter_S = len(endpoint_set & soz_in_lagpat)

    precision: Optional[float] = (n_E_inter_S / n_E) if n_E > 0 else None
    recall: Optional[float] = (n_E_inter_S / n_S) if n_S > 0 else None
    coverage: Optional[float] = (n_E / n_L) if n_L > 0 else None
    lagpat_baseline: Optional[float] = (n_S / n_L) if n_L > 0 else None
    if precision is None or lagpat_baseline is None or n_S == 0:
        enrichment: Optional[float] = None
    else:
        enrichment = precision - lagpat_baseline

    if n_S == 0 or n_S == n_L or n_E == n_L:
        typology = "degenerate"
    elif n_E_inter_S == 0:
        typology = "disjoint"
    elif n_E_inter_S == n_E and n_E_inter_S < n_S:
        typology = "E_subset_S"
    elif n_E_inter_S == n_S and n_E_inter_S < n_E:
        typology = "S_subset_E"
    else:
        typology = "partial"

    return {
        "soz_source": "clinical",
        "n_E": n_E,
        "n_S": n_S,
        "n_L": n_L,
        "n_E_inter_S": n_E_inter_S,
        "precision": precision,
        "recall_within_lagPat": recall,
        "coverage": coverage,
        "lagpat_baseline": lagpat_baseline,
        "enrichment_over_lagPat": enrichment,
        "typology": typology,
        "informative": typology != "degenerate",
        "exit_reason": None,
    }


def cohort_sign_test_enrichment(
    enrichments: Sequence[Optional[float]],
    n_boot: int = 2000,
    seed: int = 0,
) -> Dict[str, object]:
    """One-sided binomial sign test on enrichment_over_lagPat > 0 + bootstrap CI.

    None entries (degenerate subjects) are dropped before stat. n_informative
    = number of non-None entries. Sign test counts strictly positive (> 0)
    against (== 0 OR < 0); ties at 0 are conservatively treated as
    non-positive.

    Bootstrap: percentile method on the median, n_boot resamples with seed-
    deterministic numpy default_rng. CI = [2.5th, 97.5th] percentile.

    Returns a dict containing all primary statistics (no PASS gate).
    """
    informative = [v for v in enrichments if v is not None]
    n_inf = len(informative)
    if n_inf == 0:
        return {
            "n_informative": 0,
            "n_positive": 0,
            "sign_test_p": None,
            "median_enrichment": None,
            "bootstrap_ci_lo": None,
            "bootstrap_ci_hi": None,
            "n_boot": int(n_boot),
            "seed": int(seed),
        }
    arr = np.asarray(informative, dtype=float)
    n_pos = int(np.sum(arr > 0))
    sign_p = float(stats.binomtest(n_pos, n_inf, p=0.5, alternative="greater").pvalue)
    median = float(np.median(arr))
    rng = np.random.default_rng(seed)
    boot_medians = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(arr, size=n_inf, replace=True)
        boot_medians[i] = np.median(sample)
    ci_lo = float(np.percentile(boot_medians, 2.5))
    ci_hi = float(np.percentile(boot_medians, 97.5))
    return {
        "n_informative": n_inf,
        "n_positive": n_pos,
        "sign_test_p": sign_p,
        "median_enrichment": median,
        "bootstrap_ci_lo": ci_lo,
        "bootstrap_ci_hi": ci_hi,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


# ---------------------------------------------------------------------------
# PR-6-sup1 — first-rank entropy / symmetry-breaking diagnostic
# Plan v3 (2026-05-10):
#   docs/archive/topic1/pr6_template_anchoring/
#       pr6_supplementary_rank_entropy_plan_2026-05-10.md
#
# Tier: Topic 4 / SBA mechanism preflight, descriptive only.
# Sharp prediction: under noise-driven confluence-point dynamics, rank
# vector entropy should be high at endpoint positions {1, n_valid} and
# low in the middle ("seed jitter, backbone stable").
# ---------------------------------------------------------------------------


_MIN_N_VALID_RANK_ENTROPY = 4
_MIN_KEPT_EVENTS = 50
_HIGH_DROP_RATE_WARN = 0.5


def compute_rank_position_entropy(
    R_cluster: np.ndarray,
    n_valid: int,
) -> np.ndarray:
    """Per-position normalized Shannon entropy of channel identity.

    Parameters
    ----------
    R_cluster : (n_events, n_valid) integer array; each row is a permutation
        of {1, ..., n_valid}.
    n_valid : int (>= 4).

    Returns
    -------
    H_p_norm : (n_valid,) float array, each entry in [0, 1].
        H_p_norm[p-1] = Shannon entropy (base 2) of the channel-at-position-p
        distribution divided by log_2(n_valid).
    """
    if n_valid < _MIN_N_VALID_RANK_ENTROPY:
        raise ValueError(
            f"n_valid={n_valid} < {_MIN_N_VALID_RANK_ENTROPY}; need at least "
            f"2 endpoint + 2 middle positions."
        )
    R = np.asarray(R_cluster, dtype=int)
    if R.ndim != 2:
        raise ValueError(f"R_cluster must be 2D, got shape {R.shape}")
    n_events, n_cols = R.shape
    if n_events == 0:
        raise ValueError("R_cluster has n_events=0")
    if n_cols != n_valid:
        raise ValueError(
            f"R_cluster has n_cols={n_cols} but n_valid={n_valid}"
        )

    log2_nvalid = float(np.log2(n_valid))
    H_p_norm = np.empty(n_valid, dtype=float)
    for p_zero in range(n_valid):
        # rank position is 1-indexed; mask events whose channel-at-rank=p+1
        # is each candidate channel c
        rank_value = p_zero + 1
        # For each event, find which channel got rank=rank_value:
        #   channel index c = where(R[event, :] == rank_value)[0][0]
        # Vectorized: argmax over equality (each row has unique rank_value)
        eq = R == rank_value
        # eq has shape (n_events, n_valid); each row has exactly one True
        channel_at_p = np.argmax(eq, axis=1)
        counts = np.bincount(channel_at_p, minlength=n_valid)
        probs = counts / float(n_events)
        # Shannon entropy in bits (base 2); 0 * log(0) := 0
        positive = probs[probs > 0]
        H = float(-np.sum(positive * np.log2(positive))) if positive.size else 0.0
        H_p_norm[p_zero] = H / log2_nvalid

    return H_p_norm


def compute_endpoint_middle_entropy_delta(
    H_p_norm: np.ndarray,
) -> Tuple[float, float]:
    """Δ = mean(H at endpoints) - mean(H in middle); asymmetry = H_1 - H_n.

    Endpoint positions are {1, n_valid} (rank-1 and rank-n_valid).
    Middle positions are {2, ..., n_valid - 1}.
    """
    H = np.asarray(H_p_norm, dtype=float)
    n_valid = H.size
    if n_valid < _MIN_N_VALID_RANK_ENTROPY:
        raise ValueError(
            f"n_valid={n_valid} < {_MIN_N_VALID_RANK_ENTROPY}"
        )
    H_endpoint = (H[0] + H[-1]) / 2.0
    H_middle = float(np.mean(H[1:-1]))
    delta = H_endpoint - H_middle
    asymmetry = float(H[0] - H[-1])
    return float(delta), float(asymmetry)


def rank_entropy_null_N0(
    R_cluster: np.ndarray,
    n_valid: int,
    n_perm: int = 1000,
    base_seed: int = 0,
) -> np.ndarray:
    """N0 null: per-event independent rank shuffle.

    For each surrogate, replace every row of R_cluster with an independent
    uniform random permutation of {1, ..., n_valid}, recompute H_p_norm and
    Δ, and return the resulting Δ_null distribution.

    Determinism: ``np.random.RandomState(base_seed + i)`` per surrogate i.
    """
    if n_valid < _MIN_N_VALID_RANK_ENTROPY:
        raise ValueError(
            f"n_valid={n_valid} < {_MIN_N_VALID_RANK_ENTROPY}"
        )
    R = np.asarray(R_cluster, dtype=int)
    n_events = R.shape[0]
    if n_events == 0:
        raise ValueError("R_cluster has n_events=0")

    deltas = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        rng = np.random.RandomState(int(base_seed) + i)
        # Generate n_events independent permutations of 1..n_valid
        R_perm = np.empty_like(R)
        for e in range(n_events):
            R_perm[e] = rng.permutation(n_valid) + 1
        H = compute_rank_position_entropy(R_perm, n_valid)
        d, _ = compute_endpoint_middle_entropy_delta(H)
        deltas[i] = d
    return deltas


def rank_entropy_null_N1_pseudo_endpoint(
    H_p_norm: np.ndarray,
) -> Dict[str, Any]:
    """N1 cluster-level pseudo-endpoint position null (exact enumeration).

    Holds H_p_norm fixed; enumerates all C(n_valid, 2) unordered position
    pairs (p1, p2); for each pair computes
        Δ_pair = mean(H[p1], H[p2]) - mean(H[q] for q ∉ {p1, p2}).
    Reports endpoint pair {0, n_valid - 1} (1-indexed: {1, n_valid}) rank,
    percentile, p-value, and the floor min_attainable_p_N1 = 1 / C(n_valid, 2).

    The floor matters: at n_valid=6, min_attainable_p_N1 = 1/15 = 0.0667 —
    higher than the conventional 0.05, so a hard `p_N1 < 0.05` gate
    mechanically excludes n_valid=6 subjects. Use percentile / max-flag
    instead. Plan v3 §7.2.3.

    Returns dict per plan §7.2.1 schema.
    """
    H = np.asarray(H_p_norm, dtype=float)
    n_valid = int(H.size)
    if n_valid < _MIN_N_VALID_RANK_ENTROPY:
        raise ValueError(
            f"n_valid={n_valid} < {_MIN_N_VALID_RANK_ENTROPY}"
        )

    pairs: List[Tuple[int, int]] = list(combinations(range(n_valid), 2))
    delta_pair = np.empty(len(pairs), dtype=float)
    endpoint_pair = (0, n_valid - 1)
    endpoint_pair_idx = -1
    sum_H = float(np.sum(H))
    for k, (i, j) in enumerate(pairs):
        H_pair = (H[i] + H[j]) / 2.0
        # Middle = all positions except (i, j); n_valid - 2 of them.
        H_mid_sum = sum_H - H[i] - H[j]
        H_mid_mean = H_mid_sum / (n_valid - 2)
        delta_pair[k] = H_pair - H_mid_mean
        if (i, j) == endpoint_pair:
            endpoint_pair_idx = k

    if endpoint_pair_idx < 0:
        raise RuntimeError("endpoint pair (0, n_valid-1) not in enumeration")

    delta_obs = float(delta_pair[endpoint_pair_idx])
    n_pairs = len(pairs)
    # Rank under "ties count as equal-or-better"; rank=1 means strictly max.
    # We use ≥ in p_N1; rank is 1-indexed by # of pairs with delta_pair ≥ delta_obs.
    n_ge = int(np.sum(delta_pair >= delta_obs - 1e-12))
    p_N1 = n_ge / n_pairs
    # endpoint_pair_rank: smallest 1-indexed rank achievable given ties; the
    # endpoint pair sits among the ties at the top, so rank = 1 if it is
    # tied for max, else rank = 1 + #{Δ_pair > delta_obs (strict)}.
    n_strict_greater = int(np.sum(delta_pair > delta_obs + 1e-12))
    endpoint_pair_rank = 1 + n_strict_greater
    endpoint_pair_percentile = (n_pairs - endpoint_pair_rank + 1) / n_pairs
    # is_endpoint_pair_max: endpoint pair is among the maxima (ties allowed)
    max_val = float(np.max(delta_pair))
    is_endpoint_pair_max = bool(delta_obs >= max_val - 1e-12)

    return {
        "delta_obs": delta_obs,
        "delta_pair_dist": delta_pair,
        "endpoint_pair_rank": int(endpoint_pair_rank),
        "endpoint_pair_percentile": float(endpoint_pair_percentile),
        "is_endpoint_pair_max": is_endpoint_pair_max,
        "p_N1": float(p_N1),
        "min_attainable_p_N1": 1.0 / n_pairs,
        "n_valid": n_valid,
    }


def rank_entropy_null_N1_subject_level(
    H_p_norm_per_cluster: List[np.ndarray],
) -> Dict[str, Any]:
    """N1 subject-level joint enumeration (Option B locked 2026-05-10 v3).

    Enumerates the full Cartesian product of cluster-level position pairs:
        n_combos = C(n_valid_0, 2) × C(n_valid_1, 2)
    For each combo (pair_0, pair_1) computes
        Δ_combo = mean(Δ_pair_0, Δ_pair_1)
    The observed value is the (endpoint_0, endpoint_1) combo:
        Δ_obs_subject = mean(Δ_obs_0, Δ_obs_1)

    Locked at the stable_k=2 cohort: requires exactly 2 entries in
    ``H_p_norm_per_cluster``.
    """
    if not isinstance(H_p_norm_per_cluster, (list, tuple)):
        raise ValueError("H_p_norm_per_cluster must be a list of np.ndarray")
    if len(H_p_norm_per_cluster) != 2:
        raise ValueError(
            f"plan v3 locks subject-level enumeration to exactly 2 clusters; "
            f"got {len(H_p_norm_per_cluster)}"
        )

    cluster_results = []
    pair_deltas: List[np.ndarray] = []
    for H in H_p_norm_per_cluster:
        out = rank_entropy_null_N1_pseudo_endpoint(H)
        cluster_results.append(out)
        pair_deltas.append(out["delta_pair_dist"])

    delta_pair_0 = pair_deltas[0]
    delta_pair_1 = pair_deltas[1]
    n_combos = int(delta_pair_0.size * delta_pair_1.size)
    # Δ_combo grid via outer mean
    grid = (delta_pair_0[:, None] + delta_pair_1[None, :]) / 2.0
    delta_obs_subject = (cluster_results[0]["delta_obs"]
                         + cluster_results[1]["delta_obs"]) / 2.0

    n_ge = int(np.sum(grid >= delta_obs_subject - 1e-12))
    p_N1_subject = n_ge / n_combos
    n_strict_greater = int(np.sum(grid > delta_obs_subject + 1e-12))
    subject_combo_rank = 1 + n_strict_greater
    subject_combo_percentile = (n_combos - subject_combo_rank + 1) / n_combos
    max_val = float(np.max(grid))
    is_subject_combo_max = bool(delta_obs_subject >= max_val - 1e-12)

    return {
        "delta_obs_subject": float(delta_obs_subject),
        "n_combos": int(n_combos),
        "subject_combo_rank": int(subject_combo_rank),
        "subject_combo_percentile": float(subject_combo_percentile),
        "is_subject_combo_max": is_subject_combo_max,
        "p_N1_subject": float(p_N1_subject),
        "min_attainable_p_N1_subject": 1.0 / n_combos,
        "cluster_level": cluster_results,
    }


def compute_rank_position_entropy_with_absent(
    ranks: np.ndarray,
    bools: np.ndarray,
    cluster_event_idx: np.ndarray,
    valid_mask: np.ndarray,
) -> Dict[str, Any]:
    """Per-position entropy that **includes non-participating events** by
    treating their empty slots as a virtual "absent" alphabet member.

    Plan v3 §6.0 Option B (all-valid-participating filter) drops ~98 % of
    cohort events on average — the high drop rate inflates apparent
    slow-end stereotypy. This variant keeps every cluster event:

    - For each event, dense-rank the participating valid channels
      (ranks 1..n_part_e). The remaining (n_valid − n_part_e) slots at
      the deeper rank positions are filled with the "absent" sentinel.
    - At rank position p ∈ {1..n_valid}, the alphabet is the
      n_valid valid channels plus "absent" — alphabet size n_valid + 1.
    - H_p_norm normalised by log₂(n_valid + 1).

    Returns dict with fields:
      H_p_norm        : (n_valid,) — normalized Shannon entropy per rank
      P_absent        : (n_valid,) — fraction of events with "absent" at
                        each rank (= recording-level emptiness)
      P_real_renorm   : (n_valid, n_valid) — count(channel, p) / n_events
      n_events_total  : int — every cluster event counted
      n_part_dist     : np.ndarray — distribution of n_part across events
    """
    valid_idx = np.where(valid_mask)[0]
    n_valid = int(valid_idx.size)
    if n_valid < _MIN_N_VALID_RANK_ENTROPY:
        raise ValueError(
            f"n_valid={n_valid} < {_MIN_N_VALID_RANK_ENTROPY}"
        )
    cluster_event_idx = np.asarray(cluster_event_idx, dtype=int)
    n_events = int(cluster_event_idx.size)
    if n_events == 0:
        raise ValueError("cluster_event_idx empty")

    # counts shape (n_valid + 1, n_valid) — last row is "absent"
    counts = np.zeros((n_valid + 1, n_valid), dtype=int)
    n_part_arr = np.zeros(n_events, dtype=int)

    bools_valid = bools[valid_idx, :][:, cluster_event_idx]   # (n_valid, n_events)
    ranks_valid = ranks[valid_idx, :][:, cluster_event_idx].astype(float)

    for ei in range(n_events):
        part_local_mask = bools_valid[:, ei]  # (n_valid,)
        n_part = int(part_local_mask.sum())
        n_part_arr[ei] = n_part
        if n_part == 0:
            counts[n_valid, :] += 1  # all positions absent
            continue
        # Dense-rank the participating channels by their lagPat rank values
        part_local_idx = np.where(part_local_mask)[0]
        part_rank_vals = ranks_valid[part_local_idx, ei]
        order = np.argsort(part_rank_vals, kind="stable")
        # order[0] is the local index of the channel with smallest rank
        for k in range(n_part):
            c_local = int(part_local_idx[order[k]])
            counts[c_local, k] += 1
        # Positions n_part..n_valid-1 are "absent" for this event
        if n_part < n_valid:
            counts[n_valid, n_part:] += 1

    log2_alphabet = float(np.log2(n_valid + 1))
    H_p_norm = np.empty(n_valid, dtype=float)
    P_absent = np.empty(n_valid, dtype=float)
    for p in range(n_valid):
        col = counts[:, p].astype(float)
        total = col.sum()
        probs = col / total if total > 0 else col
        positive = probs[probs > 0]
        H = float(-np.sum(positive * np.log2(positive))) if positive.size else 0.0
        H_p_norm[p] = H / log2_alphabet
        P_absent[p] = float(probs[n_valid])

    # Real-channel renorm (drops the "absent" row, renormalises)
    P_real_renorm = np.zeros((n_valid, n_valid), dtype=float)
    for p in range(n_valid):
        real_col = counts[:n_valid, p].astype(float)
        total_real = real_col.sum()
        if total_real > 0:
            P_real_renorm[:, p] = real_col / total_real

    return {
        "H_p_norm": H_p_norm,
        "P_absent": P_absent,
        "P_real_renorm": P_real_renorm,
        "n_events_total": n_events,
        "n_part_dist": n_part_arr,
    }


def run_subject_rank_entropy_with_absent(
    subject_data: Dict[str, Any],
) -> Dict[str, Any]:
    """Variant of run_subject_rank_entropy: no Option B filter, uses
    "absent" sentinel + alphabet size n_valid + 1.
    Returns per-cluster H_p_norm + P_absent + n_part_distribution.
    """
    n_clusters = int(subject_data["n_clusters"])
    if n_clusters != 2:
        raise ValueError("plan v3 locks sup1 to stable_k=2")

    ranks = np.asarray(subject_data["ranks"])
    bools = np.asarray(subject_data["bools"], dtype=bool)
    labels = np.asarray(subject_data["labels"], dtype=int)
    valid_mask = np.asarray(subject_data["valid_mask"], dtype=bool)

    cluster_payload: Dict[str, Any] = {}
    for k in range(n_clusters):
        cluster_idx = np.where(labels == k)[0].astype(int)
        if cluster_idx.size == 0:
            cluster_payload[str(k)] = {"n_events_total": 0, "H_p_norm": None}
            continue
        out = compute_rank_position_entropy_with_absent(
            ranks, bools, cluster_idx, valid_mask
        )
        cluster_payload[str(k)] = {
            "n_events_total": int(out["n_events_total"]),
            "H_p_norm": [float(x) for x in out["H_p_norm"]],
            "P_absent": [float(x) for x in out["P_absent"]],
            "n_part_median": float(np.median(out["n_part_dist"])),
            "n_part_max": int(np.max(out["n_part_dist"])),
        }

    return {
        "clusters_with_absent": cluster_payload,
        "n_valid": int(valid_mask.sum()),
    }


def _filter_all_valid_participating_events(
    ranks: np.ndarray,
    bools: np.ndarray,
    cluster_event_idx: np.ndarray,
    valid_mask: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    """Plan v3 §6.0 Option B: keep only events whose all valid_mask channels
    participated. Returns (R_kept, n_events_total, n_events_kept).

    R_kept has shape (n_events_kept, n_valid_channels) where
    n_valid_channels = sum(valid_mask). Each row is the integer rank vector
    (per-event min-subtracted argsort-of-argsort if needed; here we assume
    ranks are already integer 1..n_ch with sentinel 0 for non-participating).
    """
    valid_idx = np.where(valid_mask)[0]
    n_valid = int(valid_idx.size)
    n_total = int(cluster_event_idx.size)
    if n_total == 0:
        return np.empty((0, n_valid), dtype=int), 0, 0

    # bools[valid_idx, e] all True ⇒ keep event e
    bools_valid = bools[valid_idx[:, None], cluster_event_idx[None, :]]
    keep_mask = np.all(bools_valid, axis=0)
    kept_event_idx = cluster_event_idx[keep_mask]
    n_kept = int(kept_event_idx.size)

    if n_kept == 0:
        return np.empty((0, n_valid), dtype=int), n_total, 0

    # Slice ranks to (n_valid, n_kept), then re-rank densely per event so
    # every row is a permutation of {1, ..., n_valid} regardless of the
    # raw lagPat-rank scale (which may include skipped values from the
    # original n_ch > n_valid context).
    ranks_kept = ranks[valid_idx[:, None], kept_event_idx[None, :]].astype(float)
    # Per-event dense rank: argsort-of-argsort on (n_valid,) vector + 1
    R = np.empty((n_kept, n_valid), dtype=int)
    for k in range(n_kept):
        order = np.argsort(np.argsort(ranks_kept[:, k], kind="stable"), kind="stable")
        R[k] = order + 1

    return R, n_total, n_kept


def run_subject_rank_entropy(
    subject_data: Dict[str, Any],
    *,
    n_perm_N0: int = 1000,
    base_seed_N0: int = 0,
    min_kept_events: int = _MIN_KEPT_EVENTS,
    high_drop_rate_warn: float = _HIGH_DROP_RATE_WARN,
) -> Dict[str, Any]:
    """Run sup1 entropy pipeline on a single subject (stable_k=2).

    Required keys in subject_data:
      ranks         : (n_ch, n_events) integer or float
      bools         : (n_ch, n_events) bool
      labels        : (n_events,) int cluster labels in {0, ..., n_clusters-1}
      valid_mask    : (n_ch,) bool
      n_clusters    : int (locked at 2)
      channel_names : list of length n_ch

    Per cluster k: filter to all-valid-participating events (§6.0 Option B);
    compute H_p_norm + Δ + asymmetry + N0 + N1 cluster-level. If both clusters
    pass eligibility, also compute subject-level Option B null.

    Returns nested dict; see plan §7 + §6.0 for schema.
    """
    n_clusters = int(subject_data["n_clusters"])
    if n_clusters != 2:
        raise ValueError(
            f"plan v3 locks sup1 to stable_k=2; got n_clusters={n_clusters}"
        )

    ranks = np.asarray(subject_data["ranks"])
    bools = np.asarray(subject_data["bools"], dtype=bool)
    labels = np.asarray(subject_data["labels"], dtype=int)
    valid_mask = np.asarray(subject_data["valid_mask"], dtype=bool)
    n_valid = int(valid_mask.sum())

    cluster_payload: Dict[str, Any] = {}
    H_per_cluster: List[Optional[np.ndarray]] = [None, None]
    eligibility = []

    for k in range(n_clusters):
        cluster_idx = np.where(labels == k)[0].astype(int)
        R, n_total, n_kept = _filter_all_valid_participating_events(
            ranks, bools, cluster_idx, valid_mask
        )
        drop_rate = (
            float(1.0 - n_kept / n_total) if n_total > 0 else float("nan")
        )

        if n_kept < min_kept_events:
            flag = "excluded_low_kept_events"
        elif drop_rate > high_drop_rate_warn:
            flag = "high_drop_rate_warning"
        else:
            flag = "ok"

        cluster_record: Dict[str, Any] = {
            "n_events_total_k": int(n_total),
            "n_events_kept_k": int(n_kept),
            "drop_rate_k": drop_rate,
            "eligibility_flag": flag,
        }

        if flag == "excluded_low_kept_events":
            cluster_record.update(
                {
                    "delta": float("nan"),
                    "asymmetry": float("nan"),
                    "H_p_norm": None,
                    "p_N0": float("nan"),
                    "N1_cluster": None,
                }
            )
            eligibility.append(False)
        else:
            H_p_norm = compute_rank_position_entropy(R, n_valid)
            delta, asymmetry = compute_endpoint_middle_entropy_delta(H_p_norm)
            null_N0 = rank_entropy_null_N0(
                R, n_valid, n_perm=n_perm_N0, base_seed=base_seed_N0
            )
            n_ge = int(np.sum(null_N0 >= delta - 1e-12))
            p_N0 = (1 + n_ge) / (n_perm_N0 + 1)
            N1 = rank_entropy_null_N1_pseudo_endpoint(H_p_norm)
            cluster_record.update(
                {
                    "delta": float(delta),
                    "asymmetry": float(asymmetry),
                    "H_p_norm": [float(x) for x in H_p_norm],
                    "p_N0": float(p_N0),
                    "N1_cluster": {
                        # serializable subset (drop the array)
                        "delta_obs": N1["delta_obs"],
                        "endpoint_pair_rank": N1["endpoint_pair_rank"],
                        "endpoint_pair_percentile": N1[
                            "endpoint_pair_percentile"
                        ],
                        "is_endpoint_pair_max": N1["is_endpoint_pair_max"],
                        "p_N1": N1["p_N1"],
                        "min_attainable_p_N1": N1["min_attainable_p_N1"],
                        "n_valid": N1["n_valid"],
                    },
                }
            )
            H_per_cluster[k] = H_p_norm
            eligibility.append(True)

        cluster_payload[str(k)] = cluster_record

    subject_payload: Dict[str, Any]
    if all(eligibility) and all(H is not None for H in H_per_cluster):
        N1_subject = rank_entropy_null_N1_subject_level(
            [H for H in H_per_cluster if H is not None]
        )
        subject_payload = {
            "delta_obs_subject": N1_subject["delta_obs_subject"],
            "n_combos": N1_subject["n_combos"],
            "subject_combo_rank": N1_subject["subject_combo_rank"],
            "subject_combo_percentile": N1_subject[
                "subject_combo_percentile"
            ],
            "is_subject_combo_max": N1_subject["is_subject_combo_max"],
            "p_N1_subject": N1_subject["p_N1_subject"],
            "min_attainable_p_N1_subject": N1_subject[
                "min_attainable_p_N1_subject"
            ],
            "subject_eligibility_flag": "ok",
        }
    else:
        subject_payload = {
            "subject_eligibility_flag": "excluded_one_or_both_clusters",
        }

    return {
        "clusters": cluster_payload,
        "subject": subject_payload,
        "params": {
            "n_clusters": n_clusters,
            "n_valid": n_valid,
            "n_perm_N0": int(n_perm_N0),
            "base_seed_N0": int(base_seed_N0),
            "min_kept_events": int(min_kept_events),
            "high_drop_rate_warn": float(high_drop_rate_warn),
        },
    }
