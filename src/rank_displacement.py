"""Per-channel signed rank displacement metrics for cluster template comparison.

Supplementary to PR-6 endpoint anchoring (forward/reverse template geometry).
Continuous version of PR-6 discrete swap_node count.

Plan: docs/archive/topic1/pr6_supplementary_rank_displacement_plan_2026-05-06.md

Sign anchor (per plan §3.0): T_a = cluster with smaller cluster_id (engineering
only). Δr = rank_T_b − rank_T_a is per-subject only; never aggregate signed
values across subjects.
"""
from __future__ import annotations

from math import floor
from typing import Dict, Optional, Sequence

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
