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
