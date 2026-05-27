"""SEF-ITP framework Phase 3 — per-seizure peri-ictal recruitment / expansion.

Plan: docs/superpowers/plans/2026-05-24-topic4-phase3-h5-per-seizure-recruitment-plan.md
Framework: docs/topic4_sef_itp_framework.md v1.0.7 §3.5 H5 v1.0.7 spec.

Phase 3 asks: in seizure-adjacent windows (pre-ictal / post-ictal), does the
swap-k endpoint set grow in number (Δdecision_k > 0) and/or in spatial spread
(source-side and/or sink-side spatial radius ↑) relative to time-of-day matched
baseline windows? Per-seizure is the primary reporting unit; cohort inference
uses wild cluster bootstrap (primary at n<10) + cluster-robust SE (companion)
+ subject-level Wilcoxon (sanity). All three must agree on direction for
SUPPORTED.

**Design pivots from plan v0 §3** (advisor 2026-05-24, ratified during B0):
  - Cross-block windows allowed; track `effective_seconds` per window; coverage
    floor = 0.75 × 55min (= 41.25min).
  - Sliding baseline candidates at 15-min stride (not block-aligned). Plan §3.2
    "aligned baseline" never produces candidates for Epilepsiae blocks (<60min).
  - Guard band 12h primary, 4h sensitivity. B0 audit showed 12h works for all
    6 qualifying primary subjects; 4h is sensitivity for case 1146 where 12h
    drops 1 seizure.
  - Wild cluster bootstrap = PRIMARY inference at n<10 (not sensitivity).
    statsmodels OLS(cov_type='cluster') reported as companion.

Reused from Phase 2 (CLAUDE.md §6.1 question-match):
  - `_per_cluster_template_rank` — per-cluster template rank (legacy hist mean rank →
    argsort(argsort)) + cluster valid_mask. Same primitive PR-6 anchoring + Phase 2 H4 use.
  - `compute_endpoint_spatial_radius` — centroid RMS + mean pairwise + min enclosing ball.
    Phase 3 uses centroid_rms + mean_pairwise as primary; MEB kept in JSON for diagnostic
    only (plan §1 v4 catch + framework §3.4 v1.0.7 MEB k>3 caveat).
  - `compute_swap_score_sweep` — per-window swap sweep → decision_k + swap_class_window.

Phase 3-specific helpers (new):
  - `_load_seizure_times` — Yuquan + Epilepsiae unified CSV loader.
  - `_enumerate_peri_ictal_windows` — per-seizure pre + post windows with `effective_seconds`.
  - `_pick_matched_baseline_windows` — sliding 55-min @ 15-min stride, hour-of-day matched,
    guard from any seizure.
  - `compute_window_metrics` — per-window pipeline returning all 4 metric families.
  - `compute_delta_metrics` — per-seizure Δ-from-baseline aggregation (scalar median for
    decision_k/rate/radius; per-baseline-then-median for set Jaccard / new_node_fraction).
  - `wild_cluster_bootstrap_p` — Cameron-Gelbach-Miller wild cluster bootstrap (Rademacher
    weights), one-sided primary.
  - `cluster_robust_se_p` — statsmodels OLS cov_type='cluster' wrapper, one-sided.
  - `bh_fdr` — Benjamini-Hochberg q<α correction.
  - `compute_phase3_verdict` — plan §6 verdict logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Reused Phase 2 + rank_displacement primitives.
from src.sef_itp_phase2 import (
    _per_cluster_template_rank,
    compute_endpoint_spatial_radius,
    compute_source_sink_centroid_distance,
)
from src.rank_displacement import compute_swap_score_sweep

__version__ = "v1.0.0"

# Plan §3.1 lock + advisor 2026-05-24 design pivot.
PRE_OFFSET_MIN, PRE_BUFFER_MIN = 60, 5
POST_BUFFER_MIN, POST_OFFSET_MIN = 5, 60
PERI_NOMINAL_MIN = 55
COVERAGE_FLOOR = 0.75
HOUR_TOLERANCE_H = 2
GUARD_HOURS_PRIMARY = 12.0
GUARD_HOURS_SENSITIVITY = 4.0
BASELINE_WINDOW_MIN = 55
BASELINE_STRIDE_MIN = 15
N_BASELINE_MIN = 5
MIN_EVENTS_PER_WINDOW = 30  # locked at B0 audit time; median primary peri windows have hundreds
MIN_EVENTS_PER_CLUSTER_WINDOW = 10  # below this, swap_sweep is too noisy
SWAP_SWEEP_N_PERM_PRIMARY = 300  # n_perm budget for swap_sweep (perf vs precision; baseline windows aggregated to median, primary peri reports the single value)
SWAP_SWEEP_N_PERM_BASELINE = 200


# =============================================================================
# Seizure time loaders (Yuquan + Epilepsiae unified)
# =============================================================================


def _load_seizure_times(
    dataset: str,
    subject_id: str,
    yuquan_csv: Path = Path("results/dataset_inventory/yuquan_seizure_inventory.csv"),
    epi_csv: Path = Path("results/epilepsiae_seizure_inventory.csv"),
) -> List[Dict[str, Any]]:
    """Return [{'id': str, 'onset': float, 'offset': float, 'classification': str|None}]
    sorted by onset.

    Yuquan inventory has NaN local_hour fields (we compute hour-of-day from epoch +
    Asia/Shanghai). Epilepsiae inventory provides `classification` (CP / UC / s / FBTC etc).
    Both filtered to `has_complete_eeg_interval = True` and finite onset/offset.
    """
    import pandas as pd  # late import to keep module-load light for unit tests

    if dataset == "yuquan":
        df = pd.read_csv(yuquan_csv)
        clf_col = None
    elif dataset == "epilepsiae":
        df = pd.read_csv(epi_csv)
        df["subject"] = df["subject"].astype(str)
        clf_col = "classification"
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    sub = df[df["subject"].astype(str) == str(subject_id)].copy()
    sub = sub.dropna(subset=["eeg_onset_epoch", "eeg_offset_epoch"])
    sub = sub[sub["has_complete_eeg_interval"] == True]  # noqa: E712
    sub = sub.sort_values("eeg_onset_epoch").reset_index(drop=True)
    out: List[Dict[str, Any]] = []
    for _, row in sub.iterrows():
        rec: Dict[str, Any] = {
            "id": str(row["seizure_id"]),
            "onset": float(row["eeg_onset_epoch"]),
            "offset": float(row["eeg_offset_epoch"]),
        }
        if clf_col:
            rec["classification"] = (
                str(row[clf_col]) if not (isinstance(row[clf_col], float) and np.isnan(row[clf_col])) else None
            )
        else:
            rec["classification"] = None
        out.append(rec)
    return out


def _hour_of_day(t_epoch: float, tz_name: str) -> float:
    """Local-time hour-of-day (0-24 float) under a given timezone."""
    if tz_name == "Asia/Shanghai":
        dt = datetime.fromtimestamp(t_epoch, tz=timezone(timedelta(hours=8)))
    elif tz_name == "Europe/Berlin":
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.fromtimestamp(t_epoch, tz=ZoneInfo("Europe/Berlin"))
        except Exception:
            dt = datetime.fromtimestamp(t_epoch, tz=timezone(timedelta(hours=1)))
    else:
        dt = datetime.fromtimestamp(t_epoch, tz=timezone.utc)
    return dt.hour + dt.minute / 60.0 + dt.second / 3600.0


def _tz_for_dataset(dataset: str) -> str:
    return "Asia/Shanghai" if dataset == "yuquan" else "Europe/Berlin"


def _load_real_recording_block_ranges(
    dataset: str,
    subject_id: str,
    yuquan_csv: Path = Path("results/dataset_inventory/yuquan_block_inventory.csv"),
    epi_csv: Path = Path("results/epilepsiae_block_inventory.csv"),
) -> List[Tuple[float, float]]:
    """Load real recording-block time ranges from inventory CSVs (NOT event-derived).

    User catch 2026-05-24: `load_subject_propagation_events`'s `block_time_ranges` is
    `event first/last time per block file`, NOT real recording extent. For Epilepsiae
    blocks where many events failed to be detected (e.g., short blocks), event-derived
    boundaries can be 14s long even though the real block is 60min. Using event-derived
    blocks for coverage / baseline eligibility / rate denominator conflates 'where HFO
    detected' with 'where recording exists' — that's a contract violation per AGENTS.md
    Epilepsiae trust order (SQL > head > legacy script > variable name).

    Yuquan inventory: `results/dataset_inventory/yuquan_block_inventory.csv` (EDF head)
    Epilepsiae inventory: `results/epilepsiae_block_inventory.csv` (SQL `blocks` table)

    Both CSVs have `subject`, `block_start_epoch`, `block_end_epoch` columns.
    Returns time ranges sorted by start, finite-only.
    """
    import pandas as pd

    if dataset == "yuquan":
        df = pd.read_csv(yuquan_csv)
    elif dataset == "epilepsiae":
        df = pd.read_csv(epi_csv)
        df["subject"] = df["subject"].astype(str)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    sub = df[df["subject"].astype(str) == str(subject_id)].copy()
    sub = sub.dropna(subset=["block_start_epoch", "block_end_epoch"])
    sub = sub.sort_values("block_start_epoch")
    out: List[Tuple[float, float]] = []
    for _, row in sub.iterrows():
        bs, be = float(row["block_start_epoch"]), float(row["block_end_epoch"])
        if np.isfinite(bs) and np.isfinite(be) and be > bs:
            out.append((bs, be))
    return out


def _window_effective_seconds(
    t_start: float, t_end: float, blocks: List[Tuple[float, float]]
) -> float:
    sec = 0.0
    for bs, be in blocks:
        s = max(t_start, bs)
        e = min(t_end, be)
        if e > s:
            sec += e - s
    return sec


def _hour_circular_match(h1: float, h2: float, tol_h: float = HOUR_TOLERANCE_H) -> bool:
    d = abs(h1 - h2) % 24.0
    d = min(d, 24.0 - d)
    return d <= tol_h


def _far_from_all_seizures(
    t_start: float, t_end: float,
    seizures: List[Dict[str, Any]],
    guard_hours: float,
) -> bool:
    g = guard_hours * 3600.0
    for sz in seizures:
        if not (t_end + g <= sz["onset"] or sz["offset"] + g <= t_start):
            return False
    return True


# =============================================================================
# Window enumeration
# =============================================================================


def _enumerate_peri_ictal_windows(
    seizures: List[Dict[str, Any]],
    blocks: List[Tuple[float, float]],
    pre_offset_min: float = PRE_OFFSET_MIN,
    pre_buffer_min: float = PRE_BUFFER_MIN,
    post_buffer_min: float = POST_BUFFER_MIN,
    post_offset_min: float = POST_OFFSET_MIN,
    coverage_floor: float = COVERAGE_FLOOR,
) -> List[Dict[str, Any]]:
    """Per-seizure pre + post window descriptors.

    Returns one dict per seizure with keys:
      seizure_id, onset, offset, classification,
      pre: {t_start, t_end, effective_seconds, coverage, qualifies},
      post: {t_start, t_end, effective_seconds, coverage, qualifies}

    Coverage = effective_seconds / nominal_seconds (=55min). A side qualifies iff
    coverage ≥ coverage_floor (=0.75).
    """
    nominal_sec = PERI_NOMINAL_MIN * 60.0
    floor_sec = coverage_floor * nominal_sec
    out: List[Dict[str, Any]] = []
    for sz in seizures:
        pre_t_start = sz["onset"] - pre_offset_min * 60.0
        pre_t_end = sz["onset"] - pre_buffer_min * 60.0
        post_t_start = sz["offset"] + post_buffer_min * 60.0
        post_t_end = sz["offset"] + post_offset_min * 60.0
        rec: Dict[str, Any] = {
            "seizure_id": sz["id"],
            "onset": sz["onset"],
            "offset": sz["offset"],
            "classification": sz.get("classification"),
        }
        for side, ts, te in [("pre", pre_t_start, pre_t_end), ("post", post_t_start, post_t_end)]:
            eff = _window_effective_seconds(ts, te, blocks)
            cov = eff / nominal_sec
            rec[side] = {
                "t_start": ts, "t_end": te,
                "effective_seconds": eff, "coverage": cov,
                "qualifies": eff >= floor_sec,
            }
        out.append(rec)
    return out


def _enumerate_baseline_candidates_sliding(
    blocks: List[Tuple[float, float]],
    window_minutes: float = BASELINE_WINDOW_MIN,
    stride_minutes: float = BASELINE_STRIDE_MIN,
    coverage_floor: float = COVERAGE_FLOOR,
) -> List[Tuple[float, float, float]]:
    """Sliding window candidates over total recording extent; filter by coverage.

    Returns [(t_start, t_end, effective_seconds)] for windows with coverage ≥ floor.
    """
    if not blocks:
        return []
    nominal_sec = window_minutes * 60.0
    stride_sec = stride_minutes * 60.0
    floor_sec = coverage_floor * nominal_sec
    t0 = min(b[0] for b in blocks)
    t1 = max(b[1] for b in blocks)
    out: List[Tuple[float, float, float]] = []
    n_iter = int(np.floor((t1 - t0 - nominal_sec) / stride_sec)) + 1
    if n_iter <= 0:
        return []
    for i in range(n_iter):
        ts = t0 + i * stride_sec
        te = ts + nominal_sec
        if te > t1:
            break
        eff = _window_effective_seconds(ts, te, blocks)
        if eff >= floor_sec:
            out.append((ts, te, eff))
    return out


def _pick_matched_baseline_windows(
    peri_t_start: float,
    peri_t_end: float,
    seizures: List[Dict[str, Any]],
    blocks: List[Tuple[float, float]],
    tz_name: str,
    guard_hours: float,
    hour_tolerance_h: float = HOUR_TOLERANCE_H,
    window_minutes: float = BASELINE_WINDOW_MIN,
    stride_minutes: float = BASELINE_STRIDE_MIN,
    coverage_floor: float = COVERAGE_FLOOR,
) -> List[Dict[str, Any]]:
    """Pick baseline windows matching peri window's hour-of-day, ≥guard_hours from any seizure.

    Returns [{t_start, t_end, effective_seconds, hour_of_day}] sorted by t_start.
    """
    candidates = _enumerate_baseline_candidates_sliding(
        blocks, window_minutes=window_minutes, stride_minutes=stride_minutes,
        coverage_floor=coverage_floor,
    )
    peri_hod = _hour_of_day(peri_t_start, tz_name)
    matched: List[Dict[str, Any]] = []
    for ts, te, eff in candidates:
        if not _far_from_all_seizures(ts, te, seizures, guard_hours):
            continue
        bhod = _hour_of_day(ts, tz_name)
        if not _hour_circular_match(bhod, peri_hod, hour_tolerance_h):
            continue
        matched.append({
            "t_start": ts, "t_end": te,
            "effective_seconds": eff, "hour_of_day": bhod,
        })
    return matched


# =============================================================================
# Per-window metric pipeline (plan §4.1, v4 catch swap-k node = rank-displacement)
# =============================================================================


def compute_window_metrics(
    t_start: float, t_end: float,
    event_abs_times: np.ndarray,
    labels: np.ndarray,
    ranks: np.ndarray,
    bools: np.ndarray,
    coords: Optional[np.ndarray],
    channel_names: List[str],
    cluster_a: int = 0,
    cluster_b: int = 1,
    effective_seconds: Optional[float] = None,
    n_perm: int = SWAP_SWEEP_N_PERM_PRIMARY,
    seed: int = 0,
    min_events_per_cluster: int = MIN_EVENTS_PER_CLUSTER_WINDOW,
    min_events_per_window: int = MIN_EVENTS_PER_WINDOW,
) -> Dict[str, Any]:
    """Compute the 4-step canonical per-window pipeline (plan §4.1).

    Returns:
      decision_k, swap_class_window (diagnostic only — primary stratification uses subject-level),
      swap_k_endpoint_indices (variable-k from derive_swap_endpoint logic, here computed inline
        from rank_a_dense ordering),
      source_indices, sink_indices,
      source_radius {centroid_rms, mean_pairwise, min_enclosing_radius, n_points},
      sink_radius {same fields},
      source_sink_axis_distance,
      rate (events / effective_seconds_hours),
      n_events_total, n_events_a, n_events_b,
      exit_reason: "ok" | "insufficient_events_per_cluster" | "n_valid<4"

    Notes:
      - swap-k endpoint = top-k ∪ bottom-k channels by rank_a (cluster_a's template rank).
        Source-side = lowest decision_k ranks = "earliest fire in T_a". Sink-side = highest
        decision_k ranks = "latest fire in T_a". This is the rank-displacement variable-k
        endpoint (Topic 4 H2 spatial layer source), NOT PR-6 fixed top-3 anchoring.
      - effective_seconds is the rate denominator (advisor 2026-05-24 design pivot for
        cross-block windows).
      - swap_class_window is per-window noisy (small sample); for stratification use
        subject-level full-data swap_class from masked rank_displacement JSON. Report
        per-window swap_class as diagnostic only.
    """
    mask = (event_abs_times >= t_start) & (event_abs_times < t_end)
    evt_idx = np.where(mask)[0]
    n_total = int(evt_idx.size)
    out: Dict[str, Any] = {
        "t_start": float(t_start), "t_end": float(t_end),
        "effective_seconds": float(effective_seconds) if effective_seconds is not None else float("nan"),
        "n_events_total": n_total,
        "n_events_a": 0, "n_events_b": 0,
        "decision_k": None,
        "swap_class_window": None,
        "T_obs_window": float("nan"),
        "p_fw_window": float("nan"),
        "swap_k_endpoint_indices": None,
        "source_indices": None,
        "sink_indices": None,
        "source_endpoint_channels": None,
        "sink_endpoint_channels": None,
        "source_radius": None,
        "sink_radius": None,
        "source_sink_axis_distance": float("nan"),
        "rate_per_hour": float("nan"),
        "exit_reason": "ok",
    }
    if n_total == 0:
        out["exit_reason"] = "no_events_in_window"
        return out

    sub_labels = labels[evt_idx]
    idx_a = evt_idx[sub_labels == cluster_a]
    idx_b = evt_idx[sub_labels == cluster_b]
    out["n_events_a"] = int(idx_a.size)
    out["n_events_b"] = int(idx_b.size)

    # Window-level event gate (plan §1 v4 catch, locked at B0 audit time = 30 total events).
    # B0 audit `b0_eligibility_audit_2026-05-24.csv` showed median peri events = 363
    # (epi_1146 strict) to 36 (epi_620 candidate); 30 is a conservative floor that
    # excludes windows where any per-cluster template rank estimate would be too noisy.
    if n_total < min_events_per_window:
        out["exit_reason"] = f"insufficient_events_total:{n_total}<{min_events_per_window}"
        return out

    # rate denominator
    if effective_seconds is not None and effective_seconds > 0:
        out["rate_per_hour"] = float(n_total / (effective_seconds / 3600.0))

    if idx_a.size < min_events_per_cluster or idx_b.size < min_events_per_cluster:
        out["exit_reason"] = "insufficient_events_per_cluster"
        return out

    rank_a, valid_a = _per_cluster_template_rank(ranks, bools, idx_a)
    rank_b, valid_b = _per_cluster_template_rank(ranks, bools, idx_b)
    sweep = compute_swap_score_sweep(
        rank_a.astype(float), rank_b.astype(float),
        valid_a, valid_b,
        n_perm=n_perm, seed=seed,
    )
    if sweep.get("exit_reason") != "ok":
        out["exit_reason"] = sweep["exit_reason"]
        return out
    dk = sweep.get("decision_k")
    if dk is None:
        out["exit_reason"] = "no_decision_k"
        return out
    out["decision_k"] = int(dk)
    out["swap_class_window"] = sweep.get("swap_class")
    out["T_obs_window"] = float(sweep.get("T_obs", float("nan")))
    out["p_fw_window"] = float(sweep.get("p_fw", float("nan")))

    # Quality gate (user catch 2026-05-24): compute_swap_score_sweep returns a decision_k
    # via argmin even when swap_class="none" (T_obs < score_floor=0.5 or p_fw >= alpha_candidate
    # = 0.20). That decision_k is the argmin of noisy scores, NOT a real core size. Using
    # the resulting swap-k endpoint set for Δmetrics contaminates the analysis — this is
    # the artifact that made `new_node_fraction` significant in the swap=none negative
    # control in v1.0 (cohort_run_2026-05-24.md §3.2). Gate per-window: only proceed to
    # geometry derivation when swap_class != "none" AND T_obs >= score_floor.
    score_floor = float(sweep.get("score_floor", 0.5))
    if (
        out["swap_class_window"] == "none"
        or not np.isfinite(out["T_obs_window"])
        or out["T_obs_window"] < score_floor
    ):
        out["exit_reason"] = (
            f"weak_swap_window:class={out['swap_class_window']},"
            f"T_obs={out['T_obs_window']:.3f},floor={score_floor}"
        )
        return out

    # Source / sink split via rank_a ordering (lowest dk = source, highest dk = sink).
    # rank_a here is `argsort(argsort)` from _per_cluster_template_rank, which is dense rank.
    rank_a_dense = rank_a.astype(float)
    order = np.argsort(rank_a_dense, kind="stable")
    # Only consider channels in the joint valid set of cluster_a + cluster_b (matches
    # rank_displacement.compute_swap_score_sweep's joint_valid). We pick top-dk + bottom-dk
    # from the full ordering, then filter to joint_valid.
    joint_valid = (valid_a & valid_b)
    if joint_valid.sum() < 2 * dk:
        out["exit_reason"] = "joint_valid_too_small_for_decision_k"
        return out
    # Get ordering restricted to joint_valid channels
    valid_order = [i for i in order if joint_valid[i]]
    source_idx = valid_order[:dk]
    sink_idx = valid_order[-dk:]
    swap_idx = sorted(set(source_idx) | set(sink_idx))
    out["source_indices"] = source_idx
    out["sink_indices"] = sink_idx
    out["swap_k_endpoint_indices"] = swap_idx
    out["source_endpoint_channels"] = [channel_names[i] for i in source_idx]
    out["sink_endpoint_channels"] = [channel_names[i] for i in sink_idx]

    # Spatial radius per side (need coords; intersect with mapped subset).
    if coords is not None and coords.shape[0] == len(channel_names):
        # Only use channels with finite coordinates
        coord_finite = np.all(np.isfinite(coords), axis=1)
        source_mapped = [i for i in source_idx if coord_finite[i]]
        sink_mapped = [i for i in sink_idx if coord_finite[i]]
        out["source_radius"] = compute_endpoint_spatial_radius(source_mapped, coords)
        out["sink_radius"] = compute_endpoint_spatial_radius(sink_mapped, coords)
        if source_mapped and sink_mapped:
            out["source_sink_axis_distance"] = compute_source_sink_centroid_distance(
                source_mapped, sink_mapped, coords
            )
    return out


# =============================================================================
# Per-seizure Δmetric aggregation (plan §4.2 + §4.3, v4 catch median rules)
# =============================================================================


def compute_delta_metrics(
    peri_metrics: Dict[str, Any],
    baseline_metrics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate baseline windows then compute Δmetrics for (subject, seizure, side).

    Rules (plan §4.2 v4 catch):
      - scalar metrics (decision_k, rate, source/sink centroid_rms / mean_pairwise, axis):
        median across baseline windows.
      - set metrics (Jaccard, new_node_fraction): compute per-baseline-window, then median
        the scalar across baseline windows. CANNOT median endpoint set itself.

    Returns dict with keys:
      n_baseline_qualifying,
      delta_decision_k, delta_decision_k_normalized,
      jaccard_swap_k, new_node_fraction,
      delta_source_centroid_rms, delta_source_mean_pairwise,
      delta_sink_centroid_rms, delta_sink_mean_pairwise,
      delta_axis_distance,
      delta_rate_per_hour,
      baseline_decision_k_median, peri_decision_k,
      baseline_rate_median, peri_rate,
      exit_reason
    """
    out: Dict[str, Any] = {
        "n_baseline_qualifying": 0,
        "delta_decision_k": float("nan"),
        "delta_decision_k_normalized": float("nan"),
        "jaccard_swap_k": float("nan"),
        "new_node_fraction": float("nan"),
        "delta_source_centroid_rms": float("nan"),
        "delta_source_mean_pairwise": float("nan"),
        "delta_sink_centroid_rms": float("nan"),
        "delta_sink_mean_pairwise": float("nan"),
        "delta_axis_distance": float("nan"),
        "delta_rate_per_hour": float("nan"),
        "baseline_decision_k_median": float("nan"),
        "peri_decision_k": peri_metrics.get("decision_k"),
        "baseline_rate_median": float("nan"),
        "peri_rate": peri_metrics.get("rate_per_hour"),
        "exit_reason": "ok",
    }
    if peri_metrics.get("exit_reason") != "ok":
        out["exit_reason"] = f"peri_failed:{peri_metrics.get('exit_reason')}"
        return out
    qualifying = [b for b in baseline_metrics if b.get("exit_reason") == "ok" and b.get("decision_k") is not None]
    out["n_baseline_qualifying"] = len(qualifying)
    if len(qualifying) < N_BASELINE_MIN:
        out["exit_reason"] = f"insufficient_baselines:{len(qualifying)}<{N_BASELINE_MIN}"
        return out

    # Scalar metrics: median across baseline windows
    base_dk = np.array([b["decision_k"] for b in qualifying], dtype=float)
    base_rate = np.array([b["rate_per_hour"] for b in qualifying if np.isfinite(b.get("rate_per_hour", float("nan")))], dtype=float)
    out["baseline_decision_k_median"] = float(np.median(base_dk))
    out["baseline_rate_median"] = float(np.median(base_rate)) if base_rate.size else float("nan")
    if peri_metrics.get("decision_k") is not None:
        out["delta_decision_k"] = float(peri_metrics["decision_k"]) - out["baseline_decision_k_median"]
        # normalized Δk; denominator floored at 1 to avoid divide-by-zero
        out["delta_decision_k_normalized"] = out["delta_decision_k"] / max(out["baseline_decision_k_median"], 1.0)
    if np.isfinite(peri_metrics.get("rate_per_hour", float("nan"))) and np.isfinite(out["baseline_rate_median"]):
        out["delta_rate_per_hour"] = peri_metrics["rate_per_hour"] - out["baseline_rate_median"]

    # Spatial radius: median per metric per side across baselines
    def _base_med(side: str, key: str) -> float:
        vals = []
        for b in qualifying:
            r = b.get(f"{side}_radius") or {}
            v = r.get(key)
            if v is not None and np.isfinite(v):
                vals.append(float(v))
        return float(np.median(vals)) if vals else float("nan")

    for side in ("source", "sink"):
        for key in ("centroid_rms", "mean_pairwise"):
            base_med = _base_med(side, key)
            peri_r = peri_metrics.get(f"{side}_radius") or {}
            peri_v = peri_r.get(key)
            if peri_v is not None and np.isfinite(peri_v) and np.isfinite(base_med):
                out[f"delta_{side}_{key}"] = float(peri_v) - base_med

    base_axis_vals = [b.get("source_sink_axis_distance") for b in qualifying
                      if b.get("source_sink_axis_distance") is not None
                      and np.isfinite(b.get("source_sink_axis_distance", float("nan")))]
    if base_axis_vals and np.isfinite(peri_metrics.get("source_sink_axis_distance", float("nan"))):
        out["delta_axis_distance"] = (
            peri_metrics["source_sink_axis_distance"] - float(np.median(base_axis_vals))
        )

    # Set metrics: per-baseline-window scalar then median.
    #
    # Critical contract (user catch 2026-05-24 v1.1 re-run): raw Jaccard ∈ [0,1] is never
    # 0 in practice, so testing "mean(jaccard) = 0" is pathologically biased — reverse-
    # direction p (greater) is always ~0 regardless of true recruitment, causing false
    # FAIL_IDENTITY_CONTRACTION verdicts. Same for new_node_fraction. Fix: compute
    # baseline-vs-baseline (leave-one-out) Jaccard / new_node within the qualifying
    # baseline windows themselves to get a SUBJECT-SPECIFIC natural noise floor; report
    # `_delta` = peri vs baseline reference. The delta is centered at 0 under "peri is
    # just another baseline window" (no recruitment), so H0: mean(delta) = 0 is
    # meaningful and reverse p has its proper null distribution.
    peri_swap = set(peri_metrics.get("swap_k_endpoint_indices") or [])
    if peri_swap:
        # Peri vs each baseline
        jaccards = []
        new_fracs = []
        for b in qualifying:
            base_swap = set(b.get("swap_k_endpoint_indices") or [])
            union = peri_swap | base_swap
            if union:
                jaccards.append(len(peri_swap & base_swap) / len(union))
            new_fracs.append(len(peri_swap - base_swap) / max(len(peri_swap), 1))
        peri_jaccard_med = float(np.median(jaccards)) if jaccards else float("nan")
        peri_new_frac_med = float(np.median(new_fracs)) if new_fracs else float("nan")
        out["jaccard_swap_k"] = peri_jaccard_med
        out["new_node_fraction"] = peri_new_frac_med

        # Baseline-vs-baseline reference: each baseline window vs the other baselines
        # (leave-one-out within the qualifying set), median. Subject-specific noise floor.
        base_jaccs_ref = []
        base_new_fracs_ref = []
        for i, b in enumerate(qualifying):
            b_swap = set(b.get("swap_k_endpoint_indices") or [])
            if not b_swap:
                continue
            others = [bb for j, bb in enumerate(qualifying) if j != i]
            js: List[float] = []
            nfs: List[float] = []
            for bb in others:
                bb_swap = set(bb.get("swap_k_endpoint_indices") or [])
                u = b_swap | bb_swap
                if u:
                    js.append(len(b_swap & bb_swap) / len(u))
                nfs.append(len(b_swap - bb_swap) / max(len(b_swap), 1))
            if js:
                base_jaccs_ref.append(float(np.median(js)))
            if nfs:
                base_new_fracs_ref.append(float(np.median(nfs)))
        if base_jaccs_ref:
            ref_jaccard = float(np.median(base_jaccs_ref))
            out["jaccard_swap_k_baseline_ref"] = ref_jaccard
            out["jaccard_swap_k_delta"] = peri_jaccard_med - ref_jaccard
        else:
            out["jaccard_swap_k_baseline_ref"] = float("nan")
            out["jaccard_swap_k_delta"] = float("nan")
        if base_new_fracs_ref:
            ref_new_frac = float(np.median(base_new_fracs_ref))
            out["new_node_fraction_baseline_ref"] = ref_new_frac
            out["new_node_fraction_delta"] = peri_new_frac_med - ref_new_frac
        else:
            out["new_node_fraction_baseline_ref"] = float("nan")
            out["new_node_fraction_delta"] = float("nan")
    else:
        out["jaccard_swap_k_baseline_ref"] = float("nan")
        out["jaccard_swap_k_delta"] = float("nan")
        out["new_node_fraction_baseline_ref"] = float("nan")
        out["new_node_fraction_delta"] = float("nan")

    return out


# =============================================================================
# Cohort inference: wild cluster bootstrap (primary) + cluster-robust SE (companion)
# =============================================================================


def wild_cluster_bootstrap_p(
    y: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 1999,
    alternative: str = "greater",
    seed: int = 0,
) -> Dict[str, float]:
    """Cameron-Gelbach-Miller wild cluster RESTRICTED bootstrap (WCR; null-imposed).

    Tests H0: mean(y) = 0 against alternatives "greater" / "less" / "two-sided".

    Per CGM 2008 + Roodman et al. 2019, WCR is the recommended variant at small
    n_clusters (<30). The unrestricted variant (WCU) centers y first (y - mean(y))
    which inflates bootstrap variance under H1 and gives anti-conservative p-values
    at small G; WCR avoids this by imposing H0 in the data generation:

      H0: intercept-only model with β = 0 → fitted value = 0 → residuals = y - 0 = y
      WCR draw: y_boot[k] = ε[c(k)] * y[k]   (Rademacher weights, per cluster)
      t_boot   = mean(y_boot) / cluster_robust_SE(y_boot)
      t_obs    = mean(y)      / cluster_robust_SE(y)
      p_greater = (1 + #{t_boot ≥ t_obs}) / (n_boot + 1)

    All three (p_greater / p_less / p_two_sided) are computed from the same bootstrap
    pass; `alternative` selects which is returned as `p_value`. Reverse-direction p
    is needed for FAIL detection in compute_phase3_verdict (see user catch
    2026-05-24).

    Args:
      y: per-observation Δmetric (one row per seizure × side).
      cluster_ids: per-observation subject id (string or int).
      n_boot: bootstrap replicates (default 1999 → clean fractional p-values).
      alternative: "greater" / "less" / "two-sided" — selects which p to return as
        `p_value`. All three are also returned separately.
      seed: RNG seed.

    Returns dict with:
      t_obs, p_value, p_greater, p_less, p_two_sided,
      n_obs, n_clusters, mean_y, se_cluster_robust,
      ci95_lo, ci95_hi (bootstrap percentile CI on the cluster-mean statistic),
      n_boot, alternative, exit_reason.
    """
    y = np.asarray(y, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(y)
    y = y[finite]
    cluster_ids = cluster_ids[finite]
    n_obs = len(y)
    uniq_clusters = list(np.unique(cluster_ids))
    n_cl = len(uniq_clusters)
    if n_obs < 2 or n_cl < 2:
        return {
            "t_obs": float("nan"), "p_value": float("nan"),
            "p_greater": float("nan"), "p_less": float("nan"), "p_two_sided": float("nan"),
            "n_obs": int(n_obs), "n_clusters": int(n_cl),
            "mean_y": float(np.mean(y)) if n_obs else float("nan"),
            "se_cluster_robust": float("nan"),
            "ci95_lo": float("nan"), "ci95_hi": float("nan"),
            "n_boot": int(n_boot),
            "alternative": alternative,
            "exit_reason": "insufficient_data",
        }
    mean_y = float(np.mean(y))

    def _cluster_t(values: np.ndarray) -> Tuple[float, float]:
        """Mean over all rows + cluster-robust SE → returns (mean, SE)."""
        v_mean = float(values.mean())
        resid = values - v_mean
        cl_resid_sums = np.array([resid[cluster_ids == c].sum() for c in uniq_clusters])
        var_mean = (cl_resid_sums ** 2).sum() / (n_obs ** 2)
        # Small-cluster correction (CGM 2008): G/(G-1) * (N-1)/(N-K); K=1 → G/(G-1)
        var_mean *= n_cl / max(n_cl - 1, 1)
        se = float(np.sqrt(var_mean))
        return v_mean, se

    obs_mean, obs_se = _cluster_t(y)
    t_obs = (obs_mean - 0.0) / (obs_se + 1e-18)

    rng = np.random.default_rng(seed)
    cluster_to_idx = {c: i for i, c in enumerate(uniq_clusters)}
    cluster_id_int = np.array([cluster_to_idx[c] for c in cluster_ids], dtype=int)

    t_boot = np.zeros(n_boot, dtype=float)
    boot_means = np.zeros(n_boot, dtype=float)
    for b in range(n_boot):
        eps = rng.choice([-1.0, 1.0], size=n_cl)
        # WCR (null-imposed): under H0 mean(y)=0, residuals = y - 0 = y; multiply raw y by Rademacher.
        # Critical: do NOT center y first (that would be WCU and is anti-conservative at small G).
        y_b = y * eps[cluster_id_int]
        b_mean, b_se = _cluster_t(y_b)
        t_boot[b] = (b_mean - 0.0) / (b_se + 1e-18)
        boot_means[b] = b_mean

    p_greater = (1 + (t_boot >= t_obs).sum()) / (n_boot + 1)
    p_less = (1 + (t_boot <= t_obs).sum()) / (n_boot + 1)
    p_two_sided = (1 + (np.abs(t_boot) >= abs(t_obs)).sum()) / (n_boot + 1)
    if alternative == "greater":
        p = p_greater
    elif alternative == "less":
        p = p_less
    elif alternative == "two-sided":
        p = p_two_sided
    else:
        raise ValueError(f"unknown alternative {alternative!r}")

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return {
        "t_obs": float(t_obs), "p_value": float(p),
        "p_greater": float(p_greater), "p_less": float(p_less), "p_two_sided": float(p_two_sided),
        "n_obs": int(n_obs), "n_clusters": int(n_cl),
        "mean_y": float(mean_y),
        "se_cluster_robust": float(obs_se),
        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "n_boot": int(n_boot),
        "alternative": alternative,
        "exit_reason": "ok",
    }


def cluster_robust_se_p(
    y: np.ndarray,
    cluster_ids: np.ndarray,
    alternative: str = "greater",
) -> Dict[str, float]:
    """statsmodels OLS(cov_type='cluster') with intercept-only model.

    Companion to wild_cluster_bootstrap_p. At n_clusters < ~30, this asymptotic
    test is less reliable; report together with bootstrap and Wilcoxon.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return {"p_value": float("nan"), "exit_reason": "statsmodels_unavailable"}
    y = np.asarray(y, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(y)
    y = y[finite]
    cluster_ids = cluster_ids[finite]
    n_obs = len(y)
    if n_obs < 2:
        return {"p_value": float("nan"), "exit_reason": "insufficient_data"}
    X = np.ones_like(y)
    try:
        result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": cluster_ids})
        params = result.params[0]
        se = result.bse[0]
        # statsmodels reports two-sided p by default; convert to one-sided.
        p_two = float(result.pvalues[0])
        if alternative == "greater":
            p = p_two / 2 if params > 0 else 1 - p_two / 2
        elif alternative == "less":
            p = p_two / 2 if params < 0 else 1 - p_two / 2
        else:
            p = p_two
        ci = result.conf_int(alpha=0.05)[0]
        return {
            "p_value": float(p),
            "t_obs": float(params / (se + 1e-18)),
            "mean_y": float(params), "se": float(se),
            "ci95_lo": float(ci[0]), "ci95_hi": float(ci[1]),
            "n_obs": int(n_obs), "n_clusters": int(len(np.unique(cluster_ids))),
            "alternative": alternative,
            "exit_reason": "ok",
        }
    except Exception as e:
        return {"p_value": float("nan"), "exit_reason": f"fit_failed:{type(e).__name__}:{e}"}


def subject_wilcoxon_p(
    per_seizure_y: np.ndarray,
    cluster_ids: np.ndarray,
    alternative: str = "greater",
) -> Dict[str, float]:
    """Subject-mean Wilcoxon signed-rank against 0 (sanity check).

    Aggregates per-seizure y to per-subject mean, then Wilcoxon against 0.
    """
    from scipy.stats import wilcoxon
    per_seizure_y = np.asarray(per_seizure_y, dtype=float)
    cluster_ids = np.asarray(cluster_ids)
    finite = np.isfinite(per_seizure_y)
    per_seizure_y = per_seizure_y[finite]
    cluster_ids = cluster_ids[finite]
    if per_seizure_y.size == 0:
        return {"p_value": float("nan"), "n_subjects": 0, "exit_reason": "no_data"}
    uniq = list(np.unique(cluster_ids))
    sub_means = np.array([per_seizure_y[cluster_ids == c].mean() for c in uniq])
    if sub_means.size < 2:
        return {"p_value": float("nan"), "n_subjects": int(sub_means.size), "exit_reason": "n<2"}
    try:
        result = wilcoxon(sub_means, alternative=alternative, zero_method="wilcox")
        return {
            "p_value": float(result.pvalue),
            "statistic": float(result.statistic),
            "n_subjects": int(sub_means.size),
            "median_subject_mean": float(np.median(sub_means)),
            "alternative": alternative,
            "exit_reason": "ok",
        }
    except Exception as e:
        return {"p_value": float("nan"), "exit_reason": f"wilcoxon_failed:{e}"}


def bh_fdr(p_values: Dict[str, float], q: float = 0.10) -> Dict[str, Dict[str, float]]:
    """Benjamini-Hochberg FDR control across a family of p-values.

    Returns dict per metric: {p_raw, q_value, rejected_at_q}.
    NaN p-values are skipped (counted as not-tested).
    """
    items = [(k, v) for k, v in p_values.items() if np.isfinite(v)]
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    out: Dict[str, Dict[str, float]] = {}
    if m == 0:
        return {k: {"p_raw": float(v), "q_value": float("nan"), "rejected_at_q": False}
                for k, v in p_values.items()}
    # BH q-values: q[i] = min over j≥i of (p[j] * m / (j+1)) clipped to [0, 1]
    sorted_p = np.array([v for _, v in items])
    raw_q = sorted_p * m / np.arange(1, m + 1)
    # Monotone non-increasing right-to-left
    q_vals = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_vals = np.clip(q_vals, 0.0, 1.0)
    rejected = q_vals <= q
    for (k, v), qv, rej in zip(items, q_vals, rejected):
        out[k] = {"p_raw": float(v), "q_value": float(qv), "rejected_at_q": bool(rej)}
    # Add NaN entries
    for k, v in p_values.items():
        if not np.isfinite(v):
            out[k] = {"p_raw": float(v), "q_value": float("nan"), "rejected_at_q": False}
    return out


# =============================================================================
# Phase 3 verdict (plan §6.1)
# =============================================================================


PRIMARY_BH_FDR_METRICS = (
    "jaccard_swap_k",
    "new_node_fraction",
    "source_centroid_rms",
    "source_mean_pairwise",
    "sink_centroid_rms",
    "sink_mean_pairwise",
)


def _expected_direction(metric: str) -> str:
    """SEF-ITP expected direction (Δ) for each primary metric (recruitment/expansion = positive)."""
    if metric == "jaccard_swap_k":
        return "less"  # Jaccard ↓ = recruitment
    if metric == "new_node_fraction":
        return "greater"  # ↑ = recruitment
    return "greater"  # radius ↑ = spatial expansion


def compute_phase3_verdict(
    inference_results: Dict[str, Dict[str, Any]],
    delta_k_sign: float,
    delta_k_normalized_sign: float,
    n_qualifying_subjects: int,
    q_threshold: float = 0.10,
    n_subjects_floor: int = 6,
) -> Dict[str, Any]:
    """Plan §6.1 verdict from BH-FDR-corrected primary results + Δk sign-check.

    inference_results: {metric: {"bootstrap": {"p_value": ..., "mean_y": ...}, ...}}
      Each metric must have a `bootstrap.p_value` (one-sided, in SEF-ITP direction) and
      `bootstrap.mean_y` (effect direction).
    delta_k_sign: sign of cohort mean Δk (positive = SEF-ITP-direction recruitment)
    delta_k_normalized_sign: sign of cohort mean normalized Δk

    Returns:
      verdict ∈ {SUPPORTED, NULL, FAIL_IDENTITY_CONTRACTION, FAIL_RADIUS_CONTRACTION,
        UNDERPOWERED},
      details (BH-FDR summary, sign check, reasoning)
    """
    out: Dict[str, Any] = {
        "verdict": None,
        "n_qualifying_subjects": int(n_qualifying_subjects),
        "primary_metrics_tested": list(PRIMARY_BH_FDR_METRICS),
        "delta_k_sign": float(delta_k_sign) if np.isfinite(delta_k_sign) else None,
        "delta_k_normalized_sign": float(delta_k_normalized_sign) if np.isfinite(delta_k_normalized_sign) else None,
    }
    if n_qualifying_subjects < n_subjects_floor:
        out["verdict"] = "UNDERPOWERED"
        out["reason"] = f"n_qualifying_subjects={n_qualifying_subjects} < floor={n_subjects_floor}"
        return out

    # Forward-direction BH-FDR (for SUPPORTED branch)
    p_values_forward = {
        m: float(inference_results.get(m, {}).get("bootstrap", {}).get("p_value", float("nan")))
        for m in PRIMARY_BH_FDR_METRICS
    }
    bh_forward = bh_fdr(p_values_forward, q=q_threshold)
    out["bh_fdr"] = bh_forward  # name preserved for back-compat

    # Reverse-direction BH-FDR (for FAIL branch — user catch 2026-05-24).
    # Plan §6.1 FAIL requires significant evidence in the REVERSE direction; using
    # forward p (which is ~1 under reverse effects) cannot identify FAIL. Compute the
    # reverse p from same bootstrap, BH-FDR separately, and use that for FAIL detection.
    def _reverse_p(m: str) -> float:
        boot = inference_results.get(m, {}).get("bootstrap", {})
        # `_expected_direction` = "less" for jaccard → reverse = "greater"; else reverse = "less"
        if _expected_direction(m) == "less":
            return float(boot.get("p_greater", float("nan")))
        return float(boot.get("p_less", float("nan")))

    p_values_reverse = {m: _reverse_p(m) for m in PRIMARY_BH_FDR_METRICS}
    bh_reverse = bh_fdr(p_values_reverse, q=q_threshold)
    out["bh_fdr_reverse"] = bh_reverse

    # Direction signs (descriptive)
    direction_correct: Dict[str, bool] = {}
    direction_reverse: Dict[str, bool] = {}
    for m in PRIMARY_BH_FDR_METRICS:
        mean_y = inference_results.get(m, {}).get("bootstrap", {}).get("mean_y", float("nan"))
        if not np.isfinite(mean_y):
            direction_correct[m] = False
            direction_reverse[m] = False
            continue
        if _expected_direction(m) == "less":
            direction_correct[m] = mean_y < 0
            direction_reverse[m] = mean_y > 0
        else:
            direction_correct[m] = mean_y > 0
            direction_reverse[m] = mean_y < 0
    out["direction_correct"] = direction_correct

    # FAIL detection (BH-FDR on REVERSE-direction p; direction confirmed by mean sign)
    fail_identity = (
        bh_reverse.get("jaccard_swap_k", {}).get("rejected_at_q")
        and direction_reverse.get("jaccard_swap_k", False)
    ) or (
        bh_reverse.get("new_node_fraction", {}).get("rejected_at_q")
        and direction_reverse.get("new_node_fraction", False)
    )
    fail_radius = any(
        bh_reverse.get(m, {}).get("rejected_at_q") and direction_reverse.get(m, False)
        for m in ("source_centroid_rms", "source_mean_pairwise", "sink_centroid_rms", "sink_mean_pairwise")
    )
    if fail_identity:
        out["verdict"] = "FAIL_IDENTITY_CONTRACTION"
        out["reason"] = (
            "Reverse-direction BH-FDR rejected for Jaccard ↑ or new_node_fraction ↓ — "
            "identity contraction / loss; reverse of SEF-ITP recruitment prediction"
        )
        return out
    if fail_radius:
        out["verdict"] = "FAIL_RADIUS_CONTRACTION"
        out["reason"] = (
            "Reverse-direction BH-FDR rejected for radius ↓ — spatial contraction; "
            "reverse of SEF-ITP expansion prediction"
        )
        return out

    # SUPPORTED conditions (plan §6.1 — forward-direction BH-FDR)
    identity_sig = (
        bh_forward.get("jaccard_swap_k", {}).get("rejected_at_q") and direction_correct.get("jaccard_swap_k", False)
    ) or (
        bh_forward.get("new_node_fraction", {}).get("rejected_at_q") and direction_correct.get("new_node_fraction", False)
    )
    radius_sig = any(
        bh_forward.get(m, {}).get("rejected_at_q") and direction_correct.get(m, False)
        for m in ("source_centroid_rms", "source_mean_pairwise", "sink_centroid_rms", "sink_mean_pairwise")
    )
    out["identity_recruitment_significant"] = bool(identity_sig)
    out["radius_expansion_significant"] = bool(radius_sig)

    # Δk concordance check (plan §1, §6.1)
    delta_k_concordant = (
        out["delta_k_sign"] is not None
        and out["delta_k_normalized_sign"] is not None
        and out["delta_k_sign"] > 0
        and out["delta_k_normalized_sign"] > 0
    )
    out["delta_k_concordant"] = bool(delta_k_concordant)

    if (identity_sig or radius_sig) and delta_k_concordant:
        out["verdict"] = "SUPPORTED"
        out["reason"] = "Identity recruitment OR radius expansion significant AND Δk concordant (>0 raw + normalized)"
    else:
        out["verdict"] = "NULL"
        reasons = []
        if not (identity_sig or radius_sig):
            reasons.append("no identity recruitment or radius expansion significant after BH-FDR")
        if not delta_k_concordant:
            reasons.append(
                f"Δk not concordant (raw sign={out['delta_k_sign']}, normalized sign={out['delta_k_normalized_sign']})"
            )
        out["reason"] = "; ".join(reasons)
    return out
