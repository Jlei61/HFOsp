"""SEF-ITP Phase 3 v2 — exploratory trajectory + alternative endpoints + electrode coverage.

User catch 2026-05-24: v1 (peri-vs-baseline test) reads NULL. Before declaring "no
recruitment", we should:

  1. **Continuous trajectory** — sliding 55-min windows from t_seizure − 24h to +24h
     at 30-min stride; plot geometry metrics as function of `time_to_seizure_hours`.
     Question shifts from "is peri DIFFERENT from baseline?" (NULL) to "does geometry
     EVOLVE continuously as seizure approaches?" (descriptive trajectory).

  2. **Alternative endpoint definitions** — instead of variable-k swap-k node (Topic
     4 H2 spatial layer), also compute fixed-k top-3 / top-5 by per-window template
     rank (PR-6 anchoring style). If swap-k gives flat trajectory but top-3/5 shows
     drift → swap-k is too restricted; the geometry IS evolving but at non-swap nodes.
     If BOTH flat → geometry truly stable. If BOTH drift → all geometries evolve, swap
     not special.

  3. **Electrode coverage check** — for each subject, are SEEG electrodes positioned
     near the swap-k anatomical region? Compute swap-k node coordinates, nearest
     non-swap-k neighbors, distance to clinical SOZ. If electrodes do NOT sample near
     swap-k anatomy, the v1 NULL is "real null" = measurement artifact (no recording
     at recruitment sites). If electrodes DO sample well → NULL is a real biological
     finding about swap-k stability.

This module is **descriptive / exploratory** — no cohort-level hypothesis test, no
baseline-comparison verdict. Outputs per-subject trajectories + electrode coverage
reports; cohort summary is qualitative ("does subject N show progressive drift?").
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.sef_itp_phase2 import (
    _per_cluster_template_rank,
    compute_endpoint_spatial_radius,
    compute_source_sink_centroid_distance,
)
from src.rank_displacement import compute_swap_score_sweep
from src.sef_itp_phase3 import (
    _window_effective_seconds,
    PERI_NOMINAL_MIN,
    COVERAGE_FLOOR,
)


__version__ = "v2.0.0"


TRAJECTORY_WINDOW_MIN = 55       # nominal window duration
TRAJECTORY_STRIDE_MIN = 30       # sliding stride
TRAJECTORY_RANGE_PRE_H = 24      # how many hours pre-seizure to look back
TRAJECTORY_RANGE_POST_H = 24     # how many hours post-seizure to look forward
ENDPOINT_FIXED_K_VALUES = (3, 5)  # alternative endpoint definitions to compute
ELECTRODE_COVERAGE_DISTANCE_BINS_MM = (5, 10, 15, 20, 30)  # for nearest-neighbor reporting


# =============================================================================
# Per-window endpoint extraction (3 modes: swap-k, top-3, top-5)
# =============================================================================


def _extract_endpoints_for_window(
    rank_a: np.ndarray, valid_a: np.ndarray,
    rank_b: np.ndarray, valid_b: np.ndarray,
    swap_decision_k: Optional[int],
    fixed_k_values: Tuple[int, ...] = ENDPOINT_FIXED_K_VALUES,
) -> Dict[str, Dict[str, List[int]]]:
    """Compute endpoint sets under multiple definitions (swap-k variable + top-k fixed).

    Returns dict by mode label:
      "swap_k" (if swap_decision_k given): source / sink from rank_a ordering restricted
        to joint_valid, top-k ∪ bottom-k where k = swap_decision_k.
      "top_k_{K}" for K in fixed_k_values: source = top-K by rank_a, sink = bottom-K
        by rank_a (PR-6 anchoring style, fixed K, no joint-valid restriction beyond
        rank_a's own valid_a).
    """
    out: Dict[str, Dict[str, List[int]]] = {}

    # swap-k mode (variable-k via swap_decision_k)
    if swap_decision_k is not None and swap_decision_k > 0:
        joint = valid_a & valid_b
        order = np.argsort(rank_a, kind="stable")
        valid_order = [int(i) for i in order if joint[i]]
        if len(valid_order) >= 2 * swap_decision_k:
            src = valid_order[:swap_decision_k]
            snk = valid_order[-swap_decision_k:]
            out["swap_k"] = {"source": src, "sink": snk, "k": swap_decision_k}

    # top-K fixed modes (PR-6 anchoring style; uses rank_a, valid_a only)
    order_a = np.argsort(rank_a, kind="stable")
    valid_order_a = [int(i) for i in order_a if valid_a[i]]
    for k in fixed_k_values:
        if len(valid_order_a) >= 2 * k:
            src = valid_order_a[:k]
            snk = valid_order_a[-k:]
            out[f"top_k_{k}"] = {"source": src, "sink": snk, "k": k}

    return out


PER_EVENT_K_VALUES = (2, 3, 4, 5, 6)
PER_EVENT_MIN_EVENTS_USED = 10  # window-level floor: report median only if >= 10 events qualified


def compute_per_event_endpoint_geometry(
    ranks_win: np.ndarray,
    bools_win: np.ndarray,
    coords: Optional[np.ndarray],
    k_values: Tuple[int, ...] = PER_EVENT_K_VALUES,
    min_events_used: int = PER_EVENT_MIN_EVENTS_USED,
) -> Dict[str, Any]:
    """Event-level endpoint geometry (NOT template-of-window, NOT swap-k template-pair).

    For each event in the window:
      - Filter to participating channels with finite rank AND mapped coords.
      - Per-event eligibility: n_eligible >= max(6, 2*k) — guarantees source
        and sink selections are completely non-overlapping (otherwise the
        metric measures "participating-field compactness" not "endpoint
        compactness"; user catch 2026-05-25, v2.2 → v2.3 fix).
      - Source top-k = k lowest-rank channels; sink last-k = k highest-rank.
      - Centroid RMS of k selected channel coords = "how tight this event's
        propagation endpoint is around its own centroid".

    Aggregate per window: median + p25 + p75 of per-event RMS.
    Window-level eligibility: n_events_used >= min_events_used (default 10),
    otherwise rms_median = None for that (which, k).

    Per user 2026-05-24 scientific layering: this metric measures per-event
    propagation seed/terminus compactness, NOT SOZ extent. A tight RMS only
    says "seeds are sampled at tight locations", not "SOZ is small".
    """
    out: Dict[str, Any] = {}
    if coords is None or ranks_win.size == 0:
        return out
    coord_finite = np.all(np.isfinite(coords), axis=1)
    _, n_events = ranks_win.shape
    for which in ("source", "sink"):
        for k in k_values:
            key = f"{which}_top_{k}" if which == "source" else f"{which}_last_{k}"
            min_part = max(6, 2 * k)
            event_rms: List[float] = []
            for ei in range(n_events):
                part = bools_win[:, ei].astype(bool)
                eligible = part & np.isfinite(ranks_win[:, ei]) & coord_finite
                n_elig = int(eligible.sum())
                if n_elig < min_part:
                    continue
                elig_idx = np.where(eligible)[0]
                elig_ranks = ranks_win[elig_idx, ei]
                order = np.argsort(elig_ranks)
                sel = elig_idx[order[:k] if which == "source" else order[-k:]]
                sel_coords = coords[sel]
                centroid = sel_coords.mean(axis=0)
                rms = float(np.sqrt(np.mean(np.sum((sel_coords - centroid) ** 2, axis=1))))
                event_rms.append(rms)
            n_used = len(event_rms)
            if n_used >= min_events_used:
                arr = np.asarray(event_rms)
                out[key] = {
                    "rms_median": float(np.median(arr)),
                    "rms_p25": float(np.percentile(arr, 25)),
                    "rms_p75": float(np.percentile(arr, 75)),
                    "n_events_used": int(n_used),
                }
            else:
                out[key] = {"rms_median": None, "rms_p25": None, "rms_p75": None,
                            "n_events_used": int(n_used)}
    return out


def compute_trajectory_window_metrics(
    t_start: float, t_end: float,
    event_abs_times: np.ndarray, labels: np.ndarray,
    ranks: np.ndarray, bools: np.ndarray,
    coords: Optional[np.ndarray], channel_names: List[str],
    cluster_a: int = 0, cluster_b: int = 1,
    effective_seconds: Optional[float] = None,
    n_perm: int = 200, seed: int = 0,
    min_events_per_cluster: int = 10,
    min_events_per_window: int = 30,
    fixed_k_values: Tuple[int, ...] = ENDPOINT_FIXED_K_VALUES,
) -> Dict[str, Any]:
    """Trajectory variant of compute_window_metrics: report multiple endpoint modes.

    Unlike Phase 3 v1.1's `compute_window_metrics` which gates by swap_class != "none",
    trajectory analysis DOES compute fixed top-k endpoints even when swap is weak —
    that's the whole point of the alt-endpoint comparison. Weak-swap flag stored as
    diagnostic; swap-k endpoint omitted in that case but top-k still derived.
    """
    out: Dict[str, Any] = {
        "t_start": float(t_start), "t_end": float(t_end),
        "effective_seconds": float(effective_seconds) if effective_seconds is not None else float("nan"),
        "n_events_total": 0, "n_events_a": 0, "n_events_b": 0,
        "decision_k": None, "swap_class_window": None,
        "T_obs_window": float("nan"), "p_fw_window": float("nan"),
        "weak_swap": True,
        "endpoint_modes": {},
        "per_event_endpoint_geometry": {},
        "rate_per_hour": float("nan"),
        "exit_reason": "ok",
    }
    mask = (event_abs_times >= t_start) & (event_abs_times < t_end)
    evt_idx = np.where(mask)[0]
    n_total = int(evt_idx.size)
    out["n_events_total"] = n_total
    if effective_seconds is not None and effective_seconds > 0:
        out["rate_per_hour"] = float(n_total / (effective_seconds / 3600.0))
    if n_total < min_events_per_window:
        out["exit_reason"] = f"insufficient_events_total:{n_total}<{min_events_per_window}"
        return out

    # Per-event endpoint geometry (event-level, not template-level).
    # Compute even when swap-k weak — independent of swap statistics.
    out["per_event_endpoint_geometry"] = compute_per_event_endpoint_geometry(
        ranks[:, evt_idx], bools[:, evt_idx], coords,
    )

    sub_labels = labels[evt_idx]
    idx_a = evt_idx[sub_labels == cluster_a]
    idx_b = evt_idx[sub_labels == cluster_b]
    out["n_events_a"] = int(idx_a.size)
    out["n_events_b"] = int(idx_b.size)
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
    if sweep.get("exit_reason") == "ok":
        out["decision_k"] = int(sweep.get("decision_k") or 0) or None
        out["swap_class_window"] = sweep.get("swap_class")
        out["T_obs_window"] = float(sweep.get("T_obs", float("nan")))
        out["p_fw_window"] = float(sweep.get("p_fw", float("nan")))
        score_floor = float(sweep.get("score_floor", 0.5))
        out["weak_swap"] = (
            out["swap_class_window"] == "none"
            or not np.isfinite(out["T_obs_window"])
            or out["T_obs_window"] < score_floor
        )

    # Endpoint extraction across modes — swap-k only when not weak; top-K always (if joint valid sufficient).
    rank_a_dense = rank_a.astype(float)
    endpoints = _extract_endpoints_for_window(
        rank_a=rank_a_dense, valid_a=valid_a,
        rank_b=rank_b.astype(float), valid_b=valid_b,
        swap_decision_k=(out["decision_k"] if not out["weak_swap"] else None),
        fixed_k_values=fixed_k_values,
    )

    # Compute per-mode spatial metrics
    coord_finite = np.all(np.isfinite(coords), axis=1) if coords is not None else None
    for mode, ep in endpoints.items():
        src = ep["source"]
        snk = ep["sink"]
        rec: Dict[str, Any] = {
            "k": ep["k"],
            "source_channels": [channel_names[i] for i in src],
            "sink_channels": [channel_names[i] for i in snk],
            "swap_k_endpoint_indices": sorted(set(src) | set(snk)),
        }
        if coords is not None and coord_finite is not None:
            src_mapped = [i for i in src if coord_finite[i]]
            snk_mapped = [i for i in snk if coord_finite[i]]
            rec["source_radius"] = compute_endpoint_spatial_radius(src_mapped, coords)
            rec["sink_radius"] = compute_endpoint_spatial_radius(snk_mapped, coords)
            if src_mapped and snk_mapped:
                rec["source_sink_axis_distance"] = compute_source_sink_centroid_distance(
                    src_mapped, snk_mapped, coords
                )
            else:
                rec["source_sink_axis_distance"] = float("nan")
        out["endpoint_modes"][mode] = rec
    return out


# =============================================================================
# Trajectory enumeration (sliding windows around seizure)
# =============================================================================


def _enumerate_trajectory_windows(
    seizure_onset_t: float,
    blocks: List[Tuple[float, float]],
    range_pre_h: float = TRAJECTORY_RANGE_PRE_H,
    range_post_h: float = TRAJECTORY_RANGE_POST_H,
    window_minutes: float = TRAJECTORY_WINDOW_MIN,
    stride_minutes: float = TRAJECTORY_STRIDE_MIN,
    coverage_floor: float = COVERAGE_FLOOR,
) -> List[Tuple[float, float, float, float]]:
    """Sliding window centers from t_seizure - range_pre to + range_post, stride.

    Returns [(t_start, t_end, effective_seconds, time_to_seizure_hours)] for windows
    with coverage >= floor. time_to_seizure < 0 = pre, > 0 = post.
    """
    nominal_sec = window_minutes * 60.0
    stride_sec = stride_minutes * 60.0
    floor_sec = coverage_floor * nominal_sec
    range_start = seizure_onset_t - range_pre_h * 3600.0
    range_end = seizure_onset_t + range_post_h * 3600.0
    out: List[Tuple[float, float, float, float]] = []
    n_iter = int(np.floor((range_end - range_start - nominal_sec) / stride_sec)) + 1
    for i in range(n_iter):
        ts = range_start + i * stride_sec
        te = ts + nominal_sec
        if te > range_end:
            break
        eff = _window_effective_seconds(ts, te, blocks)
        if eff >= floor_sec:
            # Time-to-seizure: window center vs onset; negative = pre, positive = post
            center = 0.5 * (ts + te)
            tts_h = (center - seizure_onset_t) / 3600.0
            out.append((ts, te, eff, tts_h))
    return out


def compute_trajectory_for_seizure(
    seizure_onset_t: float, seizure_offset_t: float,
    event_abs_times: np.ndarray, labels: np.ndarray,
    ranks: np.ndarray, bools: np.ndarray,
    coords: Optional[np.ndarray], channel_names: List[str],
    blocks: List[Tuple[float, float]],
    cluster_a: int = 0, cluster_b: int = 1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Sliding-window trajectory around a single seizure.

    Returns list of per-window dicts, sorted by time_to_seizure_hours.
    """
    windows = _enumerate_trajectory_windows(seizure_onset_t, blocks)
    out: List[Dict[str, Any]] = []
    for wi, (ts, te, eff, tts) in enumerate(windows):
        m = compute_trajectory_window_metrics(
            t_start=ts, t_end=te,
            event_abs_times=event_abs_times, labels=labels, ranks=ranks, bools=bools,
            coords=coords, channel_names=channel_names,
            cluster_a=cluster_a, cluster_b=cluster_b,
            effective_seconds=eff, n_perm=200, seed=seed + wi,
        )
        m["time_to_seizure_hours"] = tts
        # Mark whether window is within the actual seizure (ictal overlap)
        m["ictal_overlap"] = (ts < seizure_offset_t and te > seizure_onset_t)
        out.append(m)
    return out


def compute_trajectory_full_sweep(
    event_abs_times: np.ndarray, labels: np.ndarray,
    ranks: np.ndarray, bools: np.ndarray,
    coords: Optional[np.ndarray], channel_names: List[str],
    blocks: List[Tuple[float, float]],
    seizure_onsets: List[float],
    window_minutes: float = TRAJECTORY_WINDOW_MIN,
    stride_minutes: float = TRAJECTORY_STRIDE_MIN,
    coverage_floor: float = COVERAGE_FLOOR,
    cluster_a: int = 0, cluster_b: int = 1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Single sweep across the entire recording — NOT anchored to seizures.

    Returns per-window metrics dicts (same shape as compute_trajectory_window_metrics),
    each tagged with `nearest_seizure_offset_h` (signed; negative = nearest seizure
    is in the future; positive = past).
    """
    if not blocks:
        return []
    nominal_sec = window_minutes * 60.0
    stride_sec = stride_minutes * 60.0
    floor_sec = coverage_floor * nominal_sec
    record_start = min(b[0] for b in blocks)
    record_end = max(b[1] for b in blocks)
    out: List[Dict[str, Any]] = []
    wi = 0
    ts = record_start
    while ts + nominal_sec <= record_end:
        te = ts + nominal_sec
        eff = _window_effective_seconds(ts, te, blocks)
        if eff >= floor_sec:
            m = compute_trajectory_window_metrics(
                t_start=ts, t_end=te,
                event_abs_times=event_abs_times, labels=labels, ranks=ranks, bools=bools,
                coords=coords, channel_names=channel_names,
                cluster_a=cluster_a, cluster_b=cluster_b,
                effective_seconds=eff, n_perm=200, seed=seed + wi,
            )
            center = 0.5 * (ts + te)
            if seizure_onsets:
                deltas = [(center - so) / 3600.0 for so in seizure_onsets]
                nearest = min(deltas, key=abs)
                m["nearest_seizure_offset_h"] = float(nearest)
            else:
                m["nearest_seizure_offset_h"] = None
            out.append(m)
            wi += 1
        ts += stride_sec
    return out


# =============================================================================
# Electrode coverage check (SEEG vs swap-k anatomy + clinical SOZ relation)
# =============================================================================


def compute_electrode_coverage(
    channel_names: List[str],
    coords: Optional[np.ndarray],
    swap_k_channel_names: List[str],
    soz_channels: Optional[List[str]] = None,
    distance_bins_mm: Tuple[float, ...] = ELECTRODE_COVERAGE_DISTANCE_BINS_MM,
) -> Dict[str, Any]:
    """Electrode coverage diagnostic for swap-k node anatomy.

    Reports:
      n_swap_k_channels: how many swap-k nodes the subject has
      n_coords_finite: how many channels have finite 3D coordinates
      n_swap_k_with_coords: how many swap-k nodes have finite coords
      swap_k_spatial_extent: pairwise distance summary (min/median/max in mm)
      nearest_non_swap_neighbor: for each swap-k node, distance to nearest non-swap-k
        SEEG channel (in mm), reported as cohort summary (median/IQR)
      density_by_radius: for each swap-k node, count of non-swap-k SEEG channels
        within X mm (X in distance_bins_mm). Indicates "is the swap-k region densely
        sampled or sparse".
      soz_relation (if soz_channels provided):
        swap_k_in_soz: count of swap-k nodes that are in clinical SOZ list
        n_soz: total SOZ channels
        min_dist_swap_to_soz: shortest distance from any swap-k node to any SOZ node
        median_dist_swap_to_soz: median pairwise distance
    """
    out: Dict[str, Any] = {
        "n_channels_total": len(channel_names),
        "n_swap_k_channels": len(swap_k_channel_names),
        "n_coords_finite": 0,
        "n_swap_k_with_coords": 0,
        "swap_k_spatial_extent": None,
        "nearest_non_swap_neighbor": None,
        "density_by_radius": None,
        "soz_relation": None,
    }
    if coords is None:
        return out
    coord_finite = np.all(np.isfinite(coords), axis=1)
    out["n_coords_finite"] = int(coord_finite.sum())
    name_to_idx = {nm: i for i, nm in enumerate(channel_names)}
    swap_idx = [name_to_idx[nm] for nm in swap_k_channel_names if nm in name_to_idx and coord_finite[name_to_idx[nm]]]
    out["n_swap_k_with_coords"] = len(swap_idx)
    if not swap_idx:
        return out

    swap_pts = coords[swap_idx]
    # Pairwise distance within swap-k
    if len(swap_idx) >= 2:
        pdist = np.linalg.norm(swap_pts[:, None, :] - swap_pts[None, :, :], axis=-1)
        iu = np.triu_indices(len(swap_idx), k=1)
        d = pdist[iu]
        out["swap_k_spatial_extent"] = {
            "n_pairs": int(len(d)),
            "min_mm": float(d.min()),
            "median_mm": float(np.median(d)),
            "max_mm": float(d.max()),
        }

    # Nearest non-swap-k SEEG neighbor per swap-k node
    swap_set = set(swap_idx)
    non_swap_finite = [i for i in range(len(channel_names)) if coord_finite[i] and i not in swap_set]
    if non_swap_finite:
        non_swap_pts = coords[non_swap_finite]
        nearest_dists = []
        density_counts = {f"<={b}mm": 0 for b in distance_bins_mm}
        for sp in swap_pts:
            dists = np.linalg.norm(non_swap_pts - sp, axis=-1)
            nearest_dists.append(float(dists.min()))
            for b in distance_bins_mm:
                density_counts[f"<={b}mm"] += int((dists <= b).sum())
        out["nearest_non_swap_neighbor"] = {
            "median_mm": float(np.median(nearest_dists)),
            "min_mm": float(np.min(nearest_dists)),
            "max_mm": float(np.max(nearest_dists)),
            "per_swap_node_mm": nearest_dists,
        }
        out["density_by_radius"] = {
            k: v / max(len(swap_idx), 1) for k, v in density_counts.items()
        }

    if soz_channels:
        soz_idx = [name_to_idx[nm] for nm in soz_channels if nm in name_to_idx and coord_finite[name_to_idx[nm]]]
        if soz_idx:
            soz_pts = coords[soz_idx]
            soz_in_swap = sum(1 for s in soz_idx if s in swap_set)
            # Pairwise distance from each swap to each soz
            sw_to_soz = np.linalg.norm(swap_pts[:, None, :] - soz_pts[None, :, :], axis=-1)
            out["soz_relation"] = {
                "n_soz_total": len(soz_channels),
                "n_soz_with_coords": len(soz_idx),
                "n_swap_in_soz_list": soz_in_swap,
                "min_dist_swap_to_soz_mm": float(sw_to_soz.min()),
                "median_dist_swap_to_soz_mm": float(np.median(sw_to_soz)),
                "max_dist_swap_to_soz_mm": float(sw_to_soz.max()),
            }
    return out
