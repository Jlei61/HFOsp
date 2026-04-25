from __future__ import annotations

import datetime
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from zoneinfo import ZoneInfo
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest, kendalltau, spearmanr, wilcoxon
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

try:
    from diptest import diptest as hartigan_diptest
except Exception:  # pragma: no cover - optional import guard
    hartigan_diptest = None


def _record_name_from_lagpat_path(path: Path) -> str:
    name = path.name
    if name.endswith("_lagPat.npz"):
        return name[: -len("_lagPat.npz")]
    return path.stem


def _safe_float_scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.reshape(-1)[0])


def _finite_minmax(values: np.ndarray) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.min(finite)), float(np.max(finite))


def _compute_relative_lag_matrix(lag_raw: np.ndarray, bools: np.ndarray) -> np.ndarray:
    """Convert stitched centroid times into per-event relative lag."""
    lag_raw = np.asarray(lag_raw, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    if lag_raw.shape != bools.shape:
        raise ValueError("lag_raw and bools must have the same shape")

    rel = np.full(lag_raw.shape, np.nan, dtype=float)
    for ev in range(lag_raw.shape[1]):
        mask = bools[:, ev] & np.isfinite(lag_raw[:, ev])
        if not np.any(mask):
            continue
        rel[mask, ev] = lag_raw[mask, ev] - float(np.min(lag_raw[mask, ev]))
    return rel


def _event_order_alignment_summary(
    ranks: np.ndarray,
    relative_lag: np.ndarray,
    bools: np.ndarray,
    *,
    min_shared_channels: int = 3,
) -> Dict[str, Any]:
    """Check that relative lag preserves the within-event participating order."""
    ranks = np.asarray(ranks, dtype=float)
    relative_lag = np.asarray(relative_lag, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    if ranks.shape != relative_lag.shape or ranks.shape != bools.shape:
        raise ValueError("ranks, relative_lag, and bools must share shape")

    exact_match: List[float] = []
    pairwise_concordance: List[float] = []

    for ev in range(ranks.shape[1]):
        mask = bools[:, ev] & np.isfinite(relative_lag[:, ev]) & np.isfinite(ranks[:, ev])
        if int(np.sum(mask)) < int(min_shared_channels):
            continue

        rel_ev = relative_lag[mask, ev]
        rank_ev = ranks[mask, ev]
        exact_match.append(
            float(
                np.array_equal(
                    np.argsort(rel_ev, kind="mergesort"),
                    np.argsort(rank_ev, kind="mergesort"),
                )
            )
        )

        total_pairs = 0
        agree_pairs = 0
        for i in range(rel_ev.size):
            for j in range(i + 1, rel_ev.size):
                rel_delta = float(rel_ev[i] - rel_ev[j])
                rank_delta = float(rank_ev[i] - rank_ev[j])
                if abs(rel_delta) <= 1e-12 or abs(rank_delta) <= 1e-12:
                    continue
                total_pairs += 1
                if np.sign(rel_delta) == np.sign(rank_delta):
                    agree_pairs += 1
        if total_pairs > 0:
            pairwise_concordance.append(float(agree_pairs / total_pairs))

    participating_rel = relative_lag[bools & np.isfinite(relative_lag)]
    return {
        "n_events_checked": int(len(exact_match)),
        "exact_order_match_fraction": float(np.mean(exact_match)) if exact_match else np.nan,
        "pairwise_order_concordance": (
            float(np.mean(pairwise_concordance)) if pairwise_concordance else np.nan
        ),
        "nonnegative_fraction": (
            float(np.mean(participating_rel >= -1e-12)) if participating_rel.size else np.nan
        ),
        "relative_lag_min": (
            float(np.min(participating_rel)) if participating_rel.size else np.nan
        ),
        "relative_lag_max": (
            float(np.max(participating_rel)) if participating_rel.size else np.nan
        ),
    }


def _pairwise_relative_lag_correlations(
    relative_lag: np.ndarray,
    bools: np.ndarray,
    event_indices: np.ndarray,
    *,
    min_shared_channels: int,
) -> Dict[str, Any]:
    """Within-subset pairwise Pearson r on shared participating channels."""
    relative_lag = np.asarray(relative_lag, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    event_indices = np.asarray(event_indices, dtype=int)

    pearson_r_values: List[float] = []
    n_pairs_considered = 0
    n_pairs_valid = 0

    for i in range(event_indices.size):
        ev_i = int(event_indices[i])
        for j in range(i + 1, event_indices.size):
            ev_j = int(event_indices[j])
            n_pairs_considered += 1
            mask = (
                bools[:, ev_i]
                & bools[:, ev_j]
                & np.isfinite(relative_lag[:, ev_i])
                & np.isfinite(relative_lag[:, ev_j])
            )
            if int(np.sum(mask)) < int(min_shared_channels):
                continue
            x = relative_lag[mask, ev_i]
            y = relative_lag[mask, ev_j]
            if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(r):
                continue
            pearson_r_values.append(r)
            n_pairs_valid += 1

    return {
        "n_pairs_considered": int(n_pairs_considered),
        "n_pairs_valid": int(n_pairs_valid),
        "median_r": float(np.median(pearson_r_values)) if pearson_r_values else np.nan,
        "mean_r": float(np.mean(pearson_r_values)) if pearson_r_values else np.nan,
        "pearson_r_values": pearson_r_values,
    }


def _event_lag_span_summary(
    relative_lag: np.ndarray,
    bools: np.ndarray,
    event_indices: np.ndarray,
    *,
    min_participating: int,
) -> Dict[str, Any]:
    """Per-event lag span summary on relative lag vectors."""
    relative_lag = np.asarray(relative_lag, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    event_indices = np.asarray(event_indices, dtype=int)

    lag_spans: List[float] = []
    n_participating_values: List[int] = []
    for ev in event_indices:
        mask = bools[:, ev] & np.isfinite(relative_lag[:, ev])
        n_part = int(np.sum(mask))
        if n_part < int(min_participating):
            continue
        values = relative_lag[mask, ev]
        lag_spans.append(float(np.max(values) - np.min(values)))
        n_participating_values.append(n_part)

    return {
        "n_events": int(len(lag_spans)),
        "mean_lag_span": float(np.mean(lag_spans)) if lag_spans else np.nan,
        "median_lag_span": float(np.median(lag_spans)) if lag_spans else np.nan,
        "min_lag_span": float(np.min(lag_spans)) if lag_spans else np.nan,
        "max_lag_span": float(np.max(lag_spans)) if lag_spans else np.nan,
        "median_n_participating": (
            float(np.median(n_participating_values)) if n_participating_values else np.nan
        ),
    }


def _match_event_indices_by_nparticipating(
    high_indices: np.ndarray,
    low_indices: np.ndarray,
    n_participating: np.ndarray,
    *,
    seed: int,
    min_participating: int = 3,
) -> Dict[str, Any]:
    """Exact-match high/low event subsets by n_participating."""
    high_indices = np.asarray(high_indices, dtype=int)
    low_indices = np.asarray(low_indices, dtype=int)
    n_participating = np.asarray(n_participating, dtype=int)
    rng = np.random.default_rng(int(seed))

    high_indices = high_indices[n_participating[high_indices] >= int(min_participating)]
    low_indices = low_indices[n_participating[low_indices] >= int(min_participating)]

    matched_high: List[int] = []
    matched_low: List[int] = []
    matched_counts: Dict[str, int] = {}

    high_values = {int(v) for v in n_participating[high_indices].tolist()}
    low_values = {int(v) for v in n_participating[low_indices].tolist()}
    shared_values = sorted(high_values & low_values)
    for value in shared_values:
        high_local = high_indices[n_participating[high_indices] == value]
        low_local = low_indices[n_participating[low_indices] == value]
        n_match = int(min(high_local.size, low_local.size))
        if n_match <= 0:
            continue
        if high_local.size > n_match:
            high_local = rng.choice(high_local, size=n_match, replace=False)
        if low_local.size > n_match:
            low_local = rng.choice(low_local, size=n_match, replace=False)
        matched_high.extend(int(x) for x in np.sort(high_local))
        matched_low.extend(int(x) for x in np.sort(low_local))
        matched_counts[str(int(value))] = n_match

    return {
        "high_event_indices": np.asarray(sorted(matched_high), dtype=int),
        "low_event_indices": np.asarray(sorted(matched_low), dtype=int),
        "n_matched": int(len(matched_high)),
        "matched_counts_by_n_participating": matched_counts,
        "min_participating": int(min_participating),
    }


def _cluster_state_l3_summary(
    relative_lag: np.ndarray,
    bools: np.ndarray,
    *,
    lag_span_event_indices: np.ndarray,
    pearson_event_indices: np.ndarray,
    min_lag_span_participating: int,
    min_shared_channels_for_r: int,
    n_sample_pearson: int = 200,
    pearson_seed: int = 0,
) -> Dict[str, Any]:
    """L3 summary for one state within one cluster."""
    lag_span = _event_lag_span_summary(
        relative_lag,
        bools,
        lag_span_event_indices,
        min_participating=min_lag_span_participating,
    )
    sampled_pearson = _sample_event_indices(
        np.asarray(pearson_event_indices, dtype=int),
        n_sample=n_sample_pearson,
        seed=pearson_seed,
    )
    pearson = _pairwise_relative_lag_correlations(
        relative_lag,
        bools,
        sampled_pearson,
        min_shared_channels=min_shared_channels_for_r,
    )
    return {
        "lag_span_n_events": int(lag_span["n_events"]),
        "lag_span_mean": lag_span["mean_lag_span"],
        "lag_span_median": lag_span["median_lag_span"],
        "lag_span_min": lag_span["min_lag_span"],
        "lag_span_max": lag_span["max_lag_span"],
        "lag_span_median_n_participating": lag_span["median_n_participating"],
        "pearson_r_n_events": int(np.asarray(pearson_event_indices, dtype=int).size),
        "pearson_r_n_events_sampled": int(sampled_pearson.size),
        "pearson_r_n_pairs_valid": int(pearson["n_pairs_valid"]),
        "pearson_r_median": pearson["median_r"],
        "pearson_r_mean": pearson["mean_r"],
    }


def load_subject_propagation_events(subject_dir: Path) -> Dict[str, Any]:
    """Load one subject into a single event contract with timing metadata.

    Blocks are ordered by `start_t` first, then filename as a stable tie-breaker.
    Event absolute time is reconstructed as `packedTimes[:, 0] + start_t`.
    """
    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        raise FileNotFoundError(f"No *_lagPat.npz in {subject_dir}")

    raw_blocks: List[Dict[str, Any]] = []
    channel_names: List[str] = []
    channel_index: Dict[str, int] = {}

    for lp_file in lagpat_files:
        lp = np.load(lp_file, allow_pickle=True)
        ranks = np.asarray(lp["lagPatRank"], dtype=float)
        lag_raw = (
            np.asarray(lp["lagPatRaw"], dtype=float)
            if "lagPatRaw" in lp
            else np.full(ranks.shape, np.nan, dtype=float)
        )
        bools = np.asarray(lp["eventsBool"]) > 0
        chns = [str(x) for x in list(lp["chnNames"])]
        start_t = _safe_float_scalar(lp["start_t"]) if "start_t" in lp else float("nan")
        if ranks.ndim != 2 or lag_raw.ndim != 2 or bools.ndim != 2:
            continue
        if ranks.size == 0 or bools.size == 0 or lag_raw.size == 0:
            continue

        n_ev = min(ranks.shape[1], lag_raw.shape[1], bools.shape[1])
        n_ch = min(ranks.shape[0], lag_raw.shape[0], bools.shape[0], len(chns))
        if n_ev == 0 or n_ch == 0:
            continue

        ranks = ranks[:n_ch, :n_ev]
        lag_raw = lag_raw[:n_ch, :n_ev]
        bools = bools[:n_ch, :n_ev]
        chns = chns[:n_ch]

        for ch in chns:
            if ch not in channel_index:
                channel_index[ch] = len(channel_names)
                channel_names.append(ch)

        record_name = _record_name_from_lagpat_path(lp_file)
        packed_path = lp_file.with_name(f"{record_name}_packedTimes.npy")
        packed = None
        event_rel_times = np.full(n_ev, np.nan, dtype=float)
        event_abs_times = np.full(n_ev, np.nan, dtype=float)
        event_rel_end_times = np.full(n_ev, np.nan, dtype=float)
        event_abs_end_times = np.full(n_ev, np.nan, dtype=float)
        has_packed_times = packed_path.exists()
        if has_packed_times:
            packed = np.asarray(np.load(packed_path), dtype=float)
            if packed.ndim == 2 and packed.shape[1] >= 1:
                n_ev = min(n_ev, packed.shape[0])
                ranks = ranks[:, :n_ev]
                bools = bools[:, :n_ev]
                event_rel_times = packed[:n_ev, 0].astype(float, copy=False)
                if packed.shape[1] >= 2:
                    event_rel_end_times = packed[:n_ev, 1].astype(float, copy=False)
                else:
                    event_rel_end_times = event_rel_times.copy()
                if np.isfinite(start_t):
                    event_abs_times = event_rel_times + start_t
                    event_abs_end_times = event_rel_end_times + start_t

        raw_blocks.append(
            {
                "path": str(lp_file),
                "record_name": record_name,
                "start_t": start_t,
                "channel_names": chns,
                "ranks": ranks,
                "lag_raw": lag_raw,
                "bools": bools,
                "event_rel_times": event_rel_times,
                "event_abs_times": event_abs_times,
                "event_rel_end_times": event_rel_end_times,
                "event_abs_end_times": event_abs_end_times,
                "has_packed_times": has_packed_times,
            }
        )

    if not raw_blocks:
        return {
            "ranks": np.zeros((0, 0), dtype=float),
            "lag_raw": np.zeros((0, 0), dtype=float),
            "bools": np.zeros((0, 0), dtype=bool),
            "channel_names": [],
            "event_abs_times": np.zeros(0, dtype=float),
            "event_rel_times": np.zeros(0, dtype=float),
            "block_ids": np.zeros(0, dtype=int),
            "record_names": [],
            "block_boundaries": [],
            "block_start_times": np.zeros(0, dtype=float),
            "block_time_ranges": [],
            "n_blocks_total": 0,
            "n_blocks_used": 0,
        }

    raw_blocks = sorted(
        raw_blocks,
        key=lambda block: (
            block["start_t"] if np.isfinite(block["start_t"]) else float("inf"),
            os.path.basename(block["path"]),
        ),
    )

    n_union = len(channel_names)
    aligned_ranks: List[np.ndarray] = []
    aligned_lag_raw: List[np.ndarray] = []
    aligned_bools: List[np.ndarray] = []
    aligned_event_abs_times: List[np.ndarray] = []
    aligned_event_rel_times: List[np.ndarray] = []
    block_ids: List[np.ndarray] = []
    record_names: List[str] = []
    block_boundaries: List[Dict[str, Any]] = []
    block_start_times: List[float] = []
    block_time_ranges: List[Tuple[float, float]] = []
    cursor = 0

    for block_id, block in enumerate(raw_blocks):
        chns = block["channel_names"]
        ranks = block["ranks"]
        lag_raw = block["lag_raw"]
        bools = block["bools"]
        n_ev = ranks.shape[1]
        rank_block = np.zeros((n_union, n_ev), dtype=float)
        lag_raw_block = np.full((n_union, n_ev), np.nan, dtype=float)
        bool_block = np.zeros((n_union, n_ev), dtype=bool)
        for row, ch in enumerate(chns):
            idx = channel_index[ch]
            rank_block[idx, :] = ranks[row, :]
            lag_raw_block[idx, :] = lag_raw[row, :]
            bool_block[idx, :] = bools[row, :]
        aligned_ranks.append(rank_block)
        aligned_lag_raw.append(lag_raw_block)
        aligned_bools.append(bool_block)
        aligned_event_abs_times.append(np.asarray(block["event_abs_times"], dtype=float))
        aligned_event_rel_times.append(np.asarray(block["event_rel_times"], dtype=float))
        block_ids.append(np.full(n_ev, block_id, dtype=int))
        record_names.append(str(block["record_name"]))
        block_start_times.append(float(block["start_t"]))
        block_start_epoch, block_end_epoch = _finite_minmax(
            np.array(
                [
                    np.nanmin(block["event_abs_times"]) if np.any(np.isfinite(block["event_abs_times"])) else np.nan,
                    np.nanmax(block["event_abs_end_times"]) if np.any(np.isfinite(block["event_abs_end_times"])) else np.nan,
                ],
                dtype=float,
            )
        )
        block_time_ranges.append((block_start_epoch, block_end_epoch))
        block_boundaries.append(
            {
                "block_id": int(block_id),
                "record_name": str(block["record_name"]),
                "start_event_idx": int(cursor),
                "end_event_idx": int(cursor + n_ev),
                "n_events": int(n_ev),
                "start_t": float(block["start_t"]),
                "has_packed_times": bool(block["has_packed_times"]),
                "block_start_epoch": block_start_epoch,
                "block_end_epoch": block_end_epoch,
            }
        )
        cursor += n_ev

    return {
        "ranks": np.concatenate(aligned_ranks, axis=1),
        "lag_raw": np.concatenate(aligned_lag_raw, axis=1),
        "bools": np.concatenate(aligned_bools, axis=1),
        "channel_names": channel_names,
        "event_abs_times": np.concatenate(aligned_event_abs_times, axis=0),
        "event_rel_times": np.concatenate(aligned_event_rel_times, axis=0),
        "block_ids": np.concatenate(block_ids, axis=0),
        "record_names": record_names,
        "block_boundaries": block_boundaries,
        "block_start_times": np.asarray(block_start_times, dtype=float),
        "block_time_ranges": block_time_ranges,
        "n_blocks_total": len(lagpat_files),
        "n_blocks_used": len(raw_blocks),
    }


def load_subject_propagation_patterns(subject_dir: Path) -> Dict[str, Any]:
    """Backward-compatible wrapper for aligned propagation matrices."""
    return load_subject_propagation_events(subject_dir)


def label_events_by_soz(
    bools: np.ndarray,
    channel_names: Sequence[str],
    soz_channels: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    soz_set = set(soz_channels or [])
    channel_names = list(channel_names)
    soz_mask = np.array([ch in soz_set for ch in channel_names], dtype=bool)
    if bools.size == 0:
        return {
            "is_soz": np.zeros(0, dtype=bool),
            "n_soz_channels_input": len(soz_set),
            "n_soz_channels_matched": int(np.sum(soz_mask)),
        }
    is_soz = np.any(bools[soz_mask, :], axis=0) if np.any(soz_mask) else np.zeros(bools.shape[1], dtype=bool)
    out = {
        "is_soz": is_soz,
        "n_soz_channels_input": len(soz_set),
        "n_soz_channels_matched": int(np.sum(soz_mask)),
    }
    if soz_set and not np.any(soz_mask):
        out["warning"] = "no_soz_channel_match"
    return out


def _valid_event_indices(bools: np.ndarray, min_participating: int = 3) -> np.ndarray:
    if bools.size == 0:
        return np.zeros(0, dtype=int)
    n_part = np.sum(bools > 0, axis=0)
    return np.where(n_part >= min_participating)[0]


def _sample_event_indices(event_indices: np.ndarray, n_sample: int, seed: int) -> np.ndarray:
    event_indices = np.asarray(event_indices, dtype=int)
    if event_indices.size <= n_sample:
        return np.sort(event_indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(event_indices, size=n_sample, replace=False))


def _pairwise_tau_summary(
    ranks: np.ndarray,
    bools: np.ndarray,
    event_indices: np.ndarray,
    n_sample: int = 200,
    seed: int = 0,
    min_shared_channels: int = 3,
    channel_mask: Optional[np.ndarray] = None,
    return_matrix: bool = False,
) -> Dict[str, Any]:
    sampled = _sample_event_indices(event_indices, n_sample=n_sample, seed=seed)
    n_sampled = int(sampled.size)
    if n_sampled < 2:
        return {
            "event_indices": sampled,
            "tau_values": np.zeros(0, dtype=float),
            "mean_tau": np.nan,
            "n_pairs_total": 0,
            "n_pairs_valid": 0,
            "tau_matrix": np.full((n_sampled, n_sampled), np.nan, dtype=float),
        }

    tau_values: List[float] = []
    tau_matrix = np.full((n_sampled, n_sampled), np.nan, dtype=float)
    np.fill_diagonal(tau_matrix, 1.0)

    for i in range(n_sampled):
        ei = sampled[i]
        for j in range(i + 1, n_sampled):
            ej = sampled[j]
            shared = (bools[:, ei] > 0) & (bools[:, ej] > 0)
            if channel_mask is not None:
                shared &= channel_mask
            if int(np.sum(shared)) < min_shared_channels:
                continue

            x = np.asarray(ranks[shared, ei], dtype=float)
            y = np.asarray(ranks[shared, ej], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if int(np.sum(finite)) < min_shared_channels:
                continue

            tau, _ = kendalltau(x[finite], y[finite])
            if not np.isfinite(tau):
                continue
            tau = float(tau)
            tau_values.append(tau)
            tau_matrix[i, j] = tau
            tau_matrix[j, i] = tau

    mean_tau = float(np.mean(tau_values)) if tau_values else np.nan
    if return_matrix and np.isnan(mean_tau):
        tau_matrix = np.where(np.isfinite(tau_matrix), tau_matrix, 0.0)
    elif return_matrix:
        tau_matrix = np.where(np.isfinite(tau_matrix), tau_matrix, mean_tau)

    return {
        "event_indices": sampled,
        "tau_values": np.asarray(tau_values, dtype=float),
        "mean_tau": mean_tau,
        "n_pairs_total": int(n_sampled * (n_sampled - 1) // 2),
        "n_pairs_valid": int(len(tau_values)),
        "tau_matrix": tau_matrix if return_matrix else None,
    }


def _multi_seed_tau_summary(
    ranks: np.ndarray,
    bools: np.ndarray,
    event_indices: np.ndarray,
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
    channel_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    seeds = list(range(n_seeds))
    tau_samples: List[float] = []
    n_pairs_valid: List[int] = []
    n_events_sampled: List[int] = []

    for seed in seeds:
        out = _pairwise_tau_summary(
            ranks,
            bools,
            event_indices=event_indices,
            n_sample=n_sample,
            seed=seed,
            min_shared_channels=min_shared_channels,
            channel_mask=channel_mask,
            return_matrix=False,
        )
        tau_samples.append(float(out["mean_tau"]) if np.isfinite(out["mean_tau"]) else np.nan)
        n_pairs_valid.append(int(out["n_pairs_valid"]))
        n_events_sampled.append(int(out["event_indices"].size))

    tau_arr = np.asarray(tau_samples, dtype=float)
    finite = np.isfinite(tau_arr)
    if np.any(finite):
        mean_tau = float(np.nanmean(tau_arr))
        tau_sd = float(np.nanstd(tau_arr, ddof=1)) if int(np.sum(finite)) > 1 else 0.0
        tau_ci_lo = float(np.nanpercentile(tau_arr[finite], 2.5))
        tau_ci_hi = float(np.nanpercentile(tau_arr[finite], 97.5))
    else:
        mean_tau = np.nan
        tau_sd = np.nan
        tau_ci_lo = np.nan
        tau_ci_hi = np.nan

    return {
        "mean_tau": mean_tau,
        "tau_sd": tau_sd,
        "tau_ci_lo": tau_ci_lo,
        "tau_ci_hi": tau_ci_hi,
        "tau_samples": tau_samples,
        "n_events_available": int(np.asarray(event_indices).size),
        "n_events_sampled": int(np.nanmedian(n_events_sampled)) if n_events_sampled else 0,
        "n_pairs_valid_median": int(np.nanmedian(n_pairs_valid)) if n_pairs_valid else 0,
    }


def compute_pairwise_tau_values(
    ranks: np.ndarray,
    bools: np.ndarray,
    n_sample: int = 300,
    seed: int = 0,
    min_shared_channels: int = 3,
) -> np.ndarray:
    """Return flat array of sampled pairwise Kendall tau values.

    Intended for bimodality visualization — lightweight enough
    for a single subject at plot time.
    """
    valid_events = _valid_event_indices(bools, min_participating=min_shared_channels)
    out = _pairwise_tau_summary(
        ranks, bools, event_indices=valid_events,
        n_sample=n_sample, seed=seed,
        min_shared_channels=min_shared_channels,
    )
    return np.asarray(out["tau_values"], dtype=float)


def _center_rank_matrix(
    ranks: np.ndarray,
    bools: np.ndarray,
    min_participation: int = 10,
) -> Dict[str, Any]:
    n_ch = ranks.shape[0]
    counts = np.sum(bools > 0, axis=1).astype(int)
    valid_center = counts >= min_participation
    mean_rank = np.zeros(n_ch, dtype=float)

    for ch in range(n_ch):
        if not valid_center[ch]:
            continue
        vals = np.asarray(ranks[ch, bools[ch] > 0], dtype=float)
        vals = vals[np.isfinite(vals)]
        mean_rank[ch] = float(np.mean(vals)) if vals.size else 0.0

    centered = np.asarray(ranks, dtype=float).copy()
    centered[valid_center, :] = centered[valid_center, :] - mean_rank[valid_center, None]
    return {
        "centered_ranks": centered,
        "mean_rank_per_channel": mean_rank,
        "valid_center_mask": valid_center,
        "participation_count_per_channel": counts,
    }


def detect_propagation_mixture(
    ranks: np.ndarray,
    bools: np.ndarray,
    n_sample: int = 200,
    seed: int = 0,
    min_shared_channels: int = 3,
    silhouette_threshold: float = 0.15,
) -> Dict[str, Any]:
    valid_events = _valid_event_indices(bools, min_participating=min_shared_channels)
    tau_out = _pairwise_tau_summary(
        ranks,
        bools,
        event_indices=valid_events,
        n_sample=n_sample,
        seed=seed,
        min_shared_channels=min_shared_channels,
        return_matrix=True,
    )
    tau_values = tau_out["tau_values"]
    tau_matrix = tau_out["tau_matrix"]

    dip_stat = np.nan
    dip_p = np.nan
    is_mixture = False
    if tau_values.size >= 20 and hartigan_diptest is not None:
        dip_stat, dip_p = hartigan_diptest(np.asarray(tau_values, dtype=float))
        dip_stat = float(dip_stat)
        dip_p = float(dip_p)
        is_mixture = dip_p < 0.05

    silhouette_k2 = np.nan
    possible_mixture = False
    if tau_matrix.shape[0] >= 6:
        distance = 1.0 - np.asarray(tau_matrix, dtype=float)
        distance = np.clip(distance, 0.0, None)
        try:
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=2, metric="precomputed", linkage="average"
                )
            except TypeError:  # pragma: no cover - sklearn compatibility
                clustering = AgglomerativeClustering(
                    n_clusters=2, affinity="precomputed", linkage="average"
                )
            labels = clustering.fit_predict(distance)
            if len(np.unique(labels)) == 2:
                silhouette_k2 = float(
                    silhouette_score(distance, labels, metric="precomputed")
                )
                possible_mixture = (not is_mixture) and (silhouette_k2 >= silhouette_threshold)
        except Exception:
            silhouette_k2 = np.nan

    out = {
        "mean_tau": tau_out["mean_tau"],
        "n_events_available": int(valid_events.size),
        "n_events_sampled": int(tau_out["event_indices"].size),
        "n_pairs_valid": int(tau_out["n_pairs_valid"]),
        "dip_stat": dip_stat,
        "dip_p": dip_p,
        "is_mixture": bool(is_mixture),
        "silhouette_k2": silhouette_k2,
        "possible_mixture": bool(possible_mixture),
    }
    if tau_values.size < 20:
        out["warning"] = "insufficient_pairs_for_diptest"
    elif hartigan_diptest is None:
        out["warning"] = "diptest_unavailable"
    return out


def compute_centered_rank_tau(
    ranks: np.ndarray,
    bools: np.ndarray,
    n_sample: int = 200,
    min_participation: int = 10,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
) -> Dict[str, Any]:
    valid_events = _valid_event_indices(bools, min_participating=min_shared_channels)
    raw = _multi_seed_tau_summary(
        ranks,
        bools,
        event_indices=valid_events,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )
    centered_meta = _center_rank_matrix(
        ranks,
        bools,
        min_participation=min_participation,
    )
    centered = _multi_seed_tau_summary(
        centered_meta["centered_ranks"],
        bools,
        event_indices=valid_events,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
        channel_mask=centered_meta["valid_center_mask"],
    )
    raw_tau = raw["mean_tau"]
    centered_tau = centered["mean_tau"]
    bias_fraction = np.nan
    if np.isfinite(raw_tau) and abs(raw_tau) > 1e-12 and np.isfinite(centered_tau):
        bias_fraction = float((raw_tau - centered_tau) / raw_tau)

    return {
        "raw_tau": raw_tau,
        "raw_tau_ci_lo": raw["tau_ci_lo"],
        "raw_tau_ci_hi": raw["tau_ci_hi"],
        "centered_tau": centered_tau,
        "centered_tau_ci_lo": centered["tau_ci_lo"],
        "centered_tau_ci_hi": centered["tau_ci_hi"],
        "bias_fraction": bias_fraction,
        "n_valid_center_channels": int(np.sum(centered_meta["valid_center_mask"])),
        "mean_rank_per_channel": centered_meta["mean_rank_per_channel"].tolist(),
        "participation_count_per_channel": centered_meta["participation_count_per_channel"].tolist(),
    }


def compute_stereotypy_by_nparticipating(
    ranks: np.ndarray,
    bools: np.ndarray,
    bins: List[Tuple[int, int]] = [(3, 4), (5, 8), (9, 999)],
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
) -> List[Dict[str, Any]]:
    n_part = np.sum(bools > 0, axis=0).astype(int) if bools.size else np.zeros(0, dtype=int)
    out: List[Dict[str, Any]] = []
    for lo, hi in bins:
        event_indices = np.where((n_part >= lo) & (n_part <= hi))[0]
        summary = _multi_seed_tau_summary(
            ranks,
            bools,
            event_indices=event_indices,
            n_sample=n_sample,
            n_seeds=n_seeds,
            min_shared_channels=min_shared_channels,
        )
        out.append(
            {
                "bin_label": f"{lo}-{hi if hi < 999 else 'plus'}",
                "n_participating_lo": int(lo),
                "n_participating_hi": int(hi),
                "n_events": int(event_indices.size),
                "mean_tau": summary["mean_tau"],
                "tau_ci_lo": summary["tau_ci_lo"],
                "tau_ci_hi": summary["tau_ci_hi"],
                "n_pairs_valid_median": summary["n_pairs_valid_median"],
            }
        )
    return out


def compute_source_node_diagnostic(
    ranks: np.ndarray,
    bools: np.ndarray,
    channel_names: Sequence[str],
    soz_channels: Optional[Sequence[str]] = None,
    min_participation: int = 10,
) -> Dict[str, Any]:
    channel_names = list(channel_names)
    centered = _center_rank_matrix(ranks, bools, min_participation=min_participation)
    counts = np.sum(bools > 0, axis=1).astype(int)
    valid_channels = np.where(counts >= min_participation)[0]

    def _source_scores(mat: np.ndarray) -> np.ndarray:
        scores = np.full(mat.shape[0], np.nan, dtype=float)
        for i in valid_channels:
            diffs = []
            for j in valid_channels:
                if i == j:
                    continue
                shared = (bools[i] > 0) & (bools[j] > 0)
                if int(np.sum(shared)) < 3:
                    continue
                delta = np.asarray(mat[i, shared] - mat[j, shared], dtype=float)
                delta = delta[np.isfinite(delta)]
                if delta.size:
                    diffs.append(float(np.mean(delta)))
            if diffs:
                # Lower score means this channel tends to lead others.
                scores[i] = float(np.mean(diffs))
        return scores

    raw_scores = _source_scores(ranks)
    centered_scores = _source_scores(centered["centered_ranks"])

    valid_raw = np.where(np.isfinite(raw_scores))[0]
    valid_centered = np.where(np.isfinite(centered_scores))[0]
    raw_top = [channel_names[i] for i in valid_raw[np.argsort(raw_scores[valid_raw])][:3]]
    centered_top = [
        channel_names[i] for i in valid_centered[np.argsort(centered_scores[valid_centered])][:3]
    ]

    soz_set = set(soz_channels or [])
    raw_soz_top = [ch for ch in raw_top if ch in soz_set]
    centered_soz_top = [ch for ch in centered_top if ch in soz_set]

    return {
        "raw_top3_sources": raw_top,
        "centered_top3_sources": centered_top,
        "raw_top3_soz_sources": raw_soz_top,
        "centered_top3_soz_sources": centered_soz_top,
        "soz_source_erased": bool(raw_soz_top and not centered_soz_top),
    }


def compute_propagation_stereotypy(
    subject_dir: Path,
    dataset: str,
    soz_channels: Optional[List[str]] = None,
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
) -> Dict[str, Any]:
    loaded = load_subject_propagation_patterns(subject_dir)
    ranks = loaded["ranks"]
    bools = loaded["bools"]
    channel_names = loaded["channel_names"]
    valid_events = _valid_event_indices(bools, min_participating=min_shared_channels)
    soz_meta = label_events_by_soz(bools, channel_names, soz_channels=soz_channels)
    is_soz = soz_meta["is_soz"]

    result: Dict[str, Any] = {
        "dataset": dataset,
        "n_channels": int(ranks.shape[0]),
        "n_events_total": int(ranks.shape[1]),
        "n_events_valid": int(valid_events.size),
        "n_blocks_used": int(loaded["n_blocks_used"]),
        "n_soz_channels_input": int(soz_meta["n_soz_channels_input"]),
        "n_soz_channels_matched": int(soz_meta["n_soz_channels_matched"]),
    }
    if "warning" in soz_meta:
        result["warning"] = soz_meta["warning"]

    result["all"] = _multi_seed_tau_summary(
        ranks,
        bools,
        event_indices=valid_events,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )

    soz_events = valid_events[is_soz[valid_events]] if valid_events.size else np.zeros(0, dtype=int)
    nonsoz_events = valid_events[~is_soz[valid_events]] if valid_events.size else np.zeros(0, dtype=int)
    result["soz"] = _multi_seed_tau_summary(
        ranks,
        bools,
        event_indices=soz_events,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )
    result["nonsoz"] = _multi_seed_tau_summary(
        ranks,
        bools,
        event_indices=nonsoz_events,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )
    return result


def run_subject_interictal_propagation_pr1(
    subject_dir: Path,
    dataset: str,
    subject: str,
    soz_channels: Optional[List[str]] = None,
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
    min_center_participation: int = 10,
) -> Dict[str, Any]:
    loaded = load_subject_propagation_events(subject_dir)
    ranks = loaded["ranks"]
    bools = loaded["bools"]
    channel_names = loaded["channel_names"]

    if ranks.size == 0 or bools.size == 0:
        return {
            "dataset": dataset,
            "subject": subject,
            "error": "no_propagation_data",
        }

    stereotypy = compute_propagation_stereotypy(
        subject_dir=subject_dir,
        dataset=dataset,
        soz_channels=soz_channels,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )
    mixture = detect_propagation_mixture(
        ranks,
        bools,
        n_sample=n_sample,
        seed=0,
        min_shared_channels=min_shared_channels,
    )
    centered = compute_centered_rank_tau(
        ranks,
        bools,
        n_sample=n_sample,
        min_participation=min_center_participation,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )
    by_n = compute_stereotypy_by_nparticipating(
        ranks,
        bools,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )
    source_diag = compute_source_node_diagnostic(
        ranks,
        bools,
        channel_names=channel_names,
        soz_channels=soz_channels,
        min_participation=min_center_participation,
    )
    cluster_result = compute_cluster_stereotypy(
        ranks,
        bools,
        channel_names=channel_names,
        n_clusters=2,
        n_sample=n_sample,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
        min_participation=min_center_participation,
    )
    mi_result = compute_legacy_mi(ranks, bools, n_permutations=200, seed=0)
    adaptive_cluster = compute_adaptive_cluster_stereotypy(
        ranks,
        bools,
        channel_names=channel_names,
        n_sample=n_sample,
        n_tau_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
        min_participation=min_center_participation,
    )

    valid_ev = _valid_event_indices(bools, min_participating=min_shared_channels)
    ada_labels = np.array(adaptive_cluster.get("labels", []), dtype=int)
    if ada_labels.size == valid_ev.size and ada_labels.size > 0:
        within_cluster_bias = compute_within_cluster_centered_tau(
            ranks, bools, ada_labels, valid_ev,
            n_sample=n_sample, n_seeds=n_seeds,
            min_shared_channels=min_shared_channels,
            min_participation=min_center_participation,
        )
    else:
        within_cluster_bias = {"error": "labels_size_mismatch"}

    first_event_abs_time, last_event_abs_time = _finite_minmax(loaded["event_abs_times"])

    return {
        "dataset": dataset,
        "subject": subject,
        "n_channels": int(ranks.shape[0]),
        "n_events_total": int(ranks.shape[1]),
        "n_blocks_used": int(loaded["n_blocks_used"]),
        "channel_names": channel_names,
        "event_metadata": {
            "record_names": loaded["record_names"],
            "block_boundaries": loaded["block_boundaries"],
            "block_start_times": loaded["block_start_times"].tolist(),
            "first_event_abs_time": first_event_abs_time,
            "last_event_abs_time": last_event_abs_time,
        },
        "propagation_stereotypy": stereotypy,
        "mixture": mixture,
        "centered_rank": centered,
        "by_nparticipating": by_n,
        "source_diagnostic": source_diag,
        "cluster": cluster_result,
        "adaptive_cluster": adaptive_cluster,
        "within_cluster_centered": within_cluster_bias,
        "legacy_mi": mi_result,
    }


def _legacy_hist_mean_rank(ranks: np.ndarray, bools: np.ndarray) -> np.ndarray:
    """Sliding-3-bin weighted average rank template (legacy MI algorithm).

    For each channel, build a histogram of its ranks across participating events,
    find the 3 consecutive bins with the largest total count, and compute the
    count-weighted average bin index. This is the legacy template used for MI.
    """
    n_ch = ranks.shape[0]
    template = np.zeros(n_ch, dtype=float)
    for ci in range(n_ch):
        vals = ranks[ci, bools[ci] > 0] if bools.size else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            template[ci] = ci
            continue
        hist, _ = np.histogram(vals, bins=np.arange(0, n_ch + 1) - 0.5)
        if n_ch > 4:
            best_sum = 0
            best_mean = float(np.mean(vals))
            for i in range(max(1, n_ch - 2)):
                s = int(np.sum(hist[i : i + 3]))
                if s > best_sum:
                    best_sum = s
                    h3 = hist[i : i + 3].astype(float)
                    total = h3.sum()
                    if total > 0:
                        best_mean = float(np.sum(np.arange(i, i + 3) * (h3 / total)))
            template[ci] = best_mean
        else:
            template[ci] = float(np.mean(vals))
    return template


def compute_legacy_mi(
    ranks: np.ndarray,
    bools: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Legacy Matching Index: each event vs template, with row-shuffle permutation test."""
    n_ch, n_ev = ranks.shape
    template = _legacy_hist_mean_rank(ranks, bools)

    def _mi_vector(tmpl: np.ndarray, vec: np.ndarray) -> float:
        sign_t = np.sign(tmpl[:, None] - tmpl[None, :])
        sign_v = np.sign(vec[:, None] - vec[None, :])
        prod = sign_t * sign_v
        n_pairs = n_ch * (n_ch - 1) // 2
        if n_pairs == 0:
            return 0.0
        return float(np.sum(np.triu(prod, k=1))) / n_pairs

    mi_list = np.array([_mi_vector(template, ranks[:, i]) for i in range(n_ev)], dtype=float)
    ori_mean = float(np.mean(mi_list))

    rng = np.random.default_rng(seed)
    perm_means: List[float] = []
    for _ in range(n_permutations):
        shuffled = np.empty_like(ranks)
        for ri in range(n_ev):
            shuffled[:, ri] = rng.permutation(n_ch)
        perm_mi = np.array([_mi_vector(template, shuffled[:, i]) for i in range(n_ev)], dtype=float)
        perm_means.append(float(np.mean(perm_mi)))

    perm_arr = np.array(perm_means, dtype=float)
    p_value = float(np.mean(perm_arr >= ori_mean))

    return {
        "template": template.tolist(),
        "mi_mean": ori_mean,
        "mi_median": float(np.median(mi_list)),
        "mi_std": float(np.std(mi_list)),
        "mi_distribution_percentiles": {
            "p5": float(np.percentile(mi_list, 5)),
            "p25": float(np.percentile(mi_list, 25)),
            "p50": float(np.percentile(mi_list, 50)),
            "p75": float(np.percentile(mi_list, 75)),
            "p95": float(np.percentile(mi_list, 95)),
        },
        "n_events": int(n_ev),
        "n_permutations": int(n_permutations),
        "permuted_mean_median": float(np.median(perm_arr)),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def compute_cluster_stereotypy(
    ranks: np.ndarray,
    bools: np.ndarray,
    channel_names: Sequence[str],
    n_clusters: int = 2,
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
    min_participation: int = 10,
) -> Dict[str, Any]:
    """KMeans clustering of events + within-cluster stereotypy analysis.

    Replicates the legacy approach from plotting_figKura_epilepsiae958Cluster.py:
    KMeans(k=2) on lagPatRank.T, then computes per-cluster statistics.
    """
    valid_events = _valid_event_indices(bools, min_participating=min_shared_channels)
    if valid_events.size < 2 * n_clusters:
        return {"error": "too_few_valid_events", "n_valid": int(valid_events.size)}

    rank_subset = ranks[:, valid_events].T
    finite_mask = np.isfinite(rank_subset)
    rank_subset = np.where(finite_mask, rank_subset, 0.0)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(rank_subset)

    clusters: List[Dict[str, Any]] = []
    cluster_templates: List[np.ndarray] = []

    for k in range(n_clusters):
        cluster_events = valid_events[labels == k]
        n_cluster = int(cluster_events.size)

        tau_summary = _multi_seed_tau_summary(
            ranks, bools, event_indices=cluster_events,
            n_sample=min(n_sample, n_cluster),
            n_seeds=n_seeds, min_shared_channels=min_shared_channels,
        )

        centered_meta = _center_rank_matrix(ranks, bools, min_participation=min_participation)
        centered_tau_summary = _multi_seed_tau_summary(
            centered_meta["centered_ranks"], bools, event_indices=cluster_events,
            n_sample=min(n_sample, n_cluster),
            n_seeds=n_seeds, min_shared_channels=min_shared_channels,
            channel_mask=centered_meta["valid_center_mask"],
        )

        template = _legacy_hist_mean_rank(
            ranks[:, cluster_events],
            bools[:, cluster_events],
        )
        cluster_templates.append(template)

        clusters.append({
            "cluster_id": int(k),
            "n_events": n_cluster,
            "fraction": float(n_cluster / valid_events.size),
            "raw_tau": tau_summary["mean_tau"],
            "raw_tau_ci_lo": tau_summary["tau_ci_lo"],
            "raw_tau_ci_hi": tau_summary["tau_ci_hi"],
            "centered_tau": centered_tau_summary["mean_tau"],
            "template_rank": np.argsort(np.argsort(template)).tolist(),
        })

    inter_cluster_corr = np.nan
    if len(cluster_templates) == 2:
        t0, t1 = cluster_templates[0], cluster_templates[1]
        if np.std(t0) > 1e-12 and np.std(t1) > 1e-12:
            r, _ = spearmanr(t0, t1)
            inter_cluster_corr = float(r)

    overall_tau = _multi_seed_tau_summary(
        ranks, bools, event_indices=valid_events,
        n_sample=n_sample, n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )["mean_tau"]

    within_tau_values = [c["raw_tau"] for c in clusters if np.isfinite(c["raw_tau"])]
    within_tau_mean = float(np.mean(within_tau_values)) if within_tau_values else np.nan

    return {
        "n_clusters": int(n_clusters),
        "n_valid_events": int(valid_events.size),
        "clusters": clusters,
        "inter_cluster_corr": inter_cluster_corr,
        "overall_tau": overall_tau,
        "within_cluster_tau_mean": within_tau_mean,
        "uplift": float(within_tau_mean - overall_tau) if np.isfinite(within_tau_mean) and np.isfinite(overall_tau) else np.nan,
        "labels": labels.tolist(),
    }


def compute_within_cluster_centered_tau(
    ranks: np.ndarray,
    bools: np.ndarray,
    labels: np.ndarray,
    valid_events: np.ndarray,
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
    min_participation: int = 10,
) -> Dict[str, Any]:
    """Within-cluster identity-bias decomposition.

    For each cluster separately: compute channel-mean-centered tau and
    compare with raw within-cluster tau.  The bias fraction quantifies
    how much of the within-cluster stereotypy comes from channel
    identity ordering vs event-specific propagation structure.
    """
    per_cluster: Dict[str, Dict[str, Any]] = {}

    for cid in np.unique(labels):
        mask = labels == cid
        cluster_ev = valid_events[mask]
        if cluster_ev.size < 2 * min_shared_channels:
            continue

        sub_ranks = ranks[:, cluster_ev]
        sub_bools = bools[:, cluster_ev]
        sub_idx = np.arange(cluster_ev.size)

        raw = _multi_seed_tau_summary(
            sub_ranks, sub_bools, event_indices=sub_idx,
            n_sample=min(n_sample, cluster_ev.size),
            n_seeds=n_seeds, min_shared_channels=min_shared_channels,
        )

        center_info = _center_rank_matrix(
            sub_ranks, sub_bools, min_participation=min_participation,
        )

        centered = _multi_seed_tau_summary(
            center_info["centered_ranks"], sub_bools, event_indices=sub_idx,
            n_sample=min(n_sample, cluster_ev.size),
            n_seeds=n_seeds, min_shared_channels=min_shared_channels,
            channel_mask=center_info["valid_center_mask"],
        )

        raw_tau = raw["mean_tau"]
        centered_tau = centered["mean_tau"]
        bias_frac = np.nan
        if np.isfinite(raw_tau) and abs(raw_tau) > 1e-12 and np.isfinite(centered_tau):
            bias_frac = float((raw_tau - centered_tau) / raw_tau)

        per_cluster[str(int(cid))] = {
            "raw_tau": raw_tau,
            "centered_tau": centered_tau,
            "bias_fraction": bias_frac,
            "n_events": int(cluster_ev.size),
        }

    raw_taus = [v["raw_tau"] for v in per_cluster.values() if np.isfinite(v["raw_tau"])]
    cen_taus = [v["centered_tau"] for v in per_cluster.values() if np.isfinite(v["centered_tau"])]
    biases = [v["bias_fraction"] for v in per_cluster.values() if np.isfinite(v["bias_fraction"])]

    return {
        "per_cluster": per_cluster,
        "mean_raw_tau": float(np.mean(raw_taus)) if raw_taus else np.nan,
        "mean_centered_tau": float(np.mean(cen_taus)) if cen_taus else np.nan,
        "mean_bias_fraction": float(np.mean(biases)) if biases else np.nan,
    }


_MAX_SILHOUETTE_EVENTS = 2000


def _kmeans_stability_for_k(
    features: np.ndarray,
    k: int,
    n_seeds: int = 10,
    min_cluster_fraction: float = 0.10,
    stability_threshold: float = 0.70,
) -> Dict[str, Any]:
    """Assess KMeans label stability at a single *k* via multi-seed AMI.

    Silhouette is computed on a subsample of at most
    ``_MAX_SILHOUETTE_EVENTS`` rows to keep O(n²) tractable on large
    event sets.  KMeans and AMI always use the full data.
    """
    n = features.shape[0]
    empty: Dict[str, Any] = {
        "k": int(k),
        "viable": False,
        "reason": "too_few_events",
        "median_silhouette": np.nan,
        "median_ami": np.nan,
        "worst_min_cluster_fraction": 0.0,
        "passes_stability": False,
        "passes_fraction": False,
        "passes_both": False,
        "best_seed": 0,
        "best_labels": np.zeros(n, dtype=int).tolist(),
    }
    if n < 2 * k:
        return empty

    sil_rng = np.random.default_rng(0)
    all_labels: List[np.ndarray] = []
    silhouettes: List[float] = []
    min_fracs: List[float] = []

    for seed in range(n_seeds):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        lab = km.fit_predict(features)
        all_labels.append(lab)
        unique, counts = np.unique(lab, return_counts=True)
        frac = float(np.min(counts) / n) if len(unique) == k else 0.0
        min_fracs.append(frac)
        if len(unique) >= 2:
            if n > _MAX_SILHOUETTE_EVENTS:
                idx = sil_rng.choice(n, _MAX_SILHOUETTE_EVENTS, replace=False)
                silhouettes.append(float(silhouette_score(features[idx], lab[idx])))
            else:
                silhouettes.append(float(silhouette_score(features, lab)))
        else:
            silhouettes.append(np.nan)

    amis: List[float] = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            amis.append(float(adjusted_mutual_info_score(all_labels[i], all_labels[j])))

    median_ami = float(np.median(amis)) if amis else np.nan
    sil_finite = [s for s in silhouettes if np.isfinite(s)]
    median_sil = float(np.median(sil_finite)) if sil_finite else np.nan
    worst_frac = float(np.min(min_fracs)) if min_fracs else 0.0

    passes_stab = bool(median_ami >= stability_threshold) if np.isfinite(median_ami) else False
    passes_frac = worst_frac >= min_cluster_fraction

    sil_arr = np.array(silhouettes, dtype=float)
    best_idx = int(np.nanargmax(sil_arr)) if np.any(np.isfinite(sil_arr)) else 0

    return {
        "k": int(k),
        "viable": True,
        "median_silhouette": median_sil,
        "median_ami": median_ami,
        "worst_min_cluster_fraction": worst_frac,
        "passes_stability": passes_stab,
        "passes_fraction": passes_frac,
        "passes_both": bool(passes_stab and passes_frac),
        "best_seed": int(best_idx),
        "best_labels": all_labels[best_idx].tolist(),
    }


def compute_adaptive_cluster_stereotypy(
    ranks: np.ndarray,
    bools: np.ndarray,
    channel_names: Sequence[str],
    k_range: Tuple[int, int] = (2, 8),
    n_stability_seeds: int = 10,
    min_cluster_fraction: float = 0.10,
    stability_threshold: float = 0.70,
    n_sample: int = 200,
    n_tau_seeds: int = 5,
    min_shared_channels: int = 3,
    min_participation: int = 10,
) -> Dict[str, Any]:
    """Scan *k* values with stability checks, report per-cluster stereotypy.

    Returns ``stable_k`` — the *k* with the highest silhouette among
    those that pass both the AMI stability gate and the minimum-cluster-
    fraction gate.  If no *k* passes, falls back to ``k_min`` with a
    ``chosen_reason`` flag.  The output includes a full scan summary
    (without per-seed label arrays) and an inter-cluster Spearman
    correlation matrix with candidate forward/reverse pair annotations.
    """
    valid_events = _valid_event_indices(bools, min_participating=min_shared_channels)
    if valid_events.size < 2 * k_range[0]:
        return {"error": "too_few_valid_events", "n_valid": int(valid_events.size)}

    rank_features = ranks[:, valid_events].T.copy()
    rank_features = np.where(np.isfinite(rank_features), rank_features, 0.0)

    k_min, k_max = k_range
    k_max = min(k_max, max(valid_events.size // 2, k_min))

    scan_results: List[Dict[str, Any]] = []
    for k in range(k_min, k_max + 1):
        scan_results.append(
            _kmeans_stability_for_k(
                rank_features,
                k,
                n_seeds=n_stability_seeds,
                min_cluster_fraction=min_cluster_fraction,
                stability_threshold=stability_threshold,
            )
        )

    passing = [r for r in scan_results if r.get("passes_both")]
    if passing:
        stable_k_entry = max(passing, key=lambda r: r["median_silhouette"])
        stable_k: Optional[int] = stable_k_entry["k"]
    else:
        stable_k = None
        stable_k_entry = None

    chosen_k = stable_k if stable_k is not None else k_min
    chosen_entry = stable_k_entry if stable_k_entry is not None else scan_results[0]
    labels = np.array(chosen_entry["best_labels"], dtype=int)

    centered_meta = _center_rank_matrix(ranks, bools, min_participation=min_participation)

    clusters: List[Dict[str, Any]] = []
    templates: List[np.ndarray] = []
    for ci in range(chosen_k):
        cluster_events = valid_events[labels == ci]
        n_cluster = int(cluster_events.size)

        if n_cluster < 2:
            clusters.append(
                {
                    "cluster_id": int(ci),
                    "n_events": n_cluster,
                    "fraction": float(n_cluster / valid_events.size),
                    "raw_tau": np.nan,
                    "raw_tau_ci_lo": np.nan,
                    "raw_tau_ci_hi": np.nan,
                    "centered_tau": np.nan,
                    "template_rank": [],
                }
            )
            templates.append(np.zeros(ranks.shape[0], dtype=float))
            continue

        tau_sum = _multi_seed_tau_summary(
            ranks,
            bools,
            event_indices=cluster_events,
            n_sample=min(n_sample, n_cluster),
            n_seeds=n_tau_seeds,
            min_shared_channels=min_shared_channels,
        )
        cent_tau_sum = _multi_seed_tau_summary(
            centered_meta["centered_ranks"],
            bools,
            event_indices=cluster_events,
            n_sample=min(n_sample, n_cluster),
            n_seeds=n_tau_seeds,
            min_shared_channels=min_shared_channels,
            channel_mask=centered_meta["valid_center_mask"],
        )

        template = _legacy_hist_mean_rank(
            ranks[:, cluster_events],
            bools[:, cluster_events],
        )
        templates.append(template)

        clusters.append(
            {
                "cluster_id": int(ci),
                "n_events": n_cluster,
                "fraction": float(n_cluster / valid_events.size),
                "raw_tau": tau_sum["mean_tau"],
                "raw_tau_ci_lo": tau_sum["tau_ci_lo"],
                "raw_tau_ci_hi": tau_sum["tau_ci_hi"],
                "centered_tau": cent_tau_sum["mean_tau"],
                "template_rank": np.argsort(np.argsort(template)).tolist(),
            }
        )

    n_t = len(templates)
    corr_matrix = np.full((n_t, n_t), np.nan, dtype=float)
    candidate_pairs: List[Dict[str, Any]] = []
    for i in range(n_t):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n_t):
            if np.std(templates[i]) > 1e-12 and np.std(templates[j]) > 1e-12:
                r, _ = spearmanr(templates[i], templates[j])
                corr_matrix[i, j] = float(r)
                corr_matrix[j, i] = float(r)
                if r < -0.5:
                    candidate_pairs.append(
                        {
                            "cluster_a": int(i),
                            "cluster_b": int(j),
                            "spearman_r": float(r),
                            "label": "candidate_forward_reverse",
                        }
                    )

    overall_tau = _multi_seed_tau_summary(
        ranks,
        bools,
        event_indices=valid_events,
        n_sample=n_sample,
        n_seeds=n_tau_seeds,
        min_shared_channels=min_shared_channels,
    )["mean_tau"]

    within_taus = [c["raw_tau"] for c in clusters if np.isfinite(c["raw_tau"])]
    within_tau_mean = float(np.mean(within_taus)) if within_taus else np.nan
    uplift = (
        float(within_tau_mean - overall_tau)
        if np.isfinite(within_tau_mean) and np.isfinite(overall_tau)
        else np.nan
    )

    scan_summary = [{k: v for k, v in r.items() if k != "best_labels"} for r in scan_results]

    return {
        "k_range": [int(k_min), int(k_max)],
        "n_valid_events": int(valid_events.size),
        "scan": scan_summary,
        "stable_k": stable_k,
        "chosen_k": int(chosen_k),
        "chosen_reason": "stable_k" if stable_k is not None else "fallback_k_min",
        "clusters": clusters,
        "inter_cluster_corr_matrix": corr_matrix.tolist(),
        "candidate_forward_reverse_pairs": candidate_pairs,
        "overall_tau": overall_tau,
        "within_cluster_tau_mean": within_tau_mean,
        "uplift": uplift,
        "labels": chosen_entry["best_labels"],
    }


def build_cluster_templates(
    ranks: np.ndarray,
    bools: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
) -> np.ndarray:
    """Legacy-aligned template per cluster.

    Parameters
    ----------
    ranks : (n_ch, n_events)
    bools : (n_ch, n_events)
    labels : (n_events,) — cluster label per event
    n_clusters : int

    Returns
    -------
    templates : (n_clusters, n_ch) — legacy MI template with NaN for channels
    that never participate in the cluster
    """
    n_ch = ranks.shape[0]
    templates = np.full((n_clusters, n_ch), np.nan, dtype=float)
    for k in range(n_clusters):
        mask = labels == k
        if not np.any(mask):
            continue
        cluster_ranks = ranks[:, mask]
        cluster_bools = bools[:, mask]
        template = _legacy_hist_mean_rank(cluster_ranks, cluster_bools)
        missing_mask = np.sum(cluster_bools > 0, axis=1) == 0
        template = np.asarray(template, dtype=float)
        template[missing_mask] = np.nan
        templates[k] = template
    return templates


def assign_events_to_templates(
    ranks: np.ndarray,
    bools: np.ndarray,
    templates: np.ndarray,
    min_shared_channels: int = 3,
) -> np.ndarray:
    """Assign events to nearest template by masked Euclidean distance.

    Consistent with the KMeans distance metric used during clustering.
    Returns array of length n_events with cluster IDs (-1 if unassignable).
    """
    n_ch, n_events = ranks.shape
    n_clusters = templates.shape[0]
    masked_ranks = np.where(bools > 0, ranks, np.nan)

    best_dist = np.full(n_events, np.inf, dtype=float)
    assignments = np.full(n_events, -1, dtype=int)

    for k in range(n_clusters):
        diff = masked_ranks - templates[k][:, None]
        sq_diff = diff ** 2
        valid = np.isfinite(sq_diff)
        n_valid = np.sum(valid, axis=0)
        with np.errstate(invalid="ignore"):
            mean_sq = np.nansum(sq_diff, axis=0) / np.maximum(n_valid, 1)
        mean_sq = np.where(n_valid >= min_shared_channels, mean_sq, np.inf)

        improved = mean_sq < best_dist
        assignments = np.where(improved, k, assignments)
        best_dist = np.where(improved, mean_sq, best_dist)

    return assignments


def _match_templates_across_splits(
    templates_a: np.ndarray,
    templates_b: np.ndarray,
    min_finite_channels: int = 3,
) -> Dict[str, Any]:
    """Optimal matching of templates from two splits via Spearman + Hungarian."""
    n_a = templates_a.shape[0]
    n_b = templates_b.shape[0]
    k = min(n_a, n_b)

    corr_matrix = np.full((n_a, n_b), np.nan, dtype=float)
    for i in range(n_a):
        for j in range(n_b):
            both_fin = np.isfinite(templates_a[i]) & np.isfinite(templates_b[j])
            if int(np.sum(both_fin)) >= min_finite_channels:
                r, _ = spearmanr(templates_a[i, both_fin], templates_b[j, both_fin])
                corr_matrix[i, j] = float(r) if np.isfinite(r) else np.nan

    cost = np.where(np.isfinite(corr_matrix[:k, :k]), -corr_matrix[:k, :k], 1e6)
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
    matched_corrs = [
        float(corr_matrix[r, c])
        for r, c in zip(row_ind, col_ind)
        if np.isfinite(corr_matrix[r, c])
    ]

    return {
        "corr_matrix": corr_matrix,
        "mapping_a_to_b": mapping,
        "matched_corrs": matched_corrs,
        "mean_match_corr": float(np.mean(matched_corrs)) if matched_corrs else np.nan,
        "min_match_corr": float(np.min(matched_corrs)) if matched_corrs else np.nan,
    }


def _find_anticorrelated_pairs(
    templates: np.ndarray,
    threshold: float = -0.5,
    min_finite_channels: int = 3,
) -> List[Tuple[int, int]]:
    """Find template pairs with Spearman r below *threshold*."""
    n = templates.shape[0]
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            both_fin = np.isfinite(templates[i]) & np.isfinite(templates[j])
            if int(np.sum(both_fin)) < min_finite_channels:
                continue
            r, _ = spearmanr(templates[i, both_fin], templates[j, both_fin])
            if np.isfinite(r) and r < threshold:
                pairs.append((i, j))
    return pairs


def _map_pairs(pair_list: List[Tuple[int, int]], mapping: Dict[int, int]) -> List[Tuple[int, int]]:
    """Map cluster-pair indices across a template matching."""
    mapped: List[Tuple[int, int]] = []
    for i, j in pair_list:
        if i not in mapping or j not in mapping:
            continue
        a = int(mapping[i])
        b = int(mapping[j])
        mapped.append((min(a, b), max(a, b)))
    return mapped


def compute_time_split_reproducibility(
    ranks: np.ndarray,
    bools: np.ndarray,
    event_abs_times: np.ndarray,
    block_ids: np.ndarray,
    chosen_k: int,
    adaptive_labels: np.ndarray,
    valid_event_indices: np.ndarray,
    min_shared_channels: int = 3,
    fwd_rev_threshold: float = -0.5,
) -> Dict[str, Any]:
    """Assess cross-time template reproducibility via split-half and odd/even splits.

    For each split:
    1. Re-cluster each half independently at ``k=chosen_k``
    2. Build templates from each half's clustering
    3. Match templates across halves (Hungarian on Spearman correlation)
    4. Assign half-B events to half-A templates and compute label agreement

    Returns a reproducibility grade: ``strong`` / ``moderate`` / ``weak``.
    """
    labels = np.asarray(adaptive_labels, dtype=int)
    valid_event_indices = np.asarray(valid_event_indices, dtype=int)
    v_times = np.asarray(event_abs_times[valid_event_indices], dtype=float)
    if labels.size != valid_event_indices.size:
        raise ValueError("adaptive_labels size must equal valid_event_indices size")

    # Split-half must respect true event time, not whatever order the arrays happen to be in.
    order = np.argsort(np.where(np.isfinite(v_times), v_times, np.inf), kind="mergesort")
    sorted_valid = valid_event_indices[order]
    labels = labels[order]
    v_ranks = ranks[:, sorted_valid]
    v_bools = bools[:, sorted_valid]
    v_blocks = block_ids[sorted_valid]
    v_times = v_times[order]
    n_valid = sorted_valid.size

    full_templates = build_cluster_templates(v_ranks, v_bools, labels, chosen_k)
    full_fwd_rev = _find_anticorrelated_pairs(
        full_templates, fwd_rev_threshold, min_shared_channels
    )

    splits: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    mid = n_valid // 2
    if mid >= 2 * chosen_k and (n_valid - mid) >= 2 * chosen_k:
        splits["first_half_second_half"] = (np.arange(mid), np.arange(mid, n_valid))

    unique_blocks = np.unique(v_blocks)
    if len(unique_blocks) >= 2:
        even_blocks = unique_blocks[0::2]
        odd_blocks = unique_blocks[1::2]
        even_mask = np.isin(v_blocks, even_blocks)
        odd_mask = np.isin(v_blocks, odd_blocks)
        if int(np.sum(even_mask)) >= 2 * chosen_k and int(np.sum(odd_mask)) >= 2 * chosen_k:
            splits["odd_even_block"] = (
                np.where(even_mask)[0],
                np.where(odd_mask)[0],
            )

    split_results: Dict[str, Any] = {}

    for split_name, (idx_a, idx_b) in splits.items():
        feat_a = v_ranks[:, idx_a].T.copy()
        feat_a = np.where(np.isfinite(feat_a), feat_a, 0.0)
        km_a = KMeans(n_clusters=chosen_k, n_init=10, random_state=0)
        labels_a = km_a.fit_predict(feat_a)
        templates_a = build_cluster_templates(
            v_ranks[:, idx_a], v_bools[:, idx_a], labels_a, chosen_k
        )

        feat_b = v_ranks[:, idx_b].T.copy()
        feat_b = np.where(np.isfinite(feat_b), feat_b, 0.0)
        km_b = KMeans(n_clusters=chosen_k, n_init=10, random_state=0)
        labels_b = km_b.fit_predict(feat_b)
        templates_b = build_cluster_templates(
            v_ranks[:, idx_b], v_bools[:, idx_b], labels_b, chosen_k
        )

        match = _match_templates_across_splits(templates_a, templates_b, min_shared_channels)

        assigned_b = assign_events_to_templates(
            v_ranks[:, idx_b], v_bools[:, idx_b], templates_a, min_shared_channels
        )

        mapping = match["mapping_a_to_b"]
        n_assignable = int(np.sum(assigned_b >= 0))
        n_agree = 0
        for ev in range(idx_b.size):
            a_lab = int(assigned_b[ev])
            b_lab = int(labels_b[ev])
            if a_lab >= 0 and a_lab in mapping and mapping[a_lab] == b_lab:
                n_agree += 1
        agreement = float(n_agree / n_assignable) if n_assignable > 0 else np.nan

        fwd_rev_a = _find_anticorrelated_pairs(templates_a, fwd_rev_threshold, min_shared_channels)
        fwd_rev_b = _find_anticorrelated_pairs(templates_b, fwd_rev_threshold, min_shared_channels)
        fwd_rev_reproduced: Optional[bool] = None
        if full_fwd_rev:
            mapped_a_pairs = set(_map_pairs(fwd_rev_a, mapping))
            b_pairs = set((min(i, j), max(i, j)) for i, j in fwd_rev_b)
            fwd_rev_reproduced = bool(mapped_a_pairs & b_pairs)

        # PR-6 robustness: rank ONLY valid (participating) channels per cluster.
        # Non-valid channels get rank = -1 sentinel; downstream source/sink
        # extraction must filter on cluster_valid_mask_a/b.  Filling NaN with
        # the mean would give non-participating channels a fake mid-rank and
        # make split-half robustness look more stable than it is.
        n_ch_total = templates_a.shape[1]

        def _rank_with_mask(template_row: np.ndarray) -> Tuple[List[int], List[bool]]:
            valid = np.isfinite(template_row)
            rank = np.full(n_ch_total, -1, dtype=int)
            valid_idx = np.where(valid)[0]
            if valid_idx.size >= 2:
                vals = template_row[valid_idx]
                local_rank = np.argsort(np.argsort(vals))
                rank[valid_idx] = local_rank
            return rank.tolist(), valid.tolist()

        cluster_rank_a: List[List[int]] = []
        cluster_rank_b: List[List[int]] = []
        cluster_valid_mask_a: List[List[bool]] = []
        cluster_valid_mask_b: List[List[bool]] = []
        for ki in range(chosen_k):
            ra, ma = _rank_with_mask(templates_a[ki])
            rb, mb = _rank_with_mask(templates_b[ki])
            cluster_rank_a.append(ra)
            cluster_rank_b.append(rb)
            cluster_valid_mask_a.append(ma)
            cluster_valid_mask_b.append(mb)

        # PR-6 mapping alignment: cluster_rank_a[ki] is A's cluster ki, but
        # cluster_rank_b[ki] is B's raw KMeans label ki — KMeans labels are
        # arbitrary, so direct same-index Jaccard would compare unrelated
        # clusters.  Re-key B's ranks under A's cluster id via mapping_a_to_b.
        cluster_rank_b_matched_to_a: List[Optional[List[int]]] = []
        cluster_valid_mask_b_matched_to_a: List[Optional[List[bool]]] = []
        for a_id in range(chosen_k):
            b_id = mapping.get(int(a_id))
            if b_id is not None and 0 <= int(b_id) < chosen_k:
                cluster_rank_b_matched_to_a.append(cluster_rank_b[int(b_id)])
                cluster_valid_mask_b_matched_to_a.append(cluster_valid_mask_b[int(b_id)])
            else:
                cluster_rank_b_matched_to_a.append(None)
                cluster_valid_mask_b_matched_to_a.append(None)

        split_results[split_name] = {
            "n_events_a": int(idx_a.size),
            "n_events_b": int(idx_b.size),
            "template_corr_matrix": match["corr_matrix"].tolist(),
            "mapping_a_to_b": {str(k_): int(v_) for k_, v_ in mapping.items()},
            "mean_match_corr": match["mean_match_corr"],
            "min_match_corr": match["min_match_corr"],
            "assignment_agreement": agreement,
            "n_assignable": n_assignable,
            "n_agree": n_agree,
            "forward_reverse_in_a": len(fwd_rev_a),
            "forward_reverse_in_b": len(fwd_rev_b),
            "forward_reverse_reproduced": fwd_rev_reproduced,
            "cluster_rank_a": cluster_rank_a,
            "cluster_rank_b": cluster_rank_b,
            "cluster_valid_mask_a": cluster_valid_mask_a,
            "cluster_valid_mask_b": cluster_valid_mask_b,
            "cluster_rank_b_matched_to_a": cluster_rank_b_matched_to_a,
            "cluster_valid_mask_b_matched_to_a": cluster_valid_mask_b_matched_to_a,
        }

    completed = [v for v in split_results.values() if isinstance(v, dict)]
    if not completed:
        grade = "insufficient_data"
    else:
        mean_matches = [
            r["mean_match_corr"]
            for r in completed
            if np.isfinite(r.get("mean_match_corr", np.nan))
        ]
        agreements = [
            r["assignment_agreement"]
            for r in completed
            if np.isfinite(r.get("assignment_agreement", np.nan))
        ]
        avg_match = float(np.mean(mean_matches)) if mean_matches else 0.0
        avg_agree = float(np.mean(agreements)) if agreements else 0.0
        if avg_match >= 0.8 and avg_agree >= 0.70:
            grade = "strong"
        elif avg_match >= 0.5 and avg_agree >= 0.50:
            grade = "moderate"
        else:
            grade = "weak"

    return {
        "chosen_k": int(chosen_k),
        "n_valid_events": int(n_valid),
        "full_data_forward_reverse_pairs": len(full_fwd_rev),
        "splits": split_results,
        "reproducibility_grade": grade,
    }


def validate_absolute_lag_clustering(
    ranks: np.ndarray,
    lag_raw: np.ndarray,
    bools: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    *,
    valid_event_indices: Optional[np.ndarray] = None,
    n_sample: int = 200,
    seed: int = 0,
    min_shared_channels: int = 3,
    min_participating: int = 5,
) -> Dict[str, Any]:
    """Validate rank-based clusters in relative-lag space for PR-4B Step 0."""
    ranks = np.asarray(ranks, dtype=float)
    lag_raw = np.asarray(lag_raw, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    labels = np.asarray(cluster_labels, dtype=int)
    if ranks.shape != lag_raw.shape or ranks.shape != bools.shape:
        raise ValueError("ranks, lag_raw, and bools must share shape")

    if valid_event_indices is None:
        if labels.size == ranks.shape[1]:
            valid_event_indices = np.arange(ranks.shape[1], dtype=int)
        else:
            valid_event_indices = _valid_event_indices(bools, min_participating=min_shared_channels)
    valid_event_indices = np.asarray(valid_event_indices, dtype=int)
    if labels.size != valid_event_indices.size:
        raise ValueError("cluster_labels size must equal valid_event_indices size")

    relative_lag = _compute_relative_lag_matrix(lag_raw, bools)
    v_ranks = ranks[:, valid_event_indices]
    v_rel = relative_lag[:, valid_event_indices]
    v_bools = bools[:, valid_event_indices]
    n_part = np.sum(v_bools, axis=0).astype(int)

    order_summary = _event_order_alignment_summary(
        v_ranks,
        v_rel,
        v_bools,
        min_shared_channels=min_shared_channels,
    )

    npart_bins: List[Tuple[str, int, Optional[int]]] = [
        ("3-4", 3, 4),
        ("5-8", 5, 8),
        ("9+", 9, None),
    ]
    by_bin: Dict[str, Any] = {}
    eligible_values_all: List[float] = []

    for bin_idx, (label, lo, hi) in enumerate(npart_bins):
        if hi is None:
            bin_mask = n_part >= int(lo)
        else:
            bin_mask = (n_part >= int(lo)) & (n_part <= int(hi))
        min_shared = int(min_participating) if lo >= int(min_participating) else int(min_shared_channels)

        cluster_stats: List[Dict[str, Any]] = []
        bin_values: List[float] = []
        for cluster_id in range(int(n_clusters)):
            local_idx = np.where((labels == cluster_id) & bin_mask)[0]
            sampled_idx = _sample_event_indices(
                local_idx,
                n_sample=n_sample,
                seed=seed + 101 * bin_idx + 1009 * cluster_id,
            )
            stats = _pairwise_relative_lag_correlations(
                v_rel,
                v_bools,
                sampled_idx,
                min_shared_channels=min_shared,
            )
            bin_values.extend(stats["pearson_r_values"])
            cluster_stats.append(
                {
                    "cluster_id": int(cluster_id),
                    "n_events": int(local_idx.size),
                    "n_events_sampled": int(sampled_idx.size),
                    "n_pairs_valid": int(stats["n_pairs_valid"]),
                    "median_r": stats["median_r"],
                }
            )

        if lo >= int(min_participating):
            eligible_values_all.extend(bin_values)

        by_bin[label] = {
            "n_events": int(np.sum(bin_mask)),
            "n_clusters": int(n_clusters),
            "min_shared_channels_for_r": int(min_shared),
            "median_r": float(np.median(bin_values)) if bin_values else np.nan,
            "mean_r": float(np.mean(bin_values)) if bin_values else np.nan,
            "n_pairs_valid": int(len(bin_values)),
            "cluster_stats": cluster_stats,
        }

    eligible_mask = n_part >= int(min_participating)
    per_cluster_pearson_r: List[Dict[str, Any]] = []
    for cluster_id in range(int(n_clusters)):
        local_idx = np.where((labels == cluster_id) & eligible_mask)[0]
        sampled_idx = _sample_event_indices(
            local_idx,
            n_sample=n_sample,
            seed=seed + 5003 + 1009 * cluster_id,
        )
        stats = _pairwise_relative_lag_correlations(
            v_rel,
            v_bools,
            sampled_idx,
            min_shared_channels=int(min_participating),
        )
        per_cluster_pearson_r.append(
            {
                "cluster_id": int(cluster_id),
                "n_events": int(local_idx.size),
                "n_events_sampled": int(sampled_idx.size),
                "n_pairs_valid": int(stats["n_pairs_valid"]),
                "median_r": stats["median_r"],
            }
        )

    eligible_fraction = float(np.mean(eligible_mask)) if eligible_mask.size else np.nan
    eligible_median_r = (
        float(np.median(eligible_values_all)) if eligible_values_all else np.nan
    )

    cluster_fractions = np.zeros(int(n_clusters), dtype=float)
    cluster_median_rs = np.full(int(n_clusters), np.nan, dtype=float)
    total_eligible = int(np.sum(eligible_mask))
    for ci, cr in enumerate(per_cluster_pearson_r):
        cluster_fractions[ci] = float(cr["n_events"] / total_eligible) if total_eligible else 0.0
        cluster_median_rs[ci] = cr["median_r"]

    dominant_idx = int(np.argmax(cluster_fractions))
    dominant_median_r = float(cluster_median_rs[dominant_idx])
    dominant_fraction = float(cluster_fractions[dominant_idx])

    return {
        "n_valid_events": int(valid_event_indices.size),
        "n_clusters": int(n_clusters),
        "n_sample": int(n_sample),
        "min_participating": int(min_participating),
        "order_validation": order_summary,
        "within_cluster_pearson_r_by_npart": by_bin,
        "eligible_fraction": eligible_fraction,
        "eligible_median_r": eligible_median_r,
        "dominant_cluster_id": int(dominant_idx),
        "dominant_cluster_fraction": dominant_fraction,
        "dominant_cluster_median_r": dominant_median_r,
        "per_cluster_pearson_r": per_cluster_pearson_r,
        "validation_pass": bool(
            np.isfinite(dominant_median_r) and dominant_median_r > 0.7
        ),
    }


def _build_rate_state_bins(
    event_abs_times: np.ndarray,
    *,
    bin_hours: float,
    min_events_per_bin: int,
) -> Dict[str, Any]:
    times = np.asarray(event_abs_times, dtype=float)
    if bin_hours <= 0:
        raise ValueError("bin_hours must be > 0")
    if min_events_per_bin <= 0:
        raise ValueError("min_events_per_bin must be > 0")

    finite_mask = np.isfinite(times)
    finite_idx = np.where(finite_mask)[0]
    if finite_idx.size == 0:
        return {
            "event_bin_ids": np.full(times.shape, -1, dtype=int),
            "event_rate_states": np.full(times.shape, "excluded", dtype=object),
            "rate_bins": [],
            "n_bins_total": 0,
            "n_bins_eligible": 0,
            "median_eligible_rate_per_hour": np.nan,
            "high_bin_ids": [],
            "low_bin_ids": [],
        }

    bin_sec = float(bin_hours) * 3600.0
    finite_times = times[finite_mask]
    timeline_start = float(np.min(finite_times))
    event_bin_ids = np.full(times.shape, -1, dtype=int)
    event_bin_ids[finite_mask] = np.floor((finite_times - timeline_start) / bin_sec).astype(int)

    unique_bin_ids = sorted(set(event_bin_ids[finite_mask].tolist()))
    rate_bins: List[Dict[str, Any]] = []
    eligible_bins: List[Dict[str, Any]] = []
    for bin_id in unique_bin_ids:
        mask = event_bin_ids == int(bin_id)
        n_events = int(np.sum(mask))
        rate_per_hour = float(n_events / float(bin_hours))
        entry = {
            "bin_id": int(bin_id),
            "start_time": timeline_start + int(bin_id) * bin_sec,
            "end_time": timeline_start + (int(bin_id) + 1) * bin_sec,
            "n_events": n_events,
            "rate_per_hour": rate_per_hour,
            "eligible": bool(n_events >= int(min_events_per_bin)),
        }
        rate_bins.append(entry)
        if entry["eligible"]:
            eligible_bins.append(entry)

    eligible_sorted = sorted(
        eligible_bins,
        key=lambda entry: (float(entry["rate_per_hour"]), int(entry["bin_id"])),
    )
    half = len(eligible_sorted) // 2
    low_bin_ids = {int(entry["bin_id"]) for entry in eligible_sorted[:half]}
    high_bin_ids = (
        {int(entry["bin_id"]) for entry in eligible_sorted[-half:]}
        if half > 0
        else set()
    )

    event_rate_states = np.full(times.shape, "excluded", dtype=object)
    for entry in rate_bins:
        bin_id = int(entry["bin_id"])
        if bin_id in high_bin_ids:
            state = "high"
        elif bin_id in low_bin_ids:
            state = "low"
        else:
            state = "excluded"
        entry["rate_state"] = state
        event_rate_states[event_bin_ids == bin_id] = state

    eligible_rates = [
        float(entry["rate_per_hour"]) for entry in eligible_sorted
        if np.isfinite(entry["rate_per_hour"])
    ]
    return {
        "event_bin_ids": event_bin_ids,
        "event_rate_states": event_rate_states,
        "rate_bins": rate_bins,
        "n_bins_total": int(len(rate_bins)),
        "n_bins_eligible": int(len(eligible_sorted)),
        "median_eligible_rate_per_hour": (
            float(np.median(eligible_rates)) if eligible_rates else np.nan
        ),
        "high_bin_ids": sorted(high_bin_ids),
        "low_bin_ids": sorted(low_bin_ids),
    }


def _cluster_state_tau_summary(
    ranks: np.ndarray,
    bools: np.ndarray,
    event_indices: np.ndarray,
    *,
    n_sample: int,
    n_seeds: int,
    min_shared_channels: int,
    min_center_participation: int,
) -> Dict[str, Any]:
    event_indices = np.asarray(event_indices, dtype=int)
    if event_indices.size == 0:
        return {
            "n_events": 0,
            "raw_tau": np.nan,
            "centered_tau": np.nan,
            "raw_n_pairs_valid": 0,
            "centered_n_pairs_valid": 0,
        }

    sub_ranks = np.asarray(ranks[:, event_indices], dtype=float)
    sub_bools = np.asarray(bools[:, event_indices], dtype=bool)
    sub_idx = np.arange(event_indices.size, dtype=int)
    n_sample_local = int(min(int(n_sample), event_indices.size))

    raw = _multi_seed_tau_summary(
        sub_ranks,
        sub_bools,
        event_indices=sub_idx,
        n_sample=n_sample_local,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
    )
    centered_meta = _center_rank_matrix(
        sub_ranks,
        sub_bools,
        min_participation=min_center_participation,
    )
    centered = _multi_seed_tau_summary(
        centered_meta["centered_ranks"],
        sub_bools,
        event_indices=sub_idx,
        n_sample=n_sample_local,
        n_seeds=n_seeds,
        min_shared_channels=min_shared_channels,
        channel_mask=centered_meta["valid_center_mask"],
    )
    return {
        "n_events": int(event_indices.size),
        "raw_tau": raw["mean_tau"],
        "centered_tau": centered["mean_tau"],
        "raw_n_pairs_valid": int(raw["n_pairs_valid_median"]),
        "centered_n_pairs_valid": int(centered["n_pairs_valid_median"]),
    }


def _aggregate_cluster_state_metric(
    per_cluster: List[Dict[str, Any]],
    metric_key: str,
    *,
    high_key: str = "high",
    low_key: str = "low",
) -> Dict[str, Any]:
    comparable = [
        cluster for cluster in per_cluster
        if np.isfinite(cluster.get(high_key, {}).get(metric_key, np.nan))
        and np.isfinite(cluster.get(low_key, {}).get(metric_key, np.nan))
    ]
    if not comparable:
        return {
            "n_clusters_compared": 0,
            "cluster_ids_compared": [],
            "high_mean": np.nan,
            "low_mean": np.nan,
            "delta_high_minus_low": np.nan,
            "n_clusters_high_gt_low": 0,
        }

    high_vals = np.array([cluster[high_key][metric_key] for cluster in comparable], dtype=float)
    low_vals = np.array([cluster[low_key][metric_key] for cluster in comparable], dtype=float)
    delta_vals = high_vals - low_vals
    return {
        "n_clusters_compared": int(len(comparable)),
        "cluster_ids_compared": [int(cluster["cluster_id"]) for cluster in comparable],
        "high_mean": float(np.mean(high_vals)),
        "low_mean": float(np.mean(low_vals)),
        "delta_high_minus_low": float(np.mean(delta_vals)),
        "n_clusters_high_gt_low": int(np.sum(delta_vals > 0)),
    }


def compute_rate_state_coupling(
    event_abs_times: np.ndarray,
    ranks: np.ndarray,
    lag_raw: np.ndarray,
    bools: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    *,
    valid_event_indices: Optional[np.ndarray] = None,
    rate_bin_hours: float = 2.0,
    min_events_per_bin: int = 20,
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
    min_center_participation: int = 10,
    min_participating_l3: int = 5,
    match_seed: int = 42,
) -> Dict[str, Any]:
    """PR-4B Step 1-3: high-rate vs low-rate L1/L2/L3 comparison.

    Matched subsampling: for each cluster, the larger state group is
    randomly downsampled to the size of the smaller group before tau
    computation, eliminating the event-count asymmetry confound. L3 uses
    separate exact n_participating matching so lag span is not inflated by
    state-dependent participation differences.
    """
    times = np.asarray(event_abs_times, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    lag_raw = np.asarray(lag_raw, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    labels = np.asarray(cluster_labels, dtype=int)
    if ranks.shape != bools.shape or ranks.shape != lag_raw.shape:
        raise ValueError("ranks, lag_raw, and bools must share shape")

    if valid_event_indices is None:
        if labels.size == ranks.shape[1]:
            valid_event_indices = np.arange(ranks.shape[1], dtype=int)
        else:
            valid_event_indices = _valid_event_indices(
                bools,
                min_participating=min_shared_channels,
            )
    valid_event_indices = np.asarray(valid_event_indices, dtype=int)
    if labels.size != valid_event_indices.size:
        raise ValueError("cluster_labels size must equal valid_event_indices size")

    v_times = times[valid_event_indices]
    v_ranks = ranks[:, valid_event_indices]
    v_bools = bools[:, valid_event_indices]
    v_rel = _compute_relative_lag_matrix(
        lag_raw[:, valid_event_indices], v_bools
    )
    n_part = np.sum(v_bools, axis=0).astype(int)

    bin_info = _build_rate_state_bins(
        v_times,
        bin_hours=rate_bin_hours,
        min_events_per_bin=min_events_per_bin,
    )
    event_states = np.asarray(bin_info["event_rate_states"], dtype=object)
    event_bin_ids = np.asarray(bin_info["event_bin_ids"], dtype=int)
    state_masks = {
        "high": event_states == "high",
        "low": event_states == "low",
    }

    match_rng = np.random.default_rng(int(match_seed))
    cluster_counts = np.bincount(labels, minlength=int(n_clusters)) if labels.size else np.zeros(int(n_clusters), dtype=int)

    per_cluster: List[Dict[str, Any]] = []
    for cluster_id in range(int(n_clusters)):
        cluster_mask = labels == int(cluster_id)
        high_idx_all = np.where(cluster_mask & state_masks["high"])[0]
        low_idx_all = np.where(cluster_mask & state_masks["low"])[0]
        high_idx = np.asarray(high_idx_all, dtype=int)
        low_idx = np.asarray(low_idx_all, dtype=int)

        n_match = int(min(high_idx.size, low_idx.size))

        if high_idx.size > n_match and n_match > 0:
            high_idx = match_rng.choice(high_idx, size=n_match, replace=False)
        if low_idx.size > n_match and n_match > 0:
            low_idx = match_rng.choice(low_idx, size=n_match, replace=False)

        adj_center = int(min(
            int(min_center_participation),
            max(1, n_match // 5),
        ))

        cluster_entry: Dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "n_matched": n_match,
            "n_events_total": int(np.sum(cluster_mask)),
        }
        for state_name, state_idx in (("high", high_idx), ("low", low_idx)):
            cluster_entry[state_name] = _cluster_state_tau_summary(
                v_ranks,
                v_bools,
                state_idx,
                n_sample=n_sample,
                n_seeds=n_seeds,
                min_shared_channels=min_shared_channels,
                min_center_participation=adj_center,
            )
        high_raw = cluster_entry["high"]["raw_tau"]
        low_raw = cluster_entry["low"]["raw_tau"]
        high_centered = cluster_entry["high"]["centered_tau"]
        low_centered = cluster_entry["low"]["centered_tau"]
        cluster_entry["raw_delta_high_minus_low"] = (
            float(high_raw - low_raw)
            if np.isfinite(high_raw) and np.isfinite(low_raw)
            else np.nan
        )
        cluster_entry["centered_delta_high_minus_low"] = (
            float(high_centered - low_centered)
            if np.isfinite(high_centered) and np.isfinite(low_centered)
            else np.nan
        )

        l3_span_match = _match_event_indices_by_nparticipating(
            high_idx_all,
            low_idx_all,
            n_part,
            seed=int(match_seed) + 7919 * int(cluster_id) + 17,
            min_participating=int(min_shared_channels),
        )
        l3_pearson_match = _match_event_indices_by_nparticipating(
            high_idx_all,
            low_idx_all,
            n_part,
            seed=int(match_seed) + 7919 * int(cluster_id) + 43,
            min_participating=int(min_participating_l3),
        )
        cluster_entry["l3_matching"] = {
            "lag_span": {
                "n_matched": int(l3_span_match["n_matched"]),
                "matched_counts_by_n_participating": l3_span_match["matched_counts_by_n_participating"],
                "min_participating": int(l3_span_match["min_participating"]),
            },
            "pearson_r": {
                "n_matched": int(l3_pearson_match["n_matched"]),
                "matched_counts_by_n_participating": l3_pearson_match["matched_counts_by_n_participating"],
                "min_participating": int(l3_pearson_match["min_participating"]),
            },
        }
        cluster_entry["high_l3"] = _cluster_state_l3_summary(
            v_rel,
            v_bools,
            lag_span_event_indices=l3_span_match["high_event_indices"],
            pearson_event_indices=l3_pearson_match["high_event_indices"],
            min_lag_span_participating=int(min_shared_channels),
            min_shared_channels_for_r=int(min_participating_l3),
            n_sample_pearson=n_sample,
            pearson_seed=int(match_seed) + 7919 * int(cluster_id) + 101,
        )
        cluster_entry["low_l3"] = _cluster_state_l3_summary(
            v_rel,
            v_bools,
            lag_span_event_indices=l3_span_match["low_event_indices"],
            pearson_event_indices=l3_pearson_match["low_event_indices"],
            min_lag_span_participating=int(min_shared_channels),
            min_shared_channels_for_r=int(min_participating_l3),
            n_sample_pearson=n_sample,
            pearson_seed=int(match_seed) + 7919 * int(cluster_id) + 137,
        )
        high_lag_span = cluster_entry["high_l3"]["lag_span_mean"]
        low_lag_span = cluster_entry["low_l3"]["lag_span_mean"]
        high_pearson = cluster_entry["high_l3"]["pearson_r_median"]
        low_pearson = cluster_entry["low_l3"]["pearson_r_median"]
        cluster_entry["lag_span_delta_high_minus_low"] = (
            float(high_lag_span - low_lag_span)
            if np.isfinite(high_lag_span) and np.isfinite(low_lag_span)
            else np.nan
        )
        cluster_entry["pearson_r_delta_high_minus_low"] = (
            float(high_pearson - low_pearson)
            if np.isfinite(high_pearson) and np.isfinite(low_pearson)
            else np.nan
        )
        eligible_rate_bins = [entry for entry in bin_info["rate_bins"] if entry.get("eligible", False)]
        rate_values: List[float] = []
        frac_values: List[float] = []
        for entry in eligible_rate_bins:
            bin_id = int(entry["bin_id"])
            bin_mask = event_bin_ids == bin_id
            n_bin_events = int(np.sum(bin_mask))
            if n_bin_events <= 0:
                continue
            rate_values.append(float(entry["rate_per_hour"]))
            frac_values.append(float(np.sum(cluster_mask & bin_mask) / n_bin_events))
        if len(rate_values) >= 2 and len(set(np.round(rate_values, 12))) >= 2 and len(set(np.round(frac_values, 12))) >= 2:
            rho, pval = spearmanr(rate_values, frac_values)
        else:
            rho, pval = np.nan, np.nan
        cluster_entry["occupancy_rate"] = {
            "n_bins": int(len(rate_values)),
            "mean_fraction": float(np.mean(frac_values)) if frac_values else np.nan,
            "occupancy_rate_spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
            "occupancy_rate_spearman_p": float(pval) if np.isfinite(pval) else np.nan,
        }
        per_cluster.append(cluster_entry)

    l2_raw = _aggregate_cluster_state_metric(per_cluster, "raw_tau")
    l2_centered = _aggregate_cluster_state_metric(per_cluster, "centered_tau")
    l3_lag_span = _aggregate_cluster_state_metric(
        per_cluster,
        "lag_span_mean",
        high_key="high_l3",
        low_key="low_l3",
    )
    l3_pearson = _aggregate_cluster_state_metric(
        per_cluster,
        "pearson_r_median",
        high_key="high_l3",
        low_key="low_l3",
    )
    state_event_counts = {
        "high": int(np.sum(state_masks["high"])),
        "low": int(np.sum(state_masks["low"])),
        "excluded": int(np.sum(event_states == "excluded")),
    }

    rate_bin_summary = [
        {
            "bin_id": int(entry["bin_id"]),
            "n_events": int(entry["n_events"]),
            "rate_per_hour": float(entry["rate_per_hour"]),
            "rate_state": str(entry.get("rate_state", "excluded")),
        }
        for entry in bin_info["rate_bins"]
    ]

    subject_raw_delta = l2_raw.get("delta_high_minus_low", np.nan)
    subject_centered_delta = l2_centered.get("delta_high_minus_low", np.nan)
    subject_lag_span_delta = l3_lag_span.get("delta_high_minus_low", np.nan)
    subject_pearson_delta = l3_pearson.get("delta_high_minus_low", np.nan)
    dominant_cluster_id = int(np.argmax(cluster_counts)) if cluster_counts.size else -1
    dominant_cluster_l1 = next(
        (entry["occupancy_rate"] for entry in per_cluster if int(entry["cluster_id"]) == dominant_cluster_id),
        {
            "n_bins": 0,
            "mean_fraction": np.nan,
            "occupancy_rate_spearman_rho": np.nan,
            "occupancy_rate_spearman_p": np.nan,
        },
    )
    l1_rhos = [
        float(entry["occupancy_rate"]["occupancy_rate_spearman_rho"])
        for entry in per_cluster
        if np.isfinite(entry.get("occupancy_rate", {}).get("occupancy_rate_spearman_rho", np.nan))
    ]

    out: Dict[str, Any] = {
        "step_status": "step23_l1_l2_l3_complete",
        "n_valid_events": int(valid_event_indices.size),
        "n_clusters": int(n_clusters),
        "rate_bin_hours": float(rate_bin_hours),
        "min_events_per_bin": int(min_events_per_bin),
        "n_rate_bins_total": int(bin_info["n_bins_total"]),
        "n_rate_bins_eligible": int(bin_info["n_bins_eligible"]),
        "median_eligible_rate_per_hour": bin_info["median_eligible_rate_per_hour"],
        "high_bin_ids": bin_info["high_bin_ids"],
        "low_bin_ids": bin_info["low_bin_ids"],
        "state_event_counts": state_event_counts,
        "l3_eligible_fraction": (
            float(np.mean(n_part >= int(min_participating_l3))) if n_part.size else np.nan
        ),
        "rate_bin_summary": rate_bin_summary,
        "per_cluster": per_cluster,
        "subject_raw_delta": (
            float(subject_raw_delta) if np.isfinite(subject_raw_delta) else None
        ),
        "subject_centered_delta": (
            float(subject_centered_delta) if np.isfinite(subject_centered_delta) else None
        ),
        "subject_lag_span_delta": (
            float(subject_lag_span_delta) if np.isfinite(subject_lag_span_delta) else None
        ),
        "subject_pearson_r_delta": (
            float(subject_pearson_delta) if np.isfinite(subject_pearson_delta) else None
        ),
        "l2": {
            "raw": l2_raw,
            "centered": l2_centered,
        },
        "l3": {
            "matching_rule": "exact_n_participating_match",
            "lag_span": l3_lag_span,
            "pearson_r": l3_pearson,
            "pearson_r_min_participating": int(min_participating_l3),
        },
        "l1": {
            "dominant_cluster_id": dominant_cluster_id,
            "dominant_cluster": dominant_cluster_l1,
            "max_abs_spearman_rho": float(np.max(np.abs(l1_rhos))) if l1_rhos else np.nan,
            "n_clusters_with_valid_spearman": int(len(l1_rhos)),
        },
    }
    if (
        state_event_counts["high"] == 0
        or state_event_counts["low"] == 0
        or l2_raw["n_clusters_compared"] == 0
    ):
        out["warning"] = "insufficient_high_low_state_coverage"
    return out


_PR4C_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("pre_vs_baseline", "baseline", "pre"),
    ("post_vs_pre", "pre", "post"),
    ("post_vs_baseline", "baseline", "post"),
)


def _build_seizure_proximity_windows(
    event_abs_times: np.ndarray,
    seizure_times: Sequence[float],
    *,
    baseline_hours: Tuple[float, float],
    pre_ictal_hours: Tuple[float, float],
    post_ictal_hours: Tuple[float, float],
) -> Dict[str, Any]:
    """Assign each event to a single (seizure, state) and mark per-pair usability.

    Two contracts that earlier versions got wrong:

    * **Event ownership** (Fix B): we enumerate every legal ``(seizure, state)``
      candidate for an event and break ties by the smallest ``|delta_h|``,
      rather than first picking the nearest seizure and discarding the event
      if that seizure's window happens to exclude it. This recovers boundary
      events that fall outside the nearest seizure's window but inside a
      slightly farther seizure's window.
    * **Window usability** (Fix A): each pair (``pre_vs_baseline``,
      ``post_vs_pre``, ``post_vs_baseline``) is marked usable independently.
      A window with non-empty baseline+pre but empty post still feeds the
      ``pre_vs_baseline`` comparison.

    ``usable_windows`` reports windows where at least one pair is usable; the
    legacy ``usable`` boolean per window means "any pair usable" so that
    downstream cohort code keeps working without per-pair plumbing.
    """
    times = np.asarray(event_abs_times, dtype=float)
    sz_times = np.asarray(sorted(float(x) for x in seizure_times), dtype=float)
    state_names = ("baseline", "pre", "post")
    state_ranges = {
        "baseline": tuple(float(x) for x in baseline_hours),
        "pre": tuple(float(x) for x in pre_ictal_hours),
        "post": tuple(float(x) for x in post_ictal_hours),
    }
    event_states = np.full(times.shape, "excluded", dtype=object)
    event_window_ids = np.full(times.shape, -1, dtype=int)
    windows: List[Dict[str, Any]] = [
        {
            "seizure_id": int(sz_id),
            "seizure_time": float(sz_t),
            "state_event_indices": {name: [] for name in state_names},
        }
        for sz_id, sz_t in enumerate(sz_times)
    ]

    def _state_for(delta_h: float) -> Optional[str]:
        for state_name in state_names:
            lo, hi = state_ranges[state_name]
            if lo <= delta_h < hi:
                return state_name
        return None

    if sz_times.size > 0:
        for ev, event_time in enumerate(times):
            if not np.isfinite(event_time):
                continue
            best: Optional[Tuple[float, int, str]] = None
            for sz_id, sz_t in enumerate(sz_times):
                delta_h = (float(event_time) - float(sz_t)) / 3600.0
                state = _state_for(delta_h)
                if state is None:
                    continue
                key = abs(delta_h)
                if best is None or key < best[0]:
                    best = (key, int(sz_id), state)
            if best is None:
                continue
            _, sz_id, state = best
            windows[sz_id]["state_event_indices"][state].append(int(ev))
            event_states[ev] = state
            event_window_ids[ev] = int(sz_id)

    state_event_counts = {name: 0 for name in state_names}
    usable_windows: List[Dict[str, Any]] = []
    for window in windows:
        state_indices = {
            name: np.asarray(window["state_event_indices"][name], dtype=int)
            for name in state_names
        }
        state_counts = {
            name: int(state_indices[name].size)
            for name in state_names
        }
        for name in state_names:
            state_event_counts[name] += state_counts[name]
        pair_usability = {
            pair: (state_counts[a] > 0 and state_counts[b] > 0)
            for pair, a, b in _PR4C_PAIRS
        }
        usable_any_pair = any(pair_usability.values())
        usable_window = {
            "seizure_id": int(window["seizure_id"]),
            "seizure_time": float(window["seizure_time"]),
            "state_event_indices": state_indices,
            "state_event_counts": state_counts,
            "pair_usability": pair_usability,
            "usable": bool(usable_any_pair),
        }
        if usable_any_pair:
            usable_windows.append(usable_window)
        window.update(usable_window)
    state_event_counts["excluded"] = int(np.sum(event_window_ids < 0))

    return {
        "event_states": event_states,
        "event_window_ids": event_window_ids,
        "windows": windows,
        "usable_windows": usable_windows,
        "state_event_counts": state_event_counts,
        "state_ranges_hours": {
            key: [float(val[0]), float(val[1])]
            for key, val in state_ranges.items()
        },
    }


def _rate_by_template_for_window(
    window: Dict[str, Any],
    cluster_labels: np.ndarray,
    n_clusters: int,
    *,
    state_durations_hours: Dict[str, float],
) -> Dict[str, Any]:
    """Decompose per-state event counts into per-template rate (events/hour).

    Single-window version of PR-4D's gap-aware rate×type: within a seizure
    window, each state is a fixed-duration slab, so rate = count / duration.
    """
    labels = np.asarray(cluster_labels, dtype=int)
    out: Dict[str, Dict[str, Any]] = {}
    for state, duration in state_durations_hours.items():
        idx = np.asarray(
            window["state_event_indices"].get(state, []), dtype=int
        )
        if idx.size:
            counts = np.bincount(labels[idx], minlength=int(n_clusters)).astype(int)
        else:
            counts = np.zeros(int(n_clusters), dtype=int)
        total = int(counts.sum())
        dur = float(max(duration, 1e-9))
        rate = counts / dur
        frac = counts / total if total > 0 else np.full_like(counts, np.nan, dtype=float)
        out[state] = {
            "duration_hours": float(duration),
            "counts_total": int(total),
            "rate_total_per_hour": float(total / dur),
            "counts_by_template": counts.astype(int).tolist(),
            "rate_by_template_per_hour": rate.astype(float).tolist(),
            "fraction_by_template": frac.astype(float).tolist(),
        }
    return out


def _summarize_rate_by_template_records(
    records: List[Dict[str, Any]],
    n_clusters: int,
) -> Dict[str, Any]:
    """Per-subject aggregate of per-window rate×type across usable seizures."""
    if not records:
        return {"n_windows": 0}
    state_names = ("baseline", "pre", "post")

    def _stack(state: str, key: str) -> np.ndarray:
        rows = []
        for rec in records:
            state_rec = rec.get(state, {})
            vals = state_rec.get(key)
            if vals is None:
                continue
            arr = np.asarray(vals, dtype=float)
            if arr.size == int(n_clusters):
                rows.append(arr)
        return np.asarray(rows, dtype=float) if rows else np.zeros((0, int(n_clusters)))

    def _state_totals(state: str) -> List[float]:
        return [
            float(rec.get(state, {}).get("rate_total_per_hour", np.nan))
            for rec in records
            if np.isfinite(rec.get(state, {}).get("rate_total_per_hour", np.nan))
        ]

    per_state: Dict[str, Any] = {}
    for state in state_names:
        rates = _stack(state, "rate_by_template_per_hour")
        fracs = _stack(state, "fraction_by_template")
        totals = _state_totals(state)
        per_state[state] = {
            "n_windows": int(rates.shape[0]),
            "median_rate_per_hour_total": (
                float(np.median(totals)) if totals else np.nan
            ),
            "median_rate_by_template_per_hour": (
                np.nanmedian(rates, axis=0).astype(float).tolist()
                if rates.size else [np.nan] * int(n_clusters)
            ),
            "median_fraction_by_template": (
                np.nanmedian(fracs, axis=0).astype(float).tolist()
                if fracs.size else [np.nan] * int(n_clusters)
            ),
        }

    def _pair_delta(state_a: str, state_b: str) -> Dict[str, Any]:
        rate_a = _stack(state_a, "rate_by_template_per_hour")
        rate_b = _stack(state_b, "rate_by_template_per_hour")
        frac_a = _stack(state_a, "fraction_by_template")
        frac_b = _stack(state_b, "fraction_by_template")
        pairs = min(rate_a.shape[0], rate_b.shape[0])
        if pairs == 0:
            return {
                "n_windows": 0,
                "median_rate_delta_by_template": [np.nan] * int(n_clusters),
                "median_fraction_delta_by_template": [np.nan] * int(n_clusters),
                "max_abs_fraction_delta_template": np.nan,
            }
        d_rate = rate_b[:pairs] - rate_a[:pairs]
        d_frac = frac_b[:pairs] - frac_a[:pairs]
        med_rate = np.nanmedian(d_rate, axis=0).astype(float)
        med_frac = np.nanmedian(d_frac, axis=0).astype(float)
        return {
            "n_windows": int(pairs),
            "median_rate_delta_by_template": med_rate.tolist(),
            "median_fraction_delta_by_template": med_frac.tolist(),
            "max_abs_fraction_delta_template": (
                float(np.nanmax(np.abs(med_frac)))
                if np.any(np.isfinite(med_frac)) else np.nan
            ),
        }

    return {
        "n_windows": int(len(records)),
        "n_templates": int(n_clusters),
        "by_state": per_state,
        "pre_vs_baseline": _pair_delta("baseline", "pre"),
        "post_vs_pre": _pair_delta("pre", "post"),
        "post_vs_baseline": _pair_delta("baseline", "post"),
    }


def _rename_state_pair_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "n_clusters_compared": int(summary.get("n_clusters_compared", 0)),
        "cluster_ids_compared": list(summary.get("cluster_ids_compared", [])),
        "state_a_mean": summary.get("low_mean", np.nan),
        "state_b_mean": summary.get("high_mean", np.nan),
        "delta_state_b_minus_state_a": summary.get("delta_high_minus_low", np.nan),
        "n_clusters_state_b_gt_state_a": int(summary.get("n_clusters_high_gt_low", 0)),
    }


def _compare_two_event_states(
    ranks: np.ndarray,
    relative_lag: np.ndarray,
    bools: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    *,
    state_a_indices: np.ndarray,
    state_b_indices: np.ndarray,
    state_a_name: str,
    state_b_name: str,
    n_sample: int,
    n_seeds: int,
    min_shared_channels: int,
    min_center_participation: int,
    min_participating_l3: int,
    match_seed: int,
) -> Dict[str, Any]:
    """Compare two named event states using the PR-4B L1/L2/L3 contracts."""
    labels = np.asarray(cluster_labels, dtype=int)
    state_a_indices = np.asarray(state_a_indices, dtype=int)
    state_b_indices = np.asarray(state_b_indices, dtype=int)
    n_part = np.sum(np.asarray(bools, dtype=bool), axis=0).astype(int)
    cluster_counts = (
        np.bincount(labels, minlength=int(n_clusters))
        if labels.size else np.zeros(int(n_clusters), dtype=int)
    )
    match_rng = np.random.default_rng(int(match_seed))
    per_cluster: List[Dict[str, Any]] = []

    for cluster_id in range(int(n_clusters)):
        state_a_all = state_a_indices[labels[state_a_indices] == int(cluster_id)]
        state_b_all = state_b_indices[labels[state_b_indices] == int(cluster_id)]
        n_match = int(min(state_a_all.size, state_b_all.size))
        state_a_tau = np.asarray(state_a_all, dtype=int)
        state_b_tau = np.asarray(state_b_all, dtype=int)
        if state_a_tau.size > n_match and n_match > 0:
            state_a_tau = np.sort(match_rng.choice(state_a_tau, size=n_match, replace=False))
        if state_b_tau.size > n_match and n_match > 0:
            state_b_tau = np.sort(match_rng.choice(state_b_tau, size=n_match, replace=False))

        adj_center = int(min(
            int(min_center_participation),
            max(1, n_match // 5),
        ))
        cluster_entry: Dict[str, Any] = {
            "cluster_id": int(cluster_id),
            "n_events_total": int(np.sum(labels == int(cluster_id))),
            "n_matched_tau": int(n_match),
        }
        cluster_entry[state_a_name] = _cluster_state_tau_summary(
            ranks,
            bools,
            state_a_tau,
            n_sample=n_sample,
            n_seeds=n_seeds,
            min_shared_channels=min_shared_channels,
            min_center_participation=adj_center,
        )
        cluster_entry[state_b_name] = _cluster_state_tau_summary(
            ranks,
            bools,
            state_b_tau,
            n_sample=n_sample,
            n_seeds=n_seeds,
            min_shared_channels=min_shared_channels,
            min_center_participation=adj_center,
        )

        lag_span_match = _match_event_indices_by_nparticipating(
            state_b_all,
            state_a_all,
            n_part,
            seed=int(match_seed) + 7919 * int(cluster_id) + 17,
            min_participating=int(min_shared_channels),
        )
        pearson_match = _match_event_indices_by_nparticipating(
            state_b_all,
            state_a_all,
            n_part,
            seed=int(match_seed) + 7919 * int(cluster_id) + 43,
            min_participating=int(min_participating_l3),
        )
        cluster_entry["l3_matching"] = {
            "lag_span": {
                "n_matched": int(lag_span_match["n_matched"]),
                "matched_counts_by_n_participating": lag_span_match["matched_counts_by_n_participating"],
                "min_participating": int(lag_span_match["min_participating"]),
            },
            "pearson_r": {
                "n_matched": int(pearson_match["n_matched"]),
                "matched_counts_by_n_participating": pearson_match["matched_counts_by_n_participating"],
                "min_participating": int(pearson_match["min_participating"]),
            },
        }
        cluster_entry[f"{state_a_name}_l3"] = _cluster_state_l3_summary(
            relative_lag,
            bools,
            lag_span_event_indices=lag_span_match["low_event_indices"],
            pearson_event_indices=pearson_match["low_event_indices"],
            min_lag_span_participating=int(min_shared_channels),
            min_shared_channels_for_r=int(min_participating_l3),
            n_sample_pearson=n_sample,
            pearson_seed=int(match_seed) + 7919 * int(cluster_id) + 101,
        )
        cluster_entry[f"{state_b_name}_l3"] = _cluster_state_l3_summary(
            relative_lag,
            bools,
            lag_span_event_indices=lag_span_match["high_event_indices"],
            pearson_event_indices=pearson_match["high_event_indices"],
            min_lag_span_participating=int(min_shared_channels),
            min_shared_channels_for_r=int(min_participating_l3),
            n_sample_pearson=n_sample,
            pearson_seed=int(match_seed) + 7919 * int(cluster_id) + 137,
        )

        state_a_raw = cluster_entry[state_a_name]["raw_tau"]
        state_b_raw = cluster_entry[state_b_name]["raw_tau"]
        state_a_centered = cluster_entry[state_a_name]["centered_tau"]
        state_b_centered = cluster_entry[state_b_name]["centered_tau"]
        state_a_lag = cluster_entry[f"{state_a_name}_l3"]["lag_span_mean"]
        state_b_lag = cluster_entry[f"{state_b_name}_l3"]["lag_span_mean"]
        state_a_r = cluster_entry[f"{state_a_name}_l3"]["pearson_r_median"]
        state_b_r = cluster_entry[f"{state_b_name}_l3"]["pearson_r_median"]

        cluster_entry["raw_delta_state_b_minus_state_a"] = (
            float(state_b_raw - state_a_raw)
            if np.isfinite(state_a_raw) and np.isfinite(state_b_raw)
            else np.nan
        )
        cluster_entry["centered_delta_state_b_minus_state_a"] = (
            float(state_b_centered - state_a_centered)
            if np.isfinite(state_a_centered) and np.isfinite(state_b_centered)
            else np.nan
        )
        cluster_entry["lag_span_delta_state_b_minus_state_a"] = (
            float(state_b_lag - state_a_lag)
            if np.isfinite(state_a_lag) and np.isfinite(state_b_lag)
            else np.nan
        )
        cluster_entry["pearson_r_delta_state_b_minus_state_a"] = (
            float(state_b_r - state_a_r)
            if np.isfinite(state_a_r) and np.isfinite(state_b_r)
            else np.nan
        )
        cluster_entry["occupancy_fraction"] = {
            state_a_name: (
                float(state_a_all.size / state_a_indices.size)
                if state_a_indices.size else np.nan
            ),
            state_b_name: (
                float(state_b_all.size / state_b_indices.size)
                if state_b_indices.size else np.nan
            ),
        }
        per_cluster.append(cluster_entry)

    l2_raw = _rename_state_pair_summary(
        _aggregate_cluster_state_metric(
            per_cluster,
            "raw_tau",
            high_key=state_b_name,
            low_key=state_a_name,
        )
    )
    l2_centered = _rename_state_pair_summary(
        _aggregate_cluster_state_metric(
            per_cluster,
            "centered_tau",
            high_key=state_b_name,
            low_key=state_a_name,
        )
    )
    l3_lag = _rename_state_pair_summary(
        _aggregate_cluster_state_metric(
            per_cluster,
            "lag_span_mean",
            high_key=f"{state_b_name}_l3",
            low_key=f"{state_a_name}_l3",
        )
    )
    l3_pearson = _rename_state_pair_summary(
        _aggregate_cluster_state_metric(
            per_cluster,
            "pearson_r_median",
            high_key=f"{state_b_name}_l3",
            low_key=f"{state_a_name}_l3",
        )
    )
    dominant_cluster_id = int(np.argmax(cluster_counts)) if cluster_counts.size else -1
    dominant_fraction = next(
        (
            cluster["occupancy_fraction"]
            for cluster in per_cluster
            if int(cluster["cluster_id"]) == dominant_cluster_id
        ),
        {state_a_name: np.nan, state_b_name: np.nan},
    )
    fraction_shifts = [
        float(cluster["occupancy_fraction"][state_b_name] - cluster["occupancy_fraction"][state_a_name])
        for cluster in per_cluster
        if np.isfinite(cluster["occupancy_fraction"].get(state_a_name, np.nan))
        and np.isfinite(cluster["occupancy_fraction"].get(state_b_name, np.nan))
    ]

    out = {
        "state_a_name": state_a_name,
        "state_b_name": state_b_name,
        "state_event_counts": {
            state_a_name: int(state_a_indices.size),
            state_b_name: int(state_b_indices.size),
        },
        "per_cluster": per_cluster,
        "subject_raw_delta": (
            float(l2_raw["delta_state_b_minus_state_a"])
            if np.isfinite(l2_raw["delta_state_b_minus_state_a"]) else None
        ),
        "subject_centered_delta": (
            float(l2_centered["delta_state_b_minus_state_a"])
            if np.isfinite(l2_centered["delta_state_b_minus_state_a"]) else None
        ),
        "subject_lag_span_delta": (
            float(l3_lag["delta_state_b_minus_state_a"])
            if np.isfinite(l3_lag["delta_state_b_minus_state_a"]) else None
        ),
        "subject_pearson_r_delta": (
            float(l3_pearson["delta_state_b_minus_state_a"])
            if np.isfinite(l3_pearson["delta_state_b_minus_state_a"]) else None
        ),
        "l2": {
            "raw": l2_raw,
            "centered": l2_centered,
        },
        "l3": {
            "matching_rule": "exact_n_participating_match",
            "lag_span": l3_lag,
            "pearson_r": l3_pearson,
            "pearson_r_min_participating": int(min_participating_l3),
        },
        "l1": {
            "dominant_cluster_id": int(dominant_cluster_id),
            "dominant_cluster_fraction": {
                state_a_name: dominant_fraction.get(state_a_name, np.nan),
                state_b_name: dominant_fraction.get(state_b_name, np.nan),
                "delta_state_b_minus_state_a": (
                    float(dominant_fraction.get(state_b_name, np.nan) - dominant_fraction.get(state_a_name, np.nan))
                    if np.isfinite(dominant_fraction.get(state_a_name, np.nan))
                    and np.isfinite(dominant_fraction.get(state_b_name, np.nan))
                    else np.nan
                ),
            },
            "max_abs_fraction_shift": (
                float(np.max(np.abs(fraction_shifts))) if fraction_shifts else np.nan
            ),
        },
    }
    if (
        state_a_indices.size == 0
        or state_b_indices.size == 0
        or l2_raw["n_clusters_compared"] == 0
    ):
        out["warning"] = "insufficient_state_coverage"
    return out


def _summarize_state_pair_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {"n_windows": 0}
    state_a_name = str(records[0].get("state_a_name", "state_a"))
    state_b_name = str(records[0].get("state_b_name", "state_b"))

    def _collect(metric_key: str, path: Tuple[str, ...]) -> List[float]:
        values: List[float] = []
        for record in records:
            current: Any = record
            for key in path:
                current = current.get(key, {})
            value = current.get(metric_key, np.nan)
            if np.isfinite(value):
                values.append(float(value))
        return values

    def _wilcoxon_on_deltas(deltas: List[float]) -> Dict[str, Any]:
        if len(deltas) < 2:
            return {"p": np.nan, "n": int(len(deltas))}
        arr = np.asarray(deltas, dtype=float)
        nonzero = arr[np.abs(arr) > 1e-15]
        if nonzero.size < 2:
            return {"p": np.nan, "n": int(nonzero.size)}
        try:
            p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
        return {"p": p, "n": int(nonzero.size)}

    def _pair_summary(path: Tuple[str, ...]) -> Dict[str, Any]:
        state_a = _collect("state_a_mean", path)
        state_b = _collect("state_b_mean", path)
        delta = _collect("delta_state_b_minus_state_a", path)
        wil = _wilcoxon_on_deltas(delta)
        return {
            "state_a_median": float(np.median(state_a)) if state_a else np.nan,
            "state_b_median": float(np.median(state_b)) if state_b else np.nan,
            "delta_state_b_minus_state_a_median": float(np.median(delta)) if delta else np.nan,
            "n_windows_state_b_gt_state_a": int(sum(val > 0 for val in delta)),
            "wilcoxon_p": wil["p"],
            "wilcoxon_n": wil["n"],
        }

    def _fraction_summary() -> Dict[str, Any]:
        state_a_vals: List[float] = []
        state_b_vals: List[float] = []
        delta_vals: List[float] = []
        for record in records:
            frac = record.get("l1", {}).get("dominant_cluster_fraction", {})
            a_val = frac.get(state_a_name, np.nan)
            b_val = frac.get(state_b_name, np.nan)
            d_val = frac.get("delta_state_b_minus_state_a", np.nan)
            if np.isfinite(a_val):
                state_a_vals.append(float(a_val))
            if np.isfinite(b_val):
                state_b_vals.append(float(b_val))
            if np.isfinite(d_val):
                delta_vals.append(float(d_val))
        wil = _wilcoxon_on_deltas(delta_vals)
        return {
            "state_a_median": float(np.median(state_a_vals)) if state_a_vals else np.nan,
            "state_b_median": float(np.median(state_b_vals)) if state_b_vals else np.nan,
            "delta_state_b_minus_state_a_median": (
                float(np.median(delta_vals)) if delta_vals else np.nan
            ),
            "n_windows_state_b_gt_state_a": int(sum(val > 0 for val in delta_vals)),
            "wilcoxon_p": wil["p"],
            "wilcoxon_n": wil["n"],
        }

    return {
        "n_windows": int(len(records)),
        "raw_tau": _pair_summary(("l2", "raw")),
        "centered_tau": _pair_summary(("l2", "centered")),
        "l3": {
            "lag_span": _pair_summary(("l3", "lag_span")),
            "pearson_r": _pair_summary(("l3", "pearson_r")),
        },
        "dominant_cluster_fraction": _fraction_summary(),
    }


SEIZURE_PROXIMITY_CONFIGS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # PR-4C main track: retains the largest number of usable seizures across
    # the cohort (median ~7/subject) while keeping >=400 events/state on
    # median. Data-driven choice; see scripts/pr4c_window_sweep_report.py.
    "main": {
        "baseline_hours": (-4.0, -1.0),
        "pre_ictal_hours": (-1.0, -0.25),
        "post_ictal_hours": (0.25, 1.0),
    },
    # PR-4C auxiliary track: tighter peri-ictal windows for sensitivity
    # analysis. Direction of pre/post deltas should agree with main to
    # claim robustness.
    "auxiliary": {
        "baseline_hours": (-2.0, -0.5),
        "pre_ictal_hours": (-0.5, -1.0 / 12.0),
        "post_ictal_hours": (1.0 / 12.0, 1.0),
    },
}


def _intersect_seconds(
    t0: float,
    t1: float,
    coverage_ranges: Optional[Sequence[Tuple[float, float]]],
) -> float:
    """Return covered seconds inside ``[t0, t1)`` given gap-aware coverage ranges.

    When ``coverage_ranges`` is ``None`` we fall back to nominal ``t1 - t0``,
    so callers without a coverage map keep the legacy fixed-window behaviour.
    """
    if t1 <= t0:
        return 0.0
    if coverage_ranges is None:
        return float(t1 - t0)
    total = 0.0
    for lo, hi in coverage_ranges:
        a = max(t0, float(lo))
        b = min(t1, float(hi))
        if b > a:
            total += float(b - a)
    return total


def compute_seizure_proximity_coupling(
    event_abs_times: np.ndarray,
    ranks: np.ndarray,
    lag_raw: np.ndarray,
    bools: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    *,
    seizure_times: Sequence[float],
    valid_event_indices: Optional[np.ndarray] = None,
    pre_ictal_hours: Tuple[float, float] = (-1.0, -0.25),
    baseline_hours: Tuple[float, float] = (-4.0, -1.0),
    post_ictal_hours: Tuple[float, float] = (0.25, 1.0),
    coverage_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    n_sample: int = 200,
    n_seeds: int = 5,
    min_shared_channels: int = 3,
    min_center_participation: int = 10,
    min_participating_l3: int = 5,
    match_seed: int = 42,
) -> Dict[str, Any]:
    """PR-4C: seizure-proximity comparison for propagation pattern + rate metrics.

    Three corrected contracts (vs. earlier versions):

    * Per-pair window usability (Fix A): a window with non-empty
      baseline+pre but empty post still feeds ``pre_vs_baseline``.
    * Candidate-then-tie-break event ownership (Fix B): boundary events
      that fall outside the nearest seizure's window but inside a
      slightly farther seizure's window are no longer dropped.
    * Gap-aware rate denominators (Fix C): when ``coverage_ranges`` is
      provided, ``rate_by_template`` divides by the actual covered hours
      inside each state's slab, not by the nominal window width.
    """
    if not seizure_times:
        return {"warning": "no_seizure_times"}

    times = np.asarray(event_abs_times, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    lag_raw = np.asarray(lag_raw, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    labels = np.asarray(cluster_labels, dtype=int)
    if ranks.shape != bools.shape or ranks.shape != lag_raw.shape:
        raise ValueError("ranks, lag_raw, and bools must share shape")

    if valid_event_indices is None:
        if labels.size == ranks.shape[1]:
            valid_event_indices = np.arange(ranks.shape[1], dtype=int)
        else:
            valid_event_indices = _valid_event_indices(
                bools,
                min_participating=min_shared_channels,
            )
    valid_event_indices = np.asarray(valid_event_indices, dtype=int)
    if labels.size != valid_event_indices.size:
        raise ValueError("cluster_labels size must equal valid_event_indices size")

    v_times = times[valid_event_indices]
    v_ranks = ranks[:, valid_event_indices]
    v_bools = bools[:, valid_event_indices]
    v_rel = _compute_relative_lag_matrix(lag_raw[:, valid_event_indices], v_bools)
    proximity = _build_seizure_proximity_windows(
        v_times,
        seizure_times,
        baseline_hours=baseline_hours,
        pre_ictal_hours=pre_ictal_hours,
        post_ictal_hours=post_ictal_hours,
    )

    state_hours = {
        "baseline": (float(baseline_hours[0]), float(baseline_hours[1])),
        "pre": (float(pre_ictal_hours[0]), float(pre_ictal_hours[1])),
        "post": (float(post_ictal_hours[0]), float(post_ictal_hours[1])),
    }
    nominal_state_hours = {
        name: float(hi - lo) for name, (lo, hi) in state_hours.items()
    }
    coverage_norm: Optional[List[Tuple[float, float]]] = None
    if coverage_ranges is not None:
        coverage_norm = [
            (float(lo), float(hi))
            for lo, hi in coverage_ranges
            if float(hi) > float(lo)
        ]

    def _per_window_state_hours(seizure_time: float) -> Dict[str, float]:
        if coverage_norm is None:
            return dict(nominal_state_hours)
        out: Dict[str, float] = {}
        for name, (lo, hi) in state_hours.items():
            t0 = float(seizure_time + lo * 3600.0)
            t1 = float(seizure_time + hi * 3600.0)
            covered_seconds = _intersect_seconds(t0, t1, coverage_norm)
            out[name] = covered_seconds / 3600.0
        return out

    comparison_records: Dict[str, List[Dict[str, Any]]] = {
        pair: [] for pair, _, _ in _PR4C_PAIRS
    }
    rate_by_template_records: List[Dict[str, Any]] = []
    seizure_windows: List[Dict[str, Any]] = []
    n_seizures_pair_usable: Dict[str, int] = {pair: 0 for pair, _, _ in _PR4C_PAIRS}
    pair_state_indices_lookup: Dict[str, Tuple[str, str]] = {
        pair: (a, b) for pair, a, b in _PR4C_PAIRS
    }
    for window in proximity["windows"]:
        per_window_hours = _per_window_state_hours(window["seizure_time"])
        rate_decomposition = _rate_by_template_for_window(
            window,
            labels,
            n_clusters,
            state_durations_hours=per_window_hours,
        )
        pair_usability = dict(window.get("pair_usability", {}))
        window_out = {
            "seizure_id": int(window["seizure_id"]),
            "seizure_time": float(window["seizure_time"]),
            "state_event_counts": dict(window["state_event_counts"]),
            "pair_usability": pair_usability,
            "state_covered_hours": dict(per_window_hours),
            "usable": bool(window["usable"]),
            "rate_by_template": rate_decomposition,
        }
        if not window["usable"]:
            seizure_windows.append(window_out)
            continue
        rate_by_template_records.append(rate_decomposition)
        state_indices = {
            name: np.asarray(window["state_event_indices"][name], dtype=int)
            for name in ("baseline", "pre", "post")
        }
        pairwise: Dict[str, Dict[str, Any]] = {}
        for pair_name, state_a_name, state_b_name in _PR4C_PAIRS:
            if not pair_usability.get(pair_name, False):
                continue
            seed_offset = {
                "pre_vs_baseline": 11,
                "post_vs_pre": 29,
                "post_vs_baseline": 53,
            }[pair_name]
            pair_record = _compare_two_event_states(
                v_ranks,
                v_rel,
                v_bools,
                labels,
                n_clusters,
                state_a_indices=state_indices[state_a_name],
                state_b_indices=state_indices[state_b_name],
                state_a_name=state_a_name,
                state_b_name=state_b_name,
                n_sample=n_sample,
                n_seeds=n_seeds,
                min_shared_channels=min_shared_channels,
                min_center_participation=min_center_participation,
                min_participating_l3=min_participating_l3,
                match_seed=int(match_seed)
                + 1000 * int(window["seizure_id"])
                + seed_offset,
            )
            pairwise[pair_name] = pair_record
            comparison_records[pair_name].append(pair_record)
            n_seizures_pair_usable[pair_name] += 1
        window_out["pairwise_comparisons"] = pairwise
        seizure_windows.append(window_out)

    out = {
        "step_status": "pr4c_seizure_proximity_complete",
        "n_valid_events": int(valid_event_indices.size),
        "n_clusters": int(n_clusters),
        "n_seizures_total": int(len(seizure_times)),
        "n_seizures_usable": int(len(proximity["usable_windows"])),
        "n_seizures_pair_usable": n_seizures_pair_usable,
        "coverage_aware_rate": bool(coverage_norm is not None),
        "window_hours": proximity["state_ranges_hours"],
        "state_event_counts": proximity["state_event_counts"],
        "seizure_windows": seizure_windows,
        "comparison_summary": {
            key: _summarize_state_pair_records(records)
            for key, records in comparison_records.items()
        },
        "rate_by_template_summary": _summarize_rate_by_template_records(
            rate_by_template_records,
            int(n_clusters),
        ),
    }
    if not proximity["usable_windows"]:
        out["warning"] = "no_usable_seizure_windows"
    return out


def _summarize_reproducibility(valid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohort-level summary of cross-time template reproducibility."""
    repro_recs = [
        rec["time_split_reproducibility"]
        for rec in valid
        if rec.get("time_split_reproducibility")
        and "splits" in rec.get("time_split_reproducibility", {})
    ]
    if not repro_recs:
        return {"n_subjects": 0}

    grades = [r["reproducibility_grade"] for r in repro_recs]
    grade_counts = {g: int(grades.count(g)) for g in sorted(set(grades))}

    def _collect_split(key: str) -> Tuple[List[float], List[float]]:
        corrs: List[float] = []
        agrees: List[float] = []
        for r in repro_recs:
            s = r["splits"].get(key)
            if not s:
                continue
            mc = s.get("mean_match_corr", np.nan)
            ag = s.get("assignment_agreement", np.nan)
            if np.isfinite(mc):
                corrs.append(float(mc))
            if np.isfinite(ag):
                agrees.append(float(ag))
        return corrs, agrees

    sh_corrs, sh_agrees = _collect_split("first_half_second_half")
    oe_corrs, oe_agrees = _collect_split("odd_even_block")

    n_fwd_rev_full = sum(
        1 for r in repro_recs if r.get("full_data_forward_reverse_pairs", 0) > 0
    )
    n_fwd_rev_reproduced = 0
    for r in repro_recs:
        if r.get("full_data_forward_reverse_pairs", 0) == 0:
            continue
        any_repro = any(
            s.get("forward_reverse_reproduced")
            for s in r.get("splits", {}).values()
            if isinstance(s, dict)
        )
        if any_repro:
            n_fwd_rev_reproduced += 1

    return {
        "n_subjects": len(repro_recs),
        "grade_distribution": grade_counts,
        "split_half": {
            "n_subjects": len(sh_corrs),
            "median_match_corr": float(np.median(sh_corrs)) if sh_corrs else np.nan,
            "median_agreement": float(np.median(sh_agrees)) if sh_agrees else np.nan,
        },
        "odd_even_block": {
            "n_subjects": len(oe_corrs),
            "median_match_corr": float(np.median(oe_corrs)) if oe_corrs else np.nan,
            "median_agreement": float(np.median(oe_agrees)) if oe_agrees else np.nan,
        },
        "forward_reverse": {
            "n_subjects_with_pairs": n_fwd_rev_full,
            "n_reproduced": n_fwd_rev_reproduced,
        },
    }


def _summarize_absolute_lag_validation(valid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohort-level PR-4B Step 0 summary."""
    lag_recs = [
        rec["absolute_lag_validation"]
        for rec in valid
        if rec.get("absolute_lag_validation")
        and "error" not in rec.get("absolute_lag_validation", {})
    ]
    if not lag_recs:
        return {"n_subjects": 0}

    eligible_fraction = [
        float(r["eligible_fraction"])
        for r in lag_recs
        if np.isfinite(r.get("eligible_fraction", np.nan))
    ]
    eligible_median = [
        float(r["eligible_median_r"])
        for r in lag_recs
        if np.isfinite(r.get("eligible_median_r", np.nan))
    ]
    exact_order = [
        float(r["order_validation"]["exact_order_match_fraction"])
        for r in lag_recs
        if np.isfinite(r.get("order_validation", {}).get("exact_order_match_fraction", np.nan))
    ]
    pairwise_order = [
        float(r["order_validation"]["pairwise_order_concordance"])
        for r in lag_recs
        if np.isfinite(r.get("order_validation", {}).get("pairwise_order_concordance", np.nan))
    ]

    bins_summary: Dict[str, Any] = {}
    for bin_label in ("3-4", "5-8", "9+"):
        vals = [
            float(r["within_cluster_pearson_r_by_npart"][bin_label]["median_r"])
            for r in lag_recs
            if bin_label in r.get("within_cluster_pearson_r_by_npart", {})
            and np.isfinite(
                r["within_cluster_pearson_r_by_npart"][bin_label].get("median_r", np.nan)
            )
        ]
        bins_summary[bin_label] = {
            "n_subjects": int(len(vals)),
            "median_r": float(np.median(vals)) if vals else np.nan,
        }

    n_subjects_pass = sum(int(r.get("validation_pass", False)) for r in lag_recs)
    cohort_median = float(np.median(eligible_median)) if eligible_median else np.nan

    dominant_rs = [
        float(r["dominant_cluster_median_r"])
        for r in lag_recs
        if np.isfinite(r.get("dominant_cluster_median_r", np.nan))
    ]
    dominant_median = float(np.median(dominant_rs)) if dominant_rs else np.nan

    return {
        "n_subjects": int(len(lag_recs)),
        "n_subjects_pass": int(n_subjects_pass),
        "eligible_fraction_median": (
            float(np.median(eligible_fraction)) if eligible_fraction else np.nan
        ),
        "eligible_median_r_median": cohort_median,
        "dominant_cluster_median_r_median": dominant_median,
        "cohort_validation_pass": bool(
            np.isfinite(dominant_median) and dominant_median > 0.7
        ),
        "exact_order_match_fraction_median": (
            float(np.median(exact_order)) if exact_order else np.nan
        ),
        "pairwise_order_concordance_median": (
            float(np.median(pairwise_order)) if pairwise_order else np.nan
        ),
        "within_cluster_pearson_r_by_npart": bins_summary,
    }


def _summarize_rate_state_coupling(valid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohort-level PR-4B Step 1-3 summary with subject-level Wilcoxon."""
    coupling_recs = [
        rec["rate_state_coupling"]
        for rec in valid
        if rec.get("rate_state_coupling")
        and "error" not in rec.get("rate_state_coupling", {})
    ]
    if not coupling_recs:
        return {"n_subjects": 0}

    raw_deltas: List[float] = []
    centered_deltas: List[float] = []
    raw_highs: List[float] = []
    raw_lows: List[float] = []
    centered_highs: List[float] = []
    centered_lows: List[float] = []
    lag_span_deltas: List[float] = []
    lag_span_highs: List[float] = []
    lag_span_lows: List[float] = []
    pearson_deltas: List[float] = []
    pearson_highs: List[float] = []
    pearson_lows: List[float] = []
    pearson_hc_deltas: List[float] = []
    pearson_hc_highs: List[float] = []
    pearson_hc_lows: List[float] = []
    eligible_rates: List[float] = []
    l3_eligible_fraction: List[float] = []
    dominant_l1_rhos: List[float] = []
    max_abs_l1_rhos: List[float] = []

    for rec in coupling_recs:
        rd = rec.get("subject_raw_delta")
        if rd is not None and np.isfinite(rd):
            raw_deltas.append(float(rd))
        cd = rec.get("subject_centered_delta")
        if cd is not None and np.isfinite(cd):
            centered_deltas.append(float(cd))
        l2 = rec.get("l2", {})
        rh = l2.get("raw", {}).get("high_mean", np.nan)
        rl = l2.get("raw", {}).get("low_mean", np.nan)
        ch = l2.get("centered", {}).get("high_mean", np.nan)
        cl = l2.get("centered", {}).get("low_mean", np.nan)
        if np.isfinite(rh):
            raw_highs.append(float(rh))
        if np.isfinite(rl):
            raw_lows.append(float(rl))
        if np.isfinite(ch):
            centered_highs.append(float(ch))
        if np.isfinite(cl):
            centered_lows.append(float(cl))
        ls_delta = rec.get("subject_lag_span_delta")
        if ls_delta is not None and np.isfinite(ls_delta):
            lag_span_deltas.append(float(ls_delta))
        pr_delta = rec.get("subject_pearson_r_delta")
        if pr_delta is not None and np.isfinite(pr_delta):
            pearson_deltas.append(float(pr_delta))
        l3 = rec.get("l3", {})
        ls = l3.get("lag_span", {})
        pr = l3.get("pearson_r", {})
        ls_high = ls.get("high_mean", np.nan)
        ls_low = ls.get("low_mean", np.nan)
        pr_high = pr.get("high_mean", np.nan)
        pr_low = pr.get("low_mean", np.nan)
        if np.isfinite(ls_high):
            lag_span_highs.append(float(ls_high))
        if np.isfinite(ls_low):
            lag_span_lows.append(float(ls_low))
        if np.isfinite(pr_high):
            pearson_highs.append(float(pr_high))
        if np.isfinite(pr_low):
            pearson_lows.append(float(pr_low))
        if rec.get("l3_validation_pass") is True:
            if np.isfinite(pr_high):
                pearson_hc_highs.append(float(pr_high))
            if np.isfinite(pr_low):
                pearson_hc_lows.append(float(pr_low))
            if pr_delta is not None and np.isfinite(pr_delta):
                pearson_hc_deltas.append(float(pr_delta))
        l3ef = rec.get("l3_eligible_fraction", np.nan)
        if np.isfinite(l3ef):
            l3_eligible_fraction.append(float(l3ef))
        er = rec.get("median_eligible_rate_per_hour", np.nan)
        if np.isfinite(er):
            eligible_rates.append(float(er))
        dom_rho = rec.get("l1", {}).get("dominant_cluster", {}).get("occupancy_rate_spearman_rho", np.nan)
        if np.isfinite(dom_rho):
            dominant_l1_rhos.append(float(dom_rho))
        max_abs_rho = rec.get("l1", {}).get("max_abs_spearman_rho", np.nan)
        if np.isfinite(max_abs_rho):
            max_abs_l1_rhos.append(float(max_abs_rho))

    def _wilcoxon_on_deltas(deltas: List[float]) -> Dict[str, Any]:
        if len(deltas) < 2:
            return {"p": np.nan, "n": int(len(deltas))}
        arr = np.asarray(deltas, dtype=float)
        nonzero = arr[np.abs(arr) > 1e-15]
        if nonzero.size < 2:
            return {"p": np.nan, "n": int(nonzero.size)}
        try:
            p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
        return {"p": p, "n": int(nonzero.size)}

    raw_wilcoxon = _wilcoxon_on_deltas(raw_deltas)
    centered_wilcoxon = _wilcoxon_on_deltas(centered_deltas)

    return {
        "n_subjects": int(len(coupling_recs)),
        "n_subjects_with_raw_l2": int(len(raw_deltas)),
        "n_subjects_with_centered_l2": int(len(centered_deltas)),
        "median_eligible_rate_per_hour": (
            float(np.median(eligible_rates)) if eligible_rates else np.nan
        ),
        "raw_tau": {
            "high_median": float(np.median(raw_highs)) if raw_highs else np.nan,
            "low_median": float(np.median(raw_lows)) if raw_lows else np.nan,
            "delta_high_minus_low_median": (
                float(np.median(raw_deltas)) if raw_deltas else np.nan
            ),
            "n_subjects_high_gt_low": int(sum(val > 0 for val in raw_deltas)),
            "wilcoxon_p": raw_wilcoxon["p"],
            "wilcoxon_n": raw_wilcoxon["n"],
        },
        "centered_tau": {
            "high_median": float(np.median(centered_highs)) if centered_highs else np.nan,
            "low_median": float(np.median(centered_lows)) if centered_lows else np.nan,
            "delta_high_minus_low_median": (
                float(np.median(centered_deltas)) if centered_deltas else np.nan
            ),
            "n_subjects_high_gt_low": int(sum(val > 0 for val in centered_deltas)),
            "wilcoxon_p": centered_wilcoxon["p"],
            "wilcoxon_n": centered_wilcoxon["n"],
        },
        "l3": {
            "eligible_fraction_median": (
                float(np.median(l3_eligible_fraction)) if l3_eligible_fraction else np.nan
            ),
            "n_subjects_high_confidence": int(
                sum(rec.get("l3_validation_pass") is True for rec in coupling_recs)
            ),
            "lag_span": {
                "high_median": float(np.median(lag_span_highs)) if lag_span_highs else np.nan,
                "low_median": float(np.median(lag_span_lows)) if lag_span_lows else np.nan,
                "delta_high_minus_low_median": (
                    float(np.median(lag_span_deltas)) if lag_span_deltas else np.nan
                ),
                "n_subjects_high_gt_low": int(sum(val > 0 for val in lag_span_deltas)),
                "wilcoxon_p": _wilcoxon_on_deltas(lag_span_deltas)["p"],
                "wilcoxon_n": _wilcoxon_on_deltas(lag_span_deltas)["n"],
            },
            "pearson_r_exploratory": {
                "high_median": float(np.median(pearson_highs)) if pearson_highs else np.nan,
                "low_median": float(np.median(pearson_lows)) if pearson_lows else np.nan,
                "delta_high_minus_low_median": (
                    float(np.median(pearson_deltas)) if pearson_deltas else np.nan
                ),
                "n_subjects_high_gt_low": int(sum(val > 0 for val in pearson_deltas)),
                "wilcoxon_p": _wilcoxon_on_deltas(pearson_deltas)["p"],
                "wilcoxon_n": _wilcoxon_on_deltas(pearson_deltas)["n"],
            },
            "pearson_r_high_confidence": {
                "high_median": (
                    float(np.median(pearson_hc_highs)) if pearson_hc_highs else np.nan
                ),
                "low_median": (
                    float(np.median(pearson_hc_lows)) if pearson_hc_lows else np.nan
                ),
                "delta_high_minus_low_median": (
                    float(np.median(pearson_hc_deltas)) if pearson_hc_deltas else np.nan
                ),
                "n_subjects_high_gt_low": int(sum(val > 0 for val in pearson_hc_deltas)),
                "wilcoxon_p": _wilcoxon_on_deltas(pearson_hc_deltas)["p"],
                "wilcoxon_n": _wilcoxon_on_deltas(pearson_hc_deltas)["n"],
            },
        },
        "l1": {
            "dominant_cluster_rho_median": (
                float(np.median(dominant_l1_rhos)) if dominant_l1_rhos else np.nan
            ),
            "n_subjects_dominant_positive": int(sum(val > 0 for val in dominant_l1_rhos)),
            "max_abs_rho_median": (
                float(np.median(max_abs_l1_rhos)) if max_abs_l1_rhos else np.nan
            ),
        },
    }


# ---------------------------------------------------------------------------
# PR-4A: temporal cluster dynamics
# ---------------------------------------------------------------------------


def _dataset_timezone(dataset: str) -> str:
    """Canonical timezone used for local-clock summaries."""
    return "Asia/Shanghai" if dataset == "yuquan" else "Europe/Berlin"


def _epoch_to_local_hour(epoch_sec: float, timezone_name: str) -> float:
    """Convert Unix epoch to fractional local hour."""
    if not np.isfinite(epoch_sec):
        return float("nan")
    dt = datetime.datetime.fromtimestamp(float(epoch_sec), tz=ZoneInfo(timezone_name))
    return (
        float(dt.hour)
        + float(dt.minute) / 60.0
        + float(dt.second) / 3600.0
        + float(dt.microsecond) / 3.6e9
    )


def _classify_day_night(
    epoch_sec: float,
    *,
    timezone_name: str,
    day_start_hour: int,
    night_start_hour: int,
) -> str:
    hour = _epoch_to_local_hour(epoch_sec, timezone_name)
    if not np.isfinite(hour):
        return "unknown"
    return "day" if int(day_start_hour) <= hour < int(night_start_hour) else "night"


def _normalized_entropy(prob: np.ndarray) -> float:
    """Permutation-invariant concentration summary bounded to [0, 1]."""
    prob = np.asarray(prob, dtype=float)
    finite = prob[np.isfinite(prob)]
    if finite.size == 0:
        return float("nan")
    if finite.size == 1:
        return 0.0
    positive = finite[finite > 0]
    if positive.size == 0:
        return 0.0
    ent = float(-np.sum(positive * np.log(positive)))
    max_ent = float(np.log(finite.size))
    if max_ent <= 0:
        return 0.0
    return ent / max_ent


def _safe_fraction_vector(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = float(np.sum(counts))
    if total <= 0:
        return np.full(counts.shape, np.nan, dtype=float)
    return counts / total


def _summarize_fraction_vector(counts: np.ndarray) -> Dict[str, Any]:
    counts = np.asarray(counts, dtype=int)
    frac = _safe_fraction_vector(counts)
    return {
        "n_events": int(np.sum(counts)),
        "cluster_counts": counts.astype(int).tolist(),
        "cluster_fractions": frac.tolist(),
        "dominant_fraction": float(np.nanmax(frac)) if np.any(np.isfinite(frac)) else np.nan,
        "normalized_entropy": _normalized_entropy(frac),
    }


def compute_temporal_cluster_dynamics(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    dataset: str,
    coverage_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    bin_hours: float = 1.0,
    formal_day_range: Tuple[int, int] = (8, 20),
) -> Dict[str, Any]:
    """Fixed-template occupancy trajectories and day/night summaries.

    The input labels must come from a single fixed template set for the subject.
    This function only bins those labels over time; it never re-clusters per bin.
    """
    if bin_hours <= 0:
        raise ValueError("bin_hours must be > 0")
    day_start_hour, night_start_hour = formal_day_range
    if not (0 <= int(day_start_hour) < int(night_start_hour) <= 24):
        raise ValueError("formal_day_range must satisfy 0 <= start < end <= 24")

    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    if times.shape[0] != labels.shape[0]:
        raise ValueError("event_abs_times and cluster_labels must have the same length")

    valid = np.isfinite(times) & (labels >= 0) & (labels < int(n_clusters))
    times = times[valid]
    labels = labels[valid]
    timezone_name = _dataset_timezone(dataset)
    bin_sec = float(bin_hours) * 3600.0

    if times.size == 0:
        return {
            "dataset": dataset,
            "timezone_name": timezone_name,
            "day_night_rule": f"day={day_start_hour:02d}:00-{night_start_hour:02d}:00 local",
            "bin_hours": float(bin_hours),
            "n_events_used": 0,
            "n_clusters": int(n_clusters),
            "timeline_bins": [],
            "day_night_summary": {
                "day": _summarize_fraction_vector(np.zeros(int(n_clusters), dtype=int)),
                "night": _summarize_fraction_vector(np.zeros(int(n_clusters), dtype=int)),
                "total_variation_distance": np.nan,
            },
        }

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    labels = labels[order]

    t_min = float(times[0])
    t_max = float(times[-1])
    if coverage_ranges:
        raw_ranges = [
            (float(start), float(end))
            for start, end in coverage_ranges
            if np.isfinite(start) and np.isfinite(end) and float(end) >= float(start)
        ]
    else:
        raw_ranges = [(t_min, t_max)]

    selected_bin_ids = set()
    for start, end in raw_ranges:
        start_bin = int(np.floor(float(start) / bin_sec))
        end_bin = int(np.ceil(float(end) / bin_sec) - 1)
        for bin_id in range(start_bin, end_bin + 1):
            selected_bin_ids.add(int(bin_id))
    if not selected_bin_ids:
        start_bin = int(np.floor(t_min / bin_sec))
        end_bin = int(np.floor(t_max / bin_sec))
        selected_bin_ids.update(range(start_bin, end_bin + 1))

    sorted_bin_ids = np.array(sorted(selected_bin_ids), dtype=int)
    counts = np.zeros((sorted_bin_ids.size, int(n_clusters)), dtype=int)
    totals = np.zeros(sorted_bin_ids.size, dtype=int)
    bin_lookup = {int(bin_id): idx for idx, bin_id in enumerate(sorted_bin_ids)}

    raw_event_bins = np.floor(times / bin_sec).astype(int)
    for raw_bin, label in zip(raw_event_bins, labels):
        if int(raw_bin) not in bin_lookup:
            continue
        row = bin_lookup[int(raw_bin)]
        counts[row, int(label)] += 1
        totals[row] += 1

    edges = sorted_bin_ids.astype(float) * bin_sec
    centers = edges + 0.5 * bin_sec
    timeline_start = float(edges[0])
    fractions = np.full((sorted_bin_ids.size, int(n_clusters)), np.nan, dtype=float)
    valid_bins = totals > 0
    fractions[valid_bins] = counts[valid_bins] / totals[valid_bins, None]

    timeline_bins: List[Dict[str, Any]] = []
    for idx, start_edge in enumerate(edges):
        center = float(centers[idx])
        local_hour = _epoch_to_local_hour(center, timezone_name)
        timeline_bins.append(
            {
                "bin_id": int(sorted_bin_ids[idx]),
                "start_epoch": float(start_edge),
                "end_epoch": float(start_edge + bin_sec),
                "center_epoch": center,
                "hours_from_timeline_start": float((center - timeline_start) / 3600.0),
                "local_hour": local_hour,
                "day_night": _classify_day_night(
                    center,
                    timezone_name=timezone_name,
                    day_start_hour=day_start_hour,
                    night_start_hour=night_start_hour,
                ),
                "n_events": int(totals[idx]),
                "cluster_counts": counts[idx].astype(int).tolist(),
                "cluster_fractions": fractions[idx].tolist(),
            }
        )

    event_day_mask = np.array(
        [
            _classify_day_night(
                t,
                timezone_name=timezone_name,
                day_start_hour=day_start_hour,
                night_start_hour=night_start_hour,
            )
            == "day"
            for t in times
        ],
        dtype=bool,
    )
    day_counts = np.bincount(labels[event_day_mask], minlength=int(n_clusters))
    night_counts = np.bincount(labels[~event_day_mask], minlength=int(n_clusters))
    day_summary = _summarize_fraction_vector(day_counts)
    night_summary = _summarize_fraction_vector(night_counts)
    tv_distance = np.nan
    day_frac = np.asarray(day_summary["cluster_fractions"], dtype=float)
    night_frac = np.asarray(night_summary["cluster_fractions"], dtype=float)
    if np.any(np.isfinite(day_frac)) and np.any(np.isfinite(night_frac)):
        tv_distance = float(0.5 * np.nansum(np.abs(day_frac - night_frac)))

    return {
        "dataset": dataset,
        "timezone_name": timezone_name,
        "day_night_rule": f"day={day_start_hour:02d}:00-{night_start_hour:02d}:00 local",
        "bin_hours": float(bin_hours),
        "n_events_used": int(times.size),
        "n_clusters": int(n_clusters),
        "first_event_epoch": t_min,
        "last_event_epoch": t_max,
        "timeline_start_epoch": timeline_start,
        "timeline_bins": timeline_bins,
        "day_night_summary": {
            "day": day_summary,
            "night": night_summary,
            "total_variation_distance": tv_distance,
        },
    }


def _sanitize_coverage_ranges(
    coverage_ranges: Optional[Sequence[Tuple[float, float]]],
    *,
    t_min: float,
    t_max: float,
) -> List[Tuple[float, float]]:
    if coverage_ranges:
        raw = [
            (float(start), float(end))
            for start, end in coverage_ranges
            if np.isfinite(start) and np.isfinite(end) and float(end) >= float(start)
        ]
    else:
        raw = [(float(t_min), float(t_max))]
    if not raw:
        return [(float(t_min), float(t_max))]
    raw.sort(key=lambda pair: pair[0])
    merged: List[Tuple[float, float]] = []
    for start, end in raw:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            prev_start, prev_end = merged[-1]
            merged[-1] = (prev_start, max(prev_end, end))
    return merged


def _build_continuous_grid(
    ranges: Sequence[Tuple[float, float]],
    *,
    step_sec: float,
) -> np.ndarray:
    if step_sec <= 0:
        raise ValueError("step_sec must be > 0")
    points: List[np.ndarray] = []
    for start, end in ranges:
        if end < start:
            continue
        if np.isclose(end, start):
            points.append(np.array([float(start)], dtype=float))
            continue
        grid = np.arange(float(start), float(end) + 0.5 * step_sec, step_sec, dtype=float)
        if grid.size == 0 or grid[-1] < float(end):
            grid = np.append(grid, float(end))
        points.append(grid)
    if not points:
        return np.array([], dtype=float)
    merged = np.unique(np.concatenate(points))
    return np.sort(merged.astype(float))


def _run_length_stats(labels: np.ndarray) -> Dict[str, Any]:
    labels = np.asarray(labels, dtype=int)
    if labels.size == 0:
        return {
            "n_runs": 0,
            "dwell_events_median": np.nan,
            "dwell_events_q1": np.nan,
            "dwell_events_q3": np.nan,
        }
    runs: List[int] = []
    run_len = 1
    for idx in range(1, labels.size):
        if int(labels[idx]) == int(labels[idx - 1]):
            run_len += 1
        else:
            runs.append(int(run_len))
            run_len = 1
    runs.append(int(run_len))
    runs_arr = np.asarray(runs, dtype=float)
    return {
        "n_runs": int(len(runs)),
        "dwell_events_median": float(np.median(runs_arr)) if runs_arr.size else np.nan,
        "dwell_events_q1": float(np.percentile(runs_arr, 25)) if runs_arr.size else np.nan,
        "dwell_events_q3": float(np.percentile(runs_arr, 75)) if runs_arr.size else np.nan,
    }


def compute_continuous_template_dynamics(
    event_abs_times: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    dataset: str,
    *,
    coverage_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    smoothing_hours: float = 2.0,
    bin_hours: float = 1.0,
) -> Dict[str, Any]:
    """PR-4D: per-template absolute event rate over time.

    Produces two complementary views of the same data:

    * **rate_curve** — Gaussian KDE rate estimate (events/hr) for each
      template on a fine grid.  Formula at grid point *t* for template *c*:

          λ_c(t) = (3600 / (h·√(2π))) · Σ_i exp(−(t−t_i)²/(2h²)) · 𝟙[label_i=c]

      where *h* = ``smoothing_hours × 3600`` seconds.

    * **histogram** — raw event counts per template in fixed-width bins
      (width = ``bin_hours``).

    Together they answer: "when total rate changes, which template is
    responsible?"  Normalised fractions hide this because they always
    sum to 1.
    """
    if smoothing_hours <= 0:
        raise ValueError("smoothing_hours must be > 0")
    if bin_hours <= 0:
        raise ValueError("bin_hours must be > 0")

    times = np.asarray(event_abs_times, dtype=float)
    labels = np.asarray(cluster_labels, dtype=int)
    nc = int(n_clusters)
    if times.shape[0] != labels.shape[0]:
        raise ValueError("event_abs_times and cluster_labels must have the same length")

    valid = np.isfinite(times) & (labels >= 0) & (labels < nc)
    times = times[valid]
    labels = labels[valid]

    timezone_name = _dataset_timezone(dataset)
    _nan_list = lambda: [float("nan")] * nc
    if times.size == 0:
        return {
            "dataset": dataset,
            "timezone_name": timezone_name,
            "smoothing_hours": float(smoothing_hours),
            "bin_hours": float(bin_hours),
            "n_events_used": 0,
            "n_clusters": nc,
            "rate_curve": {
                "grid_hours": [],
                "total_rate": [],
                "per_template_rate": [[] for _ in range(nc)],
            },
            "histogram": {
                "bin_center_hours": [],
                "bin_width_hours": [],
                "total_count": [],
                "per_template_count": [[] for _ in range(nc)],
            },
            "summary": {
                "per_template_event_count": [0] * nc,
                "per_template_mean_rate": _nan_list(),
                "per_template_rate_fraction": _nan_list(),
                "total_mean_rate": float("nan"),
                "dominant_template_id": -1,
                "dominant_rate_fraction": float("nan"),
            },
        }

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    labels = labels[order]

    t_min = float(times[0])
    t_max = float(times[-1])
    ranges = _sanitize_coverage_ranges(coverage_ranges, t_min=t_min, t_max=t_max)

    # ---- KDE rate curves ----
    bandwidth_sec = float(smoothing_hours) * 3600.0
    step_sec = max(300.0, bandwidth_sec / 4.0)
    norm = 3600.0 / (bandwidth_sec * np.sqrt(2.0 * np.pi))

    indicator = np.zeros((nc, times.size), dtype=float)
    for c in range(nc):
        indicator[c, labels == c] = 1.0

    grid_origin = float(ranges[0][0])
    grid_parts: List[np.ndarray] = []
    rate_parts: List[np.ndarray] = []
    for ri, (start, end) in enumerate(ranges):
        seg_grid = _build_continuous_grid([(start, end)], step_sec=step_sec)
        if seg_grid.size == 0:
            continue
        seg_rate = np.zeros((nc, seg_grid.size), dtype=float)
        for gi, t in enumerate(seg_grid):
            delta = (times - float(t)) / bandwidth_sec
            kernel = np.exp(-0.5 * delta * delta)
            seg_rate[:, gi] = indicator.dot(kernel) * norm
        grid_parts.append(seg_grid)
        rate_parts.append(seg_rate)
        if ri < len(ranges) - 1:
            grid_parts.append(np.array([float("nan")], dtype=float))
            rate_parts.append(np.full((nc, 1), np.nan, dtype=float))

    grid = np.concatenate(grid_parts) if grid_parts else np.array([], dtype=float)
    per_template_rate = (
        np.concatenate(rate_parts, axis=1) if rate_parts else np.zeros((nc, 0), dtype=float)
    )
    total_rate = np.nansum(per_template_rate, axis=0)
    total_rate[np.all(np.isnan(per_template_rate), axis=0)] = np.nan
    grid_hours = ((grid - grid_origin) / 3600.0).tolist()

    # ---- Histogram bins ----
    bin_sec = float(bin_hours) * 3600.0
    hist_centers: List[float] = []
    hist_widths: List[float] = []
    hist_counts: List[np.ndarray] = []
    for start, end in ranges:
        bin_edges = np.arange(float(start), float(end) + 0.5 * bin_sec, bin_sec)
        if bin_edges.size < 2:
            bin_edges = np.array([float(start), float(end)], dtype=float)
        if bin_edges[-1] < float(end):
            bin_edges = np.append(bin_edges, float(end))
        n_bins = len(bin_edges) - 1
        seg_counts = np.zeros((nc, n_bins), dtype=int)
        for c in range(nc):
            seg_counts[c], _ = np.histogram(times[labels == c], bins=bin_edges)
        hist_counts.append(seg_counts)
        hist_centers.extend((0.5 * (bin_edges[:-1] + bin_edges[1:])).tolist())
        hist_widths.extend((bin_edges[1:] - bin_edges[:-1]).tolist())

    per_template_count = (
        np.concatenate(hist_counts, axis=1) if hist_counts else np.zeros((nc, 0), dtype=int)
    )
    total_count = per_template_count.sum(axis=0)
    bin_center_hours = ((np.asarray(hist_centers, dtype=float) - grid_origin) / 3600.0).tolist()

    # ---- Summary ----
    duration_hr = float(sum((end - start) for start, end in ranges) / 3600.0)
    total_n = int(times.size)
    per_templ_cnt = [int(np.sum(labels == c)) for c in range(nc)]
    per_templ_rate = [
        float(cnt / duration_hr) if np.isfinite(duration_hr) and duration_hr > 0
        else float("nan")
        for cnt in per_templ_cnt
    ]
    per_templ_frac = [
        float(cnt / total_n) if total_n > 0 else float("nan")
        for cnt in per_templ_cnt
    ]
    dom_id = int(np.argmax(per_templ_cnt))

    return {
        "dataset": dataset,
        "timezone_name": timezone_name,
        "smoothing_hours": float(smoothing_hours),
        "bin_hours": float(bin_hours),
        "n_events_used": total_n,
        "n_clusters": nc,
        "first_event_epoch": t_min,
        "last_event_epoch": t_max,
        "duration_hours": duration_hr,
        "coverage_ranges_hours": [
            [float((start - grid_origin) / 3600.0), float((end - grid_origin) / 3600.0)]
            for start, end in ranges
        ],
        "rate_curve": {
            "grid_hours": grid_hours,
            "total_rate": total_rate.tolist(),
            "per_template_rate": per_template_rate.tolist(),
        },
        "histogram": {
            "bin_center_hours": bin_center_hours,
            "bin_width_hours": (np.asarray(hist_widths, dtype=float) / 3600.0).tolist(),
            "total_count": total_count.tolist(),
            "per_template_count": per_template_count.tolist(),
        },
        "summary": {
            "per_template_event_count": per_templ_cnt,
            "per_template_mean_rate": per_templ_rate,
            "per_template_rate_fraction": per_templ_frac,
            "total_mean_rate": (
                float(total_n / duration_hr)
                if np.isfinite(duration_hr) and duration_hr > 0
                else float("nan")
            ),
            "dominant_template_id": dom_id,
            "dominant_rate_fraction": per_templ_frac[dom_id],
        },
    }


def _summarize_temporal_dynamics(valid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohort summary for PR-4A using label-invariant subject-level summaries."""
    temporal_recs = [
        rec["temporal_dynamics"]
        for rec in valid
        if rec.get("temporal_dynamics") and rec["temporal_dynamics"].get("day_night_summary")
    ]
    if not temporal_recs:
        return {"n_subjects": 0}

    paired_dom_day: List[float] = []
    paired_dom_night: List[float] = []
    paired_ent_day: List[float] = []
    paired_ent_night: List[float] = []
    tv_values: List[float] = []

    for td in temporal_recs:
        dn = td.get("day_night_summary", {})
        day = dn.get("day", {})
        night = dn.get("night", {})
        dom_day = day.get("dominant_fraction", np.nan)
        dom_night = night.get("dominant_fraction", np.nan)
        ent_day = day.get("normalized_entropy", np.nan)
        ent_night = night.get("normalized_entropy", np.nan)
        if (
            day.get("n_events", 0) > 0
            and night.get("n_events", 0) > 0
            and np.isfinite(dom_day)
            and np.isfinite(dom_night)
        ):
            paired_dom_day.append(float(dom_day))
            paired_dom_night.append(float(dom_night))
        if (
            day.get("n_events", 0) > 0
            and night.get("n_events", 0) > 0
            and np.isfinite(ent_day)
            and np.isfinite(ent_night)
        ):
            paired_ent_day.append(float(ent_day))
            paired_ent_night.append(float(ent_night))
        tv = dn.get("total_variation_distance", np.nan)
        if np.isfinite(tv):
            tv_values.append(float(tv))

    dom_p = np.nan
    ent_p = np.nan
    if len(paired_dom_day) >= 2:
        try:
            dom_p = float(wilcoxon(np.asarray(paired_dom_day), np.asarray(paired_dom_night)).pvalue)
        except Exception:
            dom_p = np.nan
    if len(paired_ent_day) >= 2:
        try:
            ent_p = float(wilcoxon(np.asarray(paired_ent_day), np.asarray(paired_ent_night)).pvalue)
        except Exception:
            ent_p = np.nan

    return {
        "n_subjects": int(len(temporal_recs)),
        "n_subjects_with_day_night": int(len(paired_dom_day)),
        "dominant_fraction": {
            "day_median": float(np.median(paired_dom_day)) if paired_dom_day else np.nan,
            "night_median": float(np.median(paired_dom_night)) if paired_dom_night else np.nan,
            "median_day_minus_night": float(
                np.median(np.asarray(paired_dom_day) - np.asarray(paired_dom_night))
            )
            if paired_dom_day
            else np.nan,
            "wilcoxon_p": dom_p,
        },
        "normalized_entropy": {
            "day_median": float(np.median(paired_ent_day)) if paired_ent_day else np.nan,
            "night_median": float(np.median(paired_ent_night)) if paired_ent_night else np.nan,
            "median_day_minus_night": float(
                np.median(np.asarray(paired_ent_day) - np.asarray(paired_ent_night))
            )
            if paired_ent_day
            else np.nan,
            "wilcoxon_p": ent_p,
        },
        "day_night_total_variation": {
            "median": float(np.median(tv_values)) if tv_values else np.nan,
            "q1": float(np.percentile(tv_values, 25)) if tv_values else np.nan,
            "q3": float(np.percentile(tv_values, 75)) if tv_values else np.nan,
        },
    }


def _summarize_temporal_dynamics_followup(valid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohort summary for PR-4D per-template rate decomposition."""
    recs = [
        rec.get("temporal_dynamics_followup")
        for rec in valid
        if isinstance(rec.get("temporal_dynamics_followup"), dict)
        and "summary" in rec.get("temporal_dynamics_followup", {})
    ]
    if not recs:
        return {"n_subjects": 0}

    dom_fracs: List[float] = []
    total_rates: List[float] = []
    for rec in recs:
        s = rec.get("summary", {})
        df = s.get("dominant_rate_fraction")
        tr = s.get("total_mean_rate")
        if df is not None and np.isfinite(df):
            dom_fracs.append(float(df))
        if tr is not None and np.isfinite(tr):
            total_rates.append(float(tr))

    return {
        "n_subjects": len(recs),
        "dominant_rate_fraction_median": float(np.median(dom_fracs)) if dom_fracs else np.nan,
        "dominant_rate_fraction_range": (
            [float(np.min(dom_fracs)), float(np.max(dom_fracs))] if dom_fracs else [np.nan, np.nan]
        ),
        "total_mean_rate_median": float(np.median(total_rates)) if total_rates else np.nan,
    }


def _summarize_seizure_proximity(
    valid: List[Dict[str, Any]],
    *,
    coupling_key: str = "seizure_proximity_coupling",
) -> Dict[str, Any]:
    """Cohort summary for PR-4C seizure-proximity coupling.

    ``coupling_key`` selects which per-subject field to aggregate:
    ``seizure_proximity_coupling`` (main) or ``seizure_proximity_coupling_auxiliary``.
    """
    recs = [
        rec.get(coupling_key)
        for rec in valid
        if isinstance(rec.get(coupling_key), dict)
        and "comparison_summary" in rec.get(coupling_key, {})
    ]
    if not recs:
        return {"n_subjects": 0}

    usable_windows = [
        int(rec.get("n_seizures_usable", 0))
        for rec in recs
    ]

    def _wilcoxon_on_deltas(deltas: List[float]) -> Dict[str, Any]:
        if len(deltas) < 2:
            return {"p": np.nan, "n": int(len(deltas))}
        arr = np.asarray(deltas, dtype=float)
        nonzero = arr[np.abs(arr) > 1e-15]
        if nonzero.size < 2:
            return {"p": np.nan, "n": int(nonzero.size)}
        try:
            p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
        except Exception:
            p = np.nan
        return {"p": p, "n": int(nonzero.size)}

    def _pair_metric(pair_recs: List[Dict[str, Any]], path: Tuple[str, ...]) -> Dict[str, Any]:
        state_a_vals: List[float] = []
        state_b_vals: List[float] = []
        delta_vals: List[float] = []
        for rec in pair_recs:
            current: Any = rec
            for key in path:
                current = current.get(key, {})
            a_val = current.get("state_a_median", np.nan)
            b_val = current.get("state_b_median", np.nan)
            d_val = current.get("delta_state_b_minus_state_a_median", np.nan)
            if np.isfinite(a_val):
                state_a_vals.append(float(a_val))
            if np.isfinite(b_val):
                state_b_vals.append(float(b_val))
            if np.isfinite(d_val):
                delta_vals.append(float(d_val))
        wil = _wilcoxon_on_deltas(delta_vals)
        return {
            "state_a_median": float(np.median(state_a_vals)) if state_a_vals else np.nan,
            "state_b_median": float(np.median(state_b_vals)) if state_b_vals else np.nan,
            "delta_state_b_minus_state_a_median": (
                float(np.median(delta_vals)) if delta_vals else np.nan
            ),
            "n_subjects_state_b_gt_state_a": int(sum(val > 0 for val in delta_vals)),
            "wilcoxon_p": wil["p"],
            "wilcoxon_n": wil["n"],
        }

    def _pair_summary(pair_name: str) -> Dict[str, Any]:
        pair_recs = [
            rec["comparison_summary"][pair_name]
            for rec in recs
            if pair_name in rec.get("comparison_summary", {})
        ]
        if not pair_recs:
            return {"n_subjects": 0}
        return {
            "n_subjects": int(len(pair_recs)),
            "raw_tau": _pair_metric(pair_recs, ("raw_tau",)),
            "centered_tau": _pair_metric(pair_recs, ("centered_tau",)),
            "l3": {
                "lag_span": _pair_metric(pair_recs, ("l3", "lag_span")),
                "pearson_r": _pair_metric(pair_recs, ("l3", "pearson_r")),
            },
            "dominant_cluster_fraction": _pair_metric(pair_recs, ("dominant_cluster_fraction",)),
        }

    def _rate_by_template_summary() -> Dict[str, Any]:
        state_names = ("baseline", "pre", "post")
        subject_totals: Dict[str, List[float]] = {s: [] for s in state_names}
        subject_dominant_rate: Dict[str, List[float]] = {s: [] for s in state_names}
        pair_max_abs_frac_shift: Dict[str, List[float]] = {
            "pre_vs_baseline": [], "post_vs_pre": [], "post_vs_baseline": [],
        }
        pair_dominant_rate_delta: Dict[str, List[float]] = {
            "pre_vs_baseline": [], "post_vs_pre": [], "post_vs_baseline": [],
        }
        for rec in recs:
            rbt = rec.get("rate_by_template_summary", {})
            if not rbt or rbt.get("n_windows", 0) == 0:
                continue
            by_state = rbt.get("by_state", {})
            dom_id_per_state: Dict[str, int] = {}
            for state in state_names:
                st = by_state.get(state, {})
                total = st.get("median_rate_per_hour_total", np.nan)
                if np.isfinite(total):
                    subject_totals[state].append(float(total))
                rates = np.asarray(st.get("median_rate_by_template_per_hour", []), dtype=float)
                if rates.size:
                    finite_mask = np.isfinite(rates)
                    if finite_mask.any():
                        dom_idx = int(np.nanargmax(rates))
                        dom_id_per_state[state] = dom_idx
                        subject_dominant_rate[state].append(float(rates[dom_idx]))
            for pair_name in pair_max_abs_frac_shift:
                pair_rec = rbt.get(pair_name, {})
                mabs = pair_rec.get("max_abs_fraction_delta_template", np.nan)
                if np.isfinite(mabs):
                    pair_max_abs_frac_shift[pair_name].append(float(mabs))
                rate_deltas = np.asarray(
                    pair_rec.get("median_rate_delta_by_template", []), dtype=float
                )
                state_a, state_b = {
                    "pre_vs_baseline": ("baseline", "pre"),
                    "post_vs_pre": ("pre", "post"),
                    "post_vs_baseline": ("baseline", "post"),
                }[pair_name]
                dom_idx = dom_id_per_state.get(state_b, dom_id_per_state.get(state_a))
                if dom_idx is not None and rate_deltas.size > dom_idx and np.isfinite(
                    rate_deltas[dom_idx]
                ):
                    pair_dominant_rate_delta[pair_name].append(float(rate_deltas[dom_idx]))

        def _median(vals: List[float]) -> float:
            return float(np.median(vals)) if vals else np.nan

        return {
            "n_subjects": int(
                sum(1 for rec in recs if rec.get("rate_by_template_summary", {}).get("n_windows", 0) > 0)
            ),
            "by_state": {
                state: {
                    "n_subjects": int(len(subject_totals[state])),
                    "median_rate_per_hour_total": _median(subject_totals[state]),
                    "median_dominant_template_rate_per_hour": _median(
                        subject_dominant_rate[state]
                    ),
                }
                for state in state_names
            },
            "pre_vs_baseline": {
                "max_abs_fraction_delta_median": _median(
                    pair_max_abs_frac_shift["pre_vs_baseline"]
                ),
                "dominant_template_rate_delta_median": _median(
                    pair_dominant_rate_delta["pre_vs_baseline"]
                ),
                "wilcoxon_dominant_rate": _wilcoxon_on_deltas(
                    pair_dominant_rate_delta["pre_vs_baseline"]
                ),
            },
            "post_vs_pre": {
                "max_abs_fraction_delta_median": _median(
                    pair_max_abs_frac_shift["post_vs_pre"]
                ),
                "dominant_template_rate_delta_median": _median(
                    pair_dominant_rate_delta["post_vs_pre"]
                ),
                "wilcoxon_dominant_rate": _wilcoxon_on_deltas(
                    pair_dominant_rate_delta["post_vs_pre"]
                ),
            },
            "post_vs_baseline": {
                "max_abs_fraction_delta_median": _median(
                    pair_max_abs_frac_shift["post_vs_baseline"]
                ),
                "dominant_template_rate_delta_median": _median(
                    pair_dominant_rate_delta["post_vs_baseline"]
                ),
                "wilcoxon_dominant_rate": _wilcoxon_on_deltas(
                    pair_dominant_rate_delta["post_vs_baseline"]
                ),
            },
        }

    return {
        "n_subjects": int(len(recs)),
        "n_subjects_with_usable_windows": int(sum(val > 0 for val in usable_windows)),
        "n_usable_windows_total": int(sum(usable_windows)),
        "pre_vs_baseline": _pair_summary("pre_vs_baseline"),
        "post_vs_pre": _pair_summary("post_vs_pre"),
        "post_vs_baseline": _pair_summary("post_vs_baseline"),
        "rate_by_template": _rate_by_template_summary(),
    }


def summarize_propagation_cohort(subject_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    valid = [
        rec
        for rec in subject_results.values()
        if isinstance(rec, dict) and "error" not in rec and rec.get("propagation_stereotypy")
    ]
    if not valid:
        return {"n_subjects": 0}

    strict_mix = sum(int(rec["mixture"].get("is_mixture", False)) for rec in valid)
    possible_mix = sum(int(rec["mixture"].get("possible_mixture", False)) for rec in valid)
    erased = sum(int(rec["source_diagnostic"].get("soz_source_erased", False)) for rec in valid)

    bias = [
        rec["centered_rank"].get("bias_fraction")
        for rec in valid
        if np.isfinite(rec["centered_rank"].get("bias_fraction", np.nan))
    ]
    all_tau = [
        rec["propagation_stereotypy"]["all"].get("mean_tau")
        for rec in valid
        if np.isfinite(rec["propagation_stereotypy"]["all"].get("mean_tau", np.nan))
    ]

    paired_soz = []
    paired_nonsoz = []
    for rec in valid:
        soz_tau = _safe_float_scalar(rec["propagation_stereotypy"]["soz"].get("mean_tau", np.nan))
        nonsoz_tau = _safe_float_scalar(rec["propagation_stereotypy"]["nonsoz"].get("mean_tau", np.nan))
        if np.isfinite(soz_tau) and np.isfinite(nonsoz_tau):
            paired_soz.append(float(soz_tau))
            paired_nonsoz.append(float(nonsoz_tau))

    wilcoxon_p = np.nan
    sign_greater = 0
    if len(paired_soz) >= 2:
        try:
            wilcoxon_p = float(
                wilcoxon(np.asarray(paired_soz), np.asarray(paired_nonsoz), alternative="greater").pvalue
            )
        except Exception:
            wilcoxon_p = np.nan
        sign_greater = int(np.sum(np.asarray(paired_soz) > np.asarray(paired_nonsoz)))

    by_bin_summary: List[Dict[str, Any]] = []
    first_bins = valid[0]["by_nparticipating"]
    for idx, template in enumerate(first_bins):
        vals = []
        for rec in valid:
            if idx >= len(rec["by_nparticipating"]):
                continue
            tau = rec["by_nparticipating"][idx].get("mean_tau")
            if np.isfinite(tau):
                vals.append(float(tau))
        by_bin_summary.append(
            {
                "bin_label": template["bin_label"],
                "n_subjects": int(len(vals)),
                "median_tau": float(np.median(vals)) if vals else np.nan,
            }
        )

    within_taus = [
        rec["cluster"]["within_cluster_tau_mean"]
        for rec in valid
        if rec.get("cluster") and np.isfinite(rec["cluster"].get("within_cluster_tau_mean", np.nan))
    ]
    uplifts = [
        rec["cluster"]["uplift"]
        for rec in valid
        if rec.get("cluster") and np.isfinite(rec["cluster"].get("uplift", np.nan))
    ]
    inter_corrs = [
        rec["cluster"]["inter_cluster_corr"]
        for rec in valid
        if rec.get("cluster") and np.isfinite(rec["cluster"].get("inter_cluster_corr", np.nan))
    ]
    mi_means = [
        rec["legacy_mi"]["mi_mean"]
        for rec in valid
        if rec.get("legacy_mi") and np.isfinite(rec["legacy_mi"].get("mi_mean", np.nan))
    ]
    mi_sig = sum(
        1 for rec in valid
        if rec.get("legacy_mi") and rec["legacy_mi"].get("significant", False)
    )

    return {
        "n_subjects": int(len(valid)),
        "n_strict_mixture": int(strict_mix),
        "n_possible_mixture": int(possible_mix),
        "n_soz_source_erased": int(erased),
        "frac_soz_source_erased": float(erased / len(valid)),
        "mean_tau_median": float(np.median(all_tau)) if all_tau else np.nan,
        "bias_fraction_median": float(np.median(bias)) if bias else np.nan,
        "soz_vs_nonsoz": {
            "n_pairs": int(len(paired_soz)),
            "n_soz_gt_nonsoz": int(sign_greater),
            "wilcoxon_greater_p": wilcoxon_p,
            "soz_median_tau": float(np.median(paired_soz)) if paired_soz else np.nan,
            "nonsoz_median_tau": float(np.median(paired_nonsoz)) if paired_nonsoz else np.nan,
        },
        "by_nparticipating": by_bin_summary,
        "cluster_analysis": {
            "within_cluster_tau_median": float(np.median(within_taus)) if within_taus else np.nan,
            "overall_tau_median": float(np.median(all_tau)) if all_tau else np.nan,
            "uplift_median": float(np.median(uplifts)) if uplifts else np.nan,
            "inter_cluster_corr_median": float(np.median(inter_corrs)) if inter_corrs else np.nan,
            "n_anticorrelated": int(sum(1 for c in inter_corrs if c < -0.5)),
        },
        "legacy_mi": {
            "mi_mean_median": float(np.median(mi_means)) if mi_means else np.nan,
            "n_significant": int(mi_sig),
            "n_tested": int(len(mi_means)),
        },
        "adaptive_cluster_analysis": _summarize_adaptive_clusters(valid),
        "reproducibility_analysis": _summarize_reproducibility(valid),
        "absolute_lag_validation_analysis": _summarize_absolute_lag_validation(valid),
        "rate_state_coupling_analysis": _summarize_rate_state_coupling(valid),
        "temporal_dynamics_analysis": _summarize_temporal_dynamics(valid),
        "temporal_dynamics_followup_analysis": _summarize_temporal_dynamics_followup(valid),
        "seizure_proximity_analysis": _summarize_seizure_proximity(valid),
        "seizure_proximity_analysis_auxiliary": _summarize_seizure_proximity(
            valid, coupling_key="seizure_proximity_coupling_auxiliary"
        ),
    }


def _summarize_adaptive_clusters(valid: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cohort-level summary of adaptive (stability-first) clustering."""
    adaptive_recs = [
        rec["adaptive_cluster"]
        for rec in valid
        if rec.get("adaptive_cluster") and "error" not in rec.get("adaptive_cluster", {})
    ]
    if not adaptive_recs:
        return {"n_subjects": 0}

    stable_ks = [r["stable_k"] for r in adaptive_recs if r.get("stable_k") is not None]
    chosen_ks = [r["chosen_k"] for r in adaptive_recs]
    n_fallback = sum(1 for r in adaptive_recs if r.get("chosen_reason") == "fallback_k_min")

    ada_uplifts = [
        r["uplift"] for r in adaptive_recs if np.isfinite(r.get("uplift", np.nan))
    ]
    ada_within = [
        r["within_cluster_tau_mean"]
        for r in adaptive_recs
        if np.isfinite(r.get("within_cluster_tau_mean", np.nan))
    ]
    n_fwd_rev = sum(
        len(r.get("candidate_forward_reverse_pairs", [])) for r in adaptive_recs
    )

    return {
        "n_subjects": int(len(adaptive_recs)),
        "stable_k_distribution": {
            int(k): int(stable_ks.count(k)) for k in sorted(set(stable_ks))
        } if stable_ks else {},
        "chosen_k_distribution": {
            int(k): int(chosen_ks.count(k)) for k in sorted(set(chosen_ks))
        },
        "n_stable_k_found": int(len(stable_ks)),
        "n_fallback": int(n_fallback),
        "stable_k_median": float(np.median(stable_ks)) if stable_ks else np.nan,
        "uplift_median": float(np.median(ada_uplifts)) if ada_uplifts else np.nan,
        "within_cluster_tau_median": float(np.median(ada_within)) if ada_within else np.nan,
        "n_subjects_with_forward_reverse": int(
            sum(1 for r in adaptive_recs if r.get("candidate_forward_reverse_pairs"))
        ),
        "total_forward_reverse_pairs": int(n_fwd_rev),
    }


# ---------------------------------------------------------------------------
# PR-5-A: novel-template falsification gate
# ---------------------------------------------------------------------------
#
# Contract: docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md
#   §3.2 data, §3.3 metrics, §3.5 PASS/FAIL thresholds, §5.2 code entry.
#
# The gate decides whether peri-ictal events are still in-distribution under
# the global stable template library `T_global = build_cluster_templates(...)`.
# A FAIL means H_OOD ("peri-ictal events come from a novel template family")
# cannot be ruled out, in which case PR-5-B must not start.

PR5A_R_DELTA_THRESHOLD: float = 0.05
PR5A_E_RELATIVE_THRESHOLD: float = 0.10
PR5A_WILCOXON_ALPHA: float = 0.05
PR5A_GATE_STATES: Tuple[str, ...] = ("baseline", "pre", "post")


def _per_event_gate_metrics(
    rel_vec: np.ndarray,
    rank_vec: np.ndarray,
    bool_vec: np.ndarray,
    templates: np.ndarray,
    *,
    min_shared_channels: int,
) -> Tuple[float, float, float]:
    """Per-event (best_r, best_recon_err, assignment_gap) against templates.

    * ``best_r`` — max over k of Spearman r between ``rel_vec`` on
      participating channels and ``templates[k]`` on the same channels.
    * ``best_recon_err`` — min over k of the mean squared (rank-space)
      difference on shared finite channels (>= ``min_shared_channels``).
      Identical math to ``assign_events_to_templates``.
    * ``assignment_gap`` — second-best ``recon_err`` minus best ``recon_err``.
    """
    n_clusters = int(templates.shape[0])
    bool_mask = bool_vec.astype(bool)
    masked_rel = np.where(bool_mask, rel_vec, np.nan)
    masked_rank = np.where(bool_mask, rank_vec, np.nan)

    best_r = -np.inf
    errs = np.full(n_clusters, np.inf, dtype=float)
    for k in range(n_clusters):
        tmpl = templates[k]
        finite_rel = np.isfinite(masked_rel) & np.isfinite(tmpl)
        n_rel = int(np.sum(finite_rel))
        if n_rel >= int(min_shared_channels):
            tmpl_vals = tmpl[finite_rel]
            rel_vals = masked_rel[finite_rel]
            # spearmanr requires variation in both inputs; otherwise it
            # returns NaN, which we drop into the "no usable correlation"
            # bucket together with the under-coverage cases above.
            if (
                np.unique(tmpl_vals).size >= 2
                and np.unique(rel_vals).size >= 2
            ):
                r_val, _ = spearmanr(rel_vals, tmpl_vals)
                if np.isfinite(r_val) and float(r_val) > best_r:
                    best_r = float(r_val)
        finite_rank = np.isfinite(masked_rank) & np.isfinite(tmpl)
        n_rank = int(np.sum(finite_rank))
        if n_rank >= int(min_shared_channels):
            diff = masked_rank[finite_rank] - tmpl[finite_rank]
            errs[k] = float(np.mean(diff ** 2))

    best_r_out = float(best_r) if np.isfinite(best_r) else float("nan")
    finite_errs = errs[np.isfinite(errs)]
    if finite_errs.size == 0:
        return best_r_out, float("nan"), float("nan")
    if finite_errs.size == 1:
        return best_r_out, float(finite_errs[0]), float("nan")
    sorted_errs = np.sort(finite_errs)
    return (
        best_r_out,
        float(sorted_errs[0]),
        float(sorted_errs[1] - sorted_errs[0]),
    )


def compute_novel_template_gate(
    *,
    v_ranks: np.ndarray,
    v_rel: np.ndarray,
    v_bools: np.ndarray,
    templates: np.ndarray,
    proximity_windows: Sequence[Dict[str, Any]],
    min_participating_l3: int = 5,
    min_shared_channels: int = 3,
) -> Dict[str, Any]:
    """PR-5-A novel-template falsification gate.

    For each baseline / pre / post event (after dropping events with fewer
    than ``min_participating_l3`` participating channels), compute three
    per-event diagnostics against the fixed global template library
    ``templates``:

    1. ``best_r`` — max over k of Spearman r between the event's relative-lag
       vector ``v_rel[:, i]`` (on participating channels) and ``templates[k]``
       on the same channels.
    2. ``recon_err`` — min over k of the mean squared rank-space difference
       on shared finite channels (>= ``min_shared_channels``). Identical math
       to :func:`assign_events_to_templates`.
    3. ``gap`` — second-best ``recon_err`` minus best ``recon_err``; the
       assignment confidence margin.

    Pool events across all usable seizure windows per state and return per
    state distributions, medians, and per-subject deltas (state - baseline).

    Parameters
    ----------
    v_ranks, v_rel, v_bools
        Same ``(n_ch, n_valid_events)`` arrays that
        :func:`compute_seizure_proximity_coupling` builds internally
        (already sliced by ``valid_event_indices`` with
        ``min_participating>=min_shared_channels``).
    templates
        ``T_global`` from
        ``build_cluster_templates(v_ranks, v_bools, adaptive_labels, stable_k)``;
        the same library used by PR-4D fixed-template projection. NaN
        channels are tolerated.
    proximity_windows
        Iterable of window dicts shaped like
        ``compute_seizure_proximity_coupling`` ``seizure_windows`` but **also
        carrying** ``state_event_indices`` per state (which the runner
        provides directly from ``_build_seizure_proximity_windows`` to avoid
        a second source-of-truth window builder per §3.2).
    """
    v_ranks = np.asarray(v_ranks, dtype=float)
    v_rel = np.asarray(v_rel, dtype=float)
    v_bools = np.asarray(v_bools, dtype=bool)
    templates = np.asarray(templates, dtype=float)
    if v_ranks.shape != v_rel.shape or v_ranks.shape != v_bools.shape:
        raise ValueError("v_ranks, v_rel, and v_bools must share shape")
    n_ch = v_ranks.shape[0]
    if templates.ndim != 2 or templates.shape[1] != n_ch:
        raise ValueError("templates must have shape (n_clusters, n_ch)")

    n_clusters = int(templates.shape[0])
    n_part = np.sum(v_bools > 0, axis=0)
    eligible_mask = n_part >= int(min_participating_l3)

    pooled_indices: Dict[str, List[int]] = {name: [] for name in PR5A_GATE_STATES}
    excluded_counts: Dict[str, int] = {name: 0 for name in PR5A_GATE_STATES}
    n_windows_used = 0
    for window in proximity_windows:
        if not bool(window.get("usable", True)):
            continue
        n_windows_used += 1
        state_indices = window.get("state_event_indices", {})
        for name in PR5A_GATE_STATES:
            raw = state_indices.get(name, [])
            idx = np.asarray(raw, dtype=int)
            if idx.size == 0:
                continue
            mask = eligible_mask[idx]
            for i in idx[mask]:
                pooled_indices[name].append(int(i))
            excluded_counts[name] += int(np.sum(~mask))

    per_state_metrics: Dict[str, Dict[str, List[float]]] = {
        name: {"r": [], "e": [], "gap": []} for name in PR5A_GATE_STATES
    }
    for state_name in PR5A_GATE_STATES:
        for ev in pooled_indices[state_name]:
            r_val, e_val, gap_val = _per_event_gate_metrics(
                v_rel[:, ev],
                v_ranks[:, ev],
                v_bools[:, ev],
                templates,
                min_shared_channels=int(min_shared_channels),
            )
            if np.isfinite(r_val):
                per_state_metrics[state_name]["r"].append(float(r_val))
            if np.isfinite(e_val):
                per_state_metrics[state_name]["e"].append(float(e_val))
            if np.isfinite(gap_val):
                per_state_metrics[state_name]["gap"].append(float(gap_val))

    medians: Dict[str, Dict[str, float]] = {}
    n_events: Dict[str, int] = {}
    for name in PR5A_GATE_STATES:
        n_events[name] = int(len(pooled_indices[name]))
        medians[name] = {
            metric: (
                float(np.median(per_state_metrics[name][metric]))
                if per_state_metrics[name][metric]
                else float("nan")
            )
            for metric in ("r", "e", "gap")
        }

    def _delta(state: str) -> Dict[str, float]:
        return {
            metric: float(medians[state][metric] - medians["baseline"][metric])
            for metric in ("r", "e", "gap")
        }

    return {
        "n_clusters": n_clusters,
        "n_channels": int(n_ch),
        "n_windows_used": int(n_windows_used),
        "n_events_by_state": n_events,
        "n_l3_excluded_by_state": excluded_counts,
        "min_participating_l3": int(min_participating_l3),
        "min_shared_channels": int(min_shared_channels),
        "r_by_state": {name: list(per_state_metrics[name]["r"]) for name in PR5A_GATE_STATES},
        "e_by_state": {name: list(per_state_metrics[name]["e"]) for name in PR5A_GATE_STATES},
        "gap_by_state": {name: list(per_state_metrics[name]["gap"]) for name in PR5A_GATE_STATES},
        "median_by_state": medians,
        "delta_pre_minus_baseline": _delta("pre"),
        "delta_post_minus_baseline": _delta("post"),
    }


def _wilcoxon_safe(deltas: Sequence[float]) -> Tuple[float, float, int]:
    """Paired Wilcoxon vs zero on subject-level deltas.

    Returns ``(median, p_value, n_used)`` with ``p_value=NaN`` when scipy
    refuses (e.g. all zero deltas after the wilcoxon "wsr" zero-handling).
    """
    arr = np.asarray([float(x) for x in deltas if np.isfinite(x)], dtype=float)
    n = int(arr.size)
    if n == 0:
        return float("nan"), float("nan"), 0
    median = float(np.median(arr))
    if n < 2 or np.all(arr == 0):
        return median, float("nan"), n
    try:
        _, p = wilcoxon(arr, zero_method="wilcox", alternative="two-sided")
        return median, float(p), n
    except ValueError:
        return median, float("nan"), n


def _sign_test_safe(deltas: Sequence[float]) -> Tuple[float, int]:
    """Two-sided exact sign test on non-zero subject-level deltas."""
    arr = np.asarray([float(x) for x in deltas if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), 0
    nonzero = arr[arr != 0]
    n = int(nonzero.size)
    if n == 0:
        return float("nan"), 0
    n_pos = int(np.sum(nonzero > 0))
    p = binomtest(n_pos, n=n, p=0.5, alternative="two-sided").pvalue
    return float(p), n


def _evaluate_pr5_gate_thresholds(
    cohort: Dict[str, Dict[str, Any]],
    *,
    r_delta_threshold: float,
    e_relative_threshold: float,
    wilcoxon_alpha: float,
) -> Dict[str, Any]:
    """Apply §3.5 PASS/FAIL rules to a cohort-level summary.

    Returns the per-axis (r, e, gap) PASS booleans plus the overall config
    PASS boolean. PR-5-A is intentionally conservative: failing any axis on
    any state pair (pre vs baseline OR post vs baseline) flips the gate to
    FAIL, and the threshold values are baked into the contract so reruns
    cannot drift the bar.
    """
    pairs = ("pre_vs_baseline", "post_vs_baseline")
    e_baseline_median = float(cohort.get("e_baseline_subject_median", float("nan")))

    def _axis_pass_r() -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        details: Dict[str, Dict[str, Any]] = {}
        ok = True
        for pair in pairs:
            entry = cohort["delta_summary"][pair]["r"]
            med = float(entry["median"])
            p = float(entry["wilcoxon_p"]) if np.isfinite(entry["wilcoxon_p"]) else float("nan")
            magnitude_ok = bool(np.isfinite(med) and abs(med) <= r_delta_threshold)
            wilcoxon_ok = (not np.isfinite(p)) or bool(p >= wilcoxon_alpha)
            pair_ok = magnitude_ok and wilcoxon_ok
            details[pair] = {
                "median_delta": med,
                "wilcoxon_p": p,
                "magnitude_ok": magnitude_ok,
                "wilcoxon_ok": wilcoxon_ok,
                "pair_pass": pair_ok,
            }
            ok = ok and pair_ok
        return ok, details

    def _axis_pass_e() -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        details: Dict[str, Dict[str, Any]] = {}
        ok = True
        denom = max(abs(e_baseline_median), 1e-12)
        for pair in pairs:
            entry = cohort["delta_summary"][pair]["e"]
            med = float(entry["median"])
            p = float(entry["wilcoxon_p"]) if np.isfinite(entry["wilcoxon_p"]) else float("nan")
            relative = abs(med) / denom if np.isfinite(med) else float("nan")
            magnitude_ok = bool(
                np.isfinite(relative) and relative <= e_relative_threshold
            )
            wilcoxon_ok = (not np.isfinite(p)) or bool(p >= wilcoxon_alpha)
            pair_ok = magnitude_ok and wilcoxon_ok
            details[pair] = {
                "median_delta": med,
                "median_baseline_subject_median": e_baseline_median,
                "relative_delta": relative,
                "wilcoxon_p": p,
                "magnitude_ok": magnitude_ok,
                "wilcoxon_ok": wilcoxon_ok,
                "pair_pass": pair_ok,
            }
            ok = ok and pair_ok
        return ok, details

    def _axis_pass_gap() -> Tuple[bool, Dict[str, Dict[str, Any]]]:
        # Per §3.5: only fail if peri-ictal gap is *significantly lower* than
        # baseline (i.e. negative direction with Wilcoxon p < alpha). Higher
        # gap (positive delta) is fine and direction-only failures (without
        # significance) are also fine.
        details: Dict[str, Dict[str, Any]] = {}
        ok = True
        for pair in pairs:
            entry = cohort["delta_summary"][pair]["gap"]
            med = float(entry["median"])
            p = float(entry["wilcoxon_p"]) if np.isfinite(entry["wilcoxon_p"]) else float("nan")
            wilcoxon_significant = np.isfinite(p) and p < wilcoxon_alpha
            harmful = bool(np.isfinite(med) and med < 0 and wilcoxon_significant)
            pair_ok = not harmful
            details[pair] = {
                "median_delta": med,
                "wilcoxon_p": p,
                "harmful": harmful,
                "pair_pass": pair_ok,
            }
            ok = ok and pair_ok
        return ok, details

    r_ok, r_details = _axis_pass_r()
    e_ok, e_details = _axis_pass_e()
    gap_ok, gap_details = _axis_pass_gap()

    return {
        "thresholds": {
            "r_delta": float(r_delta_threshold),
            "e_relative": float(e_relative_threshold),
            "wilcoxon_alpha": float(wilcoxon_alpha),
        },
        "axis_pass": {"r": r_ok, "e": e_ok, "gap": gap_ok},
        "axis_details": {
            "r": r_details,
            "e": e_details,
            "gap": gap_details,
        },
        "config_pass": bool(r_ok and e_ok and gap_ok),
    }


def summarize_pr5_novel_template_gate(
    per_subject_records_by_config: Dict[str, Sequence[Dict[str, Any]]],
    *,
    r_delta_threshold: float = PR5A_R_DELTA_THRESHOLD,
    e_relative_threshold: float = PR5A_E_RELATIVE_THRESHOLD,
    wilcoxon_alpha: float = PR5A_WILCOXON_ALPHA,
    min_state_events_for_gate: int = 30,
) -> Dict[str, Any]:
    """Cohort-level PR-5-A summary + PASS/FAIL for both window configs.

    Each value in ``per_subject_records_by_config`` is a list of per-subject
    gate records (output of :func:`compute_novel_template_gate`) augmented
    with at least ``subject_id`` and ``dataset`` (and optionally
    ``warning``). Records that fail the per-state ``min_state_events_for_gate``
    threshold are dropped from cohort statistics but counted as
    ``ineligible``.

    Returns ``{config_name: per_config_summary, "overall_pass": bool}`` where
    ``overall_pass`` is the conjunction of every config's ``config_pass``
    (PR-5-A sensitivity = main AND auxiliary, per §3.5 last row).
    """
    summary: Dict[str, Any] = {}
    config_passes: List[bool] = []

    for config_name, records in per_subject_records_by_config.items():
        eligible: List[Dict[str, Any]] = []
        ineligible: List[Dict[str, Any]] = []
        for rec in records:
            counts = rec.get("n_events_by_state") or {}
            min_count = min(
                int(counts.get(name, 0)) for name in PR5A_GATE_STATES
            ) if counts else 0
            label = {
                "subject_id": rec.get("subject_id"),
                "dataset": rec.get("dataset"),
                "n_events_by_state": dict(counts),
                "min_state_events": int(min_count),
            }
            if min_count < int(min_state_events_for_gate) or rec.get("warning"):
                ineligible.append(
                    {**label, "reason": rec.get("warning") or "below_min_state_events"}
                )
                continue
            eligible.append(rec)

        baseline_e_subject_medians = [
            float(rec["median_by_state"]["baseline"]["e"])
            for rec in eligible
            if np.isfinite(rec["median_by_state"]["baseline"]["e"])
        ]
        e_baseline_subject_median = (
            float(np.median(baseline_e_subject_medians))
            if baseline_e_subject_medians
            else float("nan")
        )

        delta_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for pair, delta_key in (
            ("pre_vs_baseline", "delta_pre_minus_baseline"),
            ("post_vs_baseline", "delta_post_minus_baseline"),
        ):
            pair_summary: Dict[str, Dict[str, Any]] = {}
            for metric in ("r", "e", "gap"):
                deltas = [
                    float(rec[delta_key][metric])
                    for rec in eligible
                    if np.isfinite(rec[delta_key][metric])
                ]
                med, p, n_used = _wilcoxon_safe(deltas)
                n_pos = int(np.sum(np.asarray(deltas) > 0)) if deltas else 0
                n_neg = int(np.sum(np.asarray(deltas) < 0)) if deltas else 0
                sign_p, sign_n = _sign_test_safe(deltas)
                pair_summary[metric] = {
                    "n_subjects": n_used,
                    "median": float(med) if np.isfinite(med) else float("nan"),
                    "wilcoxon_p": float(p) if np.isfinite(p) else float("nan"),
                    "sign_test_p": float(sign_p) if np.isfinite(sign_p) else float("nan"),
                    "sign_test_n": int(sign_n),
                    "n_positive": n_pos,
                    "n_negative": n_neg,
                }
            delta_summary[pair] = pair_summary

        cohort = {
            "config_name": config_name,
            "n_subjects_eligible": int(len(eligible)),
            "n_subjects_ineligible": int(len(ineligible)),
            "ineligible_subjects": ineligible,
            "min_state_events_for_gate": int(min_state_events_for_gate),
            "e_baseline_subject_median": float(e_baseline_subject_median),
            "delta_summary": delta_summary,
        }
        gate_eval = _evaluate_pr5_gate_thresholds(
            cohort,
            r_delta_threshold=r_delta_threshold,
            e_relative_threshold=e_relative_threshold,
            wilcoxon_alpha=wilcoxon_alpha,
        )
        cohort["gate_evaluation"] = gate_eval
        cohort["gate_pass"] = bool(gate_eval["config_pass"])
        config_passes.append(cohort["gate_pass"])
        summary[config_name] = cohort

    summary["overall_pass"] = bool(config_passes) and all(config_passes)
    summary["thresholds"] = {
        "r_delta": float(r_delta_threshold),
        "e_relative": float(e_relative_threshold),
        "wilcoxon_alpha": float(wilcoxon_alpha),
        "min_state_events_for_gate": int(min_state_events_for_gate),
    }
    return summary


# ---------------------------------------------------------------------------
# PR-5-B: template recruitment shift
#
# Spec: docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md §4
# Three layers:
#   1. ``compute_template_recruitment_shift`` — per-subject computation, runs
#      on the same gate-eligible event pool as PR-5-A, recomputes per-window
#      counts_by_template under the filtered pool, emits dual dominant
#      readings (global vs per-window) and the share composition diagnostic.
#   2. ``summarize_pr5_template_recruitment_shift`` — cohort aggregation,
#      paired Wilcoxon + bootstrap CI on subject-level deltas. Composition
#      diagnostic lives in a sibling field with its own schema (no
#      ``bonferroni_pass`` key) per §4.5.3 isolation contract.
#   3. ``_compute_pr5b_sensitivity_gate`` — reads only ``dominant_rate``;
#      monkey-patching ``composition_diagnostic`` MUST NOT change its output.
# ---------------------------------------------------------------------------


PR5B_STATES: Tuple[str, ...] = ("baseline", "pre", "post")
PR5B_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("pre_minus_baseline", "baseline", "pre"),
    ("post_minus_baseline", "baseline", "post"),
    ("post_minus_pre", "pre", "post"),
)
PR5B_BONFERRONI_FAMILY_SIZE = 6  # 3 pairs × 2 configs (main + auxiliary)


def _pr5b_argmax_or_neg1(counts: np.ndarray) -> int:
    if counts.size == 0 or int(counts.sum()) == 0:
        return -1
    return int(np.argmax(counts))


def _pr5b_weighted_rate(counts: Sequence[float], hours: Sequence[float]) -> float:
    h = float(np.sum([float(x) for x in hours if np.isfinite(x) and float(x) > 0.0]))
    c = float(np.sum([float(x) for x in counts if np.isfinite(x)]))
    if h <= 0.0:
        return float("nan")
    return c / h


def _pr5b_weighted_share(
    shares: Sequence[float], hours: Sequence[float]
) -> Tuple[float, int]:
    """Weight-average share with state_covered_hours weights.

    Returns ``(share, n_eligible_windows)``. Windows with zero hours or
    NaN share are excluded.
    """
    pairs = [
        (float(s), float(h))
        for s, h in zip(shares, hours)
        if np.isfinite(s) and np.isfinite(h) and float(h) > 0.0
    ]
    if not pairs:
        return float("nan"), 0
    total_h = sum(h for _, h in pairs)
    if total_h <= 0.0:
        return float("nan"), 0
    weighted = sum(s * h for s, h in pairs)
    return float(weighted / total_h), int(len(pairs))


def compute_template_recruitment_shift(
    *,
    cluster_labels: np.ndarray,
    n_part_per_event: np.ndarray,
    n_clusters: int,
    dominant_global_id: int,
    proximity_windows: Sequence[Dict[str, Any]],
    min_participating_l3: int = 5,
) -> Dict[str, Any]:
    """PR-5-B per-subject template recruitment shift.

    Reuses PR-4C P0 fix machinery (window ownership, pair usability,
    gap-aware ``state_covered_hours``) but recomputes per-window
    ``counts_by_template`` on the gate-eligible event pool
    (``min_participating_l3``), so the analysis runs on exactly the same
    events PR-5-A audited.

    Two dominant definitions run in parallel (§4.3):

    - ``dominant_global``: caller-supplied ``dominant_global_id`` (per-subject
      whole-recording occupancy max, identical to PR-4D).
    - ``dominant_per_window``: per-window argmax over **filtered** baseline
      counts; baseline empty → fall back to argmax over the window's filtered
      total counts; window with zero filtered events anywhere → ``-1`` (window
      ineligible for candidate B).

    Aggregation across a subject's windows is sum(counts) / sum(covered_hours)
    for rates, and weighted average of per-window share for the composition
    diagnostic (weight = ``state_covered_hours``, identical to the rate
    weighting per §4.5.1 contract).
    """
    cluster_labels = np.asarray(cluster_labels, dtype=int)
    n_part_per_event = np.asarray(n_part_per_event, dtype=int)
    if cluster_labels.shape != n_part_per_event.shape:
        raise ValueError("cluster_labels and n_part_per_event must share shape")

    eligible_mask = n_part_per_event >= int(min_participating_l3)
    n_clusters = int(n_clusters)
    if not (0 <= int(dominant_global_id) < n_clusters):
        raise ValueError(
            f"dominant_global_id={dominant_global_id} out of range for n_clusters={n_clusters}"
        )

    per_window: List[Dict[str, Any]] = []
    n_windows_used = 0
    n_windows_total = int(len(proximity_windows))

    for window in proximity_windows:
        if not bool(window.get("usable", True)):
            continue
        n_windows_used += 1
        state_indices = window.get("state_event_indices", {})
        covered_hours = window.get("state_covered_hours", {})

        state_counts: Dict[str, np.ndarray] = {}
        state_total: Dict[str, int] = {}
        for state in PR5B_STATES:
            raw = np.asarray(state_indices.get(state, []), dtype=int)
            if raw.size == 0:
                state_counts[state] = np.zeros(n_clusters, dtype=int)
                state_total[state] = 0
                continue
            keep = raw[eligible_mask[raw]] if raw.size else raw
            if keep.size == 0:
                state_counts[state] = np.zeros(n_clusters, dtype=int)
                state_total[state] = 0
                continue
            state_counts[state] = np.bincount(
                cluster_labels[keep], minlength=n_clusters
            ).astype(int)
            state_total[state] = int(keep.size)

        baseline_counts = state_counts["baseline"]
        if int(baseline_counts.sum()) > 0:
            dom_window_id = _pr5b_argmax_or_neg1(baseline_counts)
        else:
            total_counts = (
                state_counts["baseline"]
                + state_counts["pre"]
                + state_counts["post"]
            )
            dom_window_id = _pr5b_argmax_or_neg1(total_counts)

        per_window.append(
            {
                "seizure_id": int(window.get("seizure_id", -1)),
                "seizure_time": float(window.get("seizure_time", float("nan"))),
                "state_covered_hours": {
                    s: float(covered_hours.get(s, float("nan"))) for s in PR5B_STATES
                },
                "state_total_counts_filtered": {
                    s: int(state_total[s]) for s in PR5B_STATES
                },
                "state_counts_dom_global": {
                    s: int(state_counts[s][int(dominant_global_id)]) for s in PR5B_STATES
                },
                "state_counts_dom_window": {
                    s: int(state_counts[s][dom_window_id]) if dom_window_id >= 0 else 0
                    for s in PR5B_STATES
                },
                "state_counts_nondom_global": {
                    s: int(state_total[s] - state_counts[s][int(dominant_global_id)])
                    for s in PR5B_STATES
                },
                "dom_window_id": int(dom_window_id),
                "_state_counts_full": {s: state_counts[s].tolist() for s in PR5B_STATES},
            }
        )

    weighted_per_state: Dict[str, Dict[str, float]] = {
        "dom_global_rate_per_hour": {},
        "dom_window_rate_per_hour": {},
        "nondom_global_rate_per_hour": {},
        "dom_global_share": {},
    }
    share_state_n_eligible_windows: Dict[str, int] = {}

    for state in PR5B_STATES:
        h_all = [w["state_covered_hours"][state] for w in per_window]
        c_dom_global = [w["state_counts_dom_global"][state] for w in per_window]
        c_dom_window = [
            w["state_counts_dom_window"][state]
            for w in per_window
            if w["dom_window_id"] >= 0
        ]
        h_dom_window = [
            w["state_covered_hours"][state]
            for w in per_window
            if w["dom_window_id"] >= 0
        ]
        c_nondom = [w["state_counts_nondom_global"][state] for w in per_window]

        weighted_per_state["dom_global_rate_per_hour"][state] = _pr5b_weighted_rate(
            c_dom_global, h_all
        )
        weighted_per_state["dom_window_rate_per_hour"][state] = _pr5b_weighted_rate(
            c_dom_window, h_dom_window
        )
        weighted_per_state["nondom_global_rate_per_hour"][state] = _pr5b_weighted_rate(
            c_nondom, h_all
        )

        share_per_window: List[float] = []
        share_hours: List[float] = []
        for w in per_window:
            tot = w["state_total_counts_filtered"][state]
            h = w["state_covered_hours"][state]
            if tot <= 0 or not np.isfinite(h) or h <= 0.0:
                continue
            share_per_window.append(w["state_counts_dom_global"][state] / float(tot))
            share_hours.append(h)
        share_val, n_share = _pr5b_weighted_share(share_per_window, share_hours)
        weighted_per_state["dom_global_share"][state] = share_val
        share_state_n_eligible_windows[state] = int(n_share)

    deltas: Dict[str, Dict[str, float]] = {
        metric: {} for metric in (
            "dom_global_rate", "dom_window_rate", "nondom_global_rate", "dom_global_share"
        )
    }
    metric_to_key = {
        "dom_global_rate": "dom_global_rate_per_hour",
        "dom_window_rate": "dom_window_rate_per_hour",
        "nondom_global_rate": "nondom_global_rate_per_hour",
        "dom_global_share": "dom_global_share",
    }
    for metric, weighted_key in metric_to_key.items():
        per_state = weighted_per_state[weighted_key]
        for pair_name, state_a, state_b in PR5B_PAIRS:
            a = per_state.get(state_a, float("nan"))
            b = per_state.get(state_b, float("nan"))
            deltas[metric][pair_name] = (
                float(b - a) if np.isfinite(a) and np.isfinite(b) else float("nan")
            )

    share_pair_eligible: Dict[str, bool] = {}
    for pair_name, state_a, state_b in PR5B_PAIRS:
        share_pair_eligible[pair_name] = bool(
            share_state_n_eligible_windows[state_a] >= 1
            and share_state_n_eligible_windows[state_b] >= 1
        )

    dom_window_ids = [w["dom_window_id"] for w in per_window]
    matched = sum(1 for wid in dom_window_ids if wid == int(dominant_global_id))
    dom_agreement = (
        float(matched) / float(len(dom_window_ids)) if dom_window_ids else float("nan")
    )

    for w in per_window:
        w.pop("_state_counts_full", None)

    return {
        "n_clusters": int(n_clusters),
        "dom_global_id": int(dominant_global_id),
        "min_participating_l3": int(min_participating_l3),
        "n_windows_total": int(n_windows_total),
        "n_windows_used": int(n_windows_used),
        "per_window": per_window,
        "weighted_per_state": weighted_per_state,
        "deltas": deltas,
        "share_state_n_eligible_windows": share_state_n_eligible_windows,
        "share_pair_eligible": share_pair_eligible,
        "dom_window_ids_per_window": [int(w) for w in dom_window_ids],
        "dom_agreement": float(dom_agreement) if np.isfinite(dom_agreement) else float("nan"),
    }


def _pr5b_bootstrap_median_ci(
    deltas: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    arr = np.asarray([float(x) for x in deltas if np.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    medians = np.median(arr[idx], axis=1)
    lo = float(np.percentile(medians, 100.0 * alpha / 2.0))
    hi = float(np.percentile(medians, 100.0 * (1.0 - alpha / 2.0)))
    return lo, hi


def _pr5b_pair_summary(
    deltas: Sequence[float],
    *,
    n_boot: int,
    seed: int,
    bonferroni_alpha: Optional[float] = None,
) -> Dict[str, Any]:
    arr = np.asarray([float(x) for x in deltas if np.isfinite(x)], dtype=float)
    n = int(arr.size)
    median, p, n_used = _wilcoxon_safe(arr.tolist())
    sign_p, sign_n = _sign_test_safe(arr.tolist())
    n_pos = int(np.sum(arr > 0))
    n_neg = int(np.sum(arr < 0))
    ci_lo, ci_hi = _pr5b_bootstrap_median_ci(arr.tolist(), n_boot=n_boot, seed=seed)
    out: Dict[str, Any] = {
        "n": n,
        "n_used_wilcoxon": int(n_used),
        "median_delta": float(median) if np.isfinite(median) else float("nan"),
        "wilcoxon_p": float(p) if np.isfinite(p) else float("nan"),
        "sign_test_p": float(sign_p) if np.isfinite(sign_p) else float("nan"),
        "sign_test_n": int(sign_n),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "ci95_lo": float(ci_lo) if np.isfinite(ci_lo) else float("nan"),
        "ci95_hi": float(ci_hi) if np.isfinite(ci_hi) else float("nan"),
    }
    if bonferroni_alpha is not None:
        out["bonferroni_alpha"] = float(bonferroni_alpha)
        out["bonferroni_pass"] = bool(
            np.isfinite(p) and float(p) < float(bonferroni_alpha)
        )
    return out


def _pr5b_share_pair_summary(
    records: Sequence[Dict[str, Any]],
    pair_name: str,
    *,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    deltas: List[float] = []
    n_ineligible = 0
    for rec in records:
        if not bool(rec.get("share_pair_eligible", {}).get(pair_name, False)):
            n_ineligible += 1
            continue
        d = rec.get("deltas", {}).get("dom_global_share", {}).get(pair_name)
        if d is None or not np.isfinite(float(d)):
            n_ineligible += 1
            continue
        deltas.append(float(d))
    arr = np.asarray(deltas, dtype=float)
    median, p, n_used = _wilcoxon_safe(arr.tolist())
    sign_p, sign_n = _sign_test_safe(arr.tolist())
    n_pos = int(np.sum(arr > 0))
    n_neg = int(np.sum(arr < 0))
    ci_lo, ci_hi = _pr5b_bootstrap_median_ci(arr.tolist(), n_boot=n_boot, seed=seed)
    a_priori_consistent = (
        n_neg
        if pair_name in ("pre_minus_baseline", "post_minus_baseline")
        else None
    )
    return {
        "n": int(arr.size),
        "n_ineligible": int(n_ineligible),
        "n_used_wilcoxon": int(n_used),
        "median_delta_share": float(median) if np.isfinite(median) else float("nan"),
        "wilcoxon_p": float(p) if np.isfinite(p) else float("nan"),
        "sign_test_p": float(sign_p) if np.isfinite(sign_p) else float("nan"),
        "sign_test_n": int(sign_n),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "direction_consistent_count": (
            int(a_priori_consistent) if a_priori_consistent is not None else None
        ),
        "ci95_lo": float(ci_lo) if np.isfinite(ci_lo) else float("nan"),
        "ci95_hi": float(ci_hi) if np.isfinite(ci_hi) else float("nan"),
    }


def summarize_pr5_template_recruitment_shift(
    per_subject_records_by_config: Dict[str, Sequence[Dict[str, Any]]],
    *,
    n_boot: int = 2000,
    bootstrap_seed: int = 42,
    bonferroni_family_size: int = PR5B_BONFERRONI_FAMILY_SIZE,
    nominal_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Cohort-level PR-5-B summary (§4.4 + §4.5).

    For each window config (``main`` / ``auxiliary``) emit:

    - ``dominant_rate``: candidate A (``dominant_global``) and candidate B
      (``dominant_per_window``) per-pair Wilcoxon + sign test + bootstrap
      median CI + Bonferroni decision (alpha = ``nominal_alpha``/family_size).
    - ``composition_diagnostic.share``: subject-level paired Wilcoxon on
      dominant share deltas; **no** ``bonferroni_pass`` field per §4.5.3
      isolation contract. Subjects with ``share_pair_eligible[pair]=False``
      are dropped from the test and counted in ``n_ineligible``.

    Sensitivity gate is **not** computed here; call
    :func:`_compute_pr5b_sensitivity_gate` separately so it can be unit-
    tested for isolation from ``composition_diagnostic``.
    """
    bonferroni_alpha = float(nominal_alpha) / float(bonferroni_family_size)
    summary: Dict[str, Any] = {
        "dominant_rate_weight_key": "state_covered_hours",
        "composition_diagnostic_weight_key": "state_covered_hours",
        "thresholds": {
            "nominal_alpha": float(nominal_alpha),
            "bonferroni_family_size": int(bonferroni_family_size),
            "bonferroni_alpha": float(bonferroni_alpha),
            "n_boot": int(n_boot),
            "bootstrap_seed": int(bootstrap_seed),
        },
    }

    candidate_metric_map = {
        "candidate_a_global": "dom_global_rate",
        "candidate_b_window": "dom_window_rate",
    }

    for config_name, records in per_subject_records_by_config.items():
        records = list(records)

        dominant_rate: Dict[str, Dict[str, Any]] = {}
        for cand_name, metric_key in candidate_metric_map.items():
            cand: Dict[str, Any] = {}
            for pair_name, _, _ in PR5B_PAIRS:
                deltas = [
                    rec.get("deltas", {}).get(metric_key, {}).get(pair_name)
                    for rec in records
                ]
                deltas = [float(d) for d in deltas if d is not None and np.isfinite(float(d))]
                seed = (
                    int(bootstrap_seed)
                    + 17 * (1 + sum(ord(c) for c in cand_name))
                    + 31 * (1 + sum(ord(c) for c in pair_name))
                    + 7 * (1 + sum(ord(c) for c in config_name))
                )
                cand[pair_name] = _pr5b_pair_summary(
                    deltas,
                    n_boot=int(n_boot),
                    seed=seed,
                    bonferroni_alpha=bonferroni_alpha,
                )
            dominant_rate[cand_name] = cand

        composition: Dict[str, Dict[str, Any]] = {"share": {}}
        for pair_name, _, _ in PR5B_PAIRS:
            seed = (
                int(bootstrap_seed)
                + 41 * (1 + sum(ord(c) for c in pair_name))
                + 13 * (1 + sum(ord(c) for c in config_name))
            )
            composition["share"][pair_name] = _pr5b_share_pair_summary(
                records, pair_name, n_boot=int(n_boot), seed=seed
            )

        summary[config_name] = {
            "config_name": config_name,
            "n_subjects": int(len(records)),
            "n_subjects_dom_agreement_lt_half": int(
                sum(
                    1 for r in records
                    if np.isfinite(float(r.get("dom_agreement", float("nan"))))
                    and float(r["dom_agreement"]) < 0.5
                )
            ),
            "dominant_rate": dominant_rate,
            "composition_diagnostic": composition,
        }

    summary["sensitivity"] = _compute_pr5b_sensitivity_gate(summary)
    return summary


def _compute_pr5b_sensitivity_gate(summary: Dict[str, Any]) -> Dict[str, Any]:
    """PR-5-B §4.4 sensitivity gate, derived **only** from ``dominant_rate``.

    Strong  : candidate A post_vs_baseline same direction in main + aux,
              at least one config Bonferroni passes and the other reaches
              nominal 0.05; AND candidate B post_vs_baseline in main reaches
              nominal 0.05 with same direction.
    Medium  : candidate A satisfies above but candidate B does not.
    Descriptive : candidate A fails.

    Reads only ``summary["main"]["dominant_rate"]`` and
    ``summary["auxiliary"]["dominant_rate"]``; mutating
    ``composition_diagnostic`` cannot change the output.
    """
    nominal_alpha = float(
        summary.get("thresholds", {}).get("nominal_alpha", 0.05)
    )

    def _entry(config: str, candidate: str) -> Dict[str, Any]:
        return (
            summary.get(config, {})
            .get("dominant_rate", {})
            .get(candidate, {})
            .get("post_minus_baseline", {})
        )

    main_a = _entry("main", "candidate_a_global")
    aux_a = _entry("auxiliary", "candidate_a_global")
    main_b = _entry("main", "candidate_b_window")

    def _sign(x: Any) -> int:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return 0
        if not np.isfinite(v):
            return 0
        return int(np.sign(v))

    sign_main_a = _sign(main_a.get("median_delta"))
    sign_aux_a = _sign(aux_a.get("median_delta"))
    sign_main_b = _sign(main_b.get("median_delta"))

    same_direction_a = sign_main_a != 0 and sign_main_a == sign_aux_a

    main_a_bonf = bool(main_a.get("bonferroni_pass", False))
    aux_a_bonf = bool(aux_a.get("bonferroni_pass", False))

    def _nominal(p: Any) -> bool:
        try:
            v = float(p)
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(v) and v < nominal_alpha)

    main_a_nom = _nominal(main_a.get("wilcoxon_p"))
    aux_a_nom = _nominal(aux_a.get("wilcoxon_p"))
    main_b_nom = _nominal(main_b.get("wilcoxon_p"))

    candidate_a_strong = (
        same_direction_a
        and (main_a_bonf or aux_a_bonf)
        and (
            (main_a_bonf and aux_a_nom)
            or (aux_a_bonf and main_a_nom)
            or (main_a_bonf and aux_a_bonf)
        )
    )
    candidate_b_supports = (
        candidate_a_strong
        and main_b_nom
        and sign_main_b == sign_main_a
    )

    overall_strong = bool(candidate_a_strong and candidate_b_supports)
    overall_medium = bool(candidate_a_strong and not candidate_b_supports)
    overall_descriptive = bool(not candidate_a_strong)

    return {
        "overall_strong": overall_strong,
        "overall_medium": overall_medium,
        "overall_descriptive": overall_descriptive,
        "candidate_a_strong": bool(candidate_a_strong),
        "candidate_b_supports": bool(candidate_b_supports),
        "candidate_a_main_sign": int(sign_main_a),
        "candidate_a_aux_sign": int(sign_aux_a),
        "candidate_b_main_sign": int(sign_main_b),
        "main_a_bonferroni_pass": bool(main_a_bonf),
        "aux_a_bonferroni_pass": bool(aux_a_bonf),
        "main_a_nominal_pass": bool(main_a_nom),
        "aux_a_nominal_pass": bool(aux_a_nom),
        "main_b_nominal_pass": bool(main_b_nom),
    }
