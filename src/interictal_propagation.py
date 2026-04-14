from __future__ import annotations

import datetime
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from zoneinfo import ZoneInfo
from scipy.optimize import linear_sum_assignment
from scipy.stats import kendalltau, spearmanr, wilcoxon
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
