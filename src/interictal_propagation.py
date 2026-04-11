from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import kendalltau, spearmanr, wilcoxon
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

try:
    from diptest import diptest as hartigan_diptest
except Exception:  # pragma: no cover - optional import guard
    hartigan_diptest = None


def load_subject_propagation_patterns(subject_dir: Path) -> Dict[str, Any]:
    """Load and align lagPatRank / eventsBool across all blocks for one subject."""
    subject_dir = Path(subject_dir)
    lagpat_files = sorted(subject_dir.glob("*_lagPat.npz"))
    if not lagpat_files:
        raise FileNotFoundError(f"No *_lagPat.npz in {subject_dir}")

    raw_blocks: List[Tuple[List[str], np.ndarray, np.ndarray]] = []
    channel_names: List[str] = []
    channel_index: Dict[str, int] = {}

    for lp_file in lagpat_files:
        lp = np.load(lp_file, allow_pickle=True)
        ranks = np.asarray(lp["lagPatRank"], dtype=float)
        bools = np.asarray(lp["eventsBool"]) > 0
        chns = [str(x) for x in list(lp["chnNames"])]
        if ranks.ndim != 2 or bools.ndim != 2:
            continue
        if ranks.size == 0 or bools.size == 0:
            continue

        n_ev = min(ranks.shape[1], bools.shape[1])
        n_ch = min(ranks.shape[0], bools.shape[0], len(chns))
        if n_ev == 0 or n_ch == 0:
            continue

        ranks = ranks[:n_ch, :n_ev]
        bools = bools[:n_ch, :n_ev]
        chns = chns[:n_ch]

        for ch in chns:
            if ch not in channel_index:
                channel_index[ch] = len(channel_names)
                channel_names.append(ch)
        raw_blocks.append((chns, ranks, bools))

    if not raw_blocks:
        return {
            "ranks": np.zeros((0, 0), dtype=float),
            "bools": np.zeros((0, 0), dtype=bool),
            "channel_names": [],
            "n_blocks_used": 0,
        }

    n_union = len(channel_names)
    aligned_ranks: List[np.ndarray] = []
    aligned_bools: List[np.ndarray] = []

    for chns, ranks, bools in raw_blocks:
        n_ev = ranks.shape[1]
        rank_block = np.zeros((n_union, n_ev), dtype=float)
        bool_block = np.zeros((n_union, n_ev), dtype=bool)
        for row, ch in enumerate(chns):
            idx = channel_index[ch]
            rank_block[idx, :] = ranks[row, :]
            bool_block[idx, :] = bools[row, :]
        aligned_ranks.append(rank_block)
        aligned_bools.append(bool_block)

    return {
        "ranks": np.concatenate(aligned_ranks, axis=1),
        "bools": np.concatenate(aligned_bools, axis=1),
        "channel_names": channel_names,
        "n_blocks_used": len(raw_blocks),
    }


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
    loaded = load_subject_propagation_patterns(subject_dir)
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

    return {
        "dataset": dataset,
        "subject": subject,
        "n_channels": int(ranks.shape[0]),
        "channel_names": channel_names,
        "propagation_stereotypy": stereotypy,
        "mixture": mixture,
        "centered_rank": centered,
        "by_nparticipating": by_n,
        "source_diagnostic": source_diag,
        "cluster": cluster_result,
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
        soz_tau = rec["propagation_stereotypy"]["soz"].get("mean_tau")
        nonsoz_tau = rec["propagation_stereotypy"]["nonsoz"].get("mean_tau")
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
    }
