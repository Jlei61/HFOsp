"""Per-subject seizure clustering on Schroeder/Panagiotopoulou pathway-
dissimilarity (topic5 PR-1).

Feature  = per-channel ``t_ER_onset`` vector (NaN where CUSUM not triggered)
Distance = ``1 − Spearman(t_i, t_j)`` over channels finite in BOTH (≥5)
Linkage  = UPGMA (scipy ``method='average'``)
k select = silhouette + min_cluster_size guard, channel-wise permutation null
Outlier  = singleton clusters (size 1) split off into ``outlier_flag``;
           true subtypes require size ≥ 2 (D6)
EEG-realign sanity = value shift + same-subset comparison (D5)
PR-1 propagation cross-link = template-vs-centroid Spearman with valid_mask (D1)

This module owns the math; ``scripts/cluster_ictal_seizures.py`` owns I/O,
plotting, and CLI.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np


# ===========================================================================
# Distance


def pairwise_spearman_dissim(
    onset_matrix: np.ndarray,
    *,
    min_overlap: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairwise ``1 − Spearman`` distance between seizure columns.

    Parameters
    ----------
    onset_matrix
        ``(n_channels, n_seizures)`` ``t_onset_sec`` matrix; NaN where the
        channel did not trigger CUSUM in the given seizure.
    min_overlap
        Minimum number of channels with finite onset in BOTH columns
        required to compute a distance. Pairs below this set
        ``D[i, j] = NaN`` and ``mask[i, j] = False``.

    Returns
    -------
    D : (n_sz, n_sz) symmetric float, diagonal = 0 (or NaN if column all-NaN)
    mask : (n_sz, n_sz) bool, True where distance is valid
    n_overlap : (n_sz, n_sz) int, count of finite-in-both channels
    """
    onset = np.asarray(onset_matrix, dtype=np.float64)
    if onset.ndim != 2:
        raise ValueError("onset_matrix must be 2D (n_channels, n_seizures)")
    n_sz = onset.shape[1]
    finite = np.isfinite(onset)
    D = np.full((n_sz, n_sz), np.nan, dtype=np.float64)
    mask = np.zeros((n_sz, n_sz), dtype=bool)
    n_overlap = np.zeros((n_sz, n_sz), dtype=np.int64)

    for i in range(n_sz):
        for j in range(i, n_sz):
            both = finite[:, i] & finite[:, j]
            n_ov = int(both.sum())
            n_overlap[i, j] = n_overlap[j, i] = n_ov
            if i == j:
                if n_ov >= 1:
                    D[i, j] = 0.0
                    mask[i, j] = True
                continue
            if n_ov < int(min_overlap):
                continue
            a = onset[both, i]
            b = onset[both, j]
            ra = _rankdata(a)
            rb = _rankdata(b)
            if ra.std() == 0 or rb.std() == 0:
                continue
            rho = float(np.corrcoef(ra, rb)[0, 1])
            d_ij = 1.0 - rho
            D[i, j] = D[j, i] = d_ij
            mask[i, j] = mask[j, i] = True
    return D, mask, n_overlap


def _rankdata(arr: np.ndarray) -> np.ndarray:
    """Average-rank for ties (matches scipy.stats.rankdata default)."""
    arr = np.asarray(arr, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty_like(arr)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    # ties → average rank
    _, idx, counts = np.unique(arr, return_inverse=True, return_counts=True)
    sums = np.zeros_like(_, dtype=np.float64)
    for k, r in zip(idx, ranks):
        sums[k] += r
    avg = sums / counts
    return avg[idx] + 1.0


# ===========================================================================
# pair_isolated


def pair_isolated_mask(
    D: np.ndarray,
    *,
    threshold: float = 0.5,
) -> np.ndarray:
    """Flag seizures whose row in D has > threshold fraction of NaN distances.

    Excludes the diagonal from the count. A seizure flagged here is too
    poorly anchored against the rest to enter clustering.
    """
    D = np.asarray(D)
    n = D.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        offdiag = np.delete(D[i], i)
        if offdiag.size == 0:
            out[i] = True
            continue
        frac_nan = np.isnan(offdiag).mean()
        out[i] = frac_nan > float(threshold)
    return out


# ===========================================================================
# UPGMA clustering


def cluster_from_distance_upgma(
    D: np.ndarray,
    k: int,
) -> np.ndarray:
    """UPGMA (average-linkage) hierarchical clustering on a precomputed D.

    Returns 0-based cluster labels for each row (length n).
    Requires D to be a finite symmetric distance matrix with zero diagonal.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    D = np.asarray(D, dtype=np.float64)
    if not np.all(np.isfinite(D)):
        raise ValueError("D must be finite for UPGMA; mask non-finite rows first")
    # symmetrize defensively
    D_sym = 0.5 * (D + D.T)
    np.fill_diagonal(D_sym, 0.0)
    Z = linkage(squareform(D_sym, checks=False), method="average")
    labels_1based = fcluster(Z, t=k, criterion="maxclust")
    return labels_1based - 1


def upgma_linkage_matrix(D: np.ndarray) -> np.ndarray:
    """Just the linkage matrix Z (for plotting dendrograms)."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    D = np.asarray(D, dtype=np.float64)
    D_sym = 0.5 * (D + D.T)
    np.fill_diagonal(D_sym, 0.0)
    return linkage(squareform(D_sym, checks=False), method="average")


# ===========================================================================
# k selection


def select_k_silhouette_with_min_size(
    D: np.ndarray,
    labels_by_k: Mapping[int, np.ndarray],
    *,
    min_cluster_size: int = 2,
    max_k: int,
) -> Tuple[Optional[int], Dict[int, float]]:
    """Pick the k whose silhouette is largest, subject to:

    - ``k ≤ max_k``
    - ≥ 2 clusters present (silhouette undefined otherwise)
    - ≥ 1 cluster has size ≥ ``min_cluster_size`` (else result is purely
      singletons → no real subtypes; ``assign_outliers_and_subtypes``
      would mark everyone as outlier)

    Singletons WITHIN a partition are allowed; ``assign_outliers_and_subtypes``
    splits them off as outliers downstream. This matches the D6 semantics:
    ``[5, 1]`` is a valid "1 subtype + 1 outlier" solution and must not
    be rejected at the k-selection step.

    Tie-breaks pick the smallest k.
    """
    from sklearn.metrics import silhouette_score

    scores: Dict[int, float] = {}
    candidates: List[Tuple[float, int]] = []
    for k in sorted(labels_by_k):
        if k > max_k:
            continue
        labels = np.asarray(labels_by_k[k])
        unique, counts = np.unique(labels, return_counts=True)
        if len(unique) < 2:
            continue  # silhouette undefined
        if (counts >= int(min_cluster_size)).sum() == 0:
            continue  # all clusters singletons → no real subtypes
        try:
            s = float(silhouette_score(D, labels, metric="precomputed"))
        except ValueError:
            continue
        scores[k] = s
        candidates.append((s, k))
    if not candidates:
        return None, scores
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1], scores


# ===========================================================================
# Channel-wise permutation null (D2)


def channelwise_permutation_null(
    onset_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    B: int = 50,
    rng_seed: int = 0,
    min_overlap: int = 5,
) -> float:
    """Gap = log(within_disp_observed) − median(log(within_disp_perm)).

    Permutation preserves each channel's marginal distribution AND its
    NaN-pattern (only finite values are shuffled within the row), and
    breaks cross-channel seizure pattern. Negative gap means the
    observed clustering is no tighter than channel-shuffled noise.
    """
    onset = np.asarray(onset_matrix, dtype=np.float64)
    labels = np.asarray(labels)

    def _within_disp(D_obs: np.ndarray, lbl: np.ndarray) -> float:
        total = 0.0
        for c in np.unique(lbl):
            members = np.where(lbl == c)[0]
            if members.size < 2:
                continue
            sub = D_obs[np.ix_(members, members)]
            sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                continue
            total += float(np.nansum(sub) / 2.0)
        return total

    D_obs, _, _ = pairwise_spearman_dissim(onset, min_overlap=min_overlap)
    wd_obs = _within_disp(D_obs, labels)
    if wd_obs <= 0:
        wd_obs = 1e-12

    rng = np.random.default_rng(int(rng_seed))
    log_wd_perm = []
    for _ in range(int(B)):
        perm = onset.copy()
        for ch in range(perm.shape[0]):
            row = perm[ch]
            finite_idx = np.where(np.isfinite(row))[0]
            if finite_idx.size < 2:
                continue
            shuffled_vals = rng.permutation(row[finite_idx])
            perm[ch, finite_idx] = shuffled_vals
        D_p, _, _ = pairwise_spearman_dissim(perm, min_overlap=min_overlap)
        wd = _within_disp(D_p, labels)
        log_wd_perm.append(np.log(max(wd, 1e-12)))
    if not log_wd_perm:
        return 0.0
    return float(np.median(log_wd_perm) - np.log(wd_obs))


# ===========================================================================
# Outlier vs subtype split (D6)


def assign_outliers_and_subtypes(
    cluster_labels: np.ndarray,
    *,
    min_subtype_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Singletons → ``outlier_flag=True`` and ``subtype_label=-1``.

    Subtypes are renumbered to a contiguous 0-based index over the
    surviving clusters (those with size ≥ ``min_subtype_size``),
    sorted by descending size for stability.
    """
    cluster_labels = np.asarray(cluster_labels)
    n = cluster_labels.size
    unique, counts = np.unique(cluster_labels, return_counts=True)
    keep = unique[counts >= int(min_subtype_size)]
    # rank kept clusters by descending size for deterministic naming
    sizes = {int(u): int(c) for u, c in zip(unique, counts)}
    keep = sorted(keep.tolist(), key=lambda u: (-sizes[u], u))

    subtype_label = np.full(n, -1, dtype=np.int64)
    for new_id, old_id in enumerate(keep):
        subtype_label[cluster_labels == old_id] = new_id
    outlier_flag = subtype_label == -1
    return subtype_label, outlier_flag


# ===========================================================================
# Sentinel sanity Jaccard helpers


def outlier_jaccard(
    outlier_flag: np.ndarray,
    user_outlier_set: Set[int],
) -> float:
    """Jaccard between algo outlier indices and user-labeled outliers.

    Both empty returns 1.0 by convention.
    """
    algo_set = set(np.where(np.asarray(outlier_flag))[0].tolist())
    if not algo_set and not user_outlier_set:
        return 1.0
    inter = algo_set & set(user_outlier_set)
    union = algo_set | set(user_outlier_set)
    if not union:
        return 1.0
    return float(len(inter) / len(union))


def subtype_jaccard(
    subtype_label: np.ndarray,
    user_main_subtype_set: Set[int],
) -> float:
    """Jaccard between the largest algo subtype and user 'main' set.

    If no subtype assigned (all outliers), returns 0.0.
    """
    subtype_label = np.asarray(subtype_label)
    valid = subtype_label[subtype_label >= 0]
    if valid.size == 0:
        return 0.0
    unique, counts = np.unique(valid, return_counts=True)
    largest = int(unique[np.argmax(counts)])
    algo_main = set(np.where(subtype_label == largest)[0].tolist())
    user_set = set(user_main_subtype_set)
    if not algo_main and not user_set:
        return 1.0
    inter = algo_main & user_set
    union = algo_main | user_set
    if not union:
        return 1.0
    return float(len(inter) / len(union))


# ===========================================================================
# EEG realignment (D5)


def apply_eeg_realignment(
    onset_matrix: np.ndarray,
    eeg_clin_delta: np.ndarray,
    *,
    return_kept_mask: bool = False,
):
    """Subtract Δ_eeg_clin per seizure column to express t_onset relative to
    EEG onset instead of clinical onset.

    Columns where Δ is NaN/None are dropped (they have no EEG annotation).

    Parameters
    ----------
    onset_matrix : (n_ch, n_sz)
    eeg_clin_delta : (n_sz,) — Δ = clin_onset - eeg_onset, NaN if unknown
    return_kept_mask : if True, also return bool[n_sz] indicating columns kept
    """
    onset = np.asarray(onset_matrix, dtype=np.float64)
    deltas = np.asarray(eeg_clin_delta, dtype=np.float64)
    if onset.shape[1] != deltas.size:
        raise ValueError(
            f"onset_matrix has {onset.shape[1]} columns but "
            f"eeg_clin_delta has {deltas.size}"
        )
    kept = np.isfinite(deltas)
    onset_kept = onset[:, kept]
    delta_kept = deltas[kept]
    realigned = onset_kept - delta_kept[None, :]
    if return_kept_mask:
        return realigned, kept
    return realigned


# ===========================================================================
# Orchestrators


QUALIFYING_STATUS = "ok"
DEFAULT_MIN_OVERLAP = 5
DEFAULT_PAIR_ISOLATED_THRESHOLD = 0.5
DEFAULT_MIN_SUBTYPE_SIZE = 2
DEFAULT_MIN_N_OK = 5


def _centroid_per_subtype(
    onset_kept: np.ndarray,
    subtype_label: np.ndarray,
    channels: List[str],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Per-subtype mean t_onset across kept seizures, per channel.

    Outliers (subtype_label == -1) are excluded from centroids. Channels
    that never had a finite onset in any subtype member yield None.
    """
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for s in sorted({int(x) for x in subtype_label if x >= 0}):
        members = np.where(subtype_label == s)[0]
        if members.size == 0:
            continue
        sub = onset_kept[:, members]
        mean = np.where(np.isfinite(sub).any(axis=1),
                         np.nanmean(sub, axis=1), np.nan)
        out[f"subtype_{s}"] = {
            ch: (None if not np.isfinite(v) else float(v))
            for ch, v in zip(channels, mean)
        }
    return out


def _within_subtype_pairwise_mean(
    D_kept: np.ndarray,
    subtype_label: np.ndarray,
) -> float:
    """Mean of within-subtype pairwise (1 − Spearman) distances over all
    subtypes. Used to compute s_sz_within_subtype = 1 − this.
    """
    vals: List[float] = []
    for s in np.unique(subtype_label):
        if s < 0:
            continue
        members = np.where(subtype_label == s)[0]
        if members.size < 2:
            continue
        sub = D_kept[np.ix_(members, members)]
        triu = sub[np.triu_indices(members.size, k=1)]
        triu = triu[np.isfinite(triu)]
        vals.extend(triu.tolist())
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def _overall_pairwise_mean(D_kept: np.ndarray) -> float:
    triu = D_kept[np.triu_indices(D_kept.shape[0], k=1)]
    triu = triu[np.isfinite(triu)]
    if triu.size == 0:
        return float("nan")
    return float(np.mean(triu))


def cluster_subject_band(
    per_er_record: Mapping[str, Any],
    channels: List[str],
    *,
    band: str,
    min_overlap: int = DEFAULT_MIN_OVERLAP,
    pair_isolated_threshold: float = DEFAULT_PAIR_ISOLATED_THRESHOLD,
    min_subtype_size: int = DEFAULT_MIN_SUBTYPE_SIZE,
    min_n_ok: int = DEFAULT_MIN_N_OK,
    permutation_B: int = 50,
    permutation_seed: int = 0,
) -> Dict[str, Any]:
    """End-to-end clustering for one (subject, band) pair.

    Filters: status == "ok" → pair_isolated → UPGMA → silhouette+min_size →
    outlier vs subtype split (D6) → centroids + within/between dispersion.

    Returns SubjectBandResult dict (schema: see plan §"接口契约" + tests).
    """
    from src.atlas_loading import build_onset_matrix

    sz_records = per_er_record.get("seizure_records", [])
    onset_full, statuses, sz_ids = build_onset_matrix(per_er_record, channels)

    # Filter to ok seizures
    ok_idx = [j for j, s in enumerate(statuses) if s == QUALIFYING_STATUS]
    n_ok = len(ok_idx)
    if n_ok < int(min_n_ok):
        return {
            "band": band,
            "status": "insufficient_n",
            "n_sz_total": len(sz_records),
            "n_sz_ok": n_ok,
            "n_sz_pair_isolated": 0,
            "n_sz_effective": 0,
            "seizure_ids_kept": [],
            "seizure_ids_dropped": {
                "by_status": [sz_ids[j] for j, s in enumerate(statuses) if s != QUALIFYING_STATUS],
                "by_pair_isolated": [],
            },
            "D": None, "Z": None,
            "subtype_label": None, "outlier_flag": None,
            "n_subtypes": 0, "subtype_sizes": {}, "n_outliers": 0,
            "chosen_k": None, "silhouette_k": None, "gap_perm_k": None,
            "centroids": {},
            "s_sz_overall": None, "s_sz_within_subtype_mean": None,
        }

    onset_ok = onset_full[:, ok_idx]
    sz_ids_ok = [sz_ids[j] for j in ok_idx]

    # Pairwise distance + pair_isolated filter
    D_ok, _, _ = pairwise_spearman_dissim(onset_ok, min_overlap=min_overlap)
    isolated = pair_isolated_mask(D_ok, threshold=pair_isolated_threshold)
    keep_idx = np.where(~isolated)[0]
    n_eff = int(keep_idx.size)
    seizure_ids_dropped = {
        "by_status": [sz_ids[j] for j, s in enumerate(statuses) if s != QUALIFYING_STATUS],
        "by_pair_isolated": [sz_ids_ok[j] for j in np.where(isolated)[0]],
    }
    if n_eff < int(min_n_ok):
        status_post = "insufficient_post_filter" if n_eff > 0 else "all_pair_isolated"
        return {
            "band": band,
            "status": status_post,
            "n_sz_total": len(sz_records),
            "n_sz_ok": n_ok,
            "n_sz_pair_isolated": int(isolated.sum()),
            "n_sz_effective": n_eff,
            "seizure_ids_kept": [sz_ids_ok[j] for j in keep_idx],
            "seizure_ids_dropped": seizure_ids_dropped,
            "D": None, "Z": None,
            "subtype_label": None, "outlier_flag": None,
            "n_subtypes": 0, "subtype_sizes": {}, "n_outliers": 0,
            "chosen_k": None, "silhouette_k": None, "gap_perm_k": None,
            "centroids": {},
            "s_sz_overall": None, "s_sz_within_subtype_mean": None,
        }

    onset_kept = onset_ok[:, keep_idx]
    D_kept = D_ok[np.ix_(keep_idx, keep_idx)]
    # Replace any NaN inside D_kept (pair-overlap < min_overlap) with the
    # row-median of finite distances; UPGMA + silhouette require finite.
    D_kept = _impute_finite_with_median(D_kept)

    # Build labels for k = 2..max_k
    max_k = max(2, int(np.ceil(n_eff / 3.0)))
    labels_by_k: Dict[int, np.ndarray] = {}
    for k in range(2, max_k + 1):
        try:
            labels_by_k[k] = cluster_from_distance_upgma(D_kept, k=k)
        except (ValueError, RuntimeError):
            continue

    chosen_k, scores = select_k_silhouette_with_min_size(
        D_kept, labels_by_k,
        min_cluster_size=min_subtype_size,
        max_k=max_k,
    )
    Z = upgma_linkage_matrix(D_kept).tolist()

    if chosen_k is None:
        # No k passes min_subtype_size; treat all as outliers
        subtype_label = np.full(n_eff, -1, dtype=np.int64)
        outlier_flag = np.ones(n_eff, dtype=bool)
        gap_perm = None
        s_overall = _overall_pairwise_mean(D_kept)
        return {
            "band": band,
            "status": "ok",
            "n_sz_total": len(sz_records),
            "n_sz_ok": n_ok,
            "n_sz_pair_isolated": int(isolated.sum()),
            "n_sz_effective": n_eff,
            "seizure_ids_kept": [sz_ids_ok[j] for j in keep_idx],
            "seizure_ids_dropped": seizure_ids_dropped,
            "D": D_kept.tolist(),
            "Z": Z,
            "subtype_label": subtype_label.tolist(),
            "outlier_flag": outlier_flag.tolist(),
            "n_subtypes": 0,
            "subtype_sizes": {},
            "n_outliers": int(n_eff),
            "chosen_k": None,
            "silhouette_k": None,
            "gap_perm_k": gap_perm,
            "centroids": {},
            "s_sz_overall": (None if not np.isfinite(s_overall) else float(1.0 - s_overall)),
            "s_sz_within_subtype_mean": None,
        }

    cluster_labels = labels_by_k[chosen_k]
    subtype_label, outlier_flag = assign_outliers_and_subtypes(
        cluster_labels, min_subtype_size=min_subtype_size,
    )
    centroids = _centroid_per_subtype(onset_kept, subtype_label, channels)
    n_subtypes = len({int(s) for s in subtype_label if s >= 0})
    subtype_sizes = {
        str(int(s)): int(np.sum(subtype_label == s))
        for s in sorted({int(x) for x in subtype_label if x >= 0})
    }
    n_outliers = int(outlier_flag.sum())
    silhouette = scores.get(chosen_k)

    gap_perm = None
    if permutation_B > 0:
        try:
            gap_perm = channelwise_permutation_null(
                onset_kept, cluster_labels,
                B=permutation_B, rng_seed=permutation_seed,
                min_overlap=min_overlap,
            )
        except Exception:
            gap_perm = None

    s_overall = _overall_pairwise_mean(D_kept)
    s_within = _within_subtype_pairwise_mean(D_kept, subtype_label)
    return {
        "band": band,
        "status": "ok",
        "n_sz_total": len(sz_records),
        "n_sz_ok": n_ok,
        "n_sz_pair_isolated": int(isolated.sum()),
        "n_sz_effective": n_eff,
        "seizure_ids_kept": [sz_ids_ok[j] for j in keep_idx],
        "seizure_ids_dropped": seizure_ids_dropped,
        "D": D_kept.tolist(),
        "Z": Z,
        "subtype_label": subtype_label.tolist(),
        "outlier_flag": outlier_flag.tolist(),
        "n_subtypes": n_subtypes,
        "subtype_sizes": subtype_sizes,
        "n_outliers": n_outliers,
        "chosen_k": int(chosen_k),
        "silhouette_k": (None if silhouette is None else float(silhouette)),
        "gap_perm_k": (None if gap_perm is None else float(gap_perm)),
        "centroids": centroids,
        "s_sz_overall": (None if not np.isfinite(s_overall) else float(1.0 - s_overall)),
        "s_sz_within_subtype_mean": (
            None if not np.isfinite(s_within) else float(1.0 - s_within)
        ),
    }


def _impute_finite_with_median(D: np.ndarray) -> np.ndarray:
    """Replace NaN entries in a symmetric distance matrix with the median
    of finite entries in their row (mirrored to keep symmetry).
    """
    out = np.array(D, dtype=np.float64, copy=True)
    n = out.shape[0]
    for i in range(n):
        row = out[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            continue
        med = float(np.median(finite))
        for j in range(n):
            if i == j:
                continue
            if not np.isfinite(out[i, j]):
                out[i, j] = med
                out[j, i] = med
    np.fill_diagonal(out, 0.0)
    return out


def cluster_subject(
    per_subject_json: Mapping[str, Any],
    *,
    channels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Cluster both bands of a v2.3 per-subject JSON; report cross-band ARI.

    The dual-band ARI is *legitimate* (same seizure items, two label vectors)
    so it directly measures whether gamma + broad agree on subtyping.
    """
    from sklearn.metrics import adjusted_rand_score

    subject = per_subject_json.get("subject")
    per_er = per_subject_json.get("per_er", {})
    if channels is None:
        # Use union of channel keys across both bands' r_sz dicts
        chs: Set[str] = set()
        for band in ("gamma_ER", "broad_ER"):
            rec = per_er.get(band, {}) or {}
            chs.update((rec.get("r_sz") or {}).keys())
        channels = sorted(chs)

    per_band_result: Dict[str, Any] = {}
    for band in ("gamma_ER", "broad_ER"):
        rec = per_er.get(band)
        if not rec:
            per_band_result[band] = {
                "band": band, "status": "missing_band",
            }
            continue
        per_band_result[band] = cluster_subject_band(
            rec, list(channels), band=band,
        )

    # ARI gamma vs broad — only if both have valid subtype_label on
    # the same kept seizure subset (intersect by seizure_id).
    g = per_band_result.get("gamma_ER", {})
    b = per_band_result.get("broad_ER", {})
    ari = None
    if (g.get("status") == "ok" and b.get("status") == "ok"
            and g.get("subtype_label") and b.get("subtype_label")):
        g_ids = g.get("seizure_ids_kept", [])
        b_ids = b.get("seizure_ids_kept", [])
        common_ids = [s for s in g_ids if s in b_ids]
        if len(common_ids) >= 4:
            g_lookup = {sid: lbl for sid, lbl in zip(g_ids, g["subtype_label"])}
            b_lookup = {sid: lbl for sid, lbl in zip(b_ids, b["subtype_label"])}
            g_lab = np.array([g_lookup[s] for s in common_ids])
            b_lab = np.array([b_lookup[s] for s in common_ids])
            try:
                ari = float(adjusted_rand_score(g_lab, b_lab))
            except Exception:
                ari = None

    return {
        "schema_version": "topic5_pr1_seizure_clustering_v1",
        "subject": subject,
        "n_seizures_total": per_subject_json.get("n_seizures_total"),
        "channels_used": list(channels),
        "per_band": per_band_result,
        "ari_gamma_vs_broad": ari,
    }


# ===========================================================================
# PR-1 propagation template match with valid_mask (D1)


def match_template_to_pr1_with_valid_mask(
    centroid: Mapping[str, float],
    template_rank: Mapping[str, Any],
    valid_mask: Mapping[str, bool],
    *,
    min_overlap: int = 5,
) -> Dict[str, Any]:
    """Spearman match between an ictal-subtype centroid and a PR-1 cluster
    template_rank, restricted to channels marked valid by valid_mask.

    Per AGENTS.md "Cross-PR Contract Lookups": the PR-1 propagation
    template_rank gives fallback ranks for non-participating channels;
    must NOT include those when computing similarity.

    Returns dict::

        {
            "max_rho": float | NaN,         # Spearman ρ (NOT 1−ρ)
            "n_overlap_valid_only": int,
        }

    For multi-cluster aggregation, call once per (centroid, template) pair
    and take the max ρ at the call site.
    """
    common_chs = []
    a_vals = []
    b_vals = []
    for ch, c_val in centroid.items():
        if ch not in template_rank or ch not in valid_mask:
            continue
        if not valid_mask[ch]:
            continue
        if c_val is None or not np.isfinite(float(c_val)):
            continue
        t_val = template_rank[ch]
        if t_val is None or not np.isfinite(float(t_val)):
            continue
        common_chs.append(ch)
        a_vals.append(float(c_val))
        b_vals.append(float(t_val))
    n_ov = len(common_chs)
    if n_ov < int(min_overlap):
        return {
            "max_rho": np.nan,
            "n_overlap_valid_only": n_ov,
        }
    a = np.array(a_vals)
    b = np.array(b_vals)
    ra = _rankdata(a)
    rb = _rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return {"max_rho": np.nan, "n_overlap_valid_only": n_ov}
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return {"max_rho": rho, "n_overlap_valid_only": n_ov}
