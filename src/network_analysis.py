"""
Module 4: Network Analysis — Epilepsy Network from HFO Co-activation

**v3: Direction-First Causal Pipeline**

Phase A: Channel-Scale MVP (no MNI coordinates required)
Phase B: + Geometry (MNI coords) — stubs only
Phase C: Source Space (research frontier) — not implemented

Pipeline
--------
1. pairwise association from detections (association window)
2. direction inference before pruning (Wilcoxon + consistency)
3. fusion (broad score + direction confidence + stability)
4. physics constraint (distance + near-zero lag guard)
5. final node pruning with edge-evidence rescue
6. graph metrics (Net Outflow, Betweenness, Local Efficiency)

Public API
----------
``build_hfo_network(npz_path, **kwargs) → NetworkResult``

Design Notes
------------
- ``lag_raw[i, k] - lag_raw[j, k]``  negative ⇒ channel *i* leads channel *j*
- ``adj[i, j] > 0`` means a directed edge from node *i* to node *j*
- Edge weights are Simpson-normalised, not raw co-activation counts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NetworkResult:
    """Complete epilepsy network analysis result (direction-first)."""

    # --- Core graph ---
    adj: np.ndarray
    """(n_sel, n_sel) directed weighted adjacency. ``adj[i,j]>0`` ⇒ i→j."""

    node_names: List[str]
    """(n_sel,) selected channel names."""

    node_weights: np.ndarray
    """(n_sel,) HFO rate of selected nodes (backward compat)."""

    # --- Broad graphs (both stored for comparison) ---
    W_simpson: np.ndarray
    """(n_pool, n_pool) Simpson-normalised broad graph."""

    W_dice: np.ndarray
    """(n_pool, n_pool) Dice-normalised broad graph."""

    W_pruned: np.ndarray
    """(n_sel, n_sel) pruned subgraph (edge_method chosen)."""

    pool_names: List[str]
    """(n_pool,) all channel names in the pool."""

    selected_idx: np.ndarray
    """(n_sel,) indices of selected nodes within pool."""

    # --- Node XYZ features (for *all* pool channels) ---
    node_xyz: Dict[str, np.ndarray]
    """{'X': rate, 'Y': entropy, 'Z': epileptogenicity} arrays of len n_pool."""

    # --- Intermediate matrices (on selected subgraph) ---
    skeleton: np.ndarray
    """(n_sel, n_sel) binary (W_pruned > 0). Backward compat."""

    direction_mask: np.ndarray
    """(n_sel, n_sel) bool — True where direction was assigned."""

    stability: np.ndarray
    """(n_sel, n_sel) temporal stability weights (1 − CV)."""

    cluster_labels: np.ndarray
    """(n_pool,) spectral clustering labels. −1 = excluded."""

    # --- Metrics ---
    metrics: Dict[str, Any]
    """Graph-theory metrics (outflow, betweenness, …)."""

    edge_stats: List[Dict[str, Any]]
    """Per-edge statistical summary."""

    # --- Metadata ---
    n_pool_channels: int
    n_selected: int
    params: Dict[str, Any]
    """All hyper-parameters for reproducibility."""


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _clean_coact(coact: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf with 0, zero the diagonal."""
    out = np.where(np.isfinite(coact), coact, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  A.1 — Broad Graph Construction (Simpson / Dice)
# ═══════════════════════════════════════════════════════════════════════

def build_broad_graph(
    coact_event_count: np.ndarray,
    method: str = "simpson",
    significance_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a normalised co-activation graph from event count matrix.

    Simpson : W_ij = |Ei∩Ej| / min(|Ei|, |Ej|)
    Dice    : W_ij = 2|Ei∩Ej| / (|Ei| + |Ej|)

    Parameters
    ----------
    coact_event_count : (n, n) int
        Co-activation count matrix.  Diagonal = per-channel event count.
    method : 'simpson' | 'dice'
    significance_mask : (n, n) bool, optional
        From :func:`surrogate_significance_test`.  Edges where False are zeroed.

    Returns
    -------
    W : (n, n) float64, symmetric, diagonal=0, values in [0, 1].
    """
    intersection = coact_event_count.astype(np.float64).copy()
    events_count = np.diag(coact_event_count).astype(np.float64)
    np.fill_diagonal(intersection, 0.0)

    if method == "simpson":
        denom = np.minimum.outer(events_count, events_count)
    elif method == "dice":
        denom = np.add.outer(events_count, events_count) / 2.0
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'simpson' or 'dice'.")

    W = np.divide(
        intersection, denom,
        out=np.zeros_like(intersection),
        where=denom > 0,
    )

    # Clip to [0, 1] (safety: real data should already satisfy this).
    np.clip(W, 0.0, 1.0, out=W)

    # Symmetrise (Simpson can be slightly asymmetric due to float rounding).
    W = np.maximum(W, W.T)

    if significance_mask is not None:
        W[~significance_mask] = 0.0

    np.fill_diagonal(W, 0.0)

    n_edges = int((W > 0).sum()) // 2
    logger.info("build_broad_graph(%s): %d nodes, %d edges.", method, W.shape[0], n_edges)
    return W


def _load_detection_starts_from_gpu_npz(
    gpu_npz_path: str,
    pool_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Load per-channel detection start times from *_gpu.npz."""
    d = np.load(gpu_npz_path, allow_pickle=True)
    if "chns_names" not in d or "whole_dets" not in d:
        raise KeyError("gpu npz must contain 'chns_names' and 'whole_dets'.")
    raw_names = [str(x).upper() for x in d["chns_names"].tolist()]
    whole_dets = d["whole_dets"]
    name_to_idx = {name: idx for idx, name in enumerate(raw_names)}
    starts_by_channel: Dict[str, np.ndarray] = {}
    for name in pool_names:
        idx = name_to_idx.get(str(name).upper(), None)
        if idx is None:
            starts_by_channel[str(name)] = np.zeros((0,), dtype=np.float64)
            continue
        arr = np.asarray(whole_dets[idx], dtype=np.float64)
        if arr.size == 0:
            starts_by_channel[str(name)] = np.zeros((0,), dtype=np.float64)
            continue
        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)
        starts = np.asarray(arr[:, 0], dtype=np.float64)
        starts = starts[np.isfinite(starts)]
        starts.sort()
        starts_by_channel[str(name)] = starts
    return starts_by_channel


def _nearest_lags_within_window(
    src_starts: np.ndarray,
    dst_starts: np.ndarray,
    window_sec: float,
) -> np.ndarray:
    """Vectorised nearest-neighbour lag samples `src - dst` within window."""
    src = np.asarray(src_starts, dtype=np.float64)
    dst = np.asarray(dst_starts, dtype=np.float64)
    if src.size == 0 or dst.size == 0:
        return np.zeros((0,), dtype=np.float64)
    if not np.all(src[:-1] <= src[1:]):
        src = np.sort(src)
    if not np.all(dst[:-1] <= dst[1:]):
        dst = np.sort(dst)

    idx = np.searchsorted(dst, src, side="left")
    idx_prev = np.clip(idx - 1, 0, dst.size - 1)
    idx_next = np.clip(idx, 0, dst.size - 1)

    lag_prev = src - dst[idx_prev]
    lag_next = src - dst[idx_next]
    abs_prev = np.abs(lag_prev)
    abs_next = np.abs(lag_next)
    pick_prev = abs_prev <= abs_next

    best_lag = np.where(pick_prev, lag_prev, lag_next)
    best_abs = np.where(pick_prev, abs_prev, abs_next)
    mask = best_abs <= float(window_sec)
    return np.asarray(best_lag[mask], dtype=np.float64)


def compute_pairwise_association(
    starts_by_channel: Mapping[str, np.ndarray],
    pool_names: Sequence[str],
    *,
    assoc_window_ms: float = 40.0,
    sample_cap_per_edge: int = 50000,
) -> Dict[str, np.ndarray]:
    """Compute pairwise association support from detections (vectorised per pair).

    Notes
    -----
    - No triple loop over channel×channel×event.
    - Per pair uses `np.searchsorted` + vectorised nearest-neighbour extraction.
    """
    n = len(pool_names)
    win_sec = float(assoc_window_ms) * 1e-3
    assoc_count = np.zeros((n, n), dtype=np.int64)
    lag_median_ms = np.full((n, n), np.nan, dtype=np.float64)
    lag_consistency = np.zeros((n, n), dtype=np.float64)

    rng = np.random.default_rng(42)
    for i in range(n):
        s_i = np.asarray(starts_by_channel.get(pool_names[i], np.zeros((0,))), dtype=np.float64)
        for j in range(i + 1, n):
            s_j = np.asarray(starts_by_channel.get(pool_names[j], np.zeros((0,))), dtype=np.float64)
            if s_i.size == 0 or s_j.size == 0:
                continue

            # Query shorter side to keep per-pair cost bounded.
            if s_i.size <= s_j.size:
                lags = _nearest_lags_within_window(s_i, s_j, win_sec)
                sign_factor = 1.0
            else:
                lags = _nearest_lags_within_window(s_j, s_i, win_sec)
                sign_factor = -1.0

            if lags.size == 0:
                continue
            if sample_cap_per_edge > 0 and lags.size > int(sample_cap_per_edge):
                keep = rng.choice(lags.size, size=int(sample_cap_per_edge), replace=False)
                lags = lags[np.sort(keep)]
            lags = sign_factor * np.asarray(lags, dtype=np.float64)
            cnt = int(lags.size)
            assoc_count[i, j] = cnt
            assoc_count[j, i] = cnt

            med_ms = float(np.median(lags) * 1e3)
            lag_median_ms[i, j] = med_ms
            lag_median_ms[j, i] = -med_ms
            if med_ms == 0.0:
                cons = 0.5
            else:
                cons = float(np.mean(np.sign(lags) == np.sign(med_ms)))
            lag_consistency[i, j] = cons
            lag_consistency[j, i] = cons

    return {
        "assoc_count": assoc_count,
        "lag_median_ms": lag_median_ms,
        "lag_consistency": lag_consistency,
    }


def _apply_physics_constraint(
    edge_mask: np.ndarray,
    lag_abs_ms: np.ndarray,
    dist_matrix: Optional[np.ndarray],
    *,
    min_dist_mm: float = 10.0,
    lag_vc_ms: float = 3.0,
) -> np.ndarray:
    """Apply volume-conduction guard before final pruning."""
    if dist_matrix is None:
        return edge_mask.copy()
    dist = np.asarray(dist_matrix, dtype=np.float64)
    if dist.shape != edge_mask.shape:
        logger.warning(
            "Physics constraint skipped: dist_matrix shape %s != edge shape %s.",
            dist.shape, edge_mask.shape,
        )
        return edge_mask.copy()
    too_close = dist <= float(min_dist_mm)
    near_zero_lag = lag_abs_ms < float(lag_vc_ms)
    reject = too_close & near_zero_lag
    return edge_mask & (~reject)


def validate_propagation_velocity(
    edge_stats: List[Dict[str, Any]],
    dist_matrix: np.ndarray,
    pool_names: List[str],
    *,
    min_velocity_ms: float = 0.1,
    max_velocity_ms: float = 10.0,
) -> List[Dict[str, Any]]:
    """Annotate directed edges with propagation velocity diagnostics.

    Velocity is computed in m/s using ``dist_mm / |lag_ms|``.
    Edges are not removed here; only flagged.
    """
    dist = np.asarray(dist_matrix, dtype=np.float64)
    name_to_idx = {str(nm): i for i, nm in enumerate(pool_names)}
    out: List[Dict[str, Any]] = []

    for es in edge_stats:
        item = dict(es)
        item["velocity_ms"] = np.nan
        item["velocity_flag"] = "insufficient_data"

        if not bool(es.get("directed", False)):
            out.append(item)
            continue

        src = int(es.get("source", -1))
        tgt = int(es.get("target", -1))
        if src < 0 or tgt < 0:
            src_name = str(es.get("ch_i", ""))
            tgt_name = str(es.get("ch_j", ""))
            src = int(name_to_idx.get(src_name, -1))
            tgt = int(name_to_idx.get(tgt_name, -1))
        if src < 0 or tgt < 0 or src >= dist.shape[0] or tgt >= dist.shape[1]:
            out.append(item)
            continue

        lag_ms = abs(float(es.get("median_lag_ms", np.nan)))
        dist_mm = float(dist[src, tgt])
        if (not np.isfinite(lag_ms)) or lag_ms <= 0 or (not np.isfinite(dist_mm)) or dist_mm <= 0:
            out.append(item)
            continue

        velocity = dist_mm / lag_ms  # mm/ms == m/s
        item["velocity_ms"] = float(velocity)
        if float(min_velocity_ms) <= velocity <= float(max_velocity_ms):
            item["velocity_flag"] = "valid"
        else:
            item["velocity_flag"] = "suspicious"
        out.append(item)
    return out


# ═══════════════════════════════════════════════════════════════════════
#  A.2 — Surrogate Significance Testing
# ═══════════════════════════════════════════════════════════════════════

def surrogate_significance_test(
    events_bool: np.ndarray,
    n_surrogates: int = 200,
    p_threshold: float = 0.05,
) -> np.ndarray:
    """Test whether observed co-activation exceeds chance (circular-shift surrogates).

    Parameters
    ----------
    events_bool : (n_ch, n_events) bool
    n_surrogates : int
    p_threshold : float

    Returns
    -------
    sig_mask : (n_ch, n_ch) bool — True where co-activation is significant.
    """
    n_ch, n_ev = events_bool.shape
    eb = events_bool.astype(np.float64)
    real_coact = (eb @ eb.T) / n_ev

    surr_ge_count = np.zeros((n_ch, n_ch), dtype=np.int64)
    rng = np.random.default_rng(42)

    col_idx = np.arange(n_ev, dtype=np.int64)
    for _ in range(n_surrogates):
        shifts = rng.integers(0, n_ev, size=n_ch)
        idx = (col_idx[None, :] - shifts[:, None]) % n_ev
        shifted = eb[np.arange(n_ch)[:, None], idx]
        surr_coact = (shifted @ shifted.T) / n_ev
        surr_ge_count += (surr_coact >= real_coact).astype(np.int64)

    p_values = surr_ge_count.astype(np.float64) / n_surrogates
    sig_mask = p_values < p_threshold
    np.fill_diagonal(sig_mask, True)

    n_sig = sig_mask.sum() - n_ch
    n_total = n_ch * (n_ch - 1)
    logger.info(
        "Surrogate test: %d / %d pairs significant (p < %.3f, %d surrogates).",
        n_sig, n_total, p_threshold, n_surrogates,
    )
    return sig_mask


# ═══════════════════════════════════════════════════════════════════════
#  A.3 — Node XYZ Features
# ═══════════════════════════════════════════════════════════════════════

def compute_connection_entropy(W: np.ndarray) -> np.ndarray:
    """Normalised connection entropy for each node.

    H_i = −Σ_j p_ij ln(p_ij),  normalised by ln(N_neighbors).

    Low entropy  → specific "accomplice" connections → likely pathological.
    High entropy → diffuse connections to everything → likely artifact / noise.

    Parameters
    ----------
    W : (n, n) edge-weight matrix (diagonal = 0).

    Returns
    -------
    H_norm : (n,) float in [0, 1].  0 = maximally specific, 1 = uniform.
    """
    n = W.shape[0]
    H_norm = np.ones(n, dtype=np.float64)

    for i in range(n):
        w_i = W[i].copy()
        w_i[i] = 0.0
        total = w_i.sum()
        if total < 1e-10:
            continue
        p = w_i / total
        nonzero = p > 0
        n_neighbors = int(nonzero.sum())
        if n_neighbors < 2:
            H_norm[i] = 0.0
            continue
        H = -float(np.sum(p[nonzero] * np.log(p[nonzero])))
        H_max = np.log(n_neighbors)
        H_norm[i] = H / H_max if H_max > 0 else 1.0

    return H_norm


def compute_node_xyz(
    W_broad: np.ndarray,
    events_count: np.ndarray,
    recording_duration_min: float,
    fr_ratio: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-node XYZ pathological features.

    X = HFO Rate (events/min)
    Y = Normalised Connection Entropy (0 = specific, 1 = diffuse)
    Z = FR/(R+FR) ratio (Phase B) or zeros (Phase A)

    Returns
    -------
    X, Y, Z : (n,) float arrays.
    """
    X = events_count.astype(np.float64) / max(recording_duration_min, 1e-6)
    Y = compute_connection_entropy(W_broad)
    Z = fr_ratio.copy() if fr_ratio is not None else np.zeros_like(X)
    return X, Y, Z


# ═══════════════════════════════════════════════════════════════════════
#  A.4 — XYZ Pruning (+ spectral clustering helpers)
# ═══════════════════════════════════════════════════════════════════════

def _eigengap_n_clusters(A: np.ndarray, max_k: int = 20) -> int:
    """Estimate cluster count from the Laplacian eigengap."""
    n = A.shape[0]
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    k = min(max_k, n - 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))[:k]
    gaps = np.diff(eigenvalues)
    if len(gaps) < 2:
        return 2
    n_clusters = int(np.argmax(gaps[1:])) + 2
    return max(2, min(n_clusters, max(n // 3, 2)))


def extract_network_clusters(
    A_co: np.ndarray,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Spectral clustering on a symmetric affinity matrix.

    Returns
    -------
    labels : (n,) int   −1 = excluded, ≥ 0 = cluster id
    n_clusters_used : int
    """
    from sklearn.cluster import SpectralClustering

    n = A_co.shape[0]
    if n < max(4, min_cluster_size + 1):
        logger.warning("Too few channels (%d) for spectral clustering; keeping all.", n)
        return np.zeros(n, dtype=np.int32), 1

    if n_clusters is None:
        n_clusters = _eigengap_n_clusters(A_co)
    n_clusters = max(2, min(n_clusters, n - 1))

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=42,
    )
    labels = sc.fit_predict(A_co)

    for cl in np.unique(labels):
        if cl < 0:
            continue
        if (labels == cl).sum() < min_cluster_size:
            labels[labels == cl] = -1

    n_valid = (labels >= 0).sum()
    logger.info(
        "Spectral clustering: %d clusters, %d/%d nodes retained.", n_clusters, n_valid, n,
    )
    return labels.astype(np.int32), n_clusters


def prune_network(
    W_broad: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    *,
    min_rate: float = 0.5,
    max_entropy: float = 0.85,
    use_spectral: bool = True,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """XYZ multi-dimensional pruning.

    1. X gate — remove low-activity nodes
    2. Y gate — remove high-entropy (diffuse / artifact) nodes
    3. Z gate — spectral clustering on survivors (Phase A) or FR filter (Phase B)

    Parameters
    ----------
    W_broad : (n, n)  broad graph (Simpson or Dice).
    X, Y, Z : (n,)    per-node features.
    min_rate : float   minimum HFO rate (events/min).
    max_entropy : float  maximum normalised connection entropy.
    use_spectral : bool  apply spectral clustering after XY gating.
    min_cluster_size, n_clusters : spectral clustering params.

    Returns
    -------
    selected_idx : (n_sel,) indices into the pool.
    W_pruned : (n_sel, n_sel) subgraph.
    cluster_labels : (n,) labels (−1 = excluded).
    """
    n = W_broad.shape[0]
    labels = np.full(n, -1, dtype=np.int32)

    x_pass = X >= min_rate
    y_pass = Y <= max_entropy
    node_mask = x_pass & y_pass
    candidate_idx = np.where(node_mask)[0]

    n_x_fail = int((~x_pass).sum())
    n_y_fail = int(x_pass.sum() - node_mask.sum())
    logger.info(
        "XYZ pruning: %d/%d pass XY gate  (X-fail=%d, Y-fail=%d).",
        len(candidate_idx), n, n_x_fail, n_y_fail,
    )

    if len(candidate_idx) < max(4, min_cluster_size + 1):
        selected_idx = candidate_idx
        labels[candidate_idx] = 0
    elif use_spectral:
        W_sub = W_broad[np.ix_(candidate_idx, candidate_idx)]
        sub_labels, _ = extract_network_clusters(W_sub, min_cluster_size, n_clusters)
        for si, ci in enumerate(candidate_idx):
            labels[ci] = sub_labels[si]
        selected_idx = candidate_idx[sub_labels >= 0]
    else:
        selected_idx = candidate_idx
        labels[candidate_idx] = 0

    W_pruned = W_broad[np.ix_(selected_idx, selected_idx)]
    logger.info("Pruning result: %d nodes selected, %d edges.",
                len(selected_idx), int((W_pruned > 0).sum()) // 2)
    return selected_idx, W_pruned, labels


# ═══════════════════════════════════════════════════════════════════════
#  A.5 — Direction Injection (Wilcoxon + Consistency)
# ═══════════════════════════════════════════════════════════════════════

def inject_direction(
    W_pruned: np.ndarray,
    lag_raw: np.ndarray,
    events_bool: np.ndarray,
    *,
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Assign direction to edges in the pruned graph via statistical lag analysis.

    For every edge (i, j) where W_pruned[i,j] > 0:
    1. Wilcoxon signed-rank: is the median lag ≠ 0?
    2. Direction consistency: ≥ ``consistency_thresh`` agree on sign?
    3. Zero-lag guard: |median lag| > ``lag_thresh_ms``.

    Returns
    -------
    adj : (n, n) float   directed adjacency (consistency values).
    direction_mask : (n, n) bool
    edge_stats : list of dict
    """
    from scipy.stats import wilcoxon as wilcoxon_test

    n = W_pruned.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    direction_mask = np.zeros((n, n), dtype=bool)
    edge_stats: List[Dict[str, Any]] = []
    lag_thresh_sec = lag_thresh_ms * 1e-3

    for i in range(n):
        for j in range(i + 1, n):
            if W_pruned[i, j] <= 0:
                continue

            both = events_bool[i] & events_bool[j]
            n_coact = int(both.sum())

            if n_coact < min_events:
                edge_stats.append(_edge_stat(i, j, n_coact, reason="too_few"))
                continue

            lags = lag_raw[i, both] - lag_raw[j, both]
            lags = lags[np.isfinite(lags)]
            if len(lags) < min_events:
                edge_stats.append(_edge_stat(i, j, n_coact, reason="nan_lags"))
                continue

            try:
                _, p_val = wilcoxon_test(lags)
            except ValueError:
                p_val = 1.0
            p_val = float(p_val)

            median_lag = float(np.median(lags))

            if p_val > p_value_thresh:
                edge_stats.append(
                    _edge_stat(i, j, n_coact, median_lag * 1e3, p_val, reason="p_value"))
                continue

            if abs(median_lag) < lag_thresh_sec:
                edge_stats.append(
                    _edge_stat(i, j, n_coact, median_lag * 1e3, p_val, reason="zero_lag"))
                continue

            direction = np.sign(median_lag)
            consistency = float(np.mean(np.sign(lags) == direction))

            if consistency < consistency_thresh:
                edge_stats.append(
                    _edge_stat(i, j, n_coact, median_lag * 1e3, p_val,
                               consistency=consistency, reason="low_consistency"))
                continue

            if median_lag < 0:
                adj[i, j] = consistency
                direction_mask[i, j] = True
                src, tgt = i, j
            else:
                adj[j, i] = consistency
                direction_mask[j, i] = True
                src, tgt = j, i

            edge_stats.append(
                _edge_stat(i, j, n_coact, median_lag * 1e3, p_val,
                           consistency=consistency, directed=True, source=src, target=tgt,
                           reason="directed"))

    n_directed = int(direction_mask.sum())
    n_candidates = int((W_pruned > 0).sum()) // 2
    logger.info("Direction: %d / %d edges directed.", n_directed, n_candidates)
    return adj, direction_mask, edge_stats


def _edge_stat(
    i: int, j: int, n_coact: int,
    median_lag_ms: float = 0.0, p_value: float = 1.0, *,
    consistency: float = 0.0, directed: bool = False,
    source: int = -1, target: int = -1, reason: str = "",
) -> Dict[str, Any]:
    return {
        "i": i, "j": j, "n_coactive": n_coact,
        "median_lag_ms": median_lag_ms, "p_value": p_value,
        "consistency": consistency, "directed": directed,
        "source": source, "target": target, "reason": reason,
    }


# ═══════════════════════════════════════════════════════════════════════
#  A.6 — Stability Weights
# ═══════════════════════════════════════════════════════════════════════

def compute_stability_weights(
    lag_raw: np.ndarray,
    events_bool: np.ndarray,
    event_times: np.ndarray,
    *,
    window_sec: float = 300.0,
    min_windows: int = 3,
    min_events_per_window: int = 5,
    min_coactive_per_pair: int = 3,
) -> np.ndarray:
    """Per-edge temporal stability: ``1 − CV(consistency across time windows)``."""
    n_ch = lag_raw.shape[0]
    t_min, t_max = float(event_times.min()), float(event_times.max())
    bin_starts = np.arange(t_min, t_max, window_sec)

    if len(bin_starts) < min_windows:
        logger.info("Stability: too few windows (%d < %d). Returning uniform.", len(bin_starts), min_windows)
        return np.ones((n_ch, n_ch), dtype=np.float64)

    window_cons_list: List[np.ndarray] = []
    for t_start in bin_starts:
        t_end = t_start + window_sec
        win_mask = (event_times >= t_start) & (event_times < t_end)
        if win_mask.sum() < min_events_per_window:
            continue
        lag_win = lag_raw[:, win_mask]
        eb_win = events_bool[:, win_mask]
        cons = np.full((n_ch, n_ch), np.nan, dtype=np.float64)
        for i in range(n_ch):
            for j in range(i + 1, n_ch):
                both = eb_win[i] & eb_win[j]
                if both.sum() < min_coactive_per_pair:
                    continue
                lags = lag_win[i, both] - lag_win[j, both]
                lags = lags[np.isfinite(lags)]
                if len(lags) < min_coactive_per_pair:
                    continue
                med = np.median(lags)
                c = float(np.mean(np.sign(lags) == np.sign(med))) if med != 0 else 0.5
                cons[i, j] = cons[j, i] = c
        window_cons_list.append(cons)

    if len(window_cons_list) < min_windows:
        logger.info("Stability: too few valid windows (%d). Returning uniform.", len(window_cons_list))
        return np.ones((n_ch, n_ch), dtype=np.float64)

    import warnings
    stacked = np.stack(window_cons_list)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_c = np.nanmean(stacked, axis=0)
        std_c = np.nanstd(stacked, axis=0)
        cv = np.where(mean_c > 0, std_c / mean_c, 1.0)
        stability = 1.0 - np.clip(cv, 0.0, 1.0)

    stability = np.where(np.isfinite(stability), stability, 0.0)
    logger.info("Stability: %d windows, mean=%.3f.", len(window_cons_list), float(np.nanmean(stability)))
    return stability


# ═══════════════════════════════════════════════════════════════════════
#  A.8 — Graph Theory Metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_network_metrics(
    adj: np.ndarray,
    ch_names: List[str],
) -> Dict[str, Any]:
    """Core graph-theory metrics for the directed weighted epilepsy network."""
    import networkx as nx

    n = adj.shape[0]
    G = nx.DiGraph()
    for name in ch_names:
        G.add_node(name)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(ch_names[i], ch_names[j], weight=float(adj[i, j]))

    metrics: Dict[str, Any] = {}

    outflow: Dict[str, float] = {}
    for node in G.nodes():
        out_w = float(G.out_degree(node, weight="weight"))
        in_w = float(G.in_degree(node, weight="weight"))
        total = out_w + in_w
        outflow[node] = (out_w - in_w) / total if total > 0 else 0.0
    metrics["net_outflow"] = outflow

    metrics["out_degree"] = {n: float(d) for n, d in G.out_degree(weight="weight")}
    metrics["in_degree"] = {n: float(d) for n, d in G.in_degree(weight="weight")}
    metrics["betweenness"] = {
        k: float(v) for k, v in nx.betweenness_centrality(G, weight="weight").items()
    }

    G_und = G.to_undirected()
    metrics["global_efficiency"] = float(nx.global_efficiency(G_und))

    local_eff: Dict[str, float] = {}
    for node in G_und.nodes():
        neighbors = list(G_und.neighbors(node))
        if len(neighbors) < 2:
            local_eff[node] = 0.0
            continue
        subg = G_und.subgraph(neighbors)
        local_eff[node] = float(nx.global_efficiency(subg))
    metrics["local_efficiency"] = local_eff

    metrics["source_ranking"] = sorted(outflow.items(), key=lambda kv: kv[1], reverse=True)
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = float(nx.density(G))

    return metrics


# ═══════════════════════════════════════════════════════════════════════
#  A.9 — One-Stop API
# ═══════════════════════════════════════════════════════════════════════

def build_hfo_network(
    group_analysis_npz: str,
    dist_matrix: Optional[np.ndarray] = None,
    *,
    detections_npz_path: Optional[str] = None,
    # — Step 1: Broad graph —
    edge_method: str = "simpson",
    run_surrogate: bool = True,
    n_surrogates: int = 200,
    min_rate: float = 0.5,
    max_entropy: float = 0.85,
    use_spectral: bool = True,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
    stability_window_sec: float = 300.0,
    assoc_window_ms: float = 40.0,
    min_pair_events: int = 5,
    tau_assoc: float = 20.0,
    tau_lag_ms: float = 10.0,
    fusion_w_b: float = 0.35,
    fusion_w_d: float = 0.45,
    fusion_w_s: float = 0.20,
    d_strong: float = 0.35,
    b_min: float = 0.2,
    min_dist_mm: float = 10.0,
    lag_vc_ms: float = 3.0,
    sample_cap_per_edge: int = 50000,
) -> NetworkResult:
    """Unified entrypoint: direction-first causal pipeline."""
    return _build_network_direction_first(
        group_analysis_npz,
        detections_npz_path=detections_npz_path,
        dist_matrix=dist_matrix,
        edge_method=edge_method,
        run_surrogate=run_surrogate,
        n_surrogates=n_surrogates,
        min_rate=min_rate,
        max_entropy=max_entropy,
        use_spectral=use_spectral,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        min_events=min_events,
        lag_thresh_ms=lag_thresh_ms,
        consistency_thresh=consistency_thresh,
        p_value_thresh=p_value_thresh,
        stability_window_sec=stability_window_sec,
        assoc_window_ms=assoc_window_ms,
        min_pair_events=min_pair_events,
        tau_assoc=tau_assoc,
        tau_lag_ms=tau_lag_ms,
        fusion_w_b=fusion_w_b,
        fusion_w_d=fusion_w_d,
        fusion_w_s=fusion_w_s,
        d_strong=d_strong,
        b_min=b_min,
        min_dist_mm=min_dist_mm,
        lag_vc_ms=lag_vc_ms,
        sample_cap_per_edge=sample_cap_per_edge,
    )


def _build_network_direction_first(
    group_analysis_npz: str,
    dist_matrix: Optional[np.ndarray] = None,
    *,
    detections_npz_path: Optional[str] = None,
    edge_method: str = "simpson",
    run_surrogate: bool = True,
    n_surrogates: int = 200,
    min_rate: float = 0.5,
    max_entropy: float = 0.85,
    use_spectral: bool = True,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
    stability_window_sec: float = 300.0,
    assoc_window_ms: float = 40.0,
    min_pair_events: int = 5,
    tau_assoc: float = 20.0,
    tau_lag_ms: float = 10.0,
    fusion_w_b: float = 0.35,
    fusion_w_d: float = 0.45,
    fusion_w_s: float = 0.20,
    d_strong: float = 0.35,
    b_min: float = 0.2,
    min_dist_mm: float = 10.0,
    lag_vc_ms: float = 3.0,
    sample_cap_per_edge: int = 50000,
) -> NetworkResult:
    """Private implementation: direction-first + fusion + physics + final prune."""
    from .group_event_analysis import load_group_analysis_results

    data = load_group_analysis_results(group_analysis_npz)
    core_names: List[str] = list(data["ch_names"])
    lag_raw: np.ndarray = data["lag_raw"]
    events_bool: np.ndarray = data["events_bool"]
    event_windows: np.ndarray = data["event_windows"]

    has_all = "coact_all_event_count" in data and "coact_all_ch_names" in data
    if has_all:
        coact_count = np.asarray(data["coact_all_event_count"])
        pool_names = [str(x) for x in data["coact_all_ch_names"]]
    else:
        coact_count = np.asarray(data["coact_event_count"])
        pool_names = list(core_names)
    n_pool = len(pool_names)
    logger.info("Causal mode: pool=%d channels.", n_pool)

    W_simpson = build_broad_graph(coact_count, method="simpson")
    W_dice = build_broad_graph(coact_count, method="dice")
    if run_surrogate and events_bool is not None:
        sig_mask = surrogate_significance_test(events_bool, n_surrogates)
        core_pool_idx = []
        for cn in core_names:
            try:
                core_pool_idx.append(pool_names.index(cn))
            except ValueError:
                pass
        if len(core_pool_idx) == sig_mask.shape[0]:
            for ci_local, ci_pool in enumerate(core_pool_idx):
                for cj_local, cj_pool in enumerate(core_pool_idx):
                    if not sig_mask[ci_local, cj_local]:
                        W_simpson[ci_pool, cj_pool] = 0.0
                        W_dice[ci_pool, cj_pool] = 0.0

    method_norm = str(edge_method).strip().lower()
    if method_norm not in {"simpson", "dice"}:
        raise ValueError(
            f"Invalid edge_method='{edge_method}'. Expected 'simpson' or 'dice'."
        )
    W_active = W_simpson if method_norm == "simpson" else W_dice

    events_count = np.diag(coact_count).astype(np.float64)
    duration_min = _recording_duration_min(event_windows)
    X, Y, Z = compute_node_xyz(W_active, events_count, duration_min)

    core_name_to_idx = {name: ci for ci, name in enumerate(core_names)}
    n_events = lag_raw.shape[1]
    lag_pool = np.full((n_pool, n_events), np.nan, dtype=np.float64)
    eb_pool = np.zeros((n_pool, n_events), dtype=bool)
    for pi, name in enumerate(pool_names):
        ci = core_name_to_idx.get(name, -1)
        if ci >= 0:
            lag_pool[pi] = lag_raw[ci]
            eb_pool[pi] = events_bool[ci]

    if detections_npz_path:
        try:
            starts = _load_detection_starts_from_gpu_npz(detections_npz_path, pool_names)
            pair = compute_pairwise_association(
                starts,
                pool_names,
                assoc_window_ms=float(assoc_window_ms),
                sample_cap_per_edge=int(sample_cap_per_edge),
            )
            assoc_count = pair["assoc_count"].astype(np.float64)
        except Exception as exc:
            logger.warning(
                "Falling back to coact counts for association (%s): %s",
                detections_npz_path, exc,
            )
            assoc_count = np.asarray(coact_count, dtype=np.float64)
    else:
        assoc_count = np.asarray(coact_count, dtype=np.float64)
    np.fill_diagonal(assoc_count, 0.0)

    assoc_support = 1.0 - np.exp(-assoc_count / max(float(tau_assoc), 1e-6))
    candidate_mask = assoc_count >= float(min_pair_events)
    W_candidate = np.where(candidate_mask, assoc_support, 0.0)

    # Direction-first: infer direction on association candidates before pruning.
    adj_dir_pool, dir_mask_pool, edge_stats_pool = inject_direction(
        W_candidate,
        lag_pool,
        eb_pool,
        min_events=min_events,
        lag_thresh_ms=lag_thresh_ms,
        consistency_thresh=consistency_thresh,
        p_value_thresh=p_value_thresh,
    )

    event_times = event_windows[:, 0]
    stab_pool = compute_stability_weights(
        lag_pool, eb_pool, event_times, window_sec=stability_window_sec,
    )

    lag_abs_ms = np.full((n_pool, n_pool), np.inf, dtype=np.float64)
    for es in edge_stats_pool:
        if not es.get("directed", False):
            continue
        src = int(es.get("source", -1))
        tgt = int(es.get("target", -1))
        if src < 0 or tgt < 0:
            continue
        lag_abs_ms[src, tgt] = abs(float(es.get("median_lag_ms", 0.0)))

    w_sum = float(fusion_w_b + fusion_w_d + fusion_w_s)
    if w_sum <= 0:
        w_b, w_d, w_s = 0.35, 0.45, 0.20
    else:
        w_b = float(fusion_w_b) / w_sum
        w_d = float(fusion_w_d) / w_sum
        w_s = float(fusion_w_s) / w_sum
    D = np.where(dir_mask_pool, adj_dir_pool, 0.0)
    B = np.asarray(W_active, dtype=np.float64)
    S = np.asarray(stab_pool, dtype=np.float64)
    lag_scale_ms = max(float(tau_lag_ms), 1e-6)
    lag_amp = np.tanh(np.where(np.isfinite(lag_abs_ms), lag_abs_ms, 0.0) / lag_scale_ms)
    W_fuse_pool = assoc_support * (w_b * B + w_d * D + w_s * S) * np.maximum(lag_amp, 0.1)
    rescue_mask = (
        ((D >= float(d_strong)) & (assoc_count >= float(min_pair_events)))
        | ((B >= float(b_min)) & (assoc_count >= float(min_pair_events)))
    )
    rescue_mask &= dir_mask_pool

    physics_mask = _apply_physics_constraint(
        rescue_mask,
        lag_abs_ms,
        dist_matrix,
        min_dist_mm=min_dist_mm,
        lag_vc_ms=lag_vc_ms,
    )
    if dist_matrix is not None:
        edge_stats_pool = validate_propagation_velocity(
            edge_stats_pool,
            np.asarray(dist_matrix, dtype=np.float64),
            list(pool_names),
        )
    adj_pool = np.where(physics_mask, W_fuse_pool, 0.0)

    # Final pruning on fused evidence — with node-level rescue.
    # Step A: standard XYZ prune (same as legacy).
    W_prune_ref = np.maximum(B, assoc_support)
    selected_idx_base, _W_unused, cluster_labels = prune_network(
        W_prune_ref, X, Y, Z,
        min_rate=min_rate, max_entropy=max_entropy,
        use_spectral=use_spectral,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
    )
    # Step B: rescue nodes that are endpoints of a rescued edge.
    # adj_pool > 0 means edge survived fusion + physics; endpoints deserve to stay.
    rescued_node_mask = np.zeros(n_pool, dtype=bool)
    rescued_node_mask[np.where(np.any(adj_pool > 0, axis=1))[0]] = True  # sources
    rescued_node_mask[np.where(np.any(adj_pool > 0, axis=0))[0]] = True  # targets
    rescued_extra = np.where(rescued_node_mask & ~np.isin(np.arange(n_pool), selected_idx_base))[0]
    if rescued_extra.size > 0:
        logger.info(
            "Causal node rescue: %d extra nodes saved by edge evidence (total base=%d).",
            rescued_extra.size, len(selected_idx_base),
        )
        for ri in rescued_extra:
            cluster_labels[ri] = 0  # mark as included
    selected_idx = np.sort(np.union1d(selected_idx_base, rescued_extra).astype(int))
    n_sel = len(selected_idx)
    if n_sel < 2:
        logger.warning("Causal mode selected <2 nodes — returning empty network.")
        return _empty_result(pool_names, cluster_labels, n_pool, W_simpson, W_dice, {"X": X, "Y": Y, "Z": Z})

    sel_names = [pool_names[i] for i in selected_idx]
    adj_sel = adj_pool[np.ix_(selected_idx, selected_idx)]
    W_pruned = W_prune_ref[np.ix_(selected_idx, selected_idx)]
    dir_mask_sel = (adj_sel > 0).astype(bool)
    stab_sel = stab_pool[np.ix_(selected_idx, selected_idx)]

    metrics = compute_network_metrics(adj_sel, sel_names)
    node_weights = X[selected_idx]
    skeleton = (W_pruned > 0).astype(np.float32)

    old_to_new = {int(old): int(new) for new, old in enumerate(selected_idx)}
    edge_stats: List[Dict[str, Any]] = []
    for es in edge_stats_pool:
        if not es.get("directed", False):
            continue
        src_old = int(es.get("source", -1))
        tgt_old = int(es.get("target", -1))
        if src_old not in old_to_new or tgt_old not in old_to_new:
            continue
        src = old_to_new[src_old]
        tgt = old_to_new[tgt_old]
        if adj_sel[src, tgt] <= 0:
            continue
        item = dict(es)
        item["source"] = src
        item["target"] = tgt
        item["source_name"] = sel_names[src]
        item["target_name"] = sel_names[tgt]
        item["ch_i"] = pool_names[src_old]
        item["ch_j"] = pool_names[tgt_old]
        edge_stats.append(item)

    params = {
        "mode": "causal",
        "edge_method": method_norm,
        "run_surrogate": run_surrogate,
        "n_surrogates": n_surrogates,
        "min_rate": min_rate,
        "max_entropy": max_entropy,
        "use_spectral": use_spectral,
        "min_cluster_size": min_cluster_size,
        "n_clusters": n_clusters,
        "min_events": min_events,
        "lag_thresh_ms": lag_thresh_ms,
        "consistency_thresh": consistency_thresh,
        "p_value_thresh": p_value_thresh,
        "stability_window_sec": stability_window_sec,
        "assoc_window_ms": assoc_window_ms,
        "min_pair_events": min_pair_events,
        "tau_assoc": tau_assoc,
        "tau_lag_ms": tau_lag_ms,
        "fusion_w_b": w_b,
        "fusion_w_d": w_d,
        "fusion_w_s": w_s,
        "d_strong": d_strong,
        "b_min": b_min,
        "min_dist_mm": min_dist_mm,
        "lag_vc_ms": lag_vc_ms,
        "detections_npz_path": detections_npz_path,
    }
    metrics["causal_diagnostics"] = {
        "n_candidate_edges": int((candidate_mask.sum() - np.trace(candidate_mask)) // 2),
        "n_directed_pool_edges": int(dir_mask_pool.sum()),
        "n_physics_kept_edges": int((adj_pool > 0).sum()),
        "n_base_pruned_nodes": int(len(selected_idx_base)),
        "n_rescued_nodes": int(rescued_extra.size),
        "n_final_nodes": n_sel,
    }
    vel_vals = [
        float(es.get("velocity_ms", np.nan))
        for es in edge_stats
        if np.isfinite(float(es.get("velocity_ms", np.nan)))
    ]
    n_valid = int(sum(1 for es in edge_stats if es.get("velocity_flag") == "valid"))
    n_suspicious = int(sum(1 for es in edge_stats if es.get("velocity_flag") == "suspicious"))
    metrics["velocity_diagnostics"] = {
        "n_velocity_valid": n_valid,
        "n_velocity_suspicious": n_suspicious,
        "median_velocity_ms": float(np.median(vel_vals)) if vel_vals else None,
    }

    return NetworkResult(
        adj=adj_sel,
        node_names=sel_names,
        node_weights=node_weights,
        W_simpson=W_simpson,
        W_dice=W_dice,
        W_pruned=W_pruned,
        pool_names=pool_names,
        selected_idx=selected_idx,
        node_xyz={"X": X, "Y": Y, "Z": Z},
        skeleton=skeleton,
        direction_mask=dir_mask_sel,
        stability=stab_sel,
        cluster_labels=cluster_labels,
        metrics=metrics,
        edge_stats=edge_stats,
        n_pool_channels=n_pool,
        n_selected=n_sel,
        params=params,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Visualisation — 2D Network Topology
# ═══════════════════════════════════════════════════════════════════════

def plot_network_topology_2d(
    result: NetworkResult,
    output_path: Optional[str] = None,
    *,
    layout: str = "spring",
    figsize: Tuple[float, float] = (12, 10),
    cmap_node: str = "RdBu_r",
    cmap_edge: str = "Greys",
    title: Optional[str] = None,
) -> Any:
    """2D directed network topology (Phase A deliverable)."""
    import matplotlib.pyplot as plt
    import networkx as nx

    adj = result.adj
    names = result.node_names
    n = len(names)

    G = nx.DiGraph()
    for name in names:
        G.add_node(name)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(names[i], names[j], weight=float(adj[i, j]))

    layout_funcs = {
        "spring": lambda g: nx.spring_layout(g, seed=42, weight="weight", k=2.0),
        "circular": nx.circular_layout,
        "spectral": nx.spectral_layout,
        "kamada_kawai": lambda g: nx.kamada_kawai_layout(g, weight="weight"),
    }
    pos = layout_funcs.get(layout, layout_funcs["spring"])(G)

    outflow = result.metrics.get("net_outflow", {})
    node_colors = [outflow.get(name, 0.0) for name in names]
    node_sizes = result.node_weights.copy()
    node_sizes = 300 + 2000 * (node_sizes / (node_sizes.max() + 1e-10))

    edges = list(G.edges(data=True))
    edge_weights = [d.get("weight", 0.1) for _, _, d in edges]
    max_ew = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.5 + 4.0 * (w / max_ew) for w in edge_weights]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edges],
        width=edge_widths, alpha=0.6,
        edge_color=edge_weights, edge_cmap=plt.get_cmap(cmap_edge),
        arrows=True, arrowsize=15, connectionstyle="arc3,rad=0.1",
    )
    vmin = min(node_colors) if node_colors else -1
    vmax = max(node_colors) if node_colors else 1
    abs_max = max(abs(vmin), abs(vmax), 0.01)
    nc = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_sizes, node_color=node_colors,
        cmap=plt.get_cmap(cmap_node), vmin=-abs_max, vmax=abs_max,
        edgecolors="black", linewidths=1.0,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")
    cb = fig.colorbar(nc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Net Outflow  (← Sink | Source →)", fontsize=10)
    ax.set_title(title or "HFO Epilepsy Network (Phase A)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Visualisation — Broad Graph Comparison (Simpson vs Dice)
# ═══════════════════════════════════════════════════════════════════════

def plot_broad_graph_comparison(
    result: NetworkResult,
    output_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (18, 7),
    cmap: str = "YlOrRd",
    title: Optional[str] = None,
    exclude_zero_hfo_nodes: bool = True,
    min_rate_for_display: float = 0.0,
) -> Any:
    """Side-by-side heatmaps of Simpson and Dice broad graphs.

    Nodes with zero HFO events are excluded from display.
    """
    import matplotlib.pyplot as plt

    pool_names = result.pool_names
    X = result.node_xyz["X"]

    # Filter by configurable rate threshold; default excludes only zero-HFO nodes.
    if exclude_zero_hfo_nodes:
        keep = X > max(min_rate_for_display, 0.0)
    else:
        keep = np.ones_like(X, dtype=bool)
    if keep.sum() < 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "No active nodes", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    keep_idx = np.where(keep)[0]
    names_disp = [pool_names[i] for i in keep_idx]
    W_s = result.W_simpson[np.ix_(keep_idx, keep_idx)]
    W_d = result.W_dice[np.ix_(keep_idx, keep_idx)]
    n = len(names_disp)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    im1 = ax1.imshow(W_s, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xticklabels(names_disp, rotation=90, fontsize=6)
    ax1.set_yticklabels(names_disp, fontsize=6)
    ax1.set_title("Simpson Index", fontsize=12, fontweight="bold")
    fig.colorbar(im1, ax=ax1, shrink=0.6)

    im2 = ax2.imshow(W_d, cmap=cmap, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    ax2.set_xticklabels(names_disp, rotation=90, fontsize=6)
    ax2.set_yticklabels(names_disp, fontsize=6)
    ax2.set_title("Dice Index", fontsize=12, fontweight="bold")
    fig.colorbar(im2, ax=ax2, shrink=0.6)

    fig.suptitle(title or "Broad Graph: Simpson vs Dice (zero-HFO nodes excluded)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Visualisation — XY Scatter Diagnostic
# ═══════════════════════════════════════════════════════════════════════

def plot_xy_scatter_diagnostic(
    result: NetworkResult,
    output_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (10, 8),
    title: Optional[str] = None,
) -> Any:
    """X (Rate) vs Y (Entropy) scatter coloured by pruning outcome.

    Green = selected, Red = pruned, Grey = zero-HFO.
    Dashed lines show min_rate and max_entropy thresholds.
    """
    import matplotlib.pyplot as plt

    X = result.node_xyz["X"]
    Y = result.node_xyz["Y"]
    labels = result.cluster_labels
    pool_names = result.pool_names
    params = result.params

    min_rate = params.get("min_rate", 0.5)
    max_entropy = params.get("max_entropy", 0.85)

    fig, ax = plt.subplots(figsize=figsize)

    # Categorise nodes
    zero_mask = X <= 0
    selected_mask = labels >= 0
    pruned_mask = (~zero_mask) & (~selected_mask)

    if zero_mask.any():
        ax.scatter(X[zero_mask], Y[zero_mask], c="lightgrey", s=20, alpha=0.5,
                   label=f"Zero HFO ({zero_mask.sum()})", zorder=1)
    if pruned_mask.any():
        ax.scatter(X[pruned_mask], Y[pruned_mask], c="#e74c3c", s=40, alpha=0.7,
                   edgecolors="black", linewidths=0.3,
                   label=f"Pruned ({pruned_mask.sum()})", zorder=2)
    if selected_mask.any():
        sc = ax.scatter(X[selected_mask], Y[selected_mask], c="#2ecc71", s=80,
                        edgecolors="black", linewidths=0.5,
                        label=f"Selected ({selected_mask.sum()})", zorder=3)
        for i in np.where(selected_mask)[0]:
            ax.annotate(pool_names[i], (X[i], Y[i]),
                        fontsize=6, ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points")

    ax.axvline(min_rate, color="blue", linestyle="--", linewidth=1, alpha=0.7,
               label=f"min_rate={min_rate}")
    ax.axhline(max_entropy, color="orange", linestyle="--", linewidth=1, alpha=0.7,
               label=f"max_entropy={max_entropy}")

    ax.set_xlabel("X: HFO Rate (events/min)", fontsize=11)
    ax.set_ylabel("Y: Connection Entropy (0=specific, 1=diffuse)", fontsize=11)
    ax.set_title(title or "XY Pruning Diagnostic", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Visualisation — Outflow Bar Chart
# ═══════════════════════════════════════════════════════════════════════

def plot_outflow_bar_chart(
    result: NetworkResult,
    output_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (8, 6),
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
) -> Any:
    """Horizontal bar chart of per-node Net Outflow Index."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    outflow = result.metrics.get("net_outflow", {})
    if not outflow:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No outflow data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    sorted_items = sorted(outflow.items(), key=lambda kv: kv[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = np.array([item[1] for item in sorted_items])

    fig, ax = plt.subplots(figsize=figsize)
    cm = plt.get_cmap(cmap)
    abs_max = max(abs(values.min()), abs(values.max()), 0.01)
    norm = mcolors.Normalize(vmin=-abs_max, vmax=abs_max)
    colors = [cm(norm(v)) for v in values]

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Net Outflow Index  (← Sink | Source →)", fontsize=11)
    ax.set_title(title or "Source–Sink Ranking (Phase A)", fontsize=13, fontweight="bold")

    for bar, val in zip(bars, values):
        ha = "left" if val >= 0 else "right"
        offset = 0.02 * abs_max * (1 if val >= 0 else -1)
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}", va="center", ha=ha, fontsize=8)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_adjacency_heatmap(
    result: NetworkResult,
    output_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (9, 8),
    cmap: str = "YlOrRd",
    title: Optional[str] = None,
) -> Any:
    """Heatmap of the directed weighted adjacency matrix."""
    import matplotlib.pyplot as plt

    adj, names = result.adj, result.node_names
    n = len(names)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty network", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(adj, cmap=cmap, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Target (j)")
    ax.set_ylabel("Source (i)")
    ax.set_title(title or "Directed Weighted Adjacency  A[i→j]", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Simpson × Consistency × Stability")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_diff_adjacency_heatmap(
    new_result: NetworkResult,
    old_result: NetworkResult,
    output_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (9, 8),
    cmap: str = "RdBu_r",
    title: Optional[str] = None,
    only_changed: bool = True,
) -> Any:
    """Heatmap of `Diff_Adjacency = New_Adj - Old_Adj`.

    Parameters
    ----------
    only_changed : bool
        If True (default), only show channels that have at least one
        non-zero diff cell — keeps the plot compact and readable.
    """
    import matplotlib.pyplot as plt

    new_names = list(new_result.node_names)
    old_names = list(old_result.node_names)
    union_names = sorted(set(new_names) | set(old_names))
    n = len(union_names)
    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Empty network", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    new_map = {nm: i for i, nm in enumerate(new_names)}
    old_map = {nm: i for i, nm in enumerate(old_names)}
    M_new = np.zeros((n, n), dtype=np.float64)
    M_old = np.zeros((n, n), dtype=np.float64)
    for ui, src in enumerate(union_names):
        i_new = new_map.get(src, None)
        i_old = old_map.get(src, None)
        for uj, tgt in enumerate(union_names):
            j_new = new_map.get(tgt, None)
            j_old = old_map.get(tgt, None)
            if i_new is not None and j_new is not None:
                M_new[ui, uj] = float(new_result.adj[i_new, j_new])
            if i_old is not None and j_old is not None:
                M_old[ui, uj] = float(old_result.adj[i_old, j_old])

    diff_full = M_new - M_old

    # Filter to only channels involved in at least one changed cell.
    if only_changed:
        eps = 1e-12
        row_has_diff = np.any(np.abs(diff_full) > eps, axis=1)
        col_has_diff = np.any(np.abs(diff_full) > eps, axis=0)
        keep = row_has_diff | col_has_diff
        keep_idx = np.where(keep)[0]
        if keep_idx.size < 2:
            # Fall back to showing all if nothing changed.
            keep_idx = np.arange(n)
        disp_names = [union_names[i] for i in keep_idx]
        diff = diff_full[np.ix_(keep_idx, keep_idx)]
    else:
        disp_names = union_names
        diff = diff_full

    nd = len(disp_names)
    abs_max = max(float(np.max(np.abs(diff))), 1e-6)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(diff, cmap=cmap, aspect="auto", interpolation="nearest",
                   vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(nd))
    ax.set_yticks(range(nd))
    ax.set_xticklabels(disp_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(disp_names, fontsize=8)
    ax.set_xlabel("Target (j)")
    ax.set_ylabel("Source (i)")
    subtitle = f" ({nd}/{n} channels with changes)" if only_changed and nd < n else ""
    ax.set_title((title or "Diff Adjacency (New - Old)") + subtitle,
                 fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Positive = rescued edge weight")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def compare_ictal_interictal_networks(
    interictal_result: NetworkResult,
    ictal_result: NetworkResult,
) -> Dict[str, Any]:
    """Compute delta outflow and structural edge differences."""
    names = sorted(set(interictal_result.node_names) | set(ictal_result.node_names))
    n = len(names)
    name_to_idx = {nm: i for i, nm in enumerate(names)}

    i_mat = np.zeros((n, n), dtype=np.float64)
    t_mat = np.zeros((n, n), dtype=np.float64)
    for a, src in enumerate(interictal_result.node_names):
        for b, tgt in enumerate(interictal_result.node_names):
            i_mat[name_to_idx[src], name_to_idx[tgt]] = float(interictal_result.adj[a, b])
    for a, src in enumerate(ictal_result.node_names):
        for b, tgt in enumerate(ictal_result.node_names):
            t_mat[name_to_idx[src], name_to_idx[tgt]] = float(ictal_result.adj[a, b])

    i_out = interictal_result.metrics.get("net_outflow", {})
    t_out = ictal_result.metrics.get("net_outflow", {})
    delta_outflow = {
        nm: float(t_out.get(nm, 0.0) - i_out.get(nm, 0.0))
        for nm in names
    }

    edges_only_ictal: List[Tuple[str, str, float]] = []
    edges_only_interictal: List[Tuple[str, str, float]] = []
    shared_edges_weight_change: List[Dict[str, Any]] = []
    for i, src in enumerate(names):
        for j, tgt in enumerate(names):
            if i == j:
                continue
            w_i = float(i_mat[i, j])
            w_t = float(t_mat[i, j])
            if w_t > 0 and w_i <= 0:
                edges_only_ictal.append((src, tgt, w_t))
            elif w_i > 0 and w_t <= 0:
                edges_only_interictal.append((src, tgt, w_i))
            elif w_i > 0 and w_t > 0:
                shared_edges_weight_change.append({
                    "source": src,
                    "target": tgt,
                    "interictal_weight": w_i,
                    "ictal_weight": w_t,
                    "delta_weight": w_t - w_i,
                })

    return {
        "delta_outflow": delta_outflow,
        "edges_only_ictal": edges_only_ictal,
        "edges_only_interictal": edges_only_interictal,
        "shared_edges_weight_change": shared_edges_weight_change,
    }


def plot_delta_outflow(
    interictal_result: NetworkResult,
    ictal_result: NetworkResult,
    output_path: Optional[str] = None,
) -> Any:
    """Plot per-node interictal vs ictal outflow with ictal-rise highlight."""
    import matplotlib.pyplot as plt

    cmp = compare_ictal_interictal_networks(interictal_result, ictal_result)
    names = list(cmp["delta_outflow"].keys())
    inter = np.array(
        [float(interictal_result.metrics.get("net_outflow", {}).get(nm, 0.0)) for nm in names],
        dtype=np.float64,
    )
    ictal = np.array(
        [float(ictal_result.metrics.get("net_outflow", {}).get(nm, 0.0)) for nm in names],
        dtype=np.float64,
    )
    delta = ictal - inter

    x = np.arange(len(names))
    w = 0.4
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.45), 6))
    ax.bar(x - w / 2, inter, width=w, label="Interictal", color="#4c78a8", alpha=0.85)
    bars_ictal = ax.bar(x + w / 2, ictal, width=w, label="Ictal", color="#f58518", alpha=0.85)
    for bi, d in enumerate(delta):
        if d > 0:
            bars_ictal[bi].set_edgecolor("#d62728")
            bars_ictal[bi].set_linewidth(1.4)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Net Outflow")
    ax.set_title("Ictal vs Interictal Delta Outflow")
    ax.legend()
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_edge_direction_summary(
    result: NetworkResult,
    output_path: Optional[str] = None,
    *,
    figsize: Tuple[float, float] = (12, 5),
) -> Any:
    """Pie chart of edge disposition + lag histogram for directed edges."""
    import matplotlib.pyplot as plt

    edge_stats = result.edge_stats
    if not edge_stats:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No edge statistics", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        return fig

    reason_counts: Dict[str, int] = {}
    directed_lags: List[float] = []
    for es in edge_stats:
        r = es.get("reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1
        if es.get("directed", False):
            directed_lags.append(es.get("median_lag_ms", 0.0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    labels_list = list(reason_counts.keys())
    sizes = list(reason_counts.values())
    label_map = {
        "directed": "Directed", "p_value": "Not significant",
        "zero_lag": "Zero lag", "low_consistency": "Low consistency",
        "too_few": "Too few events", "nan_lags": "NaN lags",
    }
    color_map = {
        "directed": "#2ecc71", "p_value": "#95a5a6", "zero_lag": "#f39c12",
        "low_consistency": "#e74c3c", "too_few": "#bdc3c7", "nan_lags": "#7f8c8d",
    }
    ax1.pie(sizes,
            labels=[label_map.get(l, l) for l in labels_list],
            colors=[color_map.get(l, "#ccc") for l in labels_list],
            autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
    ax1.set_title("Edge Direction Disposition", fontsize=12, fontweight="bold")

    if directed_lags:
        ax2.hist(np.array(directed_lags), bins=30, color="#2ecc71", edgecolor="black", alpha=0.8)
        ax2.axvline(0, color="red", linewidth=1, linestyle="--")
        ax2.set_xlabel("Median Lag (ms)")
        ax2.set_ylabel("Count")
        ax2.set_title(f"Directed Edge Lags (n={len(directed_lags)})", fontsize=12, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No directed edges", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=14)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ═══════════════════════════════════════════════════════════════════════
#  Private utilities
# ═══════════════════════════════════════════════════════════════════════

def _recording_duration_min(event_windows: np.ndarray) -> float:
    if event_windows.size == 0:
        return 1.0
    return float(event_windows[-1, 1] - event_windows[0, 0]) / 60.0


def _empty_result(
    pool_names: List[str],
    cluster_labels: np.ndarray,
    n_pool: int,
    W_simpson: np.ndarray,
    W_dice: np.ndarray,
    node_xyz: Dict[str, np.ndarray],
) -> NetworkResult:
    return NetworkResult(
        adj=np.zeros((0, 0)),
        node_names=[],
        node_weights=np.array([]),
        W_simpson=W_simpson,
        W_dice=W_dice,
        W_pruned=np.zeros((0, 0)),
        pool_names=pool_names,
        selected_idx=np.array([], dtype=np.int64),
        node_xyz=node_xyz,
        skeleton=np.zeros((0, 0)),
        direction_mask=np.zeros((0, 0), dtype=bool),
        stability=np.zeros((0, 0)),
        cluster_labels=cluster_labels,
        metrics={
            "net_outflow": {}, "out_degree": {}, "in_degree": {},
            "betweenness": {}, "local_efficiency": {},
            "global_efficiency": 0.0, "source_ranking": [],
            "n_nodes": 0, "n_edges": 0, "density": 0.0,
        },
        edge_stats=[],
        n_pool_channels=n_pool,
        n_selected=0,
        params={},
    )


# ═══════════════════════════════════════════════════════════════════════
#  I/O — Save / Load NetworkResult
# ═══════════════════════════════════════════════════════════════════════

def save_network_result(result: NetworkResult, npz_path: str) -> str:
    """Persist a :class:`NetworkResult` to disk (npz + json sidecar)."""
    import json as _json

    np.savez_compressed(
        npz_path,
        adj=result.adj,
        node_names=np.asarray(result.node_names, dtype=object),
        node_weights=result.node_weights,
        W_simpson=result.W_simpson,
        W_dice=result.W_dice,
        W_pruned=result.W_pruned,
        pool_names=np.asarray(result.pool_names, dtype=object),
        selected_idx=result.selected_idx,
        node_xyz_X=result.node_xyz["X"],
        node_xyz_Y=result.node_xyz["Y"],
        node_xyz_Z=result.node_xyz["Z"],
        skeleton=result.skeleton,
        direction_mask=result.direction_mask,
        stability=result.stability,
        cluster_labels=result.cluster_labels,
        n_pool_channels=np.array([result.n_pool_channels]),
        n_selected=np.array([result.n_selected]),
    )

    json_path = str(npz_path) + ".json"
    sidecar = {
        "metrics": _serialise_metrics(result.metrics),
        "edge_stats": result.edge_stats,
        "params": result.params,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        _json.dump(sidecar, f, ensure_ascii=False, indent=2)

    logger.info("Network result saved: %s (+%s)", npz_path, json_path)
    return str(npz_path)


def load_network_result(npz_path: str) -> NetworkResult:
    """Reload a :class:`NetworkResult`."""
    import json as _json

    d = np.load(npz_path, allow_pickle=True)

    json_path = str(npz_path) + ".json"
    with open(json_path, "r", encoding="utf-8") as f:
        sidecar = _json.load(f)

    metrics = sidecar.get("metrics", {})
    if "source_ranking" in metrics:
        metrics["source_ranking"] = [tuple(item) for item in metrics["source_ranking"]]

    # Handle both v1 (no W_simpson/W_dice) and v2 formats.
    def _get(key, default_shape=(0, 0)):
        if key in d:
            return np.asarray(d[key])
        return np.zeros(default_shape)

    pool_names = [str(x) for x in d["pool_names"]] if "pool_names" in d else [str(x) for x in d["node_names"]]
    n_pool = len(pool_names)

    return NetworkResult(
        adj=np.asarray(d["adj"]),
        node_names=[str(x) for x in d["node_names"]],
        node_weights=np.asarray(d["node_weights"]),
        W_simpson=_get("W_simpson", (n_pool, n_pool)),
        W_dice=_get("W_dice", (n_pool, n_pool)),
        W_pruned=_get("W_pruned"),
        pool_names=pool_names,
        selected_idx=_get("selected_idx").astype(np.int64).ravel() if "selected_idx" in d else np.array([], dtype=np.int64),
        node_xyz={
            "X": _get("node_xyz_X", (n_pool,)).ravel(),
            "Y": _get("node_xyz_Y", (n_pool,)).ravel(),
            "Z": _get("node_xyz_Z", (n_pool,)).ravel(),
        },
        skeleton=np.asarray(d["skeleton"]),
        direction_mask=np.asarray(d["direction_mask"]).astype(bool),
        stability=np.asarray(d["stability"]),
        cluster_labels=np.asarray(d["cluster_labels"]) if "cluster_labels" in d else np.array([]),
        metrics=metrics,
        edge_stats=sidecar.get("edge_stats", []),
        n_pool_channels=int(np.asarray(d["n_pool_channels"]).ravel()[0]) if "n_pool_channels" in d else n_pool,
        n_selected=int(np.asarray(d["n_selected"]).ravel()[0]) if "n_selected" in d else 0,
        params=sidecar.get("params", {}),
    )


def _serialise_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in metrics.items():
        if isinstance(v, dict):
            sub: Dict[str, Any] = {}
            for kk, vv in v.items():
                if isinstance(vv, (np.integer, np.floating)):
                    sub[str(kk)] = vv.item()
                elif isinstance(vv, (int, float, str, bool)) or vv is None:
                    sub[str(kk)] = vv
                else:
                    sub[str(kk)] = vv
            out[k] = sub
        elif isinstance(v, list):
            out[k] = [list(item) if isinstance(item, tuple) else item for item in v]
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out
