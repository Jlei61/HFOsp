"""
Module 4: Network Analysis — Epilepsy Network from HFO Co-activation

**v2: Build-Prune-Direct Pipeline**

Phase A: Channel-Scale MVP (no MNI coordinates required)
Phase B: + Geometry (MNI coords) — stubs only
Phase C: Source Space (research frontier) — not implemented

Pipeline
--------
1. build_broad_graph   — Simpson / Dice normalised co-activation (wide net)
2. compute_node_xyz    — X=Rate, Y=Entropy, Z=Epileptogenicity
3. prune_network       — XYZ gating + optional spectral clustering
4. inject_direction    — Wilcoxon + consistency on lag_raw
5. composite weights   — Simpson × Consistency × Stability
6. graph metrics       — Net Outflow, Betweenness, Local Efficiency

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Data Classes
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NetworkResult:
    """Complete epilepsy network analysis result (v2: Build-Prune-Direct)."""

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
#  A.7 — Composite Weights
# ═══════════════════════════════════════════════════════════════════════

def compute_composite_weights(
    adj_directed: np.ndarray,
    W_pruned: np.ndarray,
    stability: np.ndarray,
) -> np.ndarray:
    """Phase A composite: Simpson × Consistency × Stability.

    ``adj_directed[i,j]`` carries the consistency value.
    ``W_pruned[i,j]`` carries the Simpson/Dice weight.
    """
    W = np.where(np.isfinite(W_pruned), W_pruned, 0.0)
    return adj_directed * W * stability


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
    # — Step 1: Broad graph —
    edge_method: str = "simpson",
    run_surrogate: bool = True,
    n_surrogates: int = 200,
    # — Step 2: XYZ pruning —
    min_rate: float = 0.5,
    max_entropy: float = 0.85,
    use_spectral: bool = True,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
    # — Step 3: Direction —
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
    # — Step 4: Stability —
    stability_window_sec: float = 300.0,
) -> NetworkResult:
    """One-stop v2 epilepsy network builder (Phase A).

    Pipeline
    --------
    1. Build both Simpson & Dice broad graphs
    2. Compute XYZ features → multi-dimensional pruning
    3. Direction injection (Wilcoxon + consistency)
    4. Stability weights → composite weight → graph metrics
    """
    from .group_event_analysis import load_group_analysis_results

    data = load_group_analysis_results(group_analysis_npz)

    # ── Resolve channel pool ──
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
    logger.info("Channel pool: %d (%s).", n_pool, "all" if has_all else "core")

    # ── Step 1: Build both broad graphs ──
    W_simpson = build_broad_graph(coact_count, method="simpson")
    W_dice = build_broad_graph(coact_count, method="dice")

    # Optional surrogate (applies to core submatrix only).
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

    W_active = W_simpson if edge_method == "simpson" else W_dice

    # ── Step 2: XYZ features + pruning ──
    events_count = np.diag(coact_count).astype(np.float64)
    duration_min = _recording_duration_min(event_windows)
    X, Y, Z = compute_node_xyz(W_active, events_count, duration_min)

    selected_idx, W_pruned, cluster_labels = prune_network(
        W_active, X, Y, Z,
        min_rate=min_rate, max_entropy=max_entropy,
        use_spectral=use_spectral,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
    )

    n_sel = len(selected_idx)
    sel_names = [pool_names[i] for i in selected_idx]
    logger.info("Selected %d / %d nodes: %s", n_sel, n_pool, sel_names)

    if n_sel < 2:
        logger.warning("Fewer than 2 nodes — returning empty network.")
        return _empty_result(pool_names, cluster_labels, n_pool,
                             W_simpson, W_dice,
                             {"X": X, "Y": Y, "Z": Z})

    # ── Map selected → core for lag / events data ──
    core_name_to_idx = {name: ci for ci, name in enumerate(core_names)}
    sel_to_core = np.full(n_sel, -1, dtype=np.int64)
    for si, name in enumerate(sel_names):
        if name in core_name_to_idx:
            sel_to_core[si] = core_name_to_idx[name]

    n_events = lag_raw.shape[1]
    lag_sel = np.full((n_sel, n_events), np.nan, dtype=np.float64)
    eb_sel = np.zeros((n_sel, n_events), dtype=bool)
    for si in range(n_sel):
        ci = sel_to_core[si]
        if ci >= 0:
            lag_sel[si] = lag_raw[ci]
            eb_sel[si] = events_bool[ci]

    # ── Step 3: Direction injection ──
    adj_dir, dir_mask, edge_stats = inject_direction(
        W_pruned, lag_sel, eb_sel,
        min_events=min_events, lag_thresh_ms=lag_thresh_ms,
        consistency_thresh=consistency_thresh, p_value_thresh=p_value_thresh,
    )

    # ── Step 4: Stability ──
    event_times = event_windows[:, 0]
    stab = compute_stability_weights(lag_sel, eb_sel, event_times,
                                     window_sec=stability_window_sec)

    # ── Step 5: Composite weights ──
    adj_weighted = compute_composite_weights(adj_dir, W_pruned, stab)

    # ── Step 6: Metrics ──
    metrics = compute_network_metrics(adj_weighted, sel_names)

    # ── Annotate edge_stats ──
    for es in edge_stats:
        es["ch_i"] = sel_names[es["i"]]
        es["ch_j"] = sel_names[es["j"]]
        if es.get("source", -1) >= 0:
            es["source_name"] = sel_names[es["source"]]
            es["target_name"] = sel_names[es["target"]]

    # ── Pack result ──
    node_weights = X[selected_idx]
    skeleton = (W_pruned > 0).astype(np.float32)

    params = {
        "edge_method": edge_method,
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
    }

    return NetworkResult(
        adj=adj_weighted,
        node_names=sel_names,
        node_weights=node_weights,
        W_simpson=W_simpson,
        W_dice=W_dice,
        W_pruned=W_pruned,
        pool_names=pool_names,
        selected_idx=selected_idx,
        node_xyz={"X": X, "Y": Y, "Z": Z},
        skeleton=skeleton,
        direction_mask=dir_mask,
        stability=stab,
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
    min_events_for_display: int = 1,
) -> Any:
    """Side-by-side heatmaps of Simpson and Dice broad graphs.

    Nodes with zero HFO events are excluded from display.
    """
    import matplotlib.pyplot as plt

    pool_names = result.pool_names
    X = result.node_xyz["X"]

    # Filter: keep nodes with at least min_events_for_display HFO
    keep = X > 0
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
            out[k] = {str(kk): float(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = [list(item) if isinstance(item, tuple) else item for item in v]
        elif isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        else:
            out[k] = v
    return out
