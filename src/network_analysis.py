"""
Module 4: Network Analysis — Epilepsy Network from HFO Co-activation

Phase A: Channel-Scale MVP (no MNI coordinates required)
Phase B: + Geometry (MNI coords) — stubs only
Phase C: Source Space (research frontier) — not implemented

This module reads the ``*_groupAnalysis.npz`` output of
:mod:`group_event_analysis` and constructs a directed, weighted epilepsy
network graph.

Pipeline
--------
load npz → node selection → skeleton → direction → weights → metrics

Public API
----------
``build_hfo_network(npz_path, **kwargs) → NetworkResult``

Design Notes
------------
- ``lag_raw[i, k] - lag_raw[j, k]``  negative ⇒ channel *i* leads channel *j*
- ``adj[i, j] > 0`` means a directed edge from node *i* to node *j*
- All channel-pair lag computations filter for mutual participation via ``events_bool``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
#  Data Classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class NetworkResult:
    """Complete epilepsy network analysis result (Phase A)."""

    # --- Core graph ---
    adj: np.ndarray
    """(n_sel, n_sel) directed weighted adjacency matrix.
    ``adj[i, j] > 0`` means i → j with that weight."""

    node_names: List[str]
    """(n_sel,) selected channel names."""

    node_weights: np.ndarray
    """(n_sel,) node importance (EigenCentrality × Rate blend)."""

    # --- Intermediate matrices ---
    skeleton: np.ndarray
    """(n_sel, n_sel) undirected binary skeleton (before direction injection)."""

    direction_mask: np.ndarray
    """(n_sel, n_sel) bool — True where a direction was successfully assigned."""

    stability: np.ndarray
    """(n_sel, n_sel) temporal stability weights (1 − CV)."""

    cluster_labels: np.ndarray
    """(n_pool,) spectral clustering labels for the full channel pool.
    −1 means excluded (too-small cluster or noise)."""

    # --- Metrics ---
    metrics: Dict[str, Any]
    """Graph-theory metrics (outflow, betweenness, local efficiency, …)."""

    edge_stats: List[Dict[str, Any]]
    """Per-edge statistical summary (lag, p-value, consistency, …)."""

    # --- Metadata ---
    n_pool_channels: int
    """Total channels in the pool (before spectral-clustering selection)."""

    n_selected: int
    """Channels that survived spectral-clustering selection."""

    params: Dict[str, Any]
    """All hyper-parameters used for reproducibility."""


# ═════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _clean_coact(coact: np.ndarray) -> np.ndarray:
    """Replace NaN / Inf with 0 in a co-activation matrix."""
    out = np.where(np.isfinite(coact), coact, 0.0)
    np.fill_diagonal(out, 0.0)
    return out


# ═════════════════════════════════════════════════════════════════════════════
#  A.1 — Node Selection (Spectral Clustering + Rate Weighting)
# ═════════════════════════════════════════════════════════════════════════════

def build_spatial_coact_graph(
    coact_ratio: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
    min_dist_mm: float = 5.0,
) -> np.ndarray:
    """Build a spatially-constrained symmetric co-activation affinity graph.

    Parameters
    ----------
    coact_ratio : (n, n)
        Channel × channel co-activation probability.
    dist_matrix : (n, n), optional
        Pairwise electrode distance in mm.  If *None* (Phase A default)
        the spatial constraint is skipped.
    min_dist_mm : float
        Electrode pairs closer than this are zeroed (volume conduction).

    Returns
    -------
    A : (n, n) symmetric affinity matrix with 0 diagonal.
    """
    A = _clean_coact(coact_ratio)

    if dist_matrix is not None:
        A[dist_matrix < min_dist_mm] = 0.0

    # Conservative symmetrisation: keep the *smaller* of A[i,j], A[j,i].
    A = np.minimum(A, A.T)
    return A


def _eigengap_n_clusters(A: np.ndarray, max_k: int = 20) -> int:
    """Estimate the number of clusters from the Laplacian eigengap.

    The largest gap among the first *max_k* eigenvalues (skipping the trivial
    zero-eigenvalue gap) determines the cluster count.
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    L = np.diag(deg) - A
    k = min(max_k, n - 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(L))[:k]
    gaps = np.diff(eigenvalues)
    if len(gaps) < 2:
        return 2
    # gaps[0] is the jump from 0 → λ₁ (always large); skip it.
    n_clusters = int(np.argmax(gaps[1:])) + 2
    return max(2, min(n_clusters, max(n // 3, 2)))


def extract_network_clusters(
    A_co: np.ndarray,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Adaptive spectral clustering on the co-activation affinity graph.

    Parameters
    ----------
    A_co : (n, n)
        Spatially-constrained symmetric affinity (from :func:`build_spatial_coact_graph`).
    min_cluster_size : int
        Clusters smaller than this are labelled −1 (noise).
    n_clusters : int, optional
        Force this many clusters.  *None* → automatic via eigengap.

    Returns
    -------
    labels : (n,) int   −1 = excluded, ≥ 0 = cluster id
    n_clusters_used : int
    """
    from sklearn.cluster import SpectralClustering

    n = A_co.shape[0]
    if n < max(4, min_cluster_size + 1):
        # Fewer nodes than the minimum cluster → keep all as one cluster.
        logger.warning(
            "Too few channels (%d) for spectral clustering; keeping all.", n,
        )
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

    # Kill small clusters.
    for cl in np.unique(labels):
        if cl < 0:
            continue
        if (labels == cl).sum() < min_cluster_size:
            labels[labels == cl] = -1

    n_valid = (labels >= 0).sum()
    logger.info(
        "Spectral clustering: %d clusters requested, %d/%d nodes retained.",
        n_clusters, n_valid, n,
    )
    return labels.astype(np.int32), n_clusters


def compute_node_weights(
    A_co: np.ndarray,
    rate_per_ch: np.ndarray,
    cluster_labels: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """Blend eigenvector-centrality (network role) and HFO rate (pathology).

    .. math::

        W_i = \\alpha \\cdot \\widehat{EC}_i
            + (1 - \\alpha) \\cdot \\widehat{\\log(1 + \\text{Rate}_i)}

    Parameters
    ----------
    A_co : (n, n)    affinity matrix.
    rate_per_ch : (n,)  HFO rate (events / min).
    cluster_labels : (n,)  from :func:`extract_network_clusters`.
    alpha : float   blend coefficient (0.6–0.7 recommended).

    Returns
    -------
    weights : (n,) — zero for excluded nodes.
    """
    import networkx as nx

    n = len(rate_per_ch)
    weights = np.zeros(n, dtype=np.float64)

    valid = cluster_labels >= 0
    n_valid = int(valid.sum())
    if n_valid < 2:
        weights[valid] = 1.0
        return weights

    # Eigenvector centrality on the valid sub-graph.
    sub = A_co[np.ix_(valid, valid)]
    G = nx.from_numpy_array(sub)
    try:
        ec = nx.eigenvector_centrality_numpy(G, weight="weight")
        ec_arr = np.array([ec[i] for i in range(n_valid)])
    except (nx.NetworkXError, nx.NetworkXException):
        ec_arr = np.ones(n_valid, dtype=np.float64) / n_valid

    # Log-normalised rate.
    rates = rate_per_ch[valid]
    rates_log = np.log1p(rates)
    denom_rate = rates_log.max() + 1e-10
    denom_ec = ec_arr.max() + 1e-10

    weights[valid] = alpha * (ec_arr / denom_ec) + (1 - alpha) * (rates_log / denom_rate)
    return weights


def select_network_nodes(
    coact_ratio: np.ndarray,
    rate_per_ch: np.ndarray,
    *,
    events_bool: Optional[np.ndarray] = None,
    dist_matrix: Optional[np.ndarray] = None,
    min_dist_mm: float = 5.0,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
    alpha: float = 0.65,
    run_surrogate: bool = False,
    n_surrogates: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Complete node-selection pipeline (spatial → spectral → rate).

    Returns
    -------
    selected_idx : (n_sel,) int   indices into the input channel pool.
    node_weights : (n_sel,) float  importance weights for selected nodes.
    cluster_labels : (n_pool,) int  full clustering labels (−1 = excluded).
    """
    A_co = build_spatial_coact_graph(coact_ratio, dist_matrix, min_dist_mm)

    # Optional surrogate significance gating.
    if run_surrogate and events_bool is not None:
        sig_mask = surrogate_significance_test(events_bool, n_surrogates)
        # sig_mask is (n_core, n_core) — may be smaller than A_co.
        n_sig = sig_mask.shape[0]
        A_co[:n_sig, :n_sig][~sig_mask] = 0.0

    labels, _ = extract_network_clusters(A_co, min_cluster_size, n_clusters)
    weights = compute_node_weights(A_co, rate_per_ch, labels, alpha)

    selected = np.where(labels >= 0)[0]
    return selected, weights[selected], labels


# ═════════════════════════════════════════════════════════════════════════════
#  A.2 — Surrogate Significance Testing
# ═════════════════════════════════════════════════════════════════════════════

def surrogate_significance_test(
    events_bool: np.ndarray,
    n_surrogates: int = 200,
    p_threshold: float = 0.05,
) -> np.ndarray:
    """Test whether observed co-activation exceeds chance (circular-shift surrogates).

    For each surrogate, every channel's event vector is independently
    circular-shifted by a random offset, then co-activation is recomputed.
    The fraction of surrogates ≥ the real value gives an empirical *p*-value.

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

    # Vectorised circular shift via fancy indexing.
    col_idx = np.arange(n_ev, dtype=np.int64)  # (n_ev,)

    for _ in range(n_surrogates):
        shifts = rng.integers(0, n_ev, size=n_ch)           # (n_ch,)
        idx = (col_idx[None, :] - shifts[:, None]) % n_ev   # (n_ch, n_ev)
        shifted = eb[np.arange(n_ch)[:, None], idx]          # (n_ch, n_ev)
        surr_coact = (shifted @ shifted.T) / n_ev
        surr_ge_count += (surr_coact >= real_coact).astype(np.int64)

    p_values = surr_ge_count.astype(np.float64) / n_surrogates
    sig_mask = p_values < p_threshold
    np.fill_diagonal(sig_mask, True)  # Self-coactivation is trivially significant.

    n_sig = sig_mask.sum() - n_ch  # exclude diagonal
    n_total = n_ch * (n_ch - 1)
    logger.info(
        "Surrogate test: %d / %d pairs significant (p < %.3f, %d surrogates).",
        n_sig, n_total, p_threshold, n_surrogates,
    )
    return sig_mask


# ═════════════════════════════════════════════════════════════════════════════
#  A.3 — Skeleton Construction (Weighted Undirected Graph)
# ═════════════════════════════════════════════════════════════════════════════

def build_skeleton(
    coact_ratio: np.ndarray,
    dist_matrix: Optional[np.ndarray] = None,
    min_coact: float = 0.10,
    min_dist_mm: float = 10.0,
) -> np.ndarray:
    """Threshold co-activation into a binary symmetric skeleton.

    Parameters
    ----------
    coact_ratio : (n, n) float   co-activation probability.
    dist_matrix : (n, n) float, optional   electrode distance (mm).
    min_coact : float   edges below this are removed.
    min_dist_mm : float   volume-conduction guard (Phase B).

    Returns
    -------
    skeleton : (n, n) float32, symmetric, 0/1.
    """
    coact = _clean_coact(coact_ratio)
    skeleton = (coact > min_coact).astype(np.float32)

    if dist_matrix is not None:
        skeleton[dist_matrix < min_dist_mm] = 0.0

    np.fill_diagonal(skeleton, 0.0)
    skeleton = np.maximum(skeleton, skeleton.T)

    n_edges = int(skeleton.sum()) // 2
    logger.info(
        "Skeleton: %d edges (min_coact=%.2f, %d nodes).",
        n_edges, min_coact, skeleton.shape[0],
    )
    return skeleton


# ═════════════════════════════════════════════════════════════════════════════
#  A.4 — Direction Injection (Wilcoxon + Consistency)
# ═════════════════════════════════════════════════════════════════════════════

def inject_direction(
    skeleton: np.ndarray,
    lag_raw: np.ndarray,
    events_bool: np.ndarray,
    *,
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Assign direction to skeleton edges via statistical lag analysis.

    For every undirected edge (i, j) in the skeleton:

    1. Extract co-active event lags: ``Δt_k = lag_raw[i, k] − lag_raw[j, k]``
    2. Wilcoxon signed-rank test: is the median lag ≠ 0?
    3. Direction consistency: do ≥ ``consistency_thresh`` fraction of events
       agree on sign?
    4. Zero-lag guard: |median lag| must exceed ``lag_thresh_ms``.

    Parameters
    ----------
    skeleton : (n, n) float   binary symmetric skeleton.
    lag_raw : (n, n_events) float   centroid time per channel per event.
    events_bool : (n, n_events) bool   participation mask.
    min_events : int   minimum co-active events to attempt direction.
    lag_thresh_ms : float   zero-lag cutoff in milliseconds.
    consistency_thresh : float   minimum directional consistency (0–1).
    p_value_thresh : float   Wilcoxon significance level.

    Returns
    -------
    adj : (n, n) float   directed adjacency; ``adj[i, j]`` = consistency of i → j.
    direction_mask : (n, n) bool   True where direction was assigned.
    edge_stats : list of dict   per-edge statistical summary.
    """
    from scipy.stats import wilcoxon as wilcoxon_test

    n = skeleton.shape[0]
    adj = np.zeros((n, n), dtype=np.float64)
    direction_mask = np.zeros((n, n), dtype=bool)
    edge_stats: List[Dict[str, Any]] = []

    lag_thresh_sec = lag_thresh_ms * 1e-3

    for i in range(n):
        for j in range(i + 1, n):
            if skeleton[i, j] == 0:
                continue

            # Mutual participation.
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

            # --- Wilcoxon signed-rank test ---
            try:
                _, p_val = wilcoxon_test(lags)
            except ValueError:
                p_val = 1.0
            p_val = float(p_val)

            median_lag = float(np.median(lags))

            if p_val > p_value_thresh:
                edge_stats.append(
                    _edge_stat(i, j, n_coact, median_lag * 1e3, p_val, reason="p_value"),
                )
                continue

            # --- Zero-lag guard ---
            if abs(median_lag) < lag_thresh_sec:
                edge_stats.append(
                    _edge_stat(i, j, n_coact, median_lag * 1e3, p_val, reason="zero_lag"),
                )
                continue

            # --- Consistency check ---
            direction = np.sign(median_lag)
            consistency = float(np.mean(np.sign(lags) == direction))

            if consistency < consistency_thresh:
                edge_stats.append(
                    _edge_stat(
                        i, j, n_coact, median_lag * 1e3, p_val,
                        consistency=consistency, reason="low_consistency",
                    ),
                )
                continue

            # === Assign direction ===
            if median_lag < 0:  # i leads j  →  edge i → j
                adj[i, j] = consistency
                direction_mask[i, j] = True
                src, tgt = i, j
            else:               # j leads i  →  edge j → i
                adj[j, i] = consistency
                direction_mask[j, i] = True
                src, tgt = j, i

            edge_stats.append(
                _edge_stat(
                    i, j, n_coact, median_lag * 1e3, p_val,
                    consistency=consistency, directed=True, source=src, target=tgt,
                    reason="directed",
                ),
            )

    n_directed = int(direction_mask.sum())
    n_skeleton = int((skeleton > 0).sum()) // 2
    logger.info(
        "Direction injection: %d / %d skeleton edges directed.", n_directed, n_skeleton,
    )
    return adj, direction_mask, edge_stats


def _edge_stat(
    i: int,
    j: int,
    n_coact: int,
    median_lag_ms: float = 0.0,
    p_value: float = 1.0,
    *,
    consistency: float = 0.0,
    directed: bool = False,
    source: int = -1,
    target: int = -1,
    reason: str = "",
) -> Dict[str, Any]:
    """Build a per-edge stat dict."""
    return {
        "i": i,
        "j": j,
        "n_coactive": n_coact,
        "median_lag_ms": median_lag_ms,
        "p_value": p_value,
        "consistency": consistency,
        "directed": directed,
        "source": source,
        "target": target,
        "reason": reason,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  A.5 — Stability Weights (Temporal Robustness)
# ═════════════════════════════════════════════════════════════════════════════

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
    """Per-edge temporal stability: ``1 − CV(consistency across time windows)``.

    Stereotypical (pathological) connections show *consistent* directionality
    across multiple 5-minute windows.  Transient / noisy connections show
    high variance → low stability → down-weighted.

    Parameters
    ----------
    lag_raw : (n_ch, n_events)   centroid lag per channel.
    events_bool : (n_ch, n_events)   participation mask.
    event_times : (n_events,)   absolute timestamp of each event (seconds).
    window_sec : float   time-window length (default 5 min).
    min_windows : int   minimum valid windows required.
    min_events_per_window : int   skip near-empty windows.
    min_coactive_per_pair : int   minimum co-active events per channel pair.

    Returns
    -------
    stability : (n_ch, n_ch) float in [0, 1].  1 = perfectly stable.
    """
    n_ch = lag_raw.shape[0]

    t_min = float(event_times.min())
    t_max = float(event_times.max())
    bin_starts = np.arange(t_min, t_max, window_sec)

    if len(bin_starts) < min_windows:
        logger.info(
            "Stability: too few time windows (%d < %d).  Returning uniform.",
            len(bin_starts), min_windows,
        )
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
                if med == 0:
                    c = 0.5
                else:
                    c = float(np.mean(np.sign(lags) == np.sign(med)))
                cons[i, j] = cons[j, i] = c

        window_cons_list.append(cons)

    if len(window_cons_list) < min_windows:
        logger.info(
            "Stability: too few valid windows (%d < %d).  Returning uniform.",
            len(window_cons_list), min_windows,
        )
        return np.ones((n_ch, n_ch), dtype=np.float64)

    import warnings

    stacked = np.stack(window_cons_list)  # (n_win, n_ch, n_ch)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_c = np.nanmean(stacked, axis=0)
        std_c = np.nanstd(stacked, axis=0)
        cv = np.where(mean_c > 0, std_c / mean_c, 1.0)
        stability = 1.0 - np.clip(cv, 0.0, 1.0)

    # Replace any residual NaN with 0 (no data → no credit).
    stability = np.where(np.isfinite(stability), stability, 0.0)

    logger.info(
        "Stability: computed over %d windows (%.0fs each). "
        "Mean stability = %.3f.",
        len(window_cons_list), window_sec, float(np.nanmean(stability)),
    )
    return stability


# ═════════════════════════════════════════════════════════════════════════════
#  A.6 — Composite Weights
# ═════════════════════════════════════════════════════════════════════════════

def compute_composite_weights(
    adj_directed: np.ndarray,
    coact_ratio: np.ndarray,
    stability: np.ndarray,
) -> np.ndarray:
    """Phase A composite weight: Coactivation × Consistency × Stability.

    .. math::

        W_{ij} = \\text{Coact}_{ij} \\times \\text{Consistency}_{ij}
                 \\times \\text{Stability}_{ij}

    ``adj_directed[i, j]`` already carries the consistency value from
    :func:`inject_direction`.  We multiply by co-activation strength and
    temporal stability.

    Phase B will add the Pathology factor: ``(1 + α · FR_ratio)``.
    """
    coact = np.where(np.isfinite(coact_ratio), coact_ratio, 0.0)
    weighted = adj_directed * coact * stability
    return weighted


# ═════════════════════════════════════════════════════════════════════════════
#  A.7 — Graph Theory Metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_network_metrics(
    adj: np.ndarray,
    ch_names: List[str],
) -> Dict[str, Any]:
    """Core graph-theory metrics for the directed weighted epilepsy network.

    Computed metrics
    ----------------
    - **net_outflow** : per-node Net Outflow Index  ``(Out − In) / (Out + In)``
    - **out_degree** / **in_degree** : weighted degree
    - **betweenness** : betweenness centrality
    - **local_efficiency** : per-node local efficiency (on undirected projection)
    - **global_efficiency** : scalar
    - **source_ranking** : nodes sorted by outflow (descending)
    - **n_nodes**, **n_edges**, **density**
    """
    import networkx as nx

    n = adj.shape[0]
    G = nx.DiGraph()
    for i, name in enumerate(ch_names):
        G.add_node(name)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(ch_names[i], ch_names[j], weight=float(adj[i, j]))

    metrics: Dict[str, Any] = {}

    # --- Net Outflow Index ---
    outflow: Dict[str, float] = {}
    for node in G.nodes():
        out_w = float(G.out_degree(node, weight="weight"))
        in_w = float(G.in_degree(node, weight="weight"))
        total = out_w + in_w
        outflow[node] = (out_w - in_w) / total if total > 0 else 0.0
    metrics["net_outflow"] = outflow

    # --- Degree ---
    metrics["out_degree"] = {n: float(d) for n, d in G.out_degree(weight="weight")}
    metrics["in_degree"] = {n: float(d) for n, d in G.in_degree(weight="weight")}

    # --- Betweenness centrality ---
    metrics["betweenness"] = {
        k: float(v) for k, v in nx.betweenness_centrality(G, weight="weight").items()
    }

    # --- Local / Global efficiency (undirected projection) ---
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

    # --- Source ranking ---
    metrics["source_ranking"] = sorted(
        outflow.items(), key=lambda kv: kv[1], reverse=True,
    )

    # --- Summary ---
    metrics["n_nodes"] = G.number_of_nodes()
    metrics["n_edges"] = G.number_of_edges()
    metrics["density"] = float(nx.density(G))

    return metrics


# ═════════════════════════════════════════════════════════════════════════════
#  A.8 — One-Stop API
# ═════════════════════════════════════════════════════════════════════════════

def build_hfo_network(
    group_analysis_npz: str,
    dist_matrix: Optional[np.ndarray] = None,
    *,
    # — Node selection (spectral clustering) —
    use_all_channels: bool = True,
    min_dist_mm: float = 5.0,
    min_cluster_size: int = 3,
    n_clusters: Optional[int] = None,
    alpha: float = 0.65,
    # — Surrogate —
    run_surrogate: bool = True,
    n_surrogates: int = 200,
    # — Skeleton —
    min_coact: float = 0.10,
    # — Direction —
    min_events: int = 5,
    lag_thresh_ms: float = 5.0,
    consistency_thresh: float = 0.6,
    p_value_thresh: float = 0.05,
    # — Stability —
    stability_window_sec: float = 300.0,
) -> NetworkResult:
    """One-stop epilepsy network builder (Phase A: Channel-Scale).

    Pipeline
    --------
    1. Load ``*_groupAnalysis.npz``
    2. Select channel pool (all channels or core only)
    3. Spectral-clustering node selection  →  ``selected_idx``
    4. Build skeleton (co-activation threshold)
    5. Inject direction (Wilcoxon + consistency on ``lag_raw``)
    6. Compute temporal stability
    7. Composite weight = Coact × Consistency × Stability
    8. Graph-theory metrics

    Parameters
    ----------
    group_analysis_npz : str
        Path to the ``*_groupAnalysis.npz`` file produced by
        :func:`group_event_analysis.compute_and_save_group_analysis`.
    dist_matrix : (n_all, n_all), optional
        Electrode distance matrix in mm.  *None* in Phase A.
    use_all_channels : bool
        If True and ``coact_all_event_ratio`` exists in the npz, use the
        full channel pool for node selection.  Otherwise fall back to
        core channels.
    min_dist_mm, min_cluster_size, n_clusters, alpha
        See :func:`select_network_nodes`.
    run_surrogate, n_surrogates
        See :func:`surrogate_significance_test`.
    min_coact
        See :func:`build_skeleton`.
    min_events, lag_thresh_ms, consistency_thresh, p_value_thresh
        See :func:`inject_direction`.
    stability_window_sec
        See :func:`compute_stability_weights`.

    Returns
    -------
    NetworkResult
    """
    from .group_event_analysis import load_group_analysis_results

    data = load_group_analysis_results(group_analysis_npz)

    # ------------------------------------------------------------------
    #  Resolve channel pool
    # ------------------------------------------------------------------
    core_names: List[str] = list(data["ch_names"])
    lag_raw: np.ndarray = data["lag_raw"]            # (n_core, n_events)
    events_bool: np.ndarray = data["events_bool"]    # (n_core, n_events)
    event_windows: np.ndarray = data["event_windows"]  # (n_events, 2)

    has_all = (
        use_all_channels
        and "coact_all_event_ratio" in data
        and "coact_all_ch_names" in data
    )

    if has_all:
        coact_pool = np.asarray(data["coact_all_event_ratio"])
        pool_names = [str(x) for x in data["coact_all_ch_names"]]
        # Rate proxy: diagonal of coact_all_event_count / duration_min
        duration_min = _recording_duration_min(event_windows)
        if "coact_all_event_count" in data:
            rate_pool = np.diag(data["coact_all_event_count"]).astype(np.float64)
            rate_pool /= max(duration_min, 1e-6)
        else:
            # Participation fraction as proxy (proportional to rate).
            diag = np.diag(_clean_coact(coact_pool) + np.eye(coact_pool.shape[0]))
            rate_pool = diag  # already 0–1, used only for ranking
        logger.info(
            "Using ALL-channel pool: %d channels (core: %d).",
            len(pool_names), len(core_names),
        )
    else:
        coact_pool = np.asarray(data["coact_event_ratio"])
        pool_names = list(core_names)
        duration_min = _recording_duration_min(event_windows)
        rate_pool = events_bool.sum(axis=1).astype(np.float64) / max(duration_min, 1e-6)
        logger.info("Using CORE-channel pool: %d channels.", len(pool_names))

    n_pool = len(pool_names)

    # ------------------------------------------------------------------
    #  A.1 — Node selection
    # ------------------------------------------------------------------
    # Surrogate test works on events_bool (core channels).  In all-channel
    # mode, we apply it to the core submatrix only.
    surrogate_eb = events_bool if run_surrogate else None

    selected_idx, node_weights, cluster_labels = select_network_nodes(
        coact_pool,
        rate_pool,
        events_bool=surrogate_eb,
        dist_matrix=dist_matrix,
        min_dist_mm=min_dist_mm,
        min_cluster_size=min_cluster_size,
        n_clusters=n_clusters,
        alpha=alpha,
        run_surrogate=run_surrogate,
        n_surrogates=n_surrogates,
    )

    n_sel = len(selected_idx)
    sel_names = [pool_names[i] for i in selected_idx]
    logger.info("Selected %d / %d nodes: %s", n_sel, n_pool, sel_names)

    if n_sel < 2:
        logger.warning("Fewer than 2 nodes selected — returning empty network.")
        return _empty_result(pool_names, cluster_labels, n_pool)

    # ------------------------------------------------------------------
    #  Prepare sub-matrices for selected nodes
    # ------------------------------------------------------------------
    coact_sel = coact_pool[np.ix_(selected_idx, selected_idx)]

    # Map selected node → core-channel index for lag / events data.
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

    # ------------------------------------------------------------------
    #  A.3 — Skeleton
    # ------------------------------------------------------------------
    skeleton = build_skeleton(coact_sel, min_coact=min_coact)

    # ------------------------------------------------------------------
    #  A.4 — Direction injection
    # ------------------------------------------------------------------
    adj_dir, dir_mask, edge_stats = inject_direction(
        skeleton,
        lag_sel,
        eb_sel,
        min_events=min_events,
        lag_thresh_ms=lag_thresh_ms,
        consistency_thresh=consistency_thresh,
        p_value_thresh=p_value_thresh,
    )

    # ------------------------------------------------------------------
    #  A.5 — Stability weights
    # ------------------------------------------------------------------
    event_times = event_windows[:, 0]  # use window start as timestamp
    stab = compute_stability_weights(
        lag_sel,
        eb_sel,
        event_times,
        window_sec=stability_window_sec,
    )

    # ------------------------------------------------------------------
    #  A.6 — Composite weights
    # ------------------------------------------------------------------
    adj_weighted = compute_composite_weights(adj_dir, coact_sel, stab)

    # ------------------------------------------------------------------
    #  A.7 — Graph metrics
    # ------------------------------------------------------------------
    metrics = compute_network_metrics(adj_weighted, sel_names)

    # ------------------------------------------------------------------
    #  Annotate edge_stats with channel names
    # ------------------------------------------------------------------
    for es in edge_stats:
        es["ch_i"] = sel_names[es["i"]]
        es["ch_j"] = sel_names[es["j"]]
        if es.get("source", -1) >= 0:
            es["source_name"] = sel_names[es["source"]]
            es["target_name"] = sel_names[es["target"]]

    # ------------------------------------------------------------------
    #  Pack result
    # ------------------------------------------------------------------
    params = {
        "use_all_channels": has_all,
        "min_dist_mm": min_dist_mm,
        "min_cluster_size": min_cluster_size,
        "n_clusters": n_clusters,
        "alpha": alpha,
        "run_surrogate": run_surrogate,
        "n_surrogates": n_surrogates,
        "min_coact": min_coact,
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


# ═════════════════════════════════════════════════════════════════════════════
#  A.9 — 2D Network Topology Visualisation
# ═════════════════════════════════════════════════════════════════════════════

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
    """Draw a 2D directed network topology (Phase A deliverable).

    - Node colour = Net Outflow (red = Source, blue = Sink)
    - Node size   = node weight (EigenCentrality × Rate)
    - Edge width  = composite weight
    - Edge colour = direction (darker = stronger)

    Parameters
    ----------
    result : NetworkResult
    output_path : str, optional   save to file (PNG/PDF/SVG).
    layout : str   'spring' | 'circular' | 'spectral' | 'kamada_kawai'
    figsize, cmap_node, cmap_edge, title : display options.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    adj = result.adj
    names = result.node_names
    n = len(names)

    G = nx.DiGraph()
    for i, name in enumerate(names):
        G.add_node(name)
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G.add_edge(names[i], names[j], weight=float(adj[i, j]))

    # --- Layout ---
    layout_funcs = {
        "spring": lambda g: nx.spring_layout(g, seed=42, weight="weight", k=2.0),
        "circular": nx.circular_layout,
        "spectral": nx.spectral_layout,
        "kamada_kawai": lambda g: nx.kamada_kawai_layout(g, weight="weight"),
    }
    pos = layout_funcs.get(layout, layout_funcs["spring"])(G)

    # --- Node attributes ---
    outflow = result.metrics.get("net_outflow", {})
    node_colors = [outflow.get(name, 0.0) for name in names]
    node_sizes = result.node_weights.copy()
    node_sizes = 300 + 2000 * (node_sizes / (node_sizes.max() + 1e-10))

    # --- Edge attributes ---
    edges = list(G.edges(data=True))
    edge_weights = [d.get("weight", 0.1) for _, _, d in edges]
    max_ew = max(edge_weights) if edge_weights else 1.0
    edge_widths = [0.5 + 4.0 * (w / max_ew) for w in edge_weights]

    # --- Draw ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Edges.
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edgelist=[(u, v) for u, v, _ in edges],
        width=edge_widths,
        alpha=0.6,
        edge_color=edge_weights,
        edge_cmap=plt.get_cmap(cmap_edge),
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    # Nodes.
    vmin = min(node_colors) if node_colors else -1
    vmax = max(node_colors) if node_colors else 1
    abs_max = max(abs(vmin), abs(vmax), 0.01)
    nc = nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        cmap=plt.get_cmap(cmap_node),
        vmin=-abs_max,
        vmax=abs_max,
        edgecolors="black",
        linewidths=1.0,
    )

    # Labels.
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    # Colour bar.
    cb = fig.colorbar(nc, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("Net Outflow  (← Sink | Source →)", fontsize=10)

    ax.set_title(
        title or "HFO Epilepsy Network (Phase A)",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Network topology saved to %s", output_path)

    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  Private utilities
# ═════════════════════════════════════════════════════════════════════════════

def _recording_duration_min(event_windows: np.ndarray) -> float:
    """Estimate recording duration in minutes from event windows."""
    if event_windows.size == 0:
        return 1.0
    return float(event_windows[-1, 1] - event_windows[0, 0]) / 60.0


def _empty_result(
    pool_names: List[str],
    cluster_labels: np.ndarray,
    n_pool: int,
) -> NetworkResult:
    """Return a trivial NetworkResult when the network is degenerate."""
    return NetworkResult(
        adj=np.zeros((0, 0)),
        node_names=[],
        node_weights=np.array([]),
        skeleton=np.zeros((0, 0)),
        direction_mask=np.zeros((0, 0), dtype=bool),
        stability=np.zeros((0, 0)),
        cluster_labels=cluster_labels,
        metrics={
            "net_outflow": {},
            "out_degree": {},
            "in_degree": {},
            "betweenness": {},
            "local_efficiency": {},
            "global_efficiency": 0.0,
            "source_ranking": [],
            "n_nodes": 0,
            "n_edges": 0,
            "density": 0.0,
        },
        edge_stats=[],
        n_pool_channels=n_pool,
        n_selected=0,
        params={},
    )
