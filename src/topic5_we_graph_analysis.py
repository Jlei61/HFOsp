"""Topology of a learned WE-SLP-RNN graph, and the nulls it has to beat.

A graph grown with ``P ∝ 1/d`` on a plane is a random geometric graph, and random
geometric graphs are clustered, modular, short-edged and spatially contiguous
before anything is trained.  So none of the topology numbers mean anything on
their own.  Three references are needed and all three live here:

* the graph the same growth rule produced before training (the growth prior),
* the graph the same prune/regrow dynamics produced with the task signal
  destroyed (the dynamics),
* a rewiring of the learned graph that preserves every unit's in- and
  out-degree and the distribution of edge lengths (the geometry).

Beating the uniform-regrowth arm alone only shows that distance-biased growth
changes topology, which is a restatement of the growth rule, not a finding.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import networkx as nx
import numpy as np

DEFAULT_LENGTH_BINS = 8


def to_graph(mask: np.ndarray, weight: np.ndarray | None = None) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(range(mask.shape[0]))
    rows, cols = np.nonzero(mask)
    if weight is None:
        g.add_edges_from(zip(rows.tolist(), cols.tolist()))
    else:
        g.add_weighted_edges_from(
            (int(i), int(j), float(weight[i, j])) for i, j in zip(rows, cols))
    return g


def modularity_q(mask: np.ndarray, seed: int = 0) -> tuple[float, list[set[int]]]:
    """Louvain modularity of the undirected projection.

    Direction is dropped here on purpose: a module is a group of units that talk
    to each other, and asking whether i→j or j→i carries the traffic is a
    different question from asking who is in the group.
    """
    undirected = nx.Graph(to_graph(np.maximum(mask, mask.T)))
    if undirected.number_of_edges() == 0:
        return float("nan"), []
    communities = nx.community.louvain_communities(undirected, seed=seed)
    return float(nx.community.modularity(undirected, communities)), communities


def clustering_coefficient(mask: np.ndarray) -> float:
    undirected = nx.Graph(to_graph(np.maximum(mask, mask.T)))
    return float(nx.average_clustering(undirected))


def small_worldness(mask: np.ndarray, seed: int = 0, n_null: int = 10) -> float:
    """``(C/C_rand) / (L/L_rand)`` against degree-preserving random rewirings."""
    undirected = nx.Graph(to_graph(np.maximum(mask, mask.T)))
    if undirected.number_of_edges() < 3:
        return float("nan")
    c_obs = nx.average_clustering(undirected)
    l_obs = _mean_path_length(undirected)
    rng = np.random.default_rng(seed)
    c_null, l_null = [], []
    for _ in range(n_null):
        shuffled = undirected.copy()
        try:
            nx.double_edge_swap(shuffled, nswap=5 * shuffled.number_of_edges(),
                                max_tries=100 * shuffled.number_of_edges(),
                                seed=int(rng.integers(1 << 30)))
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            continue
        c_null.append(nx.average_clustering(shuffled))
        l_null.append(_mean_path_length(shuffled))
    if not c_null or np.mean(c_null) == 0 or l_obs == 0:
        return float("nan")
    return float((c_obs / np.mean(c_null)) / (l_obs / max(np.mean(l_null), 1e-9)))


def _mean_path_length(graph: nx.Graph) -> float:
    lengths = []
    for component in nx.connected_components(graph):
        sub = graph.subgraph(component)
        if sub.number_of_nodes() > 1:
            lengths.append(nx.average_shortest_path_length(sub))
    return float(np.mean(lengths)) if lengths else 0.0


def participation_coefficients(mask: np.ndarray, communities: Sequence[set[int]]) -> np.ndarray:
    """Per unit, how evenly its edges are spread over the modules."""
    n = mask.shape[0]
    membership = np.full(n, -1, int)
    for index, community in enumerate(communities):
        for node in community:
            membership[node] = index
    undirected = np.maximum(mask, mask.T) > 0
    degree = undirected.sum(1)
    out = np.zeros(n)
    for i in range(n):
        if degree[i] == 0:
            continue
        share = 0.0
        for index in range(len(communities)):
            k = float(undirected[i, membership == index].sum())
            share += (k / degree[i]) ** 2
        out[i] = 1.0 - share
    return out


def length_preserving_rewire(mask: np.ndarray, distance: np.ndarray, seed: int = 0,
                             n_bins: int = DEFAULT_LENGTH_BINS,
                             n_swaps_per_edge: int = 20) -> np.ndarray:
    """Shuffle who connects to whom while holding degree and edge length fixed.

    Pairs of edges are swapped only when both replacements fall in the same
    length bins the originals came from, so the geometric budget the task spent
    is held constant and only the wiring pattern is randomised.  This is the null
    that separates "the task organised the graph" from "the plane did".
    """
    mask = np.asarray(mask) > 0
    n = mask.shape[0]
    finite = distance[np.isfinite(distance)]
    edges = np.argwhere(mask)
    if len(edges) < 4:
        return mask.astype(np.uint8)
    quantiles = np.quantile(distance[mask], np.linspace(0, 1, n_bins + 1)[1:-1])
    bin_of = lambda a, b: int(np.searchsorted(quantiles, distance[a, b]))  # noqa: E731

    rng = np.random.default_rng(seed)
    out = mask.copy()
    edge_list = [tuple(e) for e in edges]
    for _ in range(n_swaps_per_edge * len(edge_list)):
        i = rng.integers(len(edge_list))
        j = rng.integers(len(edge_list))
        if i == j:
            continue
        (a, b), (c, d) = edge_list[i], edge_list[j]
        if len({a, b, c, d}) < 4 or out[a, d] or out[c, b]:
            continue
        if bin_of(a, d) != bin_of(a, b) or bin_of(c, b) != bin_of(c, d):
            continue
        out[a, b] = out[c, d] = False
        out[a, d] = out[c, b] = True
        edge_list[i], edge_list[j] = (a, d), (c, b)
    return out.astype(np.uint8)


def contiguous_random_lesion(nodes_xy: np.ndarray, size: int, seed: int = 0) -> np.ndarray:
    """A spatially connected patch of ``size`` units, grown from a random seed.

    A Louvain module on a plane is a patch, so scattering the control lesion over
    the whole plane compares a patch against confetti and the patch wins for
    free.  The control has to be a patch too.
    """
    rng = np.random.default_rng(seed)
    n = len(nodes_xy)
    size = int(min(size, n))
    start = int(rng.integers(n))
    chosen = [start]
    remaining = set(range(n)) - {start}
    while len(chosen) < size and remaining:
        centre = nodes_xy[chosen].mean(0)
        candidates = np.array(sorted(remaining))
        d = np.linalg.norm(nodes_xy[candidates] - centre, axis=1)
        pick = int(candidates[int(np.argmin(d))])
        chosen.append(pick)
        remaining.discard(pick)
    return np.array(sorted(chosen), int)


def summarise(mask: np.ndarray, distance: np.ndarray, d0_mm: float = 10.0,
              seed: int = 0, with_small_world: bool = True) -> dict:
    mask = np.asarray(mask) > 0
    q, communities = modularity_q(mask, seed=seed)
    lengths = distance[mask]
    return {
        "n_edges": int(mask.sum()),
        "modularity_q": q,
        "n_modules": len(communities),
        "clustering": clustering_coefficient(mask),
        "small_worldness": small_worldness(mask, seed=seed) if with_small_world else float("nan"),
        "mean_edge_len_mm": float(lengths.mean()) if lengths.size else float("nan"),
        "median_edge_len_mm": float(np.median(lengths)) if lengths.size else float("nan"),
        "long_edge_fraction": float((lengths > d0_mm).mean()) if lengths.size else float("nan"),
        "participation_mean": float(participation_coefficients(mask, communities).mean())
        if communities else float("nan"),
        "connector_fraction": float(
            (participation_coefficients(mask, communities) > 0.6).mean())
        if communities else float("nan"),
    }


def module_of_each_node(mask: np.ndarray, seed: int = 0) -> tuple[np.ndarray, list[set[int]]]:
    _, communities = modularity_q(mask, seed=seed)
    membership = np.full(mask.shape[0], -1, int)
    for index, community in enumerate(communities):
        for node in community:
            membership[node] = index
    return membership, communities


def distance_controlled_similarity(similarity: np.ndarray, distance: np.ndarray,
                                   mask: np.ndarray, n_bins: int = DEFAULT_LENGTH_BINS
                                   ) -> dict:
    """Connected minus unconnected functional similarity, within distance bins.

    Nearby units read out through overlapping parts of the observation operator,
    so they look functionally alike whether or not the graph joins them.  Binning
    on distance first is what stops that from being reported as homophily.
    """
    mask = np.asarray(mask) > 0
    off = ~np.eye(mask.shape[0], dtype=bool)
    d = distance[off]
    s = similarity[off]
    connected = mask[off]
    if connected.sum() == 0 or (~connected).sum() == 0:
        return {"delta": float("nan"), "n_bins_used": 0}
    edges = np.quantile(d, np.linspace(0, 1, n_bins + 1))
    deltas, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (d >= lo) & (d <= hi)
        a, b = s[inside & connected], s[inside & ~connected]
        if a.size < 5 or b.size < 5:
            continue
        deltas.append(float(a.mean() - b.mean()))
        weights.append(float(inside.sum()))
    if not deltas:
        return {"delta": float("nan"), "n_bins_used": 0}
    return {
        "delta": float(np.average(deltas, weights=weights)),
        "per_bin": deltas,
        "n_bins_used": len(deltas),
    }
