"""Target-free graph controls for Topic 5.1 multiscale scaffold v0.5.

The primary topology null is a *refit* graph, not a post-training lesion. It
preserves the macro statistics of a frozen L3 added-edge mask while randomising
the source--target pairing before any L2m weights are trained. Matrices use the
project convention ``mask[target, source]``.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


def stable_seed(label: str, salt: int) -> int:
    digest = hashlib.sha256(f"{label}|{salt}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def reciprocity_count(mask: np.ndarray) -> int:
    directed = np.asarray(mask, dtype=bool)
    if directed.ndim != 2 or directed.shape[0] != directed.shape[1]:
        raise ValueError("mask must be square")
    return int(np.count_nonzero(directed & directed.T) // 2)


def distance_decile_labels(
    distance_mm: np.ndarray,
    candidate_pool: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Freeze ten distance strata from the complete nonlocal candidate pool."""
    distance = np.asarray(distance_mm, dtype=float)
    pool = np.asarray(candidate_pool, dtype=bool)
    if distance.shape != pool.shape or distance.ndim != 2 or distance.shape[0] != distance.shape[1]:
        raise ValueError("distance and candidate pool must be aligned square matrices")
    values = distance[pool]
    if values.size < 10 or not np.isfinite(values).all():
        raise ValueError("nonlocal pool needs at least ten finite distances")
    # Repeated cut points are retained. They are a property of the frozen
    # geometry and must not be merged after seeing graph-null feasibility.
    cutpoints = np.quantile(values, np.arange(1, 10, dtype=float) / 10.0)
    labels = np.full(distance.shape, -1, dtype=np.int8)
    labels[pool] = np.searchsorted(cutpoints, values, side="right").astype(np.int8)
    return labels, cutpoints.astype(float)


def macro_statistics(mask: np.ndarray, bin_labels: np.ndarray) -> dict[str, np.ndarray | int]:
    directed = np.asarray(mask, dtype=bool)
    bins = np.asarray(bin_labels)
    if directed.shape != bins.shape:
        raise ValueError("mask and bin_labels must align")
    active_bins = bins[directed]
    if np.any(active_bins < 0):
        raise ValueError("active edge lies outside the candidate pool")
    return {
        "edge_count": int(directed.sum()),
        "in_degree": directed.sum(axis=1).astype(np.int32),
        "out_degree": directed.sum(axis=0).astype(np.int32),
        "reciprocity_count": reciprocity_count(directed),
        "distance_bin_counts": np.bincount(active_bins, minlength=10).astype(np.int32),
    }


def exact_macro_match_audit(
    reference: np.ndarray,
    candidate: np.ndarray,
    candidate_pool: np.ndarray,
    bin_labels: np.ndarray,
) -> dict[str, object]:
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    pool = np.asarray(candidate_pool, dtype=bool)
    if reference.shape != candidate.shape or reference.shape != pool.shape:
        raise ValueError("all graph masks must align")
    ref = macro_statistics(reference, bin_labels)
    got = macro_statistics(candidate, bin_labels)
    checks: dict[str, object] = {
        "edge_count_exact": ref["edge_count"] == got["edge_count"],
        "in_degree_exact": bool(np.array_equal(ref["in_degree"], got["in_degree"])),
        "out_degree_exact": bool(np.array_equal(ref["out_degree"], got["out_degree"])),
        "reciprocity_exact": ref["reciprocity_count"] == got["reciprocity_count"],
        "distance_bin_counts_exact": bool(np.array_equal(
            ref["distance_bin_counts"], got["distance_bin_counts"]
        )),
        "within_nonlocal_pool": bool(np.all(~candidate | pool)),
        "no_self_edges": bool(not np.any(np.diag(candidate))),
        "different_from_reference": bool(not np.array_equal(reference, candidate)),
    }
    overlap = int(np.count_nonzero(reference & candidate))
    edge_count = int(reference.sum())
    checks.update({
        "reference_edge_count": edge_count,
        "candidate_edge_count": int(candidate.sum()),
        "edge_overlap_count": overlap,
        "pairing_disruption_fraction": 1.0 - overlap / max(1, edge_count),
        "reference_reciprocity_count": int(ref["reciprocity_count"]),
        "candidate_reciprocity_count": int(got["reciprocity_count"]),
        "reference_distance_bin_counts": np.asarray(ref["distance_bin_counts"]).tolist(),
        "candidate_distance_bin_counts": np.asarray(got["distance_bin_counts"]).tolist(),
    })
    checks["all_exact"] = bool(all(checks[key] for key in (
        "edge_count_exact", "in_degree_exact", "out_degree_exact",
        "reciprocity_exact", "distance_bin_counts_exact", "within_nonlocal_pool",
        "no_self_edges", "different_from_reference",
    )))
    return checks


@dataclass(frozen=True)
class MacroMatchedGraph:
    mask: np.ndarray
    bin_labels: np.ndarray
    cutpoints_mm: np.ndarray
    audit: dict[str, object]


def _pair_reciprocity(mask: np.ndarray, pair: tuple[int, int]) -> int:
    a, b = pair
    return int(bool(mask[a, b]) and bool(mask[b, a]))


def construct_macro_matched_nonlocal(
    reference_mask: np.ndarray,
    candidate_pool: np.ndarray,
    distance_mm: np.ndarray,
    seed: int,
    *,
    max_restarts: int = 100,
    attempts_per_restart: int = 10_000,
    minimum_disruption_fraction: float = 0.50,
) -> MacroMatchedGraph:
    """Randomise pairings by exact-statistic directed double-edge swaps."""
    reference = np.asarray(reference_mask, dtype=bool)
    pool = np.asarray(candidate_pool, dtype=bool)
    distance = np.asarray(distance_mm, dtype=float)
    if reference.shape != pool.shape or reference.shape != distance.shape:
        raise ValueError("reference, pool, and distance must align")
    if np.any(reference & ~pool):
        raise ValueError("reference contains edges outside the nonlocal pool")
    if int(reference.sum()) < 2:
        raise ValueError("at least two reference edges are required")
    if not 0 < float(minimum_disruption_fraction) <= 1:
        raise ValueError("minimum_disruption_fraction must lie in (0, 1]")
    labels, cutpoints = distance_decile_labels(distance, pool)
    base_reciprocity = reciprocity_count(reference)
    best_mask: np.ndarray | None = None
    best_audit: dict[str, object] | None = None
    total_accepted = 0
    for restart in range(int(max_restarts)):
        rng = np.random.default_rng(int(seed) + 104729 * restart)
        mask = reference.copy()
        edges = np.argwhere(mask).astype(np.int32)
        accepted = 0
        for _ in range(int(attempts_per_restart)):
            first, second = rng.choice(len(edges), size=2, replace=False)
            t1, s1 = map(int, edges[first])
            t2, s2 = map(int, edges[second])
            if t1 == t2 or s1 == s2:
                continue
            new1, new2 = (t1, s2), (t2, s1)
            if t1 == s2 or t2 == s1:
                continue
            if not pool[new1] or not pool[new2] or mask[new1] or mask[new2]:
                continue
            if sorted((int(labels[t1, s1]), int(labels[t2, s2]))) != sorted((
                int(labels[new1]), int(labels[new2])
            )):
                continue
            affected = {
                tuple(sorted((t1, s1))), tuple(sorted((t2, s2))),
                tuple(sorted(new1)), tuple(sorted(new2)),
            }
            before = sum(_pair_reciprocity(mask, pair) for pair in affected)
            mask[t1, s1] = False
            mask[t2, s2] = False
            mask[new1] = True
            mask[new2] = True
            after = sum(_pair_reciprocity(mask, pair) for pair in affected)
            if after != before:
                mask[new1] = False
                mask[new2] = False
                mask[t1, s1] = True
                mask[t2, s2] = True
                continue
            edges[first] = new1
            edges[second] = new2
            accepted += 1
        total_accepted += accepted
        audit = exact_macro_match_audit(reference, mask, pool, labels)
        audit.update({
            "restart": restart,
            "accepted_swaps_this_restart": accepted,
            "total_accepted_swaps": total_accepted,
            "max_restarts": int(max_restarts),
            "attempts_per_restart": int(attempts_per_restart),
            "minimum_disruption_fraction": float(minimum_disruption_fraction),
            "graph_null_seed": int(seed),
            "base_reciprocity_rechecked": bool(reciprocity_count(mask) == base_reciprocity),
        })
        if bool(audit["all_exact"]):
            if best_audit is None or float(audit["pairing_disruption_fraction"]) > float(
                best_audit["pairing_disruption_fraction"]
            ):
                best_mask, best_audit = mask.copy(), audit
            if float(audit["pairing_disruption_fraction"]) >= float(minimum_disruption_fraction):
                audit["disruption_target_met"] = True
                return MacroMatchedGraph(mask.astype(np.uint8), labels, cutpoints, audit)
    if best_mask is None or best_audit is None:
        raise RuntimeError("GRAPH_NULL_NOT_CONSTRUCTIBLE: no non-identical exact macro match")
    best_audit["disruption_target_met"] = False
    return MacroMatchedGraph(best_mask.astype(np.uint8), labels, cutpoints, best_audit)
