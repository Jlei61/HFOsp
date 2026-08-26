"""Patient graphs: a train-only directed propagation support and a geometry Laplacian.

The graph constrains which contacts may exchange information and how distance and
direction enter a message.  It is not a claim about true recurrent weights.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .event_marks import PatientEvents


@dataclass(frozen=True)
class PatientGraph:
    subject: str
    forward: np.ndarray    # (N, N) row-stochastic, i -> j when i precedes j
    reverse: np.ndarray    # (N, N) row-stochastic, transpose of the raw support
    geometry: np.ndarray   # (N, N) row-stochastic symmetric geometry kernel
    laplacian: np.ndarray  # (N, N) symmetric normalised Laplacian of the geometry kernel
    support_counts: np.ndarray   # (N, N) raw ordered co-occurrence counts (train only)
    length_scale_mm: float
    n_train_events: int
    n_geometry_mapped: int = 0
    geometry_available: bool = True

    @property
    def n_contacts(self) -> int:
        return int(self.forward.shape[0])

    def stack(self) -> np.ndarray:
        """(3, N, N) relation stack consumed by the shared message function."""
        out = np.stack([self.forward, self.reverse, self.geometry], axis=0)
        if not np.isfinite(out).all():
            raise ValueError(f"{self.subject}: graph relations contain non-finite entries")
        return out


def build_patient_graph(
    events: PatientEvents,
    *,
    split: str = "train",
    min_pair_count: int = 5,
    max_length_scale_quantile: float = 0.25,
) -> PatientGraph:
    """Directed propagation support from ``split`` events only, plus fixed geometry.

    ``split`` is checked against the frozen policy by the caller's LeakageGuard;
    this function never sees validation or test events unless explicitly asked.
    """
    mask = events.split_mask(split)
    part = events.participation[mask]
    gid = events.group_ids[mask].astype(np.int32)
    n_contacts = events.n_contacts
    counts = np.zeros((n_contacts, n_contacts), dtype=np.float64)

    # ordered co-participation: i strictly precedes j inside the same event
    for e in range(part.shape[0]):
        idx = np.flatnonzero(part[e])
        if len(idx) < 2:
            continue
        g = gid[e, idx]
        earlier = g[:, None] < g[None, :]
        counts[np.ix_(idx, idx)] += earlier
    counts[counts < min_pair_count] = 0.0

    forward = _row_stochastic(counts)
    reverse = _row_stochastic(counts.T)

    # 9 of 34 patients have unmapped contacts and 5 have no mapped contact at
    # all.  An unmapped contact gets no geometric edge rather than an invented
    # coordinate; a patient with fewer than two mapped contacts simply runs
    # without the geometry relation and is reported as its own stratum.
    coords = events.contact_coords.astype(np.float64)
    mapped = np.isfinite(coords).all(axis=1)
    n_mapped = int(mapped.sum())
    kernel = np.zeros((n_contacts, n_contacts), dtype=np.float64)
    length_scale = float("nan")
    if n_mapped >= 2:
        sub = coords[mapped]
        distance = np.linalg.norm(sub[:, None, :] - sub[None, :, :], axis=-1)
        off = distance[~np.eye(n_mapped, dtype=bool)]
        length_scale = max(float(np.quantile(off, max_length_scale_quantile)), 1e-3)
        block = np.exp(-0.5 * (distance / length_scale) ** 2)
        np.fill_diagonal(block, 0.0)
        kernel[np.ix_(mapped, mapped)] = block
    geometry = _row_stochastic(kernel)

    degree = kernel.sum(axis=1)
    inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0)
    laplacian = np.eye(n_contacts) - (inv_sqrt[:, None] * kernel * inv_sqrt[None, :])

    return PatientGraph(
        subject=events.subject,
        forward=forward.astype(np.float32),
        reverse=reverse.astype(np.float32),
        geometry=geometry.astype(np.float32),
        laplacian=laplacian.astype(np.float32),
        support_counts=counts.astype(np.float32),
        length_scale_mm=length_scale,
        n_train_events=int(mask.sum()),
        n_geometry_mapped=n_mapped,
        geometry_available=bool(n_mapped >= 2),
    )


def _row_stochastic(matrix: np.ndarray) -> np.ndarray:
    out = np.array(matrix, dtype=np.float64, copy=True)
    np.fill_diagonal(out, 0.0)
    total = out.sum(axis=1, keepdims=True)
    safe = np.where(total > 0, total, 1.0)
    normalised = out / safe
    # a node with no support keeps an all-zero row: it sends no message rather
    # than a uniform message it has no evidence for
    return normalised


#: Graph nulls for the H1 specificity question.
#:
#: "G1 beats G0" only says a relational message helped; it does not say the message
#: travelled along *this patient's* topology.  Each null keeps the message machinery
#: and the parameter count identical and changes only what the edges connect.
GRAPH_NULLS = ("patient_swapped", "degree_preserving_rewire", "identity", "geometry_only",
               "forward_only_shuffled")


def apply_graph_null(graph: PatientGraph, kind: str, *, seed: int,
                     donor: PatientGraph | None = None) -> PatientGraph:
    """Return a graph with the same shape and edge budget but a different topology.

    ``patient_swapped`` needs a ``donor`` graph from another patient; when the donor
    has a different contact count its relations are cropped or zero-padded, which
    changes the edge budget, so the caller must prefer a size-matched donor.
    """
    if kind not in GRAPH_NULLS:
        raise ValueError(f"unknown graph null {kind!r}; expected one of {GRAPH_NULLS}")
    rng = np.random.default_rng(seed)
    n = graph.n_contacts

    if kind == "identity":
        # no relational message at all: every contact only sees itself
        eye = np.eye(n, dtype=np.float64)
        return _replace_relations(graph, eye, eye, eye, f"identity")

    if kind == "geometry_only":
        # keep the fixed anatomical kernel, drop the learned propagation support
        geo = np.asarray(graph.geometry, dtype=np.float64)
        return _replace_relations(graph, geo, geo, geo, "geometry_only")

    if kind == "patient_swapped":
        if donor is None:
            raise ValueError("patient_swapped requires a donor graph from another patient")
        fwd = _fit_to(np.asarray(donor.forward, dtype=np.float64), n)
        rev = _fit_to(np.asarray(donor.reverse, dtype=np.float64), n)
        geo = np.asarray(graph.geometry, dtype=np.float64)   # anatomy stays this patient's
        return _replace_relations(graph, fwd, rev, geo, f"patient_swapped:{donor.subject}")

    if kind == "forward_only_shuffled":
        perm = rng.permutation(n)
        counts = np.asarray(graph.support_counts, dtype=np.float64)[np.ix_(perm, perm)]
        return _replace_relations(graph, _row_stochastic(counts), _row_stochastic(counts.T),
                                  np.asarray(graph.geometry, dtype=np.float64),
                                  "forward_only_shuffled")

    # degree_preserving_rewire: keep each contact's out-degree and the multiset of
    # edge weights, but re-attach the targets at random.
    counts = np.asarray(graph.support_counts, dtype=np.float64)
    rewired = np.zeros_like(counts)
    for i in range(n):
        weights = counts[i][counts[i] > 0]
        if weights.size == 0:
            continue
        targets = rng.choice([j for j in range(n) if j != i],
                             size=min(weights.size, n - 1), replace=False)
        rewired[i, targets] = rng.permutation(weights)[:len(targets)]
    return _replace_relations(graph, _row_stochastic(rewired), _row_stochastic(rewired.T),
                              np.asarray(graph.geometry, dtype=np.float64),
                              "degree_preserving_rewire")


def _fit_to(matrix: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((n, n), dtype=np.float64)
    m = min(n, matrix.shape[0])
    out[:m, :m] = matrix[:m, :m]
    return _row_stochastic(out)


def _replace_relations(graph: PatientGraph, forward: np.ndarray, reverse: np.ndarray,
                       geometry: np.ndarray, tag: str) -> PatientGraph:
    from dataclasses import replace
    for name, m in (("forward", forward), ("reverse", reverse), ("geometry", geometry)):
        if not np.isfinite(m).all():
            raise ValueError(f"{graph.subject}: {tag} produced non-finite {name}")
    return replace(graph,
                   subject=graph.subject,
                   forward=forward.astype(np.float32),
                   reverse=reverse.astype(np.float32),
                   geometry=geometry.astype(np.float32))
