"""TRAIN-frozen, rate-conditional targets for the shared S_G producer.

The slow grammar hypothesis is not that every local contact transition must
change over hours.  A stable local grammar can instead be used in different
mixtures.  This module therefore separates four target families:

* local contact grammar (handled by the frozen step-wise decoder);
* contact-community occupancy;
* transitions between those communities; and
* event-repertoire mixture plus a continuous repertoire coordinate.

Communities, PCA coordinates and repertoire centres are fitted from
CALIBRATION/FIT events only.  Future blocks are always conditional on at least
one observed event, so a higher event rate cannot manufacture a grammar gain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans

from src.topic5_group_event_state.v02.marks import build_event_marks


@dataclass(frozen=True)
class GrammarDictionary:
    community_of_contact: np.ndarray
    repertoire_centres: np.ndarray
    event_repertoire_embedding: np.ndarray
    event_repertoire_label: np.ndarray
    n_communities: int
    n_repertoires: int
    fit_event_count: int
    provenance: dict[str, Any]


@dataclass(frozen=True)
class GrammarBlockTargets:
    community_occupancy: np.ndarray
    community_valid: np.ndarray
    cross_community_coupling: np.ndarray
    coupling_valid: np.ndarray
    repertoire_mixture: np.ndarray
    repertoire_valid: np.ndarray
    repertoire_embedding_mean: np.ndarray
    repertoire_embedding_valid: np.ndarray


def _community_labels(
    participation: np.ndarray,
    fit_rows: np.ndarray,
    requested: int,
) -> np.ndarray:
    part = np.asarray(participation, dtype=np.float64)
    rows = np.asarray(fit_rows, dtype=np.int64)
    if part.ndim != 2 or rows.size < 2:
        raise ValueError("community dictionary needs a 2-D event/contact matrix and FIT events")
    n_contact = part.shape[1]
    if n_contact == 1:
        return np.zeros(1, dtype=np.int64)
    n_community = min(max(2, int(requested)), n_contact)
    x = part[rows].T
    co = x @ x.T
    scale = np.sqrt(np.maximum(np.diag(co), 1.0))
    similarity = co / np.maximum(scale[:, None] * scale[None], 1e-12)
    similarity = np.clip((similarity + similarity.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    labels = fcluster(
        linkage(squareform(distance, checks=False), method="average"),
        t=n_community,
        criterion="maxclust",
    ).astype(np.int64) - 1
    # fcluster may return fewer groups under exact ties.  Dense remapping makes
    # the realised width explicit rather than leaving empty phantom classes.
    _unique, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


def fit_grammar_dictionary(
    participation: np.ndarray,
    relative_delay: np.ndarray,
    tied_group_id: np.ndarray,
    band_features: np.ndarray,
    *,
    band_available: Sequence[bool],
    band_names: Sequence[str],
    fit_rows: np.ndarray,
    seed: int,
    requested_communities: int = 4,
    requested_repertoires: int = 6,
) -> GrammarDictionary:
    part = np.asarray(participation, dtype=bool)
    delay = np.asarray(relative_delay, dtype=np.float64)
    tied = np.asarray(tied_group_id, dtype=np.int64)
    band = np.asarray(band_features, dtype=np.float64)
    rows = np.asarray(fit_rows, dtype=np.int64)
    if not (part.shape == delay.shape == tied.shape):
        raise ValueError("participation/delay/tied-group arrays are not aligned")
    if rows.size < 8:
        raise ValueError("grammar dictionary needs at least eight FIT events")
    marks = build_event_marks(
        part,
        delay,
        band,
        band_available=band_available,
        band_names=band_names,
        train_positions=rows,
        n_components=min(8, max(1, rows.size - 1)),
        seed=int(seed),
    )
    emb_slice = marks.block_slices["embedding"]
    embedding = np.asarray(marks.continuous[:, emb_slice], dtype=np.float32)
    valid_fit = rows[marks.valid[rows]]
    if valid_fit.size < 2:
        raise ValueError("no finite FIT repertoire embedding")
    n_repertoire = min(max(2, int(requested_repertoires)), valid_fit.size)
    kmeans = KMeans(n_clusters=n_repertoire, random_state=int(seed), n_init=10)
    kmeans.fit(embedding[valid_fit])
    label = np.full(part.shape[0], -1, dtype=np.int64)
    valid = np.asarray(marks.valid, dtype=bool)
    label[valid] = kmeans.predict(embedding[valid]).astype(np.int64)
    community = _community_labels(part, rows, requested_communities)
    return GrammarDictionary(
        community_of_contact=community,
        repertoire_centres=np.asarray(kmeans.cluster_centers_, dtype=np.float32),
        event_repertoire_embedding=embedding,
        event_repertoire_label=label,
        n_communities=int(np.unique(community).size),
        n_repertoires=int(n_repertoire),
        fit_event_count=int(rows.size),
        provenance={
            "fit_rows_only": True,
            "community_method": "average-linkage on FIT co-participation cosine distance",
            "repertoire_method": "FIT-only PCA event mark plus KMeans dictionary",
            "continuous_repertoire_parallel": True,
            "seed": int(seed),
        },
    )


def _event_grammar_arrays(
    participation: np.ndarray,
    tied_group_id: np.ndarray,
    dictionary: GrammarDictionary,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    part = np.asarray(participation, dtype=bool)
    tied = np.asarray(tied_group_id, dtype=np.int64)
    community = np.asarray(dictionary.community_of_contact, dtype=np.int64)
    n_event, _n_contact = part.shape
    k = dictionary.n_communities
    occupancy = np.zeros((n_event, k), dtype=np.float32)
    coupling = np.zeros((n_event, k * k), dtype=np.float32)
    coupling_valid = np.zeros(n_event, dtype=bool)
    for e in range(n_event):
        selected = np.flatnonzero(part[e])
        if selected.size:
            occupancy[e] = np.bincount(community[selected], minlength=k) / selected.size
        groups = np.unique(tied[e, selected]) if selected.size else np.asarray([], dtype=np.int64)
        groups = np.sort(groups[groups >= 0])
        current = np.zeros((k, k), dtype=np.float64)
        for a, b in zip(groups[:-1], groups[1:]):
            left = selected[tied[e, selected] == a]
            right = selected[tied[e, selected] == b]
            if left.size == 0 or right.size == 0:
                continue
            p_left = np.bincount(community[left], minlength=k).astype(np.float64)
            p_right = np.bincount(community[right], minlength=k).astype(np.float64)
            p_left /= p_left.sum(); p_right /= p_right.sum()
            current += np.outer(p_left, p_right)
        if current.sum() > 0:
            coupling[e] = (current / current.sum()).reshape(-1).astype(np.float32)
            coupling_valid[e] = True
    return occupancy, coupling, coupling_valid


def aggregate_grammar_blocks(
    *,
    grid_time: np.ndarray,
    horizons_seconds: Sequence[float],
    future_valid: np.ndarray,
    event_time: np.ndarray,
    participation: np.ndarray,
    tied_group_id: np.ndarray,
    dictionary: GrammarDictionary,
) -> GrammarBlockTargets:
    grid = np.asarray(grid_time, dtype=np.float64)
    horizon = np.asarray(horizons_seconds, dtype=np.float64)
    event_t = np.asarray(event_time, dtype=np.float64)
    valid = np.asarray(future_valid, dtype=bool)
    if valid.shape != (grid.size, horizon.size):
        raise ValueError("future-valid mask does not match grid/horizon axes")
    if np.any(np.diff(event_t) < 0):
        raise ValueError("event times must be sorted")
    event_occ, event_coupling, event_coupling_valid = _event_grammar_arrays(
        participation, tied_group_id, dictionary,
    )
    n_grid, n_horizon = valid.shape
    kc, kr = dictionary.n_communities, dictionary.n_repertoires
    de = dictionary.event_repertoire_embedding.shape[1]
    occ = np.zeros((n_grid, n_horizon, kc), dtype=np.float32)
    occ_valid = np.zeros((n_grid, n_horizon), dtype=bool)
    coupling = np.zeros((n_grid, n_horizon, kc * kc), dtype=np.float32)
    coupling_valid = np.zeros((n_grid, n_horizon), dtype=bool)
    mixture = np.zeros((n_grid, n_horizon, kr), dtype=np.float32)
    mixture_valid = np.zeros((n_grid, n_horizon), dtype=bool)
    emb = np.zeros((n_grid, n_horizon, de), dtype=np.float32)
    emb_valid = np.zeros((n_grid, n_horizon), dtype=bool)
    for j, h in enumerate(horizon):
        left = np.searchsorted(event_t, grid, side="left")
        right = np.searchsorted(event_t, grid + h, side="left")
        for row in np.flatnonzero(valid[:, j] & (right > left)):
            rr = np.arange(left[row], right[row], dtype=np.int64)
            current = event_occ[rr].sum(axis=0)
            if current.sum() > 0:
                occ[row, j] = current / current.sum(); occ_valid[row, j] = True
            cc = rr[event_coupling_valid[rr]]
            if cc.size:
                current_c = event_coupling[cc].sum(axis=0)
                coupling[row, j] = current_c / max(float(current_c.sum()), 1e-12)
                coupling_valid[row, j] = True
            labels = dictionary.event_repertoire_label[rr]
            labels = labels[labels >= 0]
            if labels.size:
                mixture[row, j] = np.bincount(labels, minlength=kr) / labels.size
                mixture_valid[row, j] = True
            finite = rr[np.isfinite(dictionary.event_repertoire_embedding[rr]).all(axis=1)]
            if finite.size:
                emb[row, j] = dictionary.event_repertoire_embedding[finite].mean(axis=0)
                emb_valid[row, j] = True
    return GrammarBlockTargets(
        community_occupancy=occ,
        community_valid=occ_valid,
        cross_community_coupling=coupling,
        coupling_valid=coupling_valid,
        repertoire_mixture=mixture,
        repertoire_valid=mixture_valid,
        repertoire_embedding_mean=emb,
        repertoire_embedding_valid=emb_valid,
    )
