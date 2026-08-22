"""Model-internal target selection for rev12-ND same-checkpoint intervention."""
from __future__ import annotations

import numpy as np

from src.topic4_node_dualmode import (
    event_features,
    normalize_event_ranks,
    source_topology_features,
)


def early_support_probability(onset_maps: np.ndarray,
                              fraction: float = 0.10) -> np.ndarray:
    """Equal-event probability that a bin belongs to an event's earliest support."""
    maps = np.asarray(onset_maps, float)
    if maps.ndim != 3 or not len(maps):
        raise ValueError("onset maps must have shape (event, y, x) and be non-empty")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    support = np.zeros_like(maps, dtype=float)
    evaluable = np.zeros(len(maps), bool)
    for index, onset in enumerate(maps):
        finite = np.isfinite(onset)
        if not np.any(finite):
            continue
        count = int(np.sum(finite))
        early_count = min(count, max(1, int(np.ceil(float(fraction) * count))))
        threshold = np.partition(onset[finite], early_count - 1)[early_count - 1]
        support[index, finite & (onset <= threshold)] = 1.0
        evaluable[index] = True
    if not np.any(evaluable):
        raise ValueError("onset maps contain no evaluable event")
    return np.mean(support[evaluable], axis=0)


def network_balanced_early_support(onset_maps_by_network: list[np.ndarray],
                                   labels_by_network: list[np.ndarray],
                                   mode: int, fraction: float = 0.10) -> np.ndarray:
    """Average within-network early probabilities so event-rich seeds do not dominate."""
    if len(onset_maps_by_network) != len(labels_by_network):
        raise ValueError("network maps and labels do not align")
    probabilities = []
    for maps, labels in zip(onset_maps_by_network, labels_by_network):
        maps = np.asarray(maps, float)
        labels = np.asarray(labels, int)
        if maps.ndim != 3 or labels.shape != (len(maps),):
            raise ValueError("one network map bundle does not align")
        selected = maps[labels == int(mode)]
        if len(selected):
            probabilities.append(early_support_probability(selected, fraction=fraction))
    if not probabilities:
        raise ValueError(f"mode {mode} has no evaluable network")
    return np.mean(probabilities, axis=0)


def select_representative_seed(network_scores: list[dict],
                               source_counts: dict[int, int]) -> int:
    """Select a fully dual-mode seed nearest the candidate's median objective."""
    eligible = [
        row for row in network_scores
        if np.all(np.asarray(row.get("mode_counts", []), int) > 0)
        and int(source_counts.get(int(row["seed"]), 0)) >= 2
        and np.isfinite(float(row["objective"]))
    ]
    if not eligible:
        raise ValueError("no dual-mode source-evaluable confirmation seed")
    objectives = np.asarray([float(row["objective"]) for row in eligible])
    events = np.asarray([int(row["n_events"]) for row in eligible])
    objective_median = float(np.median(objectives))
    event_median = float(np.median(events))
    ranked = sorted(
        eligible,
        key=lambda row: (
            abs(float(row["objective"]) - objective_median),
            abs(int(row["n_events"]) - event_median),
            int(row["seed"]),
        ),
    )
    return int(ranked[0]["seed"])


def grid_covariates(positions_e: np.ndarray, h_e: np.ndarray,
                    baseline_spikes: np.ndarray, *, dt_ms: float,
                    sheet_mm: float, bin_mm: float) -> dict:
    """Bin h, local E density and baseline E rate on the source-topology grid."""
    positions = np.asarray(positions_e, float)
    h = np.asarray(h_e, float)
    spikes = np.asarray(baseline_spikes, bool)
    if (positions.ndim != 2 or positions.shape[1] != 2
            or h.shape != (len(positions),)
            or spikes.ndim != 2 or spikes.shape[1] != len(positions)):
        raise ValueError("positions, h and baseline spikes do not align")
    size = int(round(float(sheet_mm) / float(bin_mm)))
    if size <= 0 or not np.isclose(size * float(bin_mm), float(sheet_mm)):
        raise ValueError("bin size must tile the sheet")
    xy = np.floor(positions / float(bin_mm)).astype(int)
    xy = np.clip(xy, 0, size - 1)
    flat = xy[:, 1] * size + xy[:, 0]
    density = np.bincount(flat, minlength=size * size).astype(float)
    h_sum = np.bincount(flat, weights=h, minlength=size * size)
    duration_seconds = len(spikes) * float(dt_ms) * 1e-3
    if duration_seconds <= 0.0:
        raise ValueError("baseline window must be non-empty")
    spike_sum = np.bincount(
        flat, weights=np.sum(spikes, axis=0), minlength=size * size,
    )
    h_mean = np.divide(h_sum, density, out=np.full_like(h_sum, np.nan),
                       where=density > 0)
    rate = np.divide(
        spike_sum, density * duration_seconds,
        out=np.full_like(spike_sum, np.nan), where=density > 0,
    )
    return {
        "h_mean": h_mean.reshape(size, size),
        "e_density": density.reshape(size, size),
        "baseline_rate_hz": rate.reshape(size, size),
    }


def _grid_centers(shape: tuple[int, int], bin_mm: float) -> np.ndarray:
    y, x = np.indices(shape)
    return np.column_stack([
        (x.ravel() + 0.5) * float(bin_mm),
        (y.ravel() + 0.5) * float(bin_mm),
    ])


def select_hotspot_triplet(early_probability: np.ndarray, covariates: dict, *,
                           bin_mm: float, minimum_separation_mm: float = 3.0,
                           off_template_quantile: float = 0.25) -> dict:
    """Select dominant, separated secondary and covariate-matched control bins."""
    probability = np.asarray(early_probability, float)
    if probability.ndim != 2 or not np.all(np.isfinite(probability)):
        raise ValueError("early probability must be a finite 2D map")
    required = ("h_mean", "e_density", "baseline_rate_hz")
    arrays = {key: np.asarray(covariates[key], float) for key in required}
    if any(value.shape != probability.shape for value in arrays.values()):
        raise ValueError("covariate maps must align to the early-probability map")
    valid = arrays["e_density"] > 0
    valid &= np.logical_and.reduce([np.isfinite(value) for value in arrays.values()])
    if np.sum(valid) < 3:
        raise ValueError("fewer than three populated bins are available")
    centers = _grid_centers(probability.shape, bin_mm)
    p = probability.ravel()
    valid_flat = valid.ravel()
    dominant = int(np.flatnonzero(valid_flat)[np.argmax(p[valid_flat])])
    distance_to_dominant = np.linalg.norm(centers - centers[dominant], axis=1)
    secondary_pool = valid_flat & (distance_to_dominant >= minimum_separation_mm)
    if not np.any(secondary_pool):
        raise ValueError("no spatially separated secondary hotspot is available")
    secondary_candidates = np.flatnonzero(secondary_pool)
    secondary = int(secondary_candidates[np.argmax(p[secondary_candidates])])

    distance_to_secondary = np.linalg.norm(centers - centers[secondary], axis=1)
    threshold = float(np.quantile(p[valid_flat], float(off_template_quantile)))
    control_pool = (
        valid_flat
        & (distance_to_dominant >= minimum_separation_mm)
        & (distance_to_secondary >= minimum_separation_mm)
        & (p <= threshold)
    )
    if not np.any(control_pool):
        raise ValueError("no separated off-template control bin is available")
    covariate_matrix = np.column_stack([
        arrays[key].ravel() for key in required
    ])
    scale = np.nanpercentile(covariate_matrix[valid_flat], 75, axis=0) - np.nanpercentile(
        covariate_matrix[valid_flat], 25, axis=0,
    )
    scale = np.where(scale > 1e-12, scale, 1.0)
    control_candidates = np.flatnonzero(control_pool)
    mismatch = np.sum(
        np.abs(covariate_matrix[control_candidates] - covariate_matrix[dominant])
        / scale,
        axis=1,
    )
    # Lower template probability is the deterministic secondary key.
    order = np.lexsort((control_candidates, p[control_candidates], mismatch))
    control = int(control_candidates[order[0]])

    def record(index: int) -> dict:
        row, column = np.unravel_index(index, probability.shape)
        return {
            "flat_index": index,
            "row": int(row),
            "column": int(column),
            "xy_mm": centers[index].tolist(),
            "early_probability": float(p[index]),
            "h_mean": float(covariate_matrix[index, 0]),
            "e_density": float(covariate_matrix[index, 1]),
            "baseline_rate_hz": float(covariate_matrix[index, 2]),
        }

    return {
        "dominant": record(dominant),
        "secondary": record(secondary),
        "matched_off_template": record(control),
        "minimum_separation_mm": float(minimum_separation_mm),
        "off_template_quantile": float(off_template_quantile),
    }


def representative_event_index(onset_maps: np.ndarray, ranks: np.ndarray,
                               labels: np.ndarray, mode: int) -> int:
    """Choose a joint source-topology/contact-rank medoid without visual selection."""
    maps = np.asarray(onset_maps, float)
    ranks = np.asarray(ranks, float)
    labels = np.asarray(labels, int)
    if maps.ndim != 3 or ranks.ndim != 2 or labels.shape != (len(maps),):
        raise ValueError("event maps, ranks and labels do not align")
    selected = np.flatnonzero(labels == int(mode))
    if not len(selected):
        raise ValueError(f"mode {mode} has no evaluable event")
    source = source_topology_features(maps[selected])
    contact = event_features(normalize_event_ranks(ranks[selected]))
    source /= np.sqrt(max(1, source.shape[1]))
    contact /= np.sqrt(max(1, contact.shape[1]))
    joint = np.concatenate([source, contact], axis=1)
    centroid = np.mean(joint, axis=0)
    distance = np.linalg.norm(joint - centroid, axis=1)
    return int(selected[int(np.argmin(distance))])
