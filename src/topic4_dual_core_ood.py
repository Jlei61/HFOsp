"""Pure helpers for OOD-guided two-core Node recovery."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
from scipy.stats import qmc

from src.topic4_shaft_aware import contract_groups
from src.topic4_shaft_aware_direction import assign_direction_modes


def canonical_centers(centers_mm: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers_mm, dtype=float)
    if centers.shape != (2, 2) or not np.isfinite(centers).all():
        raise ValueError("centers_mm must be finite with shape (2, 2)")
    order = np.lexsort((centers[:, 1], centers[:, 0]))
    return centers[order]


def candidate_field_sha256(centers_mm: np.ndarray, target_count: int) -> str:
    payload = {
        "field_type": "manual_dual_core_budget_matched",
        "centers_mm": canonical_centers(centers_mm).tolist(),
        "target_count": int(target_count),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _reference_core_fractions(
    centers_mm: np.ndarray, target_count: int, *, sheet_l_mm: float = 20.0,
    reference_side: int = 160, reference_n_e: int = 32000,
) -> np.ndarray:
    axis = (np.arange(reference_side, dtype=float) + 0.5) * (
        float(sheet_l_mm) / reference_side
    )
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    centers = canonical_centers(centers_mm)
    distance = np.linalg.norm(
        positions[:, None, :] - centers[None, :, :], axis=2,
    )
    nearest = np.argmin(distance, axis=1)
    minimum = distance[np.arange(len(distance)), nearest]
    scaled_count = int(round(int(target_count) * len(positions) / reference_n_e))
    scaled_count = min(max(scaled_count, 2), len(positions))
    selected = np.argsort(minimum, kind="stable")[:scaled_count]
    counts = np.bincount(nearest[selected], minlength=2).astype(float)
    return counts / counts.sum()


def generate_sobol_candidates(family: Mapping) -> list[dict]:
    """Generate a deterministic whole-sheet five-parameter two-core library."""
    n = int(family["sobol_candidates"])
    low, high = map(float, family["sheet_bounds_mm"])
    count_low, count_high = map(int, family["target_count_bounds"])
    minimum_sep = float(family["minimum_center_separation_mm"])
    maximum_sep = float(family["maximum_center_separation_mm"])
    minimum_fraction = float(family["minimum_reference_budget_fraction_per_core"])
    sampler = qmc.Sobol(d=5, scramble=True, seed=int(family["sobol_seed"]))
    rows = []
    draw_count = 0
    while len(rows) < n:
        draw = sampler.random(1)[0]
        draw_count += 1
        centers = canonical_centers(
            low + (high - low) * np.asarray(draw[:4]).reshape(2, 2)
        )
        separation = float(np.linalg.norm(centers[0] - centers[1]))
        if not minimum_sep <= separation <= maximum_sep:
            continue
        target_count = int(round(count_low + draw[4] * (count_high - count_low)))
        fractions = _reference_core_fractions(centers, target_count)
        if float(np.min(fractions)) < minimum_fraction:
            continue
        candidate_id = f"dualcore_s{len(rows):02d}"
        rows.append({
            "candidate_id": candidate_id,
            "field_type": "manual_dual_core_budget_matched",
            "centers_mm": centers.tolist(),
            "target_count": target_count,
            "center_separation_mm": separation,
            "reference_budget_fraction_per_core": fractions.tolist(),
            "field_sha256": candidate_field_sha256(centers, target_count),
            "sobol_accepted_index": len(rows),
            "sobol_draw_index": draw_count - 1,
        })
    anchor = family["historical_anchor"]
    centers = canonical_centers(anchor["centers_mm"])
    target_count = int(anchor["target_count"])
    fractions = _reference_core_fractions(centers, target_count)
    rows.append({
        "candidate_id": str(anchor["candidate_id"]),
        "field_type": "manual_dual_core_budget_matched",
        "centers_mm": centers.tolist(),
        "target_count": target_count,
        "center_separation_mm": float(np.linalg.norm(centers[0] - centers[1])),
        "reference_budget_fraction_per_core": fractions.tolist(),
        "field_sha256": candidate_field_sha256(centers, target_count),
        "historical_anchor": True,
    })
    return rows


def load_embedding(target_npz: str) -> dict:
    with np.load(target_npz, allow_pickle=False) as loaded:
        return {
            "center": np.asarray(loaded["feature_center"], float),
            "scale": np.asarray(loaded["feature_scale"], float),
            "components": np.asarray(loaded["pca_components"], float),
            "directions": np.asarray(loaded["sw_directions"], float),
        }


def score_returned_event_support(
    onsets: np.ndarray, returned: np.ndarray, *, contract: Mapping,
    embedding: Mapping, classifier: Mapping,
) -> dict:
    """Score all returned events; unreadable returned events count as OOD."""
    onsets = np.asarray(onsets, float)
    returned = np.asarray(returned, bool)
    if onsets.ndim != 2 or returned.shape != (len(onsets),):
        raise ValueError("onsets and returned shapes disagree")
    assigned = assign_direction_modes(
        onsets, groups=contract_groups(contract), embedding=embedding,
        classifier=classifier,
    )
    labels = np.asarray(assigned["labels"], int)
    ood = np.asarray(assigned["ood"], bool)
    distance = np.asarray(assigned["ood_distance"], float)
    thresholds = np.asarray(classifier["ood_distance_thresholds"], float)
    readable = np.sum(np.isfinite(onsets), axis=1) >= 3
    in_support = returned & readable & ~ood
    n_returned = int(np.sum(returned))
    n_in_support = int(np.sum(in_support))
    counts = np.bincount(labels[in_support], minlength=2)
    normalized_distance = distance / thresholds[labels]
    mode_distance = [
        float(np.mean(normalized_distance[in_support & (labels == mode)]))
        if np.any(in_support & (labels == mode)) else None
        for mode in (0, 1)
    ]
    readable_returned = returned & readable
    return {
        "n_events": int(len(onsets)),
        "n_returned": n_returned,
        "n_returned_readable": int(np.sum(readable_returned)),
        "n_in_support": n_in_support,
        "mode_counts_in_support": counts,
        "both_modes_in_support": bool(np.all(counts > 0)),
        "ood_all_returned": (
            float(1.0 - n_in_support / n_returned) if n_returned else 1.0
        ),
        "ood_returned_readable": (
            float(np.mean(ood[readable_returned]))
            if np.any(readable_returned) else 1.0
        ),
        "unreadable_returned_fraction": (
            float(np.mean(~readable[returned])) if n_returned else 1.0
        ),
        "mean_normalized_support_distance_by_mode": mode_distance,
        "weakest_mode_normalized_support_distance": (
            float(max(mode_distance)) if all(x is not None for x in mode_distance)
            else None
        ),
        "labels": labels,
        "ood": ood,
        "readable": readable,
        "in_support": in_support,
        "normalized_support_distance": normalized_distance,
    }


def candidate_sort_key(summary: Mapping) -> tuple:
    return (
        -int(summary["networks_with_both_modes"]),
        float(summary["equal_network_ood_all_returned"]),
        float(summary.get("weakest_mode_normalized_support_distance") or np.inf),
        -float(summary["equal_network_returned_events"]),
        str(summary["candidate_id"]),
    )


def spatial_event_activity_grid(
    spikes: np.ndarray, positions_e: np.ndarray, events: list[Mapping], *,
    dt_ms: float, bin_ms: float, spatial_bins: int, sheet_l_mm: float,
    pad_before_ms: float = 20.0, pad_after_ms: float = 40.0,
) -> dict:
    """Compress per-neuron spikes around detected events into spatial grids."""
    spikes = np.asarray(spikes, bool)
    positions = np.asarray(positions_e, float)
    if spikes.ndim != 2 or spikes.shape[1] != len(positions):
        raise ValueError("spikes must have shape (time, E neuron)")
    time_steps = max(1, int(round(float(bin_ms) / float(dt_ms))))
    x = np.clip((positions[:, 0] / sheet_l_mm * spatial_bins).astype(int), 0,
                spatial_bins - 1)
    y = np.clip((positions[:, 1] / sheet_l_mm * spatial_bins).astype(int), 0,
                spatial_bins - 1)
    cells = y * spatial_bins + x
    frames, times, frame_event, event_start, event_count = [], [], [], [], []
    for event_index, event in enumerate(events):
        start = max(0, int(np.floor(
            (float(event["t_on_ms"]) - pad_before_ms) / dt_ms
        )))
        stop = min(len(spikes), int(np.ceil(
            (float(event["t_off_ms"]) + pad_after_ms) / dt_ms
        )))
        event_start.append(len(frames))
        for left in range(start, stop, time_steps):
            right = min(stop, left + time_steps)
            per_neuron = spikes[left:right].sum(axis=0)
            grid = np.bincount(
                cells, weights=per_neuron, minlength=spatial_bins ** 2,
            ).reshape(spatial_bins, spatial_bins)
            frames.append(grid.astype(np.float32))
            times.append(0.5 * (left + right) * dt_ms)
            frame_event.append(event_index)
        event_count.append(len(frames) - event_start[-1])
    return {
        "activity_grid": np.asarray(frames, np.float32).reshape(
            (-1, spatial_bins, spatial_bins)
        ),
        "activity_grid_time_ms": np.asarray(times, np.float32),
        "activity_grid_event_index": np.asarray(frame_event, np.int32),
        "activity_grid_event_start": np.asarray(event_start, np.int32),
        "activity_grid_event_count": np.asarray(event_count, np.int32),
        "activity_grid_bin_ms": np.asarray(bin_ms, float),
        "activity_grid_spatial_bins": np.asarray(spatial_bins, np.int32),
    }
