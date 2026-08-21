"""Frozen Pareto-knee selection for rev12-ND Node-only fields."""
from __future__ import annotations

import numpy as np


COORDINATE_NAMES = (
    "weakest_mode_loss",
    "negative_complete_heldout_r2",
    "same_network_mode_missing_fraction",
    "negative_source_topology_quality",
)


def selection_coordinates(row: dict) -> np.ndarray:
    """Return four lower-is-better coordinates from one aggregate row."""
    score = row["score"]
    topology = row["source_topology"]
    reproducibility = topology["mean_across_network_template_cosine"]
    separation = topology["equal_network_between_mode_distance"]
    if reproducibility is None or separation is None:
        raise ValueError("source topology is not evaluable")
    values = np.asarray([
        score["mean_weakest_mode_lse"],
        -score["model_prototype_r2_on_heldout"],
        1.0 - score["same_network_both_fraction"],
        -float(reproducibility) * float(separation),
    ], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("selection coordinates must be finite")
    return values


def pareto_mask(values: np.ndarray) -> np.ndarray:
    """Mark rows that are not dominated in all lower-is-better coordinates."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("values must be a non-empty matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("Pareto values must be finite")
    keep = np.ones(len(values), dtype=bool)
    for index, row in enumerate(values):
        dominated = np.all(values <= row, axis=1) & np.any(values < row, axis=1)
        dominated[index] = False
        keep[index] = not np.any(dominated)
    return keep


def normalized_coordinates(values: np.ndarray) -> np.ndarray:
    """Min-max normalize coordinates; a degenerate coordinate contributes zero."""
    values = np.asarray(values, dtype=float)
    low = np.min(values, axis=0)
    span = np.max(values, axis=0) - low
    output = np.zeros_like(values)
    nonzero = span > 1e-12
    output[:, nonzero] = (values[:, nonzero] - low[nonzero]) / span[nonzero]
    return output


def select_pareto_knee(rows: list[dict]) -> dict:
    """Select the fixed normalized Pareto knee with roughness as last tie-break."""
    if not rows:
        raise ValueError("selection rows are empty")
    coordinates = np.vstack([selection_coordinates(row) for row in rows])
    normalized = normalized_coordinates(coordinates)
    front = pareto_mask(coordinates)
    knee_distance = np.sqrt(np.mean(normalized ** 2, axis=1))
    eligible = np.flatnonzero(front)
    selected_index = min(
        eligible,
        key=lambda index: (
            float(knee_distance[index]),
            float(rows[index]["score"]["roughness"]),
            str(rows[index]["candidate_id"]),
        ),
    )
    details = []
    for index, row in enumerate(rows):
        details.append({
            "candidate_id": row["candidate_id"],
            "coordinates": {
                name: float(value)
                for name, value in zip(COORDINATE_NAMES, coordinates[index])
            },
            "normalized_coordinates": {
                name: float(value)
                for name, value in zip(COORDINATE_NAMES, normalized[index])
            },
            "pareto": bool(front[index]),
            "knee_distance": float(knee_distance[index]),
            "roughness": float(row["score"]["roughness"]),
        })
    return {
        "selected_candidate_id": rows[selected_index]["candidate_id"],
        "selected_index": int(selected_index),
        "coordinate_names": list(COORDINATE_NAMES),
        "details": details,
        "contract": {
            "all_coordinates_lower_is_better": True,
            "pareto_then_normalized_ideal_distance": True,
            "roughness_is_last_tie_break": True,
            "complete_returned_events_are_primary": True,
            "patient_heldout_is_development_only": True,
        },
    }
