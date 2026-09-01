"""Small deterministic helpers for the data-driven Z/M Figure 5."""
from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def stratified_random_sites(
    *, n_side: int, extent_mm: tuple[float, float], margin_mm: float, seed: int
) -> np.ndarray:
    """Draw one frozen random point per square stratum over the usable sheet."""
    if int(n_side) < 1:
        raise ValueError("n_side must be positive")
    lo, hi = map(float, extent_mm)
    margin = float(margin_mm)
    usable_lo, usable_hi = lo + margin, hi - margin
    if not usable_hi > usable_lo:
        raise ValueError("margin removes the entire sampling extent")
    edges = np.linspace(usable_lo, usable_hi, int(n_side) + 1)
    rng = np.random.default_rng(int(seed))
    sites = []
    for row in range(int(n_side)):
        for column in range(int(n_side)):
            sites.append([
                rng.uniform(edges[column], edges[column + 1]),
                rng.uniform(edges[row], edges[row + 1]),
            ])
    return np.asarray(sites, dtype=float)


def sustained_fraction_around(
    time_ms: Iterable[float],
    active_fraction: Iterable[float],
    spatial_fraction: Iterable[float],
    *,
    center_ms: float,
    window_ms: float,
    threshold: float,
) -> float:
    """Fraction of samples jointly above the global recruitment thresholds."""
    time = np.asarray(time_ms, float)
    active = np.asarray(active_fraction, float)
    spatial = np.asarray(spatial_fraction, float)
    if not (time.shape == active.shape == spatial.shape):
        raise ValueError("global-recruitment arrays must be aligned")
    half = float(window_ms) / 2.0
    keep = (time >= float(center_ms) - half) & (time < float(center_ms) + half)
    if not np.any(keep):
        return float("nan")
    valid = np.isfinite(active[keep]) & np.isfinite(spatial[keep])
    if not np.any(valid):
        return float("nan")
    return float(np.mean(
        (active[keep][valid] >= float(threshold))
        & (spatial[keep][valid] >= float(threshold))
    ))


def select_positive_identity_candidate(rows: Iterable[Mapping[str, object]]) -> dict:
    """Select the strongest positive TA/TB identity score with stable tie breaks."""
    candidates = []
    for row in rows:
        ta = float(row["ta_identity_r"])
        tb = float(row["tb_identity_r"])
        if not (np.isfinite(ta) and np.isfinite(tb)):
            continue
        winner = "TA" if ta >= tb else "TB"
        score = max(ta, tb)
        candidates.append((score, -float(row["time_ms"]), winner, dict(row)))
    if not candidates:
        raise ValueError("no finite TA/TB identity-score candidate")
    _score, _neg_time, winner, selected = max(candidates, key=lambda item: item[:2])
    selected["winning_template"] = winner
    selected["selection_score"] = float(_score)
    return selected
