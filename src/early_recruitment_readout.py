"""Model-agnostic interictal-arrival to early-recruitment-energy readout.

The module contains no Topic-4 dynamics.  Adapters provide two matched dynamic
signals, ``kick`` and ``control``, with shape ``(time, location)``.  This layer
defines the positive excess signal, the interictal half-peak arrival field, the
fixed-window early-energy proxy, spatial comparisons, and permutation nulls.

Ported verbatim into the codex/topic4-mz-slowvars worktree from
codex/topic4-early-readout (generic, numpy-only, model-agnostic) so the MZ
early-field bridge can reuse ``early_energy_field`` / ``compare_arrival_to_energy``
and the permutation primitives without cross-worktree imports (design 8h-prompt §0/§5).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from math import factorial
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ArrivalField:
    arrival_ms: np.ndarray
    peak_excess: np.ndarray
    participating: np.ndarray
    peak_fraction: float
    participation_threshold: float


@dataclass(frozen=True)
class EnergyField:
    energy: np.ndarray
    window_ms: tuple[float, float]
    status: str
    n_timepoints: int
    truncated_by_escape: bool


def _as_time_location(values, times_ms, *, name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, float)
    t = np.asarray(times_ms, float)
    if x.ndim != 2:
        raise ValueError(f"{name} must have shape (time, location); got {x.shape}")
    if t.ndim != 1 or x.shape[0] != t.size:
        raise ValueError(f"times_ms ({t.shape}) must match {name} time axis ({x.shape[0]})")
    if t.size == 0 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("times_ms must be finite and strictly increasing")
    return x, t


def positive_excess(kick, control) -> np.ndarray:
    """Positive kick-minus-control response; negative suppression contributes no energy."""
    k = np.asarray(kick, float)
    c = np.asarray(control, float)
    if k.shape != c.shape or k.ndim != 2:
        raise ValueError(f"kick/control must share (time, location) shape; got {k.shape}/{c.shape}")
    return np.maximum(k - c, 0.0)


def arrival_field(excess, times_ms, *, peak_fraction: float = 0.5,
                  participation_fraction: float = 0.1,
                  absolute_floor: float = 1e-10) -> ArrivalField:
    """First within-location half-peak crossing on an eligible positive-excess field.

    Participation is defined against the largest spatial peak, preventing every
    numerically nonzero tail from receiving a finite arrival time.
    """
    x, t = _as_time_location(excess, times_ms, name="excess")
    if not (0.0 < peak_fraction <= 1.0):
        raise ValueError("peak_fraction must lie in (0, 1]")
    if not (0.0 <= participation_fraction <= 1.0):
        raise ValueError("participation_fraction must lie in [0, 1]")
    pos = np.maximum(x, 0.0)
    peak = np.nanmax(pos, axis=0)
    global_peak = float(np.nanmax(peak)) if peak.size else 0.0
    threshold = max(float(absolute_floor), float(participation_fraction) * global_peak)
    participating = np.isfinite(peak) & (peak >= threshold) & (peak > 0.0)
    arrival = np.full(peak.shape, np.nan, float)
    for j in np.flatnonzero(participating):
        hit = np.flatnonzero(pos[:, j] >= float(peak_fraction) * peak[j])
        if hit.size:
            arrival[j] = t[hit[0]]
    participating &= np.isfinite(arrival)
    return ArrivalField(arrival, peak, participating, float(peak_fraction), float(threshold))


def early_energy_field(excess, times_ms, window_ms: Sequence[float], *,
                       escape_at_ms: float | None = None,
                       require_complete_presaturation_window: bool = True) -> EnergyField:
    """Mean squared positive excess in a fixed early window.

    The right boundary is inclusive.  If escape/saturation occurs on or before
    the window end and fail-closed behavior is requested, the returned field is
    all-NaN with status ``ineligible_escape_before_window_end``.
    """
    x, t = _as_time_location(excess, times_ms, name="excess")
    if len(window_ms) != 2:
        raise ValueError("window_ms must contain [start, end]")
    w0, w1 = map(float, window_ms)
    if not (w1 > w0):
        raise ValueError("window end must exceed start")
    escaped = escape_at_ms is not None and np.isfinite(escape_at_ms) and float(escape_at_ms) <= w1
    if escaped and require_complete_presaturation_window:
        return EnergyField(np.full(x.shape[1], np.nan), (w0, w1),
                           "ineligible_escape_before_window_end", 0, True)
    idx = np.flatnonzero((t >= w0) & (t <= w1))
    if idx.size < 2 or t[idx[0]] > w0 or t[idx[-1]] < w1:
        return EnergyField(np.full(x.shape[1], np.nan), (w0, w1),
                           "ineligible_incomplete_window", int(idx.size), bool(escaped))
    pos = np.maximum(x[idx], 0.0)
    energy = np.mean(pos ** 2, axis=0)
    return EnergyField(energy, (w0, w1), "eligible", int(idx.size), bool(escaped))


def _average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, float)
    i = 0
    while i < values.size:
        j = i
        while j + 1 < values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def _zscore(x: np.ndarray) -> np.ndarray:
    sd = float(np.std(x))
    return (x - float(np.mean(x))) / sd if sd > 0 else np.full(x.shape, np.nan)


def compare_arrival_to_energy(arrival_ms, energy, *, support_mask=None,
                              min_points: int = 3, top_k: int = 3) -> dict:
    """Compare interictal timing with target early energy on exactly matched support.

    ``arrival_energy_spearman < 0`` and ``earliness_energy_spearman > 0`` both mean
    that locations recruited earlier in the reference response are hotter early in
    the target response.
    """
    arrival = np.asarray(arrival_ms, float).ravel()
    e = np.asarray(energy, float).ravel()
    if arrival.shape != e.shape:
        raise ValueError(f"arrival/energy shape mismatch: {arrival.shape}/{e.shape}")
    support = np.ones(arrival.shape, bool) if support_mask is None else np.asarray(support_mask, bool).ravel()
    if support.shape != arrival.shape:
        raise ValueError("support_mask must align to arrival/energy")
    valid = support & np.isfinite(arrival) & np.isfinite(e)
    n = int(valid.sum())
    base = {"n": n, "min_points": int(min_points), "top_k_requested": int(top_k),
            "valid_mask": valid}
    if n < int(min_points):
        return {**base, "status": "insufficient_support", "arrival_energy_spearman": np.nan,
                "earliness_energy_spearman": np.nan, "field_cosine": np.nan,
                "top_k_used": 0, "top_k_overlap": np.nan}
    a, y = arrival[valid], e[valid]
    rho = _spearman(a, y)
    za, zy = _zscore(-a), _zscore(y)
    cosine = float(np.dot(za, zy) / (np.linalg.norm(za) * np.linalg.norm(zy))) \
        if np.all(np.isfinite(za)) and np.all(np.isfinite(zy)) else float("nan")
    k = min(int(top_k), max(1, n // 2))
    early = set(np.argsort(a, kind="mergesort")[:k].tolist())
    hot = set(np.argsort(-y, kind="mergesort")[:k].tolist())
    return {**base, "status": "eligible" if np.isfinite(rho) else "degenerate_field",
            "arrival_energy_spearman": rho,
            "earliness_energy_spearman": (-rho if np.isfinite(rho) else np.nan),
            "field_cosine": cosine, "top_k_used": int(k),
            "top_k_overlap": float(len(early & hot) / k)}


def _permutation_indices(n: int, rng: np.random.Generator, groups=None) -> np.ndarray:
    idx = np.arange(n)
    if groups is None:
        return rng.permutation(idx)
    g = np.asarray(groups)
    if g.shape != (n,):
        raise ValueError("groups must have one label per location")
    out = idx.copy()
    for label in np.unique(g):
        loc = np.flatnonzero(g == label)
        if loc.size >= 2:
            out[loc] = rng.permutation(loc)
    return out


def _permutation_groups(n: int, groups=None) -> list[np.ndarray]:
    if groups is None:
        return [np.arange(n)]
    g = np.asarray(groups)
    if g.shape != (n,):
        raise ValueError("groups must have one label per location")
    return [np.flatnonzero(g == label) for label in np.unique(g)]


def _exact_permutation_indices(n: int, groups=None):
    """Yield every index permutation while keeping values inside optional groups."""
    locs = _permutation_groups(n, groups)
    per_group = [list(permutations(loc.tolist())) for loc in locs]
    for choices in product(*per_group):
        out = np.arange(n)
        for loc, choice in zip(locs, choices):
            out[loc] = np.asarray(choice, int)
        yield out


def permutation_null(arrival_ms, energy, *, support_mask=None, groups=None,
                     n_permutations: int = 1000, seed: int = 0,
                     min_points: int = 3, max_exact_permutations: int = 50000) -> dict:
    """One-sided null for positive earliness-to-energy association.

    Energy values are shuffled; arrival and support stay fixed.  ``groups``
    implements constrained shuffles such as within-shaft permutations.
    """
    arrival = np.asarray(arrival_ms, float).ravel()
    e = np.asarray(energy, float).ravel()
    support = np.ones(arrival.shape, bool) if support_mask is None else np.asarray(support_mask, bool).ravel()
    valid = support & np.isfinite(arrival) & np.isfinite(e)
    obs = compare_arrival_to_energy(arrival, e, support_mask=valid, min_points=min_points)
    obs_stat = obs["earliness_energy_spearman"]
    if not np.isfinite(obs_stat):
        return {"status": obs["status"], "observed": obs_stat, "p_one_sided": np.nan,
                "null_median": np.nan, "null_p95": np.nan, "n_permutations": 0,
                "effective_shuffle_n": 0}
    av, ev = arrival[valid], e[valid]
    gv = None if groups is None else np.asarray(groups)[valid]
    locs = _permutation_groups(valid.sum(), gv)
    n_possible = int(np.prod([factorial(len(loc)) for loc in locs], dtype=object))
    moved = np.zeros(valid.sum(), bool)
    if n_possible <= int(max_exact_permutations):
        method = "exact"
        values = []
        for perm in _exact_permutation_indices(valid.sum(), gv):
            moved |= perm != np.arange(valid.sum())
            values.append(-_spearman(av, ev[perm]))
        null = np.asarray(values, float)
    else:
        method = "monte_carlo"
        rng = np.random.default_rng(int(seed))
        null = np.empty(int(n_permutations), float)
        for b in range(int(n_permutations)):
            perm = _permutation_indices(valid.sum(), rng, gv)
            moved |= perm != np.arange(valid.sum())
            null[b] = -_spearman(av, ev[perm])
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"status": "degenerate_null", "observed": float(obs_stat), "p_one_sided": np.nan,
                "null_median": np.nan, "null_p95": np.nan, "n_permutations": 0,
                "effective_shuffle_n": int(moved.sum())}
    p = (float(np.sum(null >= obs_stat) / null.size) if method == "exact"
         else float((1 + np.sum(null >= obs_stat)) / (1 + null.size)))
    return {"status": "eligible", "observed": float(obs_stat), "p_one_sided": p,
            "null_median": float(np.median(null)), "null_p95": float(np.percentile(null, 95)),
            "n_permutations": int(null.size), "n_unique_possible": n_possible,
            "method": method, "effective_shuffle_n": int(moved.sum())}
def register_source_grid_to_subject_sheet(
    source_xy,
    *,
    model_axis_theta_rad,
    subject_source_xy,
    subject_sink_xy,
    model_source_xy=(0.0, 0.0),
    model_axis_anchor_mm,
):
    """Place a reduced-model field in an accepted subject-SNN coordinate frame.

    This is a single similarity transform: the model source maps to the subject
    source focus, and a point ``model_axis_anchor_mm`` down the model E->E axis
    maps to the subject sink focus.  It rotates/translates/scales coordinates only;
    it never interpolates or changes the simulated field values.
    """
    xy = np.asarray(source_xy, dtype=float)
    source = np.asarray(subject_source_xy, dtype=float)
    sink = np.asarray(subject_sink_xy, dtype=float)
    origin = np.asarray(model_source_xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("source_xy must have shape (n_source, 2)")
    if source.shape != (2,) or sink.shape != (2,) or origin.shape != (2,):
        raise ValueError("source/sink/model_source coordinates must be length-2")
    anchor = float(model_axis_anchor_mm)
    if not np.isfinite(anchor) or anchor <= 0:
        raise ValueError("model_axis_anchor_mm must be finite and positive")
    subject_axis = sink - source
    subject_span = float(np.linalg.norm(subject_axis))
    if subject_span <= 1e-12:
        raise ValueError("subject source and sink foci coincide")

    theta_subject = float(np.arctan2(subject_axis[1], subject_axis[0]))
    delta = theta_subject - float(model_axis_theta_rad)
    c, s = np.cos(delta), np.sin(delta)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    scale = subject_span / anchor
    transformed = (xy - origin) @ rotation.T * scale + source
    offset = source - (origin @ rotation.T) * scale
    return transformed, {
        "scale": float(scale),
        "rotation": rotation,
        "offset": offset,
        "model_source_xy": origin,
        "model_axis_anchor_mm": anchor,
        "subject_source_xy": source,
        "subject_sink_xy": sink,
        "subject_axis_theta_deg": float(np.degrees(theta_subject)),
        "subject_axis_span_mm": subject_span,
    }
