"""Interictal propagation skeleton geometry (axis-frame, descriptive model-input).

Spec: docs/superpowers/specs/2026-06-08-sef-hfo-propagation-skeleton-geometry-design.md

All functions are pure (no file I/O). Phantom-safety: source/sink cores are
derived from a template axis whose non-participating channels are NaN, so a
phantom can never enter a core (NaN sorts out of nanargsort).
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Handles both "A'1" and "A1'" (prime before OR after the digit)
_NAME_RE = re.compile(r"^([A-Za-z]+)('?)\s*(\d+)('?)$")


def parse_shaft(channel_name: str) -> Tuple[Optional[str], Optional[int]]:
    """(shaft_prefix, ordinal) from a channel name; (None, None) if unparseable."""
    m = _NAME_RE.match(str(channel_name).strip())
    if not m:
        return (None, None)
    prime = "'" if (m.group(2) or m.group(4)) else ""
    return (m.group(1) + prime, int(m.group(3)))


def build_endpoint_cores(
    template_axis: np.ndarray,
    eligible_mask: np.ndarray,
    *,
    k_primary: int = 3,
) -> Dict[str, object]:
    """Source/sink cores from a phantom-safe template axis + eligibility tier.

    eligible_mask = participating (template non-NaN) AND coord-mapped.
    Cores are the k lowest / k highest template-axis channels AMONG eligible.
    Tier gate (spec §2): n_eff>=7 -> k=3 primary; n_eff in {5,6} -> k=2
    fallback; else descriptive_only (no cores, excluded from cohort stats).
    """
    template_axis = np.asarray(template_axis, dtype=float)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)
    eligible_idx = np.where(eligible_mask & ~np.isnan(template_axis))[0]
    n_eff = int(eligible_idx.size)

    base = {
        "n_eff": n_eff,
        "k_used": 0,
        "tier": "descriptive_only",
        "source_idx": [],
        "sink_idx": [],
        "interior_idx": [],
    }
    if n_eff >= 7:
        k, tier = k_primary, "primary"
    elif n_eff in (5, 6):
        k, tier = 2, "fallback"
    else:
        return base
    # cores must be disjoint with a non-empty interior: 2k < n_eff
    if 2 * k >= n_eff:
        return base

    order = eligible_idx[np.argsort(template_axis[eligible_idx], kind="stable")]
    source = sorted(int(i) for i in order[:k])
    sink = sorted(int(i) for i in order[-k:])
    interior = sorted(int(i) for i in order[k:-k])
    base.update(k_used=k, tier=tier, source_idx=source, sink_idx=sink,
                interior_idx=interior)
    return base


def compute_axis_frame(
    coords: np.ndarray,
    source_idx: Sequence[int],
    sink_idx: Sequence[int],
) -> Dict[str, object]:
    """Axis frame from source/sink core centroids.

    along_axis[c] = (p_c - source_centroid) . unit_axis   (0 at source, L at sink)
    off_axis[c]   = || (p_c - source_centroid) - along*unit_axis ||
    Channels with NaN coords get NaN along/off. Coincident centroids ->
    degenerate_axis=True and along/off all NaN.
    """
    coords = np.asarray(coords, dtype=float)
    src_c = np.nanmean(coords[list(source_idx)], axis=0)
    snk_c = np.nanmean(coords[list(sink_idx)], axis=0)
    axis = snk_c - src_c
    L = float(np.linalg.norm(axis))
    n = coords.shape[0]
    along = np.full(n, np.nan)
    off = np.full(n, np.nan)
    degenerate = L < 1e-9
    if not degenerate:
        u = axis / L
        rel = coords - src_c                      # (n,3)
        along = rel @ u                            # (n,)
        perp_vec = rel - np.outer(along, u)        # (n,3)
        off = np.linalg.norm(perp_vec, axis=1)
        bad = np.isnan(coords).any(axis=1)
        along[bad] = np.nan
        off[bad] = np.nan
    return {
        "source_centroid": src_c.tolist(),
        "sink_centroid": snk_c.tolist(),
        "axis_length": L,
        "along_axis": along,
        "off_axis": off,
        "degenerate_axis": bool(degenerate),
    }


def _meb_radius(points: np.ndarray) -> float:
    """Exact min-enclosing-ball radius for small point sets (k<=3 exact;
    k>=4 Ritter upper bound). Cores here are k in {2,3}."""
    pts = np.asarray(points, dtype=float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    m = pts.shape[0]
    if m == 0:
        return float("nan")
    if m == 1:
        return 0.0
    if m == 2:
        return float(np.linalg.norm(pts[0] - pts[1]) / 2.0)
    if m == 3:
        a, b, c = pts
        # MEB of 3 points: longest-side diameter if triangle non-acute,
        # else circumradius.
        sides = np.array([np.linalg.norm(b - c),
                          np.linalg.norm(a - c),
                          np.linalg.norm(a - b)])
        longest = sides.max()
        s = sides
        # non-acute (obtuse/right) at the vertex opposite the longest side:
        i = int(np.argmax(sides))
        others = [s[j] for j in range(3) if j != i]
        if longest ** 2 >= others[0] ** 2 + others[1] ** 2:
            return float(longest / 2.0)
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        if area < 1e-12:
            return float(longest / 2.0)
        return float((s[0] * s[1] * s[2]) / (4.0 * area))
    # Ritter upper bound (defensive; not expected for k in {2,3})
    center = pts.mean(axis=0)
    return float(np.max(np.linalg.norm(pts - center, axis=1)))


def core_radii(core_coords: np.ndarray, centroid: np.ndarray) -> Dict[str, object]:
    """RMS-to-centroid (primary), MEB, and max-pairwise of a core point set.

    RMS-to-centroid can read a split core's gap as a fake-large 'compact
    radius'; MEB + max-pairwise expose that (advisor #3).
    """
    pts = np.asarray(core_coords, dtype=float)
    valid = pts[~np.isnan(pts).any(axis=1)]
    centroid = np.asarray(centroid, dtype=float)
    if valid.shape[0] == 0:
        return {"rms_mm": float("nan"), "meb_mm": float("nan"),
                "max_pairwise_mm": float("nan")}
    rms = float(np.sqrt(np.mean(np.sum((valid - centroid) ** 2, axis=1))))
    if valid.shape[0] < 2:
        maxpair = 0.0
    else:
        diff = valid[:, None, :] - valid[None, :, :]
        maxpair = float(np.linalg.norm(diff, axis=-1).max())
    return {"rms_mm": rms, "meb_mm": _meb_radius(valid),
            "max_pairwise_mm": maxpair}


def perp_spread(off_axis: np.ndarray, participating_mask: np.ndarray) -> Dict[str, object]:
    """RMS (primary) + p75/p90 (robust sensitivity) of off-axis distance over
    PARTICIPATING channels. NOTE (advisor #1): off-axis spread of participating
    channels is a participation+sampling-bounded LOWER BOUND, not 'patch width'.
    Report as 'participating-channel perpendicular spread'.
    """
    off = np.asarray(off_axis, dtype=float)
    part = np.asarray(participating_mask, dtype=bool)
    vals = off[part & ~np.isnan(off)]
    if vals.size == 0:
        return {"rms_mm": float("nan"), "p75_mm": float("nan"),
                "p90_mm": float("nan"), "n": 0}
    return {
        "rms_mm": float(np.sqrt(np.mean(vals ** 2))),
        "p75_mm": float(np.percentile(vals, 75)),
        "p90_mm": float(np.percentile(vals, 90)),
        "n": int(vals.size),
    }


def classify_sampling_geometry(
    channel_names: Sequence[str],
    participating_mask: np.ndarray,
    off_axis: np.ndarray,
    *,
    spacing_mm: float = 3.5,
) -> Dict[str, object]:
    """1D (single-shaft / collinear) vs distributed sampling. Width is NOT
    measurable for 1D subjects (axis length still is). spec §4."""
    part = np.asarray(participating_mask, dtype=bool)
    off = np.asarray(off_axis, dtype=float)
    names = [channel_names[i] for i in np.where(part)[0]]
    shafts = {parse_shaft(nm)[0] for nm in names} - {None}
    vals = off[part & ~np.isnan(off)]
    p90 = float(np.percentile(vals, 90)) if vals.size else 0.0
    one_d = (len(shafts) <= 1) or (p90 < spacing_mm)
    return {
        "geometry": "1D" if one_d else "distributed",
        "n_shafts": int(len(shafts)),
        "p90_off_mm": p90,
        "measurable": not one_d,
    }


def perp_spread_participation_sweep(
    off_axis: np.ndarray,
    full_count: np.ndarray,
    *,
    thresholds: Sequence[int] = (1, 5, 10, 20),
) -> List[Dict[str, object]]:
    """Off-axis spread as the per-channel event-count threshold rises. Shows
    whether 'width' is just a participation/rate artifact (advisor #1)."""
    off = np.asarray(off_axis, dtype=float)
    cnt = np.asarray(full_count, dtype=float)
    out = []
    for t in thresholds:
        keep = (cnt >= t) & ~np.isnan(off)
        s = perp_spread(off, keep)
        out.append({"threshold": int(t), "n": s["n"], "rms_mm": s["rms_mm"],
                    "p90_mm": s["p90_mm"]})
    return out


def channel_stereotypy(masked: np.ndarray) -> np.ndarray:
    """Per-channel stereotypy = 1 - 2*std(normalized within-event rank) over
    participating events. 1 = perfectly reproducible position; ~0.42 = chance
    (uniform). NaN if <2 participating events."""
    masked = np.asarray(masked, dtype=float)
    n_ch = masked.shape[0]
    out = np.full(n_ch, np.nan)
    for c in range(n_ch):
        vals = masked[c][~np.isnan(masked[c])]
        if vals.size >= 2:
            out[c] = 1.0 - 2.0 * float(np.std(vals))
    return out


def channel_stereotypy_excess(
    masked: np.ndarray,
    bools: np.ndarray,
    *,
    rng: np.random.Generator,
    n_null: int = 200,
) -> np.ndarray:
    """Event-size-matched null z-score of per-channel stereotypy.

    For each channel c with participating events E_c, the null draws, per
    event e in E_c, a uniform random within-event rank position
    (integer in [0, m_e-1] normalized by m_e-1, m_e = #participants in e),
    recomputes stereotypy, repeats n_null times. z = (obs - null_mean)/null_std.
    Fewer events -> wider null -> smaller z (participation control).
    """
    masked = np.asarray(masked, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    obs = channel_stereotypy(masked)
    n_ch, n_ev = masked.shape
    ev_sizes = bools.sum(axis=0).astype(float)        # m_e per event
    z = np.full(n_ch, np.nan)
    for c in range(n_ch):
        ev_idx = np.where(~np.isnan(masked[c]))[0]
        if ev_idx.size < 2:
            continue
        m = ev_sizes[ev_idx]
        denom = np.maximum(m - 1.0, 1.0)
        null_vals = np.empty(n_null)
        for j in range(n_null):
            draws = rng.integers(0, np.maximum(m.astype(int), 1)) / denom
            null_vals[j] = 1.0 - 2.0 * float(np.std(draws))
        mu, sd = float(null_vals.mean()), float(null_vals.std())
        if sd > 1e-9:
            z[c] = (obs[c] - mu) / sd
    return z


def axis_stereotypy_profile(
    along_axis: np.ndarray,
    stereotypy_excess: np.ndarray,
    *,
    edges: Sequence[float],
) -> List[Dict[str, object]]:
    """Mean stereotypy-excess binned by along-axis coordinate."""
    a = np.asarray(along_axis, dtype=float)
    z = np.asarray(stereotypy_excess, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(z)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = ok & (a >= lo) & (a < hi)
        vals = z[sel]
        out.append({
            "a_lo": float(lo), "a_hi": float(hi), "n": int(vals.size),
            "mean_excess": float(vals.mean()) if vals.size else float("nan"),
            "sd_excess": float(vals.std()) if vals.size else float("nan"),
        })
    return out
