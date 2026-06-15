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
from scipy.stats import kendalltau, spearmanr

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
    if len(source_idx) == 0 or len(sink_idx) == 0:
        raise ValueError("compute_axis_frame: empty source/sink core")
    src_c = np.nanmean(coords[list(source_idx)], axis=0)
    snk_c = np.nanmean(coords[list(sink_idx)], axis=0)
    if not (np.isfinite(src_c).all() and np.isfinite(snk_c).all()):
        raise ValueError("compute_axis_frame: non-finite source/sink centroid")
    axis = snk_c - src_c
    L = float(np.linalg.norm(axis))
    if not np.isfinite(L):
        raise ValueError("compute_axis_frame: non-finite axis length")
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
    """Exact min-enclosing-ball radius for small point sets (k<=3 exact).
    Cores here are k in {2,3}; >3 points raises (loud-failure, CLAUDE.md §6)."""
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
    raise ValueError(
        f"_meb_radius expects <=3 points (cores are k<=3), got {pts.shape[0]}")


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
    parsed = [parse_shaft(nm)[0] for nm in names]
    shafts = set(parsed) - {None}
    vals = off[part & ~np.isnan(off)]
    p90 = float(np.percentile(vals, 90)) if vals.size else 0.0
    # If too many participating names don't parse to a shaft, len(shafts)<=1
    # can't be trusted to mean "1D" — flag instead of misclassifying.
    n_part = len(names)
    unparsed_frac = (
        float(sum(s is None for s in parsed)) / n_part if n_part else 0.0)
    if unparsed_frac > 0.20:
        return {
            "geometry": "shaft_parse_uncertain",
            "n_shafts": int(len(shafts)),
            "p90_off_mm": p90,
            "measurable": False,
            "unparsed_frac": unparsed_frac,
        }
    one_d = (len(shafts) <= 1) or (p90 < spacing_mm)
    return {
        "geometry": "1D" if one_d else "distributed",
        "n_shafts": int(len(shafts)),
        "p90_off_mm": p90,
        "measurable": not one_d,
        "unparsed_frac": unparsed_frac,
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


def channel_stereotypy_components(
    masked: np.ndarray,
    bools: np.ndarray,
    *,
    rng: np.random.Generator,
    n_null: int = 200,
) -> Dict[str, np.ndarray]:
    """Per-channel stereotypy decomposition (n_ch arrays).

    Returns dict with: 'obs' (observed stereotypy), 'null_mean', 'null_std',
    'excess' (= obs - null_mean; n-INVARIANT magnitude — use this for
    descriptive profiles), 'z' (= excess/null_std; a significance that GROWS
    with event count, so do NOT use it as a cross-channel magnitude).

    For each channel c with participating events E_c, the null draws, per
    event e in E_c, a uniform random within-event rank position
    (integer in [0, m_e-1] normalized by m_e-1, m_e = #participants in e),
    recomputes stereotypy, repeats n_null times. Fewer events -> wider null,
    so z shrinks (participation control) while excess stays ~constant.

    PRECONDITION: same as channel_stereotypy_excess — `masked` must be the
    per-event normalized integer-rank grid k/(m_e-1).
    """
    masked = np.asarray(masked, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    if masked.shape != bools.shape:
        raise ValueError(
            "channel_stereotypy_components: masked/bools shape mismatch")
    if not np.array_equal(~np.isnan(masked), bools):
        raise ValueError(
            "channel_stereotypy_components: masked NaN pattern does not match "
            "bools")
    obs = channel_stereotypy(masked)
    n_ch, n_ev = masked.shape
    ev_sizes = bools.sum(axis=0).astype(float)        # m_e per event
    null_mean = np.full(n_ch, np.nan)
    null_std = np.full(n_ch, np.nan)
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
        null_mean[c] = mu
        null_std[c] = sd
        if sd > 1e-9:
            z[c] = (obs[c] - mu) / sd
    excess = obs - null_mean
    return {"obs": obs, "null_mean": null_mean, "null_std": null_std,
            "excess": excess, "z": z}


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

    NOTE: z = excess*sqrt(n)-style inflates with event count and is a
    SIGNIFICANCE, not a cross-channel magnitude. For descriptive profiles use
    channel_stereotypy_components(...)["excess"] instead.

    PRECONDITION: `masked` must be the per-event normalized integer-rank grid
    k/(m_e-1), e.g. from mask_phantom_ranks(normalize=True); off-grid
    continuous input inflates z.
    """
    return channel_stereotypy_components(
        masked, bools, rng=rng, n_null=n_null)["z"]


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


# ---------------------------------------------------------------------------
# Validation layer (reuse accepted templates + anti-tautology + radius null)
# ---------------------------------------------------------------------------


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with constant/degenerate inputs -> NaN (no warning leak)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = spearmanr(x, y).correlation
    return float(rho) if rho == rho else float("nan")


def assign_events_to_templates(
    event_ranks_masked: np.ndarray,
    template_a: np.ndarray,
    template_b: np.ndarray,
) -> np.ndarray:
    """Per-event nearest-template label (0=template_a, 1=template_b, -1=unassigned).

    Each event's participating ranks (non-NaN entries of the event column) are
    Spearman-correlated, over the channels valid in BOTH the event and a given
    template (template non-NaN), against template_a and template_b. The event
    takes the label of whichever template it correlates highest with. An event
    with fewer than 3 common channels (per-template) is unassigned (-1).

    Spearman is scale-invariant, so a dense-rank template and a normalized
    event-rank column need no rescaling. Anti-correlation with a template is, by
    construction, correlation with its reverse, so an event anti-correlated with
    template_a labels to whichever template it correlates highest with.
    """
    ev = np.asarray(event_ranks_masked, dtype=float)
    ta = np.asarray(template_a, dtype=float)
    tb = np.asarray(template_b, dtype=float)
    if ev.ndim != 2:
        raise ValueError("event_ranks_masked must be 2D (n_ch, n_ev)")
    n_ch, n_ev = ev.shape
    if ta.shape != (n_ch,) or tb.shape != (n_ch,):
        raise ValueError("template_a/template_b must be length n_ch")
    labels = np.full(n_ev, -1, dtype=int)
    a_valid = ~np.isnan(ta)
    b_valid = ~np.isnan(tb)
    for e in range(n_ev):
        col = ev[:, e]
        part = ~np.isnan(col)
        ca = part & a_valid
        cb = part & b_valid
        ra = _safe_spearman(col[ca], ta[ca]) if ca.sum() >= 3 else float("nan")
        rb = _safe_spearman(col[cb], tb[cb]) if cb.sum() >= 3 else float("nan")
        # Constant event -> Spearman NaN -> treat as -inf (never picked).
        sa = ra if ra == ra else float("-inf")
        sb = rb if rb == rb else float("-inf")
        if sa == float("-inf") and sb == float("-inf"):
            labels[e] = -1
        else:
            labels[e] = 0 if sa >= sb else 1
    return labels


def core_radius_null(
    coords: np.ndarray,
    eligible_idx: Sequence[int],
    k: int,
    observed_radius_rms: float,
    *,
    n_null: int,
    rng: np.random.Generator,
) -> Dict[str, object]:
    """Left-tail compactness p-value of an observed core RMS radius.

    Draw n_null random k-subsets from eligible_idx (channels with valid coords),
    compute each subset's RMS distance to its own centroid, and report
    p_value = fraction of null radii <= observed (left tail = 'more compact than
    random'). Returns NaN p_value when eligible coords < k+1 (no null possible).
    """
    coords = np.asarray(coords, dtype=float)
    idx = np.asarray(list(eligible_idx), dtype=int)
    # Restrict to channels with finite coords (a NaN-coord channel can't be drawn).
    idx = np.array([i for i in idx if np.isfinite(coords[i]).all()], dtype=int)
    base = {
        "p_value": float("nan"),
        "null_median_mm": float("nan"),
        "null_lo_mm": float("nan"),
        "null_hi_mm": float("nan"),
        "observed_mm": float(observed_radius_rms),
    }
    if k < 1 or idx.size <= k or n_null < 1:
        return base
    null_radii = np.empty(n_null)
    for j in range(n_null):
        sub = coords[rng.choice(idx, size=k, replace=False)]
        centroid = sub.mean(axis=0)
        null_radii[j] = float(np.sqrt(np.mean(np.sum((sub - centroid) ** 2, axis=1))))
    obs = float(observed_radius_rms)
    p = float(np.mean(null_radii <= obs)) if obs == obs else float("nan")
    base.update(
        p_value=p,
        null_median_mm=float(np.median(null_radii)),
        null_lo_mm=float(np.percentile(null_radii, 2.5)),
        null_hi_mm=float(np.percentile(null_radii, 97.5)),
    )
    return base


def deterministic_event_split(
    usable_event_indices: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """确定性把可用事件二分成两半（不依赖 rng；用于 half-axis 复搭，区别于
    split_half_axis_validation 的随机 rng.permutation split）。

    usable_event_indices: 可用事件的列索引数组（按出现顺序，可以不是 0..n-1）。
    mode == "first_second": 前一半 / 后一半（奇数个时前半 = n//2，余数进后半）。
    mode == "odd_even":     偶数位 usable[::2] / 奇数位 usable[1::2]。

    返回 (a_idx, b_idx)，二者不相交且并集 == 全集（保留原始列索引，不重新编号）。
    """
    usable = np.asarray(usable_event_indices)
    if mode == "first_second":
        half = usable.size // 2
        return usable[:half], usable[half:]
    if mode == "odd_even":
        return usable[::2], usable[1::2]
    raise ValueError(f"unknown event-split mode: {mode!r} "
                     "(expected 'first_second' or 'odd_even')")


def _half_along_axis(
    half_masked: np.ndarray,
    coords: np.ndarray,
    eligible_idx: np.ndarray,
    k: int,
) -> Optional[np.ndarray]:
    """Build cores from a half's per-channel mean rank and return the along-axis
    position of every channel. Returns None if the half can't form a valid frame
    (tier descriptive_only / coincident or interleaved centroids)."""
    n_ch = coords.shape[0]
    with np.errstate(invalid="ignore"):
        mean_rank = np.array([
            np.nanmean(half_masked[c]) if np.any(~np.isnan(half_masked[c]))
            else np.nan for c in range(n_ch)])
    eligible_mask = np.zeros(n_ch, dtype=bool)
    eligible_mask[eligible_idx] = True
    eligible_mask &= ~np.isnan(mean_rank)
    cores = build_endpoint_cores(mean_rank, eligible_mask, k_primary=k)
    if cores["tier"] == "descriptive_only" or not cores["source_idx"] or not cores["sink_idx"]:
        return None
    try:
        fr = compute_axis_frame(coords, cores["source_idx"], cores["sink_idx"])
    except ValueError:
        return None
    if fr["degenerate_axis"]:
        return None
    return np.asarray(fr["along_axis"], dtype=float)


def split_half_axis_validation(
    event_ranks_masked: np.ndarray,
    coords: np.ndarray,
    eligible_idx: Sequence[int],
    k: int,
    *,
    rng: np.random.Generator,
    n_boot: int,
) -> Dict[str, object]:
    """Anti-tautology held-out test: does the spatial axis built from one half of
    the events predict firing order on the OTHER (held-out) half?

    Split assigned events into halves A and B. From half A: per-channel mean rank
    -> source/sink cores -> axis frame -> each eligible channel's along-axis
    position. From held-out half B: per-channel mean rank. rho = Spearman(
    along_axis_position, held_out_mean_rank) over eligible channels with both
    defined. A high positive HELD-OUT rho means the spatial axis predicts firing
    order on events it was NOT built from (shared stereotyped pathway, NOT a
    tautology, because half-B ranks are independent of half-A). CI by bootstrapping
    the event split n_boot times (percentile CI).

    The point estimate (rho/p/kendall) is from one canonical seeded split; the CI
    is from the bootstrap. The correlation's n is ELIGIBLE CHANNELS, not events, so
    the single-split p is usually non-significant — read the CI, not the point.
    """
    ev = np.asarray(event_ranks_masked, dtype=float)
    coords = np.asarray(coords, dtype=float)
    eligible_idx = np.asarray(list(eligible_idx), dtype=int)
    n_ch, n_ev = ev.shape
    nan = float("nan")
    base = {
        "spearman_rho": nan, "spearman_p": nan, "kendall_tau": nan,
        "rho_ci_lo": nan, "rho_ci_hi": nan,
        "n_channels": 0, "n_events_a": 0, "n_events_b": 0,
    }
    # Only events that participate (any non-NaN) are usable.
    usable = np.where(np.any(~np.isnan(ev), axis=0))[0]
    if usable.size < 4:
        return base

    def _one_split(event_order: np.ndarray) -> Optional[Dict[str, object]]:
        half = event_order.size // 2
        a_ev, b_ev = event_order[:half], event_order[half:]
        if a_ev.size < 1 or b_ev.size < 1:
            return None
        along = _half_along_axis(ev[:, a_ev], coords, eligible_idx, k)
        if along is None:
            return None
        with np.errstate(invalid="ignore"):
            held_mean = np.array([
                np.nanmean(ev[c, b_ev]) if np.any(~np.isnan(ev[c, b_ev]))
                else np.nan for c in range(n_ch)])
        elig = np.zeros(n_ch, dtype=bool)
        elig[eligible_idx] = True
        ok = elig & ~np.isnan(along) & ~np.isnan(held_mean)
        if ok.sum() < 3:
            return None
        x = along[ok]
        y = held_mean[ok]
        rho = _safe_spearman(x, y)
        if rho != rho:
            return None
        with np.errstate(invalid="ignore"):
            sp = spearmanr(x, y)
            kt = kendalltau(x, y).correlation
        return {
            "rho": rho,
            "p": float(sp.pvalue) if sp.pvalue == sp.pvalue else nan,
            "kendall": float(kt) if kt == kt else nan,
            "n_ch": int(ok.sum()), "n_a": int(a_ev.size), "n_b": int(b_ev.size),
        }

    # Canonical point estimate from the FIRST split that forms a valid frame
    # (a single unlucky permutation that interleaves cores must not zero out the
    # whole readout when most splits are fine).
    point = None
    for _ in range(max(n_boot, 50)):
        point = _one_split(rng.permutation(usable))
        if point is not None:
            break
    if point is None:
        return base
    base.update(
        spearman_rho=point["rho"], spearman_p=point["p"],
        kendall_tau=point["kendall"], n_channels=point["n_ch"],
        n_events_a=point["n_a"], n_events_b=point["n_b"])
    # Bootstrap the event split for the CI; skip splits that can't form a frame.
    boot = []
    for _ in range(n_boot):
        res = _one_split(rng.permutation(usable))
        if res is not None:
            boot.append(res["rho"])
    if len(boot) >= 2:
        base["rho_ci_lo"] = float(np.percentile(boot, 2.5))
        base["rho_ci_hi"] = float(np.percentile(boot, 97.5))
    return base
