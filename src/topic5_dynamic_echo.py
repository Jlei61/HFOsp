"""Topic 5 Stage-2b early-ictal dynamic pattern echo (PURE math, no I/O).

Spec: docs/superpowers/specs/2026-06-11-topic5-stage2b-dynamic-pattern-echo-design.md
Plan: docs/superpowers/plans/2026-06-11-topic5-stage2b-dynamic-pattern-echo.md

align_score sign is a LOCKED contract (§2.1): >0 ALWAYS means template-early
(template_rank small = source) contacts are earlier / stronger / faster.

The max-over-time null (§2.2) is the load-bearing statistical gate: echo_peak =
max_t align_score(t) is a selection over time, so each null draw recomputes the
WHOLE curve and records its max_t — a plain point-wise channel-shuffle would be
optimistic.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# §2.1 — align_score (sign-locked)
# ---------------------------------------------------------------------------
def align_score(template_rank, value, *, kind, min_ch):
    """Sign-locked template alignment (§2.1).
    kind='intensity' (activation/dZdt/AUC/slope; larger=stronger) -> -Spearman.
    kind='latency'   (latency rank; smaller=earlier)              -> +Spearman.
    Common channels = both finite; < min_ch -> NaN."""
    t = np.asarray(template_rank, float)
    v = np.asarray(value, float)
    common = np.isfinite(t) & np.isfinite(v)
    if int(common.sum()) < min_ch:
        return float("nan")
    tc, vc = t[common], v[common]
    if tc.min() == tc.max() or vc.min() == vc.max():
        return float("nan")          # constant column -> Spearman undefined (e.g. flat frame 0)
    rho = spearmanr(tc, vc).statistic
    if not np.isfinite(rho):
        return float("nan")
    return float(-rho if kind == "intensity" else rho if kind == "latency"
                 else _bad_kind(kind))


def _bad_kind(kind):
    raise ValueError(f"kind must be 'intensity' or 'latency', got {kind!r}")


# ---------------------------------------------------------------------------
# §2 — activation + Savitzky-Golay slope
# ---------------------------------------------------------------------------
SAVGOL_WIN = 5      # 0.5 s at hop=0.1
SAVGOL_POLY = 2


def activation_and_slope(z_trace, *, hop=0.1, win=SAVGOL_WIN, poly=SAVGOL_POLY):
    """(activation_z, dZdt). activation = the robust-z itself; dZdt = Savitzky-Golay
    first derivative (locked smoothing). NaN channels pass through as NaN."""
    z = np.asarray(z_trace, dtype=np.float64)
    w = min(win if win % 2 == 1 else win + 1, z.shape[1] - (1 - z.shape[1] % 2))
    if w < poly + 2:
        dz = np.gradient(z, hop, axis=1)
    else:
        dz = savgol_filter(np.nan_to_num(z, nan=0.0), window_length=w, polyorder=poly,
                           deriv=1, delta=hop, axis=1)
        dz[~np.isfinite(z)] = np.nan
    return z, dz


# ---------------------------------------------------------------------------
# §3.1 / §2.2 — echo_curve (align_score(t) + echo_peak/echo_mean)
# ---------------------------------------------------------------------------
def echo_curve(template_rank, value_by_t, t_axis, *, kind, min_ch, mean_window):
    """align_score(t) over a shared time grid. echo_peak = max_t (max-over-time, §2.2);
    echo_mean = mean over the pre-registered confirmatory window (no time selection)."""
    V = np.asarray(value_by_t, float)
    t = np.asarray(t_axis, float)
    if V.shape[1] != t.shape[0]:
        raise ValueError(f"value_by_t time axis {V.shape[1]} != len(t_axis) {t.shape[0]}")
    curve = np.array([align_score(template_rank, V[:, j], kind=kind, min_ch=min_ch)
                      for j in range(V.shape[1])])
    finite = np.isfinite(curve)
    if not finite.any():
        return {"curve": curve, "t_axis": t, "echo_peak": float("nan"),
                "t_peak": float("nan"), "echo_mean": float("nan")}
    jpeak = int(np.nanargmax(curve))
    w = (t >= mean_window[0]) & (t <= mean_window[1]) & finite
    return {"curve": curve, "t_axis": t, "echo_peak": float(curve[jpeak]),
            "t_peak": float(t[jpeak]),
            "echo_mean": float(np.nanmean(curve[w])) if w.any() else float("nan")}


# ---------------------------------------------------------------------------
# §2.2 — echo_curve_null (max-over-time / max-over-feature null)
# ---------------------------------------------------------------------------
def _permute_channels(idx, blocks, rng):
    """Channel-identity permutation. blocks=None -> full shuffle; else within-block."""
    if blocks is None:
        return rng.permutation(idx)
    out = idx.copy()
    blocks = np.asarray(blocks)
    for b in np.unique(blocks):
        m = np.where(blocks == b)[0]
        out[m] = idx[m][rng.permutation(len(m))]
    return out


def echo_curve_null(template_rank, value_by_t, t_axis, *, kind, min_ch, null_mode,
                    blocks, B, rng):
    """Max-over-time null (§2.2): each draw permutes channel identity of `value`, recomputes
    the full align_score(t) curve, and records max_t. Returns the max-null distribution.
    null_mode 'channel' = full shuffle (blocks ignored); 'within_shaft'/'anchor_matched'
    = within-block permute using `blocks`."""
    V = np.asarray(value_by_t, float)
    n_ch, n_t = V.shape
    # Hard guard: a block-null must NOT silently degrade to a channel shuffle. Missing/
    # wrong-length blocks for within_shaft/anchor_matched would falsely pass the shaft/
    # anchor null criterion that separates path replay from a shared coarse anchor (§2.2).
    if null_mode not in {"channel", "within_shaft", "anchor_matched"}:
        raise ValueError(f"unknown null_mode={null_mode!r}")
    if null_mode != "channel":
        if blocks is None:
            raise ValueError(f"null_mode={null_mode!r} requires blocks (got None)")
        blocks = np.asarray(blocks)
        if blocks.shape[0] != n_ch:
            raise ValueError(f"blocks length {blocks.shape[0]} != n_ch {n_ch}")
    idx = np.arange(n_ch)
    out = np.full(B, np.nan)
    for b in range(B):
        # channel null = full shuffle (blocks=None); within_shaft/anchor_matched = block permute
        perm = _permute_channels(idx, None if null_mode == "channel" else blocks, rng)
        Vp = V[perm]
        curve = np.array([align_score(template_rank, Vp[:, j], kind=kind, min_ch=min_ch)
                          for j in range(n_t)])
        if np.isfinite(curve).any():
            out[b] = np.nanmax(curve)
    return out[np.isfinite(out)]


def echo_peak_pvalue(observed, max_null):
    """One-sided p of observed echo_peak vs the max-null distribution."""
    mn = np.asarray(max_null, float)
    mn = mn[np.isfinite(mn)]
    if mn.size < 2 or not np.isfinite(observed):
        return float("nan")
    return float((np.sum(mn >= observed) + 1) / (mn.size + 1))


# ---------------------------------------------------------------------------
# §3.2 — slope_latencies (eligibility-gated)
# ---------------------------------------------------------------------------
def slope_latencies(z_trace, *, t_axis, z_min, delta_min, hop=0.1):
    """Per-contact growth-slope latencies in the early-ictal window. Eligibility (§3.2):
    peak_z>=z_min AND peak_z - z(T0)>=delta_min, else all latencies NaN for that contact.
    z(T0) = the first finite sample of the (already windowed) trace."""
    z = np.asarray(z_trace, float)
    t = np.asarray(t_axis, float)
    n_ch = z.shape[0]
    _, dz = activation_and_slope(z, hop=hop)
    out = {k: np.full(n_ch, np.nan) for k in ("t_max_slope", "t50_rise", "t80_rise", "t_peak")}
    for c in range(n_ch):
        zc = z[c]
        if not np.isfinite(zc).any():
            continue
        peak = np.nanmax(zc)
        z0 = zc[np.isfinite(zc)][0]
        if not (peak >= z_min and (peak - z0) >= delta_min):
            continue
        out["t_peak"][c] = t[int(np.nanargmax(zc))]
        out["t_max_slope"][c] = t[int(np.nanargmax(dz[c]))]
        for frac, key in ((0.5, "t50_rise"), (0.8, "t80_rise")):
            thr = z0 + frac * (peak - z0)
            cross = np.where(zc >= thr)[0]
            if cross.size:
                out[key][c] = t[cross[0]]
    return out


# ---------------------------------------------------------------------------
# §3.3 — ramp_strength (per-window AUC + slope)
# ---------------------------------------------------------------------------
def ramp_strength(z_trace, *, t_axis, windows=((0, 2), (2, 5), (5, 10))):
    """Per-contact early-window AUC + linear slope of robust-z."""
    z = np.asarray(z_trace, float)
    t = np.asarray(t_axis, float)
    n_ch = z.shape[0]
    auc, slope = {}, {}
    for w in windows:
        m = (t >= w[0]) & (t < w[1])
        auc[w] = np.full(n_ch, np.nan)
        slope[w] = np.full(n_ch, np.nan)
        if m.sum() < 2:
            continue
        tw = t[m]
        for c in range(n_ch):
            zc = z[c, m]
            if np.isfinite(zc).all():
                auc[w][c] = float(np.trapz(zc, tw))
                slope[w][c] = float(np.polyfit(tw, zc, 1)[0])
    return {"AUC": auc, "slope": slope}


# ---------------------------------------------------------------------------
# §3.4 — region_aggregate
# ---------------------------------------------------------------------------
def region_aggregate(value, groups, *, min_group=2):
    """Median-aggregate a per-contact value to region level. Groups with < min_group
    contacts are dropped. Returns (region_values, region_labels)."""
    v = np.asarray(value, float)
    g = np.asarray(groups)
    labels, vals = [], []
    for lab in sorted(set(g.tolist())):
        m = g == lab
        if int(m.sum()) >= min_group and np.isfinite(v[m]).any():
            labels.append(lab)
            vals.append(float(np.nanmedian(v[m])))
    return np.array(vals, float), labels
