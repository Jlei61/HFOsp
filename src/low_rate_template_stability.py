"""Low-rate-window template stability: does the propagation source->sink AXIS reproduce
the full-recording template under low sampling better than the firing COUNT reproduces its
own full ranking?

Scale-relocated from the (null) within-SOZ analysis: universe = ALL lagPat channels (not
SOZ-restricted); the payload is the INTERACTION — in low-event windows where count ranking
gets jittery, does the template hold up better.

Reversal handling (key fix): the two stable templates are mirror images, so averaging all
events blurs the axis to ~0.5. We align by flipping the reverse-template events onto a common
source->sink axis using GLOBAL per-event labels — NOT a per-window best-template pick (that
would reintroduce the banned winner's-curse bias). Direction is preserved within the axis.

Boundary conclusion preserved: static SOZ localization / within-SOZ stability did NOT support
geometry winning (soz_localization_results_2026-06-07.md). This is a different scale question.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def align_template_events(masked, labels):
    """Flip reverse-template events onto a common source->sink axis.

    masked: (n_ch, n_ev) per-event normalized propagation rank (0=source..1=sink, NaN=absent).
    labels: (n_ev,) per-event template id (from full-recording KMeans, window-independent).

    The two largest clusters' centroids are compared; if anti-correlated (reversal), the
    second cluster's events are flipped (rank -> 1-rank) so both templates' sources coincide.
    Returns (aligned, meta). No per-window decision — alignment is global.
    """
    masked = np.asarray(masked, dtype=float)
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    if uniq.size < 2:
        return masked.copy(), {"reversed": False, "flipped_label": None, "centroid_corr": float("nan")}
    order = uniq[np.argsort(-counts)]
    l0, l1 = int(order[0]), int(order[1])
    with np.errstate(invalid="ignore"):
        c0 = np.nanmean(masked[:, labels == l0], axis=1)
        c1 = np.nanmean(masked[:, labels == l1], axis=1)
    ok = np.isfinite(c0) & np.isfinite(c1)
    corr = spearmanr(c0[ok], c1[ok]).correlation if ok.sum() >= 3 else 0.0
    corr = float(corr) if corr == corr else 0.0
    aligned = masked.copy()
    reversed_ = corr < 0
    if reversed_:
        flip = labels == l1
        aligned[:, flip] = 1.0 - aligned[:, flip]
    return aligned, {"reversed": bool(reversed_),
                     "flipped_label": l1 if reversed_ else None,
                     "centroid_corr": corr}


def _spearman(a, b, min_ch):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_ch:
        return float("nan")
    rho = spearmanr(a[ok], b[ok]).correlation
    return float(rho) if rho == rho else float("nan")


def window_reproductions(aligned, bools, full_axis, full_count, window_ev, min_ch=3):
    """Reproduction of the full-recording template axis and count ranking by one window.

    FAIR comparison (common channel mask): BOTH template_repro and rate_repro are computed over
    the SAME channels — those with a propagation rank in the window (finite win_axis) AND a
    full-recording rank. This avoids the bias where a silent channel is dropped from the template
    but counted as a tied 0 in the rate (over-penalizing rate in quiet windows).

      template_repro = signed Spearman(window aligned mean-rank, full axis)  over common channels.
      rate_repro     = Spearman(window participation count, full count)       over common channels.
      rate_repro_allch = old definition (count over ALL channels incl. silent=0) — sensitivity only.
    """
    window_ev = np.asarray(window_ev, dtype=int)
    sub_a = np.asarray(aligned, dtype=float)[:, window_ev]
    sub_b = np.asarray(bools)[:, window_ev].astype(bool)
    full_axis = np.asarray(full_axis, dtype=float)
    full_count = np.asarray(full_count, dtype=float)
    with np.errstate(invalid="ignore"):
        win_axis = np.array([np.nanmean(row) if np.any(~np.isnan(row)) else np.nan for row in sub_a])
    win_count = sub_b.sum(axis=1).astype(float)
    # common channels: a propagation rank exists this window AND a full-recording rank exists
    common = np.isfinite(win_axis) & np.isfinite(full_axis)
    win_count_common = np.where(common, win_count, np.nan)
    full_count_common = np.where(common, full_count, np.nan)
    return {"template_repro": _spearman(win_axis, full_axis, min_ch),
            "rate_repro": _spearman(win_count_common, full_count_common, min_ch),
            "rate_repro_allch": _spearman(win_count, full_count, min_ch),
            "n_events": int(window_ev.size), "n_common_channels": int(common.sum())}


def count_matched_null_gap(aligned, bools, full_axis, full_count, m, n_ev, rng, n_null=100, min_ch=3):
    """Time-scrambled null: median(template_repro - rate_repro) over n_null random draws of
    m events from the WHOLE recording. This is the 'rank is just a smoother estimator than a
    small-N count' floor; the observed contiguous-window gap MINUS this null = time-structured
    drift beyond sampling. Returns NaN if no draw yields a computable pair."""
    gaps = []
    for _ in range(n_null):
        ev = rng.choice(n_ev, size=min(m, n_ev), replace=False)
        rep = window_reproductions(aligned, bools, full_axis, full_count, ev, min_ch=min_ch)
        if np.isfinite(rep["template_repro"]) and np.isfinite(rep["rate_repro"]):
            gaps.append(rep["template_repro"] - rep["rate_repro"])
    return float(np.median(gaps)) if gaps else float("nan")


def m_bucket(m):
    """Event-count bucket for M-graded reporting (the effect is not uniform across M)."""
    if m <= 2:
        return "<=2(unresolvable)"
    if m <= 4:
        return "3-4"
    if m <= 20:
        return "5-20"
    if m <= 100:
        return "21-100"
    return ">100"


def stratify_by_event_count(counts, n_strata=3):
    """Label each window low/mid/high by event count tertiles (within-subject)."""
    counts = np.asarray(counts, dtype=float)
    if counts.size == 0:
        return []
    if n_strata == 3:
        lo, hi = np.quantile(counts, [1 / 3, 2 / 3])
        return ["low" if c <= lo else ("high" if c > hi else "mid") for c in counts]
    edges = np.quantile(counts, np.linspace(0, 1, n_strata + 1)[1:-1])
    labels = np.digitize(counts, edges)
    return [int(x) for x in labels]


def time_windows(event_times, window_seconds):
    """Partition events into fixed-duration windows; returns list of event-index arrays
    (time-sorted). Windows are aligned to the first event time."""
    t = np.asarray(event_times, dtype=float)
    finite = np.isfinite(t)
    idx = np.where(finite)[0]
    if idx.size == 0:
        return []
    t0 = t[idx].min()
    bin_id = np.floor((t[idx] - t0) / float(window_seconds)).astype(int)
    out = []
    for b in np.unique(bin_id):
        out.append(idx[bin_id == b])
    return out
