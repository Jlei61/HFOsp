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

from src.lagpat_rank_audit import mask_phantom_ranks


def align_template_events(masked, labels, *, flip_threshold=0.0, min_overlap=3):
    """Flip reverse-template events onto a common source->sink axis.

    masked: (n_ch, n_ev) per-event normalized propagation rank (0=source..1=sink, NaN=absent).
    labels: (n_ev,) per-event template id (from full-recording KMeans, window-independent).

    The two largest clusters' centroids are compared; if anti-correlated (reversal), the
    second cluster's events are flipped (rank -> 1-rank) so both templates' sources coincide.
    Returns (aligned, meta). No per-window decision — alignment is global.

    flip_threshold / min_overlap: defaults (0.0, 3) preserve the read-back behaviour
    (flip iff centroid corr < 0 with >=3 overlapping channels). The de novo layer passes a
    STRICTER flip_threshold (e.g. -0.2) so a direction-pure window — whose arbitrary KMeans
    split yields two genuinely positively-correlated halves — is never spuriously flipped/blurred.
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
    corr = spearmanr(c0[ok], c1[ok]).correlation if ok.sum() >= min_overlap else 0.0
    corr = float(corr) if corr == corr else 0.0
    aligned = masked.copy()
    reversed_ = corr < flip_threshold
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


def window_endpoint_overlaps(aligned, bools, full_axis, full_count, window_ev, k=2, min_ch=3):
    """SECONDARY: does a window recover the full-recording source/sink ENDPOINT SET?

    Unlike the main analysis (whole-axis ordering reproduction), this looks at the discrete
    endpoint sets. Both full and window endpoints are defined over the SAME common channels
    (finite win_axis AND full_axis) for fairness, consistent with the main common-channel fix.

      source endpoint = k channels with lowest aligned rank; sink = k highest; endpoint = union.
      rate top-k       = k channels with highest participation count.
    Returns Jaccard(window endpoint, full endpoint) for source/sink/endpoint, and the analogous
    rate top-k Jaccard. NaN if fewer than (2k+1) common channels (can't cleanly separate ends).
    """
    from src.sef_hfo_soz_localization import topk_indices, jaccard

    window_ev = np.asarray(window_ev, dtype=int)
    sub_a = np.asarray(aligned, dtype=float)[:, window_ev]
    sub_b = np.asarray(bools)[:, window_ev].astype(bool)
    full_axis = np.asarray(full_axis, dtype=float)
    full_count = np.asarray(full_count, dtype=float)
    with np.errstate(invalid="ignore"):
        win_axis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in sub_a])
    win_count = sub_b.sum(axis=1).astype(float)
    common = np.where(np.isfinite(win_axis) & np.isfinite(full_axis))[0]
    nan = {"endpoint_jaccard": float("nan"), "source_jaccard": float("nan"),
           "sink_jaccard": float("nan"), "rate_topk_jaccard": float("nan"),
           "n_common_channels": int(common.size)}
    if common.size < max(min_ch, 2 * k + 1):
        return nan
    # restrict to common channels; topk positions index into `common`
    fa, wa = full_axis[common], win_axis[common]
    fc, wc = full_count[common], win_count[common]
    f_src, f_snk = topk_indices(fa, k, largest=False), topk_indices(fa, k, largest=True)
    w_src, w_snk = topk_indices(wa, k, largest=False), topk_indices(wa, k, largest=True)
    f_rate, w_rate = topk_indices(fc, k, largest=True), topk_indices(wc, k, largest=True)
    return {"source_jaccard": jaccard(w_src, f_src),
            "sink_jaccard": jaccard(w_snk, f_snk),
            "endpoint_jaccard": jaccard(w_src | w_snk, f_src | f_snk),
            "rate_topk_jaccard": jaccard(w_rate, f_rate),
            "n_common_channels": int(common.size)}


# ---------------------------------------------------------------------------
# De novo layer (LR-7): forbid the global-label peek. Each window re-clusters and
# orients using ONLY its own events. Only the window axis changes vs read-back; rate,
# the common mask, and the null structure are identical -> global vs de novo EXCESS is
# a PAIRED contrast = the cost of forbidding the peek. Primary metric stays SIGNED.
# ---------------------------------------------------------------------------

def _fast_2means(X, n_init=3, max_iter=20, rng_seed=0):
    """Numpy-only k=2 KMeans for the per-window de novo clustering. sklearn's KMeans carries
    ~15ms Python-level overhead per call; the de novo null makes ~10^5 tiny k=2 calls, so this
    hand-rolled version (farthest-point init + Lloyd, best-of-n_init inertia) is ~100x faster.
    Faithfulness to the global sklearn clustering is NOT required: de novo discovery is a
    window-internal operation and any reasonable 2-means is a valid discovery attempt; the null
    re-runs the SAME routine, so it is self-consistent."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)
    rng = np.random.default_rng(rng_seed)
    best_labels, best_inertia = np.zeros(n, dtype=int), np.inf
    for _ in range(n_init):
        i0 = int(rng.integers(n))
        i1 = int(np.argmax(((X - X[i0]) ** 2).sum(1)))      # farthest point from a random seed
        C = np.array([X[i0], X[i1]], dtype=float)
        labels = np.zeros(n, dtype=int)
        for it in range(max_iter):
            d0 = ((X - C[0]) ** 2).sum(1)
            d1 = ((X - C[1]) ** 2).sum(1)
            new = (d1 < d0).astype(int)
            if it > 0 and np.array_equal(new, labels):
                break
            labels = new
            for c in (0, 1):
                m = labels == c
                if m.any():
                    C[c] = X[m].mean(0)
        inertia = float(np.minimum(((X - C[0]) ** 2).sum(1), ((X - C[1]) ** 2).sum(1)).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


def denovo_window_axis(window_ranks, window_bools, *, min_cluster_events=4,
                       flip_threshold=-0.2, min_overlap=3, random_state=0):
    """Window source->sink axis discovered FROM SCRATCH (no global labels).

    Re-cluster THIS window's events into k=2, self-align (larger sub-cluster anchors the
    direction; flip the smaller only on genuine anti-correlation), return per-channel mean
    normalized rank. The small-m fallback lives HERE (not in the runner) so count-matched
    null draws hit the identical path. Direction-pure windows must not blur: an arbitrary
    2-way split of one-direction events gives two positively-correlated centroids -> no flip;
    the stricter flip_threshold guards against small-window noise spuriously anti-correlating.
    """
    window_ranks = np.asarray(window_ranks, dtype=float)
    window_bools = np.asarray(window_bools) > 0
    masked = mask_phantom_ranks(window_ranks, window_bools, normalize=True)
    m = masked.shape[1]
    if m < min_cluster_events:
        with np.errstate(invalid="ignore"):
            return np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in masked])
    # event-median imputed features (== build_masked_kmeans_features) but reuse `masked` (no 2nd mask)
    features = np.where(np.isnan(masked), 0.5, masked).T
    labels = _fast_2means(features, n_init=3, rng_seed=random_state)
    aligned, _ = align_template_events(masked, labels, flip_threshold=flip_threshold,
                                       min_overlap=min_overlap)
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in aligned])


def window_recovery_paired(aligned_global, ranks, bools, full_axis, full_count, window_ev,
                           *, min_ch=3, **denovo_kwargs):
    """Score ONE window index set under BOTH axes (advisor: draw once, score both).

    Global arm = the precomputed full-recording aligned axis (read-back; reproduces the main
    result). De novo arm = the window's self-discovered axis. Rate and the common-channel mask
    are identical for both (alignment never changes the NaN pattern), so global vs de novo EXCESS
    are paired on the same events. Primary de novo metric is SIGNED; |.| reported alongside.
    """
    window_ev = np.asarray(window_ev, dtype=int)
    aligned_global = np.asarray(aligned_global, dtype=float)
    ranks = np.asarray(ranks, dtype=float)
    bools_w = np.asarray(bools)[:, window_ev] > 0
    full_axis = np.asarray(full_axis, dtype=float)
    full_count = np.asarray(full_count, dtype=float)

    sub_g = aligned_global[:, window_ev]
    with np.errstate(invalid="ignore"):
        win_axis_global = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in sub_g])
    win_axis_denovo = denovo_window_axis(ranks[:, window_ev], bools_w, **denovo_kwargs)
    win_count = bools_w.sum(axis=1).astype(float)
    # common mask from the global axis (definitionally identical to the main runner)
    common = np.isfinite(win_axis_global) & np.isfinite(full_axis)
    win_count_c = np.where(common, win_count, np.nan)
    full_count_c = np.where(common, full_count, np.nan)
    t_denovo = _spearman(win_axis_denovo, full_axis, min_ch)
    return {"template_repro_global": _spearman(win_axis_global, full_axis, min_ch),
            "template_repro_denovo_signed": t_denovo,
            "template_repro_denovo_abs": abs(t_denovo) if t_denovo == t_denovo else float("nan"),
            "rate_repro": _spearman(win_count_c, full_count_c, min_ch),
            "n_events": int(window_ev.size),
            "n_common_channels": int(common.sum())}


def count_matched_null_gap_paired(aligned_global, ranks, bools, full_axis, full_count, m, n_ev, rng,
                                  *, n_null=100, min_ch=3, **denovo_kwargs):
    """Time-scrambled null for BOTH arms on the SAME draws. Returns medians of
    (global gap, de novo SIGNED gap, de novo ABS gap), each = template_repro - rate_repro.
    The de novo arm re-clusters every random draw through the same pipeline (incl. small-m
    fallback), so it subtracts 'de novo clustering instability at M events', not just the
    estimator-smoothness floor."""
    gg, gd, gda = [], [], []
    for _ in range(n_null):
        ev = rng.choice(n_ev, size=min(m, n_ev), replace=False)
        rep = window_recovery_paired(aligned_global, ranks, bools, full_axis, full_count, ev,
                                     min_ch=min_ch, **denovo_kwargs)
        if not np.isfinite(rep["rate_repro"]):
            continue
        if np.isfinite(rep["template_repro_global"]):
            gg.append(rep["template_repro_global"] - rep["rate_repro"])
        if np.isfinite(rep["template_repro_denovo_signed"]):
            gd.append(rep["template_repro_denovo_signed"] - rep["rate_repro"])
            gda.append(rep["template_repro_denovo_abs"] - rep["rate_repro"])
    return (float(np.median(gg)) if gg else float("nan"),
            float(np.median(gd)) if gd else float("nan"),
            float(np.median(gda)) if gda else float("nan"))


def window_endpoint_union_denovo(aligned_global, ranks, bools, full_axis, full_count, window_ev,
                                 *, k=2, min_ch=3, **denovo_kwargs):
    """Polarity-FREE endpoint secondary (de novo): does the window recover the full
    source-UNION-sink endpoint SET? The union of both extremes is invariant to axis polarity,
    so de novo's sign ambiguity is irrelevant here. Compared against rate top-(2k) over the
    same common channels (matched set sizes). NaN if < max(min_ch, 2k+1) common channels."""
    from src.sef_hfo_soz_localization import topk_indices, jaccard

    window_ev = np.asarray(window_ev, dtype=int)
    ranks = np.asarray(ranks, dtype=float)
    bools_w = np.asarray(bools)[:, window_ev] > 0
    full_axis = np.asarray(full_axis, dtype=float)
    full_count = np.asarray(full_count, dtype=float)
    win_axis = denovo_window_axis(ranks[:, window_ev], bools_w, **denovo_kwargs)
    win_count = bools_w.sum(axis=1).astype(float)
    common = np.where(np.isfinite(win_axis) & np.isfinite(full_axis))[0]
    nan = {"endpoint_union_jaccard": float("nan"), "rate_topk_jaccard": float("nan"),
           "n_common_channels": int(common.size)}
    if common.size < max(min_ch, 2 * k + 1):
        return nan
    fa, wa = full_axis[common], win_axis[common]
    fc, wc = full_count[common], win_count[common]
    f_union = topk_indices(fa, k, largest=False) | topk_indices(fa, k, largest=True)
    w_union = topk_indices(wa, k, largest=False) | topk_indices(wa, k, largest=True)
    f_rate = topk_indices(fc, 2 * k, largest=True)
    w_rate = topk_indices(wc, 2 * k, largest=True)
    return {"endpoint_union_jaccard": jaccard(w_union, f_union),
            "rate_topk_jaccard": jaccard(w_rate, f_rate),
            "n_common_channels": int(common.size)}


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
