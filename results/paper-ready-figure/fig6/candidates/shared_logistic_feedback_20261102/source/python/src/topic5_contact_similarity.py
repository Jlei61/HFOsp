"""Topic5 contact-level similarity ladder (R1 raw / R2 same-plane kernel),
grid-free counterparts of the field maxAB. See
docs/superpowers/specs/2026-06-30-topic5-contact-similarity-ladder-design.md."""
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from src.topic5_axis_alignment import (
    within_shaft_shuffle, channel_shuffle, anchor_matched_shuffle, effective_shuffle_n,
)


def kernel_smooth_at_contacts(values, source_pts, eval_pts, support, sigma):
    """Nadaraya-Watson Gaussian smoothing identical to smooth_field, but
    evaluated at arbitrary eval_pts instead of a grid (no grid -> no pixel
    density reweighting). Mirror = pass y-flipped eval_pts.

    values, support: (n_src,); source_pts/eval_pts: (n_src,D) / (n_eval,D)
    with D=2 or D=3 (or any n-D) — distance is Euclidean over ALL coord
    columns. Returns (n_eval,) with NaN where total support <= 1e-12.

    Weight per source k at eval point i:
        w_k = support_k * exp(-(sum over coord dims (x_i - x_k)^2) / sig2)
    where sig2 = 2 * sigma^2  (byte-identical to smooth_field's sig2 for the
    2D case). T_i = sum(w_k * value_k for finite k) / sum(w_k for finite k).
    Gate: if sum(w_k for all k) <= 1e-12, output is NaN (matches smooth_field S gate).

    A source contact with a NaN coordinate is excluded entirely (never
    contributes to any weight, in any eval point). An eval point with a NaN
    coordinate returns NaN. sigma must be > 0.
    """
    v = np.asarray(values, float)
    sup = np.asarray(support, float)
    src = np.asarray(source_pts, float)
    ev = np.asarray(eval_pts, float)
    if not (sigma > 0):
        raise ValueError(f"sigma must be > 0, got {sigma}")
    # numerically identical (to atol) to smooth_field's sig2 = 2.0 * sigma_xy ** 2
    sig2 = 2.0 * float(sigma) * float(sigma)
    coord_ok = np.isfinite(src).all(axis=1)     # NaN-coord sources never contribute
    out = np.full(ev.shape[0], np.nan)
    fin = np.isfinite(v) & coord_ok
    for i in range(ev.shape[0]):
        if not np.isfinite(ev[i]).all():
            continue
        d2 = ((src[coord_ok] - ev[i]) ** 2).sum(axis=1)   # n-D Euclidean
        w = sup[coord_ok] * np.exp(-d2 / sig2)
        if w.sum() <= 1e-12:                     # support gate uses all coord_ok sources (as smooth_field S)
            continue
        fi = fin[coord_ok]
        wf = w[fi]
        if wf.sum() > 1e-12:
            out[i] = float((wf * v[coord_ok][fi]).sum() / wf.sum())
    return out


def median_nn_spacing(pts):
    """Median nearest-neighbor spacing (n-D Euclidean). Returns 0.0 when all
    points coincide (or when there are fewer than 2 points). Callers treat a
    <=0 result as NA / not-computable — this helper just returns the number.
    """
    p = np.asarray(pts, float)
    if p.shape[0] < 2:
        return 0.0
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


# ---------------------------------------------------------------------------
# Task 2: polarity-free maxAB similarity (raw + same-plane kernel)
# ---------------------------------------------------------------------------

def _pearson_over_contacts(a, b):
    """Pearson r over finite, non-degenerate contacts; NaN if < 3 or zero std."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(pearsonr(a[m], b[m])[0])


def contact_corr(rank, value, *, mode, source_pts, support, sigma, mirror=False):
    """Single-template signed Pearson between rank and value fields.

    mode="raw":    plain Pearson(rank, value), no geometry.
    mode="kernel": both fields Gaussian-smoothed at contacts before correlation.
                   mirror=True flips EVAL points' y (activation field only);
                   rank field always uses identity eval (source_pts -> source_pts).
    """
    rank = np.asarray(rank, float)
    value = np.asarray(value, float)
    if mode == "raw":
        return _pearson_over_contacts(rank, value)
    pts = np.asarray(source_pts, float)
    f_rank = kernel_smooth_at_contacts(rank, pts, pts, support, sigma)   # identity eval
    eval_pts = pts.copy()
    if mirror:
        eval_pts[:, 1] = -eval_pts[:, 1]        # flip EVAL points y only (source unchanged)
    f_val = kernel_smooth_at_contacts(value, pts, eval_pts, support, sigma)
    return _pearson_over_contacts(f_rank, f_val)


def _abs_mirror(rank, value, *, mode, source_pts, support, sigma):
    """Per-template abs-mirror score: abs(max(c_identity, c_mirror)) for kernel;
    abs(pearson) for raw. Replicates _abs_corr + corr_pair_mirror_invariant."""
    if mode == "raw":
        c = contact_corr(rank, value, mode="raw", source_pts=source_pts,
                         support=support, sigma=sigma)
        return abs(c) if np.isfinite(c) else np.nan
    c_id = contact_corr(rank, value, mode="kernel", source_pts=source_pts,
                        support=support, sigma=sigma, mirror=False)
    c_mr = contact_corr(rank, value, mode="kernel", source_pts=source_pts,
                        support=support, sigma=sigma, mirror=True)
    cand = [c for c in (c_id, c_mr) if np.isfinite(c)]
    return abs(max(cand)) if cand else np.nan   # max-by-value then abs (== _abs_corr)


def polarity_free_maxab(rank_a, rank_b, value, *, mode, source_pts, support, sigma):
    """maxAB polarity-free similarity: max(_abs_mirror(A), _abs_mirror(B)).

    If rank_b is None (single-template subject), returns _abs_mirror(A).
    Replicates window_maxab from run_topic5_axis_alignment.py.
    """
    r_a = _abs_mirror(rank_a, value, mode=mode, source_pts=source_pts,
                      support=support, sigma=sigma)
    if rank_b is None:
        return float(r_a)
    r_b = _abs_mirror(rank_b, value, mode=mode, source_pts=source_pts,
                      support=support, sigma=sigma)
    vals = [v for v in (r_a, r_b) if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan


# ---------------------------------------------------------------------------
# Task 4: sequence-sanity (Spearman + Kendall), geometry-free rank similarity
# ---------------------------------------------------------------------------

def _seq_corr(rank, value, method):
    a = np.asarray(rank, float); b = np.asarray(value, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    fn = spearmanr if method == "spearman" else kendalltau
    c = fn(a[m], b[m])[0]
    return abs(float(c)) if np.isfinite(c) else np.nan


def sequence_maxab(rank_a, rank_b, value, *, method):
    """Polarity-free max(|corr_A|, |corr_B|) using Spearman or Kendall.

    rank_b is None → single-template subject, returns |corr_A|.
    method: "spearman" or "kendall".
    """
    r_a = _seq_corr(rank_a, value, method)
    if rank_b is None:
        return r_a
    r_b = _seq_corr(rank_b, value, method)
    vals = [v for v in (r_a, r_b) if np.isfinite(v)]
    return float(max(vals)) if vals else np.nan


# ---------------------------------------------------------------------------
# Task 3: per-seizure → median-over-seizures null fold (pluggable statistic)
# ---------------------------------------------------------------------------

# Maps shuffle name → lambda(v, names, anchor, rng) with correct arg shape per shuffle type
_SHUFFLE = {
    "within_shaft":  lambda v, names, anchor, rng: within_shaft_shuffle(v, names, rng),
    "channel":       lambda v, names, anchor, rng: channel_shuffle(v, rng),
    "anchor_matched": lambda v, names, anchor, rng: anchor_matched_shuffle(v, anchor, rng),
}

# Maps subject_null shuffle name → effective_shuffle_n kind string
_SHUFFLE_KIND = {
    "within_shaft": "within_shaft",
    "channel": "channel",
    "anchor_matched": "anchor",
}

MIN_EFFECTIVE_SHUFFLE_N = 4


def fold_subject(per_sz_obs, per_sz_null):
    """Fold per-seizure observations and null draws into a subject-level result.

    obs_subject = median over seizures of the per-seizure stat.
    null distribution = per-draw median over seizures (replicates _p95_med in runner).
    null_q: p5/p50/p95/p99 of that B-length distribution.
    passed = obs_subject > null_q["p95"].
    """
    obs = np.asarray(per_sz_obs, float)
    obs_subject = float(np.nanmedian(obs))
    dist = np.nanmedian(np.asarray(per_sz_null, float), axis=0)   # [B] median-over-seizures
    q = {f"p{p}": float(np.nanpercentile(dist, p)) for p in (5, 50, 95, 99)}
    return {"obs_subject": obs_subject, "null_q": q,
            "passed": bool(obs_subject > q["p95"])}


def subject_null(stat_fn, sz_value_vectors, names, *, shuffle, B, seed, anchor_by_sz=None):
    """Per-seizure × B null loop with median fold.

    stat_fn: callable(values: ndarray) -> float — recomputes the similarity stat per draw.
    sz_value_vectors: dict {seizure_idx: ndarray of contact values}.
    names: channel name list aligned to values.
    shuffle: one of "channel", "within_shaft", "anchor_matched".
    B: number of null draws.
    seed: integer seed for reproducibility.
    anchor_by_sz: optional dict {seizure_idx: anchor array} for anchor_matched shuffle.
    """
    rng = np.random.default_rng(seed)
    shuf = _SHUFFLE[shuffle]
    per_sz_obs, per_sz_null = [], []
    for idx, vals in sz_value_vectors.items():
        anchor = None if anchor_by_sz is None else anchor_by_sz.get(idx)
        r = stat_fn(vals)
        if not np.isfinite(r):
            continue
        per_sz_obs.append(r)
        per_sz_null.append([stat_fn(shuf(vals, names, anchor, rng)) for _ in range(B)])
    if not per_sz_obs:
        return {"status": "no_resolvable_seizure"}
    # Use first seizure's anchor for effective_shuffle_n (symmetric across seizures)
    first_anchor = None if anchor_by_sz is None else anchor_by_sz.get(
        next(iter(sz_value_vectors), None))
    eff = effective_shuffle_n(names, first_anchor, _SHUFFLE_KIND[shuffle])
    out = fold_subject(per_sz_obs, per_sz_null)
    out["effective_shuffle_n"] = eff
    out["n_seizures"] = len(per_sz_obs)
    out["status"] = "INSUFFICIENT_NULL" if eff < MIN_EFFECTIVE_SHUFFLE_N else "ok"
    return out
