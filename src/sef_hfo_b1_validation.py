"""Shape-comparison + spatial-geometry metrics for M3 B1 validation (B1b axis, B1d
matched-shape equivalence). Pure functions on W_shape vectors + bin positions; no SNN.

W_shape vectors are the source-excluded, per-seed-normalized early-recruitment shapes
(from src.sef_hfo_mini_w_event.build_w_shape). `pos` is the matching non-source bin centers.
"""
import numpy as np

__all__ = ["shape_similarity", "weighted_centroid", "principal_axis", "axis_angle_diff",
           "top_k_overlap", "split_half_similarity", "cross_subsample_similarity"]


def shape_similarity(a, b, metric):
    """Similarity between two shape vectors over bins (cosine / pearson / spearman)."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if metric == "cosine":
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0
    if metric == "pearson":
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])
    if metric == "spearman":
        from scipy.stats import spearmanr
        return float(spearmanr(a, b).correlation)
    raise ValueError(f"unknown metric {metric!r}")


def weighted_centroid(w, pos):
    """Weighted centroid of positions `pos` (n,2) with weights `w` (n,)."""
    w = np.asarray(w, dtype=float); pos = np.asarray(pos, dtype=float)
    s = w.sum()
    return pos.mean(0) if s <= 0 else (w[:, None] * pos).sum(0) / s


def principal_axis(w, pos):
    """(principal-axis angle in deg [0,180), anisotropy = lambda1/lambda2) of the weighted
    2nd-moment (covariance) of `pos` about its weighted centroid. Isotropic -> ratio ~1."""
    w = np.asarray(w, dtype=float); pos = np.asarray(pos, dtype=float)
    s = w.sum()
    if s <= 0:
        return float("nan"), 1.0
    d = pos - weighted_centroid(w, pos)
    cov = (w[:, None, None] * np.einsum("ni,nj->nij", d, d)).sum(0) / s
    vals, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    l2, l1 = float(vals[0]), float(vals[1])
    v = vecs[:, 1]                            # principal eigenvector
    angle = float(np.degrees(np.arctan2(v[1], v[0])) % 180.0)
    aniso = (l1 / l2) if l2 > 1e-12 else float("inf")
    return angle, float(aniso)


def axis_angle_diff(a1, a2):
    """Undirected angular difference in [0,90] (axes are mod 180; 45 vs 225 -> 0)."""
    d = abs((float(a1) - float(a2)) % 180.0)
    return float(min(d, 180.0 - d))


def top_k_overlap(a, b, k):
    """Overlap fraction of the top-k bins of a and b (|topk_a ∩ topk_b| / k)."""
    a = np.asarray(a); b = np.asarray(b)
    ta = set(np.argsort(a)[::-1][:k].tolist())
    tb = set(np.argsort(b)[::-1][:k].tolist())
    return len(ta & tb) / k


def _summ(sims):
    sims = np.asarray(sims, dtype=float)
    return {"median": float(np.median(sims)), "q25": float(np.percentile(sims, 25)),
            "q75": float(np.percentile(sims, 75)), "min": float(sims.min()),
            "max": float(sims.max())}


def split_half_similarity(rows, metric="cosine", n_splits=200, rng_seed=0):
    """Distribution of similarity between the MEAN shapes of two random halves of `rows`
    (per-seed shapes). The within-substrate upper reference for B1d equivalence."""
    rows = np.asarray(rows, dtype=float)
    n = rows.shape[0]
    if n < 2:
        return {"median": float("nan"), "q25": float("nan"), "q75": float("nan"),
                "min": float("nan"), "max": float("nan"), "n": int(n)}
    rng = np.random.default_rng(rng_seed)
    h = n // 2
    sims = []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        sims.append(shape_similarity(rows[perm[:h]].mean(0), rows[perm[h:2 * h]].mean(0),
                                     metric))
    return {**_summ(sims), "n": int(n)}


def cross_subsample_similarity(A, B, metric="cosine", n_sub=200, rng_seed=0):
    """Distribution of similarity between MEAN shapes of equal-size random subsamples of A
    and B (cross-substrate). Subsample size h = min(|A|,|B|)//2. NOTE: this is matched to the
    SMALLER substrate's split_half_similarity (which uses |B|//2); within_bare (|A|//2) may
    average over one more seed, so use min(within_bare, within_core) as the equivalence floor
    (the matched, binding reference) — not within_bare alone."""
    A = np.asarray(A, dtype=float); B = np.asarray(B, dtype=float)
    rng = np.random.default_rng(rng_seed)
    h = max(1, min(A.shape[0], B.shape[0]) // 2)
    sims = []
    for _ in range(n_sub):
        ia = rng.permutation(A.shape[0])[:h]
        ib = rng.permutation(B.shape[0])[:h]
        sims.append(shape_similarity(A[ia].mean(0), B[ib].mean(0), metric))
    return {**_summ(sims), "n_sub": int(n_sub), "subsample_size": int(h)}
