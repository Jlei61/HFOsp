"""Per-subject PNG renderer helpers for topic5 PR-1 z-ER seizure clustering.

Pure-data helpers (no matplotlib): MDS embedding, subtype color palette,
subtype-sorted seizure ordering. The plotting orchestrator lives in
``scripts/cluster_ictal_seizures.py``.

Layout per band (per user 2026-05-10 spec):
  - Left column: dendrogram (top) + sorted pairwise (1−Spearman) heatmap (bottom)
  - Right column: MDS 2D scatter (top) + cluster-stratified t_ER matrix (bottom)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def subtype_sort_indices(
    subtype_label: Sequence[int],
    outlier_flag: Sequence[bool],
) -> List[int]:
    """Return seizure indices reordered by subtype label, outliers last.

    Within each subtype, original order is preserved.
    """
    sl = list(subtype_label)
    of = list(outlier_flag)
    n = len(sl)
    if len(of) != n:
        raise ValueError("subtype_label and outlier_flag length mismatch")
    pairs = []
    for i in range(n):
        # outliers go to bucket "+inf"; subtypes sorted by label
        if of[i] or sl[i] < 0:
            bucket = float("inf")
        else:
            bucket = float(sl[i])
        pairs.append((bucket, i))
    pairs.sort(key=lambda p: (p[0], p[1]))
    return [i for _, i in pairs]


def compute_mds_2d(
    D: np.ndarray, *, random_state: int = 0,
) -> np.ndarray:
    """Compute 2D MDS embedding from precomputed dissimilarity matrix.

    Returns array of shape (n, 2). Uses sklearn MDS with precomputed
    dissimilarities and a fixed random_state for reproducibility.
    """
    from sklearn.manifold import MDS
    D_arr = np.asarray(D, dtype=np.float64)
    n = D_arr.shape[0]
    if D_arr.shape != (n, n):
        raise ValueError(f"D must be square, got {D_arr.shape}")
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)
    # Symmetrize (in case of float roundoff) and zero-diagonal
    D_arr = 0.5 * (D_arr + D_arr.T)
    np.fill_diagonal(D_arr, 0.0)
    mds = MDS(
        n_components=2,
        dissimilarity="precomputed",
        random_state=random_state,
        n_init=4,
        normalized_stress="auto",
    )
    return mds.fit_transform(D_arr)


def subtype_color_palette(
    n_subtypes: int,
    *,
    outlier_color: str = "#7f8c8d",
) -> Tuple[List[str], str]:
    """Return (subtype_colors, outlier_color).

    Uses tab10 / tab20 colormap depending on n_subtypes.
    """
    if n_subtypes <= 10:
        from matplotlib import colormaps
        cmap = colormaps["tab10"]
        cols = [
            "#" + "".join(f"{int(round(255*c)):02x}" for c in cmap(i)[:3])
            for i in range(n_subtypes)
        ]
    else:
        from matplotlib import colormaps
        cmap = colormaps["tab20"]
        cols = [
            "#" + "".join(f"{int(round(255*c)):02x}" for c in cmap(i)[:3])
            for i in range(n_subtypes)
        ]
    return cols, outlier_color


def channel_sort_by_subtype_means(
    onset_kept: np.ndarray,
    subtype_label: Sequence[int],
) -> List[int]:
    """Sort channels by their mean t_onset across the largest subtype.

    Falls back to overall mean if the largest subtype has all-NaN row for
    some channels.
    """
    sl = np.asarray(subtype_label, dtype=int)
    n_ch = onset_kept.shape[0]
    # Pick largest non-outlier subtype
    valid_subs = [int(s) for s in set(sl) if s >= 0]
    if not valid_subs:
        means = np.nanmean(onset_kept, axis=1)
    else:
        from collections import Counter
        ctr = Counter([int(s) for s in sl if s >= 0])
        biggest = ctr.most_common(1)[0][0]
        sub_idx = np.where(sl == biggest)[0]
        sub = onset_kept[:, sub_idx]
        means = np.nanmean(sub, axis=1)
        # fill NaN means from overall mean
        overall = np.nanmean(onset_kept, axis=1)
        for i in range(n_ch):
            if not np.isfinite(means[i]):
                means[i] = overall[i] if np.isfinite(overall[i]) else float("inf")
    order = np.argsort(means)
    return order.tolist()
