"""lagPatRank phantom pseudo-rank audit utilities.

Background
----------
The legacy producer (``hfo_net.py:289``) builds ``lagPatRank`` by
``argsort(argsort(per-event channel centers))`` without applying the
``eventsBool`` participation mask. Non-participating channels therefore
receive a finite integer rank drawn from spectrogram noise; the per-event
distribution of these phantom ranks is empirically U-shaped (both endpoints
biased), confirmed on chengshuai/FC10477Y (2026-05-20).

Downstream KMeans call sites in ``src/interictal_propagation.py`` apply
``np.where(np.isfinite, ranks, 0.0)`` as a NaN guard; because phantom ranks
are finite integers, this guard is a no-op. The polluted feature matrix
goes straight into KMeans, potentially driving cluster identity via
participation pattern endpoints rather than true propagation order.

This module is the audit-first fix per
``docs/superpowers/plans/2026-05-20-lagpatrank-phantom-rank-audit-and-fix.md``.

What it does NOT do
-------------------
- Does not separate "ordering signal" from "participation pattern signal".
  Midpoint imputation replaces the U-shape ordering bias with constant 0.5,
  but KMeans Euclidean can still discriminate participating vs
  non-participating channels via the constant. The audit answers "does
  endpoint rank ordering drive cluster identity?", not "is clustering
  driven by ranks or by participation?".
- Does not modify any legacy ``*_lagPat*.npz`` files. The fix is on the
  consumer side.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score


def mask_phantom_ranks(
    ranks: np.ndarray,
    bools: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Re-rank only participating channels per event; non-participating -> NaN.

    Parameters
    ----------
    ranks : (n_ch, n_ev) array
        Original ``lagPatRank`` matrix. Values for non-participating cells
        carry phantom pseudo-ranks; they are discarded.
    bools : (n_ch, n_ev) bool array
        ``eventsBool``: True where channel participates in event.
    normalize : bool
        If True, scale per-event participating ranks to [0, 1] (n_part >= 2)
        or 0.5 (n_part == 1). If False, return integer ranks in
        [0, n_part(e) - 1] with non-participating as NaN.

    Returns
    -------
    masked : (n_ch, n_ev) float array
        Non-participating cells are NaN. Events with n_participating == 0
        yield an all-NaN column (caller should drop them).
    """
    ranks = np.asarray(ranks, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    if ranks.shape != bools.shape:
        raise ValueError(
            f"ranks shape {ranks.shape} != bools shape {bools.shape}"
        )
    n_ch, n_ev = ranks.shape
    masked = np.full((n_ch, n_ev), np.nan, dtype=float)
    for e in range(n_ev):
        part_idx = np.where(bools[:, e])[0]
        n_part = part_idx.size
        if n_part == 0:
            continue
        if n_part == 1:
            masked[part_idx[0], e] = 0.5 if normalize else 0.0
            continue
        vals = ranks[part_idx, e]
        local_rank = np.argsort(np.argsort(vals, kind="stable"), kind="stable")
        if normalize:
            masked[part_idx, e] = local_rank.astype(float) / float(n_part - 1)
        else:
            masked[part_idx, e] = local_rank.astype(float)
    return masked


def build_masked_kmeans_features(
    ranks: np.ndarray,
    bools: np.ndarray,
    *,
    impute: str = "event_median",
) -> np.ndarray:
    """Build (n_ev, n_ch) KMeans feature matrix from masked normalized ranks.

    Parameters
    ----------
    ranks, bools : as in ``mask_phantom_ranks``.
    impute : str
        Strategy for filling non-participating cells in the feature matrix.
        - 'event_median' (default): fill with 0.5 (midpoint of normalized
           [0, 1] axis). Removes the U-shape endpoint bias while leaving the
           constant 0.5 as the value KMeans Euclidean sees.
        - 'channel_median': fill per-channel median normalized rank across
           events where the channel did participate. Sensitivity option.

    Returns
    -------
    X : (n_ev, n_ch) float array
        No NaNs. Ready for ``KMeans.fit_predict``. Events with
        n_participating == 0 are passed through with all values == 0.5.
    """
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    n_ch, n_ev = masked.shape
    if impute == "event_median":
        out = np.where(np.isnan(masked), 0.5, masked)
    elif impute == "channel_median":
        out = masked.copy()
        for c in range(n_ch):
            vals = masked[c, :]
            finite = vals[~np.isnan(vals)]
            fill = float(np.median(finite)) if finite.size else 0.5
            out[c, np.isnan(out[c, :])] = fill
    else:
        raise ValueError(f"unknown impute strategy: {impute}")
    return out.T  # (n_ev, n_ch)


def _seed_pairwise_ami_median(
    features: np.ndarray, k: int, *, n_seeds: int = 5
) -> Dict[str, object]:
    """KMeans on ``features`` with n_seeds random_state, pairwise median AMI."""
    labels_per_seed = []
    for seed in range(n_seeds):
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels_per_seed.append(km.fit_predict(features))
    pair_amis = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            pair_amis.append(
                adjusted_mutual_info_score(labels_per_seed[i], labels_per_seed[j])
            )
    return {
        "labels_per_seed": labels_per_seed,
        "pairwise_ami_median": float(np.median(pair_amis)) if pair_amis else float("nan"),
        "pairwise_ami_min": float(np.min(pair_amis)) if pair_amis else float("nan"),
        "pairwise_ami_max": float(np.max(pair_amis)) if pair_amis else float("nan"),
        "n_seeds": int(n_seeds),
    }


def kmeans_label_ami_audit(
    ranks: np.ndarray,
    bools: np.ndarray,
    k: int,
    *,
    n_seeds: int = 5,
    valid_event_indices: np.ndarray | None = None,
) -> Dict[str, object]:
    """Compare KMeans labels under original (phantom) vs masked feature matrices.

    Builds:
      - original features = ``ranks.T`` with non-finite -> 0.0
        (replica of ``compute_adaptive_cluster_stereotypy``'s feature prep)
      - masked features   = ``build_masked_kmeans_features(ranks, bools)``

    For each feature matrix, runs KMeans with ``n_seeds`` random_state values
    to estimate the per-feature-matrix seed-jitter AMI noise floor. Then
    computes ``AMI(original_labels@seed=0, masked_labels@seed=0)`` as the
    audit signal.

    Parameters
    ----------
    ranks, bools : (n_ch, n_ev) arrays.
    k : int
        Number of KMeans clusters (= PR-2 ``chosen_k`` for that subject).
    n_seeds : int
        Seeds for noise-floor computation. ``n_seeds * (n_seeds-1) / 2``
        pairwise AMIs are taken.
    valid_event_indices : (n_valid,) int array, optional
        Restrict to these events (typically PR-2's
        ``_valid_event_indices(bools, min_participating=3)`` output) so
        the audit operates on the same event subset as PR-2.
        If None, all events are used.

    Returns
    -------
    dict with keys:
        n_ch, n_events, phantom_fraction, k,
        ami_seed_floor_original, ami_seed_floor_masked,
        ami_audit  (= AMI(orig@seed=0, masked@seed=0)),
        ami_audit_minus_floor
            (= ami_audit - min(ami_seed_floor_original, ami_seed_floor_masked)),
        labels_original_seed0, labels_masked_seed0.
    """
    ranks = np.asarray(ranks, dtype=float)
    bools = np.asarray(bools, dtype=bool)
    if ranks.shape != bools.shape:
        raise ValueError(
            f"ranks shape {ranks.shape} != bools shape {bools.shape}"
        )
    n_ch, n_ev_total = ranks.shape

    if valid_event_indices is None:
        valid_event_indices = np.arange(n_ev_total)
    valid_event_indices = np.asarray(valid_event_indices, dtype=int)
    if valid_event_indices.size < 2 * k:
        raise ValueError(
            f"need at least 2*k={2 * k} valid events, got {valid_event_indices.size}"
        )

    ranks_v = ranks[:, valid_event_indices]
    bools_v = bools[:, valid_event_indices]
    n_ev = ranks_v.shape[1]

    # phantom_fraction = fraction of non-participating cells
    phantom_fraction = float((~bools_v).sum()) / float(bools_v.size)

    # ---- original features (PR-2 contract) ----
    original = ranks_v.T.copy()
    original = np.where(np.isfinite(original), original, 0.0)

    # ---- masked features (audit candidate) ----
    masked = build_masked_kmeans_features(ranks_v, bools_v, impute="event_median")

    floor_orig = _seed_pairwise_ami_median(original, k=k, n_seeds=n_seeds)
    floor_mask = _seed_pairwise_ami_median(masked, k=k, n_seeds=n_seeds)

    labels_orig_s0 = floor_orig["labels_per_seed"][0]
    labels_mask_s0 = floor_mask["labels_per_seed"][0]
    ami_audit = float(adjusted_mutual_info_score(labels_orig_s0, labels_mask_s0))

    floor_min = min(
        floor_orig["pairwise_ami_median"], floor_mask["pairwise_ami_median"]
    )

    return {
        "n_ch": int(n_ch),
        "n_events": int(n_ev),
        "phantom_fraction": phantom_fraction,
        "k": int(k),
        "ami_seed_floor_original": floor_orig["pairwise_ami_median"],
        "ami_seed_floor_original_min": floor_orig["pairwise_ami_min"],
        "ami_seed_floor_masked": floor_mask["pairwise_ami_median"],
        "ami_seed_floor_masked_min": floor_mask["pairwise_ami_min"],
        "ami_audit": ami_audit,
        "ami_audit_minus_floor": float(ami_audit - floor_min),
        "labels_original_seed0": labels_orig_s0,
        "labels_masked_seed0": labels_mask_s0,
        "n_seeds": int(n_seeds),
    }
