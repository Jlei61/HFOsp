"""TDD tests for src.ictal_seizure_clustering — topic5 PR-1.

Pure-function helpers for per-subject seizure clustering on Schroeder/
Panagiotopoulou pathway-dissimilarity (1 − Spearman, UPGMA).

Tests covered in this file:
- pairwise_spearman_dissim with min_overlap guard
- pair_isolated_mask
- select_k_silhouette_with_min_size
- channelwise_permutation_null gap statistic
- cluster_subject_band orchestrator
- assign_outliers_and_subtypes (D6)
- outlier_jaccard / subtype_jaccard sentinel sanity
- apply_eeg_realignment + same-subset comparison (D5)
- match_template_to_pr1_with_valid_mask (D1)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ===========================================================================
# pairwise_spearman_dissim


def test_pairwise_spearman_identical_columns_distance_zero():
    from src.ictal_seizure_clustering import pairwise_spearman_dissim
    onset = np.array([
        [-30.0, -30.0, -30.0],
        [-20.0, -20.0, -20.0],
        [-10.0, -10.0, -10.0],
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
    ])  # 5 ch × 3 sz, all columns identical
    D, mask, n_overlap = pairwise_spearman_dissim(onset, min_overlap=2)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-9)
    assert D[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert D[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert mask[0, 1] is np.True_ or mask[0, 1] == True
    assert n_overlap[0, 1] == 5


def test_pairwise_spearman_reversed_columns_distance_two():
    from src.ictal_seizure_clustering import pairwise_spearman_dissim
    onset = np.array([
        [-30.0, 10.0],
        [-20.0, 0.0],
        [-10.0, -10.0],
        [0.0, -20.0],
        [10.0, -30.0],
    ])  # column 1 is column 0 reversed → ρ = -1, distance = 2
    D, mask, _ = pairwise_spearman_dissim(onset, min_overlap=2)
    assert D[0, 1] == pytest.approx(2.0, abs=1e-9)
    assert mask[0, 1]


def test_pairwise_spearman_low_overlap_returns_nan():
    from src.ictal_seizure_clustering import pairwise_spearman_dissim
    nan = np.nan
    onset = np.array([
        [-30.0, nan],
        [-20.0, nan],
        [-10.0, -10.0],   # only 1 channel overlapping with min_overlap=5
        [nan, nan],
        [nan, nan],
    ])
    D, mask, n_overlap = pairwise_spearman_dissim(onset, min_overlap=5)
    assert np.isnan(D[0, 1])
    assert not mask[0, 1]
    assert n_overlap[0, 1] == 1


def test_pairwise_spearman_full_nan_column_yields_nan_row():
    from src.ictal_seizure_clustering import pairwise_spearman_dissim
    nan = np.nan
    onset = np.array([
        [-30.0, nan, -30.0],
        [-20.0, nan, -20.0],
        [-10.0, nan, -10.0],
        [0.0, nan, 0.0],
        [10.0, nan, 10.0],
    ])
    D, mask, _ = pairwise_spearman_dissim(onset, min_overlap=2)
    assert np.isnan(D[0, 1])
    assert np.isnan(D[1, 2])
    assert not mask[0, 1] and not mask[1, 2]
    # diag is always 0 (or NaN if column itself is all-NaN)
    # column 1 is all-NaN, so D[1, 1] should be NaN
    assert np.isnan(D[1, 1])
    assert D[0, 0] == pytest.approx(0.0)


# ===========================================================================
# pair_isolated_mask


def test_pair_isolated_mask_marks_seizure_with_majority_nan_distances():
    from src.ictal_seizure_clustering import pair_isolated_mask
    nan = np.nan
    # 4 seizures: sz3 has 3/3 NaN distances to others (isolated)
    D = np.array([
        [0.0, 0.5, 0.6, nan],
        [0.5, 0.0, 0.4, nan],
        [0.6, 0.4, 0.0, nan],
        [nan, nan, nan, 0.0],
    ])
    isolated = pair_isolated_mask(D, threshold=0.5)
    assert list(isolated) == [False, False, False, True]


def test_pair_isolated_mask_clean_dist_returns_all_false():
    from src.ictal_seizure_clustering import pair_isolated_mask
    D = np.array([
        [0.0, 0.5, 0.6],
        [0.5, 0.0, 0.4],
        [0.6, 0.4, 0.0],
    ])
    assert list(pair_isolated_mask(D, threshold=0.5)) == [False, False, False]


def test_pair_isolated_mask_handles_diagonal_nan():
    """If column has all-NaN feature, its diag is NaN — still flagged."""
    from src.ictal_seizure_clustering import pair_isolated_mask
    nan = np.nan
    D = np.array([
        [0.0, 0.5, nan],
        [0.5, 0.0, nan],
        [nan, nan, nan],
    ])
    isolated = pair_isolated_mask(D, threshold=0.5)
    assert list(isolated) == [False, False, True]


# ===========================================================================
# select_k_silhouette_with_min_size


def test_select_k_picks_largest_silhouette_with_valid_min_size():
    from src.ictal_seizure_clustering import select_k_silhouette_with_min_size
    # 4 well-separated points; expect k=2 wins
    D = np.array([
        [0.0, 0.05, 1.0, 1.05],
        [0.05, 0.0, 1.05, 1.0],
        [1.0, 1.05, 0.0, 0.05],
        [1.05, 1.0, 0.05, 0.0],
    ])
    labels_by_k = {
        2: np.array([0, 0, 1, 1]),
        3: np.array([0, 0, 1, 2]),  # cluster 1, 2 have size 1 → reject
    }
    chosen, scores = select_k_silhouette_with_min_size(
        D, labels_by_k, min_cluster_size=2, max_k=3,
    )
    assert chosen == 2
    assert 2 in scores and scores[2] > 0


def test_select_k_rejects_all_when_min_size_violated():
    from src.ictal_seizure_clustering import select_k_silhouette_with_min_size
    D = np.array([
        [0.0, 0.5, 1.0],
        [0.5, 0.0, 0.5],
        [1.0, 0.5, 0.0],
    ])
    # Only k=3 (each cluster = singleton) violates min_size=2
    labels_by_k = {
        3: np.array([0, 1, 2]),
    }
    chosen, _ = select_k_silhouette_with_min_size(
        D, labels_by_k, min_cluster_size=2, max_k=5,
    )
    assert chosen is None


def test_select_k_respects_max_k():
    from src.ictal_seizure_clustering import select_k_silhouette_with_min_size
    D = np.array([
        [0.0, 0.05, 0.9, 0.95, 1.8, 1.85],
        [0.05, 0.0, 0.95, 0.9, 1.85, 1.8],
        [0.9, 0.95, 0.0, 0.05, 0.9, 0.95],
        [0.95, 0.9, 0.05, 0.0, 0.95, 0.9],
        [1.8, 1.85, 0.9, 0.95, 0.0, 0.05],
        [1.85, 1.8, 0.95, 0.9, 0.05, 0.0],
    ])
    labels_by_k = {
        2: np.array([0, 0, 0, 0, 1, 1]),
        3: np.array([0, 0, 1, 1, 2, 2]),
    }
    # max_k=2 → reject k=3 even if better
    chosen, _ = select_k_silhouette_with_min_size(
        D, labels_by_k, min_cluster_size=2, max_k=2,
    )
    assert chosen == 2


# ===========================================================================
# channelwise_permutation_null


def test_permutation_null_clear_clusters_have_positive_gap():
    """Synthetic 2-cluster onset → permutation null should be worse."""
    from src.ictal_seizure_clustering import (
        channelwise_permutation_null,
        cluster_from_distance_upgma,
        pairwise_spearman_dissim,
    )
    rng = np.random.default_rng(0)
    n_ch = 10
    # cluster A: 5 sz with onset roughly increasing across channels
    cluster_a = np.tile(np.arange(n_ch, dtype=float)[:, None], (1, 5))
    cluster_a += rng.standard_normal((n_ch, 5)) * 0.1
    # cluster B: 5 sz with onset reversed
    cluster_b = np.tile(np.arange(n_ch, dtype=float)[::-1, None], (1, 5))
    cluster_b += rng.standard_normal((n_ch, 5)) * 0.1
    onset = np.hstack([cluster_a, cluster_b])  # 10 ch × 10 sz
    D, _, _ = pairwise_spearman_dissim(onset, min_overlap=2)
    labels = cluster_from_distance_upgma(D, k=2)
    gap = channelwise_permutation_null(
        onset, labels, B=20, rng_seed=0,
    )
    assert gap > 0.0   # observed is more compact than permuted


def test_permutation_null_random_data_gives_near_zero_gap():
    from src.ictal_seizure_clustering import (
        channelwise_permutation_null,
        cluster_from_distance_upgma,
        pairwise_spearman_dissim,
    )
    rng = np.random.default_rng(1)
    onset = rng.standard_normal((10, 10))
    D, _, _ = pairwise_spearman_dissim(onset, min_overlap=2)
    labels = cluster_from_distance_upgma(D, k=2)
    gap = channelwise_permutation_null(
        onset, labels, B=20, rng_seed=0,
    )
    assert -0.5 < gap < 0.5


# ===========================================================================
# cluster_from_distance_upgma sanity


def test_cluster_from_distance_upgma_recovers_obvious_two_cluster():
    from src.ictal_seizure_clustering import cluster_from_distance_upgma
    D = np.array([
        [0.0, 0.05, 0.05, 1.0, 1.0, 1.0],
        [0.05, 0.0, 0.05, 1.0, 1.0, 1.0],
        [0.05, 0.05, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 0.05, 0.05],
        [1.0, 1.0, 1.0, 0.05, 0.0, 0.05],
        [1.0, 1.0, 1.0, 0.05, 0.05, 0.0],
    ])
    labels = cluster_from_distance_upgma(D, k=2)
    # First 3 should share label, last 3 share another
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


# ===========================================================================
# assign_outliers_and_subtypes (D6)


def test_assign_outliers_singleton_marked_outlier_not_subtype():
    from src.ictal_seizure_clustering import assign_outliers_and_subtypes
    cluster_labels = np.array([0, 0, 0, 1, 1, 2])  # cluster 2 is singleton
    subtype_label, outlier_flag = assign_outliers_and_subtypes(
        cluster_labels, min_subtype_size=2,
    )
    # Indices: 0,1,2 → subtype 0; 3,4 → subtype 1; 5 → outlier (-1)
    assert subtype_label[5] == -1
    assert outlier_flag[5] == True
    assert subtype_label[0] == subtype_label[1] == subtype_label[2]
    assert subtype_label[3] == subtype_label[4]
    assert subtype_label[0] != subtype_label[3]
    # No outliers among the in-subtype seizures
    assert outlier_flag[:5].sum() == 0


def test_assign_outliers_no_singletons_passes_through():
    from src.ictal_seizure_clustering import assign_outliers_and_subtypes
    cluster_labels = np.array([0, 0, 1, 1])
    subtype_label, outlier_flag = assign_outliers_and_subtypes(
        cluster_labels, min_subtype_size=2,
    )
    assert outlier_flag.sum() == 0
    assert subtype_label[0] == subtype_label[1]
    assert subtype_label[2] == subtype_label[3]


def test_assign_outliers_all_singletons_all_outliers():
    from src.ictal_seizure_clustering import assign_outliers_and_subtypes
    cluster_labels = np.array([0, 1, 2, 3])
    subtype_label, outlier_flag = assign_outliers_and_subtypes(
        cluster_labels, min_subtype_size=2,
    )
    assert outlier_flag.sum() == 4
    assert (subtype_label == -1).all()


# ===========================================================================
# outlier_jaccard / subtype_jaccard


def test_outlier_jaccard_perfect_match():
    from src.ictal_seizure_clustering import outlier_jaccard
    flag = np.array([False, False, True, False, False])
    user_set = {2}
    assert outlier_jaccard(flag, user_set) == pytest.approx(1.0)


def test_outlier_jaccard_partial_match():
    from src.ictal_seizure_clustering import outlier_jaccard
    flag = np.array([False, False, True, True, False])
    user_set = {2}  # algo flagged 2 + 3, user only 2 → Jaccard = 1/2
    assert outlier_jaccard(flag, user_set) == pytest.approx(0.5)


def test_outlier_jaccard_empty_returns_one():
    """Both empty = 1.0 (degenerate but defined)."""
    from src.ictal_seizure_clustering import outlier_jaccard
    flag = np.array([False, False, False])
    assert outlier_jaccard(flag, set()) == pytest.approx(1.0)


def test_subtype_jaccard_main_subtype_match():
    from src.ictal_seizure_clustering import subtype_jaccard
    subtype = np.array([0, 0, 0, 1, 1, -1])
    # User says main pattern = {0, 1, 2, 3, 4}; algo subtype 0 = {0,1,2}, subtype 1 = {3,4}
    # Best matching subtype is largest → subtype 0; jaccard with user = |{0,1,2}∩{0,1,2,3,4}| / |union|
    user_main = {0, 1, 2, 3, 4}
    j = subtype_jaccard(subtype, user_main)
    # Picks largest subtype (0, size 3) → jaccard(set([0,1,2]), set([0,1,2,3,4])) = 3/5
    assert j == pytest.approx(0.6)


# ===========================================================================
# apply_eeg_realignment


def test_apply_eeg_realignment_value_shift():
    from src.ictal_seizure_clustering import apply_eeg_realignment
    onset = np.array([
        [-30.0, -25.0],
        [-20.0, np.nan],
    ])
    deltas = np.array([10.0, -5.0])  # eeg_clin_delta per seizure
    realigned = apply_eeg_realignment(onset, deltas)
    # cell shifts by -delta (so t=0 == eeg_onset)
    np.testing.assert_allclose(realigned[:, 0], [-40.0, -30.0])
    assert realigned[0, 1] == pytest.approx(-20.0)  # -25 - (-5) = -20
    assert np.isnan(realigned[1, 1])


def test_apply_eeg_realignment_none_delta_drops_column():
    from src.ictal_seizure_clustering import apply_eeg_realignment
    onset = np.array([
        [-30.0, -25.0, -10.0],
        [-20.0, np.nan, -5.0],
    ])
    deltas = np.array([10.0, np.nan, -5.0])
    realigned, kept_mask = apply_eeg_realignment(
        onset, deltas, return_kept_mask=True,
    )
    assert list(kept_mask) == [True, False, True]
    assert realigned.shape == (2, 2)


# ===========================================================================
# match_template_to_pr1_with_valid_mask (D1)


def test_match_template_to_pr1_uses_valid_mask():
    from src.ictal_seizure_clustering import match_template_to_pr1_with_valid_mask
    centroid = {"HL3": -50.0, "HL2": -30.0, "TBA1": -10.0, "GHOST": 0.0}
    template_rank = {"HL3": 0, "HL2": 1, "TBA1": 2, "GHOST": 3}
    valid_mask = {"HL3": True, "HL2": True, "TBA1": True, "GHOST": False}
    result = match_template_to_pr1_with_valid_mask(
        centroid, template_rank, valid_mask, min_overlap=3,
    )
    assert result["n_overlap_valid_only"] == 3
    assert result["max_rho"] == pytest.approx(1.0, abs=1e-9)


def test_match_template_to_pr1_low_overlap_returns_nan():
    from src.ictal_seizure_clustering import match_template_to_pr1_with_valid_mask
    centroid = {"HL3": -50.0, "HL2": np.nan}
    template_rank = {"HL3": 0, "HL2": 1}
    valid_mask = {"HL3": True, "HL2": True}
    result = match_template_to_pr1_with_valid_mask(
        centroid, template_rank, valid_mask, min_overlap=3,
    )
    assert result["n_overlap_valid_only"] == 1
    assert np.isnan(result["max_rho"])


# ===========================================================================
# cluster_subject_band orchestrator + cluster_subject double-band wrapper


def _synthetic_band_record(channels, onset_matrix, statuses=None):
    """Build a per_er_record dict matching v2.3 JSON shape."""
    if statuses is None:
        statuses = ["ok"] * onset_matrix.shape[1]
    sz_records = []
    for j, st in enumerate(statuses):
        co = {}
        for i, ch in enumerate(channels):
            t = onset_matrix[i, j]
            if np.isfinite(t):
                co[ch] = {"frame_idx": int((t + 300) * 10),
                          "t_onset_sec": float(t)}
            else:
                co[ch] = {"frame_idx": None, "t_onset_sec": None}
        sz_records.append({
            "seizure_idx": j,
            "seizure_id": f"sz{j:02d}",
            "status": st,
            "channel_onsets": co if st == "ok" else None,
        })
    n_ok = sum(1 for s in statuses if s == "ok")
    return {
        "n_seizures_ok": n_ok,
        "seizure_records": sz_records,
        "r_sz": {ch: float(i) for i, ch in enumerate(channels)},
    }


def test_cluster_subject_band_recovers_two_clusters():
    from src.ictal_seizure_clustering import cluster_subject_band
    rng = np.random.default_rng(0)
    channels = [f"CH{k}" for k in range(10)]
    a = np.tile(np.arange(10, dtype=float)[:, None], (1, 5)) + rng.standard_normal((10, 5)) * 0.05
    b = np.tile(np.arange(10, dtype=float)[::-1, None], (1, 5)) + rng.standard_normal((10, 5)) * 0.05
    onset = np.hstack([a, b])
    rec = _synthetic_band_record(channels, onset)
    result = cluster_subject_band(rec, channels, band="gamma_ER")
    assert result["status"] == "ok"
    assert result["chosen_k"] == 2
    assert result["n_subtypes"] == 2
    # First 5 sz should share label, last 5 share another
    s_label = result["subtype_label"]
    assert s_label[0] == s_label[1] == s_label[2] == s_label[3] == s_label[4]
    assert s_label[5] == s_label[6] == s_label[7] == s_label[8] == s_label[9]
    assert s_label[0] != s_label[5]
    # No outliers
    assert result["n_outliers"] == 0


def test_cluster_subject_band_insufficient_n_when_low_n_ok():
    from src.ictal_seizure_clustering import cluster_subject_band
    channels = ["A", "B", "C"]
    onset = np.array([
        [-30.0, -25.0, np.nan, np.nan],
        [-20.0, -22.0, np.nan, np.nan],
        [-10.0, -8.0, np.nan, np.nan],
    ])
    statuses = ["ok", "ok", "onset_unreached", "baseline_invalid"]  # only 2 ok
    rec = _synthetic_band_record(channels, onset, statuses)
    result = cluster_subject_band(rec, channels, band="gamma_ER")
    assert result["status"] == "insufficient_n"


def test_cluster_subject_band_singleton_marked_outlier():
    from src.ictal_seizure_clustering import cluster_subject_band
    channels = [f"CH{k}" for k in range(10)]
    rng = np.random.default_rng(0)
    main = np.tile(np.arange(10, dtype=float)[:, None], (1, 5)) + rng.standard_normal((10, 5)) * 0.05
    outlier = np.tile(np.arange(10, dtype=float)[::-1, None], (1, 1)) + rng.standard_normal((10, 1)) * 0.05
    onset = np.hstack([main, outlier])
    rec = _synthetic_band_record(channels, onset)
    result = cluster_subject_band(rec, channels, band="gamma_ER")
    assert result["status"] == "ok"
    # The single reversed seizure should be flagged as outlier
    assert result["outlier_flag"][5] == True
    assert result["subtype_label"][5] == -1
    # The 5 main-pattern seizures should be a single subtype
    assert (np.array(result["subtype_label"])[:5] == 0).all()


def test_cluster_subject_double_band_produces_ari():
    from src.ictal_seizure_clustering import cluster_subject
    channels = [f"CH{k}" for k in range(10)]
    rng = np.random.default_rng(0)
    a = np.tile(np.arange(10, dtype=float)[:, None], (1, 5)) + rng.standard_normal((10, 5)) * 0.05
    b = np.tile(np.arange(10, dtype=float)[::-1, None], (1, 5)) + rng.standard_normal((10, 5)) * 0.05
    onset_g = np.hstack([a, b])
    onset_b_band = np.hstack([a, b])  # identical pattern → ARI = 1
    per_subject = {
        "subject": "synthetic/test",
        "n_seizures_total": 10,
        "focal_channels": [],
        "per_er": {
            "gamma_ER": _synthetic_band_record(channels, onset_g),
            "broad_ER": _synthetic_band_record(channels, onset_b_band),
        },
    }
    result = cluster_subject(per_subject, channels=channels)
    assert result["per_band"]["gamma_ER"]["status"] == "ok"
    assert result["per_band"]["broad_ER"]["status"] == "ok"
    assert result["ari_gamma_vs_broad"] == pytest.approx(1.0, abs=1e-9)


def test_cluster_subject_disagreeing_bands_lower_ari():
    from src.ictal_seizure_clustering import cluster_subject
    channels = [f"CH{k}" for k in range(10)]
    rng = np.random.default_rng(0)
    # gamma: split [0..4] vs [5..9]; broad: split [0,1,5,6,7] vs rest
    a = np.tile(np.arange(10, dtype=float)[:, None], (1, 5)) + rng.standard_normal((10, 5)) * 0.05
    b = np.tile(np.arange(10, dtype=float)[::-1, None], (1, 5)) + rng.standard_normal((10, 5)) * 0.05
    onset_g = np.hstack([a, b])  # gamma splits at index 5
    # broad: cluster two groups but DIFFERENTLY: index [0,1,5,6,7] vs [2,3,4,8,9]
    onset_b = onset_g[:, [0, 1, 5, 6, 7, 2, 3, 4, 8, 9]]
    # remap so that index 0,1,5,6,7 share pattern, others share other
    onset_b_correct = np.full_like(onset_g, np.nan)
    pattern_a_in_broad = np.tile(np.arange(10, dtype=float)[:, None], (1, 5))
    pattern_b_in_broad = np.tile(np.arange(10, dtype=float)[::-1, None], (1, 5))
    onset_b_correct[:, [0, 1, 5, 6, 7]] = pattern_a_in_broad + rng.standard_normal((10, 5)) * 0.05
    onset_b_correct[:, [2, 3, 4, 8, 9]] = pattern_b_in_broad + rng.standard_normal((10, 5)) * 0.05
    per_subject = {
        "subject": "synthetic/test",
        "n_seizures_total": 10,
        "focal_channels": [],
        "per_er": {
            "gamma_ER": _synthetic_band_record(channels, onset_g),
            "broad_ER": _synthetic_band_record(channels, onset_b_correct),
        },
    }
    result = cluster_subject(per_subject, channels=channels)
    # The two bands disagree — ARI should be < 1
    assert result["ari_gamma_vs_broad"] < 1.0
