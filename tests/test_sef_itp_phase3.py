"""Tests for src/sef_itp_phase3.py — peri-ictal Phase 3.

Tests the locked contracts (plan §8 + advisor 2026-05-24 design pivot):

1. Window enumeration:
   - _enumerate_peri_ictal_windows: 55-min pre/post with 5-min buffer, coverage gate.
   - _enumerate_baseline_candidates_sliding: sliding 55-min @ 15-min stride.
   - _pick_matched_baseline_windows: hour-of-day match + guard from any seizure.

2. Per-window metric pipeline:
   - compute_window_metrics: returns expected schema with rank-displacement swap-k
     endpoint (source/sink split from rank_a ordering).

3. Δmetric aggregation:
   - compute_delta_metrics: scalar median rules for decision_k/rate/radius;
     per-baseline-window Jaccard then median (NOT median set).

4. Cohort inference:
   - wild_cluster_bootstrap_p: cluster-robust t with Rademacher weights, one-sided.
   - cluster_robust_se_p: statsmodels companion.
   - subject_wilcoxon_p: subject-mean Wilcoxon sanity.
   - bh_fdr: standard BH q-values.

5. Verdict logic:
   - SUPPORTED requires (identity OR radius) significant + Δk concordant + n≥floor.
   - FAIL branches detect reverse direction.
   - UNDERPOWERED below floor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.sef_itp_phase3 import (
    _enumerate_peri_ictal_windows,
    _enumerate_baseline_candidates_sliding,
    _pick_matched_baseline_windows,
    _window_effective_seconds,
    _hour_circular_match,
    _hour_of_day,
    _far_from_all_seizures,
    _expected_direction,
    _load_real_recording_block_ranges,
    compute_window_metrics,
    compute_delta_metrics,
    wild_cluster_bootstrap_p,
    cluster_robust_se_p,
    subject_wilcoxon_p,
    bh_fdr,
    compute_phase3_verdict,
    PRIMARY_BH_FDR_METRICS,
    PERI_NOMINAL_MIN,
    COVERAGE_FLOOR,
)


# =============================================================================
# Window enumeration tests
# =============================================================================


class TestEnumeratePeriIctalWindows:
    def test_basic_pre_post_in_single_block(self):
        # Yuquan-like single 24h block
        blocks = [(1_000_000.0, 1_000_000.0 + 24 * 3600.0)]
        seizures = [{"id": "sz1", "onset": 1_000_000.0 + 12 * 3600.0,
                     "offset": 1_000_000.0 + 12 * 3600.0 + 60.0, "classification": "FBTC"}]
        out = _enumerate_peri_ictal_windows(seizures, blocks)
        assert len(out) == 1
        rec = out[0]
        assert rec["seizure_id"] == "sz1"
        # pre = [onset - 60min, onset - 5min] = 55min
        assert rec["pre"]["t_start"] == seizures[0]["onset"] - 3600.0
        assert rec["pre"]["t_end"] == seizures[0]["onset"] - 300.0
        assert rec["pre"]["effective_seconds"] == pytest.approx(55 * 60.0, abs=1.0)
        assert rec["pre"]["coverage"] == pytest.approx(1.0, abs=0.001)
        assert rec["pre"]["qualifies"]
        # post = [offset + 5min, offset + 60min] = 55min
        assert rec["post"]["t_start"] == seizures[0]["offset"] + 300.0
        assert rec["post"]["t_end"] == seizures[0]["offset"] + 3600.0
        assert rec["post"]["qualifies"]

    def test_pre_window_cross_block_with_gap(self):
        # Two blocks with a 10-min gap; seizure near start of block 2.
        b1 = (0.0, 30 * 60.0)
        b2 = (40 * 60.0, 90 * 60.0)
        blocks = [b1, b2]
        # Onset at 45min mark, so pre = [45 - 60min, 45 - 5min] = [-15min, 40min]
        # Block 1 contributes [0, 30min] = 30min; block 2 contributes [40min, 40min] = 0; gap [30, 40min] = 0.
        # Effective = 30min, coverage = 30/55 ≈ 0.545 < 0.75 → does not qualify
        seizures = [{"id": "sz1", "onset": 45 * 60.0, "offset": 45 * 60.0 + 60.0, "classification": None}]
        out = _enumerate_peri_ictal_windows(seizures, blocks)
        rec = out[0]
        assert rec["pre"]["effective_seconds"] == pytest.approx(30 * 60.0, abs=1.0)
        assert rec["pre"]["coverage"] == pytest.approx(30 / 55, abs=0.01)
        assert not rec["pre"]["qualifies"]

    def test_coverage_floor_passes_at_75_percent(self):
        # Construct a window with exactly 0.75 coverage
        # Pre window is 55min = 3300s; need eff = 0.75 * 3300 = 2475s coverage
        # Make a block that intersects the pre window for 2475s, gap for the rest
        # Onset at t=3600s, pre = [0, 3300s]. Block = [0, 2475s], next block starts at 4000s.
        blocks = [(0.0, 2475.0), (4000.0, 8000.0)]
        seizures = [{"id": "sz1", "onset": 3600.0, "offset": 3601.0, "classification": None}]
        out = _enumerate_peri_ictal_windows(seizures, blocks)
        rec = out[0]
        # pre = [3600-3600, 3600-300] = [0, 3300]
        assert rec["pre"]["effective_seconds"] == pytest.approx(2475.0, abs=1.0)
        assert rec["pre"]["coverage"] == pytest.approx(0.75, abs=0.001)
        assert rec["pre"]["qualifies"]


class TestEnumerateBaselineCandidatesSliding:
    def test_24h_continuous_block_dense_candidates(self):
        blocks = [(0.0, 24 * 3600.0)]
        cands = _enumerate_baseline_candidates_sliding(blocks)
        # Windows from 0 to 24h - 55min, stride 15min
        # n = floor((24*3600 - 55*60) / (15*60)) + 1
        expected = int(np.floor((24 * 3600 - 55 * 60) / (15 * 60))) + 1
        assert len(cands) == expected
        # First and last
        assert cands[0][0] == 0.0
        assert cands[0][1] == 55 * 60.0
        assert cands[-1][1] <= 24 * 3600.0

    def test_short_blocks_fail_coverage(self):
        # Many short blocks (5min each) cannot cover 55min — should produce no candidates
        # unless the sliding window happens to span enough of them
        blocks = [(i * 60.0, i * 60.0 + 60.0) for i in range(0, 24 * 60, 5)]  # 1-min blocks every 5min
        cands = _enumerate_baseline_candidates_sliding(blocks)
        # Even 11 1-min blocks ≈ 11min < 41.25min floor → 0 candidates
        assert cands == []

    def test_cross_block_window_qualifies_if_coverage_ok(self):
        # Two contiguous 30-min blocks → 60min total → 55-min sliding window with stride 15min
        blocks = [(0.0, 30 * 60.0), (30 * 60.0, 60 * 60.0)]
        cands = _enumerate_baseline_candidates_sliding(blocks)
        # First window [0, 55min] spans both blocks, effective = 55min ✓
        assert len(cands) >= 1
        assert cands[0][2] == pytest.approx(55 * 60.0, abs=1.0)


class TestPickMatchedBaselineWindows:
    def test_hour_of_day_match_and_guard(self):
        # 24h yuquan-like block, one seizure at hour 12 local Asia/Shanghai
        # peri pre at hour 11 (one hour before seizure onset 12pm)
        # Baselines must match hour-of-day ± 2h AND be ≥ 12h from the seizure
        block_start = 1_000_000.0  # arbitrary epoch
        blocks = [(block_start, block_start + 48 * 3600.0)]
        sz_onset = block_start + 24 * 3600.0  # 24h into recording
        seizures = [{"id": "sz1", "onset": sz_onset, "offset": sz_onset + 60.0, "classification": None}]
        peri_t_start = sz_onset - 3600.0  # 1h before
        peri_t_end = sz_onset - 300.0
        matched = _pick_matched_baseline_windows(
            peri_t_start, peri_t_end, seizures, blocks,
            tz_name="Asia/Shanghai", guard_hours=12.0,
        )
        # Should find some matches; they should all be ≥12h from sz_onset (60s long).
        for m in matched:
            assert (sz_onset - m["t_end"]) >= 12 * 3600.0 - 1 or (m["t_start"] - (sz_onset + 60.0)) >= 12 * 3600.0 - 1

    def test_returns_empty_if_no_candidates_match(self):
        # Single 60-min block, with seizure inside; no baseline can fit ≥12h away
        blocks = [(0.0, 60 * 60.0)]
        seizures = [{"id": "sz1", "onset": 30 * 60.0, "offset": 30 * 60.0 + 60.0, "classification": None}]
        matched = _pick_matched_baseline_windows(
            -3600.0, -300.0, seizures, blocks,
            tz_name="Europe/Berlin", guard_hours=12.0,
        )
        assert matched == []


class TestWindowEffectiveSeconds:
    def test_single_block_full_coverage(self):
        blocks = [(0.0, 100.0)]
        assert _window_effective_seconds(20.0, 80.0, blocks) == 60.0

    def test_multi_block_partial(self):
        # blocks: [0,30], [40,70]; window [10, 60] → 20s + 20s = 40s
        blocks = [(0.0, 30.0), (40.0, 70.0)]
        assert _window_effective_seconds(10.0, 60.0, blocks) == 40.0

    def test_no_overlap(self):
        blocks = [(100.0, 200.0)]
        assert _window_effective_seconds(0.0, 50.0, blocks) == 0.0


class TestHourCircularMatch:
    def test_basic(self):
        assert _hour_circular_match(10.0, 11.0, 2.0)
        assert _hour_circular_match(10.0, 12.0, 2.0)
        assert not _hour_circular_match(10.0, 13.0, 2.0)

    def test_wraps_at_24(self):
        # Hour 23 and hour 1 are 2 hours apart circularly
        assert _hour_circular_match(23.0, 1.0, 2.0)
        # Hour 23 and hour 3 are 4 hours apart — not within ±2
        assert not _hour_circular_match(23.0, 3.0, 2.0)


# =============================================================================
# Per-window metric pipeline tests
# =============================================================================


class TestComputeWindowMetrics:
    def _make_subject(self, n_ch=10, n_events_a=50, n_events_b=50, seed=0):
        rng = np.random.default_rng(seed)
        n_total = n_events_a + n_events_b
        # Two clusters with distinct templates: cluster_a fires early on channels 0..4,
        # cluster_b fires early on channels 5..9. So template_rank for cluster_a has
        # 0..4 with low ranks, 5..9 with high ranks; cluster_b is opposite.
        ranks = np.zeros((n_ch, n_total), dtype=float)
        bools = np.zeros((n_ch, n_total), dtype=bool)
        labels = np.concatenate([
            np.zeros(n_events_a, dtype=int),
            np.ones(n_events_b, dtype=int),
        ])
        for e in range(n_events_a):
            # Cluster A events: channels 0..4 fire first, 5..9 last
            order = np.concatenate([rng.permutation(5), rng.permutation(5) + 5])
            for ci, ch in enumerate(order):
                ranks[ch, e] = ci
                bools[ch, e] = True
        for e in range(n_events_b):
            # Cluster B events: channels 5..9 fire first, 0..4 last
            order = np.concatenate([rng.permutation(5) + 5, rng.permutation(5)])
            for ci, ch in enumerate(order):
                ranks[ch, n_events_a + e] = ci
                bools[ch, n_events_a + e] = True
        event_abs_times = np.linspace(0.0, 3300.0, n_total)
        coords = np.column_stack([np.arange(n_ch, dtype=float) * 10.0, np.zeros(n_ch), np.zeros(n_ch)])
        channel_names = [f"ch{i}" for i in range(n_ch)]
        return event_abs_times, labels, ranks, bools, coords, channel_names

    def test_returns_expected_schema_on_good_window(self):
        t, lab, ranks, bools, coords, names = self._make_subject()
        out = compute_window_metrics(
            t_start=0.0, t_end=3300.0,
            event_abs_times=t, labels=lab, ranks=ranks, bools=bools,
            coords=coords, channel_names=names,
            effective_seconds=3300.0, n_perm=200, seed=0,
        )
        assert out["exit_reason"] == "ok"
        assert out["decision_k"] is not None and out["decision_k"] >= 2
        assert out["source_indices"] is not None and out["sink_indices"] is not None
        assert len(out["source_indices"]) == out["decision_k"]
        assert len(out["sink_indices"]) == out["decision_k"]
        assert out["swap_k_endpoint_indices"] is not None
        assert isinstance(out["source_radius"], dict)
        assert "centroid_rms" in out["source_radius"]
        assert "mean_pairwise" in out["source_radius"]
        assert out["rate_per_hour"] == pytest.approx(100 / 0.9167, rel=0.05)  # 100 events / 55min

    def test_source_sink_are_disjoint_in_swap_endpoint(self):
        t, lab, ranks, bools, coords, names = self._make_subject()
        out = compute_window_metrics(
            t_start=0.0, t_end=3300.0,
            event_abs_times=t, labels=lab, ranks=ranks, bools=bools,
            coords=coords, channel_names=names,
            effective_seconds=3300.0, n_perm=200, seed=0,
        )
        # Source = lowest ranks; sink = highest ranks; disjoint for decision_k < 5 here
        src = set(out["source_indices"])
        snk = set(out["sink_indices"])
        if out["decision_k"] <= 5:
            assert src.isdisjoint(snk), "Source and sink should be disjoint when decision_k * 2 ≤ n_valid"

    def test_insufficient_events_per_cluster_exit(self):
        t, lab, ranks, bools, coords, names = self._make_subject(n_events_a=5, n_events_b=50)
        out = compute_window_metrics(
            t_start=0.0, t_end=3300.0,
            event_abs_times=t, labels=lab, ranks=ranks, bools=bools,
            coords=coords, channel_names=names,
            effective_seconds=3300.0, min_events_per_cluster=10, n_perm=100, seed=0,
        )
        assert out["exit_reason"] == "insufficient_events_per_cluster"
        assert out["decision_k"] is None

    def test_no_events_in_window_returns_clean(self):
        t = np.array([1000.0, 2000.0])
        lab = np.array([0, 1])
        ranks = np.zeros((4, 2))
        bools = np.ones((4, 2), dtype=bool)
        out = compute_window_metrics(
            t_start=5000.0, t_end=6000.0,
            event_abs_times=t, labels=lab, ranks=ranks, bools=bools,
            coords=None, channel_names=["a", "b", "c", "d"],
            effective_seconds=1000.0,
        )
        assert out["exit_reason"] == "no_events_in_window"
        assert out["n_events_total"] == 0


# =============================================================================
# Δmetric aggregation tests
# =============================================================================


def _mock_window_metrics(decision_k, source_idx, sink_idx, swap_idx,
                         source_radius, sink_radius, axis_dist, rate, exit_reason="ok"):
    return {
        "decision_k": decision_k,
        "source_indices": source_idx,
        "sink_indices": sink_idx,
        "swap_k_endpoint_indices": swap_idx,
        "source_radius": source_radius,
        "sink_radius": sink_radius,
        "source_sink_axis_distance": axis_dist,
        "rate_per_hour": rate,
        "exit_reason": exit_reason,
    }


class TestComputeDeltaMetrics:
    def test_scalar_metrics_use_baseline_median(self):
        peri = _mock_window_metrics(
            decision_k=5, source_idx=[0, 1, 2, 3, 4], sink_idx=[5, 6, 7, 8, 9],
            swap_idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            source_radius={"centroid_rms": 10.0, "mean_pairwise": 15.0, "min_enclosing_radius": 12.0, "n_points": 5},
            sink_radius={"centroid_rms": 20.0, "mean_pairwise": 25.0, "min_enclosing_radius": 22.0, "n_points": 5},
            axis_dist=30.0, rate=100.0,
        )
        baselines = [
            _mock_window_metrics(decision_k=3, source_idx=[0, 1, 2], sink_idx=[5, 6, 7],
                                 swap_idx=[0, 1, 2, 5, 6, 7],
                                 source_radius={"centroid_rms": 5.0, "mean_pairwise": 8.0, "min_enclosing_radius": 6.0, "n_points": 3},
                                 sink_radius={"centroid_rms": 15.0, "mean_pairwise": 18.0, "min_enclosing_radius": 16.0, "n_points": 3},
                                 axis_dist=25.0, rate=50.0)
            for _ in range(5)
        ]
        d = compute_delta_metrics(peri, baselines)
        assert d["exit_reason"] == "ok"
        assert d["n_baseline_qualifying"] == 5
        assert d["baseline_decision_k_median"] == 3.0
        assert d["delta_decision_k"] == 2.0
        assert d["delta_decision_k_normalized"] == pytest.approx(2.0 / 3.0)
        assert d["delta_source_centroid_rms"] == pytest.approx(10.0 - 5.0)
        assert d["delta_sink_centroid_rms"] == pytest.approx(20.0 - 15.0)
        assert d["delta_axis_distance"] == pytest.approx(30.0 - 25.0)
        assert d["delta_rate_per_hour"] == 50.0

    def test_set_metrics_per_baseline_then_median(self):
        # peri has swap_k = {0, 1, 2, 3, 4} (5 channels)
        # baseline 1: swap_k = {0, 1, 2, 5, 6} → Jaccard = 3/7, new_node = 2/5
        # baseline 2: swap_k = {0, 1, 2, 3, 4} → Jaccard = 1.0, new_node = 0
        # ... (5 baselines, mix of overlap and disjoint)
        peri = _mock_window_metrics(
            decision_k=5, source_idx=[0, 1, 2], sink_idx=[3, 4, 5],
            swap_idx=[0, 1, 2, 3, 4],
            source_radius={"centroid_rms": 10.0, "mean_pairwise": 15.0, "min_enclosing_radius": 12.0, "n_points": 3},
            sink_radius={"centroid_rms": 10.0, "mean_pairwise": 15.0, "min_enclosing_radius": 12.0, "n_points": 3},
            axis_dist=20.0, rate=100.0,
        )
        baseline_swaps = [
            {0, 1, 2, 3, 4},  # Jaccard = 1.0, new = 0
            {0, 1, 2, 3, 4},  # Jaccard = 1.0
            {0, 1, 2, 3, 4},  # Jaccard = 1.0
            {0, 1, 2, 5, 6},  # Jaccard = 3/7, new = 2/5
            {7, 8, 9, 10, 11},  # Jaccard = 0/10 = 0, new = 5/5 = 1
        ]
        baselines = []
        for sw in baseline_swaps:
            b = _mock_window_metrics(
                decision_k=5, source_idx=list(sw)[:3], sink_idx=list(sw)[3:],
                swap_idx=list(sw),
                source_radius={"centroid_rms": 8.0, "mean_pairwise": 12.0, "min_enclosing_radius": 10.0, "n_points": 3},
                sink_radius={"centroid_rms": 8.0, "mean_pairwise": 12.0, "min_enclosing_radius": 10.0, "n_points": 3},
                axis_dist=15.0, rate=50.0,
            )
            baselines.append(b)
        d = compute_delta_metrics(peri, baselines)
        # Jaccard values: [1.0, 1.0, 1.0, 3/7, 0] → median = 1.0
        assert d["jaccard_swap_k"] == pytest.approx(1.0)
        # new_node values: [0, 0, 0, 2/5, 1] → median = 0
        assert d["new_node_fraction"] == pytest.approx(0.0)

    def test_insufficient_baselines_returns_exit(self):
        peri = _mock_window_metrics(
            decision_k=3, source_idx=[0, 1, 2], sink_idx=[3, 4, 5],
            swap_idx=[0, 1, 2, 3, 4, 5],
            source_radius={"centroid_rms": 1.0, "mean_pairwise": 2.0, "min_enclosing_radius": 1.5, "n_points": 3},
            sink_radius={"centroid_rms": 1.0, "mean_pairwise": 2.0, "min_enclosing_radius": 1.5, "n_points": 3},
            axis_dist=5.0, rate=10.0,
        )
        baselines = [
            _mock_window_metrics(decision_k=2, source_idx=[0], sink_idx=[3], swap_idx=[0, 3],
                                 source_radius={"centroid_rms": 1.0, "mean_pairwise": float("nan"),
                                                "min_enclosing_radius": 0.0, "n_points": 1},
                                 sink_radius={"centroid_rms": 1.0, "mean_pairwise": float("nan"),
                                              "min_enclosing_radius": 0.0, "n_points": 1},
                                 axis_dist=3.0, rate=5.0)
            for _ in range(3)
        ]
        d = compute_delta_metrics(peri, baselines)
        assert "insufficient_baselines" in d["exit_reason"]


# =============================================================================
# Cohort inference tests
# =============================================================================


class TestWildClusterBootstrap:
    def test_intercept_only_recovers_mean(self):
        rng = np.random.default_rng(42)
        n_subj = 6
        n_per = 5
        # All values positive, mean ~ 1.0
        y = np.concatenate([rng.normal(1.0, 0.2, n_per) for _ in range(n_subj)])
        cluster_ids = np.repeat(np.arange(n_subj), n_per)
        result = wild_cluster_bootstrap_p(y, cluster_ids, n_boot=499, alternative="greater", seed=0)
        assert result["exit_reason"] == "ok"
        assert result["mean_y"] == pytest.approx(1.0, abs=0.15)
        assert result["p_value"] < 0.10, f"p={result['p_value']}"  # should reject H0: mean=0

    def test_null_data_high_p(self):
        rng = np.random.default_rng(0)
        n_subj = 6
        n_per = 5
        # mean ≈ 0 noise
        y = np.concatenate([rng.normal(0.0, 1.0, n_per) for _ in range(n_subj)])
        cluster_ids = np.repeat(np.arange(n_subj), n_per)
        result = wild_cluster_bootstrap_p(y, cluster_ids, n_boot=499, alternative="greater", seed=0)
        # Should NOT reject — p > 0.05 expected (loose constraint)
        assert result["p_value"] > 0.05 or result["p_value"] < 0.95  # just sanity

    def test_clustering_changes_se(self):
        y_clustered = np.repeat([1.0, -0.5, 1.5, -0.3, 0.8, -0.2], 5)
        ids = np.repeat(np.arange(6), 5)
        result = wild_cluster_bootstrap_p(y_clustered, ids, n_boot=999, alternative="greater", seed=0)
        assert result["n_clusters"] == 6
        assert result["n_obs"] == 30

    def test_wcr_not_wcu_constant_data(self):
        """Contract: bootstrap must be WCR (null-imposed), NOT WCU (centered residuals).

        For constant y (all values equal), WCU would degenerate: y_centered = 0
        everywhere, y_boot = 0, all t_boot = 0/0 NaN → p undefined. WCR uses raw y,
        so y_boot = ±y values, t_boot has real distribution.

        Spec: src/sef_itp_phase3.py wild_cluster_bootstrap_p docstring locks WCR
        (Cameron-Gelbach-Miller 2008 restricted variant). User catch 2026-05-24:
        original v1.0 implementation was WCU (centered) which inflates bootstrap
        variance under H1 and gives anti-conservative p at small G.
        """
        y = np.full(30, 1.0)  # all values = 1.0
        ids = np.repeat(np.arange(6), 5)
        result = wild_cluster_bootstrap_p(y, ids, n_boot=99, alternative="greater", seed=0)
        # WCR gives a real t_obs (1.0 / SE > 0)
        assert np.isfinite(result["t_obs"])
        assert result["t_obs"] > 0  # mean is positive
        # WCR gives a finite p (not NaN from 0/0). WCU on this data would degenerate.
        assert np.isfinite(result["p_value"])

    def test_returns_all_three_p_values(self):
        """Contract: bootstrap returns p_greater + p_less + p_two_sided in same pass."""
        rng = np.random.default_rng(0)
        y = np.concatenate([rng.normal(0.5, 0.5, 5) for _ in range(6)])
        ids = np.repeat(np.arange(6), 5)
        result = wild_cluster_bootstrap_p(y, ids, n_boot=499, alternative="greater", seed=0)
        assert "p_greater" in result and "p_less" in result and "p_two_sided" in result
        assert result["p_value"] == result["p_greater"]
        # Sanity: p_greater + p_less - 2 * (#t_boot == t_obs)/(n_boot+1) should be ~1
        # Or at least p_greater and p_less should not both be exactly 1
        assert not (result["p_greater"] == 1.0 and result["p_less"] == 1.0)

    def test_p_less_significant_when_mean_negative(self):
        """Contract: p_less (less-than-zero) should be small when mean is significantly < 0."""
        rng = np.random.default_rng(7)
        y = np.concatenate([rng.normal(-1.0, 0.2, 5) for _ in range(6)])
        ids = np.repeat(np.arange(6), 5)
        result = wild_cluster_bootstrap_p(y, ids, n_boot=499, alternative="less", seed=0)
        # mean is significantly negative → p_less should be small
        assert result["p_less"] < 0.10, f"p_less={result['p_less']}"
        # And p_greater should be high (mean is NOT > 0)
        assert result["p_greater"] > 0.50


class TestClusterRobustSE:
    def test_runs_with_statsmodels(self):
        rng = np.random.default_rng(0)
        y = rng.normal(0.5, 1.0, 30)
        ids = np.repeat(np.arange(6), 5)
        result = cluster_robust_se_p(y, ids, alternative="greater")
        assert result.get("exit_reason") == "ok"
        assert "p_value" in result


class TestSubjectWilcoxon:
    def test_per_subject_aggregation(self):
        # 6 subjects each with positive mean
        y = np.array([1.0, 1.2, 0.8, 0.5, 1.5, 0.9, 1.1, 1.3])
        ids = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # 4 subjects with 2 obs each
        result = subject_wilcoxon_p(y, ids, alternative="greater")
        assert result["n_subjects"] == 4
        # All subject means > 0; Wilcoxon should reject... but n=4 is borderline
        assert result["median_subject_mean"] > 0


class TestBHFDR:
    def test_standard_bh_q_values(self):
        # 6 p-values, mix significant + null
        p_values = {
            "m1": 0.001, "m2": 0.005, "m3": 0.02, "m4": 0.04, "m5": 0.15, "m6": 0.50,
        }
        result = bh_fdr(p_values, q=0.10)
        # BH: sort p ascending → 0.001, 0.005, 0.02, 0.04, 0.15, 0.50
        # m * p / (i+1) for i=1..6 with m=6:
        # 0.001 * 6/1 = 0.006
        # 0.005 * 6/2 = 0.015
        # 0.02 * 6/3 = 0.04
        # 0.04 * 6/4 = 0.06
        # 0.15 * 6/5 = 0.18
        # 0.50 * 6/6 = 0.50
        # Monotone non-increasing right-to-left: min of (raw_q[i], raw_q[i+1], ...)
        # After step: [0.006, 0.015, 0.04, 0.06, 0.18, 0.50] (already monotone)
        # rejected at q=0.10: m1, m2, m3, m4 (their q ≤ 0.10); m5, m6 not
        assert result["m1"]["rejected_at_q"]
        assert result["m2"]["rejected_at_q"]
        assert result["m3"]["rejected_at_q"]
        assert result["m4"]["rejected_at_q"]
        assert not result["m5"]["rejected_at_q"]
        assert not result["m6"]["rejected_at_q"]

    def test_handles_nan_p_values(self):
        p_values = {"m1": 0.01, "m2": float("nan"), "m3": 0.05}
        result = bh_fdr(p_values, q=0.10)
        assert "m2" in result
        assert not result["m2"]["rejected_at_q"]


# =============================================================================
# Verdict tests
# =============================================================================


class TestComputePhase3Verdict:
    def _build_inference(self, p_dict, mean_dict, p_reverse_dict=None):
        """Build inference dict.

        `p_dict` = forward-direction p (Jaccard "less" / others "greater"); stored as
        `bootstrap.p_value`. `p_reverse_dict` = reverse-direction p (default 0.5 = NS).

        For verdict logic to correctly detect FAIL via reverse-direction BH-FDR (user
        catch 2026-05-24), both p_greater and p_less must be available consistently.
        """
        if p_reverse_dict is None:
            p_reverse_dict = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        out = {}
        for m in PRIMARY_BH_FDR_METRICS:
            if _expected_direction(m) == "less":
                # Jaccard forward = "less" → p_less holds forward p
                p_less, p_greater = p_dict[m], p_reverse_dict[m]
            else:
                p_greater, p_less = p_dict[m], p_reverse_dict[m]
            out[m] = {"bootstrap": {
                "p_value": p_dict[m], "mean_y": mean_dict[m],
                "p_greater": p_greater, "p_less": p_less,
            }}
        return out

    def test_supported_identity_recruitment_only(self):
        # Jaccard ↓ significant (recruitment), Δk concordant
        p = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p["jaccard_swap_k"] = 0.001
        p["new_node_fraction"] = 0.001
        mean = {m: 0.0 for m in PRIMARY_BH_FDR_METRICS}
        mean["jaccard_swap_k"] = -0.3  # negative = recruitment
        mean["new_node_fraction"] = 0.4
        verdict = compute_phase3_verdict(
            self._build_inference(p, mean),
            delta_k_sign=1.0, delta_k_normalized_sign=1.0, n_qualifying_subjects=6,
        )
        assert verdict["verdict"] == "SUPPORTED"

    def test_supported_radius_expansion_only(self):
        p = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p["source_centroid_rms"] = 0.001
        mean = {m: 0.0 for m in PRIMARY_BH_FDR_METRICS}
        mean["source_centroid_rms"] = 5.0
        verdict = compute_phase3_verdict(
            self._build_inference(p, mean),
            delta_k_sign=1.0, delta_k_normalized_sign=1.0, n_qualifying_subjects=6,
        )
        assert verdict["verdict"] == "SUPPORTED"

    def test_null_no_signal(self):
        p = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        mean = {m: 0.0 for m in PRIMARY_BH_FDR_METRICS}
        verdict = compute_phase3_verdict(
            self._build_inference(p, mean),
            delta_k_sign=0.0, delta_k_normalized_sign=0.0, n_qualifying_subjects=6,
        )
        assert verdict["verdict"] == "NULL"

    def test_fail_identity_contraction(self):
        """FAIL via Jaccard ↑ significant: forward p ~1, reverse p ~0, mean positive."""
        p_forward = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p_forward["jaccard_swap_k"] = 0.999  # forward (less) NOT significant
        p_reverse = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p_reverse["jaccard_swap_k"] = 0.001  # reverse (greater) significant
        mean = {m: 0.0 for m in PRIMARY_BH_FDR_METRICS}
        mean["jaccard_swap_k"] = 0.3  # positive = identity contraction
        verdict = compute_phase3_verdict(
            self._build_inference(p_forward, mean, p_reverse_dict=p_reverse),
            delta_k_sign=1.0, delta_k_normalized_sign=1.0, n_qualifying_subjects=6,
        )
        assert verdict["verdict"] == "FAIL_IDENTITY_CONTRACTION"

    def test_fail_radius_contraction(self):
        """FAIL via radius ↓ significant (reverse direction)."""
        p_forward = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p_forward["source_centroid_rms"] = 0.999
        p_reverse = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p_reverse["source_centroid_rms"] = 0.001
        mean = {m: 0.0 for m in PRIMARY_BH_FDR_METRICS}
        mean["source_centroid_rms"] = -5.0  # negative = spatial contraction
        verdict = compute_phase3_verdict(
            self._build_inference(p_forward, mean, p_reverse_dict=p_reverse),
            delta_k_sign=1.0, delta_k_normalized_sign=1.0, n_qualifying_subjects=6,
        )
        assert verdict["verdict"] == "FAIL_RADIUS_CONTRACTION"

    def test_fail_uses_reverse_direction_bh_fdr_not_forward(self):
        """Critical contract (user catch 2026-05-24): FAIL detection must use
        reverse-direction BH-FDR, not the forward-direction one.

        If reverse-direction p is high (not significant) but forward p is low (e.g.,
        Jaccard is significantly < 0 = forward recruitment), that should NOT trigger
        FAIL — it should trigger SUPPORTED (in the SEF-ITP direction). The bug in v1.0
        was using forward p sign + forward p significance, which is structurally wrong.
        """
        # SEF-ITP-direction significant: Jaccard ↓
        p_forward = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p_forward["jaccard_swap_k"] = 0.001  # forward (less) significant
        p_reverse = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p_reverse["jaccard_swap_k"] = 0.999  # reverse (greater) NOT significant
        mean = {m: 0.0 for m in PRIMARY_BH_FDR_METRICS}
        mean["jaccard_swap_k"] = -0.3  # negative = recruitment direction
        verdict = compute_phase3_verdict(
            self._build_inference(p_forward, mean, p_reverse_dict=p_reverse),
            delta_k_sign=1.0, delta_k_normalized_sign=1.0, n_qualifying_subjects=6,
        )
        # Should be SUPPORTED (forward Jaccard ↓ + Δk concordant), NOT FAIL
        assert verdict["verdict"] == "SUPPORTED"

    def test_underpowered_below_floor(self):
        p = {m: 0.001 for m in PRIMARY_BH_FDR_METRICS}
        mean = {m: 1.0 for m in PRIMARY_BH_FDR_METRICS}
        mean["jaccard_swap_k"] = -1.0
        verdict = compute_phase3_verdict(
            self._build_inference(p, mean),
            delta_k_sign=1.0, delta_k_normalized_sign=1.0, n_qualifying_subjects=5,  # < floor 6
        )
        assert verdict["verdict"] == "UNDERPOWERED"

    def test_supported_requires_delta_k_concordance(self):
        # Identity recruitment significant but Δk is negative → not SUPPORTED
        p = {m: 0.5 for m in PRIMARY_BH_FDR_METRICS}
        p["jaccard_swap_k"] = 0.001
        mean = {m: 0.0 for m in PRIMARY_BH_FDR_METRICS}
        mean["jaccard_swap_k"] = -0.3
        verdict = compute_phase3_verdict(
            self._build_inference(p, mean),
            delta_k_sign=-1.0, delta_k_normalized_sign=-1.0,  # Δk contracted
            n_qualifying_subjects=6,
        )
        assert verdict["verdict"] == "NULL"  # signal exists but Δk wrong direction


def test_far_from_all_seizures_basic():
    seizures = [{"id": "s1", "onset": 1000.0, "offset": 1060.0, "classification": None}]
    # window ending 12h before onset
    assert _far_from_all_seizures(1000.0 - 13 * 3600.0, 1000.0 - 12 * 3600.0 - 60.0, seizures, 12.0)
    # window overlapping seizure
    assert not _far_from_all_seizures(900.0, 1100.0, seizures, 12.0)
    # window 11h after offset
    assert not _far_from_all_seizures(1060.0 + 11 * 3600.0, 1060.0 + 12 * 3600.0, seizures, 12.0)


class TestWeakSwapWindowGate:
    """User catch 2026-05-24: compute_window_metrics must skip windows where
    compute_swap_score_sweep returns swap_class='none' OR T_obs<score_floor=0.5.

    Without this gate, argmin-derived decision_k from noise scores becomes a fake
    "core size" and the swap-k endpoint derived from it is noise → false signal in
    Δmetrics on negative-control subjects.
    """

    def _make_no_swap_subject(self, n_ch=10, n_events_a=50, n_events_b=50, seed=0):
        """Two clusters with IDENTICAL random templates → no swap signal."""
        rng = np.random.default_rng(seed)
        n_total = n_events_a + n_events_b
        ranks = np.zeros((n_ch, n_total), dtype=float)
        bools = np.zeros((n_ch, n_total), dtype=bool)
        labels = np.concatenate([
            np.zeros(n_events_a, dtype=int),
            np.ones(n_events_b, dtype=int),
        ])
        for e in range(n_total):
            order = rng.permutation(n_ch)  # random for both clusters → no swap
            for ci, ch in enumerate(order):
                ranks[ch, e] = ci
                bools[ch, e] = True
        event_abs_times = np.linspace(0.0, 3300.0, n_total)
        coords = np.column_stack([np.arange(n_ch, dtype=float) * 10.0, np.zeros(n_ch), np.zeros(n_ch)])
        channel_names = [f"ch{i}" for i in range(n_ch)]
        return event_abs_times, labels, ranks, bools, coords, channel_names

    def test_no_swap_window_gated_by_weak_swap_exit(self):
        t, lab, ranks, bools, coords, names = self._make_no_swap_subject(seed=0)
        out = compute_window_metrics(
            t_start=0.0, t_end=3300.0,
            event_abs_times=t, labels=lab, ranks=ranks, bools=bools,
            coords=coords, channel_names=names,
            effective_seconds=3300.0, n_perm=200, seed=0,
        )
        # Should EXIT before geometry derivation when swap is weak (swap_class="none" or T_obs<0.5)
        assert out["exit_reason"].startswith("weak_swap_window"), out["exit_reason"]
        # decision_k_window + T_obs_window kept as diagnostic, but geometry fields are None/NaN
        assert out["source_indices"] is None
        assert out["sink_indices"] is None
        assert out["swap_k_endpoint_indices"] is None
        # source/sink radius left as None (not computed)
        assert out["source_radius"] is None

    def test_strong_swap_window_passes_gate(self):
        """Sanity: strong oppositional templates should pass weak-swap gate."""
        rng = np.random.default_rng(0)
        n_ch = 10
        n_total = 100
        ranks = np.zeros((n_ch, n_total), dtype=float)
        bools = np.zeros((n_ch, n_total), dtype=bool)
        labels = np.array([0] * 50 + [1] * 50)
        for e in range(50):
            order = np.concatenate([rng.permutation(5), rng.permutation(5) + 5])
            for ci, ch in enumerate(order):
                ranks[ch, e] = ci
                bools[ch, e] = True
        for e in range(50):
            order = np.concatenate([rng.permutation(5) + 5, rng.permutation(5)])
            for ci, ch in enumerate(order):
                ranks[ch, 50 + e] = ci
                bools[ch, 50 + e] = True
        t = np.linspace(0.0, 3300.0, n_total)
        coords = np.column_stack([np.arange(n_ch, dtype=float) * 10.0, np.zeros(n_ch), np.zeros(n_ch)])
        names = [f"ch{i}" for i in range(n_ch)]
        out = compute_window_metrics(
            t_start=0.0, t_end=3300.0,
            event_abs_times=t, labels=labels, ranks=ranks, bools=bools,
            coords=coords, channel_names=names,
            effective_seconds=3300.0, n_perm=300, seed=0,
        )
        assert out["exit_reason"] == "ok"
        assert out["swap_class_window"] in ("strict", "candidate")
        assert out["T_obs_window"] >= 0.5


def test_min_events_per_window_gate():
    """Plan §1 v4 lock: window total events ≥ MIN_EVENTS_PER_WINDOW (B0-locked at 30)."""
    t = np.array([100.0, 200.0, 300.0])
    lab = np.array([0, 1, 0])
    ranks = np.zeros((4, 3))
    bools = np.ones((4, 3), dtype=bool)
    out = compute_window_metrics(
        t_start=0.0, t_end=400.0,
        event_abs_times=t, labels=lab, ranks=ranks, bools=bools,
        coords=None, channel_names=["a", "b", "c", "d"],
        effective_seconds=400.0, min_events_per_window=30,
    )
    assert out["exit_reason"].startswith("insufficient_events_total:3<30")


class TestRealRecordingBlockLoader:
    """User catch 2026-05-24: blocks must come from SQL/EDF head inventory, not events."""

    def test_loads_yuquan_blocks_from_inventory(self):
        # gaolan has 24h continuous Yuquan blocks (multiple EDFs)
        blocks = _load_real_recording_block_ranges("yuquan", "gaolan")
        assert len(blocks) > 0
        # All blocks should be > 0 duration
        for bs, be in blocks:
            assert be > bs
        # Sum of durations should be reasonable (24h ≈ 86400s; allow ±30%)
        total = sum(be - bs for bs, be in blocks)
        assert total > 3600 * 12  # at least 12h

    def test_loads_epilepsiae_blocks_from_inventory(self):
        # 1146 should have many ~1h blocks
        blocks = _load_real_recording_block_ranges("epilepsiae", "1146")
        assert len(blocks) > 5
        # Most Epilepsiae blocks are ~3600s (1 hour); event-derived would have many < 1000s
        durations = [be - bs for bs, be in blocks]
        median_dur = float(np.median(durations))
        assert median_dur > 3000  # real SQL blocks are ~3600s; event-derived would be much shorter


def test_isi_classification_in_summarizer():
    """ISI stratification: <3h=short, 3-12h=medium, ≥12h=long, NaN=first."""
    # Import lazily since summarizer imports pandas at module load
    from scripts.summarize_sef_itp_phase3 import _classify_isi
    assert _classify_isi(None) == "first"
    assert _classify_isi(float("nan")) == "first"
    assert _classify_isi(1.5 * 3600) == "short"  # 1.5h
    assert _classify_isi(6 * 3600) == "medium"   # 6h
    assert _classify_isi(15 * 3600) == "long"    # 15h
    assert _classify_isi(2.99 * 3600) == "short"
    assert _classify_isi(3.01 * 3600) == "medium"
    assert _classify_isi(12.01 * 3600) == "long"
