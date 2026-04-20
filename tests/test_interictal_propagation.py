from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.interictal_propagation import (
    _center_rank_matrix,
    _compute_pr5b_sensitivity_gate,
    _valid_event_indices,
    assign_events_to_templates,
    build_cluster_templates,
    compute_adaptive_cluster_stereotypy,
    compute_continuous_template_dynamics,
    compute_novel_template_gate,
    compute_rate_state_coupling,
    compute_seizure_proximity_coupling,
    compute_source_node_diagnostic,
    compute_stereotypy_by_nparticipating,
    compute_template_recruitment_shift,
    compute_temporal_cluster_dynamics,
    compute_time_split_reproducibility,
    detect_propagation_mixture,
    load_subject_propagation_events,
    run_subject_interictal_propagation_pr1,
    summarize_pr5_novel_template_gate,
    summarize_pr5_template_recruitment_shift,
    summarize_propagation_cohort,
    validate_absolute_lag_clustering,
)


def test_center_rank_matrix_ignores_low_participation_channel() -> None:
    ranks = np.array(
        [
            [1, 1, 1, 1],
            [2, 2, 2, 2],
            [0, 5, 0, 0],
        ],
        dtype=float,
    )
    bools = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 0],
        ],
        dtype=bool,
    )

    out = _center_rank_matrix(ranks, bools, min_participation=2)
    assert out["valid_center_mask"].tolist() == [True, True, False]
    assert out["mean_rank_per_channel"][2] == 0.0
    assert out["centered_ranks"][2, 1] == 5.0


def test_detect_propagation_mixture_flags_bimodal_patterns() -> None:
    pattern_a = np.tile(np.array([[1], [2], [3], [4]], dtype=float), (1, 20))
    pattern_b = np.tile(np.array([[4], [3], [2], [1]], dtype=float), (1, 20))
    ranks = np.concatenate([pattern_a, pattern_b], axis=1)
    bools = np.ones_like(ranks, dtype=bool)

    out = detect_propagation_mixture(ranks, bools, n_sample=40)
    assert out["n_pairs_valid"] > 100
    assert out["is_mixture"] or out["possible_mixture"]


def test_stereotypy_by_nparticipating_reports_bins() -> None:
    ranks = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3],
            [0, 0, 0, 4, 4, 4],
            [0, 0, 0, 5, 5, 5],
        ],
        dtype=float,
    )
    bools = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=bool,
    )

    out = compute_stereotypy_by_nparticipating(
        ranks,
        bools,
        bins=[(3, 3), (5, 5)],
        n_sample=10,
        n_seeds=3,
    )
    assert len(out) == 2
    assert out[0]["n_events"] == 3
    assert out[1]["n_events"] == 3
    assert np.isfinite(out[0]["mean_tau"])
    assert np.isfinite(out[1]["mean_tau"])


def test_source_diagnostic_can_flag_soz_source_erasure() -> None:
    ranks = np.array(
        [
            [1, 1, 1, 1, 1, 1, 3, 4, 2, 5, 2, 4],
            [2, 2, 2, 2, 2, 2, 5, 3, 3, 4, 5, 2],
            [3, 3, 3, 3, 3, 3, 1, 2, 5, 3, 1, 1],
            [4, 4, 4, 4, 4, 4, 4, 1, 1, 2, 3, 5],
            [5, 5, 5, 5, 5, 5, 2, 5, 4, 1, 4, 3],
        ],
        dtype=float,
    )
    bools = np.ones_like(ranks, dtype=bool)

    out = compute_source_node_diagnostic(
        ranks,
        bools,
        channel_names=["A", "B", "C", "D", "E"],
        soz_channels=["A"],
        min_participation=2,
    )
    assert "A" in out["raw_top3_sources"]
    assert out["soz_source_erased"] is True


def test_run_subject_interictal_propagation_pr1_smoke(tmp_path: Path) -> None:
    subject_dir = tmp_path / "548" / "all_recs"
    subject_dir.mkdir(parents=True)

    ranks = np.array(
        [
            [1, 1, 1, 4, 4, 4],
            [2, 2, 2, 1, 1, 1],
            [3, 3, 3, 2, 2, 2],
            [4, 4, 4, 3, 3, 3],
        ],
        dtype=float,
    )
    bools = np.ones_like(ranks, dtype=int)
    np.savez_compressed(
        subject_dir / "block_lagPat.npz",
        lagPatRank=ranks,
        eventsBool=bools,
        chnNames=np.array(["A", "B", "C", "D"], dtype=object),
    )

    out = run_subject_interictal_propagation_pr1(
        subject_dir=subject_dir,
        dataset="epilepsiae",
        subject="548",
        soz_channels=["A"],
        n_sample=6,
        n_seeds=3,
        min_center_participation=2,
    )
    assert out["dataset"] == "epilepsiae"
    assert out["subject"] == "548"
    assert out["propagation_stereotypy"]["all"]["n_events_available"] == 6
    assert "mixture" in out
    assert "centered_rank" in out


def test_load_subject_propagation_events_sorts_by_start_t_and_rebuilds_times(tmp_path: Path) -> None:
    subject_dir = tmp_path / "chengshuai"
    subject_dir.mkdir(parents=True)

    ranks_a = np.array([[1, 2], [2, 1]], dtype=float)
    bools_a = np.ones_like(ranks_a, dtype=int)
    np.savez_compressed(
        subject_dir / "blockB_lagPat.npz",
        lagPatRank=ranks_a,
        lagPatRaw=np.array([[10.0, 13.0], [11.0, 12.0]], dtype=float),
        eventsBool=bools_a,
        chnNames=np.array(["A", "B"], dtype=object),
        start_t=np.array(200.0),
    )
    np.save(subject_dir / "blockB_packedTimes.npy", np.array([[1.0, 1.5], [3.0, 3.5]], dtype=float))

    ranks_b = np.array([[3], [4]], dtype=float)
    bools_b = np.ones_like(ranks_b, dtype=int)
    np.savez_compressed(
        subject_dir / "blockA_lagPat.npz",
        lagPatRank=ranks_b,
        lagPatRaw=np.array([[5.0], [6.0]], dtype=float),
        eventsBool=bools_b,
        chnNames=np.array(["A", "B"], dtype=object),
        start_t=np.array(100.0),
    )
    np.save(subject_dir / "blockA_packedTimes.npy", np.array([[2.0, 2.5]], dtype=float))

    loaded = load_subject_propagation_events(subject_dir)
    assert loaded["n_blocks_used"] == 2
    assert loaded["record_names"] == ["blockA", "blockB"]
    assert loaded["block_ids"].tolist() == [0, 1, 1]
    assert loaded["event_rel_times"].tolist() == [2.0, 1.0, 3.0]
    assert loaded["event_abs_times"].tolist() == [102.0, 201.0, 203.0]
    np.testing.assert_allclose(
        loaded["lag_raw"],
        np.array([[5.0, 10.0, 13.0], [6.0, 11.0, 12.0]], dtype=float),
    )
    assert loaded["block_time_ranges"] == [(102.0, 102.5), (201.0, 203.5)]
    assert loaded["block_boundaries"] == [
        {
            "block_id": 0,
            "record_name": "blockA",
            "start_event_idx": 0,
            "end_event_idx": 1,
            "n_events": 1,
            "start_t": 100.0,
            "has_packed_times": True,
            "block_start_epoch": 102.0,
            "block_end_epoch": 102.5,
        },
        {
            "block_id": 1,
            "record_name": "blockB",
            "start_event_idx": 1,
            "end_event_idx": 3,
            "n_events": 2,
            "start_t": 200.0,
            "has_packed_times": True,
            "block_start_epoch": 201.0,
            "block_end_epoch": 203.5,
        },
    ]


def test_run_subject_interictal_propagation_pr1_reports_event_metadata_summary(tmp_path: Path) -> None:
    subject_dir = tmp_path / "548" / "all_recs"
    subject_dir.mkdir(parents=True)

    ranks = np.array([[1, 2], [2, 1]], dtype=float)
    bools = np.ones_like(ranks, dtype=int)
    np.savez_compressed(
        subject_dir / "block_lagPat.npz",
        lagPatRank=ranks,
        eventsBool=bools,
        chnNames=np.array(["A", "B"], dtype=object),
        start_t=np.array(100.0),
    )
    np.save(subject_dir / "block_packedTimes.npy", np.array([[1.0, 1.5], [4.0, 4.5]], dtype=float))

    out = run_subject_interictal_propagation_pr1(
        subject_dir=subject_dir,
        dataset="epilepsiae",
        subject="548",
        soz_channels=["A"],
        n_sample=2,
        n_seeds=2,
        min_center_participation=1,
    )
    meta = out["event_metadata"]
    assert out["n_blocks_used"] == 1
    assert meta["record_names"] == ["block"]
    assert meta["first_event_abs_time"] == 101.0
    assert meta["last_event_abs_time"] == 104.0


def test_adaptive_cluster_finds_stable_k_for_bimodal_data() -> None:
    rng = np.random.default_rng(42)
    n_ch = 4
    n_ev = 60
    pattern_a = np.tile(np.array([[1], [2], [3], [4]], dtype=float), (1, n_ev // 2))
    pattern_b = np.tile(np.array([[4], [3], [2], [1]], dtype=float), (1, n_ev // 2))
    noise = rng.normal(0, 0.05, size=(n_ch, n_ev // 2))
    ranks = np.concatenate([pattern_a + noise, pattern_b + noise], axis=1)
    bools = np.ones_like(ranks, dtype=bool)

    out = compute_adaptive_cluster_stereotypy(
        ranks,
        bools,
        channel_names=["A", "B", "C", "D"],
        k_range=(2, 5),
        n_stability_seeds=5,
        n_sample=30,
        n_tau_seeds=3,
    )
    assert "error" not in out
    assert out["chosen_k"] >= 2
    assert len(out["scan"]) >= 1
    assert len(out["clusters"]) == out["chosen_k"]
    assert np.isfinite(out["within_cluster_tau_mean"])
    assert np.isfinite(out["uplift"])
    assert out["uplift"] > 0
    assert isinstance(out["inter_cluster_corr_matrix"], list)
    assert isinstance(out["labels"], list)
    assert len(out["labels"]) == n_ev

    for entry in out["scan"]:
        assert "k" in entry
        assert "median_silhouette" in entry
        assert "median_ami" in entry
        assert "passes_both" in entry
        assert "best_labels" not in entry


def test_adaptive_cluster_returns_error_for_tiny_data() -> None:
    ranks = np.array([[1], [2]], dtype=float)
    bools = np.ones_like(ranks, dtype=bool)
    out = compute_adaptive_cluster_stereotypy(
        ranks,
        bools,
        channel_names=["A", "B"],
        k_range=(2, 4),
    )
    assert "error" in out


def test_adaptive_cluster_candidate_forward_reverse() -> None:
    rng = np.random.default_rng(7)
    n_ch = 6
    half = 30
    pattern_fwd = np.tile(np.arange(1, n_ch + 1, dtype=float).reshape(-1, 1), (1, half))
    pattern_rev = np.tile(np.arange(n_ch, 0, -1, dtype=float).reshape(-1, 1), (1, half))
    noise = rng.normal(0, 0.02, size=(n_ch, half))
    ranks = np.concatenate([pattern_fwd + noise, pattern_rev + noise], axis=1)
    bools = np.ones_like(ranks, dtype=bool)

    out = compute_adaptive_cluster_stereotypy(
        ranks,
        bools,
        channel_names=[f"ch{i}" for i in range(n_ch)],
        k_range=(2, 4),
        n_stability_seeds=5,
        n_sample=30,
        n_tau_seeds=3,
    )
    assert "error" not in out
    assert len(out["candidate_forward_reverse_pairs"]) >= 1
    pair = out["candidate_forward_reverse_pairs"][0]
    assert pair["spearman_r"] < -0.5
    assert pair["label"] == "candidate_forward_reverse"


def test_pr1_output_includes_adaptive_cluster(tmp_path: Path) -> None:
    subject_dir = tmp_path / "sub" / "all_recs"
    subject_dir.mkdir(parents=True)

    rng = np.random.default_rng(99)
    n_ch, n_ev = 4, 20
    ranks = rng.permutation(np.tile(np.arange(1, n_ch + 1).reshape(-1, 1), (1, n_ev)))
    bools = np.ones_like(ranks, dtype=int)
    np.savez_compressed(
        subject_dir / "block_lagPat.npz",
        lagPatRank=ranks.astype(float),
        eventsBool=bools,
        chnNames=np.array(["A", "B", "C", "D"], dtype=object),
    )

    out = run_subject_interictal_propagation_pr1(
        subject_dir=subject_dir,
        dataset="test",
        subject="sub",
        soz_channels=["A"],
        n_sample=10,
        n_seeds=2,
        min_center_participation=2,
    )
    assert "adaptive_cluster" in out
    ac = out["adaptive_cluster"]
    assert "scan" in ac
    assert "chosen_k" in ac
    assert "clusters" in ac


# ---------- PR-2.5: template reproducibility ----------


def test_build_cluster_templates_returns_correct_means() -> None:
    ranks = np.array(
        [[1.0, 1.0, 3.0, 3.0],
         [2.0, 2.0, 1.0, 1.0],
         [3.0, 3.0, 2.0, 2.0]],
    )
    bools = np.ones_like(ranks, dtype=bool)
    labels = np.array([0, 0, 1, 1])

    templates = build_cluster_templates(ranks, bools, labels, n_clusters=2)
    assert templates.shape == (2, 3)
    np.testing.assert_allclose(templates[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(templates[1], [3.0, 1.0, 2.0])


def test_build_cluster_templates_handles_missing_channels() -> None:
    ranks = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    bools = np.array([[1, 1], [1, 1], [0, 0]], dtype=bool)
    labels = np.array([0, 1])

    templates = build_cluster_templates(ranks, bools, labels, n_clusters=2)
    assert np.isnan(templates[0, 2])
    assert np.isnan(templates[1, 2])
    assert templates[0, 0] == 1.0
    assert templates[1, 0] == 2.0


def test_assign_events_to_templates_correct_assignment() -> None:
    templates = np.array(
        [[1.0, 2.0, 3.0, 4.0],
         [4.0, 3.0, 2.0, 1.0]],
    )
    ranks = np.array(
        [[1.0, 4.0, 1.2],
         [2.0, 3.0, 2.1],
         [3.0, 2.0, 2.9],
         [4.0, 1.0, 4.1]],
    )
    bools = np.ones_like(ranks, dtype=bool)

    assignments = assign_events_to_templates(ranks, bools, templates)
    assert assignments[0] == 0
    assert assignments[1] == 1
    assert assignments[2] == 0


def test_time_split_reproducibility_bimodal_gets_strong() -> None:
    rng = np.random.default_rng(42)
    n_ch = 6
    n_per_half = 40

    fwd = np.arange(1, n_ch + 1, dtype=float).reshape(-1, 1)
    rev = np.arange(n_ch, 0, -1, dtype=float).reshape(-1, 1)

    ranks_list = []
    labels_list = []
    block_ids_list = []
    times_list = []
    t = 0.0
    for blk in range(4):
        for _ev in range(n_per_half // 2):
            pattern = fwd if rng.random() < 0.5 else rev
            noise = rng.normal(0, 0.05, size=(n_ch, 1))
            ranks_list.append(pattern + noise)
            labels_list.append(0 if pattern is fwd else 1)
            block_ids_list.append(blk)
            times_list.append(t)
            t += 1.0

    ranks = np.concatenate(ranks_list, axis=1)
    bools = np.ones_like(ranks, dtype=bool)
    labels = np.array(labels_list, dtype=int)
    block_ids = np.array(block_ids_list, dtype=int)
    event_abs_times = np.array(times_list, dtype=float)
    valid_events = _valid_event_indices(bools, min_participating=3)

    repro = compute_time_split_reproducibility(
        ranks=ranks,
        bools=bools,
        event_abs_times=event_abs_times,
        block_ids=block_ids,
        chosen_k=2,
        adaptive_labels=labels,
        valid_event_indices=valid_events,
    )
    assert repro["reproducibility_grade"] == "strong"
    assert "first_half_second_half" in repro["splits"]
    assert "odd_even_block" in repro["splits"]

    sh = repro["splits"]["first_half_second_half"]
    assert sh["mean_match_corr"] > 0.8
    assert sh["assignment_agreement"] > 0.7


def test_time_split_reproducibility_single_block_skips_odd_even() -> None:
    n_ch, n_ev = 4, 20
    ranks = np.tile(np.arange(1, n_ch + 1, dtype=float).reshape(-1, 1), (1, n_ev))
    bools = np.ones_like(ranks, dtype=bool)
    labels = np.zeros(n_ev, dtype=int)
    block_ids = np.zeros(n_ev, dtype=int)
    event_abs_times = np.arange(n_ev, dtype=float)
    valid_events = _valid_event_indices(bools, min_participating=3)

    repro = compute_time_split_reproducibility(
        ranks=ranks,
        bools=bools,
        event_abs_times=event_abs_times,
        block_ids=block_ids,
        chosen_k=1,
        adaptive_labels=labels,
        valid_event_indices=valid_events,
    )
    assert "odd_even_block" not in repro["splits"]


def test_time_split_reproducibility_is_invariant_to_event_order_if_times_match() -> None:
    rng = np.random.default_rng(123)
    n_ch = 5
    n_ev = 40
    base_a = np.arange(1, n_ch + 1, dtype=float).reshape(-1, 1)
    base_b = np.arange(n_ch, 0, -1, dtype=float).reshape(-1, 1)

    ranks = np.concatenate(
        [
            np.tile(base_a, (1, n_ev // 2)) + rng.normal(0, 0.03, size=(n_ch, n_ev // 2)),
            np.tile(base_b, (1, n_ev // 2)) + rng.normal(0, 0.03, size=(n_ch, n_ev // 2)),
        ],
        axis=1,
    )
    bools = np.ones_like(ranks, dtype=bool)
    labels = np.array([0] * (n_ev // 2) + [1] * (n_ev // 2), dtype=int)
    times = np.linspace(100.0, 139.0, n_ev)
    blocks = np.repeat(np.arange(4), 10)
    valid_events = _valid_event_indices(bools, min_participating=3)

    repro_sorted = compute_time_split_reproducibility(
        ranks=ranks,
        bools=bools,
        event_abs_times=times,
        block_ids=blocks,
        chosen_k=2,
        adaptive_labels=labels,
        valid_event_indices=valid_events,
    )

    perm = rng.permutation(n_ev)
    repro_shuffled = compute_time_split_reproducibility(
        ranks=ranks[:, perm],
        bools=bools[:, perm],
        event_abs_times=times[perm],
        block_ids=blocks[perm],
        chosen_k=2,
        adaptive_labels=labels[perm],
        valid_event_indices=valid_events,
    )

    sh_sorted = repro_sorted["splits"]["first_half_second_half"]
    sh_shuffled = repro_shuffled["splits"]["first_half_second_half"]
    assert repro_sorted["reproducibility_grade"] == repro_shuffled["reproducibility_grade"]
    assert abs(sh_sorted["mean_match_corr"] - sh_shuffled["mean_match_corr"]) < 1e-6


def test_compute_temporal_cluster_dynamics_reports_timeline_and_daynight() -> None:
    import datetime

    tz = datetime.timezone(datetime.timedelta(hours=8))
    base = datetime.datetime(2024, 1, 1, 7, 0, 0, tzinfo=tz).timestamp()
    event_abs_times = np.array(
        [
            base + 0.0,        # 07:00 night
            base + 1800.0,     # 07:30 night
            base + 5400.0,     # 08:30 day
            base + 7200.0,     # 09:00 day
        ],
        dtype=float,
    )
    labels = np.array([0, 1, 0, 0], dtype=int)

    out = compute_temporal_cluster_dynamics(
        event_abs_times=event_abs_times,
        cluster_labels=labels,
        n_clusters=2,
        dataset="yuquan",
        bin_hours=1.0,
    )

    assert out["timezone_name"] == "Asia/Shanghai"
    assert out["n_events_used"] == 4
    assert len(out["timeline_bins"]) == 2
    assert out["timeline_bins"][0]["n_events"] == 2
    np.testing.assert_allclose(out["timeline_bins"][0]["cluster_fractions"], [0.5, 0.5])
    assert out["timeline_bins"][1]["day_night"] == "day"
    assert out["timeline_bins"][0]["hours_from_timeline_start"] == 0.5
    assert out["day_night_summary"]["day"]["n_events"] == 2
    assert out["day_night_summary"]["night"]["n_events"] == 2
    np.testing.assert_allclose(
        out["day_night_summary"]["day"]["cluster_fractions"],
        [1.0, 0.0],
    )
    np.testing.assert_allclose(
        out["day_night_summary"]["night"]["cluster_fractions"],
        [0.5, 0.5],
    )
    assert out["day_night_summary"]["total_variation_distance"] > 0.0


def test_compute_temporal_cluster_dynamics_respects_coverage_ranges_and_entropy_scale() -> None:
    event_abs_times = np.array([0.0, 3600.0 * 10.0], dtype=float)
    labels = np.array([0, 0], dtype=int)
    out = compute_temporal_cluster_dynamics(
        event_abs_times=event_abs_times,
        cluster_labels=labels,
        n_clusters=3,
        dataset="epilepsiae",
        coverage_ranges=[(0.0, 3600.0), (3600.0 * 10.0, 3600.0 * 11.0)],
        bin_hours=1.0,
    )

    assert len(out["timeline_bins"]) == 2
    assert [b["bin_id"] for b in out["timeline_bins"]] == [0, 10]
    assert out["day_night_summary"]["day"]["normalized_entropy"] == 0.0


def test_compute_continuous_template_dynamics_rate_and_histogram() -> None:
    event_abs_times = np.array([0.0, 1800.0, 3600.0, 5400.0], dtype=float)
    labels = np.array([0, 0, 1, 1], dtype=int)
    out = compute_continuous_template_dynamics(
        event_abs_times=event_abs_times,
        cluster_labels=labels,
        n_clusters=2,
        dataset="yuquan",
        coverage_ranges=[(0.0, 7200.0)],
        smoothing_hours=1.0,
        bin_hours=1.0,
    )

    assert out["n_events_used"] == 4
    assert out["n_clusters"] == 2
    assert np.isfinite(out["duration_hours"])

    rc = out["rate_curve"]
    assert len(rc["grid_hours"]) > 0
    assert len(rc["per_template_rate"]) == 2
    assert len(rc["total_rate"]) == len(rc["grid_hours"])
    total = np.array(rc["total_rate"])
    assert np.all(total >= 0)

    hist = out["histogram"]
    assert len(hist["per_template_count"]) == 2
    assert sum(sum(c) for c in hist["per_template_count"]) == 4

    s = out["summary"]
    assert s["dominant_rate_fraction"] == 0.5
    assert s["total_mean_rate"] > 0
    assert sum(s["per_template_event_count"]) == 4


def test_compute_continuous_template_dynamics_respects_coverage_gaps() -> None:
    event_abs_times = np.array([600.0, 36600.0], dtype=float)
    labels = np.array([0, 1], dtype=int)
    out = compute_continuous_template_dynamics(
        event_abs_times=event_abs_times,
        cluster_labels=labels,
        n_clusters=2,
        dataset="yuquan",
        coverage_ranges=[(0.0, 3600.0), (36000.0, 39600.0)],
        smoothing_hours=0.5,
        bin_hours=1.0,
    )

    # Exposure must be the observed time only: 2 x 1h windows = 2h.
    assert out["duration_hours"] == 2.0
    assert out["summary"]["total_mean_rate"] == 1.0
    assert out["summary"]["per_template_mean_rate"] == [0.5, 0.5]

    # Histogram should not manufacture zero-count bins inside the 9h gap.
    hist = out["histogram"]
    assert hist["bin_center_hours"] == [0.5, 10.5]
    assert hist["total_count"] == [1, 1]
    assert hist["bin_width_hours"] == [1.0, 1.0]

    # Rate curves should break across the recording gap instead of connecting.
    total_rate = np.asarray(out["rate_curve"]["total_rate"], dtype=float)
    assert np.any(np.isnan(total_rate))


def test_compute_continuous_template_dynamics_unequal_templates() -> None:
    event_abs_times = np.array([0.0, 120.0, 240.0, 7200.0], dtype=float)
    labels = np.array([0, 0, 0, 1], dtype=int)
    out = compute_continuous_template_dynamics(
        event_abs_times=event_abs_times,
        cluster_labels=labels,
        n_clusters=2,
        dataset="yuquan",
        smoothing_hours=0.5,
        bin_hours=0.5,
    )

    s = out["summary"]
    assert s["dominant_template_id"] == 0
    assert s["dominant_rate_fraction"] == 0.75
    assert s["per_template_event_count"] == [3, 1]


def test_validate_absolute_lag_clustering_reports_stratified_r_and_order_match() -> None:
    n_ch = 6
    n_ev = 10
    bools = np.zeros((n_ch, n_ev), dtype=bool)
    lag_raw = np.full((n_ch, n_ev), np.nan, dtype=float)
    ranks = np.zeros((n_ch, n_ev), dtype=float)
    labels = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 1], dtype=int)

    base_patterns = {
        0: np.array([0.00, 0.10, 0.22, 0.35, 0.48, 0.60], dtype=float),
        1: np.array([0.55, 0.44, 0.31, 0.20, 0.08, 0.00], dtype=float),
    }
    participating = [6, 6, 6, 6, 6, 6, 4, 4, 4, 4]

    for ev in range(n_ev):
        n_part = participating[ev]
        mask = np.zeros(n_ch, dtype=bool)
        mask[:n_part] = True
        bools[:, ev] = mask

        rel = base_patterns[int(labels[ev])].copy()
        rel[:n_part] += 0.01 * ev
        abs_offset = 100.0 + 7.0 * ev
        lag_raw[mask, ev] = abs_offset + rel[mask]

        rank_vals = np.argsort(np.argsort(rel[mask], kind="mergesort"), kind="mergesort") + 1
        ranks[mask, ev] = rank_vals.astype(float)

    valid_events = _valid_event_indices(bools, min_participating=3)
    out = validate_absolute_lag_clustering(
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=2,
        valid_event_indices=valid_events,
        n_sample=20,
        seed=0,
        min_shared_channels=3,
        min_participating=5,
    )

    assert out["n_valid_events"] == n_ev
    assert out["order_validation"]["exact_order_match_fraction"] == 1.0
    assert out["order_validation"]["pairwise_order_concordance"] == 1.0
    assert out["order_validation"]["nonnegative_fraction"] == 1.0
    assert out["within_cluster_pearson_r_by_npart"]["5-8"]["n_pairs_valid"] > 0
    assert out["within_cluster_pearson_r_by_npart"]["5-8"]["median_r"] > 0.99
    assert out["within_cluster_pearson_r_by_npart"]["3-4"]["n_events"] == 4
    assert out["eligible_fraction"] == 0.6
    assert out["dominant_cluster_median_r"] > 0.99
    assert out["dominant_cluster_fraction"] > 0.0
    assert out["validation_pass"] is True


def test_compute_rate_state_coupling_reports_high_vs_low_tau() -> None:
    n_ch = 5
    n_ev = 12
    ranks = np.zeros((n_ch, n_ev), dtype=float)
    bools = np.ones((n_ch, n_ev), dtype=bool)
    lag_raw = np.zeros((n_ch, n_ev), dtype=float)
    labels = np.zeros(n_ev, dtype=int)
    times = np.array(
        [
            0.0,
            1800.0,
            7200.0,
            7800.0,
            8400.0,
            9000.0,
            14400.0,
            15000.0,
            21600.0,
            22200.0,
            22800.0,
            23400.0,
        ],
        dtype=float,
    )

    pattern_stable = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    pattern_reverse = np.array([5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
    low_events = {0, 1, 6, 7}
    for ev in range(n_ev):
        pattern = pattern_stable if ev not in low_events or ev % 2 == 0 else pattern_reverse
        ranks[:, ev] = pattern
        if ev in low_events:
            rel = (
                np.array([0.0, 0.2, 0.4, 0.6, 0.8], dtype=float)
                if np.array_equal(pattern, pattern_stable)
                else np.array([0.8, 0.6, 0.4, 0.2, 0.0], dtype=float)
            )
        else:
            rel = np.array([0.0, 0.05, 0.10, 0.15, 0.20], dtype=float)
        lag_raw[:, ev] = 100.0 + 10.0 * ev + rel

    out = compute_rate_state_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        valid_event_indices=np.arange(n_ev, dtype=int),
        rate_bin_hours=2.0,
        min_events_per_bin=2,
        n_sample=20,
        n_seeds=3,
        min_shared_channels=3,
        min_center_participation=1,
    )

    assert out["n_rate_bins_total"] == 4
    assert out["n_rate_bins_eligible"] == 4
    assert out["high_bin_ids"] == [1, 3]
    assert out["low_bin_ids"] == [0, 2]
    assert out["state_event_counts"]["high"] == 8
    assert out["state_event_counts"]["low"] == 4
    assert out["l2"]["raw"]["n_clusters_compared"] == 1
    assert "per_cluster" in out
    assert len(out["per_cluster"]) == 1
    pc = out["per_cluster"][0]
    assert pc["high"]["n_events"] == pc["low"]["n_events"]
    assert out["l2"]["raw"]["delta_high_minus_low"] > 0.2
    assert "centered" in out["l2"]
    assert out["subject_raw_delta"] is not None
    assert out["l3"]["lag_span"]["n_clusters_compared"] == 1
    assert out["subject_lag_span_delta"] is not None
    assert out["subject_lag_span_delta"] < 0.0
    assert out["l3"]["pearson_r"]["n_clusters_compared"] == 1
    assert out["subject_pearson_r_delta"] is not None
    assert out["subject_pearson_r_delta"] > 0.5
    assert out["l1"]["dominant_cluster_id"] == 0
    assert "rate_bin_summary" in out
    assert "rate_bins" not in out


def test_seizure_proximity_configs_declares_main_and_auxiliary() -> None:
    """Named window configs: main=A (4/1/1), auxiliary=B (2/0.5/1)."""
    from src.interictal_propagation import SEIZURE_PROXIMITY_CONFIGS

    assert set(SEIZURE_PROXIMITY_CONFIGS) == {"main", "auxiliary"}

    main = SEIZURE_PROXIMITY_CONFIGS["main"]
    assert tuple(main["baseline_hours"]) == (-4.0, -1.0)
    assert tuple(main["pre_ictal_hours"]) == (-1.0, -0.25)
    assert tuple(main["post_ictal_hours"]) == (0.25, 1.0)

    aux = SEIZURE_PROXIMITY_CONFIGS["auxiliary"]
    assert tuple(aux["baseline_hours"]) == (-2.0, -0.5)
    assert abs(aux["pre_ictal_hours"][0] - -0.5) < 1e-9
    assert abs(aux["pre_ictal_hours"][1] - (-1.0 / 12.0)) < 1e-9
    assert abs(aux["post_ictal_hours"][0] - (1.0 / 12.0)) < 1e-9
    assert aux["post_ictal_hours"][1] == 1.0


def test_compute_seizure_proximity_coupling_default_window_is_main_config() -> None:
    """Unqualified call uses main config (A = 4/1/1 hours)."""
    n_ch = 5
    # Events spread across -4..+1h relative to seizure at t=10h
    seizure_t = 10.0 * 3600.0
    offsets_hours = [-3.5, -2.5, -1.5, -0.5, 0.5]  # each should land in a window
    times = np.array(
        [seizure_t + h * 3600.0 for h in offsets_hours], dtype=float
    )
    labels = np.zeros(times.size, dtype=int)
    bools = np.ones((n_ch, times.size), dtype=bool)
    ranks = np.tile(np.arange(1.0, n_ch + 1).reshape(-1, 1), (1, times.size))
    lag_raw = np.tile(np.linspace(0.0, 0.4, n_ch).reshape(-1, 1), (1, times.size))

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        seizure_times=[seizure_t],
        valid_event_indices=np.arange(times.size, dtype=int),
        n_sample=10,
        n_seeds=2,
        min_shared_channels=3,
        min_center_participation=1,
        min_participating_l3=5,
    )

    wh = out["window_hours"]
    assert tuple(wh["baseline"]) == (-4.0, -1.0)
    assert tuple(wh["pre"]) == (-1.0, -0.25)
    assert tuple(wh["post"]) == (0.25, 1.0)


def test_compute_seizure_proximity_coupling_handles_empty_seizure_list() -> None:
    ranks = np.tile(np.arange(1.0, 6.0).reshape(-1, 1), (1, 6))
    bools = np.ones_like(ranks, dtype=bool)
    lag_raw = np.tile(np.linspace(0.0, 0.4, 5).reshape(-1, 1), (1, 6))
    times = np.arange(6, dtype=float) * 3600.0
    labels = np.zeros(6, dtype=int)

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        seizure_times=[],
        valid_event_indices=np.arange(6, dtype=int),
    )

    assert out.get("warning") == "no_seizure_times"


def test_compute_seizure_proximity_coupling_reports_pre_vs_baseline_l2_l3() -> None:
    n_ch = 5
    times = np.array(
        [
            9.0,
            11.0,
            15.0,
            18.0,
            22.0,
            24.0,
        ],
        dtype=float,
    ) * 3600.0
    labels = np.zeros(times.size, dtype=int)
    bools = np.ones((n_ch, times.size), dtype=bool)
    ranks = np.array(
        [
            [1.0, 5.0, 1.0, 1.0, 1.0, 1.0],
            [2.0, 4.0, 2.0, 2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            [4.0, 2.0, 4.0, 4.0, 4.0, 4.0],
            [5.0, 1.0, 5.0, 5.0, 5.0, 5.0],
        ],
        dtype=float,
    )
    lag_raw = np.array(
        [
            [0.00, 0.80, 0.00, 0.00, 0.00, 0.00],
            [0.20, 0.60, 0.08, 0.09, 0.06, 0.07],
            [0.40, 0.40, 0.16, 0.18, 0.12, 0.14],
            [0.60, 0.20, 0.24, 0.27, 0.18, 0.21],
            [0.80, 0.00, 0.32, 0.36, 0.24, 0.28],
        ],
        dtype=float,
    )

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        seizure_times=[20.0 * 3600.0],
        valid_event_indices=np.arange(times.size, dtype=int),
        baseline_hours=(-12.0, -6.0),
        pre_ictal_hours=(-6.0, -1.0),
        post_ictal_hours=(1.0, 6.0),
        n_sample=20,
        n_seeds=3,
        min_shared_channels=3,
        min_center_participation=1,
        min_participating_l3=5,
    )

    assert out["n_seizures_total"] == 1
    assert out["n_seizures_usable"] == 1
    assert out["state_event_counts"] == {
        "baseline": 2,
        "pre": 2,
        "post": 2,
        "excluded": 0,
    }
    assert len(out["seizure_windows"]) == 1

    pre_vs_baseline = out["comparison_summary"]["pre_vs_baseline"]
    assert pre_vs_baseline["n_windows"] == 1
    assert pre_vs_baseline["raw_tau"]["delta_state_b_minus_state_a_median"] > 1.5
    assert "delta_state_b_minus_state_a_median" in pre_vs_baseline["centered_tau"]
    assert pre_vs_baseline["l3"]["lag_span"]["delta_state_b_minus_state_a_median"] < 0.0
    assert pre_vs_baseline["l3"]["pearson_r"]["delta_state_b_minus_state_a_median"] > 1.5
    assert pre_vs_baseline["dominant_cluster_fraction"]["state_a_median"] == 1.0
    assert pre_vs_baseline["dominant_cluster_fraction"]["state_b_median"] == 1.0


def test_compute_seizure_proximity_coupling_rate_by_template_decomposes_states() -> None:
    """PR-4C secondary descriptive layer: per-state per-template event rate."""
    n_ch = 5
    times = (
        np.array(
            [
                9.0, 10.0,
                15.0, 17.0, 19.0,
                22.0, 24.0, 25.0,
            ],
            dtype=float,
        )
        * 3600.0
    )
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 0], dtype=int)
    bools = np.ones((n_ch, times.size), dtype=bool)
    ranks = np.tile(np.arange(1.0, n_ch + 1).reshape(-1, 1), (1, times.size))
    lag_raw = np.tile(np.linspace(0.0, 0.4, n_ch).reshape(-1, 1), (1, times.size))

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=2,
        seizure_times=[20.0 * 3600.0],
        valid_event_indices=np.arange(times.size, dtype=int),
        baseline_hours=(-12.0, -6.0),
        pre_ictal_hours=(-6.0, -1.0),
        post_ictal_hours=(1.0, 6.0),
        n_sample=20,
        n_seeds=3,
        min_shared_channels=3,
        min_center_participation=1,
        min_participating_l3=5,
    )

    assert out["n_seizures_usable"] == 1
    window = out["seizure_windows"][0]
    assert "rate_by_template" in window
    baseline_state = window["rate_by_template"]["baseline"]
    pre_state = window["rate_by_template"]["pre"]
    post_state = window["rate_by_template"]["post"]
    assert baseline_state["counts_total"] == 2
    assert baseline_state["counts_by_template"] == [2, 0]
    assert baseline_state["duration_hours"] == 6.0
    assert abs(baseline_state["rate_total_per_hour"] - 2.0 / 6.0) < 1e-9
    assert pre_state["counts_by_template"] == [1, 1]
    assert post_state["counts_by_template"] == [1, 2]

    rbt_summary = out["rate_by_template_summary"]
    assert rbt_summary["n_windows"] == 1
    assert rbt_summary["n_templates"] == 2
    pre_vs_baseline = rbt_summary["pre_vs_baseline"]
    assert pre_vs_baseline["n_windows"] == 1
    assert pre_vs_baseline["median_rate_delta_by_template"][1] > 0
    assert np.isfinite(pre_vs_baseline["max_abs_fraction_delta_template"])
    post_vs_baseline = rbt_summary["post_vs_baseline"]
    assert post_vs_baseline["median_rate_delta_by_template"][1] > 0


def test_seizure_proximity_assigns_events_to_nearest_seizure_for_dense_clusters() -> None:
    """Dense seizures: an event between two seizures goes to the closer one."""
    n_ch = 5
    seizure_t = [20.0 * 3600.0, 30.0 * 3600.0]
    times = np.array(
        [
            9.0,   # baseline of sz0 (-11h)
            15.0,  # pre of sz0 (-5h). Also in baseline of sz1 (-15h) -> out of baseline range (-12,-6). Actually -15 is out of (-12,-6). Unambig -> pre(sz0).
            24.0,  # post of sz0 (+4h). Also pre of sz1 (-6h) -> boundary pre(-6,-1): -6 == lo so pre(sz1). Nearest = sz1 (6h) < sz0 (4h)? |24-20|=4, |24-30|=6, nearest=sz0, post(sz0).
            26.0,  # post of sz0 (+6h) out of [1,6). |26-20|=6 not<6; nearest=sz1 (|26-30|=4), pre(sz1)=(-4h) in (-6,-1) -> pre(sz1).
            35.0,  # post of sz1 (+5h). Nearest=sz1. post(sz1).
        ],
        dtype=float,
    ) * 3600.0
    labels = np.zeros(times.size, dtype=int)
    bools = np.ones((n_ch, times.size), dtype=bool)
    ranks = np.tile(np.arange(1.0, n_ch + 1).reshape(-1, 1), (1, times.size))
    lag_raw = np.tile(np.linspace(0.0, 0.4, n_ch).reshape(-1, 1), (1, times.size))

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        seizure_times=seizure_t,
        valid_event_indices=np.arange(times.size, dtype=int),
        baseline_hours=(-12.0, -6.0),
        pre_ictal_hours=(-6.0, -1.0),
        post_ictal_hours=(1.0, 6.0),
        n_sample=10,
        n_seeds=2,
        min_shared_channels=3,
        min_center_participation=1,
        min_participating_l3=5,
    )

    totals = out["state_event_counts"]
    assert totals["baseline"] >= 1
    assert totals["pre"] >= 1
    assert totals["post"] >= 1
    assert totals["excluded"] <= 1

    assert len(out["seizure_windows"]) == 2
    for window in out["seizure_windows"]:
        for state in ("baseline", "pre", "post"):
            assert state in window["state_event_counts"]


def test_summarize_propagation_cohort_includes_temporal_label_invariant_summary() -> None:
    subject_results = {
        "yuquan/a": {
            "dataset": "yuquan",
            "subject": "a",
            "propagation_stereotypy": {"all": {"mean_tau": 0.1}, "soz": {}, "nonsoz": {}},
            "mixture": {"is_mixture": False, "possible_mixture": False},
            "centered_rank": {"bias_fraction": 0.3},
            "source_diagnostic": {"soz_source_erased": False},
            "by_nparticipating": [{"bin_label": "3", "mean_tau": 0.1}],
            "cluster": {"within_cluster_tau_mean": 0.2, "uplift": 0.1, "overall_tau": 0.1, "inter_cluster_corr": -0.6},
            "legacy_mi": {"mi_mean": 0.2, "significant": True},
            "adaptive_cluster": {"stable_k": 2, "chosen_k": 2, "uplift": 0.1, "within_cluster_tau_mean": 0.2},
            "time_split_reproducibility": {"reproducibility_grade": "strong", "splits": {}},
            "absolute_lag_validation": {
                "eligible_fraction": 0.6,
                "eligible_median_r": 0.82,
                "dominant_cluster_median_r": 0.85,
                "dominant_cluster_fraction": 0.7,
                "dominant_cluster_id": 0,
                "validation_pass": True,
                "order_validation": {
                    "exact_order_match_fraction": 1.0,
                    "pairwise_order_concordance": 1.0,
                },
                "within_cluster_pearson_r_by_npart": {
                    "3-4": {"median_r": 0.55},
                    "5-8": {"median_r": 0.82},
                    "9+": {"median_r": np.nan},
                },
            },
            "rate_state_coupling": {
                "subject_raw_delta": 0.20,
                "subject_centered_delta": 0.10,
                "subject_lag_span_delta": -0.30,
                "subject_pearson_r_delta": 0.25,
                "l3_validation_pass": True,
                "l2": {
                    "raw": {"high_mean": 0.30, "low_mean": 0.10, "delta_high_minus_low": 0.20},
                    "centered": {"high_mean": 0.12, "low_mean": 0.02, "delta_high_minus_low": 0.10},
                },
                "l3": {
                    "lag_span": {"high_mean": 0.20, "low_mean": 0.50, "delta_high_minus_low": -0.30},
                    "pearson_r": {"high_mean": 0.90, "low_mean": 0.65, "delta_high_minus_low": 0.25},
                },
                "l1": {
                    "dominant_cluster": {"occupancy_rate_spearman_rho": 0.40},
                    "max_abs_spearman_rho": 0.50,
                },
                "median_eligible_rate_per_hour": 12.0,
                "l3_eligible_fraction": 0.60,
            },
            "temporal_dynamics": {
                "day_night_summary": {
                    "day": {"n_events": 5, "dominant_fraction": 0.8, "normalized_entropy": 0.1},
                    "night": {"n_events": 5, "dominant_fraction": 0.4, "normalized_entropy": 0.9},
                    "total_variation_distance": 0.4,
                }
            },
            "temporal_dynamics_followup": {
                "summary": {
                    "dominant_rate_fraction": 0.62,
                    "total_mean_rate": 50.0,
                    "per_template_event_count": [620, 380],
                    "per_template_mean_rate": [31.0, 19.0],
                    "per_template_rate_fraction": [0.62, 0.38],
                    "dominant_template_id": 0,
                },
            },
            "seizure_proximity_coupling": {
                "n_seizures_usable": 3,
                "comparison_summary": {
                    "pre_vs_baseline": {
                        "raw_tau": {"delta_state_b_minus_state_a_median": 0.10},
                        "centered_tau": {"delta_state_b_minus_state_a_median": 0.05},
                        "l3": {
                            "lag_span": {"delta_state_b_minus_state_a_median": -0.02},
                            "pearson_r": {"delta_state_b_minus_state_a_median": 0.20},
                        },
                        "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.01},
                    },
                    "post_vs_pre": {
                        "raw_tau": {"delta_state_b_minus_state_a_median": -0.03},
                        "centered_tau": {"delta_state_b_minus_state_a_median": -0.02},
                        "l3": {
                            "lag_span": {"delta_state_b_minus_state_a_median": 0.01},
                            "pearson_r": {"delta_state_b_minus_state_a_median": -0.10},
                        },
                        "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.00},
                    },
                    "post_vs_baseline": {
                        "raw_tau": {"delta_state_b_minus_state_a_median": 0.07},
                        "centered_tau": {"delta_state_b_minus_state_a_median": 0.03},
                        "l3": {
                            "lag_span": {"delta_state_b_minus_state_a_median": -0.01},
                            "pearson_r": {"delta_state_b_minus_state_a_median": 0.10},
                        },
                        "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.01},
                    },
                },
            },
        },
        "epilepsiae/b": {
            "dataset": "epilepsiae",
            "subject": "b",
            "propagation_stereotypy": {"all": {"mean_tau": 0.2}, "soz": {}, "nonsoz": {}},
            "mixture": {"is_mixture": True, "possible_mixture": True},
            "centered_rank": {"bias_fraction": 0.5},
            "source_diagnostic": {"soz_source_erased": False},
            "by_nparticipating": [{"bin_label": "3", "mean_tau": 0.2}],
            "cluster": {"within_cluster_tau_mean": 0.3, "uplift": 0.1, "overall_tau": 0.2, "inter_cluster_corr": -0.4},
            "legacy_mi": {"mi_mean": 0.3, "significant": True},
            "adaptive_cluster": {"stable_k": 2, "chosen_k": 2, "uplift": 0.1, "within_cluster_tau_mean": 0.3},
            "time_split_reproducibility": {"reproducibility_grade": "moderate", "splits": {}},
            "absolute_lag_validation": {
                "eligible_fraction": 0.4,
                "eligible_median_r": 0.74,
                "dominant_cluster_median_r": 0.80,
                "dominant_cluster_fraction": 0.6,
                "dominant_cluster_id": 1,
                "validation_pass": True,
                "order_validation": {
                    "exact_order_match_fraction": 0.95,
                    "pairwise_order_concordance": 0.98,
                },
                "within_cluster_pearson_r_by_npart": {
                    "3-4": {"median_r": 0.51},
                    "5-8": {"median_r": 0.74},
                    "9+": {"median_r": 0.88},
                },
            },
            "rate_state_coupling": {
                "subject_raw_delta": 0.10,
                "subject_centered_delta": 0.05,
                "subject_lag_span_delta": -0.10,
                "subject_pearson_r_delta": -0.05,
                "l3_validation_pass": False,
                "l2": {
                    "raw": {"high_mean": 0.25, "low_mean": 0.15, "delta_high_minus_low": 0.10},
                    "centered": {"high_mean": 0.08, "low_mean": 0.03, "delta_high_minus_low": 0.05},
                },
                "l3": {
                    "lag_span": {"high_mean": 0.30, "low_mean": 0.40, "delta_high_minus_low": -0.10},
                    "pearson_r": {"high_mean": 0.70, "low_mean": 0.75, "delta_high_minus_low": -0.05},
                },
                "l1": {
                    "dominant_cluster": {"occupancy_rate_spearman_rho": -0.20},
                    "max_abs_spearman_rho": 0.30,
                },
                "median_eligible_rate_per_hour": 10.0,
                "l3_eligible_fraction": 0.40,
            },
            "temporal_dynamics": {
                "day_night_summary": {
                    "day": {"n_events": 6, "dominant_fraction": 0.7, "normalized_entropy": 0.2},
                    "night": {"n_events": 6, "dominant_fraction": 0.6, "normalized_entropy": 0.4},
                    "total_variation_distance": 0.2,
                }
            },
            "temporal_dynamics_followup": {
                "summary": {
                    "dominant_rate_fraction": 0.58,
                    "total_mean_rate": 40.0,
                    "per_template_event_count": [580, 420],
                    "per_template_mean_rate": [23.2, 16.8],
                    "per_template_rate_fraction": [0.58, 0.42],
                    "dominant_template_id": 0,
                },
            },
            "seizure_proximity_coupling": {
                "n_seizures_usable": 2,
                "comparison_summary": {
                    "pre_vs_baseline": {
                        "raw_tau": {"delta_state_b_minus_state_a_median": 0.00},
                        "centered_tau": {"delta_state_b_minus_state_a_median": 0.02},
                        "l3": {
                            "lag_span": {"delta_state_b_minus_state_a_median": -0.01},
                            "pearson_r": {"delta_state_b_minus_state_a_median": 0.05},
                        },
                        "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": -0.02},
                    },
                    "post_vs_pre": {
                        "raw_tau": {"delta_state_b_minus_state_a_median": 0.02},
                        "centered_tau": {"delta_state_b_minus_state_a_median": 0.01},
                        "l3": {
                            "lag_span": {"delta_state_b_minus_state_a_median": 0.03},
                            "pearson_r": {"delta_state_b_minus_state_a_median": -0.02},
                        },
                        "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.03},
                    },
                    "post_vs_baseline": {
                        "raw_tau": {"delta_state_b_minus_state_a_median": 0.01},
                        "centered_tau": {"delta_state_b_minus_state_a_median": 0.03},
                        "l3": {
                            "lag_span": {"delta_state_b_minus_state_a_median": 0.02},
                            "pearson_r": {"delta_state_b_minus_state_a_median": 0.00},
                        },
                        "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.01},
                    },
                },
            },
        },
    }

    cohort = summarize_propagation_cohort(subject_results)
    temporal = cohort["temporal_dynamics_analysis"]
    assert temporal["n_subjects"] == 2
    assert temporal["n_subjects_with_day_night"] == 2
    assert np.isfinite(temporal["dominant_fraction"]["day_median"])
    assert np.isfinite(temporal["normalized_entropy"]["night_median"])
    assert temporal["day_night_total_variation"]["median"] == 0.30000000000000004
    lag_summary = cohort["absolute_lag_validation_analysis"]
    assert lag_summary["n_subjects"] == 2
    assert lag_summary["n_subjects_pass"] == 2
    assert lag_summary["dominant_cluster_median_r_median"] == 0.825
    assert lag_summary["cohort_validation_pass"] is True
    assert lag_summary["within_cluster_pearson_r_by_npart"]["5-8"]["median_r"] == 0.78
    coupling = cohort["rate_state_coupling_analysis"]
    assert coupling["n_subjects"] == 2
    assert coupling["raw_tau"]["delta_high_minus_low_median"] == 0.15000000000000002
    assert coupling["centered_tau"]["n_subjects_high_gt_low"] == 2
    assert "wilcoxon_p" in coupling["raw_tau"]
    assert "wilcoxon_n" in coupling["raw_tau"]
    assert coupling["raw_tau"]["wilcoxon_n"] == 2
    assert coupling["l3"]["lag_span"]["delta_high_minus_low_median"] == -0.2
    assert coupling["l3"]["pearson_r_exploratory"]["delta_high_minus_low_median"] == 0.1
    assert coupling["l3"]["pearson_r_high_confidence"]["wilcoxon_n"] == 1
    assert coupling["l1"]["max_abs_rho_median"] == 0.4
    followup = cohort["temporal_dynamics_followup_analysis"]
    assert followup["n_subjects"] == 2
    assert followup["dominant_rate_fraction_median"] == 0.6
    assert followup["total_mean_rate_median"] == 45.0
    seizure = cohort["seizure_proximity_analysis"]
    assert seizure["n_subjects"] == 2
    assert seizure["n_subjects_with_usable_windows"] == 2
    assert seizure["n_usable_windows_total"] == 5
    assert seizure["pre_vs_baseline"]["raw_tau"]["delta_state_b_minus_state_a_median"] == 0.05
    assert seizure["pre_vs_baseline"]["l3"]["lag_span"]["delta_state_b_minus_state_a_median"] == -0.015
    assert "rate_by_template" in seizure
    assert seizure["rate_by_template"]["n_subjects"] == 0


def test_summarize_propagation_cohort_reports_auxiliary_seizure_proximity_when_present() -> None:
    """Cohort aggregates seizure_proximity_coupling_auxiliary separately."""
    from src.interictal_propagation import summarize_propagation_cohort

    common = {
        "propagation_stereotypy": {"all": {"mean_tau": 0.1}, "soz": {}, "nonsoz": {}},
        "mixture": {"is_mixture": False, "possible_mixture": False},
        "centered_rank": {"bias_fraction": 0.3},
        "source_diagnostic": {"soz_source_erased": False},
        "by_nparticipating": [{"bin_label": "3", "mean_tau": 0.1}],
        "cluster": {"within_cluster_tau_mean": 0.2, "uplift": 0.1, "overall_tau": 0.1, "inter_cluster_corr": -0.6},
        "legacy_mi": {"mi_mean": 0.2, "significant": False},
        "adaptive_cluster": {"stable_k": 2, "chosen_k": 2, "uplift": 0.1, "within_cluster_tau_mean": 0.2},
    }

    def _sp_rec(raw_delta: float, n_usable: int) -> Dict[str, Any]:
        return {
            "n_seizures_usable": n_usable,
            "comparison_summary": {
                "pre_vs_baseline": {
                    "raw_tau": {"delta_state_b_minus_state_a_median": raw_delta},
                    "centered_tau": {"delta_state_b_minus_state_a_median": 0.0},
                    "l3": {
                        "lag_span": {"delta_state_b_minus_state_a_median": 0.0},
                        "pearson_r": {"delta_state_b_minus_state_a_median": 0.0},
                    },
                    "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.0},
                },
                "post_vs_pre": {
                    "raw_tau": {"delta_state_b_minus_state_a_median": 0.0},
                    "centered_tau": {"delta_state_b_minus_state_a_median": 0.0},
                    "l3": {
                        "lag_span": {"delta_state_b_minus_state_a_median": 0.0},
                        "pearson_r": {"delta_state_b_minus_state_a_median": 0.0},
                    },
                    "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.0},
                },
                "post_vs_baseline": {
                    "raw_tau": {"delta_state_b_minus_state_a_median": 0.0},
                    "centered_tau": {"delta_state_b_minus_state_a_median": 0.0},
                    "l3": {
                        "lag_span": {"delta_state_b_minus_state_a_median": 0.0},
                        "pearson_r": {"delta_state_b_minus_state_a_median": 0.0},
                    },
                    "dominant_cluster_fraction": {"delta_state_b_minus_state_a_median": 0.0},
                },
            },
        }

    subject_results = {
        "epilepsiae/a": {
            **common,
            "dataset": "epilepsiae", "subject": "a",
            "seizure_proximity_coupling": _sp_rec(raw_delta=0.10, n_usable=4),
            "seizure_proximity_coupling_auxiliary": _sp_rec(raw_delta=0.05, n_usable=6),
        },
        "epilepsiae/b": {
            **common,
            "dataset": "epilepsiae", "subject": "b",
            "seizure_proximity_coupling": _sp_rec(raw_delta=0.20, n_usable=3),
            "seizure_proximity_coupling_auxiliary": _sp_rec(raw_delta=0.08, n_usable=5),
        },
    }

    cohort = summarize_propagation_cohort(subject_results)
    main = cohort["seizure_proximity_analysis"]
    aux = cohort["seizure_proximity_analysis_auxiliary"]
    assert main["n_subjects"] == 2
    assert main["n_usable_windows_total"] == 7
    assert abs(
        main["pre_vs_baseline"]["raw_tau"]["delta_state_b_minus_state_a_median"] - 0.15
    ) < 1e-9
    assert aux["n_subjects"] == 2
    assert aux["n_usable_windows_total"] == 11
    assert abs(
        aux["pre_vs_baseline"]["raw_tau"]["delta_state_b_minus_state_a_median"] - 0.065
    ) < 1e-9


def test_seizure_proximity_window_missing_post_still_supports_pre_vs_baseline() -> None:
    """Fix A contract: per-pair usability, not all-three-states gate.

    A window with 5 baseline events + 5 pre-ictal events but 0 post-ictal
    events must still feed the pre_vs_baseline pair. Discarding it whole is a
    pure power loss that systematically biases pre_vs_baseline toward null.
    """
    n_ch = 5
    sz_t = 50.0 * 3600.0
    delta_h = np.array(
        [
            -10.0, -9.5, -9.0, -8.5, -8.0,
            -5.0, -4.0, -3.0, -2.0, -1.5,
        ],
        dtype=float,
    )
    times = sz_t + delta_h * 3600.0
    labels = np.zeros(times.size, dtype=int)
    bools = np.ones((n_ch, times.size), dtype=bool)
    rng = np.random.default_rng(0)
    ranks = np.tile(np.arange(1.0, n_ch + 1).reshape(-1, 1), (1, times.size))
    ranks = ranks + rng.normal(0.0, 0.05, size=ranks.shape)
    lag_raw = np.tile(np.linspace(0.0, 0.4, n_ch).reshape(-1, 1), (1, times.size))
    lag_raw = lag_raw + rng.normal(0.0, 0.005, size=lag_raw.shape)

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        seizure_times=[sz_t],
        valid_event_indices=np.arange(times.size, dtype=int),
        baseline_hours=(-12.0, -6.0),
        pre_ictal_hours=(-6.0, -1.0),
        post_ictal_hours=(1.0, 6.0),
        n_sample=20,
        n_seeds=2,
        min_shared_channels=3,
        min_center_participation=1,
        min_participating_l3=5,
    )

    assert len(out["seizure_windows"]) == 1
    window = out["seizure_windows"][0]
    assert window["state_event_counts"]["baseline"] == 5
    assert window["state_event_counts"]["pre"] == 5
    assert window["state_event_counts"]["post"] == 0

    pair_status = window.get("pair_usability", {})
    assert pair_status.get("pre_vs_baseline") is True, (
        "Fix A: pre_vs_baseline must be marked usable when its two states are non-empty"
    )
    assert pair_status.get("post_vs_pre") is False
    assert pair_status.get("post_vs_baseline") is False

    assert "pairwise_comparisons" in window
    assert "pre_vs_baseline" in window["pairwise_comparisons"]

    cohort = out["comparison_summary"]
    assert cohort["pre_vs_baseline"]["n_windows"] >= 1, (
        "Window contributing baseline+pre must enter the pre_vs_baseline summary"
    )
    assert cohort["post_vs_pre"]["n_windows"] == 0
    assert cohort["post_vs_baseline"]["n_windows"] == 0


def test_seizure_proximity_assigns_event_to_non_nearest_seizure_when_nearest_has_no_state() -> None:
    """Fix B contract: candidate enumeration, not nearest-first hard cut.

    Two seizures sz0=0h, sz1=8h. Windows baseline=(-12,-6), pre=(-6,-1),
    post=(1,6). Place a rescued event at sz0 + 0.5h. Distance to sz0 is
    +0.5h (no legal state — outside post.lo=1.0 and pre.hi=-1.0). Distance
    to sz1 is -7.5h (legal: in baseline=(-12,-6)). Nearest seizure is sz0
    (|0.5| < |7.5|), so the legacy "pick nearest then check state" path
    discards the event. Fix B must enumerate candidates and assign it to
    sz1.baseline.
    """
    n_ch = 5
    sz0 = 0.0
    sz1 = 8.0 * 3600.0
    seizure_t = [sz0, sz1]
    other_baseline = sz0 + np.array([-10.0, -9.0, -8.0, -7.0]) * 3600.0
    other_pre = sz0 + np.array([-5.0, -4.0, -3.0, -2.0]) * 3600.0
    other_post = sz1 + np.array([1.5, 2.0, 3.0, 4.0]) * 3600.0
    rescued_event = np.array([sz0 + 0.5 * 3600.0])
    times = np.concatenate([other_baseline, other_pre, other_post, rescued_event])
    rescued_idx = times.size - 1

    labels = np.zeros(times.size, dtype=int)
    bools = np.ones((n_ch, times.size), dtype=bool)
    ranks = np.tile(np.arange(1.0, n_ch + 1).reshape(-1, 1), (1, times.size))
    lag_raw = np.tile(np.linspace(0.0, 0.4, n_ch).reshape(-1, 1), (1, times.size))

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        seizure_times=seizure_t,
        valid_event_indices=np.arange(times.size, dtype=int),
        baseline_hours=(-12.0, -6.0),
        pre_ictal_hours=(-6.0, -1.0),
        post_ictal_hours=(1.0, 6.0),
        n_sample=10,
        n_seeds=2,
        min_shared_channels=3,
        min_center_participation=1,
        min_participating_l3=5,
    )

    counts = out["state_event_counts"]
    assert counts["excluded"] == 0, (
        "Fix B: rescued event must not be excluded (sz1 baseline is legal even though sz0 is nearest)"
    )

    sz1_window = next(
        win for win in out["seizure_windows"] if win["seizure_id"] == 1
    )
    sz0_window = next(
        win for win in out["seizure_windows"] if win["seizure_id"] == 0
    )
    assert sz1_window["state_event_counts"]["baseline"] == 1, (
        "Fix B: sz1 baseline must own exactly the rescued event"
    )
    assert sz1_window["state_event_counts"]["pre"] == 0
    assert sz1_window["state_event_counts"]["post"] == 4
    assert sz0_window["state_event_counts"]["baseline"] == 4
    assert sz0_window["state_event_counts"]["pre"] == 4
    assert sz0_window["state_event_counts"]["post"] == 0


def test_rate_by_template_uses_gap_aware_coverage_when_available() -> None:
    """Fix C contract: rate denominator uses real coverage_ranges, not fixed window width.

    Construct one seizure at 50h, baseline window (-12, -6) h relative.
    Provide coverage_ranges that cover only the second half of baseline
    (i.e. (-9, -6) h). 3 baseline events occur inside the covered second
    half. Real covered duration = 3 h, not the 6 h nominal window width.
    Therefore baseline rate must be 3 / 3 = 1.0, not 3 / 6 = 0.5.
    """
    n_ch = 5
    sz_t = 50.0 * 3600.0
    baseline_events = sz_t + np.array([-8.5, -7.5, -6.5]) * 3600.0
    pre_events = sz_t + np.array([-5.0, -3.0, -2.0]) * 3600.0
    post_events = sz_t + np.array([1.5, 2.5, 4.0]) * 3600.0
    times = np.concatenate([baseline_events, pre_events, post_events])

    labels = np.zeros(times.size, dtype=int)
    bools = np.ones((n_ch, times.size), dtype=bool)
    ranks = np.tile(np.arange(1.0, n_ch + 1).reshape(-1, 1), (1, times.size))
    lag_raw = np.tile(np.linspace(0.0, 0.4, n_ch).reshape(-1, 1), (1, times.size))

    coverage_ranges = [
        (sz_t - 9.0 * 3600.0, sz_t - 6.0 * 3600.0),
        (sz_t - 6.0 * 3600.0, sz_t - 1.0 * 3600.0),
        (sz_t + 1.0 * 3600.0, sz_t + 6.0 * 3600.0),
    ]

    out = compute_seizure_proximity_coupling(
        event_abs_times=times,
        ranks=ranks,
        lag_raw=lag_raw,
        bools=bools,
        cluster_labels=labels,
        n_clusters=1,
        seizure_times=[sz_t],
        valid_event_indices=np.arange(times.size, dtype=int),
        baseline_hours=(-12.0, -6.0),
        pre_ictal_hours=(-6.0, -1.0),
        post_ictal_hours=(1.0, 6.0),
        coverage_ranges=coverage_ranges,
        n_sample=10,
        n_seeds=2,
        min_shared_channels=3,
        min_center_participation=1,
        min_participating_l3=5,
    )

    window = out["seizure_windows"][0]
    baseline_state = window["rate_by_template"]["baseline"]
    pre_state = window["rate_by_template"]["pre"]
    post_state = window["rate_by_template"]["post"]

    assert abs(baseline_state["duration_hours"] - 3.0) < 1e-6, (
        "Fix C: baseline duration must reflect actual covered hours (3.0), not nominal window width (6.0)"
    )
    assert abs(baseline_state["rate_total_per_hour"] - 1.0) < 1e-6, (
        "Fix C: baseline rate must be counts / actual_covered_hours = 3/3 = 1.0"
    )
    assert abs(pre_state["duration_hours"] - 5.0) < 1e-6
    assert abs(post_state["duration_hours"] - 5.0) < 1e-6


def test_runner_parser_exposes_pr4c_auxiliary_flag() -> None:
    """CLI must expose --pr4c-auxiliary alongside --pr4c."""
    import importlib

    mod = importlib.import_module("scripts.run_interictal_propagation")
    parser = mod._build_parser()  # expected helper
    option_strings = {
        opt for action in parser._actions for opt in action.option_strings
    }
    assert "--pr4c" in option_strings
    assert "--pr4c-auxiliary" in option_strings


# ---------------------------------------------------------------------------
# PR-5-A: novel-template falsification gate
# ---------------------------------------------------------------------------
#
# Contract reference: docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md
#   §3.2 data contract, §3.3 metric definitions, §3.5 PASS/FAIL thresholds,
#   §5.3 tests 1-3.
#
# The gate must, for events in baseline / pre / post:
#   1. Filter to L3-eligible events (n_participating >= min_participating_l3=5).
#   2. Per event compute (a) best-template Spearman r between v_rel on
#      participating channels and the matched template's rank vector,
#      (b) min reconstruction error in rank-space (mean squared diff,
#      masked, >=min_shared_channels=3), (c) assignment gap = 2nd best
#      recon - best recon.
#   3. Pool across all usable seizure windows; take per-state medians;
#      report deltas (state - baseline).


def _pr5a_synthetic_window(
    state_indices_dict: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Build a minimal usable-window dict matching ``compute_seizure_proximity_coupling`` output."""
    counts = {name: int(idx.size) for name, idx in state_indices_dict.items()}
    return {
        "seizure_id": 0,
        "seizure_time": 0.0,
        "state_event_indices": {
            name: np.asarray(idx, dtype=int)
            for name, idx in state_indices_dict.items()
        },
        "state_event_counts": counts,
        "pair_usability": {
            "pre_vs_baseline": counts["baseline"] > 0 and counts["pre"] > 0,
            "post_vs_pre": counts["pre"] > 0 and counts["post"] > 0,
            "post_vs_baseline": counts["baseline"] > 0 and counts["post"] > 0,
        },
        "usable": True,
    }


def _pr5a_make_event_arrays(
    rng: np.random.Generator,
    base_templates: List[np.ndarray],
    n_events: int,
    *,
    rank_noise: float,
    rel_noise: float,
    rel_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample events from the supplied template set with small Gaussian noise.

    Returns (ranks (n_ch, n), rel (n_ch, n)) such that Spearman(v_rel[:, i],
    template) is near 1.0 when the chosen template matches the source.
    """
    n_ch = base_templates[0].size
    ranks = np.empty((n_ch, n_events), dtype=float)
    rel = np.empty((n_ch, n_events), dtype=float)
    for i in range(n_events):
        base = base_templates[rng.integers(0, len(base_templates))]
        ranks[:, i] = base + rng.normal(0.0, rank_noise, size=n_ch)
        r_vec = base * rel_scale + rng.normal(0.0, rel_noise, size=n_ch)
        rel[:, i] = r_vec - r_vec.min()
    return ranks, rel


def test_pr5_gate_passes_when_states_are_resampled_baseline() -> None:
    """Per §3.5: when pre/post events are drawn from the same distribution as
    baseline (no novel template), gate must observe near-zero deltas for r and
    e and the cohort thresholds (|Δr|<=0.05, |Δe/e_baseline|<=0.10) are met.

    n_per_state is set to 200 so the per-state median is tight enough to
    actually exercise the §3.5 thresholds; small samples (n=60) leak ~10%
    relative drift on e purely from sampling noise."""
    rng = np.random.default_rng(0)
    n_ch = 8
    n_per_state = 200

    template_a = np.arange(1.0, n_ch + 1, dtype=float)
    template_b = template_a[::-1].copy()
    templates = np.stack([template_a, template_b])

    n_total = n_per_state * 3
    ranks, rel = _pr5a_make_event_arrays(
        rng,
        [template_a, template_b],
        n_total,
        rank_noise=0.10,
        rel_noise=0.005,
        rel_scale=0.05,
    )
    bools = np.ones((n_ch, n_total), dtype=bool)

    state_indices = {
        "baseline": np.arange(0, n_per_state, dtype=int),
        "pre": np.arange(n_per_state, 2 * n_per_state, dtype=int),
        "post": np.arange(2 * n_per_state, 3 * n_per_state, dtype=int),
    }
    windows = [_pr5a_synthetic_window(state_indices)]

    out = compute_novel_template_gate(
        v_ranks=ranks,
        v_rel=rel,
        v_bools=bools,
        templates=templates,
        proximity_windows=windows,
        min_participating_l3=5,
        min_shared_channels=3,
    )

    assert out["n_events_by_state"] == {
        "baseline": n_per_state,
        "pre": n_per_state,
        "post": n_per_state,
    }
    assert out["n_l3_excluded_by_state"] == {"baseline": 0, "pre": 0, "post": 0}

    medians = out["median_by_state"]
    assert medians["baseline"]["r"] > 0.9
    assert medians["pre"]["r"] > 0.9
    assert medians["post"]["r"] > 0.9

    delta_pre = out["delta_pre_minus_baseline"]
    delta_post = out["delta_post_minus_baseline"]
    assert abs(delta_pre["r"]) < 0.05
    assert abs(delta_post["r"]) < 0.05

    e_base = max(abs(medians["baseline"]["e"]), 1e-9)
    assert abs(delta_pre["e"]) / e_base < 0.10
    assert abs(delta_post["e"]) / e_base < 0.10


def test_pr5_gate_fails_when_post_events_drawn_from_orthogonal_template() -> None:
    """Per §3.5: when post events come from a near-orthogonal "novel" template,
    best-template r must drop sharply and reconstruction error must rise; pre
    (drawn from the original distribution) must remain near baseline."""
    rng = np.random.default_rng(1)
    n_ch = 8
    n_per_state = 60

    template_a = np.arange(1.0, n_ch + 1, dtype=float)
    template_b = template_a[::-1].copy()
    template_c = np.array([3, 7, 2, 8, 5, 1, 6, 4], dtype=float)
    # Sanity: |Spearman(C, A)| and |Spearman(C, B)| are both near zero (~0.05),
    # so events from C should never produce a high best-template r against
    # the in-distribution library {A, B}.
    assert abs(spearmanr_check := float(np.corrcoef(template_a, template_c)[0, 1])) < 0.30
    assert abs(float(np.corrcoef(template_b, template_c)[0, 1])) < 0.30
    del spearmanr_check

    templates = np.stack([template_a, template_b])
    n_total = n_per_state * 3
    ranks = np.empty((n_ch, n_total), dtype=float)
    rel = np.empty((n_ch, n_total), dtype=float)
    bools = np.ones((n_ch, n_total), dtype=bool)

    for i in range(n_total):
        if i < 2 * n_per_state:
            base = template_a if rng.integers(0, 2) == 0 else template_b
        else:
            base = template_c
        ranks[:, i] = base + rng.normal(0.0, 0.10, size=n_ch)
        r_vec = base * 0.05 + rng.normal(0.0, 0.005, size=n_ch)
        rel[:, i] = r_vec - r_vec.min()

    state_indices = {
        "baseline": np.arange(0, n_per_state, dtype=int),
        "pre": np.arange(n_per_state, 2 * n_per_state, dtype=int),
        "post": np.arange(2 * n_per_state, 3 * n_per_state, dtype=int),
    }
    windows = [_pr5a_synthetic_window(state_indices)]

    out = compute_novel_template_gate(
        v_ranks=ranks,
        v_rel=rel,
        v_bools=bools,
        templates=templates,
        proximity_windows=windows,
        min_participating_l3=5,
        min_shared_channels=3,
    )

    medians = out["median_by_state"]
    delta_pre = out["delta_pre_minus_baseline"]
    delta_post = out["delta_post_minus_baseline"]

    # Baseline + pre should still see near-perfect r; pre delta should be small.
    assert medians["baseline"]["r"] > 0.9
    assert medians["pre"]["r"] > 0.9
    assert abs(delta_pre["r"]) < 0.10

    # Post must show a large drop in best-template r and a large rise in e.
    assert medians["post"]["r"] < 0.3
    assert delta_post["r"] < -0.5
    e_base = max(abs(medians["baseline"]["e"]), 1e-9)
    assert delta_post["e"] / e_base > 1.0


def test_pr5_gate_uses_only_l3_eligible_events() -> None:
    """Per §3.2: events with n_participating < min_participating_l3 (=5) must
    be excluded from the r/e/gap distributions even if they sit inside the
    state's window. The excluded count must be reported per state."""
    rng = np.random.default_rng(2)
    n_ch = 8

    template_a = np.arange(1.0, n_ch + 1, dtype=float)
    template_b = template_a[::-1].copy()
    template_c_bad = np.array([3, 7, 2, 8, 5, 1, 6, 4], dtype=float)
    templates = np.stack([template_a, template_b])

    n_l3 = 30
    n_bad = 15
    per_state_n = n_l3 + n_bad
    n_total = per_state_n * 3

    ranks = np.empty((n_ch, n_total), dtype=float)
    rel = np.empty((n_ch, n_total), dtype=float)
    bools = np.zeros((n_ch, n_total), dtype=bool)

    for s in range(3):
        offset = s * per_state_n
        # L3-eligible portion: drawn from {A, B}, all 8 channels active.
        for j in range(n_l3):
            base = template_a if rng.integers(0, 2) == 0 else template_b
            ranks[:, offset + j] = base + rng.normal(0.0, 0.05, size=n_ch)
            r_vec = base * 0.05 + rng.normal(0.0, 0.005, size=n_ch)
            rel[:, offset + j] = r_vec - r_vec.min()
            bools[:, offset + j] = True
        # NOT L3-eligible portion: only 3 channels participate; drawn from
        # the orthogonal "bad" template. If these leak into the gate, baseline
        # median r would crash.
        for j in range(n_bad):
            ranks[:, offset + n_l3 + j] = template_c_bad + rng.normal(0.0, 0.05, size=n_ch)
            r_vec = template_c_bad * 0.05 + rng.normal(0.0, 0.005, size=n_ch)
            rel[:, offset + n_l3 + j] = r_vec - r_vec.min()
            participating = rng.choice(n_ch, size=3, replace=False)
            bools[participating, offset + n_l3 + j] = True

    state_indices = {
        "baseline": np.arange(0, per_state_n, dtype=int),
        "pre": np.arange(per_state_n, 2 * per_state_n, dtype=int),
        "post": np.arange(2 * per_state_n, 3 * per_state_n, dtype=int),
    }
    windows = [_pr5a_synthetic_window(state_indices)]

    out = compute_novel_template_gate(
        v_ranks=ranks,
        v_rel=rel,
        v_bools=bools,
        templates=templates,
        proximity_windows=windows,
        min_participating_l3=5,
        min_shared_channels=3,
    )

    assert out["n_events_by_state"] == {
        "baseline": n_l3,
        "pre": n_l3,
        "post": n_l3,
    }
    assert out["n_l3_excluded_by_state"] == {
        "baseline": n_bad,
        "pre": n_bad,
        "post": n_bad,
    }
    # If the bad events had leaked in, baseline median r would be much lower.
    assert out["median_by_state"]["baseline"]["r"] > 0.9
    delta_pre = out["delta_pre_minus_baseline"]
    delta_post = out["delta_post_minus_baseline"]
    assert abs(delta_pre["r"]) < 0.05
    assert abs(delta_post["r"]) < 0.05


def _pr5a_summary_record(
    *,
    subject_id: str,
    baseline_e: float = 4.0,
    delta_pre_r: float = 0.01,
    delta_post_r: float = -0.01,
    delta_pre_e: float = 0.10,
    delta_post_e: float = -0.08,
    delta_pre_gap: float = 0.20,
    delta_post_gap: float = 0.15,
    n_events: int = 40,
) -> Dict[str, Any]:
    return {
        "subject_id": subject_id,
        "dataset": "synthetic",
        "config_name": "main",
        "n_events_by_state": {"baseline": n_events, "pre": n_events, "post": n_events},
        "n_l3_excluded_by_state": {"baseline": 0, "pre": 0, "post": 0},
        "n_seizures_usable": 3,
        "median_by_state": {
            "baseline": {"r": 0.80, "e": baseline_e, "gap": 1.00},
            "pre": {"r": 0.80 + delta_pre_r, "e": baseline_e + delta_pre_e, "gap": 1.00 + delta_pre_gap},
            "post": {"r": 0.80 + delta_post_r, "e": baseline_e + delta_post_e, "gap": 1.00 + delta_post_gap},
        },
        "delta_pre_minus_baseline": {"r": delta_pre_r, "e": delta_pre_e, "gap": delta_pre_gap},
        "delta_post_minus_baseline": {"r": delta_post_r, "e": delta_post_e, "gap": delta_post_gap},
        "n_clusters": 2,
    }


def test_pr5_gate_summary_reports_sign_test_and_passes() -> None:
    records = [
        _pr5a_summary_record(
            subject_id=f"s{i}",
            delta_pre_r=(0.01 if i % 2 == 0 else -0.01),
            delta_post_r=(-0.01 if i % 2 == 0 else 0.01),
            delta_pre_e=(0.08 if i < 2 else -0.06),
            delta_post_e=(-0.04 if i < 2 else 0.03),
            delta_pre_gap=(0.20 if i < 2 else 0.10),
            delta_post_gap=(0.15 if i < 2 else 0.05),
        )
        for i in range(4)
    ]
    out = summarize_pr5_novel_template_gate(
        {"main": records, "auxiliary": records},
        min_state_events_for_gate=30,
    )

    pre_r = out["main"]["delta_summary"]["pre_vs_baseline"]["r"]
    post_e = out["auxiliary"]["delta_summary"]["post_vs_baseline"]["e"]
    assert "sign_test_p" in pre_r
    assert np.isfinite(pre_r["sign_test_p"])
    assert abs(pre_r["sign_test_p"] - 1.0) < 1e-12
    assert "sign_test_p" in post_e
    assert np.isfinite(post_e["sign_test_p"])
    assert out["main"]["gate_pass"] is True
    assert out["auxiliary"]["gate_pass"] is True
    assert out["overall_pass"] is True


def test_pr5_gate_summary_fails_when_gap_is_significantly_lower() -> None:
    harmful = [
        _pr5a_summary_record(
            subject_id=f"s{i}",
            delta_pre_gap=-0.30,
            delta_post_gap=-0.30,
        )
        for i in range(6)
    ]
    out = summarize_pr5_novel_template_gate(
        {"main": harmful, "auxiliary": harmful},
        min_state_events_for_gate=30,
    )

    main_gap = out["main"]["gate_evaluation"]["axis_details"]["gap"]["pre_vs_baseline"]
    assert main_gap["harmful"] is True
    assert out["main"]["gate_pass"] is False
    assert out["overall_pass"] is False


# =============================================================================
# PR-5-B: compute_template_recruitment_shift
#
# Spec: docs/archive/topic1/pr5_template_recruitment_plan_2026-04-20.md §4 / §5.3
# tests #4-#9. Tests #1-#3 above already cover PR-5-A gate; PR-5-B adds 6 tests
# for: same gate-eligible event pool, gap-aware coverage, multi-window weighting,
# runner gate-prerequisite check, dual dominant definitions, and composition
# diagnostic isolation from main sensitivity family.
# =============================================================================


def _pr5b_synthetic_window(
    *,
    seizure_id: int,
    state_event_indices: Dict[str, np.ndarray],
    state_covered_hours: Dict[str, float],
) -> Dict[str, Any]:
    counts = {name: int(np.asarray(idx).size) for name, idx in state_event_indices.items()}
    return {
        "seizure_id": int(seizure_id),
        "seizure_time": float(seizure_id) * 3600.0,
        "state_event_indices": {
            name: np.asarray(idx, dtype=int) for name, idx in state_event_indices.items()
        },
        "state_event_counts": counts,
        "pair_usability": {
            "pre_vs_baseline": counts["baseline"] > 0 and counts["pre"] > 0,
            "post_vs_pre": counts["pre"] > 0 and counts["post"] > 0,
            "post_vs_baseline": counts["baseline"] > 0 and counts["post"] > 0,
        },
        "state_covered_hours": {k: float(v) for k, v in state_covered_hours.items()},
        "usable": True,
    }


def test_pr5_recruitment_uses_same_l3_eligible_event_pool_as_gate() -> None:
    """§4.2 / spec test #4: events with n_part < min_participating_l3 must be
    excluded from per-window counts_by_template, exactly mirroring the PR-5-A
    gate's eligibility mask."""
    n_clusters = 3
    n_per_state_eligible = 10
    n_per_state_bad = 7

    cluster_labels: List[int] = []
    n_part: List[int] = []
    for _state in range(3):
        cluster_labels.extend([0] * 6 + [1] * 3 + [2] * 1)
        n_part.extend([5] * n_per_state_eligible)
        cluster_labels.extend([2] * n_per_state_bad)
        n_part.extend([3] * n_per_state_bad)
    cluster_labels_arr = np.asarray(cluster_labels, dtype=int)
    n_part_arr = np.asarray(n_part, dtype=int)

    per_state_total = n_per_state_eligible + n_per_state_bad
    state_indices = {
        "baseline": np.arange(0, per_state_total, dtype=int),
        "pre": np.arange(per_state_total, 2 * per_state_total, dtype=int),
        "post": np.arange(2 * per_state_total, 3 * per_state_total, dtype=int),
    }
    window = _pr5b_synthetic_window(
        seizure_id=0,
        state_event_indices=state_indices,
        state_covered_hours={"baseline": 3.0, "pre": 0.75, "post": 0.75},
    )

    out = compute_template_recruitment_shift(
        cluster_labels=cluster_labels_arr,
        n_part_per_event=n_part_arr,
        n_clusters=n_clusters,
        dominant_global_id=0,
        proximity_windows=[window],
        min_participating_l3=5,
    )

    pw = out["per_window"][0]
    for state in ("baseline", "pre", "post"):
        assert pw["state_total_counts_filtered"][state] == n_per_state_eligible
        assert pw["state_counts_dom_global"][state] == 6
    expected_baseline_rate = 6.0 / 3.0
    assert abs(out["weighted_per_state"]["dom_global_rate_per_hour"]["baseline"] - expected_baseline_rate) < 1e-9
    assert abs(out["weighted_per_state"]["dom_global_share"]["baseline"] - 0.6) < 1e-12


def test_pr5_recruitment_uses_gap_aware_coverage() -> None:
    """§4.2 / spec test #5: dominant rate must divide by the gap-aware
    state_covered_hours, not the nominal window width."""
    cluster_labels = np.zeros(20, dtype=int)
    n_part = np.full(20, 5, dtype=int)
    state_indices = {
        "baseline": np.arange(0, 10, dtype=int),
        "pre": np.arange(10, 15, dtype=int),
        "post": np.arange(15, 20, dtype=int),
    }
    nominal = {"baseline": 3.0, "pre": 0.75, "post": 0.75}
    covered = {"baseline": 1.5, "pre": 0.75, "post": 0.75}
    window = _pr5b_synthetic_window(
        seizure_id=0,
        state_event_indices=state_indices,
        state_covered_hours=covered,
    )

    out = compute_template_recruitment_shift(
        cluster_labels=cluster_labels,
        n_part_per_event=n_part,
        n_clusters=1,
        dominant_global_id=0,
        proximity_windows=[window],
        min_participating_l3=5,
    )

    rate = out["weighted_per_state"]["dom_global_rate_per_hour"]["baseline"]
    assert abs(rate - (10.0 / covered["baseline"])) < 1e-9
    assert abs(rate - (10.0 / nominal["baseline"])) > 1.0


def test_pr5_recruitment_weights_per_window_by_covered_hours() -> None:
    """§4.3 / spec test #6: aggregation across windows must weight by
    state_covered_hours so a long window does not get equal vote with a tiny
    window. Equivalently: sum(counts) / sum(covered_hours)."""
    cluster_labels = np.zeros(40, dtype=int)
    n_part = np.full(40, 5, dtype=int)
    win_a = _pr5b_synthetic_window(
        seizure_id=0,
        state_event_indices={
            "baseline": np.arange(0, 8, dtype=int),
            "pre": np.arange(8, 12, dtype=int),
            "post": np.arange(12, 16, dtype=int),
        },
        state_covered_hours={"baseline": 4.0, "pre": 0.75, "post": 0.75},
    )
    win_b = _pr5b_synthetic_window(
        seizure_id=1,
        state_event_indices={
            "baseline": np.arange(16, 26, dtype=int),
            "pre": np.arange(26, 32, dtype=int),
            "post": np.arange(32, 40, dtype=int),
        },
        state_covered_hours={"baseline": 1.0, "pre": 0.75, "post": 0.75},
    )

    out = compute_template_recruitment_shift(
        cluster_labels=cluster_labels,
        n_part_per_event=n_part,
        n_clusters=1,
        dominant_global_id=0,
        proximity_windows=[win_a, win_b],
        min_participating_l3=5,
    )

    expected_rate = (8 + 10) / (4.0 + 1.0)
    got = out["weighted_per_state"]["dom_global_rate_per_hour"]["baseline"]
    assert abs(got - expected_rate) < 1e-9
    naive_avg = 0.5 * (8.0 / 4.0 + 10.0 / 1.0)
    assert abs(got - naive_avg) > 0.5


def test_pr5_recruitment_aborts_if_gate_not_passed(tmp_path) -> None:
    """§5.3 spec test #7: --pr5-recruitment must SystemExit when the gate JSON
    is missing or reports overall_pass=False; never write a recruitment JSON."""
    import json
    import sys
    from unittest import mock

    from scripts import run_interictal_propagation as runner

    results_dir = tmp_path / "results" / "interictal_propagation"
    results_dir.mkdir(parents=True)

    # Case 1: no gate JSON at all -> SystemExit
    with mock.patch.object(runner, "RESULTS_DIR", results_dir):
        with mock.patch.object(sys, "argv", ["run", "--pr5-recruitment"]):
            try:
                runner.main()
            except SystemExit as exc:
                assert exc.code != 0
            else:
                raise AssertionError("expected SystemExit when gate JSON missing")

    assert not (results_dir / "pr5b_recruitment_shift.json").exists()

    # Case 2: gate JSON present but overall_pass=False -> SystemExit
    gate_json = results_dir / "pr5a_novel_template_gate.json"
    gate_json.write_text(json.dumps({"cohort": {"overall_pass": False}}))
    with mock.patch.object(runner, "RESULTS_DIR", results_dir):
        with mock.patch.object(sys, "argv", ["run", "--pr5-recruitment"]):
            try:
                runner.main()
            except SystemExit as exc:
                assert exc.code != 0
            else:
                raise AssertionError("expected SystemExit when gate FAIL")

    assert not (results_dir / "pr5b_recruitment_shift.json").exists()


def test_pr5_recruitment_filters_to_gate_retained_subset_per_config(
    tmp_path, monkeypatch
) -> None:
    """PR-5-B must only send PR-5-A retained subjects into each config-level
    cohort summary. Retention is config-specific, so main/aux subsets may
    differ even when overall_pass=True."""
    from scripts import run_interictal_propagation as runner

    results_dir = tmp_path / "results" / "interictal_propagation"
    results_dir.mkdir(parents=True)
    per_subject_dir = results_dir / "per_subject"
    per_subject_dir.mkdir(parents=True)
    root = tmp_path / "yuquan"
    root.mkdir()

    subjects = ["keep", "drop_main"]
    for subject in subjects:
        subject_dir = root / subject
        subject_dir.mkdir()
        (subject_dir / f"{subject}_lagPat.npz").write_text("stub")
        (per_subject_dir / f"yuquan_{subject}.json").write_text(
            json.dumps(
                {
                    "adaptive_cluster": {
                        "labels": [0, 0, 0],
                        "chosen_k": 1,
                        "stable_k": 1,
                    }
                }
            )
        )

    gate_doc = {
        "per_subject": {
            "main": [
                {"dataset": "yuquan", "subject_id": "keep"},
                {"dataset": "yuquan", "subject_id": "drop_main"},
            ],
            "auxiliary": [
                {"dataset": "yuquan", "subject_id": "keep"},
                {"dataset": "yuquan", "subject_id": "drop_main"},
            ],
        },
        "cohort": {
            "overall_pass": True,
            "main": {
                "ineligible_subjects": [
                    {"dataset": "yuquan", "subject_id": "drop_main"}
                ]
            },
            "auxiliary": {"ineligible_subjects": []},
        },
    }
    (results_dir / "pr5a_novel_template_gate.json").write_text(json.dumps(gate_doc))

    def fake_loaded(_subject_dir):
        return {
            "bools": np.ones((5, 3), dtype=bool),
            "event_abs_times": np.array([1.0, 2.0, 3.0], dtype=float),
            "block_time_ranges": [(0.0, 3600.0)],
        }

    def fake_windows(*_args, **_kwargs):
        return {
            "usable_windows": [
                {
                    "seizure_time": 10.0,
                    "state_event_indices": {
                        "baseline": np.array([0], dtype=int),
                        "pre": np.array([1], dtype=int),
                        "post": np.array([2], dtype=int),
                    },
                }
            ],
            "state_ranges_hours": {
                "baseline": (-2.0, -1.0),
                "pre": (-1.0, 0.0),
                "post": (0.0, 1.0),
            },
            "state_event_counts": {"baseline": 1, "pre": 1, "post": 1},
        }

    def fake_shift(**_kwargs):
        return {
            "dom_global_id": 0,
            "dom_window_ids_per_window": [0],
            "dom_agreement": 1.0,
            "n_windows_used": 1,
            "n_windows_total": 1,
            "min_participating_l3": 5,
            "n_clusters": 1,
            "weighted_per_state": {
                "dom_global_rate_per_hour": {
                    "baseline": 1.0,
                    "pre": 1.5,
                    "post": 2.0,
                },
                "dom_window_rate_per_hour": {
                    "baseline": 1.0,
                    "pre": 1.5,
                    "post": 2.0,
                },
                "dom_global_share": {
                    "baseline": 0.5,
                    "pre": 0.4,
                    "post": 0.3,
                },
            },
            "deltas": {
                "dom_global_rate": {
                    "pre_minus_baseline": 0.5,
                    "post_minus_baseline": 1.0,
                    "post_minus_pre": 0.5,
                },
                "dom_window_rate": {
                    "pre_minus_baseline": 0.5,
                    "post_minus_baseline": 1.0,
                    "post_minus_pre": 0.5,
                },
                "dom_global_share": {
                    "pre_minus_baseline": -0.1,
                    "post_minus_baseline": -0.2,
                    "post_minus_pre": -0.1,
                },
            },
            "share_state_n_eligible_windows": {
                "baseline": 1,
                "pre": 1,
                "post": 1,
            },
            "share_pair_eligible": {
                "pre_minus_baseline": True,
                "post_minus_baseline": True,
                "post_minus_pre": True,
            },
        }

    captured = {}

    def fake_summary(per_subject_records_by_config, **_kwargs):
        for config_name, records in per_subject_records_by_config.items():
            captured[config_name] = [
                f"{rec['dataset']}/{rec['subject_id']}" for rec in records
            ]
        return {
            "main": {"n_subjects": len(per_subject_records_by_config["main"])},
            "auxiliary": {
                "n_subjects": len(per_subject_records_by_config["auxiliary"])
            },
            "sensitivity": {
                "overall_strong": False,
                "overall_medium": False,
                "overall_descriptive": True,
            },
        }

    monkeypatch.setattr(runner, "RESULTS_DIR", results_dir)
    monkeypatch.setattr(runner, "load_subject_propagation_events", fake_loaded)
    monkeypatch.setattr(
        runner,
        "_valid_event_indices",
        lambda _bools, min_participating=3: np.array([0, 1, 2], dtype=int),
    )
    monkeypatch.setattr(runner, "load_seizure_times", lambda *_args, **_kwargs: [10.0])
    monkeypatch.setattr(runner, "_build_seizure_proximity_windows", fake_windows)
    monkeypatch.setattr(runner, "compute_template_recruitment_shift", fake_shift)
    monkeypatch.setattr(
        runner, "summarize_pr5_template_recruitment_shift", fake_summary
    )
    monkeypatch.setattr(runner, "_save", lambda *_args, **_kwargs: None)

    runner._run_pr5_recruitment(
        [("yuquan", root, subjects, {})],
        per_subject_dir,
        min_state_events_for_gate=30,
        n_boot=20,
        bootstrap_seed=1,
    )

    assert captured["main"] == ["yuquan/keep"]
    assert captured["auxiliary"] == ["yuquan/keep", "yuquan/drop_main"]


def test_pr5_recruitment_dual_dominant_definitions() -> None:
    """§4.3 / spec test #8: window A baseline argmax = cluster 0, window B
    baseline argmax = cluster 1, but global dominant = cluster 0 (more events
    overall). dom_window_ids must reflect per-window argmax; dom_agreement =
    0.5; candidates A and B must produce distinguishable rates."""
    n_clusters = 2

    cluster_labels = np.concatenate([
        np.array([0] * 6 + [1] * 2, dtype=int),
        np.array([0] * 4, dtype=int),
        np.array([0] * 4, dtype=int),
        np.array([0] * 1 + [1] * 5, dtype=int),
        np.array([1] * 4, dtype=int),
        np.array([1] * 4, dtype=int),
    ])
    n_part = np.full(cluster_labels.size, 5, dtype=int)

    win_a = _pr5b_synthetic_window(
        seizure_id=0,
        state_event_indices={
            "baseline": np.arange(0, 8, dtype=int),
            "pre": np.arange(8, 12, dtype=int),
            "post": np.arange(12, 16, dtype=int),
        },
        state_covered_hours={"baseline": 1.0, "pre": 1.0, "post": 1.0},
    )
    win_b = _pr5b_synthetic_window(
        seizure_id=1,
        state_event_indices={
            "baseline": np.arange(16, 22, dtype=int),
            "pre": np.arange(22, 26, dtype=int),
            "post": np.arange(26, 30, dtype=int),
        },
        state_covered_hours={"baseline": 1.0, "pre": 1.0, "post": 1.0},
    )

    out = compute_template_recruitment_shift(
        cluster_labels=cluster_labels,
        n_part_per_event=n_part,
        n_clusters=n_clusters,
        dominant_global_id=0,
        proximity_windows=[win_a, win_b],
        min_participating_l3=5,
    )

    assert out["dom_global_id"] == 0
    assert out["dom_window_ids_per_window"] == [0, 1]
    assert abs(out["dom_agreement"] - 0.5) < 1e-12

    a_rate = out["weighted_per_state"]["dom_global_rate_per_hour"]["baseline"]
    b_rate = out["weighted_per_state"]["dom_window_rate_per_hour"]["baseline"]
    assert abs(a_rate - 7.0 / 2.0) < 1e-9
    assert abs(b_rate - (6.0 + 5.0) / 2.0) < 1e-9
    assert abs(a_rate - b_rate) > 0.5

    for cand in ("dom_global_rate", "dom_window_rate"):
        for pair in ("pre_minus_baseline", "post_minus_baseline", "post_minus_pre"):
            assert pair in out["deltas"][cand]


def _pr5b_subject_record(
    *,
    subject_id: str,
    base_rate: float = 4.0,
    delta_pre_rate: float = 0.0,
    delta_post_rate: float = 0.0,
    delta_pre_share: float = 0.0,
    delta_post_share: float = 0.0,
    base_share: float = 0.5,
    n_windows_used: int = 4,
) -> Dict[str, Any]:
    """Per-subject record matching the schema produced by
    compute_template_recruitment_shift, used to feed cohort summaries."""
    weighted = {
        "dom_global_rate_per_hour": {
            "baseline": base_rate,
            "pre": base_rate + delta_pre_rate,
            "post": base_rate + delta_post_rate,
        },
        "dom_window_rate_per_hour": {
            "baseline": base_rate,
            "pre": base_rate + delta_pre_rate,
            "post": base_rate + delta_post_rate,
        },
        "nondom_global_rate_per_hour": {
            "baseline": base_rate * 0.5,
            "pre": base_rate * 0.5,
            "post": base_rate * 0.5,
        },
        "dom_global_share": {
            "baseline": base_share,
            "pre": base_share + delta_pre_share,
            "post": base_share + delta_post_share,
        },
    }

    def _pair_deltas(metric_key: str) -> Dict[str, float]:
        b = weighted[metric_key]["baseline"]
        p = weighted[metric_key]["pre"]
        q = weighted[metric_key]["post"]
        return {
            "pre_minus_baseline": p - b,
            "post_minus_baseline": q - b,
            "post_minus_pre": q - p,
        }

    return {
        "subject_id": subject_id,
        "dataset": "synthetic",
        "config_name": "main",
        "dom_global_id": 0,
        "dom_window_ids_per_window": [0] * n_windows_used,
        "dom_agreement": 1.0,
        "n_windows_used": n_windows_used,
        "n_windows_total": n_windows_used,
        "min_participating_l3": 5,
        "n_clusters": 2,
        "weighted_per_state": weighted,
        "deltas": {
            "dom_global_rate": _pair_deltas("dom_global_rate_per_hour"),
            "dom_window_rate": _pair_deltas("dom_window_rate_per_hour"),
            "nondom_global_rate": _pair_deltas("nondom_global_rate_per_hour"),
            "dom_global_share": _pair_deltas("dom_global_share"),
        },
        "share_state_n_eligible_windows": {
            "baseline": n_windows_used,
            "pre": n_windows_used,
            "post": n_windows_used,
        },
        "share_pair_eligible": {
            "pre_minus_baseline": True,
            "post_minus_baseline": True,
            "post_minus_pre": True,
        },
    }


def test_pr5_recruitment_composition_diagnostic_isolated_from_main_family() -> None:
    """§4.5 / spec test #9: composition diagnostic is reported next to but
    physically separated from dominant_rate. Sensitivity gate must read only
    dominant_rate; ineligible windows / subjects on share are accounted for
    without touching dominant_rate accounting."""
    n_subjects = 12
    rng = np.random.default_rng(0)

    main_records = []
    aux_records = []
    for i in range(n_subjects):
        delta_post_rate_main = float(2.5 + rng.normal(0.0, 0.20))
        delta_post_share_main = float(-0.10 + rng.normal(0.0, 0.02))
        rec_main = _pr5b_subject_record(
            subject_id=f"s{i}",
            delta_pre_rate=0.05,
            delta_post_rate=delta_post_rate_main,
            delta_pre_share=-0.01,
            delta_post_share=delta_post_share_main,
        )
        rec_main["config_name"] = "main"
        main_records.append(rec_main)

        delta_post_rate_aux = float(2.0 + rng.normal(0.0, 0.20))
        delta_post_share_aux = float(-0.08 + rng.normal(0.0, 0.02))
        rec_aux = _pr5b_subject_record(
            subject_id=f"s{i}",
            delta_pre_rate=0.05,
            delta_post_rate=delta_post_rate_aux,
            delta_pre_share=-0.01,
            delta_post_share=delta_post_share_aux,
        )
        rec_aux["config_name"] = "auxiliary"
        aux_records.append(rec_aux)

    aux_records[0]["share_state_n_eligible_windows"]["post"] = 0
    aux_records[0]["share_pair_eligible"]["post_minus_baseline"] = False
    aux_records[0]["share_pair_eligible"]["post_minus_pre"] = False

    summary = summarize_pr5_template_recruitment_shift(
        {"main": main_records, "auxiliary": aux_records},
        n_boot=200,
        bootstrap_seed=7,
    )

    # Assertion 1: composition_diagnostic schema is independent + lacks bonferroni_pass.
    cd_main = summary["main"]["composition_diagnostic"]["share"]
    for pair in ("pre_minus_baseline", "post_minus_baseline", "post_minus_pre"):
        entry = cd_main[pair]
        assert "wilcoxon_p" in entry
        assert "median_delta_share" in entry
        assert "ci95_lo" in entry and "ci95_hi" in entry
        assert "direction_consistent_count" in entry
        assert "bonferroni_pass" not in entry
    assert summary["auxiliary"]["composition_diagnostic"]["share"]["post_minus_baseline"]["n_ineligible"] >= 1

    # Assertion 2: monkey-patching composition_diagnostic must not change sensitivity gate.
    sens_before = _compute_pr5b_sensitivity_gate(summary)
    summary_polluted = json.loads(json.dumps(summary))
    summary_polluted["main"]["composition_diagnostic"] = {"share": "GARBAGE"}
    summary_polluted["auxiliary"]["composition_diagnostic"] = {"share": "GARBAGE"}
    sens_after = _compute_pr5b_sensitivity_gate(summary_polluted)
    assert sens_before == sens_after
    assert sens_before["overall_strong"] is True

    # Assertion 3: cohort summary explicitly records the share weight key.
    assert summary["composition_diagnostic_weight_key"] == "state_covered_hours"
    assert summary["dominant_rate_weight_key"] == "state_covered_hours"

    # Assertion 4: ineligible window / subject accounting visible on share but
    # untouched on dominant_rate.
    cd_aux_post = summary["auxiliary"]["composition_diagnostic"]["share"]["post_minus_baseline"]
    assert cd_aux_post["n"] == n_subjects - 1
    aux_dom_post = summary["auxiliary"]["dominant_rate"]["candidate_a_global"]["post_minus_baseline"]
    assert aux_dom_post["n"] == n_subjects


def test_pr5_recruitment_share_post_minus_pre_has_no_a_priori_direction() -> None:
    records = [
        _pr5b_subject_record(
            subject_id="s0",
            delta_pre_share=-0.02,
            delta_post_share=0.05,
        ),
        _pr5b_subject_record(
            subject_id="s1",
            delta_pre_share=-0.01,
            delta_post_share=-0.04,
        ),
    ]

    summary = summarize_pr5_template_recruitment_shift(
        {"main": records, "auxiliary": records},
        n_boot=100,
        bootstrap_seed=11,
    )

    pre_post = summary["main"]["composition_diagnostic"]["share"]["post_minus_pre"]
    assert pre_post["direction_consistent_count"] is None
    assert pre_post["n_positive"] == 1
    assert pre_post["n_negative"] == 1

    pre_base = summary["main"]["composition_diagnostic"]["share"]["pre_minus_baseline"]
    assert isinstance(pre_base["direction_consistent_count"], int)
