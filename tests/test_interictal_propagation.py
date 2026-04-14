from __future__ import annotations

from pathlib import Path

import numpy as np

from src.interictal_propagation import (
    _center_rank_matrix,
    _valid_event_indices,
    assign_events_to_templates,
    build_cluster_templates,
    compute_adaptive_cluster_stereotypy,
    compute_rate_state_coupling,
    compute_source_node_diagnostic,
    compute_stereotypy_by_nparticipating,
    compute_temporal_cluster_dynamics,
    compute_time_split_reproducibility,
    detect_propagation_mixture,
    load_subject_propagation_events,
    run_subject_interictal_propagation_pr1,
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
