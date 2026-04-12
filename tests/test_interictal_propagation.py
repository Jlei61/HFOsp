from __future__ import annotations

from pathlib import Path

import numpy as np

from src.interictal_propagation import (
    _center_rank_matrix,
    _valid_event_indices,
    assign_events_to_templates,
    build_cluster_templates,
    compute_adaptive_cluster_stereotypy,
    compute_source_node_diagnostic,
    compute_stereotypy_by_nparticipating,
    compute_time_split_reproducibility,
    detect_propagation_mixture,
    load_subject_propagation_events,
    run_subject_interictal_propagation_pr1,
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
    assert loaded["block_boundaries"] == [
        {
            "block_id": 0,
            "record_name": "blockA",
            "start_event_idx": 0,
            "end_event_idx": 1,
            "n_events": 1,
            "start_t": 100.0,
            "has_packed_times": True,
        },
        {
            "block_id": 1,
            "record_name": "blockB",
            "start_event_idx": 1,
            "end_event_idx": 3,
            "n_events": 2,
            "start_t": 200.0,
            "has_packed_times": True,
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
