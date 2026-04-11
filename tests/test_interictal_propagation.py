from __future__ import annotations

from pathlib import Path

import numpy as np

from src.interictal_propagation import (
    _center_rank_matrix,
    compute_source_node_diagnostic,
    compute_stereotypy_by_nparticipating,
    detect_propagation_mixture,
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
