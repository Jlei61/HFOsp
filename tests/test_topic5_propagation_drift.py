import numpy as np
import pytest

from src.topic5_propagation_drift import (
    block_templates,
    drift_pairs,
    matched_event_distance_contrast,
    rank_residual_fraction,
    partial_spearman,
    template_similarity,
)


def test_partial_spearman_removes_a_shared_driver():
    rng = np.random.default_rng(0)
    control = rng.random(200)
    # outcome and driver are related only through the control
    outcome = 3.0 * control + 0.01 * rng.random(200)
    driver = -2.0 * control + 0.01 * rng.random(200)
    assert abs(partial_spearman(outcome, driver, control)) < 0.2


def test_partial_spearman_keeps_an_independent_driver_effect():
    rng = np.random.default_rng(1)
    control = rng.random(400)
    driver = rng.random(400)
    outcome = 2.0 * control - 3.0 * driver + 0.01 * rng.random(400)
    assert partial_spearman(outcome, driver, control) < -0.8


def test_partial_spearman_returns_none_without_enough_finite_rows():
    assert partial_spearman([1.0, 2.0], [1.0, 2.0], [1.0, 2.0]) is None


def test_partial_spearman_returns_none_when_control_absorbs_the_driver():
    control = np.arange(100.0)
    assert partial_spearman(np.arange(100.0), control.copy(), control) is None


def test_rank_residual_fraction_is_zero_for_a_fully_absorbed_driver():
    control = np.arange(100.0)
    assert rank_residual_fraction(control.copy(), control) < 1e-9


def test_rank_residual_fraction_is_near_one_for_an_independent_driver():
    rng = np.random.default_rng(5)
    assert rank_residual_fraction(rng.random(300), rng.random(300)) > 0.9


def _stream(n_events, n_contacts, source_index, abs_time, seed=0):
    rng = np.random.default_rng(seed)
    rank = rng.random((n_events, n_contacts)).astype(np.float32)
    participation = np.ones((n_events, n_contacts), dtype=np.uint8)
    return rank, participation, np.asarray(source_index), np.asarray(abs_time, float)


def test_block_templates_never_span_two_sources():
    rank, part, source, time = _stream(
        10, 3, ["recA"] * 6 + ["recB"] * 4, np.arange(10.0)
    )
    blocks = block_templates(
        rank, part, source, time, block_events=4, min_participation=0.0
    )
    assert [row["source_id"] for row in blocks] == ["recA", "recB"]
    assert [row["n_events"] for row in blocks] == [4, 4]
    # the 2 leftover events of source 0 and 0 leftover of source 1 are dropped,
    # so no block mixes the two units
    assert blocks[0]["event_stop_index"] <= 6
    assert blocks[1]["event_start_index"] >= 6


def test_block_templates_masks_non_participating_contacts():
    rank = np.array([[1.0, 5.0], [2.0, 7.0]], dtype=np.float32)
    participation = np.array([[1, 0], [1, 0]], dtype=np.uint8)
    blocks = block_templates(
        rank,
        participation,
        np.array(["recA", "recA"]),
        np.array([0.0, 1.0]),
        block_events=2,
        min_participation=0.0,
    )
    template = blocks[0]["mean_rank"]
    assert template[0] == pytest.approx(1.5)
    assert np.isnan(template[1]), "phantom rank of a never-participating contact leaked"
    assert blocks[0]["support"][1] == 0.0


def test_block_templates_records_time_and_event_bounds():
    rank, part, source, time = _stream(4, 2, ["recA"] * 4, [10.0, 20.0, 30.0, 40.0])
    blocks = block_templates(
        rank, part, source, time, block_events=4, min_participation=0.0
    )
    row = blocks[0]
    assert row["t_start"] == 10.0 and row["t_end"] == 40.0
    assert row["t_mid"] == pytest.approx(25.0)
    assert row["event_start_index"] == 0 and row["event_stop_index"] == 4
    assert row["event_mid_index"] == pytest.approx(1.5)


def test_block_templates_rejects_block_size_below_two():
    rank, part, source, time = _stream(4, 2, ["recA"] * 4, np.arange(4.0))
    with pytest.raises(ValueError):
        block_templates(rank, part, source, time, block_events=1, min_participation=0.0)


def test_template_similarity_is_one_for_identical_orderings():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    s = np.array([1.0, 1.0, 1.0, 1.0])
    assert template_similarity(a, s, a * 3.0 + 7.0, s, 0.5, 3) == pytest.approx(1.0)


def test_template_similarity_is_minus_one_for_reversed_orderings():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    s = np.ones(4)
    assert template_similarity(a, s, a[::-1].copy(), s, 0.5, 3) == pytest.approx(-1.0)


def test_template_similarity_uses_only_contacts_supported_on_both_sides():
    a = np.array([1.0, 2.0, 3.0, 99.0])
    b = np.array([1.0, 2.0, 3.0, -99.0])
    sa = np.array([1.0, 1.0, 1.0, 1.0])
    sb = np.array([1.0, 1.0, 1.0, 0.0])
    assert template_similarity(a, sa, b, sb, 0.5, 3) == pytest.approx(1.0)


def test_template_similarity_returns_none_below_minimum_shared_contacts():
    a = np.array([1.0, 2.0, np.nan])
    b = np.array([1.0, 2.0, 3.0])
    s = np.ones(3)
    assert template_similarity(a, s, b, s, 0.5, 3) is None


def test_drift_pairs_reports_event_time_and_source_separation():
    blocks = [
        {
            "source_id": "recA",
            "event_mid_index": 10.0,
            "t_mid": 100.0,
            "mean_rank": np.array([1.0, 2.0, 3.0, 4.0]),
            "support": np.ones(4),
        },
        {
            "source_id": "recA",
            "event_mid_index": 30.0,
            "t_mid": 300.0,
            "mean_rank": np.array([1.0, 2.0, 3.0, 4.0]),
            "support": np.ones(4),
        },
        {
            "source_id": "recB",
            "event_mid_index": 50.0,
            "t_mid": 90000.0,
            "mean_rank": np.array([4.0, 3.0, 2.0, 1.0]),
            "support": np.ones(4),
        },
    ]
    pairs = drift_pairs(blocks, max_pairs=100, seed=0, min_support=0.5, min_shared=3)
    assert len(pairs) == 3
    same = [row for row in pairs if row["same_source"]]
    assert len(same) == 1
    assert same[0]["d_events"] == pytest.approx(20.0)
    assert same[0]["d_seconds"] == pytest.approx(200.0)
    assert same[0]["similarity"] == pytest.approx(1.0)
    cross = [row for row in pairs if not row["same_source"]]
    assert all(row["similarity"] == pytest.approx(-1.0) for row in cross)


def test_drift_pairs_subsamples_deterministically_when_over_budget():
    rng = np.random.default_rng(0)
    blocks = [
        {
            "source_id": f"rec{index // 5}",
            "event_mid_index": float(index * 10),
            "t_mid": float(index * 100),
            "mean_rank": rng.random(6),
            "support": np.ones(6),
        }
        for index in range(20)
    ]
    first = drift_pairs(blocks, max_pairs=25, seed=7, min_support=0.5, min_shared=3)
    second = drift_pairs(blocks, max_pairs=25, seed=7, min_support=0.5, min_shared=3)
    assert len(first) == 25
    assert [row["similarity"] for row in first] == [
        row["similarity"] for row in second
    ]


def test_matched_event_distance_contrast_only_reports_populated_cells():
    pairs = [
        {"d_events": 10.0, "d_seconds": 100.0, "same_source": True, "similarity": 0.9},
        {"d_events": 12.0, "d_seconds": 120.0, "same_source": True, "similarity": 0.8},
        {"d_events": 11.0, "d_seconds": 90000.0, "same_source": False, "similarity": 0.4},
        {"d_events": 13.0, "d_seconds": 90000.0, "same_source": False, "similarity": 0.5},
        {"d_events": 900.0, "d_seconds": 9000.0, "same_source": True, "similarity": 0.1},
    ]
    cells = matched_event_distance_contrast(
        pairs, bin_edges=[0.0, 20.0, 1000.0], min_pairs_per_cell=2
    )
    assert len(cells) == 1
    cell = cells[0]
    assert cell["n_same_source"] == 2 and cell["n_cross_source"] == 2
    assert cell["median_same_source"] == pytest.approx(0.85)
    assert cell["median_cross_source"] == pytest.approx(0.45)
    assert cell["cross_minus_same"] == pytest.approx(-0.4)


def test_matched_event_distance_contrast_exposes_residual_event_imbalance():
    pairs = [
        {"d_events": 1.0, "d_seconds": 10.0, "same_source": True, "similarity": 0.9},
        {"d_events": 3.0, "d_seconds": 30.0, "same_source": True, "similarity": 0.8},
        {"d_events": 17.0, "d_seconds": 90000.0, "same_source": False, "similarity": 0.4},
        {"d_events": 19.0, "d_seconds": 90000.0, "same_source": False, "similarity": 0.5},
    ]
    cell = matched_event_distance_contrast(
        pairs, bin_edges=[0.0, 20.0], min_pairs_per_cell=2
    )[0]
    # Both arms sit in the same bin but the cross-source arm is 16 events further
    # apart, which would masquerade as a cross-recording cost if left invisible.
    assert cell["median_events_same_source"] == pytest.approx(2.0)
    assert cell["median_events_cross_source"] == pytest.approx(18.0)
    assert cell["event_imbalance"] == pytest.approx(16.0)


def test_matched_event_distance_contrast_drops_cells_missing_one_arm():
    pairs = [
        {"d_events": 5.0, "d_seconds": 50.0, "same_source": True, "similarity": 0.9},
        {"d_events": 6.0, "d_seconds": 60.0, "same_source": True, "similarity": 0.8},
    ]
    cells = matched_event_distance_contrast(
        pairs, bin_edges=[0.0, 20.0], min_pairs_per_cell=2
    )
    assert cells == []
