import numpy as np
import pytest

from src.topic5_propagation_drift_sensitivity import (
    annotated_pairs,
    partial_spearman_multi,
)


def test_partial_spearman_multi_removes_a_monotone_shared_driver():
    rng = np.random.default_rng(0)
    control = rng.random(600)
    outcome = 2.0 * control + 0.01 * rng.random(600)
    driver = -3.0 * control + 0.01 * rng.random(600)
    result = partial_spearman_multi(outcome, driver, [control])
    assert result["status"] == "RESOLVED"
    assert abs(result["rho"]) < 0.2


def test_partial_spearman_multi_only_partly_removes_a_two_control_mixture():
    """Documents a real limit of rank-based partial correlation.

    When outcome and driver are each a *linear combination of two* controls,
    their ranks are not monotone in either control alone, so regressing rank on
    rank cannot represent the confound and a large spurious association
    survives.  The frozen primary controls only for intervening event count, so
    it inherits this same limit and its residual association must not be read as
    confound-free.
    """

    rng = np.random.default_rng(0)
    first, second = rng.random(600), rng.random(600)
    outcome = 2.0 * first - 1.5 * second + 0.01 * rng.random(600)
    driver = -1.0 * first + 3.0 * second + 0.01 * rng.random(600)
    result = partial_spearman_multi(outcome, driver, [first, second])
    assert result["status"] == "RESOLVED"
    assert abs(result["rho"]) > 0.3


def test_partial_spearman_multi_keeps_an_effect_the_controls_do_not_explain():
    rng = np.random.default_rng(1)
    control = rng.random(600)
    driver = rng.random(600)
    outcome = 1.0 * control - 4.0 * driver + 0.01 * rng.random(600)
    result = partial_spearman_multi(outcome, driver, [control])
    assert result["rho"] < -0.8


def test_partial_spearman_multi_flags_a_driver_the_controls_absorb():
    control = np.arange(500.0)
    result = partial_spearman_multi(np.arange(500.0), control.copy(), [control])
    assert result["status"] == "UNRESOLVED_COLLINEAR"
    assert result["rho"] is None


def test_partial_spearman_multi_reports_surviving_driver_variation():
    rng = np.random.default_rng(2)
    result = partial_spearman_multi(
        rng.random(600), rng.random(600), [rng.random(600)]
    )
    assert 0.9 < result["residual_fraction"] <= 1.0


def test_partial_spearman_multi_needs_enough_pairs():
    rng = np.random.default_rng(3)
    result = partial_spearman_multi(
        rng.random(10), rng.random(10), [rng.random(10)], minimum_n=200
    )
    assert result["status"] == "UNRESOLVED_TOO_FEW_PAIRS"


def test_partial_spearman_multi_requires_at_least_one_control():
    with pytest.raises(ValueError):
        partial_spearman_multi([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [])


def _block(source, mid_index, t_start, t_end, ranks, support=None):
    ranks = np.asarray(ranks, dtype=float)
    return {
        "source_id": source,
        "event_mid_index": float(mid_index),
        "t_mid": 0.5 * (t_start + t_end),
        "t_start": float(t_start),
        "t_end": float(t_end),
        "mean_rank": ranks,
        "support": np.ones_like(ranks) if support is None else np.asarray(support, float),
    }


def test_annotated_pairs_reports_shared_contact_count_and_block_spans():
    blocks = [
        _block("recA", 0.0, 0.0, 100.0, [1.0, 2.0, 3.0, 4.0, 5.0]),
        _block(
            "recB",
            50.0,
            1000.0,
            1300.0,
            [1.0, 2.0, 3.0, 4.0, 5.0],
            support=[1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ]
    pairs = annotated_pairs(blocks, max_pairs=10, seed=0, min_support=0.5, min_shared=3)
    assert len(pairs) == 1
    row = pairs[0]
    assert row["n_shared_contacts"] == 5
    assert row["mean_block_span_seconds"] == pytest.approx(200.0)
    assert row["max_block_span_seconds"] == pytest.approx(300.0)
    assert row["same_source"] is False


def test_annotated_pairs_counts_only_jointly_supported_contacts():
    blocks = [
        _block("recA", 0.0, 0.0, 10.0, [1.0, 2.0, 3.0, 4.0], support=[1, 1, 1, 1]),
        _block("recA", 20.0, 20.0, 30.0, [1.0, 2.0, 3.0, 9.0], support=[1, 1, 1, 0]),
    ]
    pairs = annotated_pairs(blocks, max_pairs=10, seed=0, min_support=0.5, min_shared=3)
    assert pairs[0]["n_shared_contacts"] == 3
    assert pairs[0]["same_source"] is True


def test_annotated_pairs_drops_pairs_below_the_shared_contact_minimum():
    blocks = [
        _block("recA", 0.0, 0.0, 10.0, [1.0, 2.0, 3.0, 4.0], support=[1, 1, 0, 0]),
        _block("recA", 20.0, 20.0, 30.0, [1.0, 2.0, 3.0, 4.0], support=[1, 1, 0, 0]),
    ]
    assert annotated_pairs(
        blocks, max_pairs=10, seed=0, min_support=0.5, min_shared=3
    ) == []
