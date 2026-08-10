import numpy as np
import pytest

from src.topic4_mode_learnability import (
    binary_js_divergence,
    block_mode_reliability,
    candidate_replay_rows,
    centered_smooth_worst,
    correlation_loss,
    pareto_front_indices,
)


def test_correlation_loss_and_centered_smooth_worst_contract():
    assert correlation_loss(1.0) == pytest.approx(0.0)
    assert correlation_loss(-1.0) == pytest.approx(1.0)
    assert centered_smooth_worst([0.2, 0.2], tau=0.25) == pytest.approx(0.2)
    value = centered_smooth_worst([0.1, 0.7], tau=0.25)
    assert 0.4 < value < 0.7
    with pytest.raises(ValueError):
        centered_smooth_worst([0.1, 0.2], tau=0.0)


def test_candidate_replay_preserves_missing_descriptor_boundary():
    checkpoint = {
        "history": [{
            "generation": 2,
            "joint_loss": 0.9,
            "distance": 0.5,
            "theta": [1.0, 2.0],
            "mode": {
                "support_eligible": True,
                "cluster_counts": [10, 22],
                "matched_correlations": [0.2, 0.8],
            },
        }]
    }
    row = candidate_replay_rows(checkpoint, tau=0.25)[0]
    assert row["mode_a_loss"] == pytest.approx(0.4)
    assert row["mode_b_loss"] == pytest.approx(0.1)
    assert row["metric_availability"]["prototype_similarity"] == "retained"
    assert row["metric_availability"]["recruitment"] == "not_retained"
    assert row["metric_availability"]["precedence"] == "not_retained"


def test_pareto_front_is_support_aware_and_label_invariant():
    rows = [
        {"mode_a_loss": 0.2, "mode_b_loss": 0.5, "support_eligible": True},
        {"mode_a_loss": 0.4, "mode_b_loss": 0.3, "support_eligible": True},
        {"mode_a_loss": 0.5, "mode_b_loss": 0.6, "support_eligible": True},
        {"mode_a_loss": 0.1, "mode_b_loss": 0.1, "support_eligible": False},
    ]
    assert pareto_front_indices(rows, require_support=True) == [0, 1]
    assert pareto_front_indices(rows, require_support=False) == [3]


def test_block_mode_reliability_compares_each_block_to_complement():
    rng = np.random.default_rng(2)
    prototype_a = np.linspace(-1.0, 1.0, 9)
    prototype_b = prototype_a[::-1]
    curves, blocks, labels = [], [], []
    for block in range(4):
        for mode, prototype in enumerate((prototype_a, prototype_b)):
            for _ in range(6):
                curves.append(prototype + rng.normal(0.0, 0.01, len(prototype)))
                blocks.append(f"b{block}")
                labels.append(mode)
    result = block_mode_reliability(
        np.asarray(curves), np.asarray(blocks), np.asarray(labels),
        min_events_per_block_mode=5, bootstrap_seed=3, bootstrap_repeats=100,
    )
    assert result["n_blocks"] == 4
    assert result["modes"]["A"]["n_eligible_blocks"] == 4
    assert result["modes"]["B"]["n_eligible_blocks"] == 4
    assert result["modes"]["A"]["block_to_complement_spearman"]["median"] > 0.95
    assert result["modes"]["B"]["block_to_complement_spearman"]["median"] > 0.95
    assert result["mode_proportion_by_block"][0]["mode_a_fraction"] == pytest.approx(0.5)


def test_binary_js_is_zero_for_equal_mode_proportions_and_symmetric():
    assert binary_js_divergence(0.3, 0.3) == pytest.approx(0.0)
    assert binary_js_divergence(0.2, 0.8) == pytest.approx(
        binary_js_divergence(0.8, 0.2))
