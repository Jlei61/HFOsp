import numpy as np
import pytest

from src.topic4_rev9_review_audit import (
    binary_js_divergence,
    common_detector_metrics,
    mode_evaluability,
    network_mode_summary,
    pareto_minimize_maximize,
    response_map_spearman,
    response_site_adjudication,
)
from scripts.plot_topic4_rev9_edge_alpha_calibration import _structurally_admissible


def test_mode_evaluability_fails_low_support_or_high_ood():
    assert mode_evaluability([12, 14], 0.2)["status"] == "EVALUABLE"
    low = mode_evaluability([9, 14], 0.2)
    assert low["status"] == "NOT_EVALUABLE"
    assert "fewer_than_minimum_in_distribution_events_per_mode" in low["reasons"]
    assert mode_evaluability([12, 14], 0.6)["status"] == "NOT_EVALUABLE"


def test_network_mode_summary_uses_in_distribution_events_and_network_units():
    result = network_mode_summary(
        labels=[0, 1, 1, 0, 0],
        ood=[False, False, True, False, False],
        seed_ids=[1, 1, 1, 2, 2], seeds=[1, 2], duration_s=2.0,
        patient_mode_b_fraction=0.7, bootstrap_seed=3, bootstrap_repeats=100)
    assert result["pooled_in_distribution_counts"] == [3, 1]
    assert result["n_networks_with_both_in_distribution"] == 1
    assert result["mode_b_rate_hz"]["estimate"] == pytest.approx(0.25)
    assert result["pooled_mode_proportion_js_bits"] > 0.0
    assert binary_js_divergence(0.7, 0.7) == pytest.approx(0.0)


def test_common_detector_metrics_uses_one_absolute_threshold():
    trace = np.r_[np.zeros(20), np.full(10, 0.2), np.zeros(20)]
    result = common_detector_metrics(trace, 1.0, 0.1)
    assert result["n_events"] == 1
    assert result["time_above_fraction"] == pytest.approx(0.2)
    assert result["integrated_excess_fraction_ms"] == pytest.approx(1.0)


def test_response_map_spearman_ignores_joint_zero_background():
    left = np.asarray([[0.0, 1.0], [2.0, 3.0]])
    right = np.asarray([[0.0, 2.0], [4.0, 6.0]])
    assert response_map_spearman(left, right) == pytest.approx(1.0)


def test_response_adjudication_never_retrospectively_returns_formal_pass():
    rows = [dict(
        paired_eligible=True, source_gain_ratio=0.5,
        downstream_gain_ratio=1.0, r90_delta_mm=0.2, map_rho=0.9)
        for _ in range(10)]
    result = response_site_adjudication(rows, minimum_valid_pairs=10)
    assert result["formal_status"].startswith("UNRESOLVED")
    assert result["diagnostic_pattern"] == (
        "SOURCE_NUCLEATION_FAIL_DOWNSTREAM_PARTIAL_MATCH")


def test_alpha_structure_reference_band_is_explicit():
    admissible = {"structure": {"edge_ratio": {"min": 0.25, "max": 4.0}}}
    rejected = {"structure": {"edge_ratio": {"min": 0.249, "max": 3.0}}}
    assert _structurally_admissible(admissible)
    assert not _structurally_admissible(rejected)


def test_pareto_mask_minimizes_cost_and_maximizes_worst_mode():
    mask = pareto_minimize_maximize(
        cost=[0.4, 0.5, 0.6, 0.3], benefit=[0.2, 0.5, 0.4, 0.1])
    np.testing.assert_array_equal(mask, [True, True, False, True])
