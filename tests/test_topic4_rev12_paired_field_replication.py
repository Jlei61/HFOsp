from scripts.analyze_topic4_rev12_paired_field_replication import (
    analyze_replication,
    paired_bootstrap,
)
from tests.test_topic4_rev12_orthogonal_anchor_relative import _row


def test_paired_bootstrap_preserves_network_sign_count():
    result = paired_bootstrap(
        {"1": 1.0, "2": 2.0, "3": -1.0},
        draws=100, quantiles=(0.05, 0.95), seed=3,
    )
    assert result["n_positive"] == 2
    assert result["n_negative"] == 1
    assert result["q_low"] <= result["mean"] <= result["q_high"]


def test_replication_uses_one_joint_aggregate_and_sign_rule():
    anchor = _row(
        objective=1.0, mode0=1.0, mode1=1.0, kmeans=0.5, direction=0.4,
    )
    candidate = _row(
        objective=0.9, mode0=0.9, mode1=0.9, kmeans=0.6, direction=0.5,
    )
    anchor["source_topology"] = {
        "mean_within_network_split_half_cosine": 0.5,
        "mean_across_network_template_cosine": 0.5,
        "equal_network_between_mode_distance": 0.4,
    }
    candidate["source_topology"] = {
        "mean_within_network_split_half_cosine": 0.6,
        "mean_across_network_template_cosine": 0.6,
        "equal_network_between_mode_distance": 0.5,
    }
    result = analyze_replication({
        "stage_x_anchor": anchor, "stage_x_f14p03": candidate,
    }, decision={
        "anchor_candidate_id": "stage_x_anchor",
        "candidate_id": "stage_x_f14p03",
        "required_aggregate_endpoints": [
            "objective_utility", "patient_mode_0_utility",
            "patient_mode_1_utility", "kmeans_utility", "direction_utility",
        ],
        "required_sign_endpoints": [
            "objective_utility", "patient_mode_0_utility",
            "patient_mode_1_utility",
        ],
        "minimum_positive_networks": 2,
        "bootstrap_draws": 100,
        "bootstrap_interval": [0.05, 0.95],
        "bootstrap_seed": 4,
    })
    assert result["fit_replication_advances_to_selection_review"]
    assert result["decision"] == "OPEN_FRESH_SELECTION_REVIEW"
    assert result["patient_heldout_used_for_decision"] is False
