import numpy as np

from scripts.review_topic4_rev9l_l3_network_surrogate import (
    intended_source_distances,
    oracle_summary,
)


def test_intended_source_distances_respects_source_identity():
    result = intended_source_distances(
        ["component_1", "component_2"],
        [[0.9, 0.2], [0.3, 0.8]],
        {"A": "component_2", "B": "component_1"},
    )
    assert result == {"A": 0.3, "B": 0.2, "weak": 0.3}


def test_oracle_summary_reports_network_gap_and_deterministic_tie_break():
    values = {
        "a": {1: {"weak": 1.0}, 2: {"weak": 3.0}},
        "b": {1: {"weak": 3.0}, 2: {"weak": 1.0}},
        "c": {1: {"weak": 2.0}, 2: {"weak": 2.0}},
    }
    result = oracle_summary(values, {"a": 2.0, "b": 3.0, "c": 1.0})
    assert np.isclose(result["C_per_net_1"], 1.0)
    assert np.isclose(result["shared"]["C_shared_1"], 2.0)
    assert np.isclose(result["Delta_network_1"], 1.0)
    assert result["shared"]["tied_candidate_ids"] == ["a", "b", "c"]
    assert result["shared"]["tie_break_candidate_id"] == "c"
