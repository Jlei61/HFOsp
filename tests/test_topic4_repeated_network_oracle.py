import numpy as np
import pytest

from scripts.aggregate_topic4_rev9l_l3b_repeated_oracle import _score
from src.topic4_repeated_network_oracle import summarize_network_oracles


def test_repeated_oracle_separates_per_network_and_shared_capacity():
    result = summarize_network_oracles({
        "a": {1: 1.0, 2: 4.0},
        "b": {1: 4.0, 2: 1.0},
        "c": {1: 2.0, 2: 2.0},
    })
    assert np.isclose(result["C_per_net"], 1.0)
    assert np.isclose(result["shared"]["C_shared"], 2.0)
    assert np.isclose(result["Delta_network"], 1.0)
    assert result["shared"]["selected_candidate_id"] == "c"


def test_repeated_oracle_rejects_misaligned_or_nonfinite_matrix():
    with pytest.raises(ValueError):
        summarize_network_oracles({"a": {1: 1.0}, "b": {2: 1.0}})
    with pytest.raises(ValueError):
        summarize_network_oracles({"a": {1: np.inf}})


def test_missing_repeated_mode_is_retained_as_finite_failure():
    result = _score(
        {"mode_descriptors": None}, {}, {}, failure_objective=100.0)
    assert result["objective"] == 100.0
    assert result["readout_failure"] is True


def _floor(count, *, a_median, b_median):
    metrics = (
        "recruitment_mean_absolute_error",
        "precedence_mean_absolute_error",
        "mean_rank_profile_absolute_error",
        "event_distribution_sliced_wasserstein",
    )
    return {
        "n_events_per_mode_per_draw": count,
        "floor": {"modes": {
            "A": {name: {"median": a_median, "scale_iqr": 2.0}
                  for name in metrics},
            "B": {name: {"median": b_median, "scale_iqr": 2.0}
                  for name in metrics},
        }},
    }


def test_repeated_score_uses_mode_specific_readable_count_floor():
    metrics = (
        "recruitment_mean_absolute_error",
        "precedence_mean_absolute_error",
        "mean_rank_profile_absolute_error",
        "event_distribution_sliced_wasserstein",
    )
    row = {
        "mode_descriptors": {"modes": {
            mode: {name: 5.0 for name in metrics} for mode in ("A", "B")
        }},
        "geometry": {
            "source_a": {
                "n_curves_usable": 2, "curve_usable_fraction": 2.0 / 3.0,
                "ood_fraction": 0.0,
            },
            "source_b": {
                "n_curves_usable": 3, "curve_usable_fraction": 1.0,
                "ood_fraction": 0.0,
            },
        },
    }
    base = {
        "primary_mapping": {
            "mode_A_source": "source_a", "mode_B_source": "source_b"},
        "objective": {
            "readable_fraction_penalty_weight": 2.0,
            "weakest_mode_lse_tau": 0.25,
            "ood_weight": 0.1,
        },
    }
    result = _score(
        row,
        {2: _floor(2, a_median=1.0, b_median=20.0),
         3: _floor(3, a_median=20.0, b_median=3.0)},
        base, failure_objective=100.0)
    assert result["matched_floor_event_count_by_mode"] == {"A": 2, "B": 3}
    assert np.isclose(
        result["standardized_descriptors"]["A"]
        ["recruitment_mean_absolute_error"]["z"], 2.0)
    assert np.isclose(
        result["standardized_descriptors"]["B"]
        ["recruitment_mean_absolute_error"]["z"], 1.0)


def test_repeated_score_fails_closed_below_supported_event_count():
    row = {
        "mode_descriptors": {"modes": {"A": {}, "B": {}}},
        "geometry": {
            "source_a": {
                "n_curves_usable": 1, "curve_usable_fraction": 1.0 / 3.0,
                "ood_fraction": 0.0,
            },
            "source_b": {
                "n_curves_usable": 3, "curve_usable_fraction": 1.0,
                "ood_fraction": 0.0,
            },
        },
    }
    base = {
        "primary_mapping": {
            "mode_A_source": "source_a", "mode_B_source": "source_b"}}
    result = _score(
        row, {2: _floor(2, a_median=0.0, b_median=0.0),
              3: _floor(3, a_median=0.0, b_median=0.0)},
        base, failure_objective=100.0)
    assert result["objective"] == 100.0
    assert result["matched_floor_event_count_by_mode"] == {"A": 1, "B": 3}
