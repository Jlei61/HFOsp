import numpy as np
import pytest

from scripts.aggregate_topic4_rev9l_l3b_repeated_oracle import _score
from src.topic4_repeated_network_oracle import (
    review_repeated_capacity,
    summarize_network_oracles,
)


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
            "A": {**{name: 5.0 for name in metrics}, "n_model_events": 2},
            "B": {**{name: 5.0 for name in metrics}, "n_model_events": 3},
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


def test_repeated_score_rejects_descriptor_count_floor_count_divergence():
    metrics = (
        "recruitment_mean_absolute_error",
        "precedence_mean_absolute_error",
        "mean_rank_profile_absolute_error",
        "event_distribution_sliced_wasserstein",
    )
    # The descriptor replay kept three events while the paired-excess readout
    # reported two usable curves. Scoring three events against an n=2 floor is
    # silent mis-standardization, so the aggregator must refuse.
    row = {
        "mode_descriptors": {"modes": {
            "A": {**{name: 5.0 for name in metrics}, "n_model_events": 3},
            "B": {**{name: 5.0 for name in metrics}, "n_model_events": 3},
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
    with pytest.raises(RuntimeError, match="descriptor event counts disagree"):
        _score(
            row,
            {2: _floor(2, a_median=1.0, b_median=1.0),
             3: _floor(3, a_median=1.0, b_median=1.0)},
            base, failure_objective=100.0)


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


def test_repeated_capacity_review_rejects_objective_gain_without_mode_a_capacity():
    metrics = (
        "recruitment_mean_absolute_error",
        "precedence_mean_absolute_error",
        "mean_rank_profile_absolute_error",
        "event_distribution_sliced_wasserstein",
    )
    rows = []
    for seed in (1, 2):
        rows.append({
            "candidate_id": "candidate", "network_seed": seed,
            "score": {
                "matched_floor_event_count_by_mode": {"A": 3, "B": 3},
                "mode_scores": {"A": 2.0, "B": 1.0},
                "standardized_descriptors": {
                    "A": {name: {"raw": 2.0} for name in metrics}},
            },
        })
    payload = {
        "network_seeds": [1, 2],
        "objective_by_candidate_network": {
            "sobol_000": {"1": 3.0, "2": 3.0},
            "candidate": {"1": 2.0, "2": 2.0},
        },
        "oracle": {
            "shared": {"selected_candidate_id": "candidate"},
            "per_network": [
                {"network_seed": seed, "minimum_objective": 2.0,
                 "representative_candidate_id": "candidate"}
                for seed in (1, 2)
            ],
        },
        "candidate_network_rows": rows,
    }
    floor = _floor(3, a_median=0.0, b_median=0.0)
    for value in floor["floor"]["modes"]["A"].values():
        value["q95"] = 1.0
    review = review_repeated_capacity(payload, {3: floor})
    assert review["per_network_oracle_improved_all_networks"] is True
    assert review["shared_n_networks_improved"] == 2
    assert review["n_networks_with_mode_A_all_descriptors_within_patient_q95"] == 0
    assert review["shared_forced_capacity_supported"] is False
    assert review["status"] == "FINITE_LIBRARY_MODE_A_CAPACITY_NOT_OBSERVED"
