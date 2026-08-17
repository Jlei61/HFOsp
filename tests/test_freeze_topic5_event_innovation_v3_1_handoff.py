from scripts import freeze_topic5_event_innovation_v3_1_handoff as handoff


def test_dataset_direction_filters_minimum_support():
    rows = [
        {
            "subject": "epilepsiae_a", "eligible": True,
            "horizons": {"20": {
                "n_validation_anchors": 20,
                "observable": {"propagation_gain_standardized": 1.0},
                "true_minus_state_matched_null_gain": 2.0,
                "future_minus_past_state_gain": 3.0,
            }},
        },
        {
            "subject": "epilepsiae_b", "eligible": True,
            "horizons": {"20": {
                "n_validation_anchors": 19,
                "observable": {"propagation_gain_standardized": -10.0},
                "true_minus_state_matched_null_gain": -10.0,
                "future_minus_past_state_gain": -10.0,
            }},
        },
    ]
    result = handoff.dataset_directions(rows, "goal2", 20)
    assert result["epilepsiae"]["n_eligible"] == 1
    assert result["epilepsiae"]["propagation_gain"]["median"] == 1.0
