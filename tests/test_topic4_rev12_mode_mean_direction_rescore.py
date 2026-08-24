from scripts.rescore_topic4_rev12_mode_mean_direction import (
    corrected_equal_network_direction,
    rescore_payload,
)


def test_rescore_clips_only_after_mode_mean():
    bundle = {
        "score": 0.5,
        "per_network": [{
            "score": 0.5,
            "modes": {
                "0": {"mean_signed_axis_cosine": 0.0, "alignment_score": 0.5},
                "1": {"mean_signed_axis_cosine": 0.8, "alignment_score": 0.8},
            },
        }],
    }
    corrected = corrected_equal_network_direction(
        bundle, value_key="mean_signed_axis_cosine",
    )
    assert corrected["score"] == 0.0
    assert corrected["per_network"][0]["modes"]["0"]["alignment_score"] == 0.0
    assert corrected["per_network"][0]["modes"]["1"]["alignment_score"] == 0.8


def test_rescore_protects_weakest_mode_then_weights_networks_equally():
    bundle = {
        "score": 0.0,
        "per_network": [
            {"modes": {
                "0": {"mean_signed_axis_cosine": 0.7},
                "1": {"mean_signed_axis_cosine": 0.4},
            }},
            {"modes": {
                "0": {"mean_signed_axis_cosine": 0.2},
                "1": {"mean_signed_axis_cosine": -0.5},
            }},
        ],
    }
    corrected = corrected_equal_network_direction(
        bundle, value_key="mean_signed_axis_cosine",
    )
    assert corrected["per_network"][0]["score"] == 0.4
    assert corrected["per_network"][1]["score"] == 0.0
    assert corrected["score"] == 0.2


def _direction_bundle(mode_zero, mode_one, *, value_key):
    return {
        "score": 0.9,
        "per_network": [{
            "score": 0.9,
            "modes": {
                "0": {value_key: mode_zero, "alignment_score": 0.9},
                "1": {value_key: mode_one, "alignment_score": 0.9},
            },
        }],
    }


def test_payload_rescore_updates_only_direction_dependent_selection_terms():
    old_selection = {
        "matched_patient_loss": 1.0,
        "kmeans_direction_loss": 0.2,
        "ood_fraction": 0.4,
        "compound_fraction": 0.6,
        "k2_support": 0.3,
        "k2_support_loss": 0.7,
        "causal_direction_score": 0.9,
        "causal_direction_loss": 0.1,
        "objective": 1.4,
        "weights": {
            "matched_patient_loss": 1.0,
            "kmeans_direction_loss": 0.5,
            "ood_fraction": 0.25,
            "compound_fraction": 0.25,
            "k2_support_loss": 0.0,
            "causal_direction_loss": 0.5,
        },
    }
    payload = {
        "selection_contract": {},
        "rows": [{
            "candidate_id": "candidate",
            "selection_eligible": True,
            "patient_summary": {"sentinel": "unchanged"},
            "selection_objective": old_selection,
            "causal_direction_alignment": _direction_bundle(
                0.6, -0.2, value_key="mean_signed_axis_cosine",
            ),
            "causal_wave_monotonicity": _direction_bundle(
                0.4, 0.3, value_key="mean_signed_axis_time_spearman",
            ),
        }],
    }
    corrected = rescore_payload(payload)
    row = corrected["rows"][0]
    assert row["patient_summary"] == {"sentinel": "unchanged"}
    assert row["causal_direction_alignment"]["score"] == 0.0
    assert row["causal_wave_monotonicity"]["score"] == 0.3
    assert row["selection_objective"]["causal_direction_score"] == 0.0
    assert row["selection_objective"]["objective"] == 1.85
    assert row["retrospective_invalid_eventwise_direction_objective"] == old_selection
