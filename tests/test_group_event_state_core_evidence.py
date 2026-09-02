from __future__ import annotations

from src.topic5_group_event_state.v03.core_evidence import build_payload, validate_payload


def _summary():
    horizons = {}
    for key in ("300s", "1800s", "7200s"):
        horizons[key] = {
            "count_correct_minus_multiscale": -2.0,
            "count_correct_minus_shifted": 1.0,
            "intercept_poisson_nll": 10.0,
            "count_pair_scored_seeds": {
                "correct_vs_multiscale": 3,
                "correct_vs_shifted": 2,
                "correct_vs_state_free": 0,
            },
            "count_pair_posthoc_flagged_seeds": {
                "correct_vs_multiscale": 1,
                "correct_vs_shifted": 0,
                "correct_vs_state_free": 3,
            },
            "n_development_test_anchors": 20,
            "n_insufficient_coverage_seeds": 0,
            "n_seeds": 3,
            "continue_correct_minus_shifted": -0.01,
            "positive_size_correct_minus_shifted": 0.02,
            "subset_correct_minus_shifted": None,
        }
    return {
        "format": "group_event_state_v0_3_1_closeout_summary",
        "source_commit": "abc",
        "subjects": ["patient"],
        "nested_source_audit": {
            "model_layer_nested_contract": True,
            "measurement_layer_nested_contract": False,
        },
        "per_subject": {
            "patient": {
                "optimization_status": "mixed_or_interior",
                "selected_epochs": [2, 3, 4],
                "n_seeds": 3,
                "horizons": horizons,
            }
        },
    }


def test_v031_diagnostics_are_archived_but_do_not_populate_primary_panels():
    payload = build_payload(_summary())
    validate_payload(payload)
    h1 = payload["v0_3_1_diagnostics"]["count_rows"][0]
    assert h1["state_alone_minus_history_nll"] == -2.0
    assert h1["correct_minus_shifted_nll"] == 1.0
    assert h1["posthoc_flagged_seeds_multiscale"] == 1
    marks = {
        row["endpoint"]: row["correct_minus_shifted_nll"]
        for row in payload["v0_3_1_diagnostics"]["mark_rows"][:3]
    }
    assert marks == {"continue": -0.01, "positive_size": 0.02, "subset": None}
    assert payload["h1_future_block"]["rows"] == []
    assert payload["h2a_repertoire"]["rows"] == []
    assert "residual_gain_over_history" in payload["h1_future_block"]["required_fields"]
    assert "gain_over_best_control" in payload["h2a_repertoire"]["required_fields"]


def test_unrun_cross_task_slots_are_explicit_and_never_filled_with_demo_data():
    payload = build_payload(_summary())
    assert payload["h2b_transfer"]["status"] == "not_yet_run"
    assert payload["h2b_transfer"]["risk_rows"] == []
    assert payload["h2b_transfer"]["field_rows"] == []
    assert payload["h3_feedback"]["status"] == "not_yet_run"
    assert payload["h3_feedback"]["model_rows"] == []
    assert payload["h3_feedback"]["impulse_rows"] == []
