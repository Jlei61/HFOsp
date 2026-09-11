from scripts.aggregate_topic4_rev21_zm_canary import summarize


def _row(candidate, status, returned=3):
    return {
        "candidate_id": candidate, "topology_seed": 1, "dynamics_seed": 2,
        "model_ictal_rev21": {
            "status": status,
            "eligible": status == "MODEL_ICTAL_ELIGIBLE_REV21",
            "scientific_onset_ms": 3000.0, "failing_clauses": [],
        },
        "events": [{"returned": index < returned} for index in range(4)],
        "simulation": {"runaway_early_stop_ms": 3100.0,
                       "formal_interictal_stop_ms": 3000.0},
    }


def test_canary_proceeds_when_one_full_model_state_is_eligible():
    payload = summarize([
        _row("a", "MODEL_ICTAL_NOT_ELIGIBLE_REV21"),
        _row("b", "MODEL_ICTAL_ELIGIBLE_REV21"),
    ])
    assert payload["status"] == "REV21_CANARY_SUPPORTS_COARSE_SCREEN"
    assert payload["eligible_candidate_ids"] == ["b"]
    assert payload["candidates"][1]["n_returned_pretransition_families"] == 3
