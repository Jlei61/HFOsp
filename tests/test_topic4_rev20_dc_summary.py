from scripts.summarize_topic4_rev20_dc import summarize


def _row(candidate, seed, training, heldout, alignment, ood):
    return {
        "candidate_id": candidate,
        "seed": seed,
        "selection": {"complete_distribution_distance_training": training},
        "validation": {
            "complete_distribution_distance_reference": heldout,
            "direction_balanced_alignment": alignment,
            "ood_all_returned": ood,
            "unreadable_fraction": 0.1,
            "n_returned_families": 20,
        },
        "training_patient_floor": {"q05": 0.1, "q50": 0.2, "q95": 0.3},
        "heldout_patient_floor": {"q05": 0.11, "q50": 0.21, "q95": 0.31},
    }


def test_response_atlas_keeps_screen_and_confirmation_separate():
    config = {
        "validation": {"paired_bootstrap_draws": 32,
                       "paired_bootstrap_seed": 10},
        "claim_boundary": "test",
    }
    manifest = {"candidates": [
        {"candidate_id": "ref", "family": "reference", "level": "reference",
         "is_reference": True},
        {"candidate_id": "gain_low", "family": "node_gain", "level": 0.75,
         "is_reference": False},
        {"candidate_id": "gain_high", "family": "node_gain", "level": 1.25,
         "is_reference": False},
    ]}
    selection = {
        "reference_candidate_id": "ref",
        "candidate_ids": ["ref", "gain_low"],
        "family_winners": {"node_gain": {"candidate_id": "gain_low"}},
        "candidate_summaries": {
            "gain_high": {
                "selection_eligible": False,
                "invalid_reasons": ["RUNAWAY_SEED_2"],
            },
        },
    }
    screen_rows = []
    for seed in (1, 2):
        screen_rows.extend([
            _row("ref", seed, 1.0, 1.1, 0.7, 0.4),
            _row("gain_low", seed, 0.8, 0.9, 0.8, 0.3),
            _row("gain_high", seed, 1.2, 1.3, 0.6, 0.5),
        ])
    confirmation_rows = []
    for seed in (3, 4, 5):
        confirmation_rows.extend([
            _row("ref", seed, 1.0, 1.1, 0.7, 0.4),
            _row("gain_low", seed, 0.8, 0.9, 0.8, 0.3),
        ])
    common = {
        "validation_endpoint_status": "FROZEN_VALIDATION_OPENED",
        "heldout_opened": True,
    }
    payload = summarize(
        config, manifest, selection,
        {**common, "per_network": screen_rows},
        {**common, "per_network": confirmation_rows},
    )
    assert payload["candidates"]["gain_high"]["confirmation"] is None
    low = payload["candidates"]["gain_low"]
    assert low["screen"]["metrics"]["heldout_complete_distribution"]["mean"] == 0.9
    assert low["confirmation"]["paired_delta_vs_reference"][
        "heldout_complete_distribution"
    ]["mean"] < 0
    assert payload["candidates"]["gain_high"]["screen_eligible"] is False
    assert payload["candidates"]["gain_high"]["screen_invalid_reasons"] == [
        "RUNAWAY_SEED_2"
    ]
    curve = payload["family_curves"]["node_gain"]
    assert [row["level"] for row in curve["rows"]] == [0.75, 1.0, 1.25]
