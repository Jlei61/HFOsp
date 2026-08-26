from scripts.analyze_topic4_rev12_local_donor_interpolation import (
    candidate_audit,
    dose_main_effects,
)


ADVANCEMENT = {
    "required_positive_mean_endpoints": [
        "soft_objective", "mode_0", "mode_1", "causal_direction",
    ],
    "minimum_positive_networks_per_mode": 6,
    "direction_must_not_be_negative": True,
}


def _worker(seed, objective, mode0, mode1, direction):
    return {
        "seed": seed,
        "soft_objective": {
            "objective": objective,
            "modes": {"0": {"mean": mode0}, "1": {"mean": mode1}},
        },
        "soft_causal_direction": {"score": direction},
        "soft_causal_monotonicity": {"score": direction},
        "ood_fraction": 0.5,
        "compound_fraction": 0.4,
    }


def _row(objective, mode0, mode1, direction):
    return {
        "per_network": [
            _worker(seed, objective, mode0, mode1, direction) for seed in range(9)
        ]
    }


def test_candidate_does_not_advance_when_one_mode_improves_on_only_five_networks():
    anchor = _row(2.0, 1.5, 1.5, 0.1)
    candidate = {
        "per_network": [
            _worker(seed, 1.8, 1.4 if seed < 5 else 1.6, 1.4, 0.2)
            for seed in range(9)
        ]
    }
    result = candidate_audit(
        anchor, candidate, draws=256, confidence=0.9, seed=3,
        advancement=ADVANCEMENT,
    )
    assert result["all_required_means_positive"] is True
    assert result["both_modes_network_stable"] is False
    assert result["advances"] is False


def test_dose_main_effect_averages_other_factor_within_network():
    origin = _row(2.0, 1.5, 1.5, 0.1)
    rows, coordinates = {}, {}
    for mode in (0.0, 0.25):
        for direction in (0.0, 0.15):
            if mode == direction == 0.0:
                continue
            candidate_id = f"m{mode}_d{direction}"
            rows[candidate_id] = _row(
                2.0 - mode - direction,
                1.5 - mode,
                1.5 - 2 * mode,
                0.1 + 2 * direction,
            )
            coordinates[candidate_id] = (mode, direction)
    result = dose_main_effects(
        rows, coordinates, origin, draws=256, confidence=0.9, seed=5,
    )
    assert result["mode_1_dose"]["0.25"]["mode_1"]["mean"] > 0.0
    assert result["direction_dose"]["0.15"]["causal_direction"]["mean"] > 0.0
