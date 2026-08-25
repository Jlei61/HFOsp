from scripts.analyze_topic4_rev12_omitted_pareto_recovery import paired_recovery_audit


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


def _row(values, topology=0.8, separation=0.2):
    return {
        "per_network": values,
        "soft_topology_across_network": topology,
        "soft_topology_mode_separation": separation,
    }


def test_mode1_only_tendency_does_not_pass_balanced_stability():
    anchor = _row([_worker(seed, 2.0, 1.5, 1.5, 0.1) for seed in range(9)])
    candidate = _row([
        _worker(seed, 2.1, 1.6, 1.4 if seed < 6 else 1.6, 0.2)
        for seed in range(9)
    ])
    result = paired_recovery_audit(
        anchor, candidate, draws=256, confidence=0.9, seed=7,
    )
    assert result["status"] == "OMITTED_PARETO_MODE1_ONLY_NO_BALANCED_STABILITY"
    assert result["mode_1_only_network_tendency"] is True
    assert result["balanced_mean"] is False
    assert result["both_modes_stable"] is False


def test_balanced_candidate_requires_both_modes_on_six_of_nine_networks():
    anchor = _row([_worker(seed, 2.0, 1.5, 1.5, 0.1) for seed in range(9)])
    candidate = _row([
        _worker(seed, 1.8, 1.4, 1.4, 0.2) for seed in range(9)
    ], topology=0.85, separation=0.25)
    result = paired_recovery_audit(
        anchor, candidate, draws=256, confidence=0.9, seed=11,
    )
    assert result["status"] == "OMITTED_PARETO_BALANCED_AND_MODE_STABLE"
    assert result["balanced_mean"] is True
    assert result["both_modes_stable"] is True
