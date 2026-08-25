import numpy as np

from scripts.analyze_topic4_rev12_global_soft_field_expansion import (
    bootstrap_mean_interval,
    paired_utilities,
)


def _worker(seed, objective, mode0, mode1, direction, monotonicity):
    return {
        "seed": seed,
        "soft_objective": {
            "objective": objective,
            "modes": {"0": {"mean": mode0}, "1": {"mean": mode1}},
        },
        "soft_causal_direction": {"score": direction},
        "soft_causal_monotonicity": {"score": monotonicity},
        "ood_fraction": 0.5,
        "compound_fraction": 0.4,
    }


def test_paired_utilities_orient_lower_and_higher_endpoints_as_positive():
    anchor = {"per_network": [_worker(1, 2.0, 1.5, 1.4, 0.1, 0.2)]}
    candidate = {"per_network": [_worker(1, 1.8, 1.4, 1.3, 0.3, 0.25)]}
    values = paired_utilities(anchor, candidate)
    assert np.isclose(values["soft_objective"][0], 0.2)
    assert np.isclose(values["mode_0"][0], 0.1)
    assert np.isclose(values["mode_1"][0], 0.1)
    assert np.isclose(values["causal_direction"][0], 0.2)
    assert np.isclose(values["causal_monotonicity"][0], 0.05)


def test_bootstrap_is_deterministic_and_counts_network_signs():
    values = np.asarray([1.0, 2.0, -1.0, 3.0])
    first = bootstrap_mean_interval(values, draws=1024, confidence=0.9, seed=7)
    second = bootstrap_mean_interval(values, draws=1024, confidence=0.9, seed=7)
    assert first == second
    assert first["positive_networks"] == 3
    assert first["n_networks"] == 4
    assert first["ci_low"] < first["mean"] < first["ci_high"]
