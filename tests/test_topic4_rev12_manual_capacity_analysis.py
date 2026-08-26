import copy

from scripts.analyze_topic4_rev12_manual_capacity_replication import capacity_audit


def _worker(seed, direction=0.2, monotonicity=0.1):
    return {
        "seed": seed,
        "n_events": 20,
        "soft_objective": {
            "objective": 1.5,
            "modes": {
                "0": {"mean": 1.2, "effective_events": 8.0, "soft_occupancy": 0.4},
                "1": {"mean": 1.0, "effective_events": 12.0, "soft_occupancy": 0.6},
            },
        },
        "soft_causal_direction": {"score": direction},
        "soft_causal_monotonicity": {"score": monotonicity},
        "ood_fraction": 0.6,
        "compound_fraction": 0.4,
        "natural_kmeans_final_validation_diagnostic": {
            "status": "OK", "cluster_counts": [8, 12],
            "direction_balanced_alignment": 0.8,
            "kmeans_seed_ami_median": 1.0,
        },
    }


def _candidate(selection_eligible=False):
    return {
        "selection_eligible": selection_eligible,
        "mean_soft_objective": 1.5,
        "mean_soft_mode_0": 1.2,
        "mean_soft_mode_1": 1.0,
        "mean_soft_causal_direction": 0.2,
        "mean_soft_causal_monotonicity": 0.1,
        "mean_ood_fraction": 0.6,
        "soft_topology_across_network": 0.9,
        "soft_topology_mode_separation": 0.3,
        "per_network": [_worker(seed) for seed in range(9)],
    }


def _description():
    return {
        "minimum_effective_events_per_soft_mode": 3.0,
        "direction_sign_threshold": 0.0,
        "monotonicity_sign_threshold": 0.0,
    }


def test_full_network_direction_support_closes_capacity_positive():
    capacity = _candidate()
    comparison = copy.deepcopy(capacity)
    comparison["selection_eligible"] = True
    result = capacity_audit(
        capacity, {"comparison": comparison}, description=_description(),
        draws=200, confidence=0.9, seed=1,
    )
    assert result["full_network_sign_support"] is True
    assert result["status"].startswith("NODE_ONLY_DIRECTIONAL_CAPACITY_POSITIVE")
    assert result["capacity_support_counts"]["natural_kmeans_k2_evaluable"] == 9


def test_one_zero_direction_network_prevents_full_support():
    capacity = _candidate()
    capacity["per_network"][0]["soft_causal_direction"]["score"] = 0.0
    comparison = copy.deepcopy(capacity)
    comparison["selection_eligible"] = True
    result = capacity_audit(
        capacity, {"comparison": comparison}, description=_description(),
        draws=200, confidence=0.9, seed=2,
    )
    assert result["full_network_sign_support"] is False
    assert result["status"].startswith("NODE_ONLY_DIRECTIONAL_CAPACITY_PARTIAL")


def test_selectable_manual_control_is_rejected():
    capacity = _candidate(selection_eligible=True)
    comparison = copy.deepcopy(capacity)
    try:
        capacity_audit(
            capacity, {"comparison": comparison}, description=_description(),
            draws=20, confidence=0.9, seed=3,
        )
    except RuntimeError as exc:
        assert "selection eligible" in str(exc)
    else:
        raise AssertionError("selectable manual control was accepted")


def test_nested_nonselectable_flag_matches_real_aggregate_schema():
    capacity = _candidate()
    capacity.pop("selection_eligible")
    capacity["candidate"] = {"selection_eligible": False}
    comparison = copy.deepcopy(capacity)
    result = capacity_audit(
        capacity, {"comparison": comparison}, description=_description(),
        draws=20, confidence=0.9, seed=4,
    )
    assert result["full_network_sign_support"] is True
