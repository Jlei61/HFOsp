from scripts.analyze_topic4_rev12_orthogonal_anchor_relative import (
    anchor_relative_rows,
    local_proposals,
)


def _row(*, objective, mode0, mode1, kmeans=0.5, direction=0.4):
    seeds = (1, 2, 3)
    return {
        "selection_objective": {
            "objective": objective,
            "matched_patient_loss": objective,
            "kmeans_direction_loss": 1.0 - kmeans,
            "ood_fraction": 0.2,
            "compound_fraction": 0.1,
            "causal_direction_score": direction,
        },
        "source_topology": {
            "mean_within_network_split_half_cosine": 0.6,
            "mean_across_network_template_cosine": 0.5,
            "equal_network_between_mode_distance": 0.7,
        },
        "causal_wave_monotonicity": {
            "score": 0.3,
            "per_network": [{"score": 0.3}] * 3,
        },
        "per_seed": [
            {"seed": seed, "ood_fraction": 0.2, "compound_fraction": 0.1}
            for seed in seeds
        ],
        "matched_network_scores": [{
            "objective": objective,
            "modes": {
                "0": {"mean": mode0}, "1": {"mean": mode1},
            },
        }] * 3,
        "causal_direction_alignment": {
            "per_network": [{"score": direction}] * 3,
        },
        "per_network_natural_kmeans": {
            "rows": [
                {"seed": seed, "direction_balanced_alignment": kmeans}
                for seed in seeds
            ],
        },
    }


def test_anchor_relative_requires_improvement_over_anchor():
    rows = {
        "stage_u_anchor": _row(objective=1.0, mode0=1.0, mode1=1.0),
        "stage_u_f00_m": _row(objective=1.3, mode0=1.2, mode1=1.2),
        "stage_u_f00_p": _row(objective=1.1, mode0=0.9, mode1=0.9),
    }
    result = anchor_relative_rows(rows, minimum_same_sign=2)
    plus = next(row for row in result if row["orientation"] == 1)
    assert plus["both_patient_modes_improve_aggregate"]
    assert plus["aggregate_delta_from_anchor"]["objective_utility"] < 0.0
    assert plus["network_improvement_support"]["objective_utility"][
        "stable_sign"
    ] == -1


def test_local_proposals_enforce_both_modes_and_network_support():
    network_good = {
        endpoint: {
            str(seed): {"linear": 1.0, "quadratic": 0.0}
            for seed in (1, 2, 3)
        }
        for endpoint in (
            "objective_utility", "patient_utility", "kmeans_utility",
            "ood_utility", "compound_utility", "direction_utility",
            "monotonicity_utility", "patient_mode_0_utility",
            "patient_mode_1_utility",
        )
    }
    aggregate_good = {
        endpoint: {"linear": 1.0, "quadratic": 0.0}
        for endpoint in (
            "objective_utility", "patient_utility", "kmeans_utility",
            "ood_utility", "compound_utility", "direction_utility",
            "monotonicity_utility", "topology_reliability_utility",
            "topology_separation_utility", "patient_mode_0_utility",
            "patient_mode_1_utility",
        )
    }
    result = local_proposals([{
        "mode_index": 0, "aggregate": aggregate_good,
        "network": network_good,
    }], screen={
        "amplitude_grid": [-0.02, 0.02],
        "maximum_nonzero_modes": 1,
        "maximum_l2_amplitude": 0.03,
        "required_positive_aggregate_endpoints": [
            "objective_utility", "patient_mode_0_utility",
            "patient_mode_1_utility", "kmeans_utility", "direction_utility",
        ],
        "required_network_support_endpoints": [
            "objective_utility", "patient_mode_0_utility",
            "patient_mode_1_utility",
        ],
        "minimum_same_sign_networks": 2,
        "minimum_monotonicity_delta": 0.0,
        "protected_endpoint_weight": 0.5,
        "network_dispersion_weight": 0.15,
        "maximum_proposals": 2,
    })
    assert result["n_feasible_grid_proposals"] == 1
    assert result["selected"][0]["composition"] == [
        {"mode_index": 0, "amplitude": 0.02},
    ]
