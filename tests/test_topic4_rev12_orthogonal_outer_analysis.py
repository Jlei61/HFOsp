from scripts.analyze_topic4_rev12_orthogonal_outer_amplitude import (
    outer_response_rows,
)


def _row(patient, direction):
    return {
        "selection_objective": {
            "objective": patient - 0.5 * direction + 1.0,
            "matched_patient_loss": patient,
            "kmeans_direction_loss": 0.5,
            "ood_fraction": 0.2, "compound_fraction": 0.1,
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
            for seed in (1, 2, 3)
        ],
        "matched_network_scores": [{"objective": patient}] * 3,
        "causal_direction_alignment": {
            "per_network": [{"score": direction}] * 3,
        },
        "per_network_natural_kmeans": {"rows": [
            {"seed": seed, "direction_balanced_alignment": 0.5}
            for seed in (1, 2, 3)
        ]},
    }


def test_outer_response_retains_cross_network_improvement():
    stage_u = {
        "mode_responses": [{
            "mode_index": 0,
            "aggregate_slopes": {
                "objective_utility": 1.0, "patient_utility": 1.0,
                "kmeans_utility": 0.0, "ood_utility": 0.0,
                "compound_utility": 0.0, "direction_utility": 1.0,
                "monotonicity_utility": 0.0,
                "topology_reliability_utility": 0.0,
                "topology_separation_utility": 0.0,
            },
        }],
        "outer_followup": {
            "outer_amplitude": 0.16,
            "selected_for_outer_amplitude": [{
                "mode_index": 0, "orientation": 1, "kx": 0, "ky": 1,
                "stable_improvement": {
                    "objective_utility": True, "patient_utility": True,
                    "kmeans_utility": False, "ood_utility": False,
                    "compound_utility": False, "direction_utility": True,
                    "monotonicity_utility": False,
                },
            }],
        },
    }
    manifest = {"outer_amplitude_audit": [{
        "candidate_id": "stage_v_f00_p_a16", "mode_index": 0,
        "orientation": 1, "kx": 0, "ky": 1,
        "selection_reasons": ["champion:patient_utility"],
    }]}
    rows = {
        "stage_v_anchor": _row(patient=1.0, direction=0.2),
        "stage_v_f00_p_a16": _row(patient=0.8, direction=0.4),
    }
    result = outer_response_rows(
        stage_u, manifest, rows, minimum_same_sign=2,
    )
    assert result[0]["response_status"] == "OUTER_RESPONSE_RETAINED"
    assert set(result[0]["retained_improvement_endpoints"]) >= {
        "objective_utility", "patient_utility", "direction_utility",
    }


def test_outer_response_does_not_retain_reversal():
    stage_u = {
        "mode_responses": [{
            "mode_index": 0,
            "aggregate_slopes": {
                endpoint: (1.0 if endpoint == "patient_utility" else 0.0)
                for endpoint in (
                    "objective_utility", "patient_utility", "kmeans_utility",
                    "ood_utility", "compound_utility", "direction_utility",
                    "monotonicity_utility", "topology_reliability_utility",
                    "topology_separation_utility",
                )
            },
        }],
        "outer_followup": {
            "outer_amplitude": 0.16,
            "selected_for_outer_amplitude": [{
                "mode_index": 0, "orientation": 1, "kx": 0, "ky": 1,
                "stable_improvement": {
                    endpoint: endpoint == "patient_utility"
                    for endpoint in (
                        "objective_utility", "patient_utility", "kmeans_utility",
                        "ood_utility", "compound_utility", "direction_utility",
                        "monotonicity_utility",
                    )
                },
            }],
        },
    }
    manifest = {"outer_amplitude_audit": [{
        "candidate_id": "stage_v_f00_p_a16", "mode_index": 0,
        "orientation": 1, "kx": 0, "ky": 1,
        "selection_reasons": ["champion:patient_utility"],
    }]}
    rows = {
        "stage_v_anchor": _row(patient=1.0, direction=0.2),
        "stage_v_f00_p_a16": _row(patient=1.2, direction=0.2),
    }
    result = outer_response_rows(
        stage_u, manifest, rows, minimum_same_sign=2,
    )
    assert result[0]["response_status"] == "OUTER_RESPONSE_NOT_RETAINED"
