from scripts.analyze_topic4_rev12_orthogonal_free_field_screen import (
    mode_response_rows,
    select_outer_followup,
    stable_sign,
)


def test_stable_sign_requires_declared_network_support():
    assert stable_sign([1.0, 2.0, -0.5], minimum_same_sign=2)["stable_sign"] == 1
    assert stable_sign([-1.0, -2.0, 0.5], minimum_same_sign=2)["stable_sign"] == -1
    assert stable_sign([1.0, -2.0, 0.0], minimum_same_sign=2)["stable_sign"] == 0


def _mode(mode, patient, direction):
    endpoints = (
        "objective_utility", "patient_utility", "kmeans_utility",
        "ood_utility", "compound_utility", "direction_utility",
        "monotonicity_utility", "topology_reliability_utility",
        "topology_separation_utility",
    )
    slopes = {endpoint: 0.1 for endpoint in endpoints}
    slopes["patient_utility"] = patient
    slopes["direction_utility"] = direction
    support = {
        endpoint: {"stable_sign": (1 if slopes[endpoint] > 0 else -1)}
        for endpoint in endpoints[:7]
    }
    return {
        "mode_index": mode, "kx": mode, "ky": 0,
        "aggregate_slopes": slopes, "network_sign_support": support,
    }


def test_outer_followup_keeps_distinct_endpoint_champions():
    screen = {
        "primary_endpoints": [
            "patient_utility", "kmeans_utility", "direction_utility",
            "monotonicity_utility",
        ],
        "balanced_weights": {
            "patient_utility": 1.0, "kmeans_utility": 0.5,
            "ood_utility": 0.25, "compound_utility": 0.25,
            "direction_utility": 0.5, "monotonicity_utility": 0.5,
            "topology_reliability_utility": 0.25,
            "topology_separation_utility": 0.25,
        },
        "maximum_outer_followup_modes": 3,
        "outer_amplitude": 0.16,
    }
    result = select_outer_followup([
        _mode(0, patient=3.0, direction=0.1),
        _mode(1, patient=0.1, direction=4.0),
        _mode(2, patient=0.5, direction=0.5),
    ], screen=screen)
    selected = result["selected_for_outer_amplitude"]
    assert len(selected) <= 3
    assert {row["mode_index"] for row in selected} >= {0, 1}


def _aggregate_row(value):
    selection = {
        "objective": value,
        "matched_patient_loss": value,
        "kmeans_direction_loss": 0.5,
        "ood_fraction": 0.2,
        "compound_fraction": 0.1,
        "causal_direction_score": 0.4,
    }
    return {
        "selection_objective": selection,
        "source_topology": {
            "mean_within_network_split_half_cosine": 0.6,
            "mean_across_network_template_cosine": 0.5,
            "equal_network_between_mode_distance": 0.7,
        },
        "causal_wave_monotonicity": {
            "score": 0.3,
            "per_network": [{"score": 0.3}, {"score": 0.3}, {"score": 0.3}],
        },
        "per_seed": [
            {"seed": seed, "ood_fraction": 0.2, "compound_fraction": 0.1}
            for seed in (1, 2, 3)
        ],
        "matched_network_scores": [{"objective": value}] * 3,
        "causal_direction_alignment": {
            "per_network": [{"score": 0.4}, {"score": 0.4}, {"score": 0.4}],
        },
        "per_network_natural_kmeans": {
            "rows": [
                {"seed": seed, "direction_balanced_alignment": 0.5}
                for seed in (1, 2, 3)
            ],
        },
    }


def test_mode_response_uses_manifest_amplitude():
    rows = {
        "stage_u_f00_m": _aggregate_row(2.0),
        "stage_u_f00_p": _aggregate_row(1.0),
    }
    manifest = {"candidates": [
        {"candidate_id": f"stage_u_f00_{sign}", "node_field": {
            "residual_coordinates": {
                "residual_index": 0, "kx": 0, "ky": 1,
                "signed_log_surface_rms": amplitude,
            },
        }}
        for sign, amplitude in (("m", -0.125), ("p", 0.125))
    ]}
    result = mode_response_rows(rows, manifest, minimum_same_sign=2)
    assert result[0]["symmetric_amplitude"] == 0.125
    assert result[0]["aggregate_slopes"]["objective_utility"] == 4.0
