import numpy as np

from scripts.freeze_topic4_rev12_local_curvature_canary import build_candidates
from src.topic4_observation_invariant_spline import array_sha256


def _anchor():
    coefficients = np.zeros((18, 18), float)
    return {
        "candidate_id": "stage_u_anchor",
        "field_type": "spline_continuous", "n_basis": 18, "degree": 3,
        "coefficients": coefficients.tolist(),
        "field_sha256": array_sha256(coefficients),
        "roughness": 0.0, "component_count": None,
        "peak_count_constraint": None,
    }


def _analysis():
    delta = {endpoint: 0.01 for endpoint in (
        "objective_utility", "patient_utility", "kmeans_utility",
        "ood_utility", "compound_utility", "direction_utility",
        "monotonicity_utility", "topology_reliability_utility",
        "topology_separation_utility", "patient_mode_0_utility",
        "patient_mode_1_utility",
    )}
    network = {endpoint: {str(seed): 0.01 for seed in (1, 2, 3)}
               for endpoint in delta if endpoint not in (
                   "topology_reliability_utility",
                   "topology_separation_utility",
               )}
    return {
        "status": "REV12ND_ANCHOR_RELATIVE_LOCAL_AUDIT_COMPLETE",
        "outer_amplitude_decision": {
            "status": "NOT_OPENED_NO_ANCHOR_OBJECTIVE_IMPROVEMENT",
        },
        "local_canary_proposals": {"selected": [{
            "composition": [
                {"mode_index": 1, "amplitude": -0.01},
                {"mode_index": 2, "amplitude": 0.01},
            ],
            "predicted_delta": {"aggregate": delta, "network": network},
            "surrogate_rank_score": 0.01,
        }]},
    }


def test_local_curvature_manifest_has_attribution_and_combination_fields():
    candidates, audit = build_candidates(
        {"candidates": [{
            "candidate_id": "stage_u_anchor", "node_field": _anchor(),
        }]},
        _analysis(),
        {
            "maximum_cosine_frequency": 3, "stored_n_basis": 18,
            "degree": 3, "projection_grid_per_axis": 31,
            "single_mode_canaries": [
                {"mode_index": 0, "amplitude": 0.03},
            ],
            "maximum_surrogate_combinations": 1,
            "expected_candidate_count": 3,
            "maximum_candidate_l2_amplitude": 0.06,
        },
    )
    assert [row["candidate_id"] for row in candidates] == [
        "stage_w_anchor", "stage_w_f00p03", "stage_w_f01m01_f02p01",
    ]
    assert all(not row["selection_eligible"] for row in candidates)
    assert np.isclose(audit[0]["observed_surface_rms"], 0.03, rtol=2e-3)
    assert audit[1]["source"] == "diagonal_quadratic_surrogate"


def test_local_curvature_freezer_rejects_open_outer_extrapolation():
    analysis = _analysis()
    analysis["outer_amplitude_decision"]["status"] = "REQUIRES_SEPARATE_REVIEW"
    try:
        build_candidates(
            {"candidates": [{
                "candidate_id": "stage_u_anchor", "node_field": _anchor(),
            }]}, analysis, {},
        )
    except RuntimeError as error:
        assert "outer-amplitude" in str(error)
    else:
        raise AssertionError("open outer extrapolation must fail closed")
