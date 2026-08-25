import numpy as np

from scripts.freeze_topic4_rev12_orthogonal_outer_amplitude import (
    build_outer_candidates,
)
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
    return {
        "status": "REV12ND_ORTHOGONAL_MODE_RESPONSE_ANALYSIS_COMPLETE",
        "outer_followup": {
            "outer_amplitude": 0.16,
            "selected_for_outer_amplitude": [{
                "mode_index": 0, "kx": 0, "ky": 1, "orientation": -1,
                "selection_reasons": ["champion:patient_utility"],
                "normalized_effects": {"patient_utility": 2.0},
                "balanced_score": 1.5,
            }],
        },
    }


def _design():
    return {
        "maximum_outer_followup_modes": 6, "outer_amplitude": 0.16,
        "maximum_cosine_frequency": 3, "stored_n_basis": 18,
        "degree": 3, "projection_grid_per_axis": 31,
    }


def test_outer_amplitude_rebuilds_only_preselected_orientation():
    manifest = {"candidates": [{
        "candidate_id": "stage_u_anchor", "node_field": _anchor(),
    }]}
    candidates, audit = build_outer_candidates(
        manifest, _analysis(), _design(),
    )
    assert [row["candidate_id"] for row in candidates] == [
        "stage_v_anchor", "stage_v_f00_m_a16",
    ]
    assert all(not row["selection_eligible"] for row in candidates)
    assert audit[0]["orientation"] == -1
    assert candidates[1]["node_field"]["residual_coordinates"][
        "signed_log_surface_rms"
    ] == -0.16


def test_outer_amplitude_rejects_coordinate_drift():
    manifest = {"candidates": [{
        "candidate_id": "stage_u_anchor", "node_field": _anchor(),
    }]}
    analysis = _analysis()
    analysis["outer_followup"]["selected_for_outer_amplitude"][0]["kx"] = 3
    try:
        build_outer_candidates(manifest, analysis, _design())
    except RuntimeError as error:
        assert "coordinates drifted" in str(error)
    else:
        raise AssertionError("coordinate drift must fail closed")
