import numpy as np

from scripts.freeze_topic4_rev12_causal_field_fit import (
    build_causal_field_candidates,
)
from src.topic4_observation_invariant_spline import array_sha256


def _field(index):
    values = np.zeros((18, 18), float)
    values[index, index] = 0.1
    return {
        "candidate_id": f"source_{index}",
        "field_type": "spline_continuous", "n_basis": 18, "degree": 3,
        "coefficients": values.tolist(), "field_sha256": array_sha256(values),
        "roughness": 0.0, "component_count": None,
        "peak_count_constraint": None,
    }


def test_causal_field_fit_is_whole_sheet_continuous_and_observation_invariant():
    source = {"candidates": [
        {"candidate_id": f"source_{index}", "node_field": _field(index)}
        for index in range(3)
    ]}
    design = {
        "source_candidate_ids": ["source_0", "source_1", "source_2"],
        "pairwise_midpoints": True,
        "residual_control_grid": [4, 4],
        "residual_directions_per_anchor": 4,
        "signed_log_surface_rms": [0.08, 0.16],
        "sobol_seed": 7,
        "expected_candidate_count": 54,
    }
    candidates = build_causal_field_candidates(source, design)
    assert len(candidates) == 54
    assert len({row["node_field"]["field_sha256"] for row in candidates}) == 54
    assert all(row["node_field"]["component_count"] is None for row in candidates)
    residuals = [
        row for row in candidates
        if row["role"] == "causal_whole_sheet_smooth_residual"
    ]
    assert len(residuals) == 48
    assert all(
        row["node_field"]["residual_coordinates"][
            "observation_coordinates_used"
        ] is False
        for row in residuals
    )

