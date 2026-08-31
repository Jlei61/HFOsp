from __future__ import annotations

import copy

import numpy as np

from src.topic4_observation_invariant_spline import array_sha256
from src.topic4_rev17_dual_field_residual import (
    FORMULA,
    accepted_anchor,
    build_residual_atlas,
    mapping_sha256,
)


def _field(candidate_id: str, offset: float) -> dict:
    values = np.arange(36, dtype=float).reshape(6, 6) / 100.0 + offset
    return {
        "candidate_id": candidate_id,
        "field_type": "spline_continuous",
        "n_basis": 6,
        "degree": 3,
        "coefficients": values.tolist(),
        "field_sha256": array_sha256(values),
        "roughness": 1.0,
    }


def _exact() -> dict:
    return {
        "node_field": _field("mean", 0.0),
        "node_dispersion_field": _field("dispersion", 0.5),
        "node_mapping": {"mapping_type": "dual_continuous_mean_dispersion"},
        "source_candidate_ids": {"mean": "m", "dispersion": "d"},
    }


def test_zero_residual_anchor_preserves_both_fields_exactly():
    exact = _exact()
    anchor = accepted_anchor(exact)
    assert anchor["node_field"] == exact["node_field"]
    assert anchor["node_dispersion_field"] == exact["node_dispersion_field"]
    assert anchor["node_field"] is not exact["node_field"]
    assert anchor["node_mapping"]["mapping_type"] == (
        "dual_continuous_mean_dispersion"
    )
    assert anchor["node_mapping"]["mapping_sha256"] == mapping_sha256(
        exact["node_field"]["field_sha256"],
        exact["node_dispersion_field"]["field_sha256"],
    )


def test_atlas_perturbs_only_one_dual_channel_per_coordinate():
    exact = _exact()
    candidates, audit = build_residual_atlas(
        exact, maximum_frequency=1, amplitude=0.1,
        target_n_basis=6, degree=3, projection_grid_per_axis=21,
    )
    assert audit["formula"] == FORMULA
    assert audit["basis_mode_count"] == 3
    assert len(candidates) == 13
    anchor = candidates[0]
    for candidate in candidates[1:]:
        coordinates = candidate["residual_coordinates"]
        assert coordinates["zero_is_exact_dual_anchor"] is True
        if coordinates["channel"] == "mean":
            assert candidate["node_field"] != anchor["node_field"]
            assert candidate["node_dispersion_field"] == anchor["node_dispersion_field"]
        else:
            assert candidate["node_field"] == anchor["node_field"]
            assert candidate["node_dispersion_field"] != anchor["node_dispersion_field"]


def test_antithetic_candidates_are_symmetric_around_anchor_coefficients():
    exact = _exact()
    candidates, _ = build_residual_atlas(
        exact, maximum_frequency=1, amplitude=0.1,
        target_n_basis=6, degree=3, projection_grid_per_axis=21,
    )
    rows = {row["candidate_id"]: row for row in candidates}
    anchor = np.asarray(rows["exact_dual_anchor"]["node_field"]["coefficients"])
    negative = np.asarray(rows["mean_f00_m_a10"]["node_field"]["coefficients"])
    positive = np.asarray(rows["mean_f00_p_a10"]["node_field"]["coefficients"])
    assert np.allclose(0.5 * (negative + positive), anchor, atol=1e-12)


def test_mapping_hash_changes_when_either_channel_changes():
    exact = _exact()
    base = accepted_anchor(copy.deepcopy(exact))["node_mapping"]["mapping_sha256"]
    assert mapping_sha256("changed", exact["node_dispersion_field"]["field_sha256"]) != base
    assert mapping_sha256(exact["node_field"]["field_sha256"], "changed") != base
