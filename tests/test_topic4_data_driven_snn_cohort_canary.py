from __future__ import annotations

import numpy as np

from scripts.freeze_topic4_data_driven_snn_cohort_canary import candidate_library
from src.topic4_continuous_field import continuous_candidate_hash


def _source_manifest() -> dict:
    coefficients = np.arange(16, dtype=float).reshape(4, 4)
    node = {
        "field_type": "spline_continuous",
        "n_basis": 4,
        "degree": 3,
        "coefficients": coefficients.tolist(),
    }
    node["field_sha256"] = continuous_candidate_hash(node)
    return {
        "candidate_set": {
            "candidates": [{
                "candidate_id": "source",
                "node_field": node,
                "coefficients": np.ones((2, 6), float).tolist(),
                "raw_logit_clip": 0.75,
                "spatial_ou": {"mode": "off"},
            }],
        },
    }


def test_canary_library_rotates_field_without_patient_geometry():
    config = {
        "pretrained_source": {
            "candidate_id": "source",
            "field_rotations_deg": [0, 90, 180, 270],
            "arms": ["Node", "Node+EE+EtoI"],
        },
    }
    rows = candidate_library(config, _source_manifest())
    assert len(rows) == 8
    assert len({row["candidate_id"] for row in rows}) == 8
    source = np.arange(16, dtype=float).reshape(4, 4)
    rotated = {
        row["node_field"]["transform"]["rotation_deg"]:
        np.asarray(row["node_field"]["coefficients"])
        for row in rows if row["arm"] == "Node"
    }
    for degrees in (0, 90, 180, 270):
        np.testing.assert_array_equal(rotated[degrees], np.rot90(source, degrees // 90))
    assert all(
        not row["node_field"]["transform"]["uses_patient_target"]
        and not row["node_field"]["transform"]["uses_contact_geometry"]
        for row in rows
    )


def test_canary_node_arm_removes_edge_modulation_only():
    config = {
        "pretrained_source": {
            "candidate_id": "source",
            "field_rotations_deg": [0],
            "arms": ["Node", "Node+EE+EtoI"],
        },
    }
    rows = candidate_library(config, _source_manifest())
    node = next(row for row in rows if row["arm"] == "Node")
    joint = next(row for row in rows if row["arm"] == "Node+EE+EtoI")
    np.testing.assert_array_equal(node["coefficients"], np.zeros((2, 6)))
    np.testing.assert_array_equal(joint["coefficients"], np.ones((2, 6)))
    assert node["node_field"]["field_sha256"] == joint["node_field"]["field_sha256"]
