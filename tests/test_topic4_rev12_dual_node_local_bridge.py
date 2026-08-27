import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev12_dual_node_local_bridge import (
    affine_field,
    build_candidates,
)


def _field(name, value):
    coefficients = np.full((4, 4), float(value))
    coefficients[0, 0] += 1.0
    return affine_field(
        {
            "field_type": "spline_continuous", "n_basis": 4, "degree": 3,
            "coefficients": np.zeros((4, 4)).tolist(), "field_sha256": "left",
        },
        {
            "field_type": "spline_continuous", "n_basis": 4, "degree": 3,
            "coefficients": coefficients.tolist(), "field_sha256": name,
        },
        weight=1.0, candidate_id=name, role="test",
    )


def _candidate(candidate_id, mean_id, dispersion_id, mean_field, dispersion_field):
    return {
        "candidate_id": candidate_id,
        "role": "test",
        "selection_eligible": False,
        "source_candidate_ids": {"mean": mean_id, "dispersion": dispersion_id},
        "node_field": mean_field,
        "node_dispersion_field": dispersion_field,
        "node_mapping": {},
        "coefficients": [0.0],
        "coefficients_sha256": "unused",
    }


def test_affine_field_allows_frozen_local_extrapolation():
    left = _field("left_field", 0.0)
    right = _field("right_field", 1.0)
    result = affine_field(
        left, right, weight=1.5, candidate_id="out", role="test",
    )
    expected = np.asarray(left["coefficients"]) + 1.5 * (
        np.asarray(right["coefficients"]) - np.asarray(left["coefficients"])
    )
    assert np.allclose(result["coefficients"], expected)
    assert result["residual_coordinates"]["observation_coordinates_used"] is False


def test_local_bridge_freezes_grid_sentinels_and_historical_anchor():
    anchor = _field("anchor", 0.0)
    g10 = _field("g10", 1.0)
    g08 = _field("g08", -1.0)
    manifest = {"candidates": [
        _candidate("hist", "anchor", "anchor", anchor, anchor),
        _candidate("g10g10", "g10", "g10", g10, g10),
        _candidate("g08g08", "g08", "g08", g08, g08),
    ]}
    contract = {
        "historical_anchor_candidate_id": "hist",
        "mean_left_candidate_id": "anchor",
        "mean_right_candidate_id": "g10",
        "dispersion_left_candidate_id": "g10",
        "dispersion_right_candidate_id": "g08",
        "mean_affine_weights": [1.0, 1.25],
        "dispersion_interpolation_weights": [0.2, 0.5],
        "required_sentinels": [[1.0, 0.0], [1.0, 1.0]],
        "expected_candidate_count": 7,
        "formula": "dual",
    }
    candidates, audit = build_candidates(
        manifest,
        {"status": "DUAL_CONTINUOUS_NODE_CHANNEL_DOES_NOT_RESOLVE_TWO_MODE_TRADEOFF"},
        contract,
    )
    assert len(candidates) == 7
    assert candidates[0]["candidate_id"] == "stage_al_historical_anchor"
    assert sum(row["role"] == "dual_node_local_bridge_sentinel" for row in candidates) == 2
    assert len({row["mapping_sha256"] for row in audit["coordinates"]}) == 6


def test_local_bridge_freezer_is_directly_executable():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/freeze_topic4_rev12_dual_node_local_bridge.py"),
         "--help"],
        cwd=root, text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
