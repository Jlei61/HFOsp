import json
from pathlib import Path

import numpy as np

from scripts.freeze_topic4_rev12_orthogonal_free_field_screen import (
    build_candidates,
)
from src.topic4_continuous_field import continuous_surface
from src.topic4_node_field_search import (
    cosine_sheet_residuals,
    uniform_sheet_grid,
)


ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def test_cosine_basis_is_whole_sheet_orthonormal_and_observation_invariant():
    result = cosine_sheet_residuals(
        maximum_frequency=3, target_n_basis=18,
        projection_grid_per_axis=31,
    )
    assert result["n_modes"] == 15
    assert result["observation_coordinates_used"] is False
    assert result["maximum_absolute_gram_error"] < 1e-10
    assert max(row["projection_rmse"] for row in result["rows"]) < 1e-3
    assert {(row["kx"], row["ky"]) for row in result["rows"]} == {
        (kx, ky) for kx in range(4) for ky in range(4) if (kx, ky) != (0, 0)
    }


def test_screen_has_anchor_and_symmetric_continuous_residuals():
    source_manifest = json.loads((
        ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_s_edge_supported_field_refit/candidate_manifest.json"
    ).read_text())
    source_summary = json.loads((
        ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_s_edge_supported_field_refit/aggregate/"
        "fit_cascade_summary_mode_mean_direction.json"
    ).read_text())
    capacity_manifest = json.loads((
        ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
        "node_stage_t_causal_continuation_capacity/candidate_manifest.json"
    ).read_text())
    design = {
        "anchor_candidate_id": "stage_i_a02_d01_s00_p",
        "maximum_cosine_frequency": 3,
        "expected_mode_count": 15,
        "amplitude": 0.08,
        "stored_n_basis": 18,
        "degree": 3,
        "projection_grid_per_axis": 31,
        "expected_candidate_count": 31,
    }
    candidates, audit = build_candidates(
        source_manifest, source_summary, capacity_manifest, design,
    )
    assert len(candidates) == 31
    assert candidates[0]["candidate_id"] == "stage_u_anchor"
    assert all(
        not row["node_field"]["residual_coordinates"]["observation_coordinates_used"]
        for row in candidates
    )
    assert not any("manual" in row["candidate_id"] for row in candidates)
    assert audit["capacity_span"]["used_for_candidate_generation"] is False
    assert audit["capacity_span"]["used_for_selection"] is False
    grid = uniform_sheet_grid(31)
    anchor = np.asarray(candidates[0]["node_field"]["coefficients"], float)
    by_id = {row["candidate_id"]: row for row in candidates}
    for mode in range(15):
        minus = np.asarray(by_id[f"stage_u_f{mode:02d}_m"]["node_field"]["coefficients"])
        plus = np.asarray(by_id[f"stage_u_f{mode:02d}_p"]["node_field"]["coefficients"])
        assert np.allclose((minus + plus) / 2.0, anchor)
        delta = continuous_surface(
            (plus - minus) / 0.16, grid, n_basis=18, degree=3, L=20.0,
        )
        delta = delta - float(np.mean(delta))
        assert np.isclose(np.sqrt(np.mean(delta ** 2)), 1.0, atol=1e-10)
