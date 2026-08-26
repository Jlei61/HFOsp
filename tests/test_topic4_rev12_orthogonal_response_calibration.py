import json
from pathlib import Path

import numpy as np

from scripts.analyze_topic4_rev12_orthogonal_response_calibration import (
    solve_common_direction,
)
from scripts.freeze_topic4_rev12_orthogonal_response_calibration import (
    build_candidates,
)


ROOT = Path("/home/honglab/leijiaxin/HFOsp")
BASE = ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12"


def test_orthogonal_pairs_are_nonselectable_and_reconstruct_g04():
    stage_aa = json.loads((
        BASE / "node_stage_aa_global_soft_field_expansion/aggregate/"
        "fit_soft_global_summary.json"
    ).read_text())
    stage_ae = json.loads((
        BASE / "node_stage_ae_joint_objective_conflict_audit/"
        "joint_objective_conflict_audit.json"
    ).read_text())
    design = {
        "center_candidate_id": "stage_z_g04_m",
        "maximum_cosine_frequency": 3,
        "stored_n_basis": 18,
        "degree": 3,
        "projection_grid_per_axis": 31,
        "surface_rms_radius": 0.18,
        "expected_basis_modes": 15,
        "expected_candidate_count": 30,
        "required_stage_ae_status": (
            "OBSERVED_CROSS_MODE_TRADEOFF_NEW_PAIRED_DESIGN_REQUIRED"
        ),
    }
    candidates, audit = build_candidates(stage_aa, stage_ae, design)
    assert len(candidates) == 30
    assert all(row["selection_eligible"] is False for row in candidates)
    assert audit["manual_field_used"] is False
    assert audit["patient_or_contact_coordinates_used"] is False
    assert max(row["midpoint_max_abs_error"] for row in audit["pairs"]) < 1e-12


def test_common_direction_finds_shared_positive_halfspace():
    gradients = {
        "a": np.asarray([1.0, 0.0]),
        "b": np.asarray([0.0, 1.0]),
        "c": np.asarray([1.0, 1.0]),
    }
    result = solve_common_direction(gradients, ["a", "b", "c"])
    assert result["success"] is True
    assert result["common_normalized_margin"] > 0.6
    assert all(value > 0.0 for value in result[
        "predicted_raw_utilities_per_unit_rms"
    ].values())


def test_common_direction_reports_zero_gradient():
    result = solve_common_direction(
        {"a": np.asarray([0.0, 0.0]), "b": np.asarray([1.0, 0.0])},
        ["a", "b"],
    )
    assert result["success"] is False
    assert "zero gradient" in result["reason"]
