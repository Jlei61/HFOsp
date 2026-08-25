import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_topic4_rev12_local_donor_interpolation import build_candidates


ROOT = Path("/home/honglab/leijiaxin/HFOsp")
BASE = ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12"


def _inputs():
    stage_z = json.loads((
        BASE / "node_stage_z_global_soft_field_screen/candidate_manifest.json"
    ).read_text())
    stage_aa_summary = json.loads((
        BASE / "node_stage_aa_global_soft_field_expansion/aggregate/fit_soft_global_summary.json"
    ).read_text())
    stage_aa_audit = json.loads((
        BASE / "node_stage_aa_global_soft_field_expansion/analysis/paired_expansion_audit.json"
    ).read_text())
    stage_ab_summary = json.loads((
        BASE / "node_stage_ab_omitted_pareto_recovery/aggregate/fit_soft_global_summary.json"
    ).read_text())
    stage_ab_audit = json.loads((
        BASE / "node_stage_ab_omitted_pareto_recovery/analysis/paired_recovery_audit.json"
    ).read_text())
    return stage_z, stage_aa_summary, stage_aa_audit, stage_ab_summary, stage_ab_audit


def _design():
    return {
        "base_field_id": "stage_z_anchor",
        "balanced_donor_id": "stage_z_g04_m",
        "mode_1_donor_id": "stage_z_g11_p",
        "direction_donor_id": "stage_z_g05_p",
        "mode_1_doses": [0.0, 0.25, 0.5],
        "direction_doses": [0.0, 0.15, 0.3],
        "exclude_zero_zero": True,
        "expected_candidate_count": 8,
        "projection_grid_per_axis": 41,
        "maximum_residual_surface_rms": 0.54,
        "formula": "C = C_g04 + lambda_11 * (C_g11 - C_anchor) + lambda_05 * (C_g05 - C_anchor)",
    }


def test_interpolation_is_eight_unique_continuous_fields_inside_rms_envelope():
    candidates, audit = build_candidates(*_inputs(), _design())
    assert len(candidates) == 8
    assert len({row["node_field"]["field_sha256"] for row in candidates}) == 8
    doses = {
        (
            row["node_field"]["residual_coordinates"]["mode_1_dose"],
            row["node_field"]["residual_coordinates"]["direction_dose"],
        )
        for row in candidates
    }
    assert (0.0, 0.0) not in doses
    assert max(row["observed_surface_rms"] for row in audit["candidates"]) <= 0.54
    assert audit["patient_or_contact_coordinates_used"] is False
    assert audit["patient_heldout_used"] is False


def test_interpolation_preserves_coefficient_budget_and_is_not_discrete_core_mixture():
    stage_z, *_ = _inputs()
    source = {row["candidate_id"]: row for row in stage_z["candidates"]}
    base = np.asarray(source["stage_z_anchor"]["node_field"]["coefficients"], float)
    candidates, _ = build_candidates(*_inputs(), _design())
    for row in candidates:
        values = np.asarray(row["node_field"]["coefficients"], float)
        assert np.isclose(np.sum(values), np.sum(base), atol=1e-10, rtol=0.0)
        assert row["node_field"]["component_count"] is None
        assert row["node_field"]["field_type"] == "spline_continuous"


def test_interpolation_stops_if_stage_ab_mode1_evidence_changes():
    inputs = list(_inputs())
    inputs[-1] = copy.deepcopy(inputs[-1])
    inputs[-1]["status"] = "OMITTED_PARETO_NO_BALANCED_STABILITY"
    with pytest.raises(RuntimeError, match="mode-1 donor"):
        build_candidates(*inputs, _design())
