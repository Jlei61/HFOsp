import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev12_omitted_pareto_recovery import build_candidate


ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STAGE_Z = ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_z_global_soft_field_screen"
)
STAGE_AA = ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_aa_global_soft_field_expansion"
)


def _inputs():
    return (
        json.loads((STAGE_Z / "candidate_manifest.json").read_text()),
        json.loads((STAGE_Z / "aggregate/fit_soft_global_summary.json").read_text()),
        json.loads((STAGE_Z / "aggregate/fit_nomination_decision.json").read_text()),
        json.loads((STAGE_AA / "analysis/paired_expansion_audit.json").read_text()),
    )


def _contract():
    return {
        "candidate_id": "stage_z_g11_p",
        "required_stage_z_pareto_member": True,
        "required_absence_from_stage_z_nomination": True,
    }


def test_recovery_is_one_existing_pareto_field_omitted_by_radius_quota():
    candidates, audit = build_candidate(*_inputs(), _contract())
    assert [row["candidate_id"] for row in candidates] == ["stage_z_g11_p"]
    assert audit["omitted_by_filled_radius_quota"] is True
    assert len(audit["same_radius_nominees"]) == audit["maximum_per_radius"] == 2
    assert audit["patient_heldout_used"] is False
    assert audit["natural_kmeans_used_for_selection"] is False


def test_recovery_rejects_a_candidate_that_was_already_expanded():
    contract = _contract()
    contract["candidate_id"] = "stage_z_g05_p"
    with pytest.raises(RuntimeError, match="already expanded"):
        build_candidate(*_inputs(), contract)
