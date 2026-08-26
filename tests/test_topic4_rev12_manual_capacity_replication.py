import copy
import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev12_manual_capacity_replication import build_candidate


ROOT = Path("/home/honglab/leijiaxin/HFOsp")
BASE = ROOT / "results/topic4_sef_hfo/data_driven_node_dualmode_rev12"


def _inputs():
    source = json.loads((
        BASE / "node_stage_t_causal_continuation_capacity/candidate_manifest.json"
    ).read_text())
    audit = json.loads((
        BASE / "node_stage_ac_local_donor_interpolation/analysis/paired_response_surface_audit.json"
    ).read_text())
    return source, audit


def _contract():
    return {
        "candidate_id": "stage_t_manual_smooth_capacity",
        "required_source_role": "rigid_capacity_control_not_selectable",
        "required_selection_eligible": False,
        "required_stage_ac_status": "LOCAL_INTERPOLATION_DIRECTION_ONLY_NO_BALANCED_NODE_FIELD",
    }


def test_capacity_control_copies_exact_field_but_remains_nonselectable():
    source, audit = _inputs()
    candidates, record = build_candidate(source, audit, _contract())
    original = next(
        row for row in source["candidates"]
        if row["candidate_id"] == "stage_t_manual_smooth_capacity"
    )
    assert len(candidates) == 1
    assert candidates[0]["node_field"]["field_sha256"] == (
        original["node_field"]["field_sha256"]
    )
    assert candidates[0]["selection_eligible"] is False
    assert record["historical_geometry_used"] is True
    assert record["patient_heldout_used"] is False


def test_capacity_control_stops_if_stage_ac_finds_a_balanced_candidate():
    source, audit = _inputs()
    audit = copy.deepcopy(audit)
    audit["status"] = "LOCAL_INTERPOLATION_BALANCED_FIT_CANDIDATE_FOUND"
    with pytest.raises(RuntimeError, match="Stage-AC"):
        build_candidate(source, audit, _contract())
