import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev12_global_soft_field_expansion import build_candidates


ROOT = Path("/home/honglab/leijiaxin/HFOsp")
STAGE_Z = ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev12/"
    "node_stage_z_global_soft_field_screen"
)


def _inputs():
    return (
        json.loads((STAGE_Z / "candidate_manifest.json").read_text()),
        json.loads((STAGE_Z / "aggregate/fit_soft_global_summary.json").read_text()),
        json.loads((STAGE_Z / "aggregate/fit_nomination_decision.json").read_text()),
    )


def test_expansion_copies_predeclared_nominees_and_anchor_without_field_drift():
    manifest, summary, nomination = _inputs()
    expansion = {
        "anchor_candidate_id": "stage_z_anchor",
        "expected_nominee_count": 6,
        "expected_candidate_count": 7,
    }
    candidates, audit = build_candidates(
        manifest, summary, nomination, expansion,
    )
    assert [row["candidate_id"] for row in candidates] == [
        "stage_z_anchor", *nomination["candidate_ids"],
    ]
    assert candidates[0]["selection_eligible"] is False
    assert all(row["selection_eligible"] for row in candidates[1:])
    assert audit["field_hashes_unchanged"] is True
    assert audit["patient_heldout_used"] is False
    assert audit["natural_kmeans_used_for_selection"] is False


def test_expansion_rejects_a_nominee_without_complete_fit_coverage():
    manifest, summary, nomination = _inputs()
    candidate_id = nomination["candidate_ids"][0]
    for row in summary["rows"]:
        if row["candidate_id"] == candidate_id:
            row["fit_valid"] = False
    with pytest.raises(RuntimeError, match="invalid Stage-Z nominee"):
        build_candidates(manifest, summary, nomination, {
            "anchor_candidate_id": "stage_z_anchor",
            "expected_nominee_count": 6,
            "expected_candidate_count": 7,
        })
