from __future__ import annotations

import json

from scripts.topic5_continuous_marked_state_h2b.run_v03_full_grid_followup import (
    _claim_route_status,
)


def test_followup_claim_route_stays_closed_without_a1_and_final_a2(tmp_path) -> None:
    root = tmp_path / "v03"
    (root / "qualification").mkdir(parents=True)
    (root / "qualification/state_qualified_manifest.json").write_text(
        json.dumps({"subjects": []}), encoding="utf-8",
    )
    qualified, assay_pass = _claim_route_status(root)
    assert qualified == set()
    assert assay_pass is False


def test_followup_claim_route_requires_final_acceptance_receipt(tmp_path) -> None:
    root = tmp_path / "v03"
    (root / "qualification").mkdir(parents=True)
    (root / "assay").mkdir(parents=True)
    (root / "qualification/state_qualified_manifest.json").write_text(
        json.dumps({"subjects": ["epilepsiae_1125"]}), encoding="utf-8",
    )
    (root / "assay/type1_power_summary.json").write_text(json.dumps({
        "status": "PASS_FINAL_ASSAY_ACCEPTANCE",
        "claim_bearing_route_released": True,
    }), encoding="utf-8")
    qualified, assay_pass = _claim_route_status(root)
    assert qualified == {"epilepsiae_1125"}
    assert assay_pass is True
