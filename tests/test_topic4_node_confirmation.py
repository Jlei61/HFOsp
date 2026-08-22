import json
from pathlib import Path

import pytest

from scripts.freeze_topic4_rev12_node_confirmation import (
    build_confirmation_candidates,
)


ROOT = Path(__file__).resolve().parents[1]


def _manifest(*ids):
    return {
        "candidates": [
            {"candidate_id": candidate_id, "node_field": {"value": index}}
            for index, candidate_id in enumerate(ids)
        ]
    }


def test_confirmation_candidates_follow_frozen_selection():
    rows = build_confirmation_candidates(
        _manifest("anchor_field"),
        _manifest("stage_b_r01_m", "stage_b_r02_m"),
        {
            "status": "REV12ND_NODE_SELECTION_COMPLETE",
            "selected_candidate_id": "stage_b_r01_m",
        },
        "anchor_field",
        "stage_b_r01_m",
    )
    assert [row["candidate_id"] for row in rows] == [
        "anchor_field", "stage_b_r01_m",
    ]
    assert rows[0]["confirmation_role"] == "frozen_fig4_node_baseline"
    assert rows[1]["confirmation_role"] == "pareto_selected_node_field"


def test_confirmation_rejects_configured_selection_drift():
    with pytest.raises(RuntimeError, match="differs from frozen decision"):
        build_confirmation_candidates(
            _manifest("anchor_field"),
            _manifest("stage_b_r01_m", "stage_b_r02_m"),
            {
                "status": "REV12ND_NODE_SELECTION_COMPLETE",
                "selected_candidate_id": "stage_b_r02_m",
            },
            "anchor_field",
            "stage_b_r01_m",
        )


def test_confirmation_config_uses_worker_scientific_role():
    config = json.loads(
        (ROOT / "config/topic4_rev12_nd_node_confirmation.json").read_text()
    )
    assert config["scientific_role"] == "development_only_node_dualmode_refit"
