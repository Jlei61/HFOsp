import copy

import pytest

from scripts.freeze_topic4_rev12_native_lineage_canary import (
    build_manifest_payload,
)


def _config():
    return {
        "event_unit": {"name": "directed_spatiotemporal_lineage"},
        "search": {"contact_readout": {
            "source": "lineage_restricted_sheet_activity",
        }},
        "field_search": {"candidate_ids": ["field_a"]},
    }


def _source():
    return {"candidates": [{
        "candidate_id": "field_a",
        "node_field": {"field_sha256": "abc"},
    }]}


def test_native_canary_freezer_keeps_only_the_requested_field():
    payload = build_manifest_payload(
        _config(), _source(), config_path="config.json",
        config_sha256="cfg", inputs={}, git_commit="commit",
    )
    assert payload["status"] == "REV12ND_NATIVE_LINEAGE_CANARY_FROZEN"
    assert [row["candidate_id"] for row in payload["candidates"]] == ["field_a"]
    assert payload["simulation_rerun"] is True


def test_native_canary_freezer_rejects_unrestricted_contact_readout():
    config = copy.deepcopy(_config())
    config["search"]["contact_readout"]["source"] = (
        "full_contact_envelope_within_lineage_window"
    )
    with pytest.raises(RuntimeError, match="root-restricted"):
        build_manifest_payload(
            config, _source(), config_path="config.json",
            config_sha256="cfg", inputs={}, git_commit="commit",
        )

