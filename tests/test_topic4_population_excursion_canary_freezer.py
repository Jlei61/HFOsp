import copy

import pytest

from scripts.freeze_topic4_rev12_population_excursion_canary import (
    build_manifest_payload,
)


def _config():
    return {
        "event_unit": {
            "name": "population_excursion",
            "contact_geometry_used_for_boundary": False,
        },
        "field_search": {
            "source_candidate_ids": ["field_a", "field_b"],
            "purpose": "event-unit canary only; no candidate selection",
        },
    }


def _payload(config=None):
    return build_manifest_payload(
        _config() if config is None else config,
        {"candidates": [
            {"candidate_id": "field_a", "node_field": {"value": 1}},
            {"candidate_id": "field_b", "node_field": {"value": 2}},
            {"candidate_id": "field_c", "node_field": {"value": 3}},
        ]},
        audit_status="REV12ND_POPULATION_EXCURSION_AUDIT_COMPLETE",
        config_path="config/canary.json", config_sha256="abc",
        inputs={"source": {"path": "source.json", "sha256": "def"}},
        git_commit="123",
    )


def test_manifest_freezes_exactly_two_nonselective_legacy_fields():
    payload = _payload()
    assert payload["selection_forbidden"] is True
    assert [row["candidate_id"] for row in payload["candidates"]] == [
        "field_a", "field_b",
    ]
    assert all(
        row["event_unit_role"] == "nonselective_population_excursion_canary"
        for row in payload["candidates"]
    )


@pytest.mark.parametrize("candidate_ids", [
    ["field_a"], ["field_a", "field_a"],
    ["field_a", "field_b", "field_c"],
])
def test_manifest_rejects_wrong_or_duplicate_candidate_count(candidate_ids):
    config = copy.deepcopy(_config())
    config["field_search"]["source_candidate_ids"] = candidate_ids
    with pytest.raises(RuntimeError, match="exactly two distinct"):
        _payload(config)


def test_manifest_rejects_contact_defined_boundaries():
    config = copy.deepcopy(_config())
    config["event_unit"]["contact_geometry_used_for_boundary"] = True
    with pytest.raises(RuntimeError, match="Contact geometry|contact geometry"):
        _payload(config)
