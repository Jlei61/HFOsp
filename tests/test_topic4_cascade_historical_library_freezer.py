import copy

import pytest

from scripts.freeze_topic4_rev12_cascade_historical_library import (
    build_manifest_payload,
)


def _config():
    return {
        "event_unit": {
            "name": "spatiotemporal_cascade",
            "contact_geometry_used_for_boundary": False,
        },
        "cascade_objective": {"sample_size_per_side": 6},
        "field_search": {"new_field_parameters_released": False},
    }


def _source(n=18):
    return {"candidates": [
        {"candidate_id": f"field_{index:02d}", "node_field": {"value": index}}
        for index in range(n)
    ]}


def _payload(config=None, source=None):
    return build_manifest_payload(
        _config() if config is None else config,
        _source() if source is None else source,
        inputs={}, config_path="config/library.json",
        config_sha256="abc", git_commit="123",
    )


def test_historical_library_freezes_all_18_fields_without_new_parameters():
    payload = _payload()
    assert len(payload["candidates"]) == 18
    assert payload["new_field_parameters_released"] is False
    assert all(
        row["cascade_library_role"] == "historical_field_under_corrected_objective"
        for row in payload["candidates"]
    )


def test_historical_library_rejects_missing_field_or_contact_boundary():
    with pytest.raises(RuntimeError, match="18 unique"):
        _payload(source=_source(17))
    config = copy.deepcopy(_config())
    config["event_unit"]["contact_geometry_used_for_boundary"] = True
    with pytest.raises(RuntimeError, match="Contact geometry|contact geometry"):
        _payload(config=config)
