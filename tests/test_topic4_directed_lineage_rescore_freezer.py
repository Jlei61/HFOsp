import pytest

from scripts.freeze_topic4_rev12_directed_lineage_rescore import (
    build_manifest_payload,
)


def _config():
    return {
        "event_unit": {
            "name": "directed_spatiotemporal_lineage",
            "contact_geometry_used_for_boundary": False,
        },
        "cascade_objective": {"sample_size_per_side": 6},
    }


def _source():
    return {"candidates": [
        {"candidate_id": f"field_{index}"} for index in range(18)
    ]}


def test_directed_lineage_manifest_preserves_full_historical_field_set():
    payload = build_manifest_payload(
        _config(), _source(), config_path="config.json", config_sha256="abc",
        inputs={}, git_commit="deadbeef",
    )
    assert payload["simulation_rerun"] is False
    assert len(payload["candidates"]) == 18
    assert payload["event_unit"]["contact_geometry_used_for_boundary"] is False


def test_directed_lineage_manifest_rejects_contact_defined_boundaries():
    config = _config()
    config["event_unit"]["contact_geometry_used_for_boundary"] = True
    with pytest.raises(RuntimeError, match="Contacts cannot|contacts cannot"):
        build_manifest_payload(
            config, _source(), config_path="config.json", config_sha256="abc",
            inputs={}, git_commit="deadbeef",
        )
