import pytest

from scripts.freeze_topic4_rev12_persistent_lineage_canary import select_candidates


def test_persistent_lineage_canary_preserves_requested_field_order():
    source = {"candidates": [
        {"candidate_id": "a", "node_field": {"field_sha256": "1"}},
        {"candidate_id": "b", "node_field": {"field_sha256": "2"}},
    ]}
    assert [row["candidate_id"] for row in select_candidates(
        source, ["b", "a"],
    )] == ["b", "a"]
    with pytest.raises(RuntimeError):
        select_candidates(source, ["a", "a"])
