import pytest

from scripts.freeze_topic4_rev12_exact_neuron_causal_field_fit import (
    copy_exact_candidates,
)


def test_exact_neuron_fit_copies_field_library_without_mutation():
    source = {"candidates": [
        {"candidate_id": "a", "node_field": {"field_sha256": "1"}},
        {"candidate_id": "b", "node_field": {"field_sha256": "2"}},
    ]}
    copied = copy_exact_candidates(source, expected_count=2)
    assert copied == source["candidates"]
    with pytest.raises(RuntimeError):
        copy_exact_candidates(source, expected_count=3)

