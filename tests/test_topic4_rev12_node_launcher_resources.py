import pytest

from scripts.launch_topic4_rev12_node_workers import _resource_contract


def test_launcher_resource_override_can_reduce_frozen_parallelism():
    resources = {"maximum_workers": 24, "estimated_worker_gib": 4.0}
    assert _resource_contract(
        resources, maximum_workers=16, estimated_worker_gib=8.0,
    ) == (16, 8.0)
    assert _resource_contract(
        resources, maximum_workers=None, estimated_worker_gib=None,
    ) == (24, 4.0)


def test_launcher_rejects_nonpositive_resource_override():
    with pytest.raises(RuntimeError):
        _resource_contract(
            {"maximum_workers": 24}, maximum_workers=0,
            estimated_worker_gib=8.0,
        )

