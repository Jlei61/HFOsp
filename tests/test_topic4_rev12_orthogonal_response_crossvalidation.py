import numpy as np

from scripts.audit_topic4_rev12_orthogonal_response_crossvalidation import (
    crossvalidated_common_direction,
)


def test_crossvalidation_supports_shared_network_direction():
    seeds = list(range(9))
    base = np.asarray([1.0, 0.5, 0.2])
    gradients = {
        name: np.asarray([base + 0.01 * index for index in range(9)])
        for name in ("a", "b", "c")
    }
    result = crossvalidated_common_direction(
        gradients, seeds, ["a", "b", "c"],
        minimum_positive_networks_per_endpoint=6,
        minimum_joint_positive_networks=6,
        minimum_median_heldout_margin=0.0,
    )
    assert result["supported"] is True
    assert result["joint_positive_networks"] == 9


def test_crossvalidation_rejects_network_specific_opposition():
    seeds = list(range(9))
    gradients = {
        "a": np.asarray([[1.0, 0.0]] * 9),
        "b": np.asarray([[1.0, 0.0]] * 5 + [[-1.0, 0.0]] * 4),
    }
    result = crossvalidated_common_direction(
        gradients, seeds, ["a", "b"],
        minimum_positive_networks_per_endpoint=6,
        minimum_joint_positive_networks=6,
        minimum_median_heldout_margin=0.0,
    )
    assert result["supported"] is False
    assert result["joint_positive_networks"] < 6
