import numpy as np
import pytest
from scipy import sparse

from src.topic4_component_pair_edge import (
    GAMMA_NAMES,
    component_background_membership,
    component_pair_normalized_ee,
    gamma_matrix,
)
from src.topic4_core_connectivity import field_normalized_ee_pair, incoming_ee_weight


def _net():
    first = sparse.csc_matrix(np.asarray([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 3.0, 4.0],
    ]))
    second = sparse.csc_matrix(np.asarray([
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 1.0],
    ]))
    return {
        "NE": 3, "NI": 1, "ampa_by_delay": [first, second],
        "gaba_by_delay": [sparse.csc_matrix((4, 1))],
        "ampa_flat": ("stale",),
    }


def test_gamma_direction_is_target_row_source_column():
    np.testing.assert_array_equal(
        gamma_matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert len(GAMMA_NAMES) == 6


def test_component_membership_preserves_far_field_as_background():
    result = component_background_membership(
        np.asarray([0.8, 0.0]),
        np.asarray([[2.0, 1.0, 1.0], [1e-99, 2e-99, 3e-99]]))
    np.testing.assert_allclose(result[0], [0.4, 0.2, 0.2, 0.2])
    np.testing.assert_array_equal(result[1], [0.0, 0.0, 0.0, 1.0])


def _membership():
    return np.asarray([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def test_component_pair_mapper_preserves_all_structural_contracts():
    net = _net()
    before = incoming_ee_weight(net["ampa_by_delay"], 3)
    mapped, diagnostics = component_pair_normalized_ee(
        net, np.asarray([1.0, 1.0, 0.0]), _membership(),
        [0.5, -0.25, 1.0, -0.5, 0.2, -0.1])
    np.testing.assert_allclose(
        incoming_ee_weight(mapped["ampa_by_delay"], 3), before,
        rtol=0.0, atol=1e-14)
    assert diagnostics["topology_unchanged"]
    assert diagnostics["e_to_i_unchanged"]
    assert diagnostics["gaba_unchanged"]
    assert diagnostics["component_3_parameterized"] is False
    assert "ampa_flat" not in mapped


def test_gamma_c1_from_c2_increases_component2_source_into_component1_target():
    net = _net()
    mapped, _ = component_pair_normalized_ee(
        net, np.asarray([1.0, 1.0, 0.0]), _membership(),
        [0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    old = sum(net["ampa_by_delay"])
    new = sum(mapped["ampa_by_delay"])
    assert new[0, 1] / old[0, 1] > new[0, 2] / old[0, 2]


def test_zero_gamma_is_exact_scalar_alpha_baseline():
    net = _net()
    h = np.asarray([1.0, 0.5, 0.0])
    mapped, diagnostics = component_pair_normalized_ee(
        net, h, _membership(), np.zeros(6), alpha=0.75)
    scalar, _ = field_normalized_ee_pair(net, h, 0.75)
    for new, expected in zip(mapped["ampa_by_delay"], scalar["ampa_by_delay"]):
        np.testing.assert_array_equal(new.data, expected.data)
        np.testing.assert_array_equal(new.indices, expected.indices)
        np.testing.assert_array_equal(new.indptr, expected.indptr)
    assert diagnostics["residual_exact_noop"]
    assert diagnostics["edge_ratio"]["max"] != 1.0


@pytest.mark.parametrize(
    "gamma",
    ([1.0, 2.0], [0.0, np.nan, 0.0, 0.0, 0.0, 0.0]),
)
def test_mapper_rejects_malformed_gamma(gamma):
    with pytest.raises(ValueError):
        component_pair_normalized_ee(
            _net(), np.ones(3), _membership(), gamma)
