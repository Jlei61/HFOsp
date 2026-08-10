import numpy as np
import pytest
from scipy import sparse

from src.topic4_component_pair_edge import (
    component_pair_normalized_ee,
    eta_matrix,
    normalized_component_responsibilities,
)
from src.topic4_core_connectivity import incoming_ee_weight


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


def test_eta_direction_is_target_row_source_column():
    np.testing.assert_array_equal(
        eta_matrix([1.0, 2.0, 3.0, 4.0]),
        [[1.0, 3.0], [4.0, 2.0]])


def test_component_responsibilities_preserve_zero_mass_tail():
    result = normalized_component_responsibilities(np.asarray([
        [2.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
    np.testing.assert_allclose(result[0], [0.5, 0.25, 0.25])
    np.testing.assert_array_equal(result[1], [0.0, 0.0, 0.0])


def test_component_pair_mapper_preserves_all_structural_contracts():
    net = _net()
    responsibilities = np.eye(3)
    before = incoming_ee_weight(net["ampa_by_delay"], 3)
    mapped, diagnostics = component_pair_normalized_ee(
        net, responsibilities, [0.5, -0.25, 1.0, -0.5])
    np.testing.assert_allclose(
        incoming_ee_weight(mapped["ampa_by_delay"], 3), before,
        rtol=0.0, atol=1e-14)
    assert diagnostics["topology_unchanged"]
    assert diagnostics["e_to_i_unchanged"]
    assert diagnostics["gaba_unchanged"]
    assert diagnostics["component_3_parameterized"] is False
    assert "ampa_flat" not in mapped


def test_eta1_from_2_increases_component2_source_into_component1_target():
    net = _net()
    responsibilities = np.eye(3)
    mapped, _ = component_pair_normalized_ee(
        net, responsibilities, [0.0, 0.0, 2.0, 0.0])
    old = sum(net["ampa_by_delay"])
    new = sum(mapped["ampa_by_delay"])
    assert new[0, 1] / old[0, 1] > new[0, 2] / old[0, 2]


def test_zero_eta_is_exact_noop():
    net = _net()
    mapped, diagnostics = component_pair_normalized_ee(
        net, np.eye(3), np.zeros(4))
    for new, old in zip(mapped["ampa_by_delay"], net["ampa_by_delay"]):
        np.testing.assert_array_equal(new.data, old.data)
        np.testing.assert_array_equal(new.indices, old.indices)
        np.testing.assert_array_equal(new.indptr, old.indptr)
    assert diagnostics["exact_noop"]


@pytest.mark.parametrize("eta", ([1.0, 2.0], [0.0, np.nan, 0.0, 0.0]))
def test_mapper_rejects_malformed_eta(eta):
    with pytest.raises(ValueError):
        component_pair_normalized_ee(_net(), np.eye(3), eta)
