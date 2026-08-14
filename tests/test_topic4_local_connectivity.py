import copy

import numpy as np
from scipy import sparse

from src.topic4_local_connectivity import (
    FEATURE_NAMES,
    continuous_local_e_source_flow,
    local_pair_features,
)


def _network():
    # Four E and two I neurons. Rows are targets, columns are source-local.
    positions = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],
        [0.5, 0.2], [0.5, 0.8],
    ])
    ampa0 = sparse.csc_matrix(np.array([
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ]))
    ampa1 = sparse.csc_matrix(np.array([
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]))
    gaba = [sparse.csc_matrix(np.array([
        [1.0, 0.0], [0.0, 1.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0], [1.0, 0.0],
    ]))]
    return {
        "pos": positions,
        "NE": 4,
        "NI": 2,
        "ampa_by_delay": [ampa0, ampa1],
        "gaba_by_delay": gaba,
        "ampa_cached_operator": object(),
    }


def _incoming(matrices, rows):
    total = np.zeros(rows.stop - rows.start)
    for matrix in matrices:
        total += np.asarray(matrix[rows, :].sum(axis=1)).ravel()
    return total


def _topology(matrices):
    return [
        (matrix.tocsr().indptr.copy(), matrix.tocsr().indices.copy())
        for matrix in matrices
    ]


def test_local_pair_features_include_off_field_contrast_and_are_finite():
    features = local_pair_features(
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        length_scale=1.0,
    )
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features[0, 0] == 1.0
    assert features[1, 0] == -1.0


def test_zero_coefficients_are_exact_noop_and_drop_ampa_cache():
    net = _network()
    old_gaba = net["gaba_by_delay"][0].copy()
    transformed, audit = continuous_local_e_source_flow(
        net, net["pos"], np.linspace(0.0, 1.0, 6), np.zeros((2, 6)),
        l_ee=1.0, l_e_to_i=1.0,
    )
    assert audit["exact_noop"]
    assert audit["ampa_data_unchanged"]
    assert audit["edge_ratio"]["min"] == 1.0
    assert "ampa_cached_operator" not in transformed
    for old, new in zip(net["ampa_by_delay"], transformed["ampa_by_delay"]):
        assert (old != new).nnz == 0
        assert old is not new
    assert (old_gaba != transformed["gaba_by_delay"][0]).nnz == 0


def test_pathway_budgets_topology_delays_and_gaba_are_preserved():
    net = _network()
    old_topology = _topology(net["ampa_by_delay"])
    old_ee = _incoming(net["ampa_by_delay"], slice(0, 4))
    old_ei = _incoming(net["ampa_by_delay"], slice(4, 6))
    old_gaba = copy.deepcopy(net["gaba_by_delay"])
    coefficients = np.array([
        [0.25, 0.4, -0.1, 0.2, 0.15, -0.2],
        [-0.2, 0.3, 0.1, -0.15, -0.1, 0.25],
    ])
    transformed, audit = continuous_local_e_source_flow(
        net, net["pos"], np.array([1.0, 0.8, 0.2, 0.0, 0.7, 0.1]),
        coefficients, l_ee=1.0, l_e_to_i=0.7,
    )
    assert not audit["exact_noop"]
    assert audit["topology_unchanged"]
    assert audit["delay_assignment_unchanged"]
    assert audit["gaba_unchanged"]
    assert not audit["ampa_data_unchanged"]
    assert audit["edge_ratio"]["min"] < audit["edge_ratio"]["max"]
    np.testing.assert_allclose(_incoming(transformed["ampa_by_delay"], slice(0, 4)), old_ee)
    np.testing.assert_allclose(_incoming(transformed["ampa_by_delay"], slice(4, 6)), old_ei)
    for (old_ptr, old_idx), matrix in zip(old_topology, transformed["ampa_by_delay"]):
        csr = matrix.tocsr()
        np.testing.assert_array_equal(csr.indptr, old_ptr)
        np.testing.assert_array_equal(csr.indices, old_idx)
    for old, new in zip(old_gaba, transformed["gaba_by_delay"]):
        assert (old != new).nnz == 0


def test_ee_and_e_to_i_coefficients_are_pathway_isolated():
    net = _network()
    h = np.array([1.0, 0.8, 0.2, 0.0, 0.7, 0.1])
    ee_only = np.zeros((2, 6)); ee_only[0, 0] = 0.5
    ei_only = np.zeros((2, 6)); ei_only[1, 0] = 0.5
    mapped_ee, _ = continuous_local_e_source_flow(
        net, net["pos"], h, ee_only, l_ee=1.0, l_e_to_i=1.0,
    )
    mapped_ei, _ = continuous_local_e_source_flow(
        net, net["pos"], h, ei_only, l_ee=1.0, l_e_to_i=1.0,
    )
    for old, ee, ei in zip(net["ampa_by_delay"], mapped_ee["ampa_by_delay"], mapped_ei["ampa_by_delay"]):
        assert (old[:4, :] != ee[:4, :]).nnz > 0
        assert (old[4:, :] != ee[4:, :]).nnz == 0
        assert (old[:4, :] != ei[:4, :]).nnz == 0
        assert (old[4:, :] != ei[4:, :]).nnz > 0
