import numpy as np
import pytest
from scipy import sparse

from src.topic4_core_connectivity import incoming_ee_weight
from src.topic4_graph_edge_flow import (
    build_directed_spectral_basis,
    graph_spectral_ee_flow,
    reconstructed_spectral_field,
    sample_spectral_edge_features,
    spectral_response_weights,
    summed_ee_operator,
    two_sided_normalized_operator,
)


def _network():
    first = sparse.csc_matrix(np.asarray([
        [0.0, 2.0, 1.0, 0.0],
        [1.0, 0.0, 3.0, 0.0],
        [2.0, 1.0, 0.0, 0.0],
        [4.0, 5.0, 6.0, 0.0],
    ]))
    second = sparse.csc_matrix(np.asarray([
        [0.0, 0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]))
    return {
        "NE": 3, "NI": 1, "ampa_by_delay": [first, second],
        "gaba_by_delay": [sparse.csc_matrix((4, 1))],
        "ampa_cached": object(), "_ampa_derived_cache_keys": ["ampa_cached"],
    }


def _manual_basis(net):
    return {
        "u": np.asarray([[1.0, 0.2], [-0.3, 1.1], [0.7, -0.4]]),
        "v": np.asarray([[0.1, 1.0], [1.2, -0.2], [-0.5, 0.8]]),
        "singular_values": np.asarray([0.8, 0.5]),
        "rank": 2, "n_e": 3,
        "graph_weight_sha256": __import__(
            "src.topic4_core_connectivity", fromlist=["_hash_sparse_bins"]
        )._hash_sparse_bins(net["ampa_by_delay"], rows=slice(0, 3)),
    }


def test_summed_operator_uses_target_rows_and_source_columns():
    net = _network()
    result = summed_ee_operator(net["ampa_by_delay"], 3).toarray()
    np.testing.assert_allclose(result, np.asarray([
        [0.0, 2.0, 1.5], [1.5, 0.0, 3.0], [2.0, 1.5, 0.0],
    ]))
    normalized, incoming, outgoing = two_sided_normalized_operator(result)
    np.testing.assert_allclose(incoming, [3.5, 4.5, 3.5])
    np.testing.assert_allclose(outgoing, [3.5, 3.5, 4.5])
    assert normalized.shape == (3, 3)


def test_spectral_response_is_invariant_within_degenerate_block():
    rng = np.random.default_rng(4)
    u, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    v, _ = np.linalg.qr(rng.normal(size=(5, 2)))
    singular = np.asarray([0.8, 0.8])
    coefficients = np.asarray([0.3, -0.5])
    angle = 0.63
    rotation = np.asarray([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)],
    ])
    original = reconstructed_spectral_field(u, v, singular, coefficients)
    rotated = reconstructed_spectral_field(
        u @ rotation, v @ rotation, singular, coefficients,
    )
    np.testing.assert_allclose(rotated, original, atol=1e-12)
    response = spectral_response_weights(singular, coefficients)
    assert response[0] == pytest.approx(response[1])


def test_graph_flow_zero_is_exact_noop_and_invalidates_caches():
    net = _network()
    mapped, audit = graph_spectral_ee_flow(
        net, _manual_basis(net), np.zeros(2),
    )
    assert audit["exact_noop"]
    assert audit["ampa_data_unchanged"]
    assert "ampa_cached" not in mapped
    for old, new in zip(net["ampa_by_delay"], mapped["ampa_by_delay"]):
        np.testing.assert_array_equal(old.toarray(), new.toarray())
        assert old is not new


def test_graph_flow_preserves_topology_delays_and_target_budget():
    net = _network()
    before = incoming_ee_weight(net["ampa_by_delay"], 3)
    mapped, audit = graph_spectral_ee_flow(
        net, _manual_basis(net), np.asarray([0.25, -0.15]),
        ratio_sample_limit=4,
    )
    np.testing.assert_allclose(
        incoming_ee_weight(mapped["ampa_by_delay"], 3), before,
        atol=1e-12, rtol=0.0,
    )
    assert audit["max_abs_incoming_E_error"] <= 1e-9
    assert audit["topology_unchanged"]
    assert audit["delay_assignment_unchanged"]
    assert audit["e_to_i_unchanged"]
    assert audit["gaba_unchanged"]
    assert not audit["ampa_data_unchanged"]
    assert audit["edge_ratio"]["min"] > 0.0
    for old, new in zip(net["ampa_by_delay"], mapped["ampa_by_delay"]):
        np.testing.assert_array_equal(old.nonzero(), new.nonzero())
        np.testing.assert_allclose(old.toarray()[3], new.toarray()[3])


def test_edge_feature_sample_reconstructs_logits():
    net = _network()
    basis = _manual_basis(net)
    sample = sample_spectral_edge_features(
        net["ampa_by_delay"], basis, sample_limit=100,
    )
    coefficients = np.asarray([0.25, -0.15])
    predicted = sample["features"] @ coefficients
    observed = []
    for matrix in net["ampa_by_delay"]:
        coo = matrix.tocoo()
        selected = coo.row < 3
        rows, columns = coo.row[selected], coo.col[selected]
        u, v = basis["u"], basis["v"]
        response = spectral_response_weights(
            basis["singular_values"], coefficients,
        )
        observed.extend(np.sum(
            u[rows] * response[None, :] * v[columns], axis=1,
        ))
    np.testing.assert_allclose(predicted, observed)
    assert sample["features"].shape[1] == 2
    assert np.all(sample["feature_abs_max"] > 0.0)


def test_real_basis_is_deterministic_and_drops_leading_mode():
    rng = np.random.default_rng(8)
    dense = rng.uniform(0.2, 1.5, size=(8, 8))
    dense[rng.uniform(size=(8, 8)) < 0.25] = 0.0
    np.fill_diagonal(dense, 0.0)
    bins = [sparse.csc_matrix(dense)]
    left = build_directed_spectral_basis(
        bins, 8, rank=2, extra_modes=1, tolerance=1e-12,
    )
    right = build_directed_spectral_basis(
        bins, 8, rank=2, extra_modes=1, tolerance=1e-12,
    )
    np.testing.assert_allclose(left["u"], right["u"], atol=1e-10)
    np.testing.assert_allclose(left["v"], right["v"], atol=1e-10)
    np.testing.assert_allclose(
        np.mean(left["u"] ** 2, axis=0), np.ones(2), atol=1e-10,
    )
    assert left["leading_degree_singular_value"] > left["singular_values"][0]
    assert len(left["all_computed_singular_values"]) == 4
