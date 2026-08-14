import numpy as np
from scipy import sparse

from src.topic4_fcxr_lc6_surround import (
    EToIGraph,
    assign_frozen_target_weights,
    audit_basic_legality,
    compare_outdegree_to_c0,
    construction_q,
    empirical_edge_widths,
    extract_e_to_e,
    extract_e_to_i,
    extract_i_to_e,
    graph_sha256,
    metropolis_hastings_acceptance,
    recompute_physical_delays,
    replace_e_to_i_in_net,
    rewire_e_to_i_targetwise,
    source_outdegree_audit,
    validate_q_target,
)


def _small_net():
    ne, ni = 6, 2
    rows = np.array([6, 6, 7, 7])
    cols = np.array([0, 2, 1, 5])
    data = np.array([1.5, 1.5, 2.5, 2.5])
    d1 = sparse.csc_matrix((data[:2], (rows[:2], cols[:2])), shape=(8, 6))
    d2 = sparse.csc_matrix((data[2:], (rows[2:], cols[2:])), shape=(8, 6))
    empty = sparse.csc_matrix((8, 6))
    empty_gaba = sparse.csc_matrix((8, 2))
    return {
        "ampa_by_delay": [empty, d1, d2],
        "gaba_by_delay": [empty_gaba, empty_gaba, empty_gaba],
    }, ne, ni


def test_extract_target_first_ie_means_e_source_i_target():
    net, ne, ni = _small_net()
    graph = extract_e_to_i(net, ne, ni)
    assert graph.sources.tolist() == [[0, 2], [1, 5]]
    assert graph.delay_steps.tolist() == [[1, 1], [2, 2]]
    assert graph.weights.tolist() == [[1.5, 1.5], [2.5, 2.5]]


def test_extract_other_population_directions_and_construction_q():
    ne, ni = 2, 1
    ee = sparse.csc_matrix((np.ones(2), ([0, 1], [1, 0])), shape=(3, 2))
    ie = sparse.csc_matrix((np.ones(1), ([2], [0])), shape=(3, 2))
    ampa = ee + ie
    gaba = sparse.csc_matrix((np.ones(2), ([0, 1], [0, 0])), shape=(3, 1))
    net = {"ampa_by_delay": [ampa], "gaba_by_delay": [gaba]}
    assert extract_e_to_e(net, ne).sources.shape == (2, 1)
    assert extract_i_to_e(net, ne, ni).sources.tolist() == [[0], [0]]
    q = construction_q(
        {"sigma_parallel_mm": 3.0}, {"sigma_parallel_mm": 4.0},
        {"sigma_parallel_mm": 5.0},
    )
    assert q == 1.0


def test_weight_multiset_is_exact_and_not_distance_fitted():
    base = np.array([[1.0, 4.0, 2.0], [7.0, 7.0, 7.0]])
    sources = np.array([[1, 5, 8], [2, 3, 4]])
    first = assign_frozen_target_weights(base, sources, graph_seed=61)
    second = assign_frozen_target_weights(base, sources, graph_seed=61)
    assert np.array_equal(first, second)
    assert np.array_equal(np.sort(first, axis=1), np.sort(base, axis=1))
    assert first[1].tolist() == [7.0, 7.0, 7.0]


def test_delay_is_tau0_plus_distance_over_v_then_quantized():
    pos_e = np.array([[0.0, 0.0], [3.0, 4.0]])
    pos_i = np.array([[0.0, 0.0]])
    steps = recompute_physical_delays(
        np.array([[0, 1]]), pos_e, pos_i, tau0_ms=1.0,
        v_axon_mm_per_ms=2.0, delay_dt_ms=0.5, engine_dt_ms=0.05,
    )
    # 1.0 ms -> round(2)*10; 3.5 ms -> round(7)*10.
    assert steps.tolist() == [[20, 70]]


def test_general_hastings_term_is_not_silently_dropped():
    symmetric = metropolis_hastings_acceptance(0.0, -1.0)
    asymmetric = metropolis_hastings_acceptance(
        0.0, -1.0, log_q_reverse=np.log(4.0), log_q_forward=0.0,
    )
    assert np.isclose(symmetric, np.exp(-1.0))
    assert asymmetric == 1.0


def test_targetwise_replacement_is_reproducible_legal_and_rng_isolated():
    rng = np.random.default_rng(4)
    pos_e = rng.uniform(0, 2, size=(30, 2))
    pos_i = rng.uniform(0, 2, size=(4, 2))
    base = np.vstack([np.sort(rng.choice(30, 5, replace=False)) for _ in range(4)])
    runtime_rng = np.random.default_rng(999)
    runtime_state = repr(runtime_rng.bit_generator.state)
    a, da = rewire_e_to_i_targetwise(
        base, pos_e, pos_i, [1, 0], l_parallel=.8, l_perpendicular=.3,
        graph_seed=123, n_sweeps=3, proposal_block_size=2,
    )
    b, db = rewire_e_to_i_targetwise(
        base, pos_e, pos_i, [1, 0], l_parallel=.8, l_perpendicular=.3,
        graph_seed=123, n_sweeps=3, proposal_block_size=2,
    )
    assert np.array_equal(a, b)
    assert da == db
    assert all(len(np.unique(row)) == 5 for row in a)
    assert repr(runtime_rng.bit_generator.state) == runtime_state
    assert da["proposal"].endswith("symmetric")


def test_stratified_asymmetric_proposal_uses_hastings_and_preserves_perp_bins():
    rng = np.random.default_rng(8)
    pos_e = rng.uniform(0, 3, size=(80, 2))
    pos_i = rng.uniform(.5, 2.5, size=(3, 2))
    base = np.vstack([np.sort(rng.choice(80, 12, replace=False)) for _ in range(3)])
    axis = np.array([1.0, 0.0])
    bin_width = .5
    candidate, diag = rewire_e_to_i_targetwise(
        base, pos_e, pos_i, axis, l_parallel=1.0, l_perpendicular=.3,
        graph_seed=919, n_sweeps=2, proposal_block_size=1,
        proposal_perpendicular_bin_mm=bin_width,
    )
    for target in range(len(pos_i)):
        base_bin = np.floor((pos_e[base[target], 1] - pos_i[target, 1]) / bin_width)
        candidate_bin = np.floor((pos_e[candidate[target], 1] - pos_i[target, 1]) / bin_width)
        assert np.array_equal(np.sort(base_bin), np.sort(candidate_bin))
    assert diag["proposal"].endswith("asymmetric")
    assert "forward_reverse" in diag["hastings_correction"]


def test_graph_hash_and_off_path_are_exact():
    sources = np.array([[0, 2], [1, 5]], dtype=np.int32)
    weights = np.array([[1.5, 1.5], [2.5, 2.5]])
    delays = np.array([[1, 1], [2, 2]], dtype=np.int32)
    base = EToIGraph(sources, weights, delays)
    off = EToIGraph(sources.copy(), weights.copy(), delays.copy())
    assert graph_sha256(base) == graph_sha256(off)
    audit = audit_basic_legality(base, off, ne=6)
    assert audit["target_in_degree"] == 2
    assert audit["per_target_weight_multiset_exact"] is True


def test_replace_net_changes_only_e_to_i_and_preserves_off_path():
    net, ne, ni = _small_net()
    base = extract_e_to_i(net, ne, ni)
    updated = replace_e_to_i_in_net(net, base, ne=ne, ni=ni)
    extracted = extract_e_to_i(updated, ne, ni)
    assert graph_sha256(extracted) == graph_sha256(base)
    for old, new in zip(net["ampa_by_delay"], updated["ampa_by_delay"]):
        delta = old[:ne, :ne] - new[:ne, :ne]
        assert delta.nnz == 0


def test_outdegree_audit_and_relative_contract():
    pos = np.array([
        [.1, .1], [1., 1.], [1.5, 1.5], [1.9, 1.9], [.2, 1.8], [1.8, .2],
    ])
    sources = np.array([[0, 1, 2], [2, 3, 4], [1, 4, 5]])
    base = source_outdegree_audit(
        sources, pos, [1, 0], sheet_size_mm=2.0, edge_margin_mm=.5,
    )
    comparison = compare_outdegree_to_c0(base, base)
    assert comparison["within_contract"] is True


def test_empirical_width_and_q_target_fail_closed():
    pos_e = np.array([[0., 0.], [2., 0.], [0., 1.], [2., 1.]])
    pos_i = np.array([[1., 0.], [1., 1.]])
    sources = np.array([[0, 1], [2, 3]])
    width = empirical_edge_widths(sources, pos_e, pos_i, [1, 0], chunk_targets=1)
    assert np.isclose(width["sigma_parallel_mm"], 1.0)
    assert np.isclose(width["sigma_perpendicular_mm"], 0.0)
    validate_q_target(1.03, 1.0, 0.05)
    try:
        validate_q_target(1.08, 1.0, 0.05)
    except RuntimeError as exc:
        assert "unreachable" in str(exc)
    else:
        raise AssertionError("out-of-tolerance construction q must fail closed")
