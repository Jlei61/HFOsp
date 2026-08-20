import numpy as np
import pytest
import scipy.sparse as sp

from src.topic4_fig5_motif_reuse import (
    NOT_EVALUABLE, audit_edge_permutation, circular_shift,
    matched_off_motif_node_sets, mode_precedence_matrix, precedence_agreement,
    precedence_matrix, precedence_reuse, permute_edge_weights, rank_reuse,
    network_level_aggregate, reuse_trajectory, spearman_with_coverage,
    within_shaft_label_permutation)

SHAFTS = np.asarray(["ICL"] * 8 + ["SCL"] * 7)


def _ordered(n=15, missing=()):
    values = np.arange(float(n))
    values[list(missing)] = np.nan
    return values


def test_spearman_reports_common_coverage_not_silent_dropping():
    a = _ordered(missing=(0, 1, 2))
    b = _ordered(missing=(13, 14))
    row = spearman_with_coverage(a, b)
    assert row["status"] == "OK"
    assert row["n_common"] == 10
    assert row["rho"] == pytest.approx(1.0)


def test_spearman_needs_three_common_contacts():
    a = _ordered(missing=range(13))
    row = spearman_with_coverage(a, _ordered())
    assert row["status"] == NOT_EVALUABLE
    assert row["n_common"] == 2


def test_within_shaft_permutation_preserves_count_and_values():
    rng = np.random.default_rng(0)
    values = _ordered(missing=(1, 3, 9))
    permuted = within_shaft_label_permutation(values, SHAFTS, rng)
    for shaft in ("ICL", "SCL"):
        index = SHAFTS == shaft
        assert np.isfinite(permuted[index]).sum() == np.isfinite(values[index]).sum()
        assert sorted(permuted[index][np.isfinite(permuted[index])].tolist()) == \
            sorted(values[index][np.isfinite(values[index])].tolist())


def test_reproduced_order_beats_the_within_shaft_null():
    event = _ordered()
    early = _ordered()
    result = rank_reuse(event, early, SHAFTS, n_draws=512, seed=1)
    assert result["observed"]["rho"] == pytest.approx(1.0)
    assert result["null"]["exceeds_q95"] is True
    assert result["null"]["exceedance_probability"] < 0.01


def test_unrelated_order_does_not_beat_the_null():
    rng = np.random.default_rng(7)
    event = _ordered()
    early = rng.permutation(_ordered())
    result = rank_reuse(event, early, SHAFTS, n_draws=512, seed=2)
    assert result["null"]["status"] == "OK"
    assert result["null"]["exceedance_probability"] > 0.05


def test_precedence_matrix_is_antisymmetric_and_masks_absent_contacts():
    matrix = precedence_matrix(_ordered(missing=(2,)))
    assert np.isnan(matrix[2]).all() and np.isnan(matrix[:, 2]).all()
    finite = np.isfinite(matrix)
    assert np.allclose(matrix[finite], -matrix.T[finite])


def test_mode_precedence_weights_are_within_mode_consistency():
    consistent = np.vstack([_ordered(), _ordered()])
    mean, support = mode_precedence_matrix(consistent)
    assert np.nanmax(np.abs(mean)) == pytest.approx(1.0)
    mixed = np.vstack([_ordered(), _ordered()[::-1]])
    mean_mixed, _ = mode_precedence_matrix(mixed)
    assert np.nanmax(np.abs(mean_mixed)) < 1e-9
    assert support.max() == 2


def test_precedence_agreement_is_one_for_an_identical_order():
    reference = precedence_matrix(_ordered())
    row = precedence_agreement(reference, _ordered())
    assert row["agreement"] == pytest.approx(1.0)
    reversed_row = precedence_agreement(reference, _ordered()[::-1])
    assert reversed_row["agreement"] == pytest.approx(0.0)


def test_precedence_reuse_against_null_agrees_with_direction():
    reference = precedence_matrix(_ordered())
    result = precedence_reuse(reference, _ordered(), SHAFTS, n_draws=256, seed=3)
    assert result["observed"]["agreement"] == pytest.approx(1.0)
    assert result["null"]["exceeds_q95"] is True


def test_circular_shift_preserves_the_value_multiset():
    values = np.arange(9.0)
    shifted = circular_shift(values, 4)
    assert sorted(shifted.tolist()) == sorted(values.tolist())
    assert not np.array_equal(shifted, values)


def test_reuse_trajectory_needs_four_events():
    assert reuse_trajectory([0.1, 0.2, 0.3], [10.0, 20.0, 30.0],
                            n_draws=16, seed=0)["status"] == NOT_EVALUABLE


def test_network_aggregate_weights_every_network_equally():
    row = network_level_aggregate([0.2, 0.4, 0.6], draws=512, seed=5)
    assert row["n_networks"] == 3
    assert row["mean"] == pytest.approx(0.4)
    assert row["bootstrap_q05"] < row["mean"] < row["bootstrap_q95"]


def _synthetic_graph(n_e=40, n_i=10, n_bins=3, seed=0):
    """Small graph with a planted source-identity motif orthogonal to distance."""
    rng = np.random.default_rng(seed)
    n_total = n_e + n_i
    positions = rng.uniform(0.0, 20.0, size=(n_total, 2))
    motif_sources = np.zeros(n_e, bool)
    motif_sources[rng.choice(n_e, size=n_e // 2, replace=False)] = True
    bins = []
    for _ in range(n_bins):
        rows, cols = [], []
        for target in range(n_total):
            sources = rng.choice(n_e, size=6, replace=False)
            rows.extend([target] * len(sources))
            cols.extend(sources.tolist())
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        data = np.where(motif_sources[cols], 3.0, 1.0) * rng.uniform(
            0.9, 1.1, size=len(cols))
        bins.append(sp.coo_matrix((data, (rows, cols)),
                                  shape=(n_total, n_e)).tocsr())
    return bins, positions, motif_sources, n_e


def _motif_alignment(bins, motif_sources):
    values, flags = [], []
    for matrix in bins:
        coo = matrix.tocoo(copy=False)
        values.append(np.asarray(coo.data, float))
        flags.append(motif_sources[np.asarray(coo.col, np.int64)].astype(float))
    return float(np.corrcoef(np.concatenate(values),
                             np.concatenate(flags))[0, 1])


def test_edge_permutation_preserves_structure_and_destroys_the_motif():
    bins, positions, motif_sources, n_e = _synthetic_graph()
    before = _motif_alignment(bins, motif_sources)
    assert before > 0.8
    rng = np.random.default_rng(11)
    permuted = permute_edge_weights(bins, n_e, positions, rng=rng,
                                    n_distance_bins=4)
    after = _motif_alignment(permuted, motif_sources)
    assert abs(after) < 0.25, f"motif survived the permutation: {after}"

    report = audit_edge_permutation(bins, permuted, n_e, positions,
                                    n_distance_bins=4)
    assert report["topology_unchanged"] is True
    assert report["edge_index_sets_identical"] is True
    assert report["source_degree_identical"] is True
    assert report["target_degree_identical"] is True
    assert report["data_changed"] is True
    assert report["E_to_E_max_abs_incoming_error"] < 1e-9
    assert report["E_to_I_max_abs_incoming_error"] < 1e-9
    assert report["budget_and_degree_joint_contract"] is True


def test_edge_permutation_audit_flags_a_broken_null():
    """A permutation that also moved edges must not pass the structural audit."""
    bins, positions, _, n_e = _synthetic_graph()
    broken = [matrix.copy() for matrix in bins]
    coo = broken[0].tocoo(copy=False)
    rows = np.asarray(coo.row, np.int64).copy()
    rows[0] = (rows[0] + 1) % broken[0].shape[0]
    broken[0] = sp.coo_matrix(
        (np.asarray(coo.data, float), (rows, np.asarray(coo.col, np.int64))),
        shape=broken[0].shape).tocsr()
    report = audit_edge_permutation(bins, broken, n_e, positions)
    assert report["all_structural_clauses_pass"] is False


def test_matched_off_motif_node_sets_raises_instead_of_guessing():
    with pytest.raises(NotImplementedError, match="edge flow"):
        matched_off_motif_node_sets()


def test_a_negative_agreement_clearing_a_lower_null_is_not_called_reuse():
    """The within-shaft null can sit below zero; clearing it is not reuse."""
    from src.topic4_fig5_motif_reuse import _null_summary
    draws = np.full(1000, -0.6)
    row = _null_summary(-0.2, draws, 1000)
    assert row["exceeds_q95"] is True
    assert row["observed_is_positive"] is False
    assert row["reuse_supported"] is False
    positive = _null_summary(0.4, np.full(1000, 0.1), 1000)
    assert positive["reuse_supported"] is True


def test_precomputed_target_matrix_changes_nothing():
    reference = precedence_matrix(_ordered())
    target = _ordered(missing=(3, 7))
    plain = precedence_agreement(reference, target)
    cached = precedence_agreement(reference, target,
                                  target_matrix=precedence_matrix(target))
    assert plain == cached


def test_weight_distribution_is_a_clause_not_a_diagnostic():
    """Renormalisation after permutation moves the marginal weights; the audit
    must fail on that rather than report it beside a passing verdict."""
    bins, positions, _, n_e = _synthetic_graph()
    permuted = permute_edge_weights(bins, n_e, positions,
                                    rng=np.random.default_rng(11),
                                    n_distance_bins=4)
    strict = audit_edge_permutation(bins, permuted, n_e, positions,
                                    n_distance_bins=4,
                                    weight_quantile_relative_tolerance=0.02)
    assert strict["max_weight_quantile_relative_deviation"] > 0.02
    assert strict["weight_distribution_preserved"] is False
    assert strict["all_structural_clauses_pass"] is False
    assert strict["weight_quantiles_by_pathway_and_distance_bin"]
    loose = audit_edge_permutation(bins, permuted, n_e, positions,
                                   n_distance_bins=4,
                                   weight_quantile_relative_tolerance=10.0)
    assert loose["weight_distribution_preserved"] is True
    assert loose["all_structural_clauses_pass"] is True
