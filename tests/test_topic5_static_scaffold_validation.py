import numpy as np
import pytest

from src.topic5_static_scaffold_validation import (
    beta_binomial_participation,
    categorical_event_nll,
    contact_graph,
    contact_rank_categories,
    coherent_index_null,
    dirichlet_contact_rank_distribution,
    event_brier,
    geometry_smooth_surrogates,
    laplacian_smooth,
    partial_rank_score,
    participation_rate,
    score_signed_field,
    shaft_groups,
)


def test_within_shaft_null_never_crosses_shaft():
    names = np.asarray(["A1", "A2", "A3", "B1", "B2", "B3"])
    permutations, audit = coherent_index_null(
        names, n_draws=50, seed=7, mode="within_shaft_circular"
    )
    groups = shaft_groups(names)
    assert audit["eligible"]
    for indices in groups.values():
        assert np.all(np.isin(permutations[:, indices], indices))
    assert np.all(np.any(permutations != np.arange(len(names)), axis=1))


def test_equal_size_shaft_profile_moves_whole_profiles():
    names = np.asarray(["A1", "A2", "B1", "B2", "C1"])
    permutations, audit = coherent_index_null(
        names, n_draws=20, seed=9, mode="equal_size_shaft_profile"
    )
    assert audit["eligible"]
    assert set(permutations[0, :4]) == {0, 1, 2, 3}
    assert np.all(permutations[:, 4] == 4)


def test_geometry_surrogate_rank_matches_field():
    field = np.asarray([0.1, 0.4, 0.2, 0.9, 0.7])
    coords = np.column_stack([np.arange(5), np.zeros(5), np.zeros(5)])
    z = np.random.default_rng(3).normal(size=(25, 5))
    surrogate, scale = geometry_smooth_surrogates(
        field, coords, standard_normal=z
    )
    assert scale > 0
    assert surrogate.shape == (25, 5)
    for row in surrogate:
        assert np.allclose(np.sort(row), np.sort(field))


def test_signed_score_keeps_direction():
    field = np.arange(6, dtype=float)
    target = np.row_stack([field, field + 1])
    null = np.row_stack([field[::-1], np.roll(field, 2)])
    result = score_signed_field(field, target, null)
    assert np.isclose(result["observed_signed"], 1.0)
    assert result["null_signed"][0] < 0


def test_regularized_participation_estimators_are_bounded():
    groups = np.asarray(
        [
            [0, 1, -1, -1],
            [0, -1, 1, -1],
            [-1, 0, 1, 2],
            [0, 1, 2, -1],
        ],
        dtype=np.int16,
    )
    raw = participation_rate(groups)
    shrunk = beta_binomial_participation(groups, concentration=8.0)
    assert np.all((shrunk > 0) & (shrunk < 1))
    assert np.var(shrunk) < np.var(raw)
    assert np.isfinite(event_brier(shrunk, groups))


def test_laplacian_smoothing_reduces_graph_roughness():
    names = np.asarray(["A1", "A2", "A3", "B1"])
    graph = contact_graph(names, mode="shaft")
    field = np.asarray([0.0, 1.0, 0.0, 0.75])
    smoothed = laplacian_smooth(field, graph, penalty=2.0)
    laplacian = np.diag(graph.sum(1)) - graph
    assert float(smoothed @ laplacian @ smoothed) < float(
        field @ laplacian @ field
    )
    assert smoothed[3] == pytest.approx(field[3])


def test_dirichlet_contact_rank_distribution_scores_validation_events():
    groups = np.asarray(
        [
            [0, 1, -1],
            [0, -1, 1],
            [-1, 0, 1],
            [0, 1, 2],
        ],
        dtype=np.int16,
    )
    categories = contact_rank_categories(groups, n_rank_bins=10)
    distribution = dirichlet_contact_rank_distribution(
        categories, concentration=2.0, n_rank_bins=10
    )
    assert distribution.shape == (3, 11)
    assert np.allclose(distribution.sum(axis=1), 1.0)
    assert np.isfinite(categorical_event_nll(distribution, categories))


def test_partial_rank_score_removes_shared_linear_rank_confound():
    confound = np.arange(10, dtype=float)
    residual = np.asarray([0, 1, -1, 2, -2, 3, -3, 4, -4, 0.5])
    field = confound + 4.0 * residual
    target = np.row_stack(
        [confound + 4.0 * residual, confound + 2.0 * residual]
    )
    result = partial_rank_score(
        field, target, confound, n_null_draws=100, null_seed=5
    )
    assert result["eligible"]
    assert result["signed_rho"] > 0.8
    assert result["residual_df"] == 8
    assert result["signed_margin"] > 0.5


def test_partial_rank_score_rejects_constant_covariate():
    field = np.arange(8, dtype=float)
    target = field[None, :]
    result = partial_rank_score(field, target, np.ones(8))
    assert not result["eligible"]
    assert result["reason"] == "constant_covariate"
