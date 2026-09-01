import numpy as np

from scripts.freeze_topic4_rev10_r2_spatial_edge_selection import (
    select_diverse_pareto,
)


def _candidate(candidate_id, direction):
    return {
        "candidate_id": candidate_id,
        "coefficients": list(direction),
        "latent_whitened_direction": list(direction),
    }


def _row(candidate_id, score, *, pareto=True, runaway=0):
    return {
        "candidate_id": candidate_id,
        "selection_score_equal_network": score,
        "pareto_nondominated": pareto,
        "n_runaway_networks": runaway,
    }


def test_selection_is_fit_pareto_safe_and_keeps_directional_diversity():
    candidates = [
        _candidate("edge_noop", [0.0, 0.0]),
        _candidate("best", [1.0, 0.0]),
        _candidate("near_duplicate", [0.99, 0.01]),
        _candidate("opposite", [-1.0, 0.0]),
        _candidate("orthogonal", [0.0, 1.0]),
        _candidate("dominated", [0.5, 0.5]),
        _candidate("runaway", [0.2, 0.8]),
    ]
    rows = [
        _row("edge_noop", 1.2),
        _row("best", 0.8),
        _row("near_duplicate", 0.81),
        _row("opposite", 0.9),
        _row("orthogonal", 0.85),
        _row("dominated", 0.7, pareto=False),
        _row("runaway", 0.6, runaway=1),
    ]
    assert select_diverse_pareto(rows, candidates, 3) == [
        "best", "opposite", "orthogonal",
    ]


def test_selection_returns_empty_without_safe_nonzero_pareto_candidate():
    candidates = [
        _candidate("edge_noop", [0.0]),
        _candidate("bad", [1.0]),
    ]
    rows = [_row("edge_noop", 1.0), _row("bad", 0.5, runaway=1)]
    assert select_diverse_pareto(rows, candidates, 6) == []


def test_selection_uses_signed_not_axis_only_distance():
    candidates = [
        _candidate("edge_noop", [0.0, 0.0]),
        _candidate("positive", [1.0, 0.0]),
        _candidate("negative", [-1.0, 0.0]),
        _candidate("close", [1.0, 0.01]),
    ]
    rows = [
        _row("edge_noop", 2.0),
        _row("positive", 0.1),
        _row("negative", 0.3),
        _row("close", 0.2),
    ]
    assert select_diverse_pareto(rows, candidates, 2) == [
        "positive", "negative",
    ]


def test_zero_direction_is_rejected_for_nonzero_candidate():
    candidates = [_candidate("bad", np.zeros(2))]
    rows = [_row("bad", 1.0)]
    try:
        select_diverse_pareto(rows, candidates, 1)
    except ValueError as error:
        assert "invalid direction" in str(error)
    else:
        raise AssertionError("zero nonzero-candidate direction was accepted")
