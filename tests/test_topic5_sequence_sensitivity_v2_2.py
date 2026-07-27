from __future__ import annotations

import numpy as np

from src.topic5_sequence_sensitivity_v2_2 import (
    contact_descriptives,
    decision_rows,
    estimate_hazard,
    estimate_markov,
    evaluate_models,
    fit_shared_stop,
)


def _events() -> np.ndarray:
    return np.array(
        [
            [0, 1, 2, -1],
            [0, 1, 2, -1],
            [2, 1, 0, -1],
            [2, 1, 0, -1],
            [0, 1, 2, -1],
            [2, 1, 0, -1],
        ],
        dtype=np.int64,
    )


def test_shared_stop_uses_only_supplied_decisions_and_is_constrained() -> None:
    groups = _events()
    first = decision_rows(groups, np.array([0, 1, 2]))
    second = decision_rows(groups, np.array([3, 4]))
    stop = fit_shared_stop([first, second])
    assert stop.n_decisions == 15
    assert stop.n_terminal == 5
    assert stop.c_n >= 0
    assert np.isfinite(stop.c0)


def test_markov_recovers_opposite_direction_conditionals() -> None:
    groups = _events()
    train = np.arange(4)
    hazard = estimate_hazard(groups, train)
    transition = estimate_markov(groups, train, hazard, concentration=0.01)
    assert transition[0, 1] > transition[0, 2]
    assert transition[2, 1] > transition[2, 0]
    assert np.all((transition > 0) & (transition < 1))


def test_evaluation_is_event_first_and_finite() -> None:
    groups = _events()
    train = np.arange(4)
    heldout = np.arange(4, 6)
    stop = fit_shared_stop([decision_rows(groups, train)])
    hazard = estimate_hazard(groups, train)
    transition = estimate_markov(groups, train, hazard)
    result = evaluate_models(
        groups=groups,
        heldout_indices=heldout,
        node_hazard=hazard,
        transition=transition,
        stop=stop,
    )
    assert result["node_event_nll"].shape == (2,)
    assert result["markov_event_nll"].shape == (2,)
    assert np.isfinite(result["markov_benefit"])


def test_contact_descriptives_do_not_require_geometry() -> None:
    groups = _events()
    train = np.arange(4)
    heldout = np.arange(4, 6)
    hazard = estimate_hazard(groups, train)
    transition = estimate_markov(groups, train, hazard)
    rows = contact_descriptives(
        groups=groups,
        train_indices=train,
        heldout_indices=heldout,
        node_hazard=hazard,
        transition=transition,
    )
    assert len(rows) == groups.shape[1]
    assert rows[3]["heldout_participation_probability"] == 0.0
