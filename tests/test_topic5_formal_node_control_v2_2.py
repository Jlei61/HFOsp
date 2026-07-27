import numpy as np

from src.topic5_formal_node_control_v2_2 import (
    evaluate_node_control,
    fit_loso_stop,
    node_control_event_nll,
    stop_histogram,
)
from src.topic5_sequence_sensitivity_v2_2 import SharedStop, _event_nll


def _events():
    return np.asarray(
        [
            [0, 1, 2, -1],
            [0, 1, 1, 2],
            [2, 0, 1, -1],
        ],
        dtype=np.int64,
    )


def test_histogram_encodes_patient_balanced_event_first_weights():
    groups = _events()
    indices = np.arange(len(groups))
    histogram = stop_histogram(groups, indices)
    assert histogram.raw_decisions.sum() == 9
    assert histogram.raw_terminal.sum() == 3
    expected_weight = 0.0
    for event in groups:
        n_steps = int(np.max(event[event >= 0])) + 1
        for step in range(n_steps):
            seen = (event >= 0) & (event <= step)
            expected_weight += 1 / len(groups) / n_steps / max(1, (~seen).sum())
    assert np.isclose(histogram.decision_weight.sum(), expected_weight)
    assert np.all(histogram.terminal_weight <= histogram.decision_weight)


def test_node_event_likelihood_matches_existing_control():
    event = _events()[0]
    hazard = np.asarray([0.2, 0.3, 0.4, 0.1])
    compressed = fit_loso_stop([stop_histogram(_events(), np.arange(3))])
    expanded = SharedStop(
        c0=compressed.c0,
        c_n=compressed.c_n,
        n_decisions=compressed.n_decisions,
        n_terminal=compressed.n_terminal,
        optimizer_success=True,
    )
    observed = node_control_event_nll(event, hazard, compressed)
    expected = _event_nll(
        event=event,
        node_hazard=hazard,
        stop=expanded,
        transition=None,
    )
    assert np.isclose(observed, expected, atol=1e-12)


def test_evaluation_is_event_first_and_finite():
    groups = _events()
    stop = fit_loso_stop([stop_histogram(groups, np.arange(2))])
    values = evaluate_node_control(
        groups=groups,
        heldout_indices=np.asarray([2]),
        node_hazard=np.asarray([0.2, 0.3, 0.4, 0.1]),
        stop=stop,
    )
    assert values.shape == (1,)
    assert np.isfinite(values).all()
