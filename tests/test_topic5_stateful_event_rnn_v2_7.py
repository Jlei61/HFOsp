from src.topic5_stateful_event_rnn_v2_7 import (
    checkpoint_selection_from_trace,
    trained_patience_step,
)


def test_epoch_minus_one_cannot_age_trained_patience():
    best = None
    stale = 999
    best, stale, improved = trained_patience_step(best, 5.0, stale)
    assert improved
    assert best == 5.0
    assert stale == 0


def test_trained_improvement_resets_patience_before_static_is_beaten():
    # All scores could remain worse than an external static score of 1.0; that
    # comparator is intentionally absent from trained patience bookkeeping.
    best, stale, _ = trained_patience_step(None, 5.0, 0)
    best, stale, improved = trained_patience_step(best, 4.5, stale)
    assert improved
    assert best == 4.5
    assert stale == 0


def test_only_non_improving_trained_epochs_increment_staleness():
    best, stale, _ = trained_patience_step(None, 5.0, 0)
    best, stale, improved = trained_patience_step(best, 5.1, stale)
    assert not improved
    assert best == 5.0
    assert stale == 1
    best, stale, improved = trained_patience_step(best, 4.9, stale)
    assert improved
    assert best == 4.9
    assert stale == 0


def test_complete_trace_does_not_stop_at_minimum_while_trained_epochs_improve():
    trace = [5.0 - 0.1 * epoch for epoch in range(8)] + [4.3] * 8
    result = checkpoint_selection_from_trace(
        trace,
        static_score=1.0,
        minimum_epochs=8,
        patience=8,
    )
    # Epoch-minus-one remains the nested fallback, but trained patience starts
    # from epoch zero and therefore cannot stop at epoch seven.
    assert result["best_nested_epoch"] == -1
    assert result["best_trained_epoch"] == 7
    assert result["stopped_epoch"] == 15


def test_nested_may_select_epoch_minus_one_but_trained_never_does():
    result = checkpoint_selection_from_trace(
        [5.0, 4.0, 3.0, 3.0, 3.0],
        static_score=1.0,
        minimum_epochs=3,
        patience=2,
    )
    assert result["best_nested_epoch"] == -1
    assert result["best_trained_epoch"] == 2
