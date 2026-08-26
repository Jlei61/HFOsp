from __future__ import annotations

import numpy as np

from src.topic5_continuous_marked_state_r1.data import R1EventStream
from src.topic5_continuous_marked_state_r1.history import DeterministicHistory


def _stream() -> R1EventStream:
    return R1EventStream(
        subject="synthetic", dataset="yuquan",
        event_time=np.asarray([10.0, 20.0, 100.0]),
        split=np.asarray([0, 0, 1], dtype=np.int8),
        session=np.asarray([0, 0, 1]),
        participation=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=bool),
        group_ids=np.asarray([[0, -1], [-1, 0], [0, 1]]),
        group_count=np.asarray([1, 1, 2]),
        load=np.asarray([0.5, 0.5, 1.0], dtype=np.float32),
        contact_names=np.asarray(["A1", "A2"]),
        contact_features=np.zeros((2, 1), dtype=np.float32),
        adjacency=np.zeros((1, 2, 2), dtype=np.float32),
        source_hashes={},
    )


def test_history_is_strictly_pre_event_and_resets_by_session() -> None:
    stream = _stream()
    history = DeterministicHistory(stream, {0: 0.0, 1: 90.0})
    value = history.evaluate(
        np.asarray([10.0, 20.0, 25.0, 100.0, 101.0]),
        np.asarray([0, 0, 0, 1, 1]),
    )
    # At an event's exact time, the current event is never visible.
    assert value[0, 0] == 0.0
    assert value[1, 0] == 1.0
    np.testing.assert_array_equal(value[1, 11:13], [1.0, 0.0])
    # New session resets event history; one second after the session's event it is visible.
    assert value[3, 0] == 0.0
    assert value[4, 0] == 1.0
    np.testing.assert_array_equal(value[4, 11:13], [1.0, 1.0])
