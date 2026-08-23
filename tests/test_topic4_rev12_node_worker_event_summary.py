import numpy as np
import pytest

from scripts.run_topic4_rev12_node_worker import _event_peak_active_fraction


def test_event_peak_uses_frozen_window_without_detector_specific_fields():
    active = np.asarray([0.1, 0.4, 0.8, 0.2, 0.9])
    event = {"t_on": 2.0, "t_off": 8.0}
    assert _event_peak_active_fraction(event, active, 2.0) == 0.8


def test_event_peak_clips_terminal_window_to_recording():
    active = np.asarray([0.1, 0.4, 0.8])
    event = {"t_on": 4.0, "t_off": 10.0}
    assert _event_peak_active_fraction(event, active, 2.0) == 0.8


def test_event_peak_rejects_empty_window():
    with pytest.raises(ValueError, match="contains no population-activity sample"):
        _event_peak_active_fraction(
            {"t_on": 10.0, "t_off": 12.0}, np.asarray([0.1, 0.2]), 2.0,
        )
