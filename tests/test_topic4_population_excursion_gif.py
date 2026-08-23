import numpy as np
import pytest

from scripts.paper_figures.plot_topic4_rev12_population_excursion_gif import (
    detector_fragment_spans,
    display_frame_indices,
)


def test_display_frames_cover_complete_interval_and_reset_endpoint():
    frames = display_frame_indices(
        start_ms=18.0, stop_ms=36.0, frame_ms=2.0, n_frames=100,
    )
    assert frames[0] == 9
    assert frames[-1] == 18


def test_fragment_spans_are_reconstructed_from_population_detector():
    active = np.zeros(60)
    active[10:23] = 0.04
    active[45:60] = 0.05
    spans = detector_fragment_spans(
        active, 1.0, event_threshold=0.02,
        fragment_indices=[0, 1], trigger_ms=10.0,
    )
    assert spans == [(0.0, 12.0), (35.0, 49.0)]


def test_display_frames_reject_empty_or_reversed_interval():
    with pytest.raises(ValueError):
        display_frame_indices(
            start_ms=20.0, stop_ms=20.0, frame_ms=2.0, n_frames=100,
        )
