import numpy as np
import pytest

from scripts.paper_figures.plot_topic4_rev12_directed_lineage_gif import (
    lineage_bin_coordinates,
    lineage_display_frames,
)


def test_directed_lineage_display_includes_fixed_context_without_overrun():
    frames = lineage_display_frames(
        event_on_ms=20.0, event_off_ms=40.0, frame_ms=2.0,
        n_frames=25, context_ms=20.0,
    )
    assert frames.tolist() == list(range(25))


def test_directed_lineage_display_rejects_empty_or_reversed_window():
    with pytest.raises(ValueError):
        lineage_display_frames(
            event_on_ms=40.0, event_off_ms=20.0, frame_ms=2.0,
            n_frames=25,
        )


def test_lineage_outline_excludes_concurrent_other_root_and_collision():
    labels = np.asarray([
        [1, 1, 0],
        [2, -1, 1],
    ])
    coordinates = lineage_bin_coordinates(labels, 1)
    assert coordinates.tolist() == [[0.5, 0.5], [1.5, 0.5], [2.5, 1.5]]


def test_lineage_outline_includes_all_roots_in_complete_event():
    labels = np.asarray([[1, 0], [0, 2]])
    coordinates = lineage_bin_coordinates(labels, [1, 2])
    assert coordinates.tolist() == [[0.5, 0.5], [1.5, 1.5]]
