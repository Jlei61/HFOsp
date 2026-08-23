import numpy as np
import pytest

from scripts.paper_figures.plot_topic4_rev12_node_confirmation import (
    _settled_episode_pair,
)


def test_figure_pair_uses_distinct_nonoverlapping_settled_episodes():
    events = [
        {"detector_fragment_indices": [3, 4]},
        {"detector_fragment_indices": [7]},
    ]
    pair, fragments = _settled_episode_pair(
        events, np.asarray([0, 1]), {0: 0, 1: 1}, {0: 0, 1: 1},
    )
    assert pair == (0, 1)
    assert fragments == [{3, 4}, {7}]


def test_figure_pair_rejects_shared_detector_fragment():
    events = [
        {"detector_fragment_indices": [3, 4]},
        {"detector_fragment_indices": [4, 5]},
    ]
    with pytest.raises(RuntimeError, match="share detector fragments"):
        _settled_episode_pair(
            events, np.asarray([0, 1]), {0: 0, 1: 1}, {0: 0, 1: 1},
        )


def test_figure_pair_rejects_mode_label_mismatch():
    events = [
        {"detector_fragment_indices": [3]},
        {"detector_fragment_indices": [7]},
    ]
    with pytest.raises(RuntimeError, match="does not match"):
        _settled_episode_pair(
            events, np.asarray([1, 0]), {0: 0, 1: 1}, {0: 0, 1: 1},
        )
