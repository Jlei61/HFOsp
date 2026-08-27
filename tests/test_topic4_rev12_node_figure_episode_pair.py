import numpy as np
import pytest

from scripts.paper_figures.plot_topic4_rev12_node_confirmation import (
    _selected_seed,
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


def test_current_aggregate_selects_dual_mode_seed_nearest_median(monkeypatch):
    monkeypatch.setattr(
        "scripts.paper_figures.plot_topic4_rev12_node_confirmation._source_bundle",
        lambda _path, worker: (
            np.zeros((len(worker["labels"]), 2, 2)),
            np.asarray(worker["labels"]),
            np.arange(len(worker["labels"])),
        ),
    )
    row = {
        "per_network": [
            {"seed": 1, "n_events": 10, "soft_objective": {"objective": 1.0}},
            {"seed": 2, "n_events": 12, "soft_objective": {"objective": 2.0}},
            {"seed": 3, "n_events": 14, "soft_objective": {"objective": 10.0}},
        ],
    }
    workers = {
        seed: (None, {"labels": [0, 1, 0, 1]}) for seed in (1, 2, 3)
    }
    assert _selected_seed(row, workers) == 2
