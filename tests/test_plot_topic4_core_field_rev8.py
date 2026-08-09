"""Unit guards for the two rev8 Fig. 4 renderers."""
import numpy as np

from scripts.paper_figures.plot_fig4_data_driven_core_field_rev8 import (
    _closest_mode_pair,
)
from scripts.paper_figures.plot_fig4_data_driven_core_field_rev8_kmeans import (
    _event_order,
    _normalized_rank_matrix,
)


def test_rank_normalization_preserves_missing_contacts_and_event_scale():
    ranks = np.array([[0.0, np.nan], [2.0, 4.0], [1.0, 2.0]])
    normalized = _normalized_rank_matrix(ranks)
    assert normalized[:, 0].tolist() == [0.0, 1.0, 0.5]
    assert np.isnan(normalized[0, 1])
    assert normalized[1:, 1].tolist() == [1.0, 0.0]


def test_heatmap_order_keeps_modes_contiguous_without_changing_labels():
    curves = np.array([
        np.linspace(0, 1, 5), np.linspace(1, 0, 5),
        np.linspace(0.1, 1.1, 5), np.linspace(1.1, 0.1, 5),
    ])
    labels = np.array([1, 0, 1, 0])
    order = _event_order(curves, labels)
    assert labels[order].tolist() == [0, 0, 1, 1]


def test_direct_readout_selects_the_closest_cross_mode_pair():
    events = [
        {"mode": 0, "t_on": 100.0, "t_off": 120.0},
        {"mode": 1, "t_on": 900.0, "t_off": 920.0},
        {"mode": 0, "t_on": 850.0, "t_off": 870.0},
    ]
    pair = _closest_mode_pair(events)
    assert {event["t_on"] for event in pair} == {850.0, 900.0}


def test_direct_readout_refuses_to_invent_a_missing_second_mode():
    assert _closest_mode_pair([
        {"mode": 0, "t_on": 100.0, "t_off": 120.0},
        {"mode": 0, "t_on": 200.0, "t_off": 220.0},
    ]) is None
