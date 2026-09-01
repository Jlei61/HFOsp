"""Unit guards for the two rev8 Fig. 4 renderers."""
import numpy as np

from scripts.paper_figures.plot_fig4_data_driven_core_field_rev8 import (
    _closest_mode_pair,
    _field_landscape_grid,
    _nice_amplitude_scale,
    _verdict_label,
)
from scripts.build_topic4_core_field_stage3_rev8_figure_diagnostics import (
    conditional_hierarchical_similarity_bootstrap,
    hierarchical_bootstrap_indices,
    patient_block_mode_bands,
)
from scripts.capture_topic4_core_field_stage3_rev8_all_onsets import (
    event_equal_density,
)
from scripts.paper_figures.plot_fig4_data_driven_core_field_rev8_kmeans import (
    _event_order,
    _normalized_rank_matrix,
    _profile_stats,
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


def test_field_landscape_grid_preserves_a_planar_field():
    x, y = np.meshgrid(np.linspace(0.0, 2.0, 4), np.linspace(0.0, 2.0, 4))
    pos = np.column_stack((x.ravel(), y.ravel()))
    h = pos[:, 0] + 2.0 * pos[:, 1]
    xx, yy, zz = _field_landscape_grid(pos, h, L=2.0, resolution=9)
    assert xx.shape == yy.shape == zz.shape == (9, 9)
    assert np.allclose(zz, xx + 2.0 * yy)


def test_profile_stats_keep_all_missing_contacts_missing_without_warning():
    ranks = np.array([[np.nan, np.nan, 0.2], [0.0, 1.0, 0.5]])
    mean, std = _profile_stats(ranks, np.array([True, True, False]))
    assert np.isnan(mean[0]) and np.isnan(std[0])
    assert mean[1] == 0.5 and std[1] == 0.5


def test_internal_verdict_is_rendered_as_reader_facing_text():
    assert _verdict_label("RIGID_TEMPLATE_MATCH_NOT_BEATEN") == \
        "fails rigid-mode benchmark"


def test_common_readout_scale_uses_a_readable_one_two_five_step():
    assert _nice_amplitude_scale(43.5) == 50.0
    assert _nice_amplitude_scale(1.2) == 2.0
    assert _nice_amplitude_scale(0.0) == 1.0


def test_event_equal_onset_density_does_not_weight_large_events_more():
    small = np.array([[0.25, 0.25]])
    large = np.repeat([[1.75, 1.75]], 100, axis=0)
    density, _, count = event_equal_density([small, large], 2.0, n_bins=2)
    assert count == 2
    assert np.isclose(density.sum(), 1.0)
    assert np.isclose(density[0, 0], 0.5)
    assert np.isclose(density[1, 1], 0.5)


def test_hierarchical_resampling_keeps_whole_group_occurrences():
    groups = np.array([0, 0, 1, 1, 1])
    index = hierarchical_bootstrap_indices(groups, np.random.default_rng(4))
    assert len(index) in (4, 5, 6)
    assert np.all((0 <= index) & (index < len(groups)))


def test_conditional_hierarchical_matrix_is_deterministic_and_finite():
    base = np.linspace(0.0, 1.0, 8)
    model = np.vstack([base, base + 0.02, -base, -base - 0.02] * 3)
    labels = np.tile([0, 0, 1, 1], 3)
    groups = np.repeat(np.arange(3), 4)
    first = conditional_hierarchical_similarity_bootstrap(
        model, labels, groups, model, labels, groups, n_bootstrap=20, seed=7)
    second = conditional_hierarchical_similarity_bootstrap(
        model, labels, groups, model, labels, groups, n_bootstrap=20, seed=7)
    assert first.shape == (20, 2, 2)
    assert np.isfinite(first).all()
    assert np.array_equal(first, second)


def test_patient_block_bands_count_only_blocks_containing_each_mode():
    curves = np.array([[0, 1], [0.1, 0.9], [1, 0], [0.9, 0.1]], float)
    labels = np.array([0, 0, 1, 1])
    blocks = np.array([10, 11, 10, 10])
    low, high, counts = patient_block_mode_bands(curves, labels, blocks)
    assert low.shape == high.shape == (2, 2)
    assert counts.tolist() == [2, 1]
