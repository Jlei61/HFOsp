import numpy as np

from src.topic4_node_dualmode import (
    binned_contact_envelope,
    lineage_restricted_contact_readout,
    sheet_contact_sampling_weights,
)


def test_sheet_contact_sampler_uses_neuron_population_denominator():
    population = np.asarray([[1.0, 3.0]])
    weights = sheet_contact_sampling_weights(
        np.asarray([[1.0, 0.5]]), population,
        bin_mm=1.0, kernel_width_mm=10.0,
    )
    movie = np.asarray([[[1.0, 3.0]]])
    envelope = binned_contact_envelope(
        movie, weights, frame_ms=1.0, smooth_ms=0.01,
    )
    assert np.isclose(envelope[0, 0], 1.0)


def test_lineage_readout_ignores_a_concurrent_other_root():
    counts = np.zeros((5, 1, 3), float)
    labels = np.zeros_like(counts, int)
    counts[:, 0, 0] = 4.0
    labels[:, 0, 0] = 1
    counts[:, 0, 2] = 20.0
    labels[:, 0, 2] = 2
    population = np.ones((1, 3), float) * 20.0
    weights = sheet_contact_sampling_weights(
        np.asarray([[0.5, 0.5], [2.5, 0.5]]), population,
        bin_mm=1.0, kernel_width_mm=0.1,
    )
    result = lineage_restricted_contact_readout(
        counts, labels, [{"cascade_id": 1, "t_on": 0.0, "t_off": 5.0}],
        weights, frame_ms=1.0, smooth_ms=0.01,
        participation_margin_fraction=0.1, timing_fraction=0.5,
    )
    assert np.isfinite(result["onsets"][0, 0])
    assert np.isnan(result["onsets"][0, 1])


def test_lineage_readout_recovers_forward_contact_order():
    counts = np.zeros((6, 1, 3), float)
    labels = np.zeros_like(counts, int)
    for frame, x_index in enumerate((0, 0, 1, 1, 2, 2)):
        counts[frame, 0, x_index] = 5.0
        labels[frame, 0, x_index] = 7
    population = np.ones((1, 3), float) * 20.0
    weights = sheet_contact_sampling_weights(
        np.asarray([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]), population,
        bin_mm=1.0, kernel_width_mm=0.1,
    )
    result = lineage_restricted_contact_readout(
        counts, labels, [{"cascade_id": 7, "t_on": 0.0, "t_off": 6.0}],
        weights, frame_ms=1.0, smooth_ms=0.01,
        participation_margin_fraction=0.1, timing_fraction=0.5,
    )
    assert result["ranks"][0].tolist() == [0.0, 1.0, 2.0]
