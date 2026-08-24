import numpy as np

from src.topic4_node_dualmode import (
    bin_neuron_spikes,
    binned_contact_envelope,
    lineage_restricted_contact_readout,
    lineage_restricted_neuron_contact_readout,
    neuron_contact_sampling_weights,
    sheet_contact_sampling_weights,
)
from src.sef_hfo_observation import VirtualMontage, sample_envelopes


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


def test_exact_neuron_sampler_matches_the_frozen_virtual_contact_kernel():
    rng = np.random.default_rng(7)
    positions = rng.uniform(0.0, 3.0, size=(12, 2))
    contacts = np.asarray([[0.5, 0.5], [2.5, 2.5]])
    frames = rng.normal(size=(8, len(positions)))
    weights = neuron_contact_sampling_weights(
        positions, contacts, kernel_width_mm=0.25,
    )
    observed = (frames @ weights.T).T
    expected = sample_envelopes(
        frames, positions, VirtualMontage(contacts, ["A", "B"], "test"), 0.25,
    )
    assert np.allclose(observed, expected, atol=0.0, rtol=1e-14)


def test_exact_neuron_lineage_readout_ignores_concurrent_other_root():
    positions = np.asarray([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
    contacts = positions.copy()
    weights = neuron_contact_sampling_weights(
        positions, contacts, kernel_width_mm=0.1,
    )
    spikes = np.zeros((10, 3), bool)
    spikes[2:8:2, 0] = True
    spikes[2:8:2, 2] = True
    binned = bin_neuron_spikes(spikes, dt_ms=1.0, frame_ms=2.0)
    labels = np.zeros((5, 3, 3), int)
    labels[1:4, 0, 0] = 1
    labels[1:4, 0, 2] = 2
    result = lineage_restricted_neuron_contact_readout(
        binned, np.asarray([0, 1, 2]), labels,
        [{"cascade_id": 1, "t_on": 0.0, "t_off": 10.0}], weights,
        frame_ms=2.0, smooth_ms=0.01,
        participation_margin_fraction=0.1, timing_fraction=0.5,
    )
    assert np.isfinite(result["onsets"][0, 0])
    assert np.isnan(result["onsets"][0, 2])


def test_exact_neuron_readout_unions_only_roots_in_one_interaction_episode():
    positions = np.asarray([[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]])
    weights = neuron_contact_sampling_weights(
        positions, positions, kernel_width_mm=0.1,
    )
    spikes = np.zeros((10, 3), bool)
    spikes[2:8:2] = True
    binned = bin_neuron_spikes(spikes, dt_ms=1.0, frame_ms=2.0)
    labels = np.zeros((5, 3, 3), int)
    labels[1:4, 0, 0] = 1
    labels[1:4, 0, 1] = 2
    labels[1:4, 0, 2] = 3
    result = lineage_restricted_neuron_contact_readout(
        binned, np.asarray([0, 1, 2]), labels,
        [{"cascade_id": 1, "lineage_ids": [1, 2], "t_on": 0.0, "t_off": 10.0}],
        weights, frame_ms=2.0, smooth_ms=0.01,
        participation_margin_fraction=0.1, timing_fraction=0.5,
    )
    assert np.isfinite(result["onsets"][0, 0:2]).all()
    assert np.isnan(result["onsets"][0, 2])
