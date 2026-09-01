from __future__ import annotations

import numpy as np
import pytest

from src.sef_hfo_observation import VirtualMontage, sample_envelopes
from src.topic4_cohort_fast_readout import batched_sample_envelopes


def _case(n_frame=64, n_pixel=900, n_contact=37, seed=3):
    rng = np.random.default_rng(seed)
    frames = rng.random((n_frame, n_pixel))
    grid = rng.random((n_pixel, 2)) * 20.0
    montage = VirtualMontage(
        rng.random((n_contact, 2)) * 20.0,
        [f"c{index}" for index in range(n_contact)], "test",
    )
    return frames, grid, montage


def test_batched_readout_matches_the_frozen_envelope_sampler():
    frames, grid, montage = _case()
    reference = sample_envelopes(frames, grid, montage, 0.25)
    fast = batched_sample_envelopes(frames, grid, montage, 0.25)
    assert fast.shape == reference.shape
    np.testing.assert_allclose(fast, reference, rtol=0.0, atol=1e-12)


def test_batched_readout_preserves_contact_onset_ordering():
    """The science downstream is contact order, so ordering must be identical."""
    frames, grid, montage = _case(n_frame=128, n_contact=61, seed=11)
    reference = sample_envelopes(frames, grid, montage, 0.25)
    fast = batched_sample_envelopes(frames, grid, montage, 0.25)
    for frame in range(reference.shape[1]):
        np.testing.assert_array_equal(
            np.argsort(reference[:, frame], kind="stable"),
            np.argsort(fast[:, frame], kind="stable"),
        )
        np.testing.assert_array_equal(
            np.argmax(reference[:, frame]), np.argmax(fast[:, frame]),
        )


def test_chunk_size_moves_the_readout_only_at_float64_round_off():
    """BLAS blocks differently per shape, so the chunk size must be frozen."""
    frames, grid, montage = _case(n_contact=53)
    whole = batched_sample_envelopes(frames, grid, montage, 0.25, contact_chunk=1024)
    chunked = batched_sample_envelopes(frames, grid, montage, 0.25, contact_chunk=7)
    np.testing.assert_allclose(whole, chunked, rtol=0.0, atol=1e-12)
    for frame in range(whole.shape[1]):
        np.testing.assert_array_equal(
            np.argsort(whole[:, frame], kind="stable"),
            np.argsort(chunked[:, frame], kind="stable"),
        )


def test_batched_readout_repeats_bit_for_bit_at_a_fixed_chunk_size():
    frames, grid, montage = _case(n_contact=53)
    first = batched_sample_envelopes(frames, grid, montage, 0.25, contact_chunk=128)
    repeat = batched_sample_envelopes(frames, grid, montage, 0.25, contact_chunk=128)
    np.testing.assert_array_equal(first, repeat)


def test_batched_readout_rejects_misaligned_inputs():
    frames, grid, montage = _case()
    with pytest.raises(ValueError, match="pixel axis"):
        batched_sample_envelopes(frames[:, :-1], grid, montage, 0.25)
    with pytest.raises(ValueError, match="chunk must be positive"):
        batched_sample_envelopes(frames, grid, montage, 0.25, contact_chunk=0)
