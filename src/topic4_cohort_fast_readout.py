"""Batched contact readout for the formal cohort's 890-contact montage.

`sample_envelopes` loops over contacts and, when no hard radius is given,
indexes the frame matrix with an all-true boolean mask.  That copies the whole
(n_frame, n_pixel) array once per contact, which costs about 34 minutes for the
formal montage -- more than the simulation it reads.  This module computes the
identical distance-weighted average as one chunked matrix product instead.  It
is a separate module on purpose: the frozen observation module is part of the
provenance chain of runs that are still in flight.
"""
from __future__ import annotations

import numpy as np


def batched_sample_envelopes(source_frames: np.ndarray, grid_xy: np.ndarray,
                             montage, kernel_width: float, *,
                             contact_chunk: int = 128) -> np.ndarray:
    """Return (n_contact, n_frame) envelopes, matching `sample_envelopes`."""
    frames = np.asarray(source_frames, float)
    grid = np.asarray(grid_xy, float)
    contacts = np.asarray(montage.contacts, float)
    if frames.ndim != 2 or grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError("frames must be (n_frame, n_pixel) and grid (n_pixel, 2)")
    if frames.shape[1] != len(grid):
        raise ValueError("frames and grid do not share the pixel axis")
    if contacts.ndim != 2 or contacts.shape[1] != 2:
        raise ValueError("montage contacts must be (n_contact, 2)")
    if contact_chunk < 1:
        raise ValueError("contact chunk must be positive")

    out = np.empty((len(contacts), frames.shape[0]), float)
    transposed = frames.T
    scale = 2.0 * float(kernel_width) ** 2
    for start in range(0, len(contacts), int(contact_chunk)):
        block = contacts[start:start + int(contact_chunk)]
        squared = (
            (grid[None, :, 0] - block[:, None, 0]) ** 2
            + (grid[None, :, 1] - block[:, None, 1]) ** 2
        )
        weights = np.exp(-squared / scale)
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        out[start:start + len(block)] = weights @ transposed
    return out


def batched_snn_event_envelope(spikes: np.ndarray, positions_e: np.ndarray,
                               montage, dt: float, *, bin_ms: float = 2.0,
                               smooth_ms: float = 5.0,
                               kernel_width: float = 0.25,
                               contact_chunk: int = 128):
    """Drop-in for `snn_event_envelope` with the batched contact readout."""
    from src.sef_hfo_snn_adapter import _bin_and_smooth

    rate, frame_dt = _bin_and_smooth(spikes, dt, bin_ms, smooth_ms)
    envelope = batched_sample_envelopes(
        rate, np.asarray(positions_e, float), montage, kernel_width,
        contact_chunk=contact_chunk,
    )
    return envelope, frame_dt, envelope.mean(axis=0)
