"""Increment-3a rate-model parity adapter for the virtual-SEEG observation layer.

Lets the canonical LIF rate field (``src.sef_hfo_lif.integrate_lif_field``) feed
the SAME observation chain as the SNN (Increment 2): a finite disk-pulse stim_fn
to ignite a localized event, and a thin adapter turning the per-step field stack
(``return_frames=True``) into per-contact envelopes via the SAME
``sample_envelopes`` + ``grid_coords`` used for the SNN.

Design locks (plan 2026-06-07-...-increment3 §0 / §File-structure):
  * Drives the canonical ``integrate_lif_field`` — NOT the old sigmoid
    ``src.sef_hfo_pulse`` (that would validate the wrong substrate).
  * Reshape is C-order, aligned with ``grid_coords`` (both use ``_grid``'s
    ij-indexed, centered grid) — the field-space grid-alignment pit.
"""
import numpy as np

from src.sef_hfo_lif import _grid
from src.sef_hfo_observation import grid_coords, sample_envelopes


def pulse_stim_fn(center, radius, amp, t_on, t_off, n, L):
    """A finite disk-pulse ``stim_fn(t)`` for ``integrate_lif_field``.

    Returns a callable that, during ``t_on <= t < t_off`` (t_off EXCLUSIVE),
    yields an (n, n) field equal to ``amp`` inside ``radius`` mm of ``center``
    and 0 elsewhere; outside that window it returns the scalar ``0.0`` (the
    ``integrate_lif_field`` stim contract: an (n, n) array OR scalar 0).

    ``center`` is the confound knob: CENTER (0,0) for the C-track / isotropic
    controls (symmetric seed, no kick-position confound), OFF-CENTER for the
    kick-track control (seed-position vs connectivity-axis dissociation).

    Parameters
    ----------
    center : (cx, cy) in mm (model frame).
    radius : disk radius (mm).
    amp : pulse amplitude (added to muE inside the disk).
    t_on, t_off : on/off times (ms); on iff ``t_on <= t < t_off``.
    n, L : grid points and physical size (mm) — required to build the (n, n)
        mask; ``integrate_lif_field``'s stim must match its own n, L.
    """
    X, Y = _grid(n, L)
    cx, cy = center
    mask = ((X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2).astype(float)
    pulse = amp * mask

    def stim_fn(t):
        return pulse if (t_on <= t < t_off) else 0.0

    return stim_fn


def rate_event_envelope(rE_frames, n, L, montage, kernel_width):
    """Per-contact activity envelope from a rate-field frame stack.

    Reshapes the ``(nsteps, n, n)`` stack (from
    ``integrate_lif_field(..., return_frames=True)``) to ``(nsteps, n*n)`` in
    C-order — which matches ``grid_coords(n, L)``'s ``.ravel()`` order because
    both come from ``_grid``'s ij-indexed, centered grid — then samples it at the
    montage contacts with the SAME ``sample_envelopes`` used for the SNN.

    Returns
    -------
    envelopes : ndarray, shape ``(n_contact, nsteps)`` — distance-weighted
        per-contact activity over time, ready for ``extract_lagpat``.
    """
    frames = np.asarray(rE_frames, float)
    if frames.ndim != 3 or frames.shape[1] != n or frames.shape[2] != n:
        raise ValueError(
            f"rE_frames must be (nsteps, {n}, {n}); got {frames.shape}")
    flat = frames.reshape(frames.shape[0], -1)         # (nsteps, n*n), C-order
    return sample_envelopes(flat, grid_coords(n, L), montage, kernel_width)
