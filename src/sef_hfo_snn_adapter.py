"""SNN -> observation adapter (Increment 2).

Pure functions: bin + temporally smooth per-neuron E spikes into a rate field, sample
at virtual-electrode contacts via the Increment-1 Gaussian footprint, and turn the
aggregate into an event window. This is the ONLY model-specific piece; everything
downstream (extract_lagpat -> real pipeline -> onset_front_axis) is the shared chain.
"""
from __future__ import annotations

import numpy as np

from src.sef_hfo_observation import sample_envelopes
from src.sef_hfo_events import calibrate_detector, detect_events


def _bin_and_smooth(E_spk_bool, dt, bin_ms, smooth_ms):
    """(nsteps, NE) bool -> (n_frame, NE) Gaussian-smoothed per-neuron rate."""
    E_spk_bool = np.asarray(E_spk_bool, bool)
    nsteps, NE = E_spk_bool.shape
    bs = max(1, int(round(bin_ms / dt)))
    nf = nsteps // bs
    binned = E_spk_bool[: nf * bs].reshape(nf, bs, NE).sum(axis=1).astype(float)
    sig = max(1e-6, smooth_ms / bin_ms)
    half = int(np.ceil(3 * sig))
    x = np.arange(-half, half + 1)
    k = np.exp(-(x ** 2) / (2 * sig ** 2))
    k /= k.sum()
    sm = np.apply_along_axis(lambda col: np.convolve(col, k, mode="same"), 0, binned)
    return sm, bin_ms


def snn_event_envelope(E_spk_bool, posE, montage, dt, bin_ms=2.0, smooth_ms=5.0,
                       kernel_width=0.25):
    """Per-contact activity envelope from per-neuron E spikes.

    Returns (envelopes (n_contact, n_frame), frame_dt_ms, aggregate (n_frame,)).
    Reuses Increment-1 sample_envelopes with grid_xy = posE (E-neuron coords) and
    source_frames = the smoothed per-neuron rate. aggregate = mean over contacts (feeds
    event_window_for_run). NOTE: this is a firing-density envelope, NOT a synaptic-current
    LFP — it validates the direction read-out only (spec reporting boundary)."""
    rate, frame_dt = _bin_and_smooth(E_spk_bool, dt, bin_ms, smooth_ms)
    env = sample_envelopes(rate, np.asarray(posE, float), montage, kernel_width)
    return env, frame_dt, env.mean(axis=0)


def event_window_for_run(agg_kick, agg_ref, frame_dt, frac=0.5):
    """Per-operating-point event window (spec Item 2): recalibrate the detector bar from
    the no-kick reference + this kick run (anti-circular: floor from ref, peak from kick),
    then detect the single self-limited event. Returns (t_on, t_off) ms or None (no
    self-terminating event => INSUFFICIENT). Shared by the SNN and rate runners."""
    cal = calibrate_detector([np.asarray(agg_ref, float)], np.asarray(agg_kick, float),
                             frac=frac)
    evs = [e for e in detect_events(np.asarray(agg_kick, float), frame_dt,
                                    event_on_frac=cal["event_on_frac"]) if e["returned"]]
    if not evs:
        return None
    e = max(evs, key=lambda d: d["peak_ext"])
    return (e["t_on"], e["t_off"])
