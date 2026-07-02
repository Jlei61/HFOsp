"""Pure geometry/metric helpers for the axis-vs-core stimulation figure (Topic 4).

No SNN engine imports -- fully unit-testable. Consumed by the stim runner and the figure script.
See docs/superpowers/specs/2026-07-02-topic4-axis-vs-core-stim-difficulty-design.md."""
from __future__ import annotations

import numpy as np


def linear_montage(center, axis_unit, n_contacts=11, pitch=1.2):
    """Virtual-SEEG contacts along ``axis_unit`` through ``center`` (mm). Returns (contacts, names)."""
    center = np.asarray(center, float)
    u = np.asarray(axis_unit, float)
    offs = (np.arange(n_contacts) - (n_contacts - 1) / 2.0) * float(pitch)
    contacts = center[None, :] + offs[:, None] * u[None, :]
    return contacts, [f"C{i}" for i in range(n_contacts)]


def split_source_axis(contacts, center, core_radius):
    """Contact indices within ``core_radius`` of ``center`` (source) vs outside (axis/downstream)."""
    d = np.linalg.norm(np.asarray(contacts, float) - np.asarray(center, float)[None, :], axis=1)
    return np.flatnonzero(d <= float(core_radius)), np.flatnonzero(d > float(core_radius))


def select_footprint(contacts, center, axis_unit, source_idx, axis_idx, N):
    """Fixed-footprint contact selection (fairness contract). core = N source contacts nearest the
    centre (partial cover -> residual source); axis = N downstream contacts split symmetrically
    (N/2 nearest each side along axis_unit). Deterministic tie-break: (distance, lower index)."""
    assert N % 2 == 0, "footprint N must be even (symmetric axis split)"
    assert N < len(source_idx), f"N={N} must be < n_source_contacts={len(source_idx)} (core not fully coverable)"
    C = np.asarray(contacts, float); c = np.asarray(center, float); u = np.asarray(axis_unit, float)
    d = np.linalg.norm(C - c[None, :], axis=1)
    proj = (C - c[None, :]) @ u
    core = sorted(source_idx.tolist(), key=lambda i: (d[i], i))[:N]
    pos = sorted([i for i in axis_idx.tolist() if proj[i] > 0], key=lambda i: (d[i], i))[:N // 2]
    neg = sorted([i for i in axis_idx.tolist() if proj[i] < 0], key=lambda i: (d[i], i))[:N // 2]
    axis = pos + neg
    assert len(core) == N and len(axis) == N, "footprint could not be filled — check montage/N"
    return np.array(sorted(core)), np.array(sorted(axis))


def onset_time_field(E_spk_bool, dt):
    """Per-E-cell first-spike time in ms; nan for cells that never spiked. E_spk_bool: (nsteps, NE)."""
    spk = np.asarray(E_spk_bool, bool)
    ever = spk.any(axis=0)
    first = np.argmax(spk, axis=0).astype(float) * float(dt)
    first[~ever] = np.nan
    return first


def runaway_delay_ms(runaway_stim, runaway_nostim, T):
    """Delay = (runaway_stim or T) - runaway_nostim. nan if no baseline runaway (undefined)."""
    if runaway_nostim is None:
        return float("nan")
    rs = float(T) if runaway_stim is None else float(runaway_stim)
    return float(rs - float(runaway_nostim))


def count_events_pre_runaway(af, bin_w, runaway_ms, detect_events, *, baseline_ms=(5.0, 50.0),
                             cal_frac_burst=0.5, cal_frac_train=0.15, ramp_ms=50.0,
                             train_min_runaway_ms=300.0):
    """Count discrete events on the active-fraction series ``af`` (bin width ``bin_w`` ms).

    A LATE runaway (a train, e.g. the kick at ~757 ms) would let its big detonation peak set the
    event bar ~13x above the small near-baseline pre-runaway train and hide it (the record-peak
    confound the spontaneous runner documents). So for ``runaway_ms >= train_min_runaway_ms`` the bar
    is calibrated from the pre-runaway window (minus the last ``ramp_ms`` detonation ramp) with a
    sensitive fraction ``cal_frac_train``; for an IMMEDIATE burst the whole record is used with
    ``cal_frac_burst``. ``detect_events`` is injected (``src.sef_hfo_events.detect_events``) so this
    module stays engine-free and the count is unit-testable. Returns ``(n_events, bar)``."""
    af = np.asarray(af, float)
    nb0, nb1 = int(baseline_ms[0] / bin_w), int(baseline_ms[1] / bin_w)
    if runaway_ms is not None and runaway_ms >= train_min_runaway_ms:
        af_c = af[:max(int(round((runaway_ms - ramp_ms) / bin_w)), 1)]
        frac = cal_frac_train
    else:
        af_c = af
        frac = cal_frac_burst
    floor = float(np.percentile(af_c[nb0:nb1], 95)) if (nb1 <= len(af_c) and nb1 > nb0) else float(af_c.min())
    bar = floor + frac * (float(af_c.max()) - floor)
    return len(detect_events(af_c, bin_w, event_on_frac=bar)), bar
