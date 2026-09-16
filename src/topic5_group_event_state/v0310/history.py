"""Rebuild delayed-availability histories for any horizon from real measurement.

Clause C18. The bundle only stores H=0.5 and H=8. Every other horizon is
reconstructed from ``event_replay_blocks`` -- the same (release, times,
cumsum) arrays the original builder consumed -- using the identical
construction. ``verify_against_stored`` re-derives the two stored horizons and
requires exact equality, so a new horizon can never be a copy of H8, a
zero-padded fake, or a peek at unpublished events.
"""
from __future__ import annotations

import numpy as np

MAX_STEP_HOURS = 1 / 12


def build_history(blocks, anchor, hours, input_dim):
    """Return (inputs[T,D], dt_hours[T]) exactly as the frozen builder does.

    ``blocks`` must be ordered by block start, each with 'release', 'times'
    and 'cumsum'. Only blocks released at or before the anchor are visible.
    """
    left = anchor - hours * 3600.0
    last = left
    xs, dts = [], []
    for b in blocks:
        release = float(b['release'])
        if release > anchor or release <= left:
            continue
        cursor = int(np.searchsorted(b['times'], left))
        delta = (release - last) / 3600.0
        pieces = max(1, int(np.ceil(delta / MAX_STEP_HOURS)))
        for _ in range(pieces - 1):
            xs.append(np.zeros(input_dim))
            dts.append(delta / pieces)
        xs.append(b['cumsum'][-1] - b['cumsum'][cursor])
        dts.append(delta / pieces)
        last = release
    delta = (anchor - last) / 3600.0
    pieces = max(1, int(np.ceil(delta / MAX_STEP_HOURS)))
    for _ in range(pieces):
        xs.append(np.zeros(input_dim))
        dts.append(delta / pieces)
    return np.asarray(xs, np.float32), np.asarray(dts, np.float32)


def verify_against_stored(data, atol=0.0):
    """Clause C18 gate: rebuilt H must equal the stored H exactly."""
    blocks = data['event_replay_blocks']
    report = {}
    for key, hours in ((k, float(k)) for k in data['samples'][0]['histories']):
        worst_x = worst_dt = 0.0
        for s in data['samples']:
            x_ref, dt_ref = s['histories'][key]
            x_new, dt_new = build_history(blocks, s['anchor'], hours, data['input_dim'])
            if x_new.shape != x_ref.shape or dt_new.shape != dt_ref.shape:
                raise ValueError(f'Rebuilt H={key} changes shape; refusing to trust new horizons')
            worst_x = max(worst_x, float(np.abs(x_new - x_ref).max(initial=0.0)))
            worst_dt = max(worst_dt, float(np.abs(dt_new - dt_ref).max(initial=0.0)))
        if worst_x > atol or worst_dt > atol:
            raise ValueError(f'Rebuilt H={key} differs from stored (dx={worst_x}, ddt={worst_dt})')
        report[key] = dict(max_abs_input_difference=worst_x, max_abs_dt_difference=worst_dt, n_samples=len(data['samples']))
    return report


def past_coverage(support, anchor, hours):
    """Fraction of the past horizon with real measurement support."""
    lo = anchor - hours * 3600.0
    covered = np.maximum(0.0, np.minimum(support[:, 1], anchor) - np.maximum(support[:, 0], lo)).sum()
    return float(covered / (hours * 3600.0))


def history_matrix(data, hours, *, min_coverage=None):
    """Padded (n,T,D) inputs, (n,T) dt, lengths and per-anchor past coverage.

    ``min_coverage`` is only a report/eligibility field here; callers decide
    whether an anchor is estimable. Nothing is imputed for uncovered time --
    the state simply advances autonomously across it, exactly as the frozen
    builder does.
    """
    blocks = data['event_replay_blocks']
    samples = data['samples']
    key = str(float(hours))
    rebuilt = []
    for s in samples:
        if key in s['histories']:
            x, dt = s['histories'][key]
        else:
            x, dt = build_history(blocks, s['anchor'], hours, data['input_dim'])
        rebuilt.append((x, dt))
    lengths = np.array([len(x) for x, _ in rebuilt], int)
    if not lengths.max():
        raise ValueError('Empty history for every anchor; horizon is not estimable')
    steps = int(lengths.max())
    x_out = np.zeros((len(samples), steps, data['input_dim']), np.float32)
    dt_out = np.zeros((len(samples), steps), np.float32)
    for i, (x, dt) in enumerate(rebuilt):
        x_out[i, :len(x)] = x
        dt_out[i, :len(dt)] = dt
    coverage = np.array([past_coverage(data['observed_support'], s['anchor'], hours) for s in samples])
    eligible = np.ones(len(samples), bool) if min_coverage is None else coverage >= min_coverage
    return dict(x=x_out, dt=dt_out, lengths=lengths, coverage=coverage, eligible=eligible,
                horizon_hours=float(hours), steps=steps, rebuilt_from_replay=key not in samples[0]['histories'])
