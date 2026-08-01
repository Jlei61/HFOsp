"""Small, auditable helpers for FCXR-LC2-Core.

This module contains no simulator launch and no hidden parameter search.  It provides the R1 raw
post-X recurrent-conductance observer and deterministic offline H replay used to answer the first
scientific question: can one temporal sensor separate returning IEDs from established high states?
"""
from __future__ import annotations

import hashlib

import numpy as np


SUMMARY_NAMES = ("mean", "q50", "q90", "q99")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RawGASampler:
    """Block-average sampled and full-population post-X ``gA_raw`` without an N x dt trace.

    ``sample`` is called once per membrane step.  Every ``stride_steps`` calls, the observer stores
    float32 block averages for fixed E-cell indices and four full-population summaries.  An incomplete
    trailing block is deliberately omitted and reported by ``n_pending_steps``.
    """

    def __init__(self, n_e, sample_idx_e, stride_steps):
        self.n_e = int(n_e)
        self.sample_idx_e = np.asarray(sample_idx_e, dtype=np.int64)
        self.stride_steps = int(stride_steps)
        if self.stride_steps < 1:
            raise ValueError("stride_steps must be >=1")
        if (self.sample_idx_e.ndim != 1 or self.sample_idx_e.size == 0
                or np.unique(self.sample_idx_e).size != self.sample_idx_e.size
                or self.sample_idx_e.min() < 0 or self.sample_idx_e.max() >= self.n_e):
            raise ValueError("sample_idx_e must be nonempty unique E indices within [0,n_e)")
        self._acc = np.zeros(self.n_e, dtype=np.float64)
        self._count = 0
        self._sampled = []
        self._summary = []
        self._last_step = -1

    def sample(self, g_a_raw_e, step_i):
        g = np.asarray(g_a_raw_e, dtype=float)
        if g.shape != (self.n_e,) or not np.all(np.isfinite(g)) or np.any(g < 0.0):
            raise ValueError(f"gA_raw must be finite, nonnegative and shape ({self.n_e},)")
        if int(step_i) != self._last_step + 1:
            raise ValueError(f"non-consecutive membrane steps: got {step_i}, expected {self._last_step + 1}")
        self._last_step = int(step_i)
        self._acc += g
        self._count += 1
        if self._count == self.stride_steps:
            avg = self._acc / float(self.stride_steps)
            self._sampled.append(avg[self.sample_idx_e].astype(np.float32))
            q50, q90, q99 = np.quantile(avg, [0.50, 0.90, 0.99])
            self._summary.append(np.asarray([
                avg.mean(), q50, q90, q99
            ], dtype=np.float32))
            self._acc.fill(0.0)
            self._count = 0

    @property
    def n_pending_steps(self):
        return int(self._count)

    def arrays(self):
        return dict(
            sample_idx_E=self.sample_idx_e.copy(),
            gA_sampled=np.stack(self._sampled) if self._sampled else np.empty((0, self.sample_idx_e.size), np.float32),
            gA_population_summary=(np.stack(self._summary) if self._summary
                                   else np.empty((0, len(SUMMARY_NAMES)), np.float32)),
            summary_names=np.asarray(SUMMARY_NAMES),
            stride_steps=np.asarray([self.stride_steps], dtype=np.int64),
            n_pending_steps=np.asarray([self.n_pending_steps], dtype=np.int64),
        )


def replay_h(g_a, tau_ms, dt_ms, h0=None):
    """Exact causal first-order replay for a time x cell block-averaged input array."""
    g = np.asarray(g_a, dtype=float)
    if g.ndim != 2 or not np.all(np.isfinite(g)) or np.any(g < 0.0):
        raise ValueError("g_a must be a finite nonnegative (time,cell) array")
    if not (np.isfinite(tau_ms) and tau_ms > 0.0 and np.isfinite(dt_ms) and dt_ms > 0.0):
        raise ValueError("tau_ms and dt_ms must be finite and >0")
    h = np.zeros(g.shape[1], dtype=float) if h0 is None else np.asarray(h0, dtype=float).copy()
    if h.shape != (g.shape[1],) or not np.all(np.isfinite(h)) or np.any(h < 0.0):
        raise ValueError("h0 must be finite, nonnegative and match the cell dimension")
    decay = float(np.exp(-dt_ms / tau_ms))
    gain = 1.0 - decay
    # scipy's compiled lfilter computes the post-update state y[t]=decay*y[t-1]+gain*g[t].
    # Shift it by one row so out[t] remains h(t-), matching the membrane's causal convention.
    from scipy.signal import lfilter
    y, _zf = lfilter([gain], [1.0, -decay], g, axis=0, zi=(decay * h)[None, :])
    out = np.empty_like(g, dtype=np.float32)
    if g.shape[0]:
        out[0] = h
        out[1:] = y[:-1]
        h = np.asarray(y[-1], float)
    return out, h


def contiguous_true_intervals(x, grid):
    """Inclusive value intervals for contiguous True runs on an ordered 1-D grid."""
    x = np.asarray(x, bool)
    grid = np.asarray(grid, float)
    if x.ndim != 1 or grid.shape != x.shape or np.any(np.diff(grid) <= 0.0):
        raise ValueError("x/grid must be aligned 1-D arrays and grid strictly increasing")
    out = []
    start = None
    for i, flag in enumerate(x):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == x.size - 1):
            end = i if flag and i == x.size - 1 else i - 1
            out.append((float(grid[start]), float(grid[end])))
            start = None
    return out
