"""Finite-horizon calibration for FCXR-LC5v2.

This module is deliberately blind to seizure labels and space.  It consumes a frozen per-cell
spike stream, integrates the already-registered LC5 load equation, and returns per-cell load and
baseline fields.  The onset/window coordinates are supplied by the caller as calibration supports;
they never enter the runtime mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import numpy as np

from src.topic4_fcxr_lc5 import SparseSpikeStream
from src.topic4_mz_fcxr_pump import (
    fit_p0_shrinkage,
    pump_activation,
    rate_decile_groups,
)


CALIBRATION_DT_MS = 1.0
ACTIVATION_SAMPLE_MS = 5.0
HIST_BINS = 4096


def array_sha256(value) -> str:
    a = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(a.tobytes())
    return h.hexdigest()


def coarsen_sparse_stream(
    stream: SparseSpikeStream,
    *,
    source_dt_ms: float,
    target_dt_ms: float = CALIBRATION_DT_MS,
    stop_ms: float | None = None,
) -> SparseSpikeStream:
    """Bin an exact boolean LIF stream onto the locked calibration clock.

    Sorting is repeated because floor-binning can change the within-bin cell order.  A duplicate
    cell/bin is retained only as a count by callers outside this project; the registered E-cell
    refractory period is 2 ms, so LC5's 1 ms clock must have no duplicates and fails loudly if it
    does.
    """

    if not (np.isfinite(source_dt_ms) and source_dt_ms > 0.0):
        raise ValueError("source_dt_ms must be finite and positive")
    if not (np.isfinite(target_dt_ms) and target_dt_ms >= source_dt_ms):
        raise ValueError("target_dt_ms must be finite and >= source_dt_ms")
    ratio = float(target_dt_ms) / float(source_dt_ms)
    factor = int(round(ratio))
    if not np.isclose(ratio, factor, rtol=0.0, atol=1e-12):
        raise ValueError("target_dt_ms/source_dt_ms must be an integer")
    total_ms = stream.n_steps * float(source_dt_ms)
    end_ms = total_ms if stop_ms is None else float(stop_ms)
    if not (0.0 < end_ms <= total_ms + 1e-9):
        raise ValueError("stop_ms must lie inside the source stream")
    source_stop = min(stream.n_steps, int(round(end_ms / float(source_dt_ms))))
    right = int(np.searchsorted(stream.steps, source_stop, side="left"))
    steps = np.asarray(stream.steps[:right] // factor, dtype=np.int64)
    cells = np.asarray(stream.cells[:right], dtype=np.int64)
    if steps.size:
        order = np.lexsort((cells, steps))
        steps, cells = steps[order], cells[order]
        duplicate = (steps[1:] == steps[:-1]) & (cells[1:] == cells[:-1])
        if np.any(duplicate):
            raise ValueError("a cell spiked twice inside one calibration bin")
    n_steps = int(round(end_ms / float(target_dt_ms)))
    return SparseSpikeStream(steps, cells, n_steps, stream.n_cells)


@dataclass
class FiniteReplayResult:
    u_final: np.ndarray
    snapshots: dict[str, np.ndarray]
    block_phi_mean: dict[str, np.ndarray]
    target_phi_median: float | None
    target_sample_count: int
    excess_integral_ms: np.ndarray | None


def _histogram_median(counts: np.ndarray) -> float:
    total = int(np.sum(counts))
    if total <= 0:
        raise ValueError("activation median has no samples")
    k = (total - 1) // 2
    idx = int(np.searchsorted(np.cumsum(counts), k + 1, side="left"))
    return (idx + 0.5) / counts.size


def replay_finite_load(
    stream: SparseSpikeStream,
    *,
    dt_ms: float,
    tau_ms: float,
    a_load: float,
    h: int = 3,
    blocks: Mapping[str, tuple[int, int]] | None = None,
    target_block: str | None = None,
    sample_every_steps: int = 1,
    snapshot_steps: Mapping[int, str] | None = None,
    p0: np.ndarray | None = None,
    excess_block: str | None = None,
    u0: np.ndarray | None = None,
) -> FiniteReplayResult:
    """Replay the load on a binned stream using the engine's causal order.

    ``phi(u(t^-))`` is observed and cleared before spikes in the current bin are added.  U2 itself
    still runs at the engine's 0.05 ms clock; this deterministic 1 ms observer is only calibration.
    """

    if not (np.isfinite(dt_ms) and dt_ms > 0.0):
        raise ValueError("dt_ms must be finite and positive")
    if not (np.isfinite(tau_ms) and tau_ms > 0.0):
        raise ValueError("tau_ms must be finite and positive")
    if not (np.isfinite(a_load) and a_load >= 0.0):
        raise ValueError("a_load must be finite and non-negative")
    if int(h) < 1 or int(sample_every_steps) < 1:
        raise ValueError("h and sample_every_steps must be positive integers")
    block_map = {str(k): (int(v[0]), int(v[1])) for k, v in (blocks or {}).items()}
    for name, (lo, hi) in block_map.items():
        if not (0 <= lo < hi <= stream.n_steps):
            raise ValueError(f"invalid block {name}: {(lo, hi)}")
    if target_block is not None and target_block not in block_map:
        raise ValueError("target_block must name a registered block")
    if excess_block is not None and excess_block not in block_map:
        raise ValueError("excess_block must name a registered block")
    snaps = {int(k): str(v) for k, v in (snapshot_steps or {}).items()}
    if any(k < 0 or k >= stream.n_steps for k in snaps):
        raise ValueError("snapshot step outside stream")
    u = np.zeros(stream.n_cells, dtype=np.float64) if u0 is None else np.asarray(u0, float).copy()
    if u.shape != (stream.n_cells,) or not np.all(np.isfinite(u)) or np.any(u < 0.0):
        raise ValueError("u0 must be a finite non-negative per-cell field")
    p0_arr = None if p0 is None else np.asarray(p0, float)
    if p0_arr is not None and (p0_arr.shape != u.shape or not np.all(np.isfinite(p0_arr))):
        raise ValueError("p0 must be a finite per-cell field")

    phi_sums = {name: np.zeros(stream.n_cells) for name in block_map}
    phi_counts = {name: 0 for name in block_map}
    histogram = np.zeros(HIST_BINS, dtype=np.int64) if target_block is not None else None
    target_count = 0
    excess = np.zeros(stream.n_cells) if excess_block is not None else None
    snapshots_out: dict[str, np.ndarray] = {}
    pos = 0
    spike_counts = np.zeros(stream.n_cells)
    previous_cells = np.empty(0, dtype=np.int64)

    for step in range(stream.n_steps):
        if previous_cells.size:
            spike_counts[previous_cells] = 0.0
        phi = pump_activation(u, h)
        for name, (lo, hi) in block_map.items():
            if lo <= step < hi:
                phi_sums[name] += phi
                phi_counts[name] += 1
        if target_block is not None:
            lo, hi = block_map[target_block]
            if lo <= step < hi and (step - lo) % int(sample_every_steps) == 0:
                index = np.minimum((phi * HIST_BINS).astype(np.int64), HIST_BINS - 1)
                histogram += np.bincount(index, minlength=HIST_BINS)
                target_count += phi.size
        if excess_block is not None:
            lo, hi = block_map[excess_block]
            if lo <= step < hi:
                if p0_arr is None:
                    raise ValueError("excess_block requires p0")
                excess += np.maximum(phi - p0_arr, 0.0) * float(dt_ms)

        end = int(np.searchsorted(stream.steps, step, side="right"))
        cells = stream.cells[pos:end]
        if cells.size:
            spike_counts[cells] = 1.0
        previous_cells = cells
        pos = end
        # Match MZSlowVars/replay_sparse_loads expression and operation order exactly when the
        # calibration clock equals the engine clock.
        np.maximum(
            u + float(a_load) * spike_counts - (float(dt_ms) / float(tau_ms)) * phi,
            0.0,
            out=u,
        )
        if step in snaps:
            snapshots_out[snaps[step]] = u.copy()

    means = {name: phi_sums[name] / phi_counts[name] for name in block_map}
    median = None if histogram is None else _histogram_median(histogram)
    return FiniteReplayResult(
        u_final=u,
        snapshots=snapshots_out,
        block_phi_mean=means,
        target_phi_median=median,
        target_sample_count=int(target_count),
        excess_integral_ms=excess,
    )


def solve_a_for_window_target(
    stream: SparseSpikeStream,
    *,
    dt_ms: float,
    tau_ms: float,
    target_window: tuple[int, int],
    target: float = 0.5,
    sample_every_steps: int = 5,
    tolerance: float = 5e-4,
    max_iter: int = 40,
) -> dict:
    """Monotone deterministic bisection for the finite-window activation target."""

    if not (0.0 < target < 1.0):
        raise ValueError("target must lie in (0,1)")
    blocks = {"target": target_window}

    def evaluate(a):
        return replay_finite_load(
            stream, dt_ms=dt_ms, tau_ms=tau_ms, a_load=a, blocks=blocks,
            target_block="target", sample_every_steps=sample_every_steps,
        ).target_phi_median

    lo, hi = 0.0, 1e-4
    y_lo, y_hi = evaluate(lo), evaluate(hi)
    while y_hi < target and hi < 10.0:
        lo, y_lo = hi, y_hi
        hi *= 2.0
        y_hi = evaluate(hi)
    if y_hi < target:
        raise RuntimeError("could not bracket the finite-window activation target")
    history = [(lo, y_lo), (hi, y_hi)]
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        y_mid = evaluate(mid)
        history.append((mid, y_mid))
        if abs(y_mid - target) <= tolerance:
            return {
                "a_load": float(mid), "achieved_target": float(y_mid),
                "iterations": len(history) - 2, "bracket": [float(lo), float(hi)],
                "history": [[float(a), float(y)] for a, y in history],
            }
        if y_mid < target:
            lo, y_lo = mid, y_mid
        else:
            hi, y_hi = mid, y_mid
    mid = 0.5 * (lo + hi)
    y_mid = evaluate(mid)
    if abs(y_mid - target) > max(tolerance, 1.0 / HIST_BINS):
        raise RuntimeError("finite-window activation bisection did not converge")
    return {
        "a_load": float(mid), "achieved_target": float(y_mid),
        "iterations": int(max_iter), "bracket": [float(lo), float(hi)],
        "history": [[float(a), float(y)] for a, y in history] + [[float(mid), float(y_mid)]],
    }


def estimate_shrunken_p0(phi_baseline_blocks, baseline_rate_hz, *, n_groups=10) -> dict:
    blocks = np.asarray(phi_baseline_blocks, float)
    rates = np.asarray(baseline_rate_hz, float)
    if blocks.ndim != 2 or blocks.shape[1:] != rates.shape or blocks.shape[0] < 3:
        raise ValueError("p0 needs >=3 aligned baseline blocks")
    groups = rate_decile_groups(rates, n_groups=n_groups)
    fit = fit_p0_shrinkage(blocks, groups, n_groups=n_groups)
    p0 = np.asarray(fit["p0"], float)
    return {
        "p0": p0,
        "weight": float(fit["weight"]),
        "cv_weights": fit["cv_weights"],
        "cv_error": fit["cv_error"],
        "p0_sha256": array_sha256(p0),
        "groups_sha256": array_sha256(groups),
    }


def calibrate_episode_dose(
    *,
    unit_excess_integral_ms,
    recurrent_force_integral_ms,
    gammas=(0.10, 0.25, 0.40),
) -> dict:
    num = np.asarray(unit_excess_integral_ms, float)
    den = np.asarray(recurrent_force_integral_ms, float)
    if num.ndim != 1 or den.shape != num.shape:
        raise ValueError("dose calibration requires aligned per-cell integrals")
    if not np.all(np.isfinite(num)) or not np.all(np.isfinite(den)):
        raise ValueError("dose integrals must be finite")
    num_med, den_med = float(np.median(num)), float(np.median(den))
    if num_med <= 0.0 or den_med <= 0.0:
        raise ValueError("dose calibration medians must be positive")
    imax = {str(float(g)): float(g) * den_med / num_med for g in gammas}
    return {
        "unit_excess_integral_median_ms": num_med,
        "recurrent_force_integral_median_ms": den_med,
        "Imax_by_gamma": imax,
        "unit_excess_integral_sha256": array_sha256(num),
        "recurrent_force_integral_sha256": array_sha256(den),
    }
