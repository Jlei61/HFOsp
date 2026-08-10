"""FCXR-LC5 execution primitives.

The scientific state remains per cell.  This module only supplies deterministic sparse-spike
replay, analytic scale locking, JSON normalization, and stage-level artifact transactions.  It has
no access to spatial labels, population seizure classifiers, or the membrane update.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping

import numpy as np

from src.topic4_mz_fcxr_pump import pump_activation


@dataclass(frozen=True)
class SparseSpikeStream:
    """COO representation of an E-cell spike raster, ordered by ``(step, cell)``."""

    steps: np.ndarray
    cells: np.ndarray
    n_steps: int
    n_cells: int

    def __post_init__(self):
        steps = np.asarray(self.steps, dtype=np.int64)
        cells = np.asarray(self.cells, dtype=np.int64)
        if steps.ndim != 1 or cells.ndim != 1 or steps.shape != cells.shape:
            raise ValueError("sparse spike steps/cells must be aligned 1-D arrays")
        if int(self.n_steps) < 0 or int(self.n_cells) <= 0:
            raise ValueError("n_steps must be >=0 and n_cells must be >0")
        if steps.size:
            if steps.min() < 0 or steps.max() >= int(self.n_steps):
                raise ValueError("sparse spike step outside [0,n_steps)")
            if cells.min() < 0 or cells.max() >= int(self.n_cells):
                raise ValueError("sparse spike cell outside [0,n_cells)")
            order = np.lexsort((cells, steps))
            if not np.array_equal(order, np.arange(steps.size)):
                raise ValueError("sparse spikes must be sorted by (step,cell)")
            pairs = np.column_stack((steps, cells))
            if np.any(np.all(pairs[1:] == pairs[:-1], axis=1)):
                raise ValueError("duplicate (step,cell) spike in a boolean LIF raster")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "n_steps", int(self.n_steps))
        object.__setattr__(self, "n_cells", int(self.n_cells))

    @classmethod
    def from_dense(cls, raster):
        raster = np.asarray(raster, dtype=bool)
        if raster.ndim != 2:
            raise ValueError("dense spike raster must be n_steps x n_cells")
        steps, cells = np.nonzero(raster)
        return cls(steps=steps, cells=cells, n_steps=raster.shape[0], n_cells=raster.shape[1])

    @property
    def sha256(self):
        h = hashlib.sha256()
        h.update(np.asarray([self.n_steps, self.n_cells], dtype=np.int64).tobytes())
        h.update(self.steps.tobytes())
        h.update(self.cells.tobytes())
        return h.hexdigest()

    def active_fraction(self, *, dt_ms: float, bin_ms: float):
        """Exact sparse equivalent of dense ``any-within-bin`` active fraction."""

        if not (np.isfinite(dt_ms) and dt_ms > 0.0 and np.isfinite(bin_ms) and bin_ms > 0.0):
            raise ValueError("dt_ms and bin_ms must be finite and positive")
        steps_per_bin = max(1, int(round(float(bin_ms) / float(dt_ms))))
        n_bins = self.n_steps // steps_per_bin
        if n_bins == 0:
            return np.zeros(0, dtype=float), steps_per_bin * float(dt_ms)
        keep = self.steps < n_bins * steps_per_bin
        bins = self.steps[keep] // steps_per_bin
        cells = self.cells[keep]
        if bins.size:
            pair_code = bins * self.n_cells + cells
            counts = np.bincount(
                np.unique(pair_code) // self.n_cells, minlength=n_bins
            ).astype(float)
        else:
            counts = np.zeros(n_bins, dtype=float)
        return counts / self.n_cells, steps_per_bin * float(dt_ms)

    def per_cell_rate_hz(self, *, lo_step: int = 0, hi_step: int | None = None, dt_ms: float):
        """Per-cell firing rate over ``[lo_step, hi_step)`` without a dense raster."""

        hi = self.n_steps if hi_step is None else int(hi_step)
        lo = int(lo_step)
        if not (0 <= lo < hi <= self.n_steps):
            raise ValueError("rate interval must be a non-empty subset of the stream")
        if not (np.isfinite(dt_ms) and dt_ms > 0.0):
            raise ValueError("dt_ms must be finite and positive")
        left = int(np.searchsorted(self.steps, lo, side="left"))
        right = int(np.searchsorted(self.steps, hi, side="left"))
        counts = np.bincount(self.cells[left:right], minlength=self.n_cells)
        return counts.astype(float) / ((hi - lo) * float(dt_ms) / 1000.0)


class SparseSpikeBinaryWriter:
    """Bounded-memory sink for exact-loop sparse spike callbacks."""

    _DTYPE = np.dtype([("step", "<i8"), ("cell", "<i4")])

    def __init__(self, path, *, step_origin: int, n_steps: int, n_cells: int):
        self.path = Path(path)
        self.step_origin = int(step_origin)
        self.n_steps = int(n_steps)
        self.n_cells = int(n_cells)
        if self.step_origin < 0 or self.n_steps < 0 or self.n_cells <= 0:
            raise ValueError("invalid sparse writer dimensions")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("wb")
        self.n_spikes = 0

    def __call__(self, absolute_step, cells):
        if self._fh is None:
            raise RuntimeError("sparse spike writer is closed")
        cells = np.asarray(cells, dtype=np.int64)
        rel = int(absolute_step) - self.step_origin
        if not (0 <= rel < self.n_steps):
            raise ValueError("spike callback step outside writer interval")
        if cells.ndim != 1 or (cells.size and (cells.min() < 0 or cells.max() >= self.n_cells)):
            raise ValueError("spike callback contains an invalid E-cell index")
        if cells.size and np.any(cells[1:] <= cells[:-1]):
            raise ValueError("spike callback cells must be strictly increasing")
        rows = np.empty(cells.size, dtype=self._DTYPE)
        rows["step"] = rel
        rows["cell"] = cells.astype(np.int32, copy=False)
        rows.tofile(self._fh)
        self.n_spikes += int(cells.size)

    def close(self):
        if self._fh is not None:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
            self._fh = None

    def finalize(self, npz_path):
        self.close()
        rows = np.fromfile(self.path, dtype=self._DTYPE)
        if rows.size != self.n_spikes:
            raise IOError(f"sparse spike row count {rows.size} != callback count {self.n_spikes}")
        stream = SparseSpikeStream(
            rows["step"].astype(np.int64), rows["cell"].astype(np.int64),
            self.n_steps, self.n_cells,
        )
        out = Path(npz_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            tmp,
            steps=stream.steps,
            cells=stream.cells.astype(np.int32),
            n_steps=np.asarray([stream.n_steps], dtype=np.int64),
            n_cells=np.asarray([stream.n_cells], dtype=np.int64),
            sha256=np.asarray([stream.sha256]),
        )
        os.replace(tmp, out)
        return stream

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def load_sparse_spike_stream(path):
    """Load and verify a portable sparse stream artifact."""

    with np.load(path, allow_pickle=False) as z:
        stream = SparseSpikeStream(
            z["steps"], z["cells"], int(z["n_steps"][0]), int(z["n_cells"][0])
        )
        expected = str(z["sha256"][0])
    if stream.sha256 != expected:
        raise ValueError(f"sparse spike hash mismatch: {stream.sha256} != {expected}")
    return stream


class RecurrentDriveBlockObserver:
    """Pure per-cell block means for raw/effective recurrent E drive.

    This keeps the U1 calibration support in its native per-cell form without writing an
    ``N_cell x N_step`` tensor.  Sampling is deterministic and read-only; block means use the same
    time support for raw conductance and post-saturation force.
    """

    def __init__(self, n_cells, *, sample_every, steps_per_block, force_scale):
        self.n_cells = int(n_cells)
        self.sample_every = int(sample_every)
        self.steps_per_block = int(steps_per_block)
        self.force_scale = float(force_scale)
        if self.n_cells <= 0 or self.sample_every <= 0 or self.steps_per_block <= 0:
            raise ValueError("invalid recurrent observer dimensions")
        if not np.isfinite(self.force_scale):
            raise ValueError("force_scale must be finite")
        self._block = None
        self._raw_sum = np.zeros(self.n_cells)
        self._force_sum = np.zeros(self.n_cells)
        self._count = 0
        self.block_indices = []
        self.block_raw_mean = []
        self.block_force_mean = []

    def _flush(self):
        if self._block is None or self._count == 0:
            return
        self.block_indices.append(int(self._block))
        self.block_raw_mean.append((self._raw_sum / self._count).copy())
        self.block_force_mean.append((self._force_sum / self._count).copy())
        self._raw_sum.fill(0.0)
        self._force_sum.fill(0.0)
        self._count = 0

    def sample(self, raw_conductance, effective_conductance, absolute_step):
        step = int(absolute_step)
        if step % self.sample_every:
            return
        raw = np.asarray(raw_conductance, float)
        effective = np.asarray(effective_conductance, float)
        if raw.shape != (self.n_cells,) or effective.shape != (self.n_cells,):
            raise ValueError("recurrent observer requires aligned per-E-cell arrays")
        block = step // self.steps_per_block
        if self._block is None:
            self._block = block
        elif block != self._block:
            self._flush()
            self._block = block
        self._raw_sum += raw
        self._force_sum += effective * self.force_scale
        self._count += 1

    def arrays(self):
        self._flush()
        return {
            "block_index": np.asarray(self.block_indices, dtype=np.int32),
            "raw_conductance_mean": np.asarray(self.block_raw_mean, dtype=np.float32),
            "effective_force_mean": np.asarray(self.block_force_mean, dtype=np.float32),
        }


class ExactInputHasher:
    """Streaming hash of the actual OU state and per-cell external Poisson draw."""

    def __init__(self):
        self._hash = hashlib.sha256()
        self.n_steps = 0

    def __call__(self, absolute_step, xi, external_counts):
        ext = np.ascontiguousarray(np.asarray(external_counts, dtype=np.float64))
        self._hash.update(np.asarray([int(absolute_step)], dtype=np.int64).tobytes())
        self._hash.update(np.asarray([float(xi)], dtype=np.float64).tobytes())
        self._hash.update(ext.tobytes())
        self.n_steps += 1

    @property
    def sha256(self):
        return self._hash.hexdigest()


def replay_sparse_loads(
    stream: SparseSpikeStream,
    *,
    candidates: Mapping[str, Mapping[str, object]],
    dt_ms: float,
    snapshot_steps: Mapping[int, str] | None = None,
    blocks: Mapping[str, tuple[int, int]] | None = None,
):
    """Replay the locked discrete load equation without materializing an ``N x T`` raster.

    ``snapshot_steps`` are captured after the update for that step, matching
    :func:`src.topic4_mz_fcxr_pump.integrate_load_from_raster` and ``MZSlowVars.step``.
    """

    if not (np.isfinite(dt_ms) and dt_ms > 0.0):
        raise ValueError("dt_ms must be finite and positive")
    snapshots = {int(k): str(v) for k, v in (snapshot_steps or {}).items()}
    if any(k < 0 or k >= stream.n_steps for k in snapshots):
        raise ValueError("snapshot step outside sparse stream")
    block_map = {str(k): (int(v[0]), int(v[1])) for k, v in (blocks or {}).items()}
    for name, (lo, hi) in block_map.items():
        if not (0 <= lo < hi <= stream.n_steps):
            raise ValueError(f"invalid block {name!r}: {(lo, hi)}")

    states = {}
    for name, cfg in candidates.items():
        a_load, tau_ms = float(cfg["a_load"]), float(cfg["tau_ms"])
        h = int(cfg.get("h", 3))
        if not (np.isfinite(a_load) and a_load >= 0.0 and np.isfinite(tau_ms) and tau_ms > 0.0):
            raise ValueError(f"invalid load candidate {name!r}")
        u0 = cfg.get("u0")
        u = np.zeros(stream.n_cells) if u0 is None else np.asarray(u0, float).copy()
        if u.shape != (stream.n_cells,) or not np.all(np.isfinite(u)) or np.any(u < 0.0):
            raise ValueError(f"candidate {name!r} u0 must be finite, non-negative and per-cell")
        states[str(name)] = {
            "a_load": a_load,
            "tau_ms": tau_ms,
            "h": h,
            "u": u,
            "snapshots": {},
            "phi_sum": {b: np.zeros(stream.n_cells) for b in block_map},
            "spike_count": {b: np.zeros(stream.n_cells, dtype=np.int64) for b in block_map},
            "block_count": {b: 0 for b in block_map},
        }

    pos = 0
    spike_counts = np.zeros(stream.n_cells)
    previous_cells = np.empty(0, dtype=np.int64)
    for step in range(stream.n_steps):
        if previous_cells.size:
            spike_counts[previous_cells] = 0.0
        end = int(np.searchsorted(stream.steps, step, side="right"))
        cells = stream.cells[pos:end]
        if cells.size:
            spike_counts[cells] = 1.0
        previous_cells = cells
        pos = end

        active_blocks = [b for b, (lo, hi) in block_map.items() if lo <= step < hi]
        for state in states.values():
            u = state["u"]
            phi = pump_activation(u, state["h"])
            for block in active_blocks:
                state["phi_sum"][block] += phi
                state["spike_count"][block] += spike_counts.astype(np.int64)
                state["block_count"][block] += 1
            np.maximum(
                u + state["a_load"] * spike_counts - (dt_ms / state["tau_ms"]) * phi,
                0.0,
                out=u,
            )
            if step in snapshots:
                state["snapshots"][snapshots[step]] = u.copy()

    out = {}
    for name, state in states.items():
        means = {
            b: state["phi_sum"][b] / max(1, state["block_count"][b]) for b in block_map
        }
        out[name] = {
            "u_final": state["u"].copy(),
            "snapshots": state["snapshots"],
            "block_phi_mean": means,
            "block_spike_count": state["spike_count"],
            "a_load": state["a_load"],
            "tau_ms": state["tau_ms"],
            "h": state["h"],
            "spike_stream_sha256": stream.sha256,
        }
    return out


def lock_load_scales(
    *,
    r_hi_ref_hz: float,
    per_cell_rate_hz,
    tau_ms=(3000.0, 8000.0, 15000.0),
    target_activation=0.5,
):
    """Lock ``a_load`` and the common per-cell finite-equilibrium gate from a fresh rate field."""

    rates = np.asarray(per_cell_rate_hz, float)
    if rates.ndim != 1 or not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
        raise ValueError("per_cell_rate_hz must be a finite non-negative 1-D field")
    if not (np.isfinite(r_hi_ref_hz) and r_hi_ref_hz > 0.0):
        raise ValueError("r_hi_ref_hz must be finite and positive")
    if not (np.isfinite(target_activation) and 0.0 < target_activation < 1.0):
        raise ValueError("target_activation must lie in (0,1)")
    taus = tuple(float(t) for t in tau_ms)
    if not taus or any(not np.isfinite(t) or t <= 0.0 for t in taus):
        raise ValueError("tau_ms values must be finite and positive")

    q_star = target_activation * rates / float(r_hi_ref_hz)
    divergent = q_star >= 1.0
    q99 = float(np.quantile(q_star, 0.99)) if q_star.size else float("nan")
    admissible = bool(q_star.size and q99 < 0.90 and not np.any(divergent))
    h = hashlib.sha256(np.ascontiguousarray(q_star, dtype=np.float64).tobytes()).hexdigest()
    a_by_tau = {
        str(t): float(target_activation / ((r_hi_ref_hz / 1000.0) * t)) for t in taus
    }
    return {
        "r_hi_ref_hz": float(r_hi_ref_hz),
        "target_activation": float(target_activation),
        "a_load_by_tau_ms": a_by_tau,
        "q_star": q_star,
        "q_star_q99": q99,
        "q_star_max": float(np.max(q_star)) if q_star.size else float("nan"),
        "divergent_fraction": float(np.mean(divergent)) if q_star.size else float("nan"),
        "admissible": admissible,
        "q_star_sha256": h,
    }


def json_sanitize(value, *, max_array_elements=10000):
    """Recursively convert small provenance payloads to strict JSON-native values."""

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.size > int(max_array_elements):
            raise TypeError(f"array with {value.size} elements is too large for JSON")
        return [json_sanitize(v, max_array_elements=max_array_elements) for v in value.tolist()]
    if isinstance(value, dict):
        return {
            str(k): json_sanitize(v, max_array_elements=max_array_elements) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v, max_array_elements=max_array_elements) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


class AtomicStageBundle:
    """Publish a stage directory only after every required artifact exists.

    The temporary directory is a sibling of the final directory, so ``os.replace`` stays on one
    filesystem.  Downstream code must consume only the final directory; incomplete attempts never
    appear there.
    """

    def __init__(self, final_dir):
        self.final_dir = Path(final_dir)
        self._tmp_dir = None
        self._committed = False

    def __enter__(self):
        self.final_dir.parent.mkdir(parents=True, exist_ok=True)
        if self.final_dir.exists():
            raise FileExistsError(f"refusing to overwrite published bundle {self.final_dir}")
        self._tmp_dir = Path(
            tempfile.mkdtemp(prefix=f"{self.final_dir.name}.tmp-", dir=self.final_dir.parent)
        )
        return self

    def path(self, relative):
        if self._tmp_dir is None:
            raise RuntimeError("AtomicStageBundle must be entered before requesting paths")
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("bundle paths must stay relative to the stage directory")
        out = self._tmp_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    def commit(self, *, required):
        if self._tmp_dir is None:
            raise RuntimeError("AtomicStageBundle has not been entered")
        missing = [str(rel) for rel in required if not (self._tmp_dir / rel).is_file()]
        if missing:
            raise FileNotFoundError(f"atomic stage bundle missing required files: {missing}")
        os.replace(self._tmp_dir, self.final_dir)
        self._committed = True
        self._tmp_dir = None

    def __exit__(self, exc_type, exc, tb):
        if self._tmp_dir is not None:
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
            self._tmp_dir = None
        return False
