"""Observation-invariant continuous spatial OU modulation of E afferent rate."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


@dataclass(frozen=True)
class SpatialOUConfig:
    mode: str
    sigma_rate_per_ms: float
    tau_ms: float
    ell_mm: float
    update_interval_ms: float = 1.0
    grid_spacing_mm: float = 0.4
    seed: int = 0


class SpatialOUDrive:
    """Continuous zero-mean field with a spatial-permutation matched control.

    The latent field is an exact discrete-time OU process on a periodic regular
    grid. Gaussian smoothing sets its spatial scale; bilinear interpolation
    maps it to E neurons. ``permuted`` applies one frozen neuron permutation to
    the same local field values, preserving every update's value multiset while
    destroying adjacency. No contact or patient coordinates enter this class.
    """

    def __init__(self, positions_e, sheet_l_mm, dt_ms, config: SpatialOUConfig):
        positions = np.asarray(positions_e, float)
        if positions.ndim != 2 or positions.shape[1] != 2 or not len(positions):
            raise ValueError("positions_e must have shape (n_E, 2)")
        if config.mode not in {"local", "permuted"}:
            raise ValueError("spatial OU mode must be local or permuted")
        if config.sigma_rate_per_ms <= 0 or config.tau_ms <= 0:
            raise ValueError("spatial OU sigma and tau must be positive")
        if config.ell_mm <= 0 or config.update_interval_ms <= 0:
            raise ValueError("spatial OU length and update interval must be positive")
        if config.grid_spacing_mm <= 0 or sheet_l_mm <= 0 or dt_ms <= 0:
            raise ValueError("spatial OU geometry and dt must be positive")
        ratio = config.update_interval_ms / dt_ms
        if not np.isclose(ratio, round(ratio), atol=1e-10):
            raise ValueError("spatial OU update interval must lie on the SNN time grid")

        self.positions = positions
        self.sheet_l_mm = float(sheet_l_mm)
        self.dt_ms = float(dt_ms)
        self.config = config
        self.grid_n = max(4, int(round(self.sheet_l_mm / config.grid_spacing_mm)))
        self.grid_spacing_mm = self.sheet_l_mm / self.grid_n
        self.update_steps = int(round(ratio))
        self._next_step = self.update_steps
        self._last_step = -1
        self._rng = np.random.default_rng(int(config.seed))
        permutation_rng = np.random.default_rng(int(config.seed) + 1_000_003)
        self._permutation = permutation_rng.permutation(len(positions))

        coordinates = np.mod(positions, self.sheet_l_mm) / self.grid_spacing_mm
        self._ix0 = np.floor(coordinates[:, 0]).astype(int) % self.grid_n
        self._iy0 = np.floor(coordinates[:, 1]).astype(int) % self.grid_n
        self._ix1 = (self._ix0 + 1) % self.grid_n
        self._iy1 = (self._iy0 + 1) % self.grid_n
        self._wx = coordinates[:, 0] - np.floor(coordinates[:, 0])
        self._wy = coordinates[:, 1] - np.floor(coordinates[:, 1])

        sigma_grid = float(config.ell_mm) / self.grid_spacing_mm
        impulse = np.zeros((self.grid_n, self.grid_n), float)
        impulse[0, 0] = 1.0
        kernel = gaussian_filter(impulse, sigma=sigma_grid, mode="wrap")
        kernel -= kernel.mean()
        self._innovation_gain = float(np.sqrt(np.sum(kernel * kernel)))
        if self._innovation_gain <= 1e-12:
            raise ValueError("spatial OU smoothing removed all spatial variation")
        self._sigma_grid = sigma_grid
        self._ou_a = float(np.exp(-config.update_interval_ms / config.tau_ms))
        self._ou_b = float(np.sqrt(1.0 - self._ou_a * self._ou_a))
        self._state = config.sigma_rate_per_ms * self._innovation()
        self._cached = self._map_state()
        self._times, self._means, self._stds = [], [], []
        self._maxima, self._minima = [], []
        self._argmax_x, self._argmax_y = [], []
        self._record(0.0)

    def _innovation(self):
        white = self._rng.standard_normal((self.grid_n, self.grid_n))
        smooth = gaussian_filter(white, sigma=self._sigma_grid, mode="wrap")
        smooth -= smooth.mean()
        return smooth / self._innovation_gain

    def _map_state(self):
        x0, x1, y0, y1 = self._ix0, self._ix1, self._iy0, self._iy1
        wx, wy = self._wx, self._wy
        values = (
            (1.0 - wx) * (1.0 - wy) * self._state[x0, y0]
            + wx * (1.0 - wy) * self._state[x1, y0]
            + (1.0 - wx) * wy * self._state[x0, y1]
            + wx * wy * self._state[x1, y1]
        )
        values -= values.mean()
        if self.config.mode == "permuted":
            values = values[self._permutation]
        return values

    def _record(self, time_ms):
        values = self._cached
        index = int(np.argmax(values))
        self._times.append(float(time_ms))
        self._means.append(float(values.mean()))
        self._stds.append(float(values.std()))
        self._maxima.append(float(values[index]))
        self._minima.append(float(values.min()))
        self._argmax_x.append(float(self.positions[index, 0]))
        self._argmax_y.append(float(self.positions[index, 1]))

    def step(self, time_ms):
        step = int(round(float(time_ms) / self.dt_ms))
        if step < self._last_step:
            raise ValueError("spatial OU drive must be called in increasing time order")
        while step >= self._next_step:
            self._state = (
                self._ou_a * self._state
                + self._ou_b * self.config.sigma_rate_per_ms * self._innovation()
            )
            self._cached = self._map_state()
            self._record(self._next_step * self.dt_ms)
            self._next_step += self.update_steps
        self._last_step = step
        return self._cached

    def trace_arrays(self):
        return {
            "time_ms": np.asarray(self._times, np.float32),
            "spatial_mean_rate_per_ms": np.asarray(self._means, np.float32),
            "spatial_sd_rate_per_ms": np.asarray(self._stds, np.float32),
            "maximum_rate_per_ms": np.asarray(self._maxima, np.float32),
            "minimum_rate_per_ms": np.asarray(self._minima, np.float32),
            "argmax_x_mm": np.asarray(self._argmax_x, np.float32),
            "argmax_y_mm": np.asarray(self._argmax_y, np.float32),
        }
