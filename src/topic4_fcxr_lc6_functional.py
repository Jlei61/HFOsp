"""Pure readout helpers for the LC6A paired short functional assay."""

from __future__ import annotations

import hashlib

import numpy as np

from src.topic4_fcxr_lc6_trajectory import cell_spatial_bins, coarse_field_mean


COMPONENTS = ("F_E", "F_I", "I_syn_signed", "g_E", "g_I", "V")


def local_patch_pattern(positions, center, *, radius_mm: float) -> np.ndarray:
    positions = np.asarray(positions, float)
    center = np.asarray(center, float)
    if positions.ndim != 2 or positions.shape[1] != 2 or center.shape != (2,):
        raise ValueError("positions/center have incompatible shapes")
    if not np.isfinite(radius_mm) or float(radius_mm) <= 0.0:
        raise ValueError("radius_mm must be positive")
    pattern = (np.linalg.norm(positions - center, axis=1) <= float(radius_mm)).astype(float)
    if not np.any(pattern):
        raise RuntimeError("functional patch contains no E cells")
    return pattern


def locked_patch_centers(substrate, *, patch_radius_mm: float, core_radius_mm: float) -> dict:
    """Return graph/outcome-independent diagnostic locations on the frozen patient axis."""

    src = np.asarray(substrate["src_xy"], float)
    snk = np.asarray(substrate["snk_xy"], float)
    axis = np.asarray(substrate["axis_unit"], float)
    centers = {
        "core_adjacent": src + (float(core_radius_mm) + float(patch_radius_mm)) * axis,
        "neutral_axis": 0.5 * (src + snk),
    }
    sheet = float(substrate["L"])
    for name, center in centers.items():
        if np.any(center < patch_radius_mm) or np.any(center > sheet - patch_radius_mm):
            raise RuntimeError(f"locked {name} patch crosses the sheet boundary")
    if min(
        np.linalg.norm(centers["neutral_axis"] - src),
        np.linalg.norm(centers["neutral_axis"] - snk),
    ) <= float(core_radius_mm) + float(patch_radius_mm):
        raise RuntimeError("neutral-axis patch overlaps a low-threshold core")
    return {key: value.tolist() for key, value in centers.items()}


class FunctionalResponseRecorder:
    """Reduce full E-cell membrane terms into fixed windows without changing state."""

    def __init__(
        self,
        positions,
        *,
        patch_center,
        axis_unit,
        dt_ms: float,
        window_edges_ms=(0.0, 50.0, 150.0, 300.0),
        axis_edges_mm=None,
        transverse_half_width_mm: float = 0.75,
        sheet_size_mm: float = 20.0,
        n_map_bins_axis: int = 32,
    ):
        self.positions = np.asarray(positions, float)
        self.center = np.asarray(patch_center, float)
        self.axis = np.asarray(axis_unit, float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.perp = np.asarray([-self.axis[1], self.axis[0]])
        self.dt_ms = float(dt_ms)
        self.window_edges_ms = np.asarray(window_edges_ms, float)
        if axis_edges_mm is None:
            axis_edges_mm = np.arange(-6.0, 6.0001, 0.25)
        self.axis_edges_mm = np.asarray(axis_edges_mm, float)
        self.transverse_half_width_mm = float(transverse_half_width_mm)
        self.n_cells = int(self.positions.shape[0])
        self.n_windows = int(self.window_edges_ms.size - 1)
        self.n_ms = int(round(self.window_edges_ms[-1]))
        self.term_sums = np.zeros(
            (self.n_windows, len(COMPONENTS), self.n_cells), dtype=np.float64,
        )
        self.term_counts = np.zeros(self.n_windows, dtype=np.int64)
        self.spike_counts = np.zeros((self.n_windows, self.n_cells), dtype=np.int32)
        self.active_1ms = np.zeros((self.n_ms, self.n_cells), dtype=bool)
        self.signed_sum_1ms = np.zeros((self.n_ms, self.n_cells), dtype=np.float32)
        self.signed_count_1ms = np.zeros(self.n_ms, dtype=np.int32)
        self.map_bins, self.map_occupancy = cell_spatial_bins(
            self.positions, sheet_size_mm=sheet_size_mm, n_bins_axis=n_map_bins_axis,
        )
        relative = self.positions - self.center
        self.parallel_mm = relative @ self.axis
        self.perpendicular_mm = relative @ self.perp
        self.axis_bin = np.digitize(self.parallel_mm, self.axis_edges_mm) - 1
        self.axis_valid = (
            (self.axis_bin >= 0)
            & (self.axis_bin < self.axis_edges_mm.size - 1)
            & (np.abs(self.perpendicular_mm) <= self.transverse_half_width_mm)
        )

    def _window_index(self, step: int) -> int | None:
        t_ms = float(step) * self.dt_ms
        index = int(np.searchsorted(self.window_edges_ms, t_ms, side="right") - 1)
        return index if 0 <= index < self.n_windows else None

    def sample_membrane(self, step, voltage, drive, g_rel, _g_rev, slow):
        index = self._window_index(step)
        if index is None:
            return
        voltage = np.asarray(voltage, float)
        drive = np.asarray(drive, float)
        g_rel = np.asarray(g_rel, float)
        g_i = np.asarray(slow._gI_last_E, float)
        if slow.cfg.use_m or slow.cfg.m_frozen_E is not None or slow.cfg.use_pump:
            raise RuntimeError("LC6A functional recorder requires M=U=0")
        g_e = g_rel - g_i
        if np.any(g_e < -1e-12):
            raise RuntimeError("inferred excitatory conductance is negative")
        excitatory_signed = drive + g_e * (float(slow.cfg.E_E) - voltage)
        inhibitory_signed = g_i * (float(slow.cfg.e_gaba) - voltage)
        values = (
            np.maximum(excitatory_signed, 0.0),
            np.maximum(-inhibitory_signed, 0.0),
            excitatory_signed + inhibitory_signed,
            np.maximum(g_e, 0.0),
            g_i,
            voltage,
        )
        for component, value in enumerate(values):
            self.term_sums[index, component] += value
        self.term_counts[index] += 1
        ms = int(np.floor(float(step) * self.dt_ms))
        if 0 <= ms < self.n_ms:
            self.signed_sum_1ms[ms] += values[2].astype(np.float32)
            self.signed_count_1ms[ms] += 1

    def sample_spikes(self, step, cells):
        index = self._window_index(step)
        cells = np.asarray(cells, np.int64)
        if index is not None:
            self.spike_counts[index, cells] += 1
        ms = int(np.floor(float(step) * self.dt_ms))
        if 0 <= ms < self.n_ms:
            self.active_1ms[ms, cells] = True

    def _axis_mean(self, cell_values) -> np.ndarray:
        values = np.asarray(cell_values, float)
        n_bins = self.axis_edges_mm.size - 1
        ids = self.axis_bin[self.axis_valid]
        counts = np.bincount(ids, minlength=n_bins)
        total = np.bincount(ids, weights=values[self.axis_valid], minlength=n_bins)
        return np.divide(
            total, counts, out=np.full(n_bins, np.nan), where=counts > 0,
        )

    def finalize(self) -> dict:
        if np.any(self.term_counts == 0):
            raise RuntimeError("one or more registered functional windows are empty")
        means = self.term_sums / self.term_counts[:, None, None]
        durations_s = np.diff(self.window_edges_ms) / 1000.0
        rates = self.spike_counts / durations_s[:, None]
        signed_1ms = np.divide(
            self.signed_sum_1ms,
            self.signed_count_1ms[:, None],
            out=np.full_like(self.signed_sum_1ms, np.nan, dtype=np.float32),
            where=self.signed_count_1ms[:, None] > 0,
        )
        return {
            "components": means,
            "cell_rates_hz": rates,
            "axis_components": np.stack([
                np.stack([self._axis_mean(field) for field in window]) for window in means
            ]),
            "axis_rates_hz": np.stack([self._axis_mean(row) for row in rates]),
            "axis_signed_1ms": np.stack([self._axis_mean(row) for row in signed_1ms]),
            "map_components": np.stack([
                np.stack([
                    coarse_field_mean(field, self.map_bins, self.map_occupancy)
                    for field in window
                ]) for window in means
            ]),
            "map_rates_hz": np.stack([
                coarse_field_mean(row, self.map_bins, self.map_occupancy) for row in rates
            ]),
            "active_fraction_1ms": self.active_1ms.mean(axis=1),
            "axis_edges_mm": self.axis_edges_mm.copy(),
            "window_edges_ms": self.window_edges_ms.copy(),
            "map_occupancy": self.map_occupancy.copy(),
        }


def _first_negative_distance(curve, centers, *, direction: str, center_exclusion_mm=0.25):
    curve = np.asarray(curve, float)
    centers = np.asarray(centers, float)
    if direction == "forward":
        select = centers >= float(center_exclusion_mm)
        order = np.argsort(centers[select])
    elif direction == "backward":
        select = centers <= -float(center_exclusion_mm)
        order = np.argsort(np.abs(centers[select]))
    else:
        raise ValueError("direction must be forward or backward")
    x, y = centers[select][order], curve[select][order]
    finite = np.isfinite(y)
    hit = np.flatnonzero(finite & (y < 0.0))
    return None if not hit.size else float(x[hit[0]])


def _latency(trace, centers, region, *, fraction=0.1):
    trace = np.asarray(trace, float)
    centers = np.asarray(centers, float)
    select = region(centers)
    region_trace = np.nanmean(trace[:, select], axis=1)
    peak = float(np.nanmax(np.abs(region_trace)))
    if not np.isfinite(peak) or peak <= 0.0:
        return None
    hits = np.flatnonzero(np.abs(region_trace) >= float(fraction) * peak)
    return None if not hits.size else float(hits[0])


def paired_response(sham: dict, probe: dict) -> dict:
    for key in ("window_edges_ms", "axis_edges_mm"):
        np.testing.assert_array_equal(sham[key], probe[key])
    delta_components = probe["components"] - sham["components"]
    delta_axis = probe["axis_components"] - sham["axis_components"]
    delta_axis_rate = probe["axis_rates_hz"] - sham["axis_rates_hz"]
    delta_signed_1ms = probe["axis_signed_1ms"] - sham["axis_signed_1ms"]
    centers = 0.5 * (probe["axis_edges_mm"][:-1] + probe["axis_edges_mm"][1:])
    net_index = COMPONENTS.index("I_syn_signed")
    window_zero_crossings = []
    for curve in delta_axis[:, net_index]:
        window_zero_crossings.append({
            "forward_mm": _first_negative_distance(curve, centers, direction="forward"),
            "backward_mm": _first_negative_distance(curve, centers, direction="backward"),
        })
    regions = {
        "center": lambda x: np.abs(x) <= 0.5,
        "forward": lambda x: (x > 0.5) & (x <= 4.0),
        "backward": lambda x: (x < -0.5) & (x >= -4.0),
    }
    return {
        "delta_components": delta_components,
        "delta_axis_components": delta_axis,
        "delta_axis_rate_hz": delta_axis_rate,
        "delta_map_components": probe["map_components"] - sham["map_components"],
        "delta_map_rate_hz": probe["map_rates_hz"] - sham["map_rates_hz"],
        "delta_axis_signed_1ms": delta_signed_1ms,
        "window_zero_crossings": window_zero_crossings,
        "latency_ms": {
            name: _latency(delta_signed_1ms, centers, selector)
            for name, selector in regions.items()
        },
        "max_active_fraction_1ms_sham": float(np.max(sham["active_fraction_1ms"])),
        "max_active_fraction_1ms_probe": float(np.max(probe["active_fraction_1ms"])),
        "excess_spikes": int(
            np.rint(np.sum(probe["cell_rates_hz"] * np.diff(probe["window_edges_ms"])[:, None] / 1000.0)
                    - np.sum(sham["cell_rates_hz"] * np.diff(sham["window_edges_ms"])[:, None] / 1000.0))
        ),
    }


def array_sha256(array) -> str:
    value = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(np.asarray(value.shape, np.int64).tobytes())
    h.update(value.view(np.uint8))
    return h.hexdigest()
