"""Pure observation and spatial-readout helpers for FCXR-LC6A trajectories."""

from __future__ import annotations

import math

import numpy as np


class NaturalCurrentObserver:
    """Sample actual E-cell membrane contributions without changing the trajectory."""

    def __init__(self, *, dt_ms: float, sample_dt_ms: float):
        ratio = float(sample_dt_ms) / float(dt_ms)
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio)):
            raise ValueError("sample_dt_ms must be an integer multiple of dt_ms")
        self.dt_ms = float(dt_ms)
        self.sample_dt_ms = float(sample_dt_ms)
        self.sample_every = int(round(ratio))
        self._rows = []

    def sample(self, step, voltage, drive, g_rel, _g_rev, slow):
        if int(step) % self.sample_every:
            return
        voltage = np.asarray(voltage, float)
        drive = np.asarray(drive, float)
        g_rel = np.asarray(g_rel, float)
        g_i = np.asarray(slow._gI_last_E, float)
        if not (voltage.shape == drive.shape == g_rel.shape == g_i.shape):
            raise ValueError("natural current observer received incompatible E-cell arrays")
        g_e = g_rel - g_i
        if np.any(g_e < -1e-10):
            raise RuntimeError("inferred excitatory conductance is negative")
        excitatory = drive + np.maximum(g_e, 0.0) * (float(slow.cfg.E_E) - voltage)
        inhibitory_signed = g_i * (float(slow.cfg.e_gaba) - voltage)
        inhibitory = np.maximum(-inhibitory_signed, 0.0)
        self._rows.append((
            float(step) * self.dt_ms,
            float(np.mean(excitatory)),
            float(np.mean(inhibitory)),
            float(np.mean(excitatory + inhibitory_signed)),
            float(np.mean(np.maximum(g_e, 0.0))),
            float(np.mean(g_i)),
        ))

    def arrays(self) -> dict:
        values = np.asarray(self._rows, float)
        if values.size == 0:
            raise RuntimeError("natural current observer collected no samples")
        return {
            "current_time_ms": values[:, 0].astype(np.float32),
            "F_E_mean": values[:, 1].astype(np.float32),
            "F_I_mean": values[:, 2].astype(np.float32),
            "I_syn_signed_mean": values[:, 3].astype(np.float32),
            "g_E_mean": values[:, 4].astype(np.float32),
            "g_I_mean": values[:, 5].astype(np.float32),
        }


def observation_decision(
    *, total_ms: float, onset_ms: float | None, n_returning_ied: int,
    c0_ied_to_onset: int | None, saturated_contiguous_1s: bool,
    base_end_ms: float = 50000.0, post_onset_ms: float = 12000.0,
    hard_cap_ms: float = 65000.0, ied_multiplier: float = 1.5,
) -> dict:
    """Event-aligned continuation rule from the locked LC6A protocol."""

    total_ms = float(total_ms)
    if saturated_contiguous_1s:
        return {"continue": False, "reason": "REGISTERED_SATURATION_1S", "right_censored": False}
    if onset_ms is not None:
        target = float(onset_ms) + float(post_onset_ms)
        if total_ms >= target:
            return {"continue": False, "reason": "ONSET_PLUS_12S_OBSERVED", "right_censored": False}
        if total_ms >= float(hard_cap_ms):
            return {"continue": False, "reason": "HARD_CAP_AFTER_LATE_ONSET", "right_censored": True}
        return {"continue": True, "reason": "POST_ONSET_OBSERVATION_INCOMPLETE", "right_censored": False}
    if total_ms < float(base_end_ms):
        return {"continue": True, "reason": "NO_ONSET_BEFORE_MINIMUM", "right_censored": False}
    required = None if c0_ied_to_onset is None else int(math.ceil(
        float(ied_multiplier) * int(c0_ied_to_onset)
    ))
    if required is not None and int(n_returning_ied) >= required:
        return {
            "continue": False, "reason": "NO_ONSET_SUFFICIENT_IED_EXPOSURE",
            "right_censored": False, "required_ied_exposure": required,
        }
    if total_ms >= float(hard_cap_ms):
        return {
            "continue": False, "reason": "NO_ONSET_HARD_CAP_INSUFFICIENT_IED_EXPOSURE",
            "right_censored": True, "required_ied_exposure": required,
        }
    return {
        "continue": True, "reason": "EXTEND_FOR_IED_EXPOSURE",
        "right_censored": False, "required_ied_exposure": required,
    }


def cell_spatial_bins(positions, *, sheet_size_mm: float, n_bins_axis: int) -> tuple[np.ndarray, np.ndarray]:
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (cells, 2)")
    ij = np.floor(positions / float(sheet_size_mm) * int(n_bins_axis)).astype(int)
    ij = np.clip(ij, 0, int(n_bins_axis) - 1)
    ids = ij[:, 0] * int(n_bins_axis) + ij[:, 1]
    occupancy = np.bincount(ids, minlength=int(n_bins_axis) ** 2)
    return ids.astype(np.int32), occupancy.astype(np.int32)


def spatial_rate_maps(
    spike_steps, spike_cells, cell_bins, occupancy, *,
    n_steps: int, dt_ms: float, window_ms: float,
) -> np.ndarray:
    """Window x coarse-bin E-cell firing rates in Hz."""

    steps_per_window = int(round(float(window_ms) / float(dt_ms)))
    if steps_per_window <= 0 or n_steps % steps_per_window:
        raise ValueError("trajectory must contain an integer number of spatial windows")
    n_windows = int(n_steps // steps_per_window)
    n_bins = int(len(occupancy))
    spike_steps = np.asarray(spike_steps, np.int64)
    spike_cells = np.asarray(spike_cells, np.int64)
    time_bin = spike_steps // steps_per_window
    space_bin = np.asarray(cell_bins, np.int64)[spike_cells]
    flat = time_bin * n_bins + space_bin
    counts = np.bincount(flat, minlength=n_windows * n_bins).reshape(n_windows, n_bins)
    denominator = np.asarray(occupancy, float)[None, :] * (float(window_ms) / 1000.0)
    return np.divide(
        counts, denominator, out=np.full_like(counts, np.nan, dtype=float),
        where=denominator > 0,
    )


def per_second_cell_rates(
    spike_steps, spike_cells, *, n_steps: int, n_cells: int, dt_ms: float,
) -> np.ndarray:
    steps_per_second = int(round(1000.0 / float(dt_ms)))
    if n_steps % steps_per_second:
        raise ValueError("trajectory must contain an integer number of seconds")
    second = np.asarray(spike_steps, np.int64) // steps_per_second
    cell = np.asarray(spike_cells, np.int64)
    flat = second * int(n_cells) + cell
    return np.bincount(
        flat, minlength=(n_steps // steps_per_second) * int(n_cells),
    ).reshape(n_steps // steps_per_second, int(n_cells)).astype(float)


def local_saturation_readout(cell_rates_hz, *, refractory_ceiling_hz: float, fraction_gate: float = .05) -> dict:
    rates = np.asarray(cell_rates_hz, float)
    near = rates >= .9 * float(refractory_ceiling_hz)
    fraction = near.mean(axis=1)
    return {
        "max_near_refractory_fraction": float(fraction.max(initial=0.0)),
        "time_fraction_above_fraction_gate": float(np.mean(fraction > float(fraction_gate))),
        "per_second_near_refractory_fraction": fraction.tolist(),
        "near_refractory_rate_hz": .9 * float(refractory_ceiling_hz),
    }


def coarse_field_mean(field, cell_bins, occupancy) -> np.ndarray:
    field = np.asarray(field, float)
    total = np.bincount(cell_bins, weights=field, minlength=len(occupancy))
    return np.divide(
        total, occupancy, out=np.full_like(total, np.nan, dtype=float), where=occupancy > 0,
    )


def linear_slope(values, *, dt_s: float) -> float:
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    x = np.arange(values.size, dtype=float)[finite] * float(dt_s)
    return float(np.polyfit(x, values[finite], 1)[0])


def spatial_map_persistence(rate_maps) -> dict:
    maps = np.asarray(rate_maps, float)
    correlations = []
    for first, second in zip(maps[:-1], maps[1:]):
        finite = np.isfinite(first) & np.isfinite(second)
        if np.count_nonzero(finite) < 3 or np.std(first[finite]) == 0 or np.std(second[finite]) == 0:
            continue
        correlations.append(float(np.corrcoef(first[finite], second[finite])[0, 1]))
    return {
        "median_consecutive_correlation": (
            float(np.median(correlations)) if correlations else float("nan")
        ),
        "n_pairs": len(correlations),
    }


def largest_connected_component_area(active, occupancy, *, bin_area_mm2: float) -> float:
    """Four-neighbour component area on the frozen square coarse grid."""

    active = np.asarray(active, bool)
    occupancy = np.asarray(occupancy)
    if active.ndim != 1 or occupancy.shape != active.shape:
        raise ValueError("active and occupancy must be aligned flat maps")
    side = int(round(np.sqrt(active.size)))
    if side * side != active.size:
        raise ValueError("coarse map must be square")
    use = (active & (occupancy > 0)).reshape(side, side)
    seen = np.zeros_like(use, bool)
    largest = 0
    for i, j in np.argwhere(use):
        if seen[i, j]:
            continue
        stack = [(int(i), int(j))]
        seen[i, j] = True
        size = 0
        while stack:
            a, b = stack.pop()
            size += 1
            for c, d in ((a - 1, b), (a + 1, b), (a, b - 1), (a, b + 1)):
                if 0 <= c < side and 0 <= d < side and use[c, d] and not seen[c, d]:
                    seen[c, d] = True
                    stack.append((c, d))
        largest = max(largest, size)
    return float(largest) * float(bin_area_mm2)


def _event_window_indices(event, *, window_ms: float, n_windows: int) -> np.ndarray:
    starts = np.arange(int(n_windows), dtype=float) * float(window_ms)
    ends = starts + float(window_ms)
    return np.flatnonzero(
        (starts <= float(event["t_off"]))
        & (ends > float(event["t_on"]))
    )


def calibrate_local_classifier(
    rate_maps_100ms, occupancy, returned_events, *, onset_ms: float,
    sheet_size_mm: float, rate_quantile: float = .995, area_quantile: float = .99,
    window_ms: float = 100.0, persistence_ms: float = 500.0,
) -> dict:
    """Freeze the C0 IED-tail rate and connected-area thresholds before Q arms."""

    maps = np.asarray(rate_maps_100ms, float)
    occupancy = np.asarray(occupancy)
    if maps.ndim != 2 or maps.shape[1] != occupancy.size:
        raise ValueError("rate maps and occupancy are incompatible")
    events = [
        event for event in returned_events
        if bool(event.get("returned", False)) and float(event["t_on"]) < float(onset_ms)
    ]
    if not events:
        raise RuntimeError("C0 provides no pre-onset returning IED for local-classifier lock")
    indices = np.unique(np.concatenate([
        _event_window_indices(event, window_ms=window_ms, n_windows=maps.shape[0])
        for event in events
    ]))
    valid_bins = occupancy > 0
    sampled = maps[indices][:, valid_bins]
    finite = sampled[np.isfinite(sampled)]
    if finite.size == 0:
        raise RuntimeError("C0 IED local-rate distribution is empty")
    rate_threshold = float(np.quantile(finite, float(rate_quantile)))
    side = int(round(np.sqrt(occupancy.size)))
    bin_area = (float(sheet_size_mm) / side) ** 2
    event_maxima = []
    for event in events:
        use = _event_window_indices(event, window_ms=window_ms, n_windows=maps.shape[0])
        event_maxima.append(max(
            largest_connected_component_area(
                row >= rate_threshold, occupancy, bin_area_mm2=bin_area,
            ) for row in maps[use]
        ))
    area_threshold = float(np.quantile(event_maxima, float(area_quantile)))
    return {
        "rate_threshold_hz": rate_threshold,
        "rate_quantile": float(rate_quantile),
        "component_area_threshold_mm2": area_threshold,
        "area_quantile": float(area_quantile),
        "window_ms": float(window_ms),
        "persistence_ms": float(persistence_ms),
        "persistence_windows": int(round(float(persistence_ms) / float(window_ms))),
        "n_pre_onset_returning_events": int(len(events)),
        "n_ied_windows": int(indices.size),
        "event_component_area_maxima_mm2": [float(value) for value in event_maxima],
        "bin_area_mm2": float(bin_area),
    }


def apply_local_classifier(rate_maps_100ms, occupancy, lock) -> dict:
    maps = np.asarray(rate_maps_100ms, float)
    occupancy = np.asarray(occupancy)
    if maps.ndim != 2 or maps.shape[1] != occupancy.size:
        raise ValueError("rate maps and occupancy are incompatible")
    rate_threshold = float(lock["rate_threshold_hz"])
    area_threshold = float(lock["component_area_threshold_mm2"])
    bin_area = float(lock["bin_area_mm2"])
    q95 = np.nanquantile(maps[:, occupancy > 0], .95, axis=1)
    areas = np.asarray([
        largest_connected_component_area(
            row >= rate_threshold, occupancy, bin_area_mm2=bin_area,
        ) for row in maps
    ])
    active = (q95 > rate_threshold) & (areas > area_threshold)
    need = int(lock["persistence_windows"])
    hits = np.convolve(active.astype(int), np.ones(need, dtype=int), mode="valid") if need else np.array([])
    starts = np.flatnonzero(hits >= need)
    onset = None if not starts.size else float(starts[0] * float(lock["window_ms"]))
    return {
        "local_onset_ms": onset,
        "local_high_occupancy": float(np.mean(active)),
        "max_component_area_mm2": float(np.max(areas, initial=0.0)),
        "component_area_mm2": areas.tolist(),
        "local_rate_q95_hz": q95.tolist(),
        "max_local_rate_q95_hz": float(np.nanmax(q95)),
        "active_windows": active.tolist(),
    }
