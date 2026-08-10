"""Pure readout helpers for rev9 paired small-kick experiments."""
from __future__ import annotations

import numpy as np


def event_window_overlap(events, pulse_end_ms, windows_after_ms):
    """Return which post-pulse windows overlap any detector event."""
    windows = np.asarray(windows_after_ms, float)
    if windows.ndim != 2 or windows.shape[1] != 2:
        raise ValueError("windows_after_ms must have shape (n, 2)")
    overlap = np.zeros(len(windows), bool)
    for event in events:
        event_start = float(event["t_on"])
        event_stop = float(event["t_off"])
        if event_stop <= event_start:
            raise ValueError("event stop must be after event start")
        starts = float(pulse_end_ms) + windows[:, 0]
        stops = float(pulse_end_ms) + windows[:, 1]
        overlap |= (event_start < stops) & (event_stop > starts)
    return overlap


def fit_response_slope(amplitudes, responses):
    """OLS response-amplitude slope with intercept and a descriptive R squared."""
    amplitudes = np.asarray(amplitudes, float)
    responses = np.asarray(responses, float)
    valid = np.isfinite(amplitudes) & np.isfinite(responses)
    if valid.sum() < 2 or np.ptp(amplitudes[valid]) <= 0.0:
        return dict(slope=None, intercept=None, r2=None, n=int(valid.sum()))
    design = np.column_stack((amplitudes[valid], np.ones(valid.sum())))
    slope, intercept = np.linalg.lstsq(design, responses[valid], rcond=None)[0]
    fitted = slope * amplitudes[valid] + intercept
    residual = float(np.sum((responses[valid] - fitted) ** 2))
    total = float(np.sum((responses[valid] - responses[valid].mean()) ** 2))
    r2 = 1.0 if total <= 1e-15 and residual <= 1e-15 else (
        None if total <= 1e-15 else 1.0 - residual / total)
    return dict(
        slope=float(slope), intercept=float(intercept),
        r2=None if r2 is None else float(r2), n=int(valid.sum()))


def _spatial_maps(values, positions, *, L, n_bins):
    edges = np.linspace(0.0, float(L), int(n_bins) + 1)
    cell_count, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(edges, edges))
    mass, _, _ = np.histogram2d(
        positions[:, 1], positions[:, 0], bins=(edges, edges), weights=values)
    per_cell = np.divide(
        mass, cell_count, out=np.zeros_like(mass), where=cell_count > 0.0)
    return per_cell, edges


def paired_spike_response(kick_spikes, sham_spikes, positions, origin, axis_unit,
                          *, dt, pulse_end_ms, windows_after_ms,
                          source_radius_mm, L, spatial_bins_per_axis):
    """Measure signed and positive kick-sham response in fixed post-pulse windows."""
    kick = np.asarray(kick_spikes, bool)
    sham = np.asarray(sham_spikes, bool)
    positions = np.asarray(positions, float)
    origin = np.asarray(origin, float)
    axis = np.asarray(axis_unit, float)
    if kick.ndim != 2 or sham.ndim != 2 or kick.shape[1] != len(positions):
        raise ValueError("spike arrays and positions do not align")
    axis = axis / np.linalg.norm(axis)
    transverse = np.asarray([-axis[1], axis[0]])
    relative = positions - origin
    radial = np.linalg.norm(relative, axis=1)
    source = radial <= float(source_radius_mm)
    if not source.any() or source.all():
        raise ValueError("source disk must contain a strict subset of E cells")

    rows = []
    required_stop = 0
    for start_ms, stop_ms in windows_after_ms:
        start = int(round((float(pulse_end_ms) + float(start_ms)) / float(dt)))
        stop = int(round((float(pulse_end_ms) + float(stop_ms)) / float(dt)))
        required_stop = max(required_stop, stop)
        if start < 0 or stop <= start or stop > len(kick) or stop > len(sham):
            rows.append(dict(status="truncated", start_step=start, stop_step=stop))
            continue
        signed = (kick[start:stop].sum(axis=0).astype(float)
                  - sham[start:stop].sum(axis=0).astype(float))
        positive = np.clip(signed, 0.0, np.inf)
        positive_total = float(positive.sum())
        if positive_total > 0.0:
            order = np.argsort(radial)
            cumulative = np.cumsum(positive[order]) / positive_total
            r50 = float(radial[order][np.searchsorted(cumulative, 0.5)])
            r90 = float(radial[order][np.searchsorted(cumulative, 0.9)])
            along = relative @ axis
            across = relative @ transverse
            mean_along = float(np.sum(positive * along) / positive_total)
            mean_across = float(np.sum(positive * across) / positive_total)
            variance_along = float(
                np.sum(positive * (along - mean_along) ** 2) / positive_total)
            variance_across = float(
                np.sum(positive * (across - mean_across) ** 2) / positive_total)
            axis_ratio = variance_along / max(variance_across, 1e-12)
        else:
            r50 = r90 = axis_ratio = None
        signed_map, edges = _spatial_maps(
            signed, positions, L=L, n_bins=spatial_bins_per_axis)
        positive_map, _ = _spatial_maps(
            positive, positions, L=L, n_bins=spatial_bins_per_axis)
        rows.append(dict(
            status="ok", start_step=start, stop_step=stop,
            source_signed_per_cell=float(signed[source].sum() / source.sum()),
            downstream_signed_per_cell=float(signed[~source].sum() / (~source).sum()),
            source_positive_per_cell=float(positive[source].sum() / source.sum()),
            downstream_positive_per_cell=float(
                positive[~source].sum() / (~source).sum()),
            positive_mass=positive_total,
            r50_mm=r50, r90_mm=r90, axis_variance_ratio=axis_ratio,
            signed_map_per_cell=signed_map,
            positive_map_per_cell=positive_map,
            spatial_edges=edges,
        ))
    return dict(
        status="ok" if required_stop <= len(kick) and required_stop <= len(sham)
        else "truncated",
        n_source=int(source.sum()), n_downstream=int((~source).sum()),
        windows=rows,
    )
