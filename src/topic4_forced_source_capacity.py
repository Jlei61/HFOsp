"""Pure source-selection and paired-response helpers for rev9-L L1."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def select_source_indices(positions, source, *, n_cells,
                          component_contribution=None):
    """Select a deterministic, equal-count E-neuron source packet."""
    positions = np.asarray(positions, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (E neuron, 2)")
    n_cells = int(n_cells)
    if n_cells < 1 or n_cells > len(positions):
        raise ValueError("n_cells must lie within the E population")
    kind = source.get("kind")
    if kind == "component":
        contribution = np.asarray(component_contribution, float)
        component = int(source["component_1based"]) - 1
        if (contribution.ndim != 2 or contribution.shape[0] != len(positions)
                or component < 0 or component >= contribution.shape[1]):
            raise ValueError("component contribution does not match source contract")
        score = contribution[:, component]
        if not np.isfinite(score).all() or np.max(score) <= 0.0:
            raise ValueError("component contribution must be finite and positive")
        order = np.argsort(-score, kind="stable")
    elif kind == "matched_off_field":
        center = np.asarray(source["xy_mm"], float)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("control source needs a finite xy_mm center")
        distance = np.linalg.norm(positions - center, axis=1)
        order = np.argsort(distance, kind="stable")
    else:
        raise ValueError(f"unknown source kind: {kind}")
    return np.sort(np.asarray(order[:n_cells], int))


def paired_excess_geometry(
        forced_spikes, sham_spikes, positions, source_mask, *,
        dt_ms, start_ms, end_ms, source_center):
    """Spatial geometry of positive forced-minus-sham E spike counts."""
    forced = np.asarray(forced_spikes, bool)
    sham = np.asarray(sham_spikes, bool)
    positions = np.asarray(positions, float)
    source_mask = np.asarray(source_mask, bool)
    if forced.shape != sham.shape or forced.ndim != 2:
        raise ValueError("forced and sham spikes must share (time, E neuron) shape")
    if positions.shape != (forced.shape[1], 2) or source_mask.shape != (forced.shape[1],):
        raise ValueError("positions and source mask must align to E neurons")
    start = int(round(float(start_ms) / float(dt_ms)))
    stop = min(len(forced), int(round(float(end_ms) / float(dt_ms))))
    if start < 0 or stop <= start:
        raise ValueError("paired response window is empty")
    signed = (forced[start:stop].sum(axis=0).astype(float)
              - sham[start:stop].sum(axis=0).astype(float))
    positive = np.clip(signed, 0.0, None)
    downstream = ~source_mask
    downstream_weight = positive * downstream
    radius = np.linalg.norm(
        positions - np.asarray(source_center, float)[None, :], axis=1)

    def weighted_radius(quantile):
        valid = downstream_weight > 0.0
        if not valid.any():
            return None
        order = np.argsort(radius[valid], kind="stable")
        r = radius[valid][order]
        w = downstream_weight[valid][order]
        target = float(quantile) * float(w.sum())
        index = min(int(np.searchsorted(np.cumsum(w), target, side="left")), len(r) - 1)
        return float(r[index])

    return {
        "source_positive_spike_mass": float(positive[source_mask].sum()),
        "downstream_positive_spike_mass": float(downstream_weight.sum()),
        "downstream_positive_neurons": int(np.count_nonzero(downstream_weight > 0.0)),
        "downstream_any_positive": bool(np.any(downstream_weight > 0.0)),
        "r50_mm": weighted_radius(0.50),
        "r90_mm": weighted_radius(0.90),
        "signed_spike_count_per_E": signed,
        "positive_spike_count_per_E": positive,
    }


def exclude_injected_packet_frame(forced_spikes, sham_spikes, packet_mask,
                                  *, trigger_step):
    """Return a readout copy with only the direct injected source frame removed."""
    forced = np.asarray(forced_spikes, bool)
    sham = np.asarray(sham_spikes, bool)
    packet_mask = np.asarray(packet_mask, bool)
    if forced.shape != sham.shape or forced.ndim != 2:
        raise ValueError("forced and sham spikes must share (time, E neuron) shape")
    if packet_mask.shape != (forced.shape[1],):
        raise ValueError("packet_mask must align to E neurons")
    trigger_step = int(trigger_step)
    if trigger_step < 0 or trigger_step >= len(forced):
        raise ValueError("trigger_step lies outside the spike array")
    response = forced.copy()
    response[trigger_step, packet_mask] = sham[trigger_step, packet_mask]
    return response


def select_triggered_event(events, *, trigger_ms, max_latency_ms):
    """Earliest returned event whose onset is time-locked to the forced packet."""
    lower = float(trigger_ms)
    upper = lower + float(max_latency_ms)
    eligible = [
        event for event in events
        if bool(event.get("returned", False))
        and lower <= float(event["t_on"]) <= upper
        and float(event["t_off"]) > lower
    ]
    return None if not eligible else min(eligible, key=lambda event: float(event["t_on"]))


def select_packet_fraction(rows, *, source_ids, min_networks_per_source):
    """Freeze the smallest readable packet, with an explicit sparse fallback."""
    source_ids = [str(value) for value in source_ids]
    fractions = sorted({float(row["packet_fraction_of_E"]) for row in rows})
    summaries = []
    for fraction in fractions:
        selected = [row for row in rows
                    if np.isclose(row["packet_fraction_of_E"], fraction)]
        coverage = {}
        for source_id in source_ids:
            source_rows = [row for row in selected if row["source_id"] == source_id]
            coverage[source_id] = int(sum(
                bool(row["pretrigger_spikes_bit_identical"])
                and bool(row["paired_excess_readout"]["curve_usable"])
                and bool(row["paired_geometry"]["downstream_any_positive"])
                and row["runaway_early_stop_ms"] is None
                for row in source_rows))
        n_runaway = int(sum(row["runaway_early_stop_ms"] is not None
                            for row in selected))
        minimum = min(coverage.values()) if coverage else 0
        summaries.append({
            "packet_fraction_of_E": fraction,
            "eligible_networks_by_source": coverage,
            "minimum_source_coverage": int(minimum),
            "n_runaway": n_runaway,
            "selection_eligible": bool(
                minimum >= int(min_networks_per_source) and n_runaway == 0),
        })
    eligible = [row for row in summaries if row["selection_eligible"]]
    if eligible:
        selected = min(eligible, key=lambda row: row["packet_fraction_of_E"])
        status = "PACKET_FRACTION_FROZEN"
    else:
        selected = min(
            summaries,
            key=lambda row: (-row["minimum_source_coverage"], row["n_runaway"],
                             row["packet_fraction_of_E"]))
        status = "PACKET_FRACTION_SPARSE_FALLBACK"
    return {"status": status, "selected": selected, "fractions": summaries}


def _finite_summary(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "q05": None, "q95": None, "n": 0}
    return {
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "n": int(len(values)),
    }


def source_mode_correlation_summary(
        curves, source_ids, patient_prototypes, *, source_order):
    """Per-network source-to-patient-mode Spearman matrix.

    Source identity is kept fixed. This is an oracle/capacity readout and does
    not run KMeans or relabel a source after seeing its response.
    """
    curves = np.asarray(curves, float)
    source_ids = np.asarray(source_ids).astype(str)
    prototypes = np.asarray(patient_prototypes, float)
    source_order = [str(value) for value in source_order]
    if curves.ndim != 2 or prototypes.ndim != 2:
        raise ValueError("curves and patient_prototypes must be two-dimensional")
    if len(curves) != len(source_ids):
        raise ValueError("curves and source_ids must align")
    if curves.shape[1] != prototypes.shape[1] or prototypes.shape[0] != 2:
        raise ValueError("two patient prototypes must share the curve grid")
    unknown = sorted(set(source_ids) - set(source_order))
    if unknown:
        raise ValueError(f"source_order is missing source ids: {unknown}")

    per_source = {}
    matrix = np.full((len(source_order), 2), np.nan)
    for source_index, source in enumerate(source_order):
        selected = np.flatnonzero(source_ids == source)
        correlations = np.full((len(selected), 2), np.nan)
        for row_index, curve_index in enumerate(selected):
            curve = curves[curve_index]
            if not np.isfinite(curve).all() or np.std(curve) < 1e-12:
                continue
            for mode in range(2):
                prototype = prototypes[mode]
                if not np.isfinite(prototype).all() or np.std(prototype) < 1e-12:
                    continue
                correlations[row_index, mode] = float(
                    spearmanr(curve, prototype).statistic)
        summaries = [_finite_summary(correlations[:, mode]) for mode in range(2)]
        matrix[source_index] = [summary["median"] if summary["median"] is not None
                                else np.nan for summary in summaries]
        per_source[source] = {
            "n_total": int(len(selected)),
            "n_usable": int(np.isfinite(correlations).all(axis=1).sum()),
            "patient_A": summaries[0],
            "patient_B": summaries[1],
            "per_network_correlation_to_A_B": correlations.tolist(),
        }
    return {
        "source_order": source_order,
        "patient_mode_order": ["A", "B"],
        "median_correlation_matrix": [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix
        ],
        "sources": per_source,
    }
