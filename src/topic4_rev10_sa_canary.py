"""Pure helpers for rev10-SA contact and dual-shaft capacity canaries."""
from __future__ import annotations

import numpy as np


def matched_contact_packets(positions, contacts, *, radius_mm, requested_count,
                            minimum_count=1):
    """Select an equal number of nearest E cells inside one fixed-radius disk."""
    positions = np.asarray(positions, float)
    contacts = np.asarray(contacts, float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape (n_E, 2)")
    if contacts.ndim != 2 or contacts.shape[1] != 2:
        raise ValueError("contacts must have shape (n_contact, 2)")
    if not len(contacts):
        raise ValueError("at least one contact is required")
    if radius_mm <= 0 or requested_count < 1 or minimum_count < 1:
        raise ValueError("radius and packet counts must be positive")
    distances = np.linalg.norm(
        positions[None, :, :] - contacts[:, None, :], axis=2,
    )
    available = (distances <= float(radius_mm)).sum(axis=1)
    common_count = min(int(requested_count), int(available.min()))
    if common_count < int(minimum_count):
        raise RuntimeError(
            f"fixed-radius packet has only {common_count} common cells; "
            f"minimum is {minimum_count}"
        )
    masks = np.zeros((len(contacts), len(positions)), bool)
    selected = []
    for contact_index in range(len(contacts)):
        eligible = np.flatnonzero(distances[contact_index] <= float(radius_mm))
        order = np.argsort(distances[contact_index, eligible], kind="mergesort")
        indices = eligible[order[:common_count]]
        masks[contact_index, indices] = True
        selected.append(indices)
    return {
        "masks": masks,
        "indices": selected,
        "available_counts": available,
        "common_count": int(common_count),
        "requested_count": int(requested_count),
        "radius_mm": float(radius_mm),
    }


def lfp_kernel_audit(positions, contacts, *, cutoff_mm, rx_mm):
    """Reproduce the LFP spatial kernel support before per-contact normalization."""
    positions = np.asarray(positions, float)
    contacts = np.asarray(contacts, float)
    if cutoff_mm <= 0 or rx_mm <= 0:
        raise ValueError("kernel scales must be positive")
    rows = []
    for contact in contacts:
        distance = np.linalg.norm(positions - contact, axis=1)
        selected = distance <= float(cutoff_mm)
        if not selected.any():
            selected[np.argmin(distance)] = True
        local = np.maximum(distance[selected], 1e-4)
        weight = np.empty_like(local)
        near = local < float(rx_mm)
        weight[near] = local[near] ** -0.5
        weight[~near] = (
            (1.0 / np.sqrt(float(rx_mm)))
            * (float(rx_mm) / local[~near]) ** 2
        )
        normalized = weight / weight.sum()
        rows.append({
            "neuron_count": int(selected.sum()),
            "raw_kernel_mass": float(weight.sum()),
            "normalized_weight_ess": float(1.0 / np.sum(normalized ** 2)),
            "nearest_distance_mm": float(distance.min()),
        })
    return rows


def contact_response_metrics(
    forced_lfp,
    sham_lfp,
    forced_spikes,
    sham_spikes,
    positions,
    contact_xy,
    packet_mask,
    *,
    dt_ms,
    forced_spike_ms,
    response_stop_ms,
    baseline_window_ms,
    local_radius_mm,
):
    """Separate local neural response from the current-based contact readout."""
    forced_lfp = np.asarray(forced_lfp, float)
    sham_lfp = np.asarray(sham_lfp, float)
    forced_spikes = np.asarray(forced_spikes, bool)
    sham_spikes = np.asarray(sham_spikes, bool)
    positions = np.asarray(positions, float)
    packet_mask = np.asarray(packet_mask, bool)
    if forced_lfp.shape != sham_lfp.shape or forced_lfp.ndim != 1:
        raise ValueError("local LFP traces must be aligned vectors")
    if forced_spikes.shape != sham_spikes.shape:
        raise ValueError("forced/sham spike arrays must align")
    if forced_spikes.shape[1] != len(positions) or packet_mask.shape != (len(positions),):
        raise ValueError("spikes, positions, and packet mask do not align")
    trigger_step = int(round(float(forced_spike_ms) / float(dt_ms)))
    response_start = min(trigger_step + 1, len(forced_lfp))
    response_stop = min(
        len(forced_lfp), int(round(float(response_stop_ms) / float(dt_ms))),
    )
    baseline_start = max(0, int(round(float(baseline_window_ms[0]) / float(dt_ms))))
    baseline_stop = min(
        len(sham_lfp), int(round(float(baseline_window_ms[1]) / float(dt_ms))),
    )
    if response_stop <= response_start or baseline_stop <= baseline_start:
        raise ValueError("response or baseline window is empty")
    lfp_excess = forced_lfp - sham_lfp
    response = lfp_excess[response_start:response_stop]
    baseline = sham_lfp[baseline_start:baseline_stop]
    baseline_mean = float(baseline.mean())
    baseline_sd = float(baseline.std(ddof=1)) if len(baseline) > 1 else 0.0
    detector_threshold = baseline_mean + 5.0 * baseline_sd
    forced_peak = float(forced_lfp[response_start:response_stop].max())
    peak_excess = float(np.max(response, initial=0.0))
    local_mask = np.linalg.norm(
        positions - np.asarray(contact_xy, float), axis=1,
    ) <= float(local_radius_mm)
    spike_delta = (
        forced_spikes[response_start:response_stop].astype(np.int16)
        - sham_spikes[response_start:response_stop].astype(np.int16)
    )
    local_signed = float(spike_delta[:, local_mask].sum())
    local_positive = float(np.clip(spike_delta[:, local_mask], 0, None).sum())
    packet_signed = float(spike_delta[:, packet_mask].sum())
    packet_positive = float(np.clip(spike_delta[:, packet_mask], 0, None).sum())
    n_packet = int(packet_mask.sum())
    n_local = int(local_mask.sum())
    return {
        "peak_lfp_excess": peak_excess,
        "integrated_positive_lfp_excess": float(
            np.clip(response, 0.0, None).sum() * float(dt_ms)
        ),
        "peak_lfp_excess_per_packet_cell": peak_excess / n_packet,
        "baseline_lfp_mean": baseline_mean,
        "baseline_lfp_sd": baseline_sd,
        "lfp_peak_snr": peak_excess / max(baseline_sd, 1e-12),
        "absolute_detector_threshold_mean_plus_5sd": detector_threshold,
        "absolute_detector_margin": forced_peak - detector_threshold,
        "local_neuron_count": n_local,
        "local_signed_spike_excess_per_cell": local_signed / max(n_local, 1),
        "local_positive_spike_excess_per_cell": local_positive / max(n_local, 1),
        "packet_signed_respike_excess_per_cell": packet_signed / n_packet,
        "packet_positive_respike_excess_per_cell": packet_positive / n_packet,
        "response_window_ms": [
            float(response_start * dt_ms), float(response_stop * dt_ms),
        ],
        "baseline_window_ms": [
            float(baseline_start * dt_ms), float(baseline_stop * dt_ms),
        ],
    }


def paired_shaft_ratio(rows, value_key, shaft_key="shaft_id"):
    """Median SCL/ICL ratio for one paired network; negative values are clipped."""
    values = {
        shaft: np.asarray([
            max(0.0, float(row[value_key])) for row in rows
            if row[shaft_key] == shaft
        ]) for shaft in ("ICL", "SCL")
    }
    if not len(values["ICL"]) or not len(values["SCL"]):
        raise ValueError("both shafts are required")
    denominator = float(np.median(values["ICL"]))
    numerator = float(np.median(values["SCL"]))
    return {
        "ICL_median": denominator,
        "SCL_median": numerator,
        "SCL_over_ICL": numerator / max(denominator, 1e-12),
    }


def classify_contact_detectability(lfp_ratio, neural_ratio, *, reference_ratio=0.5):
    """Descriptive SA5 branch label; the ratio is not a formal acceptance gate."""
    values = np.asarray([lfp_ratio, neural_ratio, reference_ratio], float)
    if not np.isfinite(values).all() or reference_ratio <= 0:
        raise ValueError("contact-detectability ratios must be finite and positive")
    if neural_ratio < reference_ratio:
        return "SCL_LOCAL_NETWORK_RESPONSE_LIMIT"
    if lfp_ratio < reference_ratio:
        return "VIRTUAL_CONTACT_OBSERVATION_FAIL"
    return "SCL_READOUT_NOT_PRIMARY_LIMIT"
