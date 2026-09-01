"""Pure helpers for rev10-SA contact and dual-shaft capacity canaries."""
from __future__ import annotations

import hashlib

import numpy as np

from src.topic4_core_field_stage3 import params_to_h, unpack


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


def _theta_sha256(theta):
    return hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest()


def shaft_geometry(contact_xy):
    """Return a deterministic midpoint and principal shaft direction."""
    values = np.asarray(contact_xy, float)
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < 2:
        raise ValueError("shaft geometry requires at least two 2-D contacts")
    midpoint = values.mean(axis=0)
    _, _, right = np.linalg.svd(values - midpoint, full_matrices=False)
    unit = right[0]
    if unit[0] < 0.0 or (unit[0] == 0.0 and unit[1] < 0.0):
        unit = -unit
    return {
        "midpoint": midpoint,
        "unit": unit,
        "phi": float(np.mod(np.arctan2(unit[1], unit[0]), np.pi)),
    }


def equal_mode_earliest_shaft_centroid(onsets, labels, shaft_indices, contact_xy):
    """Average earliest-contact locations per mode, then weight modes equally."""
    values = np.asarray(onsets, float)
    labels = np.asarray(labels, int)
    indices = np.asarray(shaft_indices, int)
    xy = np.asarray(contact_xy, float)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("onsets and labels do not align")
    if xy.shape != (values.shape[1], 2) or not len(indices):
        raise ValueError("contact geometry does not align with onsets")
    mode_centroids = []
    mode_counts = []
    for mode in (0, 1):
        event_centroids = []
        for row in values[labels == mode]:
            local = row[indices]
            finite = np.isfinite(local)
            if not finite.any():
                continue
            first = float(np.min(local[finite]))
            earliest = indices[finite & np.isclose(local, first, atol=1e-12)]
            event_centroids.append(xy[earliest].mean(axis=0))
        if not event_centroids:
            raise RuntimeError(f"mode {mode} has no recruited shaft contact")
        mode_centroids.append(np.mean(event_centroids, axis=0))
        mode_counts.append(len(event_centroids))
    return {
        "centroid": np.mean(mode_centroids, axis=0),
        "mode_centroids": np.asarray(mode_centroids),
        "mode_event_counts": np.asarray(mode_counts, int),
    }


def matched_offshaft_center(frozen_center, target_center, contact_xy, *, L=20.0,
                            margin_mm=2.0):
    """Rotate the relocation vector by 90 degrees without changing its length."""
    origin = np.asarray(frozen_center, float)
    target = np.asarray(target_center, float)
    contacts = np.asarray(contact_xy, float)
    delta = target - origin
    rotations = (
        np.asarray([-delta[1], delta[0]]),
        np.asarray([delta[1], -delta[0]]),
    )
    midpoint = np.full(2, float(L) / 2.0)
    radius = float(L) / 2.0 - float(margin_mm)
    candidates = [origin + value for value in rotations]
    admissible = [value for value in candidates
                  if np.linalg.norm(value - midpoint) <= radius + 1e-12]
    if not admissible:
        raise RuntimeError("no matched off-shaft relocation lies in the center disc")
    distance_to_contacts = [
        float(np.min(np.linalg.norm(contacts - value, axis=1)))
        for value in admissible
    ]
    selected = admissible[int(np.argmax(distance_to_contacts))]
    if not np.isclose(np.linalg.norm(selected - origin), np.linalg.norm(delta)):
        raise RuntimeError("off-shaft relocation distance changed")
    return selected


def replace_component(theta, component_index, *, center, sigma_par=None,
                      sigma_perp=None, phi=None, weight=None, K=3, L=20.0):
    """Replace one physical Gaussian component and optionally its mixture mass."""
    output = np.asarray(theta, float).copy()
    if output.size != 5 * int(K) + int(K) - 1:
        raise ValueError("theta length does not match K")
    index = int(component_index)
    if index < 0 or index >= int(K):
        raise ValueError("component index is outside K")
    components = unpack(output, K=K, L=L)
    base = 5 * index
    output[base:base + 2] = np.asarray(center, float)
    if sigma_par is not None:
        output[base + 2] = np.log(float(sigma_par))
    if sigma_perp is not None:
        output[base + 3] = np.log(float(sigma_perp))
    if phi is not None:
        output[base + 4] = float(phi)
    if weight is not None:
        selected = float(weight)
        if not 0.0 < selected < 1.0:
            raise ValueError("component weight must lie in (0, 1)")
        old = np.asarray([row["weight"] for row in components], float)
        remaining = np.delete(old, index)
        remaining = (1.0 - selected) * remaining / remaining.sum()
        weights = np.insert(remaining, index, selected)
        output[5 * K:5 * K + K - 1] = np.log(weights[:-1] / weights[-1])
    return output


def build_dual_shaft_candidates(
    frozen_theta,
    *,
    scl_midpoint,
    scl_earliest_centroid,
    scl_phi,
    contact_xy,
    mass_fractions=(0.15, 0.25, 0.35),
    sigma_parallel_mm=(1.5, 3.0, 4.5),
    K=3,
    L=20.0,
):
    """Build the 21 deterministic fixed-budget SA6 field candidates."""
    frozen = np.asarray(frozen_theta, float)
    components = unpack(frozen, K=K, L=L)
    scl_index = int(K) - 1
    original = components[scl_index]
    midpoint = np.asarray(scl_midpoint, float)
    earliest = np.asarray(scl_earliest_centroid, float)
    offshaft = matched_offshaft_center(
        original["center"], midpoint, contact_xy, L=L,
    )

    rows = []

    def add(candidate_id, role, theta, **metadata):
        decoded = []
        for component in unpack(theta, K=K, L=L):
            decoded.append({
                **component,
                "center": np.asarray(component["center"], float).tolist(),
            })
        rows.append({
            "candidate_id": candidate_id,
            "role": role,
            "theta": np.asarray(theta, float).tolist(),
            "theta_sha256": _theta_sha256(theta),
            "components": decoded,
            **metadata,
        })

    add("frozen", "frozen_baseline", frozen, center_role="frozen")
    relocated = replace_component(
        frozen, scl_index, center=midpoint, phi=scl_phi, K=K, L=L,
    )
    add(
        "component3_scl_relocation", "matched_scl_relocation", relocated,
        center_role="scl_midpoint",
    )
    control = replace_component(
        frozen, scl_index, center=offshaft, phi=scl_phi, K=K, L=L,
    )
    add(
        "component3_offshaft_control", "matched_offshaft_control", control,
        center_role="matched_offshaft",
    )
    centers = {"mid": midpoint, "early": earliest}
    for center_slug, center in centers.items():
        for mass in mass_fractions:
            for sigma in sigma_parallel_mm:
                theta = replace_component(
                    frozen, scl_index, center=center,
                    sigma_par=float(sigma),
                    sigma_perp=float(original["sigma_perp"]),
                    phi=scl_phi, weight=float(mass), K=K, L=L,
                )
                add(
                    f"grid_{center_slug}_w{int(round(100 * mass)):02d}_s{sigma:g}"
                    .replace(".", "p"),
                    "scl_mass_width_grid", theta,
                    center_role=("scl_midpoint" if center_slug == "mid"
                                 else "patient_earliest_scl_centroid"),
                    requested_scl_mixture_weight=float(mass),
                    requested_scl_sigma_parallel_mm=float(sigma),
                )
    if len(rows) != 3 + 2 * len(mass_fractions) * len(sigma_parallel_mm):
        raise RuntimeError("unexpected SA6 candidate count")
    return {
        "candidates": rows,
        "geometry": {
            "scl_midpoint": midpoint.tolist(),
            "scl_earliest_centroid": earliest.tolist(),
            "scl_phi_rad": float(scl_phi),
            "matched_offshaft_center": offshaft.tolist(),
            "frozen_component_3_center": np.asarray(
                original["center"], float).tolist(),
            "relocation_distance_mm": float(np.linalg.norm(
                midpoint - np.asarray(original["center"], float)
            )),
        },
    }


def field_budget_summary(theta, positions, *, target_count, K=3, L=20.0):
    """Compute the exact per-network h budget and SCL-near field support."""
    h = params_to_h(theta, positions, K=K, L=L, target_count=target_count)
    return h, {
        "sum_h": float(h.sum()),
        "max_h": float(h.max(initial=0.0)),
        "n_h_ge_0p5": int(np.sum(h >= 0.5)),
        "n_h_ge_0p9": int(np.sum(h >= 0.9)),
    }
