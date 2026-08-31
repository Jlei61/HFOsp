"""Out-of-sample decoder-geometry diagnostics for H2b v0.3.

The geometry is fitted only on frozen interictal decoder trajectories available
before an outer-fold cutoff.  Seizure trajectories are projected afterwards.
This module intentionally uses small deterministic estimators so that basin,
approach, and abrupt-exit evidence remain separate and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DecoderProjection:
    centre: np.ndarray
    scale: np.ndarray
    active: np.ndarray
    loadings: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        standard = (matrix[:, self.active] - self.centre[self.active]) / self.scale[
            self.active
        ]
        return standard @ self.loadings.T


def fit_decoder_projection(
    values: np.ndarray, *, maximum_components: int = 6,
    variance_fraction: float = 0.95,
) -> DecoderProjection:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 20:
        raise ValueError("decoder projection requires at least 20 rows")
    centre = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    active = np.isfinite(scale) & (scale > 1e-8)
    if not np.any(active):
        raise ValueError("decoder projection has no active dimensions")
    scale = np.where(active, scale, 1.0)
    standard = (matrix[:, active] - centre[active]) / scale[active]
    _, singular, vt = np.linalg.svd(standard, full_matrices=False)
    variance = singular ** 2
    if not len(variance) or float(np.sum(variance)) <= 1e-12:
        raise ValueError("decoder projection is rank zero")
    cumulative = np.cumsum(variance) / float(np.sum(variance))
    count = int(np.searchsorted(cumulative, float(variance_fraction)) + 1)
    count = max(1, min(count, int(maximum_components), vt.shape[0]))
    loadings = np.array(vt[:count], copy=True)
    for index in range(len(loadings)):
        pivot = int(np.argmax(np.abs(loadings[index])))
        if loadings[index, pivot] < 0:
            loadings[index] *= -1.0
    return DecoderProjection(
        centre=centre, scale=scale, active=active, loadings=loadings,
    )


def fit_two_basins(scores: np.ndarray, *, iterations: int = 50) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2 or len(matrix) < 20:
        raise ValueError("two-basin fit requires at least 20 rows")
    first = int(np.argmin(matrix[:, 0]))
    second = int(np.argmax(matrix[:, 0]))
    if first == second:
        second = int(np.argmax(np.linalg.norm(matrix - matrix[first], axis=1)))
    centres = np.vstack([matrix[first], matrix[second]])
    for _ in range(int(iterations)):
        distance = np.sum((matrix[:, None, :] - centres[None, :, :]) ** 2, axis=2)
        label = np.argmin(distance, axis=1)
        updated = np.vstack([
            np.mean(matrix[label == group], axis=0)
            if np.any(label == group) else centres[group]
            for group in (0, 1)
        ])
        if np.allclose(updated, centres, rtol=0.0, atol=1e-8):
            centres = updated
            break
        centres = updated
    return centres


def assign_basins(scores: np.ndarray, centres: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64)
    centre = np.asarray(centres, dtype=np.float64)
    return np.argmin(
        np.sum((matrix[:, None, :] - centre[None, :, :]) ** 2, axis=2),
        axis=1,
    ).astype(np.int64)


def _nearest_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    query_value = np.asarray(query, dtype=np.float64)
    reference_value = np.asarray(reference, dtype=np.float64)
    result = np.full(len(query_value), np.inf, dtype=np.float64)
    for start in range(0, len(query_value), 256):
        stop = min(start + 256, len(query_value))
        distance = np.sum(
            (query_value[start:stop, None, :] - reference_value[None, :, :]) ** 2,
            axis=2,
        )
        result[start:stop] = np.sqrt(np.min(distance, axis=1))
    return result


def _reference_nn_distance(reference: np.ndarray) -> np.ndarray:
    value = np.asarray(reference, dtype=np.float64)
    result = np.full(len(value), np.inf, dtype=np.float64)
    for start in range(0, len(value), 256):
        stop = min(start + 256, len(value))
        distance = np.sum(
            (value[start:stop, None, :] - value[None, :, :]) ** 2, axis=2,
        )
        local = np.arange(stop - start)
        distance[local, np.arange(start, stop)] = np.inf
        result[start:stop] = np.sqrt(np.min(distance, axis=1))
    return result


def _robust_z(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float64)
    centre = float(np.median(ref))
    mad = float(np.median(np.abs(ref - centre))) * 1.4826
    scale = mad if mad > 1e-8 else max(float(np.std(ref)), 1e-8)
    return (np.asarray(value, dtype=np.float64) - centre) / scale


def _longest_true_run_minutes(mask: np.ndarray, time_epoch: np.ndarray) -> float:
    flag = np.asarray(mask, dtype=bool)
    time = np.asarray(time_epoch, dtype=np.float64)
    best = 0.0
    start = None
    for index, value in enumerate(flag):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(flag) - 1):
            stop = index if value and index == len(flag) - 1 else index - 1
            if stop > start:
                best = max(best, float(time[stop] - time[start]) / 60.0)
            elif stop == start:
                best = max(best, 0.5)
            start = None
    return best


def trajectory_features(
    time_epoch: np.ndarray,
    scores: np.ndarray,
    *,
    centres: np.ndarray,
    entry_basin: int,
    entry_centroid: np.ndarray,
    entry_direction: np.ndarray,
    reference_scores: np.ndarray,
    reference_nn: np.ndarray,
) -> dict[str, float]:
    time = np.asarray(time_epoch, dtype=np.float64)
    value = np.asarray(scores, dtype=np.float64)
    if len(time) < 3 or len(value) != len(time):
        raise ValueError("trajectory needs at least three aligned rows")
    order = np.argsort(time, kind="stable")
    time, value = time[order], value[order]
    basin = assign_basins(value, centres)
    distance = np.linalg.norm(value - entry_centroid[None, :], axis=1)
    elapsed = max(float(time[-1] - time[0]) / 60.0, 1e-6)
    delta = np.diff(value, axis=0)
    delta_norm = np.linalg.norm(delta, axis=1)
    valid = delta_norm > 1e-10
    alignment = np.sum(delta[valid] * entry_direction[None, :], axis=1) / (
        delta_norm[valid] * max(float(np.linalg.norm(entry_direction)), 1e-10)
    ) if np.any(valid) else np.asarray([0.0])
    off = _robust_z(_nearest_distance(value, reference_scores), reference_nn)
    return {
        "entry_basin_occupancy": float(np.mean(basin == int(entry_basin))),
        "entry_basin_longest_dwell_minutes": _longest_true_run_minutes(
            basin == int(entry_basin), time,
        ),
        "approach_rate_per_minute": float((distance[0] - distance[-1]) / elapsed),
        "flow_alignment": float(np.mean(alignment)),
        "median_off_manifold_z": float(np.median(off)),
        "max_off_manifold_z": float(np.max(off)),
    }


def _trajectory_rows(
    time_epoch: np.ndarray, session: np.ndarray, *, endpoint: float,
    label: int, lookback_minutes: float, maximum_gap_seconds: float = 120.0,
    minimum_coverage_fraction: float = 0.70,
) -> np.ndarray:
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(session, dtype=np.int64)
    rows = np.flatnonzero(
        (group == int(label)) & (time < float(endpoint))
        & (time >= float(endpoint) - float(lookback_minutes) * 60.0)
    )
    rows = rows[np.argsort(time[rows], kind="stable")]
    if len(rows) < 3:
        return np.empty(0, dtype=np.int64)
    gap = np.diff(time[rows])
    if len(gap) and float(np.max(gap)) > float(maximum_gap_seconds):
        return np.empty(0, dtype=np.int64)
    coverage = float(time[rows[-1]] - time[rows[0]]) / 60.0
    if coverage < float(minimum_coverage_fraction) * float(lookback_minutes):
        return np.empty(0, dtype=np.int64)
    return rows


def _clock_distance_seconds(left: float, right: np.ndarray) -> np.ndarray:
    delta = np.abs((np.asarray(right) % 86400.0) - (float(left) % 86400.0))
    return np.minimum(delta, 86400.0 - delta)


def matched_control_trajectories(
    time_epoch: np.ndarray, session: np.ndarray, *, case_onset: float,
    lookback_minutes: float, maximum_controls: int = 20,
    maximum_endpoint: float | None = None,
    forbidden_onsets: np.ndarray | None = None,
    maximum_gap_seconds: float = 120.0,
    minimum_coverage_fraction: float = 0.70,
) -> list[np.ndarray]:
    time = np.asarray(time_epoch, dtype=np.float64)
    group = np.asarray(session, dtype=np.int64)
    candidates: list[tuple[float, float, int, np.ndarray]] = []
    for label in np.unique(group):
        rows = np.flatnonzero(group == label)
        rows = rows[np.argsort(time[rows], kind="stable")]
        for endpoint_index in range(0, len(rows), 10):
            endpoint = float(time[rows[endpoint_index]])
            if maximum_endpoint is not None and endpoint > float(maximum_endpoint):
                continue
            if forbidden_onsets is not None and np.any(
                np.abs(np.asarray(forbidden_onsets, dtype=np.float64) - endpoint)
                < float(lookback_minutes) * 60.0
            ):
                continue
            trajectory = _trajectory_rows(
                time, group, endpoint=endpoint, label=int(label),
                lookback_minutes=float(lookback_minutes),
                maximum_gap_seconds=float(maximum_gap_seconds),
                minimum_coverage_fraction=float(minimum_coverage_fraction),
            )
            if len(trajectory):
                candidates.append((
                    float(_clock_distance_seconds(case_onset, np.asarray([endpoint]))[0]),
                    endpoint, int(label), trajectory,
                ))
    selected: list[np.ndarray] = []
    endpoints: list[tuple[float, int]] = []
    separation = float(lookback_minutes) * 60.0
    for _, endpoint, label, trajectory in sorted(candidates, key=lambda row: row[:2]):
        if any(
            previous_label == label and abs(endpoint - previous) < separation
            for previous, previous_label in endpoints
        ):
            continue
        selected.append(trajectory)
        endpoints.append((endpoint, label))
        if len(selected) >= int(maximum_controls):
            break
    return selected


def evaluate_oos_geometry_fold_full_grid(
    *,
    grid_time: np.ndarray,
    grid_segment: np.ndarray,
    grid_decoder: np.ndarray,
    onset_time: np.ndarray,
    onset_segment: np.ndarray,
    heldout_position: int,
    lookback_minutes: float = 30.0,
    maximum_controls: int = 20,
    grid_spacing_seconds: float = 300.0,
    clean_interictal_exclusion_minutes: float = 120.0,
) -> dict[str, Any]:
    """Evaluate one fold entirely in a common full-grid extraction domain.

    Projection, basins and control trajectories use only rows no later than the
    previous seizure.  The held-out preictal trajectory is projected after all
    geometry has been frozen.  This avoids comparing event-anchor states with
    regular-grid states, which produced arbitrarily large off-manifold scores.
    """
    time = np.asarray(grid_time, dtype=np.float64)
    segment = np.asarray(grid_segment, dtype=np.int64)
    decoder = np.asarray(grid_decoder, dtype=np.float64)
    onset = np.asarray(onset_time, dtype=np.float64)
    onset_group = np.asarray(onset_segment, dtype=np.int64)
    order = np.argsort(onset, kind="stable")
    onset, onset_group = onset[order], onset_group[order]
    position = int(heldout_position)
    if position < 2 or position >= len(onset):
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_prior_seizures"}
    cutoff = float(onset[position - 1])
    heldout = float(onset[position])
    heldout_segment = int(onset_group[position])
    clean = time <= cutoff
    exclusion_seconds = float(clean_interictal_exclusion_minutes) * 60.0
    for event in onset[:position]:
        clean &= np.abs(time - float(event)) > exclusion_seconds
    train = np.flatnonzero(clean)
    # Forty five-minute rows provide at least ~3.3 recorded hours while still
    # allowing early development seizures to contribute.  The old 100-row
    # threshold silently discarded most chronological folds and was not a
    # scientific requirement of the projection estimator (minimum 20 rows).
    if len(train) < 40:
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_past_full_grid_rows"}
    projection = fit_decoder_projection(decoder[train])
    all_score = projection.transform(decoder)
    train_score = all_score[train]
    centres = fit_two_basins(train_score)
    maximum_gap = max(1.2 * float(grid_spacing_seconds), 360.0)
    minimum_coverage = 0.55

    prior_end: list[np.ndarray] = []
    prior_direction: list[np.ndarray] = []
    for event, label in zip(onset[:position], onset_group[:position]):
        rows = _trajectory_rows(
            time, segment, endpoint=float(event), label=int(label),
            lookback_minutes=float(lookback_minutes),
            maximum_gap_seconds=maximum_gap,
            minimum_coverage_fraction=minimum_coverage,
        )
        if len(rows):
            prior_end.append(all_score[rows[-1]])
            displacement = all_score[rows[-1]] - all_score[rows[0]]
            norm = float(np.linalg.norm(displacement))
            if norm > 1e-10:
                prior_direction.append(displacement / norm)
    if len(prior_end) < 2 or not prior_direction:
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_prior_entry_trajectories"}
    prior_end_value = np.asarray(prior_end, dtype=np.float64)
    entry_label = assign_basins(prior_end_value, centres)
    entry_basin = int(np.bincount(entry_label, minlength=2).argmax())
    entry_centroid = np.mean(prior_end_value, axis=0)
    entry_direction = np.mean(np.asarray(prior_direction), axis=0)
    if float(np.linalg.norm(entry_direction)) <= 1e-10:
        return {"status": "NOT_ESTIMABLE", "reason": "entry_direction_cancels"}

    case_rows = _trajectory_rows(
        time, segment, endpoint=heldout, label=heldout_segment,
        lookback_minutes=float(lookback_minutes),
        maximum_gap_seconds=maximum_gap,
        minimum_coverage_fraction=minimum_coverage,
    )
    if not len(case_rows):
        return {"status": "NOT_ESTIMABLE", "reason": "heldout_trajectory_incomplete"}
    controls = matched_control_trajectories(
        time, segment, case_onset=heldout,
        lookback_minutes=float(lookback_minutes),
        maximum_controls=int(maximum_controls), maximum_endpoint=cutoff,
        forbidden_onsets=onset[:position + 1], maximum_gap_seconds=maximum_gap,
        minimum_coverage_fraction=minimum_coverage,
    )
    if len(controls) < 5:
        return {"status": "NOT_ESTIMABLE", "reason": "fewer_than_five_matched_controls"}
    reference = train_score
    if len(reference) > 1500:
        take = np.linspace(0, len(reference) - 1, 1500).round().astype(np.int64)
        reference = reference[take]
    reference_nn = _reference_nn_distance(reference)
    common = {
        "centres": centres, "entry_basin": entry_basin,
        "entry_centroid": entry_centroid, "entry_direction": entry_direction,
        "reference_scores": reference, "reference_nn": reference_nn,
    }
    case = trajectory_features(time[case_rows], all_score[case_rows], **common)
    control_features = [
        trajectory_features(time[rows], all_score[rows], **common)
        for rows in controls
    ]
    effects = {
        key: _effect(case[key], [row[key] for row in control_features])
        for key in case
    }
    basin_values = [
        effects[key]["signed_percentile"] for key in (
            "entry_basin_occupancy", "entry_basin_longest_dwell_minutes",
        )
    ]
    approach_values = [
        effects[key]["signed_percentile"] for key in (
            "approach_rate_per_minute", "flow_alignment",
        )
    ]
    family_scores = {
        "basin_gating": float(np.mean(basin_values)) if basin_values else None,
        "directed_approach": float(np.mean(approach_values)) if approach_values else None,
        "abrupt_transition": effects["max_off_manifold_z"]["signed_percentile"],
    }
    finite = {key: value for key, value in family_scores.items() if value is not None}
    return {
        "status": "COMPLETE_EXPLORATORY", "heldout_position": position,
        "heldout_onset_epoch": heldout, "heldout_segment": heldout_segment,
        "train_cutoff_epoch": cutoff, "n_past_full_grid_rows": int(len(train)),
        "n_prior_entry_trajectories": int(len(prior_end)),
        "n_controls": int(len(controls)), "lookback_minutes": float(lookback_minutes),
        "projection_active_dimensions": int(np.sum(projection.active)),
        "projection_components": int(len(projection.loadings)),
        "case_features": case, "effects": effects,
        "family_scores": family_scores,
        "family_score_scale": "matched_control_signed_percentile_in_minus1_plus1",
        "winning_family": max(finite, key=finite.get) if finite else None,
        "fit_read_heldout_seizure": False,
        "fit_and_case_extraction_domain_identical": True,
        "control_endpoints_no_later_than_previous_seizure": True,
        "projection_fit_clean_interictal_only": True,
        "clean_interictal_exclusion_minutes": float(
            clean_interictal_exclusion_minutes
        ),
    }


def _effect(case: float, control: list[float]) -> dict[str, float | int | None]:
    values = np.asarray(control, dtype=np.float64)
    if not len(values):
        return {"case": float(case), "control_median": None, "effect_z": None,
                "percentile": None, "n_controls": 0}
    percentile = float((np.sum(values < case) + 0.5 * np.sum(values == case)) / len(values))
    centre = float(np.median(values))
    mad = float(np.median(np.abs(values - centre))) * 1.4826
    spread = max(mad, float(np.std(values)))
    floor = max(1e-8, 1e-6 * max(1.0, abs(centre)))
    z = float((case - centre) / spread) if spread > floor else None
    return {
        "case": float(case), "control_median": centre,
        "effect_z": z, "percentile": percentile,
        "signed_percentile": float(2.0 * (percentile - 0.5)),
        "control_spread": spread, "control_scale_degenerate": spread <= floor,
        "n_controls": int(len(values)),
    }


def evaluate_oos_geometry_fold(
    *,
    interictal_time: np.ndarray,
    interictal_session: np.ndarray,
    interictal_decoder: np.ndarray,
    risk_time: np.ndarray,
    risk_segment: np.ndarray,
    risk_decoder: np.ndarray,
    onset_time: np.ndarray,
    onset_segment: np.ndarray,
    heldout_position: int,
    lookback_minutes: float = 30.0,
    maximum_controls: int = 20,
) -> dict[str, Any]:
    onset = np.asarray(onset_time, dtype=np.float64)
    onset_group = np.asarray(onset_segment, dtype=np.int64)
    order = np.argsort(onset, kind="stable")
    onset, onset_group = onset[order], onset_group[order]
    position = int(heldout_position)
    if position < 2 or position >= len(onset):
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_prior_seizures"}
    cutoff = float(onset[position - 1])
    heldout = float(onset[position])
    heldout_segment = int(onset_group[position])
    train = np.flatnonzero(np.asarray(interictal_time, dtype=np.float64) <= cutoff)
    if len(train) < 100:
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_past_interictal_rows"}
    projection = fit_decoder_projection(np.asarray(interictal_decoder)[train])
    train_score = projection.transform(np.asarray(interictal_decoder)[train])
    risk_score = projection.transform(np.asarray(risk_decoder))
    centres = fit_two_basins(train_score)

    prior_end: list[np.ndarray] = []
    prior_direction: list[np.ndarray] = []
    for event, label in zip(onset[:position], onset_group[:position]):
        rows = _trajectory_rows(
            risk_time, risk_segment, endpoint=float(event), label=int(label),
            lookback_minutes=float(lookback_minutes),
        )
        if len(rows):
            prior_end.append(risk_score[rows[-1]])
            displacement = risk_score[rows[-1]] - risk_score[rows[0]]
            norm = float(np.linalg.norm(displacement))
            if norm > 1e-10:
                prior_direction.append(displacement / norm)
    if len(prior_end) < 2 or not prior_direction:
        return {"status": "NOT_ESTIMABLE", "reason": "insufficient_prior_entry_trajectories"}
    prior_end_value = np.asarray(prior_end, dtype=np.float64)
    entry_label = assign_basins(prior_end_value, centres)
    entry_basin = int(np.bincount(entry_label, minlength=2).argmax())
    entry_centroid = np.mean(prior_end_value, axis=0)
    entry_direction = np.mean(np.asarray(prior_direction), axis=0)
    if float(np.linalg.norm(entry_direction)) <= 1e-10:
        return {"status": "NOT_ESTIMABLE", "reason": "entry_direction_cancels"}

    case_rows = _trajectory_rows(
        risk_time, risk_segment, endpoint=heldout, label=heldout_segment,
        lookback_minutes=float(lookback_minutes),
    )
    if not len(case_rows):
        return {"status": "NOT_ESTIMABLE", "reason": "heldout_trajectory_incomplete"}
    controls = matched_control_trajectories(
        np.asarray(interictal_time)[train], np.asarray(interictal_session)[train],
        case_onset=heldout, lookback_minutes=float(lookback_minutes),
        maximum_controls=int(maximum_controls),
    )
    if len(controls) < 5:
        return {"status": "NOT_ESTIMABLE", "reason": "fewer_than_five_matched_controls"}
    reference = train_score
    if len(reference) > 1500:
        take = np.linspace(0, len(reference) - 1, 1500).round().astype(np.int64)
        reference = reference[take]
    reference_nn = _reference_nn_distance(reference)
    common = {
        "centres": centres, "entry_basin": entry_basin,
        "entry_centroid": entry_centroid, "entry_direction": entry_direction,
        "reference_scores": reference, "reference_nn": reference_nn,
    }
    case = trajectory_features(
        np.asarray(risk_time)[case_rows], risk_score[case_rows], **common,
    )
    control_features = [
        trajectory_features(
            np.asarray(interictal_time)[train][rows], train_score[rows], **common,
        ) for rows in controls
    ]
    effects = {
        key: _effect(case[key], [row[key] for row in control_features])
        for key in case
    }
    basin_values = [
        effects[key]["signed_percentile"] for key in (
            "entry_basin_occupancy", "entry_basin_longest_dwell_minutes",
        )
    ]
    approach_values = [
        effects[key]["signed_percentile"] for key in (
            "approach_rate_per_minute", "flow_alignment",
        )
    ]
    family_scores = {
        "basin_gating": float(np.mean(basin_values)) if basin_values else None,
        "directed_approach": float(np.mean(approach_values)) if approach_values else None,
        "abrupt_transition": effects["max_off_manifold_z"]["signed_percentile"],
    }
    finite = {key: value for key, value in family_scores.items() if value is not None}
    return {
        "status": "COMPLETE_EXPLORATORY", "heldout_position": position,
        "heldout_onset_epoch": heldout, "heldout_segment": heldout_segment,
        "train_cutoff_epoch": cutoff, "n_past_interictal_rows": int(len(train)),
        "n_prior_entry_trajectories": int(len(prior_end)),
        "n_controls": int(len(controls)), "lookback_minutes": float(lookback_minutes),
        "projection_active_dimensions": int(np.sum(projection.active)),
        "projection_components": int(len(projection.loadings)),
        "case_features": case, "effects": effects,
        "family_scores": family_scores,
        "family_score_scale": "matched_control_signed_percentile_in_minus1_plus1",
        "winning_family": max(finite, key=finite.get) if finite else None,
        "fit_read_heldout_seizure": False,
    }
