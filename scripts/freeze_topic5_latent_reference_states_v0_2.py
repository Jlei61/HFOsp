#!/usr/bin/env python3
"""Freeze response-blind Pass 2 reference states, controls, and N0 support."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    canonical_json_sha256,
    load_frozen_cell,
    parameter_state_sha256,
    parse_bool,
    rank_matrix_to_event_fields,
    sha256_file,
)
from src.topic5_latent_pass1_v0_2 import (  # noqa: E402
    build_future_field_data,
    event_first_phase_balanced_weights,
    interpolate_phase_vectors,
    observable_design,
    orthogonalize_field_axis,
    spline_basis,
    spline_derivative,
    weighted_ridge,
)
from src.topic5_latent_perturbation_v0_2 import (  # noqa: E402
    DOSES,
    NODE_RANGE_TOLERANCE_FLOOR,
    NODE_RANGE_TOLERANCE_FRACTION,
    PHASE_TARGETS,
    PRIMARY_DOSE,
    SUPPORT_K,
    centered_unit_field,
    jaccard,
    local_residual_normal_directions,
    residual_covariance_direction_sd,
    stable_seed,
    support_flags,
    unit_vector,
)
from scripts.run_topic5_latent_pass1_v0_2 import (  # noqa: E402
    ANALYSIS_REVISION,
    OUT,
    PARENT,
    SYSTEM,
    cell_dir,
    replay_states,
)


PASS2 = OUT / "axis_perturbation"
REFERENCE = PASS2 / "reference_freeze"
FREEZE_REVISION = "PASS2_FREEZE_R0_RESPONSE_BLIND_LOCAL_SUPPORT"
CONTROL_NAMES = tuple(
    [f"LOCAL_NORMAL_{index}" for index in range(8)]
    + ["PHASE_SHUFFLED_PROGRESS", "PHASE_SHUFFLED_FIELD"]
    + [f"PCA_{index + 1}" for index in range(3)]
    + ["C_SUFFIX_PROGRESS", "C_SUFFIX_FIELD"]
)
MAX_CHORDS_PER_STATE = 5
SUPPORT_LANDMARKS_PER_PHASE = 512
_FIELD_CACHE: dict[str, object] = {}
_CHORD_QUANTILE_CACHE: dict[str, tuple[float, float]] = {}
_OUTPUT_AXIS_CACHE: dict[str, tuple[np.ndarray, np.ndarray, bool]] = {}
_SAMPLE_MANIFEST_SHA256: str | None = None


def reference_dir(row: pd.Series) -> Path:
    return REFERENCE / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def robust_columns(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values, axis=0)
    mad = 1.4826 * np.median(np.abs(values - center[None, :]), axis=0)
    sd = np.std(values, axis=0, ddof=0)
    bad = (~np.isfinite(mad)) | (mad <= 1e-8)
    mad[bad] = sd[bad]
    mad[(~np.isfinite(mad)) | (mad <= 1e-8)] = 1.0
    return center, mad


def response_axes(
    ranks: np.ndarray,
    split: np.ndarray,
    labels: np.ndarray,
    positive_mode: int,
    negative_mode: int,
    field_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, bool]:
    _, recurrence = rank_matrix_to_event_fields(ranks)
    means = []
    for mode in (int(positive_mode), int(negative_mode)):
        with np.errstate(invalid="ignore"):
            value = np.nanmean(recurrence[(split == 0) & (labels == mode)], axis=0)
        means.append(value)
    progress, progress_ok = centered_unit_field((means[0] + means[1]) / 2.0)
    field, field_ok = centered_unit_field(field_axis)
    return progress, field, bool(progress_ok and field_ok)


def phase_shuffled_axes(
    states: dict[str, np.ndarray],
    event_u: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    knots: tuple[float, ...],
    alpha: float,
    key: str,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train = states["split"] == 0
    weights = event_first_phase_balanced_weights(
        states["event_index"], states["split"], states["phase_bin"]
    )
    y = (states["hidden"].astype(np.float64) - center[None, :]) / scale[None, :]
    u = event_u[states["event_index"]]
    rng = np.random.default_rng(stable_seed(key, "phase_shuffle"))
    shuffled_phase = states["phase"].copy()
    shuffled_phase[train] = shuffled_phase[train][rng.permutation(np.flatnonzero(train).size)]
    basis = spline_basis(shuffled_phase, knots)
    design = np.column_stack([basis, u[:, None] * basis])
    coefficient = weighted_ridge(design[train], y[train], weights[train], alpha)
    grid_basis = spline_basis(grid, knots)
    grid_derivative = spline_derivative(grid, knots)
    n_basis = grid_basis.shape[1]
    progress_raw = (grid_derivative @ coefficient[:n_basis]) * scale[None, :]
    field_raw = (grid_basis @ coefficient[n_basis:]) * scale[None, :]
    progress = np.full_like(progress_raw, np.nan)
    field = np.full_like(field_raw, np.nan)
    for index in range(len(grid)):
        progress[index], _ = unit_vector(progress_raw[index])
        field[index], _ = orthogonalize_field_axis(progress_raw[index], field_raw[index])
    return progress, field


def chord_quantiles(fit_id: str, field_data: object, split: np.ndarray) -> tuple[float, float]:
    if fit_id in _CHORD_QUANTILE_CACHE:
        return _CHORD_QUANTILE_CACHE[fit_id]
    values = np.asarray(field_data.event_coordinate_z)[split == 1]
    differences = np.abs(values[:, None] - values[None, :])
    differences = differences[np.triu_indices(len(values), 1)]
    finite = differences[np.isfinite(differences)]
    if not len(finite):
        raise RuntimeError("validation field-coordinate pair distribution is empty")
    result = (float(np.quantile(finite, 0.25)), float(np.quantile(finite, 0.75)))
    _CHORD_QUANTILE_CACHE[fit_id] = result
    return result


def select_references(states: dict[str, np.ndarray], selected_events: np.ndarray) -> np.ndarray:
    lookup = {
        (int(event), int(step)): index
        for index, (event, step) in enumerate(zip(states["event_index"], states["step"]))
    }
    rows: list[int] = []
    for event in selected_events:
        use = np.flatnonzero(states["event_index"] == int(event))
        if not len(use):
            continue
        event_phase = states["phase"][use]
        # The terminal state has no teacher-forced future and is not a reference.
        legal = use[states["step"][use] < states["step"][use].max()]
        for target in PHASE_TARGETS:
            picked = legal[np.argmin(np.abs(states["phase"][legal] - target))]
            rows.append(int(picked))
    return np.asarray(rows, dtype=int)


def run_cell(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_hash = parameter_state_sha256(model)
    decoder_hash = parameter_state_sha256(decoder)
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as source:
        ranks = np.asarray(source["ranks"])
        split = np.asarray(source["split"])
        labels = np.asarray(source["full_train_mode"])
    if str(row.fit_id) not in _FIELD_CACHE:
        _FIELD_CACHE[str(row.fit_id)] = build_future_field_data(
            ranks, split, labels,
            positive_mode=int(eligibility["positive_mode"]),
            negative_mode=int(eligibility["negative_mode"]),
            tier=str(eligibility["status"]),
            shuffle_key=str(row.fit_id),
        )
    field_data = _FIELD_CACHE[str(row.fit_id)]
    states = replay_states(model, ranks, split, sample, device, batch_size)
    reference_events = sample[sample["pass2_reference_event"].map(parse_bool)][
        "event_array_index"
    ].to_numpy(int)
    reference_rows = select_references(states, reference_events)
    if not len(reference_rows):
        raise RuntimeError("no frozen reference states")

    pass1 = cell_dir(row)
    metrics = json.loads((pass1 / "metrics.json").read_text())
    if metrics.get("analysis_revision") != ANALYSIS_REVISION:
        raise RuntimeError("Pass 1 analysis revision mismatch")
    with np.load(pass1 / "geometry_arrays.npz", allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    with np.load(pass1 / "conditional_manifold_arrays.npz", allow_pickle=False) as source:
        branch_grid = np.asarray(source["field_direction_raw"], dtype=np.float64)
    grid = arrays["phase_grid"].astype(np.float64)
    center = arrays["robust_center"].astype(np.float64)
    scale = arrays["robust_scale"].astype(np.float64)
    gamma_all = interpolate_phase_vectors(grid, arrays["gamma_raw"], states["phase"])
    branch_all = interpolate_phase_vectors(grid, branch_grid, states["phase"])
    u_all = field_data.event_coordinate_z[states["event_index"]]
    conditional_all = gamma_all + u_all[:, None] * branch_all
    residual_all = states["hidden"].astype(np.float64) - conditional_all
    progress_all = interpolate_phase_vectors(grid, arrays["progress_axes_raw"], states["phase"])
    field_all = interpolate_phase_vectors(grid, arrays["field_axes_raw"], states["phase"])
    observables = observable_design(
        states["step"], int(row.n_contacts), states["x"], states["recruited"]
    )[:, 1:]
    feature_raw = np.column_stack([
        np.einsum("ij,ij->i", residual_all, progress_all),
        np.einsum("ij,ij->i", residual_all, field_all),
        observables,
    ])
    train_validation = states["split"] <= 1
    node_min = states["hidden"][train_validation].min(axis=0).astype(np.float64)
    node_max = states["hidden"][train_validation].max(axis=0).astype(np.float64)
    node_tolerance = np.maximum(
        NODE_RANGE_TOLERANCE_FRACTION * (node_max - node_min),
        NODE_RANGE_TOLERANCE_FLOOR,
    )
    node_lower, node_upper = node_min - node_tolerance, node_max + node_tolerance

    neighbor_models: dict[int, NearestNeighbors] = {}
    feature_centers: list[np.ndarray] = []
    feature_scales: list[np.ndarray] = []
    knn_q95: list[float] = []
    residual_q95: list[float] = []
    for phase_bin in range(5):
        use = train_validation & (states["phase_bin"] == phase_bin)
        local = feature_raw[use]
        fcenter, fscale = robust_columns(local)
        train_rows = np.flatnonzero((states["split"] == 0) & (states["phase_bin"] == phase_bin))
        validation_rows = np.flatnonzero(
            (states["split"] == 1) & (states["phase_bin"] == phase_bin)
        )
        ordered = sorted(
            train_rows.tolist(),
            key=lambda state_row: (
                hashlib.sha256(
                    f"{row.fit_id}\0{int(states['event_index'][state_row])}\0"
                    f"{int(states['step'][state_row])}".encode()
                ).hexdigest(),
                state_row,
            ),
        )[:SUPPORT_LANDMARKS_PER_PHASE]
        landmarks = (feature_raw[np.asarray(ordered, dtype=int)] - fcenter[None, :]) / fscale[None, :]
        validation_features = (
            feature_raw[validation_rows] - fcenter[None, :]
        ) / fscale[None, :]
        neighbors = NearestNeighbors(
            n_neighbors=min(SUPPORT_K, len(landmarks)), algorithm="auto"
        )
        neighbors.fit(landmarks)
        distances = neighbors.kneighbors(validation_features, return_distance=True)[0][:, -1]
        neighbor_models[phase_bin] = neighbors
        feature_centers.append(fcenter)
        feature_scales.append(fscale)
        knn_q95.append(float(np.quantile(distances, 0.95)))
        residual_q95.append(float(np.quantile(
            np.linalg.norm(residual_all[use], axis=1) / np.sqrt(residual_all.shape[1]), 0.95
        )))
    feature_centers_array = np.stack(feature_centers)
    feature_scales_array = np.stack(feature_scales)

    knots = tuple(metrics["model_selection"]["PF"]["knots"])
    alpha = float(metrics["model_selection"]["PF"]["alpha"])
    shuffle_progress_grid, shuffle_field_grid = phase_shuffled_axes(
        states, field_data.event_coordinate_z, center, scale, knots, alpha,
        f"{row.fit_id}/{row.public_arm}/{int(row.seed)}", grid,
    )
    cs_arrays = None
    if str(row.public_arm) != "C-suffix":
        cs_path = SYSTEM / "per_cell" / str(row.fit_id) / "C-suffix" / f"seed{int(row.seed)}" / "geometry_arrays.npz"
        with np.load(cs_path, allow_pickle=False) as source:
            cs_arrays = {key: np.asarray(source[key]) for key in source.files}

    ref_h = states["hidden"][reference_rows].astype(np.float64)
    ref_phase = states["phase"][reference_rows].astype(np.float64)
    ref_bins = states["phase_bin"][reference_rows].astype(int)
    ref_progress = interpolate_phase_vectors(grid, arrays["progress_axes_raw"], ref_phase)
    ref_field = interpolate_phase_vectors(grid, arrays["field_axes_raw"], ref_phase)
    ref_gamma = interpolate_phase_vectors(grid, arrays["gamma_raw"], ref_phase)
    ref_branch = interpolate_phase_vectors(grid, branch_grid, ref_phase)
    ref_u = field_data.event_coordinate_z[states["event_index"][reference_rows]]
    ref_conditional = ref_gamma + ref_u[:, None] * ref_branch
    ref_feature_raw = feature_raw[reference_rows].copy()

    n_ref, hidden_dim = ref_h.shape
    axis_directions = np.stack([ref_progress, ref_field], axis=1)
    axis_sd = np.full((n_ref, 2), np.nan)
    controls = np.full((n_ref, len(CONTROL_NAMES), hidden_dim), np.nan, dtype=np.float64)
    control_sd = np.full((n_ref, len(CONTROL_NAMES)), np.nan)
    for index in range(n_ref):
        b = ref_bins[index]
        values = arrays["local_residual_eigenvalues"][b]
        components = arrays["local_residual_components"][b]
        diagonal = arrays["local_residual_diagonal"][b]
        for axis_index in range(2):
            axis_sd[index, axis_index] = residual_covariance_direction_sd(
                axis_directions[index, axis_index], values, components, diagonal
            )
        normal = local_residual_normal_directions(
            components, ref_progress[index], ref_field[index], 8
        )
        shuffled_progress = interpolate_phase_vectors(
            grid, shuffle_progress_grid, np.asarray([ref_phase[index]])
        )[0]
        shuffled_field = interpolate_phase_vectors(
            grid, shuffle_field_grid, np.asarray([ref_phase[index]])
        )[0]
        candidates = list(normal) + [shuffled_progress, shuffled_field]
        candidates += [arrays["pca_components"][:, p] for p in range(3)]
        if cs_arrays is None:
            candidates += [np.full(hidden_dim, np.nan), np.full(hidden_dim, np.nan)]
        else:
            candidates += [
                interpolate_phase_vectors(grid, cs_arrays["progress_axes_raw"], np.asarray([ref_phase[index]]))[0],
                interpolate_phase_vectors(grid, cs_arrays["field_axes_raw"], np.asarray([ref_phase[index]]))[0],
            ]
        for control_index, candidate in enumerate(candidates):
            vector, valid = unit_vector(candidate)
            if valid:
                controls[index, control_index] = vector
                control_sd[index, control_index] = residual_covariance_direction_sd(
                    vector, values, components, diagonal
                )

    def evaluate_support_batch(
        hidden: np.ndarray, reference_index: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        hidden = np.asarray(hidden, dtype=np.float64)
        reference_index = np.asarray(reference_index, dtype=int)
        delta = hidden - ref_h[reference_index]
        raw = ref_feature_raw[reference_index].copy()
        raw[:, 0] += np.einsum("ij,ij->i", delta, ref_progress[reference_index])
        raw[:, 1] += np.einsum("ij,ij->i", delta, ref_field[reference_index])
        bins = ref_bins[reference_index]
        feature = (raw - feature_centers_array[bins]) / feature_scales_array[bins]
        residual_norm = np.linalg.norm(
            hidden - ref_conditional[reference_index], axis=1
        ) / np.sqrt(hidden_dim)
        checks = np.zeros((len(hidden), 3), dtype=np.uint8)
        checks[:, 0] = (
            np.all(hidden >= node_lower[None, :], axis=1)
            & np.all(hidden <= node_upper[None, :], axis=1)
        )
        checks[:, 2] = residual_norm <= np.asarray(residual_q95)[bins]
        distance = np.full(len(hidden), np.nan, dtype=np.float64)
        finite = np.isfinite(hidden).all(axis=1) & np.isfinite(feature).all(axis=1)
        for b in range(5):
            use = np.flatnonzero((bins == b) & finite)
            if len(use):
                distance[use] = neighbor_models[b].kneighbors(
                    feature[use], return_distance=True
                )[0][:, -1]
        checks[:, 1] = distance <= np.asarray(knn_q95)[bins]
        checks[~finite] = 0
        return checks, distance

    signs = np.asarray([-1.0, 1.0], dtype=np.float64)
    axis_hidden = (
        ref_h[:, None, None, None, :]
        + axis_directions[:, :, None, None, :]
        * axis_sd[:, :, None, None, None]
        * DOSES[None, None, :, None, None]
        * signs[None, None, None, :, None]
    )
    axis_reference = np.broadcast_to(
        np.arange(n_ref)[:, None, None, None], axis_hidden.shape[:-1]
    )
    axis_checks_flat, axis_knn_flat = evaluate_support_batch(
        axis_hidden.reshape(-1, hidden_dim), axis_reference.reshape(-1)
    )
    axis_checks = axis_checks_flat.reshape(*axis_hidden.shape[:-1], 3)
    axis_knn = axis_knn_flat.reshape(axis_hidden.shape[:-1]).astype(np.float32)

    control_hidden = (
        ref_h[:, None, None, :]
        + controls[:, :, None, :] * control_sd[:, :, None, None]
        * PRIMARY_DOSE * signs[None, None, :, None]
    )
    control_reference = np.broadcast_to(
        np.arange(n_ref)[:, None, None], control_hidden.shape[:-1]
    )
    control_checks_flat, control_knn_flat = evaluate_support_batch(
        control_hidden.reshape(-1, hidden_dim), control_reference.reshape(-1)
    )
    control_checks = control_checks_flat.reshape(*control_hidden.shape[:-1], 3)
    control_knn = control_knn_flat.reshape(control_hidden.shape[:-1]).astype(np.float32)

    low_gap, high_gap = chord_quantiles(str(row.fit_id), field_data, split)
    chord_rows: list[dict[str, object]] = []
    ref_event = states["event_index"][reference_rows]
    ref_step = states["step"][reference_rows]
    ref_x = states["x"][reference_rows]
    ref_recruited = states["recruited"][reference_rows]
    for index in range(n_ref):
        candidates: list[tuple[float, int, float]] = []
        for target in range(n_ref):
            if index == target or ref_event[index] == ref_event[target]:
                continue
            if abs(ref_phase[index] - ref_phase[target]) > 0.10 or abs(int(ref_step[index]) - int(ref_step[target])) > 1:
                continue
            if abs(float(ref_recruited[index].mean() - ref_recruited[target].mean())) > max(0.10, 1.0 / int(row.n_contacts)):
                continue
            if abs(int(ref_x[index].sum()) - int(ref_x[target].sum())) > 1:
                continue
            recruited_jaccard = jaccard(ref_recruited[index], ref_recruited[target])
            current_jaccard = jaccard(ref_x[index], ref_x[target])
            if recruited_jaccard < 0.50 or current_jaccard < 0.50:
                continue
            score = (
                abs(ref_phase[index] - ref_phase[target])
                + abs(float(ref_recruited[index].mean() - ref_recruited[target].mean()))
                + (1.0 - recruited_jaccard) + (1.0 - current_jaccard)
            )
            candidates.append((score, target, float(ref_u[target] - ref_u[index])))
        for family, predicate in (
            ("HIGH_U", lambda gap: abs(gap) >= high_gap),
            ("SMALL_U", lambda gap: 0 < abs(gap) <= low_gap),
        ):
            selected_candidates = sorted(
                [item for item in candidates if predicate(item[2])],
                key=lambda item: (item[0], int(ref_event[item[1]]), int(item[1])),
            )[:MAX_CHORDS_PER_STATE]
            for rank_index, (score, target, u_difference) in enumerate(selected_candidates):
                direction = ref_h[target] - ref_h[index]
                direction_norm = float(np.linalg.norm(direction))
                chord_rows.append({
                    "reference_index": index,
                    "target_reference_index": target,
                    "family": family,
                    "pair_rank": rank_index,
                    "observable_match_score": score,
                    "u_source": float(ref_u[index]),
                    "u_target": float(ref_u[target]),
                    "u_difference": u_difference,
                    "direction_norm": direction_norm,
                    "target_values_read": False,
                })
    chord_columns = [
        "reference_index", "target_reference_index", "family", "pair_rank",
        "observable_match_score", "u_source", "u_target", "u_difference",
        "direction_norm",
        *[f"support_eta_{dose:.2f}" for dose in DOSES],
        *[f"knn_eta_{dose:.2f}" for dose in DOSES],
        "target_values_read",
    ]
    chords = pd.DataFrame(chord_rows, columns=chord_columns)
    if len(chords):
        chord_reference = chords["reference_index"].to_numpy(int)
        chord_target = chords["target_reference_index"].to_numpy(int)
        chord_direction = ref_h[chord_target] - ref_h[chord_reference]
        chord_hidden = (
            ref_h[chord_reference, None, :]
            + DOSES[None, :, None] * chord_direction[:, None, :]
        )
        checks, distances = evaluate_support_batch(
            chord_hidden.reshape(-1, hidden_dim),
            np.repeat(chord_reference, len(DOSES)),
        )
        checks = checks.reshape(len(chords), len(DOSES), 3).all(axis=2)
        distances = distances.reshape(len(chords), len(DOSES))
        for dose_index, dose in enumerate(DOSES):
            chords[f"support_eta_{dose:.2f}"] = checks[:, dose_index]
            chords[f"knn_eta_{dose:.2f}"] = distances[:, dose_index]

    if str(row.fit_id) not in _OUTPUT_AXIS_CACHE:
        _OUTPUT_AXIS_CACHE[str(row.fit_id)] = response_axes(
            ranks, split, labels, int(eligibility["positive_mode"]),
            int(eligibility["negative_mode"]), field_data.axis,
        )
    output_progress, output_field, output_axes_ok = _OUTPUT_AXIS_CACHE[str(row.fit_id)]
    phase_target = np.asarray([
        PHASE_TARGETS[index % len(PHASE_TARGETS)] for index in range(n_ref)
    ], dtype=np.float64)
    frozen = {
        "reference_event_index": ref_event.astype(np.int64),
        "reference_state_row": reference_rows.astype(np.int64),
        "step": ref_step.astype(np.int16),
        "phase": ref_phase.astype(np.float32),
        "phase_target": phase_target.astype(np.float32),
        "phase_bin": ref_bins.astype(np.int8),
        "hidden": ref_h.astype(np.float32),
        "current_x": ref_x.astype(np.uint8),
        "recruited": ref_recruited.astype(np.uint8),
        "event_u": ref_u.astype(np.float32),
        "conditional_center": ref_conditional.astype(np.float32),
        "progress_axis": ref_progress.astype(np.float32),
        "field_axis": ref_field.astype(np.float32),
        "axis_local_sd": axis_sd.astype(np.float32),
        "axis_support_checks": axis_checks,
        "axis_knn_distance": axis_knn,
        "control_directions": controls.astype(np.float32),
        "control_local_sd": control_sd.astype(np.float32),
        "control_support_checks": control_checks,
        "control_knn_distance": control_knn,
        "node_lower": node_lower.astype(np.float32),
        "node_upper": node_upper.astype(np.float32),
        "feature_center_by_phase": feature_centers_array.astype(np.float32),
        "feature_scale_by_phase": feature_scales_array.astype(np.float32),
        "knn_q95_by_phase": np.asarray(knn_q95, dtype=np.float32),
        "manifold_residual_q95_by_phase": np.asarray(residual_q95, dtype=np.float32),
        "contact_progress_axis": output_progress.astype(np.float32),
        "contact_future_field_axis": output_field.astype(np.float32),
    }
    global _SAMPLE_MANIFEST_SHA256
    if _SAMPLE_MANIFEST_SHA256 is None:
        _SAMPLE_MANIFEST_SHA256 = sha256_file(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    input_contract_sha256 = canonical_json_sha256({
        "checkpoint": str(row.checkpoint_sha256),
        "pass1_arrays": sha256_file(pass1 / "geometry_arrays.npz"),
        "conditional": sha256_file(pass1 / "conditional_manifold_arrays.npz"),
        "sample": _SAMPLE_MANIFEST_SHA256,
    })
    manifest_rows = []
    for index in range(n_ref):
        part = chords[chords["reference_index"].eq(index)] if len(chords) else chords
        manifest_rows.append({
            "patient": str(row.patient), "fit_id": str(row.fit_id),
            "geometry_view": str(row.geometry_view), "public_arm": str(row.public_arm),
            "seed": int(row.seed), "reference_index": index,
            "event_array_index": int(ref_event[index]), "step": int(ref_step[index]),
            "phase": float(ref_phase[index]), "phase_target": float(phase_target[index]),
            "q_replay_key": f"{row.fit_id}/{row.public_arm}/seed{int(row.seed)}/event{int(ref_event[index])}/step{int(ref_step[index])}",
            "progress_axis_defined": bool(np.isfinite(ref_progress[index]).all() and axis_sd[index, 0] > 1e-8),
            "field_axis_defined": bool(np.isfinite(ref_field[index]).all() and axis_sd[index, 1] > 1e-8),
            "progress_primary_both_branches_support": bool(axis_checks[index, 0, 1].all()),
            "field_primary_both_branches_support": bool(axis_checks[index, 1, 1].all()),
            "n_control_directions_defined": int(np.isfinite(controls[index]).all(axis=1).sum()),
            "n_high_u_chords": int((part["family"] == "HIGH_U").sum()) if len(part) else 0,
            "n_small_u_chords": int((part["family"] == "SMALL_U").sum()) if len(part) else 0,
            "input_contract_sha256": input_contract_sha256,
            "target_values_read": False,
        })
    reference_manifest = pd.DataFrame(manifest_rows)
    profile = {
        "contract": "topic5_pass2_reference_cell_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "freeze_revision": FREEZE_REVISION,
        "status": "PASS",
        "patient": str(row.patient), "fit_id": str(row.fit_id),
        "public_arm": str(row.public_arm), "seed": int(row.seed),
        "n_reference_events": int(np.unique(ref_event).size),
        "n_reference_states": int(n_ref),
        "n_high_u_chords": int((chords["family"] == "HIGH_U").sum()) if len(chords) else 0,
        "n_small_u_chords": int((chords["family"] == "SMALL_U").sum()) if len(chords) else 0,
        "chord_validation_q25_abs_u": low_gap,
        "chord_validation_q75_abs_u": high_gap,
        "output_axes_defined": output_axes_ok,
        "support_k": SUPPORT_K,
        "support_landmarks_per_phase": SUPPORT_LANDMARKS_PER_PHASE,
        "support_landmark_source": "axis_train_identity_hash",
        "support_threshold_source": "axis_validation_q95",
        "node_range_tolerance_fraction": NODE_RANGE_TOLERANCE_FRACTION,
        "node_range_tolerance_floor": NODE_RANGE_TOLERANCE_FLOOR,
        "primary_dose_local_sd": PRIMARY_DOSE,
        "sensitivity_doses_local_sd": [float(value) for value in DOSES],
        "model_hash_unchanged": model_hash == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_hash == parameter_state_sha256(decoder),
        "target_values_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return frozen, chords, reference_manifest, profile


def write_cell(row: pd.Series, arrays: dict[str, np.ndarray], chords: pd.DataFrame, manifest: pd.DataFrame, profile: dict[str, object]) -> None:
    target = reference_dir(row)
    target.mkdir(parents=True, exist_ok=True)
    failure = target / "FAILURE.json"
    if failure.is_file():
        failure.replace(target / "RECOVERED_FAILURE.json")
    write_npz(target / "reference_contract.npz", arrays)
    atomic_write_csv(target / "chords.csv", chords)
    atomic_write_csv(target / "reference_manifest.csv", manifest)
    atomic_write_json(target / "profile.json", profile)
    atomic_write_json(target / "DONE.json", {
        "ok": True,
        "freeze_revision": FREEZE_REVISION,
        "contract_sha256": sha256_file(target / "reference_contract.npz"),
        "chords_sha256": sha256_file(target / "chords.csv"),
        "manifest_sha256": sha256_file(target / "reference_manifest.csv"),
        "profile_sha256": sha256_file(target / "profile.json"),
        "target_values_read": False,
    })


def aggregate(manifest: pd.DataFrame) -> dict[str, object]:
    frames, profiles, missing = [], [], []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = reference_dir(row)
        if not (target / "DONE.json").is_file():
            missing.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
            continue
        frames.append(pd.read_csv(target / "reference_manifest.csv"))
        profiles.append(json.loads((target / "profile.json").read_text()))
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        atomic_write_csv(REFERENCE / "REFERENCE_STATE_MANIFEST.csv", combined)
    payload = {
        "contract": "topic5_pass2_reference_freeze_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "freeze_revision": FREEZE_REVISION,
        "status": "PASS" if len(profiles) == 630 and not missing else "INCOMPLETE",
        "scheduled_cells": 630, "completed_cells": len(profiles),
        "reference_states": int(sum(int(item["n_reference_states"]) for item in profiles)),
        "high_u_chords": int(sum(int(item["n_high_u_chords"]) for item in profiles)),
        "small_u_chords": int(sum(int(item["n_small_u_chords"]) for item in profiles)),
        "output_axis_cells": int(sum(bool(item["output_axes_defined"]) for item in profiles)),
        "missing_count": len(missing), "missing_first20": missing[:20],
        "response_values_read_before_freeze": False,
        "target_values_read": False,
    }
    atomic_write_json(REFERENCE / "REFERENCE_FREEZE_STATUS.json", payload)
    if frames and payload["status"] == "PASS":
        atomic_write_json(REFERENCE / "REFERENCE_FREEZE_SEAL.json", {
            "sealed": True,
            "freeze_revision": FREEZE_REVISION,
            "manifest_sha256": sha256_file(REFERENCE / "REFERENCE_STATE_MANIFEST.csv"),
            "status_sha256": sha256_file(REFERENCE / "REFERENCE_FREEZE_STATUS.json"),
            "response_values_read_before_freeze": False,
            "target_values_read": False,
        })
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cell-key")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    transport_audit = json.loads((OUT / "dynamical_transport/TRANSPORT_AUDIT.json").read_text())
    if transport_audit.get("status") != "PASS":
        raise RuntimeError("transport audit must pass before response-blind Pass 2 freeze")
    manifest_all = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    manifest = manifest_all.copy()
    if args.cell_key:
        fit, arm, seed_text = args.cell_key.split("/")
        manifest = manifest[
            manifest["fit_id"].eq(fit) & manifest["public_arm"].eq(arm)
            & manifest["seed"].eq(int(seed_text.removeprefix("seed")))
        ]
    elif args.limit is not None:
        manifest = manifest.iloc[: args.limit]
    sample = pd.read_csv(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv").set_index("fit_id")
    device = torch.device(args.device)
    failures = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        target = reference_dir(row)
        if (target / "DONE.json").is_file() and not args.force:
            print(f"skip {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed}", flush=True)
            continue
        try:
            frozen, chords, ref_manifest, profile = run_cell(
                row, sample[sample["fit_id"].eq(row.fit_id)].copy(),
                eligibility.loc[row.fit_id], device, args.batch_size,
            )
            write_cell(row, frozen, chords, ref_manifest, profile)
            print(
                f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} "
                f"refs={profile['n_reference_states']} chords={profile['n_high_u_chords']}/"
                f"{profile['n_small_u_chords']} {profile['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as error:
            failures.append({
                "fit_id": row.fit_id, "public_arm": row.public_arm, "seed": int(row.seed),
                "error_type": type(error).__name__, "error": str(error),
            })
            atomic_write_json(target / "FAILURE.json", failures[-1])
            print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    status = aggregate(manifest_all)
    print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
