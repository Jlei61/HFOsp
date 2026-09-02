#!/usr/bin/env python3
"""Run Topic 5.2 Pass 1 geometry over the frozen 630-cell matrix."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    canonical_json_sha256,
    load_frozen_cell,
    parameter_state_sha256,
    sha256_file,
)
from src.topic5_latent_pass1_v0_2 import (  # noqa: E402
    PHASE_BINS,
    EMERGENCE_RIDGE_GRID,
    RIDGE_GRID,
    SPLINE_KNOT_SETS,
    build_future_field_data,
    event_first_phase_balanced_weights,
    observable_design,
    orthogonalize_field_axis,
    phase_bin,
    robust_center_scale,
    spline_basis,
    spline_derivative,
    teacher_forced_hidden,
    weighted_pca,
    weighted_r2,
    weighted_r2_scalar,
    weighted_ridge,
)
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


PARENT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OUT = ROOT / "results/topic5_latent_propagation_landscape_v0_2"
SYSTEM = OUT / "system_identification"
SPLIT_NAMES = {0: "axis_train", 1: "axis_validation", 2: "heldout_test"}
ANALYSIS_REVISION = "PASS1_R1_PHASEWISE_OBSERVABLE_RESIDUALIZED_EMERGENCE"


def _jsonable(value: object) -> object:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def cell_dir(row: pd.Series) -> Path:
    return SYSTEM / "per_cell" / str(row.fit_id) / str(row.public_arm) / f"seed{int(row.seed)}"


def replay_states(
    model: torch.nn.Module,
    ranks: np.ndarray,
    split: np.ndarray,
    selected: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    *,
    reverse_chunks: bool = False,
) -> dict[str, np.ndarray]:
    selected_indices = selected["event_array_index"].to_numpy(dtype=int)
    chunks = [
        selected_indices[start:start + batch_size]
        for start in range(0, len(selected_indices), batch_size)
    ]
    if reverse_chunks:
        chunks = list(reversed(chunks))
    output: dict[str, list[np.ndarray]] = {
        name: [] for name in (
            "hidden", "event_index", "step", "phase", "phase_bin", "split", "x", "recruited"
        )
    }
    for indices in chunks:
        tensors = build_event_tensors(ranks[indices])
        x_gpu = tensors["x"].to(device)
        hidden = teacher_forced_hidden(model, x_gpu).detach().cpu().numpy()
        x = tensors["x"].numpy()
        recruited = tensors["recruited"].numpy()
        valid = tensors["valid"].numpy()
        for local, event in enumerate(indices):
            length = int(valid[local].sum())
            if length < 2:
                continue
            steps = np.arange(length, dtype=np.int16)
            phase = steps.astype(np.float64) / (length - 1)
            output["hidden"].append(hidden[local, :length].astype(np.float32, copy=False))
            output["event_index"].append(np.full(length, event, dtype=np.int64))
            output["step"].append(steps)
            output["phase"].append(phase)
            output["phase_bin"].append(phase_bin(phase).astype(np.int8))
            output["split"].append(np.full(length, int(split[event]), dtype=np.int8))
            output["x"].append(x[local, :length].astype(np.uint8, copy=False))
            output["recruited"].append(recruited[local, :length].astype(np.uint8, copy=False))
    merged = {name: np.concatenate(values, axis=0) for name, values in output.items()}
    order = np.lexsort((merged["step"], merged["event_index"]))
    return {name: values[order] for name, values in merged.items()}


def select_multivariate_model(
    y: np.ndarray,
    phase: np.ndarray,
    u: np.ndarray,
    u_shuffled: np.ndarray,
    split: np.ndarray,
    weights: np.ndarray,
    kind: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    train = split == 0
    validation = split == 1
    best: tuple[float, tuple[float, ...], float, np.ndarray] | None = None
    for knots in SPLINE_KNOT_SETS:
        p = spline_basis(phase, knots)
        if kind == "P":
            design = p
        elif kind == "PF":
            design = np.column_stack([p, u[:, None] * p])
        elif kind == "PF_NULL":
            design = np.column_stack([p, u_shuffled[:, None] * p])
        else:
            raise ValueError(kind)
        for alpha in RIDGE_GRID:
            coefficient = weighted_ridge(design[train], y[train], weights[train], alpha)
            score = weighted_r2(y[validation], design[validation] @ coefficient, weights[validation])
            loss = -score if np.isfinite(score) else float("inf")
            candidate = (loss, tuple(knots), float(alpha), coefficient)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError(f"no valid {kind} model")
    _, knots, alpha, coefficient = best
    p = spline_basis(phase, knots)
    if kind == "P":
        design = p
    elif kind == "PF":
        design = np.column_stack([p, u[:, None] * p])
    else:
        design = np.column_stack([p, u_shuffled[:, None] * p])
    # Refit train only with frozen validation-selected hyperparameters.
    coefficient = weighted_ridge(design[train], y[train], weights[train], alpha)
    return {"knots": list(knots), "alpha": alpha, "n_features": design.shape[1]}, coefficient, design


def select_observable_model(
    y: np.ndarray, observables: np.ndarray, split: np.ndarray, weights: np.ndarray
) -> tuple[dict[str, object], np.ndarray]:
    train, validation = split == 0, split == 1
    best: tuple[float, float, np.ndarray] | None = None
    for alpha in RIDGE_GRID:
        coefficient = weighted_ridge(observables[train], y[train], weights[train], alpha)
        score = weighted_r2(
            y[validation], observables[validation] @ coefficient, weights[validation]
        )
        candidate = (-score if np.isfinite(score) else float("inf"), float(alpha), coefficient)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("no observable model")
    _, alpha, _ = best
    coefficient = weighted_ridge(observables[train], y[train], weights[train], alpha)
    return {"alpha": alpha, "n_features": observables.shape[1]}, coefficient


def output_r2(
    actual_z: np.ndarray,
    predicted_z: np.ndarray,
    weights: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    model: torch.nn.Module,
) -> float:
    actual_h = center[None, :] + actual_z * scale[None, :]
    predicted_h = center[None, :] + predicted_z * scale[None, :]
    h_operator = model.H.detach().cpu().numpy().T
    gain = float(model.readout_gain.detach().cpu().item())
    bias = model.contact_bias.detach().cpu().numpy()
    actual = bias[None, :] + gain * (actual_h @ h_operator)
    predicted = bias[None, :] + gain * (predicted_h @ h_operator)
    return weighted_r2(actual, predicted, weights)


def select_emergence_model(
    kind: str,
    phase_bin_id: int,
    z: np.ndarray,
    z_incremental: np.ndarray,
    observables: np.ndarray,
    u: np.ndarray,
    split: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray,
    dimensions: list[int],
) -> dict[str, object]:
    candidates: list[tuple[float, int, float]] = []
    dimensions_use = [0] if kind == "O" else dimensions
    train = (split == 0) & (bins == int(phase_bin_id))
    validation = (split == 1) & (bins == int(phase_bin_id))
    if int(train.sum()) < 5 or int(validation.sum()) < 3:
        return {
            "kind": kind,
            "phase_bin": int(phase_bin_id),
            "status": "NOT_IDENTIFIABLE",
            "dimension": 0,
            "alpha": None,
        }
    for dimension in dimensions_use:
        for alpha in EMERGENCE_RIDGE_GRID:
            if kind == "H":
                x = np.column_stack([np.ones(len(z)), z[:, :dimension]])
            elif kind == "O":
                x = observables
            elif kind == "OH":
                x = np.column_stack([observables, z_incremental[:, :dimension]])
            else:
                raise ValueError(kind)
            coefficient = weighted_ridge(x[train], u[train, None], weights[train], alpha)
            residual = u[validation] - (x[validation] @ coefficient).reshape(-1)
            loss = float(np.sum(weights[validation] * residual**2))
            candidates.append((loss, dimension, float(alpha)))
    if not candidates:
        return {"kind": kind, "status": "NOT_IDENTIFIABLE", "dimension": 0, "alpha": None}
    loss, dimension, alpha = min(candidates)
    return {
        "kind": kind,
        "phase_bin": int(phase_bin_id),
        "status": "OK",
        "validation_weighted_sse": loss,
        "dimension": int(dimension),
        "alpha": float(alpha),
    }


def emergence_rows(
    z: np.ndarray,
    z_incremental: np.ndarray,
    observables: np.ndarray,
    u: np.ndarray,
    split: np.ndarray,
    bins: np.ndarray,
    weights: np.ndarray,
    dimensions: list[int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    selections = {
        kind: [
            select_emergence_model(
                kind, b, z, z_incremental, observables, u, split, bins, weights, dimensions
            )
            for b in range(PHASE_BINS)
        ]
        for kind in ("H", "O", "OH")
    }
    rows: list[dict[str, object]] = []
    for b in range(PHASE_BINS):
        train = (split == 0) & (bins == b)
        test = (split == 2) & (bins == b)
        row: dict[str, object] = {
            "phase_bin": b,
            "phase_start": b / PHASE_BINS,
            "phase_end": (b + 1) / PHASE_BINS,
            "n_train_states": int(train.sum()),
            "n_test_states": int(test.sum()),
        }
        for kind, phase_selections in selections.items():
            selection = phase_selections[b]
            if selection["status"] != "OK" or int(train.sum()) < 5 or int(test.sum()) < 3:
                row[f"r2_{kind.lower()}"] = float("nan")
                continue
            dimension = int(selection["dimension"])
            if kind == "H":
                x = np.column_stack([np.ones(len(z)), z[:, :dimension]])
            elif kind == "O":
                x = observables
            else:
                x = np.column_stack([observables, z_incremental[:, :dimension]])
            coefficient = weighted_ridge(
                x[train], u[train, None], weights[train], float(selection["alpha"])
            )
            prediction = (x[test] @ coefficient).reshape(-1)
            row[f"r2_{kind.lower()}"] = weighted_r2_scalar(u[test], prediction, weights[test])
        row["incremental_r2_oh_minus_o"] = row["r2_oh"] - row["r2_o"]
        rows.append(row)
    return rows, selections


def analyse_cell(
    row: pd.Series,
    sample: pd.DataFrame,
    eligibility: pd.Series,
    device: torch.device,
    batch_size: int,
    *,
    audit_chunk_order: bool = False,
) -> tuple[dict[str, object], dict[str, np.ndarray], list[dict[str, object]], dict[str, object]]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_hash_before = parameter_state_sha256(model)
    decoder_hash_before = parameter_state_sha256(decoder)
    cache = PARENT / "cache" / str(row.fit_id)
    with np.load(cache / "events.npz", allow_pickle=False) as events:
        ranks = np.asarray(events["ranks"])
        split_event = np.asarray(events["split"])
        full_train_mode = np.asarray(events["full_train_mode"])
    field = build_future_field_data(
        ranks,
        split_event,
        full_train_mode,
        positive_mode=int(eligibility["positive_mode"]),
        negative_mode=int(eligibility["negative_mode"]),
        tier=str(eligibility["status"]),
        shuffle_key=str(row.fit_id),
    )
    states = replay_states(model, ranks, split_event, sample, device, batch_size)
    chunk_audit: dict[str, object] = {"run": False}
    if audit_chunk_order:
        reverse = replay_states(
            model, ranks, split_event, sample, device, batch_size, reverse_chunks=True
        )
        exact = all(np.array_equal(states[key], reverse[key], equal_nan=True) for key in states)
        max_abs = float(np.max(np.abs(states["hidden"] - reverse["hidden"])))
        chunk_audit = {"run": True, "exact": exact, "hidden_max_abs": max_abs}
        if not exact:
            raise RuntimeError(f"chunk-order replay drift for {row.fit_id}: {max_abs}")

    weights = event_first_phase_balanced_weights(
        states["event_index"], states["split"], states["phase_bin"]
    )
    train = states["split"] == 0
    validation = states["split"] == 1
    test = states["split"] == 2
    center, scale, constant = robust_center_scale(states["hidden"][train])
    y = (states["hidden"].astype(np.float64) - center[None, :]) / scale[None, :]
    u = field.event_coordinate_z[states["event_index"]]
    # Capacity-matched null: permute only the frozen sampled events, preserving
    # their coordinate distribution exactly within each split.
    shuffled_by_event: dict[int, float] = {}
    for split_id in (0, 1, 2):
        event_ids = np.sort(np.unique(states["event_index"][states["split"] == split_id]))
        digest = __import__("hashlib").sha256(
            f"{row.fit_id}\0pass1-selected-u-null\0{split_id}".encode("utf-8")
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        permuted = field.event_coordinate_z[rng.permutation(event_ids)]
        shuffled_by_event.update({int(event): float(value) for event, value in zip(event_ids, permuted)})
    u_shuffled = np.asarray([shuffled_by_event[int(event)] for event in states["event_index"]])
    observables = observable_design(
        states["step"], int(row.n_contacts), states["x"], states["recruited"]
    )

    p_selection, p_coef, p_design = select_multivariate_model(
        y, states["phase"], u, u_shuffled, states["split"], weights, "P"
    )
    pf_selection, pf_coef, pf_design = select_multivariate_model(
        y, states["phase"], u, u_shuffled, states["split"], weights, "PF"
    )
    # Capacity-matched null uses exactly the PF hyperparameters.
    pf_knots = tuple(pf_selection["knots"])
    p_for_null = spline_basis(states["phase"], pf_knots)
    pf_null_design = np.column_stack([p_for_null, u_shuffled[:, None] * p_for_null])
    pf_null_coef = weighted_ridge(
        pf_null_design[train], y[train], weights[train], float(pf_selection["alpha"])
    )
    o_selection, o_coef = select_observable_model(y, observables, states["split"], weights)

    pred_p = p_design @ p_coef
    pred_pf = pf_design @ pf_coef
    pred_pf_null = pf_null_design @ pf_null_coef
    pred_o = observables @ o_coef

    # Residualized-state sensitivity: remove O frozen on train, then refit P/PF.
    residual = y - pred_o
    rp_selection, rp_coef, rp_design = select_multivariate_model(
        residual, states["phase"], u, u_shuffled, states["split"], weights, "P"
    )
    rpf_selection, rpf_coef, rpf_design = select_multivariate_model(
        residual, states["phase"], u, u_shuffled, states["split"], weights, "PF"
    )
    pred_rp = rp_design @ rp_coef
    pred_rpf = rpf_design @ rpf_coef

    pca_mean, pca_values, pca_components = weighted_pca(y[train], weights[train], 16)
    pca_total_variance = float(np.sum(
        weights[train, None] * (y[train] - pca_mean[None, :]) ** 2
    ) / max(float(weights[train].sum()), 1e-12))
    z = (y - pca_mean[None, :]) @ pca_components
    # Incremental hidden features are train-only residuals after observables,
    # estimated separately within each phase bin.  This is the operational
    # meaning of O+hidden and avoids a collinear raw [O, Z] design.
    z_incremental = np.full_like(z, np.nan)
    for b in range(PHASE_BINS):
        use_train = train & (states["phase_bin"] == b)
        use_all = states["phase_bin"] == b
        coefficient = weighted_ridge(
            observables[use_train], z[use_train], weights[use_train], 1e-4
        )
        z_incremental[use_all] = z[use_all] - observables[use_all] @ coefficient
    if not np.isfinite(z_incremental).all():
        raise RuntimeError("observable-residualized PCA features are nonfinite")
    dimensions = [dimension for dimension in (2, 4, 8) if dimension <= z.shape[1]]
    emergence, emergence_selection = emergence_rows(
        z, z_incremental, observables, u, states["split"], states["phase_bin"],
        weights, dimensions
    )

    grid = np.linspace(0.0, 1.0, 21)
    basis_grid = spline_basis(grid, pf_knots)
    derivative_grid = spline_derivative(grid, pf_knots)
    n_basis = basis_grid.shape[1]
    gamma_z = basis_grid @ pf_coef[:n_basis]
    gamma_raw = center[None, :] + gamma_z * scale[None, :]
    progress_raw = (derivative_grid @ pf_coef[:n_basis]) * scale[None, :]
    field_raw = (basis_grid @ pf_coef[n_basis:]) * scale[None, :]
    progress_axes = np.full_like(progress_raw, np.nan)
    field_axes = np.full_like(field_raw, np.nan)
    collinear = np.zeros(len(grid), dtype=np.uint8)
    for index in range(len(grid)):
        norm = float(np.linalg.norm(progress_raw[index]))
        if norm > 1e-10:
            progress_axes[index] = progress_raw[index] / norm
        field_axes[index], bad = orthogonalize_field_axis(
            progress_raw[index], field_raw[index]
        )
        collinear[index] = int(bad)

    # Phase-local residual covariance compressed to its leading directions.
    residual_raw = (y - pred_pf) * scale[None, :]
    local_values = np.full((PHASE_BINS, min(16, y.shape[1])), np.nan, dtype=np.float64)
    local_components = np.full(
        (PHASE_BINS, y.shape[1], min(16, y.shape[1])), np.nan, dtype=np.float64
    )
    local_diagonal = np.full((PHASE_BINS, y.shape[1]), np.nan, dtype=np.float64)
    train_validation = train | validation
    combined_weights = weights.copy()
    combined_weights[train] *= 0.5
    combined_weights[validation] *= 0.5
    for b in range(PHASE_BINS):
        use = train_validation & (states["phase_bin"] == b)
        if int(use.sum()) < 5:
            continue
        _, values, components = weighted_pca(
            residual_raw[use], combined_weights[use], local_values.shape[1]
        )
        local_values[b, :len(values)] = values
        local_components[b, :, :components.shape[1]] = components
        mean_resid = np.sum(
            combined_weights[use, None] * residual_raw[use], axis=0
        ) / max(float(combined_weights[use].sum()), 1e-12)
        local_diagonal[b] = np.sum(
            combined_weights[use, None] * (residual_raw[use] - mean_resid) ** 2,
            axis=0,
        ) / max(float(combined_weights[use].sum()), 1e-12)

    metrics: dict[str, object] = {
        "contract": "topic5_latent_pass1_cell_v0_2",
        "analysis_revision": ANALYSIS_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "patient": str(row.patient),
        "fit_id": str(row.fit_id),
        "geometry_view": str(row.geometry_view),
        "public_arm": str(row.public_arm),
        "seed": int(row.seed),
        "checkpoint_source": str(row.checkpoint_source),
        "field_axis_tier": field.tier,
        "canonical_ab": bool(eligibility["canonical_ab"]),
        "n_common_field_contacts": field.n_common_contacts,
        "field_contrast_norm": field.contrast_norm,
        "field_coordinate_train_mean": field.train_coordinate_mean,
        "field_coordinate_train_scale": field.train_coordinate_scale,
        "n_events": {
            SPLIT_NAMES[key]: int(sample["split"].eq(key).sum()) for key in SPLIT_NAMES
        },
        "n_states": {
            SPLIT_NAMES[key]: int(np.count_nonzero(states["split"] == key)) for key in SPLIT_NAMES
        },
        "constant_hidden_nodes": int(constant.sum()),
        "model_selection": {
            "O": o_selection,
            "P": p_selection,
            "PF": pf_selection,
            "residual_P": rp_selection,
            "residual_PF": rpf_selection,
            "emergence": emergence_selection,
        },
        "heldout_geometry": {
            "r2_O": weighted_r2(y[test], pred_o[test], weights[test]),
            "r2_P": weighted_r2(y[test], pred_p[test], weights[test]),
            "r2_PF": weighted_r2(y[test], pred_pf[test], weights[test]),
            "r2_PF_null": weighted_r2(y[test], pred_pf_null[test], weights[test]),
            "delta_PF_minus_P": weighted_r2(y[test], pred_pf[test], weights[test])
            - weighted_r2(y[test], pred_p[test], weights[test]),
            "delta_PF_minus_PF_null": weighted_r2(y[test], pred_pf[test], weights[test])
            - weighted_r2(y[test], pred_pf_null[test], weights[test]),
            "output_r2_P": output_r2(y[test], pred_p[test], weights[test], center, scale, model),
            "output_r2_PF": output_r2(y[test], pred_pf[test], weights[test], center, scale, model),
            "residual_r2_P": weighted_r2(residual[test], pred_rp[test], weights[test]),
            "residual_r2_PF": weighted_r2(residual[test], pred_rpf[test], weights[test]),
            "residual_delta_PF_minus_P": weighted_r2(
                residual[test], pred_rpf[test], weights[test]
            ) - weighted_r2(residual[test], pred_rp[test], weights[test]),
        },
        "pca": {
            "eigenvalues": pca_values.tolist(),
            "variance_fraction_top8": float(
                pca_values[:8].sum() / max(pca_total_variance, 1e-12)
            ),
            "total_weighted_variance": pca_total_variance,
        },
        "axis": {
            "grid_points": len(grid),
            "field_collinear_grid_points": int(collinear.sum()),
            "progress_defined_grid_points": int(np.isfinite(progress_axes).all(axis=1).sum()),
            "field_defined_grid_points": int(np.isfinite(field_axes).all(axis=1).sum()),
        },
        "chunk_order_audit": chunk_audit,
        "model_hash_unchanged": model_hash_before == parameter_state_sha256(model),
        "decoder_hash_unchanged": decoder_hash_before == parameter_state_sha256(decoder),
        "checkpoint_sha256": str(row.checkpoint_sha256),
        "split_sha256": str(row.split_sha256),
        "sample_manifest_sha256": sha256_file(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv"),
        "mode_axis_manifest_sha256": sha256_file(OUT / "MODE_AXIS_ELIGIBILITY.csv"),
        "target_values_read": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    arrays = {
        "robust_center": center.astype(np.float32),
        "robust_scale": scale.astype(np.float32),
        "constant_hidden_mask": constant.astype(np.uint8),
        "pca_mean": pca_mean.astype(np.float32),
        "pca_eigenvalues": pca_values.astype(np.float32),
        "pca_components": pca_components.astype(np.float32),
        "phase_grid": grid.astype(np.float32),
        "gamma_raw": gamma_raw.astype(np.float32),
        "progress_axes_raw": progress_axes.astype(np.float32),
        "field_axes_raw": field_axes.astype(np.float32),
        "field_axis_collinear": collinear,
        "local_residual_eigenvalues": local_values.astype(np.float32),
        "local_residual_components": local_components.astype(np.float32),
        "local_residual_diagonal": local_diagonal.astype(np.float32),
        "contact_future_field_axis": field.axis.astype(np.float32),
        "contact_train_mean_field": field.train_mean_field.astype(np.float32),
    }
    for item in emergence:
        item.update({
            "patient": str(row.patient),
            "fit_id": str(row.fit_id),
            "geometry_view": str(row.geometry_view),
            "public_arm": str(row.public_arm),
            "seed": int(row.seed),
            "field_axis_tier": field.tier,
            "canonical_ab": bool(eligibility["canonical_ab"]),
        })
    return _jsonable(metrics), arrays, [_jsonable(item) for item in emergence], chunk_audit


def write_cell(
    row: pd.Series,
    metrics: dict[str, object],
    arrays: dict[str, np.ndarray],
    emergence: list[dict[str, object]],
) -> None:
    target = cell_dir(row)
    target.mkdir(parents=True, exist_ok=True)
    temporary = target / "geometry_arrays.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(target / "geometry_arrays.npz")
    atomic_write_json(target / "metrics.json", metrics)
    atomic_write_csv(target / "emergence.csv", pd.DataFrame(emergence))
    done = {
        "ok": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_sha256": sha256_file(target / "metrics.json"),
        "arrays_sha256": sha256_file(target / "geometry_arrays.npz"),
        "emergence_sha256": sha256_file(target / "emergence.csv"),
        "target_values_read": False,
    }
    atomic_write_json(target / "DONE.json", done)


def aggregate_outputs(manifest: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    emergence: list[pd.DataFrame] = []
    failures: list[str] = []
    for item in manifest.itertuples(index=False):
        row = pd.Series(item._asdict())
        target = cell_dir(row)
        done_path = target / "DONE.json"
        if not done_path.is_file():
            failures.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}")
            continue
        done = json.loads(done_path.read_text())
        if not bool(done.get("ok")):
            failures.append(f"{item.fit_id}/{item.public_arm}/seed{item.seed}:not_ok")
            continue
        metrics = json.loads((target / "metrics.json").read_text())
        geometry = metrics.pop("heldout_geometry")
        rows.append({**metrics, **geometry})
        emergence.append(pd.read_csv(target / "emergence.csv"))
    if rows:
        frame = pd.DataFrame(rows)
        # Nested dictionaries are retained in per-cell JSON, not flattened into CSV.
        drop = [column for column in frame if frame[column].map(lambda x: isinstance(x, (dict, list))).any()]
        atomic_write_csv(SYSTEM / "PASS1_CELL_GEOMETRY.csv", frame.drop(columns=drop))
    if emergence:
        atomic_write_csv(SYSTEM / "PASS1_FUTURE_FIELD_EMERGENCE.csv", pd.concat(emergence, ignore_index=True))
    complete = len(rows) == 630 and not failures
    payload = {
        "contract": "topic5_latent_pass1_execution_v0_2",
        "analysis_revision": ANALYSIS_REVISION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if complete else "INCOMPLETE",
        "scheduled_cells": 630,
        "completed_cells": len(rows),
        "missing_or_failed_count": len(failures),
        "missing_or_failed_first20": failures[:20],
        "target_values_read": False,
    }
    atomic_write_json(SYSTEM / "PASS1_EXECUTION_STATUS.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--cell-key", help="fit_id/public_arm/seed")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--audit-chunk-order", action="store_true")
    args = parser.parse_args()
    manifest_all = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    manifest = manifest_all.copy()
    if args.cell_key:
        fit_id, arm, seed_text = args.cell_key.split("/")
        seed = int(seed_text.removeprefix("seed"))
        manifest = manifest[
            manifest["fit_id"].eq(fit_id)
            & manifest["public_arm"].eq(arm)
            & manifest["seed"].eq(seed)
        ]
    else:
        manifest = manifest.iloc[args.offset:]
        if args.limit is not None:
            manifest = manifest.iloc[:args.limit]
    if manifest.empty:
        raise RuntimeError("no cells selected")
    samples = pd.read_csv(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv")
    eligibility = pd.read_csv(OUT / "MODE_AXIS_ELIGIBILITY.csv").set_index("fit_id")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    failures: list[dict[str, object]] = []
    for position, (_, row) in enumerate(manifest.iterrows(), start=1):
        target = cell_dir(row)
        if (target / "DONE.json").is_file() and not args.force:
            print(f"skip {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed}", flush=True)
            continue
        try:
            metrics, arrays, emergence, chunk_audit = analyse_cell(
                row,
                samples[samples["fit_id"].eq(row.fit_id)].copy(),
                eligibility.loc[row.fit_id],
                device,
                args.batch_size,
                audit_chunk_order=args.audit_chunk_order and position == 1,
            )
            write_cell(row, metrics, arrays, emergence)
            print(
                f"done {position}/{len(manifest)} {row.fit_id}/{row.public_arm}/seed{row.seed} "
                f"{metrics['elapsed_seconds']:.2f}s",
                flush=True,
            )
        except Exception as error:  # keep a complete provenance-bearing failure ledger
            failures.append({
                "fit_id": row.fit_id,
                "public_arm": row.public_arm,
                "seed": int(row.seed),
                "error_type": type(error).__name__,
                "error": str(error),
            })
            atomic_write_json(target / "FAILURE.json", failures[-1])
            print(f"FAIL {row.fit_id}/{row.public_arm}/seed{row.seed}: {error}", flush=True)
    status = aggregate_outputs(manifest_all)
    print(json.dumps({"run_failures": failures, "aggregate": status}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
