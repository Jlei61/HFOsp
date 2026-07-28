"""Seed-specific, observation-only modal routing for Z/M Phase C.

This module never advances the SNN and never assigns a Phase-C phenotype.
It consumes immutable C0/C1 classifications and routes each seed to the
operator family that is mathematically compatible with that classification.
When the saved Phase-C observables are insufficient, it fails closed instead
of manufacturing a fixed-point mode from a periodic or indeterminate trace.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from src.topic4_zm_modal_operator import (
    analyze_discrete_operator,
    evaluate_operator_prediction,
    fit_discrete_operator,
    mode_axis_angle_deg,
    mode_subspace_angle_deg,
    route_operator_tool,
)


PHASEC_MODAL_VERSION = "zm_phasec_seed_modal_v1_2026-07-28"
NOT_IDENTIFIABLE = "not_identifiable_from_phasec_observables"
HELDOUT_CONTRACT = {
    "fit_noise_count_min": 2,
    "noise_heldout_count_min": 1,
    "fit_time_fraction": 0.60,
    "heldout_time_fraction": 0.20,
    "basis_fit_scope": "fit_noise_early_time_only",
    "operator_fit_scope": "fit_noise_early_time_only",
    "time_heldout_scope": "fit_noise_late_time",
    "noise_heldout_scope": "independent_future_noise_all_eligible_time",
    "cross_seed_pooling": False,
}

_PERIODIC = {
    "periodic_non_tonic_carrier",
    "clonic_or_bursting_carrier",
}
_AI = {
    "balanced_AI_tonic_candidate",
    "balanced_AI_tonic_candidate_supported",
    "balanced_AI_tonic_cell",
    "spike_AI_screen_candidate",
}
_SATURATED = {
    "refractory_saturated_branch",
    "refractory_saturated_branch_supported",
    "refractory_saturated",
}
PERIODIC_PHENOTYPES = frozenset(_PERIODIC)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def phenotype_route(phenotype: str) -> dict[str, str | None]:
    """Return the only allowed analysis route for an accepted phenotype."""

    if phenotype in _PERIODIC:
        carrier = "periodic"
        return {
            "phenotype": phenotype,
            "route": "periodic_stroboscopic",
            "operator_tool": route_operator_tool(carrier),
        }
    if phenotype in _AI:
        carrier = "stochastic"
        return {
            "phenotype": phenotype,
            "route": "AI_observational_DMD",
            "operator_tool": route_operator_tool(carrier),
        }
    if phenotype in _SATURATED:
        return {
            "phenotype": phenotype,
            "route": "saturated_sensitivity_only",
            "operator_tool": None,
        }
    return {
        "phenotype": phenotype,
        "route": "descriptive_only",
        "operator_tool": None,
    }


def derive_seed_route(
    *,
    seed: int,
    c0_seed_row: Mapping[str, Any],
    c1_cell: Mapping[str, Any] | None,
    selected_phenotype: str,
    phenotype_source: str = "auto",
) -> dict[str, Any]:
    """Derive a route without allowing the modal layer to relabel the seed."""

    c0_label = c0_seed_row.get("klass") or c0_seed_row.get("phenotype")
    c1_label = None if c1_cell is None else c1_cell.get("cell_class")
    if phenotype_source == "auto":
        source = "C1_cell" if c1_label not in {None, "missing"} else "C0_seed"
    elif phenotype_source in {"C0_seed", "C1_cell"}:
        source = phenotype_source
    else:
        raise ValueError(f"unsupported phenotype source: {phenotype_source!r}")
    if source == "C1_cell":
        if c1_label in {None, "missing"}:
            raise ValueError(f"seed {seed}: C1 phenotype source lacks a valid cell class")
        accepted = c1_label
    else:
        if c0_label in {None, "missing"}:
            raise ValueError(f"seed {seed}: C0 phenotype source lacks a valid seed class")
        accepted = c0_label
    if accepted != selected_phenotype:
        raise ValueError(
            f"seed {seed}: representative phenotype {selected_phenotype!r} "
            f"does not equal accepted C0/C1 phenotype {accepted!r}"
        )
    route = phenotype_route(str(accepted))
    return {
        "seed": int(seed),
        "input_phenotype": str(accepted),
        "output_phenotype": str(accepted),
        "phenotype_source": source,
        "c0_class": c0_label,
        "c1_class": c1_label,
        **route,
        "modal_override_allowed": False,
    }


def _state_matrix(observables: Mapping[str, Any]) -> tuple[np.ndarray, tuple[int, int]]:
    E = np.asarray(observables["E_rate_grid"], dtype=float)
    I = np.asarray(observables["I_rate_grid"], dtype=float)
    if (
        E.ndim != 3
        or I.shape != E.shape
        or E.shape[0] < 16
        or not np.isfinite(E).all()
        or not np.isfinite(I).all()
    ):
        raise ValueError("E/I rate grids must be matched finite [time,y,x] arrays")
    return np.concatenate((E.reshape(E.shape[0], -1), I.reshape(I.shape[0], -1)), axis=1), E.shape[1:]


def _split_pairs(
    state: np.ndarray,
    *,
    lag: int,
    stride: int,
    fit_fraction: float,
    heldout_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(state.shape[0])
    lag = int(lag)
    stride = int(stride)
    fit_end = int(np.floor(n * fit_fraction))
    hold_start = int(np.ceil(n * (1.0 - heldout_fraction)))
    fit_index = np.arange(0, fit_end - lag, stride, dtype=int)
    hold_first = int(np.ceil(hold_start / stride) * stride)
    hold_index = np.arange(hold_first, n - lag, stride, dtype=int)
    if lag < 1 or stride < 1 or fit_index.size < 4 or hold_index.size < 2:
        raise ValueError("trace is too short for leakage-free fit/time-heldout pairs")
    fit_x = state[fit_index]
    fit_y = state[fit_index + lag]
    hold_x = state[hold_index]
    hold_y = state[hold_index + lag]
    return fit_x, fit_y, hold_x, hold_y


def _axis_vector(grid_shape: tuple[int, int], theta_deg: float) -> np.ndarray:
    ny, nx = grid_shape
    yy, xx = np.meshgrid(
        np.arange(ny, dtype=float) - (ny - 1.0) / 2.0,
        np.arange(nx, dtype=float) - (nx - 1.0) / 2.0,
        indexing="ij",
    )
    theta = np.deg2rad(float(theta_deg))
    field = xx * np.cos(theta) + yy * np.sin(theta)
    field -= np.mean(field)
    return np.concatenate((field.ravel(), field.ravel()))


def _fit_observational_operator(
    runs: Sequence[Mapping[str, Any]],
    *,
    lag: int,
    dt_ms: float,
    pathology_axis_deg: float,
    maximum_rank: int,
    periodic: bool,
) -> dict[str, Any]:
    fit_runs = [row for row in runs if row.get("role") == "fit"]
    noise_holdout = [
        row for row in runs if row.get("role") == "noise_heldout"
    ]
    fit_ids = {str(row.get("noise")) for row in fit_runs}
    hold_ids = {str(row.get("noise")) for row in noise_holdout}
    if (
        len(fit_runs) < HELDOUT_CONTRACT["fit_noise_count_min"]
        or len(noise_holdout) < HELDOUT_CONTRACT["noise_heldout_count_min"]
        or fit_ids & hold_ids
    ):
        raise ValueError("independent fit/noise-heldout continuations are required")

    fit_x_raw, fit_y_raw, time_x_raw, time_y_raw = [], [], [], []
    hold_x_raw, hold_y_raw = [], []
    shape = None
    for row in fit_runs:
        state, row_shape = _state_matrix(row["observables"])
        shape = row_shape if shape is None else shape
        if row_shape != shape:
            raise ValueError("fit runs use different spatial grids")
        x, y, hx, hy = _split_pairs(
            state,
            lag=lag,
            stride=lag if periodic else 1,
            fit_fraction=HELDOUT_CONTRACT["fit_time_fraction"],
            heldout_fraction=HELDOUT_CONTRACT["heldout_time_fraction"],
        )
        fit_x_raw.append(x)
        fit_y_raw.append(y)
        time_x_raw.append(hx)
        time_y_raw.append(hy)
    for row in noise_holdout:
        state, row_shape = _state_matrix(row["observables"])
        if row_shape != shape:
            raise ValueError("noise-heldout run uses a different spatial grid")
        if state.shape[0] <= lag + 2:
            raise ValueError("noise-heldout trace is too short")
        index = np.arange(
            0, state.shape[0] - lag, lag if periodic else 1, dtype=int
        )
        if periodic and index.size < 3:
            raise ValueError("noise-heldout trace has fewer than two heldout cycles")
        hold_x_raw.append(state[index])
        hold_y_raw.append(state[index + lag])

    fit_x_full = np.concatenate(fit_x_raw, axis=0)
    fit_y_full = np.concatenate(fit_y_raw, axis=0)
    center = np.mean(fit_x_full, axis=0)
    x_centered = fit_x_full - center
    y_centered = fit_y_full - center
    _, singular, vt = np.linalg.svd(x_centered, full_matrices=False)
    if not singular.size or singular[0] <= np.finfo(float).eps:
        raise ValueError("fit data have zero dynamic rank")
    energy = np.cumsum(singular ** 2) / np.sum(singular ** 2)
    rank99 = int(np.searchsorted(energy, 0.99) + 1)
    rank = min(
        int(maximum_rank),
        rank99,
        int(x_centered.shape[0]),
        int(x_centered.shape[1]),
    )
    if rank < 1:
        raise ValueError("no identifiable low-rank subspace")
    basis = vt[:rank].T
    fit_x = x_centered @ basis
    fit_y = y_centered @ basis
    fitted = fit_discrete_operator(fit_x, fit_y, ridge=1e-8)

    time_x = (np.concatenate(time_x_raw, axis=0) - center) @ basis
    time_y = (np.concatenate(time_y_raw, axis=0) - center) @ basis
    noise_x = (np.concatenate(hold_x_raw, axis=0) - center) @ basis
    noise_y = (np.concatenate(hold_y_raw, axis=0) - center) @ basis
    time_eval = evaluate_operator_prediction(fitted["operator"], time_x, time_y)
    noise_eval = evaluate_operator_prediction(fitted["operator"], noise_x, noise_y)
    horizon_ms = float(lag) * float(dt_ms)
    modal = analyze_discrete_operator(
        fitted["operator"], dt_ms=horizon_ms, horizon_ms=horizon_ms
    )
    leading_spatial = np.real_if_close(
        basis @ np.asarray(modal["leading_right_mode"])
    )
    if np.iscomplexobj(leading_spatial):
        leading_spatial = np.abs(leading_spatial)
    leading_spatial = np.asarray(leading_spatial, dtype=float)
    axis_angle = mode_axis_angle_deg(
        leading_spatial, _axis_vector(shape, pathology_axis_deg)
    )
    return {
        "status": "identified",
        "evidence_type": (
            "observational_stroboscopic_lag_operator"
            if periodic else "observational_low_rank_DMD"
        ),
        "causal_perturbation_operator": False,
        "lag_bins": int(lag),
        "operator_step_ms": horizon_ms,
        "low_rank": int(rank),
        "training_relative_error": fitted["training_relative_error"],
        "time_heldout": time_eval,
        "noise_heldout": noise_eval,
        "fit_noise_ids": sorted(fit_ids),
        "noise_heldout_ids": sorted(hold_ids),
        "n_time_heldout_pairs": int(time_x.shape[0]),
        "n_noise_heldout_pairs": int(noise_x.shape[0]),
        "leading_spatial_mode": leading_spatial.tolist(),
        "spatial_grid_shape": list(shape),
        "pathology_axis_angle_deg": axis_angle,
        "operator_summary": {
            key: value
            for key, value in modal.items()
            if key not in {
                "leading_right_mode", "leading_left_mode",
                "optimal_input_mode", "optimal_output_mode",
            }
        },
        "heldout_contract": dict(HELDOUT_CONTRACT),
    }


def analyze_seed(
    route: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    *,
    bin_ms: float,
    pathology_axis_deg: float,
    period_ms: float | None = None,
    locked_sensitivity: Mapping[str, Any] | None = None,
    maximum_rank: int = 8,
) -> dict[str, Any]:
    """Execute the routed observation-only audit for one seed."""

    base = {
        **dict(route),
        "phasec_modal_version": PHASEC_MODAL_VERSION,
        "phenotype_unchanged": (
            route.get("input_phenotype") == route.get("output_phenotype")
        ),
    }
    if not base["phenotype_unchanged"]:
        raise ValueError("modal analysis cannot override the Phase-C phenotype")
    route_name = route["route"]
    if route_name == "saturated_sensitivity_only":
        if not isinstance(locked_sensitivity, Mapping):
            return {**base, "status": NOT_IDENTIFIABLE, "reason": "locked_sensitivity_missing"}
        return {
            **base,
            "status": "summarized_without_operator",
            "locked_local_gain_and_refractory_sensitivity": dict(
                locked_sensitivity
            ),
            "seizure_mode_claim": False,
        }
    if route_name == "descriptive_only":
        description = []
        for row in runs:
            try:
                state, _ = _state_matrix(row["observables"])
            except (KeyError, TypeError, ValueError):
                continue
            description.append({
                "noise": row.get("noise"),
                "mean_rate_hz": float(np.mean(state)),
                "rate_sd_hz": float(np.std(state)),
            })
        return {
            **base,
            "status": "descriptive_only",
            "descriptive_runs": description,
            "operator_identified": False,
        }
    try:
        if route_name == "periodic_stroboscopic":
            route_operator_tool("periodic", requested_tool=route["operator_tool"])
            if period_ms is None or not np.isfinite(period_ms) or period_ms <= 0:
                raise ValueError("periodic route requires a locked positive period")
            lag = max(1, int(round(float(period_ms) / float(bin_ms))))
            out = _fit_observational_operator(
                runs,
                lag=lag,
                dt_ms=bin_ms,
                pathology_axis_deg=pathology_axis_deg,
                maximum_rank=maximum_rank,
                periodic=True,
            )
            out["locked_period_ms"] = float(period_ms)
            out["heldout_cycles_required"] = True
        elif route_name == "AI_observational_DMD":
            route_operator_tool("stochastic", requested_tool=route["operator_tool"])
            out = _fit_observational_operator(
                runs,
                lag=1,
                dt_ms=bin_ms,
                pathology_axis_deg=pathology_axis_deg,
                maximum_rank=maximum_rank,
                periodic=False,
            )
        else:
            raise ValueError(f"unsupported modal route: {route_name}")
    except (KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return {
            **base,
            "status": NOT_IDENTIFIABLE,
            "reason": str(exc),
            "operator_identified": False,
            "heldout_contract": dict(HELDOUT_CONTRACT),
        }
    return {**base, **out, "operator_identified": True}


def aggregate_seed_modal(seed_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare homologous seed-specific modes without pooling eigenvalues."""

    seeds = [int(row["seed"]) for row in seed_results]
    if len(seeds) != len(set(seeds)):
        raise ValueError("duplicate seed result would pool operating points")
    phenotypes = {str(row["input_phenotype"]) for row in seed_results}
    routes = {str(row["route"]) for row in seed_results}
    comparisons = []
    for left, right in combinations(seed_results, 2):
        if (
            left.get("route") != right.get("route")
            or left.get("status") != "identified"
            or right.get("status") != "identified"
        ):
            continue
        a = np.asarray(left["leading_spatial_mode"], dtype=float)
        b = np.asarray(right["leading_spatial_mode"], dtype=float)
        if a.shape != b.shape:
            continue
        comparisons.append({
            "seeds": [int(left["seed"]), int(right["seed"])],
            "route": left["route"],
            "leading_subspace_angle_deg": mode_subspace_angle_deg(a, b[None, :]),
            "pathology_axis_angle_deg": [
                float(left["pathology_axis_angle_deg"]),
                float(right["pathology_axis_angle_deg"]),
            ],
        })
    statuses = {str(row.get("status")) for row in seed_results}
    status = (
        "complete" if seed_results and statuses <= {
            "identified", "summarized_without_operator", "descriptive_only"
        } else "partial"
    )
    return {
        "schema": PHASEC_MODAL_VERSION,
        "status": status,
        "verdict": (
            "seed_specific_modal_audit_complete"
            if status == "complete"
            else "seed_specific_modal_audit_partial"
        ),
        "class_disagreement": len(phenotypes) > 1 or len(routes) > 1,
        "phenotypes_by_seed": {
            str(row["seed"]): row["input_phenotype"] for row in seed_results
        },
        "routes_by_seed": {
            str(row["seed"]): row["route"] for row in seed_results
        },
        "same_class_spatial_comparisons": comparisons,
        "eigenvalue_pooling": "forbidden_not_performed",
        "modal_can_override_phasec": False,
        "seed_results": [dict(row) for row in seed_results],
    }
