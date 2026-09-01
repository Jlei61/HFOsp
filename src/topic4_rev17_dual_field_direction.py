"""Robust local directions in the rev17 mean/dispersion residual span."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize


CHANNELS = ("mean", "dispersion")
METRICS = {
    "A": "mode_0_mean",
    "B": "mode_1_mean",
    "J14": "j14",
    "support_A": "mode_0_effective_events",
    "support_B": "mode_1_effective_events",
}


def _unit(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(values))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("rev17 direction has zero or nonfinite norm")
    return values / norm


def response_tensor(
    finite_rows: list[Mapping[str, Any]], *, amplitude: float,
    maximum_mode_index: int = 14, maximum_nonlinearity_ratio: float = 1.0,
    derivative_floor_fraction: float = 0.1,
) -> dict[str, Any]:
    """Build equal-network gradients and a predeclared local-linearity mask."""
    amplitude = float(amplitude)
    n_modes = int(maximum_mode_index) + 1
    seeds = sorted({int(row["seed"]) for row in finite_rows})
    if len(seeds) != 3:
        raise RuntimeError("rev17 response tensor requires exactly three fit networks")
    expected = len(CHANNELS) * n_modes * len(seeds)
    if len(finite_rows) != expected:
        raise RuntimeError("rev17 finite-difference Cartesian product is incomplete")
    coordinates = [
        {"coordinate_index": ci * n_modes + mode, "channel": channel,
         "mode_index": mode}
        for ci, channel in enumerate(CHANNELS)
        for mode in range(n_modes)
    ]
    lookup = {
        (str(row["channel"]), int(row["mode_index"]), int(row["seed"])): row
        for row in finite_rows
    }
    if len(lookup) != expected:
        raise RuntimeError("rev17 finite-difference coordinate identity is duplicated")
    gradients = {
        name: np.zeros((len(seeds), len(coordinates)), dtype=float)
        for name in METRICS
    }
    curvatures = {
        name: np.zeros((len(seeds), len(coordinates)), dtype=float)
        for name in METRICS
    }
    for coordinate in coordinates:
        column = int(coordinate["coordinate_index"])
        for network_index, seed in enumerate(seeds):
            row = lookup[(coordinate["channel"], coordinate["mode_index"], seed)]
            for name, prefix in METRICS.items():
                gradients[name][network_index, column] = float(
                    row[f"{prefix}_derivative"]
                )
                curvatures[name][network_index, column] = float(
                    row[f"{prefix}_curvature"]
                )
    if not all(np.isfinite(value).all() for value in (*gradients.values(), *curvatures.values())):
        raise RuntimeError("rev17 response tensor contains nonfinite values")

    ratios = {}
    floors = {}
    for name in METRICS:
        absolute = np.abs(gradients[name])
        nonzero = absolute[absolute > 1e-12]
        typical = float(np.median(nonzero)) if len(nonzero) else 0.0
        floor = max(float(derivative_floor_fraction) * typical, 1e-9)
        floors[name] = floor
        ratios[name] = (
            0.5 * amplitude * np.abs(curvatures[name])
            / np.maximum(absolute, floor)
        )
    # The fitted direction must predict both the weak-mode and complete-score
    # response locally. B/support are protected by explicit constraints below.
    eligible = np.maximum(
        np.max(ratios["A"], axis=0), np.max(ratios["J14"], axis=0),
    ) <= float(maximum_nonlinearity_ratio)
    coordinate_rows = []
    for coordinate in coordinates:
        column = int(coordinate["coordinate_index"])
        coordinate_rows.append({
            **coordinate,
            "linear_eligible": bool(eligible[column]),
            "maximum_A_nonlinearity_ratio": float(np.max(ratios["A"][:, column])),
            "maximum_J14_nonlinearity_ratio": float(np.max(ratios["J14"][:, column])),
            "A_gradient_sign_support": int(max(
                np.sum(gradients["A"][:, column] > 0.0),
                np.sum(gradients["A"][:, column] < 0.0),
            )),
            "J14_gradient_sign_support": int(max(
                np.sum(gradients["J14"][:, column] > 0.0),
                np.sum(gradients["J14"][:, column] < 0.0),
            )),
        })
    return {
        "network_seeds": seeds,
        "coordinates": coordinate_rows,
        "amplitude": amplitude,
        "maximum_nonlinearity_ratio": float(maximum_nonlinearity_ratio),
        "derivative_floor_fraction": float(derivative_floor_fraction),
        "derivative_floors": floors,
        "linear_eligible_coordinates": np.flatnonzero(eligible).astype(int).tolist(),
        "gradients": {name: value.tolist() for name, value in gradients.items()},
        "curvatures": {name: value.tolist() for name, value in curvatures.items()},
    }


def _masked_gradients(tensor: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    gradients = {
        name: np.asarray(values, dtype=float)
        for name, values in tensor["gradients"].items()
    }
    dimensions = {value.shape for value in gradients.values()}
    if len(dimensions) != 1:
        raise RuntimeError("rev17 gradient matrices do not align")
    shape = next(iter(dimensions))
    mask = np.zeros(shape[1], dtype=bool)
    mask[np.asarray(tensor["linear_eligible_coordinates"], dtype=int)] = True
    for name in gradients:
        gradients[name][:, ~mask] = 0.0
    return gradients, mask


def maximin_direction(
    primary: np.ndarray, *, protected_losses: tuple[np.ndarray, ...] = (),
    protected_supports: tuple[np.ndarray, ...] = (),
    active_mask: np.ndarray | None = None,
    ftol: float = 1e-10, maxiter: int = 4000, tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Maximize the worst predicted descent under networkwise protections."""
    primary = np.asarray(primary, dtype=float)
    if primary.ndim != 2 or primary.shape[0] != 3:
        raise ValueError("rev17 maximin primary gradient must be 3 by D")
    dimension = primary.shape[1]
    losses = tuple(np.asarray(value, dtype=float) for value in protected_losses)
    supports = tuple(np.asarray(value, dtype=float) for value in protected_supports)
    if any(value.shape != primary.shape for value in (*losses, *supports)):
        raise ValueError("rev17 protected gradients must match primary shape")
    if active_mask is None:
        active = np.ones(dimension, dtype=bool)
    else:
        active = np.asarray(active_mask, dtype=bool)
        if active.shape != (dimension,) or not np.any(active):
            raise ValueError("rev17 active-coordinate mask is invalid")
    initial = _unit(-np.mean(primary, axis=0))
    initial_margin = min(0.0, float(np.min(-primary @ initial)))
    x0 = np.r_[initial, initial_margin]
    constraints: list[dict[str, Any]] = [{
        "type": "ineq", "fun": lambda x: 1.0 - float(np.dot(x[:-1], x[:-1])),
    }]
    for row in primary:
        constraints.append({
            "type": "ineq",
            "fun": lambda x, row=row: float(-np.dot(row, x[:-1]) - x[-1]),
        })
    for matrix in losses:
        for row in matrix:
            constraints.append({
                "type": "ineq",
                "fun": lambda x, row=row: float(-np.dot(row, x[:-1])),
            })
    for matrix in supports:
        for row in matrix:
            constraints.append({
                "type": "ineq",
                "fun": lambda x, row=row: float(np.dot(row, x[:-1])),
            })
    result = minimize(
        lambda x: -float(x[-1]), x0, method="SLSQP",
        bounds=[(-1.0, 1.0) if keep else (0.0, 0.0) for keep in active]
        + [(-100.0, 100.0)],
        constraints=constraints,
        options={"ftol": float(ftol), "maxiter": int(maxiter), "disp": False},
    )
    residuals = np.asarray([constraint["fun"](result.x) for constraint in constraints])
    vector = np.asarray(result.x[:-1], dtype=float)
    margin = float(result.x[-1])
    feasible = bool(
        result.success and np.min(residuals) >= -float(tolerance)
        and margin > 1e-8 and np.linalg.norm(vector) > 1e-12
    )
    direction = _unit(vector) if feasible else None
    return {
        "success": bool(result.success),
        "feasible_positive_margin": feasible,
        "message": str(result.message),
        "iterations": int(result.nit),
        "worst_primary_improvement_margin": margin,
        "minimum_constraint_residual": float(np.min(residuals)),
        "direction": None if direction is None else direction.tolist(),
    }


def consensus_sparse_direction(
    gradients: Mapping[str, np.ndarray], mask: np.ndarray,
    *, maximum_coordinates: int = 8,
) -> dict[str, Any]:
    """Use only coordinates whose objective direction agrees across networks."""
    selected = []
    for column in np.flatnonzero(mask):
        median = float(np.median(gradients["A"][:, column]))
        if abs(median) <= 1e-12:
            continue
        sign = -float(np.sign(median))
        changes = {name: sign * values[:, column] for name, values in gradients.items()}
        if (
            np.all(changes["A"] < 0.0)
            and np.all(changes["J14"] < 0.0)
            and np.sum(changes["B"] <= 0.0) >= 2
            and np.sum(changes["support_A"] >= 0.0) >= 2
            and np.sum(changes["support_B"] >= 0.0) >= 2
        ):
            strength = float(np.median(-changes["A"]) + np.median(-changes["J14"]))
            selected.append((strength, int(column), sign))
    selected.sort(key=lambda row: (-row[0], row[1]))
    selected = selected[:int(maximum_coordinates)]
    vector = np.zeros(mask.size, dtype=float)
    for strength, column, sign in selected:
        vector[column] = sign * max(strength, 1e-12)
    return {
        "feasible": bool(selected),
        "selected_coordinates": [row[1] for row in selected],
        "direction": _unit(vector).tolist() if selected else None,
    }


def construct_directions(
    tensor: Mapping[str, Any], *, maximum_sparse_coordinates: int = 8,
) -> dict[str, Any]:
    gradients, mask = _masked_gradients(tensor)
    if not np.any(mask):
        return {
            "analysis_status": "NO_LOCALLY_LINEAR_COORDINATE",
            "linear_eligible_coordinate_count": 0,
            "claim_boundary": (
                "The antithetic atlas is complete, but its event-level response "
                "is not locally linear at the frozen amplitude. No gradient "
                "direction is inferred from these coordinates."
            ),
        }

    def mean_direction(name: str) -> dict[str, Any]:
        vector = -np.mean(gradients[name], axis=0)
        vector[~mask] = 0.0
        direction = _unit(vector)
        return {
            "feasible": True,
            "direction": direction.tolist(),
            "predicted_A_changes": (gradients["A"] @ direction).tolist(),
            "predicted_B_changes": (gradients["B"] @ direction).tolist(),
            "predicted_J14_changes": (gradients["J14"] @ direction).tolist(),
        }

    directions = {
        "analysis_status": "LOCALLY_LINEAR_DIRECTIONS_CONSTRUCTED",
        "linear_eligible_coordinate_count": int(np.sum(mask)),
        "mean_a": mean_direction("A"),
        "mean_j14": mean_direction("J14"),
        "maximin_bprotected": maximin_direction(
            gradients["A"], protected_losses=(gradients["B"],),
            active_mask=mask,
        ),
        "maximin_supportprotected": maximin_direction(
            gradients["A"], protected_losses=(gradients["B"],),
            protected_supports=(gradients["support_A"], gradients["support_B"]),
            active_mask=mask,
        ),
        "maximin_j14_abprotected": maximin_direction(
            gradients["J14"],
            protected_losses=(gradients["A"], gradients["B"]),
            protected_supports=(gradients["support_A"], gradients["support_B"]),
            active_mask=mask,
        ),
        "consensus_sparse": consensus_sparse_direction(
            gradients, mask, maximum_coordinates=int(maximum_sparse_coordinates),
        ),
    }
    for record in directions.values():
        if not isinstance(record, Mapping):
            continue
        if record.get("direction") is None:
            continue
        direction = np.asarray(record["direction"], dtype=float)
        record.update({
            "predicted_A_changes": (gradients["A"] @ direction).tolist(),
            "predicted_B_changes": (gradients["B"] @ direction).tolist(),
            "predicted_J14_changes": (gradients["J14"] @ direction).tolist(),
            "predicted_support_A_changes": (
                gradients["support_A"] @ direction
            ).tolist(),
            "predicted_support_B_changes": (
                gradients["support_B"] @ direction
            ).tolist(),
        })
    return directions
