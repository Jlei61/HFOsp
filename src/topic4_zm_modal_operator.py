"""Trajectory-conditioned modal/operator diagnostics for the Z/M carrier.

The functions here are analysis primitives, not evidence that a carrier or a
transition exists.  In particular, operator choice is conditional on the
source-rhythm audit so a periodic trajectory cannot be collapsed to its time
average and treated as a fixed point.
"""

from __future__ import annotations

import numpy as np


MODAL_OPERATOR_VERSION = "zm_modal_operator_v1_2026-07-27"

_TOOL_BY_CARRIER = {
    "fixed": "eigen",
    "periodic": "stroboscopic_floquet",
    "stochastic": "dmd_finite_time_gain",
}


def route_operator_tool(carrier_type, requested_tool=None):
    """Return the registered operator tool for a source carrier class."""

    if carrier_type not in _TOOL_BY_CARRIER:
        raise ValueError(f"unsupported carrier type: {carrier_type!r}")
    required = _TOOL_BY_CARRIER[carrier_type]
    if requested_tool is not None and requested_tool != required:
        if carrier_type == "periodic" and requested_tool == "eigen":
            raise ValueError(
                "periodic carrier requires a stroboscopic/Floquet operator; "
                "a time-averaged fixed-state eigen analysis is forbidden"
            )
        raise ValueError(
            f"{carrier_type} carrier requires {required}, got {requested_tool}"
        )
    return required


def _zero_mean_equal_energy(field, energy):
    value = np.asarray(field, dtype=float)
    value = value - np.mean(value)
    norm2 = float(np.sum(value ** 2))
    if norm2 <= np.finfo(float).eps:
        raise ValueError("perturbation has zero spatial energy")
    return value * np.sqrt(float(energy) / norm2)


def equal_energy_perturbations(n, theta_deg, energy=1.0, random_seed=0):
    """Construct locked axial/transverse/isotropic/core/random grid modes."""

    n = int(n)
    energy = float(energy)
    if n < 4:
        raise ValueError("n must be at least 4")
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("energy must be finite and positive")

    axis = np.arange(n, dtype=float) - (n - 1.0) / 2.0
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    theta = np.deg2rad(float(theta_deg))
    parallel = xx * np.cos(theta) + yy * np.sin(theta)
    perpendicular = -xx * np.sin(theta) + yy * np.cos(theta)
    radius2 = xx ** 2 + yy ** 2
    sigma_core = max(1.0, n / 8.0)
    sigma_broad = max(1.0, n / 3.0)
    core = np.exp(-0.5 * radius2 / sigma_core ** 2)
    # A broad Mexican-hat perturbation is isotropic but not a duplicate of the
    # sharply localized core probe.
    isotropic = (
        np.exp(-0.5 * radius2 / sigma_broad ** 2)
        - 0.5 * np.exp(-0.5 * radius2 / (0.55 * sigma_broad) ** 2)
    )
    random = np.random.default_rng(random_seed).normal(size=(n, n))

    return {
        "axial": _zero_mean_equal_energy(parallel, energy),
        "transverse": _zero_mean_equal_energy(perpendicular, energy),
        "isotropic": _zero_mean_equal_energy(isotropic, energy),
        "core": _zero_mean_equal_energy(core, energy),
        "random": _zero_mean_equal_energy(random, energy),
    }


def fit_discrete_operator(x, y, ridge=0.0):
    """Fit ``y_t = A x_t`` without an intercept to perturbation responses."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ridge = float(ridge)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError("x and y must be matched [sample, state] arrays")
    if x.shape[0] < x.shape[1]:
        raise ValueError("operator fit requires at least as many samples as states")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("operator inputs must be finite")
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative")

    gram = x.T @ x
    cross = x.T @ y
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), cross
    )
    operator = coefficients.T
    residual = y - x @ operator.T
    denominator = float(np.linalg.norm(y))
    relative_error = (
        float(np.linalg.norm(residual) / denominator)
        if denominator > np.finfo(float).eps
        else None
    )
    return {
        "operator": operator,
        "training_relative_error": relative_error,
        "n_samples": int(x.shape[0]),
        "state_dimension": int(x.shape[1]),
        "ridge": ridge,
        "modal_operator_version": MODAL_OPERATOR_VERSION,
    }


def evaluate_operator_prediction(operator, x, y):
    """Evaluate a fitted operator on held-out perturbations or time windows."""

    operator = np.asarray(operator, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or x.ndim != 2
        or y.shape != x.shape
        or x.shape[1] != operator.shape[1]
    ):
        raise ValueError("operator and held-out [sample, state] arrays do not align")
    if not np.isfinite(operator).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("operator evaluation inputs must be finite")
    prediction = x @ operator.T
    denominator = float(np.linalg.norm(y))
    if denominator <= np.finfo(float).eps:
        return {
            "status": "undefined_zero_response",
            "heldout_relative_error": None,
            "n_samples": int(x.shape[0]),
            "modal_operator_version": MODAL_OPERATOR_VERSION,
        }
    return {
        "status": "ok",
        "heldout_relative_error": float(
            np.linalg.norm(prediction - y) / denominator
        ),
        "n_samples": int(x.shape[0]),
        "modal_operator_version": MODAL_OPERATOR_VERSION,
    }


def _unit_mode(vector):
    vector = np.asarray(vector)
    norm = np.linalg.norm(vector)
    if norm <= np.finfo(float).eps:
        raise ValueError("mode has zero norm")
    return vector / norm


def _phase_aligned_mode(vector):
    """Fix the arbitrary complex phase using the largest-magnitude component."""

    vector = _unit_mode(vector)
    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) > np.finfo(float).eps:
        vector = vector * np.exp(-1j * np.angle(vector[pivot]))
    return np.real_if_close(vector, tol=1000)


def analyze_discrete_operator(operator, dt_ms, horizon_ms):
    """Report asymptotic and finite-time properties of a discrete operator."""

    operator = np.asarray(operator, dtype=float)
    dt_ms = float(dt_ms)
    horizon_ms = float(horizon_ms)
    if (
        operator.ndim != 2
        or operator.shape[0] != operator.shape[1]
        or not np.isfinite(operator).all()
    ):
        raise ValueError("operator must be a finite square matrix")
    if dt_ms <= 0 or horizon_ms <= 0:
        raise ValueError("dt_ms and horizon_ms must be positive")

    eigvals, right = np.linalg.eig(operator)
    leading = int(np.argmax(np.abs(eigvals)))
    left_vals, left = np.linalg.eig(operator.T)
    left_index = int(np.argmin(np.abs(left_vals - eigvals[leading])))
    steps = max(1, int(round(horizon_ms / dt_ms)))
    propagator = np.linalg.matrix_power(operator, steps)
    _, singular_values, vh = np.linalg.svd(propagator, full_matrices=False)
    optimal_input = _unit_mode(vh[0])
    optimal_output = _unit_mode(propagator @ optimal_input)
    magnitudes = np.abs(eigvals)
    safe_magnitudes = np.maximum(magnitudes, np.finfo(float).tiny)
    rates_per_ms = np.log(safe_magnitudes) / dt_ms

    return {
        "spectral_radius": float(np.max(magnitudes)),
        "spectral_abscissa_per_ms": float(np.max(rates_per_ms)),
        "leading_eigenvalue_real": float(np.real(eigvals[leading])),
        "leading_eigenvalue_imag": float(np.imag(eigvals[leading])),
        "leading_right_mode": _phase_aligned_mode(right[:, leading]),
        "leading_left_mode": _phase_aligned_mode(left[:, left_index]),
        "finite_time_gain": float(singular_values[0]),
        "optimal_input_mode": optimal_input,
        "optimal_output_mode": optimal_output,
        "horizon_ms": horizon_ms,
        "n_steps": steps,
        "modal_operator_version": MODAL_OPERATOR_VERSION,
    }


def mode_axis_angle_deg(mode, axis):
    """Return the sign-invariant acute angle between a mode and an axis."""

    mode = np.asarray(mode)
    axis = np.asarray(axis)
    if mode.ndim != 1 or axis.ndim != 1 or mode.shape != axis.shape:
        raise ValueError("mode and axis must be matched vectors")
    mode = np.asarray(mode, dtype=complex)
    axis = np.asarray(axis, dtype=complex)
    denominator = np.linalg.norm(mode) * np.linalg.norm(axis)
    if denominator <= np.finfo(float).eps:
        raise ValueError("mode and axis must be nonzero")
    cosine = np.clip(abs(np.vdot(mode, axis)) / denominator, 0.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def infer_linearity_range(rows, max_relative_error):
    """Find the largest contiguous low-amplitude range passing prediction error."""

    threshold = float(max_relative_error)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("max_relative_error must be finite and positive")
    if not rows:
        raise ValueError("at least one amplitude row is required")
    ordered = sorted(rows, key=lambda row: float(row["amplitude"]))
    amplitudes = np.asarray([row["amplitude"] for row in ordered], dtype=float)
    errors = np.asarray(
        [row["heldout_relative_error"] for row in ordered], dtype=float
    )
    if (
        not np.isfinite(amplitudes).all()
        or not np.isfinite(errors).all()
        or np.any(amplitudes <= 0)
        or np.any(np.diff(amplitudes) <= 0)
    ):
        raise ValueError("amplitudes must be unique positive values and errors finite")
    passing = errors <= threshold
    n_contiguous = 0
    for passed in passing:
        if not passed:
            break
        n_contiguous += 1
    return {
        "status": "identified" if n_contiguous else "no_valid_linear_range",
        "maximum_linear_amplitude": (
            float(amplitudes[n_contiguous - 1]) if n_contiguous else None
        ),
        "n_passing": int(n_contiguous),
        "max_relative_error": threshold,
        "rows": [
            {
                "amplitude": float(amplitude),
                "heldout_relative_error": float(error),
                "passes": bool(passed),
            }
            for amplitude, error, passed in zip(amplitudes, errors, passing)
        ],
        "modal_operator_version": MODAL_OPERATOR_VERSION,
    }
