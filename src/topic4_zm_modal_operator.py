"""Trajectory-conditioned modal/operator diagnostics for the Z/M carrier.

The functions here are analysis primitives, not evidence that a carrier or a
transition exists.  In particular, operator choice is conditional on the
source-rhythm audit so a periodic trajectory cannot be collapsed to its time
average and treated as a fixed point.
"""

from __future__ import annotations

import copy

import numpy as np


MODAL_OPERATOR_VERSION = "zm_modal_operator_v1_2026-07-27"
MAX_SPATIAL_BASIS_CONDITION = 100.0

_TOOL_BY_CARRIER = {
    "fixed": "eigen",
    "periodic": "stroboscopic_floquet",
    "stochastic": "dmd_finite_time_gain",
}

_CARRIER_BY_SOURCE_CLASS = {
    "stationary_rate_candidate": "fixed",
    "global_periodic_candidate": "periodic",
    "phase_staggered_periodic_candidate": "periodic",
    "asynchronous_or_irregular_candidate": "stochastic",
}


def modal_probe_authorized(source_summary):
    """Fail closed until fine-source routing is replicated across two seeds."""

    return bool(
        (source_summary or {}).get("status") == "replicated"
        and (source_summary or {}).get("carrier_type") in _TOOL_BY_CARRIER
    )


def operator_horizons_ms(carrier_type, *, frequency_hz=None, bin_ms=2.0):
    """Return locked response horizons matched to the replicated carrier type."""

    route_operator_tool(carrier_type)
    bin_ms = float(bin_ms)
    if not np.isfinite(bin_ms) or bin_ms <= 0:
        raise ValueError("bin_ms must be finite and positive")
    if carrier_type == "periodic":
        if frequency_hz is None or not np.isfinite(frequency_hz) or frequency_hz <= 0:
            raise ValueError("periodic carrier requires a positive source frequency")
        period = 1000.0 / float(frequency_hz)
        return [float(max(bin_ms, round(period / bin_ms) * bin_ms))]
    if carrier_type == "stochastic":
        return [20.0, 50.0, 100.0]
    return [2.0, 10.0, 20.0]


def route_source_temporal_class(source_temporal_class):
    """Map the fine source audit onto the operator-level carrier taxonomy."""

    if source_temporal_class not in _CARRIER_BY_SOURCE_CLASS:
        raise ValueError(
            f"insufficient or unsupported source temporal class: "
            f"{source_temporal_class!r}"
        )
    return _CARRIER_BY_SOURCE_CLASS[source_temporal_class]


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


def spatial_basis_diagnostics(spatial_modes, mode_order=None):
    """Report the unit-normalized probe-basis rank and condition number."""

    if mode_order is None:
        mode_order = tuple(spatial_modes)
    else:
        mode_order = tuple(mode_order)
    if not mode_order:
        raise ValueError("spatial probe basis is empty")
    shape = None
    columns = []
    for name in mode_order:
        if name not in spatial_modes:
            raise ValueError(f"missing spatial mode {name!r}")
        mode = np.asarray(spatial_modes[name], dtype=float)
        if mode.ndim != 2 or not np.isfinite(mode).all():
            raise ValueError(f"{name}: spatial mode must be a finite 2D field")
        if shape is None:
            shape = mode.shape
        elif mode.shape != shape:
            raise ValueError("spatial modes do not share a grid")
        vector = mode.ravel()
        norm = float(np.linalg.norm(vector))
        if norm <= np.finfo(float).eps:
            raise ValueError(f"{name}: spatial mode has zero energy")
        columns.append(vector / norm)
    design = np.column_stack(columns)
    rank = int(np.linalg.matrix_rank(design))
    condition = float(np.linalg.cond(design))
    return {
        "rank": rank,
        "n_modes": len(mode_order),
        "condition_number": condition,
        "well_conditioned": bool(
            rank == len(mode_order)
            and np.isfinite(condition)
            and condition <= MAX_SPATIAL_BASIS_CONDITION
        ),
        "maximum_condition_number": MAX_SPATIAL_BASIS_CONDITION,
        "mode_order": list(mode_order),
        "modal_operator_version": MODAL_OPERATOR_VERSION,
    }


def apply_voltage_perturbation(
    state,
    field,
    posE,
    posI,
    *,
    L,
    population,
    rms_amplitude_mv=None,
    total_energy_mv2=None,
    sign,
):
    """Map a grid mode to E/I voltage with explicit RMS or total energy."""

    field = np.asarray(field, dtype=float)
    posE = np.asarray(posE, dtype=float)
    posI = np.asarray(posI, dtype=float)
    if field.ndim != 2 or field.shape[0] != field.shape[1]:
        raise ValueError("field must be a square spatial grid")
    if population not in {"E", "I"}:
        raise ValueError("population must be E or I")
    if int(sign) not in {-1, 1}:
        raise ValueError("sign must be -1 or +1")
    positions = posE if population == "E" else posI
    if (rms_amplitude_mv is None) == (total_energy_mv2 is None):
        raise ValueError(
            "provide exactly one of rms_amplitude_mv or total_energy_mv2"
        )
    if total_energy_mv2 is not None:
        energy = float(total_energy_mv2)
        if not np.isfinite(energy) or energy <= 0:
            raise ValueError("total_energy_mv2 must be finite and positive")
        amplitude = float(np.sqrt(energy / len(positions)))
    else:
        amplitude = float(rms_amplitude_mv)
        if not np.isfinite(amplitude) or amplitude <= 0:
            raise ValueError("rms_amplitude_mv must be finite and positive")
    n_grid = field.shape[0]
    ix = np.clip((positions[:, 0] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((positions[:, 1] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    delta = field[iy, ix].astype(float)
    delta -= np.mean(delta)
    rms = float(np.sqrt(np.mean(delta ** 2)))
    if rms <= np.finfo(float).eps:
        raise ValueError("grid mode is degenerate on the selected neurons")
    delta *= int(sign) * amplitude / rms

    out = copy.deepcopy(state)
    voltage = np.asarray(out["V"], dtype=float).copy()
    nE = len(posE)
    expected = nE + len(posI)
    if voltage.ndim != 1 or voltage.size != expected:
        raise ValueError("state V does not align with E/I positions")
    selected = slice(0, nE) if population == "E" else slice(nE, expected)
    voltage[selected] += delta
    out["V"] = voltage
    return out, delta


def neuron_values_to_grid(values, positions, *, L, n_grid):
    """Average a per-neuron state coordinate on the locked spatial grid."""

    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    n_grid = int(n_grid)
    if (
        values.ndim != 1
        or positions.ndim != 2
        or positions.shape != (values.size, 2)
        or not np.isfinite(values).all()
        or not np.isfinite(positions).all()
    ):
        raise ValueError("values and positions must be aligned finite arrays")
    if n_grid < 1 or not np.isfinite(L) or float(L) <= 0:
        raise ValueError("n_grid and L must be positive")
    ix = np.clip((positions[:, 0] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((positions[:, 1] / float(L) * n_grid).astype(int), 0, n_grid - 1)
    counts = np.zeros((n_grid, n_grid), dtype=int)
    total = np.zeros((n_grid, n_grid), dtype=float)
    np.add.at(counts, (iy, ix), 1)
    np.add.at(total, (iy, ix), values)
    grid = np.divide(
        total,
        counts,
        out=np.zeros_like(total),
        where=counts > 0,
    )
    return {
        "grid": grid,
        "counts": counts,
        "all_cells_observed": bool(np.all(counts > 0)),
        "modal_operator_version": MODAL_OPERATOR_VERSION,
    }


def project_voltage_state_difference(
    state,
    baseline_state,
    posE,
    posI,
    *,
    L,
    spatial_modes,
    mode_order,
):
    """Project future-minus-baseline voltage in the same E/I probe basis."""

    voltage = np.asarray(state["V"], dtype=float)
    baseline = np.asarray(baseline_state["V"], dtype=float)
    posE = np.asarray(posE, dtype=float)
    posI = np.asarray(posI, dtype=float)
    if voltage.shape != baseline.shape or voltage.ndim != 1:
        raise ValueError("state and baseline V arrays do not align")
    if voltage.size != len(posE) + len(posI):
        raise ValueError("V arrays do not align with E/I positions")
    first_mode = np.asarray(spatial_modes[tuple(mode_order)[0]], dtype=float)
    if first_mode.ndim != 2 or first_mode.shape[0] != first_mode.shape[1]:
        raise ValueError("spatial modes must use a square grid")
    n_grid = first_mode.shape[0]
    delta = voltage - baseline
    E = neuron_values_to_grid(delta[:len(posE)], posE, L=L, n_grid=n_grid)
    I = neuron_values_to_grid(delta[len(posE):], posI, L=L, n_grid=n_grid)
    if not E["all_cells_observed"] or not I["all_cells_observed"]:
        raise ValueError("modal grid contains an unobserved E or I cell")
    projected = project_ei_grid(
        E["grid"], I["grid"], spatial_modes, mode_order=mode_order
    )
    return {
        **projected,
        "E_deltaV_grid": E["grid"],
        "I_deltaV_grid": I["grid"],
    }


def project_ei_grid(E_grid, I_grid, spatial_modes, *, mode_order):
    """Project matched E/I rate differences onto the registered spatial probes."""

    E_grid = np.asarray(E_grid, dtype=float)
    I_grid = np.asarray(I_grid, dtype=float)
    mode_order = tuple(mode_order)
    if E_grid.ndim != 2 or I_grid.shape != E_grid.shape:
        raise ValueError("E_grid and I_grid must be matched 2D fields")
    if not np.isfinite(E_grid).all() or not np.isfinite(I_grid).all():
        raise ValueError("E/I grids must be finite")
    diagnostics = spatial_basis_diagnostics(spatial_modes, mode_order)
    if not diagnostics["well_conditioned"]:
        raise ValueError(
            "spatial probe basis is rank deficient or ill-conditioned "
            f"(condition={diagnostics['condition_number']:.3g})"
        )
    basis = []
    for name in mode_order:
        if name not in spatial_modes:
            raise ValueError(f"missing spatial mode {name!r}")
        mode = np.asarray(spatial_modes[name], dtype=float)
        if mode.shape != E_grid.shape or not np.isfinite(mode).all():
            raise ValueError(f"{name}: spatial mode does not align with E/I grid")
        basis.append(mode.ravel())
    design = np.column_stack(basis)
    coordinates = []
    names = []
    for population, grid in (("E", E_grid), ("I", I_grid)):
        coefficient, *_ = np.linalg.lstsq(design, grid.ravel(), rcond=None)
        coordinates.extend(float(value) for value in coefficient)
        names.extend(f"{name}_{population}" for name in mode_order)
    return {
        "coordinates": np.asarray(coordinates, dtype=float),
        "coordinate_order": names,
        "modal_operator_version": MODAL_OPERATOR_VERSION,
    }


def assemble_central_propagator(rows, *, input_order, amplitude):
    """Assemble a finite-time response operator from paired +/- voltage probes."""

    input_order = tuple(input_order)
    amplitude = float(amplitude)
    if not input_order or len(set(input_order)) != len(input_order):
        raise ValueError("input_order must contain unique mode names")
    if not np.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    input_columns = []
    output_columns = []
    bank_shas = {}
    for mode_index, name in enumerate(input_order):
        selected = [
            row for row in rows
            if row.get("input_mode") == name
            and np.isclose(float(row.get("amplitude", np.nan)), amplitude)
        ]
        if len(selected) != 2 or {int(row.get("sign", 0)) for row in selected} != {-1, 1}:
            raise ValueError(f"{name}: require one matched plus/minus pair")
        plus = next(row for row in selected if int(row["sign"]) == 1)
        minus = next(row for row in selected if int(row["sign"]) == -1)
        if plus.get("bank_sha") != minus.get("bank_sha"):
            raise ValueError(f"{name}: central pair uses unmatched future noise")
        yp = np.asarray(plus["response"], dtype=float)
        ym = np.asarray(minus["response"], dtype=float)
        if yp.ndim != 1 or yp.shape != ym.shape or not np.isfinite([*yp, *ym]).all():
            raise ValueError(f"{name}: response vectors must be matched and finite")
        if "input_coordinates" in plus or "input_coordinates" in minus:
            xp = np.asarray(plus.get("input_coordinates"), dtype=float)
            xm = np.asarray(minus.get("input_coordinates"), dtype=float)
            if (
                xp.shape != (len(input_order),)
                or xm.shape != xp.shape
                or not np.isfinite(xp).all()
                or not np.isfinite(xm).all()
            ):
                raise ValueError(f"{name}: measured input coordinates do not align")
            input_columns.append((xp - xm) / 2.0)
        else:
            nominal = np.zeros(len(input_order), dtype=float)
            nominal[mode_index] = amplitude
            input_columns.append(nominal)
        output_columns.append((yp - ym) / 2.0)
        bank_shas[name] = plus.get("bank_sha")
    X = np.column_stack(input_columns)
    Y = np.column_stack(output_columns)
    if Y.shape != (len(input_order), len(input_order)):
        raise ValueError(
            "projected response dimension must equal the registered input basis"
        )
    input_condition = float(np.linalg.cond(X))
    if (
        np.linalg.matrix_rank(X) != len(input_order)
        or not np.isfinite(input_condition)
        or input_condition > MAX_SPATIAL_BASIS_CONDITION
    ):
        raise ValueError(
            "measured input probe matrix is rank deficient or ill-conditioned"
        )
    operator = Y @ np.linalg.inv(X)
    return {
        "operator": operator,
        "input_matrix": X,
        "input_condition_number": input_condition,
        "input_order": list(input_order),
        "amplitude": amplitude,
        "bank_sha_by_mode": bank_shas,
        "modal_operator_version": MODAL_OPERATOR_VERSION,
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
