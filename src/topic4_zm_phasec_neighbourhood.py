"""Fail-closed Phase-C carrier-maturation neighbourhood helpers.

This module is deliberately simulator-free.  It defines (1) physically valid
full-field slow-state probes and (2) the evidence aggregation contract.  The
runner is responsible for restoring the fast microstate and evaluating each
probe; this module never clips a requested slow state into a different one.

The primary coordinates are the two observed full-field fast-phase trajectories

    bounded_early -> bounded_mid -> bounded_late

for ``rising`` and ``peak`` separately.  Each trajectory contains its three
observed states plus the locked 50:50 early--mid and mid--late interpolants
(five cells per phase; ten cells per seed).  Pre-entry and onset-adjacent states
are deliberately not part of this carrier-maturation audit.

The secondary shell is a sensitivity analysis, limited to +/-0.25 robust SD
around primary cells.  A shell point outside either the observed componentwise
envelope plus 0.25 IQR, or the physical z/m/S_G bounds, is retained as
``invalid_physical`` rather than silently clipped.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from scipy.stats import beta as beta_dist


PHASEC_NEIGHBOURHOOD_VERSION = "zm_phasec_neighbourhood_v1.3_2026-07-29"
PRIMARY_STAGES = ("bounded_early", "bounded_mid", "bounded_late")
SHELL_STEP_SD = 0.25
ENVELOPE_IQR_PAD = 0.25
DEFAULT_PHASES = ("rising", "peak")
DEFAULT_NOISES = ("noise_replay", "noise_resample_1", "noise_resample_2")
PRIMARY_CELL_NAMES = tuple(
    f"primary__{phase}__{name}"
    for phase in DEFAULT_PHASES
    for name in (
        "bounded_early",
        "early_mid_midpoint",
        "bounded_mid",
        "mid_late_midpoint",
        "bounded_late",
    )
)
SHELL_DIRECTIONS = (
    "fullfield_mode2",
    "fullfield_mode3",
    "pathology_parallel",
    "pathology_perpendicular",
)
SHELL_CELL_NAMES = tuple(
    f"shell__bounded_mid__{direction}__{sign}0p25sd"
    for direction in SHELL_DIRECTIONS for sign in ("minus", "plus")
)


def _as_state(state):
    """Return a copied, shape-checked full-field state without clipping."""
    z = np.asarray(state["z"], float).reshape(-1).copy()
    m = np.asarray(state["m"], float).reshape(-1).copy()
    if z.shape != m.shape or z.size == 0:
        raise ValueError("z and m must be non-empty aligned full fields")
    sg = float(state["S_G"])
    return {"z": z, "m": m, "S_G": sg}


def _pack(state):
    s = _as_state(state)
    return np.concatenate([s["z"], s["m"], [s["S_G"]]])


def _unpack(vec, n_e):
    x = np.asarray(vec, float).reshape(-1)
    if x.size != 2 * int(n_e) + 1:
        raise ValueError("packed full-field state has the wrong dimension")
    return {
        "z": x[:n_e].copy(),
        "m": x[n_e:2 * n_e].copy(),
        "S_G": float(x[-1]),
    }


SUMMARY7_NAMES = (
    "z_core",
    "z_surround",
    "delta_z_parallel",
    "m_core",
    "m_surround",
    "delta_m_parallel",
    "S_G",
)
SUMMARY7_UNITS = (
    "dimensionless",
    "dimensionless",
    "dimensionless_per_axis_SD",
    "engine_m_state",
    "engine_m_state",
    "engine_m_state_per_axis_SD",
    "dimensionless",
)
SUMMARY7_DEFINITION = (
    "z_core=mean(z_i over pathology-core E); "
    "z_surround=mean(z_i over non-core E); "
    "delta_z_parallel=sum_i[a_i*(z_i-mean(z))]/sum_i[a_i^2]; "
    "m_core and m_surround are the analogous stratum means; "
    "delta_m_parallel=sum_i[a_i*(m_i-mean(m))]/sum_i[a_i^2]; "
    "a_i is the locked pathology-axis coordinate centred to zero mean and "
    "scaled to unit population SD; S_G is the scalar shared inhibition state"
)


def _standardized_axis(axis_coord, n_e):
    axis = np.asarray(axis_coord, np.float64).reshape(-1)
    if axis.shape != (int(n_e),) or not np.all(np.isfinite(axis)):
        raise ValueError("axis_coord must be a finite vector aligned with E fields")
    axis = axis - axis.mean()
    scale = float(axis.std())
    if scale <= 1e-12:
        raise ValueError("axis_coord is degenerate")
    return axis / scale


def _axial_projection(field, axis_coord):
    value = np.asarray(field, np.float64).reshape(-1)
    axis = _standardized_axis(axis_coord, value.size)
    centered = value - value.mean()
    return float(np.dot(axis, centered) / np.dot(axis, axis))


def summary7(state, core_mask, axis_coord):
    """Seven physical summaries used by the coarse empirical envelope.

    The third and sixth coordinates are true pathology-axis projections of the
    complete z and m fields.  They are not core-minus-surround contrasts.
    """
    s = _as_state(state)
    core = np.asarray(core_mask, bool).reshape(-1)
    if core.shape != s["z"].shape or not core.any() or core.all():
        raise ValueError("core_mask must align and contain core plus surround")
    zc, zs = float(s["z"][core].mean()), float(s["z"][~core].mean())
    mc, ms = float(s["m"][core].mean()), float(s["m"][~core].mean())
    return np.asarray([
        zc,
        zs,
        _axial_projection(s["z"], axis_coord),
        mc,
        ms,
        _axial_projection(s["m"], axis_coord),
        s["S_G"],
    ], np.float64)


def fit_physical_envelopes(
    observed_states,
    core_mask,
    axis_coord,
    pad_iqr=ENVELOPE_IQR_PAD,
):
    """Lock full-field and seven-summary envelopes from six observed states.

    Full-field limits are the componentwise 0.5--99.5 percentiles expanded by
    0.25 IQR.  Summary limits use the same rule in the seven-coordinate space.
    Physical z/m/S_G bounds are checked separately and never folded into these
    empirical limits.
    """
    if not np.isclose(float(pad_iqr), ENVELOPE_IQR_PAD):
        raise ValueError("Phase-C envelope padding is locked to 0.25 IQR")
    X = np.stack([_pack(s) for s in observed_states], axis=0)
    Q = np.stack([
        summary7(s, core_mask, axis_coord) for s in observed_states
    ], axis=0)

    def limits(a):
        lo, hi = np.quantile(a, [0.005, 0.995], axis=0)
        q25, q75 = np.quantile(a, [0.25, 0.75], axis=0)
        pad = float(pad_iqr) * (q75 - q25)
        return lo - pad, hi + pad

    field_lo, field_hi = limits(X)
    summary_lo, summary_hi = limits(Q)
    return {
        "full_field": (field_lo, field_hi),
        "summary7": (summary_lo, summary_hi),
        "full_field_quantiles": [0.005, 0.995],
        "summary_quantiles": [0.005, 0.995],
        "iqr_pad": float(pad_iqr),
        "summary7_names": list(SUMMARY7_NAMES),
        "summary7_units": list(SUMMARY7_UNITS),
        "summary7_definition": SUMMARY7_DEFINITION,
    }


def physical_status(
    state,
    *,
    full_field_envelope=None,
    summary_envelope=None,
    core_mask=None,
    axis_coord=None,
    tol=1e-12,
):
    """Validate a requested point; never repair it by clipping.

    ``envelope`` is a ``(lo, hi)`` pair in packed coordinates.  Bounds are
    inclusive up to ``tol``.
    """
    s = _as_state(state)
    reasons = []
    if not (np.all(np.isfinite(s["z"])) and np.all(np.isfinite(s["m"]))
            and np.isfinite(s["S_G"])):
        reasons.append("nonfinite")
    if np.any(s["z"] < -tol) or np.any(s["z"] > 1.0 + tol):
        reasons.append("z_physical_boundary")
    if np.any(s["m"] < -tol):
        reasons.append("m_physical_boundary")
    if s["S_G"] < -tol or s["S_G"] > 1.0 + tol:
        reasons.append("S_G_physical_boundary")
    if full_field_envelope is not None:
        lo, hi = (np.asarray(v, float).reshape(-1) for v in full_field_envelope)
        x = _pack(s)
        if lo.shape != x.shape or hi.shape != x.shape:
            raise ValueError("full-field envelope must align with packed state")
        if np.any(x < lo - tol) or np.any(x > hi + tol):
            reasons.append("full_field_percentile_envelope")
    if summary_envelope is not None:
        if core_mask is None or axis_coord is None:
            raise ValueError(
                "core_mask and axis_coord are required for the summary envelope"
            )
        lo, hi = (np.asarray(v, float).reshape(-1) for v in summary_envelope)
        q = summary7(s, core_mask, axis_coord)
        if lo.shape != q.shape or hi.shape != q.shape:
            raise ValueError("summary envelope must align with seven summaries")
        if np.any(q < lo - tol) or np.any(q > hi + tol):
            reasons.append("summary7_envelope")
    return {
        "status": "valid" if not reasons else "invalid_physical",
        "reasons": sorted(set(reasons)),
        "clipped": False,
    }


def observed_envelope(observed_states, pad_iqr=ENVELOPE_IQR_PAD):
    """Legacy helper: componentwise min/max plus IQR.

    Production Phase-C coordinates use :func:`fit_physical_envelopes` instead.
    """
    X = np.stack([_pack(s) for s in observed_states], axis=0)
    q25, q75 = np.quantile(X, [0.25, 0.75], axis=0)
    pad = float(pad_iqr) * (q75 - q25)
    return X.min(axis=0) - pad, X.max(axis=0) + pad


def _nested_observed(observed_by_phase):
    """Validate and copy the locked rising/peak x early/mid/late inventory."""
    missing = []
    states = {}
    for phase in DEFAULT_PHASES:
        if phase not in observed_by_phase:
            missing.append(phase)
            continue
        states[phase] = {}
        for stage in PRIMARY_STAGES:
            if stage not in observed_by_phase[phase]:
                missing.append(f"{phase}/{stage}")
            else:
                states[phase][stage] = _as_state(observed_by_phase[phase][stage])
    if missing:
        raise ValueError(f"missing primary observed states: {missing}")
    n_e = len(states[DEFAULT_PHASES[0]][PRIMARY_STAGES[0]]["z"])
    if any(len(states[p][s]["z"]) != n_e
           for p in DEFAULT_PHASES for s in PRIMARY_STAGES):
        raise ValueError("all observed full-field states must have the same size")
    return states


def build_primary_convex_path(
    observed_by_phase,
    *,
    core_mask=None,
    axis_coord=None,
    envelopes=None,
):
    """Build the locked ten-cell rising/peak full-field convex paths."""
    states = _nested_observed(observed_by_phase)

    def anchor_status(state, source):
        """Validate an exact observed anchor against intrinsic bounds only.

        The percentile envelopes describe extrapolation away from the six
        source fields.  They cannot make one of those source fields
        ``invalid_physical``.  Exact identity is checked before the empirical
        envelope is demoted to an audit annotation.
        """
        state_hash = slow_state_sha256(state)
        source_hash = slow_state_sha256(source)
        if state_hash != source_hash or not np.array_equal(
            _pack(state), _pack(source)
        ):
            raise AssertionError("exact primary anchor/source identity drift")
        hard = physical_status(state)
        empirical = physical_status(
            state,
            full_field_envelope=(
                None if envelopes is None else envelopes["full_field"]
            ),
            summary_envelope=(
                None if envelopes is None else envelopes["summary7"]
            ),
            core_mask=core_mask,
            axis_coord=axis_coord,
        )
        return {
            **hard,
            "validity_contract": "exact_observed_anchor_hard_bounds_only",
            "exact_observed_anchor": True,
            "source_slow_state_sha256": source_hash,
            "empirical_envelope_reasons": list(empirical["reasons"]),
        }

    def midpoint_status(state):
        return {
            **physical_status(
                state,
                full_field_envelope=(
                    None if envelopes is None else envelopes["full_field"]
                ),
                summary_envelope=(
                    None if envelopes is None else envelopes["summary7"]
                ),
                core_mask=core_mask,
                axis_coord=axis_coord,
            ),
            "validity_contract": "convex_midpoint_hard_plus_empirical_envelopes",
            "exact_observed_anchor": False,
            "source_slow_state_sha256": None,
            "empirical_envelope_reasons": [],
        }

    rows = []
    for phase in DEFAULT_PHASES:
        path_index = 0
        for seg, (left, right) in enumerate(zip(PRIMARY_STAGES[:-1], PRIMARY_STAGES[1:])):
            a, b = _pack(states[phase][left]), _pack(states[phase][right])
            for lam in (0.0, 0.5):
                state = _unpack((1.0 - lam) * a + lam * b, len(states[phase][left]["z"]))
                valid = (
                    anchor_status(state, states[phase][left])
                    if lam == 0.0
                    else midpoint_status(state)
                )
                rows.append({
                    "cell_id": (
                        f"primary__{phase}__{left}"
                        if lam == 0.0 else
                        f"primary__{phase}__{left.split('_')[-1]}_"
                        f"{right.split('_')[-1]}_midpoint"
                    ),
                    "kind": "primary_convex",
                    "trajectory_id": phase,
                    "source_fast_phase": phase,
                    "path_index": path_index,
                    "path_coordinate": float(seg + lam),
                    "path_direction": "forward",
                    "left_stage": left,
                    "right_stage": right,
                    "lambda": float(lam),
                    "state": state,
                    **valid,
                })
                path_index += 1
        final = states[phase][PRIMARY_STAGES[-1]]
        valid = anchor_status(final, states[phase][PRIMARY_STAGES[-1]])
        rows.append({
            "cell_id": f"primary__{phase}__{PRIMARY_STAGES[-1]}",
            "kind": "primary_convex",
            "trajectory_id": phase,
            "source_fast_phase": phase,
            "path_index": path_index,
            "path_coordinate": float(len(PRIMARY_STAGES) - 1),
            "path_direction": "forward",
            "left_stage": PRIMARY_STAGES[-1],
            "right_stage": PRIMARY_STAGES[-1],
            "lambda": 0.0,
            "state": final,
            **valid,
        })
    if tuple(row["cell_id"] for row in rows) != PRIMARY_CELL_NAMES:
        raise AssertionError("locked primary coordinate naming/order drift")
    return rows


def _robust_sd(x, axis=0):
    x = np.asarray(x, float)
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def _standardized_distance_to_anchor_manifold(
    vector,
    nested,
    center,
    safe_scale,
    movable,
):
    """RMS robust-coordinate distance to the two locked piecewise paths."""
    point = (np.asarray(vector, np.float64) - center) / safe_scale
    point[~movable] = 0.0
    best = np.inf
    for phase in DEFAULT_PHASES:
        anchors = [
            (_pack(nested[phase][stage]) - center) / safe_scale
            for stage in PRIMARY_STAGES
        ]
        for anchor in anchors:
            anchor[~movable] = 0.0
        for left, right in zip(anchors[:-1], anchors[1:]):
            delta = right - left
            denom = float(np.dot(delta, delta))
            lam = 0.0 if denom <= 1e-20 else float(
                np.clip(np.dot(point - left, delta) / denom, 0.0, 1.0)
            )
            residual = point - (left + lam * delta)
            best = min(
                best,
                float(np.linalg.norm(residual[movable])
                      / np.sqrt(max(1, int(movable.sum())))),
            )
    return float(best)


def _standardized_reconstruction_error(
    vector,
    *,
    center,
    safe_scale,
    movable,
    basis_vectors,
):
    """RMS residual after least-squares reconstruction in the locked basis."""
    point = (np.asarray(vector, np.float64) - center) / safe_scale
    point[~movable] = 0.0
    basis = np.stack([np.asarray(v, np.float64) for v in basis_vectors], axis=1)
    coefficient, *_ = np.linalg.lstsq(basis, point, rcond=None)
    residual = point - basis @ coefficient
    return float(
        np.linalg.norm(residual[movable])
        / np.sqrt(max(1, int(movable.sum())))
    )


def pathology_directions_from_geometry(
    observed_by_phase,
    *,
    axis_coord,
    perpendicular_coord,
):
    """Construct seed-local parallel/perpendicular slow-field directions.

    Geometry supplies only spatial shape.  Per-component empirical robust
    scales from that seed's six observed states convert the two spatial maps
    into physical z/m displacements.  The scalar S_G component is zero.  The
    transverse map is Gram--Schmidt orthogonalised against the parallel map in
    standardized full-field space.
    """
    nested = _nested_observed(observed_by_phase)
    observed = [
        nested[phase][stage]
        for phase in DEFAULT_PHASES for stage in PRIMARY_STAGES
    ]
    X = np.stack([_pack(s) for s in observed])
    n_e = len(observed[0]["z"])
    along = np.asarray(axis_coord, float).reshape(-1)
    perp = np.asarray(perpendicular_coord, float).reshape(-1)
    if along.size != n_e or perp.size != n_e:
        raise ValueError("pathology geometry must align with E slow fields")

    def standardize_spatial(x):
        x = x - x.mean()
        sd = float(x.std())
        if sd <= 1e-12:
            raise ValueError("pathology geometry coordinate is degenerate")
        return x / sd

    along = standardize_spatial(along)
    perp = standardize_spatial(perp)
    perp = perp - float(np.dot(perp, along) / np.dot(along, along)) * along
    perp = standardize_spatial(perp)

    scale = _robust_sd(X, axis=0)
    q25, q75 = np.quantile(X, [0.25, 0.75], axis=0)
    fallback = (q75 - q25) / 1.349
    scale = np.where(scale > 1e-12, scale, fallback)
    z_scale = scale[:n_e]
    m_scale = scale[n_e:2 * n_e]
    parallel = np.concatenate([along * z_scale, along * m_scale, [0.0]])
    perpendicular = np.concatenate([perp * z_scale, perp * m_scale, [0.0]])
    if np.linalg.norm(parallel) <= 1e-12 or np.linalg.norm(perpendicular) <= 1e-12:
        raise ValueError("pathology direction has no empirically movable support")
    return {
        "pathology_parallel": parallel,
        "pathology_perpendicular": perpendicular,
        "axis_coord": along,
        "perpendicular_coord": perp,
    }


def build_secondary_shell(
    observed_by_phase,
    *,
    pathology_directions,
    step_sd=SHELL_STEP_SD,
    envelope_pad_iqr=ENVELOPE_IQR_PAD,
    core_mask=None,
    axis_coord=None,
    envelopes=None,
    return_metadata=False,
):
    """Build the eight locked +/-0.25-SD shell cells around bounded-mid.

    Two full-field modes are estimated only after the observed trajectory
    tangent has been projected out.  Two additional directions are supplied by
    the locked pathology-axis geometry.  All four directions are fit within
    seed; no direction is shared across seeds.  The base point is the
    deterministic mean of the rising and peak bounded-mid slow fields.
    """
    if not np.isclose(float(step_sd), SHELL_STEP_SD):
        raise ValueError("the Phase-C secondary shell is locked to +/-0.25 robust SD")
    nested = _nested_observed(observed_by_phase)
    observed = [
        nested[phase][stage]
        for phase in DEFAULT_PHASES for stage in PRIMARY_STAGES
    ]
    X = np.stack([_pack(s) for s in observed])
    n_e = len(observed[0]["z"])
    center = np.median(X, axis=0)
    scale = _robust_sd(X, axis=0)
    q25, q75 = np.quantile(X, [0.25, 0.75], axis=0)
    fallback = (q75 - q25) / 1.349
    scale = np.where(scale > 1e-12, scale, fallback)
    movable = scale > 1e-12
    safe_scale = np.where(movable, scale, 1.0)
    Xs = (X - center) / safe_scale
    Xs[:, ~movable] = 0.0
    early = 0.5 * (
        _pack(nested["rising"]["bounded_early"])
        + _pack(nested["peak"]["bounded_early"])
    )
    late = 0.5 * (
        _pack(nested["rising"]["bounded_late"])
        + _pack(nested["peak"]["bounded_late"])
    )
    tangent = (late - early) / safe_scale
    tangent[~movable] = 0.0
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-12:
        raise ValueError("observed trajectory tangent is degenerate")
    tangent /= tangent_norm
    centered = Xs - Xs.mean(axis=0)
    residual = centered - np.outer(centered @ tangent, tangent)
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    non_tangent = []
    non_tangent_sign = []
    forward_derivative_physical = late - early
    for row in vt:
        candidate = row - np.dot(row, tangent) * tangent
        for prior in non_tangent:
            candidate -= np.dot(candidate, prior) * prior
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-10:
            continue
        candidate /= norm
        candidate_physical = candidate * safe_scale
        denom = float(
            np.linalg.norm(candidate_physical)
            * np.linalg.norm(forward_derivative_physical)
        )
        alignment = (
            0.0
            if denom <= 1e-20
            else float(
                np.dot(candidate_physical, forward_derivative_physical) / denom
            )
        )
        if abs(alignment) > 1e-10:
            sign_rule = "forward_trajectory_derivative"
            flip = alignment < 0.0
        else:
            # The non-tangent construction can make the forward projection
            # numerically zero.  In that case the sign must remain independent
            # of any future activity: orient the maximum-loading component.
            sign_rule = "deterministic_max_loading_fallback"
            j = int(np.argmax(np.abs(candidate)))
            flip = candidate[j] < 0
        if flip:
            candidate *= -1.0
            alignment *= -1.0
        non_tangent.append(candidate)
        non_tangent_sign.append({
            "rule": sign_rule,
            "forward_alignment_cosine": float(alignment),
            "near_zero_threshold": 1e-10,
        })
        if len(non_tangent) == 2:
            break
    if len(non_tangent) != 2:
        raise ValueError("fewer than two non-tangent full-field modes")
    if set(pathology_directions) != {"pathology_parallel", "pathology_perpendicular"}:
        raise ValueError("both locked pathology directions are required")
    directions = {
        "fullfield_mode2": non_tangent[0],
        "fullfield_mode3": non_tangent[1],
    }
    for name in ("pathology_parallel", "pathology_perpendicular"):
        physical = np.asarray(pathology_directions[name], float).reshape(-1)
        if physical.shape != center.shape:
            raise ValueError(f"{name} must align with the packed full field")
        candidate = physical / safe_scale
        candidate[~movable] = 0.0
        norm = float(np.linalg.norm(candidate))
        if norm <= 1e-12:
            raise ValueError(f"{name} is degenerate after standardisation")
        directions[name] = candidate / norm

    if envelopes is None:
        if core_mask is None:
            raise ValueError("core_mask is required to fit production envelopes")
        envelopes = fit_physical_envelopes(
            observed, core_mask, axis_coord, pad_iqr=envelope_pad_iqr
        )
    base_vec = 0.5 * (
        _pack(nested["rising"]["bounded_mid"])
        + _pack(nested["peak"]["bounded_mid"])
    )
    base_std = (base_vec - center) / safe_scale
    base_std[~movable] = 0.0
    rows = []
    for direction_name, direction in directions.items():
        projection = centered @ direction
        projection_sd = float(_robust_sd(projection, axis=0))
        if projection_sd <= 1e-12:
            raise ValueError(f"{direction_name} has zero observed robust trajectory SD")
        for sign in (-1, 1):
            shifted_std = base_std + sign * float(step_sd) * projection_sd * direction
            x = center + shifted_std * safe_scale
            x[~movable] = base_vec[~movable]
            state = _unpack(x, n_e)
            valid = physical_status(
                state,
                full_field_envelope=envelopes["full_field"],
                summary_envelope=envelopes["summary7"],
                core_mask=core_mask,
                axis_coord=axis_coord,
            )
            sign_name = "minus" if sign < 0 else "plus"
            rows.append({
                "cell_id": (
                    f"shell__bounded_mid__{direction_name}__"
                    f"{sign_name}0p25sd"
                ),
                "kind": "secondary_shell",
                "base_cell_id": "bounded_mid_mean_rising_peak",
                "trajectory_id": "secondary_shell",
                "source_fast_phase": "both",
                "path_index": 0,
                "path_coordinate": 0.0,
                "path_direction": direction_name,
                "basis_direction": direction_name,
                "sign": sign,
                "step_robust_sd": float(step_sd),
                "projection_robust_sd": projection_sd,
                "validity_contract": (
                    "shell_hard_plus_empirical_envelopes"
                ),
                "exact_observed_anchor": False,
                "state": state,
                **valid,
            })
    if len(rows) != 8:
        raise AssertionError("locked secondary shell must contain exactly eight cells")
    if tuple(row["cell_id"] for row in rows) != SHELL_CELL_NAMES:
        raise AssertionError("locked secondary-shell naming/order drift")
    if not return_metadata:
        return rows
    metadata = {
        "center": center,
        "component_scale": scale,
        "movable": movable,
        "trajectory_tangent_standardized": tangent,
        "basis_direction_names": list(SHELL_DIRECTIONS),
        "fullfield_mode_sign_alignment": {
            "fullfield_mode2": non_tangent_sign[0],
            "fullfield_mode3": non_tangent_sign[1],
        },
        "basis_directions_standardized": np.stack([
            directions[name] for name in SHELL_DIRECTIONS
        ]),
        "full_field_envelope_lo": envelopes["full_field"][0],
        "full_field_envelope_hi": envelopes["full_field"][1],
        "summary_envelope_lo": envelopes["summary7"][0],
        "summary_envelope_hi": envelopes["summary7"][1],
    }
    return rows, metadata


def build_coordinate_set(
    observed_by_phase,
    *,
    core_mask,
    axis_coord,
    perpendicular_coord,
):
    """Build one seed's complete locked 10-primary + 8-shell coordinates."""
    nested = _nested_observed(observed_by_phase)
    observed = [
        nested[phase][stage]
        for phase in DEFAULT_PHASES for stage in PRIMARY_STAGES
    ]
    standardized_axis = _standardized_axis(axis_coord, len(observed[0]["z"]))
    envelopes = fit_physical_envelopes(
        observed, core_mask, standardized_axis
    )
    primary = build_primary_convex_path(
        nested,
        core_mask=core_mask,
        axis_coord=standardized_axis,
        envelopes=envelopes,
    )
    geometry = pathology_directions_from_geometry(
        nested, axis_coord=axis_coord, perpendicular_coord=perpendicular_coord
    )
    shell, basis = build_secondary_shell(
        nested,
        pathology_directions={
            key: geometry[key]
            for key in ("pathology_parallel", "pathology_perpendicular")
        },
        core_mask=core_mask,
        axis_coord=standardized_axis,
        envelopes=envelopes,
        return_metadata=True,
    )
    center = np.asarray(basis["center"], np.float64)
    scale = np.asarray(basis["component_scale"], np.float64)
    movable = np.asarray(basis["movable"], bool)
    safe_scale = np.where(movable, scale, 1.0)
    reconstruction_basis = [
        np.asarray(basis["trajectory_tangent_standardized"], np.float64),
        *[
            np.asarray(row, np.float64)
            for row in basis["basis_directions_standardized"]
        ],
    ]
    for cell in [*primary, *shell]:
        vector = _pack(cell["state"])
        cell["summary7"] = summary7(
            cell["state"], core_mask, standardized_axis
        )
        cell["standardized_distance_from_anchor_manifold"] = (
            _standardized_distance_to_anchor_manifold(
                vector, nested, center, safe_scale, movable
            )
        )
        cell["reconstruction_error_standardized_rms"] = (
            _standardized_reconstruction_error(
                vector,
                center=center,
                safe_scale=safe_scale,
                movable=movable,
                basis_vectors=reconstruction_basis,
            )
        )
        if cell.get("exact_observed_anchor"):
            distance = cell["standardized_distance_from_anchor_manifold"]
            if not np.isfinite(distance) or distance > 1e-12:
                raise AssertionError(
                    "exact observed anchor is not on the source manifold"
                )
            if (
                slow_state_sha256(cell["state"])
                != cell["source_slow_state_sha256"]
            ):
                raise AssertionError(
                    "exact observed anchor slow-state hash drift"
                )
            cell["exact_observed_anchor_verified"] = True
    return {
        "version": PHASEC_NEIGHBOURHOOD_VERSION,
        "primary": primary,
        "secondary_shell": shell,
        "basis": basis,
        "geometry": {
            "core_mask": np.asarray(core_mask, bool).reshape(-1).copy(),
            "axis_coord": standardized_axis,
            "perpendicular_coord": geometry["perpendicular_coord"],
        },
        "summary7_contract": {
            "names": list(SUMMARY7_NAMES),
            "units": list(SUMMARY7_UNITS),
            "definition": SUMMARY7_DEFINITION,
            "axis_coordinate_normalization": (
                "centered pathology axial coordinate, population SD=1"
            ),
        },
    }


def coordinate_array_payload(coordinates):
    """Convert a coordinate set into a lossless allow-pickle-free NPZ payload.

    Every floating slow-state or basis array is serialized as float64.  The
    semantic state hashes are therefore invariant under NPZ round-trip.
    """
    cells = list(coordinates["primary"]) + list(coordinates["secondary_shell"])
    return {
        "cell_ids": np.asarray([c["cell_id"] for c in cells], dtype="U96"),
        "tiers": np.asarray([c["kind"] for c in cells], dtype="U32"),
        "status": np.asarray([c["status"] for c in cells], dtype="U32"),
        "z": np.stack([c["state"]["z"] for c in cells]).astype(np.float64),
        "m": np.stack([c["state"]["m"] for c in cells]).astype(np.float64),
        "S_G": np.asarray([c["state"]["S_G"] for c in cells], np.float64),
        "summary7": np.stack([c["summary7"] for c in cells]).astype(np.float64),
        "summary7_names": np.asarray(SUMMARY7_NAMES, dtype="U48"),
        "summary7_units": np.asarray(SUMMARY7_UNITS, dtype="U48"),
        "standardized_distance_from_anchor_manifold": np.asarray([
            c["standardized_distance_from_anchor_manifold"] for c in cells
        ], np.float64),
        "reconstruction_error_standardized_rms": np.asarray([
            c["reconstruction_error_standardized_rms"] for c in cells
        ], np.float64),
        "basis_direction_names": np.asarray(
            coordinates["basis"]["basis_direction_names"], dtype="U48"
        ),
        "basis_directions_standardized": np.asarray(
            coordinates["basis"]["basis_directions_standardized"], np.float64
        ),
        "trajectory_tangent_standardized": np.asarray(
            coordinates["basis"]["trajectory_tangent_standardized"], np.float64
        ),
        "component_scale": np.asarray(
            coordinates["basis"]["component_scale"], np.float64
        ),
        "full_field_envelope_lo": np.asarray(
            coordinates["basis"]["full_field_envelope_lo"], np.float64
        ),
        "full_field_envelope_hi": np.asarray(
            coordinates["basis"]["full_field_envelope_hi"], np.float64
        ),
        "summary_envelope_lo": np.asarray(
            coordinates["basis"]["summary_envelope_lo"], np.float64
        ),
        "summary_envelope_hi": np.asarray(
            coordinates["basis"]["summary_envelope_hi"], np.float64
        ),
        "axis_coord": np.asarray(
            coordinates["geometry"]["axis_coord"], np.float64
        ),
        "core_mask": np.asarray(
            coordinates["geometry"]["core_mask"], bool
        ),
        "perpendicular_coord": np.asarray(
            coordinates["geometry"]["perpendicular_coord"], np.float64
        ),
        "fullfield_mode_sign_rule": np.asarray([
            coordinates["basis"]["fullfield_mode_sign_alignment"][name]["rule"]
            for name in ("fullfield_mode2", "fullfield_mode3")
        ], dtype="U48"),
        "fullfield_mode_forward_alignment_cosine": np.asarray([
            coordinates["basis"]["fullfield_mode_sign_alignment"][name][
                "forward_alignment_cosine"
            ]
            for name in ("fullfield_mode2", "fullfield_mode3")
        ], np.float64),
    }


def _npy_bytes(array):
    out = BytesIO()
    np.lib.format.write_array(out, np.asarray(array), allow_pickle=False)
    return out.getvalue()


def deterministic_npz_bytes(arrays):
    """Return deterministic NPZ bytes with fixed member order/timestamps."""
    out = BytesIO()
    with zipfile.ZipFile(
        out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for name in sorted(arrays):
            if not name or "/" in name:
                raise ValueError(f"invalid NPZ array name: {name!r}")
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            archive.writestr(info, _npy_bytes(arrays[name]), compress_type=zipfile.ZIP_DEFLATED)
    return out.getvalue()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def semantic_array_sha256(arrays):
    """Hash array names, exact dtypes/shapes, and bytes independent of ZIP."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        h.update(
            f"{name}|{array.dtype.str}|{array.shape}|".encode("utf-8")
        )
        h.update(array.tobytes())
    return h.hexdigest()


def slow_state_sha256(state):
    """Semantic hash of one exact float64 Z/M/S_G state."""
    s = _as_state(state)
    h = hashlib.sha256()
    for name in ("z", "m"):
        array = np.ascontiguousarray(np.asarray(s[name], np.float64))
        h.update(f"{name}|{array.dtype.str}|{array.shape}|".encode("utf-8"))
        h.update(array.tobytes())
    h.update(f"S_G|{float(s['S_G']):.17g}".encode("utf-8"))
    return h.hexdigest()


def canonical_json_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def write_bytes_once(path, value):
    """Write immutable bytes; exact reuse allowed, conflicts rejected."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError(f"existing coordinate artifact differs: {path}")
        return "reused"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != value:
                raise RuntimeError(f"existing coordinate artifact differs: {path}")
            return "reused"
        return "created"
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def write_json_once(path, value):
    """Write an immutable, normalized JSON coordinate manifest."""
    pretty = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    return write_bytes_once(path, pretty)


def jeffreys_summary(k, n, ci=0.95):
    """Jeffreys Beta(1/2,1/2) posterior summary."""
    k, n = int(k), int(n)
    if n < 0 or k < 0 or k > n:
        raise ValueError("require 0 <= k <= n")
    alpha = (1.0 - float(ci)) / 2.0
    return {
        "k": k,
        "n": n,
        "posterior_mean": float((k + 0.5) / (n + 1.0)),
        "posterior_median": float(beta_dist.ppf(0.5, k + 0.5, n - k + 0.5)),
        "posterior_ci": [
            float(beta_dist.ppf(alpha, k + 0.5, n - k + 0.5)),
            float(beta_dist.ppf(1.0 - alpha, k + 0.5, n - k + 0.5)),
        ],
    }


def aggregate_cell(run_rows, *, phases=DEFAULT_PHASES, noises=DEFAULT_NOISES,
                   pass_key="mature_pass", required_successes=5):
    """Aggregate exactly 2 phases x 3 paired futures; missing is indeterminate."""
    expected = {(str(p), str(n)) for p in phases for n in noises}
    by_key = {}
    duplicate = False
    for row in run_rows:
        key = (str(row.get("phase")), str(row.get("noise")))
        if key in by_key:
            duplicate = True
        by_key[key] = row
    missing = sorted(expected - set(by_key))
    extra = sorted(set(by_key) - expected)
    invalid = [
        key for key in expected & set(by_key)
        if by_key[key].get("status", "complete") != "complete"
        or by_key[key].get(pass_key) is None
    ]
    if duplicate or missing or extra or invalid:
        return {
            "status": "indeterminate",
            "reason": "incomplete_or_invalid_2x3_cell",
            "missing": missing,
            "extra": extra,
            "invalid": sorted(invalid),
            "duplicate": duplicate,
            **jeffreys_summary(0, 0),
        }
    successes = [bool(by_key[key][pass_key]) for key in sorted(expected)]
    k, n = int(sum(successes)), len(successes)
    directions = [
        str(by_key[key].get("maturation_direction"))
        for key in sorted(expected)
        if by_key[key][pass_key] and by_key[key].get("maturation_direction") is not None
    ]
    direction = None
    if directions:
        direction, n_dir = Counter(directions).most_common(1)[0]
        if n_dir < required_successes:
            direction = "mixed"
    return {
        "status": "pass" if k >= int(required_successes) else "fail",
        "maturation_direction": direction,
        "reason": f"{k}/{n} maturation replicates",
        "missing": [],
        "extra": [],
        "invalid": [],
        "duplicate": False,
        **jeffreys_summary(k, n),
    }


def aggregate_run_table(run_rows, **kwargs):
    """Group run rows by representation, seed and cell, preserving path metadata."""
    groups = defaultdict(list)
    for row in run_rows:
        groups[(str(row["representation"]), int(row["seed"]), str(row["cell_id"]))].append(row)
    out = []
    for (representation, seed, cell_id), rows in sorted(groups.items()):
        agg = aggregate_cell(rows, **kwargs)
        first = rows[0]
        out.append({
            "representation": representation,
            "seed": seed,
            "cell_id": cell_id,
            "path_index": int(first["path_index"]),
            "trajectory_id": str(first.get("trajectory_id", "primary")),
            "path_direction": str(first.get("path_direction", "forward")),
            **agg,
        })
    return out


def _seed_window(cells, expected_cell_ids, min_adjacent=2):
    by_id = {str(c["cell_id"]): c for c in cells}
    expected = {str(x) for x in expected_cell_ids}
    missing = sorted(expected - set(by_id))
    indeterminate = sorted(
        cid for cid in expected & set(by_id)
        if by_id[cid].get("status") not in {"pass", "fail"}
    )
    if missing or indeterminate:
        return {
            "status": "insufficient_coverage",
            "direction": None,
            "window": [],
            "missing": missing,
            "indeterminate": indeterminate,
        }
    passed = sorted(
        (by_id[cid] for cid in expected if by_id[cid]["status"] == "pass"),
        key=lambda x: (str(x.get("trajectory_id", "primary")), int(x["path_index"])),
    )
    for left, right in zip(passed[:-1], passed[1:]):
        if str(left.get("trajectory_id", "primary")) != str(
                right.get("trajectory_id", "primary")):
            continue
        if int(right["path_index"]) != int(left["path_index"]) + 1:
            continue
        d1 = left.get("maturation_direction") or left.get("path_direction")
        d2 = right.get("maturation_direction") or right.get("path_direction")
        if d1 == d2 and d1 not in {None, "mixed"}:
            return {
                "status": "local_window",
                "direction": str(d1),
                "window": [left["cell_id"], right["cell_id"]],
                "missing": [],
                "indeterminate": [],
            }
    return {
        "status": "no_window",
        "direction": None,
        "window": [],
        "missing": [],
        "indeterminate": [],
    }


def _representation_verdict(cells, expected_by_seed, eligible_seeds=(1, 3, 4)):
    seed_results = {}
    for seed in eligible_seeds:
        rows = [c for c in cells if int(c["seed"]) == int(seed)]
        seed_results[int(seed)] = _seed_window(rows, expected_by_seed.get(int(seed), ()))
    windows = [v for v in seed_results.values() if v["status"] == "local_window"]
    counts = Counter(v["direction"] for v in windows)
    if counts:
        direction, n_same = counts.most_common(1)[0]
        opposite = [v for v in windows if v["direction"] != direction]
        if n_same >= 2 and not opposite:
            return {
                "status": "local_window",
                "direction": direction,
                "seed_results": seed_results,
            }
        if len(counts) > 1:
            return {
                "status": "direction_conflict",
                "direction": None,
                "seed_results": seed_results,
            }
    if any(v["status"] == "insufficient_coverage" for v in seed_results.values()):
        return {
            "status": "insufficient_coverage",
            "direction": None,
            "seed_results": seed_results,
        }
    # Strict negative: all three eligible seeds complete and all explicitly no-window.
    if (len(tuple(eligible_seeds)) == 3
            and all(v["status"] == "no_window" for v in seed_results.values())):
        return {
            "status": "no_window",
            "direction": None,
            "seed_results": seed_results,
        }
    return {
        "status": "insufficient_coverage",
        "direction": None,
        "seed_results": seed_results,
    }


def adjudicate_phasec_neighbourhood(aggregated_cells, expected_by_representation,
                                    *, eligible_seeds=(1, 3, 4)):
    """Adjudicate local maturation with explicit representation sensitivity.

    A positive requires two adjacent primary cells in >=2/3 seeds in the same
    direction, with the third seed evaluated and not showing the reverse
    direction.  A strict negative requires complete 3/3 evidence.  Missing or
    indeterminate evidence can only yield ``insufficient_coverage``.
    """
    rep_results = {}
    for representation, expected_by_seed in expected_by_representation.items():
        cells = [
            c for c in aggregated_cells
            if str(c["representation"]) == str(representation)
        ]
        rep_results[str(representation)] = _representation_verdict(
            cells, expected_by_seed, eligible_seeds=eligible_seeds
        )
    statuses = {v["status"] for v in rep_results.values()}
    directions = {
        v["direction"] for v in rep_results.values()
        if v["status"] == "local_window"
    }
    if "direction_conflict" in statuses or len(directions) > 1:
        verdict = "representation_sensitive"
    elif "insufficient_coverage" in statuses:
        verdict = "insufficient_coverage"
    elif statuses == {"local_window"}:
        verdict = "local_maturation_window"
    elif statuses == {"no_window"}:
        verdict = "no_local_maturation_window"
    else:
        # A window appearing in only one representation is not a robust window.
        verdict = "representation_sensitive"
    return {
        "version": PHASEC_NEIGHBOURHOOD_VERSION,
        "verdict": verdict,
        "representation_results": rep_results,
    }
