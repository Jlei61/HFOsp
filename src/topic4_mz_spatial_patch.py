"""Shared-pool P-patch fast scaffold for the additive MZ lifecycle line.

This module first establishes a strict P=1/uniform-P parity boundary.  Each
patch carries seven Stage-0C local fast variables plus local ``z/p/m``; the
entire domain carries exactly one shared ``mu_G/S_G`` pair.  The current
implementation deliberately freezes all three local slow variables.  Their
autonomous equations are added only after the fast scaffold passes parity.

State order is field-major::

    [rE[P], rI[P], sEE[P], sEI[P], sIE[P], sII[P], rE_fast[P],
     z[P], p[P], m[P], mu_G, S_G]

so the continuous state has size ``10*P + 2``.  This is not Stage-0C batch
vectorisation: ``mu_G`` and ``S_G`` are single shared scalars, never one pair
per patch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.sef_hfo_lif import TAU_AMPA, TAU_GABA, TAU_ME, TAU_MI
from src.sef_hfo_lif import (
    C_EE,
    C_EI,
    C_IE,
    C_II,
    JX_E,
    JX_I,
    W_EE,
    W_EI,
    W_IE,
    W_II,
    nu_theta_pop,
)
from src.topic4_spatial_slowfast_stage0c import (
    S_MAX,
    TAU_FAST_MS,
    TAU_MU_MS,
    TAU_S_MS,
    PoolParameters,
    recruitment_sensor,
)
from src.topic4_spatial_slowfast_stage0c_transfer import (
    moments_from_prepared,
    prepare_pool_parameters,
)


LOCAL_FIELDS: tuple[str, ...] = (
    "rE",
    "rI",
    "sEE",
    "sEI",
    "sIE",
    "sII",
    "rE_fast",
    "z",
    "p",
    "m",
)
SHARED_FIELDS: tuple[str, ...] = ("mu_G", "S_G")


@dataclass(frozen=True)
class PatchKernels:
    """Fixed row-stochastic coupling from source-patch rates to targets."""

    K_EE: np.ndarray
    K_I: np.ndarray
    patch_weights: np.ndarray | None = None

    @property
    def n_patches(self) -> int:
        return int(np.asarray(self.K_EE).shape[0])

    def validate(self, *, tolerance: float = 1.0e-12) -> "PatchKernels":
        e = np.asarray(self.K_EE, dtype=float)
        i = np.asarray(self.K_I, dtype=float)
        if e.ndim != 2 or e.shape[0] == 0 or e.shape[0] != e.shape[1] or i.shape != e.shape:
            raise ValueError("patch kernels must be aligned non-empty square matrices")
        if not np.all(np.isfinite(e)) or not np.all(np.isfinite(i)):
            raise ValueError("patch kernels must be finite")
        if np.any(e < 0.0) or np.any(i < 0.0):
            raise ValueError("patch kernels must be non-negative")
        if not np.allclose(e.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("K_EE must preserve a constant field")
        if not np.allclose(i.sum(axis=1), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("K_I must preserve a constant field")
        weights = self.weights()
        if weights.shape != (e.shape[0],) or np.any(weights <= 0.0):
            raise ValueError("patch weights must be positive and aligned")
        if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=tolerance):
            raise ValueError("patch weights must sum to one")
        if not np.allclose(weights @ e, weights, rtol=0.0, atol=tolerance):
            raise ValueError("patch weights must be stationary for K_EE")
        if not np.allclose(weights @ i, weights, rtol=0.0, atol=tolerance):
            raise ValueError("patch weights must be stationary for K_I")
        return self

    def weights(self) -> np.ndarray:
        """Return normalized patch-area weights without changing the state."""

        if self.patch_weights is None:
            return np.full(self.n_patches, 1.0 / self.n_patches, dtype=float)
        weights = np.asarray(self.patch_weights, dtype=float)
        if weights.shape != (self.n_patches,) or not np.all(np.isfinite(weights)):
            raise ValueError("patch weights must be finite and aligned")
        return weights

    @classmethod
    def identity(cls, n_patches: int) -> "PatchKernels":
        if n_patches < 1:
            raise ValueError("n_patches must be positive")
        matrix = np.eye(int(n_patches), dtype=float)
        return cls(matrix.copy(), matrix.copy(), np.full(n_patches, 1.0 / n_patches))


@dataclass(frozen=True)
class PatchParameters:
    """Parameters shared by all patches while local z/m live in the state."""

    alpha_g: float = 15.0
    w_ee_mult: float = 1.1
    ratio: float = 1.0
    additive_max_mv: float = 1.6
    pool_p: float = 1.0

    def validate(self) -> "PatchParameters":
        values = (self.alpha_g, self.w_ee_mult, self.ratio, self.additive_max_mv, self.pool_p)
        if not all(np.isfinite(values)):
            raise ValueError("patch parameters must be finite")
        if self.alpha_g < 0.0 or self.w_ee_mult <= 0.0 or self.ratio <= 0.0:
            raise ValueError("require alpha_g>=0, w_ee_mult>0, ratio>0")
        if self.additive_max_mv < 0.0 or self.pool_p < 1.0:
            raise ValueError("require additive_max_mv>=0 and pool_p>=1")
        return self


@dataclass(frozen=True)
class PreparedPatchRHS:
    """Prevalidated immutable coefficients for the P-patch hot loop.

    Local ``z`` and ``m`` deliberately remain in the continuous state.  In
    particular, ``z*W_EI`` must never be cached here because the next gate adds
    autonomous local resource dynamics.
    """

    K_EE: np.ndarray
    K_I: np.ndarray
    patch_weights: np.ndarray
    alpha_g: float
    w_ee: float
    nu_ext: float
    additive_max_mv: float
    pool_p: float

    @property
    def n_patches(self) -> int:
        return int(self.K_EE.shape[0])


def prepare_patch_rhs(
    kernels: PatchKernels,
    parameters: PatchParameters,
) -> PreparedPatchRHS:
    """Validate once and materialize the coefficients reused at every step."""

    checked_kernels = kernels.validate()
    checked_parameters = parameters.validate()
    arrays = (
        np.asarray(checked_kernels.K_EE, dtype=float).copy(),
        np.asarray(checked_kernels.K_I, dtype=float).copy(),
        checked_kernels.weights().copy(),
    )
    for value in arrays:
        value.setflags(write=False)
    return PreparedPatchRHS(
        K_EE=arrays[0],
        K_I=arrays[1],
        patch_weights=arrays[2],
        alpha_g=float(checked_parameters.alpha_g),
        w_ee=float(checked_parameters.w_ee_mult * W_EE),
        nu_ext=float(checked_parameters.ratio * nu_theta_pop()),
        additive_max_mv=float(checked_parameters.additive_max_mv),
        pool_p=float(checked_parameters.pool_p),
    )


def state_size(n_patches: int) -> int:
    if n_patches < 1:
        raise ValueError("n_patches must be positive")
    return len(LOCAL_FIELDS) * int(n_patches) + len(SHARED_FIELDS)


def pack_patch_state(
    local: Mapping[str, Sequence[float]],
    *,
    mu_g: float,
    s_g: float,
) -> np.ndarray:
    """Pack aligned local fields and one shared pool pair."""

    if set(local) != set(LOCAL_FIELDS):
        raise ValueError(f"local fields must be exactly {LOCAL_FIELDS}")
    arrays = {name: np.asarray(local[name], dtype=float) for name in LOCAL_FIELDS}
    sizes = {value.size for value in arrays.values() if value.ndim == 1}
    if len(sizes) != 1 or any(value.ndim != 1 for value in arrays.values()):
        raise ValueError("all local fields must be aligned 1D arrays")
    n_patches = next(iter(sizes))
    if n_patches < 1 or not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("local fields must be non-empty and finite")
    if not np.isfinite(mu_g) or not np.isfinite(s_g):
        raise ValueError("shared pool states must be finite")
    return np.concatenate([*(arrays[name] for name in LOCAL_FIELDS), [float(mu_g), float(s_g)]])


def unpack_patch_state(state: Sequence[float], n_patches: int) -> tuple[dict[str, np.ndarray], float, float]:
    """Unpack a field-major P-patch state without duplicating the pool."""

    vector = np.asarray(state, dtype=float)
    if vector.shape != (state_size(n_patches),) or not np.all(np.isfinite(vector)):
        raise ValueError(f"state must be finite with shape ({state_size(n_patches)},)")
    local = {
        name: vector[index * n_patches:(index + 1) * n_patches]
        for index, name in enumerate(LOCAL_FIELDS)
    }
    return local, float(vector[-2]), float(vector[-1])


def stage0c_to_patch_state(
    stage_state: Sequence[float],
    *,
    z: float,
    additive_mv: float,
    parameters: PatchParameters,
    persistence: float = 0.0,
) -> np.ndarray:
    """Embed one nine-state Stage-0C state in the P=1 manifold."""

    stage = np.asarray(stage_state, dtype=float)
    params = parameters.validate()
    if stage.shape != (9,) or not np.all(np.isfinite(stage)):
        raise ValueError("Stage-0C state must be finite with shape (9,)")
    if not 0.0 < z <= 1.0 or additive_mv < 0.0:
        raise ValueError("require z in (0,1] and non-negative additive current")
    if params.additive_max_mv == 0.0:
        if additive_mv != 0.0:
            raise ValueError("nonzero additive current requires additive_max_mv>0")
        m = 0.0
    else:
        m = float(additive_mv) / float(params.additive_max_mv)
    if not 0.0 <= m <= 1.0 or not 0.0 <= persistence <= 1.0:
        raise ValueError("embedded p/m must lie in [0,1]")
    local = {
        name: np.asarray([stage[index]], dtype=float)
        for index, name in enumerate(LOCAL_FIELDS[:7])
    }
    local.update(z=np.asarray([z]), p=np.asarray([persistence]), m=np.asarray([m]))
    return pack_patch_state(local, mu_g=float(stage[7]), s_g=float(stage[8]))


def patch_to_stage0c_state(state: Sequence[float]) -> np.ndarray:
    """Project the P=1 fast/shared coordinates back to Stage 0C."""

    local, mu_g, s_g = unpack_patch_state(state, 1)
    return np.asarray([*(float(local[name][0]) for name in LOCAL_FIELDS[:7]), mu_g, s_g])


def _stage_rows(
    local: Mapping[str, np.ndarray],
    mu_g: float,
    s_g: float,
) -> np.ndarray:
    """Build one Stage-0C-shaped row per patch using the shared pool pair."""

    n_patches = int(local["rE"].size)
    return np.column_stack(
        [*(local[name] for name in LOCAL_FIELDS[:7]),
         np.full(n_patches, mu_g), np.full(n_patches, s_g)]
    )


def patch_rhs_and_moments(
    state: Sequence[float],
    kernels: PatchKernels,
    parameters: PatchParameters,
    transfer: Any,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate the frozen-slow P-patch RHS with one shared dynamic pool."""

    kernels.validate()
    params = parameters.validate()
    n_patches = kernels.n_patches
    local, mu_g, s_g = unpack_patch_state(state, n_patches)
    if np.any((local["z"] <= 0.0) | (local["z"] > 1.0)):
        raise ValueError("local z must lie in (0,1]")
    if np.any((local["p"] < 0.0) | (local["p"] > 1.0)):
        raise ValueError("local p must lie in [0,1]")
    if np.any((local["m"] < 0.0) | (local["m"] > 1.0)):
        raise ValueError("local m must lie in [0,1]")
    points = [
        PoolParameters(float(z), params.alpha_g, params.w_ee_mult, params.ratio)
        for z in local["z"]
    ]
    prepared = prepare_pool_parameters(points)
    stage_rows = _stage_rows(local, mu_g, s_g)
    mu_e, sigma_e, mu_i, sigma_i, s_eff = moments_from_prepared(stage_rows, prepared)
    mu_e_effective = mu_e - params.additive_max_mv * local["m"]
    target_e = np.asarray(transfer.rate(mu_e_effective, sigma_e, "E"), dtype=float)
    target_i = np.asarray(transfer.rate(mu_i, sigma_i, "I"), dtype=float)
    if target_e.shape != (n_patches,) or target_i.shape != (n_patches,):
        raise RuntimeError("transfer returned an invalid patch target shape")

    out = {name: np.zeros(n_patches, dtype=float) for name in LOCAL_FIELDS}
    out["rE"] = (-local["rE"] + target_e) / TAU_ME
    out["rI"] = (-local["rI"] + target_i) / TAU_MI
    out["sEE"] = (kernels.K_EE @ local["rE"] - local["sEE"]) / TAU_AMPA
    out["sEI"] = (kernels.K_I @ local["rI"] - local["sEI"]) / TAU_GABA
    out["sIE"] = (kernels.K_I @ local["rE"] - local["sIE"]) / TAU_AMPA
    out["sII"] = (kernels.K_I @ local["rI"] - local["sII"]) / TAU_GABA
    out["rE_fast"] = (local["rE"] - local["rE_fast"]) / TAU_FAST_MS
    # P=1 parity gate: the new slow dynamics are deliberately absent here.
    out["z"].fill(0.0)
    out["p"].fill(0.0)
    out["m"].fill(0.0)

    sensor = recruitment_sensor(local["rE_fast"])
    area_g = float(np.sum(kernels.weights() * sensor ** params.pool_p)) ** (1.0 / params.pool_p)
    d_mu_g = (-mu_g + area_g) / TAU_MU_MS
    d_s_g = (-s_g + S_MAX * mu_g) / TAU_S_MS
    rhs = pack_patch_state(out, mu_g=d_mu_g, s_g=d_s_g)
    return rhs, (mu_e_effective, sigma_e, mu_i, sigma_i, s_eff)


def patch_rhs(
    state: Sequence[float],
    kernels: PatchKernels,
    parameters: PatchParameters,
    transfer: Any,
) -> np.ndarray:
    """Convenience wrapper returning only the frozen-slow vector field."""

    return patch_rhs_and_moments(state, kernels, parameters, transfer)[0]


def patch_rhs_fast_and_moments(
    state: Sequence[float] | np.ndarray,
    prepared: PreparedPatchRHS,
    transfer: Any,
    external_e_mv: float | Sequence[float] | np.ndarray | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Evaluate one or a batch of states without validation in the hot loop.

    This is an algebraic expansion of :func:`patch_rhs_and_moments`, not a new
    approximation.  Callers must validate the initial state and audit transfer
    support / finite bounds while integrating.  A one-dimensional input returns
    one-dimensional RHS and moment vectors; a two-dimensional input is treated
    as independent forks sharing the same spatial operator.
    """

    vector = np.asarray(state, dtype=float)
    one = vector.ndim == 1
    batch = vector[None, :] if one else vector
    n_patches = prepared.n_patches
    if batch.ndim != 2 or batch.shape[1] != state_size(n_patches):
        raise ValueError(
            f"state must have shape ({state_size(n_patches)},) or (n,{state_size(n_patches)})"
        )

    def local(name: str) -> np.ndarray:
        index = LOCAL_FIELDS.index(name)
        return batch[:, index * n_patches:(index + 1) * n_patches]

    r_e = local("rE")
    r_i = local("rI")
    s_ee = local("sEE")
    s_ei = local("sEI")
    s_ie = local("sIE")
    s_ii = local("sII")
    r_e_fast = local("rE_fast")
    z = local("z")
    m = local("m")
    mu_g = batch[:, -2]
    s_g = batch[:, -1]

    divisor = 1.0 + prepared.alpha_g * s_g[:, None]
    divisor = np.where(np.isfinite(divisor) & (divisor > 0.0), divisor, np.nan)
    w_ei = z * W_EI
    recurrent_mean_e = TAU_ME * C_EE * prepared.w_ee * s_ee / divisor
    recurrent_var_e = TAU_ME * C_EE * prepared.w_ee**2 * s_ee / divisor**2
    mu_e = (
        recurrent_mean_e
        - TAU_ME * C_EI * w_ei * s_ei
        + TAU_ME * JX_E * prepared.nu_ext
        - prepared.additive_max_mv * m
    )
    if external_e_mv is not None:
        external = np.asarray(external_e_mv, dtype=float)
        if external.ndim == 0:
            external = np.full(mu_e.shape, float(external), dtype=float)
        elif external.shape == (n_patches,):
            external = np.broadcast_to(external[None, :], mu_e.shape)
        elif external.shape != mu_e.shape:
            raise ValueError(
                "external_e_mv must be scalar, patch-aligned, or batch-by-patch aligned"
            )
        if not np.all(np.isfinite(external)):
            raise ValueError("external_e_mv must be finite")
        mu_e = mu_e + external
    var_e = (
        recurrent_var_e
        + TAU_ME * C_EI * w_ei**2 * s_ei
        + TAU_ME * JX_E**2 * prepared.nu_ext
    )
    mu_i = (
        TAU_MI * (C_IE * W_IE * s_ie - C_II * W_II * s_ii)
        + TAU_MI * JX_I * prepared.nu_ext
    )
    var_i = (
        TAU_MI * (C_IE * W_IE**2 * s_ie + C_II * W_II**2 * s_ii)
        + TAU_MI * JX_I**2 * prepared.nu_ext
    )
    sigma_e = np.sqrt(np.maximum(var_e, 1e-9))
    sigma_i = np.sqrt(np.maximum(var_i, 1e-9))
    target_e = np.asarray(transfer.rate(mu_e, sigma_e, "E"), dtype=float)
    target_i = np.asarray(transfer.rate(mu_i, sigma_i, "I"), dtype=float)

    out = np.zeros_like(batch)

    def put(name: str, value: np.ndarray) -> None:
        index = LOCAL_FIELDS.index(name)
        out[:, index * n_patches:(index + 1) * n_patches] = value

    put("rE", (-r_e + target_e) / TAU_ME)
    put("rI", (-r_i + target_i) / TAU_MI)
    put("sEE", (r_e @ prepared.K_EE.T - s_ee) / TAU_AMPA)
    put("sEI", (r_i @ prepared.K_I.T - s_ei) / TAU_GABA)
    put("sIE", (r_e @ prepared.K_I.T - s_ie) / TAU_AMPA)
    put("sII", (r_i @ prepared.K_I.T - s_ii) / TAU_GABA)
    put("rE_fast", (r_e - r_e_fast) / TAU_FAST_MS)
    # The first spatial gate is frozen-slow by construction.
    put("z", np.zeros_like(z))
    put("p", np.zeros_like(z))
    put("m", np.zeros_like(z))

    sensor = recruitment_sensor(r_e_fast)
    area_g = np.sum(
        prepared.patch_weights[None, :] * sensor**prepared.pool_p,
        axis=1,
    ) ** (1.0 / prepared.pool_p)
    out[:, -2] = (-mu_g + area_g) / TAU_MU_MS
    out[:, -1] = (-s_g + S_MAX * mu_g) / TAU_S_MS
    moments = (mu_e, sigma_e, mu_i, sigma_i, np.broadcast_to(s_g[:, None], mu_e.shape))
    if one:
        return out[0], tuple(value[0] for value in moments)  # type: ignore[return-value]
    return out, moments


def patch_rhs_fast(
    state: Sequence[float] | np.ndarray,
    prepared: PreparedPatchRHS,
    transfer: Any,
    external_e_mv: float | Sequence[float] | np.ndarray | None = None,
) -> np.ndarray:
    """Hot-loop wrapper returning only the vector field."""

    return patch_rhs_fast_and_moments(
        state, prepared, transfer, external_e_mv=external_e_mv
    )[0]


def patch_rhs_to_stage0c(rhs: Sequence[float]) -> np.ndarray:
    """Project one P=1 RHS onto the nine Stage-0C derivatives."""

    return patch_to_stage0c_state(rhs)


def uniform_patch_state(
    stage_state: Sequence[float],
    *,
    n_patches: int,
    z: float,
    additive_mv: float,
    parameters: PatchParameters,
    persistence: float = 0.0,
) -> np.ndarray:
    """Embed a Stage-0C state on the constant spatial manifold."""

    if n_patches < 1:
        raise ValueError("n_patches must be positive")
    p1 = stage0c_to_patch_state(
        stage_state,
        z=z,
        additive_mv=additive_mv,
        parameters=parameters,
        persistence=persistence,
    )
    local_one, mu_g, s_g = unpack_patch_state(p1, 1)
    local = {name: np.full(n_patches, float(value[0])) for name, value in local_one.items()}
    return pack_patch_state(local, mu_g=mu_g, s_g=s_g)
