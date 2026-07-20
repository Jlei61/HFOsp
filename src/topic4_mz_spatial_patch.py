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
