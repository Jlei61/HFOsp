"""Deterministic spatial reductions for the additive MZ patch scaffold.

The primary P=2 object is the exact patch-constant Galerkin projection of the
locked periodic M3B sheet onto one pathological core and its full complement.
It is a cheap whole-sheet diagnostic, not a wavefront model: averaging the
entire complement necessarily dilutes activity confined to the core boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sef_hfo_field import convolve_periodic
from src.sef_hfo_lif import ELL_PAR, ELL_PERP, L_INH
from src.topic4_m3b_spectral_phase import (
    THETA_EE,
    Grid,
    build_kernels,
    make_core_mask,
)
from src.topic4_mz_spatial_patch import PatchKernels


@dataclass(frozen=True)
class CoreSurroundReduction:
    """Exact binary partition plus enough provenance to reproduce it."""

    kernels: PatchKernels
    core_mask: np.ndarray
    grid_n: int
    grid_L_mm: float
    grid_spacing_mm: float
    core_radius_mm: float
    core_cells: int
    surround_cells: int
    ell_parallel_mm: float
    ell_perpendicular_mm: float
    inhibitory_width_mm: float
    theta_rad: float


@dataclass(frozen=True)
class CoreAnnulusBathReduction:
    """Exact three-zone projection resolving the first recruitment shell."""

    kernels: PatchKernels
    masks: tuple[np.ndarray, np.ndarray, np.ndarray]
    patch_names: tuple[str, str, str]
    grid_n: int
    grid_L_mm: float
    grid_spacing_mm: float
    core_radius_mm: float
    outer_annulus_radius_mm: float
    patch_cells: tuple[int, int, int]
    ell_parallel_mm: float
    ell_perpendicular_mm: float
    inhibitory_width_mm: float
    theta_rad: float


def partition_block_average(
    kernel: np.ndarray,
    masks: tuple[np.ndarray, ...] | list[np.ndarray],
) -> np.ndarray:
    """Project a periodic convolution onto an exhaustive spatial partition."""

    values = np.asarray(kernel, dtype=float)
    parts = tuple(np.asarray(mask, dtype=bool) for mask in masks)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or len(parts) < 1:
        raise ValueError("kernel must be square and the partition non-empty")
    if any(part.shape != values.shape or not np.any(part) for part in parts):
        raise ValueError("every partition mask must be aligned and non-empty")
    occupancy = np.sum(np.asarray(parts, dtype=np.int8), axis=0)
    if not np.all(occupancy == 1):
        raise ValueError("partition masks must be disjoint and exhaustive")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("kernel must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("kernel must be L1 normalized")
    source_fields = [convolve_periodic(part.astype(float), values) for part in parts]
    return np.asarray(
        [[float(np.mean(source[target])) for source in source_fields] for target in parts]
    )


def binary_block_average(kernel: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    """Project a periodic convolution onto core/complement constant fields.

    Rows are target patches and columns are source patches.  There is no source
    area factor here: each patch coordinate is a per-cell rate.  Patch areas
    enter only through the shared recruitment aggregate.
    """

    core = np.asarray(core_mask, dtype=bool)
    if not np.any(core) or np.all(core):
        raise ValueError("binary reduction requires non-empty core and surround")
    return partition_block_average(kernel, (core, ~core))


def canonical_m3b_core_surround(
    *,
    grid_n: int = 48,
    grid_L_mm: float = 12.0,
    core_radius_mm: float = 1.5,
    theta_rad: float = THETA_EE,
) -> CoreSurroundReduction:
    """Build the locked .54/.27/.25-mm single-core P=2 projection."""

    grid = Grid(n=int(grid_n), L=float(grid_L_mm))
    spatial = build_kernels(
        grid,
        ar=float(ELL_PAR / ELL_PERP),
        ell_perp=float(ELL_PERP),
        l_inh=float(L_INH),
        theta=float(theta_rad),
    )
    core = make_core_mask(
        grid,
        kind="single",
        radius=float(core_radius_mm),
        theta=float(theta_rad),
    ).mask
    counts = np.asarray([np.count_nonzero(core), np.count_nonzero(~core)], dtype=float)
    weights = counts / counts.sum()
    patch = PatchKernels(
        K_EE=binary_block_average(spatial.K_EE, core),
        K_I=binary_block_average(spatial.K_I, core),
        patch_weights=weights,
    ).validate()
    return CoreSurroundReduction(
        kernels=patch,
        core_mask=core,
        grid_n=grid.n,
        grid_L_mm=grid.L,
        grid_spacing_mm=grid.spacing,
        core_radius_mm=float(core_radius_mm),
        core_cells=int(counts[0]),
        surround_cells=int(counts[1]),
        ell_parallel_mm=float(ELL_PAR),
        ell_perpendicular_mm=float(ELL_PERP),
        inhibitory_width_mm=float(L_INH),
        theta_rad=float(theta_rad),
    )


def canonical_m3b_core_annulus_bath(
    *,
    grid_n: int = 48,
    grid_L_mm: float = 12.0,
    core_radius_mm: float = 1.5,
    theta_rad: float = THETA_EE,
) -> CoreAnnulusBathReduction:
    """Resolve one equal-area recruitment annulus while retaining the far bath."""

    grid = Grid(n=int(grid_n), L=float(grid_L_mm))
    spatial = build_kernels(
        grid,
        ar=float(ELL_PAR / ELL_PERP),
        ell_perp=float(ELL_PERP),
        l_inh=float(L_INH),
        theta=float(theta_rad),
    )
    x, y = grid.coords()
    radius_squared = x**2 + y**2
    core_radius_squared = float(core_radius_mm) ** 2
    outer_radius_squared = 2.0 * core_radius_squared
    core = radius_squared <= core_radius_squared
    annulus = (radius_squared > core_radius_squared) & (radius_squared <= outer_radius_squared)
    bath = ~(core | annulus)
    masks = (core, annulus, bath)
    counts = np.asarray([np.count_nonzero(mask) for mask in masks], dtype=float)
    weights = counts / counts.sum()
    patch = PatchKernels(
        K_EE=partition_block_average(spatial.K_EE, masks),
        K_I=partition_block_average(spatial.K_I, masks),
        patch_weights=weights,
    ).validate()
    return CoreAnnulusBathReduction(
        kernels=patch,
        masks=masks,
        patch_names=("core", "annulus", "bath"),
        grid_n=grid.n,
        grid_L_mm=grid.L,
        grid_spacing_mm=grid.spacing,
        core_radius_mm=float(core_radius_mm),
        outer_annulus_radius_mm=float(np.sqrt(2.0) * float(core_radius_mm)),
        patch_cells=tuple(int(value) for value in counts),
        ell_parallel_mm=float(ELL_PAR),
        ell_perpendicular_mm=float(ELL_PERP),
        inhibitory_width_mm=float(L_INH),
        theta_rad=float(theta_rad),
    )
