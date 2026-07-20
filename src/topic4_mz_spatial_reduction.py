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


def binary_block_average(kernel: np.ndarray, core_mask: np.ndarray) -> np.ndarray:
    """Project a periodic convolution onto core/complement constant fields.

    Rows are target patches and columns are source patches.  There is no source
    area factor here: each patch coordinate is a per-cell rate.  Patch areas
    enter only through the shared recruitment aggregate.
    """

    values = np.asarray(kernel, dtype=float)
    core = np.asarray(core_mask, dtype=bool)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or core.shape != values.shape:
        raise ValueError("kernel and core mask must be aligned square arrays")
    if not np.any(core) or np.all(core):
        raise ValueError("binary reduction requires non-empty core and surround")
    if np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("kernel must be finite and non-negative")
    if not np.isclose(values.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("kernel must be L1 normalized")
    core_source = convolve_periodic(core.astype(float), values)
    matrix = np.asarray(
        [
            [float(np.mean(core_source[core])), float(np.mean(1.0 - core_source[core]))],
            [float(np.mean(core_source[~core])), float(np.mean(1.0 - core_source[~core]))],
        ]
    )
    return matrix


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
