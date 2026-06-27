"""M3 corridor region overlay: split E cells into corridor / hub / global along the
propagation axis, for the hub-gated critical scaffold (spec §5.1, plan Task 1).

Pure geometry — NO RNG, fully deterministic given positions + axis (审阅 §5: hub
selection must not be seed-dependent).

The substrate's EXCITABLE cores (low-threshold basins giving spontaneity + bidirection)
come from the runner's existing ``build_lesion_vth`` (``twoend_equal``). This module
only defines the REGION overlay used by:
  - the hub long-range broadcast edges (plan Task 2): which few E cells are hubs,
  - the degree-normalized threshold (plan Task 3): applied to all E cells,
  - the recruitment diagnostics (plan Task 6): corridor / hub / global onset timing.

Geometry (directed split along the axis):
  along = (posE - center) . axis_unit                  # signed mm, center at 0
  s     = corridor_half_frac * half                    # split point (positive along)
  corridor = {along <= s}   global = {along > s}        # full partition of E cells
  hub      = the hub_frac of corridor cells with the LARGEST along (nearest the +edge,
             the transition band adjacent to the global region).
For both excitable basins (at +/- sep_frac*half) to sit INSIDE the corridor, the caller
must use sep_frac < corridor_half_frac.
"""
from __future__ import annotations
import numpy as np


def corridor_regions(posE, center, axis_unit, half, corridor_half_frac=0.6, hub_frac=0.12,
                     global_gap_frac=0.0):
    """Partition E-cell positions into corridor / global, and pick the hub subset.

    corridor = {along <= corridor_half_frac*half}; global = {along > (corridor_half_frac +
    global_gap_frac)*half}. With global_gap_frac>0 a BUFFER band sits between them (in neither
    set) so the global region is spatially separated from the corridor — used to test whether the
    hub's long-range broadcast (not local edges) is what bridges corridor->global. global_gap_frac=0
    (default) -> adjacent, full partition.

    Returns dict(corridor_idx, global_idx, hub_idx, along) with E-local index arrays (int) and the
    signed along-axis coordinate (float, len NE)."""
    posE = np.asarray(posE, float)
    center = np.asarray(center, float)
    axis_unit = np.asarray(axis_unit, float)
    axis_unit = axis_unit / max(np.linalg.norm(axis_unit), 1e-12)
    along = (posE - center) @ axis_unit
    s = corridor_half_frac * half
    g = (corridor_half_frac + global_gap_frac) * half
    corridor_mask = along <= s
    corridor_idx = np.flatnonzero(corridor_mask)
    global_idx = np.flatnonzero(along > g)
    if corridor_idx.size:
        n_hub = max(1, int(np.ceil(hub_frac * corridor_idx.size)))
        order = corridor_idx[np.argsort(along[corridor_idx])]  # ascending along
        hub_idx = np.sort(order[-n_hub:])                      # nearest the +edge
    else:
        hub_idx = np.array([], int)
    return dict(corridor_idx=corridor_idx, global_idx=global_idx,
                hub_idx=hub_idx, along=along)


def hub_mask_E(NE, hub_idx):
    """Length-NE boolean mask marking the hub E cells."""
    m = np.zeros(int(NE), bool)
    hub_idx = np.asarray(hub_idx, int)
    if hub_idx.size:
        m[hub_idx] = True
    return m
