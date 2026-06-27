"""M3A-v2 event readout + four-state phenotype classifier + proxy phase plane.

Operationalizes §B5.6/§B5.7. Given one event's spatial recruitment field A(x), the
per-E-cell onset times, and the slow-field traces, it produces the six metrics and the
deterministic four-state label:

    interictal_axial  |  expanded_axial  |  ictal_like_candidate  |  runaway  |  INSUFFICIENT

METHODOLOGICAL LOCK (§4 / §B5.6): the axis score MUST come from the SOURCE-SPACE per-cell
onset gradient (src.sef_hfo_snn_metrics.onset_axis) -- NOT contact-space direction, NOT
collision, NOT spike-cloud elongation. The wrong instrument reads a directed recruitment
wave as "synchronous burst / no axis".

The KEY scientific boundary the gates encode (not the thresholds -- the STRUCTURE):
expanded_axial != ictal_like_candidate. A large event is still expanded_axial unless axis
dominance DROPS *and* off-axis/low-k recruitment RISES. ictal_like is gated on axis-breaking,
never on size alone. Bad data (too few onsets / undefined axis) -> INSUFFICIENT, never ictal_like.

Canonical math: docs/snn_core_model_equations.md §B5.6-B5.7.
Plan / TDD:     docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md

!!! STATUS: STUB (red-TDD scaffold, 2026-06-28). Every logic body raises
NotImplementedError -- a classifier that silently returns a plausible label contaminates a
screen (CLAUDE.md §6). The PhenotypeGates defaults are placeholder calibration values; the
gate STRUCTURE in classify_event() is the locked contract. !!!
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sef_hfo_snn_metrics import onset_axis  # source-space instrument (methodological lock)


# ---------------------------------------------------------------------------
# Per-event source-space metrics (§B5.6).
# ---------------------------------------------------------------------------
def recruitment_area(A, theta_A):
    """R_area = mean_x 1[A(x) > theta_A]: fraction of the field above an activity
    threshold. A is the per-grid recruitment field (n_grid, n_grid). Task 6."""
    raise NotImplementedError


def axis_score(posE, onset, u_axis, min_n=20):
    """S_axis = |v_event . u_axis| / |v_event|, where v_event = onset_axis(posE, onset)
    is the SOURCE-SPACE onset gradient (methodological lock). u_axis is the unit E->E
    scaffold axis. Returns NaN when onset_axis is None (too few onsets / no gradient) --
    the caller maps NaN to INSUFFICIENT, never to a state. Task 6."""
    raise NotImplementedError


def offaxis_fraction(A, grid_xy, center, u_axis, corridor_halfwidth):
    """F_offaxis = sum_{x: |perp dist to axis| > corridor_halfwidth} A(x) / sum_x A(x).
    grid_xy is the (n_grid, n_grid, 2) lattice coords; center is the axis anchor (2,);
    perpendicular distance is |(x-center) . u_perp| with u_perp orthogonal to u_axis.
    Task 7."""
    raise NotImplementedError


def participation_ratio(A):
    """G_PR = (sum_x A)^2 / (N * sum_x A^2) in (0, 1]: globality / low-k proxy. ~1/N for a
    single hot cell, ~1 for uniform recruitment. Task 7."""
    raise NotImplementedError


def event_recovery(rate, dt, t_post0, baseline, sigma_base, m=1.5, t_return=120.0):
    """returned = mean E-rate over [t_post0, t_post0+t_return) <= baseline + m*sigma_base.
    A bool. runaway = not returned (tonic pinned / tail does not fall). Task 8."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Four-state classifier (§B5.6). Thresholds are calibration; the gate STRUCTURE
# (ictal_like REQUIRES axis-breaking AND off-axis rise; bad data -> INSUFFICIENT)
# is the locked science contract.
# ---------------------------------------------------------------------------
@dataclass
class PhenotypeGates:
    min_onsets: int = 20        # < this (or NaN S_axis) -> INSUFFICIENT
    area_small: float = 0.05    # R_area below -> "small" (interictal scale)
    area_large: float = 0.30    # R_area at/above -> "large"; NECESSARY for ictal-like (size gate)
    axis_high: float = 0.70     # S_axis at/above -> axis-dominant
    axis_broken: float = 0.40   # S_axis below -> axis dominance lost
    offaxis_high: float = 0.35  # F_offaxis at/above -> off-axis recruited
    gpr_high: float = 0.30      # G_PR at/above -> broad / low-k recruitment


def classify_event(metrics, gates: PhenotypeGates | None = None):
    """Deterministic four-state label from a metrics dict with keys:
        n_onsets:int, R_area:float, S_axis:float (NaN ok), F_offaxis:float,
        G_PR:float, recovery:bool.
    Returns one of: 'interictal_axial' | 'expanded_axial' | 'ictal_like_candidate'
    | 'runaway' | 'INSUFFICIENT'.

    Contract (gate STRUCTURE, §B5.6):
      * n_onsets < min_onsets  OR  isnan(S_axis)        -> 'INSUFFICIENT'  (first, fail-closed)
      * not recovery                                    -> 'runaway'
      * R_area >= area_large AND S_axis < axis_broken AND (F_offaxis >= offaxis_high
            or G_PR >= gpr_high)                         -> 'ictal_like_candidate'
            (LARGE + axis-breaking + off-axis/low-k rise; recovery already guaranteed by the
             runaway gate above. SIZE is a NECESSARY condition -- a small off-axis blip, even with
             a broken axis, is NEVER ictal-like.)
      * R_area < area_small AND S_axis >= axis_high     -> 'interictal_axial'
      * otherwise (large axis-dominant, or unclassified) -> 'expanded_axial'
    Task 9."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Proxy phase plane (§B5.7): region effective recruitment pressure from field traces.
# ---------------------------------------------------------------------------
def region_pressure(q_I_region, g_K_region, lgr, beta_K, eps=1e-9):
    """P_R = log(lgr) - <log(q_I + eps)>_R - beta_K * <g_K>_R, a scalar per region R.
    q_I_region / g_K_region are the field values over region R's cells. Task 10."""
    raise NotImplementedError


def proxy_phase_point(field, region_masks, lgr, beta_K):
    """Returns (X, Y): X = P_axis - P_offaxis (axis-dominance: >0 axis leads, drops/negative as
    off-axis catches up = axis-breaking); Y = P_global (whole-sheet pressure; matches the spectral
    Y=alpha_global for overlay; Y up & not returning = runaway risk). `field` exposes q_I, g_K
    lattices; region_masks selects axis / offaxis / global lattice cells. Task 10."""
    raise NotImplementedError
