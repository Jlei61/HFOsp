"""M4 Pass-1 phase-plane PURE decision functions (spec 2026-07-05 rev4 §8.1, §9.1).

This module holds ONLY pure functions: given per-cell metrics (already computed from a simulation
trajectory upstream) + calibrated guard thresholds, decide the §9.1 go/no-go. It runs NO simulation.
The simulation harness that produces the metrics (and calibrates the thresholds against arm-0 TRIVIAL
instances) is a separate, gated step (scripts/run_m4_phaseplane.py, not yet written).

§9.1 recap (rev4):
  TRIVIAL-A ("low-amplitude global skirt"): act_frac high AND core_overlap>theta_core AND globality<theta_glob
  TRIVIAL-B ("expanded axial"):             f_off<theta_off AND self_limited (retreats to core)
  go(cell) = persist AND bounded AND act_frac>=act_min AND s_grad>=sgrad_min AND not TRIVIAL-A AND not TRIVIAL-B

NOTE (rev4-consistency, flagged 2026-07-05): go does NOT hard-require "breaks axial" (high f_off). rev4's
PRIMARY target is a spatially-localized bounded CORE (region 3), which is not off-axis spread; f_off enters
only TRIVIAL-B (axis-confined AND self-limited). A high-amplitude distributed synchronized burst also passes
go as a candidate (secondary per rev4 §8.5); Pass-2 adjudicates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Phase-plane axis + core mask (rev4 §8.1)
# ---------------------------------------------------------------------------
def q_core(q_I, m_core):
    """Core-weighted mean of the q_I field (rev4 §8.1 primary phase-plane axis):
    q_core = sum(m_core * q_I) / sum(m_core). `m_core` is a nonnegative weight field (the derived core
    mask); q_I and m_core must be the same shape. Raises on an empty (all-zero) mask."""
    q = np.asarray(q_I, dtype=float)
    m = np.asarray(m_core, dtype=float)
    if q.shape != m.shape:
        raise ValueError(f"q_I {q.shape} and m_core {m.shape} must match")
    s = float(m.sum())
    if s <= 0.0:
        raise ValueError("empty core mask (sum(m_core) <= 0)")
    return float((m * q).sum() / s)


def derive_core_mask(first_activation, frac=0.1):
    """Derive the core mask from a per-cell first-activation-time map (rev4 §8.1: m_core from the
    kick-triggered first-activation map). Returns a float mask (1.0 for the earliest-activating `frac`
    of cells that ACTUALLY activated, i.e. have a finite activation time; 0.0 elsewhere).
    Non-activating cells (inf/nan) are never core. Raises if nothing activated or frac not in (0,1]."""
    fa = np.asarray(first_activation, dtype=float)
    if not (0.0 < frac <= 1.0):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    finite = np.isfinite(fa)
    if not finite.any():
        raise ValueError("no cell activated (first_activation has no finite entries)")
    thr = float(np.quantile(fa[finite], frac))
    return (finite & (fa <= thr)).astype(float)


# ---------------------------------------------------------------------------
# §9.1 per-cell go/no-go (pure)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GuardThresholds:
    """§9.1 guard thresholds. CALIBRATED against arm-0 TRIVIAL-A/TRIVIAL-B reference instances (spec §9.1
    step 1-3); these are NOT free knobs. Defaults here are placeholders for unit tests only — the real
    values are pinned by the (gated) calibration before the phase-plane sweep."""
    theta_core: float = 0.7    # TRIVIAL-A: core_overlap above this = power stays core-weighted
    theta_glob: float = 0.3    # TRIVIAL-A: globality below this = low-amplitude skirt
    theta_off: float = 0.2     # TRIVIAL-B: f_off below this = axis-confined
    act_min: float = 0.1       # go: recruited-area fraction floor (large enough, not a blip)
    act_high: float = 0.6      # TRIVIAL-A: act_frac above this = whole-field
    sgrad_min: float = 0.1     # go: source-onset gradient present (a spatial ignition sequence exists)
    b_delta_eps: float = 0.05  # bounded: window/cycle-avg branching <= 1 + eps


@dataclass(frozen=True)
class CellMetrics:
    """One phase-plane cell's readouts (already computed from its simulation, upstream)."""
    persist: bool          # activity self-sustained > T_min after the kick ends
    act_frac: float        # recruited-area fraction
    s_grad: float          # source-space onset gradient score (spatial ignition sequence present?)
    f_off: float           # off-axis recruitment fraction
    core_overlap: float    # fraction of event power on the ignition core
    globality: float       # distributed-amplitude score
    self_limited: bool     # activity retreated to core / decayed after peak (vs sustained spread)
    # bounded sub-inputs (rev4 §9.1 relaxation):
    b_delta_avg: float     # window/cycle-averaged branching ratio
    monotonic_saturation: bool  # global rate saturated monotonically (runaway signature)
    tail_returns: bool     # tail returned toward baseline
    finite_energy: bool    # event energy finite (not unbounded)


def is_bounded(m: CellMetrics, th: GuardThresholds) -> bool:
    """rev4 §9.1 bounded = window-avg branching <= 1+eps AND no monotonic saturation AND tail returns
    AND finite energy (an instantaneous B>1 during a burst rising phase is NOT runaway)."""
    return ((m.b_delta_avg <= 1.0 + th.b_delta_eps)
            and (not m.monotonic_saturation)
            and m.tail_returns
            and m.finite_energy)


def is_trivial_A(m: CellMetrics, th: GuardThresholds) -> bool:
    """Low-amplitude global skirt: whole-field extent but power stays core-weighted with low distributed
    amplitude (the substrate's known coherent whole-field recruitment made rhythmic)."""
    return (m.act_frac >= th.act_high) and (m.core_overlap > th.theta_core) and (m.globality < th.theta_glob)


def is_trivial_B(m: CellMetrics, th: GuardThresholds) -> bool:
    """Expanded axial: onset axis-confined (low off-axis) AND self-limits (retreats to core) — a bigger
    interictal axial event, not a seizure. A SUSTAINED axis-elongated bounded core is NOT trivial-B."""
    return (m.f_off < th.theta_off) and m.self_limited


@dataclass(frozen=True)
class CellVerdict:
    go: bool
    trivial_A: bool
    trivial_B: bool
    bounded: bool
    label: str             # 'go' | 'trivial_A' | 'trivial_B' | 'decay' | 'runaway' | 'blip'


def classify_cell(m: CellMetrics, th: GuardThresholds) -> CellVerdict:
    """§9.1 per-cell go/no-go. go(cell) = persist AND bounded AND act_frac>=act_min AND s_grad>=sgrad_min
    AND not TRIVIAL-A AND not TRIVIAL-B. Returns the boolean breakdown + a diagnostic label."""
    bounded = is_bounded(m, th)
    ta = is_trivial_A(m, th)
    tb = is_trivial_B(m, th)
    go = (m.persist and bounded and (m.act_frac >= th.act_min) and (m.s_grad >= th.sgrad_min)
          and (not ta) and (not tb))
    if go:
        label = "go"
    elif not m.persist:
        label = "decay"
    elif not bounded:
        label = "runaway"
    elif ta:
        label = "trivial_A"
    elif tb:
        label = "trivial_B"
    elif m.act_frac < th.act_min:
        label = "blip"
    else:
        label = "other_nogo"
    return CellVerdict(go=go, trivial_A=ta, trivial_B=tb, bounded=bounded, label=label)


# ---------------------------------------------------------------------------
# §9.1 plane-level verdict: area (contiguity) + arm-2-not-arm-1 (pure)
# ---------------------------------------------------------------------------
def largest_contiguous(go_grid) -> int:
    """Size of the largest 4-connected component of True cells in a 2-D boolean grid (rev4 §9.1
    'an AREA, not a single point'). Pure flood-fill; no scipy dependency."""
    g = np.asarray(go_grid, dtype=bool)
    if g.ndim != 2:
        raise ValueError(f"go_grid must be 2-D, got {g.ndim}-D")
    seen = np.zeros_like(g, dtype=bool)
    best = 0
    ny, nx = g.shape
    for sy in range(ny):
        for sx in range(nx):
            if not g[sy, sx] or seen[sy, sx]:
                continue
            stack = [(sy, sx)]
            seen[sy, sx] = True
            size = 0
            while stack:
                y, x = stack.pop()
                size += 1
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny_, nx_ = y + dy, x + dx
                    if 0 <= ny_ < ny and 0 <= nx_ < nx and g[ny_, nx_] and not seen[ny_, nx_]:
                        seen[ny_, nx_] = True
                        stack.append((ny_, nx_))
            best = max(best, size)
    return best


def go_plane_verdict(go_arm2, go_arm1, k_min):
    """§9.1 go(plane): >= k_min CONTIGUOUS go(cell) present in arm 2 (divisive) but NOT arm 1 (subtractive
    only). Returns a dict {verdict: 'go'|'no-go', arm2_max_contiguous, arm1_max_contiguous, reason}.
    A clean 'no-go' is a valid scientific result (spec §9.1)."""
    a2 = largest_contiguous(go_arm2)
    a1 = largest_contiguous(go_arm1)
    if a2 < k_min:
        return {"verdict": "no-go", "arm2_max_contiguous": a2, "arm1_max_contiguous": a1,
                "reason": f"arm2 largest contiguous go-region {a2} < k_min {k_min} (no area / single point)"}
    if a1 >= k_min:
        return {"verdict": "no-go", "arm2_max_contiguous": a2, "arm1_max_contiguous": a1,
                "reason": f"arm1 (subtractive-only) also opens a >= k_min go-region ({a1}) -> divisive not "
                          f"necessary (only-suppresses explanation not excluded)"}
    return {"verdict": "go", "arm2_max_contiguous": a2, "arm1_max_contiguous": a1,
            "reason": f"arm2 has a contiguous go-region ({a2} >= {k_min}) absent in arm1 ({a1})"}
