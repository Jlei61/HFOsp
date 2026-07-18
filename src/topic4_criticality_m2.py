"""Topic 4 M3-v2.2 criticality Milestone 2 — two-stage ignition/spread readout.

Productionizes the M2 de-risk pilots (results/topic4_criticality_m2/pilots/*.py).
Spec: docs/superpowers/specs/2026-07-04-topic4-m3v2-2-m2-critical-mode-decomposition-design.md (rev2.1).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import yaml
import src.topic4_m3b_spectral_phase as spm
from src.topic4_criticality import _fields_from_slow, check_low_branch_continuation_between, _crit_op_context

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _REPO / "config/topic4_criticality_m2.yaml"
_M1_VERDICT_PATH = _REPO / "results/topic4_criticality/trajectory_verdict.json"


def load_m2_config(path=None) -> dict:
    cfg = yaml.safe_load(Path(path or _DEFAULT_CFG).read_text())
    if cfg["basis"].get("theta") == "THETA_EE":
        cfg["basis"]["theta"] = float(spm.THETA_EE)
    return cfg


def basis_vectors(grid, theta) -> dict:
    X, Y = grid.coords()
    e_global = np.ones(X.size); e_global /= np.linalg.norm(e_global)
    s = (X * np.cos(theta) + Y * np.sin(theta)).ravel(); s = s - s.mean()
    e_axis = s - (s @ e_global) * e_global
    e_axis /= (np.linalg.norm(e_axis) + 1e-300)
    return {"e_global": e_global, "e_axis_gradient": e_axis}


def nonaxis_direction(loading, grid, theta, min_norm):
    v = np.asarray(loading, float).ravel(); nv = float(np.linalg.norm(v))
    b = basis_vectors(grid, theta)
    proj_g = (v @ b["e_global"]) * b["e_global"]
    proj_a = (v @ b["e_axis_gradient"]) * b["e_axis_gradient"]
    residual = v - proj_g - proj_a
    rn = float(np.linalg.norm(residual))
    frac = lambda x: float(np.linalg.norm(x) / (nv + 1e-300))
    e_nonaxis = residual / rn if rn >= min_norm else None
    return e_nonaxis, frac(residual), frac(proj_g), frac(proj_a)


def shape_scores_at(res, grid, kernels, core) -> dict:
    idxs = spm.leading_subspace_indices(res.eigenvalues, min_sep=1e-3, imag_tol=1e-3)
    loading = spm.pair_loading(res.right, idxs, grid)
    th = kernels.theta; lead = res.eigenvalues[0]
    return {
        "axis_elongation": float(spm.elongation_axis_score(loading, grid, th)),
        "axis_wavevector_alignment": float(spm.phase_gradient_axis_score(loading, grid, th)),
        "off_axis": float(spm.off_axis_score(loading, grid, th)),
        "globality": float(spm.globality(loading, grid)),
        "core_overlap": float(spm.core_overlap(loading, grid, core)),
        "leading_subspace_dim": int(len(idxs)),
        "leading_is_complex_pair": bool(len(idxs) == 2 and abs(lead.imag) > 1e-3),
        "leading_eigenvalue_real": float(lead.real),
        "leading_eigenvalue_imag": float(lead.imag),
        "_loading": loading,
    }


# --------------------------------------------------------------------------- #
# Task 1: dense alpha0-crossing localization (bracket -> coarse scan -> bisect) #
# --------------------------------------------------------------------------- #
# Productionizes results/topic4_criticality_m2/pilots/m2_pilots.py's PILOT 2 (a)+(b). The
# crossing is localized on the REAL interpolated slow-space between the M1 verdict's last
# qualified low-branch point and the first saturated point that follows it -- never the 2-D
# atlas (spec rev2.1 constraint). `low_solve_fast` deliberately uses a SINGLE warm-started
# solve_operating_point, not the full 4-init solve_branches protocol: the coarse scan +
# bisection re-solve the low branch many times near a fold, and the full branch protocol is
# overkill for tracking ONE already-identified low branch densely (see m2_pilots.py's module
# docstring for the same scout-speed rationale; ~4x fewer op-solves, same low-branch physics).


def interp_slow(a, b, frac):
    """Linear interpolation of a slow_inputs dict (exactly M1's
    check_low_branch_continuation_between; ported from m2_pilots.py's `_interp_slow`)."""
    return {
        "q_global": (1 - frac) * a["q_global"] + frac * b["q_global"],
        "q_core": (1 - frac) * a["q_core"] + frac * b["q_core"],
        "hG_scalar": (1 - frac) * (a.get("hG_scalar") or 0.0) + frac * (b.get("hG_scalar") or 0.0),
        "gK_value": (None if (a.get("gK_value") is None or b.get("gK_value") is None)
                     else (1 - frac) * a["gK_value"] + frac * b["gK_value"]),
    }


def low_solve_fast(grid, kernels, core, slow_inputs, cfg, prev_op):
    """Single warm-started solve_operating_point on the low branch (ported from m2_pilots.py's
    `_low_solve_fast`)."""
    exc, inh, gK_field, hG_scalar, eta_K, eta_G = _fields_from_slow(grid, core, slow_inputs, cfg)
    init = None if prev_op is None else {"rE": prev_op.rE, "rI": prev_op.rI}
    op = spm.solve_operating_point(grid, kernels, exc, inh, gK_field=gK_field, hG_scalar=hG_scalar,
                                   eta_K=eta_K, eta_G=eta_G, init=init)
    sat = bool(op.saturated)
    return op, sat


def alpha1_and_eig(grid, kernels, op):
    """(alpha1, EigResult, J) at an OperatingPoint; alpha1=nan on unresolved/empty spectrum
    (ported verbatim from m2_pilots.py's `_alpha1_and_eig`)."""
    J = spm.build_jacobian_dense(grid, kernels, op)
    res = spm.rate_eigenpairs(J, grid)
    if res.status != "resolved" or res.eigenvalues.size == 0:
        return float("nan"), res, J
    return float(res.eigenvalues[0].real), res, J


def _op_solve_quality(op, res, op_residual_tol):
    """op_solve_quality (spec rev2.2 §3.1/§8): the op-solve is usable for a spectral read-out iff
    its fixed-point residual is within a FOLD-APPROPRIATE tolerance (``op_residual_tol``, NOT the
    solver's strict 1e-9 ``converged`` bar) AND it is not saturated AND its spectrum resolved.

    Rev2.1 keyed this off ``not op.saturated`` only, which reported quality=True for near-fold ops
    whose residual (~1e-3-4e-3) never reaches the strict 1e-9 converged bar. Rev2.2 (user decision):
    near-fold ops legitimately never fully converge, but their alpha1/mode reads ARE stable, so the
    strict ``converged`` flag would wrongly fail the §5.0 ignition gate; a residual tolerance is the
    right bar. ``op.residual`` is the OperatingPoint's stored fixed-point residual (max rate-of-change
    magnitude, src.topic4_m3b_spectral_phase.solve_operating_point), so no field_rhs recompute is
    needed. Distinct from stability_read_quality (alpha1<0) -- see localize_alpha0_crossing's
    alpha_left/alpha_right (always <0/>=0 respectively when both are numeric)."""
    return bool(op is not None and float(op.residual) <= float(op_residual_tol)
                and not op.saturated and res is not None
                and res.status == "resolved" and res.eigenvalues.size > 0)


def localize_alpha0_crossing(points, grid, kernels, core, cfg_crit, m2cfg):
    """Dense alpha0-crossing localizer (spec rev2.2 M2 Task 1).

    Brackets the last-qualified low-branch M1 point -> the first saturated M1 point that
    follows it, coarse-scans ``densification.coarse_K`` fracs of the interpolated slow-space
    between them (``interp_slow``), then recursively bisects the first neg->(>=0 or gone)
    sub-bracket to ``crossing_width_ms_tol`` (or ``max_bisect_hard_cap`` levels). Multi-crossing
    coarse scans are flagged (``crossing_status="multiple_alpha0_crossings"``) but only the
    first crossing is localized.

    ``op_solve_quality_left/right`` use a fold-appropriate residual bar (``op_residual_tol``, spec
    rev2.2), NOT the solver's strict 1e-9 converged flag -- near-fold ops never fully converge but
    read a stable spectrum. ``branch_identity_clean`` (spec §3.1, feeds the §5.0 ignition base gate)
    reuses M1's ``check_low_branch_continuation_between`` across the crossing bracket. Returns
    ``_crossing_op``/``_crossing_res`` for later tasks (shape scoring at the crossing).
    """
    dcfg = m2cfg["densification"]
    op_residual_tol = dcfg["op_residual_tol"]
    q = [p for p in points if p.get("qualified") and p.get("branch_id") == "low_branch"]
    last_q = q[-1]
    trans = next(p for p in points if p.get("saturated") and p["time_ms"] > last_q["time_ms"])
    a, b = last_q["slow_inputs"], trans["slow_inputs"]
    span_ms = trans["time_ms"] - last_q["time_ms"]

    # branch_identity_clean (spec §3.1; feeds the §5.0 ignition base gate): reuse M1's
    # check_low_branch_continuation_between across the SAME last-qualified-low -> first-saturated
    # bracket the crossing is localized in. The low branch is a clean single continuous low-approach
    # (no branch jump) iff that check reports it stayed continuous and smoothly REACHED alpha0
    # ("low_branch_reaches_alpha0_before_jump"). The other two M1 statuses both involve a jump that
    # replaces a smooth crossing -- "..._disappears_before_alpha0" is a mid-bracket fold, and
    # "..._remains_far_from_alpha0_until_jump" literally jumps (its name ends _until_jump) without a
    # low-branch approach -- so neither is branch-identity-clean for the purpose of trusting the
    # localized crossing. Mirrors M1's own `checked is True and status == ...` defensive idiom;
    # bounded (reuses M1's helper, does NOT rebuild solve_branches). The raw status is surfaced as
    # `_branch_continuation_status` (private-key convention, like `_crossing_op`) for auditability.
    cont = check_low_branch_continuation_between(last_q, trans, cfg_crit)
    branch_identity_clean = bool(
        cont.get("branch_continuation_checked") is True
        and cont.get("continuation_status") == "low_branch_reaches_alpha0_before_jump")
    _branch_continuation_status = cont.get("continuation_status")

    # (a) coarse scan (warm-start chain along the low branch)
    fracs = list(np.linspace(0.0, 1.0, dcfg["coarse_K"]))
    prev, scan = None, []
    for fr in fracs:
        op, sat = low_solve_fast(grid, kernels, core, interp_slow(a, b, fr), cfg_crit, prev)
        if not sat and op is not None: prev = op
        a1 = (float("nan") if sat else alpha1_and_eig(grid, kernels, op)[0])
        scan.append({"frac": float(fr), "alpha1": (None if (sat or not np.isfinite(a1)) else float(a1)), "sat": sat})
    defined = [(s["frac"], s["alpha1"]) for s in scan if s["alpha1"] is not None]
    sign_changes = sum(1 for (_, x), (_, y) in zip(defined, defined[1:]) if (x < 0) != (y < 0))
    status = "multiple_alpha0_crossings" if sign_changes > 1 else ("single" if defined else "none")

    # first neg -> (>=0 or gone) sub-bracket (m2_pilots.py:pilot2 lines 201-208)
    lo_fr, hi_fr = None, None
    for i in range(len(scan) - 1):
        cur, nxt = scan[i], scan[i + 1]
        cur_neg = (cur["alpha1"] is not None and cur["alpha1"] < 0)
        nxt_cross = (nxt["alpha1"] is None) or (nxt["alpha1"] is not None and nxt["alpha1"] >= 0)
        if cur_neg and nxt_cross:
            lo_fr, hi_fr = cur["frac"], nxt["frac"]
            break

    if lo_fr is None:
        return {"alpha0_crossing_time_ms": None, "alpha0_crossing_slow_state": None,
                "crossing_frac": None, "crossing_width_ms": None,
                "alpha_left": None, "alpha_right": None, "crossing_status": status,
                "op_solve_quality_left": False, "op_solve_quality_right": False,
                "branch_identity_clean": branch_identity_clean,
                "_branch_continuation_status": _branch_continuation_status,
                "_crossing_op": None, "_crossing_res": None}

    def alpha_at(frac, prev_op):
        s = interp_slow(a, b, frac)
        op, sat = low_solve_fast(grid, kernels, core, s, cfg_crit, prev_op)
        if sat:
            return None, None, None, True, s        # low branch gone (saturated) at this frac
        a1, res, _J = alpha1_and_eig(grid, kernels, op)
        return a1, op, res, sat, s

    # (b) bisect the sub-bracket (m2_pilots.py:pilot2 lines 210-233, verbatim except: (1)
    # max_bisect_hard_cap replaces the hard-coded 8-level cap; (2) a width-in-ms stop
    # ((hi_fr-lo_fr)*span_ms < crossing_width_ms_tol) replaces the frac/alpha tolerance).
    # warm-start the bisection from lo_fr's low op
    alpha_left, op_left, res_left, _sa, s_lo = alpha_at(lo_fr, None)
    prevb = op_left
    best = {"frac": lo_fr, "alpha1": alpha_left, "op": op_left, "res": res_left, "slow": s_lo}
    # Initial hi_fr reading (NOT in pilot2, which never re-derives it -- added here because
    # alpha_right/op_solve_quality_right are new output fields the pilot never had: without this,
    # a non-monotone dip near the crossing can keep every bisected midpoint on the negative side,
    # leaving alpha_right stuck at a placeholder None even though the coarse scan already resolved
    # a genuine >=0 reading at hi_fr). Refined below whenever the loop actually revisits the hi side.
    alpha_right, op_right, res_right, _sa2, _s_hi = alpha_at(hi_fr, op_left)
    if alpha_right is not None and not np.isfinite(alpha_right):
        # non-finite alpha1 from a non-saturated-but-unresolved near-fold op is not a usable
        # hi-side reading -- treat it as "low gone" (same as the loop's NaN guard below).
        alpha_right, op_right, res_right = None, None, None
    for _lvl in range(dcfg["max_bisect_hard_cap"]):
        mid = 0.5 * (lo_fr + hi_fr)
        a1m, opm, resm, sat_any, sm = alpha_at(mid, prevb)
        if opm is None or a1m is None or not np.isfinite(a1m):
            # low branch gone (saturated -> opm/a1m None) OR a non-finite alpha1 from a
            # non-converged/unresolved near-fold op -> tighten hi_fr. FIX 3: a NaN a1m must NOT
            # fall through to the `else` and be mis-read as a crossed (>=0) reading (nan<0 is
            # False in Python), which would narrow hi_fr off a meaningless value and could set the
            # returned crossing alpha1 to NaN. Matches the coarse-scan's np.isfinite guard.
            hi_fr = mid
            alpha_right, op_right, res_right = None, None, None
            continue
        prevb = opm
        if a1m < 0:
            lo_fr = mid
            best = {"frac": mid, "alpha1": a1m, "op": opm, "res": resm, "slow": sm}
            alpha_left, op_left, res_left = a1m, opm, resm
        else:
            hi_fr = mid
            best = {"frac": mid, "alpha1": a1m, "op": opm, "res": resm, "slow": sm}
            alpha_right, op_right, res_right = a1m, opm, resm
        if (hi_fr - lo_fr) * span_ms < dcfg["crossing_width_ms_tol"]:
            best = {"frac": mid, "alpha1": a1m, "op": opm, "res": resm, "slow": sm}
            break

    # width in ms over the idx14->idx15 time bracket:
    crossing_width_ms = (hi_fr - lo_fr) * span_ms
    t_cross = (1 - best["frac"]) * last_q["time_ms"] + best["frac"] * trans["time_ms"]
    return {"alpha0_crossing_time_ms": float(t_cross), "alpha0_crossing_slow_state": best["slow"],
            "crossing_frac": float(best["frac"]), "crossing_width_ms": float(crossing_width_ms),
            "alpha_left": alpha_left, "alpha_right": alpha_right, "crossing_status": status,
            "op_solve_quality_left": _op_solve_quality(op_left, res_left, op_residual_tol),
            "op_solve_quality_right": _op_solve_quality(op_right, res_right, op_residual_tol),
            "branch_identity_clean": branch_identity_clean,
            "_branch_continuation_status": _branch_continuation_status,
            "_crossing_op": best["op"], "_crossing_res": best["res"]}


# --------------------------------------------------------------------------- #
# Task 2: linear_ignition readout + two-core symmetry-break confirmation        #
# --------------------------------------------------------------------------- #
# Productionizes results/topic4_criticality_m2/pilots/m2_pilots_round2.py's PILOT A (two-core
# crossing-mode decomposition). `_region_masks`/`_region_breakdown` are ported (the pilots dir is
# gitignored scratch, spec §8 -- not importable) from pilotA's own helpers of the same name
# (m2_pilots_round2.py lines 66-89), unchanged except that `h`/`radius` come from the caller
# (cfg-driven here) instead of pilotA's module-level TWO_CORE_SEP/TWO_CORE_RADIUS constants.


def _region_masks(grid, core2, theta, *, h, radius):
    """core-A (axis-positive), core-B (axis-negative), on-axis corridor-between, off-core-rest
    (ported verbatim from m2_pilots_round2.py's `_region_masks`, lines 66-75)."""
    X, Y = grid.coords()
    s = X * np.cos(theta) + Y * np.sin(theta)              # along-axis coord
    perp = -X * np.sin(theta) + Y * np.cos(theta)          # perpendicular coord
    maskA = core2.mask & (s > 0)
    maskB = core2.mask & (s < 0)
    corridor = (~core2.mask) & (np.abs(s) <= h) & (np.abs(perp) <= radius)   # axial strip between cores
    rest = ~(maskA | maskB | corridor)
    return {"coreA": maskA, "coreB": maskB, "corridor_axial": corridor, "offcore_rest": rest}, s


def _region_breakdown(loading, grid, core2, theta, *, h, radius):
    """Fraction of E-power (loading**2) in each region + along-axis power profile (ported
    verbatim from m2_pilots_round2.py's `_region_breakdown`, lines 78-89)."""
    p = np.abs(loading) ** 2
    tot = float(p.sum()) + 1e-300
    regions, s = _region_masks(grid, core2, theta, h=h, radius=radius)
    frac = {k: float(p[m].sum() / tot) for k, m in regions.items()}
    # along-axis power profile (coarse bins over s)
    order = np.argsort(s.ravel())
    s_sorted = s.ravel()[order]
    p_sorted = p.ravel()[order]
    prof = [{"s": float(sv), "power_frac": float(pv / tot)} for sv, pv in zip(s_sorted, p_sorted) if pv / tot > 1e-4]
    return frac, prof


def _classify_ignition(*, core_overlap, globality, axis_elongation, off_axis,
                       corridor_power, n_core_peaks, m2cfg):
    """Ignition class classifier (spec §5.1; brief Step 3, verbatim). `core_overlap`+`globality`
    are the ONLY gate for `core_localized` -- axis_elongation/off_axis/corridor_power/n_core_peaks
    only disambiguate `delocalized_subtype` once the mode has already failed that gate."""
    ig = m2cfg["ignition"]
    if core_overlap >= ig["core_localized_overlap_thresh"] and globality <= ig["core_localized_globality_thresh"]:
        return "core_localized", None
    if globality >= ig["delocalized_globality_thresh"]:
        if corridor_power >= ig["corridor_lit_thresh"]:
            return "delocalized", "corridor_lit"
        if abs(axis_elongation) < ig["iso_thresh"] and off_axis < ig["iso_thresh"]:
            return "delocalized", "global_like"
        if n_core_peaks >= 2:
            return "delocalized", "multi_core"
        return "delocalized", "global_like"
    return "ambiguous", None


def _ignition_sensitivity(core_overlap, globality, m2cfg):
    """Threshold-sweep stability check for the `core_localized` gate (spec §5.1
    `ignition_sensitivity`; brief Step 3, verbatim)."""
    ig = m2cfg["ignition"]; flips = []
    for ot in ig["overlap_sweep"]:
        for gt in ig["globality_sweep"]:
            flips.append(core_overlap >= ot and globality <= gt)
    return "core_localized but threshold-sensitive" if not all(flips) and any(flips) else "stable"


def read_linear_ignition(crossing, grid, kernels, core, cfg_crit, m2cfg, points) -> dict:
    """linear_ignition readout at the alpha0 crossing (spec §3.2/§1) + two-core symmetry-break
    confirmation (spec §3.2 "two-core 确认"; §0-f near-fold caveat + §0-g symmetric-disinhibition
    approximation caveat).

    ``crossing`` is T1's ``localize_alpha0_crossing`` output for the SINGLE-core ``core`` -- its
    ``_crossing_res`` is scored here to derive the primary ignition class. The two-core
    confirmation is a CLEAN REUSE of T1's own ``localize_alpha0_crossing``, called again with a
    two-core mask standing in for ``core``: `low_solve_fast`/`_fields_from_slow` thread whatever
    ``core`` they are given into `build_excitability_field`/`build_inhibition_field`, so swapping
    in a two-core mask re-solves the low branch under a two-core geometry where BOTH cores are
    disinhibited by the SAME scalar `q_core` (spec §0-g "symmetric-disinhibition approximation")
    and finds THAT geometry's own alpha0 crossing along the same M1 last-qualified->first-saturated
    bracket (expected earlier than the single-core crossing, ~frac 0.53 per pilotA -- the two-core
    system saturates sooner). This re-runs M1's `check_low_branch_continuation_between` a second
    time (~another minute); acceptable for a one-shot per-subject readout (task brief).
    """
    scores = shape_scores_at(crossing["_crossing_res"], grid, kernels, core)

    # Primary (single-core) ignition class: on a single-core geometry there is no "corridor"
    # (nothing sits "between" one core) and exactly one physical core region exists, so
    # corridor_power=0.0 / n_core_peaks=1 for THIS call. Neither value can change the outcome
    # here: on the real crossing core_overlap/globality already satisfy `core_localized`, which
    # short-circuits _classify_ignition before either argument is read.
    cls, subtype = _classify_ignition(
        core_overlap=scores["core_overlap"], globality=scores["globality"],
        axis_elongation=scores["axis_elongation"], off_axis=scores["off_axis"],
        corridor_power=0.0, n_core_peaks=1, m2cfg=m2cfg)

    tcc = m2cfg["two_core_confirm"]
    theta = kernels.theta
    core2 = spm.make_core_mask(grid, kind="two", radius=tcc["radius"], separation=tcc["separation"])
    two_crossing = localize_alpha0_crossing(points, grid, kernels, core2, cfg_crit, m2cfg)
    two_scores = shape_scores_at(two_crossing["_crossing_res"], grid, kernels, core2)
    region_frac, axis_profile = _region_breakdown(
        two_scores["_loading"], grid, core2, theta, h=0.5 * tcc["separation"], radius=tcc["radius"])

    corridor_axial = region_frac["corridor_axial"]
    max_single_core = max(region_frac["coreA"], region_frac["coreB"])
    two_core_symmetry_break = bool(max_single_core >= tcc["single_core_thresh"]
                                    and corridor_axial <= tcc["corridor_dark_thresh"])

    near_fold_note = (
        "two-core crossing alpha1 is a post-fold first-positive value, not a precise "
        "alpha0~0 critical shape; two cores share one q_core (symmetric-disinhibition "
        "approximation) -> proves corridor stays dark given an axial two-core opportunity, "
        "not subject1146 dual-source reproduction.")

    sensitivity = _ignition_sensitivity(scores["core_overlap"], scores["globality"], m2cfg)

    return {
        "class": cls,
        "delocalized_subtype": subtype,
        "core_overlap": scores["core_overlap"],
        "globality": scores["globality"],
        "two_core_symmetry_break": two_core_symmetry_break,
        "corridor_power": corridor_axial,
        "shape_descriptors": {
            "axis_elongation": scores["axis_elongation"],
            "off_axis": scores["off_axis"],
            "axis_wavevector_alignment": scores["axis_wavevector_alignment"],
        },
        "near_fold_note": near_fold_note,
        "ignition_sensitivity": sensitivity,
        "_two_core_region_frac": region_frac,
        "_two_core_axis_profile": axis_profile,
        "_two_core_crossing": two_crossing,
    }


# --------------------------------------------------------------------------- #
# Task 3: projected operator gain/leak + nonaxis off_axis sentinel             #
# --------------------------------------------------------------------------- #
# Productionizes spec §2.3/§3.3 (rev2.3). `embed_rE` places a grid-space E-rate direction into the
# rE block of the full 6-field state (rE,rI,sEE,sEI,sIE,sII per spm.STATE_FIELDS), the other 5
# blocks zero. `projected_gains` reads the FULL-STATE self-gain ‖exp(JT)·embed_rE(e)‖/‖e‖ via M1's
# `spm.transient_gain` -- NOT the spec §3.3 literal ‖P_Y·rE(...)‖ E-rate-block projection, which is
# DEFERRED (spec rev2.3 §3.3 sign-off: full-state self-gain is the intended reading, consistent with
# M1's own `finite_time_gain`/`core_perturbation_vector` precedent; the sentinel is a negative-
# control whose PRIMARY is the score gate, so the gain gate is only a secondary relative comparison).
# `e_nonaxis` is downgraded to a SENTINEL/negative-control (spec §0-c/§2.3): `off_axis` only ever
# reaches "present" when BOTH the shape-score gate and the gain-excess/ratio gate break; below that,
# no propagation conclusion may be drawn (spec §2.3/§7 "禁写任何侧向/离轴传播结论"). The sentinel
# reads the ASYMPTOTIC-TAIL horizons (T >= gain.sentinel_min_horizon_ms) and requires tail agreement
# (rev2.3 §2.3): short/mid horizons carry a compact-e_nonaxis core-compactness transient that made a
# single-horizon read non-monotone/fragile; see `_off_axis_tail_agreement`.

_NONAXIS_ANNOTATION = ("nonaxis_residual = core-compactness residual in a core-localized mode, "
                       "NOT sideways propagation")


def embed_rE(e, grid, kernels):
    """Embed a grid-space (N,) E-rate direction into the full 6N state (rE block=e, else 0)."""
    z = np.zeros(6 * grid.n * grid.n)     # 6 fields rE,rI,sEE,sEI,sIE,sII
    z[: grid.n * grid.n] = e
    return z


def projected_gains(J, grid, kernels, dirs, horizons):
    """{direction_name: {horizon_ms: gain}}: full-state self-gain ‖exp(J*T)·embed_rE(e)‖/‖e‖ per
    named direction, one curve over `horizons` (M1 `spm.transient_gain` precedent; spec rev2.3 §3.3
    sign-off -- full-state self-gain, NOT the deferred ‖P_Y·rE(...)‖ E-rate-block projection)."""
    out = {}
    for name, e in dirs.items():
        b = embed_rE(e, grid, kernels)
        out[name] = {int(T): float(spm.transient_gain(J, b, float(T))) for T in horizons}
    return out


def _off_axis_decision(*, off_axis_score, gain_nonaxis, gain_axis, gain_global, m2cfg):
    """Both-gates off_axis three-state decision (spec §2.3, verbatim thresholds). `present` only
    if BOTH the shape-score gate (`off_axis_score_tol`) and the gain gate (excess AND ratio over
    max(axis, global)) break; neither breaks -> `absent`; exactly one (or a boundary case) ->
    `undetermined`. Below `present`, callers must never read this as a propagation conclusion."""
    bcfg = m2cfg["basis"]
    denom = max(gain_axis, gain_global, 1e-300)
    score_gate = off_axis_score >= bcfg["off_axis_score_tol"]
    gain_gate = ((gain_nonaxis - max(gain_axis, gain_global)) >= bcfg["nonaxis_gain_excess_tol"]
                 and (gain_nonaxis / denom) >= bcfg["nonaxis_gain_ratio_tol"])
    if score_gate and gain_gate:
        return "present"
    if not score_gate and not gain_gate:
        return "absent"
    return "undetermined"


def _off_axis_tail_agreement(gains, off_axis_score, m2cfg):
    """Asymptotic-tail agreement rule for the off_axis sentinel (spec rev2.3 §2.3/§8).

    Returns ``(verdict, tail_horizons, per_tail_decisions)``. The single-horizon read was fragile:
    ``e_nonaxis`` is a spatially compact direction, so it shows a short/mid-horizon local-
    amplification burst (the pilot-doc "core-compactness read as spread" artifact,
    docs/superpowers/specs/2026-07-04-topic4-m2-pilot-findings.md §3/§5 item 5) that makes the gain
    gate NON-MONOTONE across horizons -- on the real crossing the raw `_off_axis_decision` reads
    undetermined at 10/50/100ms but absent at 25/250/500ms. rev2.3 reads only the ASYMPTOTIC-tail
    horizons (``T >= gain.sentinel_min_horizon_ms``, = [250, 500]) where the transient has decayed,
    and REQUIRES them to agree (mirrors §4.3 epsilon_sensitivity's across-sweep agreement): the
    shared verdict if every tail decision is identical, else ``"undetermined"``. ``off_axis_score``
    is horizon-independent (a static shape score), so the score gate is constant across the tail;
    only the gain gate can vary between tail horizons."""
    gcfg = m2cfg["gain"]
    min_T = gcfg["sentinel_min_horizon_ms"]
    tail = [int(T) for T in gcfg["horizons_ms"] if int(T) >= min_T]
    decisions = [
        _off_axis_decision(off_axis_score=off_axis_score,
                           gain_nonaxis=gains["e_nonaxis"][T],
                           gain_axis=gains["e_axis_gradient"][T],
                           gain_global=gains["e_global"][T], m2cfg=m2cfg)
        for T in tail
    ]
    verdict = decisions[0] if len(set(decisions)) == 1 else "undetermined"
    return verdict, tail, decisions


def off_axis_sentinel(crossing, grid, kernels, core, m2cfg) -> dict:
    """`e_nonaxis` off_axis sentinel / negative-control at the alpha0 crossing (spec §2.3/§3.3).

    Re-derives the crossing's shape scores + loading (`shape_scores_at` on
    ``crossing["_crossing_res"]`` -- the same source T2's `read_linear_ignition` scores, but
    recomputed here because this function takes the raw T1 `crossing` dict directly, not T2's
    `linear_ignition` output) and the crossing Jacobian (`spm.build_jacobian_dense` on
    ``crossing["_crossing_op"]``). `e_nonaxis` is T0's `nonaxis_direction` residual outside
    span(e_global, e_axis_gradient); when its norm is below `nonaxis_direction_min_norm`, the
    direction carries no meaningful energy to test -- this is NOT filled with a random control
    direction (spec §2.3): `nonaxis_source_policy` records the reason, `nonaxis_gain=NaN`, and
    `off_axis` is hard-set to `"absent"` WITHOUT going through the tail rule (a NaN gain must never
    reach the both-gates arithmetic); its `off_axis_per_tail_decision` is `None` (short-circuited).

    The verdict comes from `_off_axis_tail_agreement` over the ASYMPTOTIC-tail horizons (spec rev2.3
    §2.3): both tail horizons must agree, else `undetermined`. The chosen tail + the per-tail
    decision list are persisted (`sentinel_tail_horizons_ms`, `off_axis_per_tail_decision`) so a
    reviewer can see WHY the verdict came out as it did. The reported scalar `axis_gain`/
    `global_gain`/`nonaxis_gain` are the representative values at the tail anchor (min tail
    horizon).
    """
    bcfg = m2cfg["basis"]
    theta = kernels.theta

    scores = shape_scores_at(crossing["_crossing_res"], grid, kernels, core)
    loading = scores["_loading"]
    off_axis_score = scores["off_axis"]

    e_nonaxis, _frac_resid, _frac_global, _frac_axis = nonaxis_direction(
        loading, grid, theta, bcfg["nonaxis_direction_min_norm"])

    b = basis_vectors(grid, theta)
    dirs = {"e_axis_gradient": b["e_axis_gradient"], "e_global": b["e_global"]}
    if e_nonaxis is not None:
        dirs["e_nonaxis"] = e_nonaxis

    J = spm.build_jacobian_dense(grid, kernels, crossing["_crossing_op"])
    gains = projected_gains(J, grid, kernels, dirs, m2cfg["gain"]["horizons_ms"])
    gcfg = m2cfg["gain"]
    anchor = int(gcfg["sentinel_min_horizon_ms"])          # min tail horizon = representative scalar
    tail_horizons = [int(T) for T in gcfg["horizons_ms"] if int(T) >= anchor]
    gain_axis = gains["e_axis_gradient"][anchor]
    gain_global = gains["e_global"][anchor]

    if e_nonaxis is None:
        return {
            "off_axis": "absent",
            "nonaxis_gain": float("nan"),
            "axis_gain": gain_axis,
            "global_gain": gain_global,
            "annotation": _NONAXIS_ANNOTATION,
            "nonaxis_source_policy": "unavailable_low_residual_energy",
            "sentinel_tail_horizons_ms": tail_horizons,
            "off_axis_per_tail_decision": None,           # low-residual short-circuit; tail rule not run
        }

    verdict, tail, per_tail = _off_axis_tail_agreement(gains, off_axis_score, m2cfg)
    return {
        "off_axis": verdict,
        "nonaxis_gain": gains["e_nonaxis"][anchor],
        "axis_gain": gain_axis,
        "global_gain": gain_global,
        "annotation": _NONAXIS_ANNOTATION,
        "nonaxis_source_policy": "available_residual_direction",
        "sentinel_tail_horizons_ms": tail,
        "off_axis_per_tail_decision": per_tail,
    }


# --------------------------------------------------------------------------- #
# Task 4: nonlinear-footprint spread readout (the SPREAD ADJUDICATOR)         #
# --------------------------------------------------------------------------- #
# Productionizes spec §4.1/§4.2/§4.3 (rev2.3) + results/topic4_criticality_m2/pilots/
# m2_pilots_round2.py's PILOT B (`_integrate_footprint`/`_footprint_metrics`, ported verbatim below).
# The field_rhs shift-gap itself (spec §4.1's hard JVP gate) is fixed directly in
# src/topic4_m3b_spectral_phase.py -- `field_rhs` gained `gK_field`/`hG_scalar`/`eta_K`/`eta_G` kwargs
# mirroring `solve_operating_point`'s own shift (see tests/test_topic4_m3b_spectral_phase.py::
# test_field_rhs_jvp_matches_jacobian_on_shifted_op); that is a SHARED-function fix, not part of this
# module. `integrate_footprint` below THREADS that shift into its own `field_rhs` calls (spec §4.1
# "M2b 是通用工具", a general-purpose tool): on the real crossing hG_scalar~2.9e-7 and gK_field=None,
# so the shift is numerically a no-op on THIS trajectory, but the integrator must stay correct at any
# op the pipeline is later pointed at.
_JUST_PAST_FRAC = 0.75          # spec §4.2: second of >=2 "past-critical" depths (at_crossing + this)


def _get_bracket(points):
    """(last_q, trans, a, b) -- the SAME M1 last-qualified-low -> first-saturated bracket
    `localize_alpha0_crossing` (T1) brackets on (ported from m2_pilots_round2.py's `_get_bracket`)."""
    q = [p for p in points if p.get("qualified") and p.get("branch_id") == "low_branch"]
    last_q = q[-1]
    trans = next(p for p in points if p.get("saturated") and p["time_ms"] > last_q["time_ms"])
    return last_q, trans, last_q["slow_inputs"], trans["slow_inputs"]


def _shift_from_slow(grid, core, slow_inputs, cfg_crit):
    """(gK_field, hG_scalar, eta_K, eta_G) -- the `field_rhs` shift kwargs at a slow-state. Drops the
    `exc`/`inh` fields `_fields_from_slow` also returns: the caller's `op` already carries its own
    excitability/inhibition, so only the shift SCALARS/FIELDS are needed to thread the SAME shift
    `low_solve_fast` used to build `op` into `field_rhs` (deterministic pure function of
    ``(grid, core, slow_inputs, cfg_crit)``, so re-deriving it here reproduces the exact values
    `low_solve_fast` used, without re-solving the operating point)."""
    _exc, _inh, gK_field, hG_scalar, eta_K, eta_G = _fields_from_slow(grid, core, slow_inputs, cfg_crit)
    return gK_field, hG_scalar, eta_K, eta_G


def _footprint_metrics(dRE, grid, core, theta):
    """Per-sample footprint descriptors (ported verbatim from m2_pilots_round2.py's
    `_footprint_metrics`)."""
    return {"globality": float(spm.globality(dRE, grid)),
            "elongation_axis": float(spm.elongation_axis_score(dRE, grid, theta)),
            "off_axis": float(spm.off_axis_score(dRE, grid, theta)),
            "wavevec_axis": float(spm.phase_gradient_axis_score(dRE, grid, theta)),
            "core_overlap": float(spm.core_overlap(dRE, grid, core)),
            "peak_dRE": float(np.max(np.abs(dRE)))}


def integrate_footprint(grid, kernels, op, core, theta, v, *, eps, dt, t_max, sample_ms,
                        gK_field=None, hG_scalar=0.0, eta_K=1.0, eta_G=1.0,
                        return_rate_frames=False, frame_dt_ms=1.0):
    """Integrate the (shift-fixed) `field_rhs` from z*+eps*v (and a v=0 control), report the
    kick-minus-control footprint delta_rE(t) at sample times (ported verbatim from
    m2_pilots_round2.py's `_integrate_footprint`, spec §4.1/§4.2 -- the ONLY change from the pilot is
    threading `gK_field`/`hG_scalar`/`eta_K`/`eta_G` into every `field_rhs` call). Escape when max rE
    > `spm._SAT_RATE_KHZ`.

    ``return_rate_frames`` is an additive observation hook for the shared early-recruitment
    readout.  When enabled, uniformly sampled raw ``rE_kick``/``rE_control`` fields are returned
    without changing the legacy footprint samples or classifiers.  The default is off, preserving
    M2's existing output byte-for-byte.
    """
    z0 = spm.op_state_vector(op, kernels, grid)
    fix_resid = float(np.linalg.norm(spm.field_rhs(z0, grid, kernels, op, gK_field=gK_field,
                                                   hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G)))
    z_kick = z0 + eps * v
    z_ctrl = z0.copy()
    nsteps = int(t_max / dt)
    sample_steps = {int(round(t / dt)) for t in sample_ms}
    frame_steps = set()
    frame_times, frame_kick, frame_ctrl = [], [], []
    if return_rate_frames:
        frame_stride = int(round(float(frame_dt_ms) / float(dt)))
        if frame_stride < 1 or not np.isclose(frame_stride * dt, float(frame_dt_ms), atol=1e-9):
            raise ValueError("frame_dt_ms must be a positive integer multiple of dt")
        frame_steps = set(range(0, nsteps + 1, frame_stride))
        frame_steps.add(nsteps)
    op_rE = op.rE
    traj = []
    escaped_at = None
    for it in range(nsteps + 1):
        footprint_sample = it in sample_steps or it == nsteps
        frame_sample = it in frame_steps
        if footprint_sample or frame_sample:
            rE_kick = spm.unpack_state(z_kick, grid)["rE"]
            rE_ctrl = spm.unpack_state(z_ctrl, grid)["rE"]
            dRE = rE_kick - rE_ctrl                    # perturbation response isolated from op drift
        if footprint_sample:
            fm = _footprint_metrics(dRE, grid, core, theta)
            fm["t_ms"] = float(it * dt)
            fm["max_rE_kick"] = float(np.max(rE_kick))
            fm["active_frac"] = float(np.mean(rE_kick > op_rE + 1e-4))
            traj.append(fm)
            if escaped_at is None and np.max(rE_kick) > spm._SAT_RATE_KHZ:
                escaped_at = float(it * dt)
        if frame_sample:
            frame_times.append(float(it * dt))
            frame_kick.append(np.asarray(rE_kick, float).copy())
            frame_ctrl.append(np.asarray(rE_ctrl, float).copy())
            # The readout's pre-saturation eligibility needs finer escape timing than M2's sparse
            # diagnostic samples.  This only changes the additive returned timestamp when frame
            # capture is requested; the default M2 path retains its legacy semantics.
            if escaped_at is None and np.max(rE_kick) > spm._SAT_RATE_KHZ:
                escaped_at = float(it * dt)
        if it == nsteps:
            break
        z_kick = z_kick + dt * spm.field_rhs(z_kick, grid, kernels, op, gK_field=gK_field,
                                             hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G)
        z_ctrl = z_ctrl + dt * spm.field_rhs(z_ctrl, grid, kernels, op, gK_field=gK_field,
                                             hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G)
        if not np.all(np.isfinite(z_kick)):
            escaped_at = float(it * dt); break
    out = {"fixedpoint_residual": fix_resid, "escaped_at_ms": escaped_at, "trajectory": traj}
    if return_rate_frames:
        out["rate_frames"] = {
            "times_ms": np.asarray(frame_times, float),
            "rE_kick": np.asarray(frame_kick, float),
            "rE_control": np.asarray(frame_ctrl, float),
        }
    return out


def _spread_onset(traj, m2cfg):
    """Onset classifier (spec §4.3, brief Step 7 verbatim). Evaluated on ONE footprint trajectory
    (one depth x one eps_rel x one polarity, `core_kick` direction)."""
    sp = m2cfg["spread"]; bcfg = m2cfg["basis"]
    act = [fm["active_frac"] for fm in traj]
    rose = (max(act) - act[0]) >= sp["expand_active_delta"]
    exp = [fm for fm in traj if fm["active_frac"] > act[0] + 1e-9] or traj
    elong = np.mean([fm["elongation_axis"] for fm in exp])
    offax = np.mean([fm["off_axis"] for fm in exp])
    glob0 = traj[min(2, len(traj) - 1)]["globality"]
    if offax >= bcfg["off_axis_score_tol"]:
        return "off_axis"
    if not rose:
        return "core_only"
    if elong > sp["axial_onset_thresh"] and offax < bcfg["off_axis_score_tol"]:
        return "axial"
    if glob0 >= sp["global_thresh"] and elong <= sp["axial_onset_thresh"]:
        return "global_first"
    return "undetermined"


def _spread_endgame(traj, escaped, m2cfg):
    """Endgame classifier (spec §4.3, brief Step 7 verbatim)."""
    sp = m2cfg["spread"]; act = [fm["active_frac"] for fm in traj]
    if act[-1] >= sp["flood_active_thresh"]:
        return "global_flooding"
    if escaped is None and min(act[act.index(max(act)):]) <= sp["self_limit_active_thresh"]:
        return "self_limited"
    return "marginal"


def _spread_off_axis(traj, onset, m2cfg):
    """Full-trajectory off_axis sentinel for ONE footprint run (spec §4.3). Distinct from
    `_spread_onset`'s own "off_axis" ONSET label, which only tests the EXPANSION-WINDOW MEAN.
    `"absent"` iff off_axis(t) never approaches the tolerance at ANY sampled step (spec's literal
    "off_axis 全程 < off_axis_score_tol"). Spec text says present/undetermined follow §2.3's two-gate
    rule, but §2.3's second gate is a LINEAR GAIN comparison with no analogue on a nonlinear footprint
    trajectory (no "gain" is computed here -- a documented judgment call, see task-4-report.md
    Concerns). This realizes "two gates" as two DIFFERENT statistics over the SAME
    off_axis_score_tol threshold: a peak-based full-trajectory gate (this function) and the
    mean-based expansion-window gate `_spread_onset` already computed. `"present"` only when BOTH
    fire (`_spread_onset` ALSO already read "off_axis", i.e. the excess is SUSTAINED through the
    expansion window, not a single-step blip); otherwise `"undetermined"` -- never a propagation
    conclusion below both gates (spec §7)."""
    bcfg = m2cfg["basis"]
    if max(fm["off_axis"] for fm in traj) < bcfg["off_axis_score_tol"]:
        return "absent"
    return "present" if onset == "off_axis" else "undetermined"


_EPS_ENDGAME_MAJORITY_MIN = 3   # spec §4.3 "endgame majority >=3/4" over the locked 4-combo sweep


def _all_agree(labels):
    """Shared label if every entry is identical, else None (spec §4.3 `epsilon_onset_agreement=all`;
    governs BOTH `onset` and `off_axis` per spec's "onset ∧ off_axis 在 4 组里全一致")."""
    uniq = set(labels)
    return labels[0] if len(uniq) == 1 else None


def _majority(labels, min_count):
    """Modal label if its count is >= min_count, else None (spec §4.3
    `epsilon_endgame_agreement=majority`)."""
    for lbl in set(labels):
        if labels.count(lbl) >= min_count:
            return lbl
    return None


def _aggregate_depth(detail):
    """Per-depth epsilon-sensitivity aggregation over its 4 (eps_rel x polarity) combos (spec §4.3):
    `pass` iff onset AND off_axis are UNANIMOUS across all 4 AND endgame has a >=3/4 majority."""
    onset = _all_agree([d["onset"] for d in detail])
    off_axis = _all_agree([d["off_axis"] for d in detail])
    endgame = _majority([d["endgame"] for d in detail], _EPS_ENDGAME_MAJORITY_MIN)
    return {"pass": bool(onset is not None and off_axis is not None and endgame is not None),
           "onset": onset, "endgame": endgame, "off_axis": off_axis}


# spec rev2.4 §4.3 (C decision) — the EXACT caveat string the descriptive note must carry.
_IGNITING_NOTE_CAVEAT = ("DESCRIPTIVE ONLY — primary nonlinear_spread verdict is undetermined "
                         "(pre-registered §4.3); NOT a spread claim")


def _label_unanimous_or_distribution(labels):
    """A single label string if every entry agrees, else a ``{label: count}`` distribution dict
    (spec rev2.4 §4.3 `igniting_onset` "点火子集内 onset，若一致报之" — unanimous → scalar, else
    report the distribution). Both forms are JSON-serializable."""
    counts = {}
    for lbl in labels:
        counts[lbl] = counts.get(lbl, 0) + 1
    return labels[0] if len(counts) == 1 else counts


def _descriptive_igniting_note(epsilon_detail, epsilon_sensitivity):
    """DESCRIPTIVE-ONLY igniting-subset note (spec rev2.4 §4.3, user decision C 2026-07-05).

    PURELY ADDITIVE — it NEVER changes the primary `onset`/`endgame`/`off_axis`/`epsilon_sensitivity`
    verdict (those stay pre-registered per §5; CLAUDE.md §5 "no post-hoc rule change"). It only
    SUMMARIZES the per-(depth,eps,pol) breakdown the sweep already produced, for the specific case
    where the epsilon gate FAILED *because* an onset disagreement is (at least partly) explained by
    perturbations that did not ignite (`core_only` = active_frac did not rise past `expand_active_delta`).

    Returns None unless ALL of: (a) `epsilon_sensitivity == "epsilon_sensitive"`, (b) onset is
    NON-unanimous at the primary `at_crossing` depth (i.e. onset — not off_axis/endgame — is what
    broke the gate), (c) at least one `at_crossing` combo is `core_only` (so the disagreement DOES
    stem from non-ignition; a disagreement purely between igniting classes returns None, per the
    spec's "disagreement isn't from non-igniting -> None"). `igniting_onset`/`igniting_endgame`/
    `non_igniting_combos` are read at the PRIMARY `at_crossing` depth (the depth whose pass/fail gates
    the verdict — matches the reviewer's scalar examples); `n_igniting_of_total` spans every non-empty
    depth for a robustness glance. NEVER a spread claim (caveat string + CLAUDE.md §6.3)."""
    if epsilon_sensitivity != "epsilon_sensitive":
        return None
    primary = epsilon_detail.get("at_crossing") or []
    if not primary:
        return None
    if len({d["onset"] for d in primary}) == 1:
        return None                        # onset AGREED — the sensitivity is off_axis/endgame, not onset
    igniting = [d for d in primary if d["onset"] != "core_only"]
    non_igniting = [d for d in primary if d["onset"] == "core_only"]
    if not non_igniting:
        return None                        # disagreement is between igniting classes, NOT from non-ignition
    n_igniting_of_total = {
        depth: f"{sum(1 for d in det if d['onset'] != 'core_only')}/{len(det)}"
        for depth, det in epsilon_detail.items() if det
    }
    non_igniting_combos = [
        {"eps_rel": d["eps_rel"], "polarity": d["polarity"],
         "reason": f"{'suppressing' if d['polarity'] < 0 else 'excitatory'} kick, active_frac did not rise"}
        for d in non_igniting
    ]
    return {
        "n_igniting_of_total": n_igniting_of_total,
        "igniting_onset": _label_unanimous_or_distribution([d["onset"] for d in igniting]),
        "igniting_endgame": _label_unanimous_or_distribution([d["endgame"] for d in igniting]),
        "non_igniting_combos": non_igniting_combos,
        "caveat": _IGNITING_NOTE_CAVEAT,
    }


def _run_depth_sweep(grid, kernels, op, core, theta, b_core, m2cfg, *,
                     gK_field, hG_scalar, eta_K, eta_G):
    """The full epsilon_rel x polarity `core_kick` sweep (spec §4.2/§4.3, "扰动方向...core_kick 为
    主") at ONE op/depth. Returns (detail, representative_trajectory): `detail` has one dict per
    (eps_rel, polarity) combo (feeds `_aggregate_depth` + the review's epsilon-agreement audit); the
    representative trajectory is the (largest eps_rel, last-listed polarity) run, used for
    `footprint_trajectory` plotting/audit (mirrors m2_pilots_round2.py's pilotB, whose single fixed
    run used eps_rel=0.05 at the UNMODIFIED, i.e. polarity=+1, `b_core` direction)."""
    pcfg = m2cfg["perturbation"]
    z0 = spm.op_state_vector(op, kernels, grid)
    eps_rel_list, pol_list = pcfg["epsilon_rel"], pcfg["polarities"]
    rep_eps_rel, rep_pol = eps_rel_list[-1], pol_list[-1]
    detail, rep_traj = [], None
    for eps_rel in eps_rel_list:
        eps = float(eps_rel) * float(np.linalg.norm(z0))
        for pol in pol_list:
            v = float(pol) * b_core
            r = integrate_footprint(grid, kernels, op, core, theta, v, eps=eps, dt=pcfg["dt_ms"],
                                    t_max=pcfg["max_time_ms"],
                                    sample_ms=m2cfg["spread"]["footprint_sample_ms"],
                                    gK_field=gK_field, hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G)
            traj = r["trajectory"]
            onset = _spread_onset(traj, m2cfg)
            endgame = _spread_endgame(traj, r["escaped_at_ms"], m2cfg)
            off_axis = _spread_off_axis(traj, onset, m2cfg)
            detail.append({"eps_rel": float(eps_rel), "polarity": int(pol), "onset": onset,
                          "endgame": endgame, "off_axis": off_axis,
                          "escaped_at_ms": r["escaped_at_ms"],
                          "fixedpoint_residual": r["fixedpoint_residual"]})
            if eps_rel == rep_eps_rel and pol == rep_pol:
                rep_traj = traj
    return detail, rep_traj


def _depth_footprint(op, res, shift, grid, kernels, core, theta, b_core, m2cfg):
    """Run the full `core_kick` epsilon x polarity sweep + the single `critical_mode` representative
    run (spec §4.2 "辅"/auxiliary direction) at ONE depth's (op, res, shift). Returns
    (epsilon_detail, depth_aggregate, trajectories_dict)."""
    gK_field, hG_scalar, eta_K, eta_G = shift
    detail, rep_traj = _run_depth_sweep(grid, kernels, op, core, theta, b_core, m2cfg,
                                        gK_field=gK_field, hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G)
    agg = _aggregate_depth(detail)

    crit_traj = None
    if res is not None and res.right.size:
        vcrit = np.real(res.right[:, 0]).astype(float)
        nrm = float(np.linalg.norm(vcrit))
        if nrm > 0:
            vcrit = vcrit / nrm
            z0 = spm.op_state_vector(op, kernels, grid)
            pcfg = m2cfg["perturbation"]
            eps_rep = float(pcfg["epsilon_rel"][-1]) * float(np.linalg.norm(z0))
            r_crit = integrate_footprint(grid, kernels, op, core, theta, vcrit, eps=eps_rep,
                                         dt=pcfg["dt_ms"], t_max=pcfg["max_time_ms"],
                                         sample_ms=m2cfg["spread"]["footprint_sample_ms"],
                                         gK_field=gK_field, hG_scalar=hG_scalar,
                                         eta_K=eta_K, eta_G=eta_G)
            crit_traj = r_crit["trajectory"]
    return detail, agg, {"core_kick": rep_traj, "critical_mode": crit_traj}


def read_nonlinear_spread(crossing, points, grid, kernels, core, b_core, cfg_crit, m2cfg) -> dict:
    """nonlinear_spread readout (spec §4/§1): the SPREAD ADJUDICATOR -- does the instability spread,
    and how, read from `field_rhs` footprint integration (never from the linear mode shape, spec
    "Spread verdict comes from the nonlinear footprint, never from the linear mode").

    Runs >=2 depths (`at_crossing` = T1's own bisected crossing op/res, reused directly with NO
    re-solve; `just_past` = a fresh cold-start solve at frac=0.75 along the SAME M1 bracket, matching
    m2_pilots_round2.py's pilotB, which solves each depth independently with `prev=None`) x
    `epsilon_rel` x `polarities` (spec §4.2/§4.3, `_run_depth_sweep`), classifies onset/endgame/
    off_axis per combo, and aggregates with the epsilon-sensitivity pass/fail rule (spec §4.3,
    `_aggregate_depth`). The reported top-level `onset`/`endgame`/`off_axis` are `at_crossing`'s own
    aggregated values (the crossing IS the primary object of the whole M2 pipeline); `just_past` only
    feeds `depth_dependent` (spec's own example: "at_crossing 自限、just_past 漫开 -> true").
    `control_minus_kick` is unconditionally True: `integrate_footprint` always reports the
    kick-minus-(v=0-control) delta_rE, never the raw kicked trajectory.
    """
    theta = kernels.theta
    _last_q, _trans, a, b = _get_bracket(points)

    footprint_trajectory = {}
    epsilon_detail = {}
    depth_agg = {}

    # --- at_crossing: reuse T1's already-localized/bisected op/res (spec's crossing IS this depth;
    # no re-solve -- mirrors T2/T3's own reuse of crossing["_crossing_op"]/["_crossing_res"]) ---
    op_cross = crossing["_crossing_op"]
    res_cross = crossing["_crossing_res"]
    if op_cross is not None:
        shift_cross = _shift_from_slow(grid, core, crossing["alpha0_crossing_slow_state"], cfg_crit)
        (epsilon_detail["at_crossing"], depth_agg["at_crossing"],
         footprint_trajectory["at_crossing"]) = _depth_footprint(
            op_cross, res_cross, shift_cross, grid, kernels, core, theta, b_core, m2cfg)
    else:
        # T1 crossing not localized -- mirrors T2/read_linear_ignition's own deferred-to-T5 gap
        # (progress.md), but fails CLOSED here (undetermined) rather than crashing.
        epsilon_detail["at_crossing"] = []
        depth_agg["at_crossing"] = {"pass": False, "onset": None, "endgame": None, "off_axis": None}
        footprint_trajectory["at_crossing"] = {"crossing_not_localized": True}

    # --- just_past: fresh cold-start solve at frac=0.75 along the SAME bracket (pilotB's own
    # protocol: each depth solved independently, prev=None -- no warm-start chaining between depths) ---
    slow_jp = interp_slow(a, b, _JUST_PAST_FRAC)
    op_jp, sat_jp = low_solve_fast(grid, kernels, core, slow_jp, cfg_crit, None)
    if sat_jp:
        epsilon_detail["just_past"] = []
        depth_agg["just_past"] = {"pass": False, "onset": None, "endgame": None, "off_axis": None}
        footprint_trajectory["just_past"] = {"saturated": True}
    else:
        _a1_jp, res_jp, _J_jp = alpha1_and_eig(grid, kernels, op_jp)
        shift_jp = _shift_from_slow(grid, core, slow_jp, cfg_crit)
        (epsilon_detail["just_past"], depth_agg["just_past"],
         footprint_trajectory["just_past"]) = _depth_footprint(
            op_jp, res_jp, shift_jp, grid, kernels, core, theta, b_core, m2cfg)

    cross_pass = depth_agg["at_crossing"]["pass"]
    onset = depth_agg["at_crossing"]["onset"] if cross_pass else "undetermined"
    endgame = depth_agg["at_crossing"]["endgame"] if cross_pass else "undetermined"
    off_axis = depth_agg["at_crossing"]["off_axis"] if cross_pass else "undetermined"
    epsilon_sensitivity = "pass" if cross_pass else "epsilon_sensitive"

    # depth_dependent compares the RAW per-depth endgame majority (`depth_agg[label]["endgame"]`),
    # NOT the gated top-level `endgame` above. `endgame`'s own majority can be well-defined (>=3/4
    # agreement) even when `cross_pass` is False because a DIFFERENT metric (onset) failed its
    # all-agree check at that depth -- comparing the gated "undetermined" placeholder against the
    # other depth's real endgame would manufacture a spurious depth_dependent=True driven by onset's
    # failure, not a genuine differing endgame. Both per-depth endgames must be independently
    # well-defined (not None) for the comparison to mean anything.
    ac_endgame = depth_agg["at_crossing"]["endgame"]
    jp_endgame = depth_agg["just_past"]["endgame"]
    depth_dependent = bool(ac_endgame is not None and jp_endgame is not None and ac_endgame != jp_endgame)

    # DESCRIPTIVE-ONLY (spec rev2.4 §4.3, decision C): when the epsilon gate failed by an onset
    # disagreement driven by non-igniting perturbations, additionally report the igniting-subset
    # observation. This does NOT touch the primary verdict above (which stays pre-registered
    # undetermined); None whenever the gate passed or the failure wasn't onset-by-non-ignition.
    descriptive_igniting_note = _descriptive_igniting_note(epsilon_detail, epsilon_sensitivity)

    return {
        "onset": onset,
        "endgame": endgame,
        "off_axis": off_axis,
        "depth_dependent": depth_dependent,
        "footprint_trajectory": footprint_trajectory,
        "control_minus_kick": True,
        "epsilon_sensitivity": epsilon_sensitivity,
        "descriptive_igniting_note": descriptive_igniting_note,
        "_epsilon_sweep_detail": epsilon_detail,
        "_depth_aggregate": depth_agg,
    }


# --------------------------------------------------------------------------- #
# Task 5: two-stage verdict assembly (ignition/spread) + M1 CSD co-display    #
# --------------------------------------------------------------------------- #
# Wires T1 (localize_alpha0_crossing) -> T2 (read_linear_ignition) + T3 (off_axis_sentinel) ->
# T4 (read_nonlinear_spread) into the spec §1 two-stage verdict. `csd_verdict` is M1's OWN
# unresolved_operating_point verdict (results/topic4_criticality/trajectory_verdict.json), read
# fresh here and displayed alongside -- M2 never recomputes or overrides it (spec §0-b/§7).


def _ignition_base_gate(crossing) -> bool:
    """spec §5.0 ignition base gate: the alpha0 crossing must be localized (both `_crossing_op`/
    `_crossing_res` populated -- T1's real "no crossing found" path returns both None) AND
    `op_solve_quality_left`/`op_solve_quality_right` are SEPARATE fields that must BOTH read True
    (T1 review note, progress.md: do not read one side alone as "fully clean") AND
    `branch_identity_clean`. When all three hold, the 5 continuous shape scores (spec §5.0's 4th
    conjunct) are guaranteed computable -- `shape_scores_at` only needs a resolved `_crossing_res`,
    which `op_solve_quality_left`/`_right` already required -- so there is no separate "5 scores
    present" check to perform here (it would always be a no-op once the other three hold)."""
    if crossing.get("_crossing_op") is None or crossing.get("_crossing_res") is None:
        return False
    return bool(crossing["op_solve_quality_left"]) and bool(crossing["op_solve_quality_right"]) \
        and bool(crossing["branch_identity_clean"])


def _undetermined_linear_ignition() -> dict:
    """`linear_ignition` when the §5.0 base gate fails (spec §5.4): nothing could be scored, so
    `class` falls into the pre-registered `ambiguous` bucket (no 4th enum value invented -- spec §0
    fixes `class` in {core_localized, delocalized, ambiguous}) and every descriptive/derived field
    is None. Unreachable on the actual v2.2 SIMULATION crossing (T1-T4 all confirm
    `base_gate_passed` is True there); kept for a future subject/trajectory whose crossing
    genuinely fails to localize or breaks its quality/branch-identity gate."""
    return {
        "class": "ambiguous", "delocalized_subtype": None,
        "core_overlap": None, "globality": None,
        "two_core_symmetry_break": None, "corridor_power": None,
        "shape_descriptors": None, "near_fold_note": None,
        "ignition_sensitivity": None, "off_axis_sentinel": None,
    }


def _undetermined_nonlinear_spread() -> dict:
    """`nonlinear_spread` when the §5.0 ignition base gate fails: there is no localized crossing to
    define the `at_crossing` depth (spec's crossing IS the primary object T4 integrates from), so
    the whole segment reads undetermined -- the same shape `read_nonlinear_spread` itself already
    reports on its own internal `op_cross is None` fallback (spec §5.4), reproduced here directly
    so T5 need not call into T4 (and pay for its `just_past`-only partial compute) at all."""
    return {
        "onset": "undetermined", "endgame": "undetermined", "off_axis": "undetermined",
        "depth_dependent": None, "footprint_trajectory": None,
        "control_minus_kick": None, "epsilon_sensitivity": "epsilon_sensitive",
        "descriptive_igniting_note": None,
    }


def _unresolved_subreason(crossing, base_gate_passed, sp):
    """spec §1/§5.4 `unresolved_subreason`, decoupled per segment (ignition checked first since
    `nonlinear_spread`'s own `at_crossing` depth depends on the SAME crossing T1 localizes;
    `epsilon_sensitivity` is an orthogonal spread-only gate checked second). `alpha0_not_localized`
    is the SPECIFIC "T1 found no crossing at all" case; `ignition_not_localized` covers the base
    gate failing for any other reason (quality/branch-identity)."""
    if not base_gate_passed:
        return ("alpha0_not_localized" if crossing.get("_crossing_op") is None
                else "ignition_not_localized")
    if sp["epsilon_sensitivity"] != "pass":
        return "unresolved_nonlinear_spread"
    return None


def _interpretation(ig, sp) -> str:
    """spec §5.3 mechanical compose (task brief Step 3, verbatim): glue `nonlinear_spread`'s own
    onset/endgame/off_axis onto `linear_ignition`'s class -- NEVER re-glue spread onto the linear
    mode itself (e.g. never "the critical mode is axial"; spec §5.3/§7)."""
    return (f"{ig['class']} ignition followed by {sp['onset']} transient and {sp['endgame']}; "
            f"off_axis {sp['off_axis']}")


def build_ignition_spread_verdict(points, cfg_crit, m2cfg) -> dict:
    """The spec §1 two-stage verdict: `csd_verdict` (M1, unchanged, read fresh from
    ``results/topic4_criticality/trajectory_verdict.json``) co-displayed with `linear_ignition`
    (T2 `read_linear_ignition`, plus T3 `off_axis_sentinel` nested at
    `linear_ignition["off_axis_sentinel"]` -- spec §3.3 is part of the same "M2a" ignition bucket
    as §3.2) and `nonlinear_spread` (T4 `read_nonlinear_spread`, passed through unchanged -- its
    own return dict already matches spec §1's `nonlinear_spread` schema field-for-field, so no
    reshaping is needed). `interpretation` is the §5.3 mechanical compose. `base_gate_passed`/
    `unresolved_subreason` implement §5.0/§5.4: when the alpha0 crossing is not localized or fails
    its quality/branch-identity gate, `read_linear_ignition`/`off_axis_sentinel`/
    `read_nonlinear_spread` are NOT called (two of the three would crash on a None
    `_crossing_res`) -- both segments report their own undetermined shape instead (§5.4, the two
    segments stay decoupled).

    `linear_ignition["crossing"]` carries T1's full (unstripped) crossing dict forward -- needed by
    the CLI to re-derive the crossing-mode-loading/basis-sanity figures without a second expensive
    op-solve (mirrors T3's own "recompute shape_scores_at, cheap, no re-solve" precedent) and kept
    for traceability (`alpha0_crossing_time_ms`/`crossing_frac`/etc.). Private (`_`-prefixed,
    non-JSON-serializable) fields anywhere in the returned tree -- `crossing`'s own `_crossing_op`/
    `_crossing_res`, `linear_ignition`'s `_two_core_crossing`/`_two_core_region_frac`/
    `_two_core_axis_profile`, `nonlinear_spread`'s `_epsilon_sweep_detail`/`_depth_aggregate` -- are
    intentionally NOT stripped here (mirrors M1's own build/write split, run_topic4_crit_verdict.py):
    stripping happens at JSON-write time in the CLI, so the in-memory return stays fully
    introspectable for tests/figures.
    """
    grid, kernels, core, b_core = _crit_op_context(cfg_crit)
    csd_verdict = json.loads(_M1_VERDICT_PATH.read_text())["verdict"]

    crossing = localize_alpha0_crossing(points, grid, kernels, core, cfg_crit, m2cfg)
    base_gate_passed = _ignition_base_gate(crossing)

    if base_gate_passed:
        ig = read_linear_ignition(crossing, grid, kernels, core, cfg_crit, m2cfg, points)
        ig["off_axis_sentinel"] = off_axis_sentinel(crossing, grid, kernels, core, m2cfg)
        sp = read_nonlinear_spread(crossing, points, grid, kernels, core, b_core, cfg_crit, m2cfg)
    else:
        ig = _undetermined_linear_ignition()
        sp = _undetermined_nonlinear_spread()
    ig["crossing"] = crossing

    return {
        "csd_verdict": csd_verdict,
        "linear_ignition": ig,
        "nonlinear_spread": sp,
        "interpretation": _interpretation(ig, sp),
        "base_gate_passed": base_gate_passed,
        "unresolved_subreason": _unresolved_subreason(crossing, base_gate_passed, sp),
    }
