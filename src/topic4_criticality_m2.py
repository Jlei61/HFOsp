"""Topic 4 M3-v2.2 criticality Milestone 2 — two-stage ignition/spread readout.

Productionizes the M2 de-risk pilots (results/topic4_criticality_m2/pilots/*.py).
Spec: docs/superpowers/specs/2026-07-04-topic4-m3v2-2-m2-critical-mode-decomposition-design.md (rev2.1).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import yaml
import src.topic4_m3b_spectral_phase as spm
from src.topic4_criticality import _fields_from_slow, check_low_branch_continuation_between

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _REPO / "config/topic4_criticality_m2.yaml"


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
