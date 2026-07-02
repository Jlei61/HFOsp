"""Topic 4 M3-v2.2 approach-criticality config loader (Task 0).

Loads the config-of-record `config/topic4_criticality.yaml`: operator units,
verdict thresholds + threshold-sweep, quality-gate floors, branching policy,
mode-selection policy, finite-time-gain horizons, the slow_to_ratefield entry
terminology lock, slow_sensitivity finite-difference steps, atlas grid, and
the virtual_seeg estimator-reuse contract.

This module will be heavily extended by later tasks (spec
docs/superpowers/specs/2026-07-02-topic4-m3v2-2-approach-criticality-design.md);
kept to the config loader only for now.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "topic4_criticality.yaml"


def load_crit_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load the topic4 criticality config YAML as a dict.

    path=None resolves to config/topic4_criticality.yaml relative to the repo root.
    """
    cfg_path = Path(path) if path is not None else _DEFAULT_CONFIG_PATH
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# slow_to_ratefield sign-test deliverable (Task 2.5, #P1-1)                      #
# --------------------------------------------------------------------------- #


def slow_to_ratefield_sign_ok(cfg: Dict[str, Any]) -> Dict[str, Dict[str, bool]]:
    """#P1-1 sign-test: for each slow_to_ratefield var, raising it must NOT raise E excitability.

    Reads ``eta_K``/``eta_G`` off ``cfg["slow_to_ratefield"]`` (config/topic4_criticality.yaml
    #P1-1 lock) and probes each var independently on a small single-core operating point.

    Returns a STRUCTURED per-var dict of component bools (not one flat pass/fail) because q_I is
    intentionally held to a rate-only criterion while g_K/h_G are held to the strict
    alpha_1+rate criterion -- a flat bool would make a downstream reader (e.g. T3c attribution)
    misread q_I as having passed the SAME strict gate g_K/h_G passed:

    * ``q_I`` -- already wired via ``InhibitionField.q`` scaling ``W_EI`` (target=E_inhibition,
      pre-existing path); raise ``q_global``. Criterion: ``rate_not_higher`` (mean rE not higher)
      -- this is q_I's own pre-existing, documented contract ("Lower q -> weaker brake -> more
      event-prone", see ``build_inhibition_field``'s module comment). ``W_EI`` enters BOTH muE
      (subtracted) and varE (added, squared), so raising q_I also raises sigmaE; near a
      threshold-adjacent operating point the LIF gain's sigma-sensitivity can transiently outweigh
      its mu-sensitivity, so alpha_1 is not required to fall in lockstep (verified empirically:
      robust across a q in [0.84, 1.0] scan, not a probe-point fluke) -- the controller's own
      sign-test intent note scopes the stronger "AND lower alpha_1" claim to hG/gK only, never to
      q_I. Its dict carries the fixed marker ``alpha1_not_required=True`` (not a probe result) so
      downstream code can tell q_I's gate apart from g_K/h_G's without re-deriving the physics.
    * ``g_K`` -- Task 2.5 wiring, ``muE -= eta_K*gK_field`` (target=E_current, per-cell); raise a
      uniform field. Criteria: ``alpha1_not_higher`` AND ``rate_not_higher`` (#P1-1's literal claim).
    * ``h_G`` -- Task 2.5 wiring, ``muE -= eta_G*hG_scalar`` (target=E_current, global scalar);
      raise the scalar. Criteria: ``alpha1_not_higher`` AND ``rate_not_higher`` (#P1-1's literal
      claim). g_K/h_G are pure muE-only additive shifts (varE has zero dependence on either), so
      both criteria hold together cleanly -- unlike q_I's weight-scaling path.

    Only consumes ``src.topic4_m3b_spectral_phase``'s generic ``gK_field``/``hG_scalar``/``eta_K``/
    ``eta_G`` keywords -- that module has no knowledge of this config's schema.
    """
    from src.topic4_m3b_spectral_phase import (
        Grid, build_kernels, make_core_mask, build_excitability_field, build_inhibition_field,
        solve_operating_point, build_jacobian_dense, rate_eigenpairs,
    )

    block = cfg["slow_to_ratefield"]
    eta_K = float(block["g_K"]["eta_K"])
    eta_G = float(block["h_G"]["eta_G"])

    grid = Grid(n=6, L=5.0)
    kernels = build_kernels(grid, ell_perp=0.6)
    core = make_core_mask(grid, kind="single", radius=0.9)
    exc = build_excitability_field(grid, core, mu_core=1.0)
    inh_lo = build_inhibition_field(grid, core, q_global=0.94)

    def alpha1_and_rate(op) -> tuple[float, float]:
        a1 = float(rate_eigenpairs(build_jacobian_dense(grid, kernels, op), grid).eigenvalues[0].real)
        return a1, float(op.rE.mean())

    a0, r0 = alpha1_and_rate(solve_operating_point(grid, kernels, exc, inh_lo))

    def rate_ok(op_hi) -> bool:
        _, r1 = alpha1_and_rate(op_hi)
        return bool(r1 <= r0 + 1e-9)

    def alpha1_ok(op_hi) -> bool:
        a1, _ = alpha1_and_rate(op_hi)
        return bool(a1 <= a0 + 1e-9)

    inh_hi = build_inhibition_field(grid, core, q_global=0.99)
    op_qI = solve_operating_point(grid, kernels, exc, inh_hi)

    gK_field = np.full((grid.n, grid.n), 2.0)
    op_gK = solve_operating_point(grid, kernels, exc, inh_lo, gK_field=gK_field, eta_K=eta_K)

    op_hG = solve_operating_point(grid, kernels, exc, inh_lo, hG_scalar=2.0, eta_G=eta_G)

    return {
        "q_I": {"rate_not_higher": rate_ok(op_qI), "alpha1_not_required": True},
        "g_K": {"rate_not_higher": rate_ok(op_gK), "alpha1_not_higher": alpha1_ok(op_gK)},
        "h_G": {"rate_not_higher": rate_ok(op_hG), "alpha1_not_higher": alpha1_ok(op_hG)},
    }


# --------------------------------------------------------------------------- #
# M3A-v2.2 -> M3B-R2 interface export (Task 1): fail-closed handoff wiring       #
# --------------------------------------------------------------------------- #
# The v2.2 approach-to-criticality sim feeds the SAME canonical M3A->M3B handoff
# contract as A2 (src/sef_hfo_m3_interface.py + src/sef_hfo_m3a_export.py). The
# real export is EXPECTED to refuse the phase-map overlay because the slow->rate
# mapping for this sim is NOT calibrated -- that refusal is a science outcome, not
# an adapter bug. export_fixture_handoff isolates "the machinery works" from "real
# data legitimately refuses" by feeding a hand-built sign-calibrated mapping that
# passes all four overlay conditions.


def _fixture_calibrated_mapping_and_ranges(mapping_id: str):
    """A hand-built SIGN-calibrated mapping (+ranges) that passes all four overlay
    conditions -- the KNOWN-GOOD control for export_fixture_handoff.

    Starts from the canonical uncalibrated placeholder (schema-valid, physically sensible
    reciprocal-affine transforms) and flips only the two on-axis coords to
    calibration_status='passed' with a passing sign_test. The transforms are strictly
    monotone over their calibrated input domain, so check_sign_direction holds.
    """
    from src.sef_hfo_m3_interface import ON_AXIS_COORDS
    from src.sef_hfo_m3a_export import default_precalib_mapping_and_ranges
    mapping, ranges = default_precalib_mapping_and_ranges(mapping_id)
    for coord in ON_AXIS_COORDS:
        c = mapping["coordinates"][coord]
        t = c["transform"]
        c["calibration_status"] = "passed"
        c["sign_tests"] = [{
            "name": f"{coord}_sign_cal", "coord": coord, "input_var": t["input_var"],
            "expected_direction": t["expected_direction"],
            "observed_slope_sign": ("negative" if t["expected_direction"] == "decreasing_in_input"
                                    else "positive"),
            "passed": True, "engine_sha": "fixture",
        }]
    return mapping, ranges


def _fixture_landmark_rows() -> list:
    """Landmark rows whose slow values sit solidly inside the calibrated input domain
    (q in [0.25,1] -> phase in [0,1]) with canonical event_stages, so every trajectory
    row is phase_coord_valid and in-range (satisfies cond3)."""
    return [
        {"time_ms": 0.0, "event_id": 0, "event_stage": "onset", "q_core": 0.90, "q_global": 0.90, "g_K": 0.10},
        {"time_ms": 10.0, "event_id": 0, "event_stage": "peak", "q_core": 0.55, "q_global": 0.65, "g_K": 0.30},
        {"time_ms": 20.0, "event_id": 0, "event_stage": "end", "q_core": 0.40, "q_global": 0.50, "g_K": 0.50},
    ]


def _fixture_passing_summary(mapping_id: str) -> dict:
    """Summary satisfying cond4 (STRICT A2 gate): gate_A PASS + rate_matched passed +
    rate_matched_group recorded."""
    return {
        "slow_to_rate_mapping_id": mapping_id,
        "gate_A_trajectory": "PASS",
        "gate_B_seizure_like": "INCONCLUSIVE",
        "trajectory_robustness": "robust",
        "rate_matched_control": "passed",
        "rate_matched_group": {"n": 8, "source": "fixture"},
        "out_of_range_fraction": 0.0,
        "forbidden_claims": [],
    }


def export_fixture_handoff(out_dir) -> str:
    """Write a KNOWN-GOOD handoff (calibrated mapping + passing phenotype summary) and
    return its overlay_verdict. Guaranteed 'phase_map_trajectory' -- proves the M3A->M3B
    interface machinery is wired correctly, so a 'refused' verdict on real data is a
    science outcome, not an adapter bug."""
    from src.sef_hfo_m3a_export import write_handoff_artifacts
    mapping_id = "m3a_v2_2_fixture"
    mapping, ranges = _fixture_calibrated_mapping_and_ranges(mapping_id)
    audit = write_handoff_artifacts(
        str(out_dir),
        landmark_rows=_fixture_landmark_rows(),
        mapping=mapping, ranges=ranges,
        summary=_fixture_passing_summary(mapping_id),
    )
    return audit["overlay_verdict"]


def export_v2_2_handoff(out_dir, cfg: Dict[str, Any]) -> str:
    """Run the v2.2 transition sim and write the fail-closed M3A->M3B handoff artifacts.

    Uses the DEFAULT uncalibrated mapping (build_handoff_from_sim mapping/ranges=None) so the
    self-audit legitimately REFUSES the phase-map overlay: the slow->rate mapping for this
    approach-to-criticality sim is not calibrated and the phenotype gate is INCONCLUSIVE.
    The mapping is NOT weakened to force a pass -- refusal here is the honest verdict.
    Returns the overlay_verdict (expected 'refused' / 'mechanism_candidate_only').
    """
    os.makedirs(out_dir, exist_ok=True)
    from src.sef_hfo_transition_sim import run_transition, sim_dict_for_handoff
    from src.sef_hfo_m3a_export import build_handoff_from_sim, write_handoff_artifacts
    res = run_transition(cfg)
    h = build_handoff_from_sim(
        sim_dict_for_handoff(res), res["events"], res["dt_ms"],
        mapping_id="m3a_v2_2_approach", gk_enabled=cfg["use_gK"],
    )
    audit = write_handoff_artifacts(str(out_dir), **h)
    return audit["overlay_verdict"]


# --------------------------------------------------------------------------- #
# Conditional 2-D atlas (Task 2): VISUALIZATION/CONTEXT ONLY -- NEVER the verdict #
# --------------------------------------------------------------------------- #
# This runs BEFORE g_K/h_G are wired into solve_operating_point (Task 2.5). The atlas's two
# axes -- phase_x_core and phase_y_global -- are both inhibition-efficacy knobs
# (src.topic4_m3b_spectral_phase.build_inhibition_field's q_core/q_global) already handled by
# solve_operating_point; phase_recovery (g_K) is FIXED/projected-out, not injected. The verdict
# for whether the M3-v2.2 trajectory approaches criticality NEVER comes from this atlas -- it
# comes from the actual trajectory (Task 3a-5); the `verdict_source` meta field is the guard
# against a future reader mistaking this 2-D slice for that verdict.


def _invert_phase_transform(transform: Dict[str, Any], phase: float) -> float:
    """Invert a mapping coordinate transform: normalized phase in [0,1] -> raw slow-var value.

    Mirrors src.sef_hfo_m3_interface._apply_transform's TRANSFORM_TYPES enum (identity/affine/
    reciprocal_affine) but solves for the input, not the output. Needed because the atlas is
    built over normalized phase nodes while solve_operating_point needs the raw knob (q_core /
    q_global), and no inverse of that transform exists elsewhere in the codebase.
    """
    ttype = transform["type"]
    a = float(transform.get("a", 1.0))
    b = float(transform.get("b", 0.0))
    if ttype == "identity":
        return float(phase)
    if ttype == "affine":
        return (float(phase) - b) / a
    if ttype == "reciprocal_affine":
        return a / (float(phase) - b)
    raise ValueError(f"unknown transform.type {ttype!r}")


def _resolve_phase_recovery(cfg: Dict[str, Any]) -> tuple[str, float]:
    """Resolve the FIXED phase_recovery slice the atlas is conditioned on.

    Only policy=trajectory_median is implemented (the current config-of-record,
    config/topic4_criticality.yaml atlas.phase_recovery_condition): the median of the T1
    trajectory's raw g_K trace (trace_gK_axial), from the SAME default v2.2 transition run
    T1's golden fixture is captured against (re-use, not re-invent, per CLAUDE.md 6.1). g_K is
    NOT injected into the atlas op-solve (Task 2.5 wires that); this value is provenance
    recorded into atlas_name only.
    """
    cond = cfg["atlas"]["phase_recovery_condition"]
    policy = cond["policy"]
    if policy != "trajectory_median":
        raise NotImplementedError(
            f"phase_recovery_condition.policy={policy!r} not implemented; "
            "only 'trajectory_median' is wired (config-of-record).")
    from src.sef_hfo_transition_sim import run_transition, default_transition_config
    res = run_transition(default_transition_config())
    value = float(np.median(res["trace_gK_axial"]))
    return policy, value


def build_conditional_atlas(mapping: Dict[str, Any], ranges: Dict[str, Any],
                            cfg: Dict[str, Any], out_dir) -> Dict[str, Any]:
    """Build the conditional 2-D phase atlas and write finite_jacobian_grid.json.

    Scans a normalized phase_x_core x phase_y_global grid in [0,1] (atlas.normalized_grid_n
    nodes per axis) at a FIXED phase_recovery slice (provenance recorded in atlas_name),
    inverting the mapping's on-axis transforms to the raw q_core / q_global inhibition knobs
    solve_operating_point already understands via build_inhibition_field -- mu_core stays 0.0
    and g_K is never injected (not wired until Task 2.5). VISUALIZATION/CONTEXT ONLY: the
    verdict_source meta field always reads "actual_trajectory_not_atlas".
    """
    import src.topic4_m3b_spectral_phase as spm

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, phase_recovery_value = _resolve_phase_recovery(cfg)
    mapping_id = mapping["slow_to_rate_mapping_id"]

    n = int(cfg["atlas"]["normalized_grid_n"])
    x_range = ranges["phase_x_core"]
    y_range = ranges["phase_y_global"]
    phase_x_vals = np.linspace(float(x_range["min"]), float(x_range["max"]), n)
    phase_y_vals = np.linspace(float(y_range["min"]), float(y_range["max"]), n)
    x_transform = mapping["coordinates"]["phase_x_core"]["transform"]
    y_transform = mapping["coordinates"]["phase_y_global"]["transform"]

    # Modest spatial grid -- the atlas is viz-only, a large grid would make ~n^2 phase nodes
    # minutes-to-hours. n=8, L=grid_L matches the established M3B primary phase-map grid
    # (scripts/build_m3b_spectral_outputs.py).
    grid = spm.Grid(n=8, L=float(cfg["atlas"]["grid_L"]))
    kernels = spm.build_kernels(grid)
    core = spm.make_core_mask(grid, kind="single", radius=0.9)

    points: list = []
    rows: list = []
    for py in phase_y_vals:
        q_global = _invert_phase_transform(y_transform, float(py))
        for px in phase_x_vals:
            q_core = _invert_phase_transform(x_transform, float(px))
            p = spm.analyze_spectral_point(grid, kernels, core, mu_core=0.0,
                                           q_global=q_global, q_core=q_core)
            points.append(p)
            row = p.as_row()
            row["phase_x_core"] = float(px)
            row["phase_y_global"] = float(py)
            rows.append(row)

    meta = {
        "m3a_overlay_consumable": True,
        "atlas_name": f"conditional_2d_atlas_at_phase_recovery={policy}:{phase_recovery_value:.6g}",
        "verdict_source": "actual_trajectory_not_atlas",
        "axes_built_from_slow_to_rate_mapping_id": mapping_id,
        "axis_space": "normalized_unit",
        "x_axis": "phase_x_core (normalized core inhibition-exhaustion phase; raw knob q_core)",
        "y_axis": "phase_y_global (normalized global inhibition-exhaustion phase; raw knob q_global)",
        "phase_recovery_policy": policy,
        "phase_recovery_value": phase_recovery_value,
        "normalized_grid_n": n,
        "grid_n": grid.n, "grid_L": grid.L,
        "phase_x_values": [float(v) for v in phase_x_vals],
        "phase_y_values": [float(v) for v in phase_y_vals],
        "unresolved_fraction": spm.unresolved_fraction(points),
        "mode_class_counts": {c: sum(1 for p in points if p.mode_class == c)
                              for c in sorted({p.mode_class for p in points})},
        "points": rows,
    }
    (out_dir / "finite_jacobian_grid.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- #
# Operating-point quality gate (Task 3a-1, #5/#7/#8)                            #
# --------------------------------------------------------------------------- #


def rate_mismatch(rate_sim, z_star, rate_scale_floor):
    import numpy as np
    a = np.asarray(rate_sim, float).ravel()
    b = np.asarray(z_star, float).ravel()
    rms = float(np.sqrt(np.mean((a - b) ** 2)))
    scale = max(float(np.median(np.abs(b))), float(rate_scale_floor))     # #8 quiet-branch floor
    return rms, rms / scale


def adiabatic_index(slow_speed, alpha1, slow_scale, eps=1e-9):
    tf = (-1.0 / alpha1) if alpha1 < 0 else float("inf")
    return float(slow_speed) * tf / (float(slow_scale) + eps)


_REQ = ["converged", "saturated", "residual_rms", "rate_mismatch_abs", "rate_mismatch_rel",
        "slow_mismatch_rel", "adiabatic_index", "alpha_drift_index"]

_NUMERIC_REQ = ["residual_rms", "rate_mismatch_abs", "rate_mismatch_rel",
                "slow_mismatch_rel", "adiabatic_index", "alpha_drift_index"]


def qualify_point(f, cfg):
    g = cfg["quality_gate"]
    for k in _REQ:
        if k not in f or f[k] is None:
            return (False, f"missing_{k}")                               # #5 fail-closed
    for k in _NUMERIC_REQ:
        if not np.isfinite(f[k]):
            return (False, f"nonfinite_{k}")                              # user 1-1: NaN/inf fail-closed
    if not f["converged"]:
        return (False, "nonconverged")
    if f["saturated"]:
        return (False, "saturated")
    if f["residual_rms"] >= g["residual_rms_tol"]:
        return (False, "high_residual")
    if f["rate_mismatch_abs"] >= g["rate_mismatch_abs_tol"] and f["rate_mismatch_rel"] >= g["rate_mismatch_rel_tol"]:
        return (False, "rate_mismatch")                                   # #8 both abs AND rel
    if f["slow_mismatch_rel"] >= g["slow_mismatch_rel_tol"]:
        return (False, "slow_mismatch")
    if f["adiabatic_index"] >= g["adiabatic_index_tol"]:
        return (False, "not_quasistatic")
    if f["alpha_drift_index"] >= g["alpha_drift_index_tol"]:
        return (False, "alpha_drift_too_fast")                            # #7
    return (True, "qualified")


# --------------------------------------------------------------------------- #
# Branch-aware operating-point protocol (Task 3a-2, #9 field-distance / #11 deterministic seed) #
# --------------------------------------------------------------------------- #
# spec §5: solve_operating_point's steady-state solve can land on different rate branches
# depending on where it starts (a quiescent low-rate fixed point vs. a saturated/high-rate one).
# solve_branches probes several warm starts (cfg["branching"]["solve_inits"]) and clusters the
# results so CSD reads only the low/approach branch, never silently averaging across a fold.

_BRANCH_LOW_RATE_SEED_KHZ: float = 1e-3    # "low_rate" warm start -- same magnitude as
                                            # solve_operating_point's own pre-mean_field fallback.
_BRANCH_HIGH_RATE_SEED_KHZ: float = 0.20   # "high_rate" warm start -- ~2x m3b's saturation
                                            # threshold (topic4_m3b_spectral_phase._SAT_RATE_KHZ=0.10
                                            # kHz), biasing the solve toward a high/saturated branch.
_BRANCH_RANDOM_SMALL_MAX_KHZ: float = 0.01  # "random_small" warm start upper bound -- small
                                             # relative to saturation; probes alternate near-quiescent ICs.


@dataclass
class Branch:
    """One clustered rate-branch found by ``solve_branches`` from a single (grid, exc, inh) point.

    The six plan fields (``branch_id``.. ``branch_selected_reason``) are the JSON-serializable
    summary. ``op`` (the cluster's representative ``OperatingPoint``) is carried alongside because
    a branch is only USABLE downstream (T3a-5's low-branch spectral read-out, and its own
    ``previous_point`` warm-start on the NEXT trajectory point) if the actual operating point is
    reachable, not just its scalar summary.
    """
    branch_id: int
    branch_rate_mean: float
    branch_field_distance_to_low: float
    branch_alpha1: float
    branch_residual: float
    branch_selected_reason: str
    op: Any


def _branch_field_distance(a_rE: np.ndarray, b_rE: np.ndarray, floor: float) -> float:
    """#9 FIELD-level (not scalar-rate) branch distance: RMS over the whole rE spatial field,
    normalized by the larger of the two fields' own median-|rE| scale (or ``floor`` when both are
    quiet/near-zero) -- so two structurally different spatial solutions cannot be conflated just
    because their means happen to agree, and a near-zero denominator cannot blow up the ratio.
    Reuses ``quality_gate.rate_scale_floor`` (the same "quiet-branch absolute floor" ``rate_mismatch``
    already applies) rather than inventing a second floor concept.
    """
    rms = float(np.sqrt(np.mean((a_rE - b_rE) ** 2)))
    scale = max(float(floor), float(np.median(np.abs(a_rE))), float(np.median(np.abs(b_rE))))
    return rms / scale


def solve_branches(grid, kernels, exc, inh, cfg: Dict[str, Any], *, prev=None, seed_key=None,
                   gK_field=None, hG_scalar: float = 0.0, eta_K: float = 1.0,
                   eta_G: float = 1.0) -> list:
    """Branch-aware operating-point protocol (spec §5, plan T3a-2).

    Solves ``cfg["branching"]["solve_inits"]`` from several initial conditions via
    ``solve_operating_point(init=...)`` (T3a-2 #10), clusters the results by FIELD distance (#9,
    ``_branch_field_distance`` / ``branch_cluster_field_tol``), and labels each cluster
    ``saturated_branch`` (any member ``op.saturated``) / ``low_branch`` / ``high_branch`` (by rate,
    among the remaining clusters) / ``ambiguous_branch`` (a cluster mixes saturated and
    non-saturated members, or -- rare, unexercised by the T3a-2 config -- a 3rd+ distinct
    non-saturated regime that is neither the lowest nor the highest).

    ``prev``: an ``OperatingPoint`` (e.g. the previous trajectory point's selected low branch)
    enables the ``previous_point`` warm start; when ``prev is None`` that ``solve_inits`` entry is
    skipped (nothing to warm-start from -- e.g. the first point of a trajectory).

    ``gK_field``/``hG_scalar``/``eta_K``/``eta_G`` are the T2.5 ``slow_to_ratefield`` shift, forwarded
    to EVERY per-init ``solve_operating_point`` so the branch protocol solves at the SAME shifted
    operating point the trajectory eval reads out (T3a-5b: g_K's per-cell field / h_G's global scalar
    become load-bearing here -- the T2.5 review flagged this as the point where a heterogeneous g_K
    field is exercised). The Jacobian/eigen read-out downstream needs no change: ``build_jacobian_dense``
    reads the gains off the already-shifted op. Defaults (``gK_field=None``, ``hG_scalar=0.0``) are
    additive-zero -> byte-parity with every existing caller.

    ``seed_key`` seeds ``random_small`` DETERMINISTICALLY via ``np.random.default_rng(seed_key)``
    directly. numpy's ``SeedSequence`` hashes an int/tuple seed STABLY across processes; Python's
    built-in ``hash()`` is PROCESS-SALTED (``PYTHONHASHSEED``) and would silently break the "same
    seed_key -> identical branches" contract (#11) the first time this ran under a different
    hash seed, even though two calls in the SAME process would appear deterministic. Never use
    ``hash()`` here. ``seed_key=None`` falls back to the fixed literal seed ``0`` (never
    ``hash(None)``).
    """
    from src.topic4_m3b_spectral_phase import solve_operating_point, build_jacobian_dense, rate_eigenpairs

    bc = cfg["branching"]
    floor = float(cfg["quality_gate"]["rate_scale_floor"])
    tol = float(bc["branch_cluster_field_tol"])
    shape = (grid.n, grid.n)

    solved: list = []                                          # [(init_name, OperatingPoint), ...]
    for name in bc["solve_inits"]:
        if name == "low_rate":
            init = {"rE": _BRANCH_LOW_RATE_SEED_KHZ, "rI": _BRANCH_LOW_RATE_SEED_KHZ}
        elif name == "high_rate":
            init = {"rE": _BRANCH_HIGH_RATE_SEED_KHZ, "rI": _BRANCH_HIGH_RATE_SEED_KHZ}
        elif name == "previous_point":
            if prev is None:
                continue                                        # nothing to warm-start from yet
            init = {"rE": prev.rE, "rI": prev.rI}
        elif name == "random_small":
            rng = np.random.default_rng(0 if seed_key is None else seed_key)   # #11 -- stable, no hash()
            init = {"rE": rng.uniform(0.0, _BRANCH_RANDOM_SMALL_MAX_KHZ, size=shape),
                     "rI": rng.uniform(0.0, _BRANCH_RANDOM_SMALL_MAX_KHZ, size=shape)}
        else:
            raise ValueError(f"unknown branching.solve_inits entry {name!r}")
        solved.append((name, solve_operating_point(
            grid, kernels, exc, inh, init=init,
            gK_field=gK_field, hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G)))

    # --- #9 cluster by FIELD distance: greedy, compare each new point against each existing
    # cluster's first (deterministic, solve_inits-order) member. ---
    clusters: list = []                                          # list[list[int]] (indices into solved)
    for i, (_, op) in enumerate(solved):
        placed = False
        for cluster in clusters:
            ref_op = solved[cluster[0]][1]
            if _branch_field_distance(op.rE, ref_op.rE, floor) <= tol:
                cluster.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])

    # --- label: saturated / ambiguous (mixed-saturation) / provisional rate-branch ---
    reasons: dict = {}
    provisional: list = []                                        # [(cluster_idx, members), ...]
    for ci, members in enumerate(clusters):
        sat_flags = {solved[m][1].saturated for m in members}
        if len(sat_flags) > 1:
            reasons[ci] = "ambiguous_branch"
        elif True in sat_flags:
            reasons[ci] = "saturated_branch"
        else:
            provisional.append((ci, members))

    # low/high by rate, across the provisional (non-saturated, non-mixed) clusters.
    provisional.sort(key=lambda cm: solved[cm[1][0]][1].rE.mean())
    for rank, (ci, _members) in enumerate(provisional):
        if rank == 0:
            reasons[ci] = "low_branch"
        elif rank == len(provisional) - 1:
            reasons[ci] = "high_branch"
        else:
            reasons[ci] = "ambiguous_branch"      # 3rd+ distinct non-saturated regime: fits neither
                                                    # binary label cleanly (documented fallback).

    # --- #9 distance-to-low reference: the (post-label) low_branch cluster's representative op;
    # if none was labeled low_branch (e.g. every solve saturated), fall back to the deterministic
    # "low_rate"-seeded solve itself when present, else the first solved item. ---
    low_ci = next((ci for ci, r in reasons.items() if r == "low_branch"), None)
    if low_ci is not None:
        low_ref_rE = solved[clusters[low_ci][0]][1].rE
    else:
        low_rate_idx = next((i for i, (name, _) in enumerate(solved) if name == "low_rate"), 0)
        low_ref_rE = solved[low_rate_idx][1].rE

    branches: list = []
    for ci, members in enumerate(clusters):
        rep_op = solved[members[0]][1]
        # F3 EMPTY-GUARD (matches evaluate_actual_trajectory_points, review finding): an unresolved /
        # empty spectrum cannot be indexed [0] -- fall back to NaN rather than crash. branch_alpha1 is
        # a descriptive JSON field computed AFTER `reasons` (branch selection) is already finalized
        # above, so a NaN here cannot perturb which cluster is low/high/saturated/ambiguous.
        _eig = rate_eigenpairs(build_jacobian_dense(grid, kernels, rep_op), grid)
        alpha1 = (float(_eig.eigenvalues[0].real)
                  if (_eig.status == "resolved" and _eig.eigenvalues.size > 0)
                  else float("nan"))
        branches.append(Branch(
            branch_id=ci,
            branch_rate_mean=float(rep_op.rE.mean()),
            branch_field_distance_to_low=_branch_field_distance(rep_op.rE, low_ref_rE, floor),
            branch_alpha1=alpha1,
            branch_residual=float(rep_op.residual),
            branch_selected_reason=reasons[ci],
            op=rep_op,
        ))
    return branches


# --------------------------------------------------------------------------- #
# Non-normality: numerical abscissa + directional finite-time gain (Task 3a-4, #15/#16/#17) #
# --------------------------------------------------------------------------- #
# spec §non-normality: a stable operator (every eigenvalue's real part < 0) can still transiently
# amplify a perturbation before it decays if J is non-normal. numerical_abscissa bounds that growth
# from the SYMMETRIC (Hermitian) part of J -- it can be positive even when alpha_1 < 0.
# directional_finite_time_gain_curve tracks the amplification along a specific direction b (the
# core-perturbation direction, not the full operator norm -- #15 review-resolved as directional, not
# operator, per plan Task 3a-4 "directional-vs-operator") over a set of horizons.
# transient_amplification_present flags a stable-but-transiently-amplifying operator, with alpha_1>=0
# guarded off (that is modal growth, not a stable transient -- #17).


def numerical_abscissa(J) -> float:
    """Max eigenvalue of J's Hermitian part -- can be positive even when every eigenvalue of
    (non-normal) J has negative real part (#16 -- .conj().T keeps this complex-safe)."""
    Jm = np.asarray(J)
    S = 0.5 * (Jm + Jm.conj().T)
    return float(np.max(np.linalg.eigvalsh(S).real))


def directional_finite_time_gain_curve(J, b, horizons_ms) -> Dict[str, float]:
    """{horizon_ms: ||exp(J*T) b|| / ||b||} along direction b, one entry per horizon.

    REUSES topic4_m3b_spectral_phase.transient_gain: its matrix-free ||exp(M*T) b||/||b||
    (via expm_multiply) already IS this per-horizon directional gain (#15) -- not
    re-implemented here (§6.1)."""
    from src.topic4_m3b_spectral_phase import transient_gain
    return {str(int(T)): transient_gain(J, b, T) for T in horizons_ms}


def transient_amplification_present(curve, alpha1, gain_thresh: float = 1.5) -> bool:
    """True iff J is stable (alpha1 < 0) AND some horizon's directional gain exceeds
    gain_thresh. alpha1 >= 0 is modal growth, not a stable transient (#17)."""
    if alpha1 >= 0:
        return False
    return max(curve.values()) > gain_thresh


# --------------------------------------------------------------------------- #
# Trajectory verdict (Task 3a-5a) -- the 3-way PRE-REGISTERED classification.    #
# --------------------------------------------------------------------------- #
# spec §0/§1/§4: given per-point frozen-Jacobian read-outs along an approach
# trajectory, decide ONE of three PRE-REGISTERED verdicts -- smooth_CSD /
# hard_jump_no_CSD / unresolved_operating_point. The verdict is NEUTRAL: no
# outcome presupposes alpha1->0; "saturated"/"runaway" is a saturation LABEL, not
# an alpha1 reading. Pure logic over a list of point-dicts (no SNN here -- the SNN
# evaluator that PRODUCES the point-dicts is T3a-5b).

_CONTINUATION_OK = frozenset({                       # spec §4 / #2/#3
    "low_branch_disappears_before_alpha0",
    "low_branch_remains_far_from_alpha0_until_jump",
})


def _tau_ms_of(alpha1: float) -> float:
    """tau = -1/alpha1 for alpha1<0 (spec §1/§2); alpha1>=0 -> +inf (no finite decay time).
    Same tau_fast convention as adiabatic_index (line 359)."""
    return (-1.0 / alpha1) if alpha1 < 0 else float("inf")


def _saturated_transition_after(points, last_q_time, jump_window_ms):
    """The saturated point that FOLLOWS the last qualified point within jump_window_ms
    (spec §4 "sim enters saturated/runaway within jump_window"), else None. #18: a
    saturated point OUTSIDE the window does not count. The returned point is where the
    branch-continuation flags live (per the brief's hard/noc fixtures)."""
    for p in points:
        if p.get("saturated") is True:
            dt = p["time_ms"] - last_q_time
            if 0.0 <= dt <= jump_window_ms:
                return p
    return None


def _trajectory_verdict(points, q, cfg, alpha_near_zero_tol, alpha_margin_hard) -> str:
    """The pure 3-way decision for ONE (alpha_near_zero_tol, alpha_margin_hard) pair.
    Called once for the primary thresholds and once per threshold_sweep cell (#4). The
    #19/#20 gates do not depend on the two thresholds, so a gated trajectory returns
    unresolved for every sweep cell (making its stability visible)."""
    from scipy.stats import spearmanr

    gate = cfg["quality_gate"]
    vc = cfg["verdict"]
    n_total = len(points)

    # Clause #19 (count/fraction gate) -- denominator is ALL points, not just low-branch.
    if len(q) < gate["min_qualified_points"]:
        return "unresolved_operating_point"
    if n_total == 0 or (len(q) / n_total) < gate["min_qualified_fraction"]:
        return "unresolved_operating_point"

    last_q_time = q[-1]["time_ms"]
    jw = vc["jump_window_ms"]

    # Clause #20 (ambiguity) -- a qualified point on ambiguous_branch is NOT in q (see the q
    # filter in classify_trajectory), but any ambiguous_branch point within jump_window_ms of
    # the last qualified low-branch point means branch identity is not clean -> unresolved.
    for p in points:
        if p.get("branch_id") == "ambiguous_branch" and abs(p["time_ms"] - last_q_time) <= jw:
            return "unresolved_operating_point"

    alphas = [p["alpha1"] for p in q]
    max_alpha = max(alphas)                                                   # #3.1 closest-to-0
    last_alpha = alphas[-1]

    # smooth_CSD (spec §0): leading real-part eigenvalue smoothly approaches 0 along q --
    # (a) closest-to-0 within tol, (b) monotone rise (Spearman), (c) tau (=-1/alpha1) grows.
    tau_growth = _tau_ms_of(alphas[-1]) / _tau_ms_of(alphas[0])
    rho = spearmanr(alphas, list(range(len(alphas)))).correlation   # NaN if degenerate -> fails >=
    if (max_alpha >= -alpha_near_zero_tol
            and rho >= vc["smooth_min_alpha_spearman"]
            and tau_growth >= vc["smooth_min_tau_growth_ratio"]):
        return "smooth_CSD"

    # hard_jump_no_CSD (spec §4, clauses #2/#3): last qualified low-branch point still has a
    # clear margin from 0, NO alpha1 trend reached near-zero, sim saturates within the window,
    # AND branch continuation confirmed no skipped low-branch alpha1~=0 point. The two
    # continuation flags are read from the SATURATED/transition point (brief hard/noc fixtures).
    trans = _saturated_transition_after(points, last_q_time, jw)             # #18 window-gated
    if (last_alpha < -alpha_margin_hard
            and max_alpha < -alpha_near_zero_tol
            and trans is not None
            and trans.get("branch_continuation_checked") is True             # #2 absence/False -> unresolved
            and trans.get("continuation_status") in _CONTINUATION_OK):       # #3
        return "hard_jump_no_CSD"

    # spec §0/§4: operating point / branch identity / adiabatic conditions not cleanly met.
    return "unresolved_operating_point"


def classify_trajectory(points, cfg) -> dict:
    """3-way pre-registered verdict over an approach-trajectory of frozen-Jacobian point-dicts.

    Each point dict carries at least ``time_ms``, ``alpha1`` (continuous-time leading
    real-part eigenvalue, per-ms; None on saturated points), ``qualified`` (bool from
    qualify_point), ``branch_id`` ("low_branch"/"ambiguous_branch"/"saturated_branch"/...),
    and -- on the saturated/transition point of a hard jump -- ``branch_continuation_checked``
    + ``continuation_status`` (spec §4).

    Returns the primary ``verdict`` (spec §0/§4), the report fields (§1, names per #3.1),
    per-point ``tau_ms``/``instability_growth_time_ms``, and ``threshold_sensitivity`` (#4:
    the verdict re-run over cfg["verdict"]["threshold_sweep"]).
    """
    # Review (Important): callers are not guaranteed to hand in points pre-sorted by time_ms
    # (e.g. assembled from more than one source). Sort FIRST so every downstream read of the
    # ordered sequence -- q[-1]/last_q_time, the Spearman index order, tau-growth (first-q vs
    # last-q) -- is computed on a deterministic time order, not caller-dependent list order.
    # A no-op on an already-sorted or empty list.
    points = sorted(points, key=lambda p: p["time_ms"])

    vc = cfg["verdict"]

    # Clause q: qualified low-branch points only. A qualified point on ambiguous_branch is
    # excluded HERE (and separately triggers the #20 gate inside _trajectory_verdict).
    q = [p for p in points if p.get("qualified") is True and p.get("branch_id") == "low_branch"]

    # Primary verdict at the config-of-record thresholds.
    verdict = _trajectory_verdict(points, q, cfg,
                                  vc["alpha_near_zero_tol_per_ms"],
                                  vc["alpha_margin_hard_per_ms"])

    # Report fields (#3.1: alpha1_closest_to_zero_pre_onset is MAX, since for alpha1<0 the value
    # closest to 0 is the largest). None when q is empty (nothing to summarize).
    if q:
        alphas = [p["alpha1"] for p in q]
        alpha1_closest = max(alphas)
        last_stable = alphas[-1]
        jump_distance = abs(last_stable)
    else:
        alpha1_closest = last_stable = jump_distance = None

    # tau_ms (only where alpha1<0) / instability_growth_time_ms (only where alpha1>0), over every
    # point carrying a finite numeric alpha1 (§1). The alpha1>=0 (unqualified, unstable) points
    # get an instability growth time rather than a decay tau (progress.md T3a-5).
    tau_ms = []
    instability_growth_time_ms = []
    for p in points:
        a = p.get("alpha1")
        if a is None or not np.isfinite(a):
            continue
        if a < 0:
            tau_ms.append({"time_ms": p["time_ms"], "tau_ms": -1.0 / a})
        elif a > 0:
            instability_growth_time_ms.append(
                {"time_ms": p["time_ms"], "instability_growth_time_ms": 1.0 / a})

    # threshold_sensitivity (#4): re-run the verdict over the alpha_near_zero_tol x
    # alpha_margin_hard sweep grids so verdict stability across thresholds is visible.
    sweep = vc["threshold_sweep"]
    threshold_sensitivity = [
        {"alpha_near_zero_tol_per_ms": nz, "alpha_margin_hard_per_ms": mh,
         "verdict": _trajectory_verdict(points, q, cfg, nz, mh)}
        for nz in sweep["alpha_near_zero_tol_per_ms"]
        for mh in sweep["alpha_margin_hard_per_ms"]
    ]

    return {
        "verdict": verdict,
        "n_qualified_points": len(q),
        "qualified_fraction": (len(q) / len(points)) if points else 0.0,
        "alpha1_closest_to_zero_pre_onset": alpha1_closest,
        "last_stable_alpha1": last_stable,
        "jump_distance_to_alpha0": jump_distance,
        "tau_ms": tau_ms,
        "instability_growth_time_ms": instability_growth_time_ms,
        "threshold_sensitivity": threshold_sensitivity,
    }


# --------------------------------------------------------------------------- #
# Real 3-D trajectory evaluation (Task 3a-5b) -- the ACTUAL verdict SOURCE.      #
# --------------------------------------------------------------------------- #
# spec §2/§3/§4/§5: run the frozen-Jacobian read-out on the ACTUAL M3A-v2.2 slow
# trajectory (q_I(t) disinhibition + h_G(t) global recovery; g_K(t) fatigue only
# when the sim coupled it), NOT by sampling the 2-D atlas. For each SUBSAMPLED
# landmark: build the reduced operating point at that slow-state, run the branch
# protocol (warm-started), read the leading eigenvalue + eigen-metrics on the LOW
# branch, gate the point (spec §3), and finally classify_trajectory the sequence.
# The verdict carries verdict_source="actual_trajectory" (Hard-QC #1: NOT the atlas).

_N_TRAJECTORY_LANDMARKS: int = 48        # subsample of the ~10^4-step SNN trace (PERF, T2 review)
_CRIT_OP_GRID_N: int = 6                 # modest spatial grid for the op-solves (atlas-family)
_CRIT_CORE_RADIUS: float = 0.9           # single-core mask radius (atlas-consistent)


def _crit_op_context(cfg: Dict[str, Any]):
    """The reduced frozen-Jacobian op family shared by the trajectory eval AND the branch-
    continuation check, so both re-solve on a byte-identical (grid, kernels, core, b_core).
    mu_core=0.0 (atlas-consistent, build_conditional_atlas): the approach is driven by q_I
    disinhibition + h_G/g_K shifts, NOT by a core excitability bump."""
    import src.topic4_m3b_spectral_phase as spm
    grid = spm.Grid(n=_CRIT_OP_GRID_N, L=float(cfg["atlas"]["grid_L"]))
    kernels = spm.build_kernels(grid)
    core = spm.make_core_mask(grid, kind="single", radius=_CRIT_CORE_RADIUS)
    b_core = spm.core_perturbation_vector(grid, core)
    return grid, kernels, core, b_core


def _slow_inputs_at(sim, idx: int, mapping: Dict[str, Any], *, inject_gK: bool, inject_hG: bool):
    """Slow-state -> reduced-op knobs at trace index ``idx``.

    q_global = sheet-mean q_I; q_core = (min q_I / mean q_I) -- the CORE-vs-sheet gradient expressed
    as build_inhibition_field's core MULTIPLIER (q[core] = q_global*q_core). Both are clamped to the
    mapping's calibrated input domain [input_min, input_max] (the only use of ``mapping`` here -- the
    raw-value path needs no forward transform). g_K/h_G are injected only when the sim dynamically
    coupled them (use_gK is False by default in the v2.2 config, so g_K's fatigue trace accumulates
    but did NOT feed back -> not injected, keeping the reduced op faithful to the actual trajectory)."""
    import numpy as np
    dom = mapping["coordinates"]["phase_y_global"]["transform"]
    qlo, qhi = float(dom["input_min"]), float(dom["input_max"])
    q_mean = float(sim["trace_qI_mean"][idx])
    q_min = float(sim["trace_qI_min"][idx])
    q_global = float(np.clip(q_mean, qlo, qhi))
    q_core = float(np.clip(q_min / max(q_mean, 1e-9), qlo, qhi))
    gK_value = float(np.clip(float(sim["trace_gK_axial"][idx]), 0.0, 1.0)) if inject_gK else None
    hG_scalar = float(np.clip(float(sim["trace_hG"][idx]), 0.0, 1.0)) if inject_hG else 0.0
    return {"q_global": q_global, "q_core": q_core, "gK_value": gK_value, "hG_scalar": hG_scalar}


def _fields_from_slow(grid, core, slow_inputs: Dict[str, Any], cfg: Dict[str, Any]):
    """(exc, inh, gK_field, hG_scalar, eta_K, eta_G) for solve_branches from a slow-input dict."""
    import numpy as np
    import src.topic4_m3b_spectral_phase as spm
    exc = spm.build_excitability_field(grid, core, mu_core=0.0)
    inh = spm.build_inhibition_field(grid, core, q_global=float(slow_inputs["q_global"]),
                                     q_core=float(slow_inputs["q_core"]))
    stf = cfg["slow_to_ratefield"]
    eta_K = float(stf["g_K"]["eta_K"])
    eta_G = float(stf["h_G"]["eta_G"])
    gk = slow_inputs.get("gK_value")
    gK_field = np.full((grid.n, grid.n), float(gk)) if gk is not None else None
    hG_scalar = float(slow_inputs.get("hG_scalar") or 0.0)
    return exc, inh, gK_field, hG_scalar, eta_K, eta_G


def _low_branch_at(grid, kernels, core, slow_inputs, cfg, *, prev, seed_key):
    """solve_branches at one slow-state -> (low_Branch or None, branches, sat_any, dominant_reason).

    ``dominant_reason`` labels a point with NO low branch: saturated_branch if any branch saturated
    (a runaway/fold -- so classify_trajectory can see the jump), else ambiguous_branch if any branch
    is ambiguous (triggers the #20 gate), else the highest-rate branch's own reason."""
    exc, inh, gK_field, hG_scalar, eta_K, eta_G = _fields_from_slow(grid, core, slow_inputs, cfg)
    branches = solve_branches(grid, kernels, exc, inh, cfg, prev=prev, seed_key=seed_key,
                              gK_field=gK_field, hG_scalar=hG_scalar, eta_K=eta_K, eta_G=eta_G)
    low = next((b for b in branches if b.branch_selected_reason == "low_branch"), None)
    sat_any = any(b.op.saturated for b in branches)
    if sat_any:
        dominant = "saturated_branch"
    elif any(b.branch_selected_reason == "ambiguous_branch" for b in branches):
        dominant = "ambiguous_branch"
    elif branches:
        dominant = max(branches, key=lambda b: b.branch_rate_mean).branch_selected_reason
    else:
        dominant = "saturated_branch"
    return low, branches, sat_any, dominant


def _neighbors(arr, i):
    """(left, right) immediate indices (i-1 / i+1) with FINITE arr, else None on that side."""
    n = len(arr)
    left = i - 1 if (i - 1 >= 0 and np.isfinite(arr[i - 1])) else None
    right = i + 1 if (i + 1 < n and np.isfinite(arr[i + 1])) else None
    return left, right


def _scalar_central_diff(arr, t, i):
    """Central finite difference d(arr)/d(t) at i using immediate finite neighbors; one-sided at an
    end or across a None/NaN neighbor; None if neither side is usable (isolated finite point)."""
    left, right = _neighbors(arr, i)
    if left is not None and right is not None:
        return float((arr[right] - arr[left]) / (t[right] - t[left]))
    if right is not None and np.isfinite(arr[i]):
        return float((arr[right] - arr[i]) / (t[right] - t[i]))
    if left is not None and np.isfinite(arr[i]):
        return float((arr[i] - arr[left]) / (t[i] - t[left]))
    return None


def _vec_central_diff(mat, t, i):
    """Row-wise central difference of a (L, d) slow-state matrix wrt t at i (the slow velocity). Every
    column is finite at every landmark (slow vars exist regardless of op resolution), so always defined."""
    n = mat.shape[0]
    if n == 1:
        return np.zeros(mat.shape[1])
    if i == 0:
        return (mat[1] - mat[0]) / (t[1] - t[0])
    if i == n - 1:
        return (mat[i] - mat[i - 1]) / (t[i] - t[i - 1])
    return (mat[i + 1] - mat[i - 1]) / (t[i + 1] - t[i - 1])


def evaluate_actual_trajectory_points(sim, mapping, cfg) -> list:
    """Frozen-Jacobian read-out on the ACTUAL M3A-v2.2 slow trajectory (#1 -- the verdict SOURCE).

    Runs the branch-aware operating-point protocol + leading-eigenvalue read-out at ~``
    _N_TRAJECTORY_LANDMARKS`` subsampled landmarks of ``sim`` (from ``run_transition``), returning one
    JSON-serializable point dict per landmark. Each carries at least ``time_ms``, ``alpha1``
    (continuous-time leading real-part eigenvalue, per-ms; None on saturated / unresolved points),
    ``qualified`` (Python bool), ``branch_id`` + the eigen-metrics (``alpha_gap``,
    ``left_mode_input_projection``, ``numerical_abscissa``, ``directional_gain``, ``mode_class``) and
    the quality-gate fields (``rate_mismatch_*``, ``slow_mismatch_rel``, ``adiabatic_index``,
    ``alpha_drift_index``). Consumed by ``classify_trajectory`` (via ``build_trajectory_verdict``).

    Contract highlights (T3a-5b brief + T3a-5a review carry-forward):
    * warm start -- each landmark's branch solve is seeded from the PREVIOUS landmark's low-branch op.
    * F3 EMPTY-GUARD -- on an unresolved / empty spectrum the eigen-metric fns are NEVER called
      (they index [0] / argmax an empty array); the point is marked unqualified ("eig_unresolved").
    * finite-alpha1 invariant -- a ``qualified`` point ALWAYS carries a finite ``alpha1`` (asserted).
    * Python bool -- ``qualified`` is a built-in bool, not np.bool_ (classify_trajectory identity check).
    * alpha1>=0 (unstable) points keep their finite positive alpha1 but get adiabatic_index=+inf ->
      qualify_point rejects them (correct: modal growth, excluded from the CSD trend).
    """
    import numpy as np
    import src.topic4_m3b_spectral_phase as spm

    grid, kernels, core, b_core = _crit_op_context(cfg)
    inject_gK = bool(sim.get("use_gK", False))
    inject_hG = bool(sim.get("use_hG", True))
    floor = float(cfg["quality_gate"]["rate_scale_floor"])
    min_sep = float(cfg["mode"]["next_distinct_min_sep_per_ms"])
    imag_tol = float(cfg["mode"]["imag_tol_per_ms"])
    horizons = cfg["finite_time_gain"]["horizons_ms"]

    times = np.asarray(sim["times"], float)
    rate_E = np.asarray(sim["rate_E"], float)
    nsteps = times.size
    land = np.unique(np.linspace(0, nsteps - 1, _N_TRAJECTORY_LANDMARKS).astype(int))

    # --- Pass 1: per-landmark op-solve + low-branch eigen read-out (warm-started) ---
    recs: list = []
    prev_low_op = None
    for i, idx in enumerate(land):
        idx = int(idx)
        slow_inputs = _slow_inputs_at(sim, idx, mapping, inject_gK=inject_gK, inject_hG=inject_hG)
        low, branches, sat_any, dominant = _low_branch_at(
            grid, kernels, core, slow_inputs, cfg, prev=prev_low_op, seed_key=(1000 + i,))
        rec = {
            "time_ms": float(times[idx]), "slow_inputs": slow_inputs,
            "q_global": slow_inputs["q_global"], "q_core": slow_inputs["q_core"],
            "n_branches_found": int(len(branches)),
            "snn_rate_kHz": float(rate_E[idx]) / 1000.0,
            "_alpha1": float("nan"),
            "_slow_vec": np.array([slow_inputs["q_global"], slow_inputs["q_core"],
                                   slow_inputs["hG_scalar"], (slow_inputs["gK_value"] or 0.0)], float),
        }
        if low is None:
            # No low branch (fold / all-saturated): unqualified. Carry the saturation LABEL so
            # classify_trajectory's _saturated_transition_after can see the jump.
            rec.update({"alpha1": None, "qualified": False, "branch_id": dominant,
                        "saturated": bool(sat_any), "reason": "no_low_branch",
                        "op_rate_kHz": None, "residual_rms": None, "converged": False,
                        "rate_mismatch_abs": None, "rate_mismatch_rel": None, "slow_mismatch_rel": None,
                        "alpha_gap": None, "left_mode_input_projection": None,
                        "numerical_abscissa": None, "directional_gain": None,
                        "directional_gain_peak": None, "freq_hz": None,
                        "mode_class": ("runaway" if sat_any else "unresolved")})
            recs.append(rec)
            continue

        op = low.op
        prev_low_op = op                                                  # warm-start next landmark
        J = spm.build_jacobian_dense(grid, kernels, op)
        res = spm.rate_eigenpairs(J, grid)
        op_rate_kHz = float(op.rE.mean())
        # rate_mismatch (spec §3): rate_sim = SNN population rate (kHz); z_star = reduced-op fixed
        # point rate (kHz). Both in the op's native kHz (rate_scale_floor is kHz); SNN rate_E is Hz.
        rm_abs, rm_rel = rate_mismatch(np.array([rec["snn_rate_kHz"]]), np.array([op_rate_kHz]), floor)
        rec.update({"branch_id": "low_branch", "saturated": bool(op.saturated),
                    "op_rate_kHz": op_rate_kHz, "residual_rms": float(op.residual),
                    "converged": bool(op.converged), "rate_mismatch_abs": float(rm_abs),
                    "rate_mismatch_rel": float(rm_rel),
                    # op solved AT the exact sim slow-state -> no slow re-derivation mismatch (spec §3).
                    "slow_mismatch_rel": 0.0})

        # F3 EMPTY-GUARD (T3a-3 review, matches analyze_spectral_point:1069-1070): never call the
        # eigen-metric fns on an unresolved / empty spectrum -- they crash on an empty array.
        if res.status != "resolved" or res.eigenvalues.size == 0:
            rec.update({"alpha1": None, "qualified": False, "reason": "eig_unresolved",
                        "alpha_gap": None, "left_mode_input_projection": None,
                        "numerical_abscissa": None, "directional_gain": None,
                        "directional_gain_peak": None, "freq_hz": None, "mode_class": "unresolved"})
            recs.append(rec)
            continue

        alpha1 = float(res.eigenvalues[0].real)
        rec["_alpha1"] = alpha1
        # leading invariant SUBSPACE (a complex conjugate pair, or a near-degenerate real group).
        # Read the mode SHAPE off the NON-NEGATIVE subspace loading (pair_loading), NOT a single
        # signed eigenvector -- otherwise a leading complex pair's shape flips with the arbitrary
        # eigenvector sign/phase (spec §6: mode class on the invariant-subspace energy, sign-free).
        idxs = spm.leading_subspace_indices(res.eigenvalues, min_sep=min_sep, imag_tol=imag_tol)
        loading = spm.pair_loading(res.right, idxs, grid)               # (n,n) non-negative subspace E-loading
        mode_core_overlap = float(spm.core_overlap(loading, grid, core))
        mode_globality = float(spm.globality(loading, grid))
        gain_curve = directional_finite_time_gain_curve(J, b_core, horizons)
        ftg20 = spm.transient_gain(J, b_core, 20.0)                      # atlas-consistent mode-class window
        mode_class = spm.classify_mode(
            growth=alpha1, core_overlap_=mode_core_overlap, globality_=mode_globality,
            elongation_axis=spm.elongation_axis_score(loading, grid, kernels.theta),
            off_axis=spm.off_axis_score(loading, grid, kernels.theta),
            finite_time_gain_=float(ftg20), saturated=bool(op.saturated))
        rec.update({
            "alpha1": alpha1,
            "alpha_gap": float(spm.next_distinct_gap(res.eigenvalues, min_sep)),
            "left_mode_input_projection": float(
                spm.left_mode_input_projection(res.left, res.right, idxs, b_core)),
            "numerical_abscissa": float(numerical_abscissa(J)),
            "directional_gain": {k: float(v) for k, v in gain_curve.items()},
            "directional_gain_peak": float(max(gain_curve.values())),
            "freq_hz": float(spm.mode_frequency_hz(res.eigenvalues[0])),
            "mode_core_overlap": mode_core_overlap, "mode_globality": mode_globality,
            "leading_subspace_dim": int(len(idxs)),
            "mode_class": str(mode_class),
        })
        recs.append(rec)

    # --- Pass 2: finite-difference gate fields (slow_speed / adiabatic / alpha-drift) + qualify ---
    t_arr = np.array([r["time_ms"] for r in recs], float)
    a_arr = np.array([r["_alpha1"] for r in recs], float)               # NaN where alpha1 is None
    slow_mat = np.array([r["_slow_vec"] for r in recs], float)          # (L, 4), always finite
    points: list = []
    for i, r in enumerate(recs):
        r["slow_speed"] = float(np.linalg.norm(_vec_central_diff(slow_mat, t_arr, i)))
        if r["alpha1"] is None:
            # already unqualified upstream (no_low_branch / eig_unresolved) -- no gate re-run.
            r.update({"adiabatic_index": None, "alpha_drift_index": None})
        else:
            slow_scale = max(float(np.linalg.norm(slow_mat[i])), 1e-6)
            adiab = adiabatic_index(r["slow_speed"], r["alpha1"], slow_scale)     # +inf when alpha1>=0
            dadt = _scalar_central_diff(a_arr, t_arr, i)
            adrift = (abs(dadt) / (r["alpha1"] ** 2 + 1e-9)) if dadt is not None else 0.0
            r["adiabatic_index"] = float(adiab) if np.isfinite(adiab) else float("inf")
            r["alpha_drift_index"] = float(adrift)
            fields = {"converged": r["converged"], "saturated": r["saturated"],
                      "residual_rms": r["residual_rms"], "rate_mismatch_abs": r["rate_mismatch_abs"],
                      "rate_mismatch_rel": r["rate_mismatch_rel"],
                      "slow_mismatch_rel": r["slow_mismatch_rel"],
                      "adiabatic_index": r["adiabatic_index"], "alpha_drift_index": r["alpha_drift_index"]}
            ok, reason = qualify_point(fields, cfg)
            # INVARIANT (T3a-5a review): a qualified point MUST carry a finite alpha1 (classify_
            # trajectory's max(alphas) needs it). Structurally true on this branch; assert defensively.
            if ok and not np.isfinite(r["alpha1"]):
                ok, reason = False, "nonfinite_alpha1"
            r["qualified"] = bool(ok)                                    # Python bool (identity check)
            r["reason"] = reason
        r.pop("_alpha1", None)
        r.pop("_slow_vec", None)
        points.append(r)
    return points


def check_low_branch_continuation_between(pt_a, pt_b, cfg) -> dict:
    """Branch-continuation bisection between the last-qualified low-branch point ``pt_a`` and the
    first saturated/transition point ``pt_b`` (spec §4, clauses #2/#3).

    Interpolates the slow state across ``cfg["verdict"]["branch_continuation_n_bisect"]`` midpoints,
    re-solves the low branch (warm-started from pt_a's low branch), and reports whether the low branch
    either (a) DISAPPEARS before its alpha1 reaches near-zero (fold), or (b) REMAINS far from alpha1=0
    all the way to the jump -- both confirm no near-critical low-branch state was skipped, so
    ``hard_jump_no_CSD`` is admissible. If some interpolated low branch DOES reach near-zero, that is a
    skipped alpha1~=0 point: the status is NOT in ``_CONTINUATION_OK`` and classify_trajectory falls to
    ``unresolved`` (never a false hard jump).

    Returns ``{branch_continuation_checked: True (Python bool -- classify_trajectory checks `is True`,
    and np.bool_(True) is True -> False), continuation_status, n_bisect, bisection_max_low_alpha1}``.
    """
    import src.topic4_m3b_spectral_phase as spm

    grid, kernels, core, _b = _crit_op_context(cfg)
    n_bisect = int(cfg["verdict"]["branch_continuation_n_bisect"])
    near_zero_tol = float(cfg["verdict"]["alpha_near_zero_tol_per_ms"])
    a = pt_a["slow_inputs"]
    b = pt_b["slow_inputs"]

    # warm-start from pt_a's own low branch (re-solved from its stored slow-inputs).
    low0, _br, _sa, _dm = _low_branch_at(grid, kernels, core, a, cfg, prev=None, seed_key=(8000,))
    prev = low0.op if low0 is not None else None

    disappeared = False
    reached_zero = False
    max_low_alpha = None
    for k in range(1, n_bisect + 1):
        frac = k / (n_bisect + 1)
        s = {
            "q_global": (1 - frac) * a["q_global"] + frac * b["q_global"],
            "q_core": (1 - frac) * a["q_core"] + frac * b["q_core"],
            "hG_scalar": (1 - frac) * (a.get("hG_scalar") or 0.0) + frac * (b.get("hG_scalar") or 0.0),
            "gK_value": (None if (a.get("gK_value") is None or b.get("gK_value") is None)
                         else (1 - frac) * a["gK_value"] + frac * b["gK_value"]),
        }
        low, _br, _sat, _dm = _low_branch_at(grid, kernels, core, s, cfg, prev=prev, seed_key=(8100 + k,))
        if low is None:
            disappeared = True                                          # low branch gone before alpha0
            break
        prev = low.op
        res = spm.rate_eigenpairs(spm.build_jacobian_dense(grid, kernels, low.op), grid)
        if res.status == "resolved" and res.eigenvalues.size > 0:
            a1 = float(res.eigenvalues[0].real)
            max_low_alpha = a1 if max_low_alpha is None else max(max_low_alpha, a1)
            if a1 >= -near_zero_tol:
                reached_zero = True                                     # a skipped low-branch alpha0 point
                break

    if reached_zero:
        status = "low_branch_reaches_alpha0_before_jump"                # NOT in _CONTINUATION_OK
    elif disappeared:
        status = "low_branch_disappears_before_alpha0"
    else:
        status = "low_branch_remains_far_from_alpha0_until_jump"
    return {"branch_continuation_checked": True, "continuation_status": status,
            "n_bisect": n_bisect, "bisection_max_low_alpha1": max_low_alpha}


def build_trajectory_verdict(sim, mapping, cfg) -> tuple:
    """Orchestrate the T3a-5b verdict (spec §2/§4): evaluate the real trajectory, run the
    branch-continuation check across the last-qualified -> first-saturated jump (attaching its flags
    on the transition point where classify_trajectory reads them), classify_trajectory the sequence,
    and assemble the ``trajectory_verdict.json`` payload.

    verdict_source="actual_trajectory" (Hard-QC #1: NOT the 2-D atlas). operator_type=
    continuous_jacobian, alpha_units=per_ms (spec §2 unit lock). Returns (payload, points)."""
    import numpy as np

    points = evaluate_actual_trajectory_points(sim, mapping, cfg)

    # last qualified low-branch point + first saturated point AFTER it -> continuation check;
    # attach the flags on the SATURATED transition point (classify_trajectory reads them there).
    q = [p for p in points if p.get("qualified") is True and p.get("branch_id") == "low_branch"]
    continuation = None
    if q:
        last_q = q[-1]
        trans = next((p for p in points if p.get("saturated") is True
                      and p["time_ms"] > last_q["time_ms"]), None)
        if trans is not None:
            continuation = check_low_branch_continuation_between(last_q, trans, cfg)
            trans.update(continuation)

    verdict = classify_trajectory(points, cfg)

    # per-point leading alpha1 series (per-ms) over finite-alpha1 points (Hard-QC #2 alpha1_per_ms).
    alpha1_per_ms = [{"time_ms": p["time_ms"], "alpha1_per_ms": p["alpha1"]}
                     for p in points if p.get("alpha1") is not None]

    payload = dict(verdict)
    payload.update({
        "verdict_source": "actual_trajectory",                          # Hard-QC #1 guard (NOT atlas)
        "operator_type": cfg["operator"]["type"],                       # continuous_jacobian (spec §2)
        "alpha_units": cfg["operator"]["alpha_units"],                  # per_ms
        "alpha1_per_ms": alpha1_per_ms,
        "operator_gain_computed": False,                                # Hard-QC #8 (directional, not ||exp(JT)||2)
        "finite_time_gain_kind": cfg["finite_time_gain"]["mode"],       # directional_core -> 'directional_gain' (#10)
        "tier": cfg.get("tier"),
        "branch_continuation": continuation,
        "provenance": {
            "mapping_id": mapping.get("slow_to_rate_mapping_id"),
            "n_landmarks": len(points),
            "n_qualified": verdict["n_qualified_points"],
            "grid_n": _CRIT_OP_GRID_N, "grid_L": float(cfg["atlas"]["grid_L"]),
            "core_radius": _CRIT_CORE_RADIUS, "mu_core": 0.0,
            "slow_vars_injected": (["q_I"]
                                   + (["g_K"] if bool(sim.get("use_gK", False)) else [])
                                   + (["h_G"] if bool(sim.get("use_hG", True)) else [])),
            "dt_ms": (float(sim["dt_ms"]) if sim.get("dt_ms") is not None else None),
            "n_sim_steps": int(np.asarray(sim["times"]).size),
        },
        "points": points,
    })
    return payload, points
