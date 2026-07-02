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
