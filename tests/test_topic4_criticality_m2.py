"""TDD tests for Topic 4 M3-v2.2 criticality Milestone 2 (Task 0, plan Step 2).

Contract (verbatim from task brief .superpowers/sdd/task-0-brief.md Step 2):
  - shape scores behave sanely on synthetic anisotropic blobs (port of pilot4).
  - basis_vectors returns unit-norm, mutually orthogonal e_global / e_axis_gradient.
  - load_m2_config resolves the "THETA_EE" sentinel to a float.
"""
import json

import numpy as np

from src.topic4_criticality import load_crit_config, _crit_op_context
import src.topic4_criticality_m2 as m2


def _ctx():
    grid, kernels, core, b_core = _crit_op_context(load_crit_config())
    return grid, kernels, core, b_core


def _gauss(grid, theta, sig_par, sig_perp, ang):
    X, Y = grid.coords(); cx, cy = X.mean(), Y.mean()
    u = (X - cx) * np.cos(ang) + (Y - cy) * np.sin(ang)
    w = -(X - cx) * np.sin(ang) + (Y - cy) * np.cos(ang)
    return np.exp(-0.5 * ((u / sig_par) ** 2 + (w / sig_perp) ** 2))


def test_shape_scores_sanity_on_synthetic_blobs():
    grid, kernels, core, _ = _ctx(); th = kernels.theta
    import src.topic4_m3b_spectral_phase as spm
    along = _gauss(grid, th, 1.6, 0.5, th)
    perp = _gauss(grid, th, 1.6, 0.5, th + np.pi / 2)
    uniform = np.ones_like(along)
    assert spm.elongation_axis_score(along, grid, th) > 0.3
    assert spm.elongation_axis_score(along, grid, th) > spm.off_axis_score(along, grid, th)
    assert spm.off_axis_score(perp, grid, th) > 0.3 and spm.elongation_axis_score(perp, grid, th) < 0
    assert spm.globality(uniform, grid) > 0.9 and abs(spm.elongation_axis_score(uniform, grid, th)) < 0.05


def test_basis_vectors_unit_norm_and_orthogonal():
    grid, kernels, _, _ = _ctx()
    b = m2.basis_vectors(grid, kernels.theta)
    assert abs(np.linalg.norm(b["e_global"]) - 1.0) < 1e-9
    assert abs(np.linalg.norm(b["e_axis_gradient"]) - 1.0) < 1e-9
    assert abs(float(b["e_global"] @ b["e_axis_gradient"])) < 1e-9   # axis grad is zero-mean


def test_load_m2_config_resolves_theta():
    cfg = m2.load_m2_config()
    assert cfg["basis"]["off_axis_score_tol"] == 0.05
    assert isinstance(cfg["basis"]["theta"], float)      # "THETA_EE" -> np.pi/4


# --- Task 1: dense alpha0 localization (bracket -> coarse scan -> bisect), verbatim from
# task brief .superpowers/sdd/task-1-brief.md Step 1 ---
def _points():
    p = m2._REPO / "results/topic4_criticality/trajectory_verdict.json"   # M1 deliverable
    return json.loads(p.read_text())["points"]


def test_localize_alpha0_crossing_brackets_zero():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    out = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    assert out["crossing_status"] in ("single", "multiple_alpha0_crossings")
    assert out["alpha_left"] < 0.0                       # last neg before crossing
    assert out["crossing_frac"] is not None
    assert 470.0 < out["alpha0_crossing_time_ms"] < 520.0  # M1 idx14->idx15 window


# --- T1 review FIX 1 (Critical): op_solve_quality is a FOLD-APPROPRIATE residual bar, not the
# solver's strict 1e-9 converged flag. Near-fold ops (residual ~1e-3-4e-3) never hit converged=True
# yet read a stable spectrum, so both sides must still be quality=True. ---
def test_op_solve_quality_is_residual_based_not_strict_converged():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    out = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    assert out["op_solve_quality_left"] is True
    assert out["op_solve_quality_right"] is True
    op = out["_crossing_op"]
    assert op is not None
    # documents the rule is residual-based: the crossing op is NOT strictly converged (>1e-9) yet
    # sits within the fold-appropriate tolerance -> quality=True despite converged=False.
    assert float(op.residual) > 1e-9
    assert float(op.residual) <= m2cfg["densification"]["op_residual_tol"]
    assert op.converged is False


# --- T1 review FIX 2 (spec §3.1 output gap): branch_identity_clean must be present in the return
# (feeds the §5.0 ignition base gate) and True on the real crossing, where M1's continuation check
# reports the low branch stays continuous and reaches alpha0 (no branch jump/fold). ---
def test_branch_identity_clean_present_and_true_on_real_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    out = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    assert "branch_identity_clean" in out
    assert out["branch_identity_clean"] is True
    assert out["_branch_continuation_status"] == "low_branch_reaches_alpha0_before_jump"


# --- Task 2: linear_ignition readout + two-core symmetry-break confirmation, verbatim from
# task brief .superpowers/sdd/task-2-brief.md Step 1 ---
def test_linear_ignition_core_localized_on_real_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config(); pts = _points()
    crossing = m2.localize_alpha0_crossing(pts, grid, kernels, core, cfg, m2cfg)
    ig = m2.read_linear_ignition(crossing, grid, kernels, core, cfg, m2cfg, pts)
    assert ig["class"] == "core_localized"
    assert ig["core_overlap"] >= m2cfg["ignition"]["core_localized_overlap_thresh"]
    assert ig["globality"] <= m2cfg["ignition"]["core_localized_globality_thresh"]
    assert ig["two_core_symmetry_break"] is True
    assert ig["corridor_power"] <= m2cfg["two_core_confirm"]["corridor_dark_thresh"]
    assert "post-fold" in ig["near_fold_note"] and "symmetric" in ig["near_fold_note"]


def test_ignition_class_delocalized_on_global_loading():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    cls, sub = m2._classify_ignition(core_overlap=0.2, globality=0.8,
                                     axis_elongation=0.0, off_axis=0.0,
                                     corridor_power=0.0, n_core_peaks=1, m2cfg=m2cfg)
    assert cls == "delocalized" and sub == "global_like"


# --- Task 3: projected gain/leak + nonaxis off_axis sentinel, verbatim from task brief
# .superpowers/sdd/task-3-brief.md Step 1 ---
def test_off_axis_sentinel_absent_on_core_localized_crossing():
    cfg = load_crit_config(); grid, kernels, core, _ = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    crossing = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    s = m2.off_axis_sentinel(crossing, grid, kernels, core, m2cfg)
    assert s["off_axis"] == "absent"
    assert "core-compactness residual" in s["annotation"]
    # rev2.3: verdict comes from asymptotic-tail agreement, self-documented for audit. On the real
    # crossing BOTH tail horizons read absent, so the verdict is a robust (not single-horizon) absent.
    assert s["sentinel_tail_horizons_ms"] == [250, 500]
    assert s["off_axis_per_tail_decision"] == ["absent", "absent"]


def test_off_axis_present_requires_both_gates():
    m2cfg = m2.load_m2_config()
    # score gate open but gain gate closed -> NOT present
    v = m2._off_axis_decision(off_axis_score=0.09, gain_nonaxis=0.10,
                              gain_axis=0.20, gain_global=0.05, m2cfg=m2cfg)
    assert v == "undetermined"
    v2 = m2._off_axis_decision(off_axis_score=0.09, gain_nonaxis=0.40,
                               gain_axis=0.10, gain_global=0.05, m2cfg=m2cfg)
    assert v2 == "present"
    v3 = m2._off_axis_decision(off_axis_score=0.01, gain_nonaxis=0.01,
                               gain_axis=0.20, gain_global=0.05, m2cfg=m2cfg)
    assert v3 == "absent"


# --- Task 3 rev2.3: asymptotic-tail agreement rule. When the two tail horizons yield DIFFERENT
# per-horizon decisions the sentinel must return "undetermined" (kills the single-horizon fragility).
def test_off_axis_tail_disagreement_yields_undetermined():
    m2cfg = m2.load_m2_config()
    # off_axis_score is horizon-independent, so the score gate is constant across the tail; only the
    # gain gate varies. Here score gate is CLOSED (0.01 < 0.05 tol) at both tail horizons, but the
    # gain gate is closed at 250 (nonaxis 0.01 << axis 0.20 -> absent) and open at 500 (nonaxis 0.40,
    # excess 0.30 >= 0.10 tol, ratio 4.0 >= 1.25 tol -> undetermined, since score gate stays closed).
    gains = {
        "e_axis_gradient": {250: 0.20, 500: 0.10},
        "e_global": {250: 0.05, 500: 0.05},
        "e_nonaxis": {250: 0.01, 500: 0.40},
    }
    verdict, tail, per_tail = m2._off_axis_tail_agreement(gains, off_axis_score=0.01, m2cfg=m2cfg)
    assert tail == [250, 500]
    assert per_tail == ["absent", "undetermined"]   # tail horizons disagree
    assert verdict == "undetermined"


# --- Task 4: field_rhs shift fix + JVP hard gate (tests/test_topic4_m3b_spectral_phase.py) +
# nonlinear footprint spread, verbatim from task brief .superpowers/sdd/task-4-brief.md Steps 1/5 ---

# Fast synthetic unit tests (no real-crossing solve) for the epsilon-sensitivity aggregation rule +
# the standalone off_axis sentinel -- the brief gives `_spread_onset`/`_spread_endgame` verbatim but
# NOT the "all agree"/"majority" aggregation or the full-trajectory off_axis verdict; these lock the
# judgment calls this task had to make (mirrors Task 3's own precedent of unit-testing
# `_off_axis_decision`/`_off_axis_tail_agreement` directly with synthetic data).
def test_all_agree_and_majority_aggregation_rules():
    assert m2._all_agree(["axial", "axial", "axial", "axial"]) == "axial"
    assert m2._all_agree(["axial", "axial", "core_only", "axial"]) is None   # not unanimous
    assert m2._majority(["self_limited", "self_limited", "self_limited", "global_flooding"], 3) == "self_limited"
    assert m2._majority(["self_limited", "self_limited", "global_flooding", "global_flooding"], 3) is None  # 2-2 split


def test_spread_onset_and_endgame_on_synthetic_trajectory():
    m2cfg = m2.load_m2_config()
    # rises then falls back near baseline, elongated along axis, off_axis stays 0 throughout
    # -> axial onset, self_limited endgame, off_axis sentinel absent.
    traj = [
        {"active_frac": 0.0, "elongation_axis": 0.0, "off_axis": 0.0, "globality": 0.05},
        {"active_frac": 0.1, "elongation_axis": 0.1, "off_axis": 0.0, "globality": 0.05},
        {"active_frac": 0.5, "elongation_axis": 0.4, "off_axis": 0.0, "globality": 0.2},
        {"active_frac": 0.05, "elongation_axis": 0.5, "off_axis": 0.0, "globality": 0.05},
    ]
    assert m2._spread_onset(traj, m2cfg) == "axial"
    assert m2._spread_endgame(traj, None, m2cfg) == "self_limited"
    assert m2._spread_off_axis(traj, "axial", m2cfg) == "absent"

    # active_frac never rises above its initial value -> core_only (no expansion at all).
    flat = [dict(fm, active_frac=0.02) for fm in traj]
    assert m2._spread_onset(flat, m2cfg) == "core_only"

    # active_frac floods to near-1.0 at the end -> global_flooding (checked before self-limit).
    flood = [dict(fm) for fm in traj]; flood[-1]["active_frac"] = 0.95
    assert m2._spread_endgame(flood, None, m2cfg) == "global_flooding"

    # off_axis sustained through the expansion window (not just a single-step blip) -> onset reads
    # "off_axis" (sentinel breaks) and the full-trajectory verdict reads "present".
    offax = [dict(fm, off_axis=0.2) for fm in traj]
    onset_off = m2._spread_onset(offax, m2cfg)
    assert onset_off == "off_axis"
    assert m2._spread_off_axis(offax, onset_off, m2cfg) == "present"

    # a single-step off_axis blip that does NOT survive into the expansion-window mean -> onset
    # stays "axial" (the mean is still < tol), but the full-trajectory PEAK gate still breaks ->
    # "undetermined" (peak gate fired, expansion-mean gate did not -- never "present" on one gate).
    blip = [dict(fm) for fm in traj]; blip[0]["off_axis"] = 0.2   # only the FIRST (pre-expansion) sample
    onset_blip = m2._spread_onset(blip, m2cfg)
    assert onset_blip == "axial"
    assert m2._spread_off_axis(blip, onset_blip, m2cfg) == "undetermined"


# --- Task 4 rev2.4 (review decision C): descriptive-only igniting-subset note. Fast synthetic unit
# test (no 90s solve) for `_descriptive_igniting_note` — emit case + all three None branches + the
# non-unanimous igniting_onset distribution branch. The REAL-crossing assertion (note present on the
# actual v2.2 crossing + primary verdict unchanged) lives in test_nonlinear_spread_* below, reusing
# that test's already-computed `sp` rather than paying the 90s solve twice. ---
_IGNITING_CAVEAT = ("DESCRIPTIVE ONLY — primary nonlinear_spread verdict is undetermined "
                    "(pre-registered §4.3); NOT a spread claim")


def test_descriptive_igniting_note_emit_and_none_cases():
    # 3/4 ignite (axial), 1/4 non-igniting (core_only, suppressing pol=-1) at BOTH depths — the exact
    # shape of the real-crossing breakdown -> emit; igniting_onset unanimous "axial",
    # igniting_endgame "self_limited", one non-igniting (0.05, -1) combo.
    detail = {
        "at_crossing": [
            {"eps_rel": 0.01, "polarity": -1, "onset": "axial", "endgame": "self_limited"},
            {"eps_rel": 0.01, "polarity": 1, "onset": "axial", "endgame": "self_limited"},
            {"eps_rel": 0.05, "polarity": -1, "onset": "core_only", "endgame": "self_limited"},
            {"eps_rel": 0.05, "polarity": 1, "onset": "axial", "endgame": "self_limited"},
        ],
        "just_past": [
            {"eps_rel": 0.01, "polarity": -1, "onset": "axial", "endgame": "self_limited"},
            {"eps_rel": 0.01, "polarity": 1, "onset": "axial", "endgame": "self_limited"},
            {"eps_rel": 0.05, "polarity": -1, "onset": "core_only", "endgame": "self_limited"},
            {"eps_rel": 0.05, "polarity": 1, "onset": "axial", "endgame": "global_flooding"},
        ],
    }
    note = m2._descriptive_igniting_note(detail, "epsilon_sensitive")
    assert note is not None
    assert note["n_igniting_of_total"] == {"at_crossing": "3/4", "just_past": "3/4"}
    assert note["igniting_onset"] == "axial"
    assert note["igniting_endgame"] == "self_limited"
    assert note["non_igniting_combos"] == [
        {"eps_rel": 0.05, "polarity": -1, "reason": "suppressing kick, active_frac did not rise"}]
    assert note["caveat"] == _IGNITING_CAVEAT

    # None cases:
    assert m2._descriptive_igniting_note(detail, "pass") is None                 # gate PASSED
    agreed = {"at_crossing": [dict(d, onset="axial") for d in detail["at_crossing"]]}
    assert m2._descriptive_igniting_note(agreed, "epsilon_sensitive") is None    # onset AGREED (not onset-driven)
    no_coreonly = {"at_crossing": [                                              # disagreement between
        {"eps_rel": 0.01, "polarity": -1, "onset": "axial", "endgame": "self_limited"},   # two IGNITING
        {"eps_rel": 0.01, "polarity": 1, "onset": "global_first", "endgame": "marginal"}, # classes, no
        {"eps_rel": 0.05, "polarity": -1, "onset": "axial", "endgame": "self_limited"},   # core_only
        {"eps_rel": 0.05, "polarity": 1, "onset": "axial", "endgame": "self_limited"},
    ]}
    assert m2._descriptive_igniting_note(no_coreonly, "epsilon_sensitive") is None

    # igniting subset itself split (axial + global_first) WITH a non-igniting core_only present ->
    # still emit (disagreement DOES include non-ignition); igniting_onset reports the DISTRIBUTION.
    split = {"at_crossing": [
        {"eps_rel": 0.01, "polarity": -1, "onset": "axial", "endgame": "self_limited"},
        {"eps_rel": 0.01, "polarity": 1, "onset": "global_first", "endgame": "marginal"},
        {"eps_rel": 0.05, "polarity": -1, "onset": "core_only", "endgame": "self_limited"},
        {"eps_rel": 0.05, "polarity": 1, "onset": "axial", "endgame": "self_limited"},
    ]}
    note2 = m2._descriptive_igniting_note(split, "epsilon_sensitive")
    assert note2 is not None
    assert note2["igniting_onset"] == {"axial": 2, "global_first": 1}   # non-unanimous -> distribution


def test_nonlinear_spread_axial_onset_off_axis_absent():
    cfg = load_crit_config(); grid, kernels, core, b_core = _crit_op_context(cfg)
    m2cfg = m2.load_m2_config()
    crossing = m2.localize_alpha0_crossing(_points(), grid, kernels, core, cfg, m2cfg)
    sp = m2.read_nonlinear_spread(crossing, _points(), grid, kernels, core, b_core, cfg, m2cfg)
    assert sp["onset"] in ("axial", "core_only", "global_first", "off_axis", "undetermined")
    assert sp["off_axis"] in ("absent", "present", "undetermined")
    assert sp["control_minus_kick"] is True
    # trajectory sanity: off-axis power stays ~0 across all sampled steps
    for fm in sp["footprint_trajectory"]["at_crossing"]["core_kick"]:
        assert fm["off_axis"] < 0.1

    # rev2.4 decision C: on the ACTUAL v2.2 SIMULATION crossing the epsilon gate fails by an
    # onset-vs-non-ignition disagreement (3/4 igniting axial, the eps_rel=0.05/pol=-1 suppressing
    # kick doesn't ignite), so the descriptive-only note MUST be present and report the 3/4 split +
    # unanimous igniting onset — AND the primary verdict must stay undetermined (note doesn't leak).
    assert sp["epsilon_sensitivity"] == "epsilon_sensitive"
    assert sp["onset"] == "undetermined"
    assert sp["endgame"] == "undetermined"
    assert sp["off_axis"] == "undetermined"
    note = sp["descriptive_igniting_note"]
    assert note is not None
    assert note["n_igniting_of_total"] == {"at_crossing": "3/4", "just_past": "3/4"}
    assert note["igniting_onset"] == "axial"
    assert note["caveat"] == _IGNITING_CAVEAT
