"""Unit tests for the M4 Pass-1 phase-plane PURE decision functions (spec rev4 §8.1, §9.1).
No simulation is run here — only the go/no-go decision logic + core-mask/q_core math."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.sef_hfo_m4_phaseplane import (  # noqa: E402
    q_core, derive_core_mask, GuardThresholds, CellMetrics, CellVerdict,
    is_bounded, is_trivial_A, is_trivial_B, classify_cell,
    largest_contiguous, go_plane_verdict, calibrate_guards_from_references,
)


# ---- helper: a baseline "go" cell (localized bounded core) that all field tests perturb ----
def _core_cell(**kw):
    base = dict(persist=True, act_frac=0.25, s_grad=0.5, f_off=0.05, core_overlap=0.6, globality=0.5,
                self_limited=False, b_delta_avg=0.98, monotonic_saturation=False, tail_returns=True,
                finite_energy=True)
    base.update(kw)
    return CellMetrics(**base)


# ---------------------------------------------------------------- q_core / core mask
def test_q_core_weighted_mean():
    q = np.array([[1.0, 0.5], [0.25, 0.0]])
    m = np.array([[1.0, 0.0], [1.0, 0.0]])              # core = two cells (q=1.0, q=0.25)
    assert np.isclose(q_core(q, m), (1.0 + 0.25) / 2)


def test_q_core_empty_mask_and_shape_mismatch_raise():
    with pytest.raises(ValueError, match="empty core mask"):
        q_core(np.ones((2, 2)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="must match"):
        q_core(np.ones((2, 2)), np.ones((3, 3)))


def test_derive_core_mask_earliest_activators_only():
    # 3x3 first-activation map; inf = never activated. frac=0.5 of the finite activators = earliest half.
    fa = np.array([[0.0, 1.0, 2.0],
                   [3.0, np.inf, np.inf],
                   [np.inf, np.inf, np.inf]])
    m = derive_core_mask(fa, frac=0.5)
    assert m[0, 0] == 1.0 and m[0, 1] == 1.0            # earliest two are core
    assert m[1, 1] == 0.0                               # never-activated cell is never core
    assert m.sum() == 2.0                               # quantile(0.5) of [0,1,2,3] = 1.5 -> {0,1}


def test_derive_core_mask_raises_when_nothing_activated():
    with pytest.raises(ValueError, match="no cell activated"):
        derive_core_mask(np.full((2, 2), np.inf))
    with pytest.raises(ValueError, match="frac must be"):
        derive_core_mask(np.zeros((2, 2)), frac=0.0)


# ---------------------------------------------------------------- bounded / trivial predicates
def test_is_bounded_relaxed_branching():
    th = GuardThresholds()
    assert is_bounded(_core_cell(b_delta_avg=1.04), th)            # <= 1+eps(0.05) -> bounded
    assert not is_bounded(_core_cell(b_delta_avg=1.20), th)        # window-avg branching too high
    assert not is_bounded(_core_cell(monotonic_saturation=True), th)
    assert not is_bounded(_core_cell(tail_returns=False), th)
    assert not is_bounded(_core_cell(finite_energy=False), th)


def test_trivial_A_low_amplitude_global_skirt():
    th = GuardThresholds()
    skirt = _core_cell(act_frac=0.9, core_overlap=0.85, globality=0.15)   # whole-field, core-weighted, low amp
    assert is_trivial_A(skirt, th)
    # high-amplitude distributed burst is NOT trivial-A (high globality)
    assert not is_trivial_A(_core_cell(act_frac=0.9, core_overlap=0.4, globality=0.7), th)


def test_trivial_B_needs_axis_confined_AND_self_limited():
    th = GuardThresholds()
    assert is_trivial_B(_core_cell(f_off=0.05, self_limited=True), th)        # axis-confined + retreats
    assert not is_trivial_B(_core_cell(f_off=0.05, self_limited=False), th)   # sustained axial core != trivial-B
    assert not is_trivial_B(_core_cell(f_off=0.5, self_limited=True), th)     # off-axis spread != trivial-B


# ---------------------------------------------------------------- classify_cell (the core §9.1 logic)
def test_localized_bounded_core_passes_go():
    # rev4 PRIMARY target: sustained, bounded, moderate-area, spatially-structured core with LOW f_off.
    # Must pass go (f_off is NOT a hard go requirement).
    v = classify_cell(_core_cell(), GuardThresholds())
    assert v.go and v.label == "go"


def test_high_amplitude_distributed_burst_passes_go_as_candidate():
    v = classify_cell(_core_cell(act_frac=0.8, f_off=0.6, core_overlap=0.3, globality=0.7), GuardThresholds())
    assert v.go                                                     # secondary candidate (Pass-2 adjudicates)


def test_trivial_A_and_B_are_not_go():
    th = GuardThresholds()
    va = classify_cell(_core_cell(act_frac=0.9, core_overlap=0.85, globality=0.15), th)
    assert (not va.go) and va.label == "trivial_A"
    vb = classify_cell(_core_cell(f_off=0.05, self_limited=True), th)
    assert (not vb.go) and vb.label == "trivial_B"


def test_decay_runaway_blip_labels():
    th = GuardThresholds()
    assert classify_cell(_core_cell(persist=False), th).label == "decay"
    assert classify_cell(_core_cell(monotonic_saturation=True), th).label == "runaway"
    assert classify_cell(_core_cell(act_frac=0.02), th).label == "blip"


def test_s_grad_gate_excludes_simultaneous_ignition():
    th = GuardThresholds()
    # no spatial onset sequence (s_grad ~ 0) -> not a structured event -> not go
    assert not classify_cell(_core_cell(s_grad=0.0), th).go


# ---------------------------------------------------------------- plane-level verdict
def test_largest_contiguous_4connected():
    g = np.array([[1, 1, 0, 1],
                  [1, 0, 0, 1],
                  [0, 0, 1, 0]], dtype=bool)
    assert largest_contiguous(g) == 3                              # top-left L of 3; the 2 on the right are separate
    assert largest_contiguous(np.zeros((3, 3), bool)) == 0


def test_calibrate_guards_excludes_reference_instances():
    # arm-0 TRIVIAL-A flood + TRIVIAL-B axial-retreat references -> calibrated guards must FLAG them.
    flood = _core_cell(act_frac=0.9, core_overlap=0.85, globality=0.15)
    axial = _core_cell(f_off=0.05, self_limited=True)
    g = calibrate_guards_from_references(flood, axial, margin=0.05)
    assert is_trivial_A(flood, g)          # the flood reference is now caught as TRIVIAL-A
    assert is_trivial_B(axial, g)          # the axial-retreat reference is caught as TRIVIAL-B
    # a genuine localized bounded core (moderate act_frac, high globality, off-axis some) is NOT flagged
    core = _core_cell(act_frac=0.3, core_overlap=0.6, globality=0.5, f_off=0.3, self_limited=False)
    assert not is_trivial_A(core, g) and not is_trivial_B(core, g)


def test_go_plane_verdict_go_and_nogo_cases():
    k = 3
    arm2 = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)   # contiguous 3
    arm1_empty = np.zeros((3, 3), dtype=bool)
    v = go_plane_verdict(arm2, arm1_empty, k_min=k)
    assert v["verdict"] == "go" and v["arm2_max_contiguous"] == 3

    single = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool)  # single point < k_min
    v2 = go_plane_verdict(single, arm1_empty, k_min=k)
    assert v2["verdict"] == "no-go" and "single point" in v2["reason"]

    v3 = go_plane_verdict(arm2, arm2, k_min=k)                       # arm1 also opens -> only-suppresses not excluded
    assert v3["verdict"] == "no-go" and "arm1" in v3["reason"]
