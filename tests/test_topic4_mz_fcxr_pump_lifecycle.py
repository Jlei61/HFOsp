"""FCXR pump-lifecycle gate adjudicators — Task 7 (Gate I-a/I-b) and Task 9 (Gate T) TDD.

Every test encodes one thing the adjudicator must REFUSE to call a pass. The recurring failure mode
these guard against is an upstream green being propagated downstream: engineering tests passing, one
pretty trajectory, two agreeing seeds, a falling rate or a rendered figure must never move a verdict.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.topic4_mz_fcxr_pump_lifecycle as LC  # noqa: E402


def _ok_parity(**kw):
    d = dict(byte_parity_pass=True, zmx_update_order_pass=True, blessed_hashes_match=True,
             causal_order_pass=True)
    d.update(kw)
    return d


def _ok_baseline(**kw):
    d = dict(candidate_admissible=True, equivalence_all_within=True, n_metrics_outside=0)
    d.update(kw)
    return d


def _ok_readout(**kw):
    d = dict(identifiability_status="IDENTIFIABLE_AS_PROXY", identity_max_abs_err=0.0,
             band_power_pump=1e-6, band_power_no_direct_pump=1.0)
    d.update(kw)
    return d


# ============================== Gate I-a ==============================
def test_gate_Ia_passes_only_with_all_three_evidence_blocks():
    v = LC.adjudicate_gate_Ia(_ok_parity(), _ok_baseline(), _ok_readout())
    assert v["status"] == "PASS"


def test_gate_Ia_is_unresolved_not_pass_when_evidence_is_missing():
    """Missing evidence must never default to a pass -- the equivalence result simply not existing
    yet is the common case while the held-out run is still going."""
    v = LC.adjudicate_gate_Ia(_ok_parity(), _ok_baseline(equivalence_all_within=None), _ok_readout())
    assert v["status"] == "UNRESOLVED" and "baseline.equivalence_all_within" in v["missing"]


def test_gate_Ia_parity_and_update_order_fail_before_anything_else():
    assert LC.adjudicate_gate_Ia(_ok_parity(byte_parity_pass=False), _ok_baseline(),
                                 _ok_readout())["status"] == "FAIL_PARITY"
    assert LC.adjudicate_gate_Ia(_ok_parity(blessed_hashes_match=False), _ok_baseline(),
                                 _ok_readout())["status"] == "FAIL_PARITY"
    assert LC.adjudicate_gate_Ia(_ok_parity(zmx_update_order_pass=False), _ok_baseline(),
                                 _ok_readout())["status"] == "FAIL_UPDATE_ORDER"
    assert LC.adjudicate_gate_Ia(_ok_parity(causal_order_pass=False), _ok_baseline(),
                                 _ok_readout())["status"] == "FAIL_UPDATE_ORDER"


def test_gate_Ia_fails_readout_identifiability_when_no_direct_pump_cannot_be_built():
    v = LC.adjudicate_gate_Ia(_ok_parity(), _ok_baseline(),
                              _ok_readout(identifiability_status="READOUT_NOT_IDENTIFIABLE"))
    assert v["status"] == "FAIL_READOUT_IDENTIFIABILITY"
    v2 = LC.adjudicate_gate_Ia(_ok_parity(), _ok_baseline(), _ok_readout(identity_max_abs_err=1e-3))
    assert v2["status"] == "FAIL_READOUT_IDENTIFIABILITY"


def test_gate_Ia_fails_readout_contamination_when_the_pump_term_carries_the_band_power():
    """The forbidden outcome: the 1-80 Hz content lives in the direct pump current instead of being
    produced by network activity."""
    v = LC.adjudicate_gate_Ia(_ok_parity(), _ok_baseline(),
                              _ok_readout(band_power_pump=5.0, band_power_no_direct_pump=1.0))
    assert v["status"] == "FAIL_READOUT_CONTAMINATION" and v["contamination_ratio"] == 5.0


def test_gate_Ia_fails_baseline_on_inadmissible_candidate_or_broken_equivalence():
    assert LC.adjudicate_gate_Ia(_ok_parity(), _ok_baseline(candidate_admissible=False),
                                 _ok_readout())["status"] == "FAIL_BASELINE"
    assert LC.adjudicate_gate_Ia(_ok_parity(),
                                 _ok_baseline(equivalence_all_within=False, n_metrics_outside=2),
                                 _ok_readout())["status"] == "FAIL_BASELINE"


# ============================== Gate I-b (non-blocking) ==============================
def test_gate_Ib_not_run_forbids_the_response_mode_claim_without_blocking_anything():
    v = LC.adjudicate_gate_Ib(None, None)
    assert v["status"] == "NOT_RUN" and v["response_mode_claim_allowed"] is False


def test_gate_Ib_passes_only_when_the_operator_is_repeatable_on_all_three_axes():
    good = dict(epsilon_linear=True, noise_replay_repeatable=True, binning_stable=True)
    assert LC.adjudicate_gate_Ib(dict(regime="X"), good)["response_mode_claim_allowed"] is True
    bad = dict(good, binning_stable=False)
    v = LC.adjudicate_gate_Ib(dict(regime="X"), bad)
    assert v["status"] == "FAIL_OPERATOR" and v["response_mode_claim_allowed"] is False


# ============================== Gate T ==============================
def _cell(D, rho, ic, label, dP=1.0, field="shaped", P=None, unsafe=False, runaway=None):
    return dict(D=D, rho_u=rho, ic=ic, label=label, field=field,
                P=(rho * 0.4 if P is None else P),
                numerical=dict(numerical_unsafe=unsafe, runaway_early_stop_ms=runaway),
                slow_flow=dict(dP_dt=dP, dZ_dt=0.0, P=(rho * 0.4 if P is None else P), Z=1.0 - D))


def _grid(high_labels, low_labels=None, **kw):
    """D=0 healthy row (always low) + D=0.15 impaired row with the supplied high-IC labels."""
    rhos = [0.0, 0.33, 0.67, 1.0]
    low_labels = low_labels or ["INTERICTAL_WORKPOINT"] * 4
    cells = []
    for r in rhos:
        cells += [_cell(0.0, r, "low", "INTERICTAL_WORKPOINT", **kw),
                  _cell(0.0, r, "high", "INTERICTAL_WORKPOINT", **kw)]
    for r, hl, ll in zip(rhos, high_labels, low_labels):
        cells += [_cell(0.15, r, "low", ll, **kw), _cell(0.15, r, "high", hl, **kw)]
    return cells


def test_gate_T_passes_on_a_selective_exit_corridor():
    cells = _grid(["FINITE_HIGH_ORBIT", "FINITE_HIGH_ORBIT", "INTERICTAL_WORKPOINT",
                   "INTERICTAL_WORKPOINT"])
    v = LC.adjudicate_gate_T(cells)
    assert v["status"] == "PASS"
    assert v["exit"]["rho_u"] == 0.67 and v["exit"]["low_label"] == "INTERICTAL_WORKPOINT"
    assert v["healthy_low_preserved"] and v["flow_toward_exit"]


def test_gate_T_no_go_when_the_pump_kills_the_low_branch_at_the_same_time():
    """The failure the spec names explicitly: 'pump 同时移除 low/high'. Both branches gone is
    suppression, not a selective exit."""
    cells = _grid(["FINITE_HIGH_ORBIT", "FINITE_HIGH_ORBIT", "METASTABLE_TRANSIENT",
                   "METASTABLE_TRANSIENT"],
                  low_labels=["INTERICTAL_WORKPOINT", "INTERICTAL_WORKPOINT",
                              "METASTABLE_TRANSIENT", "METASTABLE_TRANSIENT"])
    v = LC.adjudicate_gate_T(cells)
    assert v["status"] == "TOPOLOGY_NO_GO"
    assert "does not select" in " ".join(v["reasons"])


def test_gate_T_no_go_when_the_high_branch_is_never_removed():
    cells = _grid(["FINITE_HIGH_ORBIT"] * 4)
    assert LC.adjudicate_gate_T(cells)["status"] == "TOPOLOGY_NO_GO"


def test_gate_T_reports_no_high_branch_instead_of_a_false_exit():
    """If the impaired corner never reaches a sustained high branch there is nothing to exit FROM;
    calling that an exit corridor would be the cheapest possible false positive."""
    cells = _grid(["INTERICTAL_WORKPOINT"] * 4)
    v = LC.adjudicate_gate_T(cells)
    assert v["status"] == "NO_HIGH_BRANCH"


def test_gate_T_refuses_to_read_topology_off_numerically_unsafe_cells():
    cells = _grid(["FINITE_HIGH_ORBIT", "FINITE_HIGH_ORBIT", "INTERICTAL_WORKPOINT",
                   "INTERICTAL_WORKPOINT"])
    cells[3]["numerical"]["numerical_unsafe"] = True
    assert LC.adjudicate_gate_T(cells)["status"] == "UNSAFE"
    cells2 = _grid(["FINITE_HIGH_ORBIT", "FINITE_HIGH_ORBIT", "INTERICTAL_WORKPOINT",
                    "INTERICTAL_WORKPOINT"])
    cells2[5]["numerical"]["runaway_early_stop_ms"] = 300.0
    assert LC.adjudicate_gate_T(cells2)["status"] == "UNSAFE"


def test_gate_T_is_unresolved_when_the_high_branch_flow_does_not_point_at_the_exit():
    cells = _grid(["FINITE_HIGH_ORBIT", "FINITE_HIGH_ORBIT", "INTERICTAL_WORKPOINT",
                   "INTERICTAL_WORKPOINT"], dP=-1.0)
    v = LC.adjudicate_gate_T(cells)
    assert v["status"] == "UNRESOLVED" and not v["flow_toward_exit"]


def test_gate_T_is_unresolved_when_a_grid_cell_lacks_an_initial_condition():
    cells = [c for c in _grid(["FINITE_HIGH_ORBIT"] * 4) if not (c["D"] == 0.15 and c["ic"] == "low")]
    assert LC.adjudicate_gate_T(cells)["status"] == "UNRESOLVED"


def test_field_controls_need_two_arms_with_an_exit_before_claiming_distinguishability():
    shaped = _grid(["FINITE_HIGH_ORBIT", "FINITE_HIGH_ORBIT", "INTERICTAL_WORKPOINT",
                    "INTERICTAL_WORKPOINT"])
    only_shaped = LC.compare_field_controls(shaped)
    assert only_shaped["fields_present"] == ["shaped"] and not only_shaped["distinguishable"]
    uniform = _grid(["FINITE_HIGH_ORBIT", "INTERICTAL_WORKPOINT", "INTERICTAL_WORKPOINT",
                     "INTERICTAL_WORKPOINT"], field="uniform")
    both = LC.compare_field_controls(shaped + uniform)
    assert set(both["fields_present"]) == {"shaped", "uniform"} and both["distinguishable"]


# ============================== conclusion language ==============================
def test_conclusion_language_never_skips_a_gate():
    ia = dict(status="PASS")
    assert "Gate I-a only" in LC.gate_conclusion_language({"Ia": ia})
    assert "Gate I-a+T:" in LC.gate_conclusion_language({"Ia": ia, "T": dict(status="PASS")})
    # a downstream pass without its upstream gate must NOT be reported as reached
    txt = LC.gate_conclusion_language({"Ia": ia, "T": dict(status="TOPOLOGY_NO_GO"),
                                       "C": dict(status="PASS"), "S": dict(status="PASS")})
    assert "Gate I-a only" in txt


def test_conclusion_language_says_nothing_is_available_when_Ia_fails():
    txt = LC.gate_conclusion_language({"Ia": dict(status="FAIL_BASELINE"), "T": dict(status="PASS")})
    assert "did not pass" in txt and "no topology or lifecycle claim" in txt


def test_prefix_hashes_change_when_the_prefix_changes():
    import numpy as np

    class _Slow:
        trace_z_mean = [1.0, 0.9]
        trace_u_mean = [0.1, 0.2]

    res_a = dict(rate_E=np.array([1.0, 2.0, 3.0]), E_spk_bool=np.zeros((3, 4), bool))
    res_b = dict(rate_E=np.array([1.0, 2.0, 9.0]), E_spk_bool=np.zeros((3, 4), bool))
    ha = LC.prefix_hashes(res_a, _Slow(), 2)
    hb = LC.prefix_hashes(res_b, _Slow(), 2)
    assert ha["rate_sha"] == hb["rate_sha"]                 # identical prefix -> identical hash
    assert LC.prefix_hashes(res_a, _Slow(), 3)["rate_sha"] != \
        LC.prefix_hashes(res_b, _Slow(), 3)["rate_sha"]     # divergence past the prefix shows up
