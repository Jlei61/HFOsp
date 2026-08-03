import numpy as np
import pytest

from scripts.analyze_topic4_zm_carrier_state_specificity import (
    REFERENCE_STATE, adjudicate, arm_key, traces_identical,
)


def _summary(state, g_max, *, e_exc=60., seed=1):
    return {
        "state": state,
        "T_ms": 2500.,
        "mechanism": {
            "pv_som_inhibitory_subtypes": {
                "som_source_fraction_realized": .25,
                "som_slow_integrated_budget_fraction": .35,
                "som_recruit_delay_scale": 3.,
                "tau_d_som_ms": 60.,
                "seed": seed,
            },
            "state_selective_mode_H": {
                "rho_mode_H": 0.,
                "tau_mode_H": 250.,
                "tau_mode_H_down": 250.,
                "mode_H_common_subtraction": 0.,
                "mode_H_persistent_g_max": g_max,
                "mode_H_persistent_e_exc": e_exc,
                "m_mode_half": 30.,
            },
        },
    }


def test_arm_key_carries_both_the_operating_point_and_the_dose():
    assert arm_key(_summary("bounded_late__peak", .32)) == (
        "bounded_late__peak", 0.32
    )
    assert arm_key(_summary(REFERENCE_STATE, 0.)) == (REFERENCE_STATE, 0.)


def test_arm_key_rejects_arms_outside_the_locked_comparison():
    assert arm_key(_summary("bounded_late__peak", .32, e_exc=40.)) is None
    assert arm_key(_summary("bounded_late__peak", .32, seed=2)) is None
    off_panel = _summary("bounded_late__peak", .32)
    off_panel["mechanism"]["state_selective_mode_H"]["rho_mode_H"] = .5
    assert arm_key(off_panel) is None
    short = _summary("bounded_late__peak", .32)
    short["T_ms"] = 12000.
    assert arm_key(short) is None


def test_traces_identical_is_exact_not_approximate():
    a = {"fine_core_rate_hz": np.arange(10.), "trace_S_G": np.zeros(4)}
    b = {"fine_core_rate_hz": np.arange(10.), "trace_S_G": np.zeros(4)}
    assert traces_identical(a, b)
    b["fine_core_rate_hz"] = b["fine_core_rate_hz"] + 1e-12
    assert not traces_identical(a, b)


def _rows(passing_states, *, ref_inert=True, g_at_ref=0.0, ref_gate_open=0.02):
    rows = {}
    for state in ("bounded_mid__rising", "bounded_mid__peak",
                  "bounded_late__rising", "bounded_late__peak"):
        rows[(state, 0.32)] = {
            "credible_carrier": state in passing_states,
            "persistent_g_core_mean_peak": 0.18,
            "z_gate_open_at_freeze": 0.99,
        }
    rows[(REFERENCE_STATE, 0.32)] = {
        "credible_carrier": False,
        "persistent_g_core_mean_peak": g_at_ref,
        "identical_to_no_mechanism": ref_inert,
        "z_gate_open_at_freeze": ref_gate_open,
    }
    rows[(REFERENCE_STATE, 0.0)] = {
        "credible_carrier": False, "persistent_g_core_mean_peak": 0.0,
        "z_gate_open_at_freeze": ref_gate_open,
    }
    return rows


def test_a_carrier_at_every_traversed_point_with_an_inert_reference_point():
    verdict = adjudicate(_rows((
        "bounded_mid__rising", "bounded_mid__peak",
        "bounded_late__rising", "bounded_late__peak",
    )))
    assert verdict["verdict"] == "STATE_SELECTIVE_CARRIER_ACROSS_TRAVERSED_POINTS"
    assert verdict["carrier_states"] == [
        "bounded_late__peak", "bounded_late__rising",
        "bounded_mid__peak", "bounded_mid__rising",
    ]
    assert verdict["reference_point_inert"] is True
    assert verdict["selectivity_testable"] is True


def test_a_carrier_at_only_one_point_cannot_carry_a_lifecycle():
    verdict = adjudicate(_rows(("bounded_late__peak",)))
    assert verdict["verdict"] == "CARRIER_CONFINED_TO_ONE_OPERATING_POINT"


def test_a_mechanism_that_also_fires_at_a_closed_gate_is_not_state_selective():
    rows = _rows(("bounded_late__peak", "bounded_late__rising"),
                 ref_inert=False, g_at_ref=0.15)
    verdict = adjudicate(rows)
    assert verdict["verdict"] == "MECHANISM_NOT_STATE_SELECTIVE"
    assert verdict["reference_point_inert"] is False


def test_a_reference_point_whose_gate_is_already_open_cannot_test_selectivity():
    """Every sampled point sits past the gate, so the panel has no off state."""
    rows = _rows(("bounded_late__peak", "bounded_late__rising"),
                 ref_inert=False, g_at_ref=0.12, ref_gate_open=0.99)
    verdict = adjudicate(rows)
    assert verdict["selectivity_testable"] is False
    assert verdict["verdict"] == "CARRIER_ON_THE_LATE_ARC_SELECTIVITY_UNTESTED"
    assert "no operating point" in verdict["next_coordinate"]


def test_no_carrier_anywhere_is_reported_without_a_selectivity_claim():
    verdict = adjudicate(_rows(()))
    assert verdict["verdict"] == "NO_CARRIER_AT_ANY_FROZEN_OPERATING_POINT"
    assert verdict["carrier_states"] == []


def test_adjudication_requires_the_reference_pair():
    rows = _rows(("bounded_late__peak",))
    del rows[(REFERENCE_STATE, 0.0)]
    with pytest.raises(RuntimeError):
        adjudicate(rows)


def test_the_swept_dose_comes_from_the_reference_pair_not_the_dose_band():
    """A dose band at one operating point must not redefine the swept dose."""
    rows = _rows(("bounded_late__peak", "bounded_mid__peak"))
    for stray in (0.04, 0.08, 0.48):
        rows[("bounded_late__peak", stray)] = {
            "credible_carrier": False, "persistent_g_core_mean_peak": .01,
            "z_gate_open_at_freeze": .99,
        }
    verdict = adjudicate(rows)
    assert verdict["dose"] == 0.32
    assert verdict["carrier_states"] == ["bounded_late__peak", "bounded_mid__peak"]


def test_an_ambiguous_reference_dose_is_refused_rather_than_guessed():
    rows = _rows(("bounded_late__peak",))
    rows[(REFERENCE_STATE, 0.16)] = {
        "credible_carrier": False, "persistent_g_core_mean_peak": .0,
        "z_gate_open_at_freeze": .02,
    }
    with pytest.raises(RuntimeError):
        adjudicate(rows)
