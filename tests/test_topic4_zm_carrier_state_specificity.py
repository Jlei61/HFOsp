import numpy as np
import pytest

from scripts.analyze_topic4_zm_carrier_state_specificity import (
    INTERICTAL_STATE, adjudicate, arm_key, traces_identical,
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
    assert arm_key(_summary(INTERICTAL_STATE, 0.)) == (INTERICTAL_STATE, 0.)


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


def _rows(passing_states, *, gate_inert=True, g_at_interictal=0.0):
    rows = {}
    for state in ("bounded_mid__rising", "bounded_mid__peak",
                  "bounded_late__rising", "bounded_late__peak"):
        rows[(state, 0.32)] = {
            "credible_carrier": state in passing_states,
            "persistent_g_core_mean_peak": 0.18,
        }
    rows[(INTERICTAL_STATE, 0.32)] = {
        "credible_carrier": False,
        "persistent_g_core_mean_peak": g_at_interictal,
        "identical_to_no_mechanism": gate_inert,
    }
    rows[(INTERICTAL_STATE, 0.0)] = {
        "credible_carrier": False, "persistent_g_core_mean_peak": 0.0,
    }
    return rows


def test_a_carrier_at_every_traversed_point_with_an_inert_interictal_point():
    verdict = adjudicate(_rows((
        "bounded_mid__rising", "bounded_mid__peak",
        "bounded_late__rising", "bounded_late__peak",
    )))
    assert verdict["verdict"] == "STATE_SELECTIVE_CARRIER_ACROSS_TRAVERSED_POINTS"
    assert verdict["carrier_states"] == [
        "bounded_late__peak", "bounded_late__rising",
        "bounded_mid__peak", "bounded_mid__rising",
    ]
    assert verdict["interictal_point_inert"] is True


def test_a_carrier_at_only_one_point_cannot_carry_a_lifecycle():
    verdict = adjudicate(_rows(("bounded_late__peak",)))
    assert verdict["verdict"] == "CARRIER_CONFINED_TO_ONE_OPERATING_POINT"


def test_a_mechanism_that_also_fires_interictally_is_not_state_selective():
    rows = _rows(("bounded_late__peak", "bounded_late__rising"),
                 gate_inert=False, g_at_interictal=0.15)
    rows[(INTERICTAL_STATE, 0.32)]["credible_carrier"] = True
    verdict = adjudicate(rows)
    assert verdict["verdict"] == "MECHANISM_NOT_STATE_SELECTIVE"
    assert verdict["interictal_point_inert"] is False


def test_no_carrier_anywhere_is_reported_without_a_selectivity_claim():
    verdict = adjudicate(_rows(()))
    assert verdict["verdict"] == "NO_CARRIER_AT_ANY_FROZEN_OPERATING_POINT"
    assert verdict["carrier_states"] == []


def test_adjudication_requires_the_interictal_pair():
    rows = _rows(("bounded_late__peak",))
    del rows[(INTERICTAL_STATE, 0.0)]
    with pytest.raises(RuntimeError):
        adjudicate(rows)
