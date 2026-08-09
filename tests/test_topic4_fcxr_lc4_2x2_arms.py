"""The 2x2's arms have to differ in one thing each, or neither comparison means anything.

The experiment asks two questions at once — does the cooperative curve keep the interictal state,
and does the slow release let the wear clear.  Both are confounded the moment the arms stop being
matched: if the linear arm delivers a bigger dose during the discharge, its damage to the
interictal side is a dose result, not a mechanism result.  These are the checks that would fail if
that happened, and they run without simulating anything.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_topic4_fcxr_lc4_cooperative_2x2 as L4  # noqa: E402


@pytest.fixture(scope="module")
def arms():
    P = L4._parameters()
    return P, {a["arm"]: a for a in L4._arms(P)}


def test_the_two_actuators_deliver_the_same_current_during_the_discharge(arms):
    """Force matching is the whole basis of the comparison."""
    P, A = arms
    hill = P["g_m_max"] * P["a_ictal"]
    lin = A["linear_fast"]["eta_m"] * P["m_ictal"]
    lin_slow = A["linear_slow"]["eta_m"] * P["m_ictal"] * 10.0   # its load runs 10x longer
    assert hill == pytest.approx(lin, rel=1e-9)
    assert hill == pytest.approx(lin_slow, rel=1e-9)


def test_the_release_arms_differ_only_in_the_release(arms):
    _, A = arms
    fast, slow = A["hill_fast"], A["hill_slow"]
    assert fast["tau_a_off"] < slow["tau_a_off"]
    for k in ("m_hill_K", "m_hill_n", "tau_a_on", "g_m_max", "tau_adp"):
        assert fast[k] == slow[k], f"{k} differs between the two release arms"


def test_the_actuator_arms_differ_only_in_the_actuator(arms):
    _, A = arms
    assert A["hill_slow"]["tau_adp"] == A["linear_fast"]["tau_adp"]
    assert A["linear_fast"].get("m_hill_K") is None
    assert A["hill_slow"]["m_hill_K"] is not None


def test_the_half_activation_sits_between_the_two_measured_states(arms):
    """Placed inside the gap, not on top of either distribution -- and the gap is only about 2x,
    so a half-point off by a factor lands inside one of the states."""
    P, _ = arms
    sep = L4.GEO._load_json(L4.SEP)["by_tau"][f"{L4.TAU_M_MS:g}"]["separation"]
    assert sep["quiet_max"] < P["K"] < sep["ictal_min"]


def test_the_predicted_interictal_load_stays_under_the_dose_that_was_measured_to_suppress(arms):
    """A per-cell current of 0.75% of the recurrent excitatory scale was measured to cut the
    interictal event rate 26-fold.  The cooperative arms must sit well under it by construction,
    and the linear arms must not -- that contrast is what the 2x2 is for."""
    P, A = arms
    hill_interictal = P["g_m_max"] * P["interictal_mean_activation"] / L4.I_EE_SCALE
    sep = L4.GEO._load_json(L4.SEP)
    m_interictal = sep["rates"]["interictal"]["median"] * L4.TAU_M_MS / 1000.0
    lin_interictal = A["linear_fast"]["eta_m"] * m_interictal / L4.I_EE_SCALE
    assert hill_interictal < 0.0075, f"the cooperative arm is already in the suppressing range: {hill_interictal:.4%}"
    assert lin_interictal > hill_interictal, "the linear arm must carry more interictal current"


def test_the_cooperativity_key_is_read_as_an_integer():
    """`f"{10.0:g}".rstrip("0")` is "1" -- a silent read of the wrong curve."""
    assert L4._hill_key(4.0) == "4"
    assert L4._hill_key(10.0) == "10"
    with pytest.raises(ValueError, match="integer cooperativities"):
        L4._hill_key(3.5)


def test_the_run_is_long_enough_for_the_wear_to_clear_after_a_stop(arms):
    """Thirteen seconds of near-silence take the wear from a discharge down past the level that
    departs on its own; a run that ends before that cannot report the recovery leg either way."""
    assert L4.RUN_MS >= 40000.0
