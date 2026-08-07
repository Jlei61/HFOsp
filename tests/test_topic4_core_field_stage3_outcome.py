"""Task 3 -- the verdict must not hide anything behind its own headline.

Stage 2 returned one short-circuit label, SIMULATOR_OVERFIT fired, and the
question the stage existed to answer went unanswered. Reordering the rules
without also returning the components would replay that occlusion in the
opposite direction, so most of these tests are about what is still visible
after the primary label has been decided.
"""
import pytest

from src.topic4_core_field_stage3_outcome import (ALLOWED_STATEMENTS,
                                                  OUTCOME_ORDER,
                                                  classify_stage3)


def _base(**over):
    r = dict(integrity_ok=True, restart_field_corr_median=0.9,
             restart_r_bar_sd=0.3, train_delta=0.2, heldout_delta=0.15,
             r_bar=0.4, c_axis_2mm=0.85, heldout_delta_vs_onaxis=0.05)
    r.update(over)
    return r


def test_taxonomy_is_ordered_and_every_label_has_a_statement():
    assert OUTCOME_ORDER[0] == "FAIL_CLOSED"
    assert OUTCOME_ORDER.index("POSITION_UNIDENTIFIABLE") < OUTCOME_ORDER.index("SIMULATOR_OVERFIT")
    assert set(ALLOWED_STATEMENTS) == set(OUTCOME_ORDER)


# ------------------------------------------------------------ each outcome
def test_missing_artefact_fails_closed():
    v = classify_stage3(_base(integrity_ok=False))
    assert v["primary_outcome"] == "FAIL_CLOSED"


def test_nan_input_fails_closed_and_names_the_field():
    v = classify_stage3(_base(r_bar=float("nan")))
    assert v["primary_outcome"] == "FAIL_CLOSED"
    assert "r_bar" in v["missing_fields"]


def test_unstable_position_is_reported_as_such():
    v = classify_stage3(_base(restart_r_bar_sd=2.5))
    assert v["primary_outcome"] == "POSITION_UNIDENTIFIABLE"
    assert v["position_stable"] is False


def test_dissimilar_restart_fields_also_count_as_unidentifiable():
    v = classify_stage3(_base(restart_field_corr_median=0.2))
    assert v["primary_outcome"] == "POSITION_UNIDENTIFIABLE"


def test_training_gain_that_does_not_transfer():
    v = classify_stage3(_base(train_delta=0.4, heldout_delta=-0.05))
    assert v["primary_outcome"] == "SIMULATOR_OVERFIT"
    assert v["transfers_to_heldout"] is False


def test_field_lands_near_the_axis():
    v = classify_stage3(_base(r_bar=0.4, c_axis_2mm=0.85))
    assert v["primary_outcome"] == "AXIS_REDISCOVERED"
    assert v["axis_relation"] == "near"


def test_field_lands_off_the_axis_without_losing_score():
    v = classify_stage3(_base(r_bar=5.0, c_axis_2mm=0.1,
                              heldout_delta_vs_onaxis=0.02))
    assert v["primary_outcome"] == "AXIS_NOT_REQUIRED"
    assert v["axis_relation"] == "off"


def test_off_axis_but_worse_than_on_axis_is_inconclusive():
    v = classify_stage3(_base(r_bar=5.0, c_axis_2mm=0.1,
                              heldout_delta_vs_onaxis=-0.3))
    assert v["primary_outcome"] == "AXIS_INCONCLUSIVE"
    assert v["axis_relation"] == "off"


def test_middle_ground_is_inconclusive_not_forced():
    v = classify_stage3(_base(r_bar=1.5, c_axis_2mm=0.5))
    assert v["primary_outcome"] == "AXIS_INCONCLUSIVE"
    assert v["axis_relation"] == "inconclusive"


# ---------------------------------------------------------------- ordering
def test_position_instability_outranks_overfitting():
    v = classify_stage3(_base(restart_r_bar_sd=3.0, train_delta=0.4,
                              heldout_delta=-0.1))
    assert v["primary_outcome"] == "POSITION_UNIDENTIFIABLE"


def test_integrity_outranks_everything():
    v = classify_stage3(_base(integrity_ok=False, restart_r_bar_sd=3.0,
                              train_delta=0.4, heldout_delta=-0.1))
    assert v["primary_outcome"] == "FAIL_CLOSED"


# --------------------------------------------- the headline hides nothing
def test_the_primary_label_does_not_swallow_the_other_facts():
    # THE contract for this task: two failures hold at once and both stay visible
    v = classify_stage3(_base(restart_r_bar_sd=3.0, train_delta=0.4,
                              heldout_delta=-0.1))
    assert v["primary_outcome"] == "POSITION_UNIDENTIFIABLE"
    assert v["position_stable"] is False
    assert v["transfers_to_heldout"] is False
    assert {1, 2} <= set(v["all_triggered_conditions"])
    # and the axis fact survives too, even though it lost the short circuit
    assert v["axis_relation"] == "near" and 3 in v["all_triggered_conditions"]


def test_axis_relation_is_computed_even_when_the_verdict_is_a_failure():
    v = classify_stage3(_base(train_delta=0.4, heldout_delta=-0.1,
                              r_bar=0.2, c_axis_2mm=0.9))
    assert v["primary_outcome"] == "SIMULATOR_OVERFIT"
    assert v["axis_relation"] == "near"          # not None, not withheld
    assert 3 in v["all_triggered_conditions"]


def test_measurements_are_passed_through_for_the_reader():
    v = classify_stage3(_base(r_bar=0.4))
    assert v["measurements"]["r_bar"] == pytest.approx(0.4)
    assert set(v["thresholds"]) >= {"near_axis_mm", "field_corr_min"}


def test_success_still_lists_its_trigger():
    v = classify_stage3(_base())
    assert v["primary_outcome"] == "AXIS_REDISCOVERED"
    assert v["all_triggered_conditions"] == [3]
    assert v["position_stable"] is True and v["transfers_to_heldout"] is True


# ------------------------------------------------------------ wording lock
def test_rediscovery_wording_does_not_claim_independence():
    # Stage 3 does not remove the circularity: the anisotropy is still fixed to
    # an axis fitted from patient ranks, and the score target comes from the
    # same ranks (spec 9.0)
    s = ALLOWED_STATEMENTS["AXIS_REDISCOVERED"]
    assert "independent" not in s.lower()
    assert "already set from the" in s
