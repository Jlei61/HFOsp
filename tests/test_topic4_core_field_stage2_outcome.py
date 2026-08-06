import numpy as np
import pytest

from src.topic4_core_field_stage2_outcome import (
    OUTCOME_ORDER, classify_stage2, equivalent_optimum_family)


def _base(**over):
    """A run that would classify as RECOVERED_NONTRIVIAL_FIELD unless overridden."""
    d = dict(
        train_delta=+0.20, heldout_delta=+0.18,
        vs_axis_only=dict(mean=+0.15, n_above=11, n=12),
        vs_uniform=dict(mean=+0.12, n_above=11, n=12),
        vs_manual_projected=dict(mean=+0.02, n_above=8, n=12),
        bidirectional_gate=dict(passed=True),
        family=dict(median_field_corr=0.82, n_members=5),
        restart_field_corr_median=0.75,
        coverage=dict(learned=0.62, manual_smooth=0.60, margin=0.10),
        integrity_ok=True,
    )
    d.update(over)
    return d


def test_the_taxonomy_is_an_ORDERED_short_circuit():
    """Order is part of the contract: it decides which label is reported when
    several conditions hold at once (spec 8.1)."""
    assert OUTCOME_ORDER[0] == "FAIL_CLOSED"
    assert OUTCOME_ORDER.index("SIMULATOR_OVERFIT") < OUTCOME_ORDER.index("ONE_DIRECTION_ONLY")
    assert OUTCOME_ORDER.index("ONE_DIRECTION_ONLY") < OUTCOME_ORDER.index("FIELD_NONIDENTIFIABLE")
    assert OUTCOME_ORDER[-1] == "RECOVERED_NONTRIVIAL_FIELD"


def test_a_clean_run_is_recovered_nontrivial_field():
    assert classify_stage2(_base())["outcome"] == "RECOVERED_NONTRIVIAL_FIELD"


def test_integrity_failure_short_circuits_everything():
    assert classify_stage2(_base(integrity_ok=False))["outcome"] == "FAIL_CLOSED"


def test_training_gain_without_heldout_gain_is_simulator_overfit():
    """The one the earlier draft was missing: fitting the simulator's own noise."""
    r = classify_stage2(_base(train_delta=+0.30, heldout_delta=-0.01))
    assert r["outcome"] == "SIMULATOR_OVERFIT"


def test_overfit_outranks_a_failing_bidirectional_gate():
    r = classify_stage2(_base(train_delta=+0.30, heldout_delta=-0.01,
                              bidirectional_gate=dict(passed=False)))
    assert r["outcome"] == "SIMULATOR_OVERFIT"


def test_a_failing_bidirectional_gate_caps_the_claim():
    r = classify_stage2(_base(bidirectional_gate=dict(passed=False)))
    assert r["outcome"] == "ONE_DIRECTION_ONLY"
    assert "one propagation direction" in r["allowed_statement"]
    assert "repertoire" not in r["allowed_statement"]


def test_dissimilar_equivalent_optima_are_field_nonidentifiable():
    """Different shapes with the same held-out score: the score does not pin the
    field, so no single solution may be shown (spec 8.2)."""
    r = classify_stage2(_base(family=dict(median_field_corr=0.31, n_members=6)))
    assert r["outcome"] == "FIELD_NONIDENTIFIABLE"


def test_unstable_fields_across_restarts_are_unidentifiable():
    r = classify_stage2(_base(restart_field_corr_median=0.22))
    assert r["outcome"] == "UNIDENTIFIABLE"


def test_never_beating_pure_geometry_is_axis_only_sufficient():
    r = classify_stage2(_base(vs_axis_only=dict(mean=-0.04, n_above=3, n=12)))
    assert r["outcome"] == "AXIS_ONLY_SUFFICIENT"


def test_beating_geometry_but_not_the_uniform_corridor():
    r = classify_stage2(_base(vs_uniform=dict(mean=-0.01, n_above=4, n=12)))
    assert r["outcome"] == "UNIFORM_CORRIDOR_SUFFICIENT"


def test_losing_to_the_manual_heuristic_is_reported_as_such():
    r = classify_stage2(_base(vs_manual_projected=dict(mean=-0.09, n_above=1, n=12)))
    assert r["outcome"] == "MANUAL_HEURISTIC_RETAINED"


def test_a_low_coverage_win_is_not_allowed_to_count():
    r = classify_stage2(_base(coverage=dict(learned=0.40, manual_smooth=0.60, margin=0.10)))
    assert r["outcome"] == "LOW_COVERAGE_WIN"


def test_classify_is_a_pure_function():
    d = _base()
    import copy
    snap = copy.deepcopy(d)
    classify_stage2(d)
    assert d == snap


def test_every_outcome_carries_an_allowed_statement():
    for name in OUTCOME_ORDER:
        assert name in classify_stage2.ALLOWED_STATEMENTS
        assert classify_stage2.ALLOWED_STATEMENTS[name]


def test_equivalent_optimum_family_takes_everything_within_one_paired_sd():
    scores = np.array([0.50, 0.49, 0.48, 0.20, 0.10])
    fields = [np.array([1.0, 0, 0]), np.array([0.9, 0.1, 0]),
              np.array([0.8, 0.2, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])]
    fam = equivalent_optimum_family(scores, fields, paired_sd=0.05)
    assert fam["n_members"] == 3
    assert fam["median_field_corr"] > 0.5


def test_equivalent_optimum_family_flags_dissimilar_members():
    scores = np.array([0.50, 0.49, 0.48])
    fields = [np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0])]
    fam = equivalent_optimum_family(scores, fields, paired_sd=0.05)
    assert fam["n_members"] == 3
    assert fam["median_field_corr"] < 0.5
