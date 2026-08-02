import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location(
        "finalize_topic4_fcxr_lc2_gx1",
        ROOT / "scripts" / "finalize_topic4_fcxr_lc2_gx1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_conditional_logic_covers_the_preregistered_two_by_two():
    g = _module()
    assert g.choose_next_hypothesis(
        "NATURAL_SELECTIVITY_WINDOW_CANDIDATE",
        "X_PATH_REACHABLE_RANGE_INSUFFICIENT") == "KEEP_H_EQUATION_CALIBRATE_X_RANGE"
    assert g.choose_next_hypothesis(
        "NATURAL_SELECTIVITY_WINDOW_CANDIDATE",
        "H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN") == "SHARED_PATH_X_H_COUPLING_ONLY"
    assert g.choose_next_hypothesis(
        "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP",
        "X_PATH_REACHABLE_RANGE_INSUFFICIENT") == "LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE"
    assert g.choose_next_hypothesis(
        "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP",
        "H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN") == "CAUSAL_2X2_D_GATE_BY_SHARED_X_H_PATH"


def test_unresolved_measurement_never_authorizes_structure():
    g = _module()
    assert g.choose_next_hypothesis(
        "SELECTIVITY_STRIP_NUMERICAL_FAILURE",
        "X_AUTHORITY_UNRESOLVED") == "MEASUREMENT_REPAIR_NO_STRUCTURAL_CLAIM"


def test_candidate_verdict_keeps_dynamic_and_morphology_unclaimed():
    g = _module()
    arm = {"numerical_failure": False}
    strip = dict(verdict="NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP",
                 n_rows=3, n_points=1, n_pass=0, n_window_points=0,
                 point_rows=[dict(arms=[arm, arm, arm])])
    xmap = dict(verdict="H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN",
                n_rows=4, rows=[arm, arm, arm, arm])
    out = g.build_candidate_verdict(strip, xmap)
    assert out["numerical_safe_rows"] == 7
    assert out["dynamic_lifecycle_tested"] is False
    assert out["morphology_tested"] is False
    assert out["authorized_next_hypothesis"] == "CAUSAL_2X2_D_GATE_BY_SHARED_X_H_PATH"
