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
    assert out["spatial_instability_tested"] is False
    assert out["morphology_tested"] is False
    assert out["canonical_verdict"] == "GX1_MECHANISM_MAP_ACCEPTED"
    assert out["preregistered_next_hypothesis"] == "CAUSAL_2X2_D_GATE_BY_SHARED_X_H_PATH"
    assert out["authorized_next_program"] == "LC3_DX_STATE_PLANE_AND_SPATIAL_INSTABILITY_AUDIT"
    assert out["entry_geometry"]["same_D_bistability_required_for_lifecycle"] is False
    assert out["entry_geometry"]["explicit_d_gate_status"] == \
        "DEFERRED_PENDING_LC3_CURRENT_EQUATION_AUDIT"
    assert out["x_authority"]["coupled_D_X_offset_status"] == "UNTESTED"
    assert "D_SELECTIVE_ONSET_CANDIDATE" in out["mechanism_map_labels"]


def test_entry_summary_does_not_upgrade_one_way_ignition_to_dual_basin():
    g = _module()
    def arm(name, label):
        return {"arm": name, "workpoint_label": label}
    strip = {
        "verdict": "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP",
        "n_window_points": 0,
        "point_rows": [{
            "point_id": "H1_ts1.25_r025",
            "arms": [arm("healthy_low", "INTERICTAL_WORKPOINT"),
                     arm("susceptible_low", "FINITE_HIGH_ORBIT"),
                     arm("susceptible_high", "FINITE_HIGH_FIXED")],
        }],
    }
    out = g.summarize_entry_geometry(strip)
    assert out["component_label"] == "D_SELECTIVE_ONE_WAY_IGNITION_WITHOUT_DUAL_BASIN"
    assert out["natural_dual_basin_window"] is False
    assert out["explicit_d_gate_status"].endswith("NOT_PROVEN_SUFFICIENT")


def _xmap():
    return {
        "returning_availabilities": [0.0, 0.1],
        "rows": [
            {"x_availability": 1.0, "required_low_workpoint_label": "FINITE_HIGH_ORBIT"},
            {"x_availability": 0.5, "required_low_workpoint_label": "FINITE_HIGH_FIXED"},
            {"x_availability": 0.1, "required_low_workpoint_label": "INTERICTAL_WORKPOINT"},
            {"x_availability": 0.0, "required_low_workpoint_label": "INTERICTAL_WORKPOINT"},
        ],
    }


def _forks_map(point_id="H6_k05_r10", labels=("FINITE_HIGH_ORBIT", "FINITE_HIGH_FIXED")):
    return {"rows": [
        {"candidate_run_id": point_id, "arm": "C", "x_availability": 1.0,
         "required_low_workpoint_label": "FINITE_HIGH_FIXED",
         "state_tail_1s": {"rate_mean_hz": 101.6}},
        {"candidate_run_id": point_id, "arm": "D1", "x_availability": 0.872,
         "required_low_workpoint_label": labels[0], "state_tail_1s": {"rate_mean_hz": 97.6}},
        {"candidate_run_id": point_id, "arm": "D2", "x_availability": 0.786,
         "required_low_workpoint_label": labels[1], "state_tail_1s": {"rate_mean_hz": 94.9}},
        {"candidate_run_id": "other", "arm": "D1", "x_availability": 0.872,
         "required_low_workpoint_label": "INTERICTAL_WORKPOINT",
         "state_tail_1s": {"rate_mean_hz": 0.1}},
    ]}


def test_archived_relay_loads_are_read_from_the_matching_anchor_only():
    g = _module()
    loads = g.archived_relay_loads(_forks_map(), "H6_k05_r10")
    assert [d["x_availability"] for d in loads] == [0.872, 0.786]   # x=1 is not a load arm
    assert all(d["returned_to_interictal"] is False for d in loads)


def test_x_summary_reports_reachable_bracket_without_physiology_claim():
    g = _module()
    out = g.summarize_x_authority(_xmap(), g.archived_relay_loads(_forks_map(), "H6_k05_r10"))
    assert out["current_x_path_reachable"] is True
    assert out["h_actuator_bypasses_x"] is False
    assert out["experimental_return_bracket"] == [0.1, 0.5]
    assert out["physiological_validity_of_returning_probe"] == "NOT_ESTABLISHED"
    assert out["archived_range_status"] == "INSUFFICIENT_FOR_THIS_H_BRANCH"


def test_archived_range_status_follows_the_archive_not_a_constant():
    g = _module()
    sufficient = _forks_map(labels=("INTERICTAL_WORKPOINT", "FINITE_HIGH_FIXED"))
    out = g.summarize_x_authority(_xmap(), g.archived_relay_loads(sufficient, "H6_k05_r10"))
    assert out["archived_range_status"] == "SUFFICIENT_FOR_THIS_H_BRANCH"
    empty = g.summarize_x_authority(_xmap(), [])
    assert empty["archived_range_status"] == "NO_ARCHIVED_LOAD_AT_THIS_ANCHOR"


def test_strip_resolution_flags_the_pinned_gate_arm():
    """A pinned-open gate means the arm resolves rho only, so it must not be counted as 12 conditions."""
    g = _module()
    def cell(arm, rho, rate, gate):
        return {"arm": arm, "rho": rho, "gH_trace": [rho * gate, rho * gate],
                "state_tail_1s": {"rate_mean_hz": rate}}
    strip = {"point_rows": [
        {"arms": [cell("healthy_low", 0.54, 4.2, 0.0), cell("susceptible_low", 0.54, 54.8, 0.0),
                  cell("susceptible_high", 0.54, 58.66, 1.0)]},
        {"arms": [cell("healthy_low", 0.54, 35.6, 0.7), cell("susceptible_low", 0.54, 62.1, 0.5),
                  cell("susceptible_high", 0.54, 58.66, 1.0)]},
    ]}
    out = g.summarize_strip_resolution(strip)
    assert out["gate_pinned_arms"] == ["susceptible_high"]
    assert out["per_arm"]["susceptible_high"]["n_points"] == 2
    assert out["per_arm"]["susceptible_high"]["n_distinct_tail_rates"] == 1
    assert "healthy_low" in out["arms_that_resolve_tau_and_theta"]


def test_x_initial_condition_records_the_head_start_gap():
    g = _module()
    xmap = {"rows": [{"x_availability": 1.0, "theta": 1.1123, "tau_ms": 632.4555,
                      "h_init_scale": 2.0, "h_trace": [2.2245, 6.8167], "T_ms": 5059.6,
                      "post_offset_required_ms": 2000.0}]}
    out = g.summarize_x_initial_condition(xmap)
    assert out["head_start_ratio"] > 3.0
    assert 0.6 < out["extra_above_theta_decay_s_if_started_converged"] < 0.8
    assert out["margin_ok"] is True
