"""Contract-layer TDD for the M3A <-> M3B-R2 interface gate.

Canonical contract: docs/superpowers/specs/2026-06-27-sef-hfo-m3-interface-contract.md
Module under test:  src/sef_hfo_m3_interface.py

Every test here is CONTRACT-LAYER: it asserts on pure JSON/CSV-shaped data (dicts / list[dict])
with NO SNN run. The single shared module is imported by BOTH the M3A exporter and the M3B
axis-builder/overlay so the two lines cannot drift. Runner-layer tests (need the SNN) live in the
M3A-A2 worktree and are listed in the contract doc §7 — they are deliberately NOT faked here.

Each test names the plan test(s) it implements and the blocker/finding it closes.
"""
import math

import pytest

import src.sef_hfo_m3_interface as itf


# ---------------------------------------------------------------------------
# Fixture builders: a fully valid handoff, then mutate for fail-closed cases.
# ---------------------------------------------------------------------------
def _coord(input_var, ttype="reciprocal_affine", a=0.2, b=0.0,
           dirn="decreasing_in_input", input_min=0.2, input_max=1.0,
           variables=None, calib="passed", shunt=None, sign_pass=True,
           drop_sign_tests=False):
    variables = variables or [input_var]
    slope = -1 if dirn == "decreasing_in_input" else 1
    sign_tests = [] if drop_sign_tests else [{
        "name": f"{input_var}_{dirn}", "coord": input_var, "input_var": input_var,
        "expected_direction": dirn, "observed_slope_sign": slope,
        "passed": sign_pass, "engine_sha": "deadbeef",
    }]
    d = {
        "transform": {"type": ttype, "input_var": input_var, "a": a, "b": b,
                      "clip": [0.0, 1.0], "input_min": input_min, "input_max": input_max,
                      "expected_direction": dirn},
        "units": "dimensionless", "valid_range": [0.0, 1.0], "variables": variables,
        "calibration_status": calib, "sign_tests": sign_tests,
    }
    if shunt is not None:
        d["shunt_path_active"] = shunt
    return d


def _mapping(mid="m3a_a1_20260627_abc", **overrides):
    m = {
        "slow_to_rate_mapping_id": mid,
        "source": "M3A-A1 quasi-static SNN calibration",
        "substrate": "stage3_twoend_equal",
        "axis_space": "normalized_unit",
        "two_core_reduction": "source_core",
        "coordinates": {
            "phase_x_core": _coord("q_core"),
            "phase_y_global": _coord("q_global"),
            "phase_recovery": _coord("phi", ttype="affine", a=0.5, b=0.0,
                                     dirn="increasing_in_input", input_min=0.0,
                                     input_max=2.0, variables=["phi", "g_K"]),
        },
    }
    m.update(overrides)
    return m


def _ranges(mid="m3a_a1_20260627_abc"):
    return {
        "slow_to_rate_mapping_id": mid,
        "phase_x_core": {"min": 0.0, "max": 1.0, "source": "A1 sweep"},
        "phase_y_global": {"min": 0.0, "max": 1.0, "source": "A1 sweep"},
        "phase_recovery": {"min": 0.0, "max": 1.0, "source": "A1 sweep"},
    }


def _traj_rows(mid="m3a_a1_20260627_abc", n=10, out_frac=0.0):
    n_out = int(round(out_frac * n))
    rows = []
    for i in range(n):
        rows.append({
            "time_ms": float(i), "event_id": -1, "event_stage": "inter_event",
            "phase_x_core": 0.5, "phase_y_global": 0.5, "phase_recovery": 0.5,
            "phase_coord_valid": True, "phase_coord_out_of_range": (i < n_out),
            "slow_to_rate_mapping_id": mid,
        })
    return rows


def _evt_rows(mid="m3a_a1_20260627_abc"):
    return [{
        "event_id": 0, "event_stage": "onset",
        "phase_x_core": 0.5, "phase_y_global": 0.5, "phase_recovery": 0.5,
        "phase_coord_valid": True, "phase_coord_out_of_range": False,
        "slow_to_rate_mapping_id": mid, "return_to_baseline": True,
        "tail_to_baseline_ratio": 0.1, "R_class": "R3",
    }]


def _summary(mid="m3a_a1_20260627_abc", gate_A="PASS", gate_B="FAIL",
             rate_matched="passed", robustness="robust", **overrides):
    s = {
        "slow_to_rate_mapping_id": mid, "gate_A_trajectory": gate_A,
        "gate_B_seizure_like": gate_B, "trajectory_robustness": robustness,
        "rate_matched_control": rate_matched, "out_of_range_fraction": 0.0,
        "rate_matched_group": "matched_peak_rate", "forbidden_claims": [],
        "m3b_ready": True, "m3b_ready_reason": "calibrated + gate_A PASS",
    }
    s.update(overrides)
    return s


def _axes_meta(mid="m3a_a1_20260627_abc", mapping=None):
    mapping = mapping or _mapping(mid)
    return {
        "axes_built_from_slow_to_rate_mapping_id": mid,
        "axis_space": "normalized_unit",
        "axis_transforms": {c: dict(mapping["coordinates"][c]["transform"])
                            for c in itf.ON_AXIS_COORDS},
    }


def _audit(**over):
    """Build an audit by running the real auditor on a (mutable) valid handoff."""
    mid = over.pop("mid", "m3a_a1_20260627_abc")
    mapping = over.pop("mapping", _mapping(mid))
    ranges = over.pop("ranges", _ranges(mid))
    trajectory = over.pop("trajectory", _traj_rows(mid))
    summary = over.pop("summary", _summary(mid))
    # lazy: don't build the default axes_meta (which dereferences mapping) when one is supplied
    axes_meta = over.pop("axes_meta") if "axes_meta" in over else _axes_meta(mid, mapping)
    return itf.audit_m3a_interface(mapping=mapping, ranges=ranges,
                                   trajectory_rows=trajectory, summary=summary,
                                   axes_meta=axes_meta, **over)


# ===========================================================================
# Mapping schema + closed-enum transform  (A1: test_slow_to_rate_mapping_schema_required_keys,
# test_slow_state_to_phase_coords_transform_is_documented; closes B3, M9)
# ===========================================================================
def test_slow_to_rate_mapping_schema_required_keys():
    itf.validate_slow_to_rate_mapping(_mapping())  # valid passes
    bad = _mapping()
    del bad["coordinates"]["phase_y_global"]["valid_range"]
    with pytest.raises(ValueError):
        itf.validate_slow_to_rate_mapping(bad)


def test_slow_state_to_phase_coords_transform_is_closed_enum():
    # free-text / unknown transform type is rejected (no eval of a formula string)
    bad = _mapping()
    bad["coordinates"]["phase_x_core"]["transform"]["type"] = "1/q_core  # TODO"
    with pytest.raises(ValueError):
        itf.validate_slow_to_rate_mapping(bad)
    assert "1/q_core  # TODO" not in itf.TRANSFORM_TYPES


def test_evaluate_phase_coord_is_normalized_and_deterministic():
    m = _mapping()
    hi = itf.evaluate_phase_coord(m, "phase_y_global", {"q_global": 0.2})
    lo = itf.evaluate_phase_coord(m, "phase_y_global", {"q_global": 1.0})
    assert 0.0 <= lo <= hi <= 1.0
    assert hi == pytest.approx(1.0) and lo == pytest.approx(0.2)


# ===========================================================================
# Signed sign-direction (NOT monotonicity)  (A1: test_q_*_maps_monotonically_*; closes B4)
# ===========================================================================
def test_phase_y_global_strictly_decreasing_in_q_global_signed():
    assert itf.check_sign_direction(_mapping(), "phase_y_global") is True
    # backwards axis: declared decreasing but transform actually increases -> caught
    flipped = _mapping()
    flipped["coordinates"]["phase_y_global"]["transform"]["type"] = "affine"
    flipped["coordinates"]["phase_y_global"]["transform"]["a"] = 1.0  # affine a>0 => INCREASING
    assert itf.check_sign_direction(flipped, "phase_y_global") is False
    with pytest.raises(ValueError):
        itf.validate_slow_to_rate_mapping(flipped)


def test_phase_x_core_strictly_decreasing_in_q_core_signed():
    assert itf.check_sign_direction(_mapping(), "phase_x_core") is True


def test_phase_recovery_signed_direction_increasing_in_phi():
    assert itf.check_sign_direction(_mapping(), "phase_recovery") is True
    flipped = _mapping()
    flipped["coordinates"]["phase_recovery"]["transform"]["a"] = -0.5  # now decreasing
    assert itf.check_sign_direction(flipped, "phase_recovery") is False


def test_recovery_variables_carry_suppressive_sign():
    # phi (a recovery var) must NOT appear on a disinhibition axis
    bad = _mapping()
    bad["coordinates"]["phase_y_global"]["variables"] = ["q_global", "phi"]
    with pytest.raises(ValueError):
        itf.validate_slow_to_rate_mapping(bad)


# ===========================================================================
# Fail-closed calibration / sign_tests  (A1: test_mapping_sign_tests_fail_closed,
# test_uncalibrated_variable_exports_phase_coord_invalid; closes B3)
# ===========================================================================
def test_calibration_passed_requires_nonempty_all_pass_sign_tests():
    # calibration says passed but sign_tests empty -> schema violation AND predicate False
    bad = _mapping()
    bad["coordinates"]["phase_x_core"] = _coord("q_core", drop_sign_tests=True, calib="passed")
    with pytest.raises(ValueError):
        itf.validate_slow_to_rate_mapping(bad)
    assert itf.mapping_sign_tests_passed(bad, "phase_x_core") is False


def test_mapping_sign_tests_fail_closed_on_not_applicable_or_failed():
    m = _mapping()
    m["coordinates"]["phase_x_core"]["calibration_status"] = "not_applicable"
    assert itf.mapping_sign_tests_passed(m, "phase_x_core") is False
    m2 = _mapping()
    m2["coordinates"]["phase_y_global"]["sign_tests"][0]["passed"] = False
    assert itf.mapping_sign_tests_passed(m2, "phase_y_global") is False


def test_uncalibrated_variable_exports_phase_coord_invalid():
    m = _mapping()
    m["coordinates"]["phase_x_core"]["calibration_status"] = "failed"
    # a sample that uses the (now uncalibrated) x-core axis is invalid
    valid = itf.sample_phase_coord_valid(m, _ranges(), {"q_core": 0.5, "q_global": 0.5},
                                         axes_used=("phase_x_core", "phase_y_global"))
    assert valid is False


def test_e_gaba_axis_fails_closed_without_shunt_path():
    m = _mapping()
    m["coordinates"]["phase_y_global"] = _coord("e_GABA", shunt=False, calib="passed",
                                                variables=["e_GABA"])
    # depolarized e_GABA is disinhibition ONLY in the shunt path -> shunt off => not trustworthy
    assert itf.mapping_sign_tests_passed(m, "phase_y_global") is False
    # positive control (H4): shunt on + calibrated + monotone -> trusted (pins the shunt read both ways)
    m_ok = _mapping()
    m_ok["coordinates"]["phase_y_global"] = _coord("e_GABA", shunt=True, variables=["e_GABA"])
    assert itf.mapping_sign_tests_passed(m_ok, "phase_y_global") is True


def test_validate_rejects_e_gaba_schema_violations():
    # (H3) e_GABA without a recorded shunt_path_active -> reject
    m = _mapping()
    m["coordinates"]["phase_y_global"] = _coord("e_GABA", variables=["e_GABA"])  # no shunt key
    with pytest.raises(ValueError):
        itf.validate_slow_to_rate_mapping(m)
    # e_GABA claiming calibrated disinhibition while shunt is off -> reject
    m2 = _mapping()
    m2["coordinates"]["phase_y_global"] = _coord("e_GABA", variables=["e_GABA"], shunt=False,
                                                 calib="passed")
    with pytest.raises(ValueError):
        itf.validate_slow_to_rate_mapping(m2)
    # positive control: shunt on validates clean
    m3 = _mapping()
    m3["coordinates"]["phase_y_global"] = _coord("e_GABA", variables=["e_GABA"], shunt=True)
    itf.validate_slow_to_rate_mapping(m3)


# ===========================================================================
# out-of-range against INPUT domain (closes M16) + ranges id (A1: ranges_reference_same_mapping_id)
# ===========================================================================
def test_coord_out_of_range_checks_input_domain_even_when_output_clipped():
    m, r = _mapping(), _ranges()
    assert itf.coord_out_of_range(m, r, "phase_y_global", {"q_global": 0.5}) is False
    # q_global below input_min (0.2): clipped output still lands in [0,1] but input extrapolates
    assert itf.coord_out_of_range(m, r, "phase_y_global", {"q_global": 0.05}) is True


def test_phase_coord_ranges_reference_same_mapping_id():
    itf.validate_phase_coord_ranges(_ranges("X"), mapping_id="X")
    with pytest.raises(ValueError):
        itf.validate_phase_coord_ranges(_ranges("X"), mapping_id="Y")


# ===========================================================================
# mapping_id consistency  (closes B5)
# ===========================================================================
def test_mapping_id_identical_across_all_artifacts():
    mid = "m3a_a1_20260627_abc"
    got = itf.assert_mapping_id_consistent(_mapping(mid), _ranges(mid), _traj_rows(mid),
                                           _evt_rows(mid), _summary(mid), _axes_meta(mid))
    assert got == mid
    with pytest.raises(ValueError):
        itf.assert_mapping_id_consistent(_mapping(mid), _ranges("OTHER"))


# ===========================================================================
# trajectory / event / summary schema + canonical enums  (A2-1b/1c; closes M2,M3,M4,m2)
# ===========================================================================
def test_event_stage_values_in_canonical_enum():
    itf.validate_phase_trajectory(_traj_rows())
    bad = _traj_rows()
    bad[0]["event_stage"] = "ramp_up"  # not canonical
    with pytest.raises(ValueError):
        itf.validate_phase_trajectory(bad)
    assert "inter_event" in itf.canonical_event_stages()
    assert "baseline" in itf.canonical_event_stages()


def test_phase_trajectory_requires_out_of_range_flag_column():
    bad = _traj_rows()
    del bad[0]["phase_coord_out_of_range"]
    with pytest.raises(ValueError):
        itf.validate_phase_trajectory(bad)


def test_return_to_baseline_field_name_canonical():
    itf.validate_event_phase_samples(_evt_rows())
    bad = _evt_rows()
    bad[0]["returned"] = bad[0].pop("return_to_baseline")  # legacy name only
    with pytest.raises(ValueError):
        itf.validate_event_phase_samples(bad)


def test_phase_export_writes_NA_not_zero_for_disabled_mechanisms():
    # disabled g_K must be NA, not 0.0; and a derived coord with all-disabled contributors is NA
    good = {"g_K": itf.NA_SENTINEL, "phase_recovery": itf.NA_SENTINEL}
    itf.assert_disabled_mechanisms_na(good, disabled_vars=["g_K", "phi"],
                                      derived={"phase_recovery": ["phi", "g_K"]})
    bad = {"g_K": 0.0, "phase_recovery": itf.NA_SENTINEL}
    with pytest.raises(ValueError):
        itf.assert_disabled_mechanisms_na(bad, disabled_vars=["g_K", "phi"],
                                          derived={"phase_recovery": ["phi", "g_K"]})
    # isolate the DERIVED branch (H2): raw vars NA, only the derived coord wrongly 0.0
    bad2 = {"g_K": itf.NA_SENTINEL, "phi": itf.NA_SENTINEL, "phase_recovery": 0.0}
    with pytest.raises(ValueError):
        itf.assert_disabled_mechanisms_na(bad2, disabled_vars=["g_K", "phi"],
                                          derived={"phase_recovery": ["phi", "g_K"]})


def test_dynamic_summary_gate_fields_in_enum():
    itf.validate_dynamic_slowvars_summary(_summary())
    bad = _summary(gate_A="maybe")
    with pytest.raises(ValueError):
        itf.validate_dynamic_slowvars_summary(bad)


def test_rate_matched_group_present_when_gate_A_claimed():
    bad = _summary(gate_A="PASS", rate_matched="not_run")
    with pytest.raises(ValueError):
        itf.validate_dynamic_slowvars_summary(bad)
    bad2 = _summary(gate_A="PASS")
    del bad2["rate_matched_group"]
    with pytest.raises(ValueError):
        itf.validate_dynamic_slowvars_summary(bad2)


def test_phase_trajectory_requires_per_core_q_and_reduction_when_two_core():
    mid = "two"
    rows = _traj_rows(mid)
    for r in rows:
        r["q_core_L"], r["q_core_R"] = 0.6, 0.7
    m = _mapping(mid)
    itf.validate_phase_trajectory(rows, two_core=True, mapping=m)
    # missing reduction rule -> reject
    m_no_rule = _mapping(mid)
    del m_no_rule["two_core_reduction"]
    with pytest.raises(ValueError):
        itf.validate_phase_trajectory(rows, two_core=True, mapping=m_no_rule)
    # missing per-core columns -> reject
    rows_flat = _traj_rows(mid)
    with pytest.raises(ValueError):
        itf.validate_phase_trajectory(rows_flat, two_core=True, mapping=m)


# ===========================================================================
# overlay verdict truth table  (closes B1)
# ===========================================================================
def test_compute_overlay_verdict_truth_table():
    V = itf.compute_overlay_verdict
    assert V(True, True, True, True) == "phase_map_trajectory"
    # phenotype real but calibration/provenance/range incomplete -> candidate, not trajectory
    assert V(False, True, True, True) == "mechanism_candidate_only"
    assert V(True, False, True, True) == "mechanism_candidate_only"
    assert V(True, True, False, True) == "mechanism_candidate_only"
    # no phenotype movement -> refused
    assert V(True, True, True, False) == "refused"
    # missing condition (None) is fail-closed
    assert V(None, True, True, True) == "mechanism_candidate_only"
    assert V(True, True, True, None) == "refused"


def test_m3a_interface_audit_records_four_conditions_and_verdict():
    a = _audit()
    for k in ("cond1_sign_tests_passed", "cond2_same_mapping_and_ranges",
              "cond3_in_range_or_flagged", "cond4_phenotype_movement_beyond_rate"):
        assert isinstance(a[k], bool)
    assert a["overlay_verdict"] in itf.OVERLAY_VERDICT_VALUES
    assert a["overlay_allowed"] == (a["overlay_verdict"] == "phase_map_trajectory")
    assert a["overlay_verdict"] == "phase_map_trajectory"
    assert a["gate_used"] == "A"
    assert a["on_axis_coords"] == ["phase_x_core", "phase_y_global"]
    assert a["projected_out_coords"] == ["phase_recovery"]
    itf.validate_interface_audit(a)


def test_audit_always_returns_dict_failclosed_on_type_confused_inputs():
    # the audit must never CRASH on garbage; it returns a refused/candidate verdict (never a trajectory).
    # deep type-confusion: a present key whose value is None/scalar, or non-dict rows.
    bad_mappings = [None, "str", {"slow_to_rate_mapping_id": "m", "axis_space": "normalized_unit",
                                  "coordinates": None}]
    for bm in bad_mappings:
        a = _audit(mapping=bm, axes_meta=_axes_meta())  # explicit axes_meta: exercise the module, not the helper
        assert isinstance(a, dict) and a["overlay_verdict"] != "phase_map_trajectory"
    for bad_traj in (None, [None, None], [5, 6]):
        a = _audit(trajectory=bad_traj)
        assert isinstance(a, dict) and a["cond3_in_range_or_flagged"] is False
    for bad_sum in (None, "str", {"gate_A_trajectory": None}):
        a = _audit(summary=bad_sum)
        assert isinstance(a, dict) and a["cond4_phenotype_movement_beyond_rate"] is False


def test_interface_audit_schema_rejects_missing_condition():
    a = _audit()
    del a["cond3_in_range_or_flagged"]  # a missing condition must NOT default to true
    with pytest.raises(ValueError):
        itf.validate_interface_audit(a)


# ===========================================================================
# present-but-failed refusals  (closes M14) + strict cond4 (B6) + cond2 identity (B5/B8)
# ===========================================================================
def test_overlay_refuses_present_but_failed_sign_tests():
    m = _mapping()
    m["coordinates"]["phase_x_core"]["sign_tests"][0]["passed"] = False
    a = _audit(mapping=m, axes_meta=_axes_meta(mapping=m))
    assert a["cond1_sign_tests_passed"] is False
    assert a["overlay_verdict"] != "phase_map_trajectory"


def test_condition4_strict_equality_refuses_not_run_inconclusive_missing():
    assert _audit(summary=_summary(gate_A="INCONCLUSIVE"))["cond4_phenotype_movement_beyond_rate"] is False
    assert _audit(summary=_summary(rate_matched="not_run"))["cond4_phenotype_movement_beyond_rate"] is False
    s = _summary()
    del s["rate_matched_control"]
    assert _audit(summary=s)["cond4_phenotype_movement_beyond_rate"] is False


def test_overlay_mapping_id_and_transform_descriptor_must_match_axes():
    # id mismatch
    a = _audit(axes_meta=_axes_meta("DIFFERENT"))
    assert a["cond2_same_mapping_and_ranges"] is False
    # id matches but the M3B axis transform coefficient differs from the mapping -> still refused
    am = _axes_meta()
    am["axis_transforms"]["phase_x_core"]["a"] = 0.9
    a2 = _audit(axes_meta=am)
    assert a2["cond2_same_mapping_and_ranges"] is False


def test_phase_trajectory_samples_in_map_or_refused_by_fraction():
    assert _audit(trajectory=_traj_rows(n=20, out_frac=0.0))["cond3_in_range_or_flagged"] is True
    # pin the exact 0.05 boundary in BOTH directions (H1): 1/20==0.05 and 1/40==0.025 pass; 2/20==0.10 fails
    assert _audit(trajectory=_traj_rows(n=20, out_frac=0.05))["cond3_in_range_or_flagged"] is True
    assert _audit(trajectory=_traj_rows(n=40, out_frac=0.025))["cond3_in_range_or_flagged"] is True
    assert _audit(trajectory=_traj_rows(n=20, out_frac=0.10))["cond3_in_range_or_flagged"] is False
    # missing flag column -> schema-invalid -> cond3 fails closed
    rows = _traj_rows()
    for r in rows:
        del r["phase_coord_out_of_range"]
    assert _audit(trajectory=rows)["cond3_in_range_or_flagged"] is False


def test_audit_refuses_all_invalid_trajectory_even_if_in_range():
    # every row untrustworthy (phase_coord_valid False) but 0% out-of-range: invalid samples count
    # as refused, NOT silently dropped (contract §5 M1 / F9)
    rows = _traj_rows()
    for r in rows:
        r["phase_coord_valid"] = False
    a = _audit(trajectory=rows)
    assert a["cond3_in_range_or_flagged"] is False
    assert a["overlay_verdict"] != "phase_map_trajectory"


def test_audit_refuses_trajectory_missing_phase_coords():
    # a trajectory with no phase coordinates must never be stamped phase_map_trajectory (F1)
    rows = [{"slow_to_rate_mapping_id": "m3a_a1_20260627_abc", "phase_coord_out_of_range": False}
            for _ in range(5)]
    a = _audit(trajectory=rows)
    assert a["cond3_in_range_or_flagged"] is False
    assert a["overlay_verdict"] != "phase_map_trajectory"


def test_audit_refuses_non_normalized_axis_space():
    # a consistently-declared raw-unit axis_space still violates the D1 lock (F5)
    m = _mapping()
    m["axis_space"] = "raw_physical_units"
    am = {"axes_built_from_slow_to_rate_mapping_id": m["slow_to_rate_mapping_id"],
          "axis_space": "raw_physical_units",
          "axis_transforms": {c: dict(m["coordinates"][c]["transform"]) for c in itf.ON_AXIS_COORDS}}
    assert _audit(mapping=m, axes_meta=am)["overlay_verdict"] != "phase_map_trajectory"


def test_audit_refuses_summary_missing_rate_matched_group():
    # gate_A==PASS without the rate_matched_group recorded is schema-invalid -> cond4 fails closed (F8)
    s = _summary(gate_A="PASS")
    del s["rate_matched_group"]
    assert _audit(summary=s)["cond4_phenotype_movement_beyond_rate"] is False


# ===========================================================================
# overlay output  (closes B2, M18) + structural no-claim (M14)
# ===========================================================================
def test_overlay_csv_has_full_min_columns_plus_flags():
    a = _audit()
    rows = itf.build_slow_trajectory_overlay(_traj_rows(), a, readout_fn=lambda r: ("axial", -0.2))
    assert len(rows) == 10
    need = set(itf.required_min_columns()) | {
        "phase_coord_out_of_range", "slow_to_rate_mapping_id", "in_map",
        "leading_mode_class", "alpha_1"}
    assert need.issubset(rows[0].keys())
    # lossless: all three phase coords present
    for c in ("phase_x_core", "phase_y_global", "phase_recovery"):
        assert c in rows[0]
    # readout + in_map are actually computed, not ignored (H6)
    assert rows[0]["leading_mode_class"] == "axial"
    assert rows[0]["alpha_1"] == pytest.approx(-0.2)
    assert rows[0]["in_map"] is True
    # an out-of-range sample (within the 5% gate) flips in_map False for that row
    big = _traj_rows(n=20, out_frac=0.05)            # exactly 1/20 out-of-range
    rows2 = itf.build_slow_trajectory_overlay(big, _audit(trajectory=big),
                                              readout_fn=lambda r: ("local", -0.1))
    oor = [r for r in rows2 if r["phase_coord_out_of_range"]]
    assert len(oor) == 1 and oor[0]["in_map"] is False


def test_m3a_overlay_refuses_missing_slow_traces():
    a = _audit()
    with pytest.raises(ValueError):
        itf.build_slow_trajectory_overlay([], a)


def test_m3a_overlay_refuses_when_verdict_not_trajectory():
    a = _audit(summary=_summary(gate_A="INCONCLUSIVE"))  # cond4 False -> refused
    assert a["overlay_verdict"] != "phase_map_trajectory"
    assert itf.build_slow_trajectory_overlay(_traj_rows(), a) == []  # no claim drawn


def test_overlay_refuses_forged_audit():
    # a truncated/forged audit (verdict string only, no condition booleans) must not emit (F2)
    with pytest.raises(ValueError):
        itf.build_slow_trajectory_overlay(_traj_rows(), {"overlay_verdict": "phase_map_trajectory"})


def test_overlay_rebinds_to_audited_rows_toctou():
    # audit a CLEAN trajectory, then try to draw a DIRTY (foreign-id) row set -> refused (F3)
    a = _audit()
    with pytest.raises(ValueError):
        itf.build_slow_trajectory_overlay(_traj_rows("OTHER_ID"), a)


def test_m3a_overlay_refuses_missing_slow_to_rate_mapping():
    # an audit produced from a missing mapping never reaches phase_map_trajectory
    a = itf.audit_m3a_interface(mapping=None, ranges=_ranges(), trajectory_rows=_traj_rows(),
                                summary=_summary(), axes_meta=_axes_meta())
    assert a["cond1_sign_tests_passed"] is False
    assert a["overlay_verdict"] != "phase_map_trajectory"


def test_phenotype_positive_without_mapping_is_mechanism_candidate_only():
    # phenotype real (gate_A PASS + rate matched) but no calibrated mapping -> candidate, not trajectory
    a = itf.audit_m3a_interface(mapping=None, ranges=_ranges(), trajectory_rows=_traj_rows(),
                                summary=_summary(gate_A="PASS", rate_matched="passed"),
                                axes_meta=_axes_meta())
    assert a["cond4_phenotype_movement_beyond_rate"] is True
    assert a["overlay_verdict"] == "mechanism_candidate_only"
    assert a["overlay_allowed"] is False


# ===========================================================================
# min-columns resolvable via documented join  (closes M5)
# ===========================================================================
def test_m3b_min_columns_resolvable_from_a2_artifacts():
    traj = _traj_rows()
    traj[0]["event_id"] = 0  # one row belongs to event 0
    traj[0]["event_stage"] = "onset"
    merged = itf.join_trajectory_and_event_samples(traj, _evt_rows())
    assert all(set(itf.required_min_columns()).issubset(r.keys()) for r in merged)
    ev_row = next(r for r in merged if r["event_id"] == 0)
    assert ev_row["R_class"] == "R3" and ev_row["return_to_baseline"] is True
    # inter-event rows carry NA for per-event fields, not a silent default
    inter = next(r for r in merged if r["event_id"] == -1)
    assert itf.is_na(inter["R_class"])


# ===========================================================================
# m3b_ready truth table (closes M15) + per-sample validity (M1) + tier guard (M13)
# ===========================================================================
def test_m3b_ready_flag_logic_truth_table():
    m = _mapping()
    assert itf.m3b_ready(_summary(gate_A="PASS", rate_matched="passed"), m)[0] is True
    assert itf.m3b_ready(_summary(gate_A="FAIL", rate_matched="passed"), m)[0] is False
    assert itf.m3b_ready(_summary(gate_A="PASS", rate_matched="not_run"), m)[0] is False
    m_unc = _mapping()
    m_unc["coordinates"]["phase_x_core"]["calibration_status"] = "failed"
    assert itf.m3b_ready(_summary(gate_A="PASS", rate_matched="passed"), m_unc)[0] is False


def test_per_sample_valid_is_and_of_axes_and_out_of_range_orthogonal():
    m, r = _mapping(), _ranges()
    # calibrated axes -> valid wherever the sample sits
    assert itf.sample_phase_coord_valid(m, r, {"q_core": 0.5, "q_global": 0.5},
                                        ("phase_x_core", "phase_y_global")) is True
    # an extrapolated sample is STILL valid (mapping is trusted) but is flagged out_of_range:
    # validity (calibration) and range are orthogonal — out_of_range must NOT force valid=False
    assert itf.sample_phase_coord_valid(m, r, {"q_core": 0.05, "q_global": 0.5},
                                        ("phase_x_core", "phase_y_global")) is True
    assert itf.coord_out_of_range(m, r, "phase_x_core", {"q_core": 0.05}) is True
    # an UNCALIBRATED axis makes the sample invalid (calibration gates, not range)
    m_bad = _mapping()
    m_bad["coordinates"]["phase_x_core"]["calibration_status"] = "failed"
    assert itf.sample_phase_coord_valid(m_bad, r, {"q_core": 0.5, "q_global": 0.5},
                                        ("phase_x_core", "phase_y_global")) is False


def test_gate_A_pass_not_reported_as_gate_B_seizure_claim():
    # overlay is Gate-A tier only; a Gate-B (seizure-like) claim needs gate_B PASS
    a = _audit(summary=_summary(gate_A="PASS", gate_B="FAIL"))
    assert a["gate_used"] == "A"
    assert itf.overlay_is_seizure_like_claim(a, _summary(gate_B="FAIL")) is False
    assert itf.overlay_is_seizure_like_claim(a, _summary(gate_B="PASS")) is True


# ===========================================================================
# classification crosswalk (closes M11)
# ===========================================================================
def test_classification_crosswalk_covers_all_vocabularies():
    cw = itf.classification_crosswalk()
    assert {"R_class", "phenotype_label", "mode_class"}.issubset(cw.keys())
    # every R_class member maps to a phenotype_label and a mode_class
    for rc in itf.R_CLASS_VALUES:
        assert rc in cw["R_class"]
