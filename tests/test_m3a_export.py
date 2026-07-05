"""M3A-A2 runner-layer export: build canonical handoff artifacts from a dynamic run.

This is the M3A side of the M3A<->M3B-R2 contract (src/sef_hfo_m3_interface.py
§6): it turns per-event landmark rows (from sample_event_landmarks) into the
canonical phase_trajectory schema, computes phase coords via the SHARED
evaluate_phase_coord transform, and self-audits with audit_m3a_interface so the
overlay is REFUSED until the mapping is calibrated.

Pure -- synthetic traces, no SNN. The science-gated pieces (lgr, tail absolute,
two_core_reduction, e_GABA axis -- contract §9) are deferred; this round proves
the wiring emits contract-valid artifacts that fail closed pre-calibration.
"""
import sys, os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.sef_hfo_a2 import sample_event_landmarks  # noqa: E402
from src.sef_hfo_m3_interface import (  # noqa: E402
    validate_phase_trajectory, validate_slow_to_rate_mapping, validate_interface_audit,
    validate_dynamic_slowvars_summary, validate_event_phase_samples, NA_SENTINEL,
)
from src.sef_hfo_m3a_export import (  # noqa: E402
    build_phase_trajectory_rows, build_self_audit,
    default_precalib_mapping_and_ranges, build_handoff_from_sim, write_handoff_artifacts,
    build_event_phase_samples, assemble_event_metrics,
)


def _coord(input_var, a, b, direction, imin, imax):
    return {
        "transform": {"type": "affine", "input_var": input_var, "a": a, "b": b,
                      "clip": [0.0, 1.0], "input_min": imin, "input_max": imax,
                      "expected_direction": direction},
        "units": "dimensionless", "valid_range": [0.0, 1.0], "variables": [input_var],
        "calibration_status": "not_applicable", "sign_tests": [],
    }


def _precalib_mapping():
    """Schema-valid mapping with sensible transforms but NOT yet calibrated."""
    return {
        "slow_to_rate_mapping_id": "m3a_a1_precalib",
        "source": "M3A-A2 dynamic run (pre-calibration)",
        "substrate": "stage3_twoend_equal",
        "axis_space": "normalized_unit",
        "coordinates": {
            "phase_x_core": _coord("q_core", -1.0, 1.0, "decreasing_in_input", 0.25, 1.0),
            "phase_y_global": _coord("q_global", -1.0, 1.0, "decreasing_in_input", 0.25, 1.0),
            "phase_recovery": _coord("g_K", 1.0, 0.0, "increasing_in_input", 0.0, 1.0),
        },
    }


def _ranges():
    return {
        "slow_to_rate_mapping_id": "m3a_a1_precalib",
        "phase_x_core": {"min": 0.0, "max": 1.0, "source": "A2 sweep"},
        "phase_y_global": {"min": 0.0, "max": 1.0, "source": "A2 sweep"},
        "phase_recovery": {"min": 0.0, "max": 1.0, "source": "A2 sweep"},
    }


def _summary():
    return {
        "slow_to_rate_mapping_id": "m3a_a1_precalib",
        "gate_A_trajectory": "INCONCLUSIVE", "gate_B_seizure_like": "INCONCLUSIVE",
        "trajectory_robustness": "not_tested", "rate_matched_control": "not_run",
        "out_of_range_fraction": 0.0, "forbidden_claims": [],
    }


def _landmark_rows():
    traces = {
        "q_core": [1.0 - 0.001 * t for t in range(300)],
        "q_global": [1.0 - 0.0008 * t for t in range(300)],
        "g_K": [0.002 * t for t in range(300)],
    }
    return sample_event_landmarks(
        traces, 1.0, [{"event_id": 1, "onset_ms": 100, "peak_ms": 150, "end_ms": 200}])


def test_build_phase_trajectory_rows_conform_to_canonical_schema():
    rows = build_phase_trajectory_rows(_landmark_rows(), _precalib_mapping(), _ranges())
    validate_phase_trajectory(rows)  # canonical schema -> no raise


def test_precalibration_rows_are_invalid():
    rows = build_phase_trajectory_rows(_landmark_rows(), _precalib_mapping(), _ranges())
    assert all(r["phase_coord_valid"] is False for r in rows)


def test_disabled_mechanism_writes_na_phase_coord():
    lr = _landmark_rows()
    lr[0]["g_K"] = "NA"  # recovery mechanism disabled for this sample
    rows = build_phase_trajectory_rows(lr, _precalib_mapping(), _ranges())
    assert rows[0]["phase_recovery"] == NA_SENTINEL


def test_out_of_range_flag_set_when_input_beyond_domain():
    lr = _landmark_rows()
    lr.append({"event_id": 1, "event_stage": "post_1s", "time_ms": 999,
               "q_core": 1.2, "q_global": 0.8, "g_K": 0.3})  # q_core beyond input_max
    rows = build_phase_trajectory_rows(lr, _precalib_mapping(), _ranges())
    assert rows[-1]["phase_coord_out_of_range"] is True


def test_self_audit_refuses_overlay_precalibration():
    rows = build_phase_trajectory_rows(_landmark_rows(), _precalib_mapping(), _ranges())
    audit = build_self_audit(_precalib_mapping(), _ranges(), rows, _summary())
    validate_interface_audit(audit)  # structurally valid audit
    assert audit["overlay_verdict"] in ("refused", "mechanism_candidate_only")
    assert audit["overlay_allowed"] is False


# --------------------------------------------------------------------------- #
# runner-side wiring (pure functions; the live runner just calls these)        #
# --------------------------------------------------------------------------- #
def _sim_with_traces(T=300):
    return {
        "trace_core": [1.0 - 0.001 * t for t in range(T)],
        "trace_global": [1.0 - 0.0008 * t for t in range(T)],
        "trace_gk": [0.0 for t in range(T)],
    }


def _read_events():
    # read_events() shape: t_on / t_off in ms, no event_id, no peak.
    return [{"t_on": 100.0, "t_off": 200.0}, {"t_on": 400.0, "t_off": 480.0}]


def test_default_precalib_mapping_is_schema_valid_but_uncalibrated():
    mapping, ranges = default_precalib_mapping_and_ranges("m3a_a2_demo")
    validate_slow_to_rate_mapping(mapping)  # schema-valid -> no raise
    assert mapping["slow_to_rate_mapping_id"] == "m3a_a2_demo"
    assert ranges["slow_to_rate_mapping_id"] == "m3a_a2_demo"
    assert all(mapping["coordinates"][c]["calibration_status"] == "not_applicable"
               for c in mapping["coordinates"])


def test_build_handoff_from_sim_produces_failclosed_inputs():
    h = build_handoff_from_sim(_sim_with_traces(), _read_events(), 1.0,
                               mapping_id="m3a_a2_demo", gk_enabled=False)
    assert len(h["landmark_rows"]) == 2 * 7  # two events x seven stages
    validate_dynamic_slowvars_summary(h["summary"])
    assert h["summary"]["gate_A_trajectory"] == "INCONCLUSIVE"
    # gk disabled -> recovery coordinate is NA, never 0.0
    rows = build_phase_trajectory_rows(h["landmark_rows"], h["mapping"], h["ranges"])
    validate_phase_trajectory(rows)
    assert all(r["phase_recovery"] == NA_SENTINEL for r in rows)


def test_write_handoff_artifacts_emits_canonical_files_and_refuses(tmp_path):
    h = build_handoff_from_sim(_sim_with_traces(), _read_events(), 1.0,
                               mapping_id="m3a_a2_demo", gk_enabled=True)
    audit = write_handoff_artifacts(str(tmp_path), landmark_rows=h["landmark_rows"],
                                    mapping=h["mapping"], ranges=h["ranges"], summary=h["summary"])
    for fname in ("slow_to_rate_mapping.json", "phase_coord_ranges.json",
                  "phase_trajectory.csv", "dynamic_slowvars_summary.json",
                  "m3a_interface_audit.json"):
        assert (tmp_path / fname).exists(), fname
    assert audit["overlay_verdict"] == "refused"
    assert audit["overlay_allowed"] is False
    # the written audit re-validates against the canonical schema
    validate_interface_audit(json.load(open(tmp_path / "m3a_interface_audit.json")))


def test_build_handoff_uses_provided_calibrated_mapping():
    from src.sef_hfo_m3a_calibration import apply_calibration, evaluate_engine_sign_test
    mapping, ranges = default_precalib_mapping_and_ranges("m3a_a2_cal")
    cal = apply_calibration(mapping, {
        "phase_x_core": evaluate_engine_sign_test([0.4, 0.7, 1.0], [5, 3, 1], variable="q_core",
                        coord="phase_x_core", expected_direction="decreasing_in_input", engine_sha="x"),
        "phase_y_global": evaluate_engine_sign_test([0.4, 0.7, 1.0], [6, 4, 2], variable="q_global",
                        coord="phase_y_global", expected_direction="decreasing_in_input", engine_sha="x"),
    })
    h = build_handoff_from_sim(_sim_with_traces(), _read_events(), 1.0, mapping_id="ignored",
                               mapping=cal, ranges=ranges, gk_enabled=False)
    prov = h["summary"]["provenance"]
    assert h["mapping"]["coordinates"]["phase_x_core"]["calibration_status"] == "passed"
    assert prov["handoff_kind"] == "calibrated_handoff"
    assert "mapping_calibration" not in prov["undefined_science_decisions"]
    assert "sign" in prov["calibration_caveat"].lower()         # P1-2: sign-only, not a fitted curve
    assert h["summary"]["slow_to_rate_mapping_id"] == cal["slow_to_rate_mapping_id"]


def test_summary_records_precalibration_provenance():
    h = build_handoff_from_sim(_sim_with_traces(), _read_events(), 1.0,
                               mapping_id="m3a_a2_demo", gk_enabled=False)
    prov = h["summary"]["provenance"]
    assert prov["handoff_kind"] == "pre_calibration_scaffold"
    # decisions A & B are now pinned; only calibration + R_class remain undefined
    assert prov["tail_to_baseline_definition"].startswith("absolute")
    assert prov["two_core_reduction"] == "mean_q"
    assert prov["peak_landmark_source"] == "window_midpoint_placeholder"  # _read_events has no t_peak
    assert "event_phase_samples.csv" in prov["deferred_artifacts"]
    assert "mapping_calibration" in prov["undefined_science_decisions"]
    assert prov["expected_overlay_verdict"] == "refused"
    validate_dynamic_slowvars_summary(h["summary"])  # extra provenance key is tolerated


def test_default_mapping_declares_mean_q_two_core_reduction():
    mapping, _ = default_precalib_mapping_and_ranges("m3a_a2_demo")
    assert mapping["two_core_reduction"] == "mean_q"


def test_exporter_collapses_per_core_q_by_mean():
    # decision B: q_core_L / q_core_R collapse to phase_x_core by AVERAGING
    mapping, ranges = default_precalib_mapping_and_ranges("m3a_a2_demo")
    base = {"event_id": 1, "event_stage": "peak", "time_ms": 100, "g_K": 0.0, "q_global": 0.9}
    via_LR = build_phase_trajectory_rows([dict(base, q_core_L=0.8, q_core_R=0.6)], mapping, ranges)[0]
    via_mean = build_phase_trajectory_rows([dict(base, q_core=0.7)], mapping, ranges)[0]
    assert via_LR["phase_x_core"] == pytest.approx(via_mean["phase_x_core"])


def test_build_handoff_uses_real_peak_when_event_carries_t_peak():
    # decision C: the canonical peak is the real activity peak (here passed as t_peak), not midpoint
    events = [{"t_on": 100.0, "t_off": 200.0, "t_peak": 130.0}]
    h = build_handoff_from_sim(_sim_with_traces(), events, 1.0, mapping_id="m3a_a2_demo", gk_enabled=False)
    assert h["summary"]["provenance"]["peak_landmark_source"] == "activity_fraction_peak"
    peak_rows = [r for r in h["landmark_rows"] if r["event_stage"] == "peak"]
    assert peak_rows[0]["time_ms"] == 130.0


def test_write_handoff_emits_status_md_marking_scaffold(tmp_path):
    h = build_handoff_from_sim(_sim_with_traces(), _read_events(), 1.0,
                               mapping_id="m3a_a2_demo", gk_enabled=False)
    write_handoff_artifacts(str(tmp_path), landmark_rows=h["landmark_rows"],
                            mapping=h["mapping"], ranges=h["ranges"], summary=h["summary"])
    assert (tmp_path / "STATUS.md").exists()
    text = (tmp_path / "STATUS.md").read_text().lower()
    for marker in ("pre-calibration", "refused", "event_phase_samples", "midpoint", "two_core_reduction"):
        assert marker in text, marker


def _classify_metrics_R2(tail=1.1):
    """A returned, local, ignited event -> classify_event gives R2."""
    return {"event_detected": True, "returned": True, "runaway": False,
            "r95_ea": 3.0, "far_ea": 0.1, "active_peak": 0.5,
            "sustained_front_score": 0.0, "tail_to_baseline_ratio": tail}


def test_build_event_phase_samples_produces_canonical_rows():
    traj = build_phase_trajectory_rows(_landmark_rows(), _precalib_mapping(), _ranges())
    rows = build_event_phase_samples(traj, {1: _classify_metrics_R2()})
    validate_event_phase_samples(rows)  # canonical schema -> no raise
    assert all(r["R_class"] == "R2" for r in rows)
    assert all(r["return_to_baseline"] is True for r in rows)
    assert all(r["tail_to_baseline_ratio"] == pytest.approx(1.1) for r in rows)


def test_build_event_phase_samples_skips_unclassified_rows():
    traj = build_phase_trajectory_rows(_landmark_rows(), _precalib_mapping(), _ranges())
    assert build_event_phase_samples(traj, {}) == []  # no metrics -> no event samples


def test_event_phase_samples_R_class_uses_canonical_classify_event():
    traj = build_phase_trajectory_rows(_landmark_rows(), _precalib_mapping(), _ranges())
    not_returned_with_front = dict(_classify_metrics_R2(), returned=False, sustained_front_score=0.9)
    rows = build_event_phase_samples(traj, {1: not_returned_with_front})
    assert all(r["R_class"] == "R4a" for r in rows)  # sustained + front = R4a


def test_assemble_event_metrics_feeds_valid_event_phase_samples():
    import numpy as np
    # 8x8 source grid in a 4mm square; an event where the lower half fires in [100,200] ms
    xs, ys = np.meshgrid(np.linspace(0.2, 3.8, 8), np.linspace(0.2, 3.8, 8))
    posE = np.column_stack([xs.ravel(), ys.ravel()])
    N = posE.shape[0]
    T = 300
    spk = np.zeros((T, N), bool)
    spk[100:200, : N // 2] = True
    af = np.zeros(T); af[100:200] = 0.5
    metrics = assemble_event_metrics([{"t_on": 100.0, "t_off": 200.0}],
                                     spk=spk, posE=posE, af=af, bin_w=1.0, dt_ms=1.0,
                                     L=4.0, n_bins_per_axis=4)
    assert set(metrics) == {0}
    for k in ("event_detected", "returned", "runaway", "r95_ea", "far_ea", "active_peak",
              "sustained_front_score", "tail_to_baseline_ratio", "return_to_baseline"):
        assert k in metrics[0]
    # the assembled metrics feed build_event_phase_samples -> canonical schema
    lm = sample_event_landmarks(
        {"q_core": [1.0 - 0.001 * t for t in range(T)],
         "q_global": [1.0 - 0.0008 * t for t in range(T)], "g_K": [0.0] * T},
        1.0, [{"event_id": 0, "onset_ms": 100, "peak_ms": 150, "end_ms": 200}])
    traj = build_phase_trajectory_rows(lm, _precalib_mapping(), _ranges())
    rows = build_event_phase_samples(traj, metrics)
    validate_event_phase_samples(rows)


def test_event_phase_samples_returned_column_uses_absolute_recovery():
    # R_class uses the event_props 'returned' (sustained-ness); the return_to_baseline COLUMN uses the
    # decision-A absolute recovery check -- they answer different questions and can differ.
    traj = build_phase_trajectory_rows(_landmark_rows(), _precalib_mapping(), _ranges())
    m = dict(_classify_metrics_R2(), returned=True, return_to_baseline=False)
    rows = build_event_phase_samples(traj, {1: m})
    assert all(r["return_to_baseline"] is False for r in rows)  # column = absolute recovery
    assert all(r["R_class"] == "R2" for r in rows)              # R_class = event_props returned
