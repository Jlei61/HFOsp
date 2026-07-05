"""M3A-A2 runner-layer export: build the canonical M3A->M3B handoff artifacts.

This is the M3A side of the shared contract in src/sef_hfo_m3_interface.py. It is
GLUE ONLY: it calls the canonical evaluate_phase_coord / coord_out_of_range /
sample_phase_coord_valid / audit_m3a_interface so the M3A exporter and the M3B
reader cannot diverge (contract §6, B8). It never re-implements a contract rule.

A disabled mechanism (NA slow value) yields an NA phase coordinate, never 0.0.
Pre-calibration (calibration_status != 'passed'), every row is phase_coord_valid
False and the self-audit verdict is 'refused' -- the wiring is fail-closed.
"""
import csv
import json
import os

from src.sef_hfo_a2 import sample_event_landmarks, tail_to_baseline_absolute
from src.sef_hfo_m3_interface import (
    PHASE_COORDS, ON_AXIS_COORDS, NA_SENTINEL, is_na,
    evaluate_phase_coord, coord_out_of_range, sample_phase_coord_valid,
    audit_m3a_interface, mapping_sign_tests_passed,
)

_META_KEYS = ("event_id", "event_stage", "time_ms")
_PASSTHROUGH_Q = ("q_core_L", "q_core_R")


def _slow_values(landmark_row):
    return {k: v for k, v in landmark_row.items() if k not in _META_KEYS}


def _reduce_two_core(q_L, q_R, method):
    """Collapse the two equal cores to one core-excitability input (user decision B)."""
    if method == "mean_q":
        return 0.5 * (q_L + q_R)
    raise NotImplementedError(
        f"two_core_reduction={method!r} not implemented; only 'mean_q' is decided "
        "(source_core / min_q are deferred science decisions, contract §9).")


def build_phase_trajectory_rows(landmark_rows, mapping, ranges):
    """Per-event landmark rows -> canonical phase_trajectory rows.

    Each phase coordinate is computed by the SHARED evaluate_phase_coord. A
    coordinate whose input slow-var is NA (disabled mechanism) is written NA
    (never 0.0). phase_coord_valid is the mapping-trust predicate over the
    on-axis coordinates; phase_coord_out_of_range is OR over coords whose input
    leaves the calibrated domain or whose output leaves the axis range.
    """
    mid = mapping["slow_to_rate_mapping_id"]
    reduction = mapping.get("two_core_reduction")
    rows = []
    for lr in landmark_rows:
        sv = _slow_values(lr)
        if reduction and "q_core_L" in sv and "q_core_R" in sv \
                and not is_na(sv["q_core_L"]) and not is_na(sv["q_core_R"]):
            sv = dict(sv)
            sv["q_core"] = _reduce_two_core(float(sv["q_core_L"]), float(sv["q_core_R"]), reduction)
        row = {
            "time_ms": lr["time_ms"], "event_id": lr["event_id"],
            "event_stage": lr["event_stage"], "slow_to_rate_mapping_id": mid,
        }
        any_oor = False
        for coord in PHASE_COORDS:
            input_var = mapping["coordinates"][coord]["transform"]["input_var"]
            v = sv.get(input_var)
            if v is None or is_na(v):
                row[coord] = NA_SENTINEL          # disabled mechanism -> NA, not 0.0
                continue
            row[coord] = evaluate_phase_coord(mapping, coord, sv)
            if coord_out_of_range(mapping, ranges, coord, sv):
                any_oor = True
        on_axis_na = any(is_na(row[c]) for c in ON_AXIS_COORDS)
        row["phase_coord_valid"] = (
            False if on_axis_na else bool(sample_phase_coord_valid(mapping, ranges, sv, ON_AXIS_COORDS)))
        row["phase_coord_out_of_range"] = any_oor
        for k in _PASSTHROUGH_Q:
            if k in lr:
                row[k] = lr[k]
        rows.append(row)
    return rows


def _self_axes_meta(mapping):
    """The M3B grid axes as if built from THIS mapping (provenance self-consistency)."""
    return {
        "axes_built_from_slow_to_rate_mapping_id": mapping["slow_to_rate_mapping_id"],
        "axis_space": mapping["axis_space"],
        "axis_transforms": {c: mapping["coordinates"][c]["transform"] for c in ON_AXIS_COORDS},
    }


def build_self_audit(mapping, ranges, trajectory_rows, summary):
    """Run the canonical overlay audit against this run's own artifacts.

    Pre-calibration this yields overlay_verdict='refused' (cond1/cond3 False),
    i.e. the runner refuses to claim a phase-map trajectory until the mapping is
    calibrated and the dynamic phenotype gate passes.
    """
    return audit_m3a_interface(
        mapping=mapping, ranges=ranges, trajectory_rows=trajectory_rows,
        summary=summary, axes_meta=_self_axes_meta(mapping))


# --------------------------------------------------------------------------- #
# Runner-side wiring: dynamic run -> fail-closed canonical handoff artifacts    #
# --------------------------------------------------------------------------- #
def _precalib_coord(input_var, ttype, a, b, direction, imin, imax):
    return {
        "transform": {"type": ttype, "input_var": input_var, "a": a, "b": b,
                      "clip": [0.0, 1.0], "input_min": imin, "input_max": imax,
                      "expected_direction": direction},
        "units": "dimensionless", "valid_range": [0.0, 1.0], "variables": [input_var],
        "calibration_status": "not_applicable", "sign_tests": [],
    }


def default_precalib_mapping_and_ranges(mapping_id):
    """A schema-valid but UNCALIBRATED mapping (+ ranges) for the fail-closed handoff.

    Transforms are physically sensible placeholders (phase ~ 1/q on the
    excitability axes, g_K monotone on recovery), but calibration_status is
    'not_applicable' on every coordinate -- so the self-audit refuses the overlay
    until A1 calibrates and pins the sign tests (contract §4).
    """
    mapping = {
        "slow_to_rate_mapping_id": mapping_id,
        "source": "M3A-A2 dynamic run (pre-calibration placeholder)",
        "substrate": "stage3_twoend_equal",
        "axis_space": "normalized_unit",
        "two_core_reduction": "mean_q",   # decision B: equal cores -> average (symmetric)
        "coordinates": {
            # phase_x_core ~ 1/q_core: (1/3)/q - 1/3 maps q in [0.25,1] -> [1,0] (decreasing)
            "phase_x_core": _precalib_coord("q_core", "reciprocal_affine", 1.0 / 3.0, -1.0 / 3.0,
                                            "decreasing_in_input", 0.25, 1.0),
            "phase_y_global": _precalib_coord("q_global", "reciprocal_affine", 1.0 / 3.0, -1.0 / 3.0,
                                              "decreasing_in_input", 0.25, 1.0),
            # phase_recovery: g_K increasing (placeholder identity-affine over [0,1])
            "phase_recovery": _precalib_coord("g_K", "affine", 1.0, 0.0, "increasing_in_input", 0.0, 1.0),
        },
    }
    ranges = {
        "slow_to_rate_mapping_id": mapping_id,
        "phase_x_core": {"min": 0.0, "max": 1.0, "source": "A2 sweep (pre-calibration)"},
        "phase_y_global": {"min": 0.0, "max": 1.0, "source": "A2 sweep (pre-calibration)"},
        "phase_recovery": {"min": 0.0, "max": 1.0, "source": "A2 sweep (pre-calibration)"},
    }
    return mapping, ranges


def build_handoff_from_sim(sim, events, dt_ms, *, mapping_id, gk_enabled=False,
                           af=None, bin_w=None, L=None, n_bins_per_axis=12,
                           mapping=None, ranges=None):
    """Assemble the fail-closed handoff inputs from a dynamic run's traces + events.

    Reads sim['trace_core'/'trace_global'/'trace_gk'] (full per-step q traces) and
    read_events() output (t_on/t_off in ms). The 'peak' landmark is the window
    midpoint (placeholder; refine to the activity peak when calibration lands).
    When the recovery mechanism is disabled (gk_enabled False) the g_K trace is NA,
    so phase_recovery exports NA not 0.0 (contract M12).
    """
    T = len(sim["trace_core"])
    gk_trace = list(sim["trace_gk"]) if gk_enabled else [NA_SENTINEL] * T
    traces = {"q_core": list(sim["trace_core"]),
              "q_global": list(sim["trace_global"]), "g_K": gk_trace}
    ev = []
    real_peak = bool(events)
    for i, e in enumerate(events):
        t_on, t_off = float(e["t_on"]), float(e["t_off"])
        if "t_peak" in e:                       # real activity peak (decision C), supplied by the runner
            peak = float(e["t_peak"])
        else:
            peak = 0.5 * (t_on + t_off)         # midpoint fallback if no peak was computed
            real_peak = False
        ev.append({"event_id": i, "onset_ms": t_on, "peak_ms": peak, "end_ms": t_off})
    landmark_rows = sample_event_landmarks(traces, dt_ms, ev)
    if mapping is None or ranges is None:
        mapping, ranges = default_precalib_mapping_and_ranges(mapping_id)
    mid = mapping["slow_to_rate_mapping_id"]
    # P1-2: this is SIGN calibration (direction locked), not a fitted rate response curve.
    calibrated = mapping_sign_tests_passed(mapping, None)
    event_metrics = None
    if af is not None and bin_w is not None and L is not None and events:
        event_metrics = assemble_event_metrics(events, spk=sim["spk"], posE=sim["posE"],
                                                af=af, bin_w=bin_w, dt_ms=dt_ms, L=L,
                                                n_bins_per_axis=n_bins_per_axis)
    have_rclass = event_metrics is not None
    undefined = (([] if calibrated else ["mapping_calibration"])
                 + ([] if have_rclass else ["R_class_classification"]))
    summary = {
        "slow_to_rate_mapping_id": mid,
        "gate_A_trajectory": "INCONCLUSIVE", "gate_B_seizure_like": "INCONCLUSIVE",
        "trajectory_robustness": "not_tested", "rate_matched_control": "not_run",
        "out_of_range_fraction": 0.0, "forbidden_claims": [],
        # honest provenance (extra key, tolerated by validate_dynamic_slowvars_summary).
        "provenance": {
            "handoff_kind": "calibrated_handoff" if calibrated else "pre_calibration_scaffold",
            "mapping_calibration": "sign_calibrated_direction_only" if calibrated else "not_applicable",
            "calibration_caveat": (
                "sign-calibrated normalized phase mapping: DIRECTION locked, quantitative "
                "response curve NOT fitted (P1-2)" if calibrated else None),
            "peak_landmark_source": "activity_fraction_peak" if real_peak else "window_midpoint_placeholder",
            "tail_to_baseline_definition": "absolute_vs_fixed_baseline_5_50ms_gate_1.5",
            "two_core_reduction": mapping.get("two_core_reduction"),
            "deferred_artifacts": [] if have_rclass else ["event_phase_samples.csv"],
            "undefined_science_decisions": undefined,
            # cond4 (gate_A PASS + rate-matched) is a science outcome of the real sweep, still pending
            "expected_overlay_verdict": "refused",
        },
    }
    return {"landmark_rows": landmark_rows, "mapping": mapping, "ranges": ranges,
            "summary": summary, "event_metrics": event_metrics}


def build_event_phase_samples(trajectory_rows, event_metrics):
    """Join per-event regime classification onto the phase-trajectory landmark rows.

    Reuses the canonical classify_event (src/sef_hfo_mu_basin) -- the SAME R_class
    producer the quasistatic runner uses -- so R_class is consistent across M3A.
    `event_metrics` maps event_id -> a metrics dict carrying the classify_event keys
    (event_detected, returned, runaway, r95_ea, far_ea, active_peak,
    sustained_front_score) PLUS tail_to_baseline_ratio (absolute, decision D6).
    Trajectory rows whose event_id has no metrics (e.g. inter-event rows) are skipped.
    Returns canonical event_phase_samples rows.
    """
    from src.sef_hfo_mu_basin import classify_event
    rows = []
    for r in trajectory_rows:
        m = event_metrics.get(r["event_id"])
        if m is None:
            continue
        rows.append({
            "event_id": r["event_id"],
            "event_stage": r["event_stage"],
            "phase_x_core": r["phase_x_core"],
            "phase_y_global": r["phase_y_global"],
            "phase_recovery": r["phase_recovery"],
            "phase_coord_valid": r["phase_coord_valid"],
            "phase_coord_out_of_range": r["phase_coord_out_of_range"],
            "slow_to_rate_mapping_id": r["slow_to_rate_mapping_id"],
            # return_to_baseline COLUMN = decision-A absolute recovery; R_class uses the
            # event_props sustained-ness 'returned' inside classify_event(m).
            "return_to_baseline": bool(m.get("return_to_baseline", m["returned"])),
            "tail_to_baseline_ratio": float(m["tail_to_baseline_ratio"]),
            "R_class": classify_event(m),
        })
    return rows


def assemble_event_metrics(events, *, spk, posE, af, bin_w, dt_ms, L,
                           n_bins_per_axis=12, far_radius_mm=None):
    """Per-event classify_event metrics (SOURCE space) + the absolute recovery check.

    Faithful reuse of the canonical helpers the quasistatic runner uses, so R_class
    is computed identically: spatial_bins (source bins from posE), event_props
    (returned/sustained/peak), _event_spatial (r95/far/front over source bins), and
    tail_to_baseline_absolute (decision A). The `r95_ea`/`far_ea` keys are the
    classify_event dict names but carry SOURCE-space values. Returns
    {event_id (enumerate index): metrics} for build_event_phase_samples.
    """
    import numpy as np
    from src.topic4_propagation_operator import spatial_bins
    from src.sef_hfo_a2 import _bin_spike_counts, _spatial_extent
    from src.sef_hfo_mu_basin import event_props

    def _event_spatial(E_spk, bin_of_cell, n_bins, bin_centers, src_bin,
                       far_radius, t_lo, t_hi, dt):
        lo = int(np.floor(float(t_lo) / float(dt)))
        hi = int(np.ceil(float(t_hi) / float(dt)))
        bins = _bin_spike_counts(E_spk, bin_of_cell, n_bins, lo, hi)
        n_act, r95, far = _spatial_extent(bins, bin_centers, src_bin, far_radius)
        tail_lo = max(float(t_lo), float(t_hi) - 50.0)
        tlo = int(np.floor(tail_lo / float(dt)))
        thi = int(np.ceil(float(t_hi) / float(dt)))
        tail_bins = _bin_spike_counts(E_spk, bin_of_cell, n_bins, tlo, thi)
        front_score = 1.0 - float(np.sum(tail_bins > 0)) / max(int(n_bins), 1)
        return float(r95), float(far), int(n_act), float(front_score)

    posE = np.asarray(posE, float)
    bins = spatial_bins(posE, n_bins_per_axis)
    bin_centers = bins["bin_centers"]
    bin_of_cell = bins["bin_of_cell"]
    n_bins = int(bin_centers.shape[0])
    core_center = bin_centers.mean(axis=0)
    src_bin = int(np.argmin(np.linalg.norm(bin_centers - core_center[None, :], axis=1)))
    far_radius = far_radius_mm if far_radius_mm is not None else 0.35 * float(L)
    af = np.asarray(af, float)
    n_rec_bins = len(af)
    out = {}
    for i, e in enumerate(events):
        t_on, t_off = float(e["t_on"]), float(e["t_off"])
        b0 = max(0, int(round(t_on / bin_w)))
        b1 = min(n_rec_bins - 1, int(round(t_off / bin_w)))
        ep = event_props(af, (b0, b1), bin_w, n_rec_bins)
        r95, far, n_act, front = _event_spatial(spk, bin_of_cell, n_bins, bin_centers,
                                                 src_bin, far_radius, t_on, t_off, dt_ms)
        tail_ratio, returned_abs = tail_to_baseline_absolute(af, bin_w, t_off)
        out[i] = {
            "event_detected": True,
            "returned": bool(ep["returned"]),          # sustained-ness -> classify_event
            "runaway": bool(ep["sustained"]),
            "r95_ea": float(r95), "far_ea": float(far),
            "active_peak": float(ep["peak_active"]),
            "sustained_front_score": float(front),
            "tail_to_baseline_ratio": float(tail_ratio),
            "return_to_baseline": bool(returned_abs),   # decision-A absolute -> CSV column
        }
    return out


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def _write_csv(path, rows):
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _status_md(summary):
    """Render the honest pre-calibration STATUS.md from the summary provenance."""
    p = summary.get("provenance", {})
    src = p.get("peak_landmark_source", "")
    peak_line = (
        "- peak landmark            PLACEHOLDER = window midpoint; must become the real\n"
        "  activity peak before calibration.\n"
        if "midpoint" in src else
        f"- peak landmark            DECIDED = {src} (real activity-fraction peak).\n"
    )
    eps_line = (
        "- event_phase_samples.csv  DEFERRED -- needs R_class classification (contract §9). Not fabricated.\n"
        if "event_phase_samples.csv" in p.get("deferred_artifacts", []) else
        "- event_phase_samples.csv  WRITTEN -- per-event R_class via classify_event + absolute recovery.\n"
    )
    calibrated = p.get("handoff_kind") == "calibrated_handoff"
    if calibrated:
        title = "# M3A->M3B-R2 handoff: SIGN-CALIBRATED (overlay still pending gate_A)\n\n"
        intro = (
            "The mapping is SIGN-calibrated (direction locked, NOT a fitted response curve, P1-2). The\n"
            "overlay audit's cond1 can pass, but `overlay_verdict` stays `refused` until cond4 --\n"
            "gate_A PASS + rate-matched -- which is a SCIENTIFIC outcome of the real sweep.\n\n"
        )
        cal_line = (f"- mapping calibration       {p.get('mapping_calibration')} -- "
                    f"{p.get('calibration_caveat')}\n")
    else:
        title = "# M3A->M3B-R2 handoff: PRE-CALIBRATION SCAFFOLD\n\n"
        intro = (
            "This directory is NOT a complete M3B-ready handoff. It is a pre-calibration scaffold:\n"
            "the phase-trajectory schema + the overlay audit gate, with the mapping UNCALIBRATED,\n"
            "so the self-audit `overlay_verdict` is `refused` BY DESIGN.\n\n"
        )
        cal_line = (
            "- mapping calibration       not_applicable -- run scripts/calibrate_a2_mapping.py and pass\n"
            "  --calibration-dir to the sweep; overlay stays refused until calibrated + gate_A PASS.\n"
        )
    return (
        title + intro +
        "## canonical contract artifacts\n"
        "- slow_to_rate_mapping.json / phase_coord_ranges.json / phase_trajectory.csv\n"
        "- dynamic_slowvars_summary.json / m3a_interface_audit.json (overlay_verdict = refused)\n\n"
        "## Decided 2026-06-27 (user)\n"
        f"- tail_to_baseline         DECIDED = {p.get('tail_to_baseline_definition')}\n"
        f"- two_core_reduction       DECIDED = {p.get('two_core_reduction')}\n"
        f"{peak_line}\n"
        "## Artifact + calibration status\n"
        f"{eps_line}"
        f"{cal_line}"
    )


def write_handoff_artifacts(out_dir, *, landmark_rows, mapping, ranges, summary, event_metrics=None):
    """Build phase_trajectory + self-audit and write the pre-calibration scaffold files.

    Writes the 5 canonical contract artifacts (slow_to_rate_mapping.json,
    phase_coord_ranges.json, phase_trajectory.csv, dynamic_slowvars_summary.json,
    m3a_interface_audit.json) plus STATUS.md recording the deferrals. Returns the audit.
    event_phase_samples.csv is not written here yet: the pure builder
    (build_event_phase_samples) exists and reuses classify_event, but the runner-side
    per-event metric assembly (event_props) is pending -- see the layer note in the recap.
    """
    os.makedirs(out_dir, exist_ok=True)
    trajectory_rows = build_phase_trajectory_rows(landmark_rows, mapping, ranges)
    audit = build_self_audit(mapping, ranges, trajectory_rows, summary)
    _write_json(os.path.join(out_dir, "slow_to_rate_mapping.json"), mapping)
    _write_json(os.path.join(out_dir, "phase_coord_ranges.json"), ranges)
    _write_json(os.path.join(out_dir, "dynamic_slowvars_summary.json"), summary)
    _write_json(os.path.join(out_dir, "m3a_interface_audit.json"), audit)
    _write_csv(os.path.join(out_dir, "phase_trajectory.csv"), trajectory_rows)
    if event_metrics:   # R-class wired -> the 5th canonical artifact
        evt_rows = build_event_phase_samples(trajectory_rows, event_metrics)
        _write_csv(os.path.join(out_dir, "event_phase_samples.csv"), evt_rows)
    with open(os.path.join(out_dir, "STATUS.md"), "w") as f:
        f.write(_status_md(summary))
    return audit
