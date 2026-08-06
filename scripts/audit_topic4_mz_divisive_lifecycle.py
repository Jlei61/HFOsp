#!/usr/bin/env python3
"""Strict post-hoc audit for the MZ divisive lifecycle screen.

This script never simulates.  It reclassifies saved downsampled traces against the paired seed-1
slow-off reference and writes a separate machine-readable artifact, preserving the original screen
labels for provenance.
"""
from __future__ import annotations

import argparse
import datetime as dt_datetime
import hashlib
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.topic4_mz_divisive_lifecycle import audit_lifecycle_against_reference  # noqa: E402


RESULT_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_divisive_lifecycle")
DEFAULT_REFERENCE = os.path.join(
    RESULT_ROOT,
    "runs",
    "20260719T133522.454363Z_6ce230e_26dda6ca76_observer",
    "summary.json",
)


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _atomic_json(payload, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _resolve_latest():
    pointer = _load(os.path.join(RESULT_ROOT, "latest_slow_gate_m.json"))
    return os.path.join(ROOT, pointer["summary"])


def _key(label, field):
    safe = label.replace(".", "p").replace("-", "_")
    return f"{safe}__{field}"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=None, help="v3 summary.json; default latest_slow_gate_m")
    parser.add_argument("--reference-summary", default=DEFAULT_REFERENCE)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    summary_path = os.path.abspath(args.summary or _resolve_latest())
    reference_path = os.path.abspath(args.reference_summary)
    summary = _load(summary_path)
    reference = _load(reference_path)
    trace_path = os.path.join(os.path.dirname(summary_path), "traces_downsampled.npz")
    reference_trace_path = os.path.join(os.path.dirname(reference_path), "traces_downsampled.npz")

    with np.load(trace_path) as traces, np.load(reference_trace_path) as ref_traces:
        ref_row = next(row for row in reference["rows"] if row["label"] == "slowoff_p1")
        ref_rate = np.asarray(ref_traces[_key("slowoff_p1", "rate")], float)
        ref_dt_ms = float(ref_row["T_ms"]) / float(ref_rate.size)

        audited = []
        for row in summary["rows"]:
            label = row["label"]
            rate = np.asarray(traces[_key(label, "rate")], float)
            dt_ms = float(row["T_ms"]) / float(rate.size)
            slow = {}
            for field in ("z_mean", "z_min", "m_mean", "adap", "SG", "AG", "TG", "UTG"):
                name = _key(label, field)
                if name in traces.files:
                    slow[field] = np.asarray(traces[name], float)
            strict = audit_lifecycle_against_reference(
                rate,
                dt_ms,
                reference_rate_hz=ref_rate,
                reference_dt_ms=ref_dt_ms,
                runaway_ms=row.get("runaway_ms"),
                slow_traces=slow,
            )
            audited.append(
                dict(
                    label=label,
                    eta_m=float(row["cfg"]["eta_m"]),
                    original_phenotype=row["phenotype"],
                    trace_dt_ms=dt_ms,
                    **strict,
                )
            )

    anchor = audited[0]
    anchor_is_finite_window_high = bool(
        anchor["runaway_ms"] is None
        and anchor["strict_phenotype"].startswith("bounded_recruited")
        and anchor["offset_ms"] is None
    )
    m_off_returns = bool(anchor["returned_to_same_seed_slowoff"])
    candidates = []
    if summary.get("phase") != "slow_gate_m":
        verdict = "strict_trace_reclassification_only"
        interpretation = "no_m_exit_inference_for_this_phase"
    else:
        candidates = [
            row["label"]
            for row in audited[1:]
            if row["strict_phenotype"] in {
                "terminate_bursting_strict",
                "terminate_nonrhythmic_strict",
            }
            and row["m_rise_before_rate_decay"] is True
            and not m_off_returns
            and anchor_is_finite_window_high
        ]
        if anchor["runaway_ms"] is not None:
            verdict = "m_exit_not_interpretable_m_off_anchor_delayed_runaway"
            interpretation = "containment_sensitivity_only"
            candidates = []
        elif not anchor_is_finite_window_high:
            verdict = "m_exit_not_interpretable_m_off_anchor_not_persistent_high"
            interpretation = "containment_sensitivity_only"
            candidates = []
        elif candidates:
            verdict = "strict_posthoc_m_dependent_lifecycle_screen_hit"
            interpretation = "requires_cross_seed_and_state_fork_causal_controls"
        else:
            verdict = "no_strict_lifecycle_candidate_in_locked_eta_ladder"
            interpretation = "single_seed_scoped_no_go_not_universal_mechanism_rejection"

    payload = dict(
        experiment="MZ slow-gated recurrent divisor + exact M ladder strict post-hoc audit",
        generated_utc=dt_datetime.datetime.now(dt_datetime.timezone.utc).isoformat(),
        verdict=verdict,
        interpretation=interpretation,
        source_summary=os.path.relpath(summary_path, ROOT),
        source_trace=os.path.relpath(trace_path, ROOT),
        reference_summary=os.path.relpath(reference_path, ROOT),
        reference_trace=os.path.relpath(reference_trace_path, ROOT),
        source_hashes={
            "summary": _sha(summary_path),
            "trace": _sha(trace_path),
            "reference_summary": _sha(reference_path),
            "reference_trace": _sha(reference_trace_path),
            "auditor": _sha(__file__),
            "classifier": _sha(os.path.join(ROOT, "src", "topic4_mz_divisive_lifecycle.py")),
        },
        reference_contract=dict(
            label="slowoff_p1",
            seed=int(ref_row["seed"]),
            T_ms=float(ref_row["T_ms"]),
            trace_dt_ms=ref_dt_ms,
            return_definition=(
                "first and final 2 s means <= paired slow-off 2 s rolling Q99, at least one brief "
                "returning-event-like excursion after the 2 s recovery window, and no rebound "
                "recruited macro-state"
            ),
        ),
        m_off_anchor_finite_window_high=anchor_is_finite_window_high,
        m_off_anchor_stationarity_claim=False,
        lifecycle_candidates=candidates,
        rows=audited,
    )
    output = os.path.abspath(args.output or os.path.join(os.path.dirname(summary_path), "strict_audit.json"))
    _atomic_json(payload, output)
    print(json.dumps(dict(output=output, verdict=verdict, candidates=candidates), indent=2))


if __name__ == "__main__":
    main()
