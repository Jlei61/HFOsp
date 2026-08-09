#!/usr/bin/env python3
"""LC4b D1/D2: exact-dead-zone paired baseline and conditional frozen-D onset."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts", ROOT / "src" / "snn_engine"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc4_gate as L4  # noqa: E402
from src.topic4_fcxr_lc4_gate import baseline_gate, onset_surface_gate  # noqa: E402
from src.topic4_fcxr_lc4b_deadzone import sha256_file  # noqa: E402


OUT = str(Path(E01.OUT) / "lc4b_deadzone_lifecycle")
LOCK = Path(OUT) / "candidate_lock.json"
OLD = Path(E01.OUT) / "lc4_lifecycle_gate"
CONTROL_JSON = OLD / "runs/baseline_control.json"
CONTROL_TRACE = OLD / "runs/baseline_control_traces.npz"
POSITIVE_D10 = Path(E01.OUT) / "quiet_watch/quiet_D10.json"
EXPECTED = {
    CONTROL_JSON: "6b406476178c2601b3d6013719e412056b49bdd286c2da705590a82c270da6a1",
    CONTROL_TRACE: "89fe2337ab3df36bc95fc7daf6133f5f95be4a8354245322334f463cc8fa985d",
    POSITIVE_D10: "42f00ce8e8aa20b06cdbae0c8d265b8b111b86346405a2e58c43cfcf1ebb92e3",
}

# Reuse the already-tested 40k execution primitive, but put every new artifact in the LC4b root.
L4.OUT = OUT


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _candidate() -> dict:
    d = _load_json(LOCK)
    if d.get("status") != "D0_PASS" or d.get("verdict") != "DEADZONE_IDENTIFIABLE":
        raise SystemExit("D1 blocked: candidate_lock.json did not pass D0")
    return d["candidate"]


def _preflight(stage: str) -> None:
    E01.FCXR._assert_engine_blessed()
    L4._preflight(stage)
    for path, expected in EXPECTED.items():
        live = sha256_file(path)
        if live != expected:
            raise SystemExit(f"provenance drift: {path}: {live} != {expected}")


def stage_baseline() -> dict:
    _preflight("D1_BASELINE")
    candidate = _candidate()
    control = _load_json(CONTROL_JSON)
    fields, _ = GEO._primary_fields()
    L4.GEO._write_json(Path(OUT) / "D1_RUNNING.json", {
        "status": "RUNNING", "pid": os.getpid(), "candidate": candidate,
        "control_reused": str(CONTROL_JSON.relative_to(ROOT)), "started": GEO._now(),
    })
    row = L4._run_frozen(
        tag="baseline_deadzone", role="baseline_candidate", d_label="D_healthy",
        d_field=fields["D_healthy"], candidate=candidate, run_ms=L4.BASELINE_MS)
    row = dict(row)
    gate = baseline_gate(
        row["summary"], control["summary"],
        numerical_safe=not bool(row["numerical"].get("numerical_unsafe")),
        sustained_bout=bool(row["departed"]), max_current=float(row["adap_current_max"]),
        recurrent_scale=L4.I_EE_SCALE)
    cand_trace = np.load(Path(OUT) / "runs/baseline_deadzone_traces.npz")
    ctrl_trace = np.load(CONTROL_TRACE)
    exact_zero = bool(row["adap_current_max"] == 0.0
                      and np.asarray(cand_trace["a_mean"]).size > 0
                      and np.all(cand_trace["a_mean"] == 0.0))
    rate_identical = bool(np.array_equal(cand_trace["rate_E"], ctrl_trace["rate_E"]))
    af_identical = bool(np.array_equal(cand_trace["af"], ctrl_trace["af"]))
    gate["clauses"].update(exact_zero_actuator=exact_zero,
                           rate_trace_byte_identical=rate_identical,
                           active_fraction_byte_identical=af_identical)
    gate["passed"] = bool(all(gate["clauses"].values()))
    gate["verdict"] = ("DEADZONE_BASELINE_INERT" if gate["passed"]
                       else "DEADZONE_BASELINE_NOT_INERT")
    row["gate"] = gate
    verdict = {
        "status": "COMPLETE", "stage": "D1", "control": control, "candidate_row": row,
        "selected_candidate": candidate if gate["passed"] else None,
        "verdict": gate["verdict"], "stopped": not gate["passed"],
        "completed": GEO._now(),
    }
    GEO._write_json(Path(OUT) / "baseline_verdict.json", verdict)
    GEO._write_json(Path(OUT) / "D1_DONE.json", {
        "status": "DONE", "verdict": gate["verdict"], "finished": GEO._now()})
    return verdict


def _positive_control() -> dict:
    d = _load_json(POSITIVE_D10)
    return {
        "role": "positive_control", "d_label": "D10", "departed": bool(d["departed"]),
        "departure_ms": d["departure_ms"], "source": str(POSITIVE_D10.relative_to(ROOT)),
        "source_sha256": EXPECTED[POSITIVE_D10], "lifecycle": d["lifecycle"],
    }


def stage_onset() -> dict:
    _preflight("D2_ONSET")
    base = _load_json(Path(OUT) / "baseline_verdict.json")
    candidate = base.get("selected_candidate")
    if candidate is None:
        raise SystemExit("D2 blocked: D1 did not select the dead-zone candidate")
    fields, _ = GEO._primary_fields()
    healthy = dict(base["candidate_row"], role="candidate", d_label="D_healthy")
    rows = [_positive_control(), healthy]
    GEO._write_json(Path(OUT) / "D2_RUNNING.json", {
        "status": "RUNNING", "pid": os.getpid(), "candidate": candidate,
        "conditional_order": ["D10", "D30", "D50"], "started": GEO._now(),
    })
    for label in ("D10", "D30", "D50"):
        row = L4._run_frozen(
            tag=f"onset_deadzone_{label}", role="candidate", d_label=label,
            d_field=fields[label], candidate=candidate, run_ms=L4.ONSET_MS)
        rows.append(row)
        if row["departed"]:
            break
    gate = onset_surface_gate(rows)
    verdict = {
        "status": "COMPLETE", "stage": "D2", "candidate": candidate, "rows": rows,
        "gate": gate, "stopped": not gate["passed"], "completed": GEO._now(),
    }
    GEO._write_json(Path(OUT) / "onset_surface_verdict.json", verdict)
    GEO._write_json(Path(OUT) / "D2_DONE.json", {
        "status": "DONE", "verdict": gate["verdict"], "finished": GEO._now()})
    return verdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", required=True, choices=("baseline", "onset"))
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k LC4b execution requires --confirm-run")
    with L4._stage_lock(f"lc4b_{args.stage}"):
        result = stage_baseline() if args.stage == "baseline" else stage_onset()
    print(json.dumps({"stage": result["stage"], "verdict": (
        result["verdict"] if args.stage == "baseline" else result["gate"]["verdict"]),
                      "stopped": result["stopped"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
