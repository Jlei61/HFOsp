#!/usr/bin/env python3
"""Execute LC4e lock, shared-executor screen and its conditional lifecycle."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc4_lifecycle as LC4  # noqa: E402
import run_topic4_fcxr_lc4d as D  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
from src.topic4_fcxr_lc4e import (  # noqa: E402
    adjudicate_shared_screen,
    derive_shared_candidate,
    sha256_file,
)


OUT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
       / "lc4e_spatially_shared_terminator")
LOCK = OUT / "candidate_lock.json"
LOCAL = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
         / "lc4d_offset_latency_alignment")
LOCAL_LOCK = LOCAL / "candidate_lock.json"
LOCAL_RECORD = LOCAL / "latency_screen.json"
LOCAL_TRACE = LOCAL / "latency_screen_traces.npz"

D.OUT = OUT
D.LOCK = LOCK
D.LC4.OUT = str(OUT)
LC4.OUT = str(OUT)

SOURCE_PATHS = (
    "src/snn_engine/mz_slow_vars.py",
    "src/topic4_fcxr_lc4e.py",
    "src/topic4_fcxr_lc4d.py",
    "scripts/run_topic4_fcxr_lc4_lifecycle.py",
    "scripts/run_topic4_fcxr_lc4d.py",
    "scripts/run_topic4_fcxr_lc4e.py",
    "scripts/run_topic4_fcxr_lc4e_autopilot.sh",
    "docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4e-spatially-shared-terminator-design.md",
    "docs/superpowers/plans/2026-08-10-topic4-fcxr-lc4e-spatially-shared-terminator.md",
)


def stage_lock() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    local_lock = json.loads(LOCAL_LOCK.read_text())
    local_record = json.loads(LOCAL_RECORD.read_text())
    with np.load(LOCAL_TRACE) as trace:
        derived = derive_shared_candidate(local_lock, local_record, trace)
    payload = dict(
        status="E0_PASS", verdict="SHARED_EXECUTOR_IDENTIFIABLE",
        candidate=derived["candidate"], derivation={k: v for k, v in derived.items()
                                                     if k != "candidate"},
        source_artifacts={
            str(p.relative_to(ROOT)): sha256_file(p)
            for p in (LOCAL_LOCK, LOCAL_RECORD, LOCAL_TRACE)
        },
        sources={rel: sha256_file(ROOT / rel) for rel in SOURCE_PATHS},
        no_candidate_list=True, created=GEO._now(),
    )
    GEO._write_json(LOCK, payload)
    return payload


def _candidate() -> dict:
    if not LOCK.exists():
        raise SystemExit("LC4e requires candidate_lock.json")
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "E0_PASS" or lock.get("verdict") != "SHARED_EXECUTOR_IDENTIFIABLE":
        raise SystemExit("LC4e requires a passing E0 lock")
    for rel, expected in lock["sources"].items():
        got = sha256_file(ROOT / rel)
        if got != expected:
            raise SystemExit(f"LC4e source drift: {rel}: {got} != {expected}")
    for rel, expected in lock["source_artifacts"].items():
        got = sha256_file(ROOT / rel)
        if got != expected:
            raise SystemExit(f"LC4e source artifact drift: {rel}: {got} != {expected}")
    return lock["candidate"]


D._candidate = _candidate
D.LC4._candidate = _candidate
LC4._candidate = _candidate


def stage_screen() -> dict:
    rec = D.stage_screen()
    with np.load(LOCAL_TRACE) as local_trace, np.load(rec["traces"]) as shared_trace:
        gate = adjudicate_shared_screen(
            local_record=json.loads(LOCAL_RECORD.read_text()), shared_record=rec,
            local_trace=local_trace, shared_trace=shared_trace)
    GEO._write_json(OUT / "architecture_verdict.json", gate)
    GEO._write_json(OUT / "E1_DONE.json", dict(
        status="DONE", verdict=gate["verdict"], passed=gate["passed"], finished=GEO._now()))
    return dict(status="COMPLETE", stage="E1_SCREEN", gate=gate, shared_record=rec)


def _require_e1() -> None:
    path = OUT / "architecture_verdict.json"
    if not path.exists() or not bool(json.loads(path.read_text()).get("passed")):
        raise SystemExit("LC4e lifecycle is blocked: E1 shared-executor screen did not pass")


def _clear(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", choices=("lock", "screen", "nominal", "confirm"), required=True)
    args = ap.parse_args()
    if args.stage == "lock":
        print(json.dumps(stage_lock(), indent=2))
        return
    if not args.confirm_run:
        raise SystemExit("40k LC4e execution requires --confirm-run")
    if args.stage == "screen":
        with LC4._stage_lock("lc4e_screen"):
            try:
                result = stage_screen()
            except BaseException as exc:
                GEO._write_json(OUT / "E1_FAILED.json", dict(
                    status="FAILED", error=repr(exc), finished=GEO._now()))
                _clear(OUT / "L1_RUNNING.json")
                raise
        gate = result["gate"]
    else:
        _require_e1()
        with LC4._stage_lock(f"lc4e_{args.stage}"):
            try:
                result = LC4.stage_nominal() if args.stage == "nominal" else LC4.stage_confirm()
            except BaseException as exc:
                GEO._write_json(OUT / f"E2_{args.stage.upper()}_FAILED.json", dict(
                    status="FAILED", error=repr(exc), finished=GEO._now()))
                _clear(OUT / f"F2_{args.stage.upper()}_RUNNING.json")
                raise
        _clear(OUT / f"F2_{args.stage.upper()}_RUNNING.json")
        gate = result.get("nominal_gate") or result.get("gate") or {}
    print(json.dumps(dict(status=result.get("status"), stage=result.get("stage"),
                          verdict=gate.get("verdict"), passed=gate.get("passed")), indent=2))


if __name__ == "__main__":
    main()
