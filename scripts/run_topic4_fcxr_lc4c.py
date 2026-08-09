#!/usr/bin/env python3
"""Execute the locked FCXR-LC4c entry gate and conditional lifecycle chain."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc4_lifecycle as LC4  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger  # noqa: E402
from src.topic4_fcxr_lc4c import adjudicate_entry  # noqa: E402
from src.topic4_fcxr_lc4b_deadzone import sha256_file  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4c_entry_offset_alignment"
LOCK = OUT / "candidate_lock.json"
ENTRY_MS = 15000.0
LC4.OUT = str(OUT)


def _candidate() -> dict:
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "C0_PASS" or lock.get("verdict") != "ENTRY_OFFSET_REPAIR_IDENTIFIABLE":
        raise SystemExit("LC4c requires a passing C0 candidate lock")
    for rel, expected in lock["sources"].items():
        got = sha256_file(ROOT / rel)
        if got != expected:
            raise SystemExit(f"LC4c source artifact drift: {rel}: {got} != {expected}")
    return lock["candidate"]


LC4._candidate = _candidate


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def stage_entry() -> dict:
    LC4._preflight("C1_ENTRY")
    candidate = _candidate()
    out_json = OUT / "entry_gate.json"
    if out_json.exists():
        prior = json.loads(out_json.read_text())
        if prior.get("status") == "COMPLETE":
            return prior
    GEO._write_json(OUT / "C1_RUNNING.json", dict(
        status="RUNNING", pid=os.getpid(), run_ms=ENTRY_MS, candidate=candidate,
        no_kick=True, no_reset=True, no_parameter_step=True, started=GEO._now()))
    S, slow = LC4._fresh_context(candidate, ENTRY_MS)
    p = dataclasses.replace(S["p"], T=ENTRY_MS, dt=LC4.DT)
    t0 = time.time()
    run = run_fcxr_loop(
        p, S["net"], slow=slow, n_steps=int(round(ENTRY_MS / LC4.DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    res, wins, numerical, rate, lifecycle, events, af, af_dt, floor = LC4._reduce(
        run, S, baseline)
    bout = LC4.first_ictal_bout(lifecycle["regimes"], float(baseline["band"]["win_ms"]))
    if bout is None:
        ceiling = 0.0
    else:
        onset_ms = float(bout[0] * baseline["band"]["win_ms"])
        ceiling = LC4.refractory_ceiling_fraction(
            run["E_spk_bool"], dt_ms=LC4.DT, onset_ms=onset_ms, offset_ms=ENTRY_MS,
            tau_ref_ms=float(S["p"].tau_ref_E))
    slow_f = run["checkpoint"].slow
    gate = adjudicate_entry(
        regimes=lifecycle["regimes"], win_ms=float(baseline["band"]["win_ms"]),
        events=events, current_trace=slow_f.trace_adap_current, current_dt_ms=LC4.DT,
        numerical_safe=not bool(numerical.get("numerical_unsafe")),
        refractory_fraction=ceiling)
    trace_path = OUT / "entry_gate_traces.npz"
    table = LC4._regional_trace_npz(
        trace_path, slow_f, S, rate=rate, af=af, af_dt=af_dt)
    r_base = float(np.median(rate[:max(1, int(round((gate.get("onset_ms") or ENTRY_MS)
                                                     / LC4.DT)))]))
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=floor,
        rate_hz=rate, dt_ms=LC4.DT, r_base_hz=r_base, table=table,
        onset_ms=gate.get("onset_ms"), offset_ms=None, total_ms=ENTRY_MS)
    rec = dict(
        status="COMPLETE", stage="C1_ENTRY", candidate=candidate, run_ms=ENTRY_MS,
        connection_seed=1, noise_seed=LC4.NOISE_SEED,
        no_kick=True, no_reset=True, no_parameter_step=True,
        gate=gate, lifecycle_classifier=lifecycle, numerical=numerical,
        event_ledger=ledger, traces=str(trace_path),
        max_rate_hz=float(np.max(rate)), mean_rate_hz=float(np.mean(rate)),
        max_adap_current=float(np.max(slow_f.trace_adap_current)),
        wall_s=time.time() - t0, peak_rss_gib=_peak_rss_gib(), finished=GEO._now(),
    )
    GEO._write_json(out_json, rec)
    GEO._write_json(OUT / "C1_DONE.json", dict(
        status="DONE", verdict=gate["verdict"], passed=gate["passed"], finished=GEO._now()))
    LC4._resource("C1_ENTRY_DONE", wall_s=rec["wall_s"], peak_rss_gib=rec["peak_rss_gib"])
    del run, res, S
    gc.collect()
    return rec


def _require_c1() -> None:
    path = OUT / "entry_gate.json"
    if not path.exists() or not bool((json.loads(path.read_text()).get("gate") or {}).get("passed")):
        raise SystemExit("LC4c nominal is blocked: C1 entry gate did not pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", choices=("entry", "nominal", "confirm"), required=True)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k LC4c execution requires --confirm-run")
    if args.stage == "entry":
        with LC4._stage_lock("lc4c_entry"):
            try:
                result = stage_entry()
            except BaseException as exc:
                GEO._write_json(OUT / "C1_FAILED.json", dict(
                    status="FAILED", error=repr(exc), finished=GEO._now()))
                raise
        gate = result["gate"]
    else:
        _require_c1()
        with LC4._stage_lock(f"lc4c_{args.stage}"):
            try:
                result = LC4.stage_nominal() if args.stage == "nominal" else LC4.stage_confirm()
            except BaseException as exc:
                GEO._write_json(OUT / f"C2_{args.stage.upper()}_FAILED.json", dict(
                    status="FAILED", error=repr(exc), finished=GEO._now()))
                raise
        gate = result.get("nominal_gate") or result.get("gate") or {}
    print(json.dumps(dict(status=result.get("status"), stage=result.get("stage"),
                          verdict=gate.get("verdict"), passed=gate.get("passed")), indent=2))


if __name__ == "__main__":
    main()
