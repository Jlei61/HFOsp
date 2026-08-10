#!/usr/bin/env python3
"""Execute the locked FCXR-LC4d latency screen and conditional lifecycle chain."""
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
from src.topic4_fcxr_lc4_lifecycle import (  # noqa: E402
    _smooth_isolated,
    first_ictal_bout,
    refractory_ceiling_fraction,
)
from src.topic4_fcxr_lc4d import adjudicate_latency_screen  # noqa: E402
from src.topic4_fcxr_lc4b_deadzone import sha256_file  # noqa: E402


OUT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
       / "lc4d_offset_latency_alignment")
LOCK = OUT / "candidate_lock.json"
SCREEN_MS = 18000.0
LC4.OUT = str(OUT)


def _candidate() -> dict:
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "L0_PASS" or lock.get("verdict") != "OFFSET_LATENCY_REPAIR_IDENTIFIABLE":
        raise SystemExit("LC4d requires a passing L0 candidate lock")
    for rel, expected in lock["sources"].items():
        got = sha256_file(ROOT / rel)
        if got != expected:
            raise SystemExit(f"LC4d source artifact drift: {rel}: {got} != {expected}")
    return lock["candidate"]


LC4._candidate = _candidate


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _clear_running_sentinel(path: Path) -> None:
    """Remove a stage-running marker after a terminal marker is durable."""
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def stage_screen() -> dict:
    LC4._preflight("L1_SCREEN")
    candidate = _candidate()
    out_json = OUT / "latency_screen.json"
    if out_json.exists():
        prior = json.loads(out_json.read_text())
        if prior.get("status") == "COMPLETE":
            return prior
    GEO._write_json(OUT / "L1_RUNNING.json", dict(
        status="RUNNING", pid=os.getpid(), run_ms=SCREEN_MS, candidate=candidate,
        no_kick=True, no_reset=True, no_parameter_step=True, started=GEO._now()))
    S, slow = LC4._fresh_context(candidate, SCREEN_MS)
    p = dataclasses.replace(S["p"], T=SCREEN_MS, dt=LC4.DT)
    t0 = time.time()
    run = run_fcxr_loop(
        p, S["net"], slow=slow, n_steps=int(round(SCREEN_MS / LC4.DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    _, _, numerical, rate, lifecycle, events, af, af_dt, floor = LC4._reduce(
        run, S, baseline)

    win_ms = float(baseline["band"]["win_ms"])
    sm = _smooth_isolated(lifecycle["regimes"])
    bout = first_ictal_bout(sm, win_ms)
    if bout is None:
        onset_ms = offset_ms = None
        pre_rate = float(np.mean(rate))
        post_rate = float("nan")
        ceiling = 0.0
    else:
        b0, b1 = bout
        onset_ms = float(b0 * win_ms)
        ended = b1 + 1 < len(sm)
        offset_ms = float((b1 + 1) * win_ms) if ended else None
        i_on = max(1, int(round(onset_ms / LC4.DT)))
        pre_rate = float(np.mean(rate[:i_on]))
        bout_stop = offset_ms if offset_ms is not None else SCREEN_MS
        if offset_ms is None:
            post_rate = float("nan")
        else:
            j0 = int(round(offset_ms / LC4.DT))
            j1 = min(rate.size, j0 + int(round(2000.0 / LC4.DT)))
            post_rate = float(np.mean(rate[j0:j1])) if j1 > j0 else float("nan")
        ceiling = refractory_ceiling_fraction(
            run["E_spk_bool"], dt_ms=LC4.DT, onset_ms=onset_ms,
            offset_ms=bout_stop, tau_ref_ms=float(S["p"].tau_ref_E))

    slow_f = run["checkpoint"].slow
    gate = adjudicate_latency_screen(
        regimes=lifecycle["regimes"], win_ms=win_ms, events=events,
        current_trace=slow_f.trace_adap_current, current_dt_ms=LC4.DT,
        numerical_safe=not bool(numerical.get("numerical_unsafe")),
        refractory_fraction=ceiling, pre_rate_hz=pre_rate, post_rate_hz=post_rate)
    trace_path = OUT / "latency_screen_traces.npz"
    table = LC4._regional_trace_npz(
        trace_path, slow_f, S, rate=rate, af=af, af_dt=af_dt)
    r_base = float(np.median(rate[:max(1, int(round((gate.get("onset_ms") or SCREEN_MS)
                                                     / LC4.DT)))]))
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=floor,
        rate_hz=rate, dt_ms=LC4.DT, r_base_hz=r_base, table=table,
        onset_ms=gate.get("onset_ms"), offset_ms=gate.get("offset_ms"),
        total_ms=SCREEN_MS)
    rec = dict(
        status="COMPLETE", stage="L1_SCREEN", candidate=candidate, run_ms=SCREEN_MS,
        connection_seed=1, noise_seed=LC4.NOISE_SEED,
        no_kick=True, no_reset=True, no_parameter_step=True,
        gate=gate, lifecycle_classifier=lifecycle, numerical=numerical,
        event_ledger=ledger, traces=str(trace_path),
        max_rate_hz=float(np.max(rate)), mean_rate_hz=float(np.mean(rate)),
        max_adap_current=float(np.max(slow_f.trace_adap_current)),
        wall_s=time.time() - t0, peak_rss_gib=_peak_rss_gib(), finished=GEO._now(),
    )
    GEO._write_json(out_json, rec)
    GEO._write_json(OUT / "L1_DONE.json", dict(
        status="DONE", verdict=gate["verdict"], passed=gate["passed"],
        finished=GEO._now()))
    _clear_running_sentinel(OUT / "L1_RUNNING.json")
    LC4._resource("L1_SCREEN_DONE", wall_s=rec["wall_s"], peak_rss_gib=rec["peak_rss_gib"])
    del run, S
    gc.collect()
    return rec


def _require_l1() -> None:
    path = OUT / "latency_screen.json"
    if not path.exists() or not bool((json.loads(path.read_text()).get("gate") or {}).get("passed")):
        raise SystemExit("LC4d nominal is blocked: L1 latency screen did not pass")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", choices=("screen", "nominal", "confirm"), required=True)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k LC4d execution requires --confirm-run")
    if args.stage == "screen":
        with LC4._stage_lock("lc4d_screen"):
            try:
                result = stage_screen()
            except BaseException as exc:
                GEO._write_json(OUT / "L1_FAILED.json", dict(
                    status="FAILED", error=repr(exc), finished=GEO._now()))
                _clear_running_sentinel(OUT / "L1_RUNNING.json")
                raise
        gate = result["gate"]
    else:
        _require_l1()
        with LC4._stage_lock(f"lc4d_{args.stage}"):
            try:
                result = LC4.stage_nominal() if args.stage == "nominal" else LC4.stage_confirm()
            except BaseException as exc:
                GEO._write_json(OUT / f"L2_{args.stage.upper()}_FAILED.json", dict(
                    status="FAILED", error=repr(exc), finished=GEO._now()))
                raise
        gate = result.get("nominal_gate") or result.get("gate") or {}
    print(json.dumps(dict(status=result.get("status"), stage=result.get("stage"),
                          verdict=gate.get("verdict"), passed=gate.get("passed")), indent=2))


if __name__ == "__main__":
    main()
