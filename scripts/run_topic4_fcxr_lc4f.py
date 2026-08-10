#!/usr/bin/env python3
"""Execute the locked LC4f X-depth screen and conditional lifecycle."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc4_lifecycle as LC4  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger  # noqa: E402
from src.topic4_fcxr_lc4_lifecycle import (  # noqa: E402
    _smooth_isolated, first_ictal_bout, refractory_ceiling_fraction,
)
from src.topic4_fcxr_lc4f import adjudicate_screen, derive_candidate  # noqa: E402
from src.topic4_fcxr_lc4b_deadzone import sha256_file  # noqa: E402

OUT = (ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
       / "lc4f_x_depth_closure")
BASE = ROOT / "results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability"
LOCK = OUT / "candidate_lock.json"
SCREEN_MS = 22000.0
LC4.OUT = str(OUT)

SOURCES = (
    "src/topic4_fcxr_lc4f.py", "scripts/run_topic4_fcxr_lc4f.py",
    "scripts/run_topic4_fcxr_lc4f_autopilot.sh",
    "docs/superpowers/specs/2026-08-10-topic4-fcxr-lc4f-x-depth-closure-design.md",
    "docs/superpowers/plans/2026-08-10-topic4-fcxr-lc4f-x-depth-closure.md",
    "src/snn_engine/mz_slow_vars.py", "scripts/run_topic4_fcxr_lc4_lifecycle.py",
)


def _load(path):
    return json.loads(Path(path).read_text())


def _artifact_key(path: Path) -> str:
    """Keep repo artifacts relative and cross-worktree evidence explicit."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def stage_lock():
    lc4c = _load(BASE / "lc4c_entry_offset_alignment/candidate_lock.json")
    rows = [_load(BASE / f"hill_placement_sweep/arm_gate76p64_Ky{k}p00.json")
            for k in (3, 4, 5)]
    dx = _load(BASE / "dx_arbitration_probe/dx_arbitration.json")
    sensor = _load(E01.ARTIFACTS["lc1_sensor"])
    candidate = derive_candidate(lc4c, *rows, dx, y_gate=sensor["y_gate_q999"])
    artifacts = [BASE / "lc4c_entry_offset_alignment/candidate_lock.json",
                 *[BASE / f"hill_placement_sweep/arm_gate76p64_Ky{k}p00.json"
                   for k in (3, 4, 5)], BASE / "dx_arbitration_probe/dx_arbitration.json",
                 Path(E01.ARTIFACTS["lc1_sensor"])]
    payload = dict(status="X0_PASS", verdict="X_DEPTH_CANDIDATE_IDENTIFIABLE",
                   candidate=candidate,
                   artifacts={_artifact_key(p): sha256_file(p) for p in artifacts},
                   sources={p: sha256_file(ROOT / p) for p in SOURCES},
                   created=GEO._now())
    GEO._write_json(LOCK, payload)
    return payload


def _candidate():
    lock = _load(LOCK)
    if lock.get("status") != "X0_PASS":
        raise SystemExit("LC4f requires X0_PASS lock")
    for name, expected in lock["artifacts"].items():
        path = Path(name) if Path(name).is_absolute() else ROOT / name
        if sha256_file(path) != expected:
            raise SystemExit(f"LC4f artifact drift: {name}")
    for rel, expected in lock["sources"].items():
        if sha256_file(ROOT / rel) != expected:
            raise SystemExit(f"LC4f source drift: {rel}")
    return lock["candidate"]


def _cfg(candidate):
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    cfg.update(use_m=False, theta_h_lc2=float(candidate["theta_h_lc2"]),
               y_gate=float(candidate["y_gate"]), K_y=float(candidate["K_y"]),
               tau_y=float(candidate["tau_y"]), tau_x_down=float(candidate["tau_x_down"]),
               tau_x_up=float(candidate["tau_x_up"]), x_min=float(candidate["x_min"]),
               hill_n=int(candidate["hill_n"]))
    return cfg


LC4._candidate = _candidate
LC4._cfg = _cfg


def stage_screen():
    LC4._preflight("X1_SCREEN")
    candidate = _candidate()
    out_json = OUT / "x_depth_screen.json"
    GEO._write_json(OUT / "X1_RUNNING.json", dict(status="RUNNING", pid=os.getpid(),
                        run_ms=SCREEN_MS, candidate=candidate, started=GEO._now()))
    S, slow = LC4._fresh_context(candidate, SCREEN_MS)
    p = dataclasses.replace(S["p"], T=SCREEN_MS, dt=LC4.DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], slow=slow, n_steps=int(round(SCREEN_MS / LC4.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    baseline = _load(E01.ARTIFACTS["lc1_baseline"])
    _, _, numerical, rate, lifecycle, events, af, af_dt, floor = LC4._reduce(run, S, baseline)
    sm = _smooth_isolated(lifecycle["regimes"])
    bout = first_ictal_bout(sm, float(baseline["band"]["win_ms"]))
    if bout is None:
        pre_rate, post_rate, ceiling = float(np.mean(rate)), float("nan"), 0.0
    else:
        b0, b1 = bout; onset = b0 * float(baseline["band"]["win_ms"])
        offset = ((b1 + 1) * float(baseline["band"]["win_ms"])
                  if b1 + 1 < len(sm) else None)
        pre_rate = float(np.mean(rate[:max(1, int(round(onset / LC4.DT)))]))
        if offset is None:
            post_rate = float("nan")
        else:
            j0 = int(round(offset / LC4.DT)); j1 = min(rate.size, j0 + int(round(2000 / LC4.DT)))
            post_rate = float(np.mean(rate[j0:j1]))
        ceiling = refractory_ceiling_fraction(run["E_spk_bool"], dt_ms=LC4.DT,
                    onset_ms=onset, offset_ms=(offset or SCREEN_MS), tau_ref_ms=float(S["p"].tau_ref_E))
    gate = adjudicate_screen(regimes=lifecycle["regimes"],
              win_ms=float(baseline["band"]["win_ms"]), events=events,
              numerical_safe=not numerical.get("numerical_unsafe", True),
              refractory_fraction=ceiling, pre_rate_hz=pre_rate, post_rate_hz=post_rate,
              m_current_max=float(np.max(run["checkpoint"].slow.trace_adap_current)))
    trace_path = OUT / "x_depth_screen_traces.npz"
    table = LC4._regional_trace_npz(trace_path, run["checkpoint"].slow, S, rate=rate,
            af=af, af_dt=af_dt, extra={"x_mean": np.asarray(run["checkpoint"].slow.trace_x_relay_mean[::200], np.float32)})
    ledger = build_event_ledger(events=events, af=af, af_bin_ms=af_dt, floor_af=floor,
             rate_hz=rate, dt_ms=LC4.DT, r_base_hz=float(np.median(rate)), table=table,
             onset_ms=gate.get("onset_ms"), offset_ms=gate.get("offset_ms"), total_ms=SCREEN_MS)
    rec = dict(status="COMPLETE", stage="X1_SCREEN", candidate=candidate, run_ms=SCREEN_MS,
               connection_seed=1, noise_seed=401, no_kick=True, no_reset=True,
               no_parameter_step=True, gate=gate, lifecycle_classifier=lifecycle,
               numerical=numerical, event_ledger=ledger, traces=str(trace_path),
               final_X_mean=float(np.mean(run["checkpoint"].slow.x_relay)),
               min_X_mean=float(np.min(run["checkpoint"].slow.trace_x_relay_mean)),
               wall_s=time.time()-t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
               finished=GEO._now())
    GEO._write_json(out_json, rec)
    GEO._write_json(OUT / "X1_DONE.json", dict(status="DONE", verdict=gate["verdict"],
                        passed=gate["passed"], finished=GEO._now()))
    try: (OUT / "X1_RUNNING.json").unlink()
    except FileNotFoundError: pass
    LC4._resource("X1_SCREEN_DONE", wall_s=rec["wall_s"], peak_rss_gib=rec["peak_rss_gib"])
    del run, S; gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", choices=("lock", "screen", "nominal", "confirm"), required=True)
    a = ap.parse_args()
    if a.stage == "lock":
        print(json.dumps(stage_lock(), indent=2)); return
    if not a.confirm_run: raise SystemExit("40k LC4f execution requires --confirm-run")
    with LC4._stage_lock(f"lc4f_{a.stage}"):
        result = stage_screen() if a.stage == "screen" else (LC4.stage_nominal() if a.stage == "nominal" else LC4.stage_confirm())
    gate = result.get("gate") or result.get("nominal_gate") or {}
    print(json.dumps({"status": result.get("status"), "stage": result.get("stage"),
                      "verdict": gate.get("verdict"), "passed": gate.get("passed")}, indent=2))


if __name__ == "__main__": main()
