#!/usr/bin/env python3
"""Cheap-first FCXR-LC4 functional baseline and frozen-D onset gates.

No simulation runs on import.  Long runs require ``--confirm-run`` and this runner is deliberately
single-worker: every row stores a 40k spike raster long enough to score event statistics.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import fcntl
import gc
import json
import sys
import time
from contextlib import contextmanager

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc4_gate import (  # noqa: E402
    baseline_gate,
    force_matched_candidates,
    onset_surface_gate,
    select_candidate,
    summarize_returning_events,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402


OUT = os.path.join(E01.OUT, "lc4_lifecycle_gate")
SEP = os.path.join(E01.OUT, "percell_separation", "separation_readjudicated.json")
BASELINE_MS = 12000.0
BASELINE_BURN_MS = 2000.0
ONSET_MS = 12000.0
NOISE_SEED = 401
I_EE_SCALE = 272.75518960107513
DT = E01.DT


@contextmanager
def _stage_lock(name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f".{name}.lock")
    with open(path, "w") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"stage {name} is already running") from exc
        yield


def _resource(stage, **extra):
    row = dict(stage=stage, epoch=time.time(), **GEO._meminfo(), **extra)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "resource_log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush(); os.fsync(f.fileno())
    return row


def _preflight(stage):
    m = _resource(f"{stage}_PREFLIGHT")
    if m["mem_available_gib"] < 128.0:
        raise SystemExit(f"{stage}: MemAvailable {m['mem_available_gib']:.1f} GiB < 128 GiB")
    if not os.path.isfile(SEP):
        raise SystemExit(f"missing measured per-cell separation artifact: {SEP}")
    if not os.path.isfile(E01.ARTIFACTS["lc1_baseline"]):
        raise SystemExit("missing frozen LC1 baseline contract")
    return m


def _candidates():
    return force_matched_candidates(GEO._load_json(SEP), recurrent_scale=I_EE_SCALE)


def _cfg(d_field, candidate=None):
    point = GEO._point(GEO.H1_POINT_ID)
    cfg = E01._dynamic_cfg(point)
    d = np.asarray(d_field, dtype=float)
    cfg.update(
        use_z=False, z_frozen_E=1.0 - d,
        use_x=True, x_relay_frozen_E=np.ones(d.size, dtype=float),
    )
    if candidate is not None:
        cfg.update(
            use_m=True, tau_adp=float(candidate["tau_adp_ms"]), eta_m=0.0,
            m_hill_K=float(candidate["K"]), m_hill_n=float(candidate["n"]),
            tau_a_on=float(candidate["tau_a_on_ms"]),
            tau_a_off=float(candidate["tau_a_off_ms"]),
            g_m_max=float(candidate["g_m_max"]),
        )
        if candidate.get("deadzone") is not None:
            cfg["m_hill_deadzone"] = float(candidate["deadzone"])
    return cfg


def _run_frozen(*, tag, role, d_label, d_field, candidate, run_ms):
    arm_dir = os.path.join(OUT, "runs")
    out_json = os.path.join(arm_dir, f"{tag}.json")
    if os.path.isfile(out_json):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior

    os.makedirs(arm_dir, exist_ok=True)
    S = PP.build_substrate(1)
    install_registered_noise_rng(S["net"])
    cfg = _cfg(d_field, candidate)
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S))
    S["net"]["rng"] = np.random.default_rng(NOISE_SEED)
    p = dataclasses.replace(S["p"], T=float(run_ms), dt=DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], slow=slow, n_steps=int(round(run_ms / DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])

    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    wins, numerical, _ = LC1R._reduce_run_windows(
        res, run["checkpoint"].slow, S, DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    events, af, af_dt, _floor, _ = OLD._events_from_res(
        res, DT, event_bar=float(baseline["frozen_event_bar"]))
    burn = BASELINE_BURN_MS if role in ("baseline_control", "baseline_candidate") else 0.0
    summary = summarize_returning_events(events, start_ms=burn, end_ms=run_ms)
    slow_f = run["checkpoint"].slow
    current_max = (float(max(slow_f.trace_adap_current))
                   if slow_f.trace_adap_current else 0.0)
    rec = dict(
        status="COMPLETE", tag=tag, role=role, d_label=d_label,
        mean_D=float(np.mean(np.asarray(d_field, float))), run_ms=float(run_ms),
        connection_seed=1, noise_seed=NOISE_SEED,
        candidate=candidate, no_kick=True, no_reset=True, no_parameter_step=True,
        lifecycle=lifecycle, departed=bool(lifecycle.get("bout") is not None),
        numerical=numerical, summary=summary,
        adap_current_max=current_max,
        adap_current_fraction=current_max / I_EE_SCALE,
        max_rate_hz=float(np.max(run["rate_E"])), mean_rate_hz=float(np.mean(run["rate_E"])),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        finished=GEO._now(),
    )
    GEO._write_json(out_json, rec)
    stride = max(1, int(round(10.0 / DT)))
    np.savez_compressed(
        out_json.replace(".json", "_traces.npz"),
        rate_dt_ms=np.asarray([10.0], np.float32),
        rate_E=np.asarray(run["rate_E"][::stride], np.float32),
        af=np.asarray(af, np.float32), af_bin_ms=np.asarray([af_dt], np.float32),
        adap_current=np.asarray(slow_f.trace_adap_current[::stride], np.float32),
        a_mean=np.asarray(slow_f.trace_a_mean[::stride], np.float32),
    )
    _resource(f"{tag}_DONE", wall_s=rec["wall_s"], peak_rss_gib=rec["peak_rss_gib"])
    del run, res, S
    gc.collect()
    return rec


def stage_baseline():
    _preflight("BASELINE")
    fields, _ = GEO._primary_fields()
    specs = [dict(tag="baseline_control", role="baseline_control", candidate=None)]
    specs += [dict(tag=f"baseline_n{int(c['n'])}", role="baseline_candidate", candidate=c)
              for c in _candidates()]
    GEO._write_json(os.path.join(OUT, "BASELINE_RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), rows=[s["tag"] for s in specs],
                         run_ms=BASELINE_MS, started=GEO._now()))
    rows = []
    for s in specs:
        r = _run_frozen(d_label="D_healthy", d_field=fields["D_healthy"],
                        run_ms=BASELINE_MS, **s)
        rows.append(r)
        print(f"[baseline] {r['tag']}: {r['summary']['n_returning']} returning, "
              f"{r['summary']['event_rate_hz']:.2f}/s, current {r['adap_current_fraction']:.4%}",
              flush=True)
    control = next(r for r in rows if r["role"] == "baseline_control")
    judged = []
    for r in rows:
        if r["role"] != "baseline_candidate":
            continue
        r = dict(r)
        r["gate"] = baseline_gate(
            r["summary"], control["summary"],
            numerical_safe=not bool(r["numerical"].get("numerical_unsafe")),
            sustained_bout=bool(r["departed"]), max_current=float(r["adap_current_max"]),
            recurrent_scale=I_EE_SCALE)
        judged.append(r)
    selected = select_candidate(judged)
    verdict = dict(
        status="COMPLETE", stage="F0", control=control, candidates=judged,
        selected_candidate=(None if selected is None else selected["candidate"]),
        verdict=("BASELINE_CANDIDATE_SELECTED" if selected is not None
                 else "NO_BASELINE_PRESERVING_HILL_CANDIDATE"),
        stopped=selected is None, completed=GEO._now())
    GEO._write_json(os.path.join(OUT, "baseline_verdict.json"), verdict)
    GEO._write_json(os.path.join(OUT, "BASELINE_DONE.json"),
                    dict(status="DONE", verdict=verdict["verdict"], finished=GEO._now()))
    return verdict


def stage_onset():
    _preflight("ONSET")
    bp = os.path.join(OUT, "baseline_verdict.json")
    if not os.path.isfile(bp):
        raise SystemExit("F1 needs baseline_verdict.json")
    base = GEO._load_json(bp)
    candidate = base.get("selected_candidate")
    if candidate is None:
        raise SystemExit("F1 blocked: no baseline-preserving candidate")
    fields, _ = GEO._primary_fields()
    specs = [dict(tag="onset_control_D10", role="positive_control", d_label="D10",
                  d_field=fields["D10"], candidate=None)]
    specs += [dict(tag=f"onset_n{int(candidate['n'])}_{label}", role="candidate",
                   d_label=label, d_field=fields[label], candidate=candidate)
              for label in ("D_healthy", "D10", "D30", "D50")]
    GEO._write_json(os.path.join(OUT, "ONSET_RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), rows=[s["tag"] for s in specs],
                         run_ms=ONSET_MS, started=GEO._now()))
    rows = []
    for s in specs:
        r = _run_frozen(run_ms=ONSET_MS, **s); rows.append(r)
        print(f"[onset] {r['tag']}: D={r['mean_D']:.4f}, departed={r['departed']}, "
              f"label={r['lifecycle']['label']}", flush=True)
    gate = onset_surface_gate(rows)
    verdict = dict(status="COMPLETE", stage="F1", candidate=candidate, rows=rows,
                   gate=gate, stopped=not gate["passed"], completed=GEO._now())
    GEO._write_json(os.path.join(OUT, "onset_surface_verdict.json"), verdict)
    GEO._write_json(os.path.join(OUT, "ONSET_DONE.json"),
                    dict(status="DONE", verdict=gate["verdict"], finished=GEO._now()))
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", choices=("baseline", "onset"), required=True)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k LC4 gate requires --confirm-run")
    with _stage_lock(args.stage):
        try:
            result = stage_baseline() if args.stage == "baseline" else stage_onset()
        except BaseException as exc:
            GEO._write_json(os.path.join(OUT, f"{args.stage.upper()}_FAILED.json"),
                            dict(status="FAILED", error=repr(exc), finished=GEO._now()))
            raise
    print(json.dumps({k: result.get(k) for k in ("status", "stage", "verdict", "stopped")},
                     indent=2))


if __name__ == "__main__":
    main()
