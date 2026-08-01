#!/usr/bin/env python3
"""FCXR-LC2 E5 dynamic Z/H/X no-kick pilot, unlocked only by frozen geometry."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_mz_fcxr as FCXR  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402


OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core", "closed_loop_exploration")
BASELINE_CONTRACT = ("/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-lc1/results/"
                     "topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure/"
                     "baseline_contract_seed1.json")
DT = 0.05
T_MS = 20000.0
MAX_DYNAMIC_CANDIDATES = 1
X_CFG = dict(x_min=0.1, tau_y=120.0, tau_x_down=1000.0, tau_x_up=10000.0,
             K_y=5.0, y_gate=76.63856219587187, hill_n=4)
Z_CFG = dict(regime="q75", I_th_EI=95.19851312666987, tau_z=5000.0)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load(path):
    with open(path) as f:
        return json.load(f)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _rolling(x, n):
    x = np.asarray(x, float)
    if x.size < n:
        return np.empty(0)
    return np.convolve(x, np.ones(n) / n, mode="valid")


def contiguous_intervals(mask, dt, minimum_ms):
    mask = np.asarray(mask, bool)
    padded = np.r_[False, mask, False]
    start = np.flatnonzero(np.diff(padded.astype(np.int8)) == 1)
    end = np.flatnonzero(np.diff(padded.astype(np.int8)) == -1)
    return [(float(a * dt), float(b * dt)) for a, b in zip(start, end)
            if (b - a) * dt >= float(minimum_ms)]


def lifecycle_readout(rate, dt, events, *, high_bar_hz=20.0):
    env_dt = 10.0
    raw_n = max(1, int(round(env_dt / dt)))
    n = (len(rate) // raw_n) * raw_n
    env = np.asarray(rate[:n], float).reshape(-1, raw_n).mean(axis=1) if n else np.empty(0)
    smooth_n = max(1, int(round(300.0 / env_dt)))
    smooth = _rolling(env, smooth_n)
    centre_offset = 0.5 * (smooth_n - 1) * env_dt
    intervals = [(a + centre_offset, b + centre_offset)
                 for a, b in contiguous_intervals(smooth >= high_bar_hz, env_dt, 1000.0)]
    onset = intervals[0][0] if intervals else None
    offset = intervals[0][1] if intervals else None
    pre = float(onset) if onset is not None else None
    post_events = [e for e in events if offset is not None and e.get("returned", False)
                   and float(e["t_on_ms"]) >= offset + 250.0]
    post_span = (float(post_events[-1]["t_off_ms"] - post_events[0]["t_on_ms"])
                 if len(post_events) >= 2 else 0.0)
    return dict(high_intervals_ms=intervals, onset_ms=onset, offset_ms=offset,
                pre_interictal_ms=pre, post_returning_events=len(post_events),
                post_event_span_ms=post_span, env_dt_ms=env_dt,
                env_trace=env[::5].tolist(), env_trace_dt_ms=env_dt * 5)


def recovery_stats(events, offset_ms, T_ms, baseline):
    start = None if offset_ms is None else float(offset_ms) + 250.0
    ret = [] if start is None else [e for e in events if e.get("returned", False)
                                    and float(e["t_on_ms"]) >= start]
    exposure_s = 0.0 if start is None else max(0.0, (float(T_ms) - start) / 1000.0)
    onsets = np.asarray([float(e["t_on_ms"]) for e in ret], float)
    iei = np.diff(onsets)
    iei_cv = float(np.std(iei) / np.mean(iei)) if iei.size >= 2 and np.mean(iei) > 0 else float("nan")
    rate = float(len(ret) / exposure_s) if exposure_s > 0 else float("nan")
    dur = float(np.median([e["dur_ms"] for e in ret])) if ret else float("nan")
    part = float(np.median([e["peak_ext"] for e in ret])) if ret else float("nan")
    base_rate = float(baseline["n_returning"] / (baseline["T"] / 1000.0))
    base_dur = float(np.median(baseline["event_durations_ms"]))
    base_part = float(np.median(baseline["event_participation"]))
    match = bool(exposure_s >= 8.0 and np.isfinite(rate) and 0.5 * base_rate <= rate <= 2.0 * base_rate
                 and np.isfinite(dur) and 0.5 * base_dur <= dur <= 2.0 * base_dur
                 and np.isfinite(part) and 0.5 * base_part <= part <= 2.0 * base_part
                 and np.isfinite(iei_cv) and iei_cv >= 0.30)
    return dict(n_returning=len(ret), exposure_s=exposure_s, event_rate_hz=rate,
                median_duration_ms=dur, median_participation=part, iei_cv=iei_cv,
                baseline=dict(event_rate_hz=base_rate, median_duration_ms=base_dur,
                              median_participation=base_part), statistical_neighbourhood_match=match)


def cmd_manifest(_args):
    fmap = os.path.join(OUT, "frozen_fork_map.json")
    if not os.path.isfile(fmap):
        raise SystemExit("frozen_fork_map.json is required")
    f = _load(fmap)
    candidates = [v for v in f["candidate_verdicts"] if v["label"] == "H_BASIN_CANDIDATE"]
    if not candidates:
        payload = dict(stage="E5_DYNAMIC", status="NOT_UNLOCKED", rows=[], created=_now())
    else:
        by_id = {r["candidate_run_id"]: r for r in f["rows"] if r["arm"] == "C"}
        rows = []
        for v in candidates[:MAX_DYNAMIC_CANDIDATES]:
            c = by_id[v["candidate_run_id"]]
            for arm in ("X_on", "X_off_matched_sensor"):
                rows.append(dict(index=len(rows), arm=arm, candidate_run_id=v["candidate_run_id"],
                                 tau_ms=c["tau_ms"], theta=c["theta"], k=c["k"], rho=c["rho"],
                                 connection_seed=1, noise_seed=401, T_ms=T_MS, no_kick=True,
                                 z=Z_CFG, x=X_CFG))
        payload = dict(stage="E5_DYNAMIC", status="LOCKED", code_head=_git_head(),
                       baseline_contract=dict(path=BASELINE_CONTRACT, sha256=_sha(BASELINE_CONTRACT)),
                       max_dynamic_candidates=MAX_DYNAMIC_CANDIDATES, rows=rows, created=_now())
    FCXR._write_json(os.path.join(OUT, "dynamic_pilot_manifest.json"), payload)
    print(json.dumps(payload, indent=2))


def _cfg(row, NE):
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True,
                       fail_on_clip=False, rec_sat_g=21.6)
    cfg.update(use_z=True, tau_z=float(Z_CFG["tau_z"]), I_th_EI=float(Z_CFG["I_th_EI"]),
               use_x=True, use_h_lc2=True, tau_h_lc2=float(row["tau_ms"]),
               theta_h_lc2=float(row["theta"]), k_h_lc2=float(row["k"]),
               rho_h_lc2=float(row["rho"]), h_lc2_init_E=np.zeros(NE), **X_CFG)
    if row["arm"] == "X_off_matched_sensor":
        cfg["x_relay_frozen_E"] = np.ones(NE)
    return cfg


def _run(row):
    S = PP.build_substrate(int(row["connection_seed"]))
    p = dataclasses.replace(S["p"], T=float(row["T_ms"]), dt=DT)
    snapshot_steps = {int(round(t / DT)): f"t{int(t)}" for t in np.arange(0.0, row["T_ms"], 100.0)}
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**_cfg(row, S["NE"])), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    S["net"]["rng"] = np.random.default_rng(int(row["noise_seed"]))
    t0 = time.time()
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"],
                        early_stop_runaway=False)
    wall = time.time() - t0
    baseline = _load(BASELINE_CONTRACT)
    bar = float(baseline["frozen_event_bar"])
    events, _, _, _, _ = OLD._events_from_res(res, DT, event_bar=bar)
    rate = np.asarray(res["rate_E"], float)
    readout = lifecycle_readout(rate, DT, events)
    readout["recovery"] = recovery_stats(events, readout["offset_ms"], row["T_ms"], baseline)
    onset_snapshot = None
    if readout["onset_ms"] is not None and slow.snapshots:
        target = float(readout["onset_ms"])
        label, snap = min(slow.snapshots.items(), key=lambda kv: abs(kv[1]["step"] * DT - target))
        snap_path = os.path.join(OUT, "dynamic_cells",
                                 f"{row['candidate_run_id']}__{row['arm']}_onset_snapshot.npz")
        FCXR._write_npz(snap_path, **{k: v for k, v in snap.items() if isinstance(v, np.ndarray)})
        onset_snapshot = dict(label=label, time_ms=float(snap["step"] * DT), path=snap_path,
                              z_mean=float(np.mean(snap["z_E"])), h_mean=float(np.mean(snap["h_E"])),
                              x_mean=float(np.mean(snap["x_E"])), y_mean=float(np.mean(snap["y_E"])))
    numerical = LC1._numerical(S, res, slow, DT)
    stride = max(1, int(round(10.0 / DT)))
    payload = dict(
        **row, numerical=numerical, readout=readout, onset_snapshot=onset_snapshot,
        returning_events=[e for e in events if e.get("returned", False)],
        mean_rate_hz=float(np.mean(rate)), end_rate_hz=float(np.mean(rate[-int(1000 / DT):])),
        h_max=float(np.max(slow.trace_h_lc2_max)), x_min_reached=float(np.min(slow.trace_x_relay_min)),
        y_max=float(np.max(slow.trace_y_max)), z_end_mean=float(np.mean(slow.z[:S["NE"]])),
        wall_s=round(wall, 2), peak_rss_gib=round(_rss_gib(), 3), trace_dt_ms=10.0,
        rate_trace=rate[::stride].tolist(), h_trace=np.asarray(slow.trace_h_lc2_mean)[::stride].tolist(),
        x_trace=np.asarray(slow.trace_x_relay_mean)[::stride].tolist(),
        z_trace=np.asarray(slow.trace_z_mean)[::stride].tolist(), finished=_now())
    path = os.path.join(OUT, "dynamic_cells", f"{row['candidate_run_id']}__{row['arm']}.json")
    FCXR._write_json(path, payload)
    return payload


def classify_pair(rows):
    by = {r["arm"]: r for r in rows}
    on, off = by["X_on"], by["X_off_matched_sensor"]
    if on["numerical"]["numerical_unsafe"] or off["numerical"]["numerical_unsafe"]:
        label = "dynamic_numerical_failure"
    elif on["readout"]["onset_ms"] is None:
        label = "dynamic_z_misses_h_basin"
    elif on["readout"]["pre_interictal_ms"] < 8000.0:
        label = "pre_interictal_too_short"
    elif on["readout"]["offset_ms"] is None:
        label = "x_on_no_offset"
    elif (off["readout"]["offset_ms"] is not None and
          off["readout"]["offset_ms"] < on["readout"]["offset_ms"] + 1000.0):
        label = "x_no_causal_duration_extension"
    elif not on["readout"]["recovery"]["statistical_neighbourhood_match"]:
        label = "offset_without_8s_returning_ied_recovery"
    else:
        label = "CORE_LIFECYCLE_CANDIDATE"
    return dict(label=label, x_on=on["readout"], x_off=off["readout"])


def cmd_run(args):
    if not args.confirm_run:
        raise SystemExit("--confirm-run is required")
    FCXR._assert_engine_blessed()
    m = _load(os.path.join(OUT, "dynamic_pilot_manifest.json"))
    if m["status"] != "LOCKED":
        raise SystemExit("dynamic stage is not unlocked")
    os.makedirs(os.path.join(OUT, "dynamic_cells"), exist_ok=True)
    rows = []
    for r in m["rows"]:
        path = os.path.join(OUT, "dynamic_cells", f"{r['candidate_run_id']}__{r['arm']}.json")
        out = _load(path) if os.path.isfile(path) else _run(r)
        rows.append(out)
        print(f"[E5] {r['candidate_run_id']} {r['arm']} onset={out['readout']['onset_ms']} "
              f"offset={out['readout']['offset_ms']} RSS={out['peak_rss_gib']:.2f}GiB", flush=True)
    verdicts = []
    for cid in sorted({r["candidate_run_id"] for r in rows}):
        verdicts.append(dict(candidate_run_id=cid,
                             **classify_pair([r for r in rows if r["candidate_run_id"] == cid])))
    payload = dict(stage="E5_DYNAMIC", status="COMPLETE", rows=rows,
                   candidate_verdicts=verdicts, finished=_now())
    FCXR._write_json(os.path.join(OUT, "dynamic_pilot.json"), payload)
    FCXR._write_json(os.path.join(OUT, "E5_DONE.json"),
                     dict(stage="E5", status="COMPLETE", candidate_verdicts=verdicts, finished=_now()))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    run = sub.add_parser("run")
    run.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    cmd_manifest(args) if args.cmd == "manifest" else cmd_run(args)


if __name__ == "__main__":
    main()
