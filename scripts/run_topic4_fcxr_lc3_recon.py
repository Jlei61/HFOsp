#!/usr/bin/env python3
"""FCXR-LC3 E4: exactly three no-kick dynamic reconnaissance trajectories."""
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
import hashlib
import json
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import H1_POINT_ID  # noqa: E402
from src.topic4_fcxr_lc3_recon import (  # noqa: E402
    nearest_snapshot_labels,
    reconnaissance_verdict,
    select_landmark_times,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402


OUT = os.path.join(E01.OUT, "dynamic_reconnaissance")
LOCK = os.path.join(OUT, "execution_lock.json")
MANIFEST = os.path.join(OUT, "manifest.json")
NOISES = (401, 405, 406)
T1_MS = 32000.0
T_CAP_MS = 45000.0
SNAP_MS = 250.0
RECON_SOURCES = (
    "src/topic4_fcxr_lc3.py",
    "src/topic4_fcxr_lc3_recon.py",
    "src/snn_engine/mz_slow_vars.py",
    "scripts/run_topic4_fcxr_lc3.py",
    "scripts/run_topic4_fcxr_lc3_geometry.py",
    "scripts/run_topic4_fcxr_lc3_recon.py",
    "scripts/run_topic4_fcxr_lc3_recon_autopilot.sh",
    "docs/superpowers/specs/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md",
    "docs/superpowers/plans/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability.md",
)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _meminfo():
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(mem_available_gib=d["MemAvailable"] / 1024.0 / 1024.0,
                swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0,
                self_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0)


@contextmanager
def _stage_lock(name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f".{name}.lock"), "a+") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"recon stage already running: {name}") from exc
        yield


def _load(path):
    with open(path) as f:
        return json.load(f)


def cmd_lock(_args):
    geometry_lock = GEO.GEOMETRY_LOCK
    if not os.path.isfile(geometry_lock) or _load(geometry_lock).get("status") != "LOCKED":
        raise SystemExit("geometry execution lock is required; geometry outcome is not required")
    missing = [rel for rel in RECON_SOURCES if not os.path.isfile(os.path.join(ROOT, rel))]
    if missing:
        raise SystemExit("missing recon sources: " + ", ".join(missing))
    initial = _load(os.path.join(E01.OUT, "execution_lock.json"))
    payload = dict(
        status="LOCKED", schema="fcxr-lc3-recon-lock-1.0",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip(),
        geometry_lock_sha256=_sha(geometry_lock),
        baseline_sha256=_sha(E01.ARTIFACTS["lc1_baseline"]),
        sensor_sha256=_sha(E01.ARTIFACTS["lc1_sensor"]),
        sources={rel: _sha(os.path.join(ROOT, rel)) for rel in RECON_SOURCES},
        engine_hashes=initial["engine_hashes"], resource_at_lock=_meminfo(), locked_at=_now(),
        outcome_scope="reconnaissance_not_parameter_acceptance",
    )
    _write_json(LOCK, payload)
    print(json.dumps(dict(status="LOCKED", git_head=payload["git_head"]), indent=2))


def _assert_lock():
    if not os.path.isfile(LOCK):
        raise SystemExit("missing recon execution lock")
    lock = _load(LOCK)
    if lock.get("status") != "LOCKED":
        raise SystemExit("recon lock is not active")
    if _sha(GEO.GEOMETRY_LOCK) != lock["geometry_lock_sha256"]:
        raise SystemExit("geometry lock drift")
    if _sha(E01.ARTIFACTS["lc1_baseline"]) != lock["baseline_sha256"]:
        raise SystemExit("baseline drift")
    for rel, expected in lock["sources"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"recon source drift: {rel}")
    for rel, expected in lock["engine_hashes"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"engine drift: {rel}")
    return lock


def cmd_manifest(_args):
    lock = _assert_lock()
    rows = [dict(
        index=i, row_id=f"H1_q75_unretunedX_noise{noise}", connection_seed=1,
        noise_seed=noise, point_id=H1_POINT_ID, Z="archived_q75",
        X="current_unretuned_LC1", M=False, K=False, A=False, ELR=False,
        no_kick=True, no_reset=True, no_parameter_step=True,
        min_ms=T1_MS, cap_ms=T_CAP_MS,
        output_path=os.path.join(OUT, f"recon_noise{noise}.json"),
    ) for i, noise in enumerate(NOISES)]
    payload = dict(status="LOCKED", schema="fcxr-lc3-recon-manifest-1.0", rows=rows,
                   lock_sha256=_sha(LOCK), git_head=lock["git_head"], created=_now())
    _write_json(MANIFEST, payload)
    print(json.dumps(dict(status="LOCKED", n_rows=3, noises=list(NOISES)), indent=2))


def _assert_manifest():
    _assert_lock()
    if not os.path.isfile(MANIFEST):
        raise SystemExit("missing recon manifest")
    m = _load(MANIFEST)
    if m.get("status") != "LOCKED" or len(m.get("rows", [])) != 3:
        raise SystemExit("invalid recon manifest")
    if [row["noise_seed"] for row in m["rows"]] != list(NOISES):
        raise SystemExit("recon noise set drift")
    if m["lock_sha256"] != _sha(LOCK):
        raise SystemExit("recon manifest lock drift")
    return m


def _run_once(row):
    out_json = row["output_path"]
    done_json = out_json.replace(".json", ".DONE.json")
    if os.path.isfile(out_json) and os.path.isfile(done_json):
        prior = _load(out_json)
        if prior.get("status") == "COMPLETE":
            return prior
    before = _meminfo()
    if before["mem_available_gib"] < 128.0:
        raise RuntimeError("recon resource gate requires 128 GiB MemAvailable")
    running = out_json.replace(".json", ".RUNNING.json")
    _write_json(running, dict(status="RUNNING", pid=os.getpid(), row_id=row["row_id"],
                              resource=before, started=_now()))
    S = PP.build_substrate(1)
    point = GEO._point(H1_POINT_ID)
    cfg = E01._dynamic_cfg(point)
    snapshot_steps = {int(round(t / E01.DT)): f"t{int(t)}"
                      for t in np.arange(0.0, T_CAP_MS + SNAP_MS, SNAP_MS)}
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    S["net"]["rng"] = np.random.default_rng(int(row["noise_seed"]))
    p1 = dataclasses.replace(S["p"], T=T1_MS, dt=E01.DT)
    t0 = time.time()
    first = run_fcxr_loop(
        p1, S["net"], slow=slow, n_steps=int(round(T1_MS / E01.DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"],
    )
    baseline = _load(E01.ARTIFACTS["lc1_baseline"])
    wins1, num1, _ = LC1R._reduce_run_windows(
        first, first["checkpoint"].slow, S, E01.DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle1 = classify_lifecycle(wins1, baseline["band"])
    extend = lifecycle1.get("bout") is not None
    if extend:
        extra_ms = T_CAP_MS - T1_MS
        p2 = dataclasses.replace(S["p"], T=extra_ms, dt=E01.DT)
        second = run_fcxr_loop(
            p2, S["net"], start=first["checkpoint"],
            n_steps=int(round(extra_ms / E01.DT)), capture_final=True,
            store_spikes=True, v_th_per_neuron=S["vth"],
        )
        rate_e = np.concatenate([first["rate_E"], second["rate_E"]])
        rate_i = np.concatenate([first["rate_I"], second["rate_I"]])
        spikes = np.concatenate([first["E_spk_bool"], second["E_spk_bool"]], axis=0)
        final = second
        total_ms = T_CAP_MS
    else:
        rate_e, rate_i, spikes = first["rate_E"], first["rate_I"], first["E_spk_bool"]
        final = first
        total_ms = T1_MS
    res = dict(rate_E=rate_e, rate_I=rate_i, E_spk_bool=spikes)
    wins, numerical, _ = LC1R._reduce_run_windows(
        res, final["checkpoint"].slow, S, E01.DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    events, _af, _af_dt, _floor, _ = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))

    bout = lifecycle.get("bout")
    ceiling_frac = 0.0
    if bout is not None:
        b0 = int(round(bout[0] * baseline["band"]["win_ms"] / E01.DT))
        b1 = int(round((bout[1] + 1) * baseline["band"]["win_ms"] / E01.DT))
        seg = spikes[b0:min(b1, spikes.shape[0])]
        if seg.size:
            per_cell = seg.sum(axis=0) / (seg.shape[0] * E01.DT * 1e-3)
            ceiling_frac = float(np.mean(per_cell >= 0.8 * 1000.0 / S["p"].tau_ref_E))
    slow_final = final["checkpoint"].slow
    xtrace = np.asarray(slow_final.trace_x_relay_mean, dtype=float)
    x_peak_ms = float(np.argmin(xtrace) * E01.DT) if xtrace.size else None
    onset_ms = float(bout[0] * baseline["band"]["win_ms"]) if bout is not None else None
    x_after = (None if onset_ms is None or x_peak_ms is None
               else bool(xtrace.min() < xtrace[0] and x_peak_ms >= onset_ms))
    verdict = reconnaissance_verdict(
        lifecycle=lifecycle, numerical_unsafe=bool(numerical["numerical_unsafe"]),
        refractory_ceiling_fraction=ceiling_frac, x_activates_after_onset=x_after)

    snap_times = {label: float(snap["step"] * E01.DT)
                  for label, snap in slow_final.snapshots.items()
                  if float(snap["step"] * E01.DT) <= total_ms}
    targets = select_landmark_times(lifecycle, win_ms=float(baseline["band"]["win_ms"]),
                                    total_ms=total_ms)
    selected = nearest_snapshot_labels(snap_times, targets)
    snaps = [slow_final.snapshots[selected[name]["snapshot_label"]] for name in selected]
    names = list(selected)
    d_fields = np.stack([1.0 - np.asarray(s["z_E"], float) for s in snaps])
    x_fields = np.stack([np.asarray(s["x_E"], float) for s in snaps])
    h_fields = np.stack([np.asarray(s["h_E"], float) for s in snaps])
    y_fields = np.stack([np.asarray(s["y_E"], float) for s in snaps])
    masks = GEO._region_masks(S)
    field_summaries = []
    for i, name in enumerate(names):
        field_summaries.append(dict(
            landmark=name, **selected[name], mean_D=float(d_fields[i].mean()),
            mean_a_X=float(x_fields[i].mean()), mean_H=float(h_fields[i].mean()),
            mean_y=float(y_fields[i].mean()),
            D_regions={k: float(d_fields[i][mask].mean()) for k, mask in masks.items()},
            X_regions={k: float(x_fields[i][mask].mean()) for k, mask in masks.items()},
        ))
    first_passage = np.full(S["NE"], np.nan, dtype=np.float32)
    if onset_ms is not None:
        i0 = int(round(onset_ms / E01.DT))
        i1 = min(spikes.shape[0], i0 + int(round(1000.0 / E01.DT)))
        seg = spikes[i0:i1]
        any_spike = seg.any(axis=0)
        first_passage[any_spike] = np.argmax(seg[:, any_spike], axis=0) * E01.DT
    npz_path = out_json.replace(".json", "_traces_and_fields.npz")
    stride = max(1, int(round(10.0 / E01.DT)))
    _write_npz(
        npz_path, rate_dt_ms=np.asarray([10.0], np.float32),
        rate_E=rate_e[::stride].astype(np.float32), rate_I=rate_i[::stride].astype(np.float32),
        landmark_names=np.asarray(names),
        landmark_times_ms=np.asarray([selected[n]["snapshot_time_ms"] for n in names], np.float32),
        D_fields=d_fields.astype(np.float32), X_fields=x_fields.astype(np.float32),
        H_fields=h_fields.astype(np.float32), Y_fields=y_fields.astype(np.float32),
        first_passage_from_onset_ms=first_passage,
    )
    after = _meminfo()
    record = dict(
        status="COMPLETE", row_id=row["row_id"], connection_seed=1,
        noise_seed=row["noise_seed"], T_ms=total_ms, extended_to_cap=extend,
        point_id=H1_POINT_ID, Z="q75", X="unretuned_current_LC1",
        no_kick=True, no_reset=True, no_parameter_step=True,
        initial_32s_lifecycle=lifecycle1, lifecycle=lifecycle, verdict=verdict,
        numerical=numerical, refractory_ceiling_fraction=ceiling_frac,
        x_peak_depletion_ms=x_peak_ms, x_activates_after_onset=x_after,
        x_start=float(xtrace[0]) if xtrace.size else None,
        x_min=float(xtrace.min()) if xtrace.size else None,
        events=[dict(t_on_ms=float(e["t_on"]), t_off_ms=float(e["t_off"]),
                     dur_ms=float(e["dur_ms"]), peak_ext=float(e["peak_ext"]),
                     returned=bool(e.get("returned", False))) for e in events],
        field_landmarks=field_summaries, output_npz=npz_path,
        output_npz_sha256=_sha(npz_path), wall_s=time.time() - t0,
        resources=dict(start=before, end=after), source_lock_git_head=_load(LOCK)["git_head"],
        finished=_now(),
    )
    _write_json(out_json, record)
    _write_json(done_json, dict(status="DONE", row_id=row["row_id"],
                                output_sha256=_sha(out_json), finished=_now()))
    if os.path.exists(running):
        os.replace(running, running.replace(".RUNNING.json", ".RUNNING.superseded.json"))
    del first, final, res, spikes, S
    gc.collect()
    return record


def cmd_row(args):
    if not args.confirm_run:
        raise SystemExit("40k reconnaissance requires --confirm-run")
    manifest = _assert_manifest()
    rows = [row for row in manifest["rows"] if row["noise_seed"] == args.noise]
    if len(rows) != 1:
        raise SystemExit(f"unknown noise {args.noise}")
    with _stage_lock(f"recon_noise{args.noise}"):
        record = _run_once(rows[0])
    print(json.dumps(dict(noise=args.noise, verdict=record["verdict"],
                          lifecycle=record["lifecycle"]["label"], T_ms=record["T_ms"]), indent=2))


def cmd_all(args):
    if not args.confirm_run:
        raise SystemExit("40k reconnaissance requires --confirm-run")
    manifest = _assert_manifest()
    with _stage_lock("recon_all"):
        swap0 = _meminfo()["swap_used_mib"]
        rows = []
        for row in manifest["rows"]:
            while _meminfo()["swap_used_mib"] - swap0 >= 256.0:
                time.sleep(30.0)
            rows.append(_run_once(row))
        payload = dict(
            status="COMPLETE", n_rows=3,
            verdict_counts={v: sum(r["verdict"] == v for r in rows)
                            for v in sorted({r["verdict"] for r in rows})},
            lifecycle_counts={v: sum(r["lifecycle"]["label"] == v for r in rows)
                              for v in sorted({r["lifecycle"]["label"] for r in rows})},
            rows=[dict(noise_seed=r["noise_seed"], verdict=r["verdict"],
                       lifecycle=r["lifecycle"]["label"], T_ms=r["T_ms"],
                       output=r["row_id"]) for r in rows],
            claim_boundary="reconnaissance only; not a lifecycle candidate or parameter acceptance",
            completed=_now(),
        )
        _write_json(os.path.join(OUT, "aggregate.json"), payload)
    print(json.dumps(payload, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    sub.add_parser("manifest")
    row = sub.add_parser("row")
    row.add_argument("--noise", type=int, choices=NOISES, required=True)
    row.add_argument("--confirm-run", action="store_true")
    allp = sub.add_parser("all")
    allp.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "lock": cmd_lock(args)
    elif args.cmd == "manifest": cmd_manifest(args)
    elif args.cmd == "row": cmd_row(args)
    elif args.cmd == "all": cmd_all(args)


if __name__ == "__main__":
    main()
