#!/usr/bin/env python3
"""FCXR-LC3 E3 local slow-vector probes at frozen-map landmarks."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import fcntl
import hashlib
import json
import resource
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
from src.topic4_fcxr_lc3 import replace_frozen_fields, run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    install_registered_noise_rng,
    load_prepared_checkpoint,
)
from src.topic4_fcxr_lc3_slowflow import select_slowflow_landmarks  # noqa: E402


OUT = os.path.join(E01.OUT, "slow_vector_field")
MANIFEST = os.path.join(OUT, "manifest.json")
PROBE_MS = 300.0
FIT_START_MS = 50.0


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


@contextmanager
def _lock(name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f".{name}.lock"), "a+") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"slow-flow stage already running: {name}") from exc
        yield


def _load(path):
    with open(path) as f:
        return json.load(f)


def _assert_geometry():
    GEO._assert_lock()
    path = os.path.join(E01.OUT, "geometry_map.json")
    if not os.path.isfile(path):
        raise SystemExit("complete geometry_map.json is required")
    geometry = _load(path)
    if geometry.get("status") != "COMPLETE" or len(geometry.get("rows", [])) != 102:
        raise SystemExit("geometry map is incomplete")
    return path, geometry


def cmd_manifest(_args):
    path, geometry = _assert_geometry()
    selected = select_slowflow_landmarks(geometry["rows"])
    rows = []
    for i, src in enumerate(selected):
        row_id = f"slow_{src['row_id']}"
        rows.append(dict(
            index=i, row_id=row_id, source_geometry_row=src["row_id"],
            d_label=src["d_label"], a_x=float(src["a_x"]),
            state_kind=src["state_kind"], prepared_state_hash=src["prepared_state_hash"],
            source_resolved_label=src["resolved_label"], probe_ms=PROBE_MS,
            fit_start_ms=FIT_START_MS,
            output_path=os.path.join(OUT, "cells", f"{row_id}.json"),
            done_path=os.path.join(OUT, "cells", f"{row_id}.DONE.json"),
            geometry_map_sha256=_sha(path),
        ))
    payload = dict(
        status="LOCKED", schema="fcxr-lc3-slowflow-manifest-1.0", rows=rows,
        geometry_map_sha256=_sha(path), geometry_lock_sha256=_sha(GEO.GEOMETRY_LOCK),
        selection_rule="boundary endpoints then locked 12-point fallback; maximum 20",
        created=_now(),
    )
    _write(MANIFEST, payload)
    print(json.dumps(dict(status="LOCKED", n_rows=len(rows)), indent=2))


def _assert_manifest():
    path, _geometry = _assert_geometry()
    if not os.path.isfile(MANIFEST):
        raise SystemExit("missing slow-flow manifest")
    m = _load(MANIFEST)
    if m.get("status") != "LOCKED" or not (12 <= len(m.get("rows", [])) <= 20):
        raise SystemExit("invalid slow-flow manifest")
    if m["geometry_map_sha256"] != _sha(path):
        raise SystemExit("geometry map drift after slow-flow lock")
    return m


def _slope(trace, start_i):
    y = np.asarray(trace, dtype=float)
    x = np.arange(y.size, dtype=float) * E01.DT
    use = np.arange(y.size) >= int(start_i)
    return 1000.0 * float(np.polyfit(x[use], y[use], 1)[0])


def _run(row):
    if os.path.isfile(row["output_path"]) and os.path.isfile(row["done_path"]):
        old = _load(row["output_path"])
        done = _load(row["done_path"])
        if (old.get("status") == "COMPLETE"
                and old.get("geometry_map_sha256") == row["geometry_map_sha256"]
                and done.get("output_sha256") == _sha(row["output_path"])):
            return old
    S = PP.build_substrate(1)
    # build_substrate omits the noise generator and this probe steps the network.
    install_registered_noise_rng(S["net"])
    fields, _records = GEO._primary_fields()
    prepared = GEO._prepared_records()[(GEO.H1_POINT_ID, row["state_kind"])]
    payload = load_prepared_checkpoint(
        prepared["checkpoint"]["path"],
        expected_file_sha256=prepared["checkpoint"]["file_sha256"])
    child = replace_frozen_fields(
        payload["state"], d_field=fields[row["d_label"]],
        x_field=np.full(S["NE"], float(row["a_x"])))
    if GEO.configured_state_hash(child) != row["prepared_state_hash"]:
        # The manifest binds the canonical state BEFORE its registered D/X replacement.
        if payload["configured_state_hash"] != row["prepared_state_hash"]:
            raise RuntimeError(f"{row['row_id']}: prepared state hash mismatch")
    slow = child.slow
    start_D = 1.0 - slow.z[:slow.NE].copy()
    start_X = slow.x_relay.copy()
    slow.cfg.use_z = True
    slow.cfg.z_frozen_E = None
    slow.cfg.x_relay_frozen_E = None
    slow._snap_steps = None
    slow.snapshots = {}
    p = dataclasses.replace(S["p"], T=PROBE_MS, dt=E01.DT)
    t0 = time.time()
    out = run_fcxr_loop(
        p, S["net"], start=child, n_steps=int(round(PROBE_MS / E01.DT)),
        capture_final=True, store_spikes=False, v_th_per_neuron=S["vth"])
    final = out["checkpoint"].slow
    dtrace = 1.0 - np.asarray(final.trace_z_mean[-out["n_steps"]:], dtype=float)
    xtrace = np.asarray(final.trace_x_relay_mean[-out["n_steps"]:], dtype=float)
    fit_i = int(round(FIT_START_MS / E01.DT))
    masks = GEO._region_masks(S)
    end_D = 1.0 - final.z[:final.NE]
    end_X = final.x_relay
    clip = np.asarray(final.trace_conductance_clip_frac[-out["n_steps"]:], dtype=float)
    tau = np.asarray(final.trace_tau_eff_ratio_min[-out["n_steps"]:], dtype=float)
    record = dict(
        status="COMPLETE", **row,
        dot_mean_D_per_s=_slope(dtrace, fit_i), dot_mean_a_X_per_s=_slope(xtrace, fit_i),
        mean_D_start=float(start_D.mean()), mean_D_end=float(end_D.mean()),
        mean_a_X_start=float(start_X.mean()), mean_a_X_end=float(end_X.mean()),
        regional_D_change={k: float((end_D - start_D)[m].mean()) for k, m in masks.items()},
        regional_X_change={k: float((end_X - start_X)[m].mean()) for k, m in masks.items()},
        finite=bool(np.all(np.isfinite(dtrace)) and np.all(np.isfinite(xtrace))),
        clip_frac_max=float(clip.max()) if clip.size else 0.0,
        tau_eff_min_ms=float(S["p"].tau_m_E * tau.min()) if tau.size else None,
        trace_dt_ms=5.0,
        D_trace=dtrace[::max(1, int(round(5.0 / E01.DT)))].tolist(),
        X_trace=xtrace[::max(1, int(round(5.0 / E01.DT)))].tolist(),
        wall_s=time.time() - t0,
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
        finished=_now(),
    )
    _write(row["output_path"], record)
    _write(row["done_path"], dict(
        status="DONE", row_id=row["row_id"],
        output_sha256=_sha(row["output_path"]), finished=_now()))
    return record


def cmd_all(args):
    if not args.confirm_run:
        raise SystemExit("40k slow-flow probes require --confirm-run")
    m = _assert_manifest()
    with _lock("slowflow_all"):
        rows = []
        for i, row in enumerate(m["rows"], 1):
            out = _run(row); rows.append(out)
            print(f"[slowflow] {i}/{len(m['rows'])} {row['row_id']} "
                  f"dD={out['dot_mean_D_per_s']:.4g}/s dX={out['dot_mean_a_X_per_s']:.4g}/s",
                  flush=True)
        safe = all(r["finite"] and r["clip_frac_max"] == 0.0
                   and r["tau_eff_min_ms"] >= 2.0 * E01.DT for r in rows)
        payload = dict(
            status="COMPLETE" if safe else "NUMERICAL_BLOCKED", n_rows=len(rows),
            geometry_map_sha256=m["geometry_map_sha256"],
            interpretation="local 50-300 ms drift only; not proof of a closed orbit",
            rows=rows, completed=_now(),
        )
        _write(os.path.join(OUT, "slow_vector_field.json"), payload)
        _write(os.path.join(OUT, "DONE.json"),
               dict(status=payload["status"], n_rows=len(rows), finished=_now()))
        if not safe:
            raise RuntimeError("slow-flow numerical safety failure")
    print(json.dumps(dict(status=payload["status"], n_rows=len(rows)), indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("manifest")
    allp = sub.add_parser("all")
    allp.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "manifest": cmd_manifest(args)
    elif args.cmd == "all": cmd_all(args)


if __name__ == "__main__":
    main()
