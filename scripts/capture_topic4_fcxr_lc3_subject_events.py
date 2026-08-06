#!/usr/bin/env python
"""Capture per-cell onset maps for the interictal events that precede the runaway.

The middle two columns of the subject figure each show one representative propagation event as a
point cloud coloured by when each cell first fired.  The long run measures every pre-entry
event's direction but keeps only the summary, because holding a 32000-cell onset map for every
event across 45 s is not worth the memory.

Entry is over within the first few seconds, so this re-simulates just that opening window at the
same connection seed, the same noise seed and the same registered configuration, and keeps the
onset map of every returning event before entry.  Determinism makes the two runs the same
trajectory, and the capture refuses to publish unless the events it finds match the ones the long
run recorded -- same guard, same reason, as the entry ledger.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_subject_runaway as SUBJ  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_reproduction import events_reproduce  # noqa: E402

CAPTURE_MS = 8000.0      # entry lands within the first few seconds; this covers it with room
SNAP_MS = 250.0


def _run(conn_seed):
    out_json = os.path.join(SUBJ.OUT, f"seed{conn_seed}_events.json")
    long_json = os.path.join(SUBJ.OUT, f"seed{conn_seed}.json")
    if not os.path.isfile(long_json):
        raise SystemExit(f"the long run for seed {conn_seed} is not on disk yet")
    long_rec = GEO._load_json(long_json)

    S = PP.build_substrate(conn_seed)
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    snapshot_steps = {int(round(t / E01.DT)): f"t{int(t)}"
                      for t in np.arange(0.0, CAPTURE_MS + SNAP_MS, SNAP_MS)}
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    S["net"]["rng"] = np.random.default_rng(SUBJ.NOISE)
    p = dataclasses.replace(S["p"], T=CAPTURE_MS, dt=E01.DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], slow=slow, n_steps=int(round(CAPTURE_MS / E01.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])

    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    events, _af, _bw, _floor, _ = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))
    guard = events_reproduce(events, long_rec["event_ledger"]["events"], cut_ms=CAPTURE_MS)

    spikes = run["E_spk_bool"]
    posE = np.asarray(S["posE"], float)
    axis_unit = np.asarray(S["axis_unit"], float)
    src_xy = np.asarray(S["src_xy"], float)
    onset_ms = long_rec["onset_ms"]

    maps, meta = [], []
    for e in events:
        if not e.get("returned"):
            continue
        if onset_ms is not None and e["t_off"] >= onset_ms:
            continue
        i0 = int(round(e["t_on"] / E01.DT))
        i1 = min(spikes.shape[0], int(round(e["t_off"] / E01.DT)) + 1)
        seg = spikes[i0:i1]
        if seg.size == 0:
            continue
        fired = seg.any(axis=0)
        onset = np.full(S["NE"], np.nan, dtype=np.float32)
        onset[fired] = np.argmax(seg[:, fired], axis=0) * E01.DT
        r, n_part = SUBJ._event_direction(spikes, e, axis_unit, posE, E01.DT, src_xy)
        maps.append(onset)
        meta.append(dict(t_on_ms=float(e["t_on"]), t_off_ms=float(e["t_off"]),
                         dur_ms=float(e["dur_ms"]), peak_ext=float(e["peak_ext"]),
                         axis_corr=r, n_participating=int(n_part),
                         direction=(None if r is None else
                                    ("forward" if r > 0 else "reverse"))))

    record = dict(
        status="COMPLETE", connection_seed=conn_seed, noise_seed=SUBJ.NOISE,
        subject=PP.SUBJECT, capture_ms=CAPTURE_MS,
        reproduces_long_run=guard["reproduces"], reproduction_detail=guard["detail"],
        onset_ms=onset_ms, n_events_captured=len(meta), events=meta,
        wall_s=time.time() - t0, finished=GEO._now())
    GEO._write_json(out_json, record)
    if maps:
        np.savez_compressed(out_json.replace(".json", "_onsets.npz"),
                            onset_maps=np.asarray(maps, np.float32),
                            posE=posE.astype(np.float32),
                            t_on_ms=np.asarray([m["t_on_ms"] for m in meta], np.float32),
                            axis_corr=np.asarray([np.nan if m["axis_corr"] is None
                                                  else m["axis_corr"] for m in meta], np.float32))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--seeds", default="3,1")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k event capture requires --confirm-run")
    for s in [int(x) for x in args.seeds.split(",") if x.strip()]:
        r = _run(s)
        fwd = sum(1 for e in r["events"] if e["direction"] == "forward")
        rev = sum(1 for e in r["events"] if e["direction"] == "reverse")
        print(f"[capture] seed {s}: {r['n_events_captured']} pre-entry events "
              f"({fwd} forward, {rev} reverse); reproduces long run="
              f"{r['reproduces_long_run']} — {r['reproduction_detail']}", flush=True)
    print(json.dumps({"done": True}, indent=2))


if __name__ == "__main__":
    main()
