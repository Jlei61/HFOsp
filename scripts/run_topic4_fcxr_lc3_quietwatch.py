#!/usr/bin/env python
"""Watch the frozen quiet state for longer than ignition takes.

Every one of the 42 frozen quiet cells ran 1500 ms and none departed.  The
no-kick trajectories ignite at 4000-6000 ms.  So the map's quiet side reports
its own screen window, and the registered temporal-geometry label is
``DX_MAP_UNRESOLVED`` rather than "the quiet state never departs".

This resolves it.  Same prepared quiet state, same wear fields, same relay level
the trajectories actually sit at before onset (mean relay 0.99999979 at
pre-onset, so the a_X = 1.00 column is the matched one and the only matched one),
run to 12000 ms -- twice the slowest observed ignition.

Two things are deliberately different from the map's own cell runner.

**The whole window is scanned, not the tail.**  The map classifies the last
500 ms or 2000 ms, which answers "where did it settle".  The question here is
"did it ever leave", and a cell that ignites at 5 s and self-terminates by 8 s
settles quiet while having departed.  So the lifecycle detector the trajectories
are judged by runs over the full record and reports a bout anywhere in it.

**Wear and relay stay frozen.**  That is the experiment, not a limitation: the
trajectories ignite while wear is climbing, and holding it still asks whether
the climb was necessary or merely concurrent.  A departure here would mean the
map was under-screened; no departure means entry needs wear in motion.
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
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402

import numpy as np  # noqa: E402

import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
from src.topic4_fcxr_lc3 import replace_frozen_fields, run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    choose_map_workers,
    configured_state_hash,
    install_registered_noise_rng,
    load_prepared_checkpoint,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "quiet_watch")
WATCH_MS = 12000.0        # 2x the slowest observed no-kick ignition (6000 ms)
MATCHED_A_X = 1.00        # the relay the trajectories sit at before onset
GIB_PER_SIM_SECOND = 0.596
BASE_RSS_GIB = 5.9
_CTX = {}


def _rss_estimate():
    return BASE_RSS_GIB + GIB_PER_SIM_SECOND * (WATCH_MS / 1000.0)


def _context():
    if not _CTX:
        S, fields, prepared = GEO._worker_context()
        install_registered_noise_rng(S)
        _CTX.update(S=S, fields=fields, prepared=prepared)
    return _CTX["S"], _CTX["fields"], _CTX["prepared"]


def _prior_cell(row_id):
    path = os.path.join(GEO.CELL_DIR, f"{row_id}.json")
    return GEO._load_json(path) if os.path.isfile(path) else None


def _run_cell(d_label):
    # Only the relay is written with 'p'; the point id keeps its own dots.
    row_id = f"{GEO.H1_POINT_ID}_{d_label}_aX" + f"{MATCHED_A_X:.2f}".replace(".", "p")
    out_json = os.path.join(OUT, f"quiet_{d_label}.json")
    if os.path.isfile(out_json):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior

    S, fields, prepared = _context()
    prep = prepared[(GEO.H1_POINT_ID, "low")]
    loaded = load_prepared_checkpoint(
        prep["checkpoint"]["path"],
        expected_file_sha256=prep["checkpoint"]["file_sha256"])
    child = replace_frozen_fields(
        loaded["state"], d_field=fields[d_label],
        x_field=np.full(S["NE"], float(MATCHED_A_X)))
    point = GEO._point(GEO.H1_POINT_ID)
    p = dataclasses.replace(S["p"], T=WATCH_MS, dt=GEO.DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], start=child, n_steps=int(round(WATCH_MS / GEO.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])

    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    wins, numerical, _ = LC1R._reduce_run_windows(
        res, run["checkpoint"].slow, S, GEO.DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    bout = lifecycle.get("bout")
    # The tail reading is kept alongside so this row sits on the map's own scale,
    # but departure is decided by the whole-window bout, not by where it settled.
    tail = GEO._tail_observables(run, S, point, tail_ms=GEO.EXTENDED_TAIL_MS,
                                 analysis_start_ms=WATCH_MS - GEO.EXTENDED_TAIL_MS)
    prior = _prior_cell(f"{row_id}_low")
    stride = max(1, int(round(10.0 / GEO.DT)))
    result = dict(
        status="COMPLETE", d_label=d_label, a_x=MATCHED_A_X, state_kind="low",
        watch_ms=WATCH_MS, source_row_id=f"{row_id}_low",
        departed=bool(bout is not None),
        departure_ms=(None if bout is None else float(bout[0] * baseline["band"]["win_ms"])),
        lifecycle=lifecycle, numerical=numerical,
        tail_label=tail["label"], tail_scale="map EXTENDED_TAIL_MS on the final window",
        prior_screen=(None if prior is None else dict(
            total_ms=prior["total_ms"], resolved_label=prior["resolved_label"],
            max_rate_hz=prior["max_rate_hz"], mean_rate_hz=prior["mean_rate_hz"])),
        max_rate_hz=float(np.max(run["rate_E"])),
        mean_rate_hz=float(np.mean(run["rate_E"])),
        rate_trace_dt_ms=10.0,
        rate_trace=run["rate_E"][::stride].astype(float).tolist(),
        final_configured_state_hash=configured_state_hash(run["checkpoint"]),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        source_lock_git_head=GEO._load_json(GEO.GEOMETRY_LOCK)["git_head"],
        claim_boundary=("frozen wear and frozen relay; a departure would mean the "
                        "map was under-screened, no departure means entry needs "
                        "wear in motion"),
        finished=GEO._now())
    GEO._write_json(out_json, result)
    del run, res
    gc.collect()
    return result


def _workers(n_pending):
    mem = GEO._meminfo()
    return max(1, min(n_pending, choose_map_workers(
        mem_available_gib=mem["mem_available_gib"], swap_used_mib=mem["swap_used_mib"],
        swap_baseline_mib=_workers.swap0, single_rss_gib=_rss_estimate(),
        cpu_count=os.cpu_count() or 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--fields", default=",".join(GEO.PRIMARY_D_LABELS))
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k quiet watch requires --confirm-run")
    os.makedirs(OUT, exist_ok=True)
    _workers.swap0 = GEO._meminfo()["swap_used_mib"]
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    GEO._write_json(os.path.join(OUT, "RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), watch_ms=WATCH_MS,
                         fields=fields, started=GEO._now()))

    rows, pending = [], list(fields)
    with ProcessPoolExecutor(max_workers=GEO.MAX_MAP_WORKERS) as pool:
        futures = {}
        while pending or futures:
            while pending and len(futures) < _workers(len(pending)):
                f = pending.pop(0)
                futures[pool.submit(_run_cell, f)] = f
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                name = futures.pop(fut)
                r = fut.result()
                rows.append(r)
                print(f"[quiet] {name}: departed={r['departed']} "
                      f"at={r['departure_ms']} tail={r['tail_label']} "
                      f"max={r['max_rate_hz']:.1f} Hz "
                      f"(prior 1500 ms: {(r['prior_screen'] or {}).get('resolved_label')})",
                      flush=True)

    n_dep = sum(1 for r in rows if r["departed"])
    payload = dict(
        status="COMPLETE", watch_ms=WATCH_MS, a_x=MATCHED_A_X, n_cells=len(rows),
        n_departed=n_dep,
        verdict=("QUIET_DEPARTS_WHEN_WATCHED_LONGER" if n_dep else
                 "QUIET_HOLDS_WITH_WEAR_FROZEN"),
        interpretation=(
            "at least one frozen quiet cell departed once given past the observed "
            "ignition time, so the map's quiet side was under-screened"
            if n_dep else
            "no frozen quiet cell departed in twice the slowest observed ignition "
            "time, so the trajectories' entry is not a property of the quiet state "
            "at fixed wear; wear had to be in motion"),
        rows=sorted(rows, key=lambda r: r["d_label"]), completed=GEO._now())
    GEO._write_json(os.path.join(OUT, "quiet_watch.json"), payload)
    GEO._write_json(os.path.join(OUT, "DONE.json"),
                    dict(status="DONE", finished=GEO._now()))
    print(json.dumps({k: payload[k] for k in
                      ("status", "n_cells", "n_departed", "verdict", "interpretation")},
                     indent=2))


if __name__ == "__main__":
    main()
