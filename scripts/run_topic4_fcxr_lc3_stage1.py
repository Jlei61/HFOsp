#!/usr/bin/env python
"""Stage 1: does the tissue enter by itself over a window of parameters, or at one point?

Entry replicates across three noise seeds, but all three sit at the same working
point, so "the tissue enters by itself" is so far a statement about one cell of
parameter space.  The gate asks for a robust non-singleton window.

The relay is frozen at full availability throughout.  That is what isolates
entry: with the relay unable to fall, nothing can brake the discharge, so
whatever happens is attributable to wear accumulating under the local feedback
and nothing else.  Only two knobs move --

  tau_z          how fast wear accumulates, registered 5000 ms
  theta_h_lc2    the local feedback threshold, registered 1.5743 at this point

-- and the registered values sit at the centre of the grid, so the centre cell
must reproduce the trajectory already on disk or the grid is not measuring what
it claims to.

Each cell runs 15 s from t=0 with no kick, no reset and no parameter step during
the run.  15 s is 2.5x the slowest ignition observed at the registered point; a
cell that has not entered by then is recorded as not having entered *within
15 s*, and the adequacy of that window is carried in the record rather than
assumed, because on this stage a screen window shorter than the phenomenon has
already once been read as the phenomenon being absent.
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

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import choose_map_workers, install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_pathlabel import window_is_adequate  # noqa: E402
from src.topic4_fcxr_lc3_stage import lifecycle_stage  # noqa: E402
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "stage1_entry_window")
RUN_MS = 15000.0                 # 2.5x the slowest ignition at the registered point
SNAP_MS = 250.0
NOISE = 401                      # the primary seed, so the centre cell is comparable
TAU_Z_GRID = (2500.0, 5000.0, 10000.0)      # registered 5000 in the middle
THETA_SCALE = (0.90, 1.00, 1.10)            # registered theta scaled around 1.00
GIB_PER_SIM_SECOND = 0.596
BASE_RSS_GIB = 5.9
_CTX = {}


def _cell_id(tau_z, scale):
    return f"tauz{int(round(tau_z))}_theta{int(round(scale * 100))}"


def _context():
    if not _CTX:
        S = PP.build_substrate(1)
        install_registered_noise_rng(S)
        _CTX["S"] = S
    return _CTX["S"]


def _run_cell(spec):
    tau_z, scale = spec
    cell = _cell_id(tau_z, scale)
    out_json = os.path.join(OUT, f"cell_{cell}.json")
    if os.path.isfile(out_json):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior

    S = _context()
    point = GEO._point(GEO.H1_POINT_ID)
    cfg = E01._dynamic_cfg(point)
    theta_registered = float(cfg["theta_h_lc2"])
    cfg["tau_z"] = float(tau_z)
    cfg["theta_h_lc2"] = theta_registered * float(scale)
    # Frozen at full availability: the relay cannot fall, so it cannot brake.
    cfg["x_relay_frozen_E"] = np.ones(int(S["NE"]), dtype=float)
    snapshot_steps = {int(round(t / E01.DT)): f"t{int(t)}"
                      for t in np.arange(0.0, RUN_MS + SNAP_MS, SNAP_MS)}
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    S["net"]["rng"] = np.random.default_rng(NOISE)
    p = dataclasses.replace(S["p"], T=RUN_MS, dt=E01.DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], slow=slow, n_steps=int(round(RUN_MS / E01.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])

    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    wins, numerical, _ = LC1R._reduce_run_windows(
        res, run["checkpoint"].slow, S, E01.DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    events, af, af_dt, _floor, _ = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))
    bout = lifecycle.get("bout")
    onset_ms = None if bout is None else float(bout[0] * baseline["band"]["win_ms"])
    offset_ms = (None if bout is None
                 else float((bout[1] + 1) * baseline["band"]["win_ms"]))
    masks = GEO._region_masks(S)
    quiet_end = int(round(float(onset_ms if onset_ms is not None else RUN_MS) / E01.DT))
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=run["rate_E"], dt_ms=E01.DT,
        r_base_hz=float(np.median(run["rate_E"][:max(quiet_end, 1)])),
        table=snapshot_table(run["checkpoint"].slow.snapshots, E01.DT, masks),
        onset_ms=onset_ms, offset_ms=offset_ms, total_ms=RUN_MS)
    # The relay is frozen, so no arm here can reach offset; the stage label is
    # about entry only and says so.
    stage = lifecycle_stage(onset_ms=onset_ms, offset_ms=None,
                            n_returning_before_onset=ledger["n_returning_before_onset"])

    result = dict(
        status="COMPLETE", cell_id=cell, tau_z=float(tau_z), theta_scale=float(scale),
        theta_h_lc2=float(cfg["theta_h_lc2"]), theta_registered=theta_registered,
        is_registered_cell=bool(abs(tau_z - 5000.0) < 1e-9 and abs(scale - 1.0) < 1e-9),
        noise_seed=NOISE, point_id=GEO.H1_POINT_ID, run_ms=RUN_MS,
        relay="frozen at 1.0 throughout, so nothing can brake the discharge",
        no_kick=True, no_reset=True, no_parameter_step=True,
        onset_ms=onset_ms, lifecycle=lifecycle, numerical=numerical,
        stage=stage["stage"], stage_reason=stage["reason"],
        stage_scope="entry only; a frozen relay cannot terminate, so later stages "
                    "are unreachable by construction rather than unobserved",
        n_events=ledger["n_events"], n_returning_before_onset=ledger["n_returning_before_onset"],
        Q_af_to_onset=ledger["Q_af_to_onset"], Q_rate_to_onset=ledger["Q_rate_to_onset"],
        entry_class=ledger["entry_class"], event_ledger=ledger,
        max_rate_hz=float(np.max(run["rate_E"])),
        mean_rate_hz=float(np.mean(run["rate_E"])),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        source_lock_git_head=GEO._load_json(GEO.GEOMETRY_LOCK)["git_head"],
        finished=GEO._now())
    GEO._write_json(out_json, result)
    del run, res
    gc.collect()
    return result


def _workers(n_pending, swap0):
    mem = GEO._meminfo()
    single = BASE_RSS_GIB + GIB_PER_SIM_SECOND * (RUN_MS / 1000.0)
    return max(1, min(n_pending, choose_map_workers(
        mem_available_gib=mem["mem_available_gib"], swap_used_mib=mem["swap_used_mib"],
        swap_baseline_mib=swap0, single_rss_gib=single, cpu_count=os.cpu_count() or 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k stage-1 grid requires --confirm-run")
    os.makedirs(OUT, exist_ok=True)
    swap0 = GEO._meminfo()["swap_used_mib"]
    specs = [(t, s) for t in TAU_Z_GRID for s in THETA_SCALE]
    GEO._write_json(os.path.join(OUT, "RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), n_cells=len(specs),
                         run_ms=RUN_MS, tau_z_grid=list(TAU_Z_GRID),
                         theta_scale=list(THETA_SCALE), started=GEO._now()))

    rows, pending = [], list(specs)
    with ProcessPoolExecutor(max_workers=GEO.MAX_MAP_WORKERS) as pool:
        futures = {}
        while pending or futures:
            while pending and len(futures) < _workers(len(pending), swap0):
                spec = pending.pop(0)
                futures[pool.submit(_run_cell, spec)] = spec
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut)
                r = fut.result()
                rows.append(r)
                print(f"[stage1] {r['cell_id']}: {r['stage']} onset={r['onset_ms']} "
                      f"events_before={r['n_returning_before_onset']} "
                      f"class={r['entry_class']}", flush=True)

    entered = [r for r in rows if r["onset_ms"] is not None]
    cumulative = [r for r in entered if r["entry_class"] == "CUMULATIVE"]
    centre = next((r for r in rows if r["is_registered_cell"]), None)
    # A cell that did not enter is only informative if it was watched past the
    # ignition times entry actually takes here.
    adequacy = window_is_adequate(
        [dict(total_ms=RUN_MS)],
        reference_ms=max((r["onset_ms"] for r in entered), default=RUN_MS))
    payload = dict(
        status="COMPLETE", n_cells=len(rows), n_entered=len(entered),
        n_cumulative=len(cumulative),
        registered_cell=(None if centre is None else dict(
            cell_id=centre["cell_id"], onset_ms=centre["onset_ms"],
            n_returning_before_onset=centre["n_returning_before_onset"],
            entry_class=centre["entry_class"])),
        window_adequacy=adequacy,
        verdict=("ENTRY_WINDOW_NON_SINGLETON" if len(cumulative) > 1 else
                 "ENTRY_WINDOW_SINGLETON_OR_ABSENT"),
        claim_boundary=("entry leg only, one noise seed per cell, relay frozen at 1.0; "
                        "not a lifecycle claim and not a parameter acceptance"),
        rows=sorted(rows, key=lambda r: (r["tau_z"], r["theta_scale"])),
        completed=GEO._now())
    GEO._write_json(os.path.join(OUT, "stage1_entry_window.json"), payload)
    GEO._write_json(os.path.join(OUT, "DONE.json"),
                    dict(status="DONE", finished=GEO._now()))
    print(json.dumps({k: payload[k] for k in
                      ("n_cells", "n_entered", "n_cumulative", "registered_cell",
                       "verdict")}, indent=2))


if __name__ == "__main__":
    main()
