#!/usr/bin/env python
"""Given a discharge already running, does the brake stop it and does the wear then clear?

The 2x2 from rest answered one of its two questions and could not reach the other: all four arms
kept the tissue below the tipping point for the whole 45 s, so no arm ever had a discharge to
terminate.  What it did settle is that the cooperative actuator is 18-36x gentler on the interictal
train than the linear one at a matched discharge dose.

This asks the other half directly, by starting inside the discharge instead of waiting for one.
It gives up the entry question -- deliberately, since that one is already answered without the
mechanism and confounded with it.

Two readouts, and they are not the same question:

* **does it stop** -- the discharge counts as present while the recruited fraction, smoothed at
  100 ms, stays above the interictal spread.  That smoothing is the registered carrier definition
  (sustained + array-wide + high-frequency), not a convenience: at 1 ms this state reads 26%
  occupancy and at 100 ms it reads 99.9%, and the criterion has to name which one it means.
* **does the wear clear** -- stopping is not recovering.  Every previous arm that stopped landed on
  a smoulder whose firing held the wear up at 0.089, and a frozen field at 0.047 departs on its own
  within 7 s.  From the 0.436 this state carries, falling under that needs about eleven seconds of
  near-silence.  The number reported is when, if ever, the whole-array wear crosses it.

The brake starts from rest here, so it needs a few load time constants to engage -- that is a
handicap of the fork, not a property of the mechanism, and the engagement time is reported.
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
import run_topic4_fcxr_lc3_gba as GBA  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_phase_map as PM  # noqa: E402
import run_topic4_fcxr_lc4_cooperative_2x2 as L4  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_ledger import snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_statefork import load_into  # noqa: E402

OUT = os.path.join(E01.OUT, "lc4_from_discharge")
REF = os.path.join(PM.OUT, "ref_ictal.npz")     # a real trajectory 12 s in, wear 0.436
RUN_MS = 30000.0
SNAP_MS = 250.0
NOISE = 4102
SMOOTH_MS = 100.0        # the scale the carrier definition is stated at
DEPARTURE_D = 0.047      # the lowest frozen wear that still departs on its own, within 7 s
QUIET_TAIL_MS = 5000.0   # a stop has to still be a stop this long later


def _arms(P):
    """The release axis, plus the linear actuator at the matched discharge dose as its control.

    linear_slow is dropped: from rest it was the weakest arm on every readout, and the actuator
    contrast it would add here is already carried by linear_fast at the same discharge current.
    """
    hill = dict(m_hill_K=P["K"], m_hill_n=P["n"], tau_a_on=L4.TAU_ON_MS, g_m_max=P["g_m_max"])
    return (
        dict(arm="hill_slow", tau_adp=L4.TAU_M_MS, **hill, tau_a_off=L4.TAU_OFF_SLOW_MS),
        dict(arm="hill_fast", tau_adp=L4.TAU_M_MS, **hill, tau_a_off=L4.TAU_ON_MS),
        dict(arm="linear_fast", tau_adp=L4.TAU_M_MS, eta_m=P["eta_linear"]),
    )


def _rekey_snapshots(state, run_ms, snap_ms, dt_ms):
    """Re-key the snapshot schedule onto the forked state's own clock, and drop what the template left.

    A loaded state resumes the step counter of the trajectory it came from, but the schedule was
    built from zero, so on a fork it fires only where the two ranges happen to overlap -- here the
    first 18 s of a 30 s run -- and the table also still carries the template's t=0 capture, whose
    wear is zero because nothing had run yet.  That single row is enough to report the starting
    wear as 0, the minimum as 0, and the first crossing of the departure level as 0 ms.

    Returns the step the fork starts at, so times can be reported relative to it.
    """
    base = int(getattr(state.slow, "_step_i", 0))
    state.slow.snapshots.clear()
    state.slow._snap_steps = {base + int(round(t / dt_ms)): f"t{int(t)}"
                              for t in np.arange(0.0, run_ms + snap_ms, snap_ms)}
    return base


def _stopped(af, bin_ms, ceiling):
    """When the discharge stops, read at the scale the carrier definition is stated at."""
    w = max(1, int(round(SMOOTH_MS / bin_ms)))
    s = np.convolve(np.asarray(af, float), np.ones(w) / w, mode="same")
    high = s > ceiling
    if not high.any():
        return dict(present_at_start=False, stop_ms=None, stopped=False,
                    occupancy=float(high.mean()))
    last = int(np.flatnonzero(high)[-1]) * bin_ms
    tail = (len(af) * bin_ms) - last
    return dict(present_at_start=bool(high[:int(round(500 / bin_ms))].any()),
                stop_ms=float(last), stopped=bool(tail >= QUIET_TAIL_MS),
                quiet_tail_ms=float(tail), occupancy=float(high.mean()))


def _run_arm(spec):
    out_json = os.path.join(OUT, f"arm_{spec['arm']}.json")
    if os.path.isfile(out_json) and GEO._load_json(out_json).get("status") == "COMPLETE":
        return GEO._load_json(out_json)

    S = GBA._context()
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    cfg.update(use_m=True, **{k: v for k, v in spec.items() if k != "arm"})
    snapshot_steps = {int(round(t / E01.DT)): f"t{int(t)}"
                      for t in np.arange(0.0, RUN_MS + SNAP_MS, SNAP_MS)}
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    start = load_into(REF, PM._seed_template(S, slow))
    base_step = _rekey_snapshots(start, RUN_MS, SNAP_MS, E01.DT)
    wear_start = float(np.mean(1.0 - np.asarray(start.slow.z[:S["NE"]], float)))
    S["net"]["rng"] = np.random.default_rng(NOISE)
    p = dataclasses.replace(S["p"], T=RUN_MS, dt=E01.DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], start=start, n_steps=int(round(RUN_MS / E01.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])

    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    events, af, af_dt, _floor, _ = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))
    ceiling = GEO._load_json(os.path.join(PM.OUT, "reference.json"))["interictal_ceiling_af"]
    stop = _stopped(af, af_dt, float(ceiling))

    slow_f = run["checkpoint"].slow
    ne = int(slow_f.NE)
    table = snapshot_table(slow_f.snapshots, E01.DT, GEO._region_masks(S))
    # Relative to the fork, not to the trajectory the state came from.
    t_ms = np.asarray([r["t_ms"] for r in table], float) - base_step * E01.DT
    wear = np.asarray([r["D"]["all"] for r in table], float)
    below = t_ms[wear < DEPARTURE_D]
    a_tr = np.asarray(slow_f.trace_a_mean, float)
    engaged = (float(np.flatnonzero(a_tr > 0.5 * a_tr.max())[0] * E01.DT)
               if a_tr.size and a_tr.max() > 0 else None)
    npz_path = out_json.replace(".json", "_traces.npz")
    record = dict(
        status="COMPLETE", arm=spec["arm"], run_ms=RUN_MS, noise_seed=NOISE,
        forked_from=REF, forked_at_ms=12000.0, point_id=GEO.H1_POINT_ID,
        config={k: v for k, v in spec.items() if k != "arm"},
        cooperative=bool(spec.get("m_hill_K") is not None),
        smoothing_ms=SMOOTH_MS, interictal_ceiling_af=float(ceiling), **stop,
        # Read off the forked state, never inferred from the snapshot table -- that is where the
        # template's zero row got in and reported a discharge as starting from no wear at all.
        wear_start=wear_start,
        wear_end=float(np.mean(1.0 - np.asarray(slow_f.z[:ne], float))),
        wear_min=float(wear.min()) if wear.size else None,
        wear_first_below_departure_ms=(float(below[0]) if below.size else None),
        departure_threshold=DEPARTURE_D,
        relay_end=float(np.mean(np.asarray(slow_f.x_relay[:ne], float))),
        brake_engaged_ms=engaged,
        a_max=(float(max(slow_f.trace_a_max)) if slow_f.trace_a_max else None),
        adap_current_max=(float(max(slow_f.trace_adap_current))
                          if slow_f.trace_adap_current else None),
        n_events=len(events), max_rate=float(np.max(run["rate_E"])),
        mean_rate=float(np.mean(run["rate_E"])),
        claim_boundary=("one seed, one fork point.  Entry is not tested here and was not meant to "
                        "be; and the brake starts from rest at the fork, so the seconds before it "
                        "engages are an artefact of the protocol, not of the mechanism."),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        finished=GEO._now())
    GEO._write_json(out_json, record)
    try:
        stride = max(1, int(round(10.0 / E01.DT)))
        GBA._write_npz(
            npz_path,
            rate_dt_ms=np.asarray([10.0], np.float32),
            rate_E=run["rate_E"][::stride].astype(np.float32),
            af=np.asarray(af, np.float32), af_bin_ms=np.asarray([af_dt], np.float32),
            a_trace_dt_ms=np.asarray([10.0], np.float32),
            a_mean=np.asarray(slow_f.trace_a_mean[::stride], np.float32),
            a_max_trace=np.asarray(slow_f.trace_a_max[::stride], np.float32),
            adap_current=np.asarray(slow_f.trace_adap_current[::stride], np.float32),
            snapshot_t_ms=t_ms.astype(np.float32),
            **{f"snapshot_{v}_{rg}": np.asarray([r[v][rg] for r in table], np.float32)
               for v in ("D", "H", "X", "y")
               for rg in ("core_A", "core_B", "axial", "off_axis", "all")})
    except Exception as exc:                                   # noqa: BLE001
        print(f"[fork] {spec['arm']}: traces not written ({exc}); the record stands", flush=True)
    del run, res
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k fork-from-discharge arms require --confirm-run")
    if not os.path.isfile(REF):
        raise SystemExit(f"the fork needs a real discharge state: {REF} is missing")
    P = L4._parameters()
    arms = _arms(P)
    per = GBA.BASE_RSS_GIB + GBA.GIB_PER_SIM_SECOND * (RUN_MS / 1000.0)
    mem = GEO._meminfo()["mem_available_gib"]
    if mem < args.workers * per + 40.0:
        raise SystemExit(f"{args.workers} workers need {args.workers * per + 40.0:.0f} GiB "
                         f"({per:.0f} each); {mem:.0f} available")
    os.makedirs(OUT, exist_ok=True)
    GEO._write_json(os.path.join(OUT, "RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), run_ms=RUN_MS, parameters=P,
                         arms=[a["arm"] for a in arms],
                         dropped=["linear_slow — weakest on every readout from rest; the actuator "
                                  "contrast is carried by linear_fast at the same dose"],
                         started=GEO._now()))
    print(f"[fork] from {REF} (12 s into a real discharge, wear 0.436)\n"
          f"[fork] {len(arms)} arms, {args.workers} workers, {per:.0f} GiB each, "
          f"{mem:.0f} GiB available; linear_slow dropped", flush=True)

    rows, pending = [], list(arms)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        while pending or futures:
            while pending and len(futures) < args.workers:
                futures[pool.submit(_run_arm, pending.pop(0))] = True
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut)
                r = fut.result()
                rows.append(r)
                print(f"[fork] {r['arm']:>12}: stopped={r['stopped']} at {r['stop_ms']} ms "
                      f"(occupancy {r['occupancy']:.2f}); wear {r['wear_start']:.3f} -> "
                      f"{r['wear_end']:.4f} (min {r['wear_min']:.4f}), below {DEPARTURE_D} at "
                      f"{r['wear_first_below_departure_ms']}; brake engaged {r['brake_engaged_ms']} ms",
                      flush=True)

    GEO._write_json(os.path.join(OUT, "from_discharge.json"),
                    dict(status="COMPLETE", run_ms=RUN_MS, parameters=P, rows=rows,
                         completed=GEO._now()))
    GEO._write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=GEO._now()))
    print(json.dumps({r["arm"]: dict(stopped=r["stopped"], stop_ms=r["stop_ms"],
                                     wear_end=r["wear_end"], wear_min=r["wear_min"],
                                     cleared_at=r["wear_first_below_departure_ms"])
                      for r in rows}, indent=2))


if __name__ == "__main__":
    main()
