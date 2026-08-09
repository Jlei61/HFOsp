#!/usr/bin/env python
"""Does the selectivity come from the curve, and the recovery from the slow release?

Two claims, one 2x2, no arm answering both:

* **cooperative vs linear** -- whether the interictal state survives.  The two states' loads are
  separated by only about 2x at their extremes, so a linear actuator delivering a terminating dose
  hands the interictal side roughly a fourteenth of it, and a fourteenth is already several times
  what was measured to suppress interictal events.  The cooperative curve buys the difference by
  the shape of the distributions rather than by their gap: only 0.01% of interictal cells ever
  reach the top, so the population carries almost no activation while the discharge carries all
  of it.  If that reasoning is right, the linear arms damage the interictal side at a matched
  ictal dose and the cooperative arms do not.
* **slow vs fast release** -- whether the tissue gets back.  Terminating is not enough: the wear
  left by a discharge takes about thirteen seconds of near-silence to fall below the level that
  departs on its own, and every previous arm that stopped landed instead on a smoulder firing
  denser than the interictal train.  If that reasoning is right, the fast-release arms stop and
  smoulder and the slow-release arms stop and clear.

Each arm runs the whole lifecycle from rest, so one run reports all three legs -- whether it still
enters on its own, whether it stops, and whether events come back inside the reference band.  An
arm that never enters has failed the interictal leg, which is a result and not a wasted run.

**Claim boundary.** The high state on this substrate is a train that re-ignites from silence every
86 ms, which the project's own criterion excludes from counting as an ictal carrier.  Whatever
happens here is "stopped a re-ignition train and let the wear clear", never "terminated a seizure".
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
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_stage import (  # noqa: E402
    lifecycle_stage,
    reference_band,
    returned_to_reference,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "lc4_cooperative_2x2")
SEP = os.path.join(E01.OUT, "percell_separation", "separation_readjudicated.json")
# Entry lands near 5 s; the discharge and any termination inside 15 s; then thirty seconds of tail,
# which is more than twice the near-silence the wear needs to fall out of the range that departs.
RUN_MS = 45000.0
SNAP_MS = 250.0
NOISE = 401
TAU_M_MS = 1000.0        # the load's own clearance; the separation was measured at this value
TAU_ON_MS = 100.0        # a few times one interictal event (10-35 ms), so the transient is not followed
TAU_OFF_SLOW_MS = 10000.0
HILL_N = 4.0
DOSE_FRAC = 0.20         # of the recurrent excitatory current scale -- the top of the swept calibration
I_EE_SCALE = 272.75518960107513


def _hill_key(n):
    """The measurement wrote its cooperativity keys as plain integers.

    Stripping zeros off a formatted float looks equivalent and is not: n=10 becomes "1", which
    silently reads the wrong row and would place the half-activation against a curve nobody ran.
    """
    if float(n) != int(n):
        raise ValueError(f"the measured grid holds integer cooperativities only, got {n}")
    return str(int(n))


def _parameters():
    """Half-activation, strength and the linear arms' matched gain, from the measurement.

    Hard-coding these would let the mechanism drift away from the distributions it was placed
    against; deriving them here means a re-measurement moves the arms with it.
    """
    if not os.path.isfile(SEP):
        raise SystemExit(f"the placement needs the measured load distributions: {SEP} is missing")
    sep = GEO._load_json(SEP)
    row = sep["by_tau"][f"{TAU_M_MS:g}"]
    K = float(row["K_midgap"])
    a_ictal = float(row["hill"][_hill_key(HILL_N)]["ictal_mean"])
    g_m_max = DOSE_FRAC * I_EE_SCALE
    # Force-match: the linear arm must deliver the same current during the discharge, or a failure
    # on the interictal side would just mean it was given a bigger dose.
    ictal_current = g_m_max * a_ictal
    m_ictal = float(row["separation"]["ictal_median"])            # load at this tau, per cell
    return dict(K=K, n=HILL_N, g_m_max=g_m_max, a_ictal=a_ictal,
                ictal_current=ictal_current, m_ictal=m_ictal,
                eta_linear=ictal_current / m_ictal,
                eta_linear_slow=ictal_current / (m_ictal * 10.0),
                interictal_mean_activation=float(
                    row["hill"][_hill_key(HILL_N)]["interictal_mean"]))


def _arms(P):
    hill = dict(m_hill_K=P["K"], m_hill_n=P["n"], tau_a_on=TAU_ON_MS, g_m_max=P["g_m_max"])
    return (
        dict(arm="hill_slow", tau_adp=TAU_M_MS, **hill, tau_a_off=TAU_OFF_SLOW_MS),
        dict(arm="hill_fast", tau_adp=TAU_M_MS, **hill, tau_a_off=TAU_ON_MS),
        dict(arm="linear_fast", tau_adp=TAU_M_MS, eta_m=P["eta_linear"]),
        # The linear actuator has no release of its own -- its release IS the load's clearance, so
        # its slow arm is a ten-fold slower load, re-matched to the same discharge current.
        dict(arm="linear_slow", tau_adp=TAU_M_MS * 10.0, eta_m=P["eta_linear_slow"]),
    )


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
    win_ms = float(baseline["band"]["win_ms"])
    onset_ms = None if bout is None else float(bout[0] * win_ms)
    offset_ms = None if bout is None else float((bout[1] + 1) * win_ms)
    masks = GEO._region_masks(S)
    quiet_end = int(round(float(onset_ms if onset_ms is not None else RUN_MS) / E01.DT))
    table = snapshot_table(run["checkpoint"].slow.snapshots, E01.DT, masks)
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=run["rate_E"], dt_ms=E01.DT,
        r_base_hz=float(np.median(run["rate_E"][:max(quiet_end, 1)])),
        table=table, onset_ms=onset_ms, offset_ms=offset_ms, total_ms=RUN_MS)

    band = reference_band(baseline)
    after = [e for e in events
             if offset_ms is not None and e["t_on"] >= offset_ms and e.get("returned")]
    tail_s = (RUN_MS - offset_ms) / 1000.0 if offset_ms is not None else 0.0
    check = (returned_to_reference(
        n_returning_after_offset=len(after),
        event_rate_hz=(len(after) / tail_s if tail_s > 0 else 0.0), band=band,
        durations_ms=[float(e["dur_ms"]) for e in after],
        participation=[float(e["peak_ext"]) for e in after]) if offset_ms is not None else None)
    stage = lifecycle_stage(onset_ms=onset_ms, offset_ms=offset_ms,
                            n_returning_before_onset=ledger["n_returning_before_onset"],
                            return_check=check)

    slow_f = run["checkpoint"].slow
    ne = int(slow_f.NE)
    wear = np.asarray([r["D"]["all"] for r in table], float)
    t_ms = np.asarray([r["t_ms"] for r in table], float)
    # The one number that decides whether a return is even possible: a frozen wear field at 0.047
    # departs on its own within 7 s, and everything above it departs sooner.
    below = t_ms[wear < 0.047]
    npz_path = out_json.replace(".json", "_traces.npz")
    record = dict(
        status="COMPLETE", arm=spec["arm"], run_ms=RUN_MS, noise_seed=NOISE,
        point_id=GEO.H1_POINT_ID, config={k: v for k, v in spec.items() if k != "arm"},
        cooperative=bool(spec.get("m_hill_K") is not None),
        no_kick=True, no_reset=True, no_parameter_step=True,
        lifecycle=lifecycle, numerical=numerical, onset_ms=onset_ms, offset_ms=offset_ms,
        n_returning_before_onset=ledger["n_returning_before_onset"],
        Q_af_to_onset=ledger["Q_af_to_onset"], entry_class=ledger["entry_class"],
        tail_window_s=tail_s, n_returning_after_offset=len(after),
        tail_event_rate_hz=(len(after) / tail_s if tail_s > 0 else 0.0),
        return_check=check, stage=stage["stage"], stage_reason=stage["reason"],
        wear_end=float(np.mean(1.0 - np.asarray(slow_f.z[:ne], float))),
        wear_min=float(wear.min()) if wear.size else None,
        wear_first_below_departure_ms=(float(below[0]) if below.size else None),
        relay_end=float(np.mean(np.asarray(slow_f.x_relay[:ne], float))),
        a_end=(float(np.mean(np.asarray(slow_f.a[:ne], float)))
               if spec.get("m_hill_K") is not None else None),
        a_max=(float(max(slow_f.trace_a_max)) if slow_f.trace_a_max else None),
        adap_current_max=(float(max(slow_f.trace_adap_current))
                          if slow_f.trace_adap_current else None),
        max_rate=float(np.max(run["rate_E"])), mean_rate=float(np.mean(run["rate_E"])),
        event_ledger=ledger, output_npz=npz_path,
        claim_boundary=("one noise seed at one point.  The high state here is the 86 ms "
                        "re-ignition train, not a qualified ictal carrier, so a stop is a stop of "
                        "that train.  And an arm that never enters has failed the interictal leg "
                        "rather than passed the termination one."),
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
        print(f"[lc4] {spec['arm']}: traces not written ({exc}); the record stands", flush=True)
    del run, res
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--arms", default="")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k cooperative-actuator 2x2 requires --confirm-run")
    P = _parameters()
    arms = _arms(P)
    if args.arms:
        arms = tuple(a for a in arms if a["arm"] in set(args.arms.split(",")))
    per = GBA.BASE_RSS_GIB + GBA.GIB_PER_SIM_SECOND * (RUN_MS / 1000.0)
    mem = GEO._meminfo()["mem_available_gib"]
    if mem < args.workers * per + 40.0:
        raise SystemExit(f"{args.workers} workers need {args.workers * per + 40.0:.0f} GiB "
                         f"({per:.0f} each); {mem:.0f} available")
    os.makedirs(OUT, exist_ok=True)
    GEO._write_json(os.path.join(OUT, "RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), run_ms=RUN_MS,
                         parameters=P, arms=[a["arm"] for a in arms], started=GEO._now()))
    print(f"[lc4] half-activation {P['K']:.2f} at n={P['n']:g}; strength {P['g_m_max']:.1f} "
          f"(= {DOSE_FRAC:.0%} of the recurrent excitatory scale at full opening)\n"
          f"      predicted: interictal population activation {P['interictal_mean_activation']:.4f}, "
          f"discharge {P['a_ictal']:.3f} -> matched discharge current {P['ictal_current']:.1f}\n"
          f"[lc4] {len(arms)} arms, {args.workers} workers, {per:.0f} GiB each, "
          f"{mem:.0f} GiB available", flush=True)

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
                print(f"[lc4] {r['arm']:>12}: {r['stage']} — entered {r['onset_ms']}, "
                      f"stopped {r['offset_ms']}, {r['n_returning_after_offset']} back over "
                      f"{r['tail_window_s']:.0f} s; wear end {r['wear_end']:.4f} "
                      f"min {r['wear_min']}, below-departure at "
                      f"{r['wear_first_below_departure_ms']}", flush=True)

    GEO._write_json(os.path.join(OUT, "cooperative_2x2.json"),
                    dict(status="COMPLETE", run_ms=RUN_MS, parameters=P,
                         stages={r["arm"]: r["stage"] for r in rows}, rows=rows,
                         completed=GEO._now()))
    GEO._write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=GEO._now()))
    print(json.dumps({r["arm"]: dict(stage=r["stage"], onset=r["onset_ms"],
                                     offset=r["offset_ms"], wear_end=r["wear_end"],
                                     wear_min=r["wear_min"],
                                     tail_rate=r["tail_event_rate_hz"])
                      for r in rows}, indent=2))


if __name__ == "__main__":
    main()
