#!/usr/bin/env python
"""Does a brake only seizure-scale recruitment can charge let the loop close?

The loop stalls at its last leg.  Left alone the tissue enters by itself, discharges, stops,
sheds most of its wear and refills its relay -- and then sits in a smouldering train at
3.55-4.71 events/s that it never leaves, with wear pinned at 0.089-0.092.  That is a fixed
point of the slow subsystem, not a slow decay: 69 s is fourteen wear time constants and the
wear does not fall further.  And a frozen field at that wear departs again after 2.0 s, so the
tissue cannot be quiet there; the firing holds the wear up and the wear drives the firing.

Nothing keyed to how OFTEN events arrive can break that, because the smoulder is DENSER than
the train that produced entry -- 212-282 ms between events against 255-372 ms for the final
gaps before onset.  A rate threshold fires before entry, not after it.

What separates them is how much tissue each event takes.  Pooled over the three no-kick
trajectories and both 70 s tail arms, the pre-entry train peaks at 0.095 of the array while
the smoulder's median is 0.178-0.281 and the discharge's is 0.390, and across gates from 0.12
to 0.25 the pre-entry train crosses zero times while the smoulder crosses 2-3.5 times a
second.  So the brake is gated on recruitment, and the interictal train never charges it.

**Why the whole trajectory and not the tail arms.**  The tail arms seed from a late-bout
checkpoint whose discharge collapses within a second, so a slow brake would never charge and
the experiment would measure its own setup.  These runs start from t=0 with nothing held: the
tissue enters on its own around 5 s, discharges for tens of seconds -- charging the brake --
and only then is asked whether it can come back.

Arms: the brake off (the trajectory already on disk, re-run for a like-for-like window),
sensor-only (the brake watches but cannot act, so its selectivity is measured before it is
trusted), and acting at two strengths.
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
from src.topic4_fcxr_lc3_geometry import install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_stage import (  # noqa: E402
    lifecycle_stage,
    reference_band,
    returned_to_reference,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "global_burst_adaptation")
RUN_MS = 90000.0        # entry (~5 s) + the discharge that charges the brake + time to come back
SNAP_MS = 250.0
NOISE = 401
GATE = 0.15             # recruited fraction; zero pre-entry crossings across 0.12-0.25
TAU_CHARGE_MS = 2000.0
TAU_RELEASE_MS = 30000.0   # must exceed tau_z=5000; wear needs 3.2-7.5 s of quiet to clear
GIB_PER_SIM_SECOND = 0.596
BASE_RSS_GIB = 5.9

ARMS = (
    dict(arm="brake_off", use_gba=False, eta_gba=0.0),
    dict(arm="sensor_only", use_gba=True, eta_gba=0.0),
    dict(arm="acting_weak", use_gba=True, eta_gba=2.0),
    dict(arm="acting_strong", use_gba=True, eta_gba=6.0),
)
_CTX = {}


def _context():
    if not _CTX:
        S = PP.build_substrate(1)
        install_registered_noise_rng(S)
        _CTX["S"] = S
    return _CTX["S"]


def _run_arm(spec):
    out_json = os.path.join(OUT, f"arm_{spec['arm']}.json")
    if os.path.isfile(out_json):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior

    S = _context()
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    if spec["use_gba"]:
        cfg.update(use_gba=True, gba_gate=GATE, tau_gba_charge=TAU_CHARGE_MS,
                   tau_gba_release=TAU_RELEASE_MS, eta_gba=float(spec["eta_gba"]))
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
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=run["rate_E"], dt_ms=E01.DT,
        r_base_hz=float(np.median(run["rate_E"][:max(quiet_end, 1)])),
        table=snapshot_table(run["checkpoint"].slow.snapshots, E01.DT, masks),
        onset_ms=onset_ms, offset_ms=offset_ms, total_ms=RUN_MS)

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
    record = dict(
        status="COMPLETE", arm=spec["arm"], use_gba=bool(spec["use_gba"]),
        eta_gba=float(spec["eta_gba"]), gba_gate=GATE,
        tau_gba_charge=TAU_CHARGE_MS, tau_gba_release=TAU_RELEASE_MS,
        noise_seed=NOISE, point_id=GEO.H1_POINT_ID, run_ms=RUN_MS,
        no_kick=True, no_reset=True, no_parameter_step=True,
        lifecycle=lifecycle, numerical=numerical,
        onset_ms=onset_ms, offset_ms=offset_ms,
        n_returning_before_onset=ledger["n_returning_before_onset"],
        Q_af_to_onset=ledger["Q_af_to_onset"], entry_class=ledger["entry_class"],
        tail_window_s=tail_s, n_returning_after_offset=len(after),
        tail_event_rate_hz=(len(after) / tail_s if tail_s > 0 else 0.0),
        return_check=check, stage=stage["stage"], stage_reason=stage["reason"],
        wear_end=float(np.mean(1.0 - np.asarray(slow_f.z[:ne], float))),
        relay_end=float(np.mean(np.asarray(slow_f.x_relay[:ne], float))),
        gba_a_end=(None if slow_f.gba_a is None else float(slow_f.gba_a)),
        max_rate=float(np.max(run["rate_E"])), mean_rate=float(np.mean(run["rate_E"])),
        event_ledger=ledger,
        claim_boundary=("one noise seed at one point; a closed loop here is a mechanism "
                        "demonstration, not a parameter acceptance"),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        finished=GEO._now())
    GEO._write_json(out_json, record)
    del run, res
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--arms", default=",".join(a["arm"] for a in ARMS))
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k global-burst adaptation requires --confirm-run")
    per = BASE_RSS_GIB + GIB_PER_SIM_SECOND * (RUN_MS / 1000.0)
    mem = GEO._meminfo()["mem_available_gib"]
    if mem < args.workers * per + 40.0:
        raise SystemExit(f"{args.workers} workers need {args.workers*per+40.0:.0f} GiB "
                         f"({per:.0f} each); {mem:.0f} available")
    os.makedirs(OUT, exist_ok=True)
    want = [a for a in ARMS if a["arm"] in set(args.arms.split(","))]
    GEO._write_json(os.path.join(OUT, "RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), run_ms=RUN_MS,
                         arms=[a["arm"] for a in want], started=GEO._now()))
    print(f"[gba] {len(want)} arms, {args.workers} workers, {per:.0f} GiB each, "
          f"{mem:.0f} GiB available", flush=True)

    rows, pending = [], list(want)
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
                print(f"[gba] {r['arm']}: {r['stage']} — onset {r['onset_ms']}, "
                      f"offset {r['offset_ms']}, {r['n_returning_after_offset']} back over "
                      f"{r['tail_window_s']:.0f} s at {r['tail_event_rate_hz']:.3f}/s; "
                      f"wear {r['wear_end']:.4f}, relay {r['relay_end']:.3f}, "
                      f"brake {r['gba_a_end']}", flush=True)

    GEO._write_json(os.path.join(OUT, "global_burst_adaptation.json"),
                    dict(status="COMPLETE", run_ms=RUN_MS, gate=GATE,
                         tau_charge=TAU_CHARGE_MS, tau_release=TAU_RELEASE_MS,
                         stages={r["arm"]: r["stage"] for r in rows},
                         rows=rows, completed=GEO._now()))
    GEO._write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=GEO._now()))
    print(json.dumps({r["arm"]: dict(stage=r["stage"], onset=r["onset_ms"],
                                     offset=r["offset_ms"],
                                     tail_rate=r["tail_event_rate_hz"],
                                     wear_end=r["wear_end"], brake=r["gba_a_end"])
                      for r in rows}, indent=2))


if __name__ == "__main__":
    main()
