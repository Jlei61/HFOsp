#!/usr/bin/env python
"""Does the quiet at the end of the free-wear arm hold, and is it interictal?

The 40 s free-wear arm at relay gate 64 is the first on this stage to reach the
interictal workpoint tier: it terminates, the relay refills from 0.394 to 0.928
and wear sheds from 0.663 to 0.108, all on their own.  Two things stop that from
being a closed lifecycle.

**Its evidence of coming back is two seconds long.**  The tier is read from the
last 2000 ms of the record, and the quiet-state watch shows that a wear field of
0.089 departs again after 2.0 s.  The arm ends at 0.108 -- beyond every level
that watch covered -- so two seconds of quiet cannot distinguish "settled" from
"about to leave".  This is the same window problem that has already produced
three retractions here, arriving at the most consequential place yet.

**The registered return test was never run on it.**  Returning is judged against
the frozen baseline's own event distribution -- count, duration, participation --
and that arm carries no event ledger, so its return is unmeasured rather than
passed.  A mean rate does not substitute: 17.5 Hz over 40 s is the discharge it
opened with.

So this re-runs the same arm to 70 s with the ledger, and adjudicates with
`returned_to_reference` and `lifecycle_stage` rather than a workpoint label.
Gate 60 runs as the contrast: it ended `ELEVATED_EVENT_TRAIN` with its local
feedback still climbing at +0.43/s and end-occupancy 0.99, so it should not come
back and the two should part company.
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

import numpy as np  # noqa: E402

import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_returngate as RG  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_dxprobe import freeze_dynamic_state, set_hill_placement  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_stage import (  # noqa: E402
    lifecycle_stage,
    reference_band,
    returned_to_reference,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "lifecycle_tail")
RUN_MS = 70000.0        # the 40 s arm plus 30 s of whatever it settled into
GATES = (64.0, 60.0)
GIB_PER_SIM_SECOND = 0.596
BASE_RSS_GIB = 5.9


def _run_gate(y_gate):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"tail_gate{int(y_gate)}.json")
    if os.path.isfile(path):
        prior = json.load(open(path))
        if prior.get("status") == "COMPLETE":
            return prior
    S, seed, baseline = RG._context()
    # Both slow variables travel with the state and keep evolving; only the
    # relay's response curve moves. This is the 40 s arm, run longer.
    child = freeze_dynamic_state(seed, freeze_x=False, freeze_d=False)
    set_hill_placement(child, y_gate=float(y_gate))

    t0 = time.time()
    p = dataclasses.replace(S["p"], T=RUN_MS, dt=E01.DT)
    out = run_fcxr_loop(p, S["net"], start=child, n_steps=int(round(RUN_MS / E01.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])

    res = dict(rate_E=out["rate_E"], rate_I=out["rate_I"], E_spk_bool=out["E_spk_bool"])
    wins, numerical, _ = LC1R._reduce_run_windows(
        res, out["checkpoint"].slow, S, E01.DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    events, af, af_dt, _floor, _ = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))

    # The arm is seeded mid-discharge, so onset is the start of the record and the
    # offset is whatever the detector finds -- not a settle window chosen by hand.
    bout = lifecycle.get("bout")
    win_ms = float(baseline["band"]["win_ms"])
    offset_ms = None if bout is None else float((bout[1] + 1) * win_ms)
    masks = GEO._region_masks(S)
    table = snapshot_table(out["checkpoint"].slow.snapshots, E01.DT, masks)
    after = [e for e in events if offset_ms is not None and e["t_on"] >= offset_ms]
    tail_start = int(round((offset_ms if offset_ms is not None else 0.0) / E01.DT))
    r_base = float(np.median(out["rate_E"][tail_start:])) if tail_start < out["rate_E"].size else 0.0
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=out["rate_E"], dt_ms=E01.DT, r_base_hz=r_base, table=table,
        onset_ms=0.0, offset_ms=offset_ms, total_ms=RUN_MS,
        r_base_definition="median population rate after the detected offset")

    ret = [e for e in after if e.get("returned")]
    tail_s = (RUN_MS - (offset_ms or 0.0)) / 1000.0
    band = reference_band(baseline)
    check = returned_to_reference(
        n_returning_after_offset=len(ret),
        event_rate_hz=(len(ret) / tail_s if tail_s > 0 else 0.0), band=band,
        durations_ms=[float(e["dur_ms"]) for e in ret],
        participation=[float(e["peak_ext"]) for e in ret])
    stage = lifecycle_stage(onset_ms=0.0, offset_ms=offset_ms,
                            n_returning_before_onset=None, return_check=check)

    slow = out["checkpoint"].slow
    ne = int(slow.NE)
    d_end = float(np.mean(1.0 - np.asarray(slow.z[:ne], float)))
    x_end = float(np.mean(np.asarray(slow.x_relay[:ne], float)))
    record = dict(
        status="COMPLETE", y_gate=float(y_gate), run_ms=RUN_MS,
        relay="free", wear="free", seed_state=RG.SEED_STATE,
        lifecycle=lifecycle, numerical=numerical,
        offset_ms=offset_ms,
        tail_window_s=tail_s,
        n_returning_after_offset=len(ret),
        tail_event_rate_hz=(len(ret) / tail_s if tail_s > 0 else 0.0),
        return_check=check, stage=stage["stage"], stage_reason=stage["reason"],
        wear_start=0.6629, wear_end=d_end, relay_end=x_end,
        reference=dict(n=band.get("n_reference_events"),
                       event_rate=[band["event_rate_lo"], band["event_rate_hi"]],
                       duration_ms=[band.get("dur_lo_ms"), band.get("dur_hi_ms")],
                       participation=[band.get("part_lo"), band.get("part_hi")]),
        event_ledger=ledger,
        claim_boundary=("one arm from one late-bout state; a closed lifecycle here "
                        "is a mechanism demonstration, not a parameter acceptance"),
        wall_s=time.time() - t0, peak_rss_gib=RG._meminfo()["self_peak_rss_gib"],
        finished=RG._now())
    RG._write_json(path, record)
    del out, res
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--gates", default=",".join(str(g) for g in GATES))
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k lifecycle tail requires --confirm-run")
    need = BASE_RSS_GIB + GIB_PER_SIM_SECOND * (RUN_MS / 1000.0)
    mem = RG._meminfo()["mem_available_gib"]
    if mem < need + 30.0:
        raise SystemExit(f"a {RUN_MS/1000:.0f} s arm peaks near {need:.0f} GiB; "
                         f"{mem:.0f} GiB available")
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for gate in [float(g) for g in args.gates.split(",") if g.strip()]:
        r = _run_gate(gate)
        rows.append(r)
        print(f"[tail] gate {gate:.0f}: {r['stage']} — offset {r['offset_ms']}, "
              f"{r['n_returning_after_offset']} returning events over "
              f"{r['tail_window_s']:.0f} s at {r['tail_event_rate_hz']:.3f}/s; "
              f"wear {r['wear_end']:.3f}, relay {r['relay_end']:.3f}", flush=True)
    RG._write_json(os.path.join(OUT, "lifecycle_tail.json"),
                   dict(status="COMPLETE", run_ms=RUN_MS, n_arms=len(rows),
                        stages={r["y_gate"]: r["stage"] for r in rows},
                        rows=rows, completed=RG._now()))
    RG._write_json(os.path.join(OUT, "DONE.json"),
                   dict(status="DONE", finished=RG._now()))
    print(json.dumps({str(r["y_gate"]): dict(
        stage=r["stage"], reason=r["stage_reason"],
        n_returning=r["n_returning_after_offset"],
        rate_hz=r["tail_event_rate_hz"]) for r in rows}, indent=2))


if __name__ == "__main__":
    main()
