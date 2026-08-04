#!/usr/bin/env python
"""Does a relay depth that stops the discharge also return the tissue to its own rhythm?

Clamping the relay terminates. Whether the tissue then behaves interictally is a
separate question, and the frozen LC1 baseline holds the reference it has to be
compared against: 34 returning events, 8-22 ms long, participation 0.045-0.080, at
0.086-3.15 events per second.

So this runs the terminating clamps long enough to accumulate a comparable sample and
puts the events side by side with that reference. The first seconds are the collapse
out of the discharge, not interictal activity, so they are excluded and the exclusion
is recorded rather than assumed.

Measured 2026-08-04: with wear pinned, every terminating clamp - 0.380, 0.350, 0.300
and the 0.100 floor - produced zero returning events across 18 s. An earlier reading
that 0.350 "lands at 3.07 Hz, right at the canonical 2.81" came from averaging over a
5 s window that opens with the collapse out of a 72 Hz discharge; with the collapse
excluded the same clamp sits at 0.093 Hz. At this wear the tissue discharges above
0.395 and is silent below 0.380, with no interictal branch between.

``--free-wear`` asks the obvious follow-up. Wear relaxes back whenever the inhibitory
sensor falls below threshold, so a silenced tissue should shed it on its own; pinning
wear may be the very reason nothing returns.

Usage:
    python scripts/run_topic4_fcxr_lc3_returngate.py --confirm-run
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "src", "snn_engine"), os.path.join(ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_dxprobe import freeze_dynamic_state  # noqa: E402
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    EXTENDED_ROW_RSS_SCALE,
    H1_POINT_ID,
    choose_map_workers,
    install_registered_noise_rng,
    load_prepared_checkpoint,
)
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402

OUT = os.path.join(E01.OUT, "return_gate_probe")
SEED_STATE = os.path.join(
    E01.OUT, "dynamic_reconnaissance", "exact_landmarks", "noise401_step895001.pkl")
RUN_MS = 20000.0            # long enough to accumulate a sample comparable to the 34
SETTLE_MS = 2000.0          # the collapse out of the discharge is not interictal activity
GIB_PER_SIM_SECOND = 0.596
CLAMPS = (0.380, 0.350, 0.300, 0.100)

_SEED = None
_SUBSTRATE = None
_BASELINE = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _meminfo():
    import resource
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(mem_available_gib=d["MemAvailable"] / 1048576.0,
                swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0,
                self_peak_rss_gib=resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1048576.0)


def _context():
    global _SEED, _SUBSTRATE, _BASELINE
    if _SUBSTRATE is None:
        _SUBSTRATE = PP.build_substrate(1)
        install_registered_noise_rng(_SUBSTRATE["net"])
        from src.topic4_fcxr_lc3 import _constants
        _constants(_SUBSTRATE["p"], _SUBSTRATE["net"])
    if _SEED is None:
        _SEED = load_prepared_checkpoint(SEED_STATE)["state"]
    if _BASELINE is None:
        _BASELINE = json.load(open(E01.ARTIFACTS["lc1_baseline"]))
    return _SUBSTRATE, _SEED, _BASELINE


def _tag(v):
    return f"{float(v):.3f}".replace(".", "p")


def _quantile_overlap(sample, reference):
    """Fraction of the sample lying inside the reference's own observed range."""
    s, r = np.asarray(sample, float), np.asarray(reference, float)
    if s.size == 0 or r.size == 0:
        return None
    return float(np.mean((s >= r.min()) & (s <= r.max())))


def _run_arm(arm):
    free_wear = bool(arm.get("free_wear", False))
    suffix = "_freewear" if free_wear else ""
    path = os.path.join(OUT, f"clamp_{_tag(arm['x'])}{suffix}.json")
    if os.path.isfile(path):
        prior = json.load(open(path))
        if prior.get("status") == "COMPLETE":
            return prior
    S, seed, baseline = _context()
    child = freeze_dynamic_state(seed, x_field=float(arm["x"]), freeze_d=not free_wear)
    t0 = time.time()
    p = dataclasses.replace(S["p"], T=RUN_MS, dt=E01.DT)
    out = run_fcxr_loop(p, S["net"], start=child, n_steps=int(round(RUN_MS / E01.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    res = dict(rate_E=out["rate_E"], rate_I=out["rate_I"], E_spk_bool=out["E_spk_bool"])
    events, af, af_dt, _floor, _rate = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))

    masks = GEO._region_masks(S)
    table = snapshot_table(out["checkpoint"].slow.snapshots, E01.DT, masks)
    settled = [e for e in events if e["t_on"] >= SETTLE_MS]
    r_base = float(np.median(out["rate_E"][int(round(SETTLE_MS / E01.DT)):]))
    ledger = build_event_ledger(
        events=settled, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=out["rate_E"], dt_ms=E01.DT, r_base_hz=r_base, table=table,
        onset_ms=None, offset_ms=None, total_ms=RUN_MS,
        r_base_definition=f"median population rate after the {SETTLE_MS:.0f} ms settle window")

    zf = np.asarray(out["checkpoint"].slow.z[:int(out["checkpoint"].slow.NE)], float)
    d_end = float(np.mean(1.0 - zf))
    ret = [e for e in settled if e["returned"]]
    dur = [float(e["dur_ms"]) for e in ret]
    part = [float(e["peak_ext"]) for e in ret]
    window_s = (RUN_MS - SETTLE_MS) / 1000.0
    ref_dur = baseline["event_durations_ms"]
    ref_part = baseline["event_participation"]
    band = baseline["band"]
    rate_per_s = len(ret) / window_s
    record = dict(
        status="COMPLETE", x_clamp=float(arm["x"]), free_wear=free_wear,
        wear_start=0.6629, wear_end=d_end,
        run_ms=RUN_MS, settle_ms=SETTLE_MS,
        n_events=len(settled), n_returning=len(ret),
        event_rate_per_s=rate_per_s,
        mean_population_rate_hz=float(np.mean(out["rate_E"][int(round(SETTLE_MS / E01.DT)):])),
        duration_ms=dict(median=(float(np.median(dur)) if dur else None),
                         min=(min(dur) if dur else None), max=(max(dur) if dur else None)),
        participation=dict(median=(float(np.median(part)) if part else None),
                           min=(min(part) if part else None), max=(max(part) if part else None)),
        reference=dict(n=len(ref_dur),
                       duration_ms=[min(ref_dur), float(np.median(ref_dur)), max(ref_dur)],
                       participation=[min(ref_part), float(np.median(ref_part)), max(ref_part)],
                       event_rate_band=[band["event_rate_lo"], band["event_rate_hi"]]),
        inside_reference=dict(
            duration_fraction=_quantile_overlap(dur, ref_dur),
            participation_fraction=_quantile_overlap(part, ref_part),
            event_rate_in_band=bool(band["event_rate_lo"] <= rate_per_s <= band["event_rate_hi"]),
        ),
        event_ledger=ledger,
        wall_s=time.time() - t0, peak_rss_gib=_meminfo()["self_peak_rss_gib"],
        finished=_now(),
    )
    _write_json(path, record)
    del out, res
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--clamps", default="")
    ap.add_argument("--free-wear", action="store_true",
                    help="let wear relax instead of pinning it; a silenced tissue sheds "
                         "wear on its own, so pinning it builds the answer into the setup")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k return-gate probe requires --confirm-run")
    clamps = ([float(v) for v in args.clamps.split(",") if v.strip()]
              if args.clamps.strip() else list(CLAMPS))
    if not all(0.0 <= v <= 1.0 for v in clamps):
        raise SystemExit("clamps must lie in [0,1]")
    mem0 = _meminfo()
    if mem0["mem_available_gib"] < 96.0:
        raise SystemExit("return-gate probe requires 96 GiB MemAvailable")
    os.makedirs(OUT, exist_ok=True)
    arms = [dict(x=v, free_wear=bool(args.free_wear)) for v in clamps]
    _write_json(os.path.join(OUT, "RUNNING.json"),
                dict(status="RUNNING", pid=os.getpid(), n_arms=len(arms), started=_now()))

    smoke = _run_arm(arms[0])
    # One continuous leg, so the footprint is the base plus the whole record's spikes;
    # divided by the helper's own fixed factor so it reserves 1.35x that, not 2.7x.
    worst = float(smoke["peak_rss_gib"])
    swap0 = mem0["swap_used_mib"]
    workers = choose_map_workers(
        mem_available_gib=_meminfo()["mem_available_gib"], swap_used_mib=swap0,
        swap_baseline_mib=swap0, single_rss_gib=worst / EXTENDED_ROW_RSS_SCALE,
        cpu_count=os.cpu_count() or 2)
    print(f"[return] clamp {smoke['x_clamp']:.3f}: {smoke['n_returning']} returning events, "
          f"{smoke['event_rate_per_s']:.2f}/s, rss={worst:.1f} GiB -> {workers} workers",
          flush=True)

    rows = [smoke]
    with ProcessPoolExecutor(max_workers=max(workers, 1)) as pool:
        futures = {pool.submit(_run_arm, a): a["x"] for a in arms[1:]}
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                x = futures.pop(fut)
                rec = fut.result()
                rows.append(rec)
                ins = rec["inside_reference"]
                print(f"[return] clamp {x:.3f}: {rec['n_returning']} returning, "
                      f"{rec['event_rate_per_s']:.2f}/s (band {ins['event_rate_in_band']}), "
                      f"wear {rec['wear_start']:.3f}->{rec['wear_end']:.3f}, "
                      f"dur inside {ins['duration_fraction']}", flush=True)

    aggregate = dict(
        status="COMPLETE", schema="fcxr-lc3-return-gate-1.0",
        seed_state=SEED_STATE, run_ms=RUN_MS, settle_ms=SETTLE_MS,
        rows=sorted(rows, key=lambda r: -r["x_clamp"]),
        claim_boundary=("relay held at a fixed depth, wear pinned at one late-bout state "
                        "of one noise seed; this asks whether the returned activity looks "
                        "interictal, not whether a trajectory reaches it on its own"),
        completed=_now(),
    )
    _write_json(os.path.join(OUT, "return_gate.json"), aggregate)
    running = os.path.join(OUT, "RUNNING.json")
    if os.path.exists(running):
        os.replace(running, os.path.join(OUT, "RUNNING.superseded.json"))
    _write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=_now()))
    print(json.dumps([{k: r[k] for k in
                       ("x_clamp", "n_returning", "event_rate_per_s", "inside_reference")}
                      for r in aggregate["rows"]], indent=2))


if __name__ == "__main__":
    main()
