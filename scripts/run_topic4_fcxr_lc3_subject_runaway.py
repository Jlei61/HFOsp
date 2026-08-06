#!/usr/bin/env python
"""Put the lifecycle mechanism on the patient's own geometry and let it run away.

The substrate this whole line already runs on IS the patient's: two low-threshold cores sitting
at the centroids of the earliest contacts of that subject's two interictal propagation templates
(source SCL9/ICL9/ICL11, sink ICL1/ICL2/ICL3), with the fifteen real contacts registered onto the
same sheet by a plane fit.  What has not been done is to run the slow variables there at the
connection seed whose interictal events are **bidirectional** -- the accepted readout at that seed
gives 14 events, 6 forward and 8 reverse, direction purity 1.0 -- and ask whether that same tissue
carries itself into a seizure.

So each arm starts from t=0 with nothing held, nothing kicked and no parameter step, and records:
the interictal train and each event's direction along the source-sink axis, whether the tissue
enters on its own, and the per-contact readout the fifteen electrodes would see throughout.

Two seeds, because they answer different questions.  Seed 1 is the connectivity every result in
this line was measured on, so it is the bridge.  Seed 3 is the one whose interictal events were
shown to be bidirectional, so it is the one the figure needs.

The contact readout is **spike-weighted**, not the current-based local field the subject figure
uses: the slow-variable loop does not expose per-step currents, and a spike-weighted contact trace
is in any case closer to what the event detector reads.  Same distance weighting, different
quantity, and the metadata says so.
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
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "subject_runaway")
RUN_MS = 45000.0
SNAP_MS = 250.0
NOISE = 401
CONNECTION_SEEDS = (3, 1)      # 3 = the bidirectional interictal readout; 1 = this line's bridge
GIB_PER_SIM_SECOND = 0.596
BASE_RSS_GIB = 5.9
LANDMARK_MS = (None, None, 500.0)   # filled per arm: pre-entry, entry, entry+500 ms


def _contact_weights(posE, contacts, Rr, rx):
    """The montage's own distance weighting, applied to spikes instead of currents."""
    from lfp import _shape_f
    idx, wts = [], []
    for s in np.asarray(contacts, float):
        d = np.linalg.norm(posE - s, axis=1)
        m = d <= Rr
        if not np.any(m):
            j = int(np.argmin(d))
            idx.append(np.array([j])), wts.append(np.array([1.0]))
            continue
        w = _shape_f(np.maximum(d[m], 1e-4), rx)
        idx.append(np.where(m)[0]), wts.append(w / w.sum())
    return idx, wts


def _contact_trace(spikes, idx, wts, bin_steps):
    """Per-contact weighted spike rate, binned so the trace is readable at figure scale."""
    n_bins = spikes.shape[0] // bin_steps
    out = np.zeros((n_bins, len(idx)), dtype=np.float32)
    for b in range(n_bins):
        seg = spikes[b * bin_steps:(b + 1) * bin_steps]
        counts = seg.sum(axis=0)
        for k, (i, w) in enumerate(zip(idx, wts)):
            out[b, k] = float(np.dot(w, counts[i]))
    return out


def _event_direction(spikes, event, axis_unit, posE, dt, src_xy):
    """Signed direction of one event along the source-to-sink axis.

    Taken from the per-cell first-spike time inside the event against its position along the
    source-to-sink axis.  Cells near the source have the small axis coordinate, so a sweep that
    starts at the source gives EARLY times at SMALL coordinates -- a POSITIVE correlation.  The
    sign convention was inverted in the first version and a synthetic source-first sweep read as
    reverse, which would have flipped every direction in the figure.
    """
    i0, i1 = int(round(event["t_on"] / dt)), int(round(event["t_off"] / dt)) + 1
    seg = spikes[i0:min(i1, spikes.shape[0])]
    if seg.size == 0:
        return None, 0
    fired = seg.any(axis=0)
    if fired.sum() < 20:
        return None, int(fired.sum())
    first = np.argmax(seg[:, fired], axis=0) * dt
    proj = (posE[fired] - src_xy) @ axis_unit
    if np.std(proj) < 1e-9 or np.std(first) < 1e-9:
        return None, int(fired.sum())
    r = float(np.corrcoef(proj, first)[0, 1])
    return r, int(fired.sum())


def _run_seed(conn_seed):
    os.makedirs(OUT, exist_ok=True)
    out_json = os.path.join(OUT, f"seed{conn_seed}.json")
    if os.path.isfile(out_json):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior

    S = PP.build_substrate(conn_seed)
    reg = S["reg"]
    ms = reg["montage_sheet"]
    contacts = np.asarray(ms.contacts, float)
    names = [str(n) for n in ms.names]
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
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

    spikes = run["E_spk_bool"]
    posE = np.asarray(S["posE"], float)
    axis_unit = np.asarray(S["axis_unit"], float)
    src_xy = np.asarray(S["src_xy"], float)
    # direction of every returning event before the tissue enters
    dirs = []
    for e in events:
        if onset_ms is not None and e["t_off"] >= onset_ms:
            continue
        if not e.get("returned"):
            continue
        r, n_part = _event_direction(spikes, e, axis_unit, posE, E01.DT, src_xy)
        dirs.append(dict(t_on_ms=float(e["t_on"]), dur_ms=float(e["dur_ms"]),
                         peak_ext=float(e["peak_ext"]), axis_corr=r, n_participating=n_part,
                         direction=(None if r is None else ("forward" if r > 0 else "reverse"))))
    n_fwd = sum(1 for d in dirs if d["direction"] == "forward")
    n_rev = sum(1 for d in dirs if d["direction"] == "reverse")

    masks = GEO._region_masks(S)
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=run["rate_E"], dt_ms=E01.DT,
        r_base_hz=float(np.median(run["rate_E"][:max(int(round(
            (onset_ms if onset_ms is not None else RUN_MS) / E01.DT)), 1)])),
        table=snapshot_table(run["checkpoint"].slow.snapshots, E01.DT, masks),
        onset_ms=onset_ms, offset_ms=offset_ms, total_ms=RUN_MS)

    slow_f = run["checkpoint"].slow
    ne = int(slow_f.NE)
    record = dict(
        status="COMPLETE", subject=PP.SUBJECT, montage=PP.MONTAGE,
        connection_seed=conn_seed, noise_seed=NOISE, run_ms=RUN_MS,
        no_kick=True, no_reset=True, no_parameter_step=True,
        source_names=list(reg["source_names"]), sink_names=list(reg["sink_names"]),
        contact_names=names, n_contacts_offsheet=int(reg["n_contacts_offsheet"]),
        inter_core_mm_sheet=float(reg["inter_core_mm_sheet"]),
        lifecycle=lifecycle, numerical=numerical,
        onset_ms=onset_ms, offset_ms=offset_ms,
        ran_away=bool(onset_ms is not None),
        n_returning_before_onset=ledger["n_returning_before_onset"],
        entry_class=ledger["entry_class"], Q_af_to_onset=ledger["Q_af_to_onset"],
        interictal_directions=dict(n_forward=n_fwd, n_reverse=n_rev,
                                   n_undetermined=sum(1 for d in dirs if d["direction"] is None),
                                   bidirectional=bool(n_fwd > 0 and n_rev > 0), events=dirs),
        wear_end=float(np.mean(1.0 - np.asarray(slow_f.z[:ne], float))),
        relay_end=float(np.mean(np.asarray(slow_f.x_relay[:ne], float))),
        max_rate=float(np.max(run["rate_E"])), mean_rate=float(np.mean(run["rate_E"])),
        event_ledger=ledger,
        readout_kind=("spike-weighted per-contact rate using the montage's own distance "
                      "weighting; NOT the current-based local field the subject figure uses"),
        claim_boundary=("one noise seed per connectivity; a runaway here is a mechanism "
                        "demonstration on this subject's geometry, not a cohort claim"),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        finished=GEO._now())
    GEO._write_json(out_json, record)

    try:
        idx, wts = _contact_weights(posE, contacts, S["p"].Rr, S["p"].rx)
        bin_steps = max(1, int(round(2.0 / E01.DT)))          # 2 ms bins
        trace = _contact_trace(spikes, idx, wts, bin_steps)
        lm = [0.0 if onset_ms is None else max(0.0, onset_ms - 500.0),
              0.0 if onset_ms is None else onset_ms,
              (RUN_MS - SNAP_MS if onset_ms is None else min(onset_ms + 500.0, RUN_MS - SNAP_MS))]
        snaps = slow_f.snapshots
        def _snap_at(t_ms):
            key = min(snaps, key=lambda k: abs(snaps[k]["step"] * E01.DT - t_ms))
            return snaps[key], float(snaps[key]["step"] * E01.DT)
        fields, times = [], []
        for t in lm:
            s, tt = _snap_at(t)
            fields.append(1.0 - np.asarray(s["z_E"], float)); times.append(tt)
        first_passage = np.full(ne, np.nan, dtype=np.float32)
        if onset_ms is not None:
            i0 = int(round(onset_ms / E01.DT))
            seg = spikes[i0:min(spikes.shape[0], i0 + int(round(1000.0 / E01.DT)))]
            any_spk = seg.any(axis=0)
            first_passage[any_spk] = np.argmax(seg[:, any_spk], axis=0) * E01.DT
        GEO._write_json  # noqa: B018  (keep the import obvious)
        np.savez_compressed(
            out_json.replace(".json", "_traces.npz"),
            posE=posE.astype(np.float32), contacts=contacts.astype(np.float32),
            contact_names=np.asarray(names), vth=np.asarray(S["vth"], np.float32),
            src_xy=src_xy.astype(np.float32), snk_xy=np.asarray(S["snk_xy"], np.float32),
            axis_unit=axis_unit.astype(np.float32), L=np.asarray([S["L"]], np.float32),
            core_r=np.asarray([PP.CORE_R], np.float32),
            contact_trace=trace, contact_bin_ms=np.asarray([bin_steps * E01.DT], np.float32),
            rate_E=run["rate_E"][::max(1, int(round(10.0 / E01.DT)))].astype(np.float32),
            rate_dt_ms=np.asarray([10.0], np.float32),
            af=np.asarray(af, np.float32), af_bin_ms=np.asarray([af_dt], np.float32),
            wear_fields=np.asarray(fields, np.float32),
            wear_field_times_ms=np.asarray(times, np.float32),
            first_passage_from_onset_ms=first_passage)
    except Exception as exc:                                   # noqa: BLE001
        print(f"[subj] seed {conn_seed}: traces not written ({exc}); the record stands", flush=True)

    del run, res, spikes
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--seeds", default=",".join(str(s) for s in CONNECTION_SEEDS))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k subject runaway requires --confirm-run")
    per = BASE_RSS_GIB + GIB_PER_SIM_SECOND * (RUN_MS / 1000.0)
    mem = GEO._meminfo()["mem_available_gib"]
    if mem < args.workers * per + 40.0:
        raise SystemExit(f"{args.workers} workers need {args.workers*per+40.0:.0f} GiB "
                         f"({per:.0f} each); {mem:.0f} available")
    os.makedirs(OUT, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    GEO._write_json(os.path.join(OUT, "RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), seeds=seeds, run_ms=RUN_MS,
                         subject=PP.SUBJECT, started=GEO._now()))
    print(f"[subj] {PP.SUBJECT} {PP.MONTAGE}: seeds {seeds}, {args.workers} workers, "
          f"{per:.0f} GiB each, {mem:.0f} GiB available", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_seed, s): s for s in seeds}
        while futures:
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut)
                r = fut.result()
                rows.append(r)
                d = r["interictal_directions"]
                print(f"[subj] seed {r['connection_seed']}: ran_away={r['ran_away']} "
                      f"onset={r['onset_ms']}  interictal {d['n_forward']}fwd/{d['n_reverse']}rev"
                      f" bidirectional={d['bidirectional']}  wear_end={r['wear_end']:.4f}",
                      flush=True)

    GEO._write_json(os.path.join(OUT, "subject_runaway.json"),
                    dict(status="COMPLETE", subject=PP.SUBJECT, run_ms=RUN_MS,
                         rows=rows, completed=GEO._now()))
    GEO._write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=GEO._now()))
    print(json.dumps({str(r["connection_seed"]): dict(
        ran_away=r["ran_away"], onset_ms=r["onset_ms"],
        bidirectional=r["interictal_directions"]["bidirectional"],
        n_forward=r["interictal_directions"]["n_forward"],
        n_reverse=r["interictal_directions"]["n_reverse"]) for r in rows}, indent=2))


if __name__ == "__main__":
    main()
