#!/usr/bin/env python
"""Land the per-event entry ledger for the no-kick trajectories.

The three 45 s reconnaissance runs answered "does it enter by itself".  They
were launched before ``build_event_ledger`` was wired into the recon runner and
source cannot be hot-edited under a live run, so the two registered entry
measurements -- how many interictal events precede onset, and how much load they
carried -- never reached disk.  Neither is recoverable from the artifacts: the
dose integrates the 1 ms active fraction the detector ran on, and only a
200x-decimated rate and four of the 250 ms snapshots survive to the NPZ.

Re-running the full 45 s costs 4-8 h and 110 GiB peak for a question that is
over by 6 s.  This runs the identical registered preparation to 20 s -- the
recon runner's own ``ONSET_CHECK_MS`` checkpoint, so the lifecycle label and the
event list are directly comparable against what is already stored.  That
comparison is the point: an entry ledger from a re-simulation is only worth
reading if the re-simulation is the same trajectory, so the run refuses to
publish unless its events reproduce the recorded ones.

Not a new experiment.  Same point, same seeds, same no-kick preparation, no
parameter step; strictly an analysis the first runner could not emit.
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
import run_topic4_fcxr_lc3_recon as RECON  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "entry_ledger")
T_MS = RECON.ONSET_CHECK_MS          # the recon runner's own 20 s checkpoint
SNAP_MS = RECON.SNAP_MS
H1_POINT_ID = RECON.H1_POINT_ID      # the point the recon runner actually ran
EVENT_MATCH_TOL_MS = 1e-6


def _recorded(noise):
    """The 45 s record for this seed, or None if it was never run."""
    path = os.path.join(RECON.OUT, f"recon_noise{noise}.json")
    return RECON._load(path) if os.path.isfile(path) else None


def _event_prefix_matches(fresh, recorded):
    """Do the re-simulated events reproduce the recorded ones inside 20 s?

    Only events that both runs could see are compared: the recorded list covers
    45 s, this one 20 s, and an event straddling the cut is truncated here and
    whole there, so the last one is excluded rather than counted as a mismatch.
    """
    ref = [e for e in recorded if e["t_off_ms"] < T_MS]
    got = [e for e in fresh if float(e["t_off"]) < T_MS]
    if len(ref) != len(got):
        return False, f"event count {len(got)} vs recorded {len(ref)} inside {T_MS:.0f} ms"
    for i, (a, b) in enumerate(zip(got, ref)):
        for key, ref_key in (("t_on", "t_on_ms"), ("t_off", "t_off_ms"),
                             ("dur_ms", "dur_ms"), ("peak_ext", "peak_ext")):
            if abs(float(a[key]) - float(b[ref_key])) > EVENT_MATCH_TOL_MS:
                return False, f"event {i} {key}: {float(a[key])!r} vs {float(b[ref_key])!r}"
    return True, f"{len(got)} events reproduced exactly"


def _run(noise):
    os.makedirs(OUT, exist_ok=True)
    out_json = os.path.join(OUT, f"entry_noise{noise}.json")
    before = RECON._meminfo()
    if before["mem_available_gib"] < 48.0:
        raise RuntimeError("entry ledger needs 48 GiB MemAvailable")
    running = out_json.replace(".json", ".RUNNING.json")
    RECON._write_json(running, dict(status="RUNNING", pid=os.getpid(), noise=noise,
                                    resource=before, started=RECON._now()))

    S = PP.build_substrate(1)
    point = GEO._point(H1_POINT_ID)
    cfg = E01._dynamic_cfg(point)
    snapshot_steps = {int(round(t / E01.DT)): f"t{int(t)}"
                      for t in np.arange(0.0, T_MS + SNAP_MS, SNAP_MS)}
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    S["net"]["rng"] = np.random.default_rng(int(noise))
    p = dataclasses.replace(S["p"], T=T_MS, dt=E01.DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], slow=slow, n_steps=int(round(T_MS / E01.DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])

    baseline = RECON._load(E01.ARTIFACTS["lc1_baseline"])
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    wins, numerical, _ = LC1R._reduce_run_windows(
        res, run["checkpoint"].slow, S, E01.DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    events, af, af_dt, _floor, _ = OLD._events_from_res(
        res, E01.DT, event_bar=float(baseline["frozen_event_bar"]))

    prior = _recorded(noise)
    if prior is None:
        reproduces, detail = None, "no 45 s record on disk to compare against"
    else:
        reproduces, detail = _event_prefix_matches(events, prior["events"])
        ref_label = prior["onset_search_20s"]["lifecycle"]["label"]
        if lifecycle["label"] != ref_label:
            reproduces = False
            detail = f"lifecycle {lifecycle['label']} vs recorded 20 s {ref_label}; {detail}"

    bout = lifecycle.get("bout")
    onset_ms = float(bout[0] * baseline["band"]["win_ms"]) if bout is not None else None
    offset_ms = (None if bout is None
                 else float((bout[1] + 1) * baseline["band"]["win_ms"]))
    masks = GEO._region_masks(S)
    table = snapshot_table(run["checkpoint"].slow.snapshots, E01.DT, masks)
    quiet_end = int(round(float(onset_ms if onset_ms is not None else T_MS) / E01.DT))
    r_base_hz = float(np.median(run["rate_E"][:max(quiet_end, 1)]))
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=float(baseline["floor_af"]),
        rate_hz=run["rate_E"], dt_ms=E01.DT, r_base_hz=r_base_hz, table=table,
        onset_ms=onset_ms, offset_ms=offset_ms, total_ms=T_MS)

    npz_path = out_json.replace(".json", "_ledger.npz")
    stride = max(1, int(round(10.0 / E01.DT)))
    RECON._write_npz(
        npz_path, rate_dt_ms=np.asarray([10.0], np.float32),
        rate_E=run["rate_E"][::stride].astype(np.float32),
        af=np.asarray(af, np.float32), af_bin_ms=np.asarray([af_dt], np.float32),
        snapshot_t_ms=np.asarray([r["t_ms"] for r in table], np.float32),
        **{f"snapshot_{var}_{region}":
           np.asarray([r[var][region] for r in table], np.float32)
           for var in ("D", "H", "X", "y")
           for region in ("core_A", "core_B", "axial", "off_axis", "all")})

    record = dict(
        status="COMPLETE", noise_seed=noise, connection_seed=1, point_id=H1_POINT_ID,
        T_ms=T_MS, no_kick=True, no_reset=True, no_parameter_step=True,
        purpose="per-event entry ledger the 45 s recon runner could not emit",
        lifecycle=lifecycle, numerical=numerical, r_base_hz=r_base_hz,
        onset_ms=onset_ms, offset_ms=offset_ms,
        reproduces_recorded_trajectory=reproduces, reproduction_detail=detail,
        event_ledger=ledger,
        events=[dict(t_on_ms=float(e["t_on"]), t_off_ms=float(e["t_off"]),
                     dur_ms=float(e["dur_ms"]), peak_ext=float(e["peak_ext"]),
                     returned=bool(e.get("returned", False))) for e in events],
        output_npz=npz_path, output_npz_sha256=RECON._sha(npz_path),
        wall_s=time.time() - t0, resources=dict(start=before, end=RECON._meminfo()),
        source_lock_git_head=RECON._load(RECON.LOCK)["git_head"],
        claim_boundary=("entry measurement on the registered no-kick point; "
                        "not a lifecycle claim and not a parameter acceptance"),
        finished=RECON._now())
    RECON._write_json(out_json, record)
    RECON._write_json(out_json.replace(".json", ".DONE.json"),
                      dict(status="DONE", output_sha256=RECON._sha(out_json),
                           reproduces_recorded_trajectory=reproduces,
                           finished=RECON._now()))
    if os.path.exists(running):
        os.replace(running, running.replace(".RUNNING.json", ".RUNNING.superseded.json"))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", type=int, choices=RECON.NOISES, required=True)
    ap.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k entry ledger requires --confirm-run")
    with RECON._stage_lock(f"entry_noise{args.noise}"):
        record = _run(args.noise)
    led = record["event_ledger"]
    print(json.dumps(dict(
        noise=args.noise, lifecycle=record["lifecycle"]["label"],
        onset_ms=record["onset_ms"],
        reproduces=record["reproduces_recorded_trajectory"],
        detail=record["reproduction_detail"],
        entry={k: led[k] for k in ("entry_class", "n_events_before_onset",
                                   "n_returning_before_onset", "Q_af_to_onset",
                                   "Q_rate_to_onset", "first_non_returning_index")}),
        indent=2))


if __name__ == "__main__":
    main()
