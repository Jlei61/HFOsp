#!/usr/bin/env python
"""Per-cell spike-triggered adaptation at the lifecycle working point.

The recruitment-gated brake this replaces is a single global scalar: one sensor reading
whole-field recruitment, one conductance handed to all 32000 E cells alike.  A cell that had
just fired twenty times got exactly what a cell that fired once got.  At up to 0.39 of leak it
moved every measured quantity by 8-14% and never terminated.

That outcome was already predicted by an accepted result.  The 2026-07-26 HEO-line acceptance
(`docs/archive/topic4/sef_hfo/mz_fcxr_heo_line_acceptance_2026-07-26.md` §1.3) found that
replacing each cell's own `m_i(t)` with the load-matched population mean collapsed the effect to
zero, and concluded: keep per-cell load/recovery state, do not substitute a single global scalar.
This run puts the per-cell variable back.

`m` is spike-triggered and per-cell: every E spike adds 1 to that cell's own m, which relaxes on
its own recovery time, and drives the same K-reversal conductance the brake used -- one actuator,
so the two are directly comparable at equal dose.

Strength comes from the measured rates, not from round numbers.  Equilibrium m is rate x tau by
construction; at tau_adp = 2000 ms the pre-entry train (4.84 Hz per cell) sits at m ~ 9.7 and the
discharge (70.5 Hz) at m ~ 141, a 15x separation.  In leak-relative conductance
(eta_m * m / (v_match - e_k), leak = 1):

    eta_m = 0.05  ->  0.39 during the discharge, 0.027 interictal   <- the strongest brake's dose
    eta_m = 0.15  ->  1.18                       0.081
    eta_m = 0.45  ->  3.53                       0.242

So selectivity is not something this needs a gate for: accumulation supplies it.  The 0.05 arm is
the contract test -- same dose as the global brake, delivered per cell instead of uniformly.  The
0.45 arm carries a non-negligible interictal load and its entry must be checked, not assumed.

What the arms decide is stated without a prediction attached: whether the pre-entry train still
lands inside the frozen reference distribution (entry intact), whether entry happens, whether the
discharge terminates on its own, and whether returning events come back into the band.
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
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3_geometry import install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_stage import (  # noqa: E402
    lifecycle_stage,
    reference_band,
    returned_to_reference,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402

OUT = os.path.join(E01.OUT, "percell_adaptation")
# Entry lands ~5 s in.  65 s of discharge was not enough to see what a slow variable does, and a
# window shorter than the phenomenon has already produced three retractions on this line.
RUN_MS = 100000.0
SNAP_MS = 250.0
NOISE = 401
# sAHP timescale; 23x the 86 ms inter-burst interval, so it integrates across bursts
# rather than tracking one.
TAU_ADP_MS = 2000.0
GRID = 16               # snapshot fields are binned to GRID x GRID for the spatial-mode panels
GIB_PER_SIM_SECOND = 0.596
BASE_RSS_GIB = 5.9

ARMS = (
    # m_sense accumulates but actuates nothing: byte-parity control, and the m field the
    # ladder's arithmetic assumes.  If its plateau is not near 141, the ladder is mis-scaled.
    dict(arm="m_sense", eta_m=0.0, mean_field=False),
    dict(arm="m_e005", eta_m=0.05, mean_field=False),        # ~0.39 of leak -- the brake's dose
    dict(arm="m_e015", eta_m=0.15, mean_field=False),        # ~1.2 of leak
    dict(arm="m_e045", eta_m=0.45, mean_field=False),        # ~3.5 of leak
    # registered control: per-cell m replaced by the population mean each step
    dict(arm="m_e015_meanfield", eta_m=0.15, mean_field=True),
)


def _write_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


_CTX = {}


def _context():
    if not _CTX:
        S = PP.build_substrate(1)
        install_registered_noise_rng(S)
        _CTX["S"] = S
    return _CTX["S"]


def _grid_fields(snapshots, posE, L, grid=GRID):
    """Bin each snapshot's per-cell m and z onto a fixed grid.

    The spatial panels need a field, not a region mean, and re-running a five-arm sweep to get one
    is not a trade worth making -- so the binning happens here while the per-cell arrays are alive.
    """
    ix = np.clip((np.asarray(posE)[:, 0] / L * grid).astype(int), 0, grid - 1)
    iy = np.clip((np.asarray(posE)[:, 1] / L * grid).astype(int), 0, grid - 1)
    flat = iy * grid + ix
    counts = np.bincount(flat, minlength=grid * grid).astype(float)
    counts[counts == 0] = np.nan
    labels = sorted(snapshots, key=lambda k: snapshots[k]["step"])
    out = {}
    for var, key in (("m", "m_E"), ("z", "z_E")):
        stack = []
        for lab in labels:
            v = np.asarray(snapshots[lab][key], float)
            stack.append((np.bincount(flat, weights=v, minlength=grid * grid) / counts)
                         .reshape(grid, grid))
        out[var] = np.asarray(stack, np.float32)
    steps = np.asarray([snapshots[lab]["step"] for lab in labels], np.float32)
    return out["m"], out["z"], steps


def _run_arm(spec):
    # A delayed arm is a different experiment from the same eta_m applied from t=0, so it gets its
    # own record rather than resuming into the undelayed one's file.
    enable = spec.get("m_enable_ms")
    name = spec["arm"] if enable is None else f"{spec['arm']}_on{int(enable / 1000)}s"
    out_json = os.path.join(OUT, f"arm_{name}.json")
    if os.path.isfile(out_json):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior

    S = _context()
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    cfg.update(use_m=True, tau_adp=TAU_ADP_MS, eta_m=float(spec["eta_m"]),
               m_mean_field=bool(spec["mean_field"]))
    if enable is not None:
        # HEO2's construction: let the high state establish first, then switch adaptation on.  It
        # also passes that line's baseline gate by construction -- adaptation is off through the
        # interictal train, so it cannot be what stops the train accumulating.
        cfg["m_enable_ms"] = float(enable)
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

    # A bout whose end is the end of the record has not terminated.  Saying so explicitly here is
    # the same fix the brake adjudication needed: without it every arm, including the control that
    # never stops, reports as "terminated and silenced".
    terminated = bool(offset_ms is not None and offset_ms < RUN_MS - 0.5 * win_ms)

    slow_f = run["checkpoint"].slow
    ne = int(slow_f.NE)
    npz_path = out_json.replace(".json", "_traces.npz")
    record = dict(
        status="COMPLETE", arm=name, eta_arm=spec["arm"], use_m=True,
        eta_m=float(spec["eta_m"]), m_enable_ms=enable,
        m_mean_field=bool(spec["mean_field"]), tau_adp_ms=TAU_ADP_MS,
        use_gba=False, noise_seed=NOISE, point_id=GEO.H1_POINT_ID, run_ms=RUN_MS,
        no_kick=True, no_reset=True, no_parameter_step=True,
        lifecycle=lifecycle, numerical=numerical,
        onset_ms=onset_ms, offset_ms=offset_ms, terminated=terminated,
        n_returning_before_onset=ledger["n_returning_before_onset"],
        Q_af_to_onset=ledger["Q_af_to_onset"], entry_class=ledger["entry_class"],
        tail_window_s=tail_s, n_returning_after_offset=len(after),
        tail_event_rate_hz=(len(after) / tail_s if tail_s > 0 else 0.0),
        return_check=check, stage=stage["stage"], stage_reason=stage["reason"],
        wear_end=float(np.mean(1.0 - np.asarray(slow_f.z[:ne], float))),
        relay_end=float(np.mean(np.asarray(slow_f.x_relay[:ne], float))),
        m_end_mean=float(np.mean(np.asarray(slow_f.m[:ne], float))),
        m_end_max=float(np.max(np.asarray(slow_f.m[:ne], float))),
        output_npz=npz_path,
        max_rate=float(np.max(run["rate_E"])), mean_rate=float(np.mean(run["rate_E"])),
        event_ledger=ledger,
        claim_boundary=("one noise seed at one point; a closed loop here is a mechanism "
                        "demonstration, not a parameter acceptance"),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        finished=GEO._now())
    GEO._write_json(out_json, record)
    # Traces last, and never fatally -- the record is what a lost diagnostic must not cost.
    try:
        table = snapshot_table(slow_f.snapshots, E01.DT, masks)
        stride = max(1, int(round(10.0 / E01.DT)))
        m_grid, z_grid, snap_steps = _grid_fields(
            slow_f.snapshots, S["posE"], float(S["L"]), GRID)
        _write_npz(
            npz_path,
            rate_dt_ms=np.asarray([10.0], np.float32),
            rate_E=run["rate_E"][::stride].astype(np.float32),
            af=np.asarray(af, np.float32), af_bin_ms=np.asarray([af_dt], np.float32),
            # the phase plane's two axes, at snapshot resolution
            snapshot_t_ms=np.asarray([r["t_ms"] for r in table], np.float32),
            snapshot_step=snap_steps,
            m_mean=np.asarray([float(np.mean(slow_f.snapshots[k]["m_E"]))
                               for k in sorted(slow_f.snapshots,
                                               key=lambda x: slow_f.snapshots[x]["step"])],
                              np.float32),
            # the spatial-mode panels' input
            m_grid=m_grid, z_grid=z_grid, grid_n=np.asarray([GRID], np.int32),
            L_mm=np.asarray([float(S["L"])], np.float32),
            **{f"snapshot_{v}_{rg}": np.asarray([r[v][rg] for r in table], np.float32)
               for v in ("D", "H", "X", "y")
               for rg in ("core_A", "core_B", "axial", "off_axis", "all")})
    except Exception as exc:                                   # noqa: BLE001
        print(f"[adapt] {spec['arm']}: traces not written ({exc}); the record stands", flush=True)
    del run, res
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--arms", default=",".join(a["arm"] for a in ARMS))
    ap.add_argument("--m-enable-ms", type=float, default=None,
                    help="switch adaptation on this late, so the discharge establishes first")
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k-neuron sweep requires --confirm-run")

    want = [dict(a, m_enable_ms=args.m_enable_ms)
            for a in ARMS if a["arm"] in set(args.arms.split(","))]
    per = BASE_RSS_GIB + GIB_PER_SIM_SECOND * RUN_MS / 1000.0
    avail = GEO._meminfo()["mem_available_gib"]
    # 40 GiB headroom, the same margin the sibling runners keep: this box has run out of memory
    # mid-sweep before, and a killed worker costs the whole arm.
    workers = args.workers or max(1, min(len(want), int((avail - 40.0) // per)))
    if workers * per + 40.0 > avail:
        raise SystemExit(f"{workers} workers need {workers * per + 40.0:.0f} GiB "
                         f"({per:.0f} each + 40 headroom); {avail:.0f} available")
    os.makedirs(OUT, exist_ok=True)
    GEO._write_json(os.path.join(OUT, "RUNNING.json"),
                    dict(status="RUNNING", arms=[a["arm"] for a in want], run_ms=RUN_MS,
                         tau_adp_ms=TAU_ADP_MS, started=GEO._now()))
    print(f"[adapt] {len(want)} arms, {workers} workers, {per:.0f} GiB each, "
          f"{avail:.0f} GiB available", flush=True)

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_arm, a): a["arm"] for a in want}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"[adapt] {r['arm']}: {r['stage']} — eta_m {r['eta_m']}, "
                  f"on at {r.get('m_enable_ms')}, "
                  f"onset {r['onset_ms']}, offset {r['offset_ms']}, "
                  f"terminated={r['terminated']}, m_end mean {r['m_end_mean']:.1f} "
                  f"max {r['m_end_max']:.1f}, wear {r['wear_end']:.4f}", flush=True)

    GEO._write_json(os.path.join(OUT, "percell_adaptation.json"),
                    dict(status="COMPLETE", run_ms=RUN_MS, tau_adp_ms=TAU_ADP_MS,
                         stages={r["arm"]: r["stage"] for r in rows},
                         terminated={r["arm"]: r["terminated"] for r in rows}, rows=rows))
    GEO._write_json(os.path.join(OUT, "DONE.json"),
                    dict(status="DONE", finished=GEO._now()))
    print(json.dumps({r["arm"]: dict(stage=r["stage"], terminated=r["terminated"],
                                     m_end_mean=r["m_end_mean"]) for r in rows}, indent=2))


if __name__ == "__main__":
    main()
