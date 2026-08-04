#!/usr/bin/env python
"""Is the seizure sustained by D leaving the mapped region, or by X failing to brake?

The first no-kick trajectory ran an ictal bout from 5 s to the end of its 45 s record
without terminating.  At 44.75 s the wear field sat at mean D = 0.663 -- 6.8x beyond
the largest level the 102-row frozen map ever covered -- while every single cell was
already below that map's a_X = 0.65 return boundary.  The map cannot arbitrate:
its D axis does not reach where the trajectory went.

A 2x2 does arbitrate.  Seed every arm from the same byte-parity-verified late-bout
state, freeze both slow fields, and cross them:

                      X = observed (mean 0.394)   X = 0.10 (the x_min floor)
    D = 0.663 observed          control                   max_brake
    D = 0.097 map maximum        map_D                 map_D_max_brake

- control must persist, otherwise the frozen replica does not reproduce the run;
- max_brake persisting means X cannot terminate at this D even at full depth;
- map_D terminating means the same X would have worked at mapped wear levels;
- the fourth cell separates a main effect from an interaction.

Each arm uses the map's own protocol: a 1.5 s screen with a 500 ms tail, extended to
5 s with a 2 s tail when the label leaves the high basin, adjudicated by the same
classifier, so the outcomes are directly comparable to the 102 map rows.

Usage:
    python scripts/run_topic4_fcxr_lc3_dxprobe.py --confirm-run
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for path in (ROOT, os.path.join(ROOT, "src", "snn_engine"), os.path.join(ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_dxprobe import DXPROBE_SCHEMA, freeze_dynamic_state, probe_summary  # noqa: E402
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    EXTENDED_MS,
    EXTENDED_TAIL_MS,
    H1_POINT_ID,
    SCREEN_MS,
    SCREEN_TAIL_MS,
    extension_required,
    install_registered_noise_rng,
    load_prepared_checkpoint,
)

OUT = os.path.join(E01.OUT, "dx_arbitration_probe")
SEED_STATE = os.path.join(
    E01.OUT, "dynamic_reconnaissance", "exact_landmarks", "noise401_step895001.pkl")
X_FLOOR = 0.10          # x_min in the registered dynamic config; X can never go below it
MAP_MAX_D_LABEL = "Dmax"  # the largest wear level the 102-row frozen map covered


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
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(mem_available_gib=d["MemAvailable"] / 1048576.0,
                swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0)


def _x_tag(value):
    return f"{float(value):.3f}".replace(".", "p")


def _ladder_arms(ne, x_values):
    """Bracket the X depth that terminates at the observed wear, one uniform level each.

    The 2x2 shows the trajectory's own X (mean 0.394) sustains the bout while the
    0.10 floor kills it, so the termination threshold lies between them.  Where it
    lies is what says whether the gap X has to close is small or large.
    """
    return [dict(arm_id=f"ladder_X{_x_tag(v)}", d=None, x=np.full(ne, float(v)),
                 d_desc="observed late-bout field",
                 x_desc=f"uniform {float(v):.3f}")
            for v in x_values]


def _arms(seed_slow, map_max_d):
    """The 2x2. ``None`` means 'freeze where the trajectory left it'."""
    ne = int(seed_slow.NE)
    observed_d = 1.0 - np.asarray(seed_slow.z[:ne], dtype=float)
    observed_x = np.asarray(seed_slow.x_relay, dtype=float)
    return [
        dict(arm_id="control", d=None, x=None,
             d_desc="observed late-bout field", x_desc="observed late-bout field"),
        dict(arm_id="max_brake", d=None, x=np.full(ne, X_FLOOR),
             d_desc="observed late-bout field", x_desc=f"uniform {X_FLOOR} (x_min floor)"),
        dict(arm_id="map_D", d=map_max_d, x=None,
             d_desc=f"frozen map {MAP_MAX_D_LABEL} field", x_desc="observed late-bout field"),
        dict(arm_id="map_D_max_brake", d=map_max_d, x=np.full(ne, X_FLOOR),
             d_desc=f"frozen map {MAP_MAX_D_LABEL} field", x_desc=f"uniform {X_FLOOR} (x_min floor)"),
    ], observed_d, observed_x


def _run_arm(S, point, seed_state, arm):
    child = freeze_dynamic_state(seed_state, d_field=arm["d"], x_field=arm["x"])
    ne = int(child.slow.NE)
    d_used = 1.0 - np.asarray(child.slow.z[:ne], dtype=float)
    x_used = np.asarray(child.slow.x_relay, dtype=float)

    p1 = dataclasses.replace(S["p"], T=SCREEN_MS, dt=E01.DT)
    t0 = time.time()
    first = run_fcxr_loop(p1, S["net"], start=child, n_steps=int(round(SCREEN_MS / E01.DT)),
                          capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    cls1 = GEO._tail_observables(first, S, point, tail_ms=SCREEN_TAIL_MS, analysis_start_ms=0.0)
    extended = extension_required(state_kind="high", label=cls1["label"])
    final, total_ms, cls2 = first, SCREEN_MS, None
    if extended:
        extra = EXTENDED_MS - SCREEN_MS
        p2 = dataclasses.replace(S["p"], T=extra, dt=E01.DT)
        second = run_fcxr_loop(p2, S["net"], start=first["checkpoint"],
                               n_steps=int(round(extra / E01.DT)), capture_final=True,
                               store_spikes=True, v_th_per_neuron=S["vth"])
        combined_rate = np.concatenate([first["rate_E"], second["rate_E"]])
        combined_spikes = np.concatenate([first["E_spk_bool"], second["E_spk_bool"]], axis=0)
        final = dict(second)
        final["rate_E"] = combined_rate
        final["E_spk_bool"] = combined_spikes
        final["n_steps"] = combined_rate.size
        cls2 = GEO._tail_observables(final, S, point, tail_ms=EXTENDED_TAIL_MS,
                                     analysis_start_ms=EXTENDED_MS - EXTENDED_TAIL_MS)
        total_ms = EXTENDED_MS
    resolved = cls2 or cls1
    record = probe_summary(arm_id=arm["arm_id"], d_field=d_used, x_field=x_used,
                           classification=resolved, total_ms=total_ms, extended=extended)
    record.update(
        d_source=arm["d_desc"], x_source=arm["x_desc"],
        initial_screen=cls1, extended_classification=cls2,
        mean_rate_hz=float(np.mean(final["rate_E"])),
        max_rate_hz=float(np.max(final["rate_E"])),
        wall_s=time.time() - t0, resource=_meminfo(), finished=_now(),
    )
    del first, final, child
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--x-ladder", default="",
                    help="comma-separated uniform X levels to bracket the termination "
                         "threshold at the observed wear, e.g. 0.35,0.30,0.25,0.20,0.15")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k D/X arbitration probe requires --confirm-run")
    if not os.path.isfile(SEED_STATE):
        raise SystemExit(f"missing byte-verified late-bout seed state: {SEED_STATE}")
    before = _meminfo()
    if before["mem_available_gib"] < 96.0:
        raise SystemExit("probe requires 96 GiB MemAvailable")

    seed_state = load_prepared_checkpoint(SEED_STATE)["state"]
    fields, _records = GEO._primary_fields()
    map_max_d = np.asarray(fields[MAP_MAX_D_LABEL], dtype=float)
    arms, observed_d, observed_x = _arms(seed_state.slow, map_max_d)
    if args.x_ladder.strip():
        levels = [float(v) for v in args.x_ladder.split(",") if v.strip()]
        if not all(0.0 <= v <= 1.0 for v in levels):
            raise SystemExit("--x-ladder levels must lie in [0,1]")
        arms = _ladder_arms(int(seed_state.slow.NE), levels)

    S = PP.build_substrate(1)
    install_registered_noise_rng(S["net"])
    point = GEO._point(H1_POINT_ID)
    os.makedirs(OUT, exist_ok=True)
    _write_json(os.path.join(OUT, "RUNNING.json"),
                dict(status="RUNNING", pid=os.getpid(), n_arms=len(arms), started=_now()))

    rows = []
    for arm in arms:
        path = os.path.join(OUT, f"arm_{arm['arm_id']}.json")
        if os.path.isfile(path):
            prior = json.load(open(path))
            if prior.get("resolved_label"):
                print(f"[dxprobe] resume {arm['arm_id']} -> {prior['resolved_label']}", flush=True)
                rows.append(prior)
                continue
        rec = _run_arm(S, point, seed_state, arm)
        _write_json(path, rec)
        rows.append(rec)
        print(f"[dxprobe] {arm['arm_id']:16s} D={rec['D_mean']:.4f} X={rec['X_mean']:.4f} "
              f"-> {rec['resolved_label']}  ({rec['wall_s']:.0f}s)", flush=True)

    by = {r["arm_id"]: r["resolved_label"] for r in rows}
    high = {"FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT"}
    aggregate = dict(
        schema=DXPROBE_SCHEMA, status="COMPLETE",
        seed_state=SEED_STATE, seed_state_time_ms=float(int(seed_state.t) * E01.DT),
        observed_D_mean=float(observed_d.mean()), observed_X_mean=float(observed_x.mean()),
        map_max_D_mean=float(map_max_d.mean()), x_floor=X_FLOOR,
        labels=by,
        claim_boundary=("frozen arbitration seeded from one real late-bout state of one "
                        "noise seed; not a lifecycle result and not a parameter acceptance"),
        rows=rows, completed=_now(),
    )
    if {"control", "max_brake", "map_D"} <= set(by):
        aggregate.update(
            control_reproduces_the_bout=bool(by["control"] in high),
            x_can_terminate_at_observed_D=bool(by["max_brake"] not in high),
            x_would_have_terminated_at_mapped_D=bool(by["map_D"] not in high),
        )
    ladder = sorted(((r["X_mean"], r["arm_id"], r["resolved_label"]) for r in rows
                     if r["arm_id"].startswith("ladder_X")), reverse=True)
    if ladder:
        persists = [x for x, _a, lab in ladder if lab in high]
        terminates = [x for x, _a, lab in ladder if lab not in high]
        aggregate["termination_bracket"] = dict(
            ladder=[dict(X=x, arm_id=a, label=lab) for x, a, lab in ladder],
            lowest_X_that_still_persists=(min(persists) if persists else None),
            highest_X_that_terminates=(max(terminates) if terminates else None),
            note=("the trajectory's own X settled at "
                  f"{float(observed_x.mean()):.3f} and its bout persisted"),
        )
    _write_json(os.path.join(OUT, "dx_arbitration.json"), aggregate)
    running = os.path.join(OUT, "RUNNING.json")
    if os.path.exists(running):
        os.replace(running, os.path.join(OUT, "RUNNING.superseded.json"))
    _write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=_now()))
    print(json.dumps({k: aggregate[k] for k in
                      ("labels", "control_reproduces_the_bout",
                       "x_can_terminate_at_observed_D",
                       "x_would_have_terminated_at_mapped_D")}, indent=2))


if __name__ == "__main__":
    main()
