#!/usr/bin/env python
"""Can a moved relay Hill let X descend on its own to the depth that terminates?

The first no-kick trajectory ran a 40 s discharge it never stopped. Its relay
availability settled at mean 0.3945; clamping it to 0.380 terminates the same state
and 0.395 does not, so it missed self-termination by under 4%. It settled there
because the sensor mean never reached the registered gate (59.4 Hz against 76.64), so
the Hill drive never saturated.

This sweep pins wear at the value the trajectory actually reached and leaves X free,
varying only where the Hill sits:

    x_inf = 1 - (1 - x_min) * Hill([y - y_gate]+ ; K_y, n)

Every arm seeds from the same byte-parity-verified 44.75 s late-bout state and uses the
map's screen/extend protocol and classifier, so outcomes sit on the same scale as the
102 map rows and the arbitration arms.

An arm is only evidence for the mechanism if it terminates *and* its relay actually
descended past the bracketed threshold; a termination with X still above it would mean
something other than relay depth did the work, so both are recorded.

Usage:
    python scripts/run_topic4_fcxr_lc3_hillsweep.py --confirm-run
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
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_dxprobe import (  # noqa: E402
    freeze_dynamic_state,
    set_hill_placement,
)
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    EXTENDED_MS,
    EXTENDED_TAIL_MS,
    H1_POINT_ID,
    SCREEN_MS,
    SCREEN_TAIL_MS,
    choose_map_workers,
    extension_required,
    install_registered_noise_rng,
    load_prepared_checkpoint,
)

OUT = os.path.join(E01.OUT, "hill_placement_sweep")
SEED_STATE = os.path.join(
    E01.OUT, "dynamic_reconnaissance", "exact_landmarks", "noise401_step895001.pkl")
# Registered placement of the relay Hill in the E4 configuration.
BASE_K_Y = 5.0
BASE_Y_GATE = 76.63856219587187
# Bracketed on the same seed state by scripts/run_topic4_fcxr_lc3_dxprobe.py.
TERMINATION_X_UPPER = 0.395     # persists at and above this uniform depth
TERMINATION_X_LOWER = 0.380     # terminates at and below it
OBSERVED_X_MEAN = 0.3945
HIGH_LABELS = ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")

Y_GATES = (BASE_Y_GATE, 72.0, 68.0, 64.0, 60.0)
K_YS = (BASE_K_Y, 4.0, 3.0)

_SEED = None
_SUBSTRATE = None
_POINT = None


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
    import resource
    return dict(mem_available_gib=d["MemAvailable"] / 1048576.0,
                swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0,
                self_peak_rss_gib=resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1048576.0)


def _context():
    """Substrate, seed state and workpoint, built once per worker process."""
    global _SEED, _SUBSTRATE, _POINT
    if _SUBSTRATE is None:
        _SUBSTRATE = PP.build_substrate(1)
        install_registered_noise_rng(_SUBSTRATE["net"])
        from src.topic4_fcxr_lc3 import _constants
        _constants(_SUBSTRATE["p"], _SUBSTRATE["net"])
    if _SEED is None:
        _SEED = load_prepared_checkpoint(SEED_STATE)["state"]
    if _POINT is None:
        _POINT = GEO._point(H1_POINT_ID)
    return _SUBSTRATE, _SEED, _POINT


def _tag(value):
    return f"{float(value):.2f}".replace(".", "p")


def _arms():
    return [dict(arm_id=f"gate{_tag(g)}_Ky{_tag(k)}", y_gate=float(g), K_y=float(k),
                 is_control=bool(g == BASE_Y_GATE and k == BASE_K_Y))
            for g in Y_GATES for k in K_YS]


def _arm_path(arm_id):
    return os.path.join(OUT, f"arm_{arm_id}.json")


def _run_arm(arm):
    prior_path = _arm_path(arm["arm_id"])
    if os.path.isfile(prior_path):
        prior = json.load(open(prior_path))
        if prior.get("resolved_label"):
            return prior
    S, seed, point = _context()
    child = freeze_dynamic_state(seed, freeze_x=False)     # wear pinned where it landed
    set_hill_placement(child, K_y=arm["K_y"], y_gate=arm["y_gate"])

    t0 = time.time()
    p1 = dataclasses.replace(S["p"], T=SCREEN_MS, dt=E01.DT)
    first = run_fcxr_loop(p1, S["net"], start=child, n_steps=int(round(SCREEN_MS / E01.DT)),
                          capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    cls1 = GEO._tail_observables(first, S, point, tail_ms=SCREEN_TAIL_MS,
                                 analysis_start_ms=0.0)
    extended = extension_required(state_kind="high", label=cls1["label"])
    final, total_ms, cls2 = first, SCREEN_MS, None
    if extended:
        extra = EXTENDED_MS - SCREEN_MS
        p2 = dataclasses.replace(S["p"], T=extra, dt=E01.DT)
        second = run_fcxr_loop(p2, S["net"], start=first["checkpoint"],
                               n_steps=int(round(extra / E01.DT)), capture_final=True,
                               store_spikes=True, v_th_per_neuron=S["vth"])
        final = dict(second)
        final["rate_E"] = np.concatenate([first["rate_E"], second["rate_E"]])
        final["E_spk_bool"] = np.concatenate(
            [first["E_spk_bool"], second["E_spk_bool"]], axis=0)
        final["n_steps"] = final["rate_E"].size
        cls2 = GEO._tail_observables(final, S, point, tail_ms=EXTENDED_TAIL_MS,
                                     analysis_start_ms=EXTENDED_MS - EXTENDED_TAIL_MS)
        total_ms = EXTENDED_MS
    resolved = cls2 or cls1

    slow = final["checkpoint"].slow
    n = int(final["n_steps"])
    xtrace = np.asarray(slow.trace_x_relay_mean[-n:], dtype=float)
    terminated = resolved["label"] not in HIGH_LABELS
    x_min = float(xtrace.min()) if xtrace.size else float("nan")
    record = dict(
        arm_id=arm["arm_id"], y_gate=arm["y_gate"], K_y=arm["K_y"],
        is_control=arm["is_control"],
        resolved_label=resolved["label"], terminated=bool(terminated),
        x_mean_start=float(xtrace[0]) if xtrace.size else None,
        x_mean_min=x_min, x_mean_final=float(xtrace[-1]) if xtrace.size else None,
        x_reached_termination_bracket=bool(x_min <= TERMINATION_X_LOWER),
        x_below_observed_setpoint=bool(x_min < OBSERVED_X_MEAN),
        mechanism_consistent=bool(terminated == (x_min <= TERMINATION_X_LOWER)),
        refractory_ceiling_fraction=resolved["refractory_ceiling_fraction"],
        h_mean=resolved["h_mean"], numerical_unsafe=resolved["numerical_unsafe"],
        mean_rate_hz=float(np.mean(final["rate_E"])),
        total_ms=float(total_ms), extended=bool(extended),
        initial_screen=cls1, extended_classification=cls2,
        wall_s=time.time() - t0, peak_rss_gib=_meminfo()["self_peak_rss_gib"],
        finished=_now(),
    )
    _write_json(prior_path, record)
    del first, final, child
    gc.collect()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k Hill-placement sweep requires --confirm-run")
    if not os.path.isfile(SEED_STATE):
        raise SystemExit(f"missing late-bout seed state: {SEED_STATE}")
    mem0 = _meminfo()
    if mem0["mem_available_gib"] < 96.0:
        raise SystemExit("sweep requires 96 GiB MemAvailable")
    os.makedirs(OUT, exist_ok=True)
    arms = _arms()
    _write_json(os.path.join(OUT, "RUNNING.json"),
                dict(status="RUNNING", pid=os.getpid(), n_arms=len(arms), started=_now()))

    # The control arm doubles as the smoke row that sizes the pool, exactly as the
    # geometry map does: one measured footprint beats an assumed one.
    control = next(a for a in arms if a["is_control"])
    smoke = _run_arm(control)
    single_rss = float(smoke["peak_rss_gib"])
    swap0 = mem0["swap_used_mib"]
    workers = choose_map_workers(
        mem_available_gib=_meminfo()["mem_available_gib"], swap_used_mib=swap0,
        swap_baseline_mib=swap0, single_rss_gib=single_rss, cpu_count=os.cpu_count() or 2)
    pool_size = max(workers, 1)
    print(f"[hill] control {smoke['resolved_label']} x_min={smoke['x_mean_min']:.4f} "
          f"rss={single_rss:.1f} GiB -> {workers} workers", flush=True)

    pending = [a for a in arms if not a["is_control"]]
    rows = [smoke]
    with ProcessPoolExecutor(max_workers=max(pool_size, 8)) as pool:
        active, cursor, last_swap = {}, 0, swap0
        while cursor < len(pending) or active:
            mem = _meminfo()
            swap_delta = mem["swap_used_mib"] - swap0
            if swap_delta >= 512.0 and mem["swap_used_mib"] > last_swap:
                raise RuntimeError(f"swap hard stop: +{swap_delta:.1f} MiB and rising")
            last_swap = mem["swap_used_mib"]
            workers = min(max(pool_size, 8), choose_map_workers(
                mem_available_gib=mem["mem_available_gib"],
                swap_used_mib=mem["swap_used_mib"], swap_baseline_mib=swap0,
                single_rss_gib=single_rss, cpu_count=os.cpu_count() or 2))
            allow = swap_delta < 256.0 and mem["mem_available_gib"] >= 96.0
            while allow and cursor < len(pending) and len(active) < workers:
                arm = pending[cursor]
                cursor += 1
                active[pool.submit(_run_arm, arm)] = arm["arm_id"]
            if not active:
                time.sleep(30.0)
                continue
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                arm_id = active.pop(future)
                rec = future.result()
                rows.append(rec)
                print(f"[hill] {len(rows)}/{len(arms)} {arm_id:22s} gate={rec['y_gate']:.2f} "
                      f"K_y={rec['K_y']:.2f} -> {rec['resolved_label']:22s} "
                      f"x_min={rec['x_mean_min']:.4f} rate={rec['mean_rate_hz']:7.2f} Hz",
                      flush=True)

    terminating = [r for r in rows if r["terminated"]]
    inconsistent = [r["arm_id"] for r in rows if not r["mechanism_consistent"]]
    aggregate = dict(
        status="COMPLETE", schema="fcxr-lc3-hill-sweep-1.0",
        seed_state=SEED_STATE, base_K_y=BASE_K_Y, base_y_gate=BASE_Y_GATE,
        observed_X_mean=OBSERVED_X_MEAN,
        termination_bracket=dict(terminates_at_or_below=TERMINATION_X_LOWER,
                                 persists_at_or_above=TERMINATION_X_UPPER),
        n_arms=len(rows), n_terminating=len(terminating),
        terminating_arms=[dict(arm_id=r["arm_id"], y_gate=r["y_gate"], K_y=r["K_y"],
                               x_mean_min=r["x_mean_min"],
                               mean_rate_hz=r["mean_rate_hz"]) for r in terminating],
        mechanism_inconsistent_arms=inconsistent,
        claim_boundary=("wear pinned at one real late-bout state of one noise seed; a "
                        "terminating arm is a mechanism demonstration, not a lifecycle "
                        "result and not a parameter acceptance"),
        rows=rows, completed=_now(),
    )
    _write_json(os.path.join(OUT, "hill_sweep.json"), aggregate)
    running = os.path.join(OUT, "RUNNING.json")
    if os.path.exists(running):
        os.replace(running, os.path.join(OUT, "RUNNING.superseded.json"))
    _write_json(os.path.join(OUT, "DONE.json"), dict(status="DONE", finished=_now()))
    print(json.dumps({k: aggregate[k] for k in
                      ("n_arms", "n_terminating", "terminating_arms",
                       "mechanism_inconsistent_arms")}, indent=2))


if __name__ == "__main__":
    main()
