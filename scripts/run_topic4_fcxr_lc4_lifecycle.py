#!/usr/bin/env python3
"""FCXR-LC4 F2: one no-kick lifecycle and its exact final-D confirmation.

No tuning lives here.  The candidate must already have passed F0 and F1.  ``nominal`` runs the one
authorised 70 s trajectory; only an eligible result unlocks ``confirm``, a 12 s continuation from
the exact final state with the measured spatial D field frozen.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import fcntl
import gc
import json
import sys
import time
from contextlib import contextmanager

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_phase_map as PM  # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC1R  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop  # noqa: E402
from src.topic4_fcxr_lc3_geometry import install_registered_noise_rng  # noqa: E402
from src.topic4_fcxr_lc3_ledger import build_event_ledger, snapshot_table  # noqa: E402
from src.topic4_fcxr_lc3_stage import reference_band  # noqa: E402
from src.topic4_fcxr_lc3_statefork import load_into, save_loop_state  # noqa: E402
from src.topic4_fcxr_lc4_lifecycle import (  # noqa: E402
    adjudicate_frozen_D,
    adjudicate_nominal,
    first_ictal_bout,
    refractory_ceiling_fraction,
)
from src.topic4_mz_fcxr_lifecycle import classify_lifecycle  # noqa: E402


OUT = os.path.join(E01.OUT, "lc4_lifecycle_gate")
NOMINAL_MS = 70000.0
CONFIRM_MS = 12000.0
CONFIRM_BURN_MS = 2000.0
SNAP_MS = 250.0
NOISE_SEED = 401
DT = E01.DT


@contextmanager
def _stage_lock(name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f".{name}.lock"), "w") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"stage {name} is already running") from exc
        yield


def _resource(stage, **extra):
    row = dict(stage=stage, epoch=time.time(), **GEO._meminfo(), **extra)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "resource_log.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush(); os.fsync(f.fileno())
    return row


def _preflight(stage):
    E01.FCXR._assert_engine_blessed()
    m = _resource(f"{stage}_PREFLIGHT")
    if m["mem_available_gib"] < 128.0:
        raise SystemExit(f"{stage}: MemAvailable {m['mem_available_gib']:.1f} GiB < 128 GiB")
    if not os.path.isfile(E01.ARTIFACTS["lc1_baseline"]):
        raise SystemExit("missing frozen LC1 baseline contract")
    return m


def _candidate():
    bp = os.path.join(OUT, "baseline_verdict.json")
    op = os.path.join(OUT, "onset_surface_verdict.json")
    if not (os.path.isfile(bp) and os.path.isfile(op)):
        raise SystemExit("F2 requires complete F0 and F1 verdicts")
    b = GEO._load_json(bp)
    o = GEO._load_json(op)
    c = b.get("selected_candidate")
    if c is None:
        raise SystemExit("F2 blocked: no F0 candidate")
    if not bool((o.get("gate") or {}).get("passed", False)):
        raise SystemExit("F2 blocked: F1 onset surface did not pass")
    if o.get("candidate") != c:
        raise SystemExit("F2 blocked: F0/F1 candidate mismatch")
    return c


def _cfg(candidate):
    cfg = E01._dynamic_cfg(GEO._point(GEO.H1_POINT_ID))
    cfg.update(
        use_m=True, tau_adp=float(candidate["tau_adp_ms"]), eta_m=0.0,
        m_hill_K=float(candidate["K"]), m_hill_n=float(candidate["n"]),
        tau_a_on=float(candidate["tau_a_on_ms"]),
        tau_a_off=float(candidate["tau_a_off_ms"]),
        g_m_max=float(candidate["g_m_max"]),
    )
    if candidate.get("deadzone") is not None:
        cfg["m_hill_deadzone"] = float(candidate["deadzone"])
    if candidate.get("theta_h_lc2") is not None:
        cfg["theta_h_lc2"] = float(candidate["theta_h_lc2"])
    return cfg


def _fresh_context(candidate, run_ms):
    S = PP.build_substrate(1)
    install_registered_noise_rng(S["net"])
    snaps = {int(round(t / DT)): f"t{int(t)}"
             for t in np.arange(0.0, float(run_ms) + SNAP_MS, SNAP_MS)}
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**_cfg(candidate)), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snaps)
    S["net"]["rng"] = np.random.default_rng(NOISE_SEED)
    return S, slow


def _reduce(run, S, baseline):
    res = dict(rate_E=run["rate_E"], rate_I=run["rate_I"], E_spk_bool=run["E_spk_bool"])
    wins, numerical, rate = LC1R._reduce_run_windows(
        res, run["checkpoint"].slow, S, DT,
        float(baseline["frozen_event_bar"]), baseline["band"])
    lifecycle = classify_lifecycle(wins, baseline["band"])
    events, af, af_dt, floor, _ = OLD._events_from_res(
        res, DT, event_bar=float(baseline["frozen_event_bar"]))
    return res, wins, numerical, rate, lifecycle, events, af, af_dt, floor


def _regional_trace_npz(path, slow, S, *, rate, af, af_dt, extra=None):
    table = snapshot_table(slow.snapshots, DT, GEO._region_masks(S))
    stride = max(1, int(round(10.0 / DT)))
    payload = dict(
        rate_dt_ms=np.asarray([10.0], np.float32),
        rate_E=np.asarray(rate[::stride], np.float32),
        af=np.asarray(af, np.float32), af_bin_ms=np.asarray([af_dt], np.float32),
        trace_dt_ms=np.asarray([10.0], np.float32),
        adap_current=np.asarray(slow.trace_adap_current[::stride], np.float32),
        a_mean=np.asarray(slow.trace_a_mean[::stride], np.float32),
        snapshot_t_ms=np.asarray([r["t_ms"] for r in table], np.float32),
        **{f"snapshot_{v}_{rg}": np.asarray([r[v][rg] for r in table], np.float32)
           for v in ("D", "H", "X", "y")
           for rg in ("core_A", "core_B", "axial", "off_axis", "all")},
    )
    payload.update(extra or {})
    np.savez_compressed(path, **payload)
    return table


def stage_nominal():
    _preflight("F2_NOMINAL")
    candidate = _candidate()
    out_json = os.path.join(OUT, "nominal_lifecycle.json")
    state_path = os.path.join(OUT, "nominal_final_exact_state.npz")
    trace_path = os.path.join(OUT, "nominal_lifecycle_traces.npz")
    if os.path.isfile(out_json) and os.path.isfile(state_path):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior

    GEO._write_json(os.path.join(OUT, "F2_NOMINAL_RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), run_ms=NOMINAL_MS,
                         candidate=candidate, started=GEO._now()))
    S, slow = _fresh_context(candidate, NOMINAL_MS)
    p = dataclasses.replace(S["p"], T=NOMINAL_MS, dt=DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], slow=slow, n_steps=int(round(NOMINAL_MS / DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    band = reference_band(baseline)
    res, wins, numerical, rate, lifecycle, events, af, af_dt, floor = _reduce(run, S, baseline)

    regimes = lifecycle["regimes"]
    bout = first_ictal_bout(regimes, float(baseline["band"]["win_ms"]))
    if bout is None:
        onset_ms = offset_ms = None
        pre_rate = float(np.mean(rate))
        post_rate = float("nan")
        ceiling = 0.0
    else:
        onset_ms = float(bout[0] * baseline["band"]["win_ms"])
        ended = bout[1] + 1 < len(regimes)
        offset_ms = (float((bout[1] + 1) * baseline["band"]["win_ms"])
                     if ended else None)
        i_on = max(1, int(round(onset_ms / DT)))
        pre_rate = float(np.mean(rate[:i_on]))
        if offset_ms is None:
            post_rate = float("nan")
            bout_stop = NOMINAL_MS
        else:
            j0 = int(round(offset_ms / DT))
            j1 = min(rate.size, j0 + int(round(2000.0 / DT)))
            post_rate = float(np.mean(rate[j0:j1])) if j1 > j0 else float("nan")
            bout_stop = offset_ms
        ceiling = refractory_ceiling_fraction(
            run["E_spk_bool"], dt_ms=DT, onset_ms=onset_ms, offset_ms=bout_stop,
            tau_ref_ms=float(S["p"].tau_ref_E))

    gate = adjudicate_nominal(
        regimes=regimes, win_ms=float(baseline["band"]["win_ms"]), events=events,
        total_ms=NOMINAL_MS, reference_band=band,
        numerical_safe=not bool(numerical.get("numerical_unsafe")),
        refractory_fraction=ceiling, pre_rate_hz=pre_rate,
        postictal_rate_hz=post_rate)

    slow_f = run["checkpoint"].slow
    ne = int(slow_f.NE)
    d_final = 1.0 - np.asarray(slow_f.z[:ne], float)
    final_hash = save_loop_state(state_path, run["checkpoint"])
    table = _regional_trace_npz(
        trace_path, slow_f, S, rate=rate, af=af, af_dt=af_dt,
        extra=dict(final_D=d_final.astype(np.float32)))
    r_base = float(np.median(rate[:max(1, int(round((gate.get("onset_ms") or NOMINAL_MS) / DT)))]))
    ledger = build_event_ledger(
        events=events, af=af, af_bin_ms=af_dt, floor_af=floor,
        rate_hz=rate, dt_ms=DT, r_base_hz=r_base, table=table,
        onset_ms=gate.get("onset_ms"), offset_ms=gate.get("offset_ms"),
        total_ms=NOMINAL_MS)
    rec = dict(
        status="COMPLETE", stage="F2_NOMINAL", candidate=candidate,
        run_ms=NOMINAL_MS, connection_seed=1, noise_seed=NOISE_SEED,
        no_kick=True, no_reset=True, no_parameter_step=True,
        nominal_gate=gate, lifecycle_classifier=lifecycle, numerical=numerical,
        event_ledger=ledger, exact_final_state=state_path,
        exact_final_state_hash=final_hash, traces=trace_path,
        final_D_mean=float(d_final.mean()), final_D_max=float(d_final.max()),
        final_X_mean=float(np.mean(slow_f.x_relay)),
        final_a_mean=float(np.mean(slow_f.a[:ne])),
        max_rate_hz=float(np.max(rate)), mean_rate_hz=float(np.mean(rate)),
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        claim_boundary=("single-seed nominal eligibility only; complete lifecycle requires the "
                        "exact-state frozen-D continuation"),
        finished=GEO._now(),
    )
    GEO._write_json(out_json, rec)
    GEO._write_json(os.path.join(OUT, "F2_NOMINAL_DONE.json"),
                    dict(status="DONE", verdict=gate["verdict"],
                         eligible_for_frozen_D=gate["eligible_for_frozen_D"],
                         finished=GEO._now()))
    _resource("F2_NOMINAL_DONE", wall_s=rec["wall_s"], peak_rss_gib=rec["peak_rss_gib"])
    del run, res, S
    gc.collect()
    return rec


def _rekey_confirmation(state):
    base = int(state.slow._step_i)
    state.slow.snapshots.clear()
    state.slow._snap_steps = {base + int(round(t / DT)): f"t{int(t)}"
                              for t in np.arange(0.0, CONFIRM_MS + SNAP_MS, SNAP_MS)}


def stage_confirm():
    _preflight("F2_CONFIRM")
    candidate = _candidate()
    nominal_path = os.path.join(OUT, "nominal_lifecycle.json")
    if not os.path.isfile(nominal_path):
        raise SystemExit("F2 confirmation needs nominal_lifecycle.json")
    nominal = GEO._load_json(nominal_path)
    if not bool((nominal.get("nominal_gate") or {}).get("eligible_for_frozen_D", False)):
        raise SystemExit("F2 confirmation blocked: nominal trajectory is not eligible")
    state_path = nominal["exact_final_state"]
    if not os.path.isfile(state_path):
        raise SystemExit("F2 confirmation missing exact final state")

    out_json = os.path.join(OUT, "frozen_D_confirmation.json")
    if os.path.isfile(out_json):
        prior = GEO._load_json(out_json)
        if prior.get("status") == "COMPLETE":
            return prior
    GEO._write_json(os.path.join(OUT, "F2_CONFIRM_RUNNING.json"),
                    dict(status="RUNNING", pid=os.getpid(), run_ms=CONFIRM_MS,
                         source_state=state_path, started=GEO._now()))

    S, template_slow = _fresh_context(candidate, CONFIRM_MS)
    template = PM._seed_template(S, template_slow)
    start = load_into(state_path, template)
    ne = int(start.slow.NE)
    z_final = np.asarray(start.slow.z[:ne], float).copy()
    start.slow.cfg = dataclasses.replace(
        start.slow.cfg, use_z=False, z_frozen_E=z_final.copy())
    start.slow.z[:ne] = z_final
    _rekey_confirmation(start)
    p = dataclasses.replace(S["p"], T=CONFIRM_MS, dt=DT)
    t0 = time.time()
    run = run_fcxr_loop(p, S["net"], start=start, n_steps=int(round(CONFIRM_MS / DT)),
                        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"])
    baseline = GEO._load_json(E01.ARTIFACTS["lc1_baseline"])
    band = reference_band(baseline)
    res, wins, numerical, rate, lifecycle, events, af, af_dt, floor = _reduce(run, S, baseline)
    ceiling = refractory_ceiling_fraction(
        run["E_spk_bool"], dt_ms=DT, onset_ms=0.0, offset_ms=CONFIRM_MS,
        tau_ref_ms=float(S["p"].tau_ref_E))
    gate = adjudicate_frozen_D(
        regimes=lifecycle["regimes"], win_ms=float(baseline["band"]["win_ms"]),
        events=events, total_ms=CONFIRM_MS, burn_ms=CONFIRM_BURN_MS,
        reference_band=band, numerical_safe=not bool(numerical.get("numerical_unsafe")),
        refractory_fraction=ceiling)
    trace_path = os.path.join(OUT, "frozen_D_confirmation_traces.npz")
    _regional_trace_npz(trace_path, run["checkpoint"].slow, S,
                        rate=rate, af=af, af_dt=af_dt,
                        extra=dict(frozen_D=(1.0 - z_final).astype(np.float32)))
    rec = dict(
        status="COMPLETE", stage="F2_CONFIRM", candidate=candidate,
        run_ms=CONFIRM_MS, burn_ms=CONFIRM_BURN_MS,
        continued_from_exact_state=state_path,
        frozen_D_hash=E01._arr_hash(1.0 - z_final),
        frozen_D_mean=float(np.mean(1.0 - z_final)),
        frozen_D_max=float(np.max(1.0 - z_final)),
        gate=gate, lifecycle_classifier=lifecycle, numerical=numerical,
        n_events=len(events), traces=trace_path,
        wall_s=time.time() - t0, peak_rss_gib=GEO._meminfo()["self_peak_rss_gib"],
        claim_boundary=("candidate complete lifecycle at one connection/noise seed only; "
                        "confirmation seeds remain a later sprint"),
        finished=GEO._now(),
    )
    GEO._write_json(out_json, rec)
    GEO._write_json(os.path.join(OUT, "lifecycle_verdict.json"),
                    dict(status="COMPLETE", verdict=gate["verdict"], passed=gate["passed"],
                         nominal=nominal_path, frozen_D=out_json, finished=GEO._now()))
    GEO._write_json(os.path.join(OUT, "F2_CONFIRM_DONE.json"),
                    dict(status="DONE", verdict=gate["verdict"], finished=GEO._now()))
    _resource("F2_CONFIRM_DONE", wall_s=rec["wall_s"], peak_rss_gib=rec["peak_rss_gib"])
    del run, res, S
    gc.collect()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--stage", choices=("nominal", "confirm"), required=True)
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k LC4 lifecycle requires --confirm-run")
    with _stage_lock(f"f2_{args.stage}"):
        try:
            result = stage_nominal() if args.stage == "nominal" else stage_confirm()
        except BaseException as exc:
            GEO._write_json(os.path.join(OUT, f"F2_{args.stage.upper()}_FAILED.json"),
                            dict(status="FAILED", error=repr(exc), finished=GEO._now()))
            raise
    gate = result.get("nominal_gate") or result.get("gate") or {}
    print(json.dumps(dict(status=result.get("status"), stage=result.get("stage"),
                          verdict=gate.get("verdict"), passed=gate.get("passed")), indent=2))


if __name__ == "__main__":
    main()
