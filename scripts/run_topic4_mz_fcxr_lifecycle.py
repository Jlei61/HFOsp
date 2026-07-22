"""FCXR-LC1 runner — dynamic slow-feedback lifecycle closure.

Nothing runs on import; every simulation requires --confirm-run. Fixed dt=0.05. Accepted FCXR-RC1 substrate
(external additive FF + recurrent conductance + recurrent-only smooth saturation g_sat=21.6), seeds 1/3.
The blessed engine (kick_probe.py) is NEVER edited; the asymmetric relay (tau_x_down/tau_x_up) rides the
non-blessed mz_slow_vars plugin. Phases are run EXPLICITLY (pilot-first gating), not one autonomous driver.

Commands (this file, E0/E1):
  dry-run   report task count / T / workers / est raster memory, NO simulation (§14.1 pre-nohup gate).
  smoke     one short continuous run end-to-end + timing + peak RSS (plumbing validation).
  baseline  slow-off statistical interictal contract per seed -> baseline_contract.json (E1 anchor + band).
Outputs: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure/
Design: docs/superpowers/plans/2026-07-22-topic4-mz-fcxr-lc1.md.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-fcxr-lc1")

import argparse
import dataclasses
import gc
import json
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP          # noqa: E402  build_substrate, R_KICK
import run_topic4_mz_slowvars as OLD    # noqa: E402  build_core_masks, compute_baseline_ref, C.*, event detection
import run_topic4_mz_fcxr as FCXR       # noqa: E402  _fc_cfg + scaffolding (flock/bless/run_id/io/mem/plan_workers)
from kick_probe import simulate_kick    # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_fcxr_dynamics import (  # noqa: E402
    rolling_rate_upper, load_onset_depletion_pi, assert_field_substrate_aligned, frozen_z_field,
)
from src.topic4_mz_fcxr_lifecycle import (  # noqa: E402
    build_windows, classify_lifecycle, depletion_coordinate, _smooth_isolated, LC_THRESHOLDS,
)

# ---- locked LC1 constants ----
G_SAT = 21.6
DT_LC = 0.05
LC_WIN_MS = 1000.0          # classifier analysis-window granularity (fine bout resolution)
LC_LOOKBACK_MS = 8000.0     # trailing window for the event-rate estimate (sparse-IED robust)
SEEDS = (1, 3)
OUT = os.path.join(FCXR.OUT_ROOT, "lifecycle_closure")
D_Z_SNAP_MS = 100.0         # D_Z(t) phase-portrait snapshot cadence
BOUNDED_MAX_HZ = 60.0       # end-of-run mean rate above this = clearly unbounded (soft flag; safety is `finite`)
SNAP_ZA_FMT = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility",
                           "snapshots", "zA_q75_tz5000", "seed_{seed}.npz")   # p_i weights for D_Z
Z_REGIMES = {   # existing calibration (results/topic4_sef_hfo/mz_slowvars/calibration.json); NOT invented
    "q75": dict(I_th_EI=95.19851312666987, tau_z=5000.0),    # primary — mid depletion (zA_q75_tz5000)
    "q50": dict(I_th_EI=1.6652801609959704, tau_z=10000.0),  # sensitivity — strong depletion (zA_q50_tz10000)
}
D_DENSE = 0.15              # frozen dense operating point (Stage-D metastable dense region D~=0.145-0.15)
SENSOR_TAU_Y = 120.0       # persistence-sensor time constant (primary; E3 sensitivity 80/200)
SENSOR_K_Y, SENSOR_HILL_N = 5.0, 4

# ---- resource watchdog thresholds (§13.3; relative to the per-launcher swap baseline) ----
SOFT_MEM_GB, HARD_MEM_GB = 64.0, 32.0
SOFT_SWAP_MB, HARD_SWAP_MB = 256.0, 512.0


# ----------------------------------------------------------------- resource watchdog / sentinels
def _swap_used_mb():
    _, swap_gb = FCXR._meminfo()
    return swap_gb * 1024.0


def _resource_state(swap_base_mb):
    """('ok'|'soft'|'hard', info) relative to the per-launcher swap baseline (§13.3)."""
    avail_gb, swap_gb = FCXR._meminfo()
    swap_mb = swap_gb * 1024.0
    swap_delta = swap_mb - swap_base_mb
    info = dict(mem_available_gb=round(avail_gb, 2), swap_used_mb=round(swap_mb, 1),
                swap_delta_mb=round(swap_delta, 1))
    if avail_gb < HARD_MEM_GB or swap_delta >= HARD_SWAP_MB:
        return "hard", info
    if avail_gb < SOFT_MEM_GB or swap_delta >= SOFT_SWAP_MB:
        return "soft", info
    return "ok", info


def _raster_gb_lc(T_ms):
    """True per-run E-raster GB at the LC dt=0.05 (FCXR._raster_gb hardcodes dt=0.1 -> half)."""
    return (float(T_ms) / DT_LC) * 32000.0 / (1024.0 ** 3)


def _strict_plan_workers(T_ms, requested):
    """FCXR worker plan with the STRICTER LC1 cap: T>=20s -> 1 worker always (not relaxed to 2), <=2 overall.
    T is scaled by (FCXR.DT/DT_LC)=2 so FCXR's internal raster budget matches the TRUE dt=0.05 raster."""
    plan = FCXR._plan_workers(float(T_ms) * (FCXR.DT / DT_LC), requested)
    hard_cap = 1 if float(T_ms) >= 20000.0 else 2
    plan["lc_hard_cap"] = hard_cap
    plan["workers"] = min(int(plan["workers"]), hard_cap)
    plan["T_ms_real"] = float(T_ms)
    plan["raster_gb_real"] = round(_raster_gb_lc(T_ms), 2)
    return plan


def _self_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _atomic_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=FCXR._jsonable)
    os.replace(tmp, path)


def _sentinel(run_dir, name, **payload):
    payload.setdefault("t", datetime.now(timezone.utc).isoformat())
    _atomic_json(os.path.join(run_dir, name), payload)


def _launch_baseline(run_dir, command):
    """Record the per-launcher resource baseline (§13.1): the swap watchdog is relative to THIS value."""
    avail_gb, swap_gb = FCXR._meminfo()
    base = dict(t=datetime.now(timezone.utc).isoformat(), pid=os.getpid(), command=command,
                worktree=ROOT, git_sha=FCXR._git_sha(), mem_available_gb=round(avail_gb, 2),
                swap_used_baseline_mb=round(swap_gb * 1024.0, 1))
    _atomic_json(os.path.join(run_dir, "launch_baseline.json"), base)
    FCXR._resource_log(run_dir, "launch", base)
    return base["swap_used_baseline_mb"]


# ----------------------------------------------------------------- continuous run + numerical safety
def _lc_run(S, cfg_dict, T_ms, *, seed, dt=DT_LC, snapshot_steps=None):
    """One purely slow-driven continuous run (NO kick: KICK_BOOST=0, t_kick=1e9). early_stop OFF so a bounded
    high state is not killed as operational runaway."""
    p = dataclasses.replace(S["p"], T=float(T_ms), dt=float(dt))
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg_dict), NE=S["NE"],
                      core_mask_E=OLD.build_core_masks(S), snapshot_steps=snapshot_steps)
    S["net"]["rng"] = np.random.default_rng(int(seed))
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"], early_stop_runaway=False)
    return res, slow


def _numerical(S, res, slow, dt):
    taur = np.asarray(slow.trace_tau_eff_ratio_min, float)
    clipf = np.asarray(slow.trace_conductance_clip_frac, float)
    finite = bool(np.all(np.isfinite(res["rate_E"])))
    tau_eff_min_ms = float(S["p"].tau_m_E * taur.min()) if taur.size else float("nan")
    max_clip = float(clipf.max()) if clipf.size else 0.0
    unsafe = bool((not finite) or max_clip > 0.0 or (np.isfinite(tau_eff_min_ms) and tau_eff_min_ms < 2.0 * dt))
    return dict(finite=finite, tau_eff_min_ms=tau_eff_min_ms, clip_frac_max=max_clip, numerical_unsafe=unsafe)


def _slowoff_cfg():
    """Accepted RC1 slow-off workpoint (arm C: additive FF + recurrent conductance + smooth saturation)."""
    return FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False, rec_sat_g=G_SAT)


def _pct(a, q, d=0.0):
    a = np.asarray(a, float)
    return float(np.percentile(a, q)) if a.size else d


# ----------------------------------------------------------------- E1 baseline statistical contract
def _reduce_baseline(res, slow, S, dt):
    """Slow-off run -> (frozen bar, returning events, af, rate, roll_hi, numerical, per-window band)."""
    rate = np.asarray(res["rate_E"], float)
    frozen_bar = OLD.slowoff_event_bar(res, dt)
    events, af, af_bin, floor, _ = OLD._events_from_res(res, dt, event_bar=frozen_bar)
    af = np.asarray(af, float)
    ret = [e for e in events if e["returned"]]
    roll_hi = rolling_rate_upper(rate, dt)
    num = _numerical(S, res, slow, dt)
    bwin = build_windows(rate, dt, af, af_bin, roll_hi, ret, LC_WIN_MS,
                         event_lookback_ms=LC_LOOKBACK_MS, finite=num["finite"])
    recs = [w["recruit_frac"] for w in bwin]
    ers = [w["event_rate_hz"] for w in bwin]
    band = dict(
        win_ms=LC_WIN_MS, event_lookback_ms=LC_LOOKBACK_MS, roll_hi=roll_hi,
        recruit_p90=_pct(recs, 90, 0.0),
        event_rate_lo=max(0.05, 0.3 * _pct(ers, 10, 0.0)),
        event_rate_hi=1.8 * _pct(ers, 90, 0.0) if ers else 0.0,
    )
    return dict(frozen_bar=float(frozen_bar), roll_hi=float(roll_hi), n_returning=len(ret),
                af_bin_ms=float(af_bin), floor_af=float(floor), numerical=num, band=band,
                base_windows=bwin, returning_events=ret, rate=rate)


def cmd_baseline(args):
    """E1: slow-off statistical interictal contract per seed. Builds the classifier band + workpoint gate."""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        run_dir = os.path.join(OUT, "runs", FCXR._run_id(f"baseline_seed{args.seed}_T{int(args.T)}"))
        swap_base = _launch_baseline(run_dir, f"baseline --seed {args.seed} --T {args.T}")
        _sentinel(run_dir, "RUNNING.json", phase="baseline", seed=args.seed, T=args.T)
        state, info = _resource_state(swap_base)
        if state == "hard":
            _sentinel(run_dir, "ABORTED.json", reason="resource hard-stop before start", **info)
            raise SystemExit(f"[baseline] resource hard-stop before start: {info}")
        print(f"[baseline] seed={args.seed} T={args.T} dt={DT_LC}; {info}", flush=True)
        S = PP.build_substrate(args.seed)
        t0 = time.time()
        res, slow = _lc_run(S, _slowoff_cfg(), args.T, seed=args.seed)
        red = _reduce_baseline(res, slow, S, DT_LC)
        wall = round(time.time() - t0, 1)
        del res
        gc.collect()
        # workpoint gate: under this band, the slow-off run must classify as an interictal baseline (no ictal)
        lc = classify_lifecycle(red["base_windows"], red["band"])
        num = red["numerical"]
        gate_ok = bool((not num["numerical_unsafe"]) and num["clip_frac_max"] == 0.0
                       and red["n_returning"] >= OLD.MIN_BASE_EVENTS
                       and lc["label"] == "INTERICTAL_BASELINE")   # workpoint must read as a clean interictal baseline
        rate_stride = max(1, int(np.ceil(red["rate"].size / 4000)))
        FCXR._write_npz(os.path.join(OUT, f"baseline_trace_seed{args.seed}.npz"),
                        rate_dt_ms=np.asarray([DT_LC * rate_stride], np.float32),
                        rate_E=red["rate"][::rate_stride].astype(np.float32))
        contract = dict(
            seed=args.seed, T=args.T, dt=DT_LC, g_sat=G_SAT, wall_s=wall, peak_rss_gb=round(_self_rss_gb(), 2),
            frozen_event_bar=red["frozen_bar"], roll_hi_hz=red["roll_hi"], n_returning=red["n_returning"],
            min_base_events=OLD.MIN_BASE_EVENTS, af_bin_ms=red["af_bin_ms"], floor_af=red["floor_af"],
            numerical=num, band=red["band"], win_ms=LC_WIN_MS, event_lookback_ms=LC_LOOKBACK_MS,
            n_windows=len(red["base_windows"]), slowoff_lifecycle_label=lc["label"],
            slowoff_regimes=lc["regimes"], base_windows=red["base_windows"], workpoint_gate_pass=gate_ok,
            event_durations_ms=[round(float(e["dur_ms"]), 1) for e in red["returning_events"]],
            event_participation=[round(float(e["peak_ext"]), 4) for e in red["returning_events"]],
        )
        out = os.path.join(OUT, f"baseline_contract_seed{args.seed}.json")
        _atomic_json(out, contract)
        FCXR._resource_log(run_dir, "baseline_done", dict(wall_s=wall, gate_ok=gate_ok, **num))
        _sentinel(run_dir, "DONE.json", phase="baseline", seed=args.seed, gate_ok=gate_ok, out=out, wall_s=wall)
        print(f"[baseline] seed{args.seed}: n_ret={red['n_returning']} roll_hi={red['roll_hi']:.1f}Hz "
              f"band_recruit_p90={red['band']['recruit_p90']:.3f} er[{red['band']['event_rate_lo']:.2f},"
              f"{red['band']['event_rate_hi']:.2f}] slowoff={lc['label']} clip={num['clip_frac_max']:.3g} "
              f"tau_eff_min={num['tau_eff_min_ms']:.3f}ms GATE={'PASS' if gate_ok else 'FAIL'} "
              f"-> {out}  ({wall}s, peak {contract['peak_rss_gb']}GB)", flush=True)


# ----------------------------------------------------------------- E2 dynamic Z-only
def _zonly_cfg(regime):
    """Dynamic Z-only config for a calibration regime (X/M/phi off). q50 overrides the _fc_cfg-hardcoded I_th_EI."""
    r = Z_REGIMES[regime]
    cfg = _slowoff_cfg()
    cfg.update(use_z=True, tau_z=float(r["tau_z"]), I_th_EI=float(r["I_th_EI"]))
    return cfg


def _load_baseline_contract(seed):
    p = os.path.join(OUT, f"baseline_contract_seed{seed}.json")
    if not os.path.exists(p):
        raise SystemExit(f"missing baseline contract {p}; run `baseline --seed {seed} --confirm-run` first")
    return json.load(open(p))


def _reduce_run_windows(res, slow, S, dt, frozen_bar, band):
    """Reduce a dynamic run to classifier windows using the SEED's frozen slow-off bar + baseline band."""
    rate = np.asarray(res["rate_E"], float)
    num = _numerical(S, res, slow, dt)
    events, af, af_bin, floor, _ = OLD._events_from_res(res, dt, event_bar=frozen_bar)
    ret = [e for e in events if e["returned"]]
    wins = build_windows(rate, dt, np.asarray(af, float), float(af_bin), float(band["roll_hi"]), ret,
                         float(band["win_ms"]), event_lookback_ms=float(band["event_lookback_ms"]),
                         finite=num["finite"])
    return wins, num, rate


def cmd_zonly(args):
    """E2: dynamic Z-only (X/M off, no kick). Does Z preserve interictal first, then self-drive into the
    metastable dense-event region, bounded? Is D_Z(t) an event-locked staircase or a hard-threshold jump?"""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        run_dir = os.path.join(OUT, "runs", FCXR._run_id(f"zonly_seed{args.seed}_{args.regime}_T{int(args.T)}"))
        swap_base = _launch_baseline(run_dir, f"zonly --seed {args.seed} --regime {args.regime} --T {args.T}")
        _sentinel(run_dir, "RUNNING.json", phase="zonly", seed=args.seed, regime=args.regime, T=args.T)
        state, info = _resource_state(swap_base)
        if state == "hard":
            _sentinel(run_dir, "ABORTED.json", reason="resource hard-stop before start", **info)
            raise SystemExit(f"[zonly] resource hard-stop before start: {info}")
        bc = _load_baseline_contract(args.seed)
        band, frozen_bar = bc["band"], bc["frozen_event_bar"]
        r = Z_REGIMES[args.regime]
        print(f"[zonly] seed={args.seed} regime={args.regime} I_th_EI={r['I_th_EI']:.3f} tau_z={r['tau_z']:.0f} "
              f"T={args.T} dt={DT_LC}; {info}", flush=True)
        S = PP.build_substrate(args.seed)
        pk = load_onset_depletion_pi(SNAP_ZA_FMT.format(seed=args.seed))
        assert_field_substrate_aligned(pk, S)                 # STOP if the D_Z weight field is mis-registered
        p_i = pk["p_i"]
        snap = {int(round(t / DT_LC)): f"t{t}" for t in range(0, int(args.T), int(D_Z_SNAP_MS))}
        t0 = time.time()
        res, slow = _lc_run(S, _zonly_cfg(args.regime), args.T, seed=args.seed, snapshot_steps=snap)
        wins, num, rate = _reduce_run_windows(res, slow, S, DT_LC, frozen_bar, band)
        lc = classify_lifecycle(wins, band)
        end_rate = float(np.mean(rate[-max(1, int(round(500.0 / DT_LC))):]))
        z_mean_trace = np.asarray(slow.trace_z_mean, float)
        del res
        gc.collect()
        # D_Z(t) phase-portrait coordinate from the z snapshots (p_i-weighted depletion)
        snaps = sorted(slow.snapshots.values(), key=lambda s: s["step"])
        DZ = np.array([[s["step"] * DT_LC, depletion_coordinate(s["z_E"], p_i)] for s in snaps], float)
        dz = DZ[:, 1] if DZ.size else np.zeros(1)
        ddz = np.diff(dz) if dz.size > 1 else np.zeros(1)
        # onset = first window that leaves the interictal band (DENSE or ICTAL); pre = interictal preserved before it
        regimes = lc["regimes"]
        sm = _smooth_isolated(regimes)                        # ignore isolated baseline bursts -> SUSTAINED onset
        onset_idx = next((i for i, rg in enumerate(sm) if rg in ("DENSE", "ICTAL")), None)
        pre_interictal_ms = (onset_idx if onset_idx is not None else len(sm)) * band["win_ms"]
        entered_sustained = lc["label"] not in ("INTERICTAL_BASELINE", "PERMANENT_SILENCE", "UNRESOLVED")
        bounded = bool(num["finite"] and (not num["numerical_unsafe"]) and end_rate < BOUNDED_MAX_HZ)
        wall = round(time.time() - t0, 1)
        summary = dict(
            seed=args.seed, regime=args.regime, I_th_EI=r["I_th_EI"], tau_z=r["tau_z"], T=args.T, dt=DT_LC,
            wall_s=wall, peak_rss_gb=round(_self_rss_gb(), 2), numerical=num, lifecycle_label=lc["label"],
            regimes=regimes, n_windows=len(regimes), entered_sustained_dense=entered_sustained,
            onset_window_idx=onset_idx, pre_interictal_ms=pre_interictal_ms, end_rate_hz=end_rate, bounded=bounded,
            D_Z_start=float(dz[0]), D_Z_end=float(dz[-1]), D_Z_max=float(dz.max()),
            D_Z_max_step=float(np.max(np.abs(ddz))), D_Z_monotone=bool(np.all(ddz >= -1e-6)),
            z_mean_start=float(z_mean_trace[0]) if z_mean_trace.size else float("nan"),
            z_mean_end=float(z_mean_trace[-1]) if z_mean_trace.size else float("nan"),
            band=band,
        )
        out = os.path.join(OUT, f"z_only_summary_seed{args.seed}_{args.regime}.json")
        _atomic_json(out, summary)
        rstride = max(1, int(np.ceil(rate.size / 4000)))
        FCXR._write_npz(os.path.join(run_dir, "zonly_traces.npz"),
                        rate_dt_ms=np.asarray([DT_LC * rstride], np.float32), rate_E=rate[::rstride].astype(np.float32),
                        DZ_t_ms=DZ[:, 0].astype(np.float32), DZ=dz.astype(np.float32),
                        z_mean=z_mean_trace[::max(1, int(np.ceil(z_mean_trace.size / 4000)))].astype(np.float32))
        _sentinel(run_dir, "DONE.json", phase="zonly", seed=args.seed, regime=args.regime, label=lc["label"], out=out)
        FCXR._resource_log(run_dir, "zonly_done", dict(wall_s=wall, label=lc["label"], bounded=bounded, **num))
        print(f"[zonly] seed{args.seed} {args.regime}: label={lc['label']} entered_sustained={entered_sustained} "
              f"pre_interictal={pre_interictal_ms:.0f}ms end_rate={end_rate:.1f}Hz bounded={bounded} "
              f"D_Z {summary['D_Z_start']:.3f}->{summary['D_Z_end']:.3f} (max_step={summary['D_Z_max_step']:.3f}, "
              f"monotone={summary['D_Z_monotone']}) z_mean {summary['z_mean_start']:.3f}->{summary['z_mean_end']:.3f} "
              f"clip={num['clip_frac_max']:.3g} -> {out} ({wall}s, peak {summary['peak_rss_gb']}GB)", flush=True)


# ----------------------------------------------------------------- E3 X sensor separation
def _region_masks(S):
    """E-cell core / axis-band / off-axis masks (source-anchored), for regional y recruitment order."""
    posE = np.asarray(S["posE"], float)[:S["NE"]]; src = np.asarray(S["src_xy"], float); axis = np.asarray(S["axis_unit"], float)
    core = np.linalg.norm(posE - src, axis=1) <= PP.CORE_R
    rel = posE - src; along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    axis_band = (perp <= PP.CORE_R) & (~core)
    return core, axis_band, (~core & ~axis_band)


def _sensor_run(S, p_i, D, seed, T, tau_y):
    """Frozen-Z run at coordinate D with the persistence sensor ACTIVE but the relay NEUTRAL (x_min=1 ->
    x stays 1 -> connectivity unchanged, byte-identical firing to no-relay). Records only the y_j sensor
    (peak trace + O(N_snap) regional means) -> is the sensor above baseline at the dense operating point?"""
    cfg = _slowoff_cfg()
    cfg.update(use_x=True, x_min=1.0, tau_y=float(tau_y), tau_x=1e9, K_y=SENSOR_K_Y, y_gate=0.0, hill_n=SENSOR_HILL_N)
    cfg["z_frozen_E"] = frozen_z_field(p_i, D)                 # use_z stays False -> a frozen field is APPLIED
    snap = {int(round(t / DT_LC)): f"t{t}" for t in range(0, int(T), int(D_Z_SNAP_MS))}
    res, slow = _lc_run(S, cfg, T, seed=seed, snapshot_steps=snap)
    num = _numerical(S, res, slow, DT_LC)
    rate = np.asarray(res["rate_E"], float)
    end_rate = float(np.mean(rate[-max(1, int(round(500.0 / DT_LC))):]))
    ymax = np.asarray(slow.trace_y_max, float)
    core, axm, offm = _region_masks(S)
    snaps = sorted(slow.snapshots.values(), key=lambda s: s["step"])
    tms = np.array([s["step"] * DT_LC for s in snaps], float)
    yreg = lambda m: np.array([float(s["y_E"][m].mean()) for s in snaps], float)
    yc, ya, yo = yreg(core), yreg(axm), yreg(offm)
    del res
    gc.collect()
    return dict(numerical=num, end_rate=end_rate, ymax=ymax, tms=tms, y_core=yc, y_axis=ya, y_off=yo,
                y_max_peak=float(ymax.max()) if ymax.size else 0.0,
                y_max_q999=float(np.percentile(ymax, 99.9)) if ymax.size else 0.0)


def cmd_sensor(args):
    """E3: does the persistence sensor y_j separate the interictal workpoint (frozen D=0) from the dense
    region (frozen D=D_dense)? x_min=1 keeps X neutral -> pure sensor calibration. y_gate = baseline y q99.9."""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        run_dir = os.path.join(OUT, "runs", FCXR._run_id(f"sensor_seed{args.seed}_ty{int(args.tau_y)}_T{int(args.T)}"))
        swap_base = _launch_baseline(run_dir, f"sensor --seed {args.seed} --tau-y {args.tau_y} --T {args.T}")
        _sentinel(run_dir, "RUNNING.json", phase="sensor", seed=args.seed, tau_y=args.tau_y, T=args.T)
        state, info = _resource_state(swap_base)
        if state == "hard":
            _sentinel(run_dir, "ABORTED.json", reason="resource hard-stop before start", **info)
            raise SystemExit(f"[sensor] resource hard-stop before start: {info}")
        print(f"[sensor] seed={args.seed} tau_y={args.tau_y} D_dense={D_DENSE} T={args.T}; {info}", flush=True)
        S = PP.build_substrate(args.seed)
        pk = load_onset_depletion_pi(SNAP_ZA_FMT.format(seed=args.seed))
        assert_field_substrate_aligned(pk, S)
        p_i = pk["p_i"]
        t0 = time.time()
        r0 = _sensor_run(S, p_i, 0.0, args.seed, args.T, args.tau_y)          # interictal workpoint
        rd = _sensor_run(S, p_i, D_DENSE, args.seed, args.T, args.tau_y)      # metastable dense region
        y_gate = r0["y_max_q999"]                                             # baseline y q99.9
        dense_occ = float(np.mean(rd["ymax"] > y_gate)) if rd["ymax"].size else 0.0
        base_occ = float(np.mean(r0["ymax"] > y_gate)) if r0["ymax"].size else 0.0
        # recruitment order at the dense operating point: first snapshot each region's mean y crosses half the
        # dense core peak (core should cross first, then axis, then off) -> interpretable core->axis->off wave.
        thr = 0.5 * float(np.max(rd["y_core"])) if rd["y_core"].size else 0.0

        def _cross(y):
            idx = next((i for i, v in enumerate(y) if v > thr), None)
            return float(rd["tms"][idx]) if idx is not None else None
        tc, ta, to = _cross(rd["y_core"]), _cross(rd["y_axis"]), _cross(rd["y_off"])
        order_ok = bool(tc is not None and (ta is None or tc <= ta) and (ta is None or to is None or ta <= to))
        # Principled separation: baseline almost never crosses its own q99.9 gate; the dense state crosses it
        # MUCH more often and reaches higher. (A hard dense_occ>0.5 is wrong for a metastable dense region that
        # is only intermittently active -> y crosses during dense bursts, not continuously.)
        occ_ratio = dense_occ / base_occ if base_occ > 0 else float("inf")
        separated = bool(base_occ < 0.02 and dense_occ > max(0.05, 5.0 * base_occ) and rd["y_max_peak"] > 1.3 * y_gate)
        wall = round(time.time() - t0, 1)
        summary = dict(
            seed=args.seed, tau_y=args.tau_y, D_dense=D_DENSE, T=args.T, dt=DT_LC, wall_s=wall,
            peak_rss_gb=round(_self_rss_gb(), 2), y_gate_q999=y_gate,
            dense_occ_above_gate=dense_occ, base_occ_above_gate=base_occ, occ_ratio=occ_ratio,
            interictal_y_peak=r0["y_max_peak"], dense_y_peak=rd["y_max_peak"],
            interictal_end_rate=r0["end_rate"], dense_end_rate=rd["end_rate"],
            interictal_numerical=r0["numerical"], dense_numerical=rd["numerical"],
            recruit_cross_ms=dict(core=tc, axis=ta, off=to), recruit_order_core_first=order_ok,
            separated=separated,
        )
        out = os.path.join(OUT, f"sensor_separation_seed{args.seed}_ty{int(args.tau_y)}.json")
        _atomic_json(out, summary)
        FCXR._write_npz(os.path.join(run_dir, "sensor_traces.npz"),
                        tms=rd["tms"].astype(np.float32), dense_y_core=rd["y_core"].astype(np.float32),
                        dense_y_axis=rd["y_axis"].astype(np.float32), dense_y_off=rd["y_off"].astype(np.float32),
                        base_y_core=r0["y_core"].astype(np.float32))
        _sentinel(run_dir, "DONE.json", phase="sensor", seed=args.seed, separated=separated, out=out)
        FCXR._resource_log(run_dir, "sensor_done", dict(wall_s=wall, separated=separated))
        print(f"[sensor] seed{args.seed} ty{args.tau_y}: y_gate={y_gate:.2f} dense_occ_above={dense_occ:.3f} "
              f"base_occ_above={base_occ:.3f} y_peak base={r0['y_max_peak']:.1f} dense={rd['y_max_peak']:.1f} "
              f"recruit(core,axis,off)={tc},{ta},{to} order_ok={order_ok} SEPARATED={separated} "
              f"-> {out} ({wall}s)", flush=True)


# ----------------------------------------------------------------- smoke + dry-run
def cmd_smoke(args):
    """One short continuous slow-off run: plumbing + timing (ms/1k-steps) + peak RSS. NOT a contract."""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        run_dir = os.path.join(OUT, "runs", FCXR._run_id("smoke"))
        swap_base = _launch_baseline(run_dir, f"smoke --seed {args.seed} --T {args.T}")
        print(f"[smoke] build seed={args.seed} T={args.T} dt={DT_LC}", flush=True)
        S = PP.build_substrate(args.seed)
        t0 = time.time()
        res, slow = _lc_run(S, _slowoff_cfg(), args.T, seed=args.seed)
        num = _numerical(S, res, slow, DT_LC)
        wall = round(time.time() - t0, 2)
        steps = int(round(args.T / DT_LC))
        rss = round(_self_rss_gb(), 2)
        del res
        gc.collect()
        _sentinel(run_dir, "DONE.json", phase="smoke", wall_s=wall, **num)
        print(f"[smoke] finite={num['finite']} unsafe={num['numerical_unsafe']} "
              f"clip={num['clip_frac_max']:.3g} tau_eff_min={num['tau_eff_min_ms']:.3f}ms "
              f"{wall}s for {steps} steps ({wall / steps * 1e6:.1f} us/step, {wall / (steps / 1000):.2f} s/1k-steps) "
              f"peak_rss={rss}GB", flush=True)


def cmd_dryrun(args):
    """Report the plan (task count / T / workers / est raster memory) WITHOUT running any simulation (§14.1)."""
    raster_gb = _raster_gb_lc(args.T)
    plan = _strict_plan_workers(args.T, args.workers)
    avail_gb, swap_gb = FCXR._meminfo()
    print(f"[dry-run] T={args.T}ms dt={DT_LC} -> {int(args.T / DT_LC)} steps/run; "
          f"raster ~{raster_gb:.1f}GB/run + substrate ~6.8GB", flush=True)
    print(f"[dry-run] workers requested={args.workers} -> planned={plan['workers']} "
          f"(lc_hard_cap={plan['lc_hard_cap']}, slots={plan['slots']}, other_40k={plan['other_40k_tasks']}); "
          f"MemAvailable={avail_gb:.1f}GB swap_used={swap_gb * 1024.0:.0f}MB", flush=True)
    if plan["workers"] == 0:
        print("[dry-run] planned workers == 0 -> submission would be BLOCKED (do not force max(1,.))", flush=True)


# ----------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="FCXR-LC1 — dynamic slow-feedback lifecycle closure (dt=0.05).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry-run"); d.add_argument("--seed", type=int, default=1)
    d.add_argument("--T", type=float, default=24000.0); d.add_argument("--workers", type=int, default=1)
    for name, defT in (("smoke", 2000.0), ("baseline", 24000.0)):  # --confirm-run AFTER the subcommand (§14.2 usage)
        pr = sub.add_parser(name)
        pr.add_argument("--seed", type=int, default=1)
        pr.add_argument("--T", type=float, default=defT)
        pr.add_argument("--confirm-run", action="store_true", help="required to run any simulation")
    z = sub.add_parser("zonly")
    z.add_argument("--seed", type=int, default=1); z.add_argument("--T", type=float, default=24000.0)
    z.add_argument("--regime", choices=["q75", "q50"], default="q75")
    z.add_argument("--confirm-run", action="store_true", help="required to run any simulation")
    sen = sub.add_parser("sensor")
    sen.add_argument("--seed", type=int, default=1); sen.add_argument("--T", type=float, default=8000.0)
    sen.add_argument("--tau-y", dest="tau_y", type=float, default=SENSOR_TAU_Y)
    sen.add_argument("--confirm-run", action="store_true", help="required to run any simulation")
    args = ap.parse_args(argv)
    if args.cmd == "dry-run":                                  # dry-run performs NO simulation -> no gate
        return cmd_dryrun(args)
    if not getattr(args, "confirm_run", False):
        raise SystemExit("REFUSING: simulations require --confirm-run")
    {"smoke": cmd_smoke, "baseline": cmd_baseline, "zonly": cmd_zonly, "sensor": cmd_sensor}[args.cmd](args)


if __name__ == "__main__":
    main()
