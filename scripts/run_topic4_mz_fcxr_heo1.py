"""FCXR-HEO1 runner — high-energy oscillatory branch acquisition.

Nothing runs on import; every simulation requires --confirm-run. Fixed dt=0.05, accepted FCXR-RC1
arm-C substrate (external additive FF + recurrent conductance + recurrent-only smooth saturation,
g_sat=21.6) on the locked E1146 / L=20 / N=40000 substrate. The new OFF-by-default cooperative
recurrent-conductance gate (mz_slow_vars.cooperative_u_tilde) boosts the RAW recurrent conductance in
a mid-activity band before the tanh saturation. Frozen slow state z_i(D)=clip(1-D*p_i,0,1) reuses the
Stage-D locked onset-depletion field. Virtual-SEEG = LFPRecorder on the real E1146 registered montage.

Modes:
  smoke     one short L=20 cell: A_c=0 parity vs coop-off, A_c>0 finite, LFP/histogram plumbing,
            montage/SCL wiring, gErec_raw quantile probe, per-step timing.
  baseline  F0 slow-off (A_c=0, D=0, seed1, T=8000): baseline_spectral_contract + rec_hist + lfp.
Design lock: docs/superpowers/plans/2026-07-24-topic4-mz-fcxr-heo1.md
Outputs: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/high_energy_oscillatory_branch/
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-fcxr-heo1")

import argparse
import dataclasses
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP          # noqa: E402  build_substrate, R_KICK
import run_topic4_mz_slowvars as OLD    # noqa: E402  build_core_masks
import run_topic4_mz_fcxr as FCXR       # noqa: E402  _fc_cfg + io/resource/flock/bless scaffolding
from kick_probe import simulate_kick    # noqa: E402
from lfp import LFPRecorder             # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig, gerec_baseline_quantiles  # noqa: E402
from src.topic4_mz_fcxr_dynamics import (  # noqa: E402
    load_onset_depletion_pi, assert_field_substrate_aligned, frozen_z_field,
    rolling_rate_upper, workpoint_metrics, classify_run_workpoint,
)
from src.topic4_mz_fcxr_heo1 import build_baseline_reference, classify_heo  # noqa: E402

import multiprocessing as mp  # noqa: E402

_CTX = {}   # fork-shared context (substrate + baseline ref); set before the pool (COW inheritance)

# ---- locked HEO1 constants (design lock) ----
G_SAT = 21.6
DT = 0.05
T_KICK_MS = 120.0
KICK3, KICK12 = 3.0, 12.0                       # high1 / high2 basin-probe kick amplitudes
ES_THRESH_HZ = 250.0                            # operational-runaway early stop: >250 Hz ...
ES_DUR_MS = 100.0                               #   ... sustained 100 ms -> OPERATIONAL_RUNAWAY
GATE_QUANTILES = [0.999, 0.9999]                # u_c calibration (Q99.9 / Q99.99 of baseline gErec_raw)
A_GRID = [1.0, 2.0, 4.0, 8.0]
D_SCREEN = [0.13, 0.15]
# fixed-edge gErec_raw histogram: the slow-off tail exceeds g_sat (smoke max ~25), so span [0,40]
# (~2x g_sat, > observed max) at 0.005 resolution + a trailing inf overflow bin. F0 must show
# Q99.9/Q99.99 not in overflow (else widen G_HIST_MAX + rerun F0).
G_HIST_MAX = 40.0
GEREC_EDGES = np.concatenate([np.linspace(0.0, G_HIST_MAX, 8001), [np.inf]])   # width 0.005, +overflow
OUT = os.path.join(FCXR.OUT_ROOT, "high_energy_oscillatory_branch")
SNAP_FMT = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility",
                        "snapshots", "zA_q75_tz5000", "seed_{seed}.npz")


# ----------------------------------------------------------------- substrate + montage + field
def _shaft(name):
    m = re.match(r"[A-Za-z]+", str(name))
    return m.group(0) if m else str(name)


def _build_and_align(seed):
    """RC1 substrate + seed-matched locked onset-depletion field p_i (STOP if mis-registered, §6)."""
    S = PP.build_substrate(int(seed))
    pk = load_onset_depletion_pi(SNAP_FMT.format(seed=int(seed)))
    assert_field_substrate_aligned(pk, S)
    return S, pk["p_i"]


def _montage(S):
    """Real E1146 registered virtual-SEEG montage: (contacts Nx2, names, scl_mask)."""
    mont = S["reg"]["montage_sheet"]
    contacts = np.asarray(mont.contacts, float)
    names = list(mont.names)
    scl = np.array([_shaft(n) == "SCL" for n in names], bool)
    return contacts, names, scl


def _heo_cfg(A_c, u_c, D, p_i, *, record_hist=False, edges=None):
    """Arm-C FCXR config + cooperative gate + optional frozen field. fail_on_clip=False (record clip)."""
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False, rec_sat_g=G_SAT)
    cfg.update(coop_A=float(A_c), coop_uc=float(u_c), coop_Kc=0.25 * float(u_c), coop_n=4,
               record_clip_identity=True, record_gerec_hist=bool(record_hist), gerec_hist_edges=edges)
    if D is not None and float(D) != 0.0:
        cfg["z_frozen_E"] = frozen_z_field(p_i, float(D))       # slow state frozen at z_i(D)
    return cfg


def _heo_run(S, cfg, T_ms, *, kick_boost, t_kick, seed, lfp_sites, early_stop=True):
    p = dataclasses.replace(S["p"], T=float(T_ms), dt=DT)
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=lfp_sites)
    S["net"]["rng"] = np.random.default_rng(int(seed))
    res = simulate_kick(p, S["net"], float(kick_boost), slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=float(t_kick), V_th_per_neuron=S["vth"],
                        lfp_recorder=rec, early_stop_runaway=bool(early_stop),
                        es_thresh_hz=ES_THRESH_HZ, es_dur_ms=ES_DUR_MS)
    return res, slow


def _numerical(S, res, slow, dt=DT):
    """Numerical-safety row (Stage-D _numerical_dt + runaway flag). numerical_unsafe drops HEO (Gate E)."""
    taur = np.asarray(slow.trace_tau_eff_ratio_min, float)
    clipf = np.asarray(slow.trace_conductance_clip_frac, float)
    finite = bool(np.all(np.isfinite(res["rate_E"])))
    tau_eff_min_ms = float(S["p"].tau_m_E * taur.min()) if taur.size else float("nan")
    max_clip = float(clipf.max()) if clipf.size else 0.0
    runaway_ms = res.get("runaway_early_stop_ms")
    unsafe = bool((not finite) or max_clip > 0.0
                  or (np.isfinite(tau_eff_min_ms) and tau_eff_min_ms < 2.0 * dt))
    return dict(finite=finite, tau_eff_min_ms=tau_eff_min_ms, clip_frac_max=max_clip,
                numerical_unsafe=unsafe, runaway_early_stop_ms=runaway_ms)


# ----------------------------------------------------------------- sentinels
def _sentinel(run_dir, name, payload):
    FCXR._write_json(os.path.join(run_dir, name), payload)


# ----------------------------------------------------------------- smoke (F1)
def cmd_smoke(a):
    t0 = time.time()
    S, p_i = _build_and_align(a.seed)
    contacts, names, scl = _montage(S)
    build_s = time.time() - t0
    shafts = sorted({_shaft(n) for n in names})
    print(f"[smoke] substrate seed={a.seed} N={S['N']} NE={S['NE']} build={build_s:.1f}s")
    print(f"[smoke] montage: {len(names)} contacts, shafts={shafts}, SCL={int(scl.sum())} "
          f"names={names}")

    T = float(a.t_ms)
    # A_c=0 slow-off (RC1) with LFP + histogram plumbing
    t1 = time.time()
    cfg0 = _heo_cfg(0.0, 0.0, None, p_i, record_hist=True, edges=GEREC_EDGES)
    res0, slow0 = _heo_run(S, cfg0, T, kick_boost=0.0, t_kick=1e9, seed=a.seed, lfp_sites=contacts,
                           early_stop=False)
    wall0 = time.time() - t1
    nsteps = len(res0["rate_E"])
    per_step_ms = wall0 / nsteps * 1e3
    q = gerec_baseline_quantiles(slow0.gerec_hist_overall, GEREC_EDGES, [0.5, 0.99, 0.999, 0.9999])
    print(f"[smoke] A_c=0: wall={wall0:.1f}s nsteps={nsteps} per_step={per_step_ms:.3f}ms "
          f"lfp_shape={None if res0['lfp_trace'] is None else res0['lfp_trace'].shape}")
    print(f"[smoke] gErec_raw max={float(slow0.max_raw_gErec.max()):.4f} "
          f"Q50={q[0.5]:.4f} Q99={q[0.99]:.4f} Q99.9={q[0.999]:.4f} Q99.99={q[0.9999]:.4f}")
    print(f"[smoke] extrapolated 8000ms wall ~= {per_step_ms * (8000/DT) / 60000:.1f} min")

    # A_c>0 finite (gate engages if u_c below the quantile probe)
    u_c = max(q[0.999], 1e-3)
    cfg1 = _heo_cfg(4.0, u_c, None, p_i, record_hist=False)
    res1, slow1 = _heo_run(S, cfg1, T, kick_boost=0.0, t_kick=1e9, seed=a.seed, lfp_sites=contacts,
                           early_stop=True)
    fin1 = bool(np.all(np.isfinite(res1["rate_E"])))
    eng = float(np.mean(slow1.trace_coop_engaged_frac)) if slow1.trace_coop_engaged_frac else 0.0
    print(f"[smoke] A_c=4 u_c={u_c:.4f}: finite={fin1} mean_engaged_frac={eng:.4g} "
          f"runaway_ms={res1.get('runaway_early_stop_ms')}")
    print(f"[smoke] TOTAL {time.time() - t0:.1f}s  -> wiring OK")


# ----------------------------------------------------------------- baseline (F0)
def cmd_baseline(a):
    if not a.confirm_run:
        raise SystemExit("baseline: pass --confirm-run to launch the F0 8000ms sim")
    FCXR._assert_engine_blessed()
    run_dir = os.path.join(OUT)
    os.makedirs(run_dir, exist_ok=True)
    with FCXR._launcher_lock():
        plan = FCXR._plan_workers(a.t_ms, 1)
        FCXR._resource_log(run_dir, "baseline_start", plan)
        _sentinel(run_dir, "RUNNING.json", dict(mode="baseline", seed=a.seed, t_ms=a.t_ms, dt=DT,
                  pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(), plan=plan))
        with open(os.path.join(run_dir, "launcher.pid"), "w") as f:
            f.write(str(os.getpid()))
        try:
            S, p_i = _build_and_align(a.seed)
            contacts, names, scl = _montage(S)
            _sentinel(run_dir, "launch_baseline.json", dict(
                seed=a.seed, t_ms=a.t_ms, dt=DT, n_contacts=len(names),
                shafts=sorted({_shaft(n) for n in names}), scl_n=int(scl.sum()), names=names,
                raster_gb=round(FCXR._raster_gb(a.t_ms), 2)))
            t1 = time.time()
            cfg = _heo_cfg(0.0, 0.0, None, p_i, record_hist=True, edges=GEREC_EDGES)
            res, slow = _heo_run(S, cfg, a.t_ms, kick_boost=0.0, t_kick=1e9, seed=a.seed,
                                 lfp_sites=contacts, early_stop=False)
            wall = time.time() - t1
            FCXR._resource_log(run_dir, "baseline_sim_done", dict(wall_s=round(wall, 1)))

            rate = np.asarray(res["rate_E"], float)
            num = _numerical(S, res, slow)
            roll_hi = float(rolling_rate_upper(rate, DT))
            wm = workpoint_metrics(rate, DT, roll_hi)
            wp_label = classify_run_workpoint(dict(numerical_unsafe=num["numerical_unsafe"], **wm))
            qs = [0.5, 0.9, 0.99, 0.999, 0.9999]
            gq = gerec_baseline_quantiles(slow.gerec_hist_overall, GEREC_EDGES, qs)
            u_c = {str(gate): float(gerec_baseline_quantiles(slow.gerec_hist_overall, GEREC_EDGES, [gate])[gate])
                   for gate in GATE_QUANTILES}
            in_range = bool(np.all(np.isfinite(list(u_c.values()))))

            contract = dict(
                seed=a.seed, t_ms=a.t_ms, dt=DT, wall_s=round(wall, 1), n_steps=len(rate),
                numerical=num, rate_roll_hi=roll_hi, workpoint_metrics=wm, workpoint_label=wp_label,
                gErec_raw_quantiles={str(k): v for k, v in gq.items()},
                gErec_raw_max=float(slow.max_raw_gErec.max()),
                u_c=u_c, u_c_in_range=in_range,
                n_contacts=len(names), scl_n=int(scl.sum()), shafts=sorted({_shaft(n) for n in names}),
                end_rate_hz=float(rate[-1]), mean_rate_hz=float(rate.mean()),
                baseline_preserved=bool((not num["numerical_unsafe"]) and in_range),
            )
            FCXR._write_json(os.path.join(run_dir, f"baseline_spectral_contract_seed{a.seed}.json"), contract)
            FCXR._write_npz(os.path.join(run_dir, f"baseline_rec_hist_seed{a.seed}.npz"),
                            edges=GEREC_EDGES, overall=slow.gerec_hist_overall,
                            core=slow.gerec_hist_core, surround=slow.gerec_hist_surround)
            FCXR._write_npz(os.path.join(run_dir, f"baseline_lfp_seed{a.seed}.npz"),
                            lfp_trace=np.asarray(res["lfp_trace"], np.float32), rate_E=rate.astype(np.float32),
                            contacts=contacts, names=np.array(names, object), scl_mask=scl, dt=DT)
            FCXR._resource_log(run_dir, "baseline_saved")
            _sentinel(run_dir, "DONE.json", dict(mode="baseline", seed=a.seed, wall_s=round(wall, 1),
                      workpoint_label=wp_label, baseline_preserved=contract["baseline_preserved"],
                      u_c=u_c, u_c_in_range=in_range, finished=datetime.now(timezone.utc).isoformat()))
            print(f"[baseline] seed{a.seed} wall={wall:.1f}s label={wp_label} "
                  f"preserved={contract['baseline_preserved']} u_c={u_c} in_range={in_range}")
            if os.path.exists(os.path.join(run_dir, "RUNNING.json")):
                os.remove(os.path.join(run_dir, "RUNNING.json"))
        except Exception as exc:  # loud failure -> FAILED sentinel (never silent)
            _sentinel(run_dir, "FAILED.json", dict(mode="baseline", seed=a.seed, error=repr(exc),
                      failed=datetime.now(timezone.utc).isoformat()))
            raise


# ----------------------------------------------------------------- screen (F2) / confirm (F3)
def _load_baseline(seed):
    """F0 outputs -> (HEO baseline reference, u_c per gate_quantile, rate_roll_hi, scl_mask, contacts)."""
    d = np.load(os.path.join(OUT, f"baseline_lfp_seed{seed}.npz"), allow_pickle=True)
    ref = build_baseline_reference(np.asarray(d["lfp_trace"], float), np.asarray(d["rate_E"], float), DT)
    contract = json.load(open(os.path.join(OUT, f"baseline_spectral_contract_seed{seed}.json")))
    u_c = {float(k): float(v) for k, v in contract["u_c"].items()}
    return (ref, u_c, float(contract["rate_roll_hi"]),
            np.asarray(d["scl_mask"], bool), np.asarray(d["contacts"], float))


def _kick_for_ic(ic):
    if ic == "nokick":
        return 0.0, 1e9
    if ic == "kick3":
        return KICK3, T_KICK_MS
    if ic == "kick12":
        return KICK12, T_KICK_MS
    raise ValueError(f"unknown ic {ic!r}")


def _region_rates(S, res):
    """Core vs surround E-cell mean firing rate (Hz) from the raster (platform coverage is separate)."""
    E = np.asarray(res["E_spk_bool"])
    core = OLD.build_core_masks(S)
    to_hz = 1000.0 / DT
    return dict(core_rate_hz=float(E[:, core].mean() * to_hz) if core.any() else float("nan"),
                surround_rate_hz=float(E[:, ~core].mean() * to_hz) if (~core).any() else float("nan"),
                core_frac=float(core.mean()))


def _cell_row(gate_q, A_c, D, ic, u_c, res, slow, num, verdict, regions):
    rate = np.asarray(res["rate_E"], float)
    eng = np.asarray(slow.trace_coop_engaged_frac, float)
    row = dict(gate_quantile=gate_q, A_c=A_c, D=D, ic=ic, u_c=u_c, K_c=0.25 * u_c,
               end_rate_hz=float(rate[-1]), mean_rate_hz=float(rate.mean()), max_rate_hz=float(rate.max()),
               engaged_frac_mean=float(eng.mean()) if eng.size else 0.0,
               engaged_frac_max=float(eng.max()) if eng.size else 0.0,
               gErec_raw_max=float(slow.max_raw_gErec.max()),
               HEO_BRANCH=verdict["HEO_BRANCH"], plateau=verdict["plateau"], oscillation=verdict["oscillation"],
               max_platform_contacts=verdict["max_platform_contacts"],
               max_platform_scl=verdict["max_platform_scl"],
               platform_window_frac=verdict["platform_window_frac"])
    row.update({k: num[k] for k in ("finite", "tau_eff_min_ms", "clip_frac_max",
                                    "numerical_unsafe", "runaway_early_stop_ms")})
    row.update({k: verdict[k] for k in ("gate_A_plateau", "gate_B_broadband", "gate_C_platform",
                                        "gate_D_oscillation", "gate_E_numerical")})
    row.update(regions)
    return row


def _workpoint_cell(task):
    gate_q, A_c = task
    S, p_i, contacts = _CTX["S"], _CTX["p_i"], _CTX["contacts"]
    roll_hi, u_c = _CTX["roll_hi"], _CTX["u_c"][gate_q]
    label = f"wp_gq{gate_q:g}_A{A_c:g}"
    t0 = time.time()
    cfg = _heo_cfg(A_c, u_c, None, p_i)                    # D=0 (no frozen field) workpoint probe
    res, slow = _heo_run(S, cfg, 4000.0, kick_boost=0.0, t_kick=1e9, seed=1,
                         lfp_sites=contacts, early_stop=True)
    num = _numerical(S, res, slow)
    rate = np.asarray(res["rate_E"], float)
    wm = workpoint_metrics(rate, DT, roll_hi)
    wp = classify_run_workpoint(dict(numerical_unsafe=num["numerical_unsafe"], **wm))
    eng = np.asarray(slow.trace_coop_engaged_frac, float)
    preserved = bool((not num["numerical_unsafe"]) and wp == "INTERICTAL_WORKPOINT"
                     and num["runaway_early_stop_ms"] is None)
    row = dict(gate_quantile=gate_q, A_c=A_c, u_c=u_c, label=label, workpoint_label=wp,
               preserved=preserved, mean_rate_hz=float(rate.mean()), end_rate_hz=float(rate[-1]),
               max_rate_hz=float(rate.max()), engaged_frac_mean=float(eng.mean()) if eng.size else 0.0,
               wall_s=round(time.time() - t0, 1))
    row.update(num); row.update(wm)
    FCXR._write_json(os.path.join(OUT, "screen_cells", f"{label}.json"), row)
    return row


def _screen_cell(task):
    gate_q, A_c, D, ic = task
    S, p_i, contacts, scl, ref = _CTX["S"], _CTX["p_i"], _CTX["contacts"], _CTX["scl"], _CTX["ref"]
    u_c = _CTX["u_c"][gate_q]
    kb, tk = _kick_for_ic(ic)
    label = f"gq{gate_q:g}_A{A_c:g}_D{D:g}_{ic}"
    t0 = time.time()
    cfg = _heo_cfg(A_c, u_c, D, p_i)
    res, slow = _heo_run(S, cfg, 4000.0, kick_boost=kb, t_kick=tk, seed=1, lfp_sites=contacts, early_stop=True)
    num = _numerical(S, res, slow)
    verdict = classify_heo(np.asarray(res["lfp_trace"], float), np.asarray(res["rate_E"], float),
                           DT, scl, ref, num)
    regions = _region_rates(S, res)
    row = _cell_row(gate_q, A_c, D, ic, u_c, res, slow, num, verdict, regions)
    row.update(label=label, wall_s=round(time.time() - t0, 1))
    FCXR._write_json(os.path.join(OUT, "screen_cells", f"{label}.json"), row)
    FCXR._write_npz(os.path.join(OUT, "screen_cells", f"{label}_trace.npz"),
                    rate_E=np.asarray(res["rate_E"], np.float32),
                    lfp_trace=np.asarray(res["lfp_trace"], np.float32),
                    engaged_frac=np.asarray(slow.trace_coop_engaged_frac, np.float32))
    return row


def _run_tasks(fn, tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    ctx = mp.get_context("fork")            # COW-share the ~7 GB substrate + baseline ref via _CTX
    with ctx.Pool(workers) as pool:
        return pool.map(fn, tasks)


def _pick_workers(a):
    plan = FCXR._plan_workers(4000.0, a.workers)
    workers = min(plan["workers"], 2)       # task §7: T<=8000 -> at most 2 workers
    return workers, plan


def _select_candidate(rows):
    """Lexicographic minimal-mechanism-deviation HEO candidate (design lock F3)."""
    heo = [r for r in rows if r["HEO_BRANCH"]]
    if not heo:
        return None
    ic_rank = {"nokick": 0, "kick3": 1, "kick12": 2}
    heo.sort(key=lambda r: (r["A_c"], 0 if r["gate_quantile"] == 0.999 else 1, ic_rank.get(r["ic"], 9), r["D"]))
    return heo[0]


def cmd_screen(a):
    if not a.confirm_run:
        raise SystemExit("screen: pass --confirm-run to launch the F2 grid")
    FCXR._assert_engine_blessed()
    os.makedirs(os.path.join(OUT, "screen_cells"), exist_ok=True)
    with FCXR._launcher_lock():
        ref, u_c, roll_hi, scl0, contacts0 = _load_baseline(1)
        S, p_i = _build_and_align(1)
        contacts, names, scl = _montage(S)
        if not (np.array_equal(scl, scl0) and contacts.shape == contacts0.shape):
            raise SystemExit("screen: montage mismatch vs F0 baseline (SCL/contacts) -> STOP")
        _CTX.update(S=S, p_i=p_i, contacts=contacts, scl=scl, ref=ref, u_c=u_c, roll_hi=roll_hi)
        workers, plan = _pick_workers(a)
        FCXR._resource_log(OUT, "screen_start", dict(plan=plan, workers=workers))
        _sentinel(OUT, "RUNNING.json", dict(mode="screen", pid=os.getpid(), workers=workers,
                  started=datetime.now(timezone.utc).isoformat(), plan=plan))
        with open(os.path.join(OUT, "launcher.pid"), "w") as f:
            f.write(str(os.getpid()))
        try:
            # phase 1 — workpoint gate (D=0/no-kick) prunes arms that break the baseline
            wp_tasks = [(gq, A) for gq in GATE_QUANTILES for A in A_GRID]
            wp_rows = _run_tasks(_workpoint_cell, wp_tasks, workers)
            survivors = [(r["gate_quantile"], r["A_c"]) for r in wp_rows if r["preserved"]]
            FCXR._write_json(os.path.join(OUT, "workpoint_gate.json"),
                             dict(rows=wp_rows, survivors=survivors, n_survivors=len(survivors)))
            FCXR._resource_log(OUT, "workpoint_done", dict(n_survivors=len(survivors)))
            print(f"[screen] workpoint survivors {len(survivors)}/{len(wp_tasks)}: {survivors}", flush=True)
            # phase 2 — screen surviving arms over frozen D x IC
            screen_tasks = [(gq, A, D, ic) for (gq, A) in survivors for D in D_SCREEN
                            for ic in ("nokick", "kick3", "kick12")]
            screen_rows = _run_tasks(_screen_cell, screen_tasks, workers)
            candidate = _select_candidate(screen_rows)
            n_heo = sum(1 for r in screen_rows if r["HEO_BRANCH"])
            FCXR._write_json(os.path.join(OUT, "branch_map.json"),
                             dict(workpoint=wp_rows, survivors=survivors, cells=screen_rows,
                                  n_cells=len(screen_rows), n_heo=n_heo))
            FCXR._write_json(os.path.join(OUT, "candidate_verdict.json"),
                             dict(candidate=candidate, n_heo=n_heo, n_cells=len(screen_rows),
                                  verdict="HEO_CANDIDATE" if candidate else "NO_GO_no_HEO_in_screen"))
            _sentinel(OUT, "DONE.json", dict(mode="screen", n_cells=len(screen_rows), n_heo=n_heo,
                      candidate=candidate, finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(OUT, "RUNNING.json")):
                os.remove(os.path.join(OUT, "RUNNING.json"))
            print(f"[screen] DONE cells={len(screen_rows)} n_heo={n_heo} candidate={candidate}", flush=True)
        except Exception as exc:
            _sentinel(OUT, "FAILED.json", dict(mode="screen", error=repr(exc),
                      failed=datetime.now(timezone.utc).isoformat()))
            raise


def cmd_confirm(a):
    if not a.confirm_run:
        raise SystemExit("confirm: pass --confirm-run")
    FCXR._assert_engine_blessed()
    conf_dir = os.path.join(OUT, "confirm")
    os.makedirs(conf_dir, exist_ok=True)
    ref, u_c_map, roll_hi, scl0, contacts0 = _load_baseline(1)
    u_c = u_c_map[a.gate_q]
    ics = ["nokick", "kick3", "kick12"] if a.seed == 1 else ["nokick", a.min_ic]
    rows = []
    for coop_on in (True, False):                          # candidate + matched cooperative-OFF control
        S, p_i = _build_and_align(a.seed)
        contacts, names, scl = _montage(S)
        for ic in ics:
            kb, tk = _kick_for_ic(ic)
            A_c = a.A if coop_on else 0.0
            uc = u_c if coop_on else 0.0
            label = f"seed{a.seed}_{'coopON' if coop_on else 'coopOFF'}_gq{a.gate_q:g}_A{A_c:g}_D{a.D:g}_{ic}"
            t0 = time.time()
            cfg = _heo_cfg(A_c, uc, a.D, p_i)
            res, slow = _heo_run(S, cfg, a.t_ms, kick_boost=kb, t_kick=tk, seed=a.seed,
                                 lfp_sites=contacts, early_stop=True)
            num = _numerical(S, res, slow)
            verdict = classify_heo(np.asarray(res["lfp_trace"], float), np.asarray(res["rate_E"], float),
                                   DT, scl, ref, num)
            row = _cell_row(a.gate_q, A_c, a.D, ic, uc, res, slow, num, verdict, _region_rates(S, res))
            row.update(label=label, seed=a.seed, coop_on=coop_on, wall_s=round(time.time() - t0, 1))
            FCXR._write_json(os.path.join(conf_dir, f"{label}.json"), row)
            FCXR._write_npz(os.path.join(conf_dir, f"{label}_trace.npz"),
                            rate_E=np.asarray(res["rate_E"], np.float32),
                            lfp_trace=np.asarray(res["lfp_trace"], np.float32),
                            engaged_frac=np.asarray(slow.trace_coop_engaged_frac, np.float32))
            rows.append(row)
            print(f"[confirm] {label} HEO={verdict['HEO_BRANCH']} wall={row['wall_s']}s", flush=True)
    FCXR._write_json(os.path.join(conf_dir, f"confirm_seed{a.seed}.json"), dict(rows=rows))


def main():
    ap = argparse.ArgumentParser(description="FCXR-HEO1 runner")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sm = sub.add_parser("smoke"); sm.add_argument("--seed", type=int, default=1)
    sm.add_argument("--t-ms", type=float, default=800.0); sm.set_defaults(fn=cmd_smoke)
    bl = sub.add_parser("baseline"); bl.add_argument("--seed", type=int, default=1)
    bl.add_argument("--t-ms", type=float, default=8000.0); bl.add_argument("--confirm-run", action="store_true")
    bl.set_defaults(fn=cmd_baseline)
    sc = sub.add_parser("screen"); sc.add_argument("--workers", type=int, default=2)
    sc.add_argument("--confirm-run", action="store_true"); sc.set_defaults(fn=cmd_screen)
    cf = sub.add_parser("confirm"); cf.add_argument("--seed", type=int, default=1)
    cf.add_argument("--gate-q", dest="gate_q", type=float, required=True)
    cf.add_argument("--A", type=float, required=True); cf.add_argument("--D", type=float, required=True)
    cf.add_argument("--min-ic", dest="min_ic", default="kick3"); cf.add_argument("--t-ms", type=float, default=8000.0)
    cf.add_argument("--confirm-run", action="store_true"); cf.set_defaults(fn=cmd_confirm)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
