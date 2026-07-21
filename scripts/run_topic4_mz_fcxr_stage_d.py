"""FCXR Stage D runner — frozen fast-branch map (D1) + dt=0.05 baseline re-anchor (D0.2) + 3-point pilot.

Nothing runs on import; every simulation requires --confirm-run. Fixed dt=0.05. Accepted FCXR-RC1 base
(arm C: external additive FF + recurrent conductance + recurrent-only smooth saturation, g_sat=21.6).
ALL slow variables frozen: Z held at z_i(D)=clip(1-D*p_i,0,1) (p_i = locked onset-depletion pattern),
M/phi/x OFF. The 120Hz/100ms phenotype early-stop is DISABLED (a bounded high state must not be killed
as operational runaway; only nonfinite / conductance-clip / tau_eff numerical stops remain).

Complements the unsaturated slow-fast-transition line (sharp cliff at D~=0.087): D1 asks whether the RC1
smooth saturation converts the past-transition runaway into a BOUNDED finite-high branch. Engine untouched
(no re-bless); all new code rides the non-blessed mz_slow_vars plugin + this runner.

Commands:
  smoke     one cell end-to-end + timing (validation).
  baseline  arm-C saturated slow-off at dt=0.05 -> baseline_ref.json (classifier anchor).
  pilot     3 D bracketing the transition x {low, high, high2}; rows for classification.
Outputs: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/fast_slow_dynamics/
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-fcxr-stage-d")

import argparse
import dataclasses
import gc
import multiprocessing as mp
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP          # noqa: E402  build_substrate, R_KICK, KICK
import run_topic4_mz_slowvars as OLD    # noqa: E402  build_core_masks, compute_baseline_ref, C.active_fraction
import run_topic4_mz_fcxr as FCXR       # noqa: E402  _fc_cfg + scaffolding (flock / bless / run_id / io / resource)
from kick_probe import simulate_kick    # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_fcxr_dynamics import (  # noqa: E402
    load_onset_depletion_pi, assert_field_substrate_aligned, frozen_z_field,
    classify_run_provisional, classify_run_envelope, envelope_metrics, resolve_high_ic,
    rolling_rate_upper, workpoint_metrics, classify_run_workpoint, WP_THRESHOLDS,
    classify_branch_D, THRESHOLDS,
)
from src.topic4_mz_conductance import oscillation_metrics  # noqa: E402

# ---- locked Stage-D constants ----
G_SAT = 21.6
DT_D = 0.05
D_GRID = [0.0, 0.05, 0.075, 0.085, 0.09, 0.10, 0.125, 0.15]   # scalar D ~= mean depletion; straddles ~0.087
PILOT_D = [0.075, 0.10, 0.125]                                 # below / just above / well above the transition
T_KICK_MS = 120.0
DUR_KICK_MS = 18.0                                             # engine constant kick_probe.DUR_KICK
KICK_HIGH1, KICK_HIGH2 = 3.0, 12.0                           # moderate (RC1 default) + strong (4x) high-IC probes
END_WIN_MS = 500.0                                            # trailing window for end-of-run high state
OUT = os.path.join(FCXR.OUT_ROOT, "fast_slow_dynamics")
SNAP_FMT = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility",
                        "snapshots", "zA_q75_tz5000", "seed_{seed}.npz")


# ----------------------------------------------------------------- substrate + field
def _build_and_align(seed):
    """Build the RC1 substrate for `seed` and load+verify the seed-matched locked onset-depletion field."""
    S = PP.build_substrate(int(seed))
    pk = load_onset_depletion_pi(SNAP_FMT.format(seed=int(seed)))
    assert_field_substrate_aligned(pk, S)                     # STOP if the field is mis-registered (§6)
    return S, pk["p_i"]


# ----------------------------------------------------------------- dt-aware run + numerical safety
def _stage_d_run(S, cfg, T_ms, *, kick_boost, t_kick, seed, dt=DT_D):
    p = dataclasses.replace(S["p"], T=float(T_ms), dt=float(dt))
    slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
    S["net"]["rng"] = np.random.default_rng(int(seed))
    res = simulate_kick(p, S["net"], float(kick_boost), slow=slow,
                        kick_center=list(S["src_xy"]), r_kick=PP.R_KICK, t_kick=float(t_kick),
                        V_th_per_neuron=S["vth"], early_stop_runaway=False)   # DESIGN: early-stop OFF
    return res, slow


def _numerical_dt(S, res, slow, dt):
    taur = np.asarray(slow.trace_tau_eff_ratio_min, float)
    clipf = np.asarray(slow.trace_conductance_clip_frac, float)
    finite = bool(np.all(np.isfinite(res["rate_E"])))
    tau_eff_min_ms = float(S["p"].tau_m_E * taur.min()) if taur.size else float("nan")
    max_clip = float(clipf.max()) if clipf.size else 0.0
    unsafe = bool((not finite) or max_clip > 0.0 or (np.isfinite(tau_eff_min_ms) and tau_eff_min_ms < 2.0 * dt))
    return dict(finite=finite, tau_eff_min_ms=tau_eff_min_ms, clip_frac_max=max_clip, numerical_unsafe=unsafe)


def _region_masks(S):
    """E-cell core / axis-band / off-axis masks for regional participation traces (reviewer P1)."""
    posE = np.asarray(S["posE"], float); src = np.asarray(S["src_xy"], float); axis = np.asarray(S["axis_unit"], float)
    core = np.linalg.norm(posE - src, axis=1) <= PP.CORE_R
    rel = posE - src; along = rel @ axis
    perp = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    axis_band = (perp <= PP.CORE_R) & (~core)
    return core, axis_band, (~core & ~axis_band)


def _branch_row(S, res, slow, bref, dt, analysis_start_ms, *, D, ic, kick_boost, seed, T_ms):
    """Reduce one completed run to the classifier observable row + a few downsampled scalar traces
    (population + core/axis/off-axis participation) so dynamics type (fixed / orbit / transient / event-train)
    can be judged later. NO T x N array is kept."""
    num = _numerical_dt(S, res, slow, dt)
    rate = np.asarray(res["rate_E"], float)
    spk = res["E_spk_bool"]
    af, af_bin = OLD.C.active_fraction(spk, dt, OLD.C.BIN_MS)
    af = np.asarray(af, float); af_bin = float(af_bin)
    BR, BS, Q95 = float(bref["baseline_rate"]), float(bref["sigma_rate"]), float(bref["floor_af"])
    om = oscillation_metrics(rate, dt, analysis_start_ms=analysis_start_ms, baseline_rate=BR, baseline_sigma=BS,
                             active_fraction=af, af_bin_ms=af_bin, baseline_af_q95=Q95,
                             runaway=bool(not num["finite"]))
    env = envelope_metrics(af, af_bin, analysis_start_ms, Q95)
    wm = workpoint_metrics(rate, dt, float(bref["rate_roll_hi"]), analysis_start_ms)   # primary (reviewer P0)
    endn = max(1, int(round(END_WIN_MS / dt)))
    end_rate = float(np.mean(rate[-endn:])) if rate.size else float("nan")
    afn = max(1, int(round(END_WIN_MS / af_bin)))
    af_tail = float(np.mean(af[-afn:])) if af.size else float("nan")
    tail_high_frac = float(np.mean(af[-afn:] > Q95)) if af.size else float("nan")
    a0_bin = max(0, int(round(analysis_start_ms / af_bin)))
    core, axm, offm = _region_masks(S)

    def _reg(mask):
        a, _ = OLD.C.active_fraction(spk[:, mask], dt, OLD.C.BIN_MS)
        return np.asarray(a, float)

    # af/regions kept at NATIVE bin (already coarse ~1ms; a 4s run is ~4000 pts) so af_bin_ms stays correct;
    # only the fine rate_E (dt=0.05) is downsampled, with its matching rate_dt_ms (fixes the plotter time axis).
    rate_stride = max(1, int(np.ceil(rate.size / 2000)))
    traces = dict(af_bin_ms=np.asarray([af_bin], np.float32),
                  rate_dt_ms=np.asarray([dt * rate_stride], np.float32),
                  rate_E=rate[::rate_stride].astype(np.float32),
                  af=af.astype(np.float32), af_core=_reg(core).astype(np.float32),
                  af_axis=_reg(axm).astype(np.float32), af_off=_reg(offm).astype(np.float32))
    row = dict(
        D=float(D), ic=ic, kick_boost=float(kick_boost), seed=int(seed), T_ms=float(T_ms), dt=float(dt),
        analysis_start_ms=float(analysis_start_ms),
        finite=num["finite"], clip_frac_max=num["clip_frac_max"], tau_eff_min_ms=num["tau_eff_min_ms"],
        numerical_unsafe=num["numerical_unsafe"],
        high_duration_ms=float(om["high_duration_ms"]), modulation=float(om["modulation"]),
        oscillatory_candidate=bool(om["oscillatory_candidate"]), tail_rate_band=bool(om["tail_rate_band"]),
        recruitment_pass=bool(om["recruitment_pass"]), spectral_pass=bool(om.get("spectral_pass", False)),
        dominant_hz=float(om["dominant_hz"]), tail_mean_hz=float(om["tail_mean_hz"]),
        end_rate_hz=end_rate, af_tail=af_tail, tail_high_frac=tail_high_frac, af_bin_ms=af_bin,
        n_bins_high=int(np.sum(af[a0_bin:] > Q95)),
        baseline_rate=BR, baseline_sigma=BS, baseline_af_q95=Q95,
        env_high_ms=float(env["env_high_ms"]), env_end_occ=float(env["env_end_occ"]),
        env_occ=float(env["env_occ"]), env_modulation=float(env["env_modulation"]),
        env_window_ms=float(env["env_window_ms"]),
        roll_occ=float(wm["roll_occ"]), roll_end_occ=float(wm["roll_end_occ"]),
        roll_high_ms=float(wm["roll_high_ms"]), roll_modulation=float(wm["roll_modulation"]),
        window_ms=float(wm["window_ms"]), baseline_roll_hi=float(bref["rate_roll_hi"]),
    )
    return row, traces


def run_branch_cell(S, pi, bref, *, D, ic, kick_boost, T_post_ms, seed, dt=DT_D, g_sat=G_SAT):
    """One frozen-Z arm-C saturated cell. ic='low' -> no kick (native low); ic='high' -> short kick at the
    source core then full release (early-stop OFF). Returns (row, slow); slow carries max_raw_gErec for D2."""
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False, rec_sat_g=g_sat)
    cfg["z_frozen_E"] = frozen_z_field(pi, D)
    cfg["record_clip_identity"] = True                        # O(NE) g_raw proxy for D2 (pure side-effect)
    if ic == "high":
        t_kick, kb, a0 = T_KICK_MS, float(kick_boost), T_KICK_MS + DUR_KICK_MS
    elif ic == "low":
        t_kick, kb, a0 = 1e9, 0.0, 0.0
    else:
        raise ValueError(f"ic must be 'low' or 'high', got {ic!r}")
    T_ms = a0 + float(T_post_ms)
    res, slow = _stage_d_run(S, cfg, T_ms, kick_boost=kb, t_kick=t_kick, seed=seed, dt=dt)
    row, traces = _branch_row(S, res, slow, bref, dt, a0, D=D, ic=ic, kick_boost=kb, seed=seed, T_ms=T_ms)
    del res
    gc.collect()
    return row, slow, traces


# ----------------------------------------------------------------- commands
def _baseline_ref_path():
    return os.path.join(OUT, "baseline_ref.json")


def cmd_baseline(args):
    """D0.2: arm-C saturated slow-off at dt=0.05 -> baseline_rate / sigma / af_q95 (classifier anchor)."""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        run_dir = os.path.join(OUT, "runs", FCXR._run_id(f"baseline_seed{args.seed}_dt{DT_D}"))
        plan = FCXR._plan_workers(args.T * (FCXR.DT / DT_D), 1)
        FCXR._resource_log(run_dir, "baseline_start", plan)
        print(f"[baseline] build seed={args.seed} dt={DT_D} T={args.T}; {plan}", flush=True)
        S = PP.build_substrate(args.seed)
        cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False, rec_sat_g=G_SAT)
        t0 = time.time()
        res, slow = _stage_d_run(S, cfg, args.T, kick_boost=0.0, t_kick=1e9, seed=args.seed, dt=DT_D)
        base = OLD.compute_baseline_ref(res, DT_D)
        num = _numerical_dt(S, res, slow, DT_D)
        rate = np.asarray(res["rate_E"], float)
        rate_roll_hi = rolling_rate_upper(rate, DT_D)          # interictal band upper edge (300ms rolling-mean q99)
        af, af_bin = OLD.C.active_fraction(res["E_spk_bool"], DT_D, OLD.C.BIN_MS)
        rate_stride = max(1, int(np.ceil(rate.size / 4000)))
        FCXR._write_npz(os.path.join(OUT, f"baseline_trace_seed{args.seed}.npz"),
                        rate_dt_ms=np.asarray([DT_D * rate_stride], np.float32),
                        rate_E=rate[::rate_stride].astype(np.float32),
                        af_bin_ms=np.asarray([af_bin], np.float32), af=np.asarray(af, np.float32))
        payload = dict(seed=args.seed, dt=DT_D, T=args.T, g_sat=G_SAT, wall_s=round(time.time() - t0, 1),
                       baseline_rate=base.baseline_rate, sigma_rate=base.sigma_rate, floor_af=base.floor_af,
                       rate_roll_hi=rate_roll_hi, roll_ms=WP_THRESHOLDS["ROLL_MS"], baseline_q=WP_THRESHOLDS["BASELINE_Q"],
                       n_returning=base.n_events, duration_median_ms=base.dur_med,
                       participation_lo=base.part_lo, participation_hi=base.part_hi,
                       peak_rate_lo=base.act_lo, peak_rate_hi=base.act_hi, numerical=num)
        out = _baseline_ref_path() if args.seed == 1 else os.path.join(OUT, f"baseline_ref_seed{args.seed}.json")
        FCXR._write_json(out, payload)
        FCXR._resource_log(run_dir, "baseline_done", dict(wall_s=payload["wall_s"], **num))
        print(f"[baseline] seed{args.seed}: rate={base.baseline_rate:.2f}+/-{base.sigma_rate:.2f}Hz "
              f"roll_hi(interictal band)={rate_roll_hi:.1f}Hz af_q95={base.floor_af:.4f} n_ret={base.n_events} "
              f"safe={not num['numerical_unsafe']} -> {out}  ({payload['wall_s']}s)", flush=True)


def cmd_smoke(args):
    """One-cell end-to-end validation + timing (dummy baseline; not for classification)."""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        run_dir = os.path.join(OUT, "runs", FCXR._run_id("smoke"))
        FCXR._resource_log(run_dir, "smoke_start", dict(D=args.D, ic=args.ic, T_post=args.T))
        print(f"[smoke] build seed={args.seed}; D={args.D} ic={args.ic} T_post={args.T} dt={DT_D}", flush=True)
        S, pi = _build_and_align(args.seed)
        bref = dict(baseline_rate=5.0, sigma_rate=5.0, floor_af=0.05)   # placeholder (timing/finiteness only)
        t0 = time.time()
        row, slow, _ = run_branch_cell(S, pi, bref, D=args.D, ic=args.ic, kick_boost=KICK_HIGH1,
                                       T_post_ms=args.T, seed=args.seed, dt=DT_D)
        row["wall_s"] = round(time.time() - t0, 1)
        FCXR._resource_log(run_dir, "smoke_done", dict(wall_s=row["wall_s"], finite=row["finite"],
                                                       numerical_unsafe=row["numerical_unsafe"]))
        steps = int(round(row["T_ms"] / DT_D))
        print(f"[smoke] finite={row['finite']} unsafe={row['numerical_unsafe']} "
              f"end_rate={row['end_rate_hz']:.1f}Hz af_tail={row['af_tail']:.4f} "
              f"high_dur={row['high_duration_ms']:.0f}ms  {row['wall_s']}s for {steps} steps "
              f"({row['wall_s'] / steps * 1e3:.2f} ms/1k-steps)", flush=True)


def _pilot_cells():
    for D in PILOT_D:
        yield dict(D=D, ic="low", kick_boost=0.0, label=f"D{D:g}_low")
        yield dict(D=D, ic="high", kick_boost=KICK_HIGH1, label=f"D{D:g}_high1")
        yield dict(D=D, ic="high", kick_boost=KICK_HIGH2, label=f"D{D:g}_high2")


def cmd_pilot(args):
    """3-point saturated pilot bracketing the transition (seed1); rows for the D1.5 classifier."""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        bref_path = _baseline_ref_path()
        if not os.path.exists(bref_path):
            raise SystemExit(f"missing baseline anchor {bref_path}; run `baseline --seed 1 --confirm-run` first")
        bref = FCXR.json.load(open(bref_path))
        run_dir = os.path.join(OUT, "runs", FCXR._run_id(f"pilot_seed{args.seed}_dt{DT_D}"))
        cells = list(_pilot_cells())
        plan = FCXR._plan_workers(args.T * (FCXR.DT / DT_D), args.workers)
        FCXR._resource_log(run_dir, "pilot_start", dict(n_cells=len(cells), T_post=args.T, **plan))
        print(f"[pilot] build seed={args.seed}; {len(cells)} cells (D={PILOT_D}) T_post={args.T} dt={DT_D}; {plan}",
              flush=True)
        S, pi = _build_and_align(args.seed)
        rows = []
        for i, c in enumerate(cells):
            t0 = time.time()
            row, slow, traces = run_branch_cell(S, pi, bref, D=c["D"], ic=c["ic"], kick_boost=c["kick_boost"],
                                                T_post_ms=args.T, seed=args.seed, dt=DT_D)
            row["label"] = c["label"]; row["wall_s"] = round(time.time() - t0, 1)
            FCXR._write_json(os.path.join(run_dir, "per_cell", f"{c['label']}.json"), row)
            FCXR._write_npz(os.path.join(run_dir, "per_cell", f"{c['label']}_trace.npz"),
                            max_raw_gErec=np.asarray(getattr(slow, "max_raw_gErec", []), np.float32), **traces)
            FCXR._resource_log(run_dir, f"cell_{c['label']}", dict(wall_s=row["wall_s"], finite=row["finite"],
                               numerical_unsafe=row["numerical_unsafe"], end_rate=row["end_rate_hz"]))
            print(f"[pilot] {i+1}/{len(cells)} {c['label']}: finite={row['finite']} "
                  f"unsafe={row['numerical_unsafe']} end_rate={row['end_rate_hz']:.1f}Hz "
                  f"af_tail={row['af_tail']:.3f} high_dur={row['high_duration_ms']:.0f}ms  {row['wall_s']}s",
                  flush=True)
            del slow; gc.collect()
            rows.append(row)
        # provisional labels (single window T1; two-window resolution deferred to the full grid)
        by_D = {}
        for row in rows:
            row["label_raw_contiguity"] = classify_run_provisional(row)
            row["label_envelope"] = classify_run_envelope(row)
            row["provisional_label"] = classify_run_workpoint(row)
            by_D.setdefault(row["D"], {})[row["label"].split("_")[-1]] = row
        per_D = []
        for D in PILOT_D:
            cd = by_D.get(D, {})
            low = cd.get("low", {}).get("provisional_label", "MISSING")
            highs = [cd[k]["provisional_label"] for k in ("high1", "high2") if k in cd]
            plateaus = [cd[k]["end_rate_hz"] for k in ("high1", "high2") if k in cd]
            highs_mapped = ["METASTABLE_TRANSIENT" if h == "EXCURSION_DECAYED" else h for h in highs]  # 1-window read
            d = classify_branch_D(low, highs_mapped, plateaus)
            per_D.append(dict(D=D, low=low, highs=highs, provisional=True, **d))
            print(f"[pilot] D={D:g}: low={low} high={highs} -> provisional {d['D_label']}", flush=True)
        any_finite = any(p["D_label"] in ("BISTABLE", "FINITE_HIGH") for p in per_D)
        verdict = ("CANDIDATE finite-high present (>=1 pilot D provisionally BISTABLE/FINITE_HIGH) "
                   "-> proceed to full grid + two-window (T2) + seed3" if any_finite else
                   "NO provisional finite-high at any pilot D (only low / metastable / ceiling / unsafe) "
                   "-> NO-GO leaning; RC1 saturation does not give a bounded persistent high branch")
        FCXR._write_json(os.path.join(run_dir, "pilot_rows.json"),
                         dict(seed=args.seed, dt=DT_D, T_post=args.T, D_grid=PILOT_D,
                              kicks=[KICK_HIGH1, KICK_HIGH2], thresholds=THRESHOLDS,
                              per_D=per_D, rows=rows, pilot_verdict=verdict))
        print(f"[pilot] VERDICT: {verdict}", flush=True)
        print(f"[pilot] done -> {run_dir}/pilot_rows.json", flush=True)


# ----------------------------------------------------------------- full 8-D grid (D1.7/D1.8)
_CTX = {}


def _grid_cell_task(task):
    """Run one grid cell (fork-pool worker or sequential). Writes per-cell JSON + g_raw; returns the row."""
    S, pi, bref, run_dir = _CTX["S"], _CTX["pi"], _CTX["bref"], _CTX["run_dir"]
    t0 = time.time()
    row, slow, traces = run_branch_cell(S, pi, bref, D=task["D"], ic=task["ic"], kick_boost=task["kick_boost"],
                                        T_post_ms=task["T_post"], seed=_CTX["seed"], dt=DT_D)
    row["label"] = task["label"]; row["slot"] = task["slot"]; row["window"] = task["window"]
    row["wall_s"] = round(time.time() - t0, 1)
    row["label_raw_contiguity"] = classify_run_provisional(row)      # legacy raw-contiguity (comparison)
    row["label_envelope"] = classify_run_envelope(row)              # envelope-vs-near-zero-q95 (comparison; flawed)
    row["provisional_label"] = classify_run_workpoint(row)          # workpoint-relative (PRIMARY, reviewer P0)
    FCXR._write_json(os.path.join(run_dir, "per_cell", f"{task['label']}.json"), row)
    FCXR._write_npz(os.path.join(run_dir, "per_cell", f"{task['label']}_trace.npz"),
                    max_raw_gErec=np.asarray(getattr(slow, "max_raw_gErec", []), np.float32), **traces)
    print(f"[grid] {task['label']:20s} -> {row['provisional_label']:20s} "
          f"roll_occ={row['roll_occ']:.2f} roll_end={row['roll_end_occ']:.2f} roll_hi={row['roll_high_ms']:6.0f}ms "
          f"end={row['end_rate_hz']:5.1f}Hz band={row['baseline_roll_hi']:.1f}Hz "
          f"unsafe={str(row['numerical_unsafe'])[0]} {row['wall_s']}s", flush=True)
    del slow; gc.collect()
    return row


def _run_tasks(tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        return [_grid_cell_task(t) for t in tasks]
    with mp.get_context("fork").Pool(workers) as pool:      # COW-share the 6.8GB substrate (FCXR pattern)
        return pool.map(_grid_cell_task, tasks)


def cmd_grid(args):
    """Full frozen branch map: 8 D x {low, high1, high2} at T1, two-window (T2) resolution for excursions."""
    with FCXR._launcher_lock():
        FCXR._assert_engine_blessed()
        bref_path = _baseline_ref_path()
        if not os.path.exists(bref_path):
            raise SystemExit(f"missing baseline anchor {bref_path}; run `baseline --seed 1 --confirm-run` first")
        bref = FCXR.json.load(open(bref_path))
        run_dir = os.path.join(OUT, "runs", FCXR._run_id(f"grid_seed{args.seed}_dt{DT_D}"))
        T1, T2 = float(args.T1), float(args.T2)
        if T1 < THRESHOLDS["HIGH_MS"] and not getattr(args, "allow_smoke", False):   # P0 runtime gate
            raise SystemExit(f"T1={T1:.0f}ms < persistence threshold HIGH_MS={THRESHOLDS['HIGH_MS']:.0f}ms: a "
                             "sub-threshold run cannot yield a scientific verdict (no run can reach persistent-high "
                             "in a window shorter than the threshold). Pass --allow-smoke to run it as validation.")
        base = [dict(D=D, ic=ic, kick_boost=kb, T_post=T1, window="T1", slot=slot, label=f"D{D:g}_{slot}_T1")
                for D in D_GRID for ic, kb, slot in
                (("low", 0.0, "low"), ("high", KICK_HIGH1, "high1"), ("high", KICK_HIGH2, "high2"))]
        plan = FCXR._plan_workers(T1 * (FCXR.DT / DT_D), args.workers)
        FCXR._resource_log(run_dir, "grid_start", dict(n_base=len(base), T1=T1, T2=T2, **plan))
        print(f"[grid] build seed={args.seed}; {len(base)} base cells (D={D_GRID}) T1={T1} T2={T2} "
              f"workers={plan['workers']}", flush=True)
        S, pi = _build_and_align(args.seed)
        _CTX.update(S=S, pi=pi, bref=bref, run_dir=run_dir, seed=args.seed)
        rows = _run_tasks(base, plan["workers"])
        t1_by = {(r["D"], r["slot"]): r for r in rows}
        # T2 (longer window) for any cell NOT clearly returned-to-low at T1: an excursion label OR a still-
        # elevated envelope tail (reviewer P1 -- a native-low with a rising envelope also enters T2).
        excursion = ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT", "EXCURSION_DECAYED")

        def _needs_t2(r):
            return bool(r["provisional_label"] in excursion or r["env_end_occ"] >= THRESHOLDS["HIGH_OCC"])

        slot_kick = {"low": 0.0, "high1": KICK_HIGH1, "high2": KICK_HIGH2}
        t2 = [dict(D=D, ic=("low" if slot == "low" else "high"), kick_boost=slot_kick[slot], T_post=T2,
                   window="T2", slot=slot, label=f"D{D:g}_{slot}_T2")
              for D in D_GRID for slot in ("low", "high1", "high2") if _needs_t2(t1_by[(D, slot)])]
        print(f"[grid] T1 done; {len(t2)} T2 (two-window) cells (excursion OR still-elevated tail)", flush=True)
        t2_rows = _run_tasks(t2, plan["workers"]) if t2 else []
        t2_by = {(r["D"], r["slot"]): r for r in t2_rows}

        def _resolved(D, slot):
            r = t1_by[(D, slot)]; p1 = r["provisional_label"]
            if _needs_t2(r):                                     # two-window resolution (clause 4)
                r2 = t2_by.get((D, slot))
                return (resolve_high_ic(p1, r2["provisional_label"]) if r2 is not None
                        else ("METASTABLE_TRANSIENT" if p1 == "EXCURSION_DECAYED" else p1))
            return p1                                            # clearly returned-to-low at T1

        per_D = []
        for D in D_GRID:
            low = _resolved(D, "low")
            resolved = [_resolved(D, "high1"), _resolved(D, "high2")]
            plateaus = [t1_by[(D, "high1")]["end_rate_hz"], t1_by[(D, "high2")]["end_rate_hz"]]
            d = classify_branch_D(low, resolved, plateaus)
            per_D.append(dict(D=D, low=low, high_resolved=resolved, **d))
            print(f"[grid] D={D:g}: low={low} high={resolved} -> {d['D_label']}", flush=True)
        is_smoke = T1 < THRESHOLDS["HIGH_MS"]
        landmarks = [p["D"] for p in per_D if p["D_label"] in ("BISTABLE", "FINITE_HIGH")]
        if is_smoke:                                             # P0: a sub-threshold run is NOT a scientific verdict
            verdict = (f"SMOKE_ONLY: T1={T1:.0f}ms < persistence threshold {THRESHOLDS['HIGH_MS']:.0f}ms — "
                       "plumbing/pool validation only, NOT a scientific branch-map verdict")
            out_name = "branch_map_SMOKE.json"
        else:
            verdict = (f"FINITE-HIGH/BISTABLE at D={landmarks} -> proceed to seed3 confirm + D2" if landmarks else
                       "CLEAN NO-GO: no persistent finite-high / bistable at any D in [0,0.15] (seed1); RC1 smooth "
                       "saturation bounds the transient (no runaway/clip) but gives no high-state attractor")
            out_name = "branch_map.json"
        FCXR._write_json(os.path.join(run_dir, out_name),
                         dict(seed=args.seed, dt=DT_D, T1=T1, T2=T2, D_grid=D_GRID, kicks=[KICK_HIGH1, KICK_HIGH2],
                              thresholds=THRESHOLDS, per_D=per_D, landmarks=landmarks, smoke_only=is_smoke,
                              base_rows=rows, t2_rows=t2_rows, verdict=verdict))
        print(f"[grid] VERDICT: {verdict}", flush=True)
        print(f"[grid] done -> {run_dir}/{out_name}", flush=True)


# ----------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description="FCXR Stage D — frozen fast-branch map (dt=0.05).")
    ap.add_argument("--confirm-run", action="store_true", help="required to run any simulation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline"); b.add_argument("--seed", type=int, default=1); b.add_argument("--T", type=float, default=8000.0)
    s = sub.add_parser("smoke"); s.add_argument("--seed", type=int, default=1); s.add_argument("--D", type=float, default=0.0)
    s.add_argument("--ic", choices=["low", "high"], default="low"); s.add_argument("--T", type=float, default=300.0)
    p = sub.add_parser("pilot"); p.add_argument("--seed", type=int, default=1); p.add_argument("--T", type=float, default=4000.0)
    p.add_argument("--workers", type=int, default=1)
    g = sub.add_parser("grid"); g.add_argument("--seed", type=int, default=1)
    g.add_argument("--T1", type=float, default=4000.0); g.add_argument("--T2", type=float, default=8000.0)
    g.add_argument("--workers", type=int, default=2)
    g.add_argument("--allow-smoke", action="store_true",
                   help="permit T1<HIGH_MS as plumbing/pool validation (writes branch_map_SMOKE.json only)")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("REFUSING: simulations require --confirm-run")
    {"baseline": cmd_baseline, "smoke": cmd_smoke, "pilot": cmd_pilot, "grid": cmd_grid}[args.cmd](args)


if __name__ == "__main__":
    main()
