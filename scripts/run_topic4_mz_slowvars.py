"""M4-MZ scientific runner — per-neuron adaptation m_i + inhibitory efficacy z_i on the E1146 substrate.

*** THIS RUNS SIMULATIONS. *** Nothing runs on import; every simulation subcommand is gated by
--confirm-run (import-safe; safe to unit-test). Design contract:
docs/superpowers/specs/2026-07-18-topic4-mz-per-neuron-slowvars-design.md.

Subcommands:
  calibrate   slow-off baselines (seeds 1/3/4) -> baseline-anchor gate + I_th_EI quantiles + eta_m
              -> results/topic4_sef_hfo/mz_slowvars/calibration.json     (design §6)
  rss-audit   one full-density cell -> peak RSS + wall time (set worker count before the sweep, §7)
  discovery   arms A (z-only) / B (m-only) / C (z+m, pre-registered 3x3) at seed=1 -> discovery_summary.json (§7)
  multiseed   selected bounded/boundary cells at seeds 1/3/4, T=15s -> per_seed/ (+ readout_ready/ for bounded) (§11)

Reuse (not reinvent): PP.build_substrate (E1146/narrow/template_source/twoend_equal/L20/dens100/AR2),
cmrun C.{active_fraction,detect_events,BIN_MS,BASELINE_MS,CAL_FRAC,per_neuron_onset,valid_mask},
M4.{_first_sustained,_smooth} (the exact 120Hz/100ms runaway criterion), topic4_m3a_v2_phenotype.event_recovery,
src.topic4_mz_slowvars (classifier + calibration helpers), mz_slow_vars.MZSlowVars.

The new module mz_slow_vars.py is NOT in the engine_versions.json guard and this runner edits NONE of the
6 guarded engine files -> no re-bless. Engine SHAs are recorded in provenance as evidence only.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")             # memory caveat: parallel numpy MUST OMP=1
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                            # noqa: E402
import csv                                                 # noqa: E402
import dataclasses                                         # noqa: E402
import hashlib                                             # noqa: E402
import json                                                # noqa: E402
import multiprocessing as mp                               # noqa: E402
import resource                                            # noqa: E402
import subprocess                                          # noqa: E402
import sys                                                 # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP                             # noqa: E402  (build_substrate, R_KICK, CORE_R, SUBJECT, MONTAGE)
import run_sef_hfo_snn_cm_spontaneous_readout as C        # noqa: E402  (active_fraction / detect_events / readout)
import run_m4_dynamic_qi as M4                             # noqa: E402  (_first_sustained / _smooth: runaway criterion)
from kick_probe import simulate_kick                       # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig      # noqa: E402
from src.topic4_mz_slowvars import (                       # noqa: E402
    MZBaseline, MZPhenotypeGates, classify_mz_run,
    pooled_quantiles_from_hist, replay_adaptation_peak, eta_m_from_frac, select_by_targets,
)
from src.topic4_m3a_v2_phenotype import event_recovery    # noqa: E402  (reuse: returned-to-baseline-band)

OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slowvars")
DT = 0.1

# ---- phase durations (ms) ----
T_CALIB, T_DISCOVERY, T_MULTISEED = 8000.0, 12000.0, 15000.0
CALIB_SEEDS = (1, 3, 4)                                    # 1,3 = clean denominator; 4 = stress (design §6/pitfall)
MIN_BASE_EVENTS = 3                                        # baseline-anchor gate (returning events / seed)

# ---- calibration grids (design §6/§7) ----
# E-cell I_I / I_E are heavy-tailed to hundreds of mV during events (current-based engine injects
# ~w*tau_m/tau_r ~ 12 mV per inhibitory spike). Linear [0,100] truncated ~25% of I_I -> quantiles wrong.
# Log-spaced [0, ~5000] mV: fine resolution near the median (~1 mV) AND captures the tail (overflow ~0).
HIST_EDGES = np.concatenate([[0.0], np.logspace(-2.0, 3.7, 400)])   # 0 + 0.01 .. ~5012 mV, 400 bins
I_TH_QS = (0.5, 0.75, 0.9)                                # q50 strong / q75 mid / q90 weak z-depletion
TAU_Z_GRID = (2500.0, 5000.0, 10000.0)                   # ms
TAU_ADP_GRID = (500.0, 2000.0, 5000.0)                   # ms
ETA_M_FRACS = (0.05, 0.10, 0.20)                         # low / mid / high adaptation-current level
ARMC_Z_TARGETS = (0.8, 0.5, 0.2)                         # arm-C weak/mid/strong by realized z_min
ARMC_M_TARGETS = (0.05, 0.10, 0.20)                      # arm-C weak/mid/strong by realized adaptation-current fraction

_GUARDED_ENGINE = ("kick_probe.py", "params.py", "model.py", "connectivity.py", "connectivity_rot.py", "lfp.py")


# ============================================================ provenance
def _git_sha():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _engine_shas():
    d = {}
    eng = os.path.join(ROOT, "src", "snn_engine")
    for f in _GUARDED_ENGINE:
        p = os.path.join(eng, f)
        try:
            d[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
        except Exception:
            d[f] = None
    return d


def _provenance(extra=None):
    prov = dict(git_sha=_git_sha(), engine_shas=_engine_shas(), argv=sys.argv,
                subject=PP.SUBJECT, montage=PP.MONTAGE, dt=DT)
    if extra:
        prov.update(extra)
    return prov


# ============================================================ substrate + core masks
def build_core_masks(S):
    """E-indexed (length NE) union of the two low-V_th cores (source + sink), geometric (design §3)."""
    posE = np.asarray(S["posE"], float)
    src = np.asarray(S["src_xy"], float)
    snk = np.asarray(S["snk_xy"], float)
    return ((np.linalg.norm(posE - src, axis=1) <= PP.CORE_R)
            | (np.linalg.norm(posE - snk, axis=1) <= PP.CORE_R))


def run_mz_cell(S, cfg, T, *, early_stop=True, lfp_recorder=None):
    """Run ONE spontaneous (no-kick) MZ cell on substrate S for T ms. Returns (res, mz).
    Same network seed AND same noise seed as every other arm (net rng reset to S['seed'])."""
    p = dataclasses.replace(S["p"], T=float(T))
    core_mask_E = build_core_masks(S)
    mz = MZSlowVars(S["N"], 18.0, cfg, NE=S["NE"], core_mask_E=core_mask_E)
    S["net"]["rng"] = np.random.default_rng(S["seed"])            # identical noise realization across arms
    res = simulate_kick(p, S["net"], 0.0, slow=mz, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], early_stop_runaway=early_stop,
                        lfp_recorder=lfp_recorder)
    return res, mz


# ============================================================ event extraction / baseline (reuse cmrun)
def _events_from_res(res, dt):
    spk = res["E_spk_bool"]
    rate = np.asarray(res["rate_E"], float)
    af, bin_w = C.active_fraction(spk, dt, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (float(af.max()) - floor)
    events = C.detect_events(af, bin_w, event_on_frac=bar)
    return events, af, bin_w, floor, rate


def _peak_rate_in(rate, e, dt):
    s, en = int(e["t_on"] / dt), int(e["t_off"] / dt) + 1
    return float(rate[s:en].max()) if en > s and en <= len(rate) else (float(rate[s:].max()) if s < len(rate) else 0.0)


def compute_baseline_ref(res, dt, gates=None):
    """MZBaseline from a slow-off run: RETURNING-event distribution + quiet-window rate stats (design §8)."""
    events, af, bin_w, floor, rate = _events_from_res(res, dt)
    ret = [e for e in events if e["returned"]]
    durs = np.array([e["dur_ms"] for e in ret], float)
    parts = np.array([e["peak_ext"] for e in ret], float)
    acts = np.array([_peak_rate_in(rate, e, dt) for e in ret], float)
    q0, q1 = int(C.BASELINE_MS[0] / dt), int(C.BASELINE_MS[1] / dt)
    quiet = rate[q0:q1]

    def pct(a, q, d):
        return float(np.percentile(a, q)) if a.size else d

    return MZBaseline(
        n_events=len(ret),
        dur_med=pct(durs, 50, 0.0), dur_hi=pct(durs, 90, 0.0),
        part_lo=pct(parts, 10, 0.0), part_hi=pct(parts, 90, 0.0),
        act_lo=pct(acts, 10, 0.0), act_hi=pct(acts, 90, 0.0),
        floor_af=float(floor),
        baseline_rate=float(quiet.mean()) if quiet.size else 0.0,
        sigma_rate=float(quiet.std()) if quiet.size else 1.0,
    )


def extract_run_metrics(res, dt, baseline, gates=None):
    """Run-metrics dict for classify_mz_run + the raw events/af/runaway_ms. peak event = largest by
    participation; peak_returned = returns to baseline band (event_recovery, reused)."""
    g = gates or MZPhenotypeGates()
    events, af, bin_w, floor, rate = _events_from_res(res, dt)
    rate_s = M4._smooth(rate, dt)
    runaway_ms = M4._first_sustained(rate_s, dt)
    es = res.get("runaway_early_stop_ms")
    if es is not None and runaway_ms is None:
        runaway_ms = es
    if events:
        peak_ev = max(events, key=lambda e: e["peak_ext"])
        # returned = post-event silence window (t_off + settle offset) mean rate back in the baseline band
        peak_returned = bool(event_recovery(rate, dt, peak_ev["t_off"] + g.recover_offset, baseline.baseline_rate,
                                            baseline.sigma_rate, m=g.recovery_m, t_return=g.t_return))
        rm = dict(n_events=len(events), peak_dur=peak_ev["dur_ms"], peak_participation=peak_ev["peak_ext"],
                  peak_rate=_peak_rate_in(rate, peak_ev, dt), peak_returned=peak_returned,
                  max_dur=max(e["dur_ms"] for e in events), peak_af=float(af.max()))
    else:
        rm = dict(n_events=0, peak_dur=0.0, peak_participation=0.0, peak_rate=0.0,
                  peak_returned=False, max_dur=0.0, peak_af=float(af.max()))
    return rm, events, af, bin_w, runaway_ms


def _downsample(a, target=2000):
    a = np.asarray(a, np.float32)
    if a.size <= target:
        return a
    return a[:: max(1, a.size // target)]


def _cell_traces(mz, res, dt):
    """Small (downsampled) traces for figures (pickle-cheap across Pool)."""
    return dict(
        rate=_downsample(res["rate_E"]), af=_downsample(_events_from_res(res, dt)[1]),
        z_mean=_downsample(mz.trace_z_mean), z_min=_downsample(mz.trace_z_min),
        z_core=_downsample(mz.trace_z_core_mean), z_surr=_downsample(mz.trace_z_surround_mean),
        m_mean=_downsample(mz.trace_m_mean), m_max=_downsample(mz.trace_m_max),
        m_core=_downsample(mz.trace_m_core_mean), adap=_downsample(mz.trace_adap_current),
    )


# ============================================================ calibration (design §6)
def _event_step_mask(events, nsteps, dt):
    m = np.zeros(nsteps, bool)
    for e in events:
        s, en = int(e["t_on"] / dt), min(nsteps, int(e["t_off"] / dt) + 1)
        m[s:en] = True
    return m


def cmd_calibrate(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    edges = HIST_EDGES
    T = float(args.T) if args.T else T_CALIB
    per_seed = {}
    pooled_I_EI = np.zeros(len(edges) - 1, np.int64)
    pooled_I_EE = np.zeros(len(edges) - 1, np.int64)
    overflow_I_EI = overflow_I_EE = 0
    peak_m_pool = {ta: [] for ta in TAU_ADP_GRID}
    for seed in CALIB_SEEDS:
        t0 = time.time()
        S = PP.build_substrate(seed)
        cfg = MZSlowVarsConfig(use_z=False, use_m=False, record_calib=True, calib_hist_edges=edges)
        res, mz = run_mz_cell(S, cfg, T, early_stop=False)        # baseline: full trace, no early stop
        baseline = compute_baseline_ref(res, DT)
        events, af, bin_w, floor, rate = _events_from_res(res, DT)
        anchor_ok = baseline.n_events >= MIN_BASE_EVENTS
        nsteps = len(mz.calib_hist_I_EI)
        emask = _event_step_mask(events, nsteps, DT)
        n_event_steps = int(emask.sum())
        hist_I_EI = np.asarray(mz.calib_hist_I_EI)               # (nsteps, nbins)
        hist_I_EE = np.asarray(mz.calib_hist_I_EE)
        if anchor_ok and n_event_steps > 0:
            pooled_I_EI += hist_I_EI[emask].sum(axis=0)
            pooled_I_EE += hist_I_EE[emask].sum(axis=0)
            # overflow = event-step E-cell samples that fell outside [edges[0], edges[-1]]
            overflow_I_EI += int(S["NE"] * n_event_steps - hist_I_EI[emask].sum())
            overflow_I_EE += int(S["NE"] * n_event_steps - hist_I_EE[emask].sum())
            for ta in TAU_ADP_GRID:
                pk = replay_adaptation_peak(res["E_spk_bool"], DT, ta, event_step_mask=emask)
                peak_m_pool[ta].append(pk)                        # per-E-cell peak m during events
        per_seed[str(seed)] = dict(
            anchor_ok=bool(anchor_ok), n_events_total=len(events), n_returning=baseline.n_events,
            n_event_steps=n_event_steps, wall_s=round(time.time() - t0, 1),
            baseline=dataclasses.asdict(baseline),
        )
        print(f"[calib seed {seed}] returning_events={baseline.n_events} anchor_ok={anchor_ok} "
              f"event_steps={n_event_steps} wall={per_seed[str(seed)]['wall_s']}s", flush=True)

    clean = [s for s in CALIB_SEEDS if per_seed[str(s)]["anchor_ok"]]
    out = dict(
        experiment="M4-MZ calibration (slow-off baseline only; design §6)",
        rules=dict(
            baseline_anchor=f">= {MIN_BASE_EVENTS} RETURNING interictal events per seed (else insufficient; "
                            f"if all seeds fail -> clean no-go at baseline, do not run arms).",
            I_th_EI=f"pooled event-step E-cell I_I quantiles over anchor-ok seeds: q50 (strong depletion) / "
                    f"q75 (mid) / q90 (weak). z depletes when I_I >= I_th_EI.",
            I_EE_scale="pooled event-step E-cell I_E q90 (excitatory current reference scale).",
            eta_m="eta_m = frac * I_EE_scale / peak_m(tau_adp); frac in {0.05,0.10,0.20} = low/mid/high; "
                  "peak_m = P95 of per-E-cell peak adaptation count during events (offline spike replay).",
            arm_c_selection="PRE-REGISTERED (before running arm C): from arm A pick weak/mid/strong = cells "
                            "whose realized z_min is closest to (0.8,0.5,0.2); from arm B pick weak/mid/strong "
                            "= cells whose realized adaptation-current fraction is closest to (0.05,0.10,0.20); "
                            "arm C = 3x3 of those. Selection uses ONLY realized z-depletion / adaptation, never "
                            "the z+m phenotype.",
            calibration_source="slow-off baseline ONLY; never tuned from z+m results.",
        ),
        seeds=list(CALIB_SEEDS), clean_seeds=clean, T=T,
        per_seed=per_seed,
        hist_edges=[float(edges[0]), float(edges[-1]), int(len(edges) - 1)],
        overflow_frac_I_EI=round(overflow_I_EI / max(1, int(pooled_I_EI.sum() + overflow_I_EI)), 4),
        overflow_frac_I_EE=round(overflow_I_EE / max(1, int(pooled_I_EE.sum() + overflow_I_EE)), 4),
        provenance=_provenance(dict(phase="calibrate", T=T)),
    )
    if not clean:
        out["verdict"] = "baseline_anchor_FAIL"
        out["note"] = ("No seed produced >= MIN_BASE_EVENTS returning interictal events at slow-off. "
                       "Baseline phase is not characterizable -> clean no-go at baseline; do NOT run arms.")
        json.dump(out, open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)
        print("\n*** BASELINE-ANCHOR FAIL: slow-off has no sparse returning interictal events. "
              "STOP (design §10). ***", flush=True)
        return out

    I_th = pooled_quantiles_from_hist(pooled_I_EI, edges, list(I_TH_QS))
    I_EE_scale = pooled_quantiles_from_hist(pooled_I_EE, edges, [0.9])[0.9]
    peak_m = {ta: float(np.percentile(np.concatenate(peak_m_pool[ta]), 95)) for ta in TAU_ADP_GRID}
    eta_m_table = {f"tau{int(ta)}_frac{frac}": eta_m_from_frac(frac, I_EE_scale, peak_m[ta])
                   for ta in TAU_ADP_GRID for frac in ETA_M_FRACS}
    out.update(dict(
        verdict="baseline_anchor_PASS",
        I_th_EI={f"q{int(q * 100)}": I_th[q] for q in I_TH_QS},
        tau_z_grid=list(TAU_Z_GRID), tau_adp_grid=list(TAU_ADP_GRID), eta_m_fracs=list(ETA_M_FRACS),
        I_EE_scale=I_EE_scale, peak_m={f"tau{int(ta)}": peak_m[ta] for ta in TAU_ADP_GRID},
        eta_m_table=eta_m_table,
        arm_c_z_targets=list(ARMC_Z_TARGETS), arm_c_m_targets=list(ARMC_M_TARGETS),
    ))
    json.dump(out, open(os.path.join(OUT_DIR, "calibration.json"), "w"), indent=2)
    print(f"\n[calibration] PASS clean_seeds={clean} I_th_EI={out['I_th_EI']} "
          f"I_EE_scale={I_EE_scale:.3f} -> {OUT_DIR}/calibration.json", flush=True)
    return out


# ============================================================ discovery (design §7)
_S = {}                              # fork-COW shared substrate + baseline (set in main before Pool)


def _arm_a_specs(cal):
    specs = []
    for qk, qv in cal["I_th_EI"].items():
        for tz in TAU_Z_GRID:
            specs.append(dict(label=f"zA_{qk}_tz{int(tz)}", arm="A",
                              cfg=dict(use_z=True, use_m=False, I_th_EI=qv, tau_z=tz)))
    return specs


def _arm_b_specs(cal):
    specs = []
    for ta in TAU_ADP_GRID:
        for frac in ETA_M_FRACS:
            eta = cal["eta_m_table"][f"tau{int(ta)}_frac{frac}"]
            specs.append(dict(label=f"mB_ta{int(ta)}_f{int(frac * 100)}", arm="B",
                              cfg=dict(use_z=False, use_m=True, tau_adp=ta, eta_m=eta)))
    return specs


def _run_spec(spec):
    """Worker: run one cell, classify, return (row, traces). Reads _S (COW: S, baseline, I_EE_scale)."""
    S, baseline, I_EE_scale, T = _S["S"], _S["baseline"], _S["I_EE_scale"], _S["T"]
    t0 = time.time()
    cfg = MZSlowVarsConfig(**spec["cfg"])
    res, mz = run_mz_cell(S, cfg, T, early_stop=True)
    rm, events, af, bin_w, runaway_ms = extract_run_metrics(res, DT, baseline)
    label = classify_mz_run(rm, baseline, runaway_ms)
    z_min_realized = float(min(mz.trace_z_min)) if mz.trace_z_min else 1.0
    adap_peak = float(max(mz.trace_adap_current)) if mz.trace_adap_current else 0.0
    adap_frac_realized = adap_peak / I_EE_scale if I_EE_scale else 0.0
    row = dict(label=spec["label"], arm=spec["arm"], phenotype=label, runaway_ms=runaway_ms,
               n_events=rm["n_events"], peak_dur=round(rm["peak_dur"], 1),
               peak_participation=round(rm["peak_participation"], 4), peak_rate=round(rm["peak_rate"], 1),
               peak_returned=rm["peak_returned"], max_dur=round(rm["max_dur"], 1),
               peak_af=round(rm["peak_af"], 4), z_min_realized=round(z_min_realized, 4),
               adap_frac_realized=round(adap_frac_realized, 4),
               runaway_early_stop=res.get("runaway_early_stop_ms"),
               seed=S["seed"], T=T, cfg=spec["cfg"], wall_s=round(time.time() - t0, 1))
    return row, _cell_traces(mz, res, DT)


def _run_specs_pool(specs, workers):
    if workers <= 1:
        return [_run_spec(s) for s in specs]
    with mp.Pool(min(workers, len(specs))) as pool:
        return pool.map(_run_spec, specs)


def cmd_discovery(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    cal = json.load(open(os.path.join(OUT_DIR, "calibration.json")))
    if cal.get("verdict") != "baseline_anchor_PASS":
        print("*** calibration verdict is not PASS -> refusing discovery (design §10). ***", flush=True)
        sys.exit(2)
    seed = int(args.seed) if args.seed else 1
    T = float(args.T) if args.T else T_DISCOVERY
    workers = int(args.workers) if args.workers else 2
    S = PP.build_substrate(seed)
    # fresh slow-off baseline at THIS (seed, T) so n_events is apples-to-apples with the arms
    base_cfg = MZSlowVarsConfig(use_z=False, use_m=False)
    res0, _ = run_mz_cell(S, base_cfg, T, early_stop=False)
    baseline = compute_baseline_ref(res0, DT)
    if baseline.n_events < MIN_BASE_EVENTS:
        print(f"*** slow-off seed {seed} has {baseline.n_events} returning events < {MIN_BASE_EVENTS} "
              f"-> insufficient baseline; STOP (design §10). ***", flush=True)
        json.dump(dict(verdict="insufficient_baseline", seed=seed, T=T,
                       baseline=dataclasses.asdict(baseline), provenance=_provenance(dict(phase="discovery"))),
                  open(os.path.join(OUT_DIR, "discovery_summary.json"), "w"), indent=2)
        return
    _S.update(S=S, baseline=baseline, I_EE_scale=cal["I_EE_scale"], T=T)

    # ---- arms A + B ----
    specs_ab = _arm_a_specs(cal) + _arm_b_specs(cal)
    print(f"[discovery] seed={seed} T={T} baseline returning_events={baseline.n_events}; "
          f"running {len(specs_ab)} A+B cells with {workers} workers...", flush=True)
    res_ab = _run_specs_pool(specs_ab, workers)
    rows = [r for r, _ in res_ab]
    traces = {r["label"]: tr for (r, tr) in res_ab}

    # ---- arm C: PRE-REGISTERED selection from realized A/B (design §7) ----
    a_rows = [r for r in rows if r["arm"] == "A"]
    b_rows = [r for r in rows if r["arm"] == "B"]
    a_pick = select_by_targets([r["z_min_realized"] for r in a_rows], list(ARMC_Z_TARGETS))
    b_pick = select_by_targets([r["adap_frac_realized"] for r in b_rows], list(ARMC_M_TARGETS))
    z_cfgs = [(a_rows[i]["label"], a_rows[i]["cfg"]) for i in a_pick]
    m_cfgs = [(b_rows[i]["label"], b_rows[i]["cfg"]) for i in b_pick]
    specs_c = []
    for zi, (zl, zc) in zip(("w", "m", "s"), z_cfgs):
        for mi, (ml, mc) in zip(("w", "m", "s"), m_cfgs):
            specs_c.append(dict(label=f"zmC_z{zi}_m{mi}", arm="C",
                                cfg=dict(use_z=True, use_m=True, I_th_EI=zc["I_th_EI"], tau_z=zc["tau_z"],
                                         tau_adp=mc["tau_adp"], eta_m=mc["eta_m"]),
                                src_z=zl, src_m=ml))
    print(f"[discovery] arm C 3x3 from z-picks {[z[0] for z in z_cfgs]} x m-picks {[m[0] for m in m_cfgs]}",
          flush=True)
    res_c = _run_specs_pool(specs_c, workers)
    rows += [r for r, _ in res_c]
    traces.update({r["label"]: tr for (r, tr) in res_c})

    # ---- persist ----
    np.savez_compressed(os.path.join(OUT_DIR, "discovery_traces.npz"),
                        **{k: json.dumps({kk: vv.tolist() for kk, vv in v.items()}) for k, v in traces.items()})
    with open(os.path.join(OUT_DIR, "per_run.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    _write_csv(rows, os.path.join(OUT_DIR, "per_run.csv"))
    summary = dict(
        experiment="M4-MZ discovery (arms A z-only / B m-only / C z+m; design §7)",
        verdict="ran", seed=seed, T=T, baseline=dataclasses.asdict(baseline),
        arm_c_selection=dict(z_picks=[z[0] for z in z_cfgs], m_picks=[m[0] for m in m_cfgs],
                             z_targets=list(ARMC_Z_TARGETS), m_targets=list(ARMC_M_TARGETS)),
        phenotype_counts=_counts([r["phenotype"] for r in rows]),
        rows=rows, provenance=_provenance(dict(phase="discovery", T=T)),
    )
    json.dump(summary, open(os.path.join(OUT_DIR, "discovery_summary.json"), "w"), indent=2)
    print(f"\n[discovery] phenotype counts: {summary['phenotype_counts']} -> {OUT_DIR}/discovery_summary.json",
          flush=True)
    return summary


def _counts(labels):
    out = {}
    for x in labels:
        out[x] = out.get(x, 0) + 1
    return out


def _write_csv(rows, path):
    keys = ["label", "arm", "phenotype", "n_events", "peak_dur", "peak_participation", "peak_rate",
            "peak_returned", "max_dur", "peak_af", "z_min_realized", "adap_frac_realized",
            "runaway_ms", "runaway_early_stop", "seed", "T", "wall_s"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================ RSS audit (design §7)
def cmd_rss_audit(args):
    seed = int(args.seed) if args.seed else 1
    T = float(args.T) if args.T else T_DISCOVERY
    S = PP.build_substrate(seed)
    cfg = MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=5.0, tau_z=5000.0, tau_adp=2000.0, eta_m=0.1)
    t0 = time.time()
    res, mz = run_mz_cell(S, cfg, T, early_stop=True)
    wall = time.time() - t0
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0   # KB -> GB (Linux)
    print(f"[rss-audit] N={S['N']} NE={S['NE']} T={T}ms wall={wall:.1f}s peak_RSS={peak_rss_gb:.2f} GB "
          f"nsteps={len(res['rate_E'])}", flush=True)
    print(f"[rss-audit] suggested workers (leave ~20% headroom, cap 5): compute from free RAM / "
          f"{peak_rss_gb:.2f} GB per worker.", flush=True)
    return dict(N=S["N"], T=T, wall_s=round(wall, 1), peak_rss_gb=round(peak_rss_gb, 2))


# ============================================================ multiseed (design §10/§11)
def save_readout_bundle(S, cfg, tag, out_dir):
    """Re-run the candidate WITH an LFP recorder and dump a readout-ready artifact bundle (design §11)."""
    from lfp import LFPRecorder
    os.makedirs(out_dir, exist_ok=True)
    msheet = S["reg"]["montage_sheet"]
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=np.asarray(msheet.contacts))
    res, mz = run_mz_cell(S, cfg, T_MULTISEED, early_stop=False, lfp_recorder=rec)
    rm, events, af, bin_w, runaway_ms = extract_run_metrics(res, DT, _S.get("baseline"))
    onset = None
    if events:
        pk = max(events, key=lambda e: e["peak_ext"])
        onset = C.per_neuron_onset(res["E_spk_bool"], pk["t_on"], pk["t_off"], DT).astype(np.float32)
        win = (pk["t_on"], pk["t_off"])
    else:
        win = (None, None)
    np.savez_compressed(
        os.path.join(out_dir, f"readout_{tag}.npz"),
        lfp_trace=np.asarray(res.get("lfp_trace"), np.float32) if res.get("lfp_trace") is not None else np.zeros(0),
        lfp_times=np.asarray(res.get("times"), np.float32),
        contacts=np.asarray(msheet.contacts, np.float32), names=np.array(msheet.names, dtype=object),
        posE=np.asarray(S["posE"], np.float32), vth=np.asarray(S["vth"][:S["NE"]], np.float32),
        src_xy=np.asarray(S["src_xy"], np.float32), snk_xy=np.asarray(S["snk_xy"], np.float32),
        axis_unit=np.asarray(S["axis_unit"], np.float32), L=float(S["L"]),
        per_neuron_onset=(onset if onset is not None else np.zeros(0, np.float32)),
        event_window=np.asarray(win, dtype=object),
        rate=np.asarray(res["rate_E"], np.float32), af=np.asarray(af, np.float32),
        trace_z_mean=np.asarray(mz.trace_z_mean, np.float32), trace_z_min=np.asarray(mz.trace_z_min, np.float32),
        trace_m_mean=np.asarray(mz.trace_m_mean, np.float32), trace_adap=np.asarray(mz.trace_adap_current, np.float32),
        cfg=json.dumps(dataclasses.asdict(cfg)), tag=tag,
    )
    return dict(tag=tag, seed=S["seed"], runaway_ms=runaway_ms, phenotype=classify_mz_run(rm, _S.get("baseline"), runaway_ms))


def _run_cand_ms(cand):
    """Worker: run one multiseed candidate on _S (COW). Deterministic: run_mz_cell resets net rng to S['seed']."""
    S, baseline, anchor_ok, T = _S["S"], _S["baseline"], _S["anchor_ok"], _S["T"]
    cfg = MZSlowVarsConfig(**cand["cfg"])
    res, mz = run_mz_cell(S, cfg, T, early_stop=True)
    rm, events, af, bin_w, runaway_ms = extract_run_metrics(res, DT, baseline)
    ph = "insufficient" if not anchor_ok else classify_mz_run(rm, baseline, runaway_ms)
    return dict(label=cand["label"], seed=S["seed"], phenotype=ph, runaway_ms=runaway_ms,
                n_events=rm["n_events"], peak_dur=round(rm["peak_dur"], 1),
                peak_participation=round(rm["peak_participation"], 4),
                peak_rate=round(rm["peak_rate"], 1), peak_returned=rm["peak_returned"],
                baseline_returning=baseline.n_events, cfg=cand["cfg"])


def cmd_multiseed(args):
    os.makedirs(os.path.join(OUT_DIR, "per_seed"), exist_ok=True)
    cands = json.load(open(args.candidates))   # [{"label":..., "cfg":{...}, "save_readout":bool}, ...]
    T = float(args.T) if args.T else T_MULTISEED
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else ["1", "3", "4"])]
    workers = int(args.workers) if args.workers else 1
    all_rows = []
    for seed in seeds:
        S = PP.build_substrate(seed)
        res0, _ = run_mz_cell(S, MZSlowVarsConfig(use_z=False, use_m=False), T, early_stop=False)
        baseline = compute_baseline_ref(res0, DT)
        anchor_ok = baseline.n_events >= MIN_BASE_EVENTS
        _S.update(S=S, baseline=baseline, anchor_ok=anchor_ok, T=T)
        if workers > 1 and len(cands) > 1:
            with mp.Pool(min(workers, len(cands))) as pool:
                rows = pool.map(_run_cand_ms, cands)
        else:
            rows = [_run_cand_ms(c) for c in cands]
        for r in rows:
            print(f"[multiseed] {r['label']} seed={seed} -> {r['phenotype']} (runaway_ms={r['runaway_ms']})", flush=True)
        all_rows.extend(rows)
        for cand, r in zip(cands, rows):
            if cand.get("save_readout") and r["phenotype"] in ("expanded_bounded", "expanded_returned"):
                save_readout_bundle(S, MZSlowVarsConfig(**cand["cfg"]), f"{cand['label']}_seed{seed}",
                                    os.path.join(OUT_DIR, "readout_ready"))
    json.dump(dict(experiment="M4-MZ multiseed (seeds 1/3/4, T=15s; design §10)", T=T, seeds=seeds,
                   phenotype_counts=_counts([r["phenotype"] for r in all_rows]), rows=all_rows,
                   provenance=_provenance(dict(phase="multiseed", T=T))),
              open(os.path.join(OUT_DIR, "per_seed", "multiseed_summary.json"), "w"), indent=2)
    print(f"\n[multiseed] done -> {OUT_DIR}/per_seed/multiseed_summary.json", flush=True)


# ============================================================ figure capture (design §13)
# most-interesting-phenotype priority for picking one representative cell per arm
_PHENO_PRIORITY = {"expanded_returned": 6, "expanded_bounded": 5, "runaway": 4, "fragment": 3,
                   "suppress": 2, "interictal_like": 1, "insufficient": 0}


def cmd_capture_figures(args):
    """Re-run slow-off + one representative cell per arm (seed 1) saving 1D traces + a downsampled
    spatial movie, for the 3 figures. Reads discovery_summary.json to pick representatives."""
    summary = json.load(open(os.path.join(OUT_DIR, "discovery_summary.json")))
    rows = summary["rows"]
    seed = int(summary["seed"])
    T = float(summary["T"])
    S = PP.build_substrate(seed)
    out = dict(seed=seed, T=T, src_xy=np.asarray(S["src_xy"], np.float32),
               snk_xy=np.asarray(S["snk_xy"], np.float32), axis_unit=np.asarray(S["axis_unit"], np.float32),
               L=float(S["L"]), core_r=float(PP.CORE_R))

    def _capture(tag, cfg, pheno, early_stop):
        res, mz = run_mz_cell(S, cfg, T, early_stop=early_stop)
        af = _events_from_res(res, DT)[1]
        movie = M4._spatial_movie(res["E_spk_bool"], S["posE"], S["L"], DT)   # (frames, 24, 24) E-active frac
        out[f"{tag}__pheno"] = pheno
        out[f"{tag}__rate"] = _downsample(res["rate_E"], 3000)
        out[f"{tag}__af"] = _downsample(af, 3000)
        out[f"{tag}__z_mean"] = _downsample(mz.trace_z_mean, 3000)
        out[f"{tag}__z_min"] = _downsample(mz.trace_z_min, 3000)
        out[f"{tag}__z_core"] = _downsample(mz.trace_z_core_mean, 3000)
        out[f"{tag}__m_core"] = _downsample(mz.trace_m_core_mean, 3000)
        out[f"{tag}__adap"] = _downsample(mz.trace_adap_current, 3000)
        out[f"{tag}__movie"] = movie.astype(np.float32)
        print(f"[capture] {tag} pheno={pheno} movie={movie.shape}", flush=True)

    _capture("slow_off", MZSlowVarsConfig(use_z=False, use_m=False), "slow_off", early_stop=False)
    for arm in ("A", "B", "C"):
        arm_rows = [r for r in rows if r["arm"] == arm]
        if not arm_rows:
            continue
        pick = max(arm_rows, key=lambda r: _PHENO_PRIORITY.get(r["phenotype"], 0))
        _capture(f"arm{arm}_{pick['label']}", MZSlowVarsConfig(**pick["cfg"]), pick["phenotype"],
                 early_stop=(pick["phenotype"] == "runaway"))
    np.savez_compressed(os.path.join(OUT_DIR, "figure_capture.npz"), **out)
    print(f"[capture] -> {OUT_DIR}/figure_capture.npz", flush=True)


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="M4-MZ per-neuron slow-var runner (design 2026-07-18).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "rss-audit", "discovery", "multiseed", "capture-figures"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true", help="required to start any simulation")
        sp.add_argument("--seed", default=None)
        sp.add_argument("--T", default=None)
        sp.add_argument("--workers", default=None)
        sp.add_argument("--seeds", default=None, help="multiseed: comma list, default 1,3,4")
        sp.add_argument("--candidates", default=None, help="multiseed: path to candidates JSON")
    args = ap.parse_args(argv)
    if not args.confirm_run:
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run to proceed "
              f"(import-safe gate, design §5).", file=sys.stderr)
        sys.exit(2)
    if args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "rss-audit":
        cmd_rss_audit(args)
    elif args.cmd == "discovery":
        cmd_discovery(args)
    elif args.cmd == "multiseed":
        cmd_multiseed(args)
    elif args.cmd == "capture-figures":
        cmd_capture_figures(args)


if __name__ == "__main__":
    main()
