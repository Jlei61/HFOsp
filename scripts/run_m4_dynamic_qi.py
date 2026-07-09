"""M4 DYNAMIC q_I experiment (user redirect 2026-07-06): the frozen-q_I phase-plane can't reach runaway
because runaway is a *dynamic* q_I-depletion phenomenon (repeated axial replay depletes the inhibitory
resource until it crosses the separatrix). Here two SMALL axial spontaneous foci (E1146 source/sink) fire
repeatedly under background noise, q_I depletes across events (k_q>0, NOT frozen), and we watch whether q_I
crosses into runaway -- WITH vs WITHOUT the M4 shared divisive pool S_G.

Reuses: run_m4_phaseplane.build_substrate (E1146 twoend_equal, L=20, SNN Stage-5 layout);
run_sef_hfo_snn_cm_spontaneous_readout (active_fraction/detect_events); pilot_stage4_spontaneous_qI
(spontaneous dynamic-q_I mechanic + _first_sustained runaway detector + verdict). Blessed q_I params from
run_m3a_v2_step2_qI (tau_q=5000, sigma_q=1.5, q_min=0.05, tau_a=20).

Question this answers (the two the user posed):
  no_pool (S_G off): does q_I deplete to the separatrix -> runaway in the window? -> (a) just not long enough
                     before, vs (b) heterogeneity keeps q_I off the separatrix.
  pool   (S_G on) : does S_G bound the q_I-depletion-driven runaway into a bounded, sustained middle state
                    (the middle region that q_I depletion ALONE lacks -- M3A sharp-bistability, m3_stage §67)?

*** RUNS SIMULATIONS -- gated behind --confirm-run. Parallel via a fork Pool (build the substrate ONCE, share
    read-only by copy-on-write; each arm is an independent long spontaneous run). ***
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse                                            # noqa: E402
import dataclasses                                         # noqa: E402
import json                                                # noqa: E402
import subprocess                                          # noqa: E402
import multiprocessing as mp                               # noqa: E402
import sys                                                 # noqa: E402
import time                                                # noqa: E402

import numpy as np                                         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP                             # noqa: E402  (E1146 build_substrate, R_KICK etc.)
import run_sef_hfo_snn_cm_spontaneous_readout as C        # noqa: E402  (active_fraction / detect_events)
from kick_probe import simulate_kick                       # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "topic4_m4_dynamic")

# ---- dynamic q_I (blessed run_m3a_v2_step2_qI) ----
TAU_Q, TAU_A, SIGMA_Q, Q_MIN = 5000.0, 20.0, 1.5, 0.05
# ---- M4 pool sensor (r50 ~ the phase-plane-calibrated rE_fast scale; REVIEW POINT) ----
R50_PSI, N_PSI, P_POOL, TAU_MU, TAU_S, S_MAX = 0.4, 2.0, 3.0, 30.0, 80.0, 1.0
# ---- M4-3A load->shunt defaults (spec Global Constraints; docs/superpowers/specs/
#      2026-07-09-sef-hfo-m4-3-continuous-shunting-axis-coordinate-design.md); tau_n SLOW (> TAU_Q) ----
M43A_TAU_N, M43A_N50, M43A_HILL_H, M43A_A_MAX = 20000.0, 0.4, 2.0, 1.0
M43A_RHO_N, M43A_K_N, M43A_ETA_A, M43A_UN0 = 0.1, 1.0, 0.0, 0.0
M43A_GA_MAX = 20.0
M43A_EARLY_OFFSET_MS = 750.0                # P1-5 early refractory probe offset (Task 7 additive)
# ---- spontaneous run ----
T_DYN = 5000.0                                             # ms (long enough for a train of events)
DT = 0.1
MOVIE_BIN_MS = 25.0                                        # downsampled spatial-activity movie frame width
MOVIE_GRID = 24                                            # movie spatial resolution (MOVIE_GRID x MOVIE_GRID)
RUNAWAY_HZ, RUNAWAY_DUR_MS = 120.0, 100.0                  # _first_sustained runaway criterion (pilot)
# ---- reversibility / basin perturbation test (2 arms, on the aG16 bounded state; NOT a termination test) ----
T_PERTURB0, T_PERTURB1 = 8000.0, 8500.0                    # 500ms transient during the settled bounded state
QI_REFILL_VAL = 1.0                                        # qI_refill arm: reset q_I to 1.0 (directly repair slow var)
INHIB_DVTH = 15.0                                          # inhibitory_pulse arm: raise E V_th +15mV (fully suppress firing, q_I untouched)

# ---- arms: (label, k_q depletion rate, use_SG, alpha_G) ----
ARMS = [
    ("kq0.35_no_pool",   0.35, False, 0.0),
    ("kq0.35_pool_aG6",  0.35, True,  6.0),
    ("kq0.18_no_pool",   0.18, False, 0.0),
    ("kq0.18_pool_aG6",  0.18, True,  6.0),
]

# ---- --sweep: dynamic (k_q x alpha_G) phase diagram = depletion-rate x pool-strength. Each k_q row has one
#      no_pool baseline (does q_I deplete to runaway?) + an alpha_G ladder (does S_G bound it, and how strong
#      must it be?). Same substrate + same noise seed across cells (build once + fork COW). ----
KQ_GRID = [0.10, 0.18, 0.25, 0.35, 0.50]
ALPHA_GRID = [2.0, 4.0, 6.0, 8.0, 12.0]


def _sweep_arms():
    arms = []
    for kq in KQ_GRID:
        arms.append((f"kq{kq:.2f}_no_pool", kq, False, 0.0))
        for ag in ALPHA_GRID:
            arms.append((f"kq{kq:.2f}_aG{ag:04.1f}", kq, True, ag))
    return arms


def _reversibility_arms():
    """3 arms at the aG16 bounded operating point (k_q=0.10). P1-3: qI_refill and inhibitory_pulse answer
    DIFFERENT questions -> reported as separate arms, never merged. baseline = un-perturbed control."""
    base = (0.10, True, 16.0)
    return [
        ("aG16_baseline", *base, None),
        ("aG16_qI_refill", *base, dict(kind="qI_refill", t0=T_PERTURB0, t1=T_PERTURB1, val=QI_REFILL_VAL)),
        ("aG16_inhib_pulse", *base, dict(kind="inhibitory_pulse", t0=T_PERTURB0, t1=T_PERTURB1, val=INHIB_DVTH)),
    ]


# ---- center-vs-core stimulation locus (2026-07-06 user redirect): spatial inhibitory_pulse on the aG16
#      bounded state. SAME footprint / window / strength; only the stim LOCUS differs. GENTLE +8mV (not the
#      reversibility arm's stronger INHIB_DVTH) so center-vs-core differences are visible, not both压死. ----
STIM_RADIUS, STIM_DVTH = 1.5, 8.0                          # per-disk E-cell radius (mm) + inhibitory V_th bump (mV)
STIM_ON, STIM_OFF = 8000.0, 10000.0                       # 2 s pulse on the settled bounded state


def _e_disk_mask(S, centers, radius):
    """Bool mask over N (E+I): E neurons within `radius` mm of ANY center; I neurons always False."""
    posE = np.asarray(S["posE"], float); NE = posE.shape[0]
    inE = np.zeros(NE, bool)
    for c in centers:
        inE |= np.linalg.norm(posE - np.asarray(c, float), axis=1) <= float(radius)
    m = np.zeros(int(S["N"]), bool); m[:NE] = inE
    return m


def _stim_locus_arms(S, stim_on=STIM_ON, stim_off=STIM_OFF, radius=STIM_RADIUS, dvth=STIM_DVTH):
    """center-vs-core stim on the aG16 bounded state (k_q=0.10, alpha_G=16). inhibitory_pulse V_th+dvth on
    a SPATIAL target; baseline = no stim. core = source+sink cores; center = two corridor points at 0.35 /
    0.65 along source->sink (straddle the midpoint, clear of the cores). Both = 2 disks radius `radius`
    -> balanced footprint. Returns 5-tuples (label, k_q, use_SG, alpha_G, perturb) for run_arm."""
    base = (0.10, True, 16.0)
    src = np.asarray(S["src_xy"], float); snk = np.asarray(S["snk_xy"], float)
    core_mask = _e_disk_mask(S, [src, snk], radius)
    center_mask = _e_disk_mask(S, [src + 0.35 * (snk - src), src + 0.65 * (snk - src)], radius)
    win = dict(t0=stim_on, t1=stim_off, val=dvth)
    return [
        ("aG16_stim_baseline", *base, None),
        ("aG16_stim_center", *base, dict(kind="inhibitory_pulse", target_mask=center_mask, **win)),
        ("aG16_stim_core", *base, dict(kind="inhibitory_pulse", target_mask=core_mask, **win)),
    ]


def _smooth(rate, dt, win_ms=20.0):
    n = max(1, int(round(win_ms / dt)))
    return np.convolve(np.asarray(rate, float), np.ones(n) / n, mode="same")


def _first_sustained(rate, dt, threshold_hz=RUNAWAY_HZ, dur_ms=RUNAWAY_DUR_MS):
    above = np.asarray(rate) >= threshold_hz
    n = max(1, int(round(dur_ms / dt)))
    if above.size < n:
        return None
    c = np.convolve(above.astype(float), np.ones(n), mode="valid")
    idx = np.flatnonzero(c >= 0.80 * n)
    return None if idx.size == 0 else round(float(idx[0] * dt), 1)


def _spatial_movie(spk, posE, L, dt):
    """Downsampled spatial-activity movie: per MOVIE_BIN_MS frame, fraction of E neurons active per
    MOVIE_GRID x MOVIE_GRID cell. Small (~n_frames x GRID x GRID). For the runaway GIF."""
    nsteps, NE = spk.shape
    bs = int(round(MOVIE_BIN_MS / dt))
    ix = np.clip((posE[:, 0] / L * MOVIE_GRID).astype(int), 0, MOVIE_GRID - 1)
    iy = np.clip((posE[:, 1] / L * MOVIE_GRID).astype(int), 0, MOVIE_GRID - 1)
    cell = iy * MOVIE_GRID + ix
    counts = np.bincount(cell, minlength=MOVIE_GRID * MOVIE_GRID).astype(float)
    counts[counts == 0] = 1.0                                          # per-cell neuron count (for fraction)
    frames = []
    for b0 in range(0, nsteps, bs):
        active = spk[b0:b0 + bs].any(axis=0)                          # neurons active in this frame
        fc = np.bincount(cell[active], minlength=MOVIE_GRID * MOVIE_GRID).astype(float)
        frames.append((fc / counts).reshape(MOVIE_GRID, MOVIE_GRID))
    return np.asarray(frames, dtype=np.float32)


def _spatial_coverage(movie, active_thresh=0.1, tail_frames=8):
    """Spatial extent of activity from the downsampled movie (n_frames, G, G) of E-active fractions --
    answers 'did the event fill the whole sheet?' (whole-field runaway) rather than staying core-local.
    active_area = fraction of grid cells whose activity fraction exceeds active_thresh; tail = last frames."""
    if movie.size == 0:
        return dict(active_area_peak=0.0, active_area_tail=0.0, tail_frac_gt_0p5=0.0)
    per_frame = (movie > active_thresh).mean(axis=(1, 2))          # active-area fraction per frame
    tail = movie[-tail_frames:]
    return dict(active_area_peak=round(float(per_frame.max()), 4),
                active_area_tail=round(float((tail > active_thresh).mean()), 4),
                tail_frac_gt_0p5=round(float((tail > 0.5).mean()), 4))


def run_arm(S, label, k_q, use_SG, alpha_G, perturb=None, pool_extra=None,
            ee_std_u=0.0, ee_std_tau_ms=0.0, t_kick2=None, KICK_BOOST2=0.0, trace_xdep=False, T_ms=None,
            use_A=False, alpha_A=0.0, tau_n=M43A_TAU_N, k_n=M43A_K_N, rho_n=M43A_RHO_N, n_base=0.0,
            n50=M43A_N50, hill_h=M43A_HILL_H, a_max=M43A_A_MAX, eta_A=M43A_ETA_A, sigma_n=SIGMA_Q,
            u_n0=M43A_UN0, g_A_max=M43A_GA_MAX, dump_shunt_trace=False):
    p = S["p"] if T_ms is None else dataclasses.replace(S["p"], T=float(T_ms))   # M4-2: pass-2 extends T for the probe
    beta_SG = float(pool_extra.get("beta_SG", 0.0)) if pool_extra else 0.0     # matched-subtractive arm
    clamp_SG = pool_extra.get("clamp_SG") if pool_extra else None              # clamped-static-pool arm
    cfg = SpatialSlowFieldConfig(use_qI=True, use_gK=False, k_q=k_q, k_K=0.0, sigma_q=SIGMA_Q, sigma_K=0.5,
                                 q_min=Q_MIN, q_init=1.0, tau_q=TAU_Q, tau_a=TAU_A,
                                 use_SG=use_SG, alpha_G=alpha_G, beta_SG=beta_SG, clamp_SG=clamp_SG,
                                 r0_psi=0.0, r50_psi=R50_PSI,
                                 n_psi=N_PSI, p_pool=P_POOL, tau_mu=TAU_MU, tau_S=TAU_S, S_max=S_MAX,
                                 # M4-3A load->shunt (use_A=False by default -> byte-parity, §6.3 Global Constraints):
                                 use_A=use_A, alpha_A=alpha_A, tau_n=tau_n, k_n=k_n, rho_n=rho_n, n_base=n_base,
                                 n50=n50, hill_h=hill_h, a_max=a_max, eta_A=eta_A, sigma_n=sigma_n,
                                 u_n0=u_n0, g_A_max=g_A_max)
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    t0 = time.time()
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], perturb=perturb,
                        ee_std_u=ee_std_u, ee_std_tau_ms=ee_std_tau_ms, dump_ee_std_trace=trace_xdep,
                        t_kick2=t_kick2, KICK_BOOST2=KICK_BOOST2,       # M4-2: STD terminator + post-offset retrigger kick
                        early_stop_runaway=_EARLY_STOP["on"])          # spontaneous primary (KICK_BOOST=0); t_kick2 = only kick
    spk = res["E_spk_bool"]
    rate = np.asarray(res["rate_E"], float)
    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (float(af.max()) - floor)
    events = C.detect_events(af, bin_w, event_on_frac=bar)
    rate_s = _smooth(rate, DT)
    runaway = _first_sustained(rate_s, DT)
    n_pre = sum(1 for e in events if runaway is None or e["t_on"] < runaway - 20.0)
    verdict = ("no_runaway" if runaway is None
               else "train_then_runaway" if (n_pre >= 2 and runaway > 200.0)
               else "one_shot_burst" if (runaway <= 200.0 or n_pre == 0)
               else "few_events_then_runaway")
    movie = _spatial_movie(spk, S["posE"], S["L"], DT)
    out = dict(
        label=label, k_q=k_q, use_SG=use_SG, alpha_G=alpha_G, seed=S["seed"], T=p.T,
        perturb_kind=(perturb["kind"] if perturb else None),
        beta_SG=beta_SG, clamp_SG=(None if clamp_SG is None else float(clamp_SG)),
        n_events=len(events), n_pre_runaway=int(n_pre), runaway_ms=runaway, verdict=verdict,
        runaway_early_stop=res.get("runaway_early_stop_ms"),        # truncation point if the sim was early-stopped
        max_rate_hz=round(float(rate_s.max()), 1),                  # res rate_E is already Hz (kick_probe:363)
        baseline_af=round(float(floor), 5),                         # M4-2: runner baseline for classify_termination (P1-a)
        q_mean_final=round(float(slow.q_I.mean()), 4), q_min_final=round(float(slow.q_I.min()), 4),
        S_G_max=round(float(max(slow.trace_SG)) if slow.trace_SG else 0.0, 4),
        **_spatial_coverage(movie),                                # active_area_peak/tail + tail_frac_gt_0p5
        wall_s=round(time.time() - t0, 1),
        # traces (per-step) for the figure + a downsampled movie for the GIF:
        trace_qI_mean=np.asarray(slow.trace_qI_mean, np.float32),
        trace_SG=np.asarray(slow.trace_SG, np.float32) if slow.trace_SG else np.zeros(0, np.float32),
        trace_Irec=np.asarray(slow.trace_Irec_mean, np.float32) if slow.trace_Irec_mean else np.zeros(0, np.float32),
        rate=rate.astype(np.float32), af=af.astype(np.float32), bin_w=float(bin_w),
        events=[(round(e["t_on"], 1), round(e["t_off"], 1)) for e in events],
        movie=movie, q_field_final=slow.q_I.astype(np.float32),
        ee_std_u=ee_std_u, ee_std_tau_ms=ee_std_tau_ms, t_kick2_ms=t_kick2,   # M4-2 provenance
    )
    if trace_xdep:                                                    # M4-2: (x_dep, q_I) diagnostic traces
        out["xdep_mean"] = res.get("xdep_mean")
        out["xdep_min"] = res.get("xdep_min")
    if dump_shunt_trace:                                              # M4-3A: (a, n) diagnostic traces (Task 4 fields)
        out["a_trace"] = np.asarray(slow.trace_a_mean, np.float32)
        out["n_trace"] = np.asarray(slow.trace_n_mean, np.float32)
    return out


def _run_p1_timing_cell(S, out_dir, *, k_q, alpha_G, ee_std_u, ee_std_tau_ms, base_T,
                        recovery_factor, reprobe_boost):
    """P1 (spec §5) ONE timing cell: medium-STD Arm-1 on a bounded op-point, two-pass retrigger, full JSON.
    NOT the sweep (Task 5) -- one cell to measure wall-clock + smoke the P1 output contract. Arm 0 vs Arm 1
    and the full grid are the (gated) sweep."""
    from src.sef_hfo_m4_termination import run_cell_with_retrigger
    cap = {}

    def run_fn(t_kick2, kick_boost2, min_T):
        T_use = base_T if min_T is None else max(base_T, min_T)       # pass-2 MUST cover t_kick2 + probe_window
        r = run_arm(S, "p1_timing", k_q, True, alpha_G, ee_std_u=ee_std_u, ee_std_tau_ms=ee_std_tau_ms,
                    t_kick2=t_kick2, KICK_BOOST2=kick_boost2, trace_xdep=True, T_ms=T_use)
        cap["pass1" if t_kick2 is None else "pass2"] = r
        return {"af": r["af"], "runaway_ms": r["runaway_ms"], "baseline_af": r["baseline_af"]}

    recovery_ms = max(ee_std_tau_ms, TAU_Q)                           # re-trigger needs q_I recovered (slow tau_q)
    t0 = time.time()
    verdict = run_cell_with_retrigger(run_fn, C.BIN_MS, recovery_ms=recovery_ms,
                                      recovery_factor=recovery_factor, reprobe_boost=reprobe_boost)
    wall = round(time.time() - t0, 1)
    p1, p2 = cap["pass1"], cap.get("pass2")
    os.makedirs(out_dir, exist_ok=True)
    result = dict(verdict, wall_s=wall, k_q=k_q, alpha_G=alpha_G, ee_std_u=ee_std_u,
                  ee_std_tau_ms=ee_std_tau_ms, recovery_ms=recovery_ms, recovery_factor=recovery_factor,
                  reprobe_boost=reprobe_boost, bin_ms=C.BIN_MS,
                  pass1_wall_s=p1["wall_s"], pass1_verdict=p1["verdict"], pass1_runaway_ms=p1["runaway_ms"],
                  pass2_wall_s=(p2["wall_s"] if p2 else None))
    json.dump(result, open(os.path.join(out_dir, "p1_timing_cell.json"), "w"), indent=2)
    np.savez_compressed(                                              # arrays for the (x_dep, q_I) diagnostic
        os.path.join(out_dir, "p1_timing_cell.npz"),
        pass1_af=p1["af"], pass1_rate=p1["rate"], pass1_xdep_mean=p1["xdep_mean"],
        pass1_xdep_min=p1["xdep_min"], pass1_qI_mean=p1["trace_qI_mean"],
        **({"pass2_af": p2["af"], "pass2_rate": p2["rate"], "pass2_xdep_mean": p2["xdep_mean"],
            "pass2_xdep_min": p2["xdep_min"], "pass2_qI_mean": p2["trace_qI_mean"]} if p2 else {}))
    print(f"[P1 timing cell] class={verdict['termination_class']} retrigger={verdict['retrigger_probe']} "
          f"runaway_ms={verdict['runaway_ms']} t_kick2_ms={verdict['t_kick2_ms']} "
          f"pass1_wall={p1['wall_s']}s pass2_wall={p2['wall_s'] if p2 else None}s total={wall}s -> {out_dir}",
          flush=True)
    return result


_S = {}
_EARLY_STOP = {"on": False}     # set in main: truncate runaway arms (perf) EXCEPT when the full post-event
                                # trajectory is needed (reversibility / stim-locus rebound must be seen in full)


def _worker(arm):
    return run_arm(_S["S"], *arm)


def _p1_cell_worker(cell):
    """Pool worker for the P1 sweep. cell = (label, k_q, alpha_G, ee_std_u, ee_std_tau_ms, base_T,
    recovery_factor, reprobe_boost). Uses the COW-shared _S['S']; runs the two-pass retrigger for one cell.
    Returns the verdict scalars + pass-1 diagnostic traces (arrays prefixed '_')."""
    from src.sef_hfo_m4_termination import run_cell_with_retrigger
    label, k_q, alpha_G, u, tau, base_T, rf, rb = cell
    S = _S["S"]
    cap = {}

    def run_fn(t_kick2, kick_boost2, min_T):
        T_use = base_T if min_T is None else max(base_T, min_T)
        r = run_arm(S, label, k_q, True, alpha_G, ee_std_u=u, ee_std_tau_ms=tau,
                    t_kick2=t_kick2, KICK_BOOST2=kick_boost2, trace_xdep=True, T_ms=T_use)
        cap["pass1" if t_kick2 is None else "pass2"] = r
        return {"af": r["af"], "runaway_ms": r["runaway_ms"], "baseline_af": r["baseline_af"]}

    recovery_ms = max(tau, TAU_Q)
    base = dict(label=label, k_q=k_q, alpha_G=alpha_G, ee_std_u=u, ee_std_tau_ms=tau, recovery_ms=recovery_ms)
    try:
        v = run_cell_with_retrigger(run_fn, C.BIN_MS, recovery_ms=recovery_ms, recovery_factor=rf, reprobe_boost=rb)
    except Exception as e:                                            # fail-loud per cell; don't kill the whole sweep
        return dict(base, termination_class="ERROR", retrigger_probe="ERROR", error=repr(e))
    p1 = cap["pass1"]
    return dict(base, **v, pass1_wall_s=p1["wall_s"], pass1_verdict=p1["verdict"],
                pass2_wall_s=(cap["pass2"]["wall_s"] if "pass2" in cap else None),
                _af=p1["af"], _xdep_mean=p1["xdep_mean"], _xdep_min=p1["xdep_min"],
                _qI=p1["trace_qI_mean"], _rate=p1["rate"])


def _run_p1_sweep(out_dir, *, seed, k_q, alpha_G, u_grid, tau_grid, base_T, recovery_factor, reprobe_boost, workers):
    """P1 (spec §5) sweep: (ee_std_u x ee_std_tau_ms) grid at one bounded op-point + Arm 0 baseline, each a
    two-pass retrigger cell, via Pool (COW-shared net). Writes p1_sweep_summary.json + p1_sweep_traces.npz.
    Conservative `workers` for OOM safety (swap tiny)."""
    os.makedirs(out_dir, exist_ok=True)
    cells = [("p1_arm0", k_q, alpha_G, 0.0, tau_grid[0], base_T, recovery_factor, reprobe_boost)]
    for u in u_grid:
        for tau in tau_grid:
            cells.append((f"p1_u{u:g}_tau{int(tau)}", k_q, alpha_G, u, tau, base_T, recovery_factor, reprobe_boost))
    print(f"[P1 sweep] {len(cells)} cells (Arm0 + {len(u_grid)}x{len(tau_grid)}) workers={workers} T={base_T} "
          f"kq={k_q} aG={alpha_G}", flush=True)
    t0 = time.time()
    with mp.Pool(min(workers, len(cells))) as pool:
        rows = pool.map(_p1_cell_worker, cells)
    wall = round(time.time() - t0, 1)
    for r in rows:
        r["seed"] = seed                                             # provenance in EVERY row (P1-2: don't rely on dir name)
    scal = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    try:
        git_sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or None
    except Exception:
        git_sha = None
    prov = dict(seed=seed, subject=getattr(PP, "SUBJECT", None), montage=getattr(PP, "MONTAGE", None),
                git_sha=git_sha, argv=sys.argv, T=base_T)
    json.dump(dict(seed=seed, k_q=k_q, alpha_G=alpha_G, u_grid=list(u_grid), tau_grid=list(tau_grid),
                   base_T=base_T, recovery_factor=recovery_factor, reprobe_boost=reprobe_boost, wall_s=wall,
                   provenance=prov, rows=scal),
              open(os.path.join(out_dir, "p1_sweep_summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(out_dir, "p1_sweep_traces.npz"),
                        **{f"{r['label']}__{a}": r[f"_{a}"] for r in rows if "error" not in r
                           for a in ("af", "xdep_mean", "xdep_min", "qI", "rate")})
    print(f"[P1 sweep] done in {wall}s -> {out_dir}", flush=True)
    for r in scal:                                                   # console map (go = terminate_clean AND retrigger pass)
        print(f"  {r['label']:<20} class={str(r.get('termination_class')):<15} "
              f"retrigger={str(r.get('retrigger_probe')):<8} runaway={r.get('runaway_ms')} "
              f"pass1={r.get('pass1_wall_s')}s", flush=True)
    return scal


def _m43a_cell_worker(cell):
    """Pool worker for the M4-3A (alpha_A x tau_n) discovery sweep. cell = (label, k_q, alpha_G, alpha_A,
    tau_n, base_T, recovery_factor, reprobe_boost). Mirrors _p1_cell_worker but sweeps the Task-4 n->a
    load/shunt field (use_A=True, alpha_A, tau_n) instead of the ee_std STD terminator. Arm0 (alpha_A=0.0)
    is the M4-1 bounded-persist baseline (uses_shunt() False -- a computed but has zero membrane effect).
    Uses the COW-shared _S['S']; runs the two-pass + early-probe retrigger (Task 7) for one cell. Returns
    the verdict scalars + the go composite + the D_A diagnostic (spec §8) + pass-1 traces (prefixed '_')."""
    from src.sef_hfo_m4_termination import run_cell_with_retrigger
    label, k_q, alpha_G, alpha_A, tau_n, base_T, rf, rb = cell
    S = _S["S"]
    cap = {}
    try:
        def run_fn(t_kick2, kick_boost2, min_T):
            T_use = base_T if min_T is None else max(base_T, min_T)
            r = run_arm(S, label, k_q, True, alpha_G, use_A=True, alpha_A=alpha_A, tau_n=tau_n,
                        t_kick2=t_kick2, KICK_BOOST2=kick_boost2, dump_shunt_trace=True, T_ms=T_use)
            cap["pass1" if t_kick2 is None else "pass2"] = r
            return {"af": r["af"], "runaway_ms": r["runaway_ms"], "baseline_af": r["baseline_af"]}

        recovery_ms = max(tau_n, TAU_Q)                               # re-trigger needs both q_I and n recovered
        v = run_cell_with_retrigger(run_fn, C.BIN_MS, recovery_ms=recovery_ms, recovery_factor=rf,
                                    reprobe_boost=rb, early_offset_ms=M43A_EARLY_OFFSET_MS)
    except Exception as e:                                             # fail-loud per cell; don't kill the whole sweep
        return dict(label=label, alpha_A=alpha_A, tau_n=tau_n, k_q=k_q, alpha_G=alpha_G,
                    termination_class="ERROR", retrigger_probe="ERROR", go=False, error=repr(e))
    go = (v["termination_class"] == "terminate_clean"                  # spec §7/D5: late AND early both required
          and v.get("retrigger_early") == "attenuated"                 # absent (None) unless terminate_clean -> go=False
          and v["retrigger_probe"] == "reignite_bounded")               # retrigger_probe = late window (Task 7)
    p1 = cap["pass1"]
    row = dict(label=label, alpha_A=alpha_A, tau_n=tau_n, k_q=k_q, alpha_G=alpha_G,
              recovery_ms=recovery_ms, go=bool(go), **v,
              pass1_wall_s=p1["wall_s"], pass1_verdict=p1["verdict"],
              pass2_wall_s=(cap["pass2"]["wall_s"] if "pass2" in cap else None),
              _af=p1["af"], _a_trace=p1["a_trace"], _n_trace=p1["n_trace"], _rate=p1["rate"])
    a_tr = p1["a_trace"]
    if a_tr is not None and len(a_tr):                                 # diagnostic: D_A = 1 + alpha_A*a (spec §8)
        D_A = 1.0 + alpha_A * np.asarray(a_tr, float)
        row.update(D_A_mean=float(D_A.mean()), D_A_p95=float(np.percentile(D_A, 95)), D_A_max=float(D_A.max()))
    return row


def _run_m43a_sweep(out_dir, *, seed, k_q, alpha_G, alpha_grid, tau_grid, base_T, recovery_factor,
                    reprobe_boost, workers):
    """M4-3A (spec §4.1/§8) discovery sweep: (alpha_A x tau_n) grid at the fixed M4 op-point (k_q, alpha_G)
    + per-seed Arm0 baseline (alpha_A=0), each a two-pass + early-probe retrigger cell, via Pool (COW-shared
    net). Writes m43a_sweep_summary.json + m43a_sweep_traces.npz. Conservative `workers` for OOM safety."""
    os.makedirs(out_dir, exist_ok=True)
    cells = [("m43a_arm0", k_q, alpha_G, 0.0, tau_grid[0], base_T, recovery_factor, reprobe_boost)]
    for a_A in alpha_grid:
        for tau in tau_grid:
            cells.append((f"m43a_a{a_A:g}_tau{int(tau)}", k_q, alpha_G, a_A, tau, base_T,
                          recovery_factor, reprobe_boost))
    print(f"[M4-3A sweep] {len(cells)} cells (Arm0 + {len(alpha_grid)}x{len(tau_grid)}) workers={workers} "
          f"T={base_T} kq={k_q} aG={alpha_G}", flush=True)
    t0 = time.time()
    with mp.Pool(min(workers, len(cells))) as pool:
        rows = pool.map(_m43a_cell_worker, cells)
    wall = round(time.time() - t0, 1)
    for r in rows:
        r["seed"] = seed                                             # provenance in EVERY row, incl. ERROR (P1-2 pattern)
    scal = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    try:
        git_sha = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip() or None
    except Exception:
        git_sha = None
    prov = dict(seed=seed, subject=getattr(PP, "SUBJECT", None), montage=getattr(PP, "MONTAGE", None),
                git_sha=git_sha, argv=sys.argv, T=base_T)
    json.dump(dict(seed=seed, k_q=k_q, alpha_G=alpha_G, alpha_grid=list(alpha_grid), tau_grid=list(tau_grid),
                   base_T=base_T, recovery_factor=recovery_factor, reprobe_boost=reprobe_boost, wall_s=wall,
                   provenance=prov, rows=scal),
              open(os.path.join(out_dir, "m43a_sweep_summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(out_dir, "m43a_sweep_traces.npz"),
                        **{f"{r['label']}__{a}": r[f"_{a}"] for r in rows if "error" not in r
                           for a in ("af", "a_trace", "n_trace", "rate")})
    print(f"[M4-3A sweep] done in {wall}s -> {out_dir}", flush=True)
    for r in scal:                                                    # console map (go = terminate_clean AND early/late)
        print(f"  {r['label']:<20} class={str(r.get('termination_class')):<15} "
              f"early={str(r.get('retrigger_early')):<12} retrigger={str(r.get('retrigger_probe')):<15} "
              f"go={r.get('go')} D_A_mean={r.get('D_A_mean')}", flush=True)
    return scal


def main():
    ap = argparse.ArgumentParser(description="M4 dynamic q_I spontaneous experiment (RUNS SIMULATIONS)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=float, default=T_DYN)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--sweep", action="store_true",
                    help="run the (k_q x alpha_G) dynamic phase diagram instead of the 4 fixed arms")
    ap.add_argument("--cells", default=None,
                    help="explicit 'k_q:alpha_G' cells (comma-sep), e.g. '0.10:12,0.18:12'; use_SG on when aG>0. "
                         "For T=5000 survivor confirmation runs.")
    ap.add_argument("--reversibility", action="store_true",
                    help="basin perturbation test on the aG16 bounded state: baseline / qI_refill / inhibitory_pulse")
    ap.add_argument("--stim-locus", action="store_true",
                    help="center-vs-core spatial stim on the aG16 bounded state: baseline / center / core "
                         "(inhibitory_pulse V_th+dvth on a spatial target; same footprint / window / strength)")
    ap.add_argument("--stim-on", type=float, default=STIM_ON)
    ap.add_argument("--stim-off", type=float, default=STIM_OFF)
    ap.add_argument("--stim-radius", type=float, default=STIM_RADIUS)
    ap.add_argument("--stim-dvth", type=float, default=STIM_DVTH)
    ap.add_argument("--mechanism", action="store_true",
                    help="mechanism control on aG16 (k_q=0.10): no_pool / divisive-S_G / matched-subtractive / "
                         "clamped-S_G. matched-subtractive beta_SG is calibrated (Definition A: same mean removed "
                         "recurrent current at the divisive steady state) from a phase-1 divisive run.")
    ap.add_argument("--confirm-run", action="store_true")
    # ---- M4-2 P1 (spec §5): ONE timing cell (NOT the full sweep = Task 5, still gated) ----
    ap.add_argument("--p1-timing-cell", action="store_true",
                    help="P1 ONE timing cell: medium-STD Arm-1 two-pass retrigger + full JSON/npz (measures wall-clock)")
    ap.add_argument("--p1-kq", type=float, default=0.10)
    ap.add_argument("--p1-alpha-g", type=float, default=16.0)         # pass-1 confirmed-bounded strip
    ap.add_argument("--p1-ee-std-u", type=float, default=0.15)        # medium STD (not strong -> avoids suppress bias)
    ap.add_argument("--p1-ee-std-tau", type=float, default=1000.0)
    ap.add_argument("--p1-recovery-factor", type=float, default=2.0)  # t_kick2 = offset + factor*max(ee_std_tau, tau_q)
    ap.add_argument("--p1-reprobe-boost", type=float, default=3.0)
    ap.add_argument("--p1-sweep", action="store_true",
                    help="P1 (ee_std_u x ee_std_tau_ms) sweep + Arm0 at one op-point via Pool (conservative workers)")
    ap.add_argument("--p1-workers", type=int, default=5)              # OOM-safe (swap tiny; other campaign live)
    ap.add_argument("--p1-u-grid", default="0.15,0.3,0.5")
    ap.add_argument("--p1-tau-grid", default="1000,2500,5000")
    # ---- M4-3A (spec §4.1/§8): (alpha_A x tau_n) load->shunt discovery sweep, same op-point + machinery as P1 ----
    ap.add_argument("--m43a-sweep", action="store_true",
                    help="M4-3A (alpha_A x tau_n) n->a load/shunt sweep + Arm0 at the P1 op-point via Pool "
                         "(conservative workers); go = terminate_clean AND early-attenuated AND late-reignite_bounded")
    ap.add_argument("--m43a-alpha-grid", default="2,4,8")
    ap.add_argument("--m43a-tau-grid", default="5000,20000,40000")
    ap.add_argument("--m43a-workers", type=int, default=5)            # OOM-safe (swap tiny; other campaign live)
    a = ap.parse_args()
    if not a.confirm_run:
        print("REFUSED: dynamic-q_I sim gate. Re-run with --confirm-run.")
        return
    # truncate runaway arms for speed EXCEPT reversibility/stim-locus (their rebound trajectory must be seen in full)
    _EARLY_STOP["on"] = not (a.reversibility or a.stim_locus)
    if a.reversibility:
        arms = _reversibility_arms()
    elif a.stim_locus or a.mechanism:
        arms = None                                                  # built after substrate (geometry / calibration)
    elif a.cells:
        arms = []
        for tok in a.cells.split(","):
            kq, ag = (float(x) for x in tok.split(":"))
            arms.append((f"kq{kq:.2f}_aG{ag:04.1f}" if ag > 0 else f"kq{kq:.2f}_no_pool", kq, ag > 0, ag))
    elif a.sweep:
        arms = _sweep_arms()
    else:
        arms = ARMS
    if a.out == OUT_DIR:
        a.out = OUT_DIR + ("_m43a_sweep" if a.m43a_sweep
                           else "_p1_sweep" if a.p1_sweep else "_p1_timing" if a.p1_timing_cell
                           else "_reversibility" if a.reversibility
                           else "_stimlocus" if a.stim_locus
                           else "_mechanism" if a.mechanism else "_sweep" if a.sweep
                           else "_confirm" if a.cells else "")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = PP.build_substrate(a.seed)
    S["p"].T = a.T                                                    # long spontaneous window (build is T-independent)
    _S["S"] = S                                                       # set BEFORE Pool -> fork COW-shares the net
    if a.p1_timing_cell:                                              # M4-2 P1: ONE timing cell, no Pool, no sweep
        _run_p1_timing_cell(S, a.out, k_q=a.p1_kq, alpha_G=a.p1_alpha_g, ee_std_u=a.p1_ee_std_u,
                            ee_std_tau_ms=a.p1_ee_std_tau, base_T=a.T, recovery_factor=a.p1_recovery_factor,
                            reprobe_boost=a.p1_reprobe_boost)
        return
    if a.p1_sweep:                                                   # M4-2 P1: (u x tau) sweep + Arm0, Pool, OOM-safe
        _run_p1_sweep(a.out, seed=a.seed, k_q=a.p1_kq, alpha_G=a.p1_alpha_g,
                      u_grid=[float(x) for x in a.p1_u_grid.split(",")],
                      tau_grid=[float(x) for x in a.p1_tau_grid.split(",")],
                      base_T=a.T, recovery_factor=a.p1_recovery_factor, reprobe_boost=a.p1_reprobe_boost,
                      workers=a.p1_workers)
        return
    if a.m43a_sweep:                     # M4-3A: (alpha_A x tau_n) sweep + Arm0, Pool, OOM-safe, same op-point as P1
        _run_m43a_sweep(a.out, seed=a.seed, k_q=a.p1_kq, alpha_G=a.p1_alpha_g,
                        alpha_grid=[float(x) for x in a.m43a_alpha_grid.split(",")],
                        tau_grid=[float(x) for x in a.m43a_tau_grid.split(",")],
                        base_T=a.T, recovery_factor=a.p1_recovery_factor, reprobe_boost=a.p1_reprobe_boost,
                        workers=a.m43a_workers)
        return
    if a.stim_locus:
        arms = _stim_locus_arms(S, a.stim_on, a.stim_off, a.stim_radius, a.stim_dvth)
    mech_div, mech_calib = None, None
    if a.mechanism:
        # phase 1: divisive arm run SERIALLY in the parent (also caches ampa_flat for the fork). Definition A:
        # match the subtractive to remove the SAME mean recurrent current at the divisive steady state (last 40%).
        mech_div = run_arm(S, "mech_divisive", 0.10, True, 16.0)
        irec, sg = mech_div["trace_Irec"], mech_div["trace_SG"]
        w0 = int(0.6 * len(irec))
        I_ref, S_ref = float(irec[w0:].mean()), float(sg[w0:].mean())
        frac = 16.0 * S_ref / (1.0 + 16.0 * S_ref)                    # divisive fraction removed at steady state
        beta_matched = (frac * I_ref / S_ref) if S_ref > 1e-9 else 0.0
        mech_calib = dict(I_ref=round(I_ref, 5), S_ref=round(S_ref, 5), frac_removed=round(frac, 4),
                          beta_matched=round(beta_matched, 5), rule="A: beta*S_ref = frac*I_ref (same mean removed current)")
        print(f"[mechanism calib] I_ref={I_ref:.4f} S_ref={S_ref:.4f} frac_removed={frac:.3f} -> beta_matched={beta_matched:.4f}",
              flush=True)
        arms = [("mech_no_pool", 0.10, False, 0.0, None, None),
                ("mech_matched_subtractive", 0.10, True, 0.0, None, dict(beta_SG=beta_matched)),
                ("mech_clamped_SG", 0.10, True, 16.0, None, dict(clamp_SG=S_ref))]
    workers = a.workers if a.workers else min(len(arms), 40)
    print(f"substrate: E1146 {PP.MONTAGE} L={S['L']} N={S['N']} src={S['src_xy'].round(1)} snk={S['snk_xy'].round(1)} "
          f"T={a.T} n_arms={len(arms)} workers={workers} sweep={a.sweep}", flush=True)
    with mp.Pool(min(workers, len(arms))) as pool:
        rows = pool.map(_worker, arms)
    if a.mechanism:
        rows = [mech_div] + list(rows)                               # prepend the phase-1 divisive (calibration) arm
    wall = time.time() - t0
    meta = dict(experiment="M4 dynamic q_I spontaneous (two axial foci, E1146 twoend_equal)",
                subject=PP.SUBJECT, montage=PP.MONTAGE, L=float(S["L"]), N=int(S["N"]), seed=a.seed, T=a.T,
                src_xy=S["src_xy"].tolist(), snk_xy=S["snk_xy"].tolist(),
                axis_unit=S["axis_unit"].tolist(), center=S["center"].tolist(),
                qI=dict(tau_q=TAU_Q, sigma_q=SIGMA_Q, q_min=Q_MIN, tau_a=TAU_A),
                pool=dict(r50_psi=R50_PSI, tau_mu=TAU_MU, tau_S=TAU_S),
                runaway_criterion=dict(hz=RUNAWAY_HZ, dur_ms=RUNAWAY_DUR_MS), sweep=bool(a.sweep),
                mechanism_calib=mech_calib,
                kq_grid=(KQ_GRID if a.sweep else None), alpha_grid=(ALPHA_GRID if a.sweep else None),
                arms=[dict(label=a[0], k_q=a[1], use_SG=a[2], alpha_G=a[3],
                           perturb=(a[4]["kind"] if len(a) > 4 and a[4] else None)) for a in arms],
                wall_s=round(wall, 1))
    # summary JSON (small) + full npz (traces + movies) for the figure/GIF
    summary = dict(meta=meta, rows=[{k: v for k, v in r.items()
                                     if k not in ("trace_qI_mean", "trace_SG", "trace_Irec", "rate", "af", "movie", "q_field_final")}
                                    for r in rows])
    json.dump(summary, open(os.path.join(a.out, "dynamic_qi_summary.json"), "w"), indent=2)
    np.savez_compressed(os.path.join(a.out, "dynamic_qi_traces.npz"),
                        posE=S["posE"].astype(np.float32), src_xy=S["src_xy"], snk_xy=S["snk_xy"],
                        L=float(S["L"]), meta=json.dumps(meta),
                        **{f"{r['label']}__{k}": r[k] for r in rows
                           for k in ("trace_qI_mean", "trace_SG", "trace_Irec", "rate", "af", "movie", "q_field_final")},
                        **{f"{r['label']}__events": np.asarray(r["events"], float) for r in rows},
                        **{f"{r['label']}__meta": json.dumps({k: v for k, v in r.items()
                            if k not in ("trace_qI_mean", "trace_SG", "trace_Irec", "rate", "af", "movie", "q_field_final", "events")})
                           for r in rows})
    print("\n===== M4 dynamic q_I report =====")
    for r in rows:
        print(f"  {r['label']:20} verdict={r['verdict']:22} n_events={r['n_events']:3} runaway_ms={r['runaway_ms']} "
              f"max_rate={r['max_rate_hz']}Hz q_final=[{r['q_min_final']},{r['q_mean_final']}] S_G_max={r['S_G_max']}")
    print(f"wrote {a.out}/dynamic_qi_summary.json + dynamic_qi_traces.npz in {wall:.0f}s")
    print("(M4 dynamic screen -- q_I-depletion vs S_G bounding; NOT a proven seizure-mechanism claim.)")


if __name__ == "__main__":
    main()
