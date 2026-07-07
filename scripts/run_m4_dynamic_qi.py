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
import json                                                # noqa: E402
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


def run_arm(S, label, k_q, use_SG, alpha_G, perturb=None, pool_extra=None):
    p = S["p"]
    beta_SG = float(pool_extra.get("beta_SG", 0.0)) if pool_extra else 0.0     # matched-subtractive arm
    clamp_SG = pool_extra.get("clamp_SG") if pool_extra else None              # clamped-static-pool arm
    cfg = SpatialSlowFieldConfig(use_qI=True, use_gK=False, k_q=k_q, k_K=0.0, sigma_q=SIGMA_Q, sigma_K=0.5,
                                 q_min=Q_MIN, q_init=1.0, tau_q=TAU_Q, tau_a=TAU_A,
                                 use_SG=use_SG, alpha_G=alpha_G, beta_SG=beta_SG, clamp_SG=clamp_SG,
                                 r0_psi=0.0, r50_psi=R50_PSI,
                                 n_psi=N_PSI, p_pool=P_POOL, tau_mu=TAU_MU, tau_S=TAU_S, S_max=S_MAX)
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    t0 = time.time()
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], perturb=perturb)   # SPONTANEOUS (no kick)
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
    return dict(
        label=label, k_q=k_q, use_SG=use_SG, alpha_G=alpha_G, seed=S["seed"], T=p.T,
        perturb_kind=(perturb["kind"] if perturb else None),
        beta_SG=beta_SG, clamp_SG=(None if clamp_SG is None else float(clamp_SG)),
        n_events=len(events), n_pre_runaway=int(n_pre), runaway_ms=runaway, verdict=verdict,
        max_rate_hz=round(float(rate_s.max()), 1),                  # res rate_E is already Hz (kick_probe:363)
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
    )


_S = {}


def _worker(arm):
    return run_arm(_S["S"], *arm)


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
    a = ap.parse_args()
    if not a.confirm_run:
        print("REFUSED: dynamic-q_I sim gate. Re-run with --confirm-run.")
        return
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
        a.out = OUT_DIR + ("_reversibility" if a.reversibility else "_stimlocus" if a.stim_locus
                           else "_mechanism" if a.mechanism else "_sweep" if a.sweep
                           else "_confirm" if a.cells else "")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    S = PP.build_substrate(a.seed)
    S["p"].T = a.T                                                    # long spontaneous window (build is T-independent)
    _S["S"] = S                                                       # set BEFORE Pool -> fork COW-shares the net
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
