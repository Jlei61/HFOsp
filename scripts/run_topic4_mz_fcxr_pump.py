"""FCXR pump-lifecycle runner — P0 (Gate I-a instrument) and P1 (Gate T topology) stages.

Nothing runs on import; every simulation requires --confirm-run. Fixed dt=0.05 on the locked
FCXR-HEO substrate (E1146 / L=20 / N=40000, arm-C recurrent conductance + g_sat=21.6 smooth
saturation, M off, X=1). The ONLY new mechanism is the off-by-default per-cell activity-dependent
load u_i -> phi(u_i) -> baseline-compensated electrogenic current.

RNG contract (plan §2): connectivity_seed / noise_seed are split. development (1,101),
baseline calibration noise 201, final held-out equivalence noise 202, confirmatory 301-303,
perturbation/shuffle 7001.

Stages
  p0-smoke        cheap plumbing: small net + one short L=20 cell (no gate output)
  p0-baseline     sensor-only calibration trajectory (noise 201) -> load candidate scan, p0
                  shrinkage, block equivalence margins, virtual-SEEG component audit
  p0-equivalence  FINAL held-out trajectory (noise 202) pump-off vs pump-on, judged ONCE against
                  the margins locked by p0-baseline
  p0-adjudicate   Gate I-a verdict
  p1-sensor-field activity-shaped load fields on the sensor-only established-high branch
  p1-map          frozen rho_Z x mean-excess-pump-activation topology map

Design: docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md
Plan:   docs/superpowers/plans/2026-07-26-topic4-mz-fcxr-pump-lifecycle.md
Outputs: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/pump_lifecycle/
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-fcxr-pump")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import fcntl
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP                     # noqa: E402  build_substrate, R_KICK
import run_topic4_mz_slowvars as OLD               # noqa: E402  build_core_masks, _events_from_res
import run_topic4_mz_fcxr as FCXR                  # noqa: E402  _fc_cfg + io/resource/flock/bless scaffolding
from kick_probe import simulate_kick               # noqa: E402
from lfp import LFPRecorder                        # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
import src.topic4_mz_fcxr_pump as PUMP             # noqa: E402

# ---------------- locked constants ----------------
G_SAT = 21.6
DT = 0.05
CONN_SEED = 1
NOISE_CALIB, NOISE_HELDOUT = 201, 202
BURN_IN_MS = 1000.0                                # discarded before any calibration block
BLOCK_MS = 2000.0                                  # calibration / equivalence block length
N_BLOCKS = 5                                       # -> T = 1000 + 5*2000 = 11000 ms
BASELINE_RATE_HZ = 3.838                           # accepted FCXR interictal workpoint (HEO1 F0 seed1)

# Pre-registered load-candidate admissibility (spec §I1).
#   A1 visible   : PER-CELL, event-locked. Among cells participating in a detected event, the median
#                  within-event rise of their own pump activation must be >= K_VISIBLE x the median
#                  |change| of the SAME cells over a matched-length interval with no event.
#                  CORRECTION (locked 2026-07-26 before the re-run, first run superseded): the
#                  original clause used the POPULATION-MEAN activation, which dilutes a participating
#                  cell's excursion ~25x (events recruit a few percent of E cells) and whose
#                  fluctuation is dominated by residual equilibration drift, not by events.
#   A2 not pinned: the 99th percentile of per-cell phi stays below PHI_PINNED AND no cell is
#                  divergent (a_load*r_i*tau_N >= 1 has no steady state).
#   A3 headroom  : the population-mean phi (the p0 level) lies inside PHI_HEADROOM
# Selection among admissible candidates: largest tau_N (longest postictal-memory capacity), ties
# broken by the phi target closest to 0.15.
K_VISIBLE = 3.0
PHI_PINNED = 0.90
PHI_HEADROOM = (0.02, 0.35)
TAU_N_GRID = (500.0, 1000.0, 2000.0)               # ms
PHI_TARGET_GRID = (0.05, 0.15, 0.30)
PLUMBING_CANDIDATE = dict(tau_N=2000.0, phi_target=0.15)   # in-engine cross-check only

OUT = os.path.join(FCXR.OUT_ROOT, "pump_lifecycle")


def _a_load_for(phi_target, tau_N, rate_hz=BASELINE_RATE_HZ):
    """Mean-field inversion of the load steady state: phi_ss = a_load * r * tau_N (r in spikes/ms).
    A STARTING POINT only -- admissibility is decided on the measured trajectory, not on this."""
    return float(phi_target / (rate_hz * 1e-3 * tau_N))


def _candidates():
    return [dict(tau_N=float(t), phi_target=float(q), a_load=_a_load_for(q, t))
            for t in TAU_N_GRID for q in PHI_TARGET_GRID]


# ----------------------------------------------------------------- substrate / config / run
def _substrate(seed=CONN_SEED):
    return PP.build_substrate(int(seed))


def _montage(S):
    mont = S["reg"]["montage_sheet"]
    return np.asarray(mont.contacts, float), list(mont.names)


def _pump_cfg(*, sensor_only, a_load, tau_ms, Imax=0.0, p0_E=None, record_calibration=False,
              interventions=None):
    """Accepted arm-C FCXR substrate + the pump plugin. M off, X=1, no cooperative gate (P0 runs on
    the interictal workpoint, not the high branch)."""
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False,
                       rec_sat_g=G_SAT)
    cfg.update(use_pump=True, pump_sensor_only=bool(sensor_only), pump_a_load=float(a_load),
               pump_tau_ms=float(tau_ms), pump_Imax=float(Imax), pump_h=PUMP.PRIMARY_H,
               pump_p0_E=p0_E, pump_record_calibration=bool(record_calibration),
               pump_interventions=interventions)
    return cfg


def _pump_off_cfg():
    return FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False,
                        rec_sat_g=G_SAT)


def _run(S, cfg_dict, T_ms, *, noise_seed, observe=True, snapshot_steps=None):
    """One no-kick trajectory. connectivity_seed fixes the network; noise_seed fixes the drive."""
    PUMP.require_primary_h(cfg_dict.get("pump_h", PUMP.PRIMARY_H))
    p = dataclasses.replace(S["p"], T=float(T_ms), dt=DT)
    cfg = MZSlowVarsConfig(**cfg_dict)
    slow = MZSlowVars(S["N"], 18.0, cfg, NE=S["NE"], core_mask_E=OLD.build_core_masks(S),
                      snapshot_steps=snapshot_steps)
    contacts, _ = _montage(S)
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
    obs = PUMP.VirtualSeegComponentObserver(rec, cfg) if observe else None
    slow.seeg_observer = obs
    S["net"]["rng"] = np.random.default_rng(int(noise_seed))
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"],
                        lfp_recorder=rec, early_stop_runaway=False)
    return res, slow, obs


# ----------------------------------------------------------------- per-block primary metrics
def _core_masks(S):
    """Source / sink / off-axis E masks from the registered geometry (no learned labels)."""
    posE = np.asarray(S["posE"], float)
    src = np.linalg.norm(posE - np.asarray(S["src_xy"], float), axis=1) <= PP.CORE_R
    snk = np.linalg.norm(posE - np.asarray(S["snk_xy"], float), axis=1) <= PP.CORE_R
    return src, snk, ~(src | snk)


def _forward_fraction(spk_block, src_m, snk_m, events, dt, t0_ms):
    """Fraction of events whose mean first-spike time is earlier in the SOURCE core than in the SINK
    core (the model-side forward/reverse template readout). Events with no participation on one side
    are skipped, never counted as forward."""
    fwd = tot = 0
    for e in events:
        s = int((e["t_on"] - t0_ms) / dt); en = int((e["t_off"] - t0_ms) / dt) + 1
        if s < 0 or en > spk_block.shape[0] or en <= s:
            continue
        seg = spk_block[s:en]
        first = np.argmax(seg, axis=0).astype(float)
        fired = seg.any(axis=0)
        a, b = fired & src_m, fired & snk_m
        if not (a.any() and b.any()):
            continue
        tot += 1
        fwd += int(first[a].mean() < first[b].mean())
    return float(fwd) / tot if tot else float("nan"), int(tot)


def _block_metrics(res, obs_traces, S, lo, hi, dt, event_bar):
    """Primary Gate I-a baseline metrics on ONE block (steps [lo,hi)) of a trajectory."""
    spk = res["E_spk_bool"][lo:hi]
    rate = np.asarray(res["rate_E"], float)[lo:hi]
    sub = dict(E_spk_bool=spk, rate_E=rate)
    events, af, bin_w, floor, _ = OLD._events_from_res(sub, dt, event_bar=event_bar)
    ret = [e for e in events if e["returned"]]
    onsets = np.array([e["t_on"] for e in ret], float)
    iei = np.diff(onsets) if onsets.size >= 2 else np.array([])
    src_m, snk_m, off_m = _core_masks(S)
    tot = max(1, int(spk.sum()))
    win_s = (hi - lo) * dt / 1000.0
    fwd_frac, n_dir = _forward_fraction(spk, src_m, snk_m, ret, dt, 0.0)
    m = dict(
        ied_rate_hz=len(ret) / win_s,
        iei_median_ms=float(np.median(iei)) if iei.size else float("nan"),
        iei_cv=float(np.std(iei) / np.mean(iei)) if iei.size and np.mean(iei) > 0 else float("nan"),
        duration_median_ms=float(np.median([e["dur_ms"] for e in ret])) if ret else float("nan"),
        participation_median=float(np.median([e["peak_ext"] for e in ret])) if ret else float("nan"),
        peak_rate_median_hz=float(np.median([OLD._peak_rate_in(rate, e, dt) for e in ret]))
        if ret else float("nan"),
        mean_rate_hz=float(rate.mean()),
        source_spike_share=float(spk[:, src_m].sum()) / tot,
        sink_spike_share=float(spk[:, snk_m].sum()) / tot,
        offaxis_spike_share=float(spk[:, off_m].sum()) / tot,
        forward_event_fraction=fwd_frac,
        n_direction_events=n_dir,
        n_events=len(ret),
    )
    for name in ("legacy_abs", "no_direct_pump", "all_components"):
        m[f"bandpower_1_80_{name}"] = PUMP.band_power(obs_traces[name][lo:hi], dt, (1.0, 80.0))
    return m


def _blocks(n_steps, dt):
    b0 = int(round(BURN_IN_MS / dt)); w = int(round(BLOCK_MS / dt))
    return [(b0 + i * w, min(b0 + (i + 1) * w, n_steps)) for i in range(N_BLOCKS)]


# ----------------------------------------------------------------- candidate admissibility (§I1)
def _candidate_report(raster, cand, dt, blocks, events_steps, quiet_steps, rate_E_hz, quiet_pairs):
    """Offline sensor-only load replay from the ANALYTIC steady state + the three PRE-REGISTERED
    admissibility clauses. Starting at u* removes the multi-second startup transient that would
    otherwise make every p0_i a transient rather than a baseline expectation.

    A1 is PER CELL and event-locked: for every event, the load of the cells that actually fired in
    it is read at the event's start and end, and compared with the same cells' load change over a
    matched-length interval containing no event.
    """
    u0, frac_div = PUMP.analytic_steady_load(rate_E_hz, a_load=cand["a_load"], tau_N=cand["tau_N"])
    # one snapshot per boundary of every scored event and of its matched quiet control
    usable = [k for k, q in enumerate(quiet_pairs) if q is not None]
    want = sorted({s for k in usable for s in
                   (events_steps[k][0], events_steps[k][1], quiet_pairs[k][0], quiet_pairs[k][1])})
    idx_of = {s: i for i, s in enumerate(want)}
    u_fin, snaps, blk_phi, blk_spk, u_mean = PUMP.integrate_load_from_raster(
        raster, a_load=cand["a_load"], tau_N=cand["tau_N"], dt=dt, u0=u0,
        snapshot_steps=want, block_edges=blocks)

    def _phi_at(step):
        return PUMP.pump_activation(snaps[idx_of[step]])

    if usable:
        phi_on = np.stack([_phi_at(events_steps[k][0]) for k in usable])
        phi_off = np.stack([_phi_at(events_steps[k][1]) for k in usable])
        part = np.stack([raster[events_steps[k][0]:events_steps[k][1] + 1].any(axis=0)
                         for k in usable])
        phi_qa = np.stack([_phi_at(quiet_pairs[k][0]) for k in usable])
        phi_qb = np.stack([_phi_at(quiet_pairs[k][1]) for k in usable])
        vis = PUMP.event_locked_load_visibility(phi_on, phi_off, part, phi_qa, phi_qb, K_VISIBLE)
        part_frac = float(part.mean())
    else:
        vis = dict(n_events_scored=0, rise_median=float("nan"), quiet_median=float("nan"),
                   ratio=float("nan"), visible=False)
        part_frac = float("nan")

    phi_mean_trace = PUMP.pump_activation(u_mean)          # population-mean load (diagnostic only)
    phi_all = blk_phi.reshape(-1)
    pop_phi = float(blk_phi.mean())
    drift = float(abs(u_fin.mean() - u0.mean()) / max(u0.mean(), 1e-12))
    a1 = bool(vis["visible"])
    a2 = bool(np.percentile(phi_all, 99) < PHI_PINNED and frac_div == 0.0)
    a3 = bool(PHI_HEADROOM[0] <= pop_phi <= PHI_HEADROOM[1])
    return dict(tau_N=cand["tau_N"], phi_target=cand["phi_target"], a_load=cand["a_load"],
                phi_pop_mean=pop_phi, phi_q99=float(np.percentile(phi_all, 99)),
                phi_max=float(phi_all.max()), u0_mean=float(u0.mean()), u0_max=float(u0.max()),
                u_final_mean=float(u_fin.mean()), u_final_max=float(u_fin.max()),
                equilibration_drift=drift, frac_divergent_cells=frac_div,
                cell_rise_median=vis["rise_median"], cell_quiet_median=vis["quiet_median"],
                visibility_ratio=vis["ratio"], n_events_scored=vis["n_events_scored"],
                mean_participation_frac=part_frac,
                population_mean_phi_sd=float(np.std(phi_mean_trace)),   # diagnostic, NOT the clause
                A1_visible=a1, A2_not_pinned=a2, A3_headroom=a3,
                admissible=bool(a1 and a2 and a3)), blk_phi, blk_spk, u_fin, u0


def _select_candidate(reports):
    ok = [r for r in reports if r["admissible"]]
    if not ok:
        return None
    ok.sort(key=lambda r: (-r["tau_N"], abs(r["phi_target"] - 0.15)))
    return ok[0]


# ----------------------------------------------------------------- sentinels / io
@contextmanager
def _stage_lock(tag):
    """Single-instance lock PER STAGE SHARD (same intent as the FCXR launcher lock, but keyed so
    that two independent shards -- e.g. the shaped and the uniform-control map -- may run side by
    side while a duplicate submission of the SAME shard still refuses."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f".lock_{tag}")
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"stage shard {tag} already running; refusing duplicate") from exc
        lock.seek(0); lock.truncate()
        lock.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n")
        lock.flush()
        try:
            yield path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _sentinel(run_dir, name, payload):
    FCXR._write_json(os.path.join(run_dir, name), payload)


def _provenance(extra=None):
    prov = dict(git_sha=FCXR._git_sha(), engine_shas=FCXR._engine_shas() if hasattr(FCXR, "_engine_shas")
                else {k: FCXR._sha(os.path.join(ROOT, k)) for k in json.load(open(FCXR.ENGINE_VERSIONS))},
                argv=sys.argv, dt=DT, conn_seed=CONN_SEED,
                spec="docs/superpowers/specs/2026-07-26-topic4-mz-fcxr-pump-lifecycle-design.md",
                base_commit="cf0d4d1768adb1fdc80c1cb5a9e5d9a963d47450")
    if extra:
        prov.update(extra)
    return prov


# ================================================================= p0-smoke
def cmd_smoke(a):
    from params import Params
    from connectivity import place_neurons, build_connectivity
    t0 = time.time()
    p = Params(L=6.0, density=100.0, T=200.0, dt=DT, nu_ext_ratio=0.9, seed=1)
    rng = np.random.default_rng(1)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
    N = NE + NI
    sites = np.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=sites)
    cand = dict(tau_N=PLUMBING_CANDIDATE["tau_N"],
                a_load=_a_load_for(PLUMBING_CANDIDATE["phi_target"], PLUMBING_CANDIDATE["tau_N"]))
    cfg = MZSlowVarsConfig(**_pump_cfg(sensor_only=True, a_load=cand["a_load"],
                                       tau_ms=cand["tau_N"], record_calibration=True))
    slow = MZSlowVars(N, 18.0, cfg, NE=NE, core_mask_E=np.zeros(NE, bool))
    slow.seeg_observer = PUMP.VirtualSeegComponentObserver(rec, cfg)
    net["rng"] = np.random.default_rng(NOISE_CALIB)
    res = simulate_kick(p, net, 0.0, slow=slow, kick_center=np.array([3.0, 3.0]), r_kick=0.5,
                        t_kick=1e9, V_th_per_neuron=np.full(N, 18.0), lfp_recorder=rec)
    off = PUMP.integrate_load_from_raster(res["E_spk_bool"], a_load=cand["a_load"],
                                          tau_N=cand["tau_N"], dt=DT)[0]
    tr = slow.seeg_observer.stack()
    print(f"[smoke:small] NE={NE} spikes={int(res['E_spk_bool'].sum())} "
          f"u_engine_max={slow.u_pump_E.max():.6g} u_offline_max={off.max():.6g} "
          f"identical={np.allclose(slow.u_pump_E, off, rtol=0, atol=0)}")
    print(f"[smoke:small] legacy_abs==lfp_trace: {np.array_equal(tr['legacy_abs'], res['lfp_trace'])}")
    if not a.small_only:
        S = _substrate()
        contacts, names = _montage(S)
        t1 = time.time()
        res2, slow2, obs2 = _run(S, _pump_cfg(sensor_only=True, a_load=cand["a_load"],
                                              tau_ms=cand["tau_N"], record_calibration=True),
                                 a.t_ms, noise_seed=NOISE_CALIB)
        wall = time.time() - t1
        n = len(res2["rate_E"])
        print(f"[smoke:L20] N={S['N']} contacts={len(names)} T={a.t_ms}ms wall={wall:.1f}s "
              f"per_step={wall / n * 1e3:.3f}ms  ->  8000ms ~= {wall / n * (8000 / DT) / 60:.1f} min")
        print(f"[smoke:L20] mean_rate={float(np.mean(res2['rate_E'])):.3f}Hz "
              f"u_mean={slow2.u_pump_E.mean():.4g} u_max={slow2.u_pump_E.max():.4g} "
              f"phi_mean={float(PUMP.pump_activation(slow2.u_pump_E).mean()):.4g}")
    print(f"[smoke] TOTAL {time.time() - t0:.1f}s")


# ================================================================= p0-baseline
def cmd_baseline(a):
    if not a.confirm_run:
        raise SystemExit("p0-baseline: pass --confirm-run to launch the calibration trajectory")
    FCXR._assert_engine_blessed()
    run_dir = OUT
    os.makedirs(run_dir, exist_ok=True)
    T = BURN_IN_MS + N_BLOCKS * BLOCK_MS
    with FCXR._launcher_lock():
        plan = FCXR._plan_workers(T, 1)
        FCXR._resource_log(run_dir, "p0_baseline_start", plan)
        _sentinel(run_dir, "RUNNING.json", dict(stage="p0-baseline", pid=os.getpid(), t_ms=T,
                  noise_seed=NOISE_CALIB, started=datetime.now(timezone.utc).isoformat(), plan=plan))
        with open(os.path.join(run_dir, "launcher.pid"), "w") as f:
            f.write(str(os.getpid()))
        try:
            S = _substrate()
            contacts, names = _montage(S)
            plumb = dict(tau_N=PLUMBING_CANDIDATE["tau_N"],
                         a_load=_a_load_for(PLUMBING_CANDIDATE["phi_target"],
                                            PLUMBING_CANDIDATE["tau_N"]))
            t1 = time.time()
            res, slow, obs = _run(S, _pump_cfg(sensor_only=True, a_load=plumb["a_load"],
                                               tau_ms=plumb["tau_N"], record_calibration=True),
                                  T, noise_seed=NOISE_CALIB)
            wall = time.time() - t1
            FCXR._resource_log(run_dir, "p0_baseline_sim_done", dict(wall_s=round(wall, 1)))
            traces = obs.stack()
            raster = res["E_spk_bool"]
            n_steps = raster.shape[0]
            blocks = _blocks(n_steps, DT)
            rate = np.asarray(res["rate_E"], float)

            # Event segmentation. The canonical amplitude bar is floor + 0.5*(af_max - floor) where
            # `floor` is the top of the QUIET baseline, estimated by the helper from the run's first
            # 5-50 ms (the silent post-reset startup). It MUST therefore be computed on the FULL
            # trajectory: applied to a post-burn-in slice, that 45 ms window can land on a burst and
            # the bar jumps to ~78% of the peak, so no interval survives the 8 ms minimum duration
            # and zero events are detected (superseded/ run, event_bar=0.0591 -> 0 events).
            b0 = blocks[0][0]
            event_bar = OLD.slowoff_event_bar(res, DT)
            events, af, bin_w, floor, _ = OLD._events_from_res(res, DT, event_bar=event_bar)
            ret = [e for e in events if e["returned"] and int(e["t_on"] / DT) >= b0]
            ev_steps = [(int(e["t_on"] / DT), int(e["t_off"] / DT)) for e in ret]
            ev_mask = np.zeros(n_steps, bool)
            for s, e in ev_steps:
                ev_mask[s:min(e + 1, n_steps)] = True
            quiet_steps = []
            i = b0
            while i < n_steps:
                if not ev_mask[i]:
                    j = i
                    while j < n_steps and not ev_mask[j]:
                        j += 1
                    if j - i > 20:
                        quiet_steps.append((i, j))
                    i = j
                else:
                    i += 1
            quiet_pairs = PUMP.matched_quiet_intervals(ev_steps, quiet_steps)
            print(f"[p0-baseline] events: n_detected={len(events)} n_returning_post_burnin={len(ret)} "
                  f"bar={event_bar:.5g} af_max={float(af.max()):.5g} floor={floor:.5g} "
                  f"n_quiet_segments={len(quiet_steps)} n_matched={sum(q is not None for q in quiet_pairs)}",
                  flush=True)

            # ---- load-candidate scan (offline; one simulation calibrates the whole set) ----
            rate_E_hz = raster[b0:].sum(axis=0) / ((n_steps - b0) * DT / 1000.0)
            reports, packs = [], {}
            for cand in _candidates():
                rep, blk_phi, blk_spk, u_fin, u0 = _candidate_report(
                    raster, cand, DT, blocks, ev_steps, quiet_steps, rate_E_hz, quiet_pairs)
                reports.append(rep)
                packs[(cand["tau_N"], cand["phi_target"])] = (blk_phi, blk_spk, u_fin, u0)
                print(f"[p0-baseline] cand tau={cand['tau_N']:.0f} phi*={cand['phi_target']:.2f} "
                      f"a={cand['a_load']:.4g} phi_pop={rep['phi_pop_mean']:.4g} "
                      f"q99={rep['phi_q99']:.4g} rise={rep['cell_rise_median']:.3g} "
                      f"quiet={rep['cell_quiet_median']:.3g} ratio={rep['visibility_ratio']:.3g} "
                      f"drift={rep['equilibration_drift']:.3g} adm={rep['admissible']}", flush=True)
            chosen = _select_candidate(reports)
            FCXR._resource_log(run_dir, "p0_baseline_candidates_done",
                               dict(n_admissible=sum(r["admissible"] for r in reports)))

            # ---- engine/offline cross-check on the plumbing candidate ----
            off_plumb = PUMP.integrate_load_from_raster(raster, a_load=plumb["a_load"],
                                                        tau_N=plumb["tau_N"], dt=DT)[0]
            cross = dict(max_abs_diff=float(np.max(np.abs(slow.u_pump_E - off_plumb))),
                         bitwise_identical=bool(np.array_equal(slow.u_pump_E, off_plumb)))

            # ---- p0 shrinkage for the chosen candidate ----
            p0_pack = None
            if chosen is not None:
                blk_phi, blk_spk, u_fin_c, u0_c = packs[(chosen["tau_N"], chosen["phi_target"])]
                block_rate_E = blk_spk.sum(axis=0) / (N_BLOCKS * BLOCK_MS / 1000.0)
                groups = PUMP.rate_decile_groups(block_rate_E)
                fit = PUMP.fit_p0_shrinkage(blk_phi, groups)
                src_m, snk_m, _ = _core_masks(S)
                p0 = fit["p0"]
                p0_pack = dict(weight=fit["weight"], cv_weights=fit["cv_weights"],
                               cv_error=fit["cv_error"],
                               p0_mean=float(p0.mean()), p0_sd=float(p0.std()),
                               p0_source_core_mean=float(p0[src_m].mean()),
                               p0_sink_core_mean=float(p0[snk_m].mean()),
                               p0_source_minus_sink=float(p0[src_m].mean() - p0[snk_m].mean()),
                               raw_sd=float(fit["raw_p0"].std()))
                FCXR._write_npz(os.path.join(run_dir, "p0_E.npz"), p0=p0, raw_p0=fit["raw_p0"],
                                group_p0=fit["group_p0"], groups=groups,
                                block_phi_mean=blk_phi, block_spike_count=blk_spk,
                                block_rate_E=block_rate_E, u_baseline_E=u_fin_c,
                                u_analytic_star_E=u0_c, rate_E_hz=rate_E_hz,
                                tau_N=chosen["tau_N"], a_load=chosen["a_load"])

            # ---- per-block primary metrics -> equivalence margins (locked BEFORE the held-out run) ----
            blk_metrics = [_block_metrics(res, traces, S, lo, hi, DT, event_bar) for lo, hi in blocks]
            margins = PUMP.block_equivalence_margins(
                [{k: v for k, v in b.items() if np.isfinite(v)} for b in blk_metrics], k=2.0)
            ied_budget = PUMP.required_ied_count([b["n_events"] for b in blk_metrics])
            audit = PUMP.component_audit(traces, DT)

            FCXR._write_json(os.path.join(run_dir, "pump_baseline_calibration.json"), dict(
                provenance=_provenance(dict(stage="p0-baseline", noise_seed=NOISE_CALIB, t_ms=T)),
                wall_s=round(wall, 1), n_steps=n_steps, blocks=blocks,
                mean_rate_hz=float(rate.mean()), n_events_post_burnin=len(ret),
                event_bar=float(event_bar), candidate_grid=reports, chosen_candidate=chosen,
                engine_offline_cross_check=cross, p0_shrinkage=p0_pack,
                ied_budget=ied_budget,
                selection_rule=dict(K_VISIBLE=K_VISIBLE, PHI_PINNED=PHI_PINNED,
                                    PHI_HEADROOM=list(PHI_HEADROOM),
                                    rule="largest tau_N among admissible; tie -> phi_target near 0.15",
                                    prelocked=True)))
            FCXR._write_json(os.path.join(run_dir, "baseline_variability.json"), dict(
                provenance=_provenance(dict(stage="p0-baseline")), block_metrics=blk_metrics,
                margins=margins, k_sd=2.0,
                note="margins are locked here, BEFORE the held-out pump-on trajectory is run"))
            FCXR._write_json(os.path.join(run_dir, "virtual_seeg_component_audit.json"), dict(
                provenance=_provenance(dict(stage="p0-baseline")), pump_on=False, audit=audit,
                identifiability=PUMP.readout_identifiability_note()))
            FCXR._write_npz(os.path.join(run_dir, "baseline_traces_noise201.npz"),
                            rate_E=rate.astype(np.float32), contacts=contacts,
                            names=np.array(names, object), dt=DT,
                            **{f"seeg_{k}": v.astype(np.float32) for k, v in traces.items()})
            _sentinel(run_dir, "DONE_p0_baseline.json", dict(
                stage="p0-baseline", wall_s=round(wall, 1), chosen=chosen,
                n_admissible=sum(r["admissible"] for r in reports),
                finished=datetime.now(timezone.utc).isoformat()))
            FCXR._resource_log(run_dir, "p0_baseline_saved")
            if os.path.exists(os.path.join(run_dir, "RUNNING.json")):
                os.remove(os.path.join(run_dir, "RUNNING.json"))
            print(f"[p0-baseline] wall={wall:.1f}s events={len(ret)} chosen={chosen}")
        except Exception as exc:
            _sentinel(run_dir, "FAILED.json", dict(stage="p0-baseline", error=repr(exc),
                      failed=datetime.now(timezone.utc).isoformat()))
            raise


# ================================================================= p0-equivalence
def cmd_equivalence(a):
    """FINAL held-out trajectory (noise 202): pump-off vs pump-on, judged ONCE against the margins
    that p0-baseline already locked. The held-out noise never touches grouping, shrinkage strength,
    the margins or any threshold, and nothing here is re-fitted."""
    if not a.confirm_run:
        raise SystemExit("p0-equivalence: pass --confirm-run to launch the held-out trajectories")
    FCXR._assert_engine_blessed()
    run_dir = OUT
    calib = json.load(open(os.path.join(run_dir, "pump_baseline_calibration.json")))
    var = json.load(open(os.path.join(run_dir, "baseline_variability.json")))
    chosen = calib["chosen_candidate"]
    if chosen is None:
        raise SystemExit("p0-equivalence: no admissible load candidate -- Gate I-a already fails")
    pack = np.load(os.path.join(run_dir, "p0_E.npz"))
    p0_E = np.asarray(pack["p0"], float)
    u_init = np.asarray(pack["u_baseline_E"], float)
    # Imax is anchored on the BASELINE inhibitory-current proxy (not tuned to any result): at full
    # ictal load the pump would add about as much outward current as baseline inhibition supplies.
    # The same value is carried into Gate T, so equivalence is tested at the Imax actually used.
    imax = float(a.imax) if a.imax > 0 else _imax_anchor(run_dir)
    T = BURN_IN_MS + N_BLOCKS * BLOCK_MS
    with FCXR._launcher_lock():
        plan = FCXR._plan_workers(T, 1)
        FCXR._resource_log(run_dir, "p0_equivalence_start", plan)
        _sentinel(run_dir, "RUNNING.json", dict(stage="p0-equivalence", pid=os.getpid(), t_ms=T,
                  noise_seed=NOISE_HELDOUT, Imax=imax,
                  started=datetime.now(timezone.utc).isoformat(), plan=plan))
        with open(os.path.join(run_dir, "launcher.pid"), "w") as f:
            f.write(str(os.getpid()))
        try:
            S = _substrate()
            contacts, names = _montage(S)
            bar = float(calib["event_bar"])
            per_arm, keep = {}, {}
            arms = [("pump_off", _pump_off_cfg()),
                    ("pump_on", _pump_cfg(sensor_only=False, a_load=chosen["a_load"],
                                          tau_ms=chosen["tau_N"], Imax=imax,
                                          p0_E=p0_E, record_calibration=True))]
            if a.reuse_off:
                # The pump-off arm carries no pump term at all, so it is byte-identical for every
                # Imax on the same noise seed. Reuse it rather than paying 16 min to reproduce it.
                prev = json.load(open(os.path.join(run_dir, a.reuse_off)))
                assert prev["provenance"]["noise_seed"] == NOISE_HELDOUT, "reuse: noise seed mismatch"
                per_arm["pump_off"] = prev["per_arm"]["pump_off"]
                prev_npz = os.path.join(run_dir, a.reuse_off.replace("pump_baseline_equivalence", "heldout_traces_noise202").replace(".json", ".npz"))
                pz = np.load(prev_npz, allow_pickle=True) if os.path.exists(prev_npz) else {}
                keep["pump_off"] = dict(blocks=prev["blocks"], reused_from=a.reuse_off,
                                        audit=prev["readout_audit"]["pump_off"],
                                        rate=np.asarray(pz["rate_off"], np.float32) if "rate_off" in pz
                                        else np.zeros(0, np.float32),
                                        traces={k[len("off_seeg_"):]: np.asarray(pz[k], float)
                                                for k in getattr(pz, "files", []) if k.startswith("off_seeg_")}
                                        or None)
                arms = [arms[1]]
                print(f"[p0-equivalence] reusing pump_off arm from {a.reuse_off}", flush=True)
            for label, cfg in arms:
                if label == "pump_on":
                    cfg["pump_u_init_E"] = u_init
                t1 = time.time()
                res, slow, obs = _run(S, cfg, T, noise_seed=NOISE_HELDOUT)
                wall = time.time() - t1
                FCXR._resource_log(run_dir, f"p0_equivalence_{label}_done", dict(wall_s=round(wall, 1)))
                traces = obs.stack()
                blocks = _blocks(len(res["rate_E"]), DT)
                # reduce to metrics INSIDE the loop and drop the raster: holding two full
                # n_steps x NE rasters at once would peak ~14 GB for no reason.
                bm = [_block_metrics(res, traces, S, lo, hi, DT, bar) for lo, hi in blocks]
                per_arm[label] = dict(block_metrics=bm, wall_s=round(wall, 1),
                                      pooled={k: float(np.nanmean([b[k] for b in bm])) for k in bm[0]})
                keep[label] = dict(rate=np.asarray(res["rate_E"], np.float32), traces=traces,
                                   audit=PUMP.component_audit(traces, DT), blocks=blocks)
                if label == "pump_on":
                    keep[label]["u_mean"] = np.asarray(slow.trace_u_mean, float)
                    keep[label]["excess_mean"] = np.asarray(slow.trace_pump_excess_mean, float)
                print(f"[p0-equivalence] {label} wall={wall:.1f}s "
                      f"mean_rate={float(np.mean(res['rate_E'])):.3f}Hz", flush=True)
                del res, slow, obs, traces

            blocks = keep["pump_off"]["blocks"]
            budget = PUMP.baseline_neutrality_budget(
                np.asarray(np.load(os.path.join(run_dir, "p0_E.npz"))["block_phi_mean"], float), p0_E)
            margins = var["margins"]
            eq = PUMP.evaluate_baseline_equivalence(per_arm["pump_off"]["pooled"],
                                                    per_arm["pump_on"]["pooled"], margins)
            u_tr = keep["pump_on"]["u_mean"]
            ex_tr = keep["pump_on"]["excess_mean"]
            audit_on, audit_off = keep["pump_on"]["audit"], keep["pump_off"]["audit"]
            FCXR._write_json(os.path.join(run_dir, a.equiv_out), dict(
                provenance=_provenance(dict(stage="p0-equivalence", noise_seed=NOISE_HELDOUT,
                                            t_ms=T, Imax=imax)),
                chosen_candidate=chosen, Imax=imax, imax_anchor=budget, blocks=blocks,
                per_arm={k: v for k, v in per_arm.items()}, equivalence=eq,
                held_out_load=dict(u_mean_start=float(u_tr[0]), u_mean_end=float(u_tr[-1]),
                                   u_drift_rel=float(abs(u_tr[-1] - u_tr[0]) / max(u_tr[0], 1e-12)),
                                   excess_mean=float(ex_tr.mean()) if ex_tr.size else None,
                                   excess_min=float(ex_tr.min()) if ex_tr.size else None,
                                   excess_max=float(ex_tr.max()) if ex_tr.size else None),
                readout_audit=dict(pump_on=audit_on, pump_off=audit_off),
                one_shot="margins were locked by p0-baseline and are NOT refitted here"))
            off_tr = keep["pump_off"].get("traces")
            FCXR._write_npz(os.path.join(run_dir, "heldout_traces_noise202.npz"),
                            rate_off=keep["pump_off"].get("rate", np.zeros(0, np.float32)),
                            rate_on=keep["pump_on"]["rate"],
                            u_mean_on=u_tr.astype(np.float32),
                            pump_excess_mean_on=ex_tr.astype(np.float32),
                            contacts=contacts, names=np.array(names, object), dt=DT,
                            **{f"on_seeg_{k}": v.astype(np.float32)
                               for k, v in keep["pump_on"]["traces"].items()},
                            **({f"off_seeg_{k}": v.astype(np.float32) for k, v in off_tr.items()}
                               if off_tr is not None else {}))
            _sentinel(run_dir, "DONE_p0_equivalence.json", dict(
                stage="p0-equivalence", all_within=eq["all_within"], n_outside=eq["n_outside"],
                finished=datetime.now(timezone.utc).isoformat()))
            FCXR._resource_log(run_dir, "p0_equivalence_saved")
            if os.path.exists(os.path.join(run_dir, "RUNNING.json")):
                os.remove(os.path.join(run_dir, "RUNNING.json"))
            print(f"[p0-equivalence] all_within={eq['all_within']} n_outside={eq['n_outside']}")
        except Exception as exc:
            _sentinel(run_dir, "FAILED.json", dict(stage="p0-equivalence", error=repr(exc),
                      failed=datetime.now(timezone.utc).isoformat()))
            raise


# ================================================================= p0-adjudicate
def cmd_adjudicate(a):
    import subprocess
    import src.topic4_mz_fcxr_pump_lifecycle as LC
    run_dir = OUT
    calib = json.load(open(os.path.join(run_dir, "pump_baseline_calibration.json")))
    eqp = os.path.join(run_dir, "pump_baseline_equivalence.json")
    equiv = json.load(open(eqp)) if os.path.exists(eqp) else None

    # ---- parity evidence: the frozen pre-edit fixture suite + the update-order/causal-order tests ----
    tests = ["tests/test_mz_full_conductance_spatial_relay.py",
             "tests/test_mz_slow_vars.py", "tests/test_topic4_mz_fcxr_pump.py",
             "tests/test_topic4_mz_fcxr_heo1.py", "tests/test_topic4_mz_fcxr_heo2.py",
             "tests/test_topic4_mz_fcxr_heo3.py"]
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests], cwd=ROOT,
                          capture_output=True, text=True)
    parity_pass = proc.returncode == 0
    recorded = json.load(open(FCXR.ENGINE_VERSIONS))
    hashes_ok = all(FCXR._sha(os.path.join(ROOT, k)) == v for k, v in recorded.items())
    named = {"zmx": "test_existing_ZMX_update_order_unchanged",
             "causal": "test_membrane_uses_pre_step_load_and_step_applies_the_jump_after"}
    named_ok = {}
    for key, name in named.items():
        pr = subprocess.run([sys.executable, "-m", "pytest", "-q", "-k", name, *tests],
                            cwd=ROOT, capture_output=True, text=True)
        named_ok[key] = pr.returncode == 0

    parity = dict(byte_parity_pass=parity_pass, blessed_hashes_match=hashes_ok,
                  zmx_update_order_pass=named_ok["zmx"], causal_order_pass=named_ok["causal"],
                  pytest_tail=proc.stdout.strip().splitlines()[-1] if proc.stdout else "")
    baseline = dict(candidate_admissible=calib["chosen_candidate"] is not None,
                    equivalence_all_within=None if equiv is None else equiv["equivalence"]["all_within"],
                    n_metrics_outside=None if equiv is None else equiv["equivalence"]["n_outside"])
    aud = (equiv["readout_audit"]["pump_on"] if equiv is not None
           else json.load(open(os.path.join(run_dir, "virtual_seeg_component_audit.json")))["audit"])
    note = PUMP.readout_identifiability_note()
    readout = dict(identifiability_status=note["status"],
                   identity_max_abs_err=max(aud["identity_all_minus_nodp_equals_pump_max_abs_err"],
                                            aud["identity_component_sum_max_abs_err"]),
                   band_power_pump=aud["band_power_pump"],
                   band_power_no_direct_pump=aud["band_power_no_direct_pump"])
    verdict = LC.adjudicate_gate_Ia(parity, baseline, readout)
    out = dict(provenance=_provenance(dict(stage="p0-adjudicate")), status=verdict["status"],
               verdict=verdict, parity=parity, baseline=baseline, readout=readout,
               identifiability=note,
               conclusion_language=LC.gate_conclusion_language({"Ia": verdict}))
    FCXR._write_json(os.path.join(run_dir, "gate_Ia.json"), out)
    print(f"[p0-adjudicate] Gate I-a = {verdict['status']}")
    for r in verdict.get("reasons", []):
        print(f"    - {r}")


# ================================================================= P1 (Gate T)
# Locked FCXR-HEO high-branch anchor (the Phase-0 "strongest sustained-16 Hz state" pick that HEO2
# and HEO3 both continued from): gate_quantile 0.999, cooperative gain A_c=8, frozen-Z depletion
# scale D=0.15. Nothing here re-tunes the substrate; only the pump field and D are varied.
ANCHOR_GQ, ANCHOR_A, ANCHOR_D = 0.999, 8.0, 0.15
D_GRID = (0.0, 0.08, 0.13, 0.15)                   # rho_Z axis: healthy -> anchor-impaired
RHO_U_GRID = (0.0, 0.33, 0.67, 1.0)                # activity-shaped field scale (NOT the abscissa)
MAP_T_MS = 2000.0                                  # per frozen cell (HIGH_MS=1000 needs >=1 s above band)
SENSOR_FIELD_T_MS = 3500.0
FIELD_SNAPSHOT_MS = (500.0, 1000.0, 2000.0, 3000.0)
KICK_HIGH = 12.0                                   # high-IC basin probe (HEO1 kick12)
T_KICK_MS = 120.0
SHUFFLE_SEED = 7001


def _anchor_uc():
    """u_c for the locked gate quantile, read from the accepted HEO1 F0 baseline contract."""
    src = os.path.join(FCXR.OUT_ROOT, "high_energy_oscillatory_branch",
                       f"baseline_spectral_contract_seed{CONN_SEED}.json")
    if not os.path.exists(src):     # worktree results/ may be sparse; fall back to the main checkout
        src = src.replace(ROOT, os.path.dirname(os.path.dirname(ROOT)))
    return float(json.load(open(src))["u_c"][str(ANCHOR_GQ)])


def _anchor_cfg(D, *, pump=None):
    """Arm-C + cooperative gate at the locked anchor, frozen Z at depletion scale D, plus optional
    pump fields. Requires the Stage-D onset-depletion field p_i for the seed."""
    from src.topic4_mz_fcxr_dynamics import frozen_z_field
    cfg = FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True, fail_on_clip=False,
                       rec_sat_g=G_SAT)
    u_c = _anchor_uc()
    cfg.update(coop_A=ANCHOR_A, coop_uc=u_c, coop_Kc=0.25 * u_c, coop_n=4)
    if D is not None and float(D) != 0.0:
        cfg["z_frozen_E"] = frozen_z_field(_anchor_cfg._p_i, float(D))
    if pump:
        cfg.update(pump)
    return cfg


def _load_pi(S):
    from src.topic4_mz_fcxr_dynamics import load_onset_depletion_pi, assert_field_substrate_aligned
    snap = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility",
                        "snapshots", "zA_q75_tz5000", f"seed_{CONN_SEED}.npz")
    pk = load_onset_depletion_pi(snap)
    assert_field_substrate_aligned(pk, S)
    _anchor_cfg._p_i = pk["p_i"]
    return pk["p_i"]


def _chosen_and_fields(run_dir):
    calib = json.load(open(os.path.join(run_dir, "pump_baseline_calibration.json")))
    chosen = calib["chosen_candidate"]
    if chosen is None:
        raise SystemExit("Gate T requires an admissible load candidate from p0-baseline")
    pack = np.load(os.path.join(run_dir, "p0_E.npz"))
    return chosen, np.asarray(pack["p0"], float), np.asarray(pack["u_baseline_E"], float)


def _imax_anchor(run_dir):
    """Pump strength = the BASELINE-NEUTRALITY BUDGET derived from the calibration trajectory.

    ANCHOR CORRECTION (2026-07-26). The first anchor equated Imax with the mean magnitude of the
    electrode-weighted inhibitory SEEG component (48.1). That is an aggregate of a different
    quantity, and at that strength the pump is not baseline-neutral even though its p0 compensation
    is exact in the mean: a cell's own activation still wanders by q95 ~ 0.044 on the block
    timescale, which at Imax=48 is a multi-second per-cell current of ~12% of spike threshold. The
    measured consequence is recorded in pump_baseline_equivalence_imax48.json -- fewer, longer, far
    more regular events and a +1.0 Hz mean-rate shift, i.e. a reorganised interictal train.

    The replacement bounds exactly that residual and is computed from the CALIBRATION trajectory
    alone, never from the held-out equivalence outcome (src.topic4_mz_fcxr_pump.baseline_neutrality_budget).
    """
    pack = np.load(os.path.join(run_dir, "p0_E.npz"))
    return float(PUMP.baseline_neutrality_budget(np.asarray(pack["block_phi_mean"], float),
                                                 np.asarray(pack["p0"], float))["imax_budget"])


def cmd_p1_sensor_field(a):
    """Activity-shaped load fields on the SENSOR-ONLY established-high branch (Imax=0, spec §T1)."""
    if not a.confirm_run:
        raise SystemExit("p1-sensor-field: pass --confirm-run")
    FCXR._assert_engine_blessed()
    run_dir = OUT
    chosen, p0_E, u0 = _chosen_and_fields(run_dir)
    with _stage_lock("p1_sensor_field"):
        FCXR._resource_log(run_dir, "p1_sensor_field_start", FCXR._plan_workers(SENSOR_FIELD_T_MS, 1))
        _sentinel(run_dir, "RUNNING_p1_sensor_field.json", dict(stage="p1-sensor-field", pid=os.getpid(),
                  started=datetime.now(timezone.utc).isoformat()))
        try:
            S = _substrate()
            _load_pi(S)
            snap_steps = {int(round((T_KICK_MS + ms) / DT)): f"t{int(ms)}ms" for ms in FIELD_SNAPSHOT_MS}
            cfg = _anchor_cfg(ANCHOR_D, pump=dict(
                use_pump=True, pump_sensor_only=True, pump_a_load=chosen["a_load"],
                pump_tau_ms=chosen["tau_N"], pump_h=PUMP.PRIMARY_H, pump_u_init_E=u0))
            p = dataclasses.replace(S["p"], T=SENSOR_FIELD_T_MS, dt=DT)
            mzcfg = MZSlowVarsConfig(**cfg)
            slow = MZSlowVars(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=OLD.build_core_masks(S),
                              snapshot_steps=snap_steps)
            contacts, names = _montage(S)
            rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
            S["net"]["rng"] = np.random.default_rng(NOISE_CALIB)
            t1 = time.time()
            res = simulate_kick(p, S["net"], KICK_HIGH, slow=slow, kick_center=list(S["src_xy"]),
                                r_kick=PP.R_KICK, t_kick=T_KICK_MS, V_th_per_neuron=S["vth"],
                                lfp_recorder=rec, early_stop_runaway=False)
            wall = time.time() - t1
            rate = np.asarray(res["rate_E"], float)
            fields = {lab: slow.snapshots[lab]["u_E"] for lab in snap_steps.values()}
            rows = {lab: dict(P=PUMP.mean_excess_pump_activation(f, p0_E),
                              u_mean=float(f.mean()), u_max=float(f.max()),
                              phi_mean=float(PUMP.pump_activation(f).mean()))
                    for lab, f in fields.items()}
            FCXR._write_npz(os.path.join(run_dir, "p1_sensor_fields.npz"),
                            u_baseline=u0, p0=p0_E, rate_E=rate.astype(np.float32), dt=DT,
                            **{f"u_high_{lab}": f for lab, f in fields.items()})
            FCXR._write_json(os.path.join(run_dir, "p1_sensor_field.json"), dict(
                provenance=_provenance(dict(stage="p1-sensor-field")), wall_s=round(wall, 1),
                anchor=dict(gate_quantile=ANCHOR_GQ, A_c=ANCHOR_A, D=ANCHOR_D, kick=KICK_HIGH,
                            t_kick_ms=T_KICK_MS, u_c=_anchor_uc()),
                candidate=chosen, mean_rate_hz=float(rate.mean()),
                post_kick_rate_hz=float(rate[int(T_KICK_MS / DT):].mean()),
                end_rate_hz=float(rate[-int(200 / DT):].mean()), fields=rows))
            _sentinel(run_dir, "DONE_p1_sensor_field.json", dict(stage="p1-sensor-field",
                      wall_s=round(wall, 1), fields=rows,
                      finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(run_dir, "RUNNING_p1_sensor_field.json")):
                os.remove(os.path.join(run_dir, "RUNNING_p1_sensor_field.json"))
            print(f"[p1-sensor-field] wall={wall:.1f}s post_kick_rate="
                  f"{float(rate[int(T_KICK_MS / DT):].mean()):.2f}Hz fields={rows}")
        except Exception as exc:
            _sentinel(run_dir, "FAILED_p1_sensor_field.json", dict(stage="p1-sensor-field",
                      error=repr(exc), failed=datetime.now(timezone.utc).isoformat()))
            raise


def _map_cell(args):
    """One frozen (D, rho_u, field, IC) cell. Frozen load: a_load=0 and an effectively infinite
    release time hold u at the injected field (drift < 1e-8 over the window), so the fast system
    sees a STATIC per-cell pump current Imax*(phi(u_i)-p0_i)."""
    S, D, rho_u, field_kind, ic, chosen, p0_E, u0, u_high, Imax, i_th = args
    if field_kind == "shaped":
        u_f = PUMP.frozen_load_field(u0, u_high, rho_u)
    elif field_kind == "uniform":
        u_f = PUMP.matched_uniform_field(PUMP.frozen_load_field(u0, u_high, rho_u), p0_E)
    else:
        u_f = PUMP.value_matched_shuffle_field(PUMP.frozen_load_field(u0, u_high, rho_u),
                                               np.random.default_rng(SHUFFLE_SEED))
    cfg = _anchor_cfg(D, pump=dict(use_pump=True, pump_sensor_only=False, pump_a_load=0.0,
                                   pump_tau_ms=1e12, pump_Imax=float(Imax), pump_h=PUMP.PRIMARY_H,
                                   pump_p0_E=p0_E, pump_u_init_E=u_f))
    p = dataclasses.replace(S["p"], T=MAP_T_MS, dt=DT)
    mzcfg = MZSlowVarsConfig(**cfg)
    slow = MZSlowVars(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
    contacts, _ = _montage(S)
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
    obs = PUMP.VirtualSeegComponentObserver(rec, mzcfg, z_threshold=i_th)
    slow.seeg_observer = obs
    S["net"]["rng"] = np.random.default_rng(NOISE_CALIB)
    t0 = time.time()
    kick = KICK_HIGH if ic == "high" else 0.0
    res = simulate_kick(p, S["net"], kick, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=(T_KICK_MS if ic == "high" else 1e9),
                        V_th_per_neuron=S["vth"], lfp_recorder=rec, early_stop_runaway=True,
                        es_thresh_hz=250.0, es_dur_ms=100.0)
    rate = np.asarray(res["rate_E"], float)
    from src.topic4_mz_fcxr_dynamics import workpoint_metrics, classify_run_workpoint
    num = dict(finite=bool(np.all(np.isfinite(rate))),
               clip_frac_max=float(np.max(slow.trace_conductance_clip_frac) if slow.trace_conductance_clip_frac else 0.0),
               runaway_early_stop_ms=res.get("runaway_early_stop_ms"))
    num["numerical_unsafe"] = bool((not num["finite"]) or num["clip_frac_max"] > 0.0)
    wm = workpoint_metrics(rate, DT, _map_cell.roll_hi, analysis_start_ms=T_KICK_MS + 100.0)
    label = classify_run_workpoint(dict(numerical_unsafe=num["numerical_unsafe"], **wm))
    if num["runaway_early_stop_ms"] is not None:
        # The early stop truncates the window, so the persistence test would silently mis-read an
        # unbounded escape as "not sustained". Label it for what it is: a high branch with no bound.
        label = "OPERATIONAL_RUNAWAY"
    a0 = int((T_KICK_MS + 100.0) / DT)
    rate_cell = res["E_spk_bool"][a0:].sum(axis=0) / ((len(rate) - a0) * DT / 1000.0)
    flow = PUMP.branch_slow_flow(rate_cell, u_f, p0_E, slow.z[:S["NE"]], obs.frac_z_inf_high(),
                                 a_load=chosen["a_load"], tau_N=chosen["tau_N"],
                                 tau_z=mzcfg.tau_z)
    return dict(D=float(D), rho_u=float(rho_u), field=field_kind, ic=ic,
                P=PUMP.mean_excess_pump_activation(u_f, p0_E), Z=float(np.mean(slow.z[:S["NE"]])),
                Imax=float(Imax), label=label, workpoint=wm, numerical=num,
                mean_rate_hz=float(rate.mean()),
                plateau_hz=float(np.mean(rate[-int(300 / DT):])),
                slow_flow=flow, wall_s=round(time.time() - t0, 1))


def cmd_p1_map(a):
    """Frozen rho_Z x mean-excess-pump-activation topology map (spec §T2 + §T3)."""
    if not a.confirm_run:
        raise SystemExit("p1-map: pass --confirm-run")
    FCXR._assert_engine_blessed()
    run_dir = OUT
    chosen, p0_E, u0 = _chosen_and_fields(run_dir)
    Imax = a.imax if a.imax > 0 else _imax_anchor(run_dir)
    pack = np.load(os.path.join(run_dir, "p1_sensor_fields.npz"))
    u_high = np.asarray(pack[f"u_high_{a.field_label}"], float)
    fields = [f.strip() for f in a.fields.split(",") if f.strip()]
    tag = "p1_map_" + "_".join(fields)
    with _stage_lock(tag):
        plan = FCXR._plan_workers(MAP_T_MS, a.workers)
        FCXR._resource_log(run_dir, "p1_map_start", dict(plan, fields=fields, Imax=Imax))
        _sentinel(run_dir, f"RUNNING_{tag}.json", dict(stage="p1-map", pid=os.getpid(), Imax=Imax,
                  fields=fields, started=datetime.now(timezone.utc).isoformat(), plan=plan))
        with open(os.path.join(run_dir, f"{tag}.pid"), "w") as f:
            f.write(str(os.getpid()))
        try:
            S = _substrate()
            _load_pi(S)
            base = json.load(open(os.path.join(FCXR.OUT_ROOT, "high_energy_oscillatory_branch",
                                               f"baseline_spectral_contract_seed{CONN_SEED}.json")))
            _map_cell.roll_hi = float(base["rate_roll_hi"])
            i_th = MZSlowVarsConfig(**_pump_off_cfg()).I_th_EI
            tasks = [(S, D, r, fk, ic, chosen, p0_E, u0, u_high, Imax, i_th)
                     for fk in fields for D in D_GRID for r in RHO_U_GRID for ic in ("low", "high")]
            out_path = os.path.join(run_dir, f"frozen_topology_map_{'_'.join(fields)}.json")
            rows = []
            t0 = time.time()
            for k, t in enumerate(tasks):
                rows.append(_map_cell(t))
                r = rows[-1]
                print(f"[p1-map] {k + 1}/{len(tasks)} {r['field']} D={r['D']:.2f} rho={r['rho_u']:.2f}"
                      f" P={r['P']:.4f} ic={r['ic']} -> {r['label']} plateau={r['plateau_hz']:.2f}Hz"
                      f" dP={r['slow_flow']['dP_dt']:.3g} dZ={r['slow_flow']['dZ_dt']:.3g}"
                      f" ({r['wall_s']:.0f}s)", flush=True)
                FCXR._write_json(out_path, dict(
                    provenance=_provenance(dict(stage="p1-map")), Imax=Imax, candidate=chosen,
                    anchor=dict(gate_quantile=ANCHOR_GQ, A_c=ANCHOR_A, u_c=_anchor_uc()),
                    field_label=a.field_label, D_grid=list(D_GRID), rho_u_grid=list(RHO_U_GRID),
                    T_ms=MAP_T_MS, roll_hi=_map_cell.roll_hi, n_done=len(rows), n_total=len(tasks),
                    cells=rows))
                FCXR._resource_log(run_dir, "p1_map_cell", dict(k=k + 1, n=len(tasks)))
            _sentinel(run_dir, f"DONE_{tag}.json", dict(stage="p1-map", n_cells=len(rows),
                      wall_s=round(time.time() - t0, 1), fields=fields,
                      finished=datetime.now(timezone.utc).isoformat()))
            if os.path.exists(os.path.join(run_dir, f"RUNNING_{tag}.json")):
                os.remove(os.path.join(run_dir, f"RUNNING_{tag}.json"))
            print(f"[p1-map] done {len(rows)} cells in {time.time() - t0:.0f}s")
        except Exception as exc:
            _sentinel(run_dir, f"FAILED_{tag}.json", dict(stage="p1-map", error=repr(exc),
                      failed=datetime.now(timezone.utc).isoformat()))
            raise


# ================================================================= entry
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["p0-smoke", "p0-baseline", "p0-equivalence", "p0-adjudicate",
                             "p1-sensor-field", "p1-map"])
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--t-ms", type=float, default=1000.0)
    ap.add_argument("--imax", type=float, default=0.0,
                    help="0 -> anchor on the baseline inhibitory-current proxy")
    ap.add_argument("--fields", default="shaped", help="comma list: shaped,uniform,shuffle")
    ap.add_argument("--field-label", default="t2000ms", help="which sensor-field snapshot to use")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reuse-off", default="", help="reuse the pump-off arm from this equivalence JSON")
    ap.add_argument("--equiv-out", default="pump_baseline_equivalence.json")
    ap.add_argument("--small-only", action="store_true")
    a = ap.parse_args(argv)
    {"p0-smoke": cmd_smoke, "p0-baseline": cmd_baseline,
     "p0-equivalence": cmd_equivalence, "p0-adjudicate": cmd_adjudicate,
     "p1-sensor-field": cmd_p1_sensor_field, "p1-map": cmd_p1_map}[a.stage](a)


if __name__ == "__main__":
    main()
