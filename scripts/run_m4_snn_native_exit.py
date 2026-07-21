#!/usr/bin/env python
"""Stage-1a exit-boundary probe -- SNN-native M4 containment-to-exit line.

CHEAP-FIRST GATE (spec 2026-07-21 §13). Open-loop, ZERO engine change: fork the accepted
M4 bounded state (k_q=0.10, alpha_G=16) and apply a clamped threshold displacement
(`inhibitory_pulse`: raise E V_th by DVTH) over a window [t0, t1], then observe the
post-release [t1, T] behaviour. Tests the q_I-refill exit hypothesis (spec §13, §5):

  short hold  -> q_I still depleted at release -> re-ignites (rebound / runaway)   [known: M4 500ms pulse rebounds]
  long  hold  -> firing stops, q_I REFILLS toward 1 over tau_q(=5000ms) during the quiet ->
                 on release the network is in a high-q_I interictal-like basin -> stays low.

If NO hold gives clean exit-and-stay for any reachable displacement => bounded-negative gate (stop).
If a hold does => calibrates tau_p (recovery hold must be ~ q_I refill time) for the dynamic field.

Reuses run_m4_phaseplane.build_substrate + run_m4_dynamic_qi.run_arm + sef_hfo_m4_termination.
Build ONCE, Pool over cells (COW). OMP forced to 1 by run_m4_dynamic_qi import. Gated by --confirm-run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_m4_phaseplane as PP          # noqa: E402
import run_m4_dynamic_qi as M4          # noqa: E402  (forces OMP=1 at import; provides run_arm/_S/_EARLY_STOP/DT + helpers)
from kick_probe import simulate_kick    # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.sef_hfo_m4_termination import classify_termination  # noqa: E402

BASE_KQ, BASE_AG = 0.10, 16.0
ARR_KEYS = ("trace_qI_mean", "trace_SG", "trace_Irec", "rate", "af", "movie", "q_field_final")


def _sanitize(obj):
    """Recursively replace non-finite floats with None so json.dump(allow_nan=False) never raises
    (spec §12: strict JSON, non-finite -> null). NaN/inf can arise from empty readout windows."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _levels(r, t0, t1, T):
    """Bounded pre-pulse level, in-hold level, post-release level, and q_I refill readout."""
    af = np.asarray(r["af"], float)
    bw = float(r["bin_w"])
    qi = np.asarray(r["trace_qI_mean"], float)   # per-step mean q_I (dt = M4.DT)
    dt = float(M4.DT)

    def af_win(a_ms, b_ms):
        i0, i1 = max(0, int(a_ms / bw)), min(af.size, int(b_ms / bw))
        m = float(af[i0:i1].mean()) if i1 > i0 else None      # empty window (e.g. baseline post) -> None, not nan
        return round(m, 5) if m is not None else None

    def qi_at(ms):
        return round(float(qi[min(qi.size - 1, max(0, int(ms / dt)))]), 4)

    return dict(
        pre_af=af_win(t0 - 1000, t0),
        hold_af=af_win(max(t0, t1 - 500), t1),
        post_af=af_win(t1 + 500, T),
        qI_t0=qi_at(t0), qI_t1=qi_at(t1), qI_final=round(float(qi[-1]), 4),
    )


def _verdict(r, lv, t1):
    """Post-release verdict: did the bounded state stay down after the hold released?"""
    ra = r.get("runaway_ms")
    if ra is not None and ra > t1 - 50.0:
        return "rebound_runaway"
    pre, post = lv["pre_af"], lv["post_af"]
    if not pre or pre <= 1e-9:
        return "no_bounded_pre"
    if post is None:
        return "partial"
    frac = post / pre
    if frac >= 0.5:
        return "rebound_bounded"
    if frac < 0.2:
        return "exit_stay_low"
    return "partial"


def _cell_worker(cell):
    label, perturb, t0, t1, T_ms = cell
    S = M4._S["S"]
    try:
        r = M4.run_arm(S, label, BASE_KQ, True, BASE_AG, perturb=perturb, T_ms=T_ms)
    except Exception as e:  # fail-loud per cell; don't kill the sweep
        return dict(label=label, error=repr(e)), None
    lv = _levels(r, t0, t1, T_ms) if perturb is not None else _levels(r, t0, T_ms, T_ms)
    cls, info = classify_termination(np.asarray(r["af"], float), float(r["bin_w"]),
                                     baseline=r["baseline_af"], runaway_ms=r.get("runaway_ms"))
    row = dict(
        label=label, seed=S["seed"], k_q=BASE_KQ, alpha_G=BASE_AG,
        t0_ms=t0, t1_ms=t1, hold_ms=(t1 - t0 if perturb is not None else 0.0), T_ms=T_ms,
        dvth=(perturb["val"] if perturb else 0.0),
        verdict=(_verdict(r, lv, t1) if perturb is not None else "baseline_" + r["verdict"]),
        termination_class=cls, offset_ms=info["offset_ms"],
        m4_verdict=r["verdict"], runaway_ms=r.get("runaway_ms"), max_rate_hz=r["max_rate_hz"],
        q_min_final=r["q_min_final"], q_mean_final=r["q_mean_final"], S_G_max=r["S_G_max"],
        active_area_peak=r.get("active_area_peak"), active_area_tail=r.get("active_area_tail"),
        wall_s=r["wall_s"], **lv,
    )
    arrays = {k: np.asarray(r[k], np.float32) for k in ARR_KEYS if k in r}
    return row, arrays


# ===========================================================================================
# Stage-2 dynamic arms (spec §8): the M4 base (q_I depletion + S_G pool) + the persistence-gated
# recovery field p(x,t). Reuses the M4 readout helpers (M4.C / _smooth / _first_sustained /
# _spatial_*) so labels are directly comparable to run_m4_dynamic_qi. Persist params via CLI.
# ===========================================================================================
def _persist_cfg(*, k_q=BASE_KQ, use_SG=True, alpha_G=BASE_AG, use_persist=False,
                 tau_p=5000.0, theta_p=0.0, a50_p=1.0, sigma_p=1.5, eta_r=0.0,
                 p50_r=0.0, n_r=2.0, clamp_persist=None, tau_p_down=None, persist_onset_ms=0.0):
    """M4 base config (k_q depletion + S_G pool, params from run_m4_dynamic_qi) + persistence field."""
    return SpatialSlowFieldConfig(
        use_qI=True, k_q=k_q, sigma_q=M4.SIGMA_Q, sigma_K=0.5, q_min=M4.Q_MIN, q_init=1.0,
        tau_q=M4.TAU_Q, tau_a=M4.TAU_A, use_gK=False, k_K=0.0,
        use_SG=use_SG, alpha_G=alpha_G, r0_psi=0.0, r50_psi=M4.R50_PSI, n_psi=M4.N_PSI,
        p_pool=M4.P_POOL, tau_mu=M4.TAU_MU, tau_S=M4.TAU_S, S_max=M4.S_MAX,
        use_persist=use_persist, tau_p=tau_p, theta_p=theta_p, a50_p=a50_p, sigma_p=sigma_p,
        eta_r=eta_r, p50_r=p50_r, n_r=n_r, clamp_persist=clamp_persist, tau_p_down=tau_p_down,
        persist_onset_ms=persist_onset_ms)


def _run_persist_arm(S, label, cfg, T_ms, perturb=None):
    """One spontaneous (KICK_BOOST=0) arm with the persistence field; full readout + termination class."""
    p = dataclasses.replace(S["p"], T=float(T_ms))
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], cfg=cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    t0 = time.time()
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], perturb=perturb,
                        early_stop_runaway=M4._EARLY_STOP["on"])  # runaway (>120Hz sustained) truncates; clean-exit/bounded unaffected
    spk = res["E_spk_bool"]
    rate = np.asarray(res["rate_E"], float)
    af, bin_w = M4.C.active_fraction(spk, M4.DT, M4.C.BIN_MS)
    nb0, nb1 = int(M4.C.BASELINE_MS[0] / bin_w), int(M4.C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + M4.C.CAL_FRAC * (float(af.max()) - floor)
    events = M4.C.detect_events(af, bin_w, event_on_frac=bar)
    rate_s = M4._smooth(rate, M4.DT)
    runaway = M4._first_sustained(rate_s, M4.DT)
    n_pre = sum(1 for e in events if runaway is None or e["t_on"] < runaway - 20.0)
    verdict = ("no_runaway" if runaway is None else "train_then_runaway" if (n_pre >= 2 and runaway > 200.0)
               else "one_shot_burst" if (runaway <= 200.0 or n_pre == 0) else "few_events_then_runaway")
    cls, info = classify_termination(af, bin_w, baseline=floor, runaway_ms=runaway)
    movie = M4._spatial_movie(spk, S["posE"], S["L"], M4.DT)
    row = dict(
        label=label, seed=S["seed"], verdict=verdict, termination_class=cls, offset_ms=info["offset_ms"],
        runaway_ms=runaway, max_rate_hz=round(float(rate_s.max()), 1), n_events=len(events), n_pre_runaway=int(n_pre),
        q_min_final=round(float(slow.q_I.min()), 4), q_mean_final=round(float(slow.q_I.mean()), 4),
        S_G_max=round(float(max(slow.trace_SG)) if slow.trace_SG else 0.0, 4),
        p_mean_final=round(float(slow.p.mean()), 4), p_max_final=round(float(slow.p.max()), 4),
        p_peak=round(float(max(slow.trace_p_max)) if slow.trace_p_max else 0.0, 4),
        T_ms=float(T_ms), perturb_kind=(perturb["kind"] if perturb else None),
        cfg_effective=dict(use_persist=cfg.use_persist, tau_p=cfg.tau_p, tau_p_down=cfg.tau_p_down,
                           theta_p=cfg.theta_p, a50_p=cfg.a50_p, sigma_p=cfg.sigma_p, eta_r=cfg.eta_r,
                           p50_r=cfg.p50_r, n_r=cfg.n_r, k_q=cfg.k_q, use_SG=cfg.use_SG, alpha_G=cfg.alpha_G),
        wall_s=round(time.time() - t0, 1), **M4._spatial_coverage(movie),
        events=[(round(e["t_on"], 1), round(e["t_off"], 1)) for e in events],
    )
    arrays = dict(
        trace_qI_mean=np.asarray(slow.trace_qI_mean, np.float32),
        trace_SG=(np.asarray(slow.trace_SG, np.float32) if slow.trace_SG else np.zeros(0, np.float32)),
        trace_p_mean=(np.asarray(slow.trace_p_mean, np.float32) if slow.trace_p_mean else np.zeros(0, np.float32)),
        trace_p_max=(np.asarray(slow.trace_p_max, np.float32) if slow.trace_p_max else np.zeros(0, np.float32)),
        rate=rate.astype(np.float32), af=af.astype(np.float32), movie=movie,
        q_field_final=slow.q_I.astype(np.float32), p_field_final=slow.p.astype(np.float32),
    )
    return row, arrays


def _arm_worker(spec):
    label, cfg, T_ms, perturb = spec
    S = M4._S["S"]
    try:
        return _run_persist_arm(S, label, cfg, T_ms, perturb=perturb)
    except Exception as e:                                # fail-loud per arm
        return dict(label=label, error=repr(e)), None


def _build_arms(a):
    """Arms A-E (spec §8), or a D-only (tau_p:eta_r) sweep. P = calibrated persistence params (from Stage-1)."""
    P = dict(tau_p=a.tau_p, theta_p=a.theta_p, a50_p=a.a50_p, sigma_p=a.sigma_p, p50_r=a.p50_r, n_r=a.n_r,
             tau_p_down=a.tau_p_down, persist_onset_ms=a.persist_onset_ms)
    T = a.T
    base = dict(k_q=BASE_KQ, use_SG=True, alpha_G=BASE_AG)
    if a.d_sweep:                          # arm-D (tau_p:eta_r) grid, one build shared across cells
        cells = []
        if a.include_anchor:
            cells.append(("B_m4_anchor", _persist_cfg(**base), T, None))
        for tok in a.d_sweep.split(","):
            tp, er = (float(x) for x in tok.split(":"))
            # build from P (single source of persist params incl tau_p_down) with tau_p overridden per cell,
            # so a param can never be silently dropped again. Label encodes tau_p_down when asymmetric.
            lab = f"D_tau{int(tp)}_eta{er:g}" + (f"_dn{int(a.tau_p_down)}" if a.tau_p_down else "")
            cells.append((lab, _persist_cfg(**base, use_persist=True, eta_r=er, **{**P, "tau_p": tp}), T, None))
        return cells
    catalog = {
        "A_slow_off":  (_persist_cfg(k_q=0.0, use_SG=False), T),
        "A_sensor_on": (_persist_cfg(k_q=0.0, use_SG=False, use_persist=True, eta_r=0.0, **P), T),  # p on real IEDs, actuator off
        "A_persist_act": (_persist_cfg(k_q=0.0, use_SG=False, use_persist=True, eta_r=a.eta_r, **P), T),  # slow-off + candidate actuator ON -> do real IEDs survive? (prevention test)
        "B_m4_anchor": (_persist_cfg(**base), T),
        "C_sensor_on": (_persist_cfg(**base, use_persist=True, eta_r=0.0, **P), T),   # p evolves, actuator off
        "D_full":      (_persist_cfg(**base, use_persist=True, eta_r=a.eta_r, **P), T),
        "E1_no_qI":    (_persist_cfg(k_q=0.0, use_SG=True, alpha_G=BASE_AG, use_persist=True, eta_r=a.eta_r, **P), T),
        "E2_no_SG":    (_persist_cfg(k_q=BASE_KQ, use_SG=False, use_persist=True, eta_r=a.eta_r, **P), T),
        "E4_clamp_p":  (_persist_cfg(**base, use_persist=True, eta_r=a.eta_r, clamp_persist=a.clamp_val, **P), T),
    }
    want = [x for x in a.arms.split(",") if x]
    return [(name, catalog[name][0], catalog[name][1], None) for name in want if name in catalog]


def _run_arms(a):
    os.makedirs(a.out, exist_ok=True)
    M4._EARLY_STOP["on"] = a.early_stop     # runaway arms truncate (still set runaway_ms); clean-exit/bounded run full
    t_build = time.time()
    S = PP.build_substrate(a.seed)
    S["p"].T = a.T
    M4._S["S"] = S
    with open(os.path.join(a.out, f"pids_{a.tag}_seed{a.seed}.txt"), "w") as f:
        f.write(f"{os.getpid()}\n")
    specs = _build_arms(a)
    print(f"[arms] N={S['N']} seed={a.seed} tau_p={a.tau_p} theta_p={a.theta_p} a50_p={a.a50_p} "
          f"eta_r={a.eta_r} sigma_p={a.sigma_p} arms={[s[0] for s in specs]} T={a.T} "
          f"workers={a.workers} build={time.time()-t_build:.0f}s", flush=True)
    t_run = time.time()
    with mp.Pool(min(a.workers, len(specs))) as pool:
        results = pool.map(_arm_worker, specs)
    rows = [r for r, _ in results]
    tag = f"{a.tag}_seed{a.seed}"
    np.savez_compressed(os.path.join(a.out, f"arms_{tag}.npz"),          # npz FIRST (survives a json failure)
                        posE=S["posE"].astype(np.float32), src_xy=S["src_xy"], snk_xy=S["snk_xy"], L=float(S["L"]),
                        **{f"{r['label']}__{k}": arr for (r, arrs) in results if arrs for k, arr in arrs.items()})
    json.dump(_sanitize(dict(meta=dict(seed=a.seed, tau_p=a.tau_p, theta_p=a.theta_p, a50_p=a.a50_p, eta_r=a.eta_r,
                                       sigma_p=a.sigma_p, clamp_val=a.clamp_val, T=a.T, base_kq=BASE_KQ, base_ag=BASE_AG,
                                       N=int(S["N"]), axis_unit=S["axis_unit"].tolist(),
                                       wall_s=round(time.time() - t_run, 1), argv=" ".join(sys.argv)),
                             rows=rows)),
              open(os.path.join(a.out, f"arms_{tag}.json"), "w"), indent=2, allow_nan=False)
    print("\n===== Stage-2 dynamic arms =====", flush=True)
    for r in rows:
        if "error" in r:
            print(f"  {r['label']:14s} ERROR {r['error']}", flush=True)
            continue
        print(f"  {r['label']:14s} verdict={r['verdict']:18s} cls={r['termination_class']:15s} "
              f"n_ev={r['n_events']:2d} maxHz={r['max_rate_hz']:6.1f} qmin={r['q_min_final']:.2f} "
              f"SGmax={r['S_G_max']:.2f} p_peak={r['p_peak']:.2f} area_tail={r.get('active_area_tail')}", flush=True)
    print(f"wrote arms_{tag}.json + .npz to {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--mode", default="exit_atlas", choices=["exit_atlas", "arms"],
                    help="exit_atlas = Stage-1a inhibitory-pulse hold sweep; arms = Stage-2 dynamic persistence arms")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t0", type=float, default=3000.0, help="branch time (bounded state settled)")
    ap.add_argument("--dvth", type=float, default=15.0, help="inhibitory_pulse V_th raise (mV)")
    ap.add_argument("--holds", default="500,3000,6000", help="comma-sep hold durations (ms)")
    ap.add_argument("--post-obs", type=float, default=3000.0, help="post-release observation (ms)")
    ap.add_argument("--baseline", default=True, action=argparse.BooleanOptionalAction,
                    help="exit_atlas: include an unperturbed bounded-reference cell (--no-baseline to skip)")
    # ---- Stage-2 arms mode: persistence-field calibration (from Stage-1) ----
    ap.add_argument("--T", type=float, default=15000.0, help="arms mode: spontaneous window (ms)")
    ap.add_argument("--arms", default="A_slow_off,B_m4_anchor,C_sensor_on,D_full",
                    help="arms mode: comma list of A_slow_off,B_m4_anchor,C_sensor_on,D_full,E1_no_qI,E2_no_SG,E4_clamp_p")
    ap.add_argument("--d-sweep", dest="d_sweep", default=None,
                    help="arms mode: D-only 'tau_p:eta_r' grid (comma-sep), one build shared, e.g. '5000:30,8000:50'")
    ap.add_argument("--include-anchor", dest="include_anchor", default=False, action=argparse.BooleanOptionalAction,
                    help="d-sweep: also run B_m4_anchor as the un-terminated reference")
    ap.add_argument("--tau-p", dest="tau_p", type=float, default=5000.0)
    ap.add_argument("--tau-p-down", dest="tau_p_down", type=float, default=None,
                    help="asymmetric p decay time (ms); None -> symmetric. Fast charge (tau_p) + slow decay = long hold")
    ap.add_argument("--persist-onset-ms", dest="persist_onset_ms", type=float, default=0.0,
                    help="established-state fork: p inactive until this t (ms) so the M4 state forms first, then engages")
    ap.add_argument("--theta-p", dest="theta_p", type=float, default=0.0)
    ap.add_argument("--a50-p", dest="a50_p", type=float, default=1.0)
    ap.add_argument("--sigma-p", dest="sigma_p", type=float, default=1.5)
    ap.add_argument("--eta-r", dest="eta_r", type=float, default=15.0)
    ap.add_argument("--p50-r", dest="p50_r", type=float, default=0.0, help="Phi(p) Hill half-point; 0 -> linear")
    ap.add_argument("--n-r", dest="n_r", type=float, default=2.0, help="Phi(p) Hill exponent (if p50_r>0)")
    ap.add_argument("--clamp-val", dest="clamp_val", type=float, default=0.8, help="E4: frozen p value")
    ap.add_argument("--early-stop", dest="early_stop", default=True, action=argparse.BooleanOptionalAction,
                    help="exit_atlas: early-stop genuine runaway for speed (won't cut the bounded state or an exit)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="s1")
    a = ap.parse_args()
    if not a.confirm_run:
        print("REFUSED: exit sim gate. Re-run with --confirm-run.")
        return
    if a.out is None:
        sub = "stage2_arms" if a.mode == "arms" else "stage1_exit_atlas"
        a.out = os.path.join(PP.ROOT, "results", "topic4_sef_hfo", "m4_snn_native_exit", sub)
    if a.mode == "arms":
        _run_arms(a)
        return
    os.makedirs(a.out, exist_ok=True)
    holds = [float(x) for x in a.holds.split(",")]

    # early-stop truncates only genuine runaway (>=120Hz sustained 100ms); the bounded state maxes ~64-97Hz
    # and an exit stays low -> neither is cut. It still sets runaway_ms so a rebound_runaway is DETECTED,
    # just not simulated in full -> big speed win on rebound cells. Default on; --no-early-stop for full traces.
    M4._EARLY_STOP["on"] = a.early_stop
    t_build = time.time()
    S = PP.build_substrate(a.seed)
    M4._S["S"] = S
    with open(os.path.join(a.out, f"pids_{a.tag}_seed{a.seed}.txt"), "w") as f:
        f.write(f"{os.getpid()}\n")       # manifest for the resource monitor (parent; workers COW-fork)

    cells = []
    if a.baseline:                        # optional: unperturbed bounded reference (== each hold's pre_af)
        cells.append(("baseline", None, a.t0, a.t0 + a.post_obs, a.t0 + a.post_obs))
    for h in holds:
        t1 = a.t0 + h
        T = t1 + a.post_obs
        cells.append((f"hold{int(h)}", dict(kind="inhibitory_pulse", t0=a.t0, t1=t1, val=a.dvth), a.t0, t1, T))

    print(f"[exit-atlas] substrate E1146 {PP.MONTAGE} N={S['N']} seed={a.seed} t0={a.t0} dvth={a.dvth} "
          f"holds={holds} n_cells={len(cells)} workers={a.workers} build={time.time()-t_build:.0f}s", flush=True)
    t_run = time.time()
    with mp.Pool(min(a.workers, len(cells))) as pool:
        results = pool.map(_cell_worker, cells)

    rows = [r for r, _ in results]
    out_tag = f"{a.tag}_seed{a.seed}"
    np.savez_compressed(os.path.join(a.out, f"exit_atlas_{out_tag}.npz"),   # npz FIRST (survives a json failure)
                        posE=S["posE"].astype(np.float32), src_xy=S["src_xy"], snk_xy=S["snk_xy"],
                        L=float(S["L"]),
                        **{f"{r['label']}__{k}": arr for (r, arrs) in results if arrs
                           for k, arr in arrs.items()})
    json.dump(_sanitize(dict(meta=dict(seed=a.seed, t0=a.t0, dvth=a.dvth, holds=holds, post_obs=a.post_obs,
                                       base_kq=BASE_KQ, base_ag=BASE_AG, subject=PP.SUBJECT, montage=PP.MONTAGE,
                                       N=int(S["N"]), axis_unit=S["axis_unit"].tolist(),
                                       wall_s=round(time.time() - t_run, 1), argv=" ".join(sys.argv)),
                             rows=rows)),
              open(os.path.join(a.out, f"exit_atlas_{out_tag}.json"), "w"), indent=2, allow_nan=False)

    print("\n===== Stage-1a exit-boundary atlas =====", flush=True)
    for r in rows:
        if "error" in r:
            print(f"  {r['label']:12s} ERROR {r['error']}", flush=True)
            continue
        print(f"  {r['label']:12s} hold={r['hold_ms']:6.0f}ms verdict={r['verdict']:16s} "
              f"cls={r['termination_class']:15s} pre_af={r['pre_af']} post_af={r['post_af']} "
              f"qI:{r['qI_t0']:.2f}->{r['qI_t1']:.2f} qI_fin={r['qI_final']:.2f} runaway={r['runaway_ms']}", flush=True)
    print(f"wrote exit_atlas_{out_tag}.json + .npz to {a.out}", flush=True)


if __name__ == "__main__":
    main()
