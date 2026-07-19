"""Topic 4 MZ slow–fast dynamical transition — scientific runner.

*** THIS RUNS SIMULATIONS. *** Nothing runs on import; sim subcommands gated by --confirm-run.
Design contract (BINDING): docs/superpowers/specs/2026-07-20-topic4-mz-slow-fast-transition-design.md

Freeze the natural MZ slow state {z_i, m_i} at registered checkpoints and evolve ONLY the fast spiking system.
Reuse (not reinvent): src.topic4_mz_onset_dynamics (MZOnsetProbe, run_loop checkpoint/resume, score_runaway,
epsilon_c_from_ladder), run_m4_phaseplane.build_substrate, src.topic4_mz_slow_fast_transition (pure helpers).
NO engine edits (6 guarded files read-only).

Subcommands (all resumable via per-(cond,seed,state) JSON):
  pilot      1 cond x 1 seed x 1 checkpoint x few replays -> peak RSS + wall/step (resource probe)
  run        per-(cond,seed) job: natural trajectory + checkpoints + P_runaway / epsilon_c / tau_rec +
             counterfactuals + matched-D; --all fans out over the 12 units (memory-gated Pool)
  aggregate  (no sim) combine per-(cond,seed,state) JSON -> summary CSV/JSON + classification + STATUS
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse   # noqa: E402
import sys        # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import copy            # noqa: E402
import dataclasses     # noqa: E402
import hashlib         # noqa: E402
import json            # noqa: E402
import time            # noqa: E402

import numpy as np     # noqa: E402
import run_m4_phaseplane as PP                                                       # noqa: E402
from mz_slow_vars import MZSlowVarsConfig                                            # noqa: E402
from src.topic4_mz_onset_dynamics import (                                           # noqa: E402
    MZOnsetProbe, run_loop, score_runaway, epsilon_c_from_ladder,
)
from src.topic4_mz_slowvars import eta_m_from_frac                                   # noqa: E402
from src.topic4_mz_slow_fast_transition import (                                     # noqa: E402
    SCHEMA_VERSION, branch_rng_state, wilson_ci, recovery_time,
    state_step_schedule, matched_d_times, classify_transition,
)

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slow_fast_transition")
CFG_PATH = os.path.join(ROOT, "config", "topic4_mz_slow_fast_transition.yaml")
DT = 0.1
_GUARDED = ("kick_probe.py", "params.py", "model.py", "connectivity.py", "connectivity_rot.py", "lfp.py")
_TRACE_ATTRS = ("trace_z_mean", "trace_z_min", "trace_z_core_mean", "trace_z_surround_mean", "trace_m_mean",
                "trace_m_max", "trace_m_core_mean", "trace_m_surround_mean", "trace_adap_current",
                "trace_I_EI_E_mean", "trace_rate_E", "trace_rate_I", "calib_hist_I_EI", "calib_hist_I_EE")


def load_cfg():
    import yaml
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


# ============================================================ provenance + io
def _git_sha():
    import subprocess
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _file_hash(path):
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
    except Exception:
        return None


def engine_shas():
    eng = os.path.join(ROOT, "src", "snn_engine")
    return {f: _file_hash(os.path.join(eng, f)) for f in _GUARDED}


def provenance(cfg, extra=None):
    prov = dict(schema_version=SCHEMA_VERSION, git_sha=_git_sha(), engine_shas=engine_shas(),
                config_hash=_file_hash(CFG_PATH),
                module_hash=_file_hash(os.path.join(ROOT, "src", "topic4_mz_slow_fast_transition.py")),
                subject=cfg["subject"], montage=cfg["montage"], dt=DT)
    if extra:
        prov.update(extra)
    return prov


def _json_default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _dump_atomic(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


# ============================================================ condition + frozen-fork helpers
def _onset(cfg, seed):
    o = cfg["onset_ms"]
    return float(o[seed] if seed in o else o[str(seed)])


def _mzcfg(cond, cfg):
    c = dict(cfg["conditions"][cond])
    if c.get("use_m"):
        c["eta_m"] = eta_m_from_frac(cfg["A_target"], cfg["I_EE_scale"], cfg["peak_m_tau2000"])
    return MZSlowVarsConfig(**c)


def _clear_traces(slow):
    """Blank the streaming trace lists (pure outputs) so per-fork deep-copies stay cheap; z/m arrays untouched."""
    for a in _TRACE_ATTRS:
        setattr(slow, a, [])
    slow.snapshots = {}
    slow._snap_steps = None
    return slow


def _frozen_template(ck):
    """A frozen (z/m held) copy of the checkpoint's slow object, ready to fork forward from ck.t."""
    s = copy.deepcopy(ck.slow)
    s.set_branch(branch_step=int(ck.t), freeze=True)
    return _clear_traces(s)


def _run_fork(S, ck, slow, n_steps, cfg, *, rng_state=None, early_stop=True):
    """Continue the fast system from checkpoint ck for n_steps with the frozen/probed `slow`. rng_state=None
    -> native future noise (deterministic); else an independent replay branch (V/currents/z/m identical)."""
    start = ck if rng_state is None else dataclasses.replace(ck, rng_state=rng_state, slow=None)
    hz, dur = cfg["runaway_hz"], cfg["runaway_dur_ms"]
    res = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=n_steps, start=start, store_spikes=False,
                   early_stop_runaway=early_stop, es_thresh_hz=hz, es_dur_ms=dur)
    ra = score_runaway(res["rate_E"], DT, thresh_hz=hz, dur_ms=dur)
    if ra is None and res["runaway_early_stop_step"] is not None:
        ra = (res["runaway_early_stop_step"] - int(ck.t)) * DT
    return res, ra


def _p_runaway(S, ck, template, seed, cond, label, cfg):
    """Perturbation-free escape probability (design §3.1): N independent future-noise branches from the same
    frozen state, fraction meeting 120 Hz/100 ms runaway."""
    n = int(cfg["p_runaway"]["n_replay"])
    horizon = int(round(cfg["p_runaway"]["horizon_ms"] / DT))
    onsets, k = [], 0
    for b in range(n):
        s = copy.deepcopy(template)
        _res, ra = _run_fork(S, ck, s, horizon, cfg, rng_state=branch_rng_state(seed, cond, label, b))
        onsets.append(None if ra is None else round(float(ra), 1))
        k += int(ra is not None)
    lo, hi = wilson_ci(k, n)
    return dict(p_runaway=k / n, p_runaway_ci=[round(lo, 4), round(hi, 4)], n_replay=n, n_runaway=k,
                replay_onsets_ms=onsets)


def _epsilon_c(S, ck, template, gap, all_E, cfg):
    """Global nonlinear ignition threshold (design §3.2): smallest uniform 10 ms threshold-lowering probe on
    ALL E (native noise) that ignites runaway. Ladder + bisection."""
    ig = cfg["ignition"]
    ladder = [float(a) for a in ig["amplitude_ladder"]]
    probe_steps = int(round(ig["probe_ms"] / DT))
    horizon = int(round(ig["horizon_ms"] / DT))
    branch = int(ck.t)
    ran = []
    for a in ladder:
        s = copy.deepcopy(template)
        if a > 0.0:
            s.set_probe(lo=branch, hi=branch + probe_steps, target_E=all_E, delta=a * gap)
        _res, ra = _run_fork(S, ck, s, horizon, cfg, rng_state=None)      # native noise -> deterministic eps_c
        ran.append(ra is not None)
    eps = epsilon_c_from_ladder(ladder, ran)
    bis = []
    if eps["bracket"] is not None and not eps["zero_runaway"]:
        lo, hi = eps["bracket"]
        for _ in range(int(ig["bisection_refinements"])):
            mid = 0.5 * (lo + hi)
            s = copy.deepcopy(template)
            s.set_probe(lo=branch, hi=branch + probe_steps, target_E=all_E, delta=mid * gap)
            _res, ra = _run_fork(S, ck, s, horizon, cfg, rng_state=None)
            bis.append(dict(a=round(mid, 5), runaway=ra is not None))
            if ra is not None:
                hi = mid
            else:
                lo = mid
        eps["epsilon_c_refined"] = round(hi, 5)
    return dict(epsilon_c=eps["epsilon_c"], epsilon_c_refined=eps.get("epsilon_c_refined"),
                epsilon_c_censored=eps["censored"], epsilon_c_zero_runaway=eps["zero_runaway"],
                ignition_ladder=ladder, ignition_ran=ran, ignition_bisection=bis)


def _tau_rec(S, ck, template, gap, all_E, cfg):
    """Fast-rate recovery time (design §3.3): one subthreshold global pulse, time for the E-rate to return to
    the frozen pre-pulse band. Censored if the state runs away (flagged runaway_during_probe)."""
    rc = cfg["recovery"]
    branch = int(ck.t)
    pre = int(round(rc["pre_window_ms"] / DT))
    pulse = int(round(rc["pulse_ms"] / DT))
    horizon = pre + pulse + int(round(rc["horizon_ms"] / DT))
    s = copy.deepcopy(template)
    s.set_probe(lo=branch + pre, hi=branch + pre + pulse, target_E=all_E, delta=float(rc["amp"]) * gap)
    res, ra = _run_fork(S, ck, s, horizon, cfg, rng_state=None, early_stop=True)
    rate = res["rate_E"]
    if ra is not None or rate.size < pre + pulse + 10:
        return dict(tau_rec=None, tau_rec_censored=True, runaway_during_probe=(ra is not None), pre_band=None)
    pre_rate = rate[:pre]
    mu, sd = float(pre_rate.mean()), float(pre_rate.std())
    k = float(rc["band_k"])
    lo, hi = mu - k * sd, mu + k * sd
    rt = recovery_time(rate, DT, pulse_off_idx=pre + pulse, band_lo=lo, band_hi=hi,
                       smooth_ms=float(rc["smooth_ms"]), min_hold_ms=float(rc["min_hold_ms"]))
    return dict(tau_rec=(None if rt is None else round(rt, 1)), tau_rec_censored=(rt is None),
                runaway_during_probe=False, pre_band=[round(lo, 4), round(hi, 4)], pre_rate_mean=round(mu, 4))


# ============================================================ natural trajectory (survey + chained capture)
def _natural_survey(S, mzcfg, seed, O_s, tail, cfg):
    """One natural (un-frozen) run to O_s+tail. Returns full D/a/rate traces + operational-runaway crossing."""
    mz = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=None)
    S["net"]["rng"] = np.random.default_rng(seed)
    n = int(round((O_s + tail) / DT))
    res = run_loop(S["p"], S["net"], mz, S["vth"], n_steps=n, store_spikes=False, early_stop_runaway=False)
    rate = res["rate_E"]
    z_mean = np.asarray(mz.trace_z_mean, float)
    adap = np.asarray(mz.trace_adap_current, float)
    I_EE = float(cfg["I_EE_scale"])
    D_full = 1.0 - z_mean
    a_full = adap / I_EE
    t_full = np.arange(z_mean.size) * DT
    crossing = score_runaway(rate, DT, thresh_hz=cfg["runaway_hz"], dur_ms=cfg["runaway_dur_ms"])
    return dict(rate=rate, D_full=D_full, a_full=a_full, t_full=t_full, crossing_ms=crossing,
                D_max=float(D_full.max()) if D_full.size else float("nan"))


def _chain_capture(S, mzcfg, seed, cap):
    """Chain natural segments (bit-identical to one continuous run) capturing a LoopState at each requested
    step. cap = {label: step}. Returns {label: LoopState} (traces on the copies are blanked to save memory)."""
    label_by_step = {}
    for lab, st in cap.items():
        label_by_step.setdefault(int(st), []).append(lab)
    steps = sorted(label_by_step)
    mz = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=None)
    S["net"]["rng"] = np.random.default_rng(seed)
    cks, ck, prev = {}, None, 0
    for st in steps:
        delta = st - prev
        if delta > 0:
            res = run_loop(S["p"], S["net"], mz, S["vth"], n_steps=delta, start=ck, capture_final=True,
                           store_spikes=False)
            ck = res["checkpoint"]
            _clear_traces(ck.slow)
            prev = st
        for lab in label_by_step[st]:
            cks[lab] = ck
    return cks


def _downsample(arr, step):
    arr = np.asarray(arr, float)
    n = arr.size // step
    return arr[:n * step].reshape(n, step).mean(axis=1) if n else arr


# ============================================================ per-(condition, seed) unit
def run_unit(cond, seed, cfg, resume=False, verbose=True):
    """Full job for one (condition, seed): natural survey + chained checkpoint capture + per-checkpoint
    P_runaway / epsilon_c / tau_rec + state-matched counterfactuals + matched-D cross-check. Atomic writes."""
    t_job = time.time()
    per_state = os.path.join(OUT, "per_state")
    mzcfg = _mzcfg(cond, cfg)
    O_s = _onset(cfg, seed)
    S = PP.build_substrate(seed)
    NE = S["NE"]
    all_E = np.ones(NE, bool)
    gap = float(np.median(S["vth"][:NE] - S["p"].V_reset))
    I_EE = float(cfg["I_EE_scale"])
    tail = float(cfg["natural_tail_ms"])

    # --- survey: full natural trajectory + crossing (also cross-check z_only crossing vs onset anchor) ---
    survey = _natural_survey(S, mzcfg, seed, O_s, tail, cfg)
    npz_path = os.path.join(per_state, f"{cond}_seed{seed}_natural.npz")
    if not (resume and os.path.exists(npz_path)):
        os.makedirs(per_state, exist_ok=True)
        ds = max(1, int(round(5.0 / DT)))               # 5 ms downsample for the figure
        np.savez_compressed(
            npz_path, t_ms=_downsample(survey["t_full"], ds), D=_downsample(survey["D_full"], ds),
            a=_downsample(survey["a_full"], ds), rate_E_hz=_downsample(survey["rate"], ds),
            crossing_ms=(np.nan if survey["crossing_ms"] is None else float(survey["crossing_ms"])),
            onset_anchor_ms=float(O_s), D_max=float(survey["D_max"]), condition=cond, seed=int(seed))
    xcheck = None
    if cond == "z_only" and survey["crossing_ms"] is not None:
        xcheck = round(float(survey["crossing_ms"]) - O_s, 1)   # should be ~0 (validates run_loop reproduction)

    # --- capture steps: matched-time + first_crossing + matched-D ---
    sched = state_step_schedule(O_s, DT)
    cap = dict(sched)
    crossing_step = None
    if survey["crossing_ms"] is not None:
        crossing_step = int(round(survey["crossing_ms"] / DT))
        cap["first_crossing"] = crossing_step
    md_times = matched_d_times(survey["D_full"], survey["t_full"], cfg["matched_d_targets"])
    md_steps = {float(tg): int(round(tm / DT)) for tg, tm in md_times.items() if tm is not None}
    for tg, st in md_steps.items():
        cap[f"matched_d_{tg}"] = st
    cks = _chain_capture(S, mzcfg, seed, cap)

    def _coord(ck):
        z = ck.slow.z[:NE]
        m = ck.slow.m[:NE]
        return float(1.0 - z.mean()), float(mzcfg.eta_m * m.mean() / I_EE)

    # --- per matched-time state (+ first_crossing): all three probes ---
    state_labels = list(sched) + (["first_crossing"] if crossing_step is not None else [])
    written = 0
    for label in state_labels:
        outp = os.path.join(per_state, f"{cond}_seed{seed}_{label}.json")
        if resume and os.path.exists(outp):
            continue
        ck = cks[label]
        template = _frozen_template(ck)
        D, a = _coord(ck)
        pr = _p_runaway(S, ck, template, seed, cond, label, cfg)
        ec = _epsilon_c(S, ck, template, gap, all_E, cfg)
        tr = _tau_rec(S, ck, template, gap, all_E, cfg)
        _dump_atomic(dict(condition=cond, seed=int(seed), state=label, step=int(ck.t),
                          time_ms=round(int(ck.t) * DT, 1), D=round(D, 6), a=round(a, 8),
                          onset_anchor_ms=O_s, **pr, **ec, **tr,
                          provenance=provenance(cfg, dict(phase="run", condition=cond, seed=seed))), outp)
        written += 1
        if verbose:
            print(f"[run] {cond} s{seed} {label} D={D:.4f} P_run={pr['p_runaway']:.2f} "
                  f"eps_c={ec['epsilon_c']} tau_rec={tr['tau_rec']}", flush=True)

    # --- counterfactuals at pre_onset_100ms (design §4) ---
    outcf = os.path.join(OUT, "counterfactual", f"{cond}_seed{seed}.json")
    if not (resume and os.path.exists(outcf)):
        ck100 = cks["pre_onset_100ms"]
        ckmid = cks["mid_fraction"]
        z_late, m_late = ck100.slow.z[:NE].copy(), ck100.slow.m[:NE].copy()
        z_early, m_early = ckmid.slow.z[:NE].copy(), ckmid.slow.m[:NE].copy()
        settings = {
            "native_zm": (z_late, m_late),
            "native_z_reset_m": (z_late, np.zeros(NE)),
            "reset_z_native_m": (np.ones(NE), m_late),
            "late_z_early_m": (z_late, m_early),
            "early_z_late_m": (z_early, m_late),
        }
        rows = []
        for br in cfg["counterfactual"]["branches"]:
            zz, mm = settings[br]
            template = _frozen_template(ck100)
            template.z[:NE] = zz
            template.m[:NE] = mm
            pr = _p_runaway(S, ck100, template, seed, cond, f"cf_{br}", cfg)
            ec = _epsilon_c(S, ck100, template, gap, all_E, cfg)
            tr = _tau_rec(S, ck100, template, gap, all_E, cfg)
            rows.append(dict(branch=br, D=round(float(1.0 - zz.mean()), 6),
                             a=round(float(mzcfg.eta_m * mm.mean() / I_EE), 8), **pr, **ec, **tr))
            if verbose:
                print(f"[run] {cond} s{seed} cf:{br} P_run={pr['p_runaway']:.2f} eps_c={ec['epsilon_c']}",
                      flush=True)
        _dump_atomic(dict(condition=cond, seed=int(seed), branch_state="pre_onset_100ms",
                          early_state="mid_fraction", rows=rows,
                          provenance=provenance(cfg, dict(phase="counterfactual", condition=cond, seed=seed))),
                     outcf)

    # --- matched-D cross-check: P_runaway + epsilon_c only (design §2.2) ---
    outmd = os.path.join(OUT, "matched_d", f"{cond}_seed{seed}.json")
    if not (resume and os.path.exists(outmd)):
        rows = []
        for tg in sorted(md_steps):
            ck = cks[f"matched_d_{tg}"]
            template = _frozen_template(ck)
            D, a = _coord(ck)
            pr = _p_runaway(S, ck, template, seed, cond, f"md_{tg}", cfg)
            ec = _epsilon_c(S, ck, template, gap, all_E, cfg)
            rows.append(dict(target_D=tg, time_ms=round(md_times[tg], 1), D=round(D, 6), **pr, **ec))
        censored = [tg for tg, tm in md_times.items() if tm is None]
        _dump_atomic(dict(condition=cond, seed=int(seed), rows=rows, censored_targets=censored,
                          provenance=provenance(cfg, dict(phase="matched-d", condition=cond, seed=seed))),
                     outmd)

    wall = time.time() - t_job
    if verbose:
        xc = "" if xcheck is None else f" xcheck(cross-onset)={xcheck}ms"
        print(f"[run] DONE {cond} s{seed} wall={wall:.0f}s states_written={written} "
              f"crossing={survey['crossing_ms']} D_max={survey['D_max']:.4f}{xc}", flush=True)
    return dict(condition=cond, seed=seed, wall_s=round(wall, 1), crossing_ms=survey["crossing_ms"],
                D_max=survey["D_max"], onset_xcheck_ms=xcheck)


# ============================================================ CLI dispatch
def cmd_pilot(args, cfg):
    raise NotImplementedError("pilot: implemented in Task 6")


def cmd_run(args, cfg):
    if not args.only:
        raise SystemExit("run: pass --only cond:seed (single unit) [--all fan-out lands in Task 7]")
    cond, seed = args.only.split(":")
    run_unit(cond, int(seed), cfg, resume=args.resume)


def cmd_aggregate(args, cfg):
    raise NotImplementedError("aggregate: implemented in Task 7")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Topic 4 MZ slow–fast dynamical transition runner (design 2026-07-20).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pilot", "run", "aggregate"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true")
        sp.add_argument("--seeds", default=None, help="comma-separated seed subset override")
        sp.add_argument("--conditions", default=None, help="comma-separated condition subset override")
        sp.add_argument("--only", default=None, help="run: single unit 'cond:seed'")
        sp.add_argument("--all", action="store_true", help="run: fan out over all (cond,seed) units")
        sp.add_argument("--workers", type=int, default=None, help="run --all: worker count (default memory-gated)")
        sp.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    if args.cmd in ("pilot", "run") and not args.confirm_run:
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run.", file=sys.stderr)
        sys.exit(2)
    os.makedirs(OUT, exist_ok=True)
    {"pilot": cmd_pilot, "run": cmd_run, "aggregate": cmd_aggregate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
