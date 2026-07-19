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


# ============================================================ natural trajectory (single pass + capture)
def _nearest_earlier(sched, step):
    """(label, step) of the latest matched-time checkpoint at or before ``step``; (None, None) if none."""
    cands = [(lab, st) for lab, st in sched.items() if st <= step]
    return max(cands, key=lambda kv: kv[1]) if cands else (None, None)


def _resume_capture(S, base_ck, target_step, mzcfg, seed):
    """Capture a natural (un-frozen) checkpoint at ``target_step`` via a cheap short re-resume from the
    nearest earlier checkpoint (or a fresh chain from 0 if none). Bit-identical to the main trajectory."""
    if base_ck is None:
        mz = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=None)
        S["net"]["rng"] = np.random.default_rng(seed)
        res = run_loop(S["p"], S["net"], mz, S["vth"], n_steps=int(target_step), capture_final=True,
                       store_spikes=False)
    else:
        d = int(target_step) - int(base_ck.t)
        if d <= 0:
            return base_ck
        res = run_loop(S["p"], S["net"], copy.deepcopy(base_ck.slow), S["vth"], n_steps=d, start=base_ck,
                       capture_final=True, store_spikes=False)
    ck = res["checkpoint"]
    _clear_traces(ck.slow)
    return ck


def _natural_and_capture(S, mzcfg, seed, O_s, tail, cfg):
    """ONE natural run. Chain-captures the matched-time checkpoints (all pre-onset -> normal speed), then
    continues with early-stop to O_s+tail so the run stops AT the operational-runaway crossing instead of
    grinding through the slow post-runaway saturated tail. first_crossing + matched-D checkpoints are captured
    by cheap short re-resumes. Returns the full pre-onset(+crossing) trajectory + all LoopState checkpoints."""
    sched = state_step_schedule(O_s, DT)
    mz = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=None)
    S["net"]["rng"] = np.random.default_rng(seed)
    cks, rate_segs, ck, prev = {}, [], None, 0
    for label, st in sorted(sched.items(), key=lambda kv: kv[1]):
        d = st - prev
        if d > 0:
            res = run_loop(S["p"], S["net"], mz, S["vth"], n_steps=d, start=ck, capture_final=True,
                           store_spikes=False)
            ck = res["checkpoint"]
            rate_segs.append(res["rate_E"])
            prev = st
        _clear_traces(ck.slow)
        cks[label] = ck
    end_step = int(round((O_s + tail) / DT))
    tail_stop = None
    if end_step > prev:
        res = run_loop(S["p"], S["net"], mz, S["vth"], n_steps=end_step - prev, start=ck, store_spikes=False,
                       early_stop_runaway=True, es_thresh_hz=cfg["runaway_hz"], es_dur_ms=cfg["runaway_dur_ms"])
        rate_segs.append(res["rate_E"])
        tail_stop = res["runaway_early_stop_step"]
    full_rate = np.concatenate(rate_segs) if rate_segs else np.array([])
    I_EE = float(cfg["I_EE_scale"])
    z_mean = np.asarray(mz.trace_z_mean, float)
    adap = np.asarray(mz.trace_adap_current, float)
    D_full = 1.0 - z_mean
    a_full = adap / I_EE
    t_full = np.arange(z_mean.size) * DT
    crossing = score_runaway(full_rate, DT, thresh_hz=cfg["runaway_hz"], dur_ms=cfg["runaway_dur_ms"])
    if crossing is None and tail_stop is not None:
        crossing = tail_stop * DT
    crossing_step = None if crossing is None else int(round(crossing / DT))
    if crossing_step is not None:
        lab, _ = _nearest_earlier(sched, crossing_step)
        cks["first_crossing"] = _resume_capture(S, cks.get(lab), crossing_step, mzcfg, seed)
    md_times = matched_d_times(D_full, t_full, cfg["matched_d_targets"])
    md_steps = {}
    for tg, tm in md_times.items():
        if tm is None:
            continue
        st = int(round(tm / DT))
        md_steps[float(tg)] = st
        lab, _ = _nearest_earlier(sched, st)
        cks[f"matched_d_{float(tg)}"] = _resume_capture(S, cks.get(lab), st, mzcfg, seed)
    return dict(cks=cks, sched=sched, D_full=D_full, a_full=a_full, rate=full_rate, t_full=t_full,
                crossing_ms=crossing, crossing_step=crossing_step, md_times=md_times, md_steps=md_steps,
                D_max=float(D_full.max()) if D_full.size else float("nan"))


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

    # --- ONE natural pass: trajectory + all checkpoints (matched-time + first_crossing + matched-D) ---
    nat = _natural_and_capture(S, mzcfg, seed, O_s, tail, cfg)
    cks, sched = nat["cks"], nat["sched"]
    crossing_step, md_times, md_steps = nat["crossing_step"], nat["md_times"], nat["md_steps"]
    npz_path = os.path.join(per_state, f"{cond}_seed{seed}_natural.npz")
    if not (resume and os.path.exists(npz_path)):
        os.makedirs(per_state, exist_ok=True)
        ds = max(1, int(round(5.0 / DT)))               # 5 ms downsample for the figure
        np.savez_compressed(
            npz_path, t_ms=_downsample(nat["t_full"], ds), D=_downsample(nat["D_full"], ds),
            a=_downsample(nat["a_full"], ds), rate_E_hz=_downsample(nat["rate"], ds),
            crossing_ms=(np.nan if nat["crossing_ms"] is None else float(nat["crossing_ms"])),
            onset_anchor_ms=float(O_s), D_max=float(nat["D_max"]), condition=cond, seed=int(seed))
    xcheck = None
    if cond == "z_only" and nat["crossing_ms"] is not None:
        xcheck = round(float(nat["crossing_ms"]) - O_s, 1)     # should be ~0 (validates run_loop reproduction)

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
              f"crossing={nat['crossing_ms']} D_max={nat['D_max']:.4f}{xc}", flush=True)
    return dict(condition=cond, seed=seed, wall_s=round(wall, 1), crossing_ms=nat["crossing_ms"],
                D_max=nat["D_max"], onset_xcheck_ms=xcheck)


# ============================================================ resource helpers
def _rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)   # ru_maxrss KB (linux) -> GB


def _avail_ram_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024.0 ** 2)                       # KB -> GB
    except Exception:
        pass
    return float("inf")


def suggest_workers(peak_rss_gb, n_units=12):
    """Memory-gated worker count: min(nproc-2, floor((avail - max(30GB,25%avail)) / peak_rss)) (design §8)."""
    avail = _avail_ram_gb()
    margin = max(30.0, 0.25 * avail)
    w_mem = int((avail - margin) // max(peak_rss_gb, 0.05)) if np.isfinite(avail) else n_units
    return max(1, min((os.cpu_count() or 2) - 2, w_mem, n_units)), avail, margin


# ============================================================ CLI dispatch
def cmd_pilot(args, cfg):
    """1 cond x 1 seed x 1 checkpoint x few replays -> peak RSS + wall/step + projected full-run cost +
    memory-gated worker suggestion. Validates the full-substrate path (incl. z_only crossing == onset anchor)."""
    cond, seed = (args.only.split(":") if args.only else ("z_only", "1"))
    seed = int(seed)
    n_replay = int(args.replays) if getattr(args, "replays", None) else 4
    print(f"[pilot] cond={cond} seed={seed} n_replay={n_replay}", flush=True)
    t0 = time.time(); S = PP.build_substrate(seed); t_build = time.time() - t0
    NE, N = S["NE"], S["N"]
    print(f"[pilot] substrate N={N} NE={NE} NI={S['NI']} build={t_build:.1f}s rss={_rss_gb():.2f}GB", flush=True)
    mzcfg = _mzcfg(cond, cfg)
    O_s = _onset(cfg, seed)
    gap = float(np.median(S["vth"][:NE] - S["p"].V_reset))
    all_E = np.ones(NE, bool)
    tail = float(cfg["natural_tail_ms"])
    t0 = time.time(); nat = _natural_and_capture(S, mzcfg, seed, O_s, tail, cfg); t_nat = time.time() - t0
    xcheck = None if nat["crossing_ms"] is None else round(nat["crossing_ms"] - O_s, 1)
    main_steps = nat["crossing_step"] or int(round((O_s + tail) / DT))
    print(f"[pilot] natural+capture {t_nat:.1f}s ({t_nat / max(main_steps, 1) * 1e5:.2f}s/1e5steps) "
          f"crossing={nat['crossing_ms']} D_max={nat['D_max']:.4f} onset_xcheck={xcheck}ms "
          f"n_checkpoints={len(nat['cks'])}", flush=True)
    ck = nat["cks"]["pre_onset_100ms"]
    template = _frozen_template(ck)
    cfg_pilot = copy.deepcopy(cfg); cfg_pilot["p_runaway"]["n_replay"] = n_replay
    t0 = time.time(); pr = _p_runaway(S, ck, template, seed, cond, "pilot", cfg_pilot); t_pr = time.time() - t0
    t0 = time.time(); ec = _epsilon_c(S, ck, template, gap, all_E, cfg); t_ec = time.time() - t0
    t0 = time.time(); tr = _tau_rec(S, ck, template, gap, all_E, cfg); t_tr = time.time() - t0
    peak = _rss_gb()
    per_replay = t_pr / max(n_replay, 1)
    print(f"[pilot] P_runaway({n_replay})={pr['p_runaway']:.2f} {t_pr:.1f}s ({per_replay:.2f}s/replay) | "
          f"eps_c={ec['epsilon_c']} {t_ec:.1f}s | tau_rec={tr['tau_rec']} {t_tr:.1f}s", flush=True)
    print(f"[pilot] PEAK RSS = {peak:.2f} GB", flush=True)
    n_rep_full = int(cfg["p_runaway"]["n_replay"])
    full_pr = n_rep_full * per_replay
    n_states = len(state_step_schedule(O_s, DT)) + (1 if nat["crossing_ms"] is not None else 0)
    n_md = len(nat["md_steps"])
    ckpt = full_pr + t_ec + t_tr
    job_s = t_nat + n_states * ckpt + 5 * ckpt + n_md * (full_pr + t_ec)
    workers, avail, margin = suggest_workers(peak)
    print(f"[pilot] PROJECTED job ~ {job_s / 60:.1f} min/unit (n_replay={n_rep_full}; states={n_states}x{ckpt:.0f}s "
          f"+ cf 5x{ckpt:.0f}s + md {n_md}x{full_pr + t_ec:.0f}s + natural {t_nat:.0f}s)", flush=True)
    print(f"[pilot] avail_ram={avail:.0f}GB margin={margin:.0f}GB peak_rss={peak:.2f}GB -> "
          f"SUGGEST workers={workers} (nproc-2={max(1, (os.cpu_count() or 2) - 2)}); "
          f"12 units ~ {12 / max(workers, 1) * job_s / 60:.0f} min wall", flush=True)


def _unit_worker(t):
    """Pool worker (spawn-picklable): one (cond, seed) unit, fail-loud but isolated (error -> tagged dict)."""
    cond, seed, resume = t
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    cfg = load_cfg()
    try:
        return run_unit(cond, int(seed), cfg, resume=bool(resume), verbose=True)
    except Exception as e:
        import traceback
        return dict(condition=cond, seed=int(seed), error=f"{type(e).__name__}: {e}", tb=traceback.format_exc())


def cmd_run(args, cfg):
    # single unit
    if args.only and not args.all:
        cond, seed = args.only.split(":")
        run_unit(cond, int(seed), cfg, resume=args.resume)
        return
    if not args.all:
        raise SystemExit("run: pass --only cond:seed (single) OR --all (fan out over all units)")
    conds = args.conditions.split(",") if args.conditions else cfg["condition_order"]
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    units = [(c, s) for c in conds for s in seeds]
    peak = float(args.peak_rss) if getattr(args, "peak_rss", None) else 6.0
    w_cap, avail, margin = suggest_workers(peak, n_units=len(units))
    W = min(int(args.workers), w_cap) if args.workers else w_cap
    print(f"[run] {len(units)} units, workers={W} (mem-cap={w_cap}, avail={avail:.0f}GB, margin={margin:.0f}GB, "
          f"peak_rss={peak:.1f}GB, resume={args.resume})", flush=True)
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = []
    with ctx.Pool(W) as pool:
        for r in pool.imap_unordered(_unit_worker, [(c, s, args.resume) for c, s in units]):
            results.append(r)
            tag = f"{r['condition']} s{r['seed']}"
            if r.get("error"):
                print(f"[run] FAILED {tag}: {r['error']}", flush=True)
            else:
                print(f"[run] OK {tag} wall={r.get('wall_s')}s crossing={r.get('crossing_ms')} "
                      f"D_max={r.get('D_max'):.4f}", flush=True)
    failed = [r for r in results if r.get("error")]
    _dump_atomic(dict(units=results, n_units=len(units), n_failed=len(failed),
                      provenance=provenance(cfg, dict(phase="run-all", workers=W))),
                 os.path.join(OUT, "run_manifest.json"))
    if failed:
        for r in failed:
            print(f"[run] --- traceback {r['condition']} s{r['seed']} ---\n{r.get('tb')}", file=sys.stderr)
        raise SystemExit(f"{len(failed)}/{len(units)} units FAILED (see run_manifest.json)")
    print(f"[run] all {len(units)} units done", flush=True)


# ============================================================ aggregate (no sim)
_STATE_ORDER = ["baseline_1000ms", "mid_fraction", "pre_onset_2000ms", "pre_onset_1000ms",
                "pre_onset_500ms", "pre_onset_200ms", "pre_onset_100ms", "first_crossing"]


def _write_csv(rows, path, keys):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def cmd_aggregate(args, cfg):
    per_state = os.path.join(OUT, "per_state")
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    conds = cfg["condition_order"]
    # --- load per-state records + natural crossings ---
    data, crossings, dmax = {}, {}, {}
    for cond in conds:
        for seed in seeds:
            for st in _STATE_ORDER:
                p = os.path.join(per_state, f"{cond}_seed{seed}_{st}.json")
                if os.path.exists(p):
                    data[(cond, seed, st)] = json.load(open(p))
            npz = os.path.join(per_state, f"{cond}_seed{seed}_natural.npz")
            if os.path.exists(npz):
                d = np.load(npz)
                cm = float(d["crossing_ms"])
                crossings[(cond, seed)] = None if not np.isfinite(cm) else cm
                dmax[(cond, seed)] = float(d["D_max"])
    # --- summary CSV rows ---
    rows = []
    for cond in conds:
        for seed in seeds:
            for st in _STATE_ORDER:
                r = data.get((cond, seed, st))
                if r is None:
                    continue
                ci = r.get("p_runaway_ci") or [None, None]
                rows.append(dict(condition=cond, seed=seed, state=st, time_ms=r.get("time_ms"), D=r.get("D"),
                                 a=r.get("a"), p_runaway=r.get("p_runaway"), p_ci_lo=ci[0], p_ci_hi=ci[1],
                                 n_runaway=r.get("n_runaway"), epsilon_c=r.get("epsilon_c"),
                                 epsilon_c_censored=r.get("epsilon_c_censored"), tau_rec=r.get("tau_rec"),
                                 tau_rec_censored=r.get("tau_rec_censored")))
    _write_csv(rows, os.path.join(OUT, "slow_fast_transition_summary.csv"),
               ["condition", "seed", "state", "time_ms", "D", "a", "p_runaway", "p_ci_lo", "p_ci_hi",
                "n_runaway", "epsilon_c", "epsilon_c_censored", "tau_rec", "tau_rec_censored"])
    # --- classify per (cond, seed), consensus per condition ---
    per_unit, per_condition = {}, {}
    for cond in conds:
        labels = []
        for seed in seeds:
            ps = [dict(D=data[(cond, seed, st)]["D"], p_runaway=data[(cond, seed, st)]["p_runaway"],
                       epsilon_c=data[(cond, seed, st)].get("epsilon_c"),
                       tau_rec=data[(cond, seed, st)].get("tau_rec"))
                  for st in _STATE_ORDER if (cond, seed, st) in data]
            natural_crosses = crossings.get((cond, seed)) is not None
            plateau_outside = crossings.get(("mz_plateau", seed), "missing") is None
            cls = classify_transition(ps, natural_crosses=natural_crosses, plateau_outside=plateau_outside)
            per_unit[f"{cond}_seed{seed}"] = dict(label=cls["label"], features=cls["features"],
                                                  crossing_ms=crossings.get((cond, seed)),
                                                  D_max=dmax.get((cond, seed)))
            labels.append(cls["label"])
        consensus = labels[0] if labels and all(x == labels[0] for x in labels) else "seed-inconsistent"
        per_condition[cond] = dict(consensus=consensus, per_seed_labels=labels)
    # --- engine unchanged check ---
    import subprocess
    eng_ok = subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "--", "src/snn_engine"]).returncode == 0
    summary = dict(schema_version=SCHEMA_VERSION, conditions=conds, seeds=seeds,
                   per_condition=per_condition, per_unit=per_unit, natural_crossings={f"{c}_seed{s}": crossings.get((c, s)) for c in conds for s in seeds},
                   engine_unmodified=eng_ok, engine_shas=engine_shas(),
                   provenance=provenance(cfg, dict(phase="aggregate")))
    _dump_atomic(summary, os.path.join(OUT, "slow_fast_transition_summary.json"))
    _write_status(cfg, per_condition, per_unit, crossings, dmax, eng_ok)
    print(f"[aggregate] {len(rows)} state-rows; per-condition consensus: "
          f"{ {c: per_condition[c]['consensus'] for c in conds} }; engine_unmodified={eng_ok}", flush=True)


def _write_status(cfg, per_condition, per_unit, crossings, dmax, eng_ok):
    lines = ["# MZ slow–fast dynamical transition — STATUS", "",
             "Tier = model-side mechanism analysis. Operational runaway (120 Hz / 100 ms) only; NOT seizure.",
             "", "## Per-condition transition class (result-neutral, consensus across seeds 1/3/4)", ""]
    for cond in cfg["condition_order"]:
        pc = per_condition.get(cond, {})
        lines.append(f"- **{cond}** -> `{pc.get('consensus')}`  (per-seed: {pc.get('per_seed_labels')})")
    lines += ["", "## Natural operational-runaway crossing (ms) + D_max, per (condition, seed)", ""]
    for cond in cfg["condition_order"]:
        for seed in cfg["seeds"]:
            cm = crossings.get((cond, seed))
            lines.append(f"- {cond} s{seed}: crossing={cm} D_max={dmax.get((cond, seed))}")
    lines += ["", f"engine_unmodified={eng_ok}. See slow_fast_transition_summary.json for features + CIs,",
              "docs/archive/topic4/sef_hfo/mz_slow_fast_transition_2026-07-20.md for the verdict.", ""]
    with open(os.path.join(OUT, "STATUS.md"), "w") as f:
        f.write("\n".join(lines))


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
        sp.add_argument("--replays", type=int, default=None, help="pilot: P_runaway replay count (default 4)")
        sp.add_argument("--peak-rss", dest="peak_rss", type=float, default=None,
                        help="run --all: measured peak RSS (GB) per unit for the memory gate (from pilot)")
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
