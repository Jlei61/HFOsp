"""L=20 MZ-FCXR (full-conductance + persistence-gated E->E relay) staged runner.

Nothing runs on import.  Every simulation requires --confirm-run.  One launcher builds the E1146
substrate once and forks COW workers; BLAS threads pinned to one.  New results root (never overwrites
the accepted mz_conductance tree):
    results/topic4_sef_hfo/mz_full_conductance_spatial_relay/

Design: docs/superpowers/specs/2026-07-20-topic4-mz-full-conductance-spatial-relay-design.md.
Commands (staged ladder):
  smoke      Stage 0A: single full-conductance cell, RSS + numerical-safety smoke.
  workpoint  Stage 0B: c_E bracket vs the current-model slow-off reference; pick + seed3 confirm.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-fcxr")

import argparse
from contextlib import contextmanager
import dataclasses
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from kick_probe import simulate_kick  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_slowvars import classify_mz_run  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay")
ENGINE_VERSIONS = os.path.join(ROOT, "results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
DT = 0.1
I_TH_EI = 95.19851312666987          # locked Z depletion threshold (accepted conductance staircase)
MACHINE_RESERVE_GB = 64.0
PARENT_BUDGET_GB = 8.0
_CTX = {}


# ----------------------------------------------------------------- provenance / io
def _git_sha():
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _run_id(tag):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}_{os.getpid()}_{_git_sha()}_{tag}"


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    raise TypeError(type(x).__name__)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ----------------------------------------------------------------- resource contract (task §6)
def _meminfo():
    with open("/proc/meminfo") as f:
        vals = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    avail = vals["MemAvailable"] / 1024.0 / 1024.0
    swap_used = (vals["SwapTotal"] - vals["SwapFree"]) / 1024.0 / 1024.0
    return avail, swap_used


def _self_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _raster_gb(T_ms):
    return (float(T_ms) / DT) * 32000.0 / (1024.0 ** 3)


def _worker_budget_gb(T_ms):
    return max(12.0, 1.5 * (4.5 + _raster_gb(T_ms)))


def _other_40k_running():
    """Count sibling L=20 Topic4 SNN launchers/workers that are NOT this process tree."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout
    except Exception:
        return True   # fail-safe: assume contention
    me = {os.getpid(), os.getppid()}
    markers = ("run_topic4_mz_", "run_m4_", "run_topic4_sef", "mzx_", "topic4_mz_direct",
               "topic4_mz_divisive", "topic4_mz_slow", "topic4_mz_early")
    n = 0
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = int(parts[0]), int(parts[1]), parts[2]
        if pid in me or ppid in me:
            continue
        if "run_topic4_mz_fcxr" in args:      # never count my own tree
            continue
        if any(m in args for m in markers) and "python" in args.lower():
            n += 1
    return n


def _plan_workers(T_ms, requested):
    avail, swap_used = _meminfo()
    B = _worker_budget_gb(T_ms)
    slots = int((avail - MACHINE_RESERVE_GB - PARENT_BUDGET_GB) // B)
    others = _other_40k_running()
    if T_ms >= 20000.0:
        cap = 1 if others else 2
    else:
        cap = 2 if others else 4
    workers = max(0, min(int(requested), cap, slots))
    return dict(workers=workers, slots=slots, cap=cap, budget_gb=round(B, 2),
                mem_available_gb=round(avail, 2), swap_used_gb=round(swap_used, 3),
                other_40k_tasks=others, T_ms=float(T_ms))


def _resource_log(run_dir, tag, extra=None):
    avail, swap_used = _meminfo()
    row = dict(t=datetime.now(timezone.utc).isoformat(), tag=tag,
               mem_available_gb=round(avail, 2), swap_used_gb=round(swap_used, 3),
               self_rss_gb=round(_self_rss_gb(), 3))
    if extra:
        row.update(extra)
    path = os.path.join(run_dir, "resource_log.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=_jsonable) + "\n")
    return row


@contextmanager
def _launcher_lock():
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, ".l20_launcher.lock")
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another FCXR launcher is active; refusing duplicate build") from exc
        lock.seek(0); lock.truncate()
        lock.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n")
        lock.flush()
        try:
            yield path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _assert_engine_blessed():
    recorded = json.load(open(ENGINE_VERSIONS))
    for rel, expected in recorded.items():
        current = _sha(os.path.join(ROOT, rel))
        if current != expected:
            raise SystemExit(f"engine not blessed: {rel} {current[:12]} != {expected[:12]}")


# ----------------------------------------------------------------- cfg + run
def _fc_cfg(c_E, *, gamma=0.0, global_mode="additive", z=False, tau_z=2500.0,
            use_x=False, tau_x=1000.0, x_min=0.0, y_gate=0.0, K_y=5.0, tau_y=120.0,
            fail_on_clip=True):
    """FCXR locked config: full_conductance E_E=58 / V_match=18 / E_I=0 / gaba_gain=1.125 /
    z_scope=local_only / protected additive-global.  M off."""
    return dict(
        membrane_mode="full_conductance", E_E=58.0, c_E=float(c_E), v_match=18.0, e_gaba=0.0, e_k=0.0,
        use_z=bool(z), use_m=False, use_phi=False, I_th_EI=I_TH_EI, tau_z=float(tau_z),
        gaba_gain=1.125, m_conductance_gain=1.0,
        global_gaba_fraction=float(gamma), global_gaba_mode=str(global_mode), z_scope="local_only",
        max_total_conductance=99.0, fail_on_clip=bool(fail_on_clip),
        use_x=bool(use_x), tau_y=float(tau_y), tau_x=float(tau_x), x_min=float(x_min),
        y_gate=float(y_gate), K_y=float(K_y), hill_n=4,
    )


def _make_slow(S, cfg):
    return MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"], core_mask_E=OLD.build_core_masks(S))


def _run(S, cfg, T, *, kick_boost=0.0, t_kick=1e9, early_stop=True):
    p = dataclasses.replace(S["p"], T=float(T))
    slow = _make_slow(S, cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    res = simulate_kick(p, S["net"], float(kick_boost), slow=slow,
                        kick_center=list(S["src_xy"]), r_kick=PP.R_KICK, t_kick=float(t_kick),
                        V_th_per_neuron=S["vth"], early_stop_runaway=early_stop)
    return res, slow


def _numerical(S, res, slow, *, settle_ms=1000.0):
    """Numerical safety, split full-run vs settled (post-startup) so the network's first synchronous
    volley from V_reset is not conflated with the interictal operating point."""
    taur = np.asarray(slow.trace_tau_eff_ratio_min, float)
    clipf = np.asarray(slow.trace_conductance_clip_frac, float)
    finite = bool(np.all(np.isfinite(res["rate_E"])))
    tau_eff_min_ms = float(S["p"].tau_m_E * taur.min()) if taur.size else float("nan")
    max_clip = float(clipf.max()) if clipf.size else 0.0
    i0 = int(round(settle_ms / DT))
    if taur.size > i0:
        s_tau = float(S["p"].tau_m_E * taur[i0:].min())
        s_clip = float(clipf[i0:].max())
    else:  # early-stopped before the settle window -> settled == full (report honestly)
        s_tau, s_clip = tau_eff_min_ms, max_clip
    return dict(
        safe=bool(finite and tau_eff_min_ms >= 2 * DT and max_clip == 0.0),
        settled_safe=bool(finite and s_tau >= 2 * DT and s_clip == 0.0),
        finite=finite, tau_eff_min_ms=tau_eff_min_ms, max_clip_fraction=max_clip,
        settled_tau_eff_min_ms=s_tau, settled_max_clip_fraction=s_clip, settle_ms=settle_ms)


def _event_profile(events, rate, dt):
    returned = [e for e in events if bool(e.get("returned", False))]
    if not returned:
        return dict(n_returning=0, duration_median_ms=float("nan"),
                    participation_median=float("nan"), peak_rate_median_hz=float("nan"))
    return dict(
        n_returning=len(returned),
        duration_median_ms=float(np.median([e["dur_ms"] for e in returned])),
        participation_median=float(np.median([e["peak_ext"] for e in returned])),
        peak_rate_median_hz=float(np.median([OLD._peak_rate_in(rate, e, dt) for e in returned])),
    )


def _workpoint_distance(profile, baseline):
    """Distance of a c_E arm to the current-model slow-off reference workpoint (lower = closer)."""
    n_ratio = profile["n_returning"] / max(1, baseline.n_events)
    dur_ratio = profile["duration_median_ms"] / max(1e-9, baseline.dur_med)
    part = profile["participation_median"]; peak = profile["peak_rate_median_hz"]
    clauses = dict(
        event_count_ratio=bool(0.5 <= n_ratio <= 1.5),
        duration_ratio=bool(0.5 <= dur_ratio <= 2.0),
        participation_band=bool(0.5 * baseline.part_lo <= part <= 2.0 * baseline.part_hi),
        peak_rate_band=bool(0.5 * baseline.act_lo <= peak <= 2.0 * baseline.act_hi),
    )
    score = (abs(np.log(max(n_ratio, 1e-9))) + abs(np.log(max(dur_ratio, 1e-9)))
             + abs(part - 0.5 * (baseline.part_lo + baseline.part_hi)) / max(baseline.part_hi, 1e-9)
             + abs(peak - 0.5 * (baseline.act_lo + baseline.act_hi)) / max(baseline.act_hi, 1e-9))
    return dict(clauses=clauses, all_bands=bool(all(clauses.values())),
                event_count_ratio=float(n_ratio), duration_ratio=float(dur_ratio),
                baseline_distance_score=float(score))


def _small_trace(res, slow, target=4000):
    def ds(x):
        a = np.asarray(x, np.float32)
        return a[::max(1, int(np.ceil(a.size / target)))] if a.size else a
    stride = max(1, int(np.ceil(len(res["rate_E"]) / target)))
    out = dict(trace_dt_ms=np.asarray([DT * stride], np.float32),
               rate_E=ds(res["rate_E"]),
               tau_eff_ratio_min=ds(slow.trace_tau_eff_ratio_min),
               clip_frac=ds(slow.trace_conductance_clip_frac),
               gEff_mean=ds(slow.trace_gEff_mean) if slow.trace_gEff_mean else np.asarray([], np.float32),
               gErec_mean=ds(slow.trace_gErec_mean) if slow.trace_gErec_mean else np.asarray([], np.float32))
    return out


def _baseline(S, T):
    """Current-model slow-off reference workpoint (same anchor the accepted pilot used)."""
    cfg = dict(use_z=False, use_m=False, membrane_mode="current")
    res, slow = _run(S, cfg, T, early_stop=False)
    baseline = OLD.compute_baseline_ref(res, DT)
    event_bar = OLD.slowoff_event_bar(res, DT)
    rm, events, af, _, runaway = OLD.extract_run_metrics(res, DT, baseline, event_bar=event_bar)
    payload = dict(seed=S["seed"], T=T, cfg=cfg, baseline=dataclasses.asdict(baseline), event_bar=event_bar,
                   n_detected=len(events), n_returning=sum(bool(e["returned"]) for e in events),
                   runaway_ms=runaway, profile=_event_profile(events, res["rate_E"], DT))
    del res, slow, af
    gc.collect()
    return baseline, event_bar, payload


# ----------------------------------------------------------------- workers
def _cell(S, cfg, T, baseline, event_bar, label, *, kick_boost=0.0, t_kick=1e9):
    t0 = time.time()
    try:
        res, slow = _run(S, cfg, T, kick_boost=kick_boost, t_kick=t_kick, early_stop=True)
    except (FloatingPointError, ValueError) as exc:
        return dict(label=label, seed=S["seed"], T=T, cfg=cfg, numerical_safe=False, error=str(exc),
                    phenotype="numerically_unsafe", wall_s=round(time.time() - t0, 2),
                    peak_rss_gb=round(_self_rss_gb(), 3))
    rm, events, af, bin_w, runaway = OLD.extract_run_metrics(res, DT, baseline, event_bar=event_bar)
    phenotype = classify_mz_run(rm, baseline, runaway)
    num = _numerical(S, res, slow)
    profile = _event_profile(events, res["rate_E"], DT)
    wp = _workpoint_distance(profile, baseline)
    row = dict(label=label, seed=S["seed"], T=T, cfg=cfg, kick_boost=kick_boost, t_kick=t_kick,
               phenotype=phenotype, runaway_ms=runaway, numerical_safe=num["safe"], numerical=num,
               n_detected=len(events), n_returning=int(sum(bool(e["returned"]) for e in events)),
               event_profile=profile, workpoint=wp, wall_s=round(time.time() - t0, 2),
               peak_rss_gb=round(_self_rss_gb(), 3))
    _write_npz(os.path.join(_CTX["run_dir"], "per_cell", f"{label}_seed{S['seed']}_trace.npz"),
               **_small_trace(res, slow))
    _write_json(os.path.join(_CTX["run_dir"], "per_cell", f"{label}_seed{S['seed']}.json"), row)
    del res, slow, af
    gc.collect()
    return row


def _cell_task(task):
    return _cell(_CTX["S"], task["cfg"], _CTX["T"], _CTX["baseline"], _CTX["event_bar"], task["label"],
                 kick_boost=float(task.get("kick_boost", 0.0)), t_kick=float(task.get("t_kick", 1e9)))


def _pool(tasks, workers):
    if workers <= 1 or len(tasks) == 1:
        return [_cell_task(t) for t in tasks]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        return pool.map(_cell_task, tasks)


# ----------------------------------------------------------------- commands
def _cmd_smoke(args):
    run_id = _run_id("smoke")
    run_dir = os.path.join(OUT_ROOT, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    plan = _plan_workers(args.T, 1)
    _resource_log(run_dir, "smoke_start", plan)
    print(f"[smoke] build L=20 seed={args.seed}; {plan}", flush=True)
    S = PP.build_substrate(args.seed)
    cfg = _fc_cfg(args.c_E, gamma=0.0, z=False, fail_on_clip=False)   # smoke tolerates clip (reports it)
    t0 = time.time()
    res, slow = _run(S, cfg, args.T, kick_boost=args.kick_boost,
                     t_kick=(PP.T_KICK if args.kick_boost > 0 else 1e9), early_stop=True)
    num = _numerical(S, res, slow)
    row = dict(label="fc_smoke", seed=args.seed, T=args.T, cfg=cfg, kick_boost=args.kick_boost,
               numerical=num, peak_rate_hz=float(np.max(res["rate_E"])),
               mean_rate_hz=float(np.mean(res["rate_E"])),
               peak_rss_gb=round(_self_rss_gb(), 3), wall_s=round(time.time() - t0, 2))
    _resource_log(run_dir, "smoke_done", dict(peak_rss_gb=row["peak_rss_gb"]))
    summary = dict(run_id=run_id, tag="smoke", stage="0A", row=row, resource_plan=plan,
                   provenance=_provenance())
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_json(os.path.join(OUT_ROOT, "latest_smoke.json"), dict(run_id=run_id, path=run_dir))
    print(f"[smoke] safe={num['safe']} tau_min={num['tau_eff_min_ms']:.3f}ms clip={num['max_clip_fraction']:.4f} "
          f"peak={row['peak_rate_hz']:.1f}Hz rss={row['peak_rss_gb']:.2f}GiB wall={row['wall_s']}s", flush=True)
    return summary


def _cmd_workpoint(args):
    run_id = _run_id("workpoint")
    run_dir = os.path.join(OUT_ROOT, "runs", run_id)
    os.makedirs(os.path.join(run_dir, "per_cell"), exist_ok=True)
    c_es = [float(x) for x in args.c_E.split(",")]
    plan = _plan_workers(args.T, args.workers)
    _resource_log(run_dir, "workpoint_start", plan)
    print(f"[workpoint] build L=20 seed={args.seed}; c_E={c_es}; {plan}", flush=True)
    S = PP.build_substrate(args.seed)
    baseline, event_bar, base_payload = _baseline(S, args.T)
    _write_json(os.path.join(run_dir, "baseline_current.json"), base_payload)
    _resource_log(run_dir, "baseline_done", dict(self_rss_gb=round(_self_rss_gb(), 3),
                                                 n_base_events=baseline.n_events))
    if baseline.n_events < 5:
        summary = dict(run_id=run_id, verdict="baseline_anchor_fail", baseline=base_payload)
        _write_json(os.path.join(run_dir, "summary.json"), summary)
        raise SystemExit(f"baseline anchor fail: only {baseline.n_events} slow-off events")
    _CTX.update(S=S, T=float(args.T), baseline=baseline, event_bar=event_bar, run_dir=run_dir)
    # fail_on_clip=False so a c_E cell completes + is characterized (startup-volley clip must not abort an
    # otherwise-clean interictal cell); numerical safety is judged on the SETTLED window below.
    tasks = [dict(label=f"fc_cE{c:g}_glob0", cfg=_fc_cfg(c, gamma=0.0, z=False, fail_on_clip=False))
             for c in c_es]
    workers = max(1, plan["workers"])
    print(f"[workpoint] {len(tasks)} c_E cells, workers={workers}", flush=True)
    rows = _pool(tasks, workers)
    _resource_log(run_dir, "bracket_done", dict(self_rss_gb=round(_self_rss_gb(), 3)))
    # a workpoint candidate = settled-numerically-safe AND has returning interictal events
    safe = [r for r in rows if r.get("numerical", {}).get("settled_safe")
            and r.get("event_profile", {}).get("n_returning", 0) > 0
            and r.get("phenotype") not in ("runaway", "numerically_unsafe")]
    ranked = sorted(safe, key=lambda r: r["workpoint"]["baseline_distance_score"])
    pick = ranked[0]["cfg"]["c_E"] if ranked else None
    summary = dict(
        run_id=run_id, tag="workpoint", stage="0B", seed=args.seed, T=args.T,
        resource_plan=plan, baseline=base_payload,
        reference_workpoint=dict(n_events=baseline.n_events, dur_med=baseline.dur_med,
                                 part_lo=baseline.part_lo, part_hi=baseline.part_hi,
                                 act_lo=baseline.act_lo, act_hi=baseline.act_hi),
        rows=rows,
        picked_c_E=pick,
        ranking=[dict(c_E=r["cfg"]["c_E"], score=r["workpoint"]["baseline_distance_score"],
                      all_bands=r["workpoint"]["all_bands"], profile=r["event_profile"],
                      numerical_safe=r["numerical_safe"]) for r in ranked],
        verdict=("candidate" if pick is not None else "no_go_no_safe_workpoint_c_E"),
        provenance=_provenance(),
    )
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_json(os.path.join(OUT_ROOT, "latest_workpoint.json"), dict(run_id=run_id, path=run_dir))
    for r in rows:
        wp = r.get("workpoint", {})
        print(f"  {r['label']}: safe={r.get('numerical_safe')} pheno={r.get('phenotype')} "
              f"n_ret={r['event_profile']['n_returning'] if 'event_profile' in r else 'NA'} "
              f"score={wp.get('baseline_distance_score', float('nan')):.3f} bands={wp.get('all_bands')} "
              f"clip={r.get('numerical', {}).get('max_clip_fraction', 'NA')}", flush=True)
    print(f"[workpoint] verdict={summary['verdict']} picked_c_E={pick}", flush=True)
    return summary


def _provenance():
    return dict(git_sha=_git_sha(), argv=sys.argv,
                git_status=subprocess.run(["git", "-C", ROOT, "status", "--short"],
                                          capture_output=True, text=True).stdout.splitlines(),
                file_sha256={rel: _sha(os.path.join(ROOT, rel)) for rel in (
                    "src/snn_engine/kick_probe.py", "src/snn_engine/mz_slow_vars.py",
                    "scripts/run_topic4_mz_fcxr.py")})


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("smoke")
    s.add_argument("--confirm-run", action="store_true")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--T", type=float, default=1000.0)
    s.add_argument("--c-E", dest="c_E", type=float, default=1.0)
    s.add_argument("--kick-boost", type=float, default=0.0)

    w = sub.add_parser("workpoint")
    w.add_argument("--confirm-run", action="store_true")
    w.add_argument("--seed", type=int, default=1)
    w.add_argument("--T", type=float, default=8000.0)
    w.add_argument("--c-E", dest="c_E", default="0.85,1.0,1.15")
    w.add_argument("--workers", type=int, default=2)

    args = ap.parse_args(argv)
    if not getattr(args, "confirm_run", False):
        raise SystemExit("REFUSING: simulations require --confirm-run")
    _assert_engine_blessed()
    with _launcher_lock():
        if args.cmd == "smoke":
            _cmd_smoke(args)
        elif args.cmd == "workpoint":
            _cmd_workpoint(args)


if __name__ == "__main__":
    main()
