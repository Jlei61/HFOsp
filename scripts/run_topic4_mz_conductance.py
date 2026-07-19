"""L=20 MZ conductance/global-GABA cheap-first runner.

Nothing runs on import.  Every simulation command requires ``--confirm-run``.
The launcher builds one network and forks workers so the sparse substrate is
shared copy-on-write.  No nested pool and BLAS threads are pinned to one.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-mz-conductance")

import argparse
from contextlib import contextmanager
import dataclasses
import fcntl
import gc
import hashlib
import json
import multiprocessing as mp
import re
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_mz_slowvars as OLD  # noqa: E402
from kick_probe import DUR_KICK, simulate_kick  # noqa: E402
from lfp import LFPRecorder  # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_conductance import oscillation_metrics, staircase_metrics  # noqa: E402
from src.topic4_mz_slowvars import MZBaseline, classify_mz_run  # noqa: E402


OUT_ROOT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_conductance")
CONFIG_PATH = os.path.join(ROOT, "config", "topic4_mz_conductance.yaml")
CONTRACT = yaml.safe_load(open(CONFIG_PATH))
DT = 0.1
MAX_WORKERS = int(CONTRACT["resource"]["hard_worker_cap"])
_CTX = {}


def _git_sha():
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _task_hash(tasks):
    raw = json.dumps(tasks, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:10]


def _run_id(tag, tasks):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{stamp}_{os.getpid()}_{_git_sha()}_{_task_hash(tasks)}_{tag}"


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
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


def _mem_available_gb():
    with open("/proc/meminfo") as f:
        rows = {line.split(":", 1)[0]: line.split()[1:] for line in f}
    return float(rows["MemAvailable"][0]) / 1024.0 / 1024.0


def _swap_used_gb():
    with open("/proc/meminfo") as f:
        vals = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return (vals["SwapTotal"] - vals["SwapFree"]) / 1024.0 / 1024.0


def _resource_preflight(T, workers):
    if T >= 20000.0 and workers > int(CONTRACT["resource"]["long_T_ms_worker_cap"][20000]):
        raise SystemExit("T>=20s is capped at 2 workers while full E_spk_bool is retained")
    avail = _mem_available_gb()
    raster_gb = (float(T) / DT) * 32000.0 / (1024.0 ** 3)
    parent_est_gb = 8.0
    per_worker_est_gb = raster_gb + 3.0
    budget_gb = 0.70 * avail
    need_gb = parent_est_gb + int(workers) * per_worker_est_gb
    if avail < 40.0 or need_gb > budget_gb:
        raise SystemExit(
            f"resource preflight failed: available={avail:.1f}GiB, conservative need={need_gb:.1f}GiB, "
            f"70% budget={budget_gb:.1f}GiB"
        )
    return dict(mem_available_gb=round(avail, 2), swap_used_gb=round(_swap_used_gb(), 3),
                estimated_need_gb=round(need_gb, 2), budget_gb=round(budget_gb, 2))


@contextmanager
def _launcher_lock():
    os.makedirs(OUT_ROOT, exist_ok=True)
    path = os.path.join(OUT_ROOT, ".l20_launcher.lock")
    with open(path, "a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("another L=20 MZ conductance launcher is active; refusing duplicate build") from exc
        lock.seek(0); lock.truncate(); lock.write(f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n")
        lock.flush()
        try:
            yield path
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_tasks(tasks):
    labels = [str(t.get("label", "")) for t in tasks]
    if len(set(labels)) != len(labels):
        raise SystemExit("task labels must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", label) for label in labels):
        raise SystemExit("task labels may contain only letters, digits, dot, underscore and hyphen")
    by_label = {str(t["label"]): t for t in tasks}
    for task in tasks:
        cfg = task.get("cfg", {})
        if cfg.get("membrane_mode") == "conductance":
            cap = cfg.get("max_total_conductance")
            if cap is None or not np.isfinite(cap) or cap > 99.0:
                raise SystemExit(f"{task['label']}: conductance cap must be finite and <=99")
            if not cfg.get("fail_on_clip", False):
                raise SystemExit(f"{task['label']}: scientific screens require fail_on_clip=true")
        pair = task.get("paired_control_label")
        if pair is not None:
            if pair == task["label"] or pair not in by_label:
                raise SystemExit(f"{task['label']}: paired control must name a distinct task in the same spec")
            control = by_label[pair]
            if float(task.get("kick_boost", 0.0)) <= 0.0 or float(control.get("kick_boost", 0.0)) != 0.0:
                raise SystemExit(f"{task['label']}: paired lifecycle requires kicked response and zero-kick control")
            if "analysis_start_ms" not in task or task.get("analysis_start_ms") != control.get("analysis_start_ms"):
                raise SystemExit(f"{task['label']}: response/control require the same explicit analysis_start_ms")
            if cfg != control.get("cfg", {}):
                raise SystemExit(f"{task['label']}: response/control must use identical mechanism config")


def _assert_engine_blessed():
    manifest = os.path.join(ROOT, "results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
    recorded = json.load(open(manifest))
    for rel, expected in recorded.items():
        current = _sha(os.path.join(ROOT, rel))
        if current != expected:
            raise SystemExit(f"engine not blessed: {rel} {current[:12]} != {expected[:12]}")


def _core_mask(S):
    return OLD.build_core_masks(S)


def _make_slow(S, cfg):
    return MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"], core_mask_E=_core_mask(S))


def _global_source(cfg):
    mode = cfg.get("global_gaba_mode", "replace")
    return f"{mode}_mean_received_gaba_surrogate"


def _run(S, cfg, T, *, kick_boost=0.0, t_kick=1e9, early_stop=True, lfp_recorder=None):
    p = dataclasses.replace(S["p"], T=float(T))
    slow = _make_slow(S, cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    res = simulate_kick(
        p, S["net"], float(kick_boost), slow=slow,
        kick_center=list(S["src_xy"]), r_kick=PP.R_KICK, t_kick=float(t_kick),
        V_th_per_neuron=S["vth"], early_stop_runaway=early_stop,
        lfp_recorder=lfp_recorder,
    )
    return res, slow


def _onset_rank(E_spk_bool, dt, t_lo, t_hi):
    """First-spike onset and normalized onset rank in one registered window."""
    nsteps, ne = E_spk_bool.shape
    i0 = max(0, min(nsteps, int(np.floor(float(t_lo) / dt))))
    i1 = max(i0, min(nsteps, int(np.ceil(float(t_hi) / dt))))
    onset = np.full(ne, np.nan, dtype=np.float32)
    rank = np.full(ne, np.nan, dtype=np.float32)
    if i1 <= i0:
        return onset, rank
    sub = np.asarray(E_spk_bool[i0:i1], bool)
    fired = sub.any(axis=0)
    if not fired.any():
        return onset, rank
    first = np.argmax(sub[:, fired], axis=0).astype(np.float64) * float(dt) + i0 * float(dt)
    onset[fired] = first.astype(np.float32)
    order = np.argsort(np.argsort(first, kind="stable"), kind="stable").astype(np.float64)
    if order.size > 1:
        order /= float(order.size - 1)
    else:
        order[:] = 0.0
    rank[fired] = order.astype(np.float32)
    return onset, rank


def _small_trace(res, slow, target=4000):
    def ds(x):
        a = np.asarray(x, np.float32)
        return a[::max(1, int(np.ceil(a.size / target)))] if a.size else a
    stride = max(1, int(np.ceil(len(res["rate_E"]) / target)))
    return dict(
        trace_dt_ms=np.asarray([DT * stride], np.float32),
        rate_E=ds(res["rate_E"]), z_mean=ds(slow.trace_z_mean), z_min=ds(slow.trace_z_min),
        z_core_mean=ds(slow.trace_z_core_mean),
        m_mean=ds(slow.trace_m_mean), phi_mean=ds(slow.trace_phi_mean),
        gaba_received_mean=ds(slow.trace_gaba_received_mean),
        global_pre_z=ds(slow.trace_global_pre_z), z_sensor_mean=ds(slow.trace_z_sensor_mean),
        gI_mean=ds(slow.trace_gI_mean), gM_mean=ds(slow.trace_gM_mean),
        tau_eff_ratio_min=ds(slow.trace_tau_eff_ratio_min),
        clip_frac=ds(slow.trace_conductance_clip_frac),
    )


def _baseline(S, T):
    t0 = time.time()
    cfg = dict(use_z=False, use_m=False, membrane_mode="current")
    res, slow = _run(S, cfg, T, early_stop=False)
    baseline = OLD.compute_baseline_ref(res, DT)
    event_bar = OLD.slowoff_event_bar(res, DT)
    rm, events, af, _, runaway = OLD.extract_run_metrics(res, DT, baseline, event_bar=event_bar)
    payload = dict(
        seed=S["seed"], T=T, cfg=cfg, baseline=dataclasses.asdict(baseline), event_bar=event_bar,
        n_detected=len(events), n_returning=sum(bool(e["returned"]) for e in events),
        runaway_ms=runaway, run_metrics=rm, wall_s=round(time.time() - t0, 2),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3),
    )
    trace = _small_trace(res, slow)
    del res, slow, af
    gc.collect()
    return baseline, event_bar, payload, trace


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


def _workpoint_gate(profile, baseline, *, phenotype, numerical_safe):
    n_ratio = profile["n_returning"] / max(1, baseline.n_events)
    dur_ratio = profile["duration_median_ms"] / max(1e-9, baseline.dur_med)
    part = profile["participation_median"]
    peak = profile["peak_rate_median_hz"]
    clauses = dict(
        phenotype_interictal=(phenotype == "interictal_like"),
        event_count_ratio=bool(0.5 <= n_ratio <= 1.5),
        duration_ratio=bool(0.5 <= dur_ratio <= 2.0),
        participation_band=bool(0.5 * baseline.part_lo <= part <= 2.0 * baseline.part_hi),
        peak_rate_band=bool(0.5 * baseline.act_lo <= peak <= 2.0 * baseline.act_hi),
        numerical_safe=bool(numerical_safe),
    )
    score = (
        abs(np.log(max(n_ratio, 1e-9))) + abs(np.log(max(dur_ratio, 1e-9)))
        + abs(part - 0.5 * (baseline.part_lo + baseline.part_hi)) / max(baseline.part_hi, 1e-9)
        + abs(peak - 0.5 * (baseline.act_lo + baseline.act_hi)) / max(baseline.act_hi, 1e-9)
    )
    return dict(candidate=bool(all(clauses.values())), clauses=clauses,
                event_count_ratio=float(n_ratio), duration_ratio=float(dur_ratio),
                baseline_distance_score=float(score))


def _worker(task):
    S, baseline, event_bar, T, run_dir = (_CTX[k] for k in ("S", "baseline", "event_bar", "T", "run_dir"))
    label = task["label"]
    cfg = task["cfg"]
    kick_boost = float(task.get("kick_boost", 0.0))
    t_kick = float(task.get("t_kick", 1e9))
    t0 = time.time()
    try:
        res, slow = _run(S, cfg, T, kick_boost=kick_boost, t_kick=t_kick, early_stop=True)
    except (FloatingPointError, ValueError) as exc:
        row = dict(label=label, seed=S["seed"], T=T, cfg=cfg, kick_boost=kick_boost, t_kick=t_kick,
                   phenotype="numerically_unsafe", numerical_safe=False, error=str(exc),
                   wall_s=round(time.time() - t0, 2), global_source=_global_source(cfg))
        _write_json(os.path.join(run_dir, "per_cell", f"{label}_seed{S['seed']}.json"), row)
        return row
    rm, events, af, bin_w, runaway = OLD.extract_run_metrics(res, DT, baseline, event_bar=event_bar)
    phenotype = classify_mz_run(rm, baseline, runaway)
    stair = staircase_metrics(events, slow.trace_z_core_mean, DT, transition_ms=runaway) if cfg.get("use_z") else None
    if stair is not None:
        stair["coordinate"] = "registered_two_core_mean_D"
    analysis_start_ms = task.get("analysis_start_ms", t_kick + DUR_KICK)
    osc = oscillation_metrics(
        res["rate_E"], DT, analysis_start_ms=analysis_start_ms,
        baseline_rate=baseline.baseline_rate, baseline_sigma=baseline.sigma_rate,
        active_fraction=af, af_bin_ms=bin_w, baseline_af_q95=baseline.part_hi,
        runaway=runaway is not None,
    ) if (kick_boost > 0 or "analysis_start_ms" in task) else None
    tau_eff_min_ms = float(S["p"].tau_m_E * min(slow.trace_tau_eff_ratio_min))
    max_clip = float(max(slow.trace_conductance_clip_frac))
    numerical_safe = bool(np.isfinite(tau_eff_min_ms) and tau_eff_min_ms >= 2 * DT and max_clip == 0.0)
    profile = _event_profile(events, res["rate_E"], DT)
    is_workpoint_arm = bool(kick_boost == 0.0 and not cfg.get("use_z") and not cfg.get("use_m")
                            and not cfg.get("use_phi") and cfg.get("membrane_mode") == "conductance")
    workpoint = _workpoint_gate(profile, baseline, phenotype=phenotype, numerical_safe=numerical_safe) \
        if is_workpoint_arm else None
    late_cut = max(0.0, T - 2000.0)
    late_returning = int(sum(bool(e.get("returned")) and float(e["t_on"]) >= late_cut for e in events))
    recovery_candidate = bool(osc and osc["tail_rate_band"] and late_returning >= 1 and runaway is None)
    row = dict(
        label=label, seed=S["seed"], T=T, cfg=cfg, kick_boost=kick_boost, t_kick=t_kick,
        phenotype=phenotype, runaway_ms=runaway, run_metrics=rm, staircase=stair,
        oscillation=osc, event_profile=profile, workpoint=workpoint, n_detected=len(events),
        n_detector_returning=int(sum(bool(e["returned"]) for e in events)),
        late_returning_events=late_returning, recovery_candidate=recovery_candidate,
        tau_eff_min_ms=tau_eff_min_ms, max_clip_fraction=max_clip,
        numerical_safe=numerical_safe,
        wall_s=round(time.time() - t0, 2),
        peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3),
        global_source=_global_source(cfg),
    )
    cell_dir = os.path.join(run_dir, "per_cell")
    _write_json(os.path.join(cell_dir, f"{label}_seed{S['seed']}.json"), row)
    _write_npz(os.path.join(cell_dir, f"{label}_seed{S['seed']}_trace.npz"), **_small_trace(res, slow))
    return row


def _pool(tasks, workers):
    workers = max(1, min(int(workers), MAX_WORKERS, len(tasks)))
    if workers == 1:
        return [_worker(t) for t in tasks]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        return pool.map(_worker, tasks)


def _base_cfg(gain, *, gamma=0.0, e_gaba=0.0, global_mode="replace"):
    return dict(
        use_z=False, use_m=False, use_phi=False, membrane_mode="conductance",
        gaba_gain=float(gain), m_conductance_gain=1.0,
        global_gaba_fraction=float(gamma), global_gaba_mode=str(global_mode), z_scope="total",
        v_match=18.0, e_gaba=float(e_gaba), e_k=0.0,
        max_total_conductance=99.0, fail_on_clip=True,
    )


def _smoke(task, *, seed, T, resource_preflight):
    """Numerical/resource smoke only; it does not issue a phenotype or workpoint verdict."""
    run_id = _run_id("smoke", [task])
    run_dir = os.path.join(OUT_ROOT, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[smoke-build] L=20 seed={seed}; single process", flush=True)
    t0 = time.time()
    S = PP.build_substrate(seed)
    try:
        res, slow = _run(S, task["cfg"], T, early_stop=True)
        tau_eff_min_ms = float(S["p"].tau_m_E * min(slow.trace_tau_eff_ratio_min))
        max_clip = float(max(slow.trace_conductance_clip_frac))
        finite = bool(np.all(np.isfinite(res["rate_E"])))
        numerical_safe = bool(finite and tau_eff_min_ms >= 2 * DT and max_clip == 0.0)
        row = dict(
            label=task["label"], seed=seed, T=T, cfg=task["cfg"], numerical_safe=numerical_safe,
            finite_rate=finite, tau_eff_min_ms=tau_eff_min_ms, max_clip_fraction=max_clip,
            peak_rate_hz=float(np.max(res["rate_E"])), wall_s=round(time.time() - t0, 2),
            peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3),
            global_source=_global_source(task["cfg"]),
        )
        _write_npz(os.path.join(run_dir, "smoke_trace.npz"), **_small_trace(res, slow))
    except (FloatingPointError, ValueError) as exc:
        row = dict(label=task["label"], seed=seed, T=T, cfg=task["cfg"], numerical_safe=False,
                   error=str(exc), wall_s=round(time.time() - t0, 2),
                   peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3),
                   global_source=_global_source(task["cfg"]))
    summary = dict(
        run_id=run_id, tag="smoke", verdict=("numerically_safe" if row["numerical_safe"] else "unsafe"),
        row=row, resource_preflight=resource_preflight,
        substrate=dict(subject=PP.SUBJECT, montage=PP.MONTAGE, L=PP.L, density=PP.DENSITY,
                       N=S["N"], NE=S["NE"], NI=S["NI"], AR=2.0, drive=PP.DRIVE, g=PP.G),
        provenance=dict(git_sha=_git_sha(), argv=sys.argv, task_hash=_task_hash([task])),
    )
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_json(os.path.join(OUT_ROOT, "latest_smoke.json"), dict(run_id=run_id, path=run_dir))
    print(f"[smoke] safe={row['numerical_safe']} tau_min={row.get('tau_eff_min_ms', float('nan')):.3f}ms "
          f"clip={row.get('max_clip_fraction', float('nan')):.3f} rss={row['peak_rss_gb']:.2f}GiB", flush=True)
    return summary, run_dir


def _execute(tasks, *, seed, T, workers, tag, resource_preflight):
    run_id = _run_id(tag, tasks)
    run_dir = os.path.join(OUT_ROOT, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[build] L=20 seed={seed}; one parent substrate, fork/COW workers<=4", flush=True)
    S = PP.build_substrate(seed)
    print(f"[baseline] current slow-off T={T}ms", flush=True)
    baseline, event_bar, base_row, base_trace = _baseline(S, T)
    _write_json(os.path.join(run_dir, "baseline_current.json"), base_row)
    _write_npz(os.path.join(run_dir, "baseline_current_trace.npz"), **base_trace)
    if baseline.n_events < 5:
        summary = dict(run_id=run_id, verdict="baseline_anchor_fail", baseline=base_row, rows=[])
        _write_json(os.path.join(run_dir, "summary.json"), summary)
        return summary, run_dir
    _CTX.update(S=S, baseline=baseline, event_bar=event_bar, T=float(T), run_dir=run_dir)
    print(f"[screen] {len(tasks)} cells workers={min(workers, MAX_WORKERS, len(tasks))}", flush=True)
    rows = _pool(tasks, workers)
    by_label = {r["label"]: r for r in rows}
    for task in tasks:
        pair = task.get("paired_control_label")
        if not pair or task["label"] not in by_label or pair not in by_label:
            continue
        row, control = by_label[task["label"]], by_label[pair]
        ro, co = row.get("oscillation"), control.get("oscillation")
        paired_pass = bool(
            ro and co and ro["high_duration_ms"] >= co["high_duration_ms"] + 500.0
            and ro["recruitment_pass"] and ro["oscillatory_candidate"]
        )
        row["paired_control"] = dict(label=pair, response_pass=paired_pass,
                                      high_duration_delta_ms=(ro["high_duration_ms"] - co["high_duration_ms"])
                                      if ro and co else float("nan"))
        row["full_lifecycle_candidate"] = bool(paired_pass and row.get("recovery_candidate"))
        _write_json(os.path.join(run_dir, "per_cell", f"{row['label']}_seed{seed}.json"), row)
    for row in rows:
        print(f"  {row['label']}: {row['phenotype']} events={row.get('n_detected', 'NA')} "
              f"returned={row.get('n_detector_returning', 'NA')} runaway={row.get('runaway_ms')} "
              f"tau_min={row.get('tau_eff_min_ms', float('nan')):.3f}ms "
              f"clip={row.get('max_clip_fraction', float('nan')):.3f}", flush=True)
    candidates = sorted(
        (r for r in rows if (r.get("workpoint") or {}).get("candidate")),
        key=lambda r: r["workpoint"]["baseline_distance_score"],
    )
    summary = dict(
        run_id=run_id, tag=tag, seed=seed, T=T, workers=min(workers, MAX_WORKERS),
        substrate=dict(subject=PP.SUBJECT, montage=PP.MONTAGE, L=PP.L, density=PP.DENSITY,
                       N=S["N"], NE=S["NE"], NI=S["NI"], AR=2.0, drive=PP.DRIVE, g=PP.G),
        baseline=base_row, rows=rows, resource_preflight=resource_preflight,
        workpoint_candidates=[r["label"] for r in candidates[:2]],
        provenance=dict(git_sha=_git_sha(), git_status=subprocess.run(
                            ["git", "-C", ROOT, "status", "--short"], capture_output=True, text=True).stdout.splitlines(),
                        argv=sys.argv, task_hash=_task_hash(tasks),
                        global_sources=sorted({_global_source(t["cfg"]) for t in tasks}),
                        file_sha256={os.path.relpath(p, ROOT): _sha(p) for p in (
                            os.path.join(ROOT, "src", "snn_engine", "kick_probe.py"),
                            os.path.join(ROOT, "src", "snn_engine", "mz_slow_vars.py"),
                            os.path.join(ROOT, "src", "topic4_mz_conductance.py"),
                            os.path.join(ROOT, "scripts", "run_topic4_mz_conductance.py"),
                            CONFIG_PATH,
                            os.path.join(ROOT, "docs", "superpowers", "specs",
                                         "2026-07-19-topic4-mz-conductance-global-inhibition-design.md"),
                        )}),
    )
    _write_json(os.path.join(run_dir, "summary.json"), summary)
    _write_json(os.path.join(OUT_ROOT, f"latest_{tag}.json"), dict(run_id=run_id, path=run_dir))
    return summary, run_dir


def _capture_figure(task, *, seed, T, baseline_json, resource_preflight):
    """Run one deterministic L=20 cell and retain only compact paper-figure evidence.

    The full E spike raster exists only while the simulation is in memory.  The saved
    artifact contains two NE-length first-spike maps plus scalar/15-contact traces; it
    deliberately never serializes the T-by-NE raster.
    """
    base_payload = json.load(open(baseline_json))
    if int(base_payload["seed"]) != int(seed):
        raise SystemExit("figure capture baseline seed must match --seed")
    if abs(float(base_payload["T"]) - float(T)) > 1e-9:
        raise SystemExit("figure capture baseline T must match --T")
    baseline = MZBaseline(**base_payload["baseline"])
    event_bar = float(base_payload["event_bar"])

    run_id = _run_id("figure_capture", [task])
    run_dir = os.path.join(OUT_ROOT, "figure_capture", run_id)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[figure-capture build] L=20 seed={seed}; one process", flush=True)
    t0 = time.time()
    S = PP.build_substrate(seed)
    montage = S["reg"]["montage_sheet"]
    contacts = np.asarray(montage.contacts, float)
    names = np.asarray(montage.names, str)
    recorder = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"], sites=contacts)
    res, slow = _run(
        S, task["cfg"], T,
        kick_boost=float(task.get("kick_boost", 0.0)),
        t_kick=float(task.get("t_kick", 1e9)),
        early_stop=True,
        lfp_recorder=recorder,
    )
    rm, events, af, bin_w, runaway = OLD.extract_run_metrics(
        res, DT, baseline, event_bar=event_bar,
    )
    returning = [
        e for e in events
        if bool(e.get("returned", False))
        and float(e["t_on"]) >= 500.0
        and (runaway is None or float(e["t_off"]) <= float(runaway) - 200.0)
    ]
    if not returning:
        raise RuntimeError("figure capture found no eligible pre-runaway returning event")
    returning = sorted(returning, key=lambda e: (float(e["peak_ext"]), float(e["t_on"])))
    rep = returning[len(returning) // 2]  # deterministic median-participation exemplar
    ret_lo = max(0.0, float(rep["t_on"]) - 5.0)
    ret_hi = min(float(res["times"][-1]) + DT, float(rep["t_off"]) + 10.0)
    onset_ret, rank_ret = _onset_rank(res["E_spk_bool"], DT, ret_lo, ret_hi)

    if runaway is None:
        raise RuntimeError("figure capture target must contain the registered delayed runaway")
    run_lo = max(0.0, float(runaway) - 30.0)
    run_hi = min(float(res["times"][-1]) + DT, float(runaway) + 100.0)
    onset_run, rank_run = _onset_rank(res["E_spk_bool"], DT, run_lo, run_hi)

    tau_eff_min_ms = float(S["p"].tau_m_E * min(slow.trace_tau_eff_ratio_min))
    max_clip = float(max(slow.trace_conductance_clip_frac))
    numerical_safe = bool(
        np.all(np.isfinite(res["rate_E"]))
        and tau_eff_min_ms >= 2 * DT
        and max_clip == 0.0
    )
    artifact = os.path.join(run_dir, "mz_conductance_current_dynamics.npz")
    _write_npz(
        artifact,
        times=np.asarray(res["times"], np.float32),
        rate_E=np.asarray(res["rate_E"], np.float32),
        active_fraction=np.asarray(af, np.float32),
        active_fraction_bin_ms=np.asarray([bin_w], np.float32),
        z_mean=np.asarray(slow.trace_z_mean, np.float32),
        z_core_mean=np.asarray(slow.trace_z_core_mean, np.float32),
        gI_mean=np.asarray(slow.trace_gI_mean, np.float32),
        gM_mean=np.asarray(slow.trace_gM_mean, np.float32),
        lfp_trace=np.asarray(res["lfp_trace"], np.float32),
        contacts=np.asarray(contacts, np.float32),
        names=names,
        posE=np.asarray(S["posE"], np.float32),
        vth=np.asarray(S["vth"][:S["NE"]], np.float32),
        src_xy=np.asarray(S["src_xy"], np.float32),
        snk_xy=np.asarray(S["snk_xy"], np.float32),
        center=np.asarray(S["center"], np.float32),
        axis_unit=np.asarray(S["axis_unit"], np.float32),
        onset_returning_ms=onset_ret,
        onset_rank_returning=rank_ret,
        onset_runaway_ms=onset_run,
        onset_rank_runaway=rank_run,
        returning_window_ms=np.asarray([ret_lo, ret_hi], np.float32),
        runaway_window_ms=np.asarray([run_lo, run_hi], np.float32),
        returning_event_intervals_ms=np.asarray(
            [[e["t_on"], e["t_off"]] for e in returning], np.float32,
        ),
        all_event_intervals_ms=np.asarray(
            [[e["t_on"], e["t_off"]] for e in events], np.float32,
        ),
        runaway_ms=np.asarray([runaway], np.float32),
        L=np.asarray([S["L"]], np.float32),
        core_radius=np.asarray([PP.CORE_R], np.float32),
    )
    meta = dict(
        run_id=run_id,
        label=task["label"],
        seed=int(seed),
        T=float(T),
        cfg=task["cfg"],
        artifact=os.path.relpath(artifact, ROOT),
        selection_rule=(
            "median participation among returned events with t_on>=500 ms and "
            "t_off<=runaway-200 ms"
        ),
        representative_returning_event=rep,
        returning_window_ms=[ret_lo, ret_hi],
        runaway_window_ms=[run_lo, run_hi],
        n_returning_eligible=len(returning),
        n_events_total=len(events),
        runaway_ms=float(runaway),
        run_metrics=rm,
        spatial_counts=dict(
            returning=int(np.isfinite(rank_ret).sum()),
            early_runaway=int(np.isfinite(rank_run).sum()),
            NE=int(S["NE"]),
        ),
        numerical=dict(
            safe=numerical_safe,
            tau_eff_min_ms=tau_eff_min_ms,
            max_clip_fraction=max_clip,
            peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024, 3),
            wall_s=round(time.time() - t0, 2),
        ),
        resource_preflight=resource_preflight,
        baseline_source=os.path.relpath(os.path.abspath(baseline_json), ROOT),
        substrate=dict(
            subject=PP.SUBJECT, montage=PP.MONTAGE, L=PP.L, density=PP.DENSITY,
            N=S["N"], NE=S["NE"], NI=S["NI"], AR=2.0, drive=PP.DRIVE, g=PP.G,
        ),
        provenance=dict(
            git_sha=_git_sha(),
            git_status=subprocess.run(
                ["git", "-C", ROOT, "status", "--short"], capture_output=True, text=True,
            ).stdout.splitlines(),
            argv=sys.argv,
            task_hash=_task_hash([task]),
            note="full T-by-NE spike raster was not serialized",
        ),
    )
    if not numerical_safe:
        raise RuntimeError(f"figure capture numerical gate failed: {meta['numerical']}")
    meta_path = os.path.join(run_dir, "mz_conductance_current_dynamics.json")
    _write_json(meta_path, meta)
    _write_json(
        os.path.join(OUT_ROOT, "latest_figure_capture.json"),
        dict(run_id=run_id, path=run_dir, artifact=artifact, metadata=meta_path),
    )
    print(
        f"[figure-capture] returning={len(returning)} rep={rep['t_on']:.1f}-{rep['t_off']:.1f}ms "
        f"runaway={runaway:.1f}ms spatial={meta['spatial_counts']} "
        f"rss={meta['numerical']['peak_rss_gb']:.2f}GiB",
        flush=True,
    )
    del res, slow
    gc.collect()
    return meta, run_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("baseline-screen")
    b.add_argument("--confirm-run", action="store_true")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--T", type=float, default=8000.0)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--gains", default="0.5,0.75,1.0,1.25")
    b.add_argument("--e-gaba", type=float, default=0.0,
                   help="reversal in engine coordinates; 0=V_L current-equivalent primary")

    m = sub.add_parser("smoke")
    m.add_argument("--confirm-run", action="store_true")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--T", type=float, default=500.0)
    m.add_argument("--workers", type=int, default=1)
    m.add_argument("--gain", type=float, default=0.25)
    m.add_argument("--gamma", type=float, default=0.0)
    m.add_argument("--e-gaba", type=float, default=0.0)

    s = sub.add_parser("screen")
    s.add_argument("--confirm-run", action="store_true")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--T", type=float, default=12000.0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--spec", required=True, help="JSON list of {label,cfg,kick_boost?,t_kick?}")
    s.add_argument("--tag", default="mechanism_screen")

    c = sub.add_parser("capture-figure")
    c.add_argument("--confirm-run", action="store_true")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--T", type=float, default=8000.0)
    c.add_argument("--workers", type=int, default=1)
    c.add_argument("--spec", required=True, help="JSON list containing exactly one target cell")
    c.add_argument("--baseline-json", required=True, help="same-seed current baseline JSON")

    args = ap.parse_args(argv)
    if not args.confirm_run:
        raise SystemExit("REFUSING: simulations require --confirm-run")
    if args.workers < 1 or args.workers > MAX_WORKERS:
        raise SystemExit(f"--workers must be 1..{MAX_WORKERS} in the cheap-first stage")
    if args.cmd == "baseline-screen":
        gains = [float(x) for x in args.gains.split(",")]
        tasks = [dict(label=f"cond_eg{args.e_gaba:g}_g{g:g}",
                      cfg=_base_cfg(g, e_gaba=args.e_gaba)) for g in gains]
    elif args.cmd == "smoke":
        if args.workers != 1:
            raise SystemExit("smoke is single-process only")
        tasks = [dict(label=f"smoke_eg{args.e_gaba:g}_g{args.gain:g}_gamma{args.gamma:g}",
                      cfg=_base_cfg(args.gain, gamma=args.gamma, e_gaba=args.e_gaba))]
    else:
        tasks = json.load(open(args.spec))
        if not isinstance(tasks, list) or not tasks:
            raise SystemExit("--spec must contain a non-empty JSON list")
        if args.cmd == "capture-figure" and len(tasks) != 1:
            raise SystemExit("capture-figure requires exactly one task")
    _validate_tasks(tasks)
    preflight = _resource_preflight(args.T, args.workers)
    _assert_engine_blessed()
    with _launcher_lock():
        if args.cmd == "smoke":
            _smoke(tasks[0], seed=args.seed, T=args.T, resource_preflight=preflight)
        elif args.cmd == "capture-figure":
            if args.workers != 1:
                raise SystemExit("capture-figure is single-process only")
            _capture_figure(
                tasks[0], seed=args.seed, T=args.T,
                baseline_json=args.baseline_json,
                resource_preflight=preflight,
            )
        else:
            _execute(tasks, seed=args.seed, T=args.T, workers=args.workers,
                     tag=("baseline_screen" if args.cmd == "baseline-screen" else args.tag),
                     resource_preflight=preflight)


if __name__ == "__main__":
    main()
