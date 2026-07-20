#!/usr/bin/env python3
"""Cheap-first runner for current-based Z/M plus the recurrent-E divisive pool.

No simulation runs on import. Every command requires ``--confirm-run``. The runner owns only
``results/topic4_sef_hfo/mz_divisive_lifecycle`` and never edits the parallel conductance worktree.
"""
import os

# Must be forced before importing NumPy in parent and forked workers. ``setdefault`` is not enough:
# inherited site defaults of 32/64 threads would multiply across simulation workers.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_mpl_cache")

import argparse  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import dataclasses  # noqa: E402
import datetime as dt_datetime  # noqa: E402
import fcntl  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pathlib  # noqa: E402
import resource  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_dynamic_qi as M4  # noqa: E402
import run_m4_phaseplane as PP  # noqa: E402
import run_sef_hfo_snn_cm_spontaneous_readout as CM  # noqa: E402
import run_topic4_mz_slowvars as MZR  # noqa: E402
from kick_probe import _flatten_by_source, simulate_kick  # noqa: E402
from mz_divisive_pool import MZDivisivePoolConfig, MZDivisivePoolSlowVars  # noqa: E402
from src.topic4_mz_divisive_lifecycle import (  # noqa: E402
    LifecycleThresholds,
    analyze_lifecycle,
    safe_worker_count,
)


DEFAULT_CONFIG = os.path.join(ROOT, "config", "topic4_mz_divisive_lifecycle.yaml")
_GLOBAL = {}
_GUARDED_ENGINE = (
    "kick_probe.py",
    "params.py",
    "model.py",
    "connectivity.py",
    "connectivity_rot.py",
    "lfp.py",
)
_PROVENANCE_SOURCES = {
    "runner": os.path.join(ROOT, "scripts", "run_topic4_mz_divisive_lifecycle.py"),
    "classifier": os.path.join(ROOT, "src", "topic4_mz_divisive_lifecycle.py"),
    "composite": os.path.join(ROOT, "src", "snn_engine", "mz_divisive_pool.py"),
    "mz_slow_vars": os.path.join(ROOT, "src", "snn_engine", "mz_slow_vars.py"),
    "slow_field": os.path.join(ROOT, "src", "snn_engine", "slow_field.py"),
    "substrate_builder": os.path.join(ROOT, "scripts", "run_m4_phaseplane.py"),
}


def _load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return cfg


def _git_sha():
    proc = subprocess.run(
        ["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip() or None


def _git_status_short():
    proc = subprocess.run(
        ["git", "-C", ROOT, "status", "--short"], capture_output=True, text=True, check=False
    )
    return proc.stdout.splitlines()


def _file_sha(path):
    try:
        return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:12]
    except OSError:
        return None


def _provenance(cfg, phase, extra=None):
    engine_dir = os.path.join(ROOT, "src", "snn_engine")
    out = dict(
        phase=phase,
        git_sha=_git_sha(),
        argv=sys.argv,
        hostname=socket.gethostname(),
        pid=os.getpid(),
        utc=dt_datetime.datetime.now(dt_datetime.timezone.utc).isoformat(),
        config_path=os.path.relpath(args_config_path(cfg), ROOT),
        config_sha256=_file_sha(args_config_path(cfg)),
        engine_shas={name: _file_sha(os.path.join(engine_dir, name)) for name in _GUARDED_ENGINE},
        source_shas={name: _file_sha(path) for name, path in _PROVENANCE_SOURCES.items()},
        git_status_short=_git_status_short(),
        blas_threads={
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    )
    if extra:
        out.update(extra)
    return out


def args_config_path(cfg):
    return cfg.get("_config_path", DEFAULT_CONFIG)


def _atomic_json(payload, path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_csv(rows, path):
    if not rows:
        return
    scalar_keys = []
    for key in rows[0]:
        if key not in {"cfg", "thresholds"} and not isinstance(rows[0][key], (dict, list, tuple)):
            scalar_keys.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_id(phase):
    stamp = dt_datetime.datetime.now(dt_datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    token = hashlib.sha1(" ".join(sys.argv).encode()).hexdigest()[:10]
    sha = (_git_sha() or "nogit")[:7]
    return f"{stamp}_{sha}_{token}_{phase}"


@contextlib.contextmanager
def _launcher(cfg, phase):
    root = os.path.join(ROOT, cfg["result_root"])
    os.makedirs(root, exist_ok=True)
    lock_path = os.path.join(root, ".launcher.lock")
    lock = open(lock_path, "a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock.seek(0)
            owner = lock.read().strip()
            raise RuntimeError(f"another lifecycle launcher holds {lock_path}: {owner}") from exc
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(), host=socket.gethostname(), phase=phase, argv=sys.argv)))
        lock.flush()
        run_dir = os.path.join(root, "runs", _run_id(phase))
        os.makedirs(run_dir, exist_ok=False)
        manifest = dict(status="running", provenance=_provenance(cfg, phase), run_dir=run_dir)
        _atomic_json(manifest, os.path.join(run_dir, "manifest.json"))
        try:
            yield run_dir
        except BaseException as exc:
            manifest.update(status="failed", error=repr(exc), finished_utc=dt_datetime.datetime.now(
                dt_datetime.timezone.utc
            ).isoformat())
            _atomic_json(manifest, os.path.join(run_dir, "manifest.json"))
            raise
        else:
            manifest.update(status="complete", finished_utc=dt_datetime.datetime.now(
                dt_datetime.timezone.utc
            ).isoformat())
            _atomic_json(manifest, os.path.join(run_dir, "manifest.json"))
    finally:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        finally:
            lock.close()


def _meminfo_gib():
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            key, value = line.split(":", 1)
            vals[key] = float(value.strip().split()[0]) / 1024.0 / 1024.0
    return dict(
        mem_total_gib=vals["MemTotal"],
        mem_available_gib=vals["MemAvailable"],
        swap_total_gib=vals.get("SwapTotal", 0.0),
        swap_free_gib=vals.get("SwapFree", 0.0),
        swap_used_gib=vals.get("SwapTotal", 0.0) - vals.get("SwapFree", 0.0),
    )


def _resource_gate(cfg, requested, n_cells, peak_worker_gib=None):
    rcfg = cfg["resources"]
    mem = _meminfo_gib()
    peak = float(peak_worker_gib or rcfg["unmeasured_worker_gib"])
    workers = safe_worker_count(
        int(requested),
        int(n_cells),
        mem["mem_available_gib"],
        peak,
        reserve_gib=float(rcfg["reserve_gib"]),
        safety_factor=float(rcfg["rss_safety_factor"]),
        hard_cap=min(int(rcfg["hard_worker_cap"]), int(rcfg["initial_worker_cap"])),
        cpu_count=os.cpu_count(),
        cpu_reserve=int(rcfg["cpu_reserve"]),
    )
    audit = dict(requested=int(requested), selected=workers, assumed_peak_worker_gib=peak, **mem)
    if workers < 1:
        raise RuntimeError(f"resource gate refused launch: {audit}")
    return workers, audit


def _prepare_flat_cache(S):
    """Build the immutable recurrent edge cache in the parent before fork/COW."""
    net, NE, NI = S["net"], S["NE"], S["NI"]
    M = net["max_delay_steps"] + 1
    if "ampa_flat" not in net:
        bins = [d for d in range(M) if net["ampa_by_delay"][d].nnz > 0]
        net["ampa_flat"] = _flatten_by_source(net["ampa_by_delay"], bins, NE)
    if "gaba_flat" not in net:
        bins = [d for d in range(M) if net["gaba_by_delay"][d].nnz > 0]
        net["gaba_flat"] = _flatten_by_source(net["gaba_by_delay"], bins, NI)


def _core_mask(S):
    return MZR.build_core_masks(S)


def _slow_from_spec(S, spec):
    cfg = MZDivisivePoolConfig(**spec["cfg"])
    return MZDivisivePoolSlowVars(
        S["N"],
        18.0,
        S["posE"],
        S["posI"],
        S["L"],
        cfg=cfg,
        NE=S["NE"],
        core_mask_E=_core_mask(S),
    )


def _downsample(a, target=2500):
    a = np.asarray(a, np.float32)
    if a.size <= target:
        return a
    return a[:: max(1, int(np.ceil(a.size / target)))]


def _run_cell(spec):
    S = _GLOBAL["S"]
    T_ms = float(spec["T_ms"])
    p = dataclasses.replace(S["p"], T=T_ms)
    slow = _slow_from_spec(S, spec)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    t0 = time.time()
    res = simulate_kick(
        p,
        S["net"],
        0.0,
        slow=slow,
        kick_center=list(S["src_xy"]),
        r_kick=PP.R_KICK,
        t_kick=1e9,
        V_th_per_neuron=S["vth"],
        early_stop_runaway=bool(spec.get("early_stop", True)),
    )
    rate = np.asarray(res["rate_E"], float)
    rate_s = M4._smooth(rate, PP.DT)
    runaway_ms = M4._first_sustained(rate_s, PP.DT)
    if runaway_ms is None:
        runaway_ms = res.get("runaway_early_stop_ms")
    life = analyze_lifecycle(
        rate,
        PP.DT,
        baseline_rate_hz=float(_GLOBAL["baseline_rate_hz"]),
        runaway_ms=runaway_ms,
        thresholds=_GLOBAL["lifecycle_thresholds"],
    )
    af, af_bin_ms = CM.active_fraction(res["E_spk_bool"], PP.DT, CM.BIN_MS)
    roll_n = max(1, int(round(1000.0 / PP.DT)))
    if rate.size >= roll_n:
        csum = np.r_[0.0, np.cumsum(rate, dtype=float)]
        rolling_1s = (csum[roll_n:] - csum[:-roll_n]) / float(roll_n)
        rolling_t_s = (np.arange(rolling_1s.size) + roll_n) * PP.DT * 1e-3
        tail_mask = rolling_t_s >= max(rolling_t_s[-1] - 3.0, rolling_t_s[0])
        rolling_slope = (float(np.polyfit(rolling_t_s[tail_mask], rolling_1s[tail_mask], 1)[0])
                         if int(tail_mask.sum()) >= 2 else 0.0)
        rolling_max = float(rolling_1s.max())
        rolling_final = float(rolling_1s[-1])
    else:
        rolling_max = rolling_final = float(rate.mean())
        rolling_slope = 0.0
    sg = np.asarray(slow.trace_SG, float)
    ag = np.asarray(slow.trace_AG, float)
    tg = np.asarray(slow.trace_TG, float)
    utg = np.asarray(slow.trace_UTG, float)
    row = dict(
        label=spec["label"],
        arm=spec["arm"],
        seed=int(S["seed"]),
        T_ms=T_ms,
        raw_rate_n=int(rate.size),
        raw_rate_dt_ms=float(PP.DT),
        wall_s=round(time.time() - t0, 2),
        worker_peak_rss_gib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0, 3),
        max_rate_hz=round(float(rate_s.max()), 4),
        mean_tail_rate_hz=round(float(rate_s[-max(1, int(round(1000.0 / PP.DT))):].mean()), 4),
        rolling_1s_rate_max_hz=round(rolling_max, 4),
        rolling_1s_rate_final_hz=round(rolling_final, 4),
        rolling_1s_tail_slope_hz_per_s=round(rolling_slope, 4),
        z_min=round(float(np.min(slow.z[: S["NE"]])), 6),
        z_mean_final=round(float(np.mean(slow.z[: S["NE"]])), 6),
        m_mean_final=round(float(np.mean(slow.m[: S["NE"]])), 6),
        adap_current_max=round(float(max(slow.trace_adap_current, default=0.0)), 6),
        SG_max=round(float(sg.max()), 6) if sg.size else 0.0,
        SG_final=round(float(sg[-1]), 6) if sg.size else 0.0,
        SG_occupancy_gt_0p1=round(float(np.mean(sg > 0.1)), 6) if sg.size else 0.0,
        AG_max=round(float(ag.max()), 6) if ag.size else 0.0,
        TG_max=round(float(tg.max()), 6) if tg.size else 0.0,
        TG_final=round(float(tg[-1]), 6) if tg.size else 0.0,
        TG_occupancy_gt_0p01=round(float(np.mean(tg > 0.01)), 6) if tg.size else 0.0,
        UTG_max=round(float(utg.max()), 6) if utg.size else 0.0,
        rEfast_max=round(float(max(slow.trace_rEfast_max, default=0.0)), 6),
        af_peak=round(float(np.max(af)), 6),
        af_tail=round(float(np.mean(af[-max(1, int(round(1000.0 / af_bin_ms))):])), 6),
        cfg=spec["cfg"],
        **life,
    )
    traces = dict(
        rate=_downsample(rate),
        af=_downsample(af),
        z_mean=_downsample(slow.trace_z_mean),
        z_min=_downsample(slow.trace_z_min),
        m_mean=_downsample(slow.trace_m_mean),
        adap=_downsample(slow.trace_adap_current),
        SG=_downsample(sg),
        AG=_downsample(ag),
        muG=_downsample(slow.trace_muG),
        TG=_downsample(tg),
        UTG=_downsample(utg),
    )
    del res
    return row, traces


def _run_specs(specs, workers):
    if workers == 1:
        return [_run_cell(spec) for spec in specs]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers, maxtasksperchild=1) as pool:
        return pool.map(_run_cell, specs, chunksize=1)


def _save_outputs(cfg, run_dir, phase, rows, traces, resource_audit, extra=None):
    counts = {}
    for row in rows:
        counts[row["phenotype"]] = counts.get(row["phenotype"], 0) + 1
    summary = dict(
        experiment="current-based MZ + dynamic recurrent-E divisive pool",
        phase=phase,
        phenotype_counts=counts,
        rows=rows,
        resource_audit=resource_audit,
        provenance=_provenance(cfg, phase),
        trace_contract=dict(
            storage="per-array stride downsample to at most 2500 samples",
            raw_rate_dt_ms=float(PP.DT),
            effective_dt_ms_by_label={
                row["label"]: float(row["T_ms"]) / max(1, len(traces[row["label"]]["rate"]))
                for row in rows
            },
            raw_rate_n_by_label={row["label"]: int(row["raw_rate_n"]) for row in rows},
        ),
    )
    if extra:
        summary.update(extra)
    _atomic_json(summary, os.path.join(run_dir, "summary.json"))
    _write_csv(rows, os.path.join(run_dir, "per_run.csv"))
    payload = {}
    for label, trace in traces.items():
        safe = label.replace(".", "p").replace("-", "_")
        for name, arr in trace.items():
            payload[f"{safe}__{name}"] = np.asarray(arr, np.float32)
    np.savez_compressed(os.path.join(run_dir, "traces_downsampled.npz"), **payload)
    root = os.path.join(ROOT, cfg["result_root"])
    _atomic_json(
        dict(run_dir=os.path.relpath(run_dir, ROOT), phase=phase, summary=os.path.relpath(
            os.path.join(run_dir, "summary.json"), ROOT
        )),
        os.path.join(root, f"latest_{phase}.json"),
    )
    return summary


def _baseline_rate(cfg, seed):
    cal_path = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slowvars", "calibration.json")
    cal = json.load(open(cal_path))
    return float(cal["per_seed"][str(seed)]["baseline"]["baseline_rate"])


def _lifecycle_thresholds(cfg):
    screen = cfg["screen"]
    return LifecycleThresholds(
        recruited_hz=float(screen["recruited_hz"]),
        recruited_sustain_ms=float(screen["recruited_sustain_ms"]),
        min_recruited_ms=float(screen["min_recruited_ms"]),
        recovery_ms=float(screen["recovery_ms"]),
        recovery_margin_hz=float(screen.get("recovery_margin_hz", 5.0)),
        envelope_ms=float(screen.get("envelope_ms", 50.0)),
        merge_gap_ms=float(screen.get("merge_gap_ms", 100.0)),
        burst_min_peaks=int(screen["burst_min_peaks"]),
        burst_min_modulation=float(screen["burst_min_modulation"]),
        burst_band_hz=tuple(float(x) for x in screen["burst_band_hz"]),
        peak_min_separation_ms=float(screen.get("peak_min_separation_ms", 50.0)),
    )


def _common_setup(cfg, seed):
    configured_dt = float(cfg["substrate"]["dt_ms"])
    if not np.isclose(configured_dt, float(PP.DT), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"config dt_ms={configured_dt} does not match engine PP.DT={PP.DT}")
    print(f"[build] E1146 substrate seed={seed}", flush=True)
    S = PP.build_substrate(seed)
    _prepare_flat_cache(S)
    _GLOBAL.clear()
    _GLOBAL.update(
        S=S,
        baseline_rate_hz=_baseline_rate(cfg, seed),
        lifecycle_thresholds=_lifecycle_thresholds(cfg),
    )
    return S


def _base_cfg(
    cfg,
    *,
    use_z,
    use_m=False,
    eta_m=0.0,
    use_SG=True,
    alpha_G=0.0,
    p_pool=3.0,
    use_TG=False,
    alpha_TG=0.0,
    tau_TG=None,
    tau_S=None,
):
    z = cfg["z_anchor"]
    pool = cfg["pool"]
    slow_gate = cfg.get("slow_gated_divisor", {})
    return dict(
        use_z=bool(use_z),
        use_m=bool(use_m),
        I_th_EI=float(z["I_th_EI"]),
        tau_z=float(z["tau_z_ms"]),
        tau_adp=float(cfg["adaptation"]["tau_adp_ms"]),
        eta_m=float(eta_m),
        use_SG=bool(use_SG),
        alpha_G=float(alpha_G),
        r0_psi=float(pool["r0_psi"]),
        r50_psi=float(pool["r50_psi"]),
        n_psi=float(pool["n_psi"]),
        p_pool=float(p_pool),
        tau_mu=float(pool["tau_mu_ms"]),
        tau_S=float(tau_S if tau_S is not None else pool["tau_S_ms"]),
        S_max=float(pool["S_max"]),
        use_TG=bool(use_TG),
        alpha_TG=float(alpha_TG),
        AG0_TG=float(slow_gate.get("AG0", 0.15)),
        AG50_TG=float(slow_gate.get("AG50", 0.10)),
        n_TG=float(slow_gate.get("exponent", 4.0)),
        tau_TG=float(tau_TG if tau_TG is not None else slow_gate.get("tau_TG_ms", [750.0])[0]),
        TG_max=float(slow_gate.get("TG_max", 1.0)),
    )


def cmd_observer(cli, cfg):
    phase = "observer"
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or cfg["screen"]["observer_T_ms"])
        specs = []
        for p in (3.0, 1.0):
            specs.append(dict(label=f"slowoff_p{int(p)}", arm="slowoff_observer", T_ms=T, early_stop=True,
                              cfg=_base_cfg(cfg, use_z=False, use_SG=True, alpha_G=0.0, p_pool=p)))
            specs.append(dict(label=f"zanchor_p{int(p)}", arm="z_observer", T_ms=T, early_stop=True,
                              cfg=_base_cfg(cfg, use_z=True, use_SG=True, alpha_G=0.0, p_pool=p)))
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib)
        print(f"[observer] cells={len(specs)} workers={workers} resource={audit}", flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        sensor = {}
        for p in (3, 1):
            slow = next(r for r in rows if r["label"] == f"slowoff_p{p}")
            zrun = next(r for r in rows if r["label"] == f"zanchor_p{p}")
            sensor[f"p{p}"] = dict(
                recruited_driven=bool(zrun["SG_max"] >= 0.02),
                slowoff_not_tonic=bool(slow["SG_occupancy_gt_0p1"] <= 0.25),
                slowoff_SG_max=slow["SG_max"],
                zanchor_SG_max=zrun["SG_max"],
            )
        summary = _save_outputs(cfg, run_dir, phase, rows, traces, audit, dict(sensor_gate=sensor))
        print(f"[observer] done -> {run_dir}/summary.json sensor_gate={sensor}", flush=True)
        return summary


def cmd_containment(cli, cfg):
    phase = "containment"
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or cfg["screen"]["containment_T_ms"])
        specs = [dict(
            label="z_no_pool", arm="z_no_pool", T_ms=T, early_stop=True,
            cfg=_base_cfg(cfg, use_z=True, use_SG=False, alpha_G=0.0, p_pool=3.0),
        )]
        for arm, acfg in cfg["pool"]["arms"].items():
            p = float(acfg["p_pool"])
            for alpha in acfg["alpha_G"]:
                specs.append(dict(
                    label=f"{arm}_p{int(p)}_aG{float(alpha):g}", arm=arm, T_ms=T, early_stop=True,
                    cfg=_base_cfg(cfg, use_z=True, use_SG=True, alpha_G=float(alpha), p_pool=p),
                ))
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib)
        print(f"[containment] cells={len(specs)} workers={workers} resource={audit}", flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        candidates = [
            r["label"] for r in rows
            if r["phenotype"] in {"bounded_bursting", "bounded_plateau"}
            and r["recruited_duration_ms"] >= float(cfg["screen"]["min_recruited_ms"])
        ]
        summary = _save_outputs(
            cfg, run_dir, phase, rows, traces, audit,
            dict(containment_candidates=candidates, stop_if_empty=True),
        )
        for r in rows:
            print(f"  {r['label']}: {r['phenotype']} max={r['max_rate_hz']:.1f}Hz "
                  f"SG={r['SG_max']:.3f} onset={r['onset_ms']}", flush=True)
        print(f"[containment] candidates={candidates} -> {run_dir}/summary.json", flush=True)
        return summary


def cmd_boundary(cli, cfg):
    """Adaptive one-dimensional alpha refinement after the registered 0-to-8 phenotype jump."""
    phase = "boundary"
    source_path, source = _resolve_summary(cli.source_summary, cfg, "containment")
    if source.get("containment_candidates"):
        raise RuntimeError("containment already has a registered candidate; boundary refinement is unnecessary")
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or cfg["screen"]["containment_T_ms"])
        specs = []
        for arm, acfg in cfg["pool"]["boundary_refinement"].items():
            p = float(acfg["p_pool"])
            for alpha in acfg["alpha_G"]:
                specs.append(dict(
                    label=f"{arm}_p{int(p)}_aG{float(alpha):g}", arm=f"{arm}_boundary",
                    T_ms=T, early_stop=True,
                    cfg=_base_cfg(cfg, use_z=True, use_SG=True, alpha_G=float(alpha), p_pool=p),
                ))
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib)
        print(f"[boundary] cells={len(specs)} workers={workers} resource={audit}", flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        candidates = [
            r["label"] for r in rows
            if r["phenotype"] in {"bounded_bursting", "bounded_plateau"}
            and r["recruited_duration_ms"] >= float(cfg["screen"]["min_recruited_ms"])
        ]
        summary = _save_outputs(
            cfg, run_dir, phase, rows, traces, audit,
            dict(source_containment_summary=os.path.relpath(source_path, ROOT),
                 containment_candidates=candidates,
                 adaptive_reason="registered alpha=0 runaway versus alpha>=8 IED-like bracket"),
        )
        for r in rows:
            print(f"  {r['label']}: {r['phenotype']} max={r['max_rate_hz']:.1f}Hz "
                  f"tail={r['mean_tail_rate_hz']:.1f}Hz SG={r['SG_max']:.3f}", flush=True)
        print(f"[boundary] candidates={candidates} -> {run_dir}/summary.json", flush=True)
        return summary


def cmd_long_check(cli, cfg):
    """Twenty-second check that the two rising boundary traces are not delayed runaways."""
    phase = "long_check"
    source_path, source = _resolve_summary(cli.source_summary, cfg, "boundary")
    labels = list(cfg["screen"]["boundary_long_check"])
    source_rows = {row["label"]: row for row in source["rows"]}
    missing = [label for label in labels if label not in source_rows]
    if missing:
        raise RuntimeError(f"long-check source is missing registered cells: {missing}")
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or cfg["screen"]["confirm_T_ms"])
        specs = [dict(label=f"{label}_long", arm="boundary_long_check", T_ms=T, early_stop=True,
                      cfg=dict(source_rows[label]["cfg"])) for label in labels]
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib or 12.0)
        print(f"[long-check] cells={len(specs)} workers={workers} T={T} resource={audit}", flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        settled = [
            r["label"] for r in rows
            if r["runaway_ms"] is None
            and r["rolling_1s_rate_final_hz"] >= 10.0
            and abs(r["rolling_1s_tail_slope_hz_per_s"]) <= 1.0
        ]
        summary = _save_outputs(
            cfg, run_dir, phase, rows, traces, audit,
            dict(source_boundary_summary=os.path.relpath(source_path, ROOT), settled_candidates=settled,
                 settled_rule="no runaway; final rolling-1s rate >=10 Hz; |last-3s slope| <=1 Hz/s"),
        )
        for r in rows:
            print(f"  {r['label']}: {r['phenotype']} runaway={r['runaway_ms']} "
                  f"roll1s_final={r['rolling_1s_rate_final_hz']:.1f}Hz "
                  f"slope={r['rolling_1s_tail_slope_hz_per_s']:.2f}Hz/s", flush=True)
        print(f"[long-check] settled={settled} -> {run_dir}/summary.json", flush=True)
        return summary


def cmd_timescale(cli, cfg):
    """Final registered tau_S sensitivity at the two alpha-boundary coordinates."""
    phase = "timescale"
    source_path, source = _resolve_summary(cli.source_summary, cfg, "boundary")
    labels = list(cfg["screen"]["boundary_long_check"])
    source_rows = {row["label"]: row for row in source["rows"]}
    missing = [label for label in labels if label not in source_rows]
    if missing:
        raise RuntimeError(f"timescale source is missing registered cells: {missing}")
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or cfg["screen"]["termination_T_ms"])
        specs = []
        for label in labels:
            for tau_s in cfg["pool"]["tau_S_sensitivity_ms"]:
                cell_cfg = dict(source_rows[label]["cfg"])
                cell_cfg["tau_S"] = float(tau_s)
                specs.append(dict(
                    label=f"{label}_tauS{int(float(tau_s))}", arm="tau_S_sensitivity",
                    T_ms=T, early_stop=True, cfg=cell_cfg,
                ))
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib or 10.0)
        print(f"[timescale] cells={len(specs)} workers={workers} T={T} resource={audit}", flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        settled = [
            r["label"] for r in rows
            if r["runaway_ms"] is None
            and r["rolling_1s_rate_final_hz"] >= 10.0
            and abs(r["rolling_1s_tail_slope_hz_per_s"]) <= 1.0
        ]
        summary = _save_outputs(
            cfg, run_dir, phase, rows, traces, audit,
            dict(source_boundary_summary=os.path.relpath(source_path, ROOT), settled_candidates=settled,
                 final_rescue_test=True),
        )
        for r in rows:
            print(f"  {r['label']}: {r['phenotype']} runaway={r['runaway_ms']} "
                  f"roll1s={r['rolling_1s_rate_final_hz']:.1f}Hz "
                  f"slope={r['rolling_1s_tail_slope_hz_per_s']:.2f}Hz/s", flush=True)
        print(f"[timescale] settled={settled} -> {run_dir}/summary.json", flush=True)
        return summary


def cmd_slow_gate(cli, cfg):
    """Locked five-cell v2 screen: high-state-gated slow recurrent-E divisor."""
    phase = "slow_gate"
    gate = cfg["slow_gated_divisor"]
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or gate["T_ms"])
        common = dict(
            use_z=True,
            use_m=False,
            use_SG=True,
            alpha_G=float(gate["alpha_fast"]),
            p_pool=float(gate["p_pool"]),
            tau_S=float(gate["tau_S_ms"]),
            use_TG=True,
        )
        # alpha_TG=0 is a literal neutral implementation/parity anchor. The tau value is irrelevant.
        anchor_tau = float(gate["tau_TG_ms"][0])
        specs = [dict(
            label="slow_gate_anchor_aT0",
            arm="parity_anchor",
            T_ms=T,
            early_stop=True,
            cfg=_base_cfg(cfg, alpha_TG=0.0, tau_TG=anchor_tau, **common),
        )]
        for alpha_t in gate["alpha_TG"]:
            for tau_t in gate["tau_TG_ms"]:
                specs.append(dict(
                    label=f"slow_gate_aT{float(alpha_t):g}_tau{int(float(tau_t))}",
                    arm="slow_gated_divisor",
                    T_ms=T,
                    early_stop=True,
                    cfg=_base_cfg(
                        cfg,
                        alpha_TG=float(alpha_t),
                        tau_TG=float(tau_t),
                        **common,
                    ),
                ))
        if len(specs) != 5:
            raise RuntimeError(f"locked slow-gate screen must contain exactly 5 cells, got {len(specs)}")
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib or 12.0)
        print(f"[slow-gate] cells={len(specs)} workers={workers} T={T} resource={audit}", flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        anchor = rows[0]
        anchor_valid = bool(
            anchor["phenotype"] == "runaway"
            and anchor["runaway_ms"] is not None
            and 12000.0 <= float(anchor["runaway_ms"]) <= 17000.0
        )
        min_recruited = float(cfg["screen"]["min_recruited_ms"])
        candidates = [
            r["label"]
            for r in rows[1:]
            if r["phenotype"] in {"terminate_bursting", "terminate_plateau"}
            and r["returned_to_baseline"]
            and r["recruited_duration_ms"] >= min_recruited
            and r["runaway_ms"] is None
        ]
        screen_valid = anchor_valid
        summary = _save_outputs(
            cfg,
            run_dir,
            phase,
            rows,
            traces,
            audit,
            dict(
                locked_five_cell_screen=True,
                anchor_valid=anchor_valid,
                anchor_expected_runaway_window_ms=[12000.0, 17000.0],
                sensor_lock=dict(
                    slowoff_AG_max=0.111708,
                    AG0=float(gate["AG0"]),
                    slowoff_drive_is_literal_zero=bool(0.111708 < float(gate["AG0"])),
                ),
                screen_valid=screen_valid,
                lifecycle_candidates=candidates if screen_valid else [],
                stop_if_empty=True,
            ),
        )
        for r in rows:
            print(
                f"  {r['label']}: {r['phenotype']} runaway={r['runaway_ms']} "
                f"recruited={r['recruited_duration_ms']:.0f}ms returned={r['returned_to_baseline']} "
                f"TG={r['TG_max']:.3f}",
                flush=True,
            )
        print(
            f"[slow-gate] anchor_valid={anchor_valid} candidates={candidates if screen_valid else []} "
            f"-> {run_dir}/summary.json",
            flush=True,
        )
        return summary


def cmd_slow_gate_m(cli, cfg):
    """Locked v3 screen: apply the pre-existing M ladder to the finite-window v2 high state."""
    phase = "slow_gate_m"
    source_path, source = _resolve_summary(cli.source_summary, cfg, "slow_gate")
    if not source.get("anchor_valid"):
        raise RuntimeError("slow-gate source failed its parity anchor; M-exit screen is invalid")
    source_label = "slow_gate_aT4_tau750"
    matches = [r for r in source["rows"] if r["label"] == source_label]
    if len(matches) != 1:
        raise RuntimeError(f"slow-gate source must contain exactly one {source_label!r} row")
    bounded = matches[0]
    if bounded["phenotype"] != "bounded_bursting" or bounded["runaway_ms"] is not None:
        raise RuntimeError(f"registered v2 source is not bounded bursting: {bounded['phenotype']}")
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or 25000.0)
        eta_ladder = [float(x) for x in cfg["adaptation"]["eta_m"]]
        if (
            len(eta_ladder) != 6
            or eta_ladder[0] != 0.0
            or any(b <= a for a, b in zip(eta_ladder, eta_ladder[1:]))
        ):
            raise RuntimeError(
                "locked M-exit eta ladder must contain six unique strictly increasing values starting at 0"
            )
        specs = []
        for eta in eta_ladder:
            cell_cfg = dict(bounded["cfg"])
            cell_cfg.update(
                use_m=float(eta) > 0.0,
                eta_m=float(eta),
                tau_adp=float(cfg["adaptation"]["tau_adp_ms"]),
            )
            specs.append(dict(
                label=f"slow_gate_m_eta{float(eta):.5f}",
                arm="m_exit_ladder" if float(eta) > 0.0 else "m_off_long_anchor",
                T_ms=T,
                early_stop=False,
                cfg=cell_cfg,
            ))
        if len(specs) != 6:
            raise RuntimeError(f"locked M-exit screen must contain exactly 6 cells, got {len(specs)}")
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib or 15.0)
        print(f"[slow-gate-m] cells={len(specs)} workers={workers} T={T} resource={audit}", flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        anchor = rows[0]
        anchor_valid = bool(
            anchor["phenotype"] == "bounded_bursting"
            and anchor["runaway_ms"] is None
            and anchor["rolling_1s_rate_final_hz"] >= 10.0
            and abs(anchor["rolling_1s_tail_slope_hz_per_s"]) <= 1.0
        )
        min_recruited = float(cfg["screen"]["min_recruited_ms"])
        legacy_screen_hits = [
            r["label"]
            for r in rows[1:]
            if r["phenotype"] in {"terminate_bursting", "terminate_plateau"}
            and r["returned_to_baseline"]
            and r["recruited_duration_ms"] >= min_recruited
            and r["runaway_ms"] is None
        ]
        summary = _save_outputs(
            cfg,
            run_dir,
            phase,
            rows,
            traces,
            audit,
            dict(
                source_slow_gate_summary=os.path.relpath(source_path, ROOT),
                source_bounded_cell=source_label,
                legacy_m_off_rate_gate=anchor_valid,
                legacy_screen_hits=legacy_screen_hits if anchor_valid else [],
                lifecycle_candidates=[],
                strict_posthoc_required=True,
                interpretation="screen_descriptor_only_pending_paired_slowoff_posthoc",
                stop_if_empty=True,
            ),
        )
        for r in rows:
            print(
                f"  {r['label']}: {r['phenotype']} runaway={r['runaway_ms']} "
                f"onset={r['onset_ms']} offset={r['offset_ms']} returned={r['returned_to_baseline']} "
                f"m={r['m_mean_final']:.3f} slope={r['rolling_1s_tail_slope_hz_per_s']:.2f}",
                flush=True,
            )
        print(
            f"[slow-gate-m] anchor_valid={anchor_valid} "
            f"legacy_screen_hits={legacy_screen_hits if anchor_valid else []} "
            f"strict_posthoc=required -> {run_dir}/summary.json",
            flush=True,
        )
        return summary


def _resolve_summary(path_or_latest, cfg, phase):
    if path_or_latest:
        path = path_or_latest if os.path.isabs(path_or_latest) else os.path.join(ROOT, path_or_latest)
    else:
        latest = json.load(open(os.path.join(ROOT, cfg["result_root"], f"latest_{phase}.json")))
        path = os.path.join(ROOT, latest["summary"])
    return path, json.load(open(path))


def cmd_termination(cli, cfg):
    phase = "termination"
    source_path, source = _resolve_summary(cli.source_summary, cfg, "containment")
    candidates = [r for r in source["rows"] if r["label"] in source.get("containment_candidates", [])]
    if cli.candidate:
        candidates = [r for r in candidates if r["label"] == cli.candidate]
    elif candidates:
        candidates = [max(
            candidates,
            key=lambda r: (r["phenotype"] == "bounded_bursting", r["recruited_duration_ms"]),
        )]
    if not candidates:
        raise RuntimeError("no registered containment candidate; stop rule forbids adding M")
    cand = candidates[0]
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or cfg["screen"]["termination_T_ms"])
        specs = []
        for eta in cfg["adaptation"]["eta_m"]:
            cell_cfg = dict(cand["cfg"])
            cell_cfg.update(use_m=float(eta) > 0.0, eta_m=float(eta),
                            tau_adp=float(cfg["adaptation"]["tau_adp_ms"]))
            specs.append(dict(
                label=f"{cand['label']}_eta{float(eta):.5f}", arm="termination_m_ladder",
                T_ms=T, early_stop=False, cfg=cell_cfg,
            ))
        workers, audit = _resource_gate(cfg, cli.workers, len(specs), cli.worker_gib or 8.0)
        print(f"[termination] source={cand['label']} cells={len(specs)} workers={workers} resource={audit}",
              flush=True)
        results = _run_specs(specs, workers)
        rows = [x[0] for x in results]
        traces = {row["label"]: trace for row, trace in results}
        term = [r["label"] for r in rows if r["phenotype"] in {"terminate_bursting", "terminate_plateau"}
                and r["recruited_duration_ms"] >= float(cfg["screen"]["min_recruited_ms"])]
        summary = _save_outputs(
            cfg, run_dir, phase, rows, traces, audit,
            dict(source_containment_summary=os.path.relpath(source_path, ROOT), source_candidate=cand["label"],
                 termination_candidates=term),
        )
        for r in rows:
            print(f"  {r['label']}: {r['phenotype']} onset={r['onset_ms']} offset={r['offset_ms']} "
                  f"tail={r['mean_tail_rate_hz']:.2f}Hz", flush=True)
        print(f"[termination] candidates={term} -> {run_dir}/summary.json", flush=True)
        return summary


def cmd_rss_audit(cli, cfg):
    phase = "rss_audit"
    with _launcher(cfg, phase) as run_dir:
        _common_setup(cfg, cli.seed)
        T = float(cli.T or 2000.0)
        spec = dict(
            label="rss_z_p3_aG16", arm="rss", T_ms=T, early_stop=True,
            cfg=_base_cfg(cfg, use_z=True, use_SG=True, alpha_G=16.0, p_pool=3.0),
        )
        before = _meminfo_gib()
        row, traces = _run_cell(spec)
        after = _meminfo_gib()
        audit = dict(before=before, after=after, measured_peak_rss_gib=row["worker_peak_rss_gib"])
        summary = _save_outputs(cfg, run_dir, phase, [row], {row["label"]: traces}, audit)
        root = os.path.join(ROOT, cfg["result_root"])
        _atomic_json(audit, os.path.join(root, "rss_audit.json"))
        print(f"[rss-audit] peak={row['worker_peak_rss_gib']:.2f}GiB wall={row['wall_s']:.1f}s "
              f"-> {run_dir}/summary.json", flush=True)
        return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("rss-audit", "observer", "containment", "boundary", "long-check", "timescale",
                 "slow-gate", "slow-gate-m", "termination"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true")
        sp.add_argument("--seed", type=int, default=1)
        sp.add_argument("--T", type=float, default=None)
        sp.add_argument("--workers", type=int, default=8)
        sp.add_argument("--worker-gib", type=float, default=None)
        sp.add_argument("--source-summary", default=None)
        sp.add_argument("--candidate", default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    cli = parser.parse_args(argv)
    if not cli.confirm_run:
        parser.error(f"{cli.command} runs simulations; pass --confirm-run")
    cfg = _load_config(cli.config)
    cfg["_config_path"] = os.path.abspath(cli.config)
    command = cli.command.replace("-", "_")
    return globals()[f"cmd_{command}"](cli, cfg)


if __name__ == "__main__":
    main()
