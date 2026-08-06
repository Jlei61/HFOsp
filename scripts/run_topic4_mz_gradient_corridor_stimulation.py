"""MZ gradient-corridor stimulation runner — patient-layout virtual-SEEG site comparison on the Z+M SNN.

*** RUNS SIMULATIONS *** (except `geometry-audit`). Every simulation subcommand is gated by
--confirm-run. Nothing runs on import. Compute lives in src.topic4_mz_gradient_corridor_stimulation;
this runner owns geometry audit, the baseline->window->stim orchestration, the fork Pool, resume,
atomic writes, the resource log, and per-run artifacts.

Subcommands:
  geometry-audit   fingerprint + cohort + bipolar-site audit -> geometry_audit.csv + cohort_manifest.csv
  rss-audit        one full baseline run -> peak RSS + wall time (sets the worker budget)
  pilot            E1146 + zhaochenxi, seed 1: baseline + 3 sites; pre-stim parity + eligibility + RSS
  cohort           full campaign: admitted subjects x seeds x arms (baseline->window->stim), fork Pool

Model/axis contract: see src.topic4_mz_gradient_corridor_stimulation module docstring. Frozen gradient
input is READ-ONLY from the main checkout via --input-root (never written).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

import argparse            # noqa: E402
import csv                 # noqa: E402
import hashlib             # noqa: E402
import json                # noqa: E402
import multiprocessing as mp  # noqa: E402
import resource           # noqa: E402
import subprocess         # noqa: E402
import sys                 # noqa: E402
import time               # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import src.topic4_mz_gradient_corridor_stimulation as G  # noqa: E402

DEFAULT_INPUT_ROOT = "/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/template_gradient_fields"
OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_gradient_corridor_stimulation")
PER_RUN = os.path.join(OUT_DIR, "per_run")

DT = G.SNN["dt"]
STIM_DELTA_MV = 50.0
STIM_RADIUS_MM = 1.5
T_MAX_MS = 20000.0
STIM_ON_FRAC = 0.45
STIM_OFF_FRAC = 0.75
MIN_PRE_STIM_EVENTS = 3
SEEDS = (1, 3, 4)
AXIAL_BINS = 20
TRANSVERSE_BINS = 12
SPATIAL_BIN_MS = 5.0
ARMS_PRIMARY = ("gradient_endpoint_negative", "gradient_endpoint_positive", "gradient_middle")
SITE_KEY = {"gradient_endpoint_negative": "endpoint_negative",
            "gradient_endpoint_positive": "endpoint_positive",
            "gradient_middle": "middle", "gradient_offaxis_control": "offaxis"}


# ============================================================ provenance / io helpers
def _git_sha():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _engine_shas():
    d = {}
    eng = os.path.join(ROOT, "src", "snn_engine")
    for f in ("kick_probe.py", "params.py", "connectivity.py", "connectivity_rot.py",
              "mz_slow_vars.py", "lfp.py"):
        p = os.path.join(eng, f)
        try:
            d[f] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:12]
        except Exception:
            d[f] = None
    return d


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _atomic_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".tmp.{os.getpid()}"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _mem_snapshot():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")[0], line.split()[1]
                if k in ("MemAvailable", "SwapFree", "SwapTotal"):
                    info[k] = int(v) / 1024.0 / 1024.0     # GiB
    except Exception:
        pass
    la = os.getloadavg()[0]
    return dict(memavail_gib=round(info.get("MemAvailable", 0.0), 1),
                swap_used_gib=round(info.get("SwapTotal", 0.0) - info.get("SwapFree", 0.0), 2),
                loadavg=round(la, 1))


def arm_fingerprint(subject_id, seed, arm, montage, n_steps, stim_window_steps, n_target, delta, radius):
    h = hashlib.sha256()
    for x in (subject_id, seed, arm, n_steps, G.MZ_CANDIDATE, G.SNN,
              np.round(montage.src_xy, 6).tolist(), np.round(montage.snk_xy, 6).tolist(),
              stim_window_steps, n_target, delta, radius, G.SCHEMA_VERSION):
        h.update(str(x).encode())
    return h.hexdigest()[:16]


# ============================================================ substrate assembly (per subject)
def _patient_pack(subject_id, input_root, shared):
    """Load gradient record -> montage -> sites -> vth/core -> stim targets, all validated."""
    record = G.load_gradient_record(subject_id, input_root)
    montage = G.build_sheet_montage(record, L=G.SNN["L"], margin=G.SNN["sheet_margin_mm"],
                                    core_quantiles=G.SNN["core_quantiles"])
    sel = G.select_bipolar_sites(montage)
    if not sel["sites"]:
        raise RuntimeError(f"{subject_id}: {sel['reason']}")
    patient = G.build_patient_vth(shared, montage)
    targets = G.build_stim_targets(shared["posE"], sel["sites"], radius_mm=STIM_RADIUS_MM)
    return dict(record_status=record.get("status"), montage=montage, sites=sel["sites"],
                patient=patient, targets=targets, coredist=montage.core_separation_mm)


# ============================================================ single-arm execution (+ save)
def _run_and_save(shared, pack, subject_id, seed, arm, *, stim_on_ms, stim_off_ms, frozen_bar,
                  baseline_total_activity, n_steps, save_arrays=True):
    montage = pack["montage"]
    if arm == "baseline_no_stim":
        target_E, window = None, None
        n_target = 0
    else:
        skey = SITE_KEY[arm]
        if skey not in pack["targets"]["masks"]:
            return dict(arm=arm, status="no_site", subject_id=subject_id, seed=seed)
        target_E = pack["targets"]["masks"][skey]
        window = (int(stim_on_ms / DT), int(stim_off_ms / DT))
        n_target = pack["targets"]["n_target"]
    fp = arm_fingerprint(subject_id, seed, arm, montage, n_steps, window, n_target, STIM_DELTA_MV, STIM_RADIUS_MM)
    jpath = os.path.join(PER_RUN, subject_id, str(seed), f"{arm}.json")
    npath = os.path.join(PER_RUN, subject_id, str(seed), f"{arm}.npz")
    if os.path.isfile(jpath):
        try:
            prev = json.load(open(jpath))
            if prev.get("fingerprint") == fp and prev.get("status") == "ok":
                return dict(arm=arm, status="resumed", subject_id=subject_id, seed=seed,
                            summary=prev.get("summary"))
        except Exception:
            pass
    t0 = time.time()
    res, obs, slow = G.run_arm(shared, pack["patient"], montage, arm=arm, target_E=target_E,
                               stim_window_steps=window, delta_mv=STIM_DELTA_MV, n_steps=n_steps,
                               spatial_bin_ms=SPATIAL_BIN_MS, axial_bins=AXIAL_BINS,
                               transverse_bins=TRANSVERSE_BINS)
    summary = G.summarize_run(res, obs, slow, arm=arm, dt=DT, frozen_bar=frozen_bar,
                              stim_on_ms=stim_on_ms, stim_off_ms=stim_off_ms, t_max_ms=T_MAX_MS,
                              coredist_mm=pack["coredist"], core_r=G.SNN["core_r"], spatial_bin_ms=SPATIAL_BIN_MS,
                              baseline_total_activity=baseline_total_activity)
    wall = time.time() - t0
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    rec = dict(schema=G.SCHEMA_VERSION, fingerprint=fp, status="ok", subject_id=subject_id, seed=seed,
               arm=arm, n_steps=n_steps, dt=DT, t_max_ms=T_MAX_MS, delta_mv=STIM_DELTA_MV,
               stim_radius_mm=STIM_RADIUS_MM, n_target=n_target, coredist_mm=round(pack["coredist"], 3),
               stim_on_ms=stim_on_ms, stim_off_ms=stim_off_ms, frozen_event_bar=frozen_bar,
               wall_s=round(wall, 1), peak_rss_gib=round(peak_rss, 2), summary=summary,
               generated_at=_now())
    _atomic_json(jpath, rec)
    if save_arrays:
        _atomic_npz(npath, **G.arm_arrays(res, obs, slow))
    return dict(arm=arm, status="ok", subject_id=subject_id, seed=seed, summary=summary,
                wall_s=round(wall, 1), peak_rss_gib=round(peak_rss, 2), res=res, obs=obs)


# ============================================================ baseline: window + eligibility
def _baseline_and_window(shared, pack, subject_id, seed, n_steps):
    """Run the no-stim baseline, freeze the event bar, derive the arm-blind stim window, judge
    eligibility (>= MIN_PRE_STIM_EVENTS recoverable interictal events BEFORE stim_on, and a runaway)."""
    montage = pack["montage"]
    res, obs, slow = G.run_arm(shared, pack["patient"], montage, arm="baseline_no_stim", target_E=None,
                               stim_window_steps=None, delta_mv=STIM_DELTA_MV, n_steps=n_steps,
                               spatial_bin_ms=SPATIAL_BIN_MS, axial_bins=AXIAL_BINS,
                               transverse_bins=TRANSVERSE_BINS)
    af = np.asarray(obs.active_frac, float)
    bar = G.frozen_event_bar(af)
    rate = np.asarray(res["rate_E"], float)
    runaway_ms = G.score_runaway(rate, DT)
    if res.get("runaway_early_stop_step") is not None and runaway_ms is None:
        runaway_ms = res["runaway_early_stop_step"] * DT
    events = G.detect_events(af, G.ACTIVE_BIN_MS, event_on_frac=bar)
    base_total_activity = float(af.sum())
    verdict = dict(subject_id=subject_id, seed=seed, baseline_runaway_ms=runaway_ms,
                   n_events_total=len(events), frozen_event_bar=bar,
                   base_total_activity=base_total_activity)
    if runaway_ms is None:
        verdict.update(eligible=False, reason="no_runaway_in_T_max")
        return verdict, res, obs, slow, None
    stim_on = STIM_ON_FRAC * runaway_ms
    stim_off = STIM_OFF_FRAC * runaway_ms
    n_pre = sum(1 for e in events if e["t_off"] <= stim_on and e["returned"])
    # high-firing-from-start guard: mean rate in first 200 ms already runaway-like
    early_rate = float(rate[:int(200 / DT)].mean()) if rate.size else 0.0
    if early_rate >= G.SNN.get("early_high_hz", 60.0):
        verdict.update(eligible=False, reason="high_firing_from_start", stim_on_ms=stim_on, stim_off_ms=stim_off,
                       n_pre_stim_recoverable=n_pre)
        return verdict, res, obs, slow, None
    if n_pre < MIN_PRE_STIM_EVENTS:
        verdict.update(eligible=False, reason=f"only {n_pre} recoverable pre-stim events < {MIN_PRE_STIM_EVENTS}",
                       stim_on_ms=stim_on, stim_off_ms=stim_off, n_pre_stim_recoverable=n_pre)
        return verdict, res, obs, slow, None
    verdict.update(eligible=True, reason=None, stim_on_ms=stim_on, stim_off_ms=stim_off,
                   n_pre_stim_recoverable=n_pre)
    return verdict, res, obs, slow, (stim_on, stim_off, bar, base_total_activity)


# ============================================================ resource-aware worker budget
def _safe_workers(peak_rss_gib, cap=12, reserve_gib=64.0, mult=1.5):
    snap = _mem_snapshot()
    avail = snap["memavail_gib"]
    per = max(0.2, mult * peak_rss_gib)
    n = int((avail - reserve_gib) // per)
    return max(1, min(cap, n)), snap


def _resource_log_row(path, extra):
    snap = _mem_snapshot()
    row = dict(timestamp=_now(), **snap, **extra)
    hdr = not os.path.isfile(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if hdr:
            w.writeheader()
        w.writerow(row)
    return row


# ============================================================ geometry audit
def cmd_geometry_audit(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = [G.audit_subject_geometry(s, args.input_root, tier="primary_candidate") for s in G.PRIMARY_COHORT]
    rows += [G.audit_subject_geometry(s, args.input_root, tier="sensitivity") for s in G.SENSITIVITY_COHORT]
    cols = list(rows[0])
    with open(os.path.join(OUT_DIR, "geometry_audit.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    admitted = [r["subject_id"] for r in rows if r["admitted"] and r["tier"] == "primary_candidate"]
    sens_admitted = [r["subject_id"] for r in rows if r["admitted"] and r["tier"] == "sensitivity"]
    manifest = dict(contract=G.SCHEMA_VERSION, generated_at=_now(), input_root=args.input_root,
                    admitted_primary=admitted, n_admitted_primary=len(admitted),
                    admitted_sensitivity=sens_admitted,
                    excluded={r["subject_id"]: r["exclusion_reason"] for r in rows if not r["admitted"]},
                    cohort_go=bool(len(admitted) >= 4), git_sha=_git_sha(), engine_shas=_engine_shas())
    _atomic_json(os.path.join(OUT_DIR, "cohort_manifest.json"), manifest)
    with open(os.path.join(OUT_DIR, "cohort_manifest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "tier", "admitted", "core_separation_mm", "middle_pair", "exclusion_reason"])
        for r in rows:
            w.writerow([r["subject_id"], r["tier"], r["admitted"], r["core_separation_mm"],
                        r["middle_pair"], r["exclusion_reason"]])
    print(json.dumps(manifest, indent=2, default=_json_default))
    print(f"\n[geometry-audit] admitted primary (n={len(admitted)}): {admitted}")
    print(f"[geometry-audit] cohort_go={manifest['cohort_go']} -> {OUT_DIR}/geometry_audit.csv")
    return manifest


# ============================================================ rss audit (one full baseline)
def cmd_rss_audit(args):
    subject = args.subject or "epilepsiae_1146"
    seed = int(args.seed) if args.seed else 1
    n_steps = int(T_MAX_MS / DT)
    shared = G.build_shared_net(seed)
    pack = _patient_pack(subject, args.input_root, shared)
    t0 = time.time()
    verdict, res, obs, slow, win = _baseline_and_window(shared, pack, subject, seed, n_steps)
    wall = time.time() - t0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    sw, snap = _safe_workers(peak)
    out = dict(subject=subject, seed=seed, N=shared["N"], NE=shared["NE"], n_steps=n_steps,
               wall_s=round(wall, 1), peak_rss_gib=round(peak, 2), safe_workers=sw, mem=snap,
               baseline_verdict={k: verdict[k] for k in verdict if k not in ("res",)})
    print(json.dumps(out, indent=2, default=_json_default))
    print(f"\n[rss-audit] {subject} seed{seed}: wall={wall:.1f}s peak_RSS={peak:.2f} GiB -> safe_workers={sw}")
    return out


# ============================================================ pilot (parity + eligibility + RSS)
def cmd_pilot(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    subjects = (args.subjects.split(",") if args.subjects else ["epilepsiae_1146", "yuquan_zhaochenxi"])
    seed = int(args.seed) if args.seed else 1
    n_steps = int(T_MAX_MS / DT)
    shared = G.build_shared_net(seed)
    report = dict(seeds=[seed], subjects=subjects, generated_at=_now(), runs=[])
    for subject in subjects:
        pack = _patient_pack(subject, args.input_root, shared)
        verdict, bres, bobs, bslow, win = _baseline_and_window(shared, pack, subject, seed, n_steps)
        base_summary = G.summarize_run(bres, bobs, bslow, arm="baseline_no_stim", dt=DT,
                                       frozen_bar=verdict["frozen_event_bar"],
                                       stim_on_ms=verdict.get("stim_on_ms", T_MAX_MS),
                                       stim_off_ms=verdict.get("stim_off_ms", T_MAX_MS), t_max_ms=T_MAX_MS,
                                       coredist_mm=pack["coredist"], core_r=G.SNN["core_r"],
                                       spatial_bin_ms=SPATIAL_BIN_MS,
                                       baseline_total_activity=verdict["base_total_activity"])
        entry = dict(subject=subject, coredist_mm=round(pack["coredist"], 2),
                     n_target=pack["targets"]["n_target"], candidate_counts=pack["targets"]["candidate_counts"],
                     verdict={k: verdict[k] for k in verdict if k != "res"}, baseline_summary=base_summary,
                     stim_arms={})
        print(f"\n[pilot] {subject}: coredist={pack['coredist']:.2f}mm eligible={verdict['eligible']} "
              f"reason={verdict.get('reason')} runaway={verdict.get('baseline_runaway_ms')} "
              f"n_pre_recover={verdict.get('n_pre_stim_recoverable')} N_target={pack['targets']['n_target']}")
        if win is None:
            report["runs"].append(entry)
            continue
        stim_on, stim_off, bar, bta = win
        # pre-stim parity: baseline rate up to stim_on must equal a stim arm's rate up to stim_on
        parity_ref = np.asarray(bres["rate_E"], float)
        for arm in ARMS_PRIMARY:
            r = _run_and_save(shared, pack, subject, seed, arm, stim_on_ms=stim_on, stim_off_ms=stim_off,
                              frozen_bar=bar, baseline_total_activity=bta, n_steps=n_steps, save_arrays=True)
            if r["status"] in ("ok", "resumed") and "summary" in r:
                s = r["summary"]
                entry["stim_arms"][arm] = s
                # parity check up to stim_on (only when we have the live res)
                if "res" in r:
                    a = np.asarray(r["res"]["rate_E"], float)
                    ns = int(stim_on / DT)
                    m = min(len(parity_ref), len(a), ns)
                    par = bool(np.array_equal(parity_ref[:m], a[:m]))
                    entry.setdefault("pre_stim_parity", {})[arm] = par
                print(f"  [{arm}] runaway={s['runaway_ms']} censored={s['censored']} "
                      f"RRT={s['restricted_runaway_free_time_ms']} escape={s['escape_prob']} "
                      f"far={s['far_side_recruit']} local={s['n_post_local_events']} "
                      f"totact_ratio={s['total_activity_ratio']} wall={r.get('wall_s')}s "
                      f"rss={r.get('peak_rss_gib')}GiB")
        report["runs"].append(entry)
    _atomic_json(os.path.join(OUT_DIR, "pilot_report.json"), report)
    print(f"\n[pilot] -> {OUT_DIR}/pilot_report.json")
    return report


# ============================================================ cohort campaign
_W = {}   # module global for fork-COW workers (set before each Pool)


def _baseline_worker(subject_id):
    """Run (or resume) the no-stim baseline ONCE; save artifact with the embedded verdict; return the
    window + eligibility for Phase B. Resume: a matching-fingerprint baseline artifact with a stored
    verdict is reused without re-simulating."""
    shared, input_root, n_steps = _W["shared"], _W["input_root"], _W["n_steps"]
    seed = shared["seed"]
    try:
        pack = _patient_pack(subject_id, input_root, shared)
    except Exception as exc:
        return dict(subject_id=subject_id, status="pack_error", error=f"{type(exc).__name__}: {exc}")
    montage = pack["montage"]
    fp = arm_fingerprint(subject_id, seed, "baseline_no_stim", montage, n_steps, None, 0, STIM_DELTA_MV, STIM_RADIUS_MM)
    jpath = os.path.join(PER_RUN, subject_id, str(seed), "baseline_no_stim.json")
    if os.path.isfile(jpath):
        try:
            prev = json.load(open(jpath))
            if prev.get("fingerprint") == fp and prev.get("baseline_verdict"):
                v = prev["baseline_verdict"]
                v["subject_id"] = subject_id
                v["status"] = "resumed"
                v["coredist"] = pack["coredist"]
                return v
        except Exception:
            pass
    verdict, res, obs, slow, win = _baseline_and_window(shared, pack, subject_id, seed, n_steps)
    stim_on = verdict.get("stim_on_ms", T_MAX_MS) or T_MAX_MS
    stim_off = verdict.get("stim_off_ms", T_MAX_MS) or T_MAX_MS
    summary = G.summarize_run(res, obs, slow, arm="baseline_no_stim", dt=DT,
                              frozen_bar=verdict["frozen_event_bar"], stim_on_ms=stim_on, stim_off_ms=stim_off,
                              t_max_ms=T_MAX_MS, coredist_mm=pack["coredist"], core_r=G.SNN["core_r"],
                              spatial_bin_ms=SPATIAL_BIN_MS, baseline_total_activity=verdict["base_total_activity"])
    rec = dict(schema=G.SCHEMA_VERSION, fingerprint=fp, status="ok", subject_id=subject_id, seed=seed,
               arm="baseline_no_stim", n_steps=n_steps, dt=DT, t_max_ms=T_MAX_MS,
               coredist_mm=round(pack["coredist"], 3), summary=summary, baseline_verdict=verdict,
               generated_at=_now())
    _atomic_json(jpath, rec)
    _atomic_npz(os.path.join(PER_RUN, subject_id, str(seed), "baseline_no_stim.npz"),
                **G.arm_arrays(res, obs, slow))
    out = dict(verdict)
    out.update(subject_id=subject_id, status="ok", coredist=pack["coredist"])
    return out


def _stim_worker(task):
    subject_id, arm, stim_on, stim_off, bar, bta = task
    shared, input_root, n_steps = _W["shared"], _W["input_root"], _W["n_steps"]
    try:
        pack = _patient_pack(subject_id, input_root, shared)
    except Exception as exc:
        return dict(subject_id=subject_id, arm=arm, status="pack_error", error=str(exc))
    r = _run_and_save(shared, pack, subject_id, shared["seed"], arm, stim_on_ms=stim_on, stim_off_ms=stim_off,
                      frozen_bar=bar, baseline_total_activity=bta, n_steps=n_steps, save_arrays=True)
    r.pop("res", None)
    r.pop("obs", None)
    return r


def cmd_cohort(args):
    os.makedirs(PER_RUN, exist_ok=True)
    manifest = json.load(open(os.path.join(OUT_DIR, "cohort_manifest.json")))
    subjects = list(manifest["admitted_primary"])
    if args.include_sensitivity:
        subjects += list(manifest.get("admitted_sensitivity", []))
    if args.subjects:
        subjects = args.subjects.split(",")
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else SEEDS)]
    n_steps = int(T_MAX_MS / DT)
    workers = int(args.workers) if args.workers else 4
    off_axis = bool(args.offaxis)
    force_pilot = bool(getattr(args, "force_pilot", False))
    reslog = os.path.join(OUT_DIR, "resource_log.csv")
    run_manifest_rows = []
    t_start = time.time()
    print(f"[cohort] subjects={subjects} seeds={seeds} workers={workers} n_steps={n_steps} T_max={T_MAX_MS}ms")
    for seed in seeds:
        shared = G.build_shared_net(seed)
        _W.update(shared=shared, input_root=args.input_root, n_steps=n_steps)
        _resource_log_row(reslog, dict(phase=f"net_built_seed{seed}", workers=0, running=0,
                                       completed=len(run_manifest_rows)))
        # ---- Phase A: baselines (parallel) ----
        print(f"[cohort seed {seed}] Phase A: {len(subjects)} baselines with {workers} workers")
        base_res = _map_pool(_baseline_worker, subjects, workers, reslog, f"baseline_seed{seed}",
                             len(run_manifest_rows))
        eligible = {}
        for br in base_res:
            run_manifest_rows.append(dict(seed=seed, subject=br["subject_id"], arm="baseline_no_stim",
                                          status=br["status"], eligible=br.get("eligible"),
                                          reason=br.get("reason"), coredist_mm=round(br.get("coredist", 0) or 0, 2),
                                          baseline_runaway_ms=br.get("baseline_runaway_ms")))
            print(f"  [baseline] {br['subject_id']}: {br['status']} eligible={br.get('eligible')} "
                  f"reason={br.get('reason')} runaway={br.get('baseline_runaway_ms')} "
                  f"n_pre={br.get('n_pre_stim_recoverable')}")
            # strict cohort eligibility, OR (force_pilot) any subject with a defined runaway window.
            strict_ok = br["status"] in ("ok", "resumed") and br.get("eligible")
            pilot_ok = (force_pilot and br["status"] in ("ok", "resumed")
                        and br.get("baseline_runaway_ms") is not None and br.get("stim_on_ms") is not None)
            if strict_ok or pilot_ok:
                br["_tier"] = "strict" if strict_ok else "mechanism_pilot"
                eligible[br["subject_id"]] = br
        # ---- Phase B: stim arms (parallel) ----
        arms = list(ARMS_PRIMARY) + (["gradient_offaxis_control"] if off_axis else [])
        tasks = [(sid, arm, br["stim_on_ms"], br["stim_off_ms"], br["frozen_event_bar"],
                  br["base_total_activity"]) for sid, br in eligible.items() for arm in arms]
        print(f"[cohort seed {seed}] Phase B: {len(tasks)} stim arms ({len(eligible)} eligible subjects)")
        stim_res = _map_pool(_stim_worker, tasks, workers, reslog, f"stim_seed{seed}", len(run_manifest_rows))
        for sr in stim_res:
            run_manifest_rows.append(dict(seed=seed, subject=sr["subject_id"], arm=sr["arm"],
                                          status=sr["status"],
                                          rrt=(sr.get("summary") or {}).get("restricted_runaway_free_time_ms"),
                                          runaway_ms=(sr.get("summary") or {}).get("runaway_ms"),
                                          censored=(sr.get("summary") or {}).get("censored")))
        _write_run_manifest(run_manifest_rows)
    dur = time.time() - t_start
    print(f"\n[cohort] done in {dur/60:.1f} min. run_manifest rows={len(run_manifest_rows)}")
    _write_run_manifest(run_manifest_rows)
    return run_manifest_rows


def _map_pool(fn, items, workers, reslog, phase, completed_offset):
    """Fork Pool with periodic resource logging; workers write their own artifacts."""
    if not items:
        return []
    if workers <= 1:
        out = []
        for it in items:
            out.append(fn(it))
            _resource_log_row(reslog, dict(phase=phase, workers=1, running=1,
                                           completed=completed_offset + len(out)))
        return out
    results = []
    with mp.Pool(min(workers, len(items))) as pool:
        it = pool.imap_unordered(fn, items)
        last_log = 0.0
        for r in it:
            results.append(r)
            now = time.time()
            if now - last_log > 30.0:
                _resource_log_row(reslog, dict(phase=phase, workers=workers, running=min(workers, len(items)),
                                               completed=completed_offset + len(results)))
                last_log = now
    _resource_log_row(reslog, dict(phase=phase + "_done", workers=workers, running=0,
                                   completed=completed_offset + len(results)))
    return results


def _write_run_manifest(rows):
    if not rows:
        return
    cols = ["seed", "subject", "arm", "status", "eligible", "reason", "coredist_mm",
            "baseline_runaway_ms", "rrt", "runaway_ms", "censored"]
    with open(os.path.join(OUT_DIR, "run_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


# ============================================================ aggregation + statistics
def _load_arm_full(subject, seed, arm):
    """Arm summary JSON + the raw axial_act (saved un-downsampled) so cross-corridor spread can be scored
    on a COMMON window shared by all three arms — NOT each arm's own [stim_off, t_run) (which couples the
    metric to the RRT outcome: an arm that delays runaway would get a longer observation window)."""
    jp = os.path.join(PER_RUN, subject, str(seed), f"{arm}.json")
    if not os.path.isfile(jp):
        return None
    rec = json.load(open(jp))
    s = dict(rec.get("summary") or {})
    npz = os.path.join(PER_RUN, subject, str(seed), f"{arm}.npz")
    if os.path.isfile(npz) and rec.get("stim_off_ms") is not None:
        try:
            d = np.load(npz)
            s["_axial"] = np.asarray(d["axial_act"], float)
            s["_edges"] = np.asarray(d["ax_edges"], float)
            s["_stim_off"] = float(rec["stim_off_ms"])
            s["_coredist"] = float(rec["coredist_mm"])
            s["_t_run"] = float(s["runaway_ms"]) if s.get("runaway_ms") is not None else T_MAX_MS
        except Exception:
            pass
    return s


def _common_window_far_delta(neg, pos, mid, *, window_ms=1000.0):
    """far-reach-prob(middle) - mean far-reach-prob(endpoints) on the SAME window for all three arms:
    [stim_off, min(stim_off+window_ms, earliest arm runaway)). Decoupled from each arm's own RRT."""
    if not all("_axial" in a for a in (neg, pos, mid)):
        return None
    cutoff = min(neg["_stim_off"] + window_ms, neg["_t_run"], pos["_t_run"], mid["_t_run"])

    def _fr(a):
        return G.prerunaway_propagation(a["_axial"], a["_edges"], a["_coredist"], G.SNN["core_r"],
                                        SPATIAL_BIN_MS, stim_off_ms=a["_stim_off"], t_run_ms=cutoff)["far_reach_prob"]

    fm, fn, fp = _fr(mid), _fr(neg), _fr(pos)
    if any(not np.isfinite(x) for x in (fm, fn, fp)):
        return None
    return float(fm - 0.5 * (fn + fp))


def _stat_block(vals):
    """Descriptive median + exact sign-flip test ON THE MEAN + Wilcoxon (labels kept explicit so the
    p-value is never mistaken for a test on the median). n=4 exact two-sided power floor is p>=0.125."""
    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return dict(n=0, median=None, mean=None, n_positive=0)
    sf = G.paired_sign_flip_test(vals)
    wx = G.wilcoxon_signed_rank(vals)
    return dict(n=len(vals), median=float(np.median(vals)), mean=float(np.mean(vals)),
                n_positive=int(sum(1 for v in vals if v > 0)),
                exact_sign_flip_on_mean_p=sf["p_value"], wilcoxon_p=wx["p_value"],
                values=[round(v, 1) for v in vals])


def cmd_aggregate(args):
    manifest = json.load(open(os.path.join(OUT_DIR, "cohort_manifest.json")))
    subjects = list(manifest["admitted_primary"])
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else SEEDS)]

    seed_rows = []
    for subject in subjects:
        for seed in seeds:
            neg = _load_arm_full(subject, seed, "gradient_endpoint_negative")
            pos = _load_arm_full(subject, seed, "gradient_endpoint_positive")
            mid = _load_arm_full(subject, seed, "gradient_middle")
            if not (neg and pos and mid):
                continue
            rrt = {a: s["restricted_runaway_free_time_ms"] for a, s in (("neg", neg), ("pos", pos), ("mid", mid))}
            seed_rows.append(dict(
                subject=subject, seed=seed,
                c_run=rrt["mid"] - 0.5 * (rrt["neg"] + rrt["pos"]),
                c_best=rrt["mid"] - max(rrt["neg"], rrt["pos"]),
                rrt_middle=rrt["mid"], rrt_neg=rrt["neg"], rrt_pos=rrt["pos"],
                runaway_mid=mid.get("runaway_ms"), censored_mid=mid.get("censored"),
                totact_mid=mid.get("total_activity_ratio"),
                # cross-corridor spread on a COMMON 1 s post-stim window shared by all 3 arms (decoupled
                # from each arm's own runaway time so it is not an artifact of the RRT outcome)
                far_delta_common1s=_common_window_far_delta(neg, pos, mid, window_ms=1000.0)))

    if seed_rows:
        cols = list(seed_rows[0])
        with open(os.path.join(OUT_DIR, "per_seed_effects.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(seed_rows)

    seeds_present = {s: {r["subject"] for r in seed_rows if r["seed"] == s} for s in seeds}
    balanced_seeds = sorted(s for s in seeds if seeds_present[s] == set(subjects))

    def cohort_over(seed_set):
        per_subj = {}
        for subject in subjects:
            v = [r for r in seed_rows if r["subject"] == subject and r["seed"] in seed_set]
            if v:
                fd = [r["far_delta_common1s"] for r in v if r["far_delta_common1s"] is not None]
                per_subj[subject] = dict(
                    n_seeds=len(v), c_run=float(np.median([r["c_run"] for r in v])),
                    c_best=float(np.median([r["c_best"] for r in v])),
                    rrt_middle=float(np.median([r["rrt_middle"] for r in v])),
                    rrt_neg=float(np.median([r["rrt_neg"] for r in v])),
                    rrt_pos=float(np.median([r["rrt_pos"] for r in v])),
                    far_delta_common1s=(float(np.median(fd)) if fd else None))
        return dict(seeds=sorted(seed_set), n_subjects=len(per_subj), per_subject=per_subj,
                    c_run=_stat_block([v["c_run"] for v in per_subj.values()]),
                    c_best=_stat_block([v["c_best"] for v in per_subj.values()]))

    per_seed_cohort = {}
    for s in seeds:
        cr = [r["c_run"] for r in seed_rows if r["seed"] == s]
        cb = [r["c_best"] for r in seed_rows if r["seed"] == s]
        per_seed_cohort[str(s)] = dict(
            n_subjects=len(cr), subjects_present=sorted(seeds_present[s]),
            c_run_median=(float(np.median(cr)) if cr else None), c_run_n_positive=int(sum(1 for v in cr if v > 0)),
            c_best_median=(float(np.median(cb)) if cb else None), c_best_n_positive=int(sum(1 for v in cb if v > 0)))

    # Two analysis sets reported SIDE BY SIDE (neither crowned "primary" post-hoc):
    #   all_available   = pre-specified seeds 1/3/4, per-subject median over the seeds each is eligible in
    #   complete_case   = complete-case SENSITIVITY on the seeds where all 4 are eligible (here 1,3)
    all_available = cohort_over(set(seeds))
    complete_case = cohort_over(set(balanced_seeds))
    subj_rows = []
    for subject, v in all_available["per_subject"].items():
        signs = [int(np.sign(r["c_run"])) for r in seed_rows if r["subject"] == subject]
        pref = ("middle" if v["c_run"] > 0 and v["c_best"] >= 0 else
                "middle>avg_only" if v["c_run"] > 0 else "endpoint")
        cc = complete_case["per_subject"].get(subject, {})
        subj_rows.append(dict(subject=subject, tier="primary",
                              all_avail_c_run=round(v["c_run"], 1), all_avail_c_best=round(v["c_best"], 1),
                              complete_case_c_run=round(cc.get("c_run", float("nan")), 1),
                              complete_case_c_best=round(cc.get("c_best", float("nan")), 1),
                              rrt_middle=round(v["rrt_middle"], 1), rrt_neg=round(v["rrt_neg"], 1),
                              rrt_pos=round(v["rrt_pos"], 1), c_run_signs_by_seed=",".join(map(str, signs)),
                              far_delta_common1s=(round(v["far_delta_common1s"], 3)
                                                  if v["far_delta_common1s"] is not None else None),
                              site_preference=pref))
    if subj_rows:
        cols = list(subj_rows[0])
        with open(os.path.join(OUT_DIR, "subject_effects.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(subj_rows)

    same_dir = (all_available["c_run"]["median"] is not None and complete_case["c_run"]["median"] is not None
                and (all_available["c_run"]["median"] < 0) == (complete_case["c_run"]["median"] < 0))
    stats = dict(
        contract=G.SCHEMA_VERSION, generated_at=_now(), unit="subject", tier="descriptive_small_cohort",
        seeds=seeds, complete_case_seeds=balanced_seeds,
        stat_note=("median = descriptive effect. 'exact_sign_flip_on_mean_p' is an EXACT sign-flip "
                   "randomization test on the MEAN statistic, NOT the median; 'wilcoxon_p' is the "
                   "signed-rank test. n=4 exact-test two-sided power floor: p>=0.125 (cannot reach <0.05)."),
        analysis_set_note=("all_available (pre-specified seeds 1/3/4) and complete_case (seeds where all 4 "
                           "eligible = 1,3) are reported side by side; neither is crowned primary post-hoc. "
                           "Both are per-subject-one-median, so complete_case is not 'more weighted' — it "
                           "simply drops seed 4 (where E1146 has no runaway/arms and the other 3 subjects "
                           "happen to be most-positive), so its cohort median is more negative. The DIRECTION "
                           "is the same in both."),
        analysis_sets_same_direction=bool(same_dir),
        e1146_seed4_note=("epilepsiae_1146 seed 4 baseline never reached operational runaway within T_max "
                          "(20 s) -> no stim window/arms -> E1146 contributes only seeds 1,3."),
        propagation_note=("cross-corridor spread = 'far_delta_common1s' (mid minus mean-endpoint far-reach "
                          "probability) scored on a COMMON 1 s post-stim window shared by all three arms "
                          "(decoupled from each arm's own runaway time). It is NOT load-bearing: at the common "
                          "window the per-subject deltas are ~+/-0.01 and only E958 is sign-stable across "
                          "window choices; 'no reduction' therefore means 'no window-robust site difference "
                          "observed', NOT 'stim cannot limit spread'. Whole-run escape/far/span (runaway-"
                          "saturated) retained per-run only."),
        all_available=all_available, complete_case_sensitivity=complete_case,
        seed_stratified=per_seed_cohort, per_subject=subj_rows)
    _atomic_json(os.path.join(OUT_DIR, "cohort_statistics.json"), stats)

    for name, blk in (("all_available (seeds 1/3/4)", all_available), (f"complete_case (seeds {balanced_seeds})", complete_case)):
        c, cb = blk["c_run"], blk["c_best"]
        print(f"[aggregate] {name} n={blk['n_subjects']}: C_run median={c['median']:+.0f}ms (mean {c['mean']:+.0f}) "
              f"+{c['n_positive']}/{c['n']} sign-flip(mean) p={c['exact_sign_flip_on_mean_p']:.3f} "
              f"wilcoxon p={c['wilcoxon_p']:.3f} | C_best median={cb['median']:+.0f} +{cb['n_positive']}/{cb['n']}")
    print(f"  same_direction={same_dir} (both negative -> conclusion robust to the analysis set)")
    print("  seed-stratified C_run median (+pos/n):",
          {k: (None if v["c_run_median"] is None else round(v["c_run_median"]), f"+{v['c_run_n_positive']}/{v['n_subjects']}")
           for k, v in per_seed_cohort.items()})
    for r in subj_rows:
        print(f"    {r['subject']:20s} all_avail C_run={r['all_avail_c_run']:+.0f}/C_best={r['all_avail_c_best']:+.0f} "
              f"complete_case C_run={r['complete_case_c_run']:+.0f} pref={r['site_preference']} "
              f"signs=[{r['c_run_signs_by_seed']}] far_delta_common1s={r['far_delta_common1s']}")
    print(f"[aggregate] -> subject_effects.csv, cohort_statistics.json, per_seed_effects.csv")
    return stats


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("geometry-audit", "rss-audit", "pilot", "cohort", "aggregate"):
        sp = sub.add_parser(name)
        sp.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
        sp.add_argument("--confirm-run", action="store_true")
        sp.add_argument("--subject", default=None)
        sp.add_argument("--subjects", default=None)
        sp.add_argument("--seed", default=None)
        sp.add_argument("--seeds", default=None)
        sp.add_argument("--workers", default=None)
        sp.add_argument("--offaxis", action="store_true")
        sp.add_argument("--include-sensitivity", action="store_true")
        sp.add_argument("--force-pilot", action="store_true",
                        help="run stim arms on subjects that have a defined runaway window even if the "
                             "strict >=3-recoverable-pre-stim-events gate fails; results are MECHANISM-PILOT "
                             "tier (per-subject descriptive), NEVER a cohort significance claim")
    args = ap.parse_args(argv)
    if args.cmd == "geometry-audit":
        return cmd_geometry_audit(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    if not args.confirm_run:
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run.", file=sys.stderr)
        sys.exit(2)
    if args.cmd == "rss-audit":
        cmd_rss_audit(args)
    elif args.cmd == "pilot":
        cmd_pilot(args)
    elif args.cmd == "cohort":
        cmd_cohort(args)


if __name__ == "__main__":
    main()
