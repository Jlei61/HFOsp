#!/usr/bin/env python
"""Z/M minimal-carrier branch decision -- orchestration (spec rev3.1, plan Tasks 3.5/6/7/8/13).

Phases (each is crash-safe and writes atomically under results/topic4_sef_hfo/zm_branch_decision/):

  smoke     Task 3.5 vertical slice: anchor -> checkpoint -> restore -> freeze_all -> continuation
            -> source + current-vSEEG metrics. Writes ONLY under smoke/ and carries no evidence.
  anchor1   trace pass: run the locked Z/M+S_G trajectory, save slow coordinates + readouts.
  anchor2   snapshot pass: re-run the SAME trajectory and capture exact states at the slow-state
            bins x natural fast phases selected from anchor1.
  fork      Task 7 minimal carrier-subsystem matrix over arms x paired-noise replicates.
            ``--evidence-tier long_confirmation`` writes a separate 20 s central-state result.
            ``--resolution dt2 --evidence-tier dt2_confirmation`` consumes only an independently
            generated dt/2-native anchor and writes another separate result namespace.

Every phase pins the canonical config SHA, the engine SHAs, the state hash and the noise-bank SHA.
OMP/MKL/OPENBLAS/NUMEXPR are forced to 1.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import resource
import sys
import time

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _p in (_ROOT, _SCRIPTS, os.path.join(_ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import run_m4_phaseplane as PP                       # noqa: E402
import run_zm_snn_native_exit as ZM                  # noqa: E402
from kick_probe import simulate_kick                 # noqa: E402
from lfp import LFPRecorder                          # noqa: E402
from slow_field import SpatialSlowField              # noqa: E402

import src.topic4_zm_fork_state as FS                # noqa: E402
import src.topic4_zm_checkpoint as CK                # noqa: E402
import src.topic4_zm_noise_bank as NB                # noqa: E402
import src.topic4_zm_minimal_carrier as MC           # noqa: E402
import src.topic4_zm_ictal_carrier as CG             # noqa: E402

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")
PHASE0 = os.path.join(OUT, "phase0")
DT_BASE = 0.1
DT_HALF = 0.05
RESOLUTION_DT = {"dt": DT_BASE, "dt2": DT_HALF}
ES_THRESH_HZ = 250.0        # bounded-arm runaway threshold (the containment arm plateaus high)
ARM_KWARGS = dict(use_SG=True, alpha_G=16.0)


# ================================================================ helpers
def _rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 ** 2, 2)


def _mem_avail_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                return round(int(line.split()[1]) / 1024.0 ** 2, 1)
    return float("nan")


def _git_sha():
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
                          text=True).stdout.strip()


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else o)
    os.replace(tmp, path)


def save_npz_atomic(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def load_lock(seed, *, resolution="dt"):
    if resolution not in RESOLUTION_DT:
        raise ValueError(f"unknown resolution {resolution!r}; choices={sorted(RESOLUTION_DT)}")
    dt = RESOLUTION_DT[resolution]
    lock = json.load(open(os.path.join(PHASE0, "canonical_config.json")))
    parent = lock["seeds"][str(int(seed))]
    live = FS.build_canonical_config(seed=seed, I_th_EI=parent["config"]["I_th_EI"],
                                     arm_kwargs=ARM_KWARGS, dt=dt)
    live_sha = FS.config_sha(live)
    if resolution == "dt":
        if live_sha != parent["config_sha"]:
            raise SystemExit(f"config drift for seed {seed}: live {live_sha[:16]} != locked "
                             f"{parent['config_sha'][:16]}; re-run "
                             "audit_topic4_zm_dynamic_state.py")
        return parent["config"], parent["config_sha"], dt

    # A dt snapshot cannot be converted to dt/2.  Lock a separate native
    # configuration per seed before constructing the independent anchor.
    lock_path = os.path.join(PHASE0, "dt2", f"seed{int(seed)}_config.json")
    expected = dict(
        resolution="dt2", dt=dt, seed=int(seed),
        parent_config_sha=parent["config_sha"],
        config_sha=live_sha, config=live,
        created_from="independent dt/2 canonical config; no state interpolation",
    )
    if os.path.exists(lock_path):
        old = json.load(open(lock_path))
        if old.get("config_sha") != live_sha or old.get("parent_config_sha") != parent["config_sha"]:
            raise SystemExit(
                f"dt/2 config drift for seed {seed}: live {live_sha[:16]} != "
                f"locked {str(old.get('config_sha'))[:16]}"
            )
    else:
        write_json_atomic(lock_path, expected)
    return live, live_sha, dt


def build_context(seed, *, smoke=False, resolution="dt"):
    """Substrate + slow config + virtual-SEEG montage, all pinned to the Phase-0 lock."""
    # Capture this ONCE.  A multi-hour continuation may span a later commit in
    # the same worktree; resolving HEAD when the row is finally written would
    # falsely attribute already-imported code to that later commit.
    runtime_git_sha = _git_sha()
    runtime_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    cfg_locked, cfg_sha, dt = load_lock(seed, resolution=resolution)
    t0 = time.time()
    S = PP.build_substrate(seed=int(seed), dt=dt)
    S["seed"] = int(seed)
    S["I_th_EI"] = float(cfg_locked["I_th_EI"])
    mont = S["reg"]["montage_sheet"]
    rec = LFPRecorder(S["p"], S["net"]["pos"], S["net"]["labels"],
                      sites=np.asarray(mont.contacts, float))
    core = ZM._core_mask_E(S)
    along, _perp = CG.axis_transverse_coords(S["posE"], S["src_xy"], S["axis_unit"])
    print(f"[ctx] seed={seed} built in {time.time()-t0:.0f}s N={S['N']} contacts={len(mont.names)} "
          f"resolution={resolution} dt={dt:g} config_sha={cfg_sha[:16]} "
          f"rss={_rss_gb()}GB", flush=True)
    return dict(S=S, rec=rec, core=core, axis=along, contacts=list(mont.names),
                cfg_locked=cfg_locked, cfg_sha=cfg_sha, smoke=smoke,
                resolution=resolution, dt=dt,
                anchor_root=("anchors" if resolution == "dt" else "anchors_dt2"),
                runtime_git_sha=runtime_git_sha, runtime_started_at=runtime_started_at)


def make_slow(ctx, freeze_arm=None):
    cfg = ZM._zm_cfg(ctx["S"]["I_th_EI"], **ARM_KWARGS)
    slow = SpatialSlowField(ctx["S"]["N"], 18.0, ctx["S"]["posE"], ctx["S"]["posI"],
                            ctx["S"]["L"], core_mask_E=ctx["core"], cfg=cfg)
    if freeze_arm is not None:
        slow = FS.FreezeWrapper(slow, FS.FreezePolicy.for_arm(freeze_arm))
    return slow


def run_segment(ctx, slow, T_ms, *, ckpt=None, fresh_rng=True):
    S = ctx["S"]
    p = dataclasses.replace(S["p"], T=float(T_ms))
    if fresh_rng:
        S["net"]["rng"] = np.random.default_rng(S["seed"])
    return simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                         r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"], verbose=False,
                         lfp_recorder=ctx["rec"], early_stop_runaway=True,
                         es_thresh_hz=ES_THRESH_HZ, es_dur_ms=100.0, zm_ckpt=ckpt)


def segment_metrics(ctx, res, *, bin_ms=MC.BIN_MS):
    S = ctx["S"]
    m = MC.source_metrics(res["E_spk_bool"], ctx["core"], S["posE"], S["L"], ctx["dt"],
                          bin_ms=bin_ms,
                          lfp_trace=res.get("lfp_trace"), axis_coord=ctx["axis"])
    m["runaway_early_stop_ms"] = res.get("runaway_early_stop_ms")
    return m


def slow_coords(slow, n_bins, n_steps):
    """Slow coordinates on the SOURCE metric bin grid (decimated, not re-derived)."""
    s = CK._inner(slow)
    idx = np.linspace(0, max(0, n_steps - 1), max(1, n_bins)).astype(int)

    def take(tr, default=0.0):
        a = np.asarray(tr, float)
        if a.size == 0:
            return np.full(len(idx), default, np.float32)
        return a[np.clip(idx, 0, a.size - 1)].astype(np.float32)

    return dict(z_mean=take(s.trace_z_mean, 1.0), z_min=take(s.trace_z_min, 1.0),
                z_core=take(s.trace_z_core_mean, 1.0), z_surround=take(s.trace_z_surround_mean, 1.0),
                m_mean=take(s.trace_m_mean), m_max=take(s.trace_m_max),
                m_core=take(s.trace_m_core_mean), m_surround=take(s.trace_m_surround_mean),
                S_G=take(s.trace_SG), mu_G=take(s.trace_muG))


def provenance(ctx, **extra):
    p = dict(git_sha=ctx["runtime_git_sha"],
             runtime_started_at=ctx["runtime_started_at"],
             config_sha=ctx["cfg_sha"], seed=int(ctx["S"]["seed"]), dt=ctx["dt"],
             resolution=ctx["resolution"],
             engine_sha256=ctx["cfg_locked"]["engine_sha256"],
             metrics_version=MC.METRICS_VERSION, state_schema=CK.STATE_SCHEMA,
             lockpoint=ctx["cfg_locked"]["lockpoint"], arm_kwargs=ARM_KWARGS,
             es_thresh_hz=ES_THRESH_HZ, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
             peak_rss_gb=_rss_gb(), mem_available_gb=_mem_avail_gb(),
             resource_log="results/topic4_sef_hfo/zm_branch_decision/resource_log.jsonl", pid=os.getpid())
    p.update(extra)
    return p


# ================================================================ phase: anchor1 (trace pass)
def phase_anchor1(ctx, T_ms, tag="anchor"):
    t0 = time.time()
    slow = make_slow(ctx)
    res = run_segment(ctx, slow, T_ms)
    n_steps = res["E_spk_bool"].shape[0]
    met = segment_metrics(ctx, res)
    sc = slow_coords(slow, met["n_bins"], n_steps)
    lfp_ds, fs = CG.decimate_lfp(res["lfp_trace"])
    seed = ctx["S"]["seed"]
    base = os.path.join(OUT, "smoke" if ctx["smoke"] else ctx["anchor_root"])
    save_npz_atomic(os.path.join(base, f"{tag}_seed{seed}_traces.npz"),
                    **{k: np.asarray(v, np.float32) for k, v in met.items()
                       if isinstance(v, np.ndarray)},
                    **{f"slow_{k}": v for k, v in sc.items()},
                    lfp=lfp_ds.astype(np.float32), lfp_fs=np.asarray(fs))
    man = provenance(ctx, phase="anchor1", tag=tag, T_ms=float(T_ms), n_steps=int(n_steps),
                     n_bins=int(met["n_bins"]), bin_ms=MC.BIN_MS,
                     runaway_early_stop_ms=res.get("runaway_early_stop_ms"),
                     contacts=ctx["contacts"], lfp_fs=float(fs),
                     peak_r_all_hz=float(np.max(met["r_all"])),
                     tail_r_all_hz=float(np.mean(met["r_all"][-max(1, met["n_bins"] // 20):])),
                     z_core_final=float(sc["z_core"][-1]), S_G_max=float(np.max(sc["S_G"])),
                     wall_s=round(time.time() - t0, 1))
    write_json_atomic(os.path.join(base, f"{tag}_seed{seed}_traces.json"), man)
    print(f"[anchor1] seed={seed} steps={n_steps} bins={met['n_bins']} "
          f"peak={man['peak_r_all_hz']:.1f}Hz tail={man['tail_r_all_hz']:.1f}Hz "
          f"runaway={man['runaway_early_stop_ms']} wall={man['wall_s']}s rss={_rss_gb()}GB",
          flush=True)
    return man


# ================================================================ phase: anchor (trace + states)
COARSE_MS = 1000.0          # spacing of the pass-1 checkpoint grid used as resume springboards
REST_QUANTILE = 0.95        # d_rest threshold = this quantile of the anchor's own rest window
REST_DWELL_MS = 200.0       # ...held this long before the trajectory counts as back in the basin


def phase_anchor(ctx, T_ms):
    """Pass 1: full trajectory + coarse checkpoint grid + traces + source/rest locks.
    Pass 2: for each selected slow-bin x fast-phase state, resume from the nearest coarse checkpoint
    and capture the EXACT state at the target step (no second full re-run).
    """
    seed = ctx["S"]["seed"]
    base = os.path.join(
        OUT, "smoke" if ctx["smoke"] else ctx["anchor_root"], f"seed{seed}"
    )
    t0 = time.time()
    dt = ctx["dt"]
    bs = int(round(MC.BIN_MS / dt))
    n_total = int(round(T_ms / dt))
    coarse = sorted(
        set(range(int(round(COARSE_MS / dt)), n_total, int(round(COARSE_MS / dt))))
    )

    slow = make_slow(ctx)
    ck = CK.ZMCheckpoint(snapshot_steps=coarse)
    res = run_segment(ctx, slow, T_ms, ckpt=ck)
    n_steps = res["E_spk_bool"].shape[0]
    met = segment_metrics(ctx, res)
    sc = slow_coords(slow, met["n_bins"], n_steps)
    lfp_ds, fs = CG.decimate_lfp(res["lfp_trace"])
    wall1 = time.time() - t0
    print(f"[anchor:p1] seed={seed} steps={n_steps} bins={met['n_bins']} "
          f"snaps={len(ck.snapshots)} wall={wall1:.0f}s rss={_rss_gb()}GB", flush=True)

    import src.topic4_zm_anchor_states as AS
    sel = AS.select_states(met, sc, MC.BIN_MS, res.get("runaway_early_stop_ms"))
    elig = sel["eligibility"]

    # ---- source-space rest / IED locks (spec §4.4): from the anchor's own pre-escalation window
    locks = None
    if elig["eligible"]:
        ref = MC.rest_reference(met, 0, sel["rest_window"]["hi_bin"])
        d = MC.rest_distance(met, ref)
        d_rest_thresh = float(np.quantile(d[:sel["rest_window"]["hi_bin"]], REST_QUANTILE))
        locks = dict(rest_reference=ref, d_rest_thresh=d_rest_thresh, rest_dwell_ms=REST_DWELL_MS,
                     rest_quantile=REST_QUANTILE,
                     ied_lifetime_ms=elig["returning_events"]["median_duration_ms"],
                     d_rest_anchor=d.astype(float).tolist())

    save_npz_atomic(os.path.join(base, "anchor_traces.npz"),
                    **{k: np.asarray(v, np.float32) for k, v in met.items()
                       if isinstance(v, np.ndarray)},
                    **{f"slow_{k}": v for k, v in sc.items()},
                    lfp=lfp_ds.astype(np.float32), lfp_fs=np.asarray(fs))
    man = provenance(ctx, phase="anchor", T_ms=float(T_ms), n_steps=int(n_steps),
                     n_bins=int(met["n_bins"]), bin_ms=MC.BIN_MS, coarse_ms=COARSE_MS,
                     runaway_early_stop_ms=res.get("runaway_early_stop_ms"),
                     contacts=ctx["contacts"], lfp_fs=float(fs), selection=sel, locks=locks,
                     selection_version=AS.SELECTION_VERSION,
                     peak_r_all_hz=float(np.max(met["r_all"])),
                     tail_r_all_hz=float(np.mean(met["r_all"][-max(1, met["n_bins"] // 20):])),
                     z_core_final=float(sc["z_core"][-1]), S_G_max=float(np.max(sc["S_G"])),
                     wall_pass1_s=round(wall1, 1))
    write_json_atomic(os.path.join(base, "anchor.json"), man)
    if not elig["eligible"]:
        print(f"[anchor] seed={seed} NOT ELIGIBLE: {elig['reasons']}", flush=True)
        return man

    # ---- pass 2: exact state capture at each selected step -------------------------------------
    del res
    targets = []
    for st in sel["states"]:
        t_star = int(st["bin_index"]) * bs
        targets.append((t_star, st))
    targets.sort(key=lambda x: (x[0], x[1]["bin_name"], x[1]["fast_phase"]))
    springs = np.array(coarse, dtype=int)
    captured = []
    for t_star, st in targets:
        if t_star <= 0:
            continue
        cand = springs[springs <= t_star]
        c = int(cand.max()) if cand.size else None
        tag = f"{st['bin_name']}__{st['fast_phase']}"
        path = os.path.join(base, "states", f"{tag}_t{t_star}.npz")
        if c is None or c == t_star:
            state = ck.snapshots.get(t_star) or ck.snapshots.get(c)
            if state is None:
                print(f"[anchor:p2] {tag}: no springboard for step {t_star}, skipped", flush=True)
                continue
        else:
            slow2 = make_slow(ctx)
            ck2 = CK.ZMCheckpoint(initial_state=ck.snapshots[c], snapshot_steps=[t_star])
            run_segment(ctx, slow2, (t_star - c) * dt, ckpt=ck2, fresh_rng=True)
            state = ck2.snapshots.get(t_star)
            if state is None:
                print(f"[anchor:p2] {tag}: capture failed (early stop?), skipped", flush=True)
                continue
        smanifest = CK.save_state_npz(state, dict(
            config_sha=ctx["cfg_sha"], engine_sha=ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"],
            dt=dt, seed=int(seed), t_step=int(t_star), t_ms=float(t_star * dt),
            bin_name=st["bin_name"], fast_phase=st["fast_phase"], springboard_step=c,
            slow_coord=st.get("slow_coord"), git_sha=ctx["runtime_git_sha"],
            runtime_started_at=ctx["runtime_started_at"]), path)
        captured.append(dict(bin_name=st["bin_name"], fast_phase=st["fast_phase"], t_step=int(t_star),
                             t_ms=float(t_star * dt), path=os.path.relpath(path, _ROOT),
                             state_hash=smanifest["state_hash"], springboard_step=c,
                             size_mb=round(os.path.getsize(path) / 1024 ** 2, 1)))
        print(f"[anchor:p2] {tag} t={t_star * dt:.0f}ms captured "
              f"({captured[-1]['size_mb']}MB, spring={c}) rss={_rss_gb()}GB", flush=True)
    man["captured_states"] = captured
    man["wall_s"] = round(time.time() - t0, 1)
    man["peak_rss_gb"] = _rss_gb()
    write_json_atomic(os.path.join(base, "anchor.json"), man)
    print(f"[anchor] seed={seed} eligible=True states={len(captured)} "
          f"bounded_ms={elig['bounded_ms']:.0f} wall={man['wall_s']}s", flush=True)
    return man


# ================================================================ phase: fork matrix (Task 7)
T_CONT_MS = 8000.0          # spec §6.2 primary continuation length after burn-in
CHUNK_MS = 1000.0           # continuations run in exact resumable chunks so a dead arm stops early
PLATEAU_AREA_FRAC = 0.50    # saturated whole-field plateau (same constant as the carrier gate)
DEAD_DWELL_MS = 1500.0      # continuous dwell in the rest basin that means "definitively dead"
                            # (>> the ~100-300 ms inter-burst gaps of the relaxation train, so a
                            #  re-igniting train keeps running and gets its reset count measured)
ARMS_ORDER = ("freeze_all", "freeze_zm", "freeze_zsg", "freeze_z", "dynamic_replay",
              "dynamic_z_only")
#: spec §6.2 burn-in: 2 x the slowest CARRIER-CANDIDATE variable left dynamic in the arm
#: (tau_adp=500 ms for M, tau_S=120 ms for the S_G pool); Z is the entry coordinate, not a carrier
#: component, and the two dynamic-Z arms are controls -- both facts are recorded per arm.
BURN_IN_MS = {"freeze_all": 250.0, "freeze_zm": 250.0, "freeze_zsg": 1000.0, "freeze_z": 1000.0,
              "dynamic_replay": 250.0, "dynamic_z_only": 1000.0}
CONTROL_ARMS = ("dynamic_replay", "dynamic_z_only")


def _cat(chunks, key):
    return np.concatenate([np.asarray(c[key], float) for c in chunks]) if chunks else np.zeros(0)


def _count_rest_returns(d_rest, bin_ms, thresh, dwell_ms):
    need = max(1, int(round(dwell_ms / bin_ms)))
    below = np.asarray(d_rest) < thresh
    n, run = 0, 0
    for b in below:
        if b:
            run += 1
            if run == need:
                n += 1
        else:
            run = 0
    return n


def run_continuation(ctx, state0, arm, bank, locks, *, T_ms=T_CONT_MS, chunk_ms=CHUNK_MS):
    """One exact, chunked, early-stopping continuation. Chunking is byte-exact (proved by
    tests/test_topic4_zm_exact_resume.py), so a continuation that has definitively fallen back into
    the interictal basin costs seconds instead of the full window."""
    burn = BURN_IN_MS[arm]
    total_ms = burn + T_ms
    n_chunks = int(np.ceil(total_ms / chunk_ms))
    cur = state0
    chunks, end_reason, runaway_ms = [], None, None
    t0 = time.time()
    for c in range(n_chunks):
        this_ms = min(chunk_ms, total_ms - c * chunk_ms)
        slow = make_slow(ctx, freeze_arm=arm)
        ck = CK.ZMCheckpoint(initial_state=cur, return_final_state=True,
                             rng_state=(bank["rng_state"] if c == 0 else None),
                             ext_mean_only=bank["ext_mean_only"])
        res = run_segment(ctx, slow, this_ms, ckpt=ck)
        met = segment_metrics(ctx, res)
        met["_slow"] = slow_coords(slow, met["n_bins"], res["E_spk_bool"].shape[0])
        chunks.append(met)
        if res.get("runaway_early_stop_ms") is not None:
            runaway_ms = c * chunk_ms + float(res["runaway_early_stop_ms"])
            end_reason = "runaway"
            break
        cur = ck.final_state
        if cur is None:
            end_reason = "truncated_no_final_state"
            break
        # --- streaming stop checks on the concatenated series, evaluated after burn-in only ---
        # NOTE: we do NOT stop at the first rest return. A relaxation burst train returns to the
        # interictal distribution between bursts and then RE-IGNITES; stopping at the first return
        # would make `hfo_like_relaxation_train` undetectable (it needs the re-ignition count).
        # We stop only when the trajectory has stayed in the basin long enough to be dead.
        d_all = _rest_series(chunks, locks)
        b0 = int(round(burn / MC.BIN_MS))
        if MC.first_rest_return(d_all[b0:], MC.BIN_MS, locks["d_rest_thresh"],
                                DEAD_DWELL_MS) is not None:
            end_reason = "dead_in_rest_basin"
            break
        A = _cat(chunks, "A_active")[b0:]
        need = max(1, int(round(500.0 / MC.BIN_MS)))
        if A.size >= need and np.all(A[-need:] >= PLATEAU_AREA_FRAC):
            end_reason = "saturated_plateau"
            break
    return dict(chunks=chunks, end_reason=end_reason, runaway_ms=runaway_ms, burn_in_ms=burn,
                wall_s=round(time.time() - t0, 1), n_chunks_run=len(chunks))


def _rest_series(chunks, locks):
    met = {k: _cat(chunks, k) for k in MC.REST_KEYS}
    met["n_bins"] = int(len(met[MC.REST_KEYS[0]]))
    return MC.rest_distance(met, locks["rest_reference"])


def summarize_continuation(run, locks, T_ms=T_CONT_MS):
    chunks = run["chunks"]
    d = _rest_series(chunks, locks)
    b0 = int(round(run["burn_in_ms"] / MC.BIN_MS))
    d_post = d[b0:]
    surv = MC.survival(d_post, MC.BIN_MS, locks["d_rest_thresh"], locks["rest_dwell_ms"], T_ms,
                       runaway_ms=(None if run["runaway_ms"] is None
                                   else max(0.0, run["runaway_ms"] - run["burn_in_ms"])),
                       plateau_bin=None)
    if run["end_reason"] == "saturated_plateau" and surv["survived"]:
        surv = dict(lifetime_ms=float(len(d_post) * MC.BIN_MS), survived=False,
                    end_reason="saturated_plateau")
    A = _cat(chunks, "A_active")[b0:]
    E = _cat(chunks, "E_vSEEG")[b0:]
    r = _cat(chunks, "r_all")[b0:]
    H = _cat(chunks, "H_spatial")[b0:]
    slow_m = np.concatenate([c["_slow"]["m_core"] for c in chunks])[b0:]
    slow_S_G = np.concatenate([c["_slow"]["S_G"] for c in chunks])[b0:]
    drift_by_coordinate = dict(
        A_active=MC.drift_stats(A),
        E_vSEEG=MC.drift_stats(E),
        m_core=MC.drift_stats(slow_m),
        S_G=MC.drift_stats(slow_S_G),
    )
    stationarity = MC.stationarity_gate(drift_by_coordinate)
    return dict(
        survived=bool(surv["survived"]), lifetime_ms=float(surv["lifetime_ms"]),
        end_reason=surv["end_reason"] or run["end_reason"],
        rest_returns=int(_count_rest_returns(d_post, MC.BIN_MS, locks["d_rest_thresh"],
                                             locks["rest_dwell_ms"])),
        d_rest_median=float(np.median(d_post)) if d_post.size else float("nan"),
        d_rest_min=float(np.min(d_post)) if d_post.size else float("nan"),
        A_active_mean=float(np.mean(A)) if A.size else 0.0,
        A_active_max=float(np.max(A)) if A.size else 0.0,
        r_all_mean_hz=float(np.mean(r)) if r.size else 0.0,
        r_all_peak_hz=float(np.max(r)) if r.size else 0.0,
        H_spatial_mean=float(np.mean(H)) if H.size else 0.0,
        E_vSEEG_mean=float(np.mean(E)) if E.size else 0.0,
        duty_cycle=float(np.mean(r > 0.2 * np.max(r))) if r.size and np.max(r) > 0 else 0.0,
        drift_A=drift_by_coordinate["A_active"],
        drift_E=drift_by_coordinate["E_vSEEG"],
        drift_m=drift_by_coordinate["m_core"],
        drift_S_G=drift_by_coordinate["S_G"],
        stationarity_ok=bool(stationarity["passed"]),
        stationarity=stationarity,
        burn_in_ms=run["burn_in_ms"], n_bins_post_burn=int(d_post.size), wall_s=run["wall_s"])


def dump_continuation_traces(path, run, locks):
    """Per-bin series of one continuation (figure input; summaries alone cannot show a burst train
    re-igniting vs a state that simply died)."""
    chunks = run["chunks"]
    arrays = {k: _cat(chunks, k).astype(np.float32)
              for k in ("r_core", "r_surround", "r_all", "A_active", "H_spatial", "E_vSEEG",
                        "n_grid_active")}
    arrays["d_rest"] = _rest_series(chunks, locks).astype(np.float32)
    for k in ("z_core", "z_surround", "m_core", "S_G"):
        arrays[f"slow_{k}"] = np.concatenate([c["_slow"][k] for c in chunks]).astype(np.float32)
    arrays["bin_ms"] = np.asarray(MC.BIN_MS)
    arrays["burn_in_ms"] = np.asarray(run["burn_in_ms"])
    arrays["d_rest_thresh"] = np.asarray(locks["d_rest_thresh"])
    save_npz_atomic(path, **arrays)


def phase_fork(ctx, states_filter=None, arms=ARMS_ORDER, replicates=NB.PAIRED_REPLICATES,
               resume=True, T_ms=T_CONT_MS, dump_traces=False,
               evidence_tier="discovery"):
    seed = ctx["S"]["seed"]
    confirmation_roots = {
        "long_confirmation": os.path.join("confirmations", "long"),
        "dt2_confirmation": os.path.join("confirmations", "dt2"),
    }
    if evidence_tier == "discovery":
        fork_root = "smoke" if ctx["smoke"] else "forks"
    elif evidence_tier in confirmation_roots:
        fork_root = confirmation_roots[evidence_tier]
    else:
        raise ValueError(f"unknown evidence tier {evidence_tier!r}")
    root = os.path.join(OUT, fork_root, f"seed{seed}")
    anchor_path = os.path.join(OUT, ctx["anchor_root"], f"seed{seed}", "anchor.json")

    if evidence_tier != "discovery":
        expected_resolution = "dt" if evidence_tier == "long_confirmation" else "dt2"
        min_horizon = (20_000.0 if evidence_tier == "long_confirmation" else T_CONT_MS)
        central = ["bounded_mid__peak"]
        if ctx["resolution"] != expected_resolution:
            raise SystemExit(
                f"{evidence_tier} requires resolution={expected_resolution}, "
                f"got {ctx['resolution']}"
            )
        if states_filter != central:
            raise SystemExit(
                f"{evidence_tier} is locked to --states {central[0]}"
            )
        if tuple(arms) != ("freeze_all",) or tuple(replicates) != ("noise_replay",):
            raise SystemExit(
                f"{evidence_tier} requires --arms freeze_all "
                "--replicates noise_replay"
            )
        if float(T_ms) < min_horizon:
            raise SystemExit(
                f"{evidence_tier} requires T_cont >= {min_horizon:g} ms"
            )
    if not os.path.exists(anchor_path):
        raise SystemExit(f"no anchor for seed {seed}: run --phase anchor first")
    anchor = json.load(open(anchor_path))
    if not anchor["selection"]["eligibility"]["eligible"]:
        raise SystemExit(f"seed {seed} anchor not eligible: "
                         f"{anchor['selection']['eligibility']['reasons']}")
    if anchor["config_sha"] != ctx["cfg_sha"]:
        raise SystemExit("anchor config_sha != live config_sha")
    locks = anchor["locks"]
    locks["rest_reference"] = {k: v for k, v in locks["rest_reference"].items()}
    man_path = os.path.join(root, "fork_matrix.json")
    done = {}
    if resume and os.path.exists(man_path):
        done = {r["key"]: r for r in json.load(open(man_path)).get("rows", [])}

    rows = dict(done)
    for st in anchor["captured_states"]:
        tag = f"{st['bin_name']}__{st['fast_phase']}"
        if states_filter and tag not in states_filter:
            continue
        state0, sman = CK.load_state_npz(
            os.path.join(_ROOT, st["path"]), expected_config_sha=ctx["cfg_sha"],
            expected_engine_sha=ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"],
            expected_dt=ctx["dt"])
        for arm in arms:
            for rep in replicates:
                key = f"{tag}|{arm}|{rep}"
                existing = rows.get(key)
                # v1 summaries did not carry the mandatory bounded-drift /
                # bounded-variance gate.  A negative v1 continuation remains a
                # valid negative (stationarity cannot rescue a state that
                # returned to rest), but any putative surviving v1 candidate
                # must be rerun under v1.1 before it can count positively.
                needs_stationarity_rerun = MC.needs_stationarity_rerun(existing)
                needs_horizon_rerun = bool(
                    existing is not None
                    and (
                        float(existing.get("T_cont_ms", 0.0)) < float(T_ms)
                        or existing.get("evidence_tier", "discovery") != evidence_tier
                        or existing.get("resolution", "dt") != ctx["resolution"]
                    )
                )
                if (existing is not None and not needs_stationarity_rerun
                        and not needs_horizon_rerun):
                    continue
                bank = NB.build_noise_bank(ctx["cfg_sha"], seed, st["t_step"], rep)
                run = run_continuation(ctx, state0, arm, bank, locks, T_ms=T_ms)
                summ = summarize_continuation(run, locks, T_ms=T_ms)
                if dump_traces:
                    dump_continuation_traces(
                        os.path.join(root, "traces", f"{tag}__{arm}__{rep}.npz"), run, locks)
                rows[key] = dict(key=key, seed=int(seed), bin_name=st["bin_name"],
                                 fast_phase=st["fast_phase"], t_step=int(st["t_step"]),
                                 t_ms=float(st["t_ms"]), arm=arm, replicate=rep,
                                 is_control_arm=arm in CONTROL_ARMS,
                                 freeze_policy=FS.FreezePolicy.for_arm(arm).as_dict(),
                                 state_hash=st["state_hash"], bank_sha=bank["bank_sha"],
                                 config_sha=ctx["cfg_sha"], git_sha=ctx["runtime_git_sha"],
                                 runtime_started_at=ctx["runtime_started_at"],
                                 resolution=ctx["resolution"], evidence_tier=evidence_tier,
                                 dt=ctx["dt"],
                                 metrics_version=MC.METRICS_VERSION, T_cont_ms=float(T_ms),
                                 peak_rss_gb=_rss_gb(), mem_available_gb=_mem_avail_gb(), **summ)
                write_json_atomic(man_path, dict(seed=int(seed), anchor=os.path.relpath(anchor_path, _ROOT),
                                                 locks=locks, rows=sorted(rows.values(),
                                                                          key=lambda r: r["key"])))
                print(f"[fork] {key} survived={summ['survived']} "
                      f"life={summ['lifetime_ms']:.0f}ms end={summ['end_reason']} "
                      f"A={summ['A_active_mean']:.3f} r={summ['r_all_mean_hz']:.1f}Hz "
                      f"resets={summ['rest_returns']} wall={summ['wall_s']}s", flush=True)
    print(f"[fork] seed={seed} rows={len(rows)} -> {man_path}", flush=True)
    return rows


# ================================================================ phase: neighbourhood (Task 8)
NB_ARMS = ("freeze_all", "freeze_zm", "freeze_zsg")
NB_REPLICATES = ("noise_replay", "noise_resample_1")


def _load_state_fields(path, nE, cfg_sha, eng_sha):
    st, _ = CK.load_state_npz(path, expected_config_sha=cfg_sha, expected_engine_sha=eng_sha,
                              expected_dt=DT_BASE)
    return st, dict(z=np.asarray(st["slow.z"], float)[:nE],
                    m=np.asarray(st["slow.m"], float)[:nE],
                    S_G=float(np.asarray(st["slow.S_G"])))


def phase_neighbourhood(ctx, base_state="bounded_mid__peak", arms=NB_ARMS,
                        replicates=NB_REPLICATES, resume=True, T_ms=T_CONT_MS):
    """Displace the SLOW fields of a real visited snapshot inside the locked neighbourhood and ask
    the carrier question again. The fast state stays a naturally occurring microstate throughout."""
    import src.topic4_zm_neighbourhood as NBH
    import src.topic4_zm_anchor_states as AS

    seed = ctx["S"]["seed"]
    nE = int(ctx["S"]["NE"])
    eng_sha = ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"]
    anchor = json.load(open(os.path.join(OUT, "anchors", f"seed{seed}", "anchor.json")))
    locks = anchor["locks"]
    root = os.path.join(OUT, "neighbourhood", f"seed{seed}")
    man_path = os.path.join(root, "neighbourhood.json")

    # ---- representation 1: coarse decision PCA on the anchor's own 7-summary trajectory ----
    tr = np.load(os.path.join(OUT, "anchors", f"seed{seed}", "anchor_traces.npz"))
    sc = {k[5:]: tr[k] for k in tr.files if k.startswith("slow_")}
    Q = AS.slow_feature_matrix(sc)
    Q_std, std_ref = AS.robust_standardize(Q, anchor["selection"]["standardization"])
    coarse = NBH.coarse_representation(Q_std, n_modes=2)
    scale = NBH.trajectory_scale(Q_std)

    # ---- representation 2: full-field PCA over the captured states ----
    states = {f"{s['bin_name']}__{s['fast_phase']}": s for s in anchor["captured_states"]}
    fields, raws = {}, {}
    for tag, s in states.items():
        raws[tag], fields[tag] = _load_state_fields(os.path.join(_ROOT, s["path"]), nE,
                                                    ctx["cfg_sha"], eng_sha)
    order = [t for t in states if t.startswith("bounded")]
    field_pca = NBH.full_field_representation([fields[t] for t in order], n_modes=3)
    X = np.array([np.concatenate([fields[t]["z"], fields[t]["m"], [fields[t]["S_G"]]])
                  for t in order], float)
    scores = (X - field_pca["mean"]) @ field_pca["components"].T
    sd_scores = scores.std(axis=0)

    if base_state not in fields:
        raise SystemExit(f"{base_state} not captured for seed {seed}")
    base_raw = raws[base_state]
    base_vec = np.concatenate([fields[base_state]["z"], fields[base_state]["m"],
                               [fields[base_state]["S_G"]]])
    base_score = (base_vec - field_pca["mean"]) @ field_pca["components"].T

    # ---- the locked displacement families (all are field reconstructions, never scalar edits) ----
    probes = []
    for k in (0, 1):
        for sgn in (-1.0, +1.0):
            c = base_score.copy()
            c[k] += sgn * NBH.MAX_SD * sd_scores[k]
            probes.append(dict(family="field_pca", label=f"pc{k+1}{'+' if sgn > 0 else '-'}",
                               vec=field_pca["mean"] + c @ field_pca["components"]))
    for lam, other in ((0.35, "bounded_early__peak"), (0.65, "bounded_late__peak")):
        if other in fields:
            v = np.concatenate([fields[other]["z"], fields[other]["m"], [fields[other]["S_G"]]])
            probes.append(dict(family="trajectory_interp", label=f"interp_{other}_{lam}",
                               vec=NBH.interpolate_fields(base_vec, v, lam)))
    axis = np.asarray(ctx["axis"], float)
    axis_n = (axis - axis.mean()) / (np.std(axis) + 1e-12)
    for sgn in (-1.0, +1.0):
        v = base_vec.copy()
        amp = NBH.MAX_SD * float(np.std(fields[base_state]["z"]))
        v[:nE] = np.clip(v[:nE] + sgn * amp * axis_n, 0.0, 1.0)
        probes.append(dict(family="pathology_axis", label=f"axial_z{'+' if sgn > 0 else '-'}",
                           vec=v))

    done = {}
    if resume and os.path.exists(man_path):
        done = {r["key"]: r for r in json.load(open(man_path)).get("rows", [])}
    rows = dict(done)
    for pr in probes:
        f = NBH.split_full_field(np.asarray(pr["vec"], float), nE)
        st = {k: (v.copy() if hasattr(v, "copy") else v) for k, v in base_raw.items()}
        st["slow.z"] = np.asarray(st["slow.z"], float).copy()
        st["slow.m"] = np.asarray(st["slow.m"], float).copy()
        st["slow.z"][:nE] = f["z"]
        st["slow.m"][:nE] = f["m"]
        st["slow.S_G"] = np.asarray(float(f["S_G"]))
        coarse_q = dict(z_core=float(f["z"][ctx["core"]].mean()),
                        z_surround=float(f["z"][~ctx["core"]].mean()),
                        m_core=float(f["m"][ctx["core"]].mean()),
                        m_surround=float(f["m"][~ctx["core"]].mean()), S_G=float(f["S_G"]))
        axis_proj = NBH.pathology_axis_projection(f["z"], f["m"], ctx["axis"], ctx["core"])
        for arm in arms:
            for rep in replicates:
                key = f"{pr['family']}|{pr['label']}|{arm}|{rep}"
                if key in rows:
                    continue
                bank = NB.build_noise_bank(ctx["cfg_sha"], seed,
                                           int(np.asarray(base_raw["t"])), rep)
                run = run_continuation(ctx, st, arm, bank, locks, T_ms=T_ms)
                summ = summarize_continuation(run, locks, T_ms=T_ms)
                rows[key] = dict(key=key, seed=int(seed), family=pr["family"], label=pr["label"],
                                 base_state=base_state, arm=arm, replicate=rep,
                                 coarse_q=coarse_q, axis_projection=axis_proj,
                                 config_sha=ctx["cfg_sha"], git_sha=ctx["runtime_git_sha"],
                                 runtime_started_at=ctx["runtime_started_at"],
                                 bank_sha=bank["bank_sha"], T_cont_ms=float(T_ms),
                                 neighbourhood_version=NBH.NEIGHBOURHOOD_VERSION,
                                 peak_rss_gb=_rss_gb(), **summ)
                write_json_atomic(man_path, dict(
                    seed=int(seed), base_state=base_state, max_sd=NBH.MAX_SD,
                    coarse_pca=dict(explained_variance_ratio=coarse["explained_variance_ratio"],
                                    trajectory_scale=scale),
                    field_pca=dict(explained_variance_ratio=field_pca["explained_variance_ratio"],
                                   n_samples=field_pca["n_samples"], dim=field_pca["dim"],
                                   score_sd=sd_scores.tolist()),
                    standardization=std_ref, arms=list(arms), replicates=list(replicates),
                    audit=dict(
                        complete=False,
                        representations_agree=False,
                        local_carrier_window=False,
                        formal_local_negative=False,
                        family_results={},
                        reason=(
                            "exploratory scaffold only: this run has not yet "
                            "closed all onset/early/mid/late anchors, two fast "
                            "phases, four scientific arms and three paired-noise "
                            "replicates required for a formal Branch T/F verdict"
                        ),
                    ),
                    rows=sorted(rows.values(), key=lambda r: r["key"])))
                print(f"[nbh] {key} survived={summ['survived']} life={summ['lifetime_ms']:.0f}ms "
                      f"end={summ['end_reason']} A={summ['A_active_mean']:.3f} "
                      f"resets={summ['rest_returns']} wall={summ['wall_s']}s", flush=True)
    print(f"[nbh] seed={seed} rows={len(rows)} -> {man_path}", flush=True)
    return rows


# ================================================================ phase: vertical slice (Task 3.5)
def phase_smoke(ctx, T_anchor_ms=2000.0, T_cont_ms=500.0):
    """anchor -> checkpoint -> restore -> freeze_all -> continuation -> source + current-vSEEG
    metrics, with hash round-trip and observer continuity. Writes ONLY under smoke/."""
    seed = ctx["S"]["seed"]
    base = os.path.join(OUT, "smoke", "vertical_slice")
    t0 = time.time()
    dt = ctx["dt"]
    t_fork = int(round(T_anchor_ms / dt)) - int(round(500.0 / dt))
    slow = make_slow(ctx)
    ck = CK.ZMCheckpoint(snapshot_steps=[t_fork])
    res = run_segment(ctx, slow, T_anchor_ms, ckpt=ck)
    met = segment_metrics(ctx, res)
    ref = MC.rest_reference(met, 0, max(2, met["n_bins"] // 3))
    d = MC.rest_distance(met, ref)
    locks = dict(rest_reference=ref, d_rest_thresh=float(np.quantile(d, 0.95)),
                 rest_dwell_ms=REST_DWELL_MS, ied_lifetime_ms=50.0)
    state = ck.snapshots[t_fork]
    path = os.path.join(base, f"state_seed{seed}_t{t_fork}.npz")
    sman = CK.save_state_npz(state, dict(
        config_sha=ctx["cfg_sha"], dt=dt, seed=int(seed), t_step=int(t_fork),
        engine_sha=ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"]), path)
    back, bman = CK.load_state_npz(path, expected_config_sha=ctx["cfg_sha"], expected_dt=dt)
    assert bman["state_hash"] == sman["state_hash"], "state hash did not round-trip"

    bank = NB.build_noise_bank(ctx["cfg_sha"], seed, t_fork, "noise_replay")
    run = run_continuation(ctx, back, "freeze_all", bank, locks, T_ms=T_cont_ms, chunk_ms=T_cont_ms)
    summ = summarize_continuation(run, locks, T_ms=T_cont_ms)
    # observer continuity: the current-based vSEEG has no filter state (memoryless weighted sum of
    # |I_E|+|I_I|), so continuity is structural -- assert the continuation actually produced it.
    cont_met = run["chunks"][0]
    assert cont_met["E_vSEEG"].size > 0 and np.all(np.isfinite(cont_met["E_vSEEG"]))
    out = dict(provenance(ctx, phase="vertical_slice", T_anchor_ms=T_anchor_ms,
                          T_cont_ms=T_cont_ms, t_fork_step=int(t_fork)),
               state_hash=sman["state_hash"], state_path=os.path.relpath(path, _ROOT),
               bank_sha=bank["bank_sha"], locks={k: v for k, v in locks.items()
                                                 if k != "rest_reference"},
               continuation=summ, metric_keys=sorted(k for k in cont_met if not k.startswith("_")),
               wall_s=round(time.time() - t0, 1), evidence_value="none (smoke namespace)")
    write_json_atomic(os.path.join(base, f"vertical_slice_seed{seed}.json"), out)
    print(f"[smoke] vertical slice ok: state_hash={sman['state_hash'][:16]} "
          f"survived={summ['survived']} life={summ['lifetime_ms']:.0f}ms "
          f"metrics={len(out['metric_keys'])} wall={out['wall_s']}s", flush=True)
    return out


# ================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["anchor1", "anchor", "fork", "neighbourhood", "smoke"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--T", type=float, default=15000.0)
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="write under smoke/ (no evidence value)")
    ap.add_argument("--states", default=None, help="comma list of bin__phase tags (default: all)")
    ap.add_argument("--arms", default=",".join(ARMS_ORDER))
    ap.add_argument("--replicates", default=",".join(NB.PAIRED_REPLICATES))
    ap.add_argument("--resolution", choices=sorted(RESOLUTION_DT), default="dt",
                    help="dt2 always builds an independent native anchor; snapshots are never converted")
    ap.add_argument("--evidence-tier",
                    choices=("discovery", "long_confirmation", "dt2_confirmation"),
                    default="discovery")
    ap.add_argument("--T-cont", dest="T_cont", type=float, default=T_CONT_MS)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dump-traces", action="store_true",
                    help="also save per-bin continuation series (figure input)")
    a = ap.parse_args()
    if a.phase != "smoke" and not a.confirm_run:
        raise SystemExit("refusing a multi-minute N=40000 run without --confirm-run")
    if a.phase == "neighbourhood" and a.resolution != "dt":
        raise SystemExit("neighbourhood is a discovery-dt phase; dt2 is confirmation-only")
    if a.phase != "fork" and a.evidence_tier != "discovery":
        raise SystemExit("--evidence-tier applies only to --phase fork")
    ctx = build_context(
        a.seed, smoke=a.smoke or a.phase == "smoke", resolution=a.resolution
    )
    if a.phase == "anchor1":
        phase_anchor1(ctx, a.T)
    elif a.phase == "anchor":
        phase_anchor(ctx, a.T)
    elif a.phase == "smoke":
        phase_smoke(ctx)
    elif a.phase == "neighbourhood":
        phase_neighbourhood(ctx, base_state=(a.states.split(",")[0] if a.states
                                             else "bounded_mid__peak"),
                            arms=tuple(x.strip() for x in a.arms.split(",") if x.strip())
                            if a.arms != ",".join(ARMS_ORDER) else NB_ARMS,
                            replicates=tuple(x.strip() for x in a.replicates.split(",") if x.strip())
                            if a.replicates != ",".join(NB.PAIRED_REPLICATES) else NB_REPLICATES,
                            resume=not a.no_resume, T_ms=a.T_cont)
    elif a.phase == "fork":
        phase_fork(ctx,
                   states_filter=[s.strip() for s in a.states.split(",")] if a.states else None,
                   arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()),
                   replicates=tuple(x.strip() for x in a.replicates.split(",") if x.strip()),
                   resume=not a.no_resume, T_ms=a.T_cont, dump_traces=a.dump_traces,
                   evidence_tier=a.evidence_tier)


if __name__ == "__main__":
    main()
