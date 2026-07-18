"""Topic 4 state-conditioned spatial susceptibility — gated runner (design 2026-07-19).

*** SIMULATION-BEARING SUBCOMMANDS ARE GATED BY --confirm-run. *** Nothing runs on import.

Question (design §0): as inhibitory efficacy z_i evolves along one FIXED, pre-computed MZ trajectory
(candidate zA_q50_tz10000, seeds 1/3/4), does the finite-time spatial susceptibility of the fixed E1146
scaffold change in a way that preserves/strengthens the interictal propagation axis before runoff? No
result direction is a success gate.

REUSE, DO NOT REINVENT (design §1.2): the MZ substrate + detection + snapshot observer already exist.
This runner imports `run_topic4_mz_slowvars` (PP.build_substrate / M4 runaway criterion) and the
snapshot-enabled `mz_slow_vars.MZSlowVars`; it EDITS NONE of the 6 guarded engine files and does not
edit run_topic4_mz_slowvars.py. Pure mapping/probe/operator math lives in the import-safe module
`src/topic4_state_conditioned_susceptibility.py` (no simulations, no file writes there).

Subcommands:
  audit-inputs             verify artifact paths + engine-SHA parity, write snapshot_contract.json (no sim)
  smoke                    seed-1 short observer-vs-no-observer PARITY proof + wall/RSS (needs --confirm-run)
  capture-snapshots        replay candidate seeds, snapshot z_E/m_E at 5 states -> snapshots/ (needs --confirm-run)
  build-atlas              (Task 5) coarse-field -> operator -> probe atlas (needs --confirm-run)
  run-controls             (Task 5) real/uniform/rotate/shuffle/z-blocked controls (needs --confirm-run)
  run-nonlinear-spotchecks (Task 6) two-amplitude linear-regime check (needs --confirm-run)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")        # parallel numpy MUST be single-threaded (OOM + determinism)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse            # noqa: E402
import dataclasses         # noqa: E402
import hashlib             # noqa: E402
import json                # noqa: E402
import multiprocessing as mp  # noqa: E402
import resource            # noqa: E402
import subprocess          # noqa: E402
import sys                 # noqa: E402
import time                # noqa: E402

import numpy as np         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_topic4_mz_slowvars as MZR                     # noqa: E402  (build_substrate / build_core_masks / M4 / PP / DT)
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig    # noqa: E402
from kick_probe import simulate_kick                     # noqa: E402

DT = 0.1
OUT_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "state_conditioned_susceptibility")
MZ_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slowvars")
SCHEMA_VERSION = "sccs-1.0"

# Five pre-declared snapshot states (design §4.2). Times are per-seed except the fixed 1000 ms baseline.
SNAP_STATES = ["baseline_1000ms", "mid_fraction", "pre_onset_500ms", "pre_onset_100ms", "onset"]

# capture runs a hair past onset (early_stop OFF) so the onset step is always reached and the runaway
# is confirmed to persist; trajectory <= onset is byte-identical to the locked early_stop replay.
CAPTURE_TAIL_MS = 400.0

SNAP_CONVENTION = (
    "captured inside MZSlowVars.step() AFTER the z/m Euler update and the streaming trace record; "
    "step counter is 0-based, one increment per simulate_kick iteration t; time_ms = step*dt; "
    "consistency: snapshot.z_E.mean() == mz.trace_z_mean[step]."
)


# ============================================================ provenance
def _git_sha():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def _engine_shas():
    return MZR._engine_shas()


def _locked_engine_shas():
    """Engine SHAs recorded in the MZ multiseed provenance (the run whose onsets we replay)."""
    ms = json.load(open(os.path.join(MZ_DIR, "per_seed", "multiseed_summary.json")))
    return ms["provenance"]["engine_shas"]


# ============================================================ locked-candidate loader (no hard-coded numbers)
def _load_locked_candidate(label):
    """Read the committed MZ multiseed summary -> per-seed onset + cfg for `label`. No refit."""
    ms_path = os.path.join(MZ_DIR, "per_seed", "multiseed_summary.json")
    ms = json.load(open(ms_path))
    rows = [r for r in ms["rows"] if r["label"] == label]
    if not rows:
        raise ValueError(f"candidate {label!r} not found in {ms_path}")
    onsets, phenos, cfgs = {}, {}, []
    for r in rows:
        onsets[int(r["seed"])] = r["runaway_ms"]
        phenos[int(r["seed"])] = r["phenotype"]
        cfgs.append(tuple(sorted(r["cfg"].items())))
    if len(set(cfgs)) != 1:
        raise ValueError(f"candidate {label!r} has inconsistent cfg across seeds: {set(cfgs)}")
    cfg = dict(cfgs[0])
    return dict(label=label, cfg=cfg, onsets=onsets, phenotypes=phenos,
                source_artifact_paths=[os.path.relpath(ms_path, ROOT),
                                       os.path.relpath(os.path.join(MZ_DIR, "p3_candidates.json"), ROOT),
                                       os.path.relpath(os.path.join(MZ_DIR, "calibration.json"), ROOT)])


def _snapshot_times(onset):
    return {
        "baseline_1000ms": 1000.0,
        "mid_fraction": 0.5 * float(onset),
        "pre_onset_500ms": float(onset) - 500.0,
        "pre_onset_100ms": float(onset) - 100.0,
        "onset": float(onset),
    }


def _snapshot_steps(onset, dt):
    """(times, lab->step, step->lab). Raises on step collision (design: distinct captured states)."""
    times = _snapshot_times(onset)
    lab2step = {lab: int(round(times[lab] / dt)) for lab in SNAP_STATES}
    if len(set(lab2step.values())) != len(lab2step):
        raise ValueError(f"snapshot step collision for onset={onset}: {lab2step}")
    step2lab = {v: k for k, v in lab2step.items()}
    return times, lab2step, step2lab


# ============================================================ snapshot-enabled cell run (mirrors MZR.run_mz_cell)
def run_mz_cell_with_snapshots(S, cfg, T, step2lab, *, early_stop=False):
    """Byte-identical to MZR.run_mz_cell except (a) attaches the off-by-default snapshot observer,
    (b) no LFP recorder. The observer only COPIES z_E/m_E -> res is unaffected (proven in `smoke`)."""
    p = dataclasses.replace(S["p"], T=float(T))
    core_mask_E = MZR.build_core_masks(S)
    mz = MZSlowVars(S["N"], 18.0, cfg, NE=S["NE"], core_mask_E=core_mask_E, snapshot_steps=step2lab)
    S["net"]["rng"] = np.random.default_rng(S["seed"])                 # identical noise realization
    res = simulate_kick(p, S["net"], 0.0, slow=mz, kick_center=list(S["src_xy"]), r_kick=MZR.PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], early_stop_runaway=early_stop)
    return res, mz


def _runaway_ms(res):
    """Exact reuse of the MZ runaway criterion (M4._smooth + _first_sustained, early-stop fallback)."""
    rate = np.asarray(res["rate_E"], float)
    rm = MZR.M4._first_sustained(MZR.M4._smooth(rate, DT), DT)
    es = res.get("runaway_early_stop_ms")
    if es is not None and rm is None:
        rm = es
    return rm


# ============================================================ audit-inputs (no simulation)
def cmd_audit_inputs(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    label = args.candidate or "zA_q50_tz10000"
    cand = _load_locked_candidate(label)
    cur, locked = _engine_shas(), _locked_engine_shas()
    engine_ok = all(cur.get(k) == locked.get(k) for k in locked)
    seeds = sorted(cand["onsets"])
    per_seed = {}
    for s in seeds:
        onset = cand["onsets"][s]
        times, lab2step, _ = _snapshot_steps(onset, DT)
        per_seed[str(s)] = dict(onset_ms=onset, phenotype=cand["phenotypes"][s],
                                snapshot_times_ms=times, snapshot_steps=lab2step,
                                capture_T_ms=float(onset) + CAPTURE_TAIL_MS)
    contract = dict(
        schema_version=SCHEMA_VERSION, candidate=label, cfg=cand["cfg"],
        seeds=seeds, dt_ms=DT, subject=MZR.PP.SUBJECT, montage=MZR.PP.MONTAGE,
        snap_states=SNAP_STATES, snapshot_update_convention=SNAP_CONVENTION,
        capture_tail_ms=CAPTURE_TAIL_MS, early_stop_in_capture=False,
        per_seed=per_seed, source_artifact_paths=cand["source_artifact_paths"],
        engine_sha_parity=dict(ok=bool(engine_ok), current=cur, locked=locked),
        git_sha=_git_sha(), argv=sys.argv,
    )
    path = os.path.join(OUT_DIR, "snapshot_contract.json")
    json.dump(contract, open(path, "w"), indent=2)
    print(f"[audit-inputs] candidate={label} cfg={cand['cfg']}")
    for s in seeds:
        print(f"  seed {s}: onset={cand['onsets'][s]} ms phenotype={cand['phenotypes'][s]} "
              f"steps={per_seed[str(s)]['snapshot_steps']}")
    print(f"[audit-inputs] engine_sha_parity_ok={engine_ok}  -> {os.path.relpath(path, ROOT)}")
    if not engine_ok:
        print("*** ENGINE SHA DRIFT vs locked MZ run: replay would NOT reproduce locked onsets. STOP. ***",
              file=sys.stderr)
        sys.exit(3)
    return contract


# ============================================================ capture-snapshots
def _build_snapshot_arrays(mz, S, lab2step, times):
    """Stack captured E-cell z/m into [state, NE] + fixed per-seed geometry (design §4.4 NPZ schema)."""
    labels = [lab for lab in SNAP_STATES if lab in mz.snapshots]
    posE = np.asarray(S["posE"], np.float32)
    src = np.asarray(S["src_xy"], float); snk = np.asarray(S["snk_xy"], float)
    core_r = float(MZR.PP.CORE_R)
    src_core = (np.linalg.norm(posE - src, axis=1) <= core_r)
    snk_core = (np.linalg.norm(posE - snk, axis=1) <= core_r)
    arr = dict(
        snapshot_labels=np.array(labels, dtype=object),
        requested_time_ms=np.array([times[l] for l in labels], np.float64),
        actual_time_ms=np.array([mz.snapshots[l]["step"] * DT for l in labels], np.float64),
        snapshot_step=np.array([mz.snapshots[l]["step"] for l in labels], np.int64),
        z_E=np.stack([mz.snapshots[l]["z_E"] for l in labels]).astype(np.float32),
        m_E=np.stack([mz.snapshots[l]["m_E"] for l in labels]).astype(np.float32),
        pos_E=posE,
        core_mask_E=(src_core | snk_core), src_core_mask_E=src_core, snk_core_mask_E=snk_core,
        vth_E=np.asarray(S["vth"][:S["NE"]], np.float32),
        src_xy=np.asarray(S["src_xy"], np.float32), snk_xy=np.asarray(S["snk_xy"], np.float32),
        axis_unit=np.asarray(S["axis_unit"], np.float32), L=float(S["L"]), core_r=core_r,
    )
    return labels, arr


def _capture_one_seed(task):
    """Worker: replay one seed with the snapshot observer, gate, and persist NPZ+JSON. Returns summary."""
    candidate, cfg_dict, seed, onset, phenotype_locked, source_paths = task
    t0 = time.time()
    S = MZR.PP.build_substrate(seed)
    times, lab2step, step2lab = _snapshot_steps(onset, DT)
    T = float(onset) + CAPTURE_TAIL_MS
    cfg = MZSlowVarsConfig(**cfg_dict)
    res, mz = run_mz_cell_with_snapshots(S, cfg, T, step2lab, early_stop=False)
    replay_runaway = _runaway_ms(res)
    captured = set(mz.snapshots)
    missing = {lab: times[lab] for lab in SNAP_STATES if lab not in captured}
    onset_match = (replay_runaway is not None) and abs(replay_runaway - onset) <= 5.0
    # E-only invariant audit (I cells never modulated): z==1, m==0 on [NE:]
    i_pinned = bool(np.all(mz.z[S["NE"]:] == 1.0) and np.all(mz.m[S["NE"]:] == 0.0))
    z_bounds_ok = bool(all(np.all((mz.snapshots[l]["z_E"] >= 0.0) & (mz.snapshots[l]["z_E"] <= 1.0))
                           for l in captured))
    m_zero = bool(all(np.all(mz.snapshots[l]["m_E"] == 0.0) for l in captured)) if not cfg.use_m else None
    gate_pass = bool(onset_match and not missing and i_pinned and z_bounds_ok
                     and (m_zero if not cfg.use_m else True))

    labels, arr = _build_snapshot_arrays(mz, S, lab2step, times)
    cand_dir = os.path.join(OUT_DIR, "snapshots", candidate)
    os.makedirs(cand_dir, exist_ok=True)
    npz_path = os.path.join(cand_dir, f"seed_{seed}.npz")
    _atomic_savez(npz_path, **arr)
    meta = dict(
        schema_version=SCHEMA_VERSION, candidate=candidate, seed=seed,
        phenotype=phenotype_locked, replay_phenotype=("runaway" if onset_match else "MISMATCH"),
        locked_runaway_ms=onset, replay_runaway_ms=replay_runaway, onset_match=onset_match,
        dt_ms=DT, L=float(S["L"]), subject=MZR.PP.SUBJECT, montage=MZR.PP.MONTAGE, NE=int(S["NE"]), N=int(S["N"]),
        core_r=float(MZR.PP.CORE_R), config=cfg_dict, capture_T_ms=T, early_stop=False,
        n_steps_run=int(mz.n_steps_run), snapshot_labels=labels,
        snapshot_steps={l: int(lab2step[l]) for l in SNAP_STATES},
        requested_time_ms={l: times[l] for l in SNAP_STATES},
        actual_time_ms={l: mz.snapshots[l]["step"] * DT for l in captured},
        missing_snapshots=missing, snapshot_update_convention=SNAP_CONVENTION,
        gate=dict(onset_match=onset_match, all_states_captured=(not missing), i_cells_pinned=i_pinned,
                  z_in_bounds=z_bounds_ok, m_zero_for_zonly=m_zero, pass_=gate_pass),
        source_artifact_paths=source_paths, guarded_engine_sha256=_engine_shas(), git_sha=_git_sha(),
        argv=sys.argv, wall_s=round(time.time() - t0, 1),
    )
    json.dump(meta, open(os.path.join(cand_dir, f"seed_{seed}.json"), "w"), indent=2, default=_json_default)
    return dict(seed=seed, onset=onset, replay_runaway_ms=replay_runaway, onset_match=onset_match,
                missing=list(missing), gate_pass=gate_pass, wall_s=meta["wall_s"],
                src_xy=[float(x) for x in S["src_xy"]], snk_xy=[float(x) for x in S["snk_xy"]],
                axis_unit=[float(x) for x in S["axis_unit"]], L=float(S["L"]))


def cmd_capture_snapshots(args):
    label = args.candidate or "zA_q50_tz10000"
    cand = _load_locked_candidate(label)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(cand["onsets"])
    workers = min(int(args.workers) if args.workers else 2, 2)
    tasks = [(label, cand["cfg"], s, cand["onsets"][s], cand["phenotypes"][s], cand["source_artifact_paths"])
             for s in seeds]
    print(f"[capture] candidate={label} seeds={seeds} workers={workers} cfg={cand['cfg']}", flush=True)
    if workers > 1 and len(tasks) > 1:
        with mp.Pool(min(workers, len(tasks))) as pool:
            rows = pool.map(_capture_one_seed, tasks)
    else:
        rows = [_capture_one_seed(t) for t in tasks]
    for r in rows:
        print(f"  seed {r['seed']}: replay_runaway={r['replay_runaway_ms']} (locked {r['onset']}) "
              f"onset_match={r['onset_match']} missing={r['missing']} gate_pass={r['gate_pass']} "
              f"wall={r['wall_s']}s", flush=True)
    # geometry-consistency across seeds (src/snk/axis/L identical; only pos_E differs per seed)
    geom = {k: [tuple(np.round(r[k], 4)) if isinstance(r[k], list) else round(r[k], 4) for r in rows]
            for k in ("src_xy", "snk_xy", "axis_unit", "L")}
    geom_consistent = all(len(set(v)) == 1 for v in geom.values())
    summary = dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds,
                   geometry_consistent_across_seeds=bool(geom_consistent),
                   all_gates_pass=bool(all(r["gate_pass"] for r in rows)), rows=rows,
                   git_sha=_git_sha(), argv=sys.argv)
    _atomic_json(os.path.join(OUT_DIR, "snapshots", label, "capture_summary.json"), summary)
    print(f"[capture] all_gates_pass={summary['all_gates_pass']} geom_consistent={geom_consistent}", flush=True)
    return summary


# ============================================================ smoke: observer-parity proof + cost
def cmd_smoke(args):
    label = args.candidate or "zA_q50_tz10000"
    cand = _load_locked_candidate(label)
    seed = int(args.seed) if args.seed else 1
    T = float(args.T) if args.T else 1500.0                       # short, past the 1000 ms baseline
    S = MZR.PP.build_substrate(seed)
    cfg = MZSlowVarsConfig(**cand["cfg"])
    step2lab = {int(round(1000.0 / DT)): "baseline_1000ms", int(round((T - 100.0) / DT)): "smoke_end"}
    t0 = time.time()
    res_on, mz = run_mz_cell_with_snapshots(S, cfg, T, step2lab, early_stop=False)
    wall_on = time.time() - t0
    # same run WITHOUT observer (MZR.run_mz_cell path) -> must be byte-identical
    res_off, _ = MZR.run_mz_cell(S, cfg, T, early_stop=False)
    ok = (np.array_equal(res_on["rate_E"], res_off["rate_E"])
          and np.array_equal(res_on["E_spk_bool"], res_off["E_spk_bool"])
          and np.array_equal(res_on["rate_I"], res_off["rate_I"]))
    trace_ok = all(abs(mz.snapshots[l]["z_E"].mean() - mz.trace_z_mean[mz.snapshots[l]["step"]]) < 1e-12
                   for l in mz.snapshots)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    nsteps = len(res_on["rate_E"])
    onset = cand["onsets"][seed]
    est_full = wall_on / max(1, nsteps) * ((onset + CAPTURE_TAIL_MS) / DT)
    print(f"[smoke] seed={seed} T={T}ms nsteps={nsteps} observer_parity={ok} trace_index_pin={trace_ok} "
          f"peak_RSS={rss_gb:.2f}GB wall_on={wall_on:.1f}s", flush=True)
    print(f"[smoke] est full capture (T={onset + CAPTURE_TAIL_MS:.0f}ms) ~= {est_full:.0f}s/seed", flush=True)
    if not (ok and trace_ok):
        print("*** SMOKE FAIL: observer perturbed the trajectory or the step index is misaligned. STOP. ***",
              file=sys.stderr)
        sys.exit(4)
    return dict(observer_parity=ok, trace_index_pin=trace_ok, peak_rss_gb=round(rss_gb, 2),
                wall_on_s=round(wall_on, 1), est_full_s=round(est_full, 0))


# ============================================================ atlas / controls (Task 5, design §6/§7)
def _load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "config", "topic4_state_conditioned_susceptibility.yaml")))


def _load_snapshot(candidate, seed):
    p = os.path.join(OUT_DIR, "snapshots", candidate, f"seed_{seed}.npz")
    d = np.load(p, allow_pickle=True)
    return {k: d[k] for k in d.files}


def _seed_context(cfg, snap, grid_n, *, ar=None, center_which=None):
    """Build the (fixed) scaffold + probes + normalized positions for one seed at a given grid/ar."""
    import src.topic4_state_conditioned_susceptibility as M
    from src.topic4_m3b_spectral_phase import Grid
    grid = Grid(n=int(grid_n), L=float(cfg["L_norm"]))
    kw = dict(L_phys=cfg["L_phys"], L_norm=cfg["L_norm"], center_phys=cfg["center_phys"])
    src_norm = M.affine_to_norm(snap["src_xy"], **kw)
    snk_norm = M.affine_to_norm(snap["snk_xy"], **kw)
    au = np.asarray(snap["axis_unit"], float)
    theta = float(np.arctan2(au[1], au[0]))
    scaffold = M.build_fixed_scaffold(grid, src_norm, snk_norm, ell_perp=cfg["ell_perp"],
                                      ar=(cfg["ar"] if ar is None else ar), mu_core=cfg["mu_core"],
                                      core_radius=cfg["core_radius_norm"], theta=theta)
    which = center_which or cfg["probe_center"]
    center = src_norm if which == "source" else snk_norm
    probes = M.make_phase_paired_probe_dictionary(grid, p_max=cfg["p_max"], sigma=cfg["gabor_sigma"],
                                                  center=tuple(center), gabor=bool(cfg["gabor"]))
    pos_norm = M.affine_to_norm(snap["pos_E"], **kw)
    return dict(grid=grid, scaffold=scaffold, probes=probes, pos_norm=pos_norm, theta=theta,
                src_norm=src_norm, snk_norm=snk_norm)


def _summarize_field(M, zbar, ctx, cfg):
    out, arrays = M.summarize_state_susceptibility(
        zbar, ctx["grid"], ctx["scaffold"], ctx["probes"], cfg["T_windows"],
        w_ee_mult=cfg["w_ee_mult"], ratio=cfg["ratio"], q_floor=cfg["q_floor"],
        T_primary=cfg["T_primary"], op_dt=cfg["op_dt"], op_t_max=cfg["op_t_max"], op_tol=cfg["op_tol"])
    return out, arrays


def _delta_and_median(per_seed_state, metric_path, start="baseline_1000ms", end="pre_onset_100ms"):
    """Within-seed end-minus-start for a resolved metric, plus the cross-seed median (design §6.2).
    Primary endpoint is the last RESOLVED state (pre_onset_100ms); onset is the unresolved boundary."""
    deltas = {}
    for seed, states in per_seed_state.items():
        b, o = states.get(start), states.get(end)
        vb = _dig(b, metric_path); vo = _dig(o, metric_path)
        if vb is not None and vo is not None:
            deltas[seed] = float(vo - vb)
    med = float(np.median(list(deltas.values()))) if deltas else None
    return dict(per_seed=deltas, median=med, n=len(deltas), start=start, end=end)


def _dig(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None


def cmd_build_atlas(args):
    import src.topic4_state_conditioned_susceptibility as M
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else _load_locked_candidate(label)["onsets"].keys()
    seeds = sorted(seeds)
    per_seed, arrays_out, audit = {}, {}, {}
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        ctx = _seed_context(cfg, snap, cfg["grid_n"])
        labels = [str(x) for x in snap["snapshot_labels"]]
        per_state = {}
        for i, lab in enumerate(labels):
            zbar, occ, fill = M.bin_neuron_state_to_grid(snap["z_E"][i], ctx["pos_norm"], ctx["grid"])
            out, arr = _summarize_field(M, zbar, ctx, cfg)
            out["occupancy_min"] = int(occ.min()); out["fill_fraction"] = float(fill.mean())
            per_state[lab] = out
            if arr is not None:
                for k, v in arr.items():
                    arrays_out[f"{seed}__{lab}__{k}"] = np.asarray(v)
        per_seed[str(seed)] = per_state
        audit[str(seed)] = {lab: dict(op_status=per_state[lab]["op_status"],
                                      op_residual=per_state[lab]["op_residual"],
                                      eig=(per_state[lab].get("eigen") or {}).get("eig_residual_ok"),
                                      occupancy_min=per_state[lab]["occupancy_min"],
                                      fill_fraction=per_state[lab]["fill_fraction"]) for lab in per_state}
        print(f"[build-atlas] seed {seed}: " + ", ".join(f"{l}={per_state[l]['op_status']}" for l in labels), flush=True)
    # primary estimand: within-seed deltas + cross-seed median (design §6.2). Endpoint = last RESOLVED
    # state (pre_onset_100ms); baseline->onset is also reported but onset is the unresolved boundary.
    est_keys = ("axial_gain", "perp_gain", "global_gain", "peak_gain", "axis_minus_perp", "peak_k")
    Tp = cfg["T_primary"]
    estimand = dict(
        baseline_to_pre_onset_100ms={m: _delta_and_median(per_seed, ["atlas", "per_T", Tp, m],
                                                          "baseline_1000ms", "pre_onset_100ms") for m in est_keys},
        baseline_to_onset={m: _delta_and_median(per_seed, ["atlas", "per_T", Tp, m],
                                                "baseline_1000ms", "onset") for m in est_keys})
    atlas = dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds, config=cfg,
                 grid_n=cfg["grid_n"], T_windows=cfg["T_windows"], T_primary=cfg["T_primary"],
                 within_seed_deltas=estimand, per_seed=per_seed,
                 git_sha=_git_sha(), engine_shas=_engine_shas(), argv=sys.argv)
    _atomic_json(os.path.join(OUT_DIR, "susceptibility_atlas.json"), atlas)
    _atomic_savez(os.path.join(OUT_DIR, "susceptibility_arrays.npz"), **arrays_out)
    _atomic_json(os.path.join(OUT_DIR, "numerical_audit.json"),
                 dict(schema_version=SCHEMA_VERSION, candidate=label, per_seed=audit,
                      batch_vs_single=_batch_single_audit(M, cfg, label, seeds[0]), git_sha=_git_sha()))
    print(f"[build-atlas] -> susceptibility_atlas.json ; median axial dGain(baseline->pre_onset_100ms)="
          f"{estimand['baseline_to_pre_onset_100ms']['axial_gain']['median']}", flush=True)
    return atlas


def _batch_single_audit(M, cfg, label, seed):
    """Gate D6 spot check on the real baseline J: batched response == one-at-a-time."""
    snap = _load_snapshot(label, seed)
    ctx = _seed_context(cfg, snap, cfg["grid_n"])
    i0 = [str(x) for x in snap["snapshot_labels"]].index("baseline_1000ms")
    zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][i0], ctx["pos_norm"], ctx["grid"])
    op, J, q = M.state_operator(zbar, ctx["grid"], ctx["scaffold"], w_ee_mult=cfg["w_ee_mult"],
                                ratio=cfg["ratio"], q_floor=cfg["q_floor"])
    if J is None:
        return dict(status=op.status, max_abs_diff=None)
    B = M.probe_matrix(ctx["probes"][:6], ctx["grid"])
    Yb = M.batched_finite_time_response(J, B, cfg["T_primary"])
    Ys = np.column_stack([M.batched_finite_time_response(J, B[:, i:i + 1], cfg["T_primary"])[:, 0]
                          for i in range(B.shape[1])])
    return dict(status=op.status, max_abs_diff=float(np.max(np.abs(Yb - Ys))))


def cmd_run_controls(args):
    import src.topic4_state_conditioned_susceptibility as M
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(_load_locked_candidate(label)["onsets"])
    control_names = ["uniform_mean", "rotated_90", "spatial_shuffle", "z_blocked"]   # 'real' lives in the atlas
    per_seed, arrays_out = {}, {}
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        ctx = _seed_context(cfg, snap, cfg["grid_n"])
        labels = [str(x) for x in snap["snapshot_labels"]]
        per_state = {}
        for i, lab in enumerate(labels):
            zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][i], ctx["pos_norm"], ctx["grid"])
            variants = M.make_state_controls(zbar, ctx["grid"], shuffle_seed=cfg["shuffle_seed"])
            per_ctrl = {}
            for cn in control_names:
                out, arr = _summarize_field(M, variants[cn], ctx, cfg)
                per_ctrl[cn] = out
                if arr is not None:
                    arrays_out[f"{seed}__{lab}__{cn}__q_field"] = np.asarray(arr["q_field"])
            per_state[lab] = per_ctrl
        per_seed[str(seed)] = per_state
        print(f"[run-controls] seed {seed}: {len(labels)} states x {len(control_names)} controls done", flush=True)
    # AR1 isotropic control at baseline+onset (design §7): does axial preference vanish?
    ar1 = {}
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        ctx = _seed_context(cfg, snap, cfg["grid_n"], ar=1.0)
        labels = [str(x) for x in snap["snapshot_labels"]]
        ar1[str(seed)] = {}
        for lab in ("baseline_1000ms", "pre_onset_100ms", "onset"):
            if lab in labels:
                zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][labels.index(lab)], ctx["pos_norm"], ctx["grid"])
                out, _ = _summarize_field(M, zbar, ctx, cfg)
                ar1[str(seed)][lab] = out
    # resolution sensitivity n=8 vs n=12 at baseline+onset on the primary seed (design §7)
    res_seed = seeds[0]
    resolution = {}
    snap = _load_snapshot(label, res_seed)
    for gn in (cfg["grid_n_sensitivity"], cfg["grid_n"]):
        ctx = _seed_context(cfg, snap, gn)
        labels = [str(x) for x in snap["snapshot_labels"]]
        resolution[f"n{gn}"] = {}
        for lab in ("baseline_1000ms", "pre_onset_100ms", "onset"):
            if lab in labels:
                zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][labels.index(lab)], ctx["pos_norm"], ctx["grid"])
                out, _ = _summarize_field(M, zbar, ctx, cfg)
                resolution[f"n{gn}"][lab] = out
    summary = dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds, controls=control_names,
                   per_seed=per_seed, ar1_isotropic=ar1, resolution_n8_vs_n12=dict(seed=res_seed, **resolution),
                   config=cfg, git_sha=_git_sha(), argv=sys.argv)
    _atomic_json(os.path.join(OUT_DIR, "control_summary.json"), summary)
    _atomic_savez(os.path.join(OUT_DIR, "control_arrays.npz"), **arrays_out)
    print(f"[run-controls] -> control_summary.json (controls={control_names} + AR1 + resolution)", flush=True)
    return summary


def cmd_run_nonlinear(args):
    """Task 6 (P3, design §8 Gate E): two-amplitude linear-regime spot check on the primary seed."""
    import src.topic4_state_conditioned_susceptibility as M
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seed = int(args.seed) if args.seed else sorted(_load_locked_candidate(label)["onsets"])[0]
    snap = _load_snapshot(label, seed)
    ctx = _seed_context(cfg, snap, cfg["grid_n"])
    labels = [str(x) for x in snap["snapshot_labels"]]
    out = {}
    for lab in ("baseline_1000ms", "onset"):
        if lab not in labels:
            continue
        zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][labels.index(lab)], ctx["pos_norm"], ctx["grid"])
        op, J, q = M.state_operator(zbar, ctx["grid"], ctx["scaffold"], w_ee_mult=cfg["w_ee_mult"],
                                    ratio=cfg["ratio"], q_floor=cfg["q_floor"])
        if J is None:
            out[lab] = dict(op_status=op.status, note="not resolved; linear-regime check not run")
            continue
        b = M.probe_matrix(ctx["probes"][1:2], ctx["grid"])[:, 0]      # one axial-ish probe
        from scipy.sparse.linalg import expm_multiply
        a1, a2 = 1.0, 4.0                                              # fixed amplitudes (design §8: fixed before)
        y1 = np.linalg.norm(expm_multiply(J * cfg["T_primary"], a1 * b)[:ctx["grid"].size])
        y2 = np.linalg.norm(expm_multiply(J * cfg["T_primary"], a2 * b)[:ctx["grid"].size])
        ratio_scale = (y2 / y1) / (a2 / a1)
        out[lab] = dict(op_status=op.status, a1=a1, a2=a2, response_scale_ratio=float(ratio_scale),
                        linear_within_10pct=bool(abs(ratio_scale - 1.0) < 0.10))
    summary = dict(schema_version=SCHEMA_VERSION, candidate=label, seed=seed, per_state=out,
                   note="J is a LINEAR operator so exp(JT) scales exactly; this checks embedding/readout linearity.",
                   git_sha=_git_sha(), argv=sys.argv)
    _atomic_json(os.path.join(OUT_DIR, "nonlinear_spotcheck_summary.json"), summary)
    print(f"[run-nonlinear] -> nonlinear_spotcheck_summary.json {out}", flush=True)
    return summary


# ============================================================ small IO helpers
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2, default=_json_default)
    os.replace(tmp, path)


def _atomic_savez(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Topic 4 state-conditioned spatial susceptibility runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit-inputs", "smoke", "capture-snapshots", "build-atlas", "run-controls",
                 "run-nonlinear-spotchecks", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true", help="required to start any simulation")
        sp.add_argument("--candidate", default=None, help="MZ multiseed label (default zA_q50_tz10000)")
        sp.add_argument("--seeds", default=None, help="comma list (default all seeds of the candidate)")
        sp.add_argument("--seed", default=None, help="single seed (smoke)")
        sp.add_argument("--T", default=None, help="override sim duration ms (smoke)")
        sp.add_argument("--workers", default=None, help="max SNN workers, capped at 2")
        sp.add_argument("--resume", action="store_true", help="skip stages whose provenance matches")
    args = ap.parse_args(argv)
    needs_confirm = args.cmd not in ("audit-inputs",)
    if needs_confirm and not args.confirm_run:
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run (import-safe gate, design §4).",
              file=sys.stderr)
        sys.exit(2)
    if args.cmd == "audit-inputs":
        cmd_audit_inputs(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "capture-snapshots":
        cmd_capture_snapshots(args)
    elif args.cmd == "build-atlas":
        cmd_build_atlas(args)
    elif args.cmd == "run-controls":
        cmd_run_controls(args)
    elif args.cmd == "run-nonlinear-spotchecks":
        cmd_run_nonlinear(args)
    elif args.cmd == "all":
        cmd_capture_snapshots(args)
        cmd_build_atlas(args)
        cmd_run_controls(args)


if __name__ == "__main__":
    main()
