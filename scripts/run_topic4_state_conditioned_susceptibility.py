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
  capture-snapshots        replay candidate seeds, snapshot z_E/m_E on the registered trajectory grid
  build-atlas              (Task 5) coarse-field -> operator -> probe atlas (needs --confirm-run)
  run-controls             (Task 5) real/uniform/rotate/shuffle/z-blocked controls (needs --confirm-run)
  run-nonlinear-spotchecks (Task 6) two-amplitude linear-regime check (needs --confirm-run)
  plot-paper-ready        plotting-only export of Figure 5 Supplementary candidates 1/2
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
PAPER_FIG_DIR = os.path.join(
    ROOT, "results", "paper-ready-figure", "fig5_mz_spatial_dynamics_supplementary", "figures")
MZ_DIR = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_slowvars")
SCHEMA_VERSION = "sccs-1.0"

# Original atlas states remain frozen so extending the actual-time trajectory does not silently expand
# the expensive atlas/control estimand. Dense pre-onset captures are used only for the eigenmode
# timecourse, whose x-axis is true milliseconds to the locked SNN runoff time (never interpolation alpha).
ATLAS_STATES = ["baseline_1000ms", "mid_fraction", "pre_onset_500ms", "pre_onset_100ms", "onset"]
EIGEN_OFFSETS_MS = (1000, 750, 500, 300, 200, 100, 50, 20)
EIGEN_TIME_STATES = [f"pre_onset_{v}ms" for v in EIGEN_OFFSETS_MS] + ["onset"]
SNAP_STATES = list(dict.fromkeys(ATLAS_STATES[:-1] + EIGEN_TIME_STATES))

# capture runs a hair past onset (early_stop OFF) so the onset step is always reached and the runaway
# is confirmed to persist; trajectory <= onset is byte-identical to the locked early_stop replay.
CAPTURE_TAIL_MS = 400.0

SNAP_CONVENTION = (
    "captured inside MZSlowVars.step() AFTER the z/m Euler update and the streaming trace record; "
    "step counter is 0-based, one increment per simulate_kick iteration t; time_ms = step*dt; "
    "consistency: snapshot.z_E.mean() == mz.trace_z_mean[step]."
)


def _save_diagnostic_and_paper_figure(fig, diagnostic_stem, paper_stem):
    """Save one accepted canvas under both analysis and manuscript-facing names."""
    targets = (
        (os.path.join(OUT_DIR, "figures"), diagnostic_stem),
        (PAPER_FIG_DIR, paper_stem),
    )
    for directory, stem in targets:
        os.makedirs(directory, exist_ok=True)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(directory, f"{stem}.{ext}"), dpi=300, bbox_inches="tight")


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
    times = {
        "baseline_1000ms": 1000.0,
        "mid_fraction": 0.5 * float(onset),
        "onset": float(onset),
    }
    times.update({f"pre_onset_{v}ms": float(onset) - float(v) for v in EIGEN_OFFSETS_MS})
    return times


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
            if lab not in ATLAS_STATES:
                continue
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
        print(f"[build-atlas] seed {seed}: " + ", ".join(
            f"{l}={per_state[l]['op_status']}" for l in ATLAS_STATES if l in per_state), flush=True)
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
            if lab not in ATLAS_STATES:
                continue
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
        print(f"[run-controls] seed {seed}: {len(per_state)} atlas states x {len(control_names)} controls done", flush=True)
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


def cmd_run_convergence(args):
    """Grid-resolution convergence (review 2026-07-19): are the OPERATOR-based quantities (sigma1,
    eigenmode axis/globality, U1 output axis) grid-converged as n grows? peak_k with fixed p_max=4 is
    domain-limited (its low rail = the whole-sheet scale), reported for context."""
    import src.topic4_state_conditioned_susceptibility as M
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(_load_locked_candidate(label)["onsets"])
    grid_ns = [8, 12, 16, 20, 24]
    states = ["baseline_1000ms", "pre_onset_100ms"]
    per = {st: {n: [] for n in grid_ns} for st in states}
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        labels = [str(x) for x in snap["snapshot_labels"]]
        for n in grid_ns:
            ctx = _seed_context(cfg, snap, n)
            for st in states:
                if st not in labels:
                    continue
                zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][labels.index(st)], ctx["pos_norm"], ctx["grid"])
                # T=T_primary ONLY (convergence needs one window, not the 4-window atlas) -> 4x cheaper
                out, _ = M.summarize_state_susceptibility(
                    zbar, ctx["grid"], ctx["scaffold"], ctx["probes"], [cfg["T_primary"]],
                    w_ee_mult=cfg["w_ee_mult"], ratio=cfg["ratio"], q_floor=cfg["q_floor"],
                    T_primary=cfg["T_primary"], op_dt=cfg["op_dt"], op_t_max=cfg["op_t_max"], op_tol=cfg["op_tol"])
                if out.get("atlas"):
                    pt = out["atlas"]["per_T"][cfg["T_primary"]]
                    per[st][n].append(dict(sigma1=out["optimal"]["sigma1"], eig_glob=out["eigen"]["leading_globality"],
                                           eig_axis=out["eigen"]["leading_axis_score"],
                                           u1_axis=out["optimal"]["u1_output_axis"],
                                           kpar=pt["axial_gain"], kperp=pt["perp_gain"], peak_k=pt["peak_k"]))
            print(f"[convergence] seed {seed} n={n} done", flush=True)
    med = {st: {n: {k: float(np.median([r[k] for r in rows])) for k in rows[0]} if rows else None
                for n, rows in per[st].items()} for st in states}
    summary = dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds, grid_ns=grid_ns, p_max=cfg["p_max"],
                   note="operator quantities (sigma1/eig/u1) test grid convergence; peak_k low rail=2pi/L "
                        "(whole-sheet scale, domain-limited at fixed p_max), not a resolution artifact.",
                   median_over_seeds=med, git_sha=_git_sha(), argv=sys.argv)
    _atomic_json(os.path.join(OUT_DIR, "convergence_summary.json"), summary)
    _plot_convergence(med, grid_ns, states)
    print(f"[convergence] -> convergence_summary.json + figures/convergence.png", flush=True)
    return summary


def _plot_convergence(med, grid_ns, states):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, st in zip(axes, states):
        ns = [n for n in grid_ns if med[st][n] is not None]
        for key, col in (("sigma1", "#b35806"), ("kpar", "#1b7837"), ("kperp", "#762a83"),
                         ("eig_axis", "#2166ac"), ("u1_axis", "#5aae61")):
            ax.plot(ns, [med[st][n][key] for n in ns], "-o", color=col, label=key, lw=1.8, ms=4)
        axr = ax.twinx()
        axr.plot(ns, [med[st][n]["peak_k"] for n in ns], "--s", color="0.5", label="peak_k (right)", lw=1.4, ms=4)
        axr.set_ylabel("peak_k (rail=2pi/L=1.26)", fontsize=8, color="0.4")
        ax.set_title(f"{st}: operator quantities vs grid n  (representative seed)", fontsize=10)
        ax.set_xlabel("grid n (cells per side)"); ax.set_ylabel("gain / axis score")
        ax.set_xticks(grid_ns); ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="center right")
    fig.suptitle("Grid-resolution convergence — operator quantities (sigma1 / k|| / k_perp / eigenmode axis / "
                 "U1 output axis) stabilize; peak_k pinned at the whole-sheet rail (fixed p_max=4)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"convergence.{ext}"), dpi=150, bbox_inches="tight")


def cmd_run_time_response(args):
    """Final fixed-source-kick figure: baseline vs pre-onset 100 ms, same input in both states.

    The main figure contains only fixed-input readouts. The state-specific optimal-input envelope
    sigma1(T) is saved separately, because it answers a different operator question.
    """
    import src.topic4_state_conditioned_susceptibility as M
    from src.spatial_perturbation_toolkit import first_arrival_times, fit_arrival_time_distance
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(_load_locked_candidate(label)["onsets"])
    STATES = ["baseline_1000ms", "pre_onset_100ms"]
    T_sigma = [float(t) for t in range(0, 151, 5)]
    t_maps = [5.0, 15.0, 30.0, 50.0]
    t_kymo = sorted(set([float(t) for t in range(0, 101, 2)] + t_maps))
    kick_sigma = 0.6
    arrival_fraction = 0.10
    acc = {st: dict(sigma=[], maps=[], kymo=[], gain=[], source=[], sink=[], ratio=[], cumulative_ratio=[])
           for st in STATES}
    xs_ref = y_axis = None
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        ctx = _seed_context(cfg, snap, cfg["grid_n"])
        labels = [str(x) for x in snap["snapshot_labels"]]
        N = ctx["grid"].size
        b_fixed = M.make_localized_kick(ctx["grid"], tuple(ctx["src_norm"]), kick_sigma)   # SAME kick
        y_axis = float(ctx["src_norm"][1])
        for st in STATES:
            zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][labels.index(st)], ctx["pos_norm"], ctx["grid"])
            op, J, _ = M.state_operator(zbar, ctx["grid"], ctx["scaffold"], w_ee_mult=cfg["w_ee_mult"],
                                        ratio=cfg["ratio"], q_floor=cfg["q_floor"])
            if J is None:
                continue
            acc[st]["sigma"].append(M.sigma1_vs_T(J, ctx["grid"], T_sigma, N))
            evolution = M.fixed_kick_evolution(J, ctx["grid"], b_fixed, t_kymo, N)
            read = M.fixed_kick_readouts(
                evolution, ctx["grid"], source_center=ctx["src_norm"], sink_center=ctx["snk_norm"],
                region_radius=cfg["core_radius_norm"], axis_y=y_axis, axis_band=0.5,
                arrival_fraction=arrival_fraction)
            acc[st]["maps"].append(np.stack([evolution[t] for t in t_maps]))
            acc[st]["kymo"].append(read["kymograph"])
            acc[st]["gain"].append(read["fixed_gain"])
            acc[st]["source"].append(read["source_rms"])
            acc[st]["sink"].append(read["sink_rms"])
            acc[st]["ratio"].append(read["sink_source_ratio"])
            acc[st]["cumulative_ratio"].append(read["cumulative_sink_source_ratio"])
            xs_ref = read["xs"]
        print(f"[time-response] seed {seed} done", flush=True)
    out_arrays, summary = {}, {}
    for st in STATES:
        if not acc[st]["sigma"]:
            continue
        sig = np.median(np.stack(acc[st]["sigma"]), axis=0)
        out_arrays[f"{st}__sigma1_T"] = sig
        out_arrays[f"{st}__maps"] = np.median(np.stack(acc[st]["maps"]), axis=0)
        kymo = np.median(np.stack(acc[st]["kymo"]), axis=0)
        gain = np.median(np.stack(acc[st]["gain"]), axis=0)
        source = np.median(np.stack(acc[st]["source"]), axis=0)
        sink = np.median(np.stack(acc[st]["sink"]), axis=0)
        ratio_curve = np.nanmedian(np.stack(acc[st]["ratio"]), axis=0)
        cumulative_ratio = np.nanmedian(np.stack(acc[st]["cumulative_ratio"]), axis=0)
        arrival, threshold = first_arrival_times(kymo, t_kymo, threshold_fraction=arrival_fraction)
        fit = fit_arrival_time_distance(xs_ref, arrival, source_position=float(ctx["src_norm"][0]),
                                        sink_position=float(ctx["snk_norm"][0]))
        out_arrays[f"{st}__kymo"] = kymo
        out_arrays[f"{st}__fixed_gain"] = gain
        out_arrays[f"{st}__source_rms"] = source
        out_arrays[f"{st}__sink_rms"] = sink
        out_arrays[f"{st}__sink_source_ratio"] = ratio_curve
        out_arrays[f"{st}__cumulative_sink_source_ratio"] = cumulative_ratio
        out_arrays[f"{st}__arrival_ms"] = arrival
        cross = next((T_sigma[i] for i, v in enumerate(sig) if v > 1.0), None)
        ip = int(np.argmax(sig))
        ig = int(np.nanargmax(gain))
        summary[st] = dict(
            fixed_kick_gain_peak=float(gain[ig]), fixed_kick_gain_peak_ms=float(t_kymo[ig]),
            instantaneous_sink_source_ratio_at_30ms=float(ratio_curve[t_kymo.index(30.0)]),
            cumulative_sink_source_ratio_at_30ms=float(cumulative_ratio[t_kymo.index(30.0)]),
            cumulative_sink_source_ratio_at_100ms=float(cumulative_ratio[-1]),
            arrival_threshold=float(threshold),
            arrival_fit=fit, operator_envelope=dict(sigma1_cross1_ms=cross,
                                                     sigma1_peak=float(sig[ip]), T_peak_ms=T_sigma[ip]))
    out_arrays.update(T_sigma=np.array(T_sigma), t_maps=np.array(t_maps), t_kymo=np.array(t_kymo),
                      xs=xs_ref, y_axis=np.array([y_axis]),
                      src_x=np.array([float(ctx["src_norm"][0])]), snk_x=np.array([float(ctx["snk_norm"][0])]))
    _atomic_savez(os.path.join(OUT_DIR, "time_response_arrays.npz"), **out_arrays)
    _atomic_json(os.path.join(OUT_DIR, "time_response_summary.json"),
                 dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds, states=STATES,
                      kick_sigma=kick_sigma, kick_center="source_core", per_state=summary,
                      arrival_threshold_fraction=arrival_fraction,
                      model_contract="MZ z-only trajectory-derived frozen-q M3B rate-field susceptibility; "
                                     "not old qI/gK and not a direct perturbation of the full MZ SNN",
                      note="Main figure: FIXED source-core Gaussian kick evolved under exp(J_s t). "
                           "Operator sigma1(T) envelope is saved separately and is not the fixed-kick gain. "
                           "Arrival slope is threshold-defined evidence for sequential recruitment, not proof "
                           "of a continuous wavefront.", git_sha=_git_sha(), argv=sys.argv))
    _plot_time_response(out_arrays, STATES, cfg)
    print(f"[time-response] -> time_response_summary.json + figures/time_response.png + "
          f"operator_gain_envelope.png ; {summary}", flush=True)


def _plot_time_response(a, states, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D
    plt.rcParams.update({
        "font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.2,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    t_maps = a["t_maps"]; xs = a["xs"]; tk = a["t_kymo"]
    src_x, snk_x, half = float(a["src_x"][0]), float(a["snk_x"][0]), float(cfg["L_norm"]) / 2
    short = {"baseline_1000ms": "Baseline", "pre_onset_100ms": "Pre-onset"}
    # Avoid the manuscript's template-A/template-B red/blue semantic colors for state identity.
    cols = {"baseline_1000ms": "#555555", "pre_onset_100ms": "#C88719"}
    avail = [st for st in states if f"{st}__sigma1_T" in a]
    fig = plt.figure(figsize=(7.2, 7.8))
    gs = GridSpec(4, 9, figure=fig, hspace=0.46, wspace=0.36,
                  height_ratios=[1.0, 1.0, 1.08, 0.92],
                  width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 0.13],
                  top=0.975, bottom=0.075, left=0.085, right=0.955)

    def _clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.8)

    # a: same fixed kick, spatial response first.
    mmax = max([np.abs(a[f"{st}__maps"]).max() for st in avail] + [1e-12])
    for r, st in enumerate(avail):
        maps = a[f"{st}__maps"]
        for c, t in enumerate(t_maps):
            ax = fig.add_subplot(gs[r, 2 * c:2 * c + 2])
            im = ax.imshow(maps[c].T, origin="lower", extent=[-half, half, -half, half], cmap="PuOr_r",
                           vmin=-mmax, vmax=mmax, aspect="equal")
            ax.plot(src_x, float(a["y_axis"][0]), "^", color="k", ms=4.2)
            ax.plot(snk_x, float(a["y_axis"][0]), "v", color="k", ms=4.2)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{int(t)} ms", pad=3)
            if c == 0:
                ax.set_ylabel(short[st], color=cols[st], fontweight="semibold", labelpad=5)
            if r == 0 and c == 0:
                ax.text(-0.28, 1.08, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")
    cax_map = fig.add_subplot(gs[0:2, 8])
    cb = fig.colorbar(im, cax=cax_map)
    cb.set_label(r"$\Delta r_E$", labelpad=2); cb.ax.tick_params(labelsize=7.5, length=2)

    # b-c: axis-time responses. Arrival fits remain in the numeric sidecar.
    kmax = max([a[f"{st}__kymo"].max() for st in avail] + [1e-12])
    for c, st in enumerate(avail):
        ax = fig.add_subplot(gs[2, 4 * c:4 * c + 4])
        imk = ax.imshow(a[f"{st}__kymo"].T, origin="lower", aspect="auto", cmap="magma",
                        vmin=0, vmax=kmax, extent=[tk.min(), tk.max(), xs.min(), xs.max()])
        ax.axhline(src_x, color="w", ls="--", lw=0.8); ax.axhline(snk_x, color="w", ls=":", lw=0.8)
        ax.set_title(short[st], color=cols[st], fontweight="semibold", pad=3)
        ax.set_xlabel("Time (ms)")
        if c == 0:
            ax.set_ylabel("Axis position")
        else:
            ax.set_yticklabels([])
        ax.text(-0.12, 1.06, ("b" if c == 0 else "c"), transform=ax.transAxes,
                fontsize=11, fontweight="bold")
    cax_k = fig.add_subplot(gs[2, 8])
    cbk = fig.colorbar(imk, cax=cax_k)
    cbk.set_label(r"$|\Delta r_E|$", labelpad=2); cbk.ax.tick_params(labelsize=7.5, length=2)

    # d-e: only the two reader-facing scalar summaries.
    ax_gain = fig.add_subplot(gs[3, 0:4]); ax_remote = fig.add_subplot(gs[3, 5:9])
    for st in avail:
        ax_gain.plot(tk, a[f"{st}__fixed_gain"], color=cols[st], lw=2.0)
        ax_remote.plot(tk, a[f"{st}__cumulative_sink_source_ratio"], color=cols[st], lw=2.0)
    ax_gain.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax_gain.set(xlabel="Time (ms)", ylabel="Response norm", title="Fixed-kick response")
    ax_remote.set(xlabel="Time (ms)", ylabel="Energy ratio", title="Remote recruitment")
    ax_remote.set_ylim(bottom=0)
    for lab, ax in zip(("d", "e"), (ax_gain, ax_remote)):
        _clean(ax); ax.text(-0.14, 1.08, lab, transform=ax.transAxes, fontsize=11, fontweight="bold")
    handles = [Line2D([0], [0], color=cols[st], lw=2.0, label=short[st]) for st in avail]
    ax_gain.legend(handles=handles, frameon=False, loc="upper right", handlelength=2.2)

    _save_diagnostic_and_paper_figure(
        fig,
        diagnostic_stem="time_response",
        paper_stem="figure5_supplementary_1_spatial_perturbation_response",
    )
    plt.close(fig)

    # Supplement only: state-specific optimal-input upper bound, never labelled fixed-kick gain.
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for st in avail:
        ax.plot(a["T_sigma"], a[f"{st}__sigma1_T"], color=cols[st], lw=1.8, label=short[st])
    ax.axhline(1.0, color="0.55", ls="--", lw=0.8)
    ax.set_xlabel("Window (ms)"); ax.set_ylabel(r"$\sigma_1(T)$")
    ax.set_title("Operator gain")
    _clean(ax); ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"operator_gain_envelope.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def cmd_run_eigenmode_timecourse(args):
    """Instantaneous leading Jacobian mode on the actual MZ time-to-runoff trajectory.

    Unlike the older alpha continuation, every point is a captured SNN slow state at a real
    millisecond offset from the locked runoff time. Each point still uses a frozen-q rate-field
    operating point; onset is left blank when that equilibrium is unresolved.
    """
    import src.topic4_state_conditioned_susceptibility as M
    from src.spatial_perturbation_toolkit import normalized_field_overlap
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(_load_locked_candidate(label)["onsets"])
    time_labels = ["baseline_1000ms", "mid_fraction"] + \
                  [f"pre_onset_{v}ms" for v in EIGEN_OFFSETS_MS] + ["onset"]
    arrays, summary = {}, {}
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        labels = [str(x) for x in snap["snapshot_labels"]]
        missing = [lab for lab in time_labels if lab not in labels]
        if missing:
            raise RuntimeError(f"seed {seed} snapshot artifact lacks dense eigenmode states {missing}; "
                               "rerun capture-snapshots")
        ctx = _seed_context(cfg, snap, cfg["grid_n"])
        onset_t = float(snap["actual_time_ms"][labels.index("onset")])
        prev_op = None
        previous_field = None
        records, fields = [], []
        for lab in time_labels:
            i = labels.index(lab)
            t_rel = float(snap["actual_time_ms"][i]) - onset_t
            zbar, _, _ = M.bin_neuron_state_to_grid(snap["z_E"][i], ctx["pos_norm"], ctx["grid"])
            op, J, _ = M.state_operator(
                zbar, ctx["grid"], ctx["scaffold"], w_ee_mult=cfg["w_ee_mult"], ratio=cfg["ratio"],
                q_floor=cfg["q_floor"], op_dt=cfg["op_dt"], op_t_max=cfg["op_t_max"],
                op_tol=cfg["op_tol"], init=prev_op)
            rec = dict(label=lab, time_to_runoff_ms=t_rel, op_status=op.status, re=None, im=None,
                       freq_hz=None, damping_time_ms=None, axis_score=None, globality=None,
                       overlap_previous=None)
            field = np.full((ctx["grid"].n, ctx["grid"].n), np.nan)
            if J is not None:
                mode = M.leading_mode_snapshot(J, ctx["grid"], theta=ctx["theta"])
                if mode is not None:
                    field = mode.pop("field")
                    rec.update(**mode)
                    rec["damping_time_ms"] = (-1.0 / rec["re"] if rec["re"] < 0 else None)
                    if previous_field is not None:
                        rec["overlap_previous"] = normalized_field_overlap(previous_field, field)
                    previous_field = field
                prev_op = {"rE": op.rE, "rI": op.rI}
            records.append(rec); fields.append(field)
        for key in ("time_to_runoff_ms", "re", "im", "freq_hz", "damping_time_ms",
                    "axis_score", "globality", "overlap_previous"):
            arrays[f"{seed}__{key}"] = np.array([
                np.nan if r.get(key) is None else r[key] for r in records], float)
        arrays[f"{seed}__fields"] = np.stack(fields)
        arrays[f"{seed}__resolved"] = np.array([r["op_status"] == "resolved" for r in records], bool)
        summary[str(seed)] = dict(records=records,
                                  last_resolved_label=next((r["label"] for r in records[::-1]
                                                            if r["op_status"] == "resolved"), None))
        print(f"[eigenmode-timecourse] seed {seed}: last resolved={summary[str(seed)]['last_resolved_label']}",
              flush=True)
    out = dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds, labels=time_labels,
               model_contract="actual MZ z-only SNN slow-state timestamps -> frozen-q M3B rate-field Jacobian",
               mode_contract="instantaneous leading rate-branch eigenvalue and complex-pair E-loading; "
                             "not finite-time V1/U1 and not a fixed-kick response",
               per_seed=summary, git_sha=_git_sha(), argv=sys.argv)
    _atomic_savez(os.path.join(OUT_DIR, "eigenmode_timecourse_arrays.npz"), **arrays)
    _atomic_json(os.path.join(OUT_DIR, "eigenmode_timecourse_summary.json"), out)
    _plot_eigenmode_timecourse(arrays, seeds, time_labels, representative_seed=seeds[0])
    print("[eigenmode-timecourse] -> eigenmode_timecourse_summary.json + figures/eigenmode_timecourse.png",
          flush=True)


def _plot_eigenmode_timecourse(a, seeds, labels, representative_seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({
        "font.size": 9.0, "axes.titlesize": 10.0, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 7.8,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    cfg = _load_cfg()
    L = float(cfg["L_norm"]); half = L / 2
    rep_t = a[f"{representative_seed}__time_to_runoff_ms"]
    rep_fields = a[f"{representative_seed}__fields"]
    map_labels = ["baseline_1000ms", "mid_fraction", "pre_onset_500ms", "pre_onset_20ms"]
    idx = [labels.index(lab) for lab in map_labels]
    vmax = max([float(np.nanmax(rep_fields[i])) for i in idx if np.isfinite(rep_fields[i]).any()] + [1e-12])

    def _stack(key):
        return np.stack([a[f"{s}__{key}"] for s in seeds])

    t = np.nanmedian(_stack("time_to_runoff_ms"), axis=0)
    metrics = {
        "re": _stack("re"), "freq_hz": _stack("freq_hz"),
        "damping_time_ms": _stack("damping_time_ms"), "axis_score": _stack("axis_score"),
        "globality": _stack("globality"), "overlap_previous": _stack("overlap_previous"),
    }
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = GridSpec(2, 13, figure=fig, hspace=0.52, wspace=0.50,
                  height_ratios=[1.0, 0.90], width_ratios=[1] * 12 + [0.16],
                  top=0.96, bottom=0.13, left=0.08, right=0.955)
    for c, i in enumerate(idx):
        ax = fig.add_subplot(gs[0, 3 * c:3 * c + 3])
        if np.isfinite(rep_fields[i]).any():
            im = ax.imshow(rep_fields[i].T, origin="lower", extent=[-half, half, -half, half],
                           cmap="magma", vmin=0, vmax=vmax, aspect="equal")
        else:
            ax.set_facecolor("0.92"); ax.text(0.5, 0.5, "Unresolved", transform=ax.transAxes,
                                               ha="center", va="center", color="0.35")
        display = {"baseline_1000ms": "Baseline", "mid_fraction": "Midpoint",
                   "pre_onset_500ms": "Pre-onset", "pre_onset_20ms": "Pre-onset"}
        ax.set_title(f"{display[map_labels[c]]}\n{rep_t[i]:.0f} ms", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(-0.20, 1.08, "abcd"[c], transform=ax.transAxes, fontsize=11, fontweight="bold")
    cax = fig.add_subplot(gs[0, 12])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Mode amplitude", labelpad=2); cb.ax.tick_params(labelsize=7.5, length=2)

    def _column_summary(x):
        med = np.full(x.shape[1], np.nan); lo = med.copy(); hi = med.copy()
        for j in range(x.shape[1]):
            finite = x[:, j][np.isfinite(x[:, j])]
            if finite.size:
                med[j], lo[j], hi[j] = np.median(finite), finite.min(), finite.max()
        return med, lo, hi

    def _clean(ax):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(length=3, width=0.8)
        ax.set_xticks([-4000, -2000, 0])

    # e: stability; f: decay time. Frequency remains in the sidecar, not the main canvas.
    scalar_panels = [
        ("re", r"Re $\lambda$ (ms$^{-1}$)", "Stability", "#7B3294", 0.0),
        ("damping_time_ms", r"$\tau$ (ms)", "Persistence", "#2A9D8F", None),
    ]
    scalar_spans = ((0, 3), (4, 7))
    for c, (key, ylabel, title, col, zero) in enumerate(scalar_panels):
        lo_col, hi_col = scalar_spans[c]
        ax = fig.add_subplot(gs[1, lo_col:hi_col]); x = metrics[key]
        med, lo, hi = _column_summary(x)
        ax.fill_between(t, lo, hi, color=col, alpha=0.16, lw=0)
        ax.plot(t, med, color=col, lw=2.0)
        if zero is not None: ax.axhline(zero, color="0.55", ls="--", lw=0.8)
        ax.set(ylabel=ylabel, title=title)
        _clean(ax); ax.text(-0.19, 1.08, "ef"[c], transform=ax.transAxes,
                            fontsize=11, fontweight="bold")

    # g: spatial-mode metrics. Each quantity has a distinct non-template color.
    ax = fig.add_subplot(gs[1, 8:12])
    for key, col, lab in (("axis_score", "#C88719", "Axis"),
                          ("globality", "#555555", "Globality"),
                          ("overlap_previous", "#2A9D8F", "Overlap")):
        x = metrics[key]; med, _, _ = _column_summary(x)
        ax.plot(t, med, color=col, lw=1.8, label=lab)
    ax.set(ylabel="Score", title="Spatial mode")
    ax.set_ylim(-0.03, 1.03); _clean(ax)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(0.02, 0.48), handlelength=1.5)
    ax.text(-0.19, 1.08, "g", transform=ax.transAxes, fontsize=11, fontweight="bold")
    fig.supxlabel("Time to runoff (ms)", x=0.50, y=0.025, fontsize=9.5)
    _save_diagnostic_and_paper_figure(
        fig,
        diagnostic_stem="eigenmode_timecourse",
        paper_stem="figure5_supplementary_2_eigenmode_dynamics",
    )
    plt.close(fig)


def cmd_plot_paper_ready(args):
    """Rebuild the two Figure 5 supplementary candidates from accepted numeric sidecars only."""
    time_path = os.path.join(OUT_DIR, "time_response_arrays.npz")
    eigen_path = os.path.join(OUT_DIR, "eigenmode_timecourse_arrays.npz")
    eigen_summary_path = os.path.join(OUT_DIR, "eigenmode_timecourse_summary.json")
    missing = [p for p in (time_path, eigen_path, eigen_summary_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError("paper-ready export requires accepted sidecars: " + ", ".join(missing))

    cfg = _load_cfg()
    with np.load(time_path, allow_pickle=False) as time_arrays:
        _plot_time_response(time_arrays, ["baseline_1000ms", "pre_onset_100ms"], cfg)

    eigen_summary = json.load(open(eigen_summary_path))
    seeds = [int(s) for s in eigen_summary["seeds"]]
    if args.seeds:
        requested = [int(s) for s in args.seeds.split(",")]
        unknown = sorted(set(requested) - set(seeds))
        if unknown:
            raise ValueError(f"requested seeds are absent from the accepted eigenmode sidecar: {unknown}")
        seeds = requested
    with np.load(eigen_path, allow_pickle=False) as eigen_arrays:
        _plot_eigenmode_timecourse(
            eigen_arrays,
            seeds,
            eigen_summary["labels"],
            representative_seed=seeds[0],
        )
    print(f"[plot-paper-ready] -> {PAPER_FIG_DIR}", flush=True)


def _classify_continuation(traj):
    """Classify the resting->runaway transition from the leading-eigenvalue trajectory. Honest about the
    Hopf-vs-fold ambiguity when the steady solver loses the fixed point before Re crosses 0."""
    res = [r for r in traj if r["op_status"] == "resolved" and r["re"] is not None]
    if not res:
        return dict(classification="no_resolved_point", alpha_crit=None)
    cross = next((r for r in res if r["re"] >= 0.0), None)
    if cross is not None:                                          # leading Re actually crossed 0 (confirmed)
        return dict(classification=("hopf_confirmed" if cross["is_complex"] else "real_instability_confirmed"),
                    alpha_crit=cross["alpha"], freq_hz_crit=(cross["freq_hz"] if cross["is_complex"] else 0.0),
                    re_at_last_resolved=res[-1]["re"])
    # the transition is the FIRST loss of the resolved fixed point (its status distinguishes fold vs Hopf;
    # later saturated alphas are past the transition and do NOT define it)
    last_before, first_loss, seen = None, None, False
    for r in traj:
        if r["op_status"] == "resolved":
            seen = True; last_before = r
        elif seen:
            first_loss = r; break
    if first_loss is None:
        return dict(classification="stable_throughout", alpha_crit=None, re_at_last_resolved=res[-1]["re"])
    if first_loss["op_status"] == "saturated":                    # jumps to a saturated high-rate branch
        return dict(classification="saturation_jump", alpha_last_resolved=last_before["alpha"],
                    re_at_last_resolved=last_before["re"],
                    note="op jumps to a saturated high-rate branch at the first fixed-point loss")
    last = last_before                                            # low-rate loss: describe the near-critical mode
    mode = ("complex_%.0fHz" % last["freq_hz"]) if last["is_complex"] else "real"
    return dict(classification="fixed_point_loss_low_rate",
                leading_mode=("complex" if last["is_complex"] else "real"),
                freq_hz_near_crit=(last["freq_hz"] if last["is_complex"] else None),
                re_at_last_resolved=last["re"], alpha_last_resolved=last["alpha"],
                note=("resting fixed point lost (steady solver stops converging) at LOW rate (rE_max << "
                      "saturation, NOT a jump to a high-rate branch) while the leading mode is a weakly-damped "
                      f"{mode} pair (Re~{last['re']:.3f}). Consistent with an oscillatory (Hopf-type) "
                      "transition to a limit cycle, but NOT a confirmed supercritical Hopf: leading Re does not "
                      "smoothly cross 0 before the fixed point vanishes (could be a fold / subcritical). "
                      "Disambiguate fold-vs-Hopf with finer alpha near the loss + the post-onset limit-cycle "
                      "analysis (time-dependent tangent operator / Floquet)."))


def cmd_run_continuation(args):
    """Continuation z_alpha=(1-a)z_pre100 + a z_onset (review 2026-07-19): warm-start the operating point
    along the path, track the leading rate-branch eigenvalue, and classify the resting->runaway
    transition (Hopf if a complex pair's Re crosses 0; saddle-node/saturation if the fixed point
    disappears with Re still <0). Answers 'why the resting state becomes oscillatory/runaway'."""
    import src.topic4_state_conditioned_susceptibility as M
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(_load_locked_candidate(label)["onsets"])
    alphas = [float(a) for a in np.linspace(0.0, 1.0, 41)]
    per_seed, np_arrays = {}, {}
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        ctx = _seed_context(cfg, snap, cfg["grid_n"])
        labels = [str(x) for x in snap["snapshot_labels"]]
        z_pre = M.bin_neuron_state_to_grid(snap["z_E"][labels.index("pre_onset_100ms")], ctx["pos_norm"], ctx["grid"])[0]
        z_on = M.bin_neuron_state_to_grid(snap["z_E"][labels.index("onset")], ctx["pos_norm"], ctx["grid"])[0]
        traj, prev = [], None
        for a in alphas:
            z_a = (1.0 - a) * z_pre + a * z_on
            op, J, _ = M.state_operator(z_a, ctx["grid"], ctx["scaffold"], w_ee_mult=cfg["w_ee_mult"],
                                        ratio=cfg["ratio"], q_floor=cfg["q_floor"], init=prev)
            rec = dict(alpha=a, op_status=op.status, rE_max=float(op.rE.max()),
                       re=None, im=None, freq_hz=None, is_complex=None)
            if J is not None:
                le = M.leading_eigenvalue(J, ctx["grid"])
                rec.update(re=le["re"], im=le["im"], freq_hz=le["freq_hz"], is_complex=le["is_complex"])
                prev = {"rE": op.rE, "rI": op.rI}
            traj.append(rec)
        per_seed[str(seed)] = dict(trajectory=traj, **_classify_continuation(traj))
        for key in ("alpha", "re", "im", "freq_hz", "rE_max"):
            np_arrays[f"{seed}__{key}"] = np.array([r[key] if r[key] is not None else np.nan for r in traj], float)
        np_arrays[f"{seed}__resolved"] = np.array([1.0 if r["op_status"] == "resolved" else 0.0 for r in traj])
        _sc = {"resolved": 1.0, "saturated": 2.0, "unresolved": 0.0}   # status trajectory (fold vs Hopf clue)
        np_arrays[f"{seed}__status"] = np.array([_sc.get(r["op_status"], 0.0) for r in traj])
        print(f"[continuation] seed {seed}: {per_seed[str(seed)]['classification']} "
              f"alpha_crit={per_seed[str(seed)].get('alpha_crit')}", flush=True)
    summary = dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds, alphas=alphas,
                   note="continuation z_alpha=(1-a)z_pre100 + a z_onset; warm-start branch tracking; leading "
                        "rate-branch eigenvalue (Re/Im); classification per seed.",
                   per_seed={s: {k: v for k, v in d.items() if k != "trajectory"} for s, d in per_seed.items()},
                   git_sha=_git_sha(), argv=sys.argv)
    _atomic_savez(os.path.join(OUT_DIR, "continuation_arrays.npz"), **np_arrays)
    _atomic_json(os.path.join(OUT_DIR, "continuation_summary.json"), summary)
    _plot_continuation(np_arrays, seeds)
    print(f"[continuation] -> continuation_summary.json + figures/continuation.png", flush=True)


def _plot_continuation(a, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2166ac", "#b35806", "#1b7837", "#762a83"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    for i, s in enumerate(seeds):
        al = a[f"{s}__alpha"]; c = cols[i % len(cols)]
        ax[0].plot(al, a[f"{s}__re"], "-o", color=c, ms=3, label=f"seed {s}")
        ax[1].plot(al, a[f"{s}__freq_hz"], "-o", color=c, ms=3, label=f"seed {s}")
        ax[2].plot(al, a[f"{s}__rE_max"], "-o", color=c, ms=3, label=f"seed {s}")
        res = a[f"{s}__resolved"]
        lost = np.where(res < 0.5)[0]
        if lost.size:
            ax[0].axvline(al[lost[0]], color=c, ls=":", lw=1, alpha=0.7)
    ax[0].axhline(0, color="0.5", lw=0.8, ls="--"); ax[0].set_ylabel("Re(leading eigenvalue) (1/ms)")
    ax[0].set_title("leading Re vs alpha  (>=0 => linear instability; dotted = fixed point lost)", fontsize=9.5)
    ax[1].set_ylabel("leading |Im|/2pi (Hz)"); ax[1].set_title("leading-mode frequency vs alpha  (>0 => complex pair => Hopf-type)", fontsize=9.5)
    ax[2].set_ylabel("operating-point rE_max (kHz)"); ax[2].set_yscale("log")
    ax[2].set_title("operating-point rate vs alpha  (jump => saturation/branch loss)", fontsize=9.5)
    for x in ax:
        x.set_xlabel("alpha  (0 = pre_onset_100ms  ->  1 = onset)"); x.grid(alpha=0.25); x.legend(fontsize=8)
    fig.suptitle("Continuation pre_onset_100ms -> onset: is the resting->runaway transition Hopf, real-instability, or saddle-node/saturation?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"continuation.{ext}"), dpi=150, bbox_inches="tight")


def cmd_run_post_onset(args):
    """Post-onset autonomous dynamics (review 2026-07-19): integrate the rate field from the PRE-ONSET
    fixed point (user-specified init) under the ONSET q-field -> is it a ~24 Hz limit cycle or a
    saturation? Resolves the continuation's open question (fold-vs-Hopf / oscillatory-vs-saturated)."""
    import src.topic4_state_conditioned_susceptibility as M
    cfg = _load_cfg()
    label = args.candidate or "zA_q50_tz10000"
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else sorted(_load_locked_candidate(label)["onsets"])
    t_max = float(args.T) if args.T else 2500.0     # runaway is SLOW (>300ms); match solve_operating_point window
    per_seed, arrays = {}, {}
    for seed in seeds:
        snap = _load_snapshot(label, seed)
        ctx = _seed_context(cfg, snap, cfg["grid_n"])
        labels = [str(x) for x in snap["snapshot_labels"]]
        z_pre = M.bin_neuron_state_to_grid(snap["z_E"][labels.index("pre_onset_100ms")], ctx["pos_norm"], ctx["grid"])[0]
        z_on = M.bin_neuron_state_to_grid(snap["z_E"][labels.index("onset")], ctx["pos_norm"], ctx["grid"])[0]
        op_pre, _, _ = M.state_operator(z_pre, ctx["grid"], ctx["scaffold"], w_ee_mult=cfg["w_ee_mult"],
                                        ratio=cfg["ratio"], q_floor=cfg["q_floor"])
        q_on = M.zbar_to_q(z_on, cfg["q_floor"])
        tr = M.forward_integrate_ratefield(ctx["grid"], ctx["scaffold"], q_on, op_pre.rE, op_pre.rI,
                                           w_ee_mult=cfg["w_ee_mult"], ratio=cfg["ratio"], t_max=t_max,
                                           dt=0.5, record_every=4, record_state_every=125)
        cl = M.classify_post_onset(tr)
        per_seed[str(seed)] = cl
        arrays[f"{seed}__t"] = tr["t"]; arrays[f"{seed}__rE_mean"] = tr["rE_mean"]; arrays[f"{seed}__rE_max"] = tr["rE_max"]
        # frozen-J finite-time susceptibility along the actual trajectory (time-dependent operator, item 3)
        N = ctx["grid"].size
        tsig, tre = [], []
        for st in tr["states"]:
            Jt = M.jacobian_at_state(ctx["grid"], ctx["scaffold"], q_on, st, w_ee_mult=cfg["w_ee_mult"], ratio=cfg["ratio"])
            tsig.append(M.optimal_finite_time_perturbation(Jt, ctx["grid"], cfg["T_primary"], N)["sigma1"])
            le = M.leading_eigenvalue(Jt, ctx["grid"]); tre.append(le["re"] if le else np.nan)
        arrays[f"{seed}__traj_t"] = tr["state_times"]
        arrays[f"{seed}__traj_sigma1"] = np.array(tsig, float); arrays[f"{seed}__traj_re"] = np.array(tre, float)
        cl["traj_sigma1_peak"] = float(np.nanmax(tsig)); cl["traj_re_max"] = float(np.nanmax(tre))
        print(f"[post-onset] seed {seed}: {cl}", flush=True)
    summary = dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds, t_max=t_max,
                   init="pre_onset_100ms fixed point + onset q-field (user-specified)", per_seed=per_seed,
                   git_sha=_git_sha(), argv=sys.argv)
    _atomic_savez(os.path.join(OUT_DIR, "post_onset_arrays.npz"), **arrays)
    _atomic_json(os.path.join(OUT_DIR, "post_onset_summary.json"), summary)
    _plot_post_onset(arrays, per_seed, seeds)
    print(f"[post-onset] -> post_onset_summary.json + figures/post_onset.png ; "
          f"{ {s: d['outcome'] for s, d in per_seed.items()} }", flush=True)


def _plot_post_onset(a, per_seed, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = ["#2166ac", "#b35806", "#1b7837", "#762a83"]
    fig, axs = plt.subplots(2, 2, figsize=(14, 9)); ax = axs.ravel()
    for i, s in enumerate(seeds):
        t = a[f"{s}__t"]; rm = a[f"{s}__rE_mean"]; rx = a[f"{s}__rE_max"]; c = cols[i % len(cols)]
        out = per_seed[str(s)]["outcome"]; f0 = per_seed[str(s)].get("dom_freq_hz")
        lab = f"seed {s} ({out}" + (f", {f0:.0f}Hz)" if f0 else ")")
        ax[0].plot(t, rm, "-", color=c, lw=1.2, label=lab)
        ax[1].plot(t, rx, "-", color=c, lw=1.2, label=f"seed {s}")
        half = len(rm) // 2
        x = rm[half:] - np.nanmean(rm[half:])
        fr = np.fft.rfftfreq(len(x), d=(t[1] - t[0]) / 1000.0); amp = np.abs(np.fft.rfft(np.nan_to_num(x)))
        ax[2].plot(fr, amp, "-", color=c, lw=1.2, label=f"seed {s}")
        tt = a.get(f"{s}__traj_t")
        if tt is not None:
            ax[3].plot(tt, a[f"{s}__traj_sigma1"], "-o", color=c, ms=3, lw=1.4, label=f"seed {s}")
    ax[0].set_xlabel("time (ms)"); ax[0].set_ylabel("rE_mean (kHz)")
    ax[0].set_title("A: autonomous rE_mean(t) from pre-onset FP + onset q-field", fontsize=9.5)
    ax[1].set_xlabel("time (ms)"); ax[1].set_ylabel("rE_max (kHz)"); ax[1].set_yscale("log")
    ax[1].axhline(0.1, color="0.5", ls=":", lw=0.8); ax[1].set_title("B: rE_max(t)  (dotted=saturation 0.1)", fontsize=9.5)
    ax[2].set_xlabel("frequency (Hz)"); ax[2].set_ylabel("|FFT rE_mean| (2nd half)"); ax[2].set_xlim(0, 120)
    ax[2].axvline(24, color="0.5", ls=":", lw=0.8); ax[2].set_title("C: spectrum (dotted=continuation ~24Hz)", fontsize=9.5)
    ax[3].axhline(1.0, color="0.5", ls=":", lw=0.8)
    ax[3].set_xlabel("time along trajectory (ms)"); ax[3].set_ylabel("frozen-J sigma1(T=30ms)")
    ax[3].set_title("D: finite-time susceptibility along the trajectory (frozen J(t))", fontsize=9.5)
    for x in ax:
        x.grid(alpha=0.25); x.legend(fontsize=8)
    fig.suptitle("Post-onset autonomous dynamics (limit cycle vs saturation) + finite-time susceptibility along the trajectory", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, "figures", f"post_onset.{ext}"), dpi=150, bbox_inches="tight")


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
                 "run-nonlinear-spotchecks", "run-convergence", "run-time-response",
                 "run-eigenmode-timecourse", "run-continuation", "run-post-onset",
                 "plot-paper-ready", "all"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true", help="required to start any simulation")
        sp.add_argument("--candidate", default=None, help="MZ multiseed label (default zA_q50_tz10000)")
        sp.add_argument("--seeds", default=None, help="comma list (default all seeds of the candidate)")
        sp.add_argument("--seed", default=None, help="single seed (smoke)")
        sp.add_argument("--T", default=None, help="override sim duration ms (smoke)")
        sp.add_argument("--workers", default=None, help="max SNN workers, capped at 2")
        sp.add_argument("--resume", action="store_true", help="skip stages whose provenance matches")
    args = ap.parse_args(argv)
    needs_confirm = args.cmd not in ("audit-inputs", "plot-paper-ready")
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
    elif args.cmd == "run-convergence":
        cmd_run_convergence(args)
    elif args.cmd == "run-time-response":
        cmd_run_time_response(args)
    elif args.cmd == "run-eigenmode-timecourse":
        cmd_run_eigenmode_timecourse(args)
    elif args.cmd == "run-continuation":
        cmd_run_continuation(args)
    elif args.cmd == "run-post-onset":
        cmd_run_post_onset(args)
    elif args.cmd == "plot-paper-ready":
        cmd_plot_paper_ready(args)
    elif args.cmd == "all":
        cmd_capture_snapshots(args)
        cmd_build_atlas(args)
        cmd_run_controls(args)


if __name__ == "__main__":
    main()
