"""Topic 4 — MZ full-SNN state-aligned finite-time spatial mode tracking: scientific runner.

*** THIS RUNS SIMULATIONS. *** Nothing runs on import; every sim subcommand is gated by --confirm-run.
Design contract (BINDING): docs/superpowers/specs/2026-07-21-topic4-mz-m-eigenmode-tracking-design.md

Tracks the empirical finite-time spatial response of the COMPLETE current-based MZ spiking network
(≈40k E/I LIF) along the z+m plateau slow-state trajectory. REUSE (not reinvent): the direct-spatial
runner `run_topic4_mz_direct_spatial_modes` (imported as DSM) supplies build_S + the fork / fixed-kick
/ corrected-audit machinery; `src.topic4_mz_m_eigenmode_tracking` supplies the new state-registration,
m-counterfactual, and cross-state mode-tracking math; `src.topic4_mz_onset_dynamics` supplies the
checkpoint/resume loop + the D/a/rate trajectory.

Subcommands (all resumable via per-seed/per-state JSON + checkpoint pickles; re-run is idempotent):
  smoke     tiny-net end-to-end glue (register -> capture -> fixed-kick -> audit -> m-controls) — fast
  register  P0: replay -> upstream-NPZ parity -> 5-state registration -> full-state checkpoints
  run       P1 fixed-kick tracking + P2 low-k strict operator audit at every registered state
  controls  P4: minimal m-mechanism counterfactuals (native / reset / uniform / shuffle) fixed-kick
  aggregate P3 mode tracking + summaries + numerical audit + provenance + STATUS
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse   # noqa: E402
import copy       # noqa: E402
import glob       # noqa: E402
import json       # noqa: E402
import pickle     # noqa: E402
import sys        # noqa: E402
import time       # noqa: E402

import numpy as np  # noqa: E402
import yaml         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_topic4_mz_direct_spatial_modes as DSM  # noqa: E402  (build_S + fork/kick/audit machinery)
from src.topic4_mz_onset_dynamics import run_loop, natural_zm_trajectory  # noqa: E402
from src.topic4_mz_direct_spatial_modes import (  # noqa: E402
    MZSpatialProbe, real_fourier_basis_2d, spikes_to_rate_grid,
    field_axis_alignment, field_globality, normalized_field_overlap,
)
from src.topic4_m3b_spectral_phase import Grid  # noqa: E402
from src.topic4_mz_slowvars import eta_m_from_frac  # noqa: E402
from mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_mz_m_eigenmode_tracking import (  # noqa: E402
    SCHEMA_VERSION, build_zm_slow_config, register_states, trajectory_parity, apply_m_control,
    principal_angles_deg, subspace_alignment, centroid_displacement, leading_subspace,
    state_checkpoint_fingerprint,
)

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_m_eigenmode_tracking")
CKDIR = os.path.join(OUT, "per_seed", "checkpoints")
CFG_PATH = os.path.join(ROOT, "config", "topic4_mz_m_eigenmode_tracking.yaml")
REGPATH = os.path.join(OUT, "state_registration.json")
DT = 0.1
STATE_ORDER = ["baseline", "approach_25", "approach_50", "approach_75", "settled_plateau"]
ADJACENT = [("baseline", "approach_25"), ("approach_25", "approach_50"),
            ("approach_50", "approach_75"), ("approach_75", "settled_plateau")]


# ============================================================ config + provenance + io
def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


def _provenance(cfg, extra=None):
    prov = dict(schema_version=SCHEMA_VERSION, git_sha=DSM._git_sha(), engine_shas=DSM._engine_shas(),
                config_hash=DSM._file_hash(CFG_PATH),
                module_hash=DSM._file_hash(os.path.join(ROOT, "src", "topic4_mz_m_eigenmode_tracking.py")),
                dsm_module_hash=DSM._file_hash(os.path.join(ROOT, "src", "topic4_mz_direct_spatial_modes.py")),
                argv=sys.argv, subject=cfg["subject"], montage=cfg["montage"], dt=DT)
    if extra:
        prov.update(extra)
    return prov


def _dump(obj, path):
    DSM._dump(obj, path)                                        # atomic write, numpy-aware default


def _load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _seeds(args, cfg):
    return [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]


def _sr_kwargs(sr):
    """The frozen state-registration rule block -> register_states kwargs (spec §3/§9)."""
    return dict(baseline_ms=float(sr["baseline_ms"]),
                baseline_search_halfwidth_ms=float(sr["baseline_search_halfwidth_ms"]),
                approach_fracs=[float(x) for x in sr["approach_fracs"]],
                approach_search_ms=float(sr["approach_search_ms"]),
                settle_tail_ms=float(sr["settle_tail_ms"]),
                resting_win_ms=float(sr["resting_win_ms"]), resting_k=float(sr["resting_k"]),
                settled_D_ptp_max=float(sr["settled_D_ptp_max"]),
                settled_a_ptp_max=float(sr["settled_a_ptp_max"]),
                settled_min_resting_frac=float(sr["settled_min_resting_frac"]),
                D_onset_ref=float(sr["D_onset_ref"]))


def _upstream_traj(cfg, seed):
    up = cfg["upstream"]
    p = os.path.join(ROOT, up["onset_traj_dir"], up["onset_traj_template"].format(seed=seed))
    d = np.load(p, allow_pickle=True)
    return dict(D=np.asarray(d["D_allE"], float), a=np.asarray(d["a_allE"], float),
                rate=np.asarray(d["rate_E_hz"], float), eta_m=float(d["eta_m"]), path=p,
                sha=DSM._file_hash(p))


# ============================================================ checkpoint persistence
_SLOW_TRACE_ATTRS = ("trace_z_mean", "trace_z_min", "trace_z_core_mean", "trace_z_surround_mean",
                     "trace_m_mean", "trace_m_max", "trace_m_core_mean", "trace_m_surround_mean",
                     "trace_adap_current", "trace_I_EI_E_mean", "trace_rate_E", "trace_rate_I",
                     "calib_hist_I_EI", "calib_hist_I_EE")


def _strip_slow_traces(slow):
    """Clear the slow object's diagnostic trace lists (they grow to ~200k entries over the replay and
    are NOT part of the recoverable state — z/m/currents/rings/rng carry that). Stripping them shrinks
    the checkpoint pickle and makes every fork's deepcopy(ck.slow) cheap. Does not change the checkpoint
    fingerprint (traces are not hashed)."""
    for a in _SLOW_TRACE_ATTRS:
        if hasattr(slow, a):
            setattr(slow, a, [])
    return slow


def _ck_path(seed, state):
    return os.path.join(CKDIR, f"ck_seed{seed}_{state}.pkl")


def _capture_checkpoints(S, zm_cfg, states, seed, *, persist=True):
    """Segmented replay + resume capturing the full LoopState at each RESOLVED registered step (one
    pass, bit-consistent with the registration replay because it re-runs the identical seed/substrate/
    slow with the RNG reset — spec §3.6, verified by E4). persist=False (smoke) keeps checkpoints in
    memory only, so the tiny-net smoke never writes into the real CKDIR."""
    items = sorted([(st, int(d["branch_step"])) for st, d in states.items() if d.get("branch_step") is not None],
                   key=lambda x: x[1])
    slow = MZSpatialProbe(S["N"], 18.0, zm_cfg, NE=S["NE"])
    S["net"]["rng"] = np.random.default_rng(seed)
    cks, fps, start, cur = {}, {}, None, 0
    for st, step in items:
        n = max(int(step) - cur, 1)
        rep = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=n, start=start,
                       capture_final=True, store_spikes=False)
        ck = rep["checkpoint"]
        _strip_slow_traces(ck.slow)                             # lean checkpoint (traces not needed to fork)
        cks[st], fps[st] = ck, state_checkpoint_fingerprint(ck)
        if persist:
            os.makedirs(CKDIR, exist_ok=True)
            with open(_ck_path(seed, st), "wb") as f:
                pickle.dump(ck, f)
        start, cur, slow = ck, int(ck.t), copy.deepcopy(ck.slow)
    return cks, fps


def _register_done(seed, reg_all):
    """Resume: True iff `seed` is registered AND every resolved-state checkpoint exists on disk."""
    rec = reg_all.get("seeds", {}).get(str(seed))
    if rec is None:
        return False
    return all(os.path.exists(_ck_path(seed, st)) for st, d in rec["states"].items()
               if d.get("branch_step") is not None)


def _state_done(path, resume):
    """Resume-skip predicate: a completed per-state output JSON is not recomputed (idempotency, E18)."""
    return bool(resume and os.path.exists(path))


def _load_checkpoints(seed, seed_rec):
    cks = {}
    for st, d in seed_rec["states"].items():
        if d.get("branch_step") is not None and os.path.exists(_ck_path(seed, st)):
            with open(_ck_path(seed, st), "rb") as f:
                ck = pickle.load(f)
            _strip_slow_traces(ck.slow)                        # lean forks even from a pre-fix fat pickle
            cks[st] = ck
    return cks


# ============================================================ P0 register (replay + parity + capture)
def cmd_register(args, cfg):
    import gc
    zm_cfg = build_zm_slow_config(cfg["work_point"], cfg["I_EE_scale"])
    sr = cfg["state_registration"]
    replay_steps = int(round(float(sr["replay_ms"]) / DT))
    os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)
    reg_all = _load_json(REGPATH) or dict(schema_version=SCHEMA_VERSION, eta_m=zm_cfg.eta_m,
                                          work_point=cfg["work_point"], lock=sr, seeds={})
    for seed in _seeds(args, cfg):
        if args.resume and _register_done(seed, reg_all):
            print(f"[register] resume skip seed{seed}", flush=True)
            continue
        t0 = time.time()
        S = DSM.build_S(seed, cfg)
        # Pass 1: full replay -> D/a/rate traces (store_spikes=False: no NxT)
        slow = MZSpatialProbe(S["N"], 18.0, zm_cfg, NE=S["NE"])
        S["net"]["rng"] = np.random.default_rng(seed)
        res = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=replay_steps,
                       capture_final=False, store_spikes=False)
        rate_step = res["rate_E"]
        z_mean = np.asarray(slow.trace_z_mean, float)
        adap = np.asarray(slow.trace_adap_current, float)
        D_step = 1.0 - z_mean
        a_step = adap / float(cfg["I_EE_scale"])
        # parity vs upstream NPZ (downsampled)
        traj = natural_zm_trajectory(z_mean, adap, rate_step, DT, I_EE_scale=float(cfg["I_EE_scale"]),
                                     downsample_ms=float(sr["downsample_ms"]))
        up = _upstream_traj(cfg, seed)
        parity = trajectory_parity(traj["D_allE"], traj["a_allE"], traj["rate_E_hz"],
                                   up["D"], up["a"], up["rate"], rel_tol=float(sr["parity_rel_tol"]))
        np.savez_compressed(os.path.join(OUT, "per_seed", f"traj_seed{seed}.npz"),   # downsampled traj for the figure
                            t_ms=traj["t_ms"], D_allE=traj["D_allE"], a_allE=traj["a_allE"],
                            rate_E_hz=traj["rate_E_hz"])
        # registration (step resolution) + full-state checkpoint capture
        reg = register_states(D_step, a_step, rate_step, DT, **_sr_kwargs(sr))
        cks, fps = _capture_checkpoints(S, zm_cfg, reg["states"], seed)
        states_out = {}
        for st, d in reg["states"].items():
            rec = dict(d)
            rec["checkpoint_sha"] = fps.get(st)
            rec["checkpoint_file"] = os.path.relpath(_ck_path(seed, st), ROOT) if st in cks else None
            states_out[st] = rec
        reg_all["seeds"][str(seed)] = dict(
            seed=seed, wall_s=round(time.time() - t0, 1), eta_m=zm_cfg.eta_m, D_base=reg["D_base"],
            D_plateau=reg["D_plateau"], settled=reg["settled"], n_steps=reg["n_steps"],
            parity=parity, upstream=dict(path=os.path.relpath(up["path"], ROOT), sha=up["sha"], eta_m=up["eta_m"]),
            states=states_out, provenance=_provenance(cfg, dict(phase="register", seed=seed)))
        _dump(reg_all, REGPATH)
        flag = "OK" if parity["pass"] else "*** PARITY DISCREPANCY (stop & report) ***"
        print(f"[register] seed{seed} parity D/a/rate rel="
              f"{parity['D']['rel']:.2e}/{parity['a']['rel']:.2e}/{parity['rate']['rel']:.2e} {flag} "
              f"| D_base={reg['D_base']:.4f} D_plateau={reg['D_plateau']:.4f} settled={reg['settled']} "
              f"({reg_all['seeds'][str(seed)]['wall_s']}s)", flush=True)
        for st in STATE_ORDER:
            d = states_out[st]
            print(f"    {st:16s} step={d.get('branch_step')} D={d.get('D')} resolved={d.get('resolved')}", flush=True)
        del S, cks
        gc.collect()


# ============================================================ P1+P2 run (fixed-kick + strict audit)
def cmd_run(args, cfg):
    import gc
    if args.realizations:
        cfg = {**cfg, "corrected_audit": {**cfg["corrected_audit"], "n_realizations": int(args.realizations)}}
    seeds = _seeds(args, cfg)
    states = args.states.split(",") if args.states else STATE_ORDER
    workers = int(args.workers or cfg["workers"])
    freeze = bool(cfg.get("freeze_zm", True))
    reg_all = _load_json(REGPATH)
    if reg_all is None:
        raise SystemExit("run requires registration: run `register` first")
    os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)
    for seed in seeds:
        seed_rec = reg_all["seeds"].get(str(seed))
        if seed_rec is None:
            raise SystemExit(f"seed {seed} not registered; run `register --seeds {seed}` first")
        S = DSM.build_S(seed, cfg)
        DSM._ensure_flat(S)
        cks = _load_checkpoints(seed, seed_rec)
        for st in states:
            rj = os.path.join(OUT, "per_seed", f"state_seed{seed}_{st}.json")
            if _state_done(rj, args.resume):
                print(f"[run] resume skip s{seed} {st}", flush=True)
                continue
            srec = seed_rec["states"].get(st, {})
            if srec.get("branch_step") is None:
                _dump(dict(seed=seed, state=st, resolved=False, note="unresolved in registration",
                           provenance=_provenance(cfg, dict(phase="run", seed=seed, state=st))), rj)
                print(f"[run] s{seed} {st} UNRESOLVED (registration) -> skip", flush=True)
                continue
            ck, branch = cks[st], int(srec["branch_step"])
            t0 = time.time()
            fk = DSM.fixed_kick_state(S, ck, branch, cfg)                                  # P1
            ca, arrays = DSM.corrected_operator_audit(S, ck, branch, cfg, workers=workers, freeze=freeze)  # P2
            summ = dict(seed=seed, state=st, resolved=True, branch=branch, time_ms=branch * DT,
                        D=srec.get("D"), a=srec.get("a"), rate_hz=srec.get("rate_hz"),
                        src_g=[float(x) for x in S["src_g"]], snk_g=[float(x) for x in S["snk_g"]],
                        axis_g=[float(x) for x in S["axis_g"]],
                        fixed_kick={k: v for k, v in fk.items() if k != "arrays"},
                        operator_audit={k: v for k, v in ca.items() if k != "sigma1"},
                        operator_sigma1=ca.get("sigma1", {}),
                        wall_s=round(time.time() - t0, 1),
                        provenance=_provenance(cfg, dict(phase="run", seed=seed, state=st)))
            _dump(summ, rj)
            bundle = {f"fk_{k}": v for k, v in fk["arrays"].items()}
            bundle.update(arrays)                                                          # corr_u1/v1/K/Kr_*
            np.savez_compressed(os.path.join(OUT, "per_seed", f"arrays_seed{seed}_{st}.npz"), **bundle)
            print(f"[run] s{seed} {st} kick_norm={fk['response_norm']:.3g} identifiable={ca['identifiable']} "
                  f"disc={ca['linearity_discrepancy']:.3f} splithalf={ca['split_half_stability']:.3f} "
                  f"sat={ca['n_saturated_forks']}/{ca['n_forks']} ({summ['wall_s']}s)", flush=True)
        del S, cks
        gc.collect()


# ============================================================ P4 m-mechanism controls
def cmd_controls(args, cfg):
    import gc
    mc = cfg["m_controls"]
    reg_all = _load_json(REGPATH)
    if reg_all is None:
        raise SystemExit("controls require registration: run `register` first")
    for seed in _seeds(args, cfg):
        rj = os.path.join(OUT, "per_seed", f"controls_seed{seed}.json")
        if _state_done(rj, args.resume):
            print(f"[controls] resume skip s{seed}", flush=True)
            continue
        seed_rec = reg_all["seeds"].get(str(seed))
        if seed_rec is None:
            raise SystemExit(f"seed {seed} not registered")
        S = DSM.build_S(seed, cfg)
        DSM._ensure_flat(S)
        cks = _load_checkpoints(seed, seed_rec)
        rows, arrays = {}, {}
        for st in mc["states"]:
            srec = seed_rec["states"].get(st, {})
            if srec.get("branch_step") is None:
                rows[st] = dict(resolved=False)
                continue
            ck, branch = cks[st], int(srec["branch_step"])
            rows[st] = {}
            for cond in mc["conditions"]:
                ck_c = apply_m_control(ck, cond, S["NE"], seed=int(mc["shuffle_seed"]))
                fk = DSM.fixed_kick_state(S, ck_c, branch, cfg)
                rows[st][cond] = dict(response_norm=fk["response_norm"], censor=fk["censor"],
                                      region=fk["region"],
                                      distal_over_matched_off_axis=fk.get("distal_corridor_over_matched_off_axis"),
                                      cum_remote_over_source=fk.get("cum_remote_over_source_final"),
                                      arrival_eligible=(fk.get("arrival_fit") or {}).get("eligible"))
                arrays[f"{st}__{cond}__dY_full"] = fk["arrays"]["dY_full"]
                arrays[f"{st}__{cond}__dmaps"] = fk["arrays"]["dmaps"]
            print(f"[controls] s{seed} {st} norms " +
                  " ".join(f"{c}={rows[st][c]['response_norm']:.3g}" for c in mc["conditions"]), flush=True)
        _dump(dict(seed=seed, states=rows, m_control_states=mc["states"], conditions=mc["conditions"],
                   shuffle_seed=mc["shuffle_seed"],
                   provenance=_provenance(cfg, dict(phase="controls", seed=seed))), rj)
        np.savez_compressed(os.path.join(OUT, "per_seed", f"controls_arrays_seed{seed}.npz"), **arrays)
        del S, cks
        gc.collect()


# ============================================================ P3 mode tracking + aggregate + STATUS
def _state_arrays(seed, st):
    p = os.path.join(OUT, "per_seed", f"arrays_seed{seed}_{st}.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def _mode_tracking(cfg):
    """Cross-state mode trajectory (spec §4 P3): ONLY between adjacent states both strictly identifiable.
    Sign-invariant U1 overlap + leading-subspace principal angles + centroid displacement + axis /
    sigma_hat_1 change. Degenerate (gap<ratio) -> track the leading subspace, not a single vector."""
    grid = Grid(n=int(cfg["grid_n"]), L=float(cfg["L_norm"]))
    X, Y = grid.coords()
    ratio = float(cfg["degeneracy_ratio"])
    Tmid = int(round(cfg["T_windows_ms"][1]))
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "per_seed", "state_seed*_baseline.json"))):
        seed = int(os.path.basename(f).split("seed")[1].split("_")[0])
        js = {st: _load_json(os.path.join(OUT, "per_seed", f"state_seed{seed}_{st}.json")) for st in STATE_ORDER}
        for A, B in ADJACENT:
            jA, jB = js.get(A), js.get(B)
            if not (jA and jB and jA.get("resolved") and jB.get("resolved")):
                continue
            idA = (jA.get("operator_audit") or {}).get("identifiable")
            idB = (jB.get("operator_audit") or {}).get("identifiable")
            if not (idA and idB):
                rows.append(dict(seed=seed, pair=f"{A}->{B}", both_identifiable=False))
                continue
            aA, aB = _state_arrays(seed, A), _state_arrays(seed, B)
            kkey = f"corr_K_T{Tmid}"
            ukey = f"corr_u1_T{Tmid}"
            if aA is None or aB is None or kkey not in aA.files or kkey not in aB.files:
                continue
            sa, sb = leading_subspace(aA[kkey], ratio), leading_subspace(aB[kkey], ratio)
            dim = min(sa["subspace_dim"], sb["subspace_dim"])
            angles = principal_angles_deg(sa["U"][:, :dim], sb["U"][:, :dim])
            u1A, u1B = aA[ukey], aB[ukey]
            sigA = (jA.get("operator_sigma1") or {}).get(str(float(cfg["T_windows_ms"][1])), {})
            sigB = (jB.get("operator_sigma1") or {}).get(str(float(cfg["T_windows_ms"][1])), {})
            rows.append(dict(
                seed=seed, pair=f"{A}->{B}", both_identifiable=True,
                u1_overlap=subspace_alignment(u1A.ravel(), u1B.ravel()),
                principal_angles_deg=[float(x) for x in angles], max_principal_angle_deg=float(angles.max()),
                centroid_disp=centroid_displacement(u1A, u1B, X, Y),
                d_u1_axis=(sigB.get("u1_axis", np.nan) - sigA.get("u1_axis", np.nan)),
                d_sigma1=(sigB.get("sigma1", np.nan) - sigA.get("sigma1", np.nan)),
                degenerate=bool(sa["degenerate"] or sb["degenerate"]), tracked_dim=int(dim)))
    return rows


def cmd_aggregate(args, cfg):
    reg_all = _load_json(REGPATH) or {"seeds": {}}
    fk_rows, op_rows = [], []
    for f in sorted(glob.glob(os.path.join(OUT, "per_seed", "state_seed*.json"))):
        d = json.load(open(f))
        base = dict(seed=d["seed"], state=d["state"], resolved=d.get("resolved"))
        if not d.get("resolved"):
            fk_rows.append(base); op_rows.append(base); continue
        fk = d.get("fixed_kick", {})
        fk_rows.append(dict(base, D=d.get("D"), time_ms=d.get("time_ms"), censor=fk.get("censor"),
                            response_norm=fk.get("response_norm"),
                            distal_over_matched_off_axis=fk.get("distal_corridor_over_matched_off_axis"),
                            cum_remote_over_source=fk.get("cum_remote_over_source_final"),
                            arrival_eligible=(fk.get("arrival_fit") or {}).get("eligible"),
                            arrival_slope=(fk.get("arrival_fit") or {}).get("slope"),
                            arrival_r2=(fk.get("arrival_fit") or {}).get("r2"),
                            **{f"region_{k}": v for k, v in (fk.get("region") or {}).items()}))
        ca = d.get("operator_audit", {})
        row = dict(base, D=d.get("D"), time_ms=d.get("time_ms"), identifiable=ca.get("identifiable"),
                   discrepancy=ca.get("linearity_discrepancy"), disc_repeatA=ca.get("disc_repeatA"),
                   disc_repeatB=ca.get("disc_repeatB"), split_half_stability=ca.get("split_half_stability"),
                   n_saturated=ca.get("n_saturated_forks"), n_forks=ca.get("n_forks"))
        for Tms, sd in (d.get("operator_sigma1") or {}).items():
            T = int(float(Tms))
            row[f"sigma1_T{T}"] = sd.get("sigma1"); row[f"u1_axis_T{T}"] = sd.get("u1_axis")
            row[f"u1_glob_T{T}"] = sd.get("u1_globality"); row[f"u1_corridor_T{T}"] = sd.get("u1_corridor_frac")
            row[f"gap_T{T}"] = sd.get("gap")
        op_rows.append(row)
    mode_rows = _mode_tracking(cfg)
    n_ident = sum(1 for r in op_rows if r.get("identifiable"))
    _dump(dict(schema_version=SCHEMA_VERSION, rows=fk_rows,
               provenance=_provenance(cfg, dict(phase="aggregate"))),
          os.path.join(OUT, "fixed_kick_summary.json"))
    _dump(dict(schema_version=SCHEMA_VERSION, tol=float(cfg["linearity_tol"]), rows=op_rows,
               n_identifiable=n_ident, n_states=len([r for r in op_rows if r.get("resolved")]),
               mode_tracking=mode_rows, provenance=_provenance(cfg, dict(phase="aggregate"))),
          os.path.join(OUT, "operator_tracking_summary.json"))
    ctrl = sorted(glob.glob(os.path.join(OUT, "per_seed", "controls_seed*.json")))
    if ctrl:
        _dump(dict(schema_version=SCHEMA_VERSION, rows=[json.load(open(f)) for f in ctrl],
                   provenance=_provenance(cfg, dict(phase="aggregate"))),
              os.path.join(OUT, "controls_summary.json"))
    # numerical audit: basis orthonormality + parity summary + mass-conservation spot check
    P = real_fourier_basis_2d(int(cfg["grid_n"]))
    ortho = float(np.max(np.abs(P.T @ P - np.eye(P.shape[1]))))
    parity_summary = {s: rec.get("parity", {}).get("pass") for s, rec in reg_all.get("seeds", {}).items()}
    _dump(dict(basis_orthonormality_max_abs_resid=ortho, parity_pass_by_seed=parity_summary,
               n_states_resolved=len([r for r in op_rows if r.get("resolved")]), n_identifiable=n_ident,
               provenance=_provenance(cfg, dict(phase="numerical-audit"))),
          os.path.join(OUT, "numerical_audit.json"))
    # checkpoint manifest
    cks = sorted(glob.glob(os.path.join(CKDIR, "*.pkl")))
    manifest = [dict(file=os.path.relpath(p, ROOT), sha=DSM._file_hash(p),
                     size_mb=round(os.path.getsize(p) / 1e6, 1)) for p in cks]
    _dump(dict(checkpoints=manifest, provenance=_provenance(cfg, dict(phase="manifest"))),
          os.path.join(OUT, "checkpoint_manifest.json"))
    _dump(_provenance(cfg, dict(phase="aggregate", n_state_files=len(op_rows))),
          os.path.join(OUT, "provenance.json"))
    print(f"[aggregate] {len(op_rows)} state rows; {n_ident} identifiable; basis ortho={ortho:.1e}; "
          f"mode_tracking pairs={len(mode_rows)}; controls={len(ctrl)}; parity={parity_summary}", flush=True)


# ============================================================ tiny-net smoke (glue, no 40k substrate)
def cmd_smoke(args, cfg):
    scfg = dict(cfg)
    scfg["grid_n"] = 6; scfg["window_ms"] = 20.0
    scfg["T_windows_ms"] = [5.0, 10.0, 20.0]; scfg["local_map_centers_ms"] = [5.0, 10.0, 20.0]
    scfg["corrected_audit"] = {**cfg["corrected_audit"], "n_realizations": 4}
    S = DSM._tiny_S(scfg, grid_n=6)
    DSM._ensure_flat(S)
    eta_m = eta_m_from_frac(0.001, float(cfg["I_EE_scale"]), 36.6036014019694)
    zm_cfg = MZSlowVarsConfig(use_z=True, use_m=True, I_th_EI=5.0, tau_z=3000.0, tau_adp=2000.0, eta_m=eta_m)
    n = 3000
    slow = MZSpatialProbe(S["N"], 18.0, zm_cfg, NE=S["NE"])
    S["net"]["rng"] = np.random.default_rng(1)
    res = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=n, capture_final=False, store_spikes=False)
    D = 1.0 - np.asarray(slow.trace_z_mean, float)
    a = np.asarray(slow.trace_adap_current, float) / float(cfg["I_EE_scale"])
    reg = register_states(D, a, res["rate_E"], DT, baseline_ms=50.0, baseline_search_halfwidth_ms=20.0,
                          approach_fracs=[0.25, 0.5, 0.75], approach_search_ms=30.0, settle_tail_ms=50.0,
                          resting_win_ms=5.0, resting_k=0.3, settled_D_ptp_max=1.0, settled_a_ptp_max=1.0,
                          settled_min_resting_frac=0.0, D_onset_ref=0.0)
    print(f"[smoke] D_base={reg['D_base']:.4f} D_plateau={reg['D_plateau']:.4f} "
          f"steps={[reg['states'][s]['branch_step'] for s in STATE_ORDER]}", flush=True)
    cks, fps = _capture_checkpoints(S, zm_cfg, reg["states"], seed=1, persist=False)   # smoke: no CKDIR write
    st = "baseline" if reg["states"]["baseline"]["branch_step"] is not None else next(iter(cks))
    ck, br = cks[st], int(reg["states"][st]["branch_step"])
    fk = DSM.fixed_kick_state(S, ck, br, scfg)
    ca, arrays = DSM.corrected_operator_audit(S, ck, br, scfg, workers=int(args.workers or 2))
    print(f"[smoke] fixed_kick norm={fk['response_norm']:.4g} censor={fk['censor']} | "
          f"audit n_modes={ca['n_modes']} n_real={ca['n_realizations']} disc={ca['linearity_discrepancy']:.3f} "
          f"identifiable={ca['identifiable']}", flush=True)
    norms = {}
    for cond in cfg["m_controls"]["conditions"]:
        ck_c = apply_m_control(ck, cond, S["NE"], seed=int(cfg["m_controls"]["shuffle_seed"]))
        norms[cond] = DSM.fixed_kick_state(S, ck_c, br, scfg)["response_norm"]
    crn = np.isclose(norms["native_zm"], DSM.fixed_kick_state(S, ck, br, scfg)["response_norm"])
    print(f"[smoke] m-controls norms={ {k: round(v,3) for k,v in norms.items()} } native_CRN={bool(crn)}", flush=True)
    ok = fk["censor"] in ("resolved", "right_censored_native_transition") and ca["n_modes"] == 9 and bool(crn)
    print("[smoke] PASS" if ok else "[smoke] CHECK", flush=True)


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Topic 4 MZ state-aligned finite-time spatial mode tracking.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "register", "run", "controls", "aggregate"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true")
        sp.add_argument("--seeds", default=None)
        sp.add_argument("--states", default=None)
        sp.add_argument("--workers", default=None)
        sp.add_argument("--realizations", default=None, help="override n_realizations (cheap-first smoke)")
        sp.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    if args.cmd in {"smoke", "register", "run", "controls"} and not args.confirm_run:
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run.", file=sys.stderr)
        sys.exit(2)
    os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)
    {"smoke": cmd_smoke, "register": cmd_register, "run": cmd_run, "controls": cmd_controls,
     "aggregate": cmd_aggregate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
