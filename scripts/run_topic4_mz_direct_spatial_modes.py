"""Topic 4 — MZ full-SNN direct spatial mode dynamics: scientific runner.

*** THIS RUNS SIMULATIONS. *** Nothing runs on import; every sim subcommand is gated by --confirm-run.
Design contract (BINDING): docs/superpowers/specs/2026-07-19-topic4-mz-direct-spatial-modes-design.md

Directly perturbs the COMPLETE current-based MZ spiking network (≈40k E/I LIF), NOT the frozen-q
rate-field surrogate. Reuse (not reinvent): run_m4_phaseplane.build_substrate (E1146 narrow /
template_source / twoend_equal), src.topic4_mz_onset_dynamics.{MZ*Probe, run_loop, LoopState,
score_runaway} (checkpoint/resume + freeze), src.topic4_mz_direct_spatial_modes (pure operator /
carrier / readout math), src.topic4_state_conditioned_susceptibility.make_phase_paired_probe_dictionary
(the SAME Gabor dictionary, for the phase-paired probe comparison — evaluated through the empirical
operator, NOT the forbidden frozen-q Jacobian).

Subcommands (all resumable via per-seed/per-state JSON + checkpoint pickles; re-run is idempotent):
  smoke    tiny-net end-to-end (native parity + fork CRN + operator SVD sanity + fixed kick) — fast
  run      full-density P0: replay -> checkpoints -> linearity/eps -> fixed-kick + 144-dim operator
  controls P1: z+m plateau + D-matched z-only (same operator + fixed kick at the matched states)
  aggregate combine per-seed/per-state -> summaries + provenance + numerical audit
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse            # noqa: E402
import copy                # noqa: E402
import glob                # noqa: E402
import hashlib             # noqa: E402
import json                # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle              # noqa: E402
import sys                 # noqa: E402
import time                # noqa: E402

import numpy as np         # noqa: E402
import yaml                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

from src.topic4_mz_onset_dynamics import run_loop, score_runaway, _loop_consts  # noqa: E402
from src.topic4_mz_slowvars import eta_m_from_frac  # noqa: E402
from mz_slow_vars import MZSlowVarsConfig  # noqa: E402
from src.topic4_state_conditioned_susceptibility import (  # noqa: E402
    normalize_subject_coordinates, make_phase_paired_probe_dictionary,
)
from src.topic4_mz_direct_spatial_modes import (  # noqa: E402
    SCHEMA_VERSION, MZSpatialProbe, build_grid_readout, grid_pattern_to_current,
    spikes_to_rate_grid, local_window_maps, real_fourier_basis_2d, balanced_lowk_indices,
    central_difference, build_empirical_operator, field_globality, field_axis_alignment,
    normalized_field_overlap, gaussian_current_field, response_norm, region_response,
    cumulative_response_ratio, axis_kymograph, first_arrival_times, fit_arrival_distance,
    threshold_sensitivity_arrivals, linearity_discrepancy, select_epsilon, right_censoring_label,
    robust_identifiability_gate,
)

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_direct_spatial_modes")
CKDIR = os.path.join(OUT, "per_seed", "checkpoints")
CFG_PATH = os.path.join(ROOT, "config", "topic4_mz_direct_spatial_modes.yaml")
DT = 0.1
_GUARDED = ("kick_probe.py", "params.py", "model.py", "connectivity.py", "connectivity_rot.py", "lfp.py")


# ============================================================ config + provenance (mirror onset runner)
def load_cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)


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


def _engine_shas():
    eng = os.path.join(ROOT, "src", "snn_engine")
    return {f: _file_hash(os.path.join(eng, f)) for f in _GUARDED}


def _provenance(cfg, extra=None):
    prov = dict(schema_version=SCHEMA_VERSION, git_sha=_git_sha(), engine_shas=_engine_shas(),
                config_hash=_file_hash(CFG_PATH),
                module_hash=_file_hash(os.path.join(ROOT, "src", "topic4_mz_direct_spatial_modes.py")),
                argv=sys.argv, subject=cfg["subject"], montage=cfg["montage"], dt=DT)
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
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)                                  # atomic (no partial-file race)


# ============================================================ substrate + readout + regions
def build_S(seed, cfg):
    """Full E1146 substrate + coarse-grid readout + grid-space source/sink/axis + region masks."""
    import run_m4_phaseplane as PP
    S = PP.build_substrate(seed)
    S["seed"] = seed
    S["net"]["rng"] = np.random.default_rng(seed)          # the rng OBJECT must exist for run_loop
                                                           # (fresh replay re-seeds it; forks overwrite its state from the checkpoint)
    ro = build_grid_readout(S["posE"], grid_n=int(cfg["grid_n"]), L_phys=float(cfg["L_phys"]),
                            L_norm=float(cfg["L_norm"]), center_phys=cfg["center_phys"])
    src_g, _ = normalize_subject_coordinates(S["src_xy"][None, :], L_phys=cfg["L_phys"],
                                             L_norm=cfg["L_norm"], center_phys=cfg["center_phys"])
    snk_g, _ = normalize_subject_coordinates(S["snk_xy"][None, :], L_phys=cfg["L_phys"],
                                             L_norm=cfg["L_norm"], center_phys=cfg["center_phys"])
    src_g, snk_g = src_g[0], snk_g[0]
    axis_g = (snk_g - src_g) / np.linalg.norm(snk_g - src_g)
    regions = grid_region_masks(ro, src_g, snk_g, axis_g, float(cfg["core_radius_norm"]),
                                float(cfg["corridor_halfwidth_norm"]))
    S["readout"] = ro
    S["src_g"] = src_g
    S["snk_g"] = snk_g
    S["axis_g"] = axis_g
    S["regions"] = regions
    return S


def grid_region_masks(ro, src_g, snk_g, axis_g, core_r, corridor_hw):
    """(n,n) boolean grid masks: source_core, remote_sink, axis_corridor, off_axis, and (review
    2026-07-20) DISTAL_corridor (corridor beyond the source half, toward the sink) + MATCHED_off_axis
    (an off-axis band at the same distal along-axis distance) — to separate "reaches down the corridor
    toward sink" from "local + isotropic off-axis spread"."""
    X, Y = ro.grid.coords()
    pts = np.column_stack([X.ravel(), Y.ravel()])
    d_src = np.linalg.norm(pts - src_g, axis=1)
    d_snk = np.linalg.norm(pts - snk_g, axis=1)
    u = axis_g / np.linalg.norm(axis_g)
    up = np.array([-u[1], u[0]])
    rel = pts - src_g
    proj = rel @ u
    perp = np.abs(rel @ up)
    axis_len = float(np.linalg.norm(snk_g - src_g))
    corridor = (perp <= corridor_hw) & (proj >= -core_r) & (proj <= axis_len + core_r)
    distal = (proj >= 0.5 * axis_len) & (proj <= axis_len + core_r)     # far half toward sink
    distal_corridor = corridor & distal
    matched_off_axis = (perp > corridor_hw) & (perp <= 2.0 * corridor_hw) & distal   # off-axis, matched distance
    n = ro.n
    rs = lambda m: m.reshape(n, n)
    return dict(source_core=rs(d_src <= core_r), remote_sink=rs(d_snk <= core_r),
                axis_corridor=rs(corridor), off_axis=rs(~corridor),
                distal_corridor=rs(distal_corridor), matched_off_axis=rs(matched_off_axis))


def _cand(cfg, which):
    c = cfg["candidates"][which]
    return c["label"], MZSlowVarsConfig(**c["cfg"]), {int(k): float(v) for k, v in c["onset_ms"].items()}


def state_branches(onset_ms, cfg):
    """Registered branch STEP index for each primary temporal state (spec §1)."""
    o = float(onset_ms)
    return {"baseline": int(round(cfg["baseline_ms"] / DT)),
            "midpoint": int(round(cfg["mid_fraction"] * o / DT)),
            "pre_onset": int(round((o - cfg["pre_onset_100_ms"]) / DT))}


# ============================================================ replay -> checkpoints (segmented; persisted)
def _ck_path(label, seed, state):
    return os.path.join(CKDIR, f"ck_{label}_seed{seed}_{state}.pkl")


def replay_checkpoints(S, mzcfg, branches, label, *, resume=True, verbose=True):
    """Segmented native replay: capture the checkpoint at each ordered branch, reusing run_loop's
    resume so the whole trajectory is replayed only ONCE (spec §2.1). Checkpoints are persisted so a
    resumed run skips the (~90k-step) replay. The replay slow is PLAIN (natural z/m); freeze is armed
    per-fork downstream."""
    os.makedirs(CKDIR, exist_ok=True)
    order = sorted(branches, key=lambda k: branches[k])
    seed = S["seed"]
    cks, need = {}, []
    for st in order:
        p = _ck_path(label, seed, st)
        if resume and os.path.exists(p):
            with open(p, "rb") as f:
                cks[st] = pickle.load(f)
        else:
            need.append(st)
    if not need:
        if verbose:
            print(f"[replay] seed{seed} all {len(order)} checkpoints cached", flush=True)
        return cks
    # replay from t=0 (or the latest cached checkpoint before the first needed branch)
    t0 = time.time()
    done = [st for st in order if st in cks]
    start_ck = cks[done[-1]] if done else None
    cur_t = branches[done[-1]] if done else 0
    slow = copy.deepcopy(start_ck.slow) if start_ck is not None else \
        MZSpatialProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=S["regions"]["source_core"].any() and None)
    if start_ck is None:
        slow = MZSpatialProbe(S["N"], 18.0, mzcfg, NE=S["NE"])
        S["net"]["rng"] = np.random.default_rng(seed)
    for st in order:
        if st in cks:
            continue
        n = branches[st] - cur_t
        rep = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=n, start=start_ck,
                       capture_final=True, store_spikes=False)
        ck = rep["checkpoint"]
        cks[st] = ck
        with open(_ck_path(label, seed, st), "wb") as f:
            pickle.dump(ck, f)
        if verbose:
            print(f"[replay] seed{seed} -> {st}@{branches[st]} ({time.time()-t0:.0f}s)", flush=True)
        start_ck = ck
        slow = copy.deepcopy(ck.slow)
        cur_t = branches[st]
    return cks


def _ensure_flat(S):
    """Cache the flattened source-indexed edges in the parent BEFORE forking workers (COW share),
    without advancing the RNG (a resumed run skips the replay that would otherwise cache them)."""
    if "ampa_flat" not in S["net"]:
        _loop_consts(S["p"], S["net"])


# ============================================================ single fork (frozen fast-subsystem)
def _fork_run(S, ck, branch, pattern_E, *, window_steps, cur_dur_steps, freeze, store_spikes):
    """Fork from the checkpoint: arm freeze (fast-subsystem isolation) + optional additive-current
    pulse, continue `window_steps` under the SAME noise (run_loop resets rng to ck.rng_state -> common
    random numbers). Never early-stops (the full window is needed for Y_T)."""
    s = copy.deepcopy(ck.slow)
    if not isinstance(s, MZSpatialProbe):                  # a plain-onset checkpoint -> wrap for the schedule
        s = _as_spatial_probe(s, S)
    s.set_branch(branch_step=branch, freeze=freeze)
    if pattern_E is not None:
        s.set_current_schedule(lo=branch, hi=branch + cur_dur_steps, pattern_E=pattern_E)
    return run_loop(S["p"], S["net"], s, S["vth"], n_steps=window_steps, start=ck,
                    store_spikes=store_spikes, early_stop_runaway=False)


def _as_spatial_probe(slow, S):
    """Rebuild an MZSpatialProbe carrying slow's z/m/_step_i (checkpoints captured with MZOnsetProbe)."""
    sp = MZSpatialProbe(S["N"], 18.0, slow.cfg, NE=S["NE"])
    sp.z = slow.z.copy(); sp.m = slow.m.copy()
    sp._I_I_last = slow._I_I_last.copy()
    sp._step_i = slow._step_i
    return sp


def _fork_Y(S, ck, branch, pattern_E, window_steps, cur_dur_steps, T_steps_list, readout, freeze,
            run_hz, run_dur):
    """Fork -> {T_steps: Y_T (144-vector = mean E rate per grid cell over [0,T])} + runaway flag."""
    res = _fork_run(S, ck, branch, pattern_E, window_steps=window_steps, cur_dur_steps=cur_dur_steps,
                    freeze=freeze, store_spikes=True)
    spk = res["E_spk_bool"]
    Ys = {}
    for T_steps in T_steps_list:
        hi = min(T_steps, spk.shape[0])
        Ys[T_steps] = spikes_to_rate_grid(spk[:hi], readout, dt_ms=DT)["rate_hz"].ravel()
    ra = score_runaway(res["rate_E"], DT, thresh_hz=run_hz, dur_ms=run_dur)
    return Ys, ra


# ============================================================ parallel operator (COW workers)
_WS = None                                                 # worker state (inherited via fork, never pickled)


def _op_task(args):
    j, sign = args
    W = _WS
    ro = W["readout"]
    pat2d = W["basis"][:, j].reshape(W["n"], W["n"])
    cur = sign * W["eps"] * W["I_EE_scale"] * grid_pattern_to_current(pat2d, ro)
    Ys, ra = _fork_Y(W["S"], W["ck"], W["branch"], cur, W["window_steps"], W["cur_dur_steps"],
                     W["T_steps_list"], ro, W["freeze"], W["run_hz"], W["run_dur"])
    return (j, int(sign), {int(T): Ys[T] for T in Ys}, ra is not None)


def _parallel_map(fn, tasks, worker_state, workers):
    global _WS
    _WS = worker_state
    if workers <= 1:
        return [fn(t) for t in tasks]
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        return pool.map(fn, tasks, chunksize=max(1, len(tasks) // (workers * 4)))


# ============================================================ operator for one state
def operator_for_state(S, ck, branch, cfg, eps, *, workers, freeze=True):
    """Full 144-dim empirical operator at one state: +/-eps forks over the complete real Fourier basis
    -> K_T -> SVD -> sigma_hat_1 / V1 / U1 per T. Fails closed to right-censored when the no-probe
    control runs away in the window (spec §1)."""
    ro = S["readout"]
    n = ro.n
    basis = real_fourier_basis_2d(n)
    T_steps_list = [int(round(t / DT)) for t in cfg["T_windows_ms"]]
    cur_dur = int(round(cfg["current_dur_ms"] / DT))
    window_steps = int(round(cfg["window_ms"] / DT))
    run_hz, run_dur = float(cfg["runaway_hz"]), float(cfg["saturation_dur_ms"])  # within-window saturation (fits window_ms)

    # no-probe control (right-censoring + fixed-kick reference)
    Y0, ra0 = _fork_Y(S, ck, branch, None, window_steps, cur_dur, T_steps_list, ro, freeze, run_hz, run_dur)
    censor = right_censoring_label(ra0)
    if censor != "resolved":
        return dict(censor=censor, sigma1={}, arrays={}, Y0={int(t): Y0[t] for t in Y0}), None

    ws = dict(S=S, ck=ck, branch=branch, readout=ro, basis=basis, n=n, eps=eps,
              I_EE_scale=float(cfg["I_EE_scale"]), window_steps=window_steps, T_steps_list=T_steps_list,
              cur_dur_steps=cur_dur, freeze=freeze, run_hz=run_hz, run_dur=run_dur)
    tasks = [(j, s) for j in range(basis.shape[1]) for s in (+1, -1)]
    t0 = time.time()
    res = _parallel_map(_op_task, tasks, ws, workers)
    Yp = {int(round(t / DT)): [None] * basis.shape[1] for t in cfg["T_windows_ms"]}
    Ym = {int(round(t / DT)): [None] * basis.shape[1] for t in cfg["T_windows_ms"]}
    any_runaway = False
    for j, sign, Ys, ranaway in res:
        any_runaway = any_runaway or ranaway
        tgt = Yp if sign > 0 else Ym
        for T, v in Ys.items():
            tgt[T][j] = v

    out = dict(censor="resolved", sigma1={}, arrays={}, Y0={int(t): Y0[t] for t in Y0},
               any_fork_saturation=bool(any_runaway), wall_s=round(time.time() - t0, 1))
    for T_ms in cfg["T_windows_ms"]:
        Ts = int(round(T_ms / DT))
        K = np.column_stack([central_difference(Yp[Ts][j], Ym[Ts][j], eps) for j in range(basis.shape[1])])
        op = build_empirical_operator(K, basis, grid_n=n)
        out["sigma1"][float(T_ms)] = dict(
            sigma1=op["sigma1"], gap=op["gap"], degenerate=op["degenerate"], subspace_dim=op["subspace_dim"],
            singular_values=op["singular_values"],
            u1_axis=field_axis_alignment(op["u1_field"], ro, S["axis_g"]),
            u1_globality=field_globality(op["u1_field"]),
            v1_axis=field_axis_alignment(op["v1_field"], ro, S["axis_g"]))
        out["arrays"][f"K_T{int(T_ms)}"] = K
        out["arrays"][f"u1_T{int(T_ms)}"] = op["u1_field"]
        out["arrays"][f"v1_T{int(T_ms)}"] = op["v1_field"]
        out["arrays"][f"M_T{int(T_ms)}"] = K @ basis.T
    return out, basis


# ============================================================ linearity audit + eps selection
def linearity_audit_state(S, ck, branch, cfg, *, workers, freeze=True):
    """eps-ladder audit on a low-k pattern subset: for each ladder eps, form K(eps) and K(eps/2) and
    report ||K(eps)-K(eps/2)||/||K(eps/2)|| + saturation (spec §2.3)."""
    ro = S["readout"]
    n = ro.n
    basis = real_fourier_basis_2d(n)
    n_sub = int(cfg["linearity_audit_n_patterns"])
    subset = list(range(min(n_sub, basis.shape[1])))       # leading (low-k) patterns
    T_ms = float(cfg["T_windows_ms"][1])                    # audit at the mid window (30 ms)
    Ts = [int(round(T_ms / DT))]
    cur_dur = int(round(cfg["current_dur_ms"] / DT))
    window_steps = int(round(cfg["window_ms"] / DT))
    run_hz, run_dur = float(cfg["runaway_hz"]), float(cfg["saturation_dur_ms"])  # within-window saturation (fits window_ms)
    I_EE = float(cfg["I_EE_scale"])
    ladder = [float(a) for a in cfg["amplitude_ladder"]]
    discreps, saturated = [], []
    for a in ladder:
        Kf, Kh = [], []
        sat = False
        for scale, store in ((a, Kf), (a / 2.0, Kh)):
            ws = dict(S=S, ck=ck, branch=branch, readout=ro, basis=basis, n=n, eps=scale,
                      I_EE_scale=I_EE, window_steps=window_steps, T_steps_list=Ts, cur_dur_steps=cur_dur,
                      freeze=freeze, run_hz=run_hz, run_dur=run_dur)
            res = _parallel_map(_op_task, [(j, s) for j in subset for s in (+1, -1)], ws, workers)
            yp = {j: None for j in subset}; ym = {j: None for j in subset}
            for j, sign, Ys, ra in res:
                (yp if sign > 0 else ym)[j] = Ys[Ts[0]]
                sat = sat or ra
            store.append(np.column_stack([central_difference(yp[j], ym[j], scale) for j in subset]))
        discreps.append(linearity_discrepancy(Kf[0], Kh[0]))
        saturated.append(bool(sat))
    sel = select_epsilon(ladder, discreps, saturated, tol=float(cfg["linearity_tol"]))
    return dict(ladder=ladder, discrepancies=discreps, saturated=saturated, selection=sel,
                T_ms=T_ms, n_patterns=len(subset))


# ============================================================ fixed localized kick (spec §3.1)
def fixed_kick_state(S, ck, branch, cfg):
    """Fixed source-core Gaussian POSITIVE current kick vs no-probe: 5/15/30/50 ms local maps, axis
    kymograph, arrival-vs-distance, region/response readouts (spec §4)."""
    ro = S["readout"]
    fk = cfg["fixed_kick"]
    window_steps = int(round(cfg["window_ms"] / DT))
    cur_dur = int(round(cfg["current_dur_ms"] / DT))
    kick2d = gaussian_current_field(ro, center_norm=tuple(S["src_g"]), sigma=float(fk["sigma_norm"]),
                                    rms=float(fk["frac"]) * float(cfg["I_EE_scale"]))
    kick_E = grid_pattern_to_current(kick2d, ro)
    run_hz, run_dur = float(cfg["runaway_hz"]), float(cfg["saturation_dur_ms"])  # within-window saturation (fits window_ms)

    res0 = _fork_run(S, ck, branch, None, window_steps=window_steps, cur_dur_steps=cur_dur,
                     freeze=True, store_spikes=True)
    resK = _fork_run(S, ck, branch, kick_E, window_steps=window_steps, cur_dur_steps=cur_dur,
                     freeze=True, store_spikes=True)
    ra0 = score_runaway(res0["rate_E"], DT, thresh_hz=run_hz, dur_ms=run_dur)
    raK = score_runaway(resK["rate_E"], DT, thresh_hz=run_hz, dur_ms=run_dur)
    censor = right_censoring_label(ra0)                    # no-probe control fate (right-censoring)

    centers = [float(c) for c in cfg["local_map_centers_ms"]]
    m0 = local_window_maps(res0["E_spk_bool"], ro, dt_ms=DT, centers_ms=centers,
                           width_ms=float(cfg["local_map_width_ms"]))
    mK = local_window_maps(resK["E_spk_bool"], ro, dt_ms=DT, centers_ms=centers,
                           width_ms=float(cfg["local_map_width_ms"]))
    dmaps = {c: np.nan_to_num(mK[c]) - np.nan_to_num(m0[c]) for c in centers}

    stack0 = _time_stack(res0["E_spk_bool"], ro, bin_ms=1.0)
    stackK = _time_stack(resK["E_spk_bool"], ro, bin_ms=1.0)
    dstack = stackK - stack0
    ky = axis_kymograph(dstack, ro, axis_unit=S["axis_g"], src_norm=tuple(S["src_g"]),
                        snk_norm=tuple(S["snk_g"]), band=float(fk["kymograph_band_norm"]),
                        n_pos=int(fk["kymograph_n_pos"]))
    dY_full = np.nan_to_num(_cum_rate(resK["E_spk_bool"], ro) - _cum_rate(res0["E_spk_bool"], ro))
    reg = region_response(dY_full, S["regions"])
    src_series = np.abs(dstack.reshape(dstack.shape[0], -1)[:, S["regions"]["source_core"].ravel()]).sum(1)
    rem_series = np.abs(dstack.reshape(dstack.shape[0], -1)[:, S["regions"]["remote_sink"].ravel()]).sum(1)
    cumratio = cumulative_response_ratio(rem_series, src_series)
    # arrival is only defined when the kymograph peak clears an absolute response floor; below it the
    # 0.1*max threshold sits in quantization noise and fabricates a spurious front (review 2026-07-20).
    kymo_max = float(np.nanmax(np.abs(ky["kymo"])))
    min_peak = float(fk.get("arrival_min_peak_hz", 2.0))
    r2_min = float(fk.get("arrival_r2_min", 0.5))
    dir_consistent = None
    if kymo_max < min_peak:
        arrivals = np.full(ky["kymo"].shape[1], np.nan)
        fit = dict(eligible=False, n_points=0, slope=None, velocity_proxy=None, r2=None, below_floor=True)
        thr = {}
    else:
        arrivals = first_arrival_times(ky["kymo"], ky["times"], threshold=max(0.1 * kymo_max, min_peak))
        fit = fit_arrival_distance(ky["distances"], arrivals, r2_min=r2_min)
        thr = threshold_sensitivity_arrivals(ky["kymo"], ky["times"], ky["distances"],
                                             fracs=[float(x) for x in fk["arrival_thresh_fracs"]], r2_min=r2_min)
        # direction consistency: eligible + positive slope at EVERY sampled threshold (not a threshold artifact)
        dir_consistent = bool(thr and all(v.get("eligible") and (v.get("slope") or 0) > 0 for v in thr.values()))
    return dict(
        censor=censor, kick_induced_saturation_ms=raK, response_norm=response_norm(dY_full), region=reg,
        distal_corridor_over_matched_off_axis=(
            float(reg["distal_corridor"] / reg["matched_off_axis"])
            if reg.get("matched_off_axis") and reg["matched_off_axis"] > 0 else None),
        cum_remote_over_source_final=float(cumratio[-1]) if cumratio.size else float("nan"),
        arrival_fit=fit, arrival_direction_consistent=dir_consistent,
        arrival_threshold_sensitivity={str(k): v for k, v in thr.items()},
        arrays=dict(dmaps=np.stack([dmaps[c] for c in centers]), map_centers=np.array(centers),
                    kymo=ky["kymo"], kymo_dist=ky["distances"], kymo_times=ky["times"],
                    dY_full=dY_full, kick_field=kick2d, cumratio=cumratio))


def _time_stack(E_spk_bool, ro, bin_ms):
    spk = np.asarray(E_spk_bool)
    step = int(round(bin_ms / DT))
    n_t = spk.shape[0] // step
    out = np.zeros((n_t, ro.n, ro.n))
    for b in range(n_t):
        out[b] = np.nan_to_num(spikes_to_rate_grid(spk[b * step:(b + 1) * step], ro, dt_ms=DT)["rate_hz"])
    return out


def _cum_rate(E_spk_bool, ro):
    return spikes_to_rate_grid(E_spk_bool, ro, dt_ms=DT)["rate_hz"]


# ============================================================ Gabor probe scan from the operator (spec §3.3)
def gabor_scan_from_operator(M, S, cfg):
    """phase-paired probe susceptibility = ||M @ probe|| / ||probe|| for the SAME source-centered Gabor
    dictionary, evaluated through the EMPIRICAL operator M (NOT a frozen-q Jacobian). Named distinctly;
    its max is <= sigma_hat_1 by construction (control #7)."""
    g = cfg["gabor"]
    probes = make_phase_paired_probe_dictionary(S["readout"].grid, p_max=int(g["p_max"]),
                                                sigma=float(g["sigma"]), center=tuple(S["src_g"]))
    by_pq = {}
    glob = None
    for pr in probes:
        p = pr["field"].ravel()
        nrm = np.linalg.norm(p)
        gain = float(np.linalg.norm(M @ p) / nrm) if nrm > 0 else 0.0
        if pr["phase"] == "global":
            glob = gain
            continue
        by_pq.setdefault((pr["p"], pr["q"]), {})[pr["phase"]] = (gain, pr)
    paired = {}
    for pq, d in by_pq.items():
        gc = d.get("cos", (0.0, None))[0]
        gs = d.get("sin", (0.0, None))[0]
        pr = (d.get("cos") or d.get("sin"))[1]
        paired[pq] = dict(gain=float(np.hypot(gc, gs)), k_mag=pr["k_mag"], p=pr["p"], q=pr["q"])
    axial = max((v["gain"] for v in paired.values() if v["q"] == 0), default=0.0)
    perp = max((v["gain"] for v in paired.values() if v["p"] == 0), default=0.0)
    peak_pq = max(paired, key=lambda k: paired[k]["gain"]) if paired else None
    return dict(axial_gain=float(axial), perp_gain=float(perp), global_gain=float(glob or 0.0),
                axis_minus_perp=float(axial - perp), peak_pq=list(peak_pq) if peak_pq else None,
                peak_gain=float(paired[peak_pq]["gain"]) if peak_pq else 0.0,
                gains={f"{p},{q}": v["gain"] for (p, q), v in paired.items()})


# ============================================================ corrected identifiability audit (review 2026-07-20)
def _realization_state(base_seed, seed, branch, r):
    """Independent continuation-noise future r: a fresh PCG64 state, distinct per (seed,state,r)."""
    return np.random.default_rng(np.random.SeedSequence([int(base_seed), int(seed), int(branch), int(r)])).bit_generator.state


def _corr_task(args):
    j, sign, r, ai = args
    W = _WS
    ro = W["readout"]
    pat2d = W["P_rms"][:, j].reshape(W["n"], W["n"])          # unit per-cell-RMS low-k pattern
    cur = sign * W["amps"][ai] * grid_pattern_to_current(pat2d, ro)   # per-cell RMS current = amps[ai]
    ck_r = copy.copy(W["ck"]); ck_r.rng_state = W["real_states"][r]   # same branch state, realization-r future noise
    Ys, ra = _fork_Y(W["S"], ck_r, W["branch"], cur, W["window_steps"], W["cur_dur_steps"],
                     W["T_steps_list"], ro, W["freeze"], W["run_hz"], W["run_dur"])
    return (j, int(sign), r, ai, {int(T): Ys[T] for T in Ys}, ra is not None)   # ra -> within-window saturation


def corrected_operator_audit(S, ck, branch, cfg, *, workers, freeze=True):
    """Corrected identifiability audit (review 2026-07-20): balanced symmetric low-k modes, per-grid RMS
    matched to the fixed kick, ENSEMBLE-averaged over independent continuation-noise realizations (each +/-
    pair shares its future = CRN). Tests whether an ensemble linear-response operator IS identifiable at
    THIS (seed,state) once quantization noise is averaged out and the input strength matches the kick.
    Only if identifiable does it SVD -> sigma1/V1/U1 (over the low-k probed subspace)."""
    ca = cfg["corrected_audit"]
    ro = S["readout"]
    n = ro.n
    basis = real_fourier_basis_2d(n)
    lowk = balanced_lowk_indices(n, int(ca["k_max"]))
    P_lowk = basis[:, lowk]                                    # n_bins x n_modes (orthonormal columns)
    P_rms = P_lowk * float(np.sqrt(n * n))                    # unit per-cell RMS patterns
    I_EE = float(cfg["I_EE_scale"])
    a_base = float(ca["strength_frac"])
    amps = [a_base * I_EE, (a_base / 2.0) * I_EE]             # per-cell RMS current: base, half (linearity check)
    N = int(ca["n_realizations"])
    real_states = [_realization_state(ca["realization_base_seed"], S["seed"], branch, r) for r in range(N)]
    T_steps_list = [int(round(t / DT)) for t in cfg["T_windows_ms"]]
    Tmid = int(round(cfg["T_windows_ms"][1] / DT))
    ws = dict(S=S, ck=ck, branch=branch, readout=ro, P_rms=P_rms, n=n, amps=amps, real_states=real_states,
              window_steps=int(round(cfg["window_ms"] / DT)), cur_dur_steps=int(round(cfg["current_dur_ms"] / DT)),
              T_steps_list=T_steps_list, freeze=freeze, run_hz=float(cfg["runaway_hz"]),
              run_dur=float(cfg["saturation_dur_ms"]))
    n_modes = len(lowk)
    tol = float(cfg["linearity_tol"])
    tasks = [(j, s, r, ai) for j in range(n_modes) for s in (+1, -1) for r in range(N) for ai in (0, 1)]
    t0 = time.time()
    res = _parallel_map(_corr_task, tasks, ws, workers)
    # PER-REALIZATION responses (review round-2): keep each realization separate so we can post-hoc
    # average any subset (N=4/8/16 convergence) or two independent 8-realization halves, AND register
    # within-window saturation (a saturated fork = left the linear regime = contaminates the estimate).
    Yr = {}                                                    # (r, ai, sign, T) -> [n_modes arrays]
    n_sat = 0
    for j, sign, r, ai, Ys, sat in res:
        n_sat += int(sat)
        for T, y in Ys.items():
            Yr.setdefault((r, ai, sign, T), [None] * n_modes)[j] = y

    def Kr(r, ai, T):                                          # per-realization central-difference K
        yp, ym = Yr[(r, ai, +1, T)], Yr[(r, ai, -1, T)]
        return np.column_stack([(yp[j] - ym[j]) / (2.0 * amps[ai]) for j in range(n_modes)])

    def ensK(R, ai, T):                                       # ensemble-mean K over realization subset R
        return np.mean([Kr(r, ai, T) for r in R], axis=0)

    def disc_over(R):                                         # linearity discrepancy on ensemble R at Tmid
        return float(linearity_discrepancy(ensK(R, 0, Tmid), ensK(R, 1, Tmid)))

    allR = list(range(N))
    conv = {str(nn): disc_over(list(range(nn))) for nn in (4, 8, N) if nn <= N}   # N-convergence
    half = N // 2
    repA, repB = list(range(half)), list(range(half, N))      # two independent halves
    discA, discB = disc_over(repA), disc_over(repB)
    disc_full = disc_over(allR)
    split_half_stability = float(linearity_discrepancy(ensK(repA, 0, Tmid), ensK(repB, 0, Tmid)))
    any_sat = bool(n_sat > 0)
    # STRICT identifiable: full-N + both independent halves are amplitude-consistent, the two halves
    # recover the same operator, and no fork saturates.  The cross-half term was already computed in
    # round-2 but was accidentally omitted from the predicate; adding it does not change the 2/9 result.
    identifiable = robust_identifiability_gate(
        disc_full, discA, discB, split_half_stability, any_saturated=any_sat, tol=tol)
    out = dict(k_max=int(ca["k_max"]), n_modes=n_modes, n_realizations=N, strength_frac=a_base,
               per_cell_rms_current=round(amps[0], 3), linearity_discrepancy=disc_full,
               disc_convergence=conv, disc_repeatA=discA, disc_repeatB=discB,
               split_half_stability=split_half_stability, n_forks=len(tasks), n_saturated_forks=int(n_sat),
               any_fork_saturated=any_sat, identifiable=identifiable, T_mid_ms=float(cfg["T_windows_ms"][1]),
               wall_s=round(time.time() - t0, 1), sigma1={})
    arrays = {}
    corr_mask = S["regions"]["axis_corridor"].ravel()
    if identifiable:
        for T_ms in cfg["T_windows_ms"]:
            Ts = int(round(T_ms / DT))
            Kf = ensK(allR, 0, Ts)
            U, s, Vt = np.linalg.svd(Kf, full_matrices=False)
            u1 = U[:, 0].reshape(n, n)
            v1 = (P_lowk @ Vt[0, :]).reshape(n, n)
            u1sq = np.abs(u1.ravel()) ** 2
            out["sigma1"][float(T_ms)] = dict(
                sigma1=float(s[0]), gap=(float(s[0] / s[1]) if s.size > 1 and s[1] > 0 else float("inf")),
                u1_axis=field_axis_alignment(u1, ro, S["axis_g"]), u1_globality=field_globality(u1),
                u1_corridor_frac=float(u1sq[corr_mask].sum() / u1sq.sum()),   # does U1 sit on the src->sink corridor?
                v1_axis=field_axis_alignment(v1, ro, S["axis_g"]),
                singular_values=[float(x) for x in s[:6]])
            arrays[f"corr_u1_T{int(T_ms)}"] = u1
            arrays[f"corr_v1_T{int(T_ms)}"] = v1
            arrays[f"corr_K_T{int(T_ms)}"] = Kf
    # per-realization K sufficient statistics at T_mid (both amplitudes) for post-hoc reproducibility
    for ai in (0, 1):
        arrays[f"corr_Kr_a{ai}_T{Tmid}"] = np.stack([Kr(r, ai, Tmid) for r in allR])  # (N, n_bins, n_modes)
    return out, arrays


def cmd_audit(args, cfg):
    import gc
    label, mzcfg, onsets = _cand(cfg, args.candidate)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    states = args.states.split(",") if args.states else cfg["primary_states"]
    workers = int(args.workers or cfg["workers"])
    freeze = bool(cfg.get("freeze_zm", True))
    os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)
    for seed in seeds:
        S = build_S(seed, cfg)
        branches = state_branches(onsets[seed], cfg)
        cks = replay_checkpoints(S, mzcfg, branches, label, resume=args.resume)
        _ensure_flat(S)
        for st in states:
            rj = os.path.join(OUT, "per_seed", f"corrected_audit_{label}_seed{seed}_{st}.json")
            if args.resume and os.path.exists(rj):
                print(f"[audit] resume skip s{seed} {st}", flush=True)
                continue
            out, arrays = corrected_operator_audit(S, cks[st], branches[st], cfg, workers=workers, freeze=freeze)
            out.update(candidate=label, seed=seed, state=st, branch=branches[st], time_ms=branches[st] * DT,
                       provenance=_provenance(cfg, dict(phase="corrected-audit", seed=seed, state=st)))
            _dump(out, rj)
            if arrays:
                np.savez_compressed(os.path.join(OUT, "per_seed", f"corrected_audit_arrays_{label}_seed{seed}_{st}.npz"), **arrays)
            print(f"[audit] {label} s{seed} {st} disc={out['linearity_discrepancy']:.3f} "
                  f"repA/B={out['disc_repeatA']:.3f}/{out['disc_repeatB']:.3f} sat={out['n_saturated_forks']}/{out['n_forks']} "
                  f"identifiable={out['identifiable']} ({out['wall_s']}s)", flush=True)
        del S, cks
        gc.collect()


# ============================================================ tiny substrate (smoke only)
def _tiny_S(cfg, grid_n=6):
    from params import Params
    from model import build_network
    p = Params(g=3.6, L=1.0, density=2000.0, T=200.0, dt=DT, nu_ext_ratio=0.9, seed=1)
    net = build_network(p, verbose=False)
    net["rng"] = np.random.default_rng(1)
    NE, N = net["NE"], net["NE"] + net["NI"]
    posE = net["pos"][:NE]
    src_xy = np.array([0.35, 0.5]); snk_xy = np.array([0.65, 0.5])
    axis_unit = (snk_xy - src_xy) / np.linalg.norm(snk_xy - src_xy)
    vth = np.full(N, p.V_th)
    for c in (src_xy, snk_xy):
        vth[:NE][np.linalg.norm(posE - c, axis=1) <= 0.12] -= 1.0
    S = dict(p=p, net=net, NE=NE, N=N, L=1.0, posE=posE, vth=vth, src_xy=src_xy, snk_xy=snk_xy,
             center=np.array([0.5, 0.5]), axis_unit=axis_unit, seed=1)
    ro = build_grid_readout(posE, grid_n=grid_n, L_phys=1.0, L_norm=float(cfg["L_norm"]),
                            center_phys=[0.5, 0.5])
    src_g, _ = normalize_subject_coordinates(src_xy[None, :], L_phys=1.0, L_norm=cfg["L_norm"], center_phys=[0.5, 0.5])
    snk_g, _ = normalize_subject_coordinates(snk_xy[None, :], L_phys=1.0, L_norm=cfg["L_norm"], center_phys=[0.5, 0.5])
    src_g, snk_g = src_g[0], snk_g[0]
    axis_g = (snk_g - src_g) / np.linalg.norm(snk_g - src_g)
    S["readout"] = ro; S["src_g"] = src_g; S["snk_g"] = snk_g; S["axis_g"] = axis_g
    S["regions"] = grid_region_masks(ro, src_g, snk_g, axis_g, float(cfg["core_radius_norm"]),
                                     float(cfg["corridor_halfwidth_norm"]))
    return S


def cmd_smoke(args, cfg):
    """Tiny-net end-to-end: native parity + common-RNG fork + operator SVD + fixed kick + timing."""
    scfg = dict(cfg); scfg["grid_n"] = 6; scfg["window_ms"] = 20.0
    scfg["T_windows_ms"] = [5.0, 10.0, 20.0]; scfg["local_map_centers_ms"] = [5.0, 10.0, 20.0]
    scfg["linearity_audit_n_patterns"] = 8
    mzcfg = MZSlowVarsConfig(use_z=True, use_m=False, I_th_EI=5.0, tau_z=3000.0)
    S = _tiny_S(scfg, grid_n=6)
    _ensure_flat(S)
    branch = 400
    slow = MZSpatialProbe(S["N"], 18.0, mzcfg, NE=S["NE"])
    S["net"]["rng"] = np.random.default_rng(1)
    rep = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=branch, capture_final=True, store_spikes=False)
    ck = rep["checkpoint"]
    print(f"[smoke] tiny NE={S['NE']} grid={scfg['grid_n']} occ_min={S['readout'].occupancy.min()} "
          f"empty={int(S['readout'].empty_mask.sum())}", flush=True)
    # common random numbers: two identical +eps forks
    pat = 0.4 * np.ones(S["NE"])
    a = _fork_run(S, ck, branch, pat, window_steps=200, cur_dur_steps=10, freeze=True, store_spikes=True)
    b = _fork_run(S, ck, branch, pat, window_steps=200, cur_dur_steps=10, freeze=True, store_spikes=True)
    crn = bool(np.array_equal(a["E_spk_bool"], b["E_spk_bool"]))
    z0 = _fork_run(S, ck, branch, None, window_steps=200, cur_dur_steps=10, freeze=True, store_spikes=True)
    changed = not np.array_equal(a["E_spk_bool"], z0["E_spk_bool"])
    print(f"[smoke] common-RNG idempotent={crn}  perturbation_changes_output={changed}", flush=True)
    # linearity + operator (workers from CLI so the smoke also exercises the fork/pickle path)
    workers = int(args.workers or 2)
    t0 = time.time()
    aud = linearity_audit_state(S, ck, branch, scfg, workers=workers)
    eps = aud["selection"]["epsilon"] or scfg["amplitude_ladder"][1]
    op, basis = operator_for_state(S, ck, branch, scfg, eps, workers=workers)
    n_forks = 2 * basis.shape[1] + 1
    dt_fork = (time.time() - t0) / max(n_forks, 1)
    print(f"[smoke] eps={eps} mode={aud['selection']['mode']} censor={op['censor']} "
          f"discreps={[round(d,3) for d in aud['discrepancies']]}", flush=True)
    if op["censor"] == "resolved":
        for Tms, sd in op["sigma1"].items():
            print(f"[smoke]  T={Tms}ms sigma1={sd['sigma1']:.4g} gap={sd['gap']:.2f} "
                  f"u1_axis={sd['u1_axis']:.3f} u1_glob={sd['u1_globality']:.3f}", flush=True)
    fk = fixed_kick_state(S, ck, branch, scfg)
    reg_str = {k: round(v, 3) for k, v in fk["region"].items()}
    print(f"[smoke] fixed_kick norm={fk['response_norm']:.4g} censor={fk['censor']} region={reg_str} "
          f"arrival_eligible={fk['arrival_fit']['eligible']}", flush=True)
    # corrected identifiability audit mechanics (ensemble + low-k + RMS-matched)
    scfg["corrected_audit"] = dict(k_max=1, n_realizations=3, strength_frac=0.05, realization_base_seed=90001)
    ca, _ = corrected_operator_audit(S, ck, branch, scfg, workers=workers)
    print(f"[smoke] corrected-audit n_modes={ca['n_modes']} n_real={ca['n_realizations']} "
          f"disc={ca['linearity_discrepancy']:.3f} identifiable={ca['identifiable']}", flush=True)
    print(f"[smoke] ~{dt_fork*1000:.0f} ms/fork on tiny net ({n_forks} forks); OK", flush=True)
    ok = crn and changed and op["censor"] == "resolved" and ca["n_modes"] == 9
    print("[smoke] PASS" if ok else "[smoke] CHECK", flush=True)


# ============================================================ full-density run (P0)
def _lock_epsilon(aud, cfg):
    """Lock ONE eps for all cells (spec §2.3). The quiet baseline is below the identifiability floor
    (~0.1 Hz -> quantization-dominated response), so lock on the most-active state where linearity
    actually holds: pre_onset first, then midpoint. None qualify -> nonlinear_response_only globally."""
    for st in ("pre_onset", "midpoint"):
        sel = aud.get(st, {}).get("selection", {})
        if sel.get("mode") == "operator":
            return dict(epsilon=sel["epsilon"], index=sel["index"], mode="operator", lock_state=st)
    return dict(epsilon=None, index=None, mode="nonlinear_response_only", lock_state=None)


def _state_verified(aud, st, lock_index):
    """Did the locked eps pass THIS state's own linearity audit?"""
    return bool(lock_index is not None and lock_index in aud.get(st, {}).get("selection", {}).get("qualified", []))


def _process_state(S, ck, branch, cfg, label, seed, st, eps, mode, workers, freeze, linearity_verified=False):
    t0 = time.time()
    fk = fixed_kick_state(S, ck, branch, cfg)
    summ = dict(candidate=label, seed=seed, state=st, branch=int(branch), time_ms=branch * DT,
                src_g=[float(x) for x in S["src_g"]], snk_g=[float(x) for x in S["snk_g"]],
                axis_g=[float(x) for x in S["axis_g"]], linearity_verified=bool(linearity_verified),
                fixed_kick={k: v for k, v in fk.items() if k != "arrays"})
    arrays = {f"fk_{k}": v for k, v in fk["arrays"].items()}
    if mode == "operator" and eps is not None:
        op, _ = operator_for_state(S, ck, branch, cfg, eps, workers=workers, freeze=freeze)
        summ["operator"] = dict(censor=op["censor"], sigma1=op.get("sigma1", {}),
                                any_fork_saturation=op.get("any_fork_saturation"),
                                linearity_verified=bool(linearity_verified))
        if op["censor"] == "resolved":
            for k, v in op["arrays"].items():
                arrays[f"op_{k}"] = v
            Tmid = int(round(cfg["T_windows_ms"][1]))
            summ["gabor"] = gabor_scan_from_operator(op["arrays"][f"M_T{Tmid}"], S, cfg)
    else:
        summ["operator"] = dict(censor="nonlinear_response_only", note="operator skipped (no eps passed linearity)")
    summ["wall_s"] = round(time.time() - t0, 1)
    summ["provenance"] = _provenance(cfg, dict(phase="run", seed=seed, state=st, epsilon=eps))
    _dump(summ, os.path.join(OUT, "per_seed", f"state_{label}_seed{seed}_{st}.json"))
    np.savez_compressed(os.path.join(OUT, "per_seed", f"arrays_{label}_seed{seed}_{st}.npz"), **arrays)
    print(f"[run] {label} s{seed} {st} kick_norm={fk['response_norm']:.3g} "
          f"op_censor={summ['operator'].get('censor')} verified={linearity_verified} ({summ['wall_s']}s)", flush=True)


def cmd_run(args, cfg):
    import gc
    which = args.candidate
    label, mzcfg, onsets = _cand(cfg, which)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    states = args.states.split(",") if args.states else cfg["primary_states"]
    workers = int(args.workers or cfg["workers"])
    freeze = bool(cfg.get("freeze_zm", True))
    os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)
    audit_path = os.path.join(OUT, "linearity_audit.json")
    locked = json.load(open(audit_path)) if os.path.exists(audit_path) else None
    for seed in seeds:
        S = build_S(seed, cfg)
        branches = state_branches(onsets[seed], cfg)
        cks = replay_checkpoints(S, mzcfg, branches, label, resume=args.resume)
        _ensure_flat(S)
        if locked is None:
            aud = {st: linearity_audit_state(S, cks[st], branches[st], cfg, workers=workers, freeze=freeze)
                   for st in ("baseline", "midpoint", "pre_onset")}
            lock = _lock_epsilon(aud, cfg)
            verified = {st: _state_verified(aud, st, lock["index"]) for st in aud}
            locked = dict(lock_seed=seed, per_state=aud, locked_epsilon=lock["epsilon"], mode=lock["mode"],
                          index=lock["index"], lock_state=lock["lock_state"], verified=verified,
                          provenance=_provenance(cfg, dict(phase="linearity")))
            _dump(locked, audit_path)
            print(f"[run] LOCKED eps={lock['epsilon']} mode={lock['mode']} lock_state={lock['lock_state']} "
                  f"verified={verified}", flush=True)
        eps, mode = locked["locked_epsilon"], locked["mode"]
        verified = locked.get("verified", {})
        for st in states:
            rj = os.path.join(OUT, "per_seed", f"state_{label}_seed{seed}_{st}.json")
            if args.resume and os.path.exists(rj):
                print(f"[run] resume skip s{seed} {st}", flush=True)
                continue
            _process_state(S, cks[st], branches[st], cfg, label, seed, st, eps, mode, workers, freeze,
                           linearity_verified=verified.get(st, False))
        del S, cks
        gc.collect()


# ============================================================ controls (P1): plateau + D-matched z-only
def cmd_controls(args, cfg):
    which = "primary"
    label, mzcfg, onsets = _cand(cfg, which)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    workers = int(args.workers or cfg["workers"])
    freeze = bool(cfg.get("freeze_zm", True))
    cc = cfg["controls"]["zm_plateau"]
    audit_path = os.path.join(OUT, "linearity_audit.json")
    locked = json.load(open(audit_path)) if os.path.exists(audit_path) else None
    if locked is None:
        raise SystemExit("controls require a locked eps: run the P0 `run` first")
    eps, mode = locked["locked_epsilon"], locked["mode"]
    eta_m = eta_m_from_frac(float(cc["A_target"]), float(cfg["I_EE_scale"]), float(cc["peak_m_tau2000"]))
    zm_cfg = MZSlowVarsConfig(use_z=bool(cc["cfg"]["use_z"]), use_m=True, I_th_EI=float(cc["cfg"]["I_th_EI"]),
                              tau_z=float(cc["cfg"]["tau_z"]), tau_adp=float(cc["tau_adp_ms"]), eta_m=eta_m)
    for seed in seeds:
        rj = os.path.join(OUT, "per_seed", f"controls_seed{seed}.json")
        if args.resume and os.path.exists(rj):
            print(f"[controls] resume skip s{seed}", flush=True)
            continue
        S = build_S(seed, cfg)
        onset_step = int(round(onsets[seed] / DT))
        # --- z+m plateau checkpoint: replay to settle, pick a resting D~median time in the last window ---
        pk = _select_plateau_checkpoint(S, zm_cfg, cfg, onset_step)
        # --- D-matched z-only checkpoint at the SAME D (selection by D + resting only) ---
        zk = _select_dmatched_checkpoint(S, mzcfg, cfg, onset_step, pk["D_target"])
        out = dict(candidate=label, seed=seed, eta_m=eta_m,
                   plateau=dict(branch=pk["branch"], time_ms=pk["branch"] * DT, D=pk["D_target"],
                                settled=pk.get("settled")),
                   dmatched=dict(branch=zk["branch"], time_ms=zk["branch"] * DT, D=zk["D_actual"]))
        arrays = {}
        for tag, ck, br in (("plateau", pk["ck"], pk["branch"]), ("dmatched", zk["ck"], zk["branch"])):
            fk = fixed_kick_state(S, ck, br, cfg)
            out[f"{tag}_fixed_kick"] = {k: v for k, v in fk.items() if k != "arrays"}
            for k, v in fk["arrays"].items():
                arrays[f"{tag}_fk_{k}"] = v
            if mode == "operator" and eps is not None:
                op, _ = operator_for_state(S, ck, br, cfg, eps, workers=workers, freeze=freeze)
                out[f"{tag}_operator"] = dict(censor=op["censor"], sigma1=op.get("sigma1", {}))
                if op["censor"] == "resolved":
                    for k, v in op["arrays"].items():
                        arrays[f"{tag}_op_{k}"] = v
        out["provenance"] = _provenance(cfg, dict(phase="controls", seed=seed))
        _dump(out, rj)
        np.savez_compressed(os.path.join(OUT, "per_seed", f"controls_arrays_seed{seed}.npz"), **arrays)
        print(f"[controls] s{seed} plateau D={pk['D_target']:.4f}@{pk['branch']*DT:.0f}ms "
              f"dmatched D={zk['D_actual']:.4f}@{zk['branch']*DT:.0f}ms", flush=True)
        import gc
        del S
        gc.collect()


def _replay_traj(S, mz_cfg, T_ms):
    """Replay accumulating z-mean trace + population E-rate (store_spikes=False -> low memory: no
    ~N x T spike bool). Returns (rate_E_hz, D=1-z_bar) per step."""
    slow = MZSpatialProbe(S["N"], 18.0, mz_cfg, NE=S["NE"])
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    n = int(round(T_ms / DT))
    res = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=n, capture_final=False, store_spikes=False)
    return res["rate_E"], 1.0 - np.asarray(slow.trace_z_mean, float)


def _resting_mask(rate_hz, *, win_ms=20.0, k=0.3):
    """Resting (non-event) steps: 20 ms-smoothed population E-rate below floor + k*(peak-floor).
    Events = bursts above the bar. Uses the cheap population rate (no per-neuron spikes)."""
    r = np.asarray(rate_hz, float)
    w = max(1, int(round(win_ms / DT)))
    sm = np.convolve(r, np.ones(w) / w, mode="same")
    floor = float(np.percentile(sm, 20))
    peak = float(np.percentile(sm, 99))
    return sm <= floor + float(k) * (peak - floor)


def _select_plateau_checkpoint(S, zm_cfg, cfg, onset_step):
    """Replay the z+m plateau; pick a RESTING time in the plateau window (AFTER settle_ms, so the
    plateau has formed) whose D is nearest the window median (selection frozen BEFORE any spatial
    response, spec §1). Also report whether the plateau is actually settled (D roughly constant over
    the window) — an unsettled checkpoint is not the upstream stable plateau (review 2026-07-20)."""
    cc = cfg["controls"]["zm_plateau"]
    T_ms = min(onset_step * DT, 15000.0)
    rate, D = _replay_traj(S, zm_cfg, T_ms)
    rest = _resting_mask(rate)
    settle_lo = int(round(float(cc.get("settle_ms", 6000.0)) / DT))        # plateau must have formed
    win_lo = max(int(round((T_ms - float(cc["plateau_window_ms"])) / DT)), settle_lo)
    idx = np.arange(win_lo, len(D))
    idx = idx[rest[win_lo:len(D)]]
    if idx.size == 0:
        idx = np.arange(win_lo, len(D))
    med = float(np.median(D[idx]))
    branch = int(idx[np.argmin(np.abs(D[idx] - med))])
    d_win = D[win_lo:len(D)]
    settled = bool(d_win.size > 0 and np.ptp(d_win) < 0.02)                # plateau D varies < 0.02 over window
    ck = _capture_at(S, zm_cfg, branch)
    return dict(branch=branch, D_target=float(D[branch]), settled=settled, ck=ck)


def _select_dmatched_checkpoint(S, mzcfg, cfg, onset_step, D_target):
    """RESTING time in the z-only trajectory whose D is closest to the plateau D (selection by D +
    resting + time only, never the spatial response)."""
    T_ms = min(onset_step * DT, 15000.0)
    rate, D = _replay_traj(S, mzcfg, T_ms)
    rest = _resting_mask(rate)
    cand = np.where(rest)[0]
    cand = cand[cand < onset_step - 100]
    if cand.size == 0:
        cand = np.arange(0, max(1, onset_step - 100))
    branch = int(cand[np.argmin(np.abs(D[cand] - D_target))])
    ck = _capture_at(S, mzcfg, branch)
    return dict(branch=branch, D_actual=float(D[branch]), ck=ck)


def _capture_at(S, mz_cfg, branch):
    slow = MZSpatialProbe(S["N"], 18.0, mz_cfg, NE=S["NE"])
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    rep = run_loop(S["p"], S["net"], slow, S["vth"], n_steps=int(branch), capture_final=True, store_spikes=False)
    return rep["checkpoint"]


# ============================================================ aggregate + audits
def _seeds_from_files(label):
    seeds = set()
    for f in glob.glob(os.path.join(OUT, "per_seed", f"state_{label}_seed*_*.json")):
        seeds.add(int(os.path.basename(f).split("seed")[1].split("_")[0]))
    return sorted(seeds)


def _mode_overlaps(label, cfg):
    """Adjacent-state U1 output-mode overlap (spec §4, sign-invariant |cos|) at the mid window: does the
    empirical optimal-output mode reorganize between states, or stay the same shape?"""
    Tmid = int(round(cfg["T_windows_ms"][1]))
    out = []
    for seed in _seeds_from_files(label):
        u1 = {}
        for st in cfg["primary_states"]:
            f = os.path.join(OUT, "per_seed", f"arrays_{label}_seed{seed}_{st}.npz")
            if os.path.exists(f):
                d = np.load(f, allow_pickle=True)
                if f"op_u1_T{Tmid}" in d.files:
                    u1[st] = d[f"op_u1_T{Tmid}"]
        row = dict(seed=seed)
        for a, b in (("baseline", "pre_onset"), ("midpoint", "pre_onset")):
            if a in u1 and b in u1:
                row[f"u1_overlap_{a}_{b}"] = normalized_field_overlap(u1[a], u1[b])
        out.append(row)
    return out


def cmd_aggregate(args, cfg):
    label, _, _ = _cand(cfg, args.candidate)
    per = sorted(glob.glob(os.path.join(OUT, "per_seed", f"state_{label}_seed*_*.json")))
    op_rows, fk_rows, gb_rows = [], [], []
    for f in per:
        d = json.load(open(f))
        base = dict(seed=d["seed"], state=d["state"], time_ms=d["time_ms"])
        op = d.get("operator", {})
        row = dict(base, censor=op.get("censor"))
        for Tms, sd in (op.get("sigma1") or {}).items():
            row[f"sigma1_T{int(float(Tms))}"] = sd["sigma1"]
            row[f"u1_axis_T{int(float(Tms))}"] = sd["u1_axis"]
            row[f"u1_glob_T{int(float(Tms))}"] = sd["u1_globality"]
            row[f"gap_T{int(float(Tms))}"] = sd["gap"]
        op_rows.append(row)
        fk = d.get("fixed_kick", {})
        fk_rows.append(dict(base, censor=fk.get("censor"), response_norm=fk.get("response_norm"),
                            cum_remote_over_source=fk.get("cum_remote_over_source_final"),
                            arrival_eligible=(fk.get("arrival_fit") or {}).get("eligible"),
                            arrival_slope=(fk.get("arrival_fit") or {}).get("slope"),
                            arrival_r2=(fk.get("arrival_fit") or {}).get("r2"),
                            **{f"region_{k}": v for k, v in (fk.get("region") or {}).items()}))
        gb = d.get("gabor")
        if gb:
            gb_rows.append(dict(base, axial_gain=gb["axial_gain"], perp_gain=gb["perp_gain"],
                                global_gain=gb["global_gain"], axis_minus_perp=gb["axis_minus_perp"]))
    _dump(dict(schema_version=SCHEMA_VERSION, rows=op_rows, mode_overlaps=_mode_overlaps(label, cfg),
               provenance=_provenance(cfg, dict(phase="aggregate"))),
          os.path.join(OUT, "empirical_operator_summary.json"))
    _dump(dict(schema_version=SCHEMA_VERSION, rows=fk_rows), os.path.join(OUT, "fixed_kick_summary.json"))
    _dump(dict(schema_version=SCHEMA_VERSION, rows=gb_rows), os.path.join(OUT, "probe_scan_summary.json"))
    # checkpoint manifest
    cks = sorted(glob.glob(os.path.join(CKDIR, "*.pkl")))
    manifest = [dict(file=os.path.basename(p), sha=_file_hash(p),
                     size_mb=round(os.path.getsize(p) / 1e6, 1)) for p in cks]
    _dump(dict(checkpoints=manifest, provenance=_provenance(cfg, dict(phase="manifest"))),
          os.path.join(OUT, "checkpoint_manifest.json"))
    # numerical audit: basis orthonormality + mass conservation spot check
    P = real_fourier_basis_2d(int(cfg["grid_n"]))
    ortho = float(np.max(np.abs(P.T @ P - np.eye(P.shape[1]))))
    _dump(dict(basis_orthonormality_max_abs_resid=ortho, n_states=len(op_rows),
               censored_states=[r for r in op_rows if r.get("censor") != "resolved"],
               provenance=_provenance(cfg, dict(phase="numerical-audit"))),
          os.path.join(OUT, "numerical_audit.json"))
    # aggregate arrays
    _agg_arrays(label, "op_", os.path.join(OUT, "empirical_operator_arrays.npz"))
    _agg_arrays(label, "fk_", os.path.join(OUT, "fixed_kick_arrays.npz"))
    # corrected identifiability audit (ensemble / low-k / RMS-matched)
    ca_rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "per_seed", f"corrected_audit_{label}_seed*_*.json"))):
        d = json.load(open(f))
        s30 = (d.get("sigma1") or {}).get(str(float(cfg["T_windows_ms"][1])), {})
        robust_id = robust_identifiability_gate(
            d.get("linearity_discrepancy"), d.get("disc_repeatA"), d.get("disc_repeatB"),
            d.get("split_half_stability"), any_saturated=bool(d.get("any_fork_saturated")),
            tol=float(cfg["linearity_tol"]))
        half_u1_overlap = None
        arrays_path = os.path.join(
            OUT, "per_seed", f"corrected_audit_arrays_{label}_seed{d['seed']}_{d['state']}.npz")
        if robust_id and os.path.exists(arrays_path):
            aa = np.load(arrays_path)
            key = f"corr_Kr_a0_T{int(round(cfg['T_windows_ms'][1] / DT))}"
            if key in aa.files:
                kr = aa[key]
                half = kr.shape[0] // 2
                ua = np.linalg.svd(kr[:half].mean(axis=0), full_matrices=False)[0][:, 0]
                ub = np.linalg.svd(kr[half:].mean(axis=0), full_matrices=False)[0][:, 0]
                half_u1_overlap = normalized_field_overlap(ua, ub)
        ca_rows.append(dict(seed=d["seed"], state=d["state"], discrepancy=d["linearity_discrepancy"],
                            disc_repeatA=d.get("disc_repeatA"), disc_repeatB=d.get("disc_repeatB"),
                            disc_convergence=d.get("disc_convergence"), split_half_stability=d.get("split_half_stability"),
                            n_saturated_forks=d.get("n_saturated_forks"), any_fork_saturated=d.get("any_fork_saturated"),
                            identifiable=robust_id, n_modes=d["n_modes"], n_realizations=d["n_realizations"],
                            per_cell_rms_current=d.get("per_cell_rms_current"),
                            sigma1_T30=s30.get("sigma1"), u1_axis_T30=s30.get("u1_axis"),
                            u1_glob_T30=s30.get("u1_globality"), u1_corridor_frac_T30=s30.get("u1_corridor_frac"),
                            u1_half_overlap_T30=half_u1_overlap))
    if ca_rows:
        n_id = sum(1 for r in ca_rows if r["identifiable"])
        _dump(dict(schema_version=SCHEMA_VERSION, tol=float(cfg["linearity_tol"]), rows=ca_rows,
                   n_identifiable=n_id, provenance=_provenance(cfg, dict(phase="corrected-audit-aggregate"))),
              os.path.join(OUT, "corrected_audit_summary.json"))   # V1/U1 arrays stay per-seed (identifiable states only)
    ctrl = sorted(glob.glob(os.path.join(OUT, "per_seed", "controls_seed*.json")))
    if ctrl:
        _dump(dict(rows=[json.load(open(f)) for f in ctrl]), os.path.join(OUT, "controls_summary.json"))
    _dump(_provenance(cfg, dict(phase="aggregate", n_state_files=len(per))), os.path.join(OUT, "provenance.json"))
    print(f"[aggregate] {len(op_rows)} state rows; basis ortho resid={ortho:.2e}; "
          f"controls={len(ctrl)}; corrected_audit={len(ca_rows)} ({sum(1 for r in ca_rows if r['identifiable'])} identifiable)", flush=True)


def _agg_arrays(label, prefix, out_path):
    bundle = {}
    for f in sorted(glob.glob(os.path.join(OUT, "per_seed", f"arrays_{label}_seed*_*.npz"))):
        base = os.path.basename(f).replace(f"arrays_{label}_", "").replace(".npz", "")
        d = np.load(f, allow_pickle=True)
        for k in d.files:
            if k.startswith(prefix):
                bundle[f"{base}__{k}"] = d[k]
    if bundle:
        np.savez_compressed(out_path, **bundle)


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Topic 4 MZ direct spatial mode dynamics runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "run", "audit", "controls", "aggregate"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true")
        sp.add_argument("--candidate", default="primary", choices=["primary", "sensitivity"])
        sp.add_argument("--seeds", default=None)
        sp.add_argument("--states", default=None)
        sp.add_argument("--workers", default=None)
        sp.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    needs_run = {"smoke", "run", "audit", "controls"}
    if args.cmd in needs_run and not args.confirm_run:
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run.", file=sys.stderr)
        sys.exit(2)
    os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)
    {"smoke": cmd_smoke, "run": cmd_run, "audit": cmd_audit, "controls": cmd_controls,
     "aggregate": cmd_aggregate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
