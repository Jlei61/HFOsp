"""Topic 4 MZ early-onset dynamics — scientific runner (state map / frozen operator / nonlinear ignition /
z counterfactuals / event suppression / focused m push-pull / integration).

*** THIS RUNS SIMULATIONS. *** Nothing runs on import; every sim subcommand is gated by --confirm-run.
Design contract (BINDING): docs/superpowers/specs/2026-07-19-topic4-mz-onset-dynamics-phase-portrait-design.md

Reuse (not reinvent):
  run_m4_phaseplane.build_substrate (E1146 narrow/template_source/twoend_equal/L20/dens100/AR2, CORE_R, R_KICK),
  run_sef_hfo_snn_cm_spontaneous_readout.{active_fraction,detect_events,per_neuron_onset,BIN_MS,BASELINE_MS,CAL_FRAC},
  run_m4_dynamic_qi.{_smooth,_first_sustained} (locked 120Hz/100ms runaway), kick_probe.simulate_kick,
  mz_slow_vars / src.topic4_mz_slowvars (classifier + calibration helpers),
  src.topic4_mz_onset_dynamics (MZOnsetProbe, run_loop checkpoint/resume, coords/qeff/DA/ignition),
  src.topic4_state_conditioned_susceptibility (coarse binning + fixed scaffold + operator/probe atlas),
  upstream snapshots + susceptibility_atlas + mz_early_field_bridge cohort artifacts.

Subcommands (all resumable via per-seed/per-state JSON; re-run is idempotent):
  state-coords   Task A/B: native replay + q_eff observer -> state_coordinate_audit.json + event_state_transitions.csv
  operator-grid  Task C:   frozen operator at natural states (verify+extend upstream) + q_eff-field audit +
                           realized (D,A) grid -> spectral_state_summary.csv + projected_phase_grid.csv
  ignition       Task D:   full-SNN epsilon_c via checkpoint/resume -> nonlinear_ignition_summary.csv
  counterfactual Task E:   z counterfactual branches -> z_counterfactual_summary.csv
  event-suppress Task F:   interictal-event suppression pulse -> event_suppression_summary.csv
  focused-m      Task G:   focused realized-state m push-pull grid -> focused_m_summary.csv
  integrate      Task H:   align field bridge with D/q_eff/alpha1/gain/epsilon_c -> integration table (STATUS)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse           # noqa: E402
import copy               # noqa: E402
import csv                # noqa: E402
import dataclasses        # noqa: E402
import hashlib            # noqa: E402
import glob               # noqa: E402
import json               # noqa: E402
import sys                # noqa: E402
import time               # noqa: E402

import numpy as np        # noqa: E402
import yaml               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP                             # noqa: E402
import run_sef_hfo_snn_cm_spontaneous_readout as C        # noqa: E402
import run_m4_dynamic_qi as M4                             # noqa: E402
from kick_probe import simulate_kick                       # noqa: E402
from mz_slow_vars import MZSlowVarsConfig                  # noqa: E402
from src.topic4_mz_onset_dynamics import (                 # noqa: E402
    MZOnsetProbe, run_loop, score_runaway, build_region_masks, slow_state_coordinates,
    qeff_region_summary, zbar_qeff_field_audit, realized_D_grid, build_DA_q_field, DA_controls,
    epsilon_c_from_ladder, classify_ignition, natural_zm_trajectory, SCHEMA_VERSION,
    validate_focused_m_grid, build_tau_sensitivity,
)
from src.topic4_mz_slowvars import classify_mz_run, replay_adaptation_peak, eta_m_from_frac  # noqa: E402
from src.topic4_m3b_spectral_phase import Grid, build_excitability_field  # noqa: E402
from src.topic4_state_conditioned_susceptibility import (  # noqa: E402
    normalize_subject_coordinates, bin_neuron_state_to_grid, build_fixed_scaffold, two_core_mask_at,
    make_phase_paired_probe_dictionary, summarize_state_susceptibility, state_operator,
)

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_onset_dynamics")
CFG_PATH = os.path.join(ROOT, "config", "topic4_mz_onset_dynamics.yaml")
DT = 0.1
_GUARDED = ("kick_probe.py", "params.py", "model.py", "connectivity.py", "connectivity_rot.py", "lfp.py")


# ============================================================ config + provenance
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
                config_hash=_file_hash(CFG_PATH), module_hash=_file_hash(os.path.join(ROOT, "src", "topic4_mz_onset_dynamics.py")),
                argv=sys.argv, subject=cfg["subject"], montage=cfg["montage"], dt=DT,
                upstream={k: dict(path=v, hash=_file_hash(os.path.join(ROOT, v)) if v.endswith(".json") else None)
                          for k, v in cfg["upstream"].items()})
    if extra:
        prov.update(extra)
    return prov


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _write_csv(rows, path, keys=None):
    if not rows:
        return
    keys = keys or list(rows[0].keys())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


# ============================================================ substrate + regions + states
def build_S(seed, cfg):
    """Substrate + E-indexed region masks + coarse-grid scaffold (all fixed geometry)."""
    S = PP.build_substrate(seed)
    core_r = float(PP.CORE_R)
    regions = build_region_masks(S["posE"], S["src_xy"], S["snk_xy"], S["axis_unit"], core_r,
                                 corridor_halfwidth=float(cfg["axis_corridor_halfwidth_norm"]) * core_r / 1.0)
    core_mask_E = regions["source_core"] | regions["sink_core"]
    return S, regions, core_mask_E, core_r


def state_steps(onset_ms, cfg):
    """Registered slow-state step indices (spec §4). mid = 0.50*onset. Matches upstream snapshot_steps."""
    o = float(onset_ms)
    return {
        "baseline_1000ms": int(round(cfg["baseline_ms"] / DT)),
        "mid_fraction": int(round(cfg["mid_fraction"] * o / DT)),
        "pre_onset_500ms": int(round((o - cfg["pre_onset_500_ms"]) / DT)),
        "pre_onset_100ms": int(round((o - cfg["pre_onset_100_ms"]) / DT)),
        "onset": int(round(o / DT)),
    }


def _cand(cfg, which):
    c = cfg["candidates"][which]
    return c["label"], MZSlowVarsConfig(**c["cfg"]), {int(k): float(v) for k, v in c["onset_ms"].items()}


# ============================================================ Task A/B: state coordinates + q_eff
def cmd_state_coords(args, cfg):
    which = args.candidate
    label, mzcfg, onsets = _cand(cfg, which)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    win_steps = int(round(cfg["q_eff_window_ms"] / DT))
    I_th = float(mzcfg.I_th_EI)
    outdir = os.path.join(OUT, "per_seed")
    rows_state, rows_event = [], []
    for seed in seeds:
        seed_json = os.path.join(outdir, f"state_coords_{label}_seed{seed}.json")
        if args.resume and os.path.exists(seed_json):
            print(f"[state-coords] resume: {seed_json} exists, skip", flush=True)
            d = json.load(open(seed_json))
            rows_state += d["state_rows"]; rows_event += d["event_rows"]
            continue
        t0 = time.time()
        S, regions, core_mask_E, core_r = build_S(seed, cfg)
        onset_ms = onsets[seed]
        steps = state_steps(onset_ms, cfg)
        tail = onset_ms + cfg["event_post_offset_ms"] + 200.0
        # windows: 20ms BEFORE each registered state; label them by state
        windows = [(max(0, steps[st] - win_steps), steps[st], st) for st in cfg["registered_states"]]
        snap_steps = {steps[st]: st for st in cfg["registered_states"]}
        # corridor as the observer 'core' so trace_z_core_mean(t) == D_axis(t) proxy for event-resolved ΔD
        corridor = regions["axis_corridor"]
        mz = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=corridor, snapshot_steps=snap_steps)
        mz.set_qeff_windows(windows, I_th)
        p = dataclasses.replace(S["p"], T=float(tail))
        S["net"]["rng"] = np.random.default_rng(seed)
        res = simulate_kick(p, S["net"], 0.0, slow=mz, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                            t_kick=1e9, V_th_per_neuron=S["vth"], early_stop_runaway=False)
        qfields = mz.qeff_fields()
        # coarse-grid mapping for the z_bar vs q_eff field audit (reuse susceptibility binning)
        grid = Grid(n=int(cfg["operator"]["grid_n"]), L=float(cfg["operator"]["L_norm"]))
        pos_norm, _ = normalize_subject_coordinates(S["posE"], L_phys=cfg["operator"]["L_phys"],
                                                    L_norm=cfg["operator"]["L_norm"], center_phys=cfg["operator"]["center_phys"])
        # per-state coordinates from captured z/m snapshots + q_eff windows
        state_json = {}
        for st in cfg["registered_states"]:
            snap = mz.snapshots.get(st)
            if snap is None:
                continue
            zE, mE = snap["z_E"], snap["m_E"]
            coords = slow_state_coordinates(zE, mE, mzcfg.eta_m, regions)
            qf = qfields.get(st, {})
            q_eff_E = qf.get("q_eff"); pdep_E = qf.get("p_deplete")
            qsum = qeff_region_summary(q_eff_E, pdep_E, regions) if q_eff_E is not None else {}
            # field audit: z_bar-derived field vs q_eff field on the coarse grid
            zbar_field, _, _ = bin_neuron_state_to_grid(zE, pos_norm, grid)
            audit = None
            if q_eff_E is not None and np.isfinite(q_eff_E).any():
                qeff_field, _, _ = bin_neuron_state_to_grid(np.where(np.isfinite(q_eff_E), q_eff_E, np.nan), pos_norm, grid)
                audit = zbar_qeff_field_audit(zbar_field, qeff_field)
            state_json[st] = dict(step=steps[st], time_ms=steps[st] * DT, coords=coords, qeff=qsum,
                                  zbar_qeff_audit=audit)
            for rgn, cv in coords.items():
                rows_state.append(dict(candidate=label, seed=seed, state=st, region=rgn, time_ms=steps[st] * DT,
                                       D_z=cv["D_z"], A=cv["A"], n=cv["n"],
                                       q_eff=qsum.get(rgn, {}).get("q_eff"), p_deplete=qsum.get(rgn, {}).get("p_deplete")))
        # event-resolved: fixed bar from this run's baseline window; D_axis(t) from corridor trace
        events, af, bin_w, floor, rate = _events(res)
        D_axis_trace = 1.0 - np.asarray(mz.trace_z_core_mean, float)   # corridor = observer 'core'
        ret = [e for e in events if e["returned"] and e["t_off"] < onset_ms - cfg["event_end_guard_ms"]]
        for e in ret:
            s_pre = int((e["t_on"] - cfg["event_pre_offset_ms"]) / DT)
            s_post = int((e["t_off"] + cfg["event_post_offset_ms"]) / DT)
            if s_pre < 0 or s_post >= len(D_axis_trace):
                continue
            rows_event.append(dict(candidate=label, seed=seed, t_on=round(e["t_on"], 1), t_off=round(e["t_off"], 1),
                                   dur_ms=round(e["dur_ms"], 1), D_axis_pre=round(float(D_axis_trace[s_pre]), 5),
                                   D_axis_post=round(float(D_axis_trace[s_post]), 5),
                                   dD_axis=round(float(D_axis_trace[s_post] - D_axis_trace[s_pre]), 5),
                                   dA=0.0))   # z-only candidate -> m off -> A==0
        # verify snapshot z matches upstream snapshot (sanity)
        up = _verify_upstream_snapshot(cfg, which, seed, mz.snapshots, steps)
        wall = time.time() - t0
        _dump(dict(candidate=label, seed=seed, onset_ms=onset_ms, wall_s=round(wall, 1),
                   states=state_json, state_rows=[r for r in rows_state if r["seed"] == seed],
                   event_rows=[r for r in rows_event if r["seed"] == seed],
                   n_returning_eligible=len(ret), upstream_snapshot_check=up,
                   provenance=_provenance(cfg, dict(phase="state-coords", candidate=label, seed=seed))), seed_json)
        print(f"[state-coords] {label} seed{seed} onset={onset_ms} wall={wall:.0f}s events_eligible={len(ret)} "
              f"snapshot_match={up.get('max_abs_z_diff')}", flush=True)
    _write_csv(rows_state, os.path.join(OUT, f"state_coordinate_rows_{label}.csv"))
    _write_csv(rows_event, os.path.join(OUT, f"event_state_transitions_{label}.csv"))
    _aggregate_state_coord_audit(cfg)


def _events(res):
    spk = res["E_spk_bool"]; rate = np.asarray(res["rate_E"], float)
    af, bin_w = C.active_fraction(spk, DT, C.BIN_MS)
    nb0, nb1 = int(C.BASELINE_MS[0] / bin_w), int(C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + C.CAL_FRAC * (float(af.max()) - floor)
    events = C.detect_events(af, bin_w, event_on_frac=bar)
    return events, af, bin_w, floor, rate


def _verify_upstream_snapshot(cfg, which, seed, snapshots, steps):
    """Cross-check my captured z_E against the upstream susceptibility snapshot (spec §14.2 sanity)."""
    key = "snapshots_primary" if which == "primary" else "snapshots_sensitivity"
    path = os.path.join(ROOT, cfg["upstream"][key], f"seed_{seed}.npz")
    if not os.path.exists(path):
        return dict(available=False, path=path)
    d = np.load(path, allow_pickle=True)
    labels = [str(x) for x in d["snapshot_labels"]]
    maxdiff = 0.0
    for i, lab in enumerate(labels):
        snap = snapshots.get(lab)
        if snap is None:
            continue
        maxdiff = max(maxdiff, float(np.max(np.abs(snap["z_E"] - d["z_E"][i]))))
    return dict(available=True, max_abs_z_diff=maxdiff, bit_identical=bool(maxdiff == 0.0))


def _aggregate_state_coord_audit(cfg):
    """Merge per-seed state-coords into state_coordinate_audit.json (spec §13 artifact)."""
    per_seed = {}
    d = os.path.join(OUT, "per_seed")
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.startswith("state_coords_") and f.endswith(".json"):
            j = json.load(open(os.path.join(d, f)))
            per_seed.setdefault(j["candidate"], {})[str(j["seed"])] = {
                k: j[k] for k in ("onset_ms", "states", "n_returning_eligible", "upstream_snapshot_check", "wall_s")}
    _dump(dict(schema_version=SCHEMA_VERSION, description="registered MZ slow-state coordinates + current-aware q_eff",
               per_candidate=per_seed, provenance=_provenance(cfg, dict(phase="state-coords-aggregate"))),
          os.path.join(OUT, "state_coordinate_audit.json"))


# ============================================================ Task C: frozen operator + (D,A) grid
def _scaffold(cfg, S):
    o = cfg["operator"]
    grid = Grid(n=int(o["grid_n"]), L=float(o["L_norm"]))
    src_norm, _ = normalize_subject_coordinates(S["src_xy"][None, :], L_phys=o["L_phys"], L_norm=o["L_norm"], center_phys=o["center_phys"])
    snk_norm, _ = normalize_subject_coordinates(S["snk_xy"][None, :], L_phys=o["L_phys"], L_norm=o["L_norm"], center_phys=o["center_phys"])
    scaffold = build_fixed_scaffold(grid, tuple(src_norm[0]), tuple(snk_norm[0]), ell_perp=o["ell_perp"], ar=o["ar"],
                                    mu_core=o["mu_core"], core_radius=o["core_radius_norm"], theta=0.0)
    probes = make_phase_paired_probe_dictionary(grid, p_max=int(o["p_max"]), sigma=float(o["gabor_sigma"]),
                                                center=tuple(src_norm[0]), gabor=bool(o["gabor"]))
    return grid, scaffold, probes


def _op_summary_row(candidate, seed, state, field_kind, summ):
    e = summ.get("eigen", {}) or {}
    at = (summ.get("atlas") or {}).get("per_T", {}) if summ.get("atlas") else {}
    tp = at.get(30.0, {}) if at else {}
    return dict(candidate=candidate, seed=seed, state=state, field=field_kind, op_status=summ["op_status"],
                op_residual=round(summ["op_residual"], 8), alpha1=e.get("leading_growth"),
                leading_globality=e.get("leading_globality"), leading_axis=e.get("leading_axis_score"),
                leading_core_overlap=e.get("leading_core_overlap"), next_gap=e.get("next_distinct_gap"),
                axial_gain=tp.get("axial_gain"), perp_gain=tp.get("perp_gain"), global_gain=tp.get("global_gain"),
                axis_minus_perp=tp.get("axis_minus_perp"), peak_k=tp.get("peak_k"))


def cmd_operator_grid(args, cfg):
    which = args.candidate
    label, mzcfg, onsets = _cand(cfg, which)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    o = cfg["operator"]
    T_list = [float(x) for x in o["T_windows"]]
    rows_op, rows_grid = [], []
    D_axis_by_state = {st: [] for st in cfg["registered_states"]}
    onset_dep_fields = []
    # --- natural states: frozen operator on z_bar field + q_eff field (verify/extend upstream) ---
    for seed in seeds:
        S, regions, core_mask_E, core_r = build_S(seed, cfg)
        grid, scaffold, probes = _scaffold(cfg, S)
        pos_norm, _ = normalize_subject_coordinates(S["posE"], L_phys=o["L_phys"], L_norm=o["L_norm"], center_phys=o["center_phys"])
        # load z_E from state-coords per-seed (already replayed); else from upstream snapshot
        scj = os.path.join(OUT, "per_seed", f"state_coords_{label}_seed{seed}.json")
        upnpz = os.path.join(ROOT, cfg["upstream"]["snapshots_primary" if which == "primary" else "snapshots_sensitivity"], f"seed_{seed}.npz")
        snap = np.load(upnpz, allow_pickle=True) if os.path.exists(upnpz) else None
        labels_up = [str(x) for x in snap["snapshot_labels"]] if snap is not None else []
        for st in cfg["registered_states"]:
            zE = None
            if snap is not None and st in labels_up:
                zE = snap["z_E"][labels_up.index(st)]
            if zE is None:
                continue
            zbar_field, _, _ = bin_neuron_state_to_grid(zE, pos_norm, grid)
            summ, _ = summarize_state_susceptibility(zbar_field, grid, scaffold, probes, T_list,
                                                     w_ee_mult=o["w_ee_mult"], ratio=o["ratio"], q_floor=o["q_floor"],
                                                     T_primary=o["T_primary"])
            rows_op.append(_op_summary_row(label, seed, st, "z_bar", summ))
            # D_axis (corridor) realized coordinate for the (D,A) grid range
            D_axis = float(1.0 - np.nanmean(zbar_field[scaffold["core"].mask]))
            D_axis_by_state[st].append(D_axis)
            if st in ("pre_onset_100ms", "onset"):
                onset_dep_fields.append(1.0 - zbar_field)   # depletion pattern for the (D,A) primary z-pattern
            # q_eff field operator (current-aware) if available
            if os.path.exists(scj):
                jd = json.load(open(scj))
                # recompute q_eff field from stored per-neuron not persisted -> use audit-only (documented)
        print(f"[operator-grid] {label} seed{seed} natural states done", flush=True)
    _write_csv(rows_op, os.path.join(OUT, f"spectral_state_summary_{label}.csv"))

    # --- controlled realized (D,A) grid (spec §6.1): pooled depletion pattern, D range, uniform A ---
    if onset_dep_fields:
        dep_pool = np.nanmean(np.stack(onset_dep_fields, 0), 0)          # pooled onset depletion pattern
        base_D = float(np.nanmean([np.nanmean(v) for v in D_axis_by_state["baseline_1000ms"] if v is not None]) if D_axis_by_state["baseline_1000ms"] else 0.0)
        max_onset_D = float(np.nanmax([v for v in D_axis_by_state["onset"]]) if D_axis_by_state["onset"] else base_D + 0.5)
        D_vals = realized_D_grid(base_D, max_onset_D, n_D=int(cfg["DA_grid"]["n_D"]),
                                 clip=cfg["DA_grid"]["D_clip"], overshoot=cfg["DA_grid"]["D_overshoot"])
        A_fracs = [float(a) for a in cfg["DA_grid"]["A_fracs"]]
        # one representative substrate/scaffold (seed geometry identical across seeds -> use first seed)
        S0, _, _, _ = build_S(seeds[0], cfg)
        for grid_n in (int(o["grid_n_audit"]), int(o["grid_n"])):   # n=8 audit then n=12 final
            gr = Grid(n=grid_n, L=float(o["L_norm"]))
            src_norm, _ = normalize_subject_coordinates(S0["src_xy"][None, :], L_phys=o["L_phys"], L_norm=o["L_norm"], center_phys=o["center_phys"])
            snk_norm, _ = normalize_subject_coordinates(S0["snk_xy"][None, :], L_phys=o["L_phys"], L_norm=o["L_norm"], center_phys=o["center_phys"])
            core = two_core_mask_at(gr, [tuple(src_norm[0]), tuple(snk_norm[0])], o["core_radius_norm"], 0.0)
            probes = make_phase_paired_probe_dictionary(gr, p_max=int(o["p_max"]), sigma=float(o["gabor_sigma"]), center=tuple(src_norm[0]))
            # resample pooled depletion pattern onto this grid via nearest (dep_pool is grid_n(12); for n=8 rebin)
            dep_g = _rebin(dep_pool, grid_n)
            ctrls = DA_controls(dep_g, shuffle_seed=cfg["DA_grid"]["shuffle_seed"])
            scaf = build_fixed_scaffold(gr, tuple(src_norm[0]), tuple(snk_norm[0]), ell_perp=o["ell_perp"],
                                        ar=o["ar"], mu_core=o["mu_core"], core_radius=o["core_radius_norm"], theta=0.0)
            for A_frac in A_fracs:
                A_val = A_frac * float(o["mu_core"])   # proxy: uniform coarse mu decrement = A_frac * core drive
                exc = build_excitability_field(gr, core, mu_core=o["mu_core"])
                exc.mu_core = exc.mu_core - A_val
                scaf["exc"] = exc                       # swap only the A-dependent excitability; kernels/core fixed
                for ctrl_name in (["primary"] + list(cfg["DA_grid"]["controls"])):
                    pat = ctrls[ctrl_name]
                    for D in D_vals:
                        q_field, D_field = build_DA_q_field(pat, float(D))
                        summ, _ = summarize_state_susceptibility(q_field, gr, scaf, probes, T_list,
                                                                w_ee_mult=o["w_ee_mult"], ratio=o["ratio"],
                                                                q_floor=o["q_floor"], T_primary=o["T_primary"])
                        row = _op_summary_row(label, "pooled", f"D{float(D):.3f}_A{A_frac}", ctrl_name, summ)
                        row.update(grid_n=grid_n, D_target=round(float(D), 4), A_frac=A_frac, A_coarse=round(A_val, 4), control=ctrl_name)
                        rows_grid.append(row)
            print(f"[operator-grid] (D,A) grid n={grid_n} done ({len(D_vals)} D x {len(A_fracs)} A x {1+len(cfg['DA_grid']['controls'])} patterns)", flush=True)
    _write_csv(rows_grid, os.path.join(OUT, "projected_phase_grid.csv"))
    _dump(dict(schema_version=SCHEMA_VERSION, candidate=label, seeds=seeds,
               note="natural-state operator reuses+verifies upstream susceptibility; (D,A) grid is realized-coordinate.",
               provenance=_provenance(cfg, dict(phase="operator-grid", candidate=label))),
          os.path.join(OUT, "per_seed", f"operator_grid_{label}.json"))


def _rebin(field, n):
    """Nearest-neighbor resample a (m,m) field to (n,n)."""
    field = np.asarray(field, float)
    m = field.shape[0]
    if m == n:
        return field.copy()
    ii = np.linspace(0, m - 1, n).round().astype(int)
    return field[np.ix_(ii, ii)]


# ============================================================ Task D: full-SNN nonlinear ignition
def cmd_ignition(args, cfg):
    which = args.candidate
    label, mzcfg, onsets = _cand(cfg, which)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    states = args.states.split(",") if args.states else cfg["ignition_states"] if "ignition_states" in cfg else \
        ["baseline_1000ms", "mid_fraction", "pre_onset_500ms", "pre_onset_100ms"]
    ig = cfg["ignition"]
    ladder = [float(a) for a in ig["amplitude_ladder"]]
    probe_steps = int(round(ig["probe_ms"] / DT))
    horizon_steps = int(round(ig["horizon_ms"] / DT))
    rows = []
    for seed in seeds:
        seed_json = os.path.join(OUT, "per_seed", f"ignition_{label}_seed{seed}.json")
        existing = json.load(open(seed_json)) if (args.resume and os.path.exists(seed_json)) else {"cells": {}}
        S, regions, core_mask_E, core_r = build_S(seed, cfg)
        onset_ms = onsets[seed]
        steps = state_steps(onset_ms, cfg)
        # probe targets: source core, sink core, off-axis control at matched distance from sheet center
        loc_masks = _probe_targets(S, regions, cfg)
        vth_gap = {loc: float(np.median(S["vth"][:S["NE"]][m] - S["p"].V_reset)) for loc, m in loc_masks.items()}
        for st in states:
            branch = steps[st]
            # replay ONCE to branch (freeze z,m), capture checkpoint
            base = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=core_mask_E).set_branch(branch_step=branch, freeze=True)
            S["net"]["rng"] = np.random.default_rng(seed)
            t0 = time.time()
            rep = run_loop(S["p"], S["net"], base, S["vth"], n_steps=branch, capture_final=True,
                           store_spikes=False, early_stop_runaway=False)
            ck = rep["checkpoint"]
            for loc, tgt in loc_masks.items():
                cell_key = f"{st}:{loc}"
                if args.resume and cell_key in existing["cells"]:
                    rows.append(existing["cells"][cell_key]); continue
                ran = []
                for a in ladder:
                    ra = _ignite_once(S, ck, tgt, vth_gap[loc], a, branch, probe_steps, horizon_steps, ig)
                    ran.append(ra["runaway"])
                eps = epsilon_c_from_ladder(ladder, ran)
                # optional bisection if bracketed
                bis = []
                if eps["bracket"] is not None and not eps["zero_runaway"]:
                    lo, hi = eps["bracket"]
                    for _ in range(int(ig["bisection_refinements"])):
                        mid = 0.5 * (lo + hi)
                        ra = _ignite_once(S, ck, tgt, vth_gap[loc], mid, branch, probe_steps, horizon_steps, ig)
                        bis.append(dict(a=round(mid, 5), runaway=ra["runaway"], onset_ms=ra["onset_ms"]))
                        if ra["runaway"]:
                            hi = mid
                        else:
                            lo = mid
                    eps["epsilon_c_refined"] = hi
                row = dict(candidate=label, seed=seed, state=st, location=loc, time_ms=branch * DT,
                           epsilon_c=eps["epsilon_c"], epsilon_c_refined=eps.get("epsilon_c_refined"),
                           censored=eps["censored"], zero_runaway=eps["zero_runaway"], vth_gap=round(vth_gap[loc], 4),
                           ladder=ladder, ran_away=ran, bisection=bis)
                existing["cells"][cell_key] = row
                rows.append(row)
                _dump(dict(candidate=label, seed=seed, onset_ms=onset_ms, cells=existing["cells"],
                           provenance=_provenance(cfg, dict(phase="ignition", candidate=label, seed=seed))), seed_json)
                print(f"[ignition] {label} s{seed} {cell_key} eps_c={eps['epsilon_c']} censored={eps['censored']} "
                      f"zero={eps['zero_runaway']} (branch replay {time.time()-t0:.0f}s)", flush=True)
    _write_csv([_flat_ig(r) for r in rows], os.path.join(OUT, f"nonlinear_ignition_summary_{label}.csv"))


def _flat_ig(r):
    return {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in r.items()}


def _probe_targets(S, regions, cfg):
    """Focal probe disks (E masks) at source core, sink core, and one off-axis control at matched distance
    from sheet center (spec §7.2)."""
    posE = np.asarray(S["posE"], float)
    center = np.asarray(S["center"], float)
    rk = float(cfg["ignition"]["probe_radius_norm_mm"])
    src = np.asarray(S["src_xy"], float); snk = np.asarray(S["snk_xy"], float)
    d_src = np.linalg.norm(src - center)
    u = np.asarray(S["axis_unit"], float)
    perp = np.array([-u[1], u[0]])
    off_center = center + perp * d_src                     # matched distance, perpendicular to axis
    return dict(source_core=np.linalg.norm(posE - src, axis=1) <= rk,
                sink_core=np.linalg.norm(posE - snk, axis=1) <= rk,
                off_axis=np.linalg.norm(posE - off_center, axis=1) <= rk)


def _ignite_once(S, ck, tgt, gap, a, branch, probe_steps, horizon_steps, ig):
    """Fork from the branch checkpoint: frozen z/m + 10ms threshold-lowering probe (amplitude a*gap) on tgt,
    continue horizon_ms under the same noise. Score operational runaway (120Hz/100ms)."""
    s = copy.deepcopy(ck.slow)
    if a > 0.0:
        s.set_probe(lo=branch, hi=branch + probe_steps, target_E=tgt, delta=a * gap)
    res = run_loop(S["p"], S["net"], s, S["vth"], n_steps=horizon_steps, start=ck, store_spikes=True,
                   early_stop_runaway=True, es_thresh_hz=ig["runaway_hz"], es_dur_ms=ig["runaway_dur_ms"])
    ra_ms = score_runaway(res["rate_E"], DT, thresh_hz=ig["runaway_hz"], dur_ms=ig["runaway_dur_ms"])
    if ra_ms is None and res["runaway_early_stop_step"] is not None:
        ra_ms = (res["runaway_early_stop_step"] - branch) * DT
    peak = float(np.max(res["rate_E"])) if res["rate_E"].size else 0.0
    part = float(res["E_spk_bool"].any(axis=0).mean()) if res["E_spk_bool"] is not None else float("nan")
    return dict(runaway=ra_ms is not None, onset_ms=ra_ms, peak_hz=round(peak, 1), participation=round(part, 4))


# ============================================================ Task E: z counterfactuals
def cmd_counterfactual(args, cfg):
    which = args.candidate
    label, mzcfg, onsets = _cand(cfg, which)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    zc = cfg["z_counterfactuals"]
    ig = cfg["ignition"]
    probe_steps = int(round(ig["probe_ms"] / DT)); horizon_steps = int(round(ig["horizon_ms"] / DT))
    rows = []
    for seed in seeds:
        seed_json = os.path.join(OUT, "per_seed", f"counterfactual_{label}_seed{seed}.json")
        existing = json.load(open(seed_json)) if (args.resume and os.path.exists(seed_json)) else {"cells": {}}
        S, regions, core_mask_E, core_r = build_S(seed, cfg)
        onset_ms = onsets[seed]; steps = state_steps(onset_ms, cfg)
        loc_masks = _probe_targets(S, regions, cfg)
        src_tgt = loc_masks["source_core"]
        gap = float(np.median(S["vth"][:S["NE"]][src_tgt] - S["p"].V_reset))
        for st in zc["states"]:
            branch = steps[st]
            base = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=core_mask_E).set_branch(branch_step=branch, freeze=True)
            S["net"]["rng"] = np.random.default_rng(seed)
            rep = run_loop(S["p"], S["net"], base, S["vth"], n_steps=branch, capture_final=True, store_spikes=False)
            ck = rep["checkpoint"]
            z_branch = ck.slow.z[:S["NE"]].copy()
            qeff_mean = float(np.nanmean(z_branch))
            transforms = _counterfactual_transforms(z_branch, regions, qeff_mean, zc["shuffle_seed"],
                                                    posE=S["posE"], L=S["L"], grid_n=int(cfg["operator"]["grid_n"]))
            for br in zc["branches"]:
                cell_key = f"{st}:{br}"
                if args.resume and cell_key in existing["cells"]:
                    rows.append(existing["cells"][cell_key]); continue
                out = {}
                for probe_a in (0.0, ig["amplitude_ladder"][2]):   # zero + one registered source probe (~0.05)
                    s = copy.deepcopy(ck.slow)
                    if br == "native_dynamic":
                        s.set_branch(branch_step=branch, freeze=False)   # continue dynamic z
                    elif br in transforms:
                        s.set_branch(branch_step=branch, freeze=True, z_transform=transforms[br])
                        # re-arm transform (freeze already set); apply at branch
                        s._z_transform_done = False; s._branch_step = branch; s._z_transform = transforms[br]; s._freeze = True
                    if probe_a > 0:
                        s.set_probe(lo=branch, hi=branch + probe_steps, target_E=src_tgt, delta=probe_a * gap)
                    res = run_loop(S["p"], S["net"], s, S["vth"], n_steps=horizon_steps, start=ck, store_spikes=True,
                                   early_stop_runaway=True, es_thresh_hz=ig["runaway_hz"], es_dur_ms=ig["runaway_dur_ms"])
                    ra = score_runaway(res["rate_E"], DT, thresh_hz=ig["runaway_hz"], dur_ms=ig["runaway_dur_ms"])
                    if ra is None and res["runaway_early_stop_step"] is not None:
                        ra = (res["runaway_early_stop_step"] - branch) * DT
                    out[f"probe{probe_a}"] = dict(runaway=ra is not None, onset_ms=ra,
                                                  peak_hz=round(float(np.max(res["rate_E"])) if res["rate_E"].size else 0.0, 1),
                                                  early_spread=round(float(res["E_spk_bool"][:500].any(axis=0).mean()), 4))
                row = dict(candidate=label, seed=seed, state=st, branch=br, z_mean_branch=round(qeff_mean, 4), **{f"{k}_{kk}": vv for k, d in out.items() for kk, vv in d.items()})
                existing["cells"][cell_key] = row; rows.append(row)
                _dump(dict(candidate=label, seed=seed, cells=existing["cells"],
                           provenance=_provenance(cfg, dict(phase="counterfactual", seed=seed))), seed_json)
                print(f"[counterfactual] {label} s{seed} {cell_key} -> {out}", flush=True)
    _write_csv(rows, os.path.join(OUT, f"z_counterfactual_summary_{label}.csv"))


def _rotate90_coarse_field(z, posE, L, n):
    """COARSE-FIELD 90-degree rotation of a per-E-neuron scalar field via an n x n grid round-trip (task §5.1).

    Bin each E neuron to its grid cell, take the per-cell MEAN, np.rot90 the coarse grid, then read each
    neuron's rotated value back from its own cell. This is NOT identity (its whole purpose) and preserves the
    COARSE field's grid-level histogram + spatial autocorrelation. But because every neuron in a cell receives
    the same cell mean, it does NOT preserve the neuron-level z histogram when grid cells hold unequal neuron
    counts -> it is a *coarse-field spatial control*, NOT a strict state-matched causal control (a true
    state-matched rotation needs a neuron-level permutation, which a non-lattice point cloud does not admit).
    FAIL-CLOSED: if any occupied target cell's rot90-source cell was empty, raise rather than invent a value.
    """
    z = np.asarray(z, float); pos = np.asarray(posE, float)
    if pos.shape[0] != z.size:
        raise ValueError(f"rotated_90: posE has {pos.shape[0]} rows but z has {z.size} entries")
    ix = np.clip(np.floor(pos[:, 0] / float(L) * n).astype(int), 0, n - 1)
    iy = np.clip(np.floor(pos[:, 1] / float(L) * n).astype(int), 0, n - 1)
    flat = iy * n + ix                                          # row-major: iy = row, ix = col
    sums = np.bincount(flat, weights=z, minlength=n * n).astype(float).reshape(n, n)
    cnts = np.bincount(flat, minlength=n * n).astype(float).reshape(n, n)
    with np.errstate(invalid="ignore"):
        grid = np.where(cnts > 0, sums / cnts, np.nan)
    z_rot = np.rot90(grid)[iy, ix]                              # each neuron reads its rotated-in value
    if not np.all(np.isfinite(z_rot)):
        raise ValueError(f"rotated_90 FAIL-CLOSED: {int((~np.isfinite(z_rot)).sum())}/{z.size} E neurons map "
                         f"to an empty rot90-source cell on the {n}x{n} grid; cannot rotate without inventing z")
    return z_rot


def _counterfactual_transforms(z_branch, regions, qeff_mean, shuffle_seed, *, posE, L, grid_n):
    """Per-E-neuron z-counterfactual transforms (task §5). Contract fixes vs the identity-fallback version:

    - ``uniform_mean_matched`` (was the misleadingly-named ``uniform_current_matched``): fills every E cell
      with the SPATIAL MEAN of z (nanmean). It matches mean disinhibition, NOT inhibitory current, so it is
      DEMOTED from any causal main analysis until a verified current-aware match (sum(z*I_I)/sum(I_I), which
      MZOnsetProbe.qeff_fields already computes) is run behind a bit-identical state-resume (task §5.2).
    - ``rotated_90`` is a COARSE-FIELD grid round-trip rotation (fail-closed), never ``lambda z: z`` — but it
      is NOT a strict state-matched causal control (per-cell mean does not preserve the neuron-level z
      histogram); see _rotate90_coarse_field. Treat it as a coarse spatial control only.
    """
    rng = np.random.default_rng(int(shuffle_seed))
    shuf = z_branch.copy(); rng.shuffle(shuf)
    return dict(
        native_frozen=lambda z: z,                                   # freeze at branch value (the ONLY identity arm)
        uniform_mean_matched=lambda z: np.full_like(z, qeff_mean),   # spatial mean of z (NOT current-matched)
        spatial_shuffle=lambda z, s=shuf: s.copy(),
        reset_one=lambda z: np.ones_like(z),
        rotated_90=lambda z: _rotate90_coarse_field(z, posE, L, grid_n),   # coarse-field rotation; fail-closed
    )


# ============================================================ Task F: event suppression
def cmd_event_suppress(args, cfg):
    which = "primary"
    label, mzcfg, onsets = _cand(cfg, which)
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else cfg["seeds"])]
    es = cfg["event_suppression"]; ig = cfg["ignition"]
    rows = []
    for seed in seeds:
        seed_json = os.path.join(OUT, "per_seed", f"event_suppress_seed{seed}.json")
        if args.resume and os.path.exists(seed_json):
            rows += json.load(open(seed_json))["rows"]; continue
        S, regions, core_mask_E, core_r = build_S(seed, cfg)
        onset_ms = onsets[seed]; steps = state_steps(onset_ms, cfg)
        # slow-off baseline to detect/calibrate the pulse on returning events (fixed amplitude, locked before target)
        tgt = regions["source_core"] | regions["axis_corridor"]
        gap = float(np.median(S["vth"][:S["NE"]][tgt] - S["p"].V_reset))
        # native replay to find the last 3 eligible returning events ending >=200ms before onset
        base = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=core_mask_E)
        S["net"]["rng"] = np.random.default_rng(seed)
        res = simulate_kick(S["p"]._replace(T=onset_ms + 100.0) if hasattr(S["p"], "_replace") else dataclasses.replace(S["p"], T=onset_ms + 100.0),
                            S["net"], 0.0, slow=base, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK, t_kick=1e9,
                            V_th_per_neuron=S["vth"], early_stop_runaway=False)
        events, af, bin_w, floor, rate = _events(res)
        ret = [e for e in events if e["returned"] and e["t_off"] < onset_ms - es["event_end_guard_ms"]]
        target_events = ret[-int(es["n_target_events"]):]
        # calibrate pulse amplitude on slow-off events: smallest cutting peak active frac >=50% w/o >200ms silence
        amp = _calibrate_suppression(S, mzcfg, core_mask_E, tgt, gap, target_events, es, ig)
        seed_rows = []
        for e in target_events:
            branch = int((e["t_on"] - es["pulse_lead_ms"]) / DT)
            base2 = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=core_mask_E)
            S["net"]["rng"] = np.random.default_rng(seed)
            rep = run_loop(S["p"], S["net"], base2, S["vth"], n_steps=branch, capture_final=True, store_spikes=False)
            ck = rep["checkpoint"]
            cont_steps = int((e["t_off"] + 600.0 - e["t_on"] + es["pulse_lead_ms"]) / DT)
            conds = {}
            for cond in ("no_pulse", "suppress", "sham"):
                s = copy.deepcopy(ck.slow)
                if cond == "suppress" and amp is not None:
                    s.set_suppression(lo=branch, hi=branch + int(es["pulse_ms"] / DT), target_E=tgt, delta=amp * gap)
                elif cond == "sham" and amp is not None:
                    s.set_suppression(lo=branch + int((e["dur_ms"] + 300.0) / DT), hi=branch + int((e["dur_ms"] + 300.0 + es["pulse_ms"]) / DT), target_E=tgt, delta=amp * gap)
                r = run_loop(S["p"], S["net"], s, S["vth"], n_steps=cont_steps, start=ck, store_spikes=True)
                paf = float(r["E_spk_bool"][:int((e["dur_ms"] + es["pulse_lead_ms"]) / DT)].any(axis=0).mean())
                conds[cond] = dict(event_peak_af=round(paf, 4), peak_hz=round(float(np.max(r["rate_E"])), 1))
            row = dict(candidate=label, seed=seed, t_on=round(e["t_on"], 1), amp=amp,
                       no_pulse_af=conds["no_pulse"]["event_peak_af"], suppress_af=conds["suppress"]["event_peak_af"],
                       sham_af=conds["sham"]["event_peak_af"],
                       removed_frac=(round(1 - conds["suppress"]["event_peak_af"] / conds["no_pulse"]["event_peak_af"], 3)
                                     if conds["no_pulse"]["event_peak_af"] > 0 else None),
                       unresolved=(amp is None))
            seed_rows.append(row); rows.append(row)
            print(f"[event-suppress] s{seed} ev@{e['t_on']:.0f} amp={amp} removed={row['removed_frac']}", flush=True)
        _dump(dict(candidate=label, seed=seed, calibrated_amp=amp, n_target_events=len(target_events),
                   rows=seed_rows, provenance=_provenance(cfg, dict(phase="event-suppress", seed=seed))), seed_json)
    _write_csv(rows, os.path.join(OUT, "event_suppression_summary.csv"))


def _calibrate_suppression(S, mzcfg, core_mask_E, tgt, gap, events, es, ig):
    """Smallest predeclared amplitude cutting the FIRST target event's peak active fraction >=50% without
    silencing >200ms. Calibrated on slow-off-derived events only; locked before target runs (spec §9)."""
    if not events:
        return None
    e = events[0]
    branch = int((e["t_on"] - es["pulse_lead_ms"]) / DT)
    base = MZOnsetProbe(S["N"], 18.0, mzcfg, NE=S["NE"], core_mask_E=core_mask_E)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    rep = run_loop(S["p"], S["net"], base, S["vth"], n_steps=branch, capture_final=True, store_spikes=False)
    ck = rep["checkpoint"]
    cont = int((e["dur_ms"] + 400.0 + es["pulse_lead_ms"]) / DT)
    r0 = run_loop(S["p"], S["net"], copy.deepcopy(ck.slow), S["vth"], n_steps=cont, start=ck, store_spikes=True)
    ev_steps = int((e["dur_ms"] + es["pulse_lead_ms"]) / DT)
    base_af = float(r0["E_spk_bool"][:ev_steps].any(axis=0).mean())
    if base_af <= 0:
        return None
    for a in es["amplitude_candidates"]:
        s = copy.deepcopy(ck.slow)
        s.set_suppression(lo=branch, hi=branch + int(es["pulse_ms"] / DT), target_E=tgt, delta=a * gap)
        r = run_loop(S["p"], S["net"], s, S["vth"], n_steps=cont, start=ck, store_spikes=True)
        ev_af = float(r["E_spk_bool"][:ev_steps].any(axis=0).mean())
        tail_af = float(r["E_spk_bool"][ev_steps:ev_steps + int(es["max_silence_ms"] / DT)].any(axis=0).mean())
        if ev_af <= 0.5 * base_af and tail_af > 0.0:      # cut >=50% without total post-silence
            return float(a)
    return None


# ============================================================ Task G: focused m push-pull
def _save_traj(S, mz, I_EE, ds_ms, runaway_ms, events, fname, *, z_regime, A_frac, seed, eta_m):
    """Save a continuous downsampled MZ D–a trajectory npz (temporal phase-diagram §5.3). rate is
    derived from trace_rate_E (same _record_traces length as z/adap traces -> no cross-array skew)."""
    rate_hz = np.asarray(mz.trace_rate_E, float) / float(S["NE"]) / (DT / 1000.0)
    traj = natural_zm_trajectory(mz.trace_z_mean, mz.trace_adap_current, rate_hz,
                                 dt=DT, I_EE_scale=I_EE, downsample_ms=ds_ms)
    np.savez_compressed(
        os.path.join(OUT, "per_seed", fname),
        t_ms=traj["t_ms"], D_allE=traj["D_allE"], a_allE=traj["a_allE"], rate_E_hz=traj["rate_E_hz"],
        runaway_ms=(np.nan if runaway_ms is None else float(runaway_ms)),
        event_on_ms=np.array([e["t_on"] for e in events], float),
        event_off_ms=np.array([e["t_off"] for e in events], float),
        z_regime=z_regime, A_frac=float(A_frac), seed=int(seed), eta_m=float(eta_m), i_ee_scale=float(I_EE))


def _aggregate_focused_m():
    """Combine the per-seed focused_m MAIN-grid JSONs into focused_m_summary.csv (task §4.1).

    Reads ONLY `focused_m_seed*_g*.json` (the tau-sweep `focused_m_tau*` files and the old-format
    `focused_m_seed*.json` without a `_g` tag are excluded by the glob), then `validate_focused_m_grid`
    fail-loudly rejects any missing / duplicate / tau-contaminated / schema-misaligned row before the CSV
    is written. This is a single explicit step (run after the parallel per-seed/per-frac processes finish),
    so there is no parallel-writer race and no stale partial CSV.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "per_seed", "focused_m_seed*_g*.json"))):
        rows += json.load(open(f)).get("rows", [])
    rows = validate_focused_m_grid(rows)                # raises on any grid defect (missing/dup/tau-mix/misalign)
    _write_csv(rows, os.path.join(OUT, "focused_m_summary.csv"))
    print(f"[focused-m] aggregated + validated {len(rows)} MAIN-grid rows -> focused_m_summary.csv", flush=True)


def _aggregate_tau_sensitivity(a_frac=0.001, taus=(2000, 1000, 500), seeds=(1, 3, 4)):
    """Independent tau-sensitivity summary at fixed A_frac (task §4.2). tau2000 rows come from the A_frac
    cell of the MAIN `focused_m_seed{S}_g0.json` files; tau1000/tau500 from `focused_m_tau{tau}_seed{S}_g0.001.json`.
    Writes focused_m_tau_sensitivity.{csv,json} (with per-tau phenotype denominators). Fail-loud on any
    missing/duplicate/misplaced cell."""
    by = {}
    for s in seeds:
        for tau in taus:
            if int(tau) == 2000:
                fn = os.path.join(OUT, "per_seed", f"focused_m_seed{s}_g0.json")
                cand = ([r for r in json.load(open(fn)).get("rows", []) if abs(float(r["A_frac"]) - a_frac) < 1e-12]
                        if os.path.exists(fn) else [])
            else:
                fn = os.path.join(OUT, "per_seed", f"focused_m_tau{int(tau)}_seed{s}_g0.001.json")
                cand = json.load(open(fn)).get("rows", []) if os.path.exists(fn) else []
            if len(cand) != 1:
                raise ValueError(f"tau-sensitivity: expected exactly 1 row for seed{s} tau{tau} A={a_frac}, "
                                 f"got {len(cand)} from {fn}")
            by[(int(s), float(tau))] = cand[0]
    rows, denom = build_tau_sensitivity(by, seeds=tuple(seeds), taus=tuple(float(t) for t in taus), a_frac=a_frac)
    _write_csv(rows, os.path.join(OUT, "focused_m_tau_sensitivity.csv"))
    _dump(dict(experiment="focused-m tau sensitivity (fixed A_frac, vary tau_adp)", a_frac=a_frac,
               taus_ms=[float(t) for t in taus], seeds=list(seeds), phenotype_denominators=denom, rows=rows),
          os.path.join(OUT, "focused_m_tau_sensitivity.json"))
    print(f"[focused-m] tau sensitivity denominators {denom} -> focused_m_tau_sensitivity.{{csv,json}}", flush=True)


def cmd_focused_m(args, cfg):
    if getattr(args, "aggregate", False):
        _aggregate_focused_m()
        _aggregate_tau_sensitivity()
        return
    fm = cfg["focused_m"]
    seeds = [int(s) for s in (args.seeds.split(",") if args.seeds else fm["seeds"])]
    tau_adp = float(fm["tau_adp_ms"]); I_EE = float(fm["I_EE_scale"]); peak_m = float(fm["peak_m_tau2000"])
    A_fracs = [float(a) for a in (args.a_fracs.split(",") if getattr(args, "a_fracs", None) else fm["A_fracs"])]
    z_regimes = fm["z_regimes"]
    if getattr(args, "regime", None):
        z_regimes = [z for z in z_regimes if z["label"] == args.regime]
        if not z_regimes:
            raise SystemExit(f"--regime {args.regime!r} not in focused_m.z_regimes")
    tau_over = getattr(args, "tau_adp_ms", None)
    if tau_over:
        tau_adp = float(tau_over)
    T_ms = float(args.t_ms) if getattr(args, "t_ms", None) else 15000.0
    ds_ms = float(fm.get("traj_downsample_ms", 5.0))
    tag = f"_tau{int(tau_adp)}" if tau_over else ""     # keep tau-sweep artifacts distinct
    from run_topic4_mz_slowvars import compute_baseline_ref, extract_run_metrics, run_mz_cell, slowoff_event_bar
    gtag = f"g{A_fracs[0]:g}"                            # frac-group tag: unique per parallel (seed,frac-subset) process
    for seed in seeds:
        seed_json = os.path.join(OUT, "per_seed", f"focused_m{tag}_seed{seed}_{gtag}.json")
        if args.resume and os.path.exists(seed_json):
            continue
        S, regions, core_mask_E, core_r = build_S(seed, cfg)
        # slow-off baseline + FROZEN event bar (P0-2): threshold set once on slow-off, reused for all cells
        res0, mz0 = run_mz_cell(S, MZSlowVarsConfig(use_z=False, use_m=False), T_ms, early_stop=False)
        baseline = compute_baseline_ref(res0, DT)
        frozen_bar = slowoff_event_bar(res0, DT)
        _save_traj(S, mz0, I_EE, ds_ms, None, [], f"traj_slow_off{tag}_{gtag}_seed{seed}.npz",
                   z_regime="slow_off", A_frac=0.0, seed=seed, eta_m=0.0)
        seed_rows = []
        for z in z_regimes:
            for A_frac in A_fracs:
                eta_m = 0.0 if A_frac == 0 else eta_m_from_frac(A_frac, I_EE, peak_m)
                cfg_cell = MZSlowVarsConfig(use_m=(A_frac > 0), tau_adp=tau_adp, eta_m=eta_m, **z["cfg"])
                # P0/§6.3: only pure z-only (A_frac==0) may early-stop; z+m cells run full T to see if m terminates
                res, mz = run_mz_cell(S, cfg_cell, T_ms, early_stop=(A_frac == 0))
                rm, events, af, bin_w, runaway_ms = extract_run_metrics(res, DT, baseline, event_bar=frozen_bar)
                pheno = classify_mz_run(rm, baseline, runaway_ms)
                adap_peak = float(max(mz.trace_adap_current)) if mz.trace_adap_current else 0.0
                d_max = round(1.0 - float(min(mz.trace_z_mean)), 5) if mz.trace_z_mean else 0.0
                row = dict(seed=seed, z_regime=z["label"], A_frac=A_frac, tau_adp_ms=tau_adp, eta_m=round(eta_m, 5),
                           realized_a_max=round(adap_peak / I_EE, 5), D_max=d_max, phenotype=pheno,
                           runaway_ms=runaway_ms, n_events=rm["n_events"],
                           peak_participation=round(rm["peak_participation"], 4), peak_returned=rm["peak_returned"],
                           event_bar=round(frozen_bar, 6))
                seed_rows.append(row)
                _save_traj(S, mz, I_EE, ds_ms, runaway_ms, events, f"traj_{z['label']}_A{A_frac}{tag}_seed{seed}.npz",
                           z_regime=z["label"], A_frac=A_frac, seed=seed, eta_m=eta_m)
                print(f"[focused-m] s{seed} {z['label']} A_target={A_frac} eta_m={eta_m:.5f} -> {pheno} "
                      f"(realized_a_max={row['realized_a_max']} D_max={d_max} runaway={runaway_ms})", flush=True)
        _dump(dict(seeds=seed, tau_adp_ms=tau_adp, rows=seed_rows,
                   provenance=_provenance(cfg, dict(phase="focused-m", seed=seed, tau_adp_ms=tau_adp))), seed_json)


# ============================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description="Topic 4 MZ early-onset dynamics runner (design 2026-07-19).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("state-coords", "operator-grid", "ignition", "counterfactual", "event-suppress", "focused-m"):
        sp = sub.add_parser(name)
        sp.add_argument("--confirm-run", action="store_true")
        sp.add_argument("--candidate", default="primary", choices=["primary", "sensitivity"])
        sp.add_argument("--seeds", default=None)
        sp.add_argument("--states", default=None)
        sp.add_argument("--resume", action="store_true")
        sp.add_argument("--a-fracs", dest="a_fracs", default=None, help="focused-m: comma-separated A_frac subset override")
        sp.add_argument("--regime", default=None, help="focused-m: z-regime label subset override")
        sp.add_argument("--t-ms", dest="t_ms", default=None, help="focused-m: sim length override (ms); default 15000")
        sp.add_argument("--aggregate", action="store_true", help="focused-m: combine per-seed JSONs into summary CSV (no sim)")
        sp.add_argument("--tau-adp-ms", dest="tau_adp_ms", default=None, help="focused-m: tau_adp override (ms), tau-sweep")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    needs_run = {"state-coords", "operator-grid", "ignition", "counterfactual", "event-suppress", "focused-m"}
    if args.cmd in needs_run and not args.confirm_run and not getattr(args, "aggregate", False):
        print(f"REFUSING: '{args.cmd}' runs simulations. Pass --confirm-run.", file=sys.stderr)
        sys.exit(2)
    os.makedirs(os.path.join(OUT, "per_seed"), exist_ok=True)
    {"state-coords": cmd_state_coords, "operator-grid": cmd_operator_grid, "ignition": cmd_ignition,
     "counterfactual": cmd_counterfactual, "event-suppress": cmd_event_suppress, "focused-m": cmd_focused_m}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
