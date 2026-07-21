#!/usr/bin/env python
"""Stage-1a exit-boundary probe -- SNN-native M4 containment-to-exit line.

CHEAP-FIRST GATE (spec 2026-07-21 §13). Open-loop, ZERO engine change: fork the accepted
M4 bounded state (k_q=0.10, alpha_G=16) and apply a clamped threshold displacement
(`inhibitory_pulse`: raise E V_th by DVTH) over a window [t0, t1], then observe the
post-release [t1, T] behaviour. Tests the q_I-refill exit hypothesis (spec §13, §5):

  short hold  -> q_I still depleted at release -> re-ignites (rebound / runaway)   [known: M4 500ms pulse rebounds]
  long  hold  -> firing stops, q_I REFILLS toward 1 over tau_q(=5000ms) during the quiet ->
                 on release the network is in a high-q_I interictal-like basin -> stays low.

If NO hold gives clean exit-and-stay for any reachable displacement => bounded-negative gate (stop).
If a hold does => calibrates tau_p (recovery hold must be ~ q_I refill time) for the dynamic field.

Reuses run_m4_phaseplane.build_substrate + run_m4_dynamic_qi.run_arm + sef_hfo_m4_termination.
Build ONCE, Pool over cells (COW). OMP forced to 1 by run_m4_dynamic_qi import. Gated by --confirm-run.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_m4_phaseplane as PP          # noqa: E402
import run_m4_dynamic_qi as M4          # noqa: E402  (forces OMP=1 at import; provides run_arm/_S/_EARLY_STOP/DT + helpers)
from kick_probe import simulate_kick    # noqa: E402
from slow_field import SpatialSlowField, SpatialSlowFieldConfig  # noqa: E402
from src.sef_hfo_m4_termination import classify_termination  # noqa: E402
from src.sef_hfo_snn_engine_guard import record_versions, assert_versions  # noqa: E402

BASE_KQ, BASE_AG = 0.10, 16.0
ARR_KEYS = ("trace_qI_mean", "trace_SG", "trace_Irec", "rate", "af", "movie", "q_field_final")

# ---- provenance / engine drift guard (spec 2026-07-21 §5, §12) --------------------------------------
# The mechanism lives in the UNGUARDED slow_field.py (kick_probe/params/model/connectivity(_rot)/lfp stay
# frozen -> no re-bless). This bless snapshot pins exactly those guarded files; _engine_guard() fails loud
# if any drifted, so an unreviewed engine edit can't silently contaminate a run.
ENGINE_VERSIONS = os.path.join(PP.ROOT, "results", "topic4_sef_hfo", "snn_heterogeneity", "engine_versions.json")
_GUARDED_ENGINE = ("kick_probe.py", "params.py", "model.py", "connectivity.py", "connectivity_rot.py", "lfp.py")

# module-local manifest state so a COW-forked _arm_worker can drop a "running" marker without threading
# out/tag/seed through the Pool (set by _orchestrate_arms BEFORE the Pool forks).
_MANI = {}


def _engine_guard():
    """Loud fail if a GUARDED engine file drifted from the bless snapshot (spec §12). A missing snapshot
    is a warning, not a hard stop -- git history is the primary integrity record (sef_hfo_snn_engine_guard
    docstring); slow_field.py is deliberately absent (this line's mechanism edits it)."""
    if not os.path.exists(ENGINE_VERSIONS):
        print(f"[engine-guard] WARN: bless snapshot missing ({ENGINE_VERSIONS}); skipping drift check", flush=True)
        return
    assert_versions(json.loads(open(ENGINE_VERSIONS).read()))


def _provenance():
    """base_sha (git HEAD) + engine_versions (sha256 of the GUARDED engine files) + argv (spec §6/§12
    schema). Recorded in every run JSON + the manifest so a result is reproducible without the dir name."""
    try:
        base_sha = subprocess.check_output(["git", "-C", PP.ROOT, "rev-parse", "--short", "HEAD"],
                                           text=True).strip() or None
    except Exception:
        base_sha = None
    eng_dir = os.path.join(PP.ROOT, "src", "snn_engine")
    paths = [os.path.join(eng_dir, f) for f in _GUARDED_ENGINE]
    return dict(base_sha=base_sha, engine_versions=record_versions([p for p in paths if os.path.exists(p)]),
                argv=" ".join(sys.argv))


def _cfg_effective(cfg):
    """Full persistence + M4 param snapshot for a run row / manifest (spec §6 schema). Distinguishes the
    four axes the task brief §3 requires -- onset (persist_onset_ms) / tau_up (tau_p) / tau_down
    (tau_p_down) / actuator (eta_r) -- and records k_q / alpha_G. persist_onset_ms + clamp_persist were
    missing from the pre-Phase-0 inline dict (silently dropped)."""
    return dict(use_persist=cfg.use_persist, tau_p=cfg.tau_p, tau_p_down=cfg.tau_p_down,
                persist_onset_ms=cfg.persist_onset_ms, theta_p=cfg.theta_p, a50_p=cfg.a50_p,
                sigma_p=cfg.sigma_p, eta_r=cfg.eta_r, p50_r=cfg.p50_r, n_r=cfg.n_r,
                clamp_persist=cfg.clamp_persist, k_q=cfg.k_q, use_SG=cfg.use_SG, alpha_G=cfg.alpha_G)


# ---- crash-safe per-arm output + run_manifest + resume (task brief §3) --------------------------------
def _arm_dir(out, tag, seed):
    return os.path.join(out, "per_arm", f"{tag}_seed{seed}")


def _write_arm_result(arm_dir, row, arrays):
    """Land one arm's row (JSON) + arrays (NPZ) the moment it finishes (npz first so a json failure can't
    orphan arrays). Returns (json_path, npz_path). An interrupt now loses at most the in-flight arm."""
    os.makedirs(arm_dir, exist_ok=True)
    label = row["label"]
    npzp = os.path.join(arm_dir, f"{label}.npz")
    jp = os.path.join(arm_dir, f"{label}.json")
    np.savez_compressed(npzp, **{k: np.asarray(v) for k, v in (arrays or {}).items()})
    json.dump(_sanitize(row), open(jp, "w"), indent=2, allow_nan=False)
    return jp, npzp


def _load_completed_arms(arm_dir):
    """label -> row for every per-arm JSON that parses AND has no 'error' (an error row is NOT complete,
    so --resume re-runs it). Missing dir -> {}."""
    out = {}
    if not os.path.isdir(arm_dir):
        return out
    for fn in sorted(os.listdir(arm_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            row = json.load(open(os.path.join(arm_dir, fn)))
        except Exception:
            continue
        if isinstance(row, dict) and "error" not in row and "label" in row:
            out[row["label"]] = row
    return out


def _manifest_dict(specs, results, running, provenance, meta):
    """Per-arm status: complete (row in, no error) / error (row has 'error') / running (worker started) /
    pending (submitted, not started). cfg_effective snapshot per arm so onset/tau_up/tau_down/actuator are
    recoverable from the manifest alone."""
    arms = {}
    for (label, cfg, T_ms, perturb) in specs:
        r = results.get(label)
        if r is not None and "error" in r:
            status = "error"
        elif r is not None:
            status = "complete"
        elif label in running:
            status = "running"
        else:
            status = "pending"
        arms[label] = dict(status=status, cfg_effective=_cfg_effective(cfg), T_ms=float(T_ms),
                           error=(r.get("error") if r and "error" in r else None),
                           verdict=(r.get("verdict") if r else None),
                           termination_class=(r.get("termination_class") if r else None))
    return dict(provenance=provenance, meta=meta, n_arms=len(specs),
                n_complete=sum(1 for a in arms.values() if a["status"] == "complete"),
                arms=arms)


def _scan_running(arm_dir, results):
    """Labels whose worker dropped a `_running_<label>` marker and haven't produced a result yet."""
    if not os.path.isdir(arm_dir):
        return set()
    return {fn[len("_running_"):] for fn in os.listdir(arm_dir)
            if fn.startswith("_running_")} - set(results)


def _orchestrate_arms(specs, out, tag, seed, provenance, meta, workers, run_one, resume=False):
    """Run `specs` through `run_one(spec)->(row, arrays)`, landing each arm's JSON+NPZ + rewriting
    run_manifest_<tag>_seed<seed>.json AS each completes (crash-safe). resume=True skips arms whose
    per-arm JSON is already complete. workers<=1 runs serially (also the injectable-fake test path); else
    a fork Pool + imap_unordered streams completions. Returns rows for ALL target arms
    (loaded-from-disk + freshly computed)."""
    os.makedirs(out, exist_ok=True)
    arm_dir = _arm_dir(out, tag, seed)
    os.makedirs(arm_dir, exist_ok=True)
    _MANI["arm_dir"] = arm_dir                            # COW-inherited by forked workers
    results = dict(_load_completed_arms(arm_dir)) if resume else {}
    pending = [s for s in specs if s[0] not in results]

    def _flush():
        _write_manifest(out, tag, seed,
                        _manifest_dict(specs, results, _scan_running(arm_dir, results), provenance, meta))

    _flush()

    def _consume(res):
        row, arrays = res
        label = row["label"]
        _write_arm_result(arm_dir, row, arrays or {})
        mk = os.path.join(arm_dir, f"_running_{label}")
        if os.path.exists(mk):
            os.remove(mk)
        results[label] = row
        _flush()

    if workers <= 1:
        for spec in pending:
            _consume(run_one(spec))
    elif pending:
        with mp.Pool(min(workers, len(pending))) as pool:
            for res in pool.imap_unordered(run_one, pending):
                _consume(res)
    return [results[s[0]] for s in specs if s[0] in results]


def _write_manifest(out, tag, seed, manifest):
    path = os.path.join(out, f"run_manifest_{tag}_seed{seed}.json")
    json.dump(_sanitize(manifest), open(path, "w"), indent=2, allow_nan=False)
    return path


def _sanitize(obj):
    """Recursively replace non-finite floats with None so json.dump(allow_nan=False) never raises
    (spec §12: strict JSON, non-finite -> null). NaN/inf can arise from empty readout windows."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def _levels(r, t0, t1, T):
    """Bounded pre-pulse level, in-hold level, post-release level, and q_I refill readout."""
    af = np.asarray(r["af"], float)
    bw = float(r["bin_w"])
    qi = np.asarray(r["trace_qI_mean"], float)   # per-step mean q_I (dt = M4.DT)
    dt = float(M4.DT)

    def af_win(a_ms, b_ms):
        i0, i1 = max(0, int(a_ms / bw)), min(af.size, int(b_ms / bw))
        m = float(af[i0:i1].mean()) if i1 > i0 else None      # empty window (e.g. baseline post) -> None, not nan
        return round(m, 5) if m is not None else None

    def qi_at(ms):
        return round(float(qi[min(qi.size - 1, max(0, int(ms / dt)))]), 4)

    return dict(
        pre_af=af_win(t0 - 1000, t0),
        hold_af=af_win(max(t0, t1 - 500), t1),
        post_af=af_win(t1 + 500, T),
        qI_t0=qi_at(t0), qI_t1=qi_at(t1), qI_final=round(float(qi[-1]), 4),
    )


def _verdict(r, lv, t1):
    """Post-release verdict: did the bounded state stay down after the hold released?"""
    ra = r.get("runaway_ms")
    if ra is not None and ra > t1 - 50.0:
        return "rebound_runaway"
    pre, post = lv["pre_af"], lv["post_af"]
    if not pre or pre <= 1e-9:
        return "no_bounded_pre"
    if post is None:
        return "partial"
    frac = post / pre
    if frac >= 0.5:
        return "rebound_bounded"
    if frac < 0.2:
        return "exit_stay_low"
    return "partial"


def _cell_worker(cell):
    label, perturb, t0, t1, T_ms = cell
    S = M4._S["S"]
    try:
        r = M4.run_arm(S, label, BASE_KQ, True, BASE_AG, perturb=perturb, T_ms=T_ms)
    except Exception as e:  # fail-loud per cell; don't kill the sweep
        return dict(label=label, error=repr(e)), None
    lv = _levels(r, t0, t1, T_ms) if perturb is not None else _levels(r, t0, T_ms, T_ms)
    cls, info = classify_termination(np.asarray(r["af"], float), float(r["bin_w"]),
                                     baseline=r["baseline_af"], runaway_ms=r.get("runaway_ms"))
    row = dict(
        label=label, seed=S["seed"], k_q=BASE_KQ, alpha_G=BASE_AG,
        t0_ms=t0, t1_ms=t1, hold_ms=(t1 - t0 if perturb is not None else 0.0), T_ms=T_ms,
        dvth=(perturb["val"] if perturb else 0.0),
        verdict=(_verdict(r, lv, t1) if perturb is not None else "baseline_" + r["verdict"]),
        termination_class=cls, offset_ms=info["offset_ms"],
        m4_verdict=r["verdict"], runaway_ms=r.get("runaway_ms"), max_rate_hz=r["max_rate_hz"],
        q_min_final=r["q_min_final"], q_mean_final=r["q_mean_final"], S_G_max=r["S_G_max"],
        active_area_peak=r.get("active_area_peak"), active_area_tail=r.get("active_area_tail"),
        wall_s=r["wall_s"], **lv,
    )
    arrays = {k: np.asarray(r[k], np.float32) for k in ARR_KEYS if k in r}
    return row, arrays


# ===========================================================================================
# Stage-2 dynamic arms (spec §8): the M4 base (q_I depletion + S_G pool) + the persistence-gated
# recovery field p(x,t). Reuses the M4 readout helpers (M4.C / _smooth / _first_sustained /
# _spatial_*) so labels are directly comparable to run_m4_dynamic_qi. Persist params via CLI.
# ===========================================================================================
def _persist_cfg(*, k_q=BASE_KQ, use_SG=True, alpha_G=BASE_AG, use_persist=False,
                 tau_p=5000.0, theta_p=0.0, a50_p=1.0, sigma_p=1.5, eta_r=0.0,
                 p50_r=0.0, n_r=2.0, clamp_persist=None, tau_p_down=None, persist_onset_ms=0.0):
    """M4 base config (k_q depletion + S_G pool, params from run_m4_dynamic_qi) + persistence field."""
    return SpatialSlowFieldConfig(
        use_qI=True, k_q=k_q, sigma_q=M4.SIGMA_Q, sigma_K=0.5, q_min=M4.Q_MIN, q_init=1.0,
        tau_q=M4.TAU_Q, tau_a=M4.TAU_A, use_gK=False, k_K=0.0,
        use_SG=use_SG, alpha_G=alpha_G, r0_psi=0.0, r50_psi=M4.R50_PSI, n_psi=M4.N_PSI,
        p_pool=M4.P_POOL, tau_mu=M4.TAU_MU, tau_S=M4.TAU_S, S_max=M4.S_MAX,
        use_persist=use_persist, tau_p=tau_p, theta_p=theta_p, a50_p=a50_p, sigma_p=sigma_p,
        eta_r=eta_r, p50_r=p50_r, n_r=n_r, clamp_persist=clamp_persist, tau_p_down=tau_p_down,
        persist_onset_ms=persist_onset_ms)


# ---- spatial readouts (Phase 1 output list: core/surround activity + axis/transverse kymographs) --------
def _core_mask_E(S):
    """E-only bool mask of the two low-threshold cores (E cells within PP.CORE_R of source/sink centroid).
    Reuses the stim-locus disk convention (run_m4_dynamic_qi._e_disk_mask), sliced to E."""
    return M4._e_disk_mask(S, [S["src_xy"], S["snk_xy"]], PP.CORE_R)[:S["NE"]]


def _kymograph(spk_E, posE, origin, unit, bin_ms=25.0, n_space=24):
    """Active-fraction kymograph: project E positions onto `unit` (relative to `origin`), bin space x time.
    Returns (n_frames, n_space) fraction of E cells in each spatial bin active per bin_ms frame."""
    proj = (np.asarray(posE, float) - np.asarray(origin, float)) @ np.asarray(unit, float)
    edges = np.linspace(proj.min(), proj.max(), n_space + 1)
    sidx = np.clip(np.digitize(proj, edges) - 1, 0, n_space - 1)
    per_bin = np.bincount(sidx, minlength=n_space).astype(float); per_bin[per_bin == 0] = 1.0
    bs = int(round(bin_ms / M4.DT))
    frames = [np.bincount(sidx[spk_E[b0:b0 + bs].any(axis=0)], minlength=n_space).astype(float) / per_bin
              for b0 in range(0, spk_E.shape[0], bs)]
    return np.asarray(frames, np.float32)


def _core_surround_activity(spk_E, core_mask_E, bin_ms=25.0):
    """Per-bin_ms E firing rate (Hz) inside the cores vs the surround (Phase 1 output: core/surround activity)."""
    bs = int(round(bin_ms / M4.DT))
    nc, ns = max(1, int(core_mask_E.sum())), max(1, int((~core_mask_E).sum()))
    core, surr = [], []
    for b0 in range(0, spk_E.shape[0], bs):
        seg = spk_E[b0:b0 + bs]
        dt_win = seg.shape[0] * M4.DT
        core.append(float(seg[:, core_mask_E].sum()) / nc / dt_win * 1e3)
        surr.append(float(seg[:, ~core_mask_E].sum()) / ns / dt_win * 1e3)
    return np.asarray(core, np.float32), np.asarray(surr, np.float32)


def _run_persist_arm(S, label, cfg, T_ms, perturb=None):
    """One spontaneous (KICK_BOOST=0) arm with the persistence field; full readout + termination class."""
    p = dataclasses.replace(S["p"], T=float(T_ms))
    core_mask_E = _core_mask_E(S)
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], core_mask_E=core_mask_E, cfg=cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    t0 = time.time()
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]), r_kick=PP.R_KICK,
                        t_kick=1e9, V_th_per_neuron=S["vth"], perturb=perturb,
                        early_stop_runaway=M4._EARLY_STOP["on"])  # runaway (>120Hz sustained) truncates; clean-exit/bounded unaffected
    spk = res["E_spk_bool"]
    rate = np.asarray(res["rate_E"], float)
    af, bin_w = M4.C.active_fraction(spk, M4.DT, M4.C.BIN_MS)
    nb0, nb1 = int(M4.C.BASELINE_MS[0] / bin_w), int(M4.C.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + M4.C.CAL_FRAC * (float(af.max()) - floor)
    events = M4.C.detect_events(af, bin_w, event_on_frac=bar)
    rate_s = M4._smooth(rate, M4.DT)
    runaway = M4._first_sustained(rate_s, M4.DT)
    n_pre = sum(1 for e in events if runaway is None or e["t_on"] < runaway - 20.0)
    verdict = ("no_runaway" if runaway is None else "train_then_runaway" if (n_pre >= 2 and runaway > 200.0)
               else "one_shot_burst" if (runaway <= 200.0 or n_pre == 0) else "few_events_then_runaway")
    cls, info = classify_termination(af, bin_w, baseline=floor, runaway_ms=runaway)
    movie = M4._spatial_movie(spk, S["posE"], S["L"], M4.DT)
    axis_u = np.asarray(S["axis_unit"], float)                    # source->sink (connection long axis)
    perp_u = np.array([-axis_u[1], axis_u[0]])                    # transverse
    kymo_axis = _kymograph(spk, S["posE"], S["center"], axis_u)
    kymo_transverse = _kymograph(spk, S["posE"], S["center"], perp_u)
    core_act, surr_act = _core_surround_activity(spk, core_mask_E)
    row = dict(
        label=label, seed=S["seed"], verdict=verdict, termination_class=cls, offset_ms=info["offset_ms"],
        runaway_ms=runaway, max_rate_hz=round(float(rate_s.max()), 1), n_events=len(events), n_pre_runaway=int(n_pre),
        q_min_final=round(float(slow.q_I.min()), 4), q_mean_final=round(float(slow.q_I.mean()), 4),
        q_core_final=round(float(slow.trace_q_core[-1]), 4) if slow.trace_q_core else None,
        q_surround_final=round(float(slow.trace_q_surround[-1]), 4) if slow.trace_q_surround else None,
        S_G_max=round(float(max(slow.trace_SG)) if slow.trace_SG else 0.0, 4),
        p_mean_final=round(float(slow.p.mean()), 4), p_max_final=round(float(slow.p.max()), 4),
        p_core_final=round(float(slow.trace_p_core[-1]), 4) if slow.trace_p_core else None,
        p_surround_final=round(float(slow.trace_p_surround[-1]), 4) if slow.trace_p_surround else None,
        p_peak=round(float(max(slow.trace_p_max)) if slow.trace_p_max else 0.0, 4),
        T_ms=float(T_ms), perturb_kind=(perturb["kind"] if perturb else None),
        cfg_effective=_cfg_effective(cfg),
        wall_s=round(time.time() - t0, 1), **M4._spatial_coverage(movie),
        events=[(round(e["t_on"], 1), round(e["t_off"], 1)) for e in events],
    )
    arrays = dict(
        trace_qI_mean=np.asarray(slow.trace_qI_mean, np.float32),
        trace_q_core=np.asarray(slow.trace_q_core, np.float32),
        trace_q_surround=np.asarray(slow.trace_q_surround, np.float32),
        trace_SG=(np.asarray(slow.trace_SG, np.float32) if slow.trace_SG else np.zeros(0, np.float32)),
        trace_p_mean=(np.asarray(slow.trace_p_mean, np.float32) if slow.trace_p_mean else np.zeros(0, np.float32)),
        trace_p_max=(np.asarray(slow.trace_p_max, np.float32) if slow.trace_p_max else np.zeros(0, np.float32)),
        trace_p_core=(np.asarray(slow.trace_p_core, np.float32) if slow.trace_p_core else np.zeros(0, np.float32)),
        trace_p_surround=(np.asarray(slow.trace_p_surround, np.float32) if slow.trace_p_surround else np.zeros(0, np.float32)),
        kymo_axis=kymo_axis, kymo_transverse=kymo_transverse,
        core_activity=core_act, surround_activity=surr_act,
        rate=rate.astype(np.float32), af=af.astype(np.float32), movie=movie,
        q_field_final=slow.q_I.astype(np.float32), p_field_final=slow.p.astype(np.float32),
    )
    return row, arrays


def _arm_worker(spec):
    label, cfg, T_ms, perturb = spec
    ad = _MANI.get("arm_dir")
    if ad:                                                # drop a "running" marker so the manifest can
        try:                                              # distinguish in-flight from queued arms
            open(os.path.join(ad, f"_running_{label}"), "w").close()
        except Exception:
            pass
    S = M4._S["S"]
    try:
        return _run_persist_arm(S, label, cfg, T_ms, perturb=perturb)
    except Exception as e:                                # fail-loud per arm
        return dict(label=label, error=repr(e)), None


def _build_arms(a):
    """Arms A-E (spec §8), or a D-only (tau_p:eta_r) sweep. P = calibrated persistence params (from Stage-1)."""
    P = dict(tau_p=a.tau_p, theta_p=a.theta_p, a50_p=a.a50_p, sigma_p=a.sigma_p, p50_r=a.p50_r, n_r=a.n_r,
             tau_p_down=a.tau_p_down, persist_onset_ms=a.persist_onset_ms)
    T = a.T
    base = dict(k_q=BASE_KQ, use_SG=True, alpha_G=BASE_AG)
    if a.d_sweep:                          # arm-D (tau_p:eta_r) grid, one build shared across cells
        cells = []
        if a.include_anchor:
            cells.append(("B_m4_anchor", _persist_cfg(**base), T, None))
        for tok in a.d_sweep.split(","):
            tp, er = (float(x) for x in tok.split(":"))
            # build from P (single source of persist params incl tau_p_down + persist_onset_ms) with tau_p
            # overridden per cell, so a param can never be silently dropped again. Label distinguishes the
            # four brief-§3 axes: tau_up (tau{tp}) / actuator (eta{er}) / tau_down (dn) / onset (on).
            lab = (f"D_tau{int(tp)}_eta{er:g}"
                   + (f"_dn{int(a.tau_p_down)}" if a.tau_p_down else "")
                   + (f"_on{int(a.persist_onset_ms)}" if a.persist_onset_ms else ""))
            cells.append((lab, _persist_cfg(**base, use_persist=True, eta_r=er, **{**P, "tau_p": tp}), T, None))
        return cells
    catalog = {
        "A_slow_off":  (_persist_cfg(k_q=0.0, use_SG=False), T),
        "A_sensor_on": (_persist_cfg(k_q=0.0, use_SG=False, use_persist=True, eta_r=0.0, **P), T),  # p on real IEDs, actuator off
        "A_persist_act": (_persist_cfg(k_q=0.0, use_SG=False, use_persist=True, eta_r=a.eta_r, **P), T),  # slow-off + candidate actuator ON -> do real IEDs survive? (prevention test)
        "B_m4_anchor": (_persist_cfg(**base), T),
        "C_sensor_on": (_persist_cfg(**base, use_persist=True, eta_r=0.0, **P), T),   # p evolves, actuator off
        "D_full":      (_persist_cfg(**base, use_persist=True, eta_r=a.eta_r, **P), T),
        "E1_no_qI":    (_persist_cfg(k_q=0.0, use_SG=True, alpha_G=BASE_AG, use_persist=True, eta_r=a.eta_r, **P), T),
        "E2_no_SG":    (_persist_cfg(k_q=BASE_KQ, use_SG=False, use_persist=True, eta_r=a.eta_r, **P), T),
        "E4_clamp_p":  (_persist_cfg(**base, use_persist=True, eta_r=a.eta_r, clamp_persist=a.clamp_val, **P), T),
    }
    want = [x for x in a.arms.split(",") if x]
    return [(name, catalog[name][0], catalog[name][1], None) for name in want if name in catalog]


# ===========================================================================================
# Phase 2 (task brief §5): frozen dual-initial-state exit atlas. Freeze the slow coordinates
#   q_core (uniform frozen q_I via k_q=0 + q_init), S_G (clamp_SG), J_exit (frozen uniform outward
#   recovery current eta_r*Phi(1), linear Phi) -- and run the FULL SNN fast subsystem short from TWO
#   initial conditions: cold (spontaneous, no kick) and warm (a strong core kick at t~0, an
#   established-M4-fast-state surrogate; kick_probe is guarded so a true V-checkpoint isn't available).
#   cold->low AND warm->high at the same frozen coords => BISTABLE cell; both->low => monostable low
#   (no ictal branch); both->high => monostable high (no low basin to exit into). Answers the stop-rule:
#   does a low/interictal basin exist near recovered q_core + high S_G/J_exit? NO ENGINE EDIT.
# ===========================================================================================
ATLAS_KICK_BOOST, ATLAS_KICK_R, ATLAS_KICK_T = 6.0, 1.5, 50.0     # warm-IC core ignition (2x focal KICK, core-wide)


def _atlas_cfg(q_core, S_G, J_exit):
    """Frozen-slow config for one exit-atlas cell (spec §5). q_core = uniform frozen q_I (k_q=0 -> ODE
    skipped, stays q_init); S_G = clamped divisive pool; J_exit = frozen uniform outward recovery current
    eta_r*Phi(1) with linear Phi (clamp_persist=1). J_exit=0 -> no persistence coupling."""
    return SpatialSlowFieldConfig(
        use_qI=True, k_q=0.0, q_init=float(q_core), sigma_q=M4.SIGMA_Q, sigma_K=0.5, q_min=0.0,
        tau_q=M4.TAU_Q, tau_a=M4.TAU_A, use_gK=False, k_K=0.0,
        use_SG=True, alpha_G=BASE_AG, clamp_SG=float(S_G), r0_psi=0.0, r50_psi=M4.R50_PSI, n_psi=M4.N_PSI,
        p_pool=M4.P_POOL, tau_mu=M4.TAU_MU, tau_S=M4.TAU_S, S_max=M4.S_MAX,
        use_persist=(J_exit > 0.0), clamp_persist=1.0, p50_r=0.0, n_r=2.0, eta_r=float(J_exit),
        sigma_p=1.5, a50_p=1.0)


def _classify_atlas(settled_rate, settled_cv, area_tail, runaway, low_hz=4.0, cv_burst=0.6):
    """Steady-state class of a frozen-slow fast run over the settled window: runaway / low(interictal-like)
    / bounded_oscillatory (sustained but bursty) / bounded_high (sustained, broad) / fragment (some activity,
    no spatial extent). Thresholds reported raw in the row so the class is transparent + adjustable."""
    if runaway is not None:
        return "runaway"
    if settled_rate < low_hz and area_tail < 0.05:
        return "low"
    if settled_cv >= cv_burst:
        return "bounded_oscillatory"
    if area_tail >= 0.05:
        return "bounded_high"
    return "fragment"


def _build_atlas_cells(a):
    """(q_core x S_G x J_exit) grid, each from cold + warm IC. Cell = (label, cfg, T_ms, warm)."""
    qs = [float(x) for x in a.q_core_grid.split(",")]
    sgs = [float(x) for x in a.sg_grid.split(",")]
    js = [float(x) for x in a.j_exit_grid.split(",")]
    cells = []
    for q in qs:
        for sg in sgs:
            for j in js:
                cfg = _atlas_cfg(q, sg, j)
                base = f"q{q:g}_sg{sg:g}_j{j:g}"
                for warm in (False, True):
                    cells.append((f"{base}_{'warm' if warm else 'cold'}", cfg, a.T, warm))
    return cells


def _run_atlas_cell(S, label, cfg, T_ms, warm):
    """One frozen-slow short run from a cold (no kick) or warm (strong core kick at t~0) IC; classify the
    settled fast state. Saves the spatial movie (full field, not just the mean) per the atlas contract."""
    p = dataclasses.replace(S["p"], T=float(T_ms))
    core_mask_E = _core_mask_E(S)
    slow = SpatialSlowField(S["N"], 18.0, S["posE"], S["posI"], S["L"], core_mask_E=core_mask_E, cfg=cfg)
    S["net"]["rng"] = np.random.default_rng(S["seed"])
    t0 = time.time()
    kb = ATLAS_KICK_BOOST if warm else 0.0
    tk = ATLAS_KICK_T if warm else 1e9
    res = simulate_kick(p, S["net"], kb, slow=slow, kick_center=list(S["src_xy"]), r_kick=ATLAS_KICK_R,
                        t_kick=tk, V_th_per_neuron=S["vth"], early_stop_runaway=M4._EARLY_STOP["on"])
    spk = res["E_spk_bool"]
    rate_s = M4._smooth(np.asarray(res["rate_E"], float), M4.DT)
    runaway = M4._first_sustained(rate_s, M4.DT)
    movie = M4._spatial_movie(spk, S["posE"], S["L"], M4.DT)
    cov = M4._spatial_coverage(movie)
    n = rate_s.size
    settled = rate_s[int(0.66 * n):]                             # fast subsystem has settled by the last third
    settled_rate = float(settled.mean()) if settled.size else 0.0
    settled_cv = float(settled.std() / (settled.mean() + 1e-9)) if settled.size else 0.0
    cls = _classify_atlas(settled_rate, settled_cv, cov["active_area_tail"], runaway)
    core_act, surr_act = _core_surround_activity(spk, core_mask_E)
    row = dict(
        label=label, seed=S["seed"], warm=bool(warm), atlas_class=cls,
        q_core=float(cfg.q_init), S_G=float(cfg.clamp_SG), J_exit=float(cfg.eta_r if cfg.use_persist else 0.0),
        settled_rate_hz=round(settled_rate, 2), settled_cv=round(settled_cv, 3),
        peak_rate_hz=round(float(rate_s.max()), 1), runaway_ms=runaway,
        T_ms=float(T_ms), cfg_effective=_cfg_effective(cfg), wall_s=round(time.time() - t0, 1),
        **cov,
    )
    axis_u = np.asarray(S["axis_unit"], float)
    arrays = dict(rate=np.asarray(res["rate_E"], np.float32), movie=movie,
                  kymo_axis=_kymograph(spk, S["posE"], S["center"], axis_u),
                  core_activity=core_act, surround_activity=surr_act,
                  q_field_final=slow.q_I.astype(np.float32))
    return row, arrays


def _atlas_worker(spec):
    label, cfg, T_ms, warm = spec
    ad = _MANI.get("arm_dir")
    if ad:
        try:
            open(os.path.join(ad, f"_running_{label}"), "w").close()
        except Exception:
            pass
    S = M4._S["S"]
    try:
        return _run_atlas_cell(S, label, cfg, T_ms, warm)
    except Exception as e:
        return dict(label=label, error=repr(e)), None


def _run_atlas(a):
    os.makedirs(a.out, exist_ok=True)
    _engine_guard()
    M4._EARLY_STOP["on"] = a.early_stop
    t_build = time.time()
    S = PP.build_substrate(a.seed)
    S["p"].T = a.T
    M4._S["S"] = S
    with open(os.path.join(a.out, f"pids_{a.tag}_seed{a.seed}.txt"), "w") as f:
        f.write(f"{os.getpid()}\n")
    cells = _build_atlas_cells(a)
    prov = _provenance()
    print(f"[frozen-atlas] N={S['N']} seed={a.seed} n_cells={len(cells)} "
          f"q_core={a.q_core_grid} S_G={a.sg_grid} J_exit={a.j_exit_grid} T={a.T} workers={a.workers} "
          f"resume={a.resume} base_sha={prov['base_sha']} build={time.time()-t_build:.0f}s", flush=True)
    t_run = time.time()
    meta = dict(mode="frozen_atlas", seed=a.seed, T=a.T, q_core_grid=a.q_core_grid, sg_grid=a.sg_grid,
                j_exit_grid=a.j_exit_grid, kick=dict(boost=ATLAS_KICK_BOOST, r_kick=ATLAS_KICK_R, t_kick=ATLAS_KICK_T),
                base_kq=BASE_KQ, base_ag=BASE_AG, N=int(S["N"]), axis_unit=S["axis_unit"].tolist(),
                argv=" ".join(sys.argv), provenance=prov)
    rows = _orchestrate_arms(cells, a.out, a.tag, a.seed, prov, meta, a.workers, _atlas_worker, resume=a.resume)
    meta["wall_s"] = round(time.time() - t_run, 1)
    _assemble_combined(a.out, a.tag, a.seed, S, meta, rows)
    print("\n===== frozen exit atlas (q_core x S_G x J_exit; cold/warm IC) =====", flush=True)
    for r in sorted([x for x in rows if "error" not in x], key=lambda r: (r["q_core"], r["S_G"], r["J_exit"], r["warm"])):
        print(f"  q={r['q_core']:.2f} SG={r['S_G']:.2f} J={r['J_exit']:5.1f} {'warm' if r['warm'] else 'cold'} "
              f"-> {r['atlas_class']:20s} rate={r['settled_rate_hz']:6.1f}Hz cv={r['settled_cv']:.2f} "
              f"area_tail={r['active_area_tail']:.2f} runaway={r['runaway_ms']}", flush=True)
    for r in [x for x in rows if "error" in x]:
        print(f"  {r['label']:26s} ERROR {r['error']}", flush=True)
    print(f"wrote arms_{a.tag}_seed{a.seed}.json + .npz (+ per_arm/ + run_manifest) to {a.out}", flush=True)


def _assemble_combined(out, tag, seed, S, meta, rows):
    """Backward-compat combined outputs the plotters read: arms_<tag>_seed<seed>.{json,npz}. Rebuilt from
    the per-arm pieces so it reflects exactly what completed (resume included). npz first (survives a json
    failure)."""
    tagf = f"{tag}_seed{seed}"
    arm_dir = _arm_dir(out, tag, seed)
    payload = dict(posE=S["posE"].astype(np.float32), src_xy=S["src_xy"], snk_xy=S["snk_xy"], L=float(S["L"]))
    for r in rows:
        npzp = os.path.join(arm_dir, f"{r['label']}.npz")
        if os.path.exists(npzp):
            with np.load(npzp) as z:
                for k in z.files:
                    payload[f"{r['label']}__{k}"] = z[k]
    np.savez_compressed(os.path.join(out, f"arms_{tagf}.npz"), **payload)
    json.dump(_sanitize(dict(meta=meta, rows=rows)),
              open(os.path.join(out, f"arms_{tagf}.json"), "w"), indent=2, allow_nan=False)


def _run_arms(a):
    os.makedirs(a.out, exist_ok=True)
    _engine_guard()                         # spec §12: loud fail on unreviewed GUARDED-engine drift (slow_field unguarded)
    M4._EARLY_STOP["on"] = a.early_stop     # runaway arms truncate (still set runaway_ms); clean-exit/bounded run full
    t_build = time.time()
    S = PP.build_substrate(a.seed)
    S["p"].T = a.T
    M4._S["S"] = S
    with open(os.path.join(a.out, f"pids_{a.tag}_seed{a.seed}.txt"), "w") as f:
        f.write(f"{os.getpid()}\n")
    specs = _build_arms(a)
    prov = _provenance()
    print(f"[arms] N={S['N']} seed={a.seed} tau_p={a.tau_p} tau_p_down={a.tau_p_down} onset={a.persist_onset_ms} "
          f"theta_p={a.theta_p} a50_p={a.a50_p} eta_r={a.eta_r} sigma_p={a.sigma_p} arms={[s[0] for s in specs]} "
          f"T={a.T} workers={a.workers} resume={a.resume} base_sha={prov['base_sha']} "
          f"build={time.time()-t_build:.0f}s", flush=True)
    t_run = time.time()
    meta = dict(seed=a.seed, tau_p=a.tau_p, tau_p_down=a.tau_p_down, persist_onset_ms=a.persist_onset_ms,
                theta_p=a.theta_p, a50_p=a.a50_p, eta_r=a.eta_r, sigma_p=a.sigma_p, p50_r=a.p50_r, n_r=a.n_r,
                clamp_val=a.clamp_val, T=a.T, base_kq=BASE_KQ, base_ag=BASE_AG, N=int(S["N"]),
                axis_unit=S["axis_unit"].tolist(), argv=" ".join(sys.argv), provenance=prov)
    rows = _orchestrate_arms(specs, a.out, a.tag, a.seed, prov, meta, a.workers, _arm_worker, resume=a.resume)
    meta["wall_s"] = round(time.time() - t_run, 1)
    _assemble_combined(a.out, a.tag, a.seed, S, meta, rows)
    print("\n===== Stage-2 dynamic arms =====", flush=True)
    for r in rows:
        if "error" in r:
            print(f"  {r['label']:24s} ERROR {r['error']}", flush=True)
            continue
        print(f"  {r['label']:24s} verdict={r['verdict']:18s} cls={r['termination_class']:15s} "
              f"n_ev={r['n_events']:2d} maxHz={r['max_rate_hz']:6.1f} qmin={r['q_min_final']:.2f} "
              f"SGmax={r['S_G_max']:.2f} p_peak={r['p_peak']:.2f} area_tail={r.get('active_area_tail')}", flush=True)
    print(f"wrote arms_{a.tag}_seed{a.seed}.json + .npz (+ per_arm/ + run_manifest) to {a.out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--mode", default="exit_atlas", choices=["exit_atlas", "arms", "frozen_atlas"],
                    help="exit_atlas = Stage-1a inhibitory-pulse hold sweep; arms = Stage-2 dynamic persistence "
                         "arms; frozen_atlas = Phase-2 frozen dual-IC exit atlas (q_core x S_G x J_exit)")
    # ---- Phase-2 frozen atlas grid (task brief §5) ----
    ap.add_argument("--q-core-grid", dest="q_core_grid", default="0.05,0.4,0.9",
                    help="frozen_atlas: comma q_core levels (low/middle/recovered = uniform frozen q_I)")
    ap.add_argument("--sg-grid", dest="sg_grid", default="0.0,0.2,0.4",
                    help="frozen_atlas: comma S_G levels (0/intermediate/high divisive containment)")
    ap.add_argument("--j-exit-grid", dest="j_exit_grid", default="0.0,8.0,20.0",
                    help="frozen_atlas: comma J_exit levels (mV outward recovery current = eta_r*Phi(1))")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--t0", type=float, default=3000.0, help="branch time (bounded state settled)")
    ap.add_argument("--dvth", type=float, default=15.0, help="inhibitory_pulse V_th raise (mV)")
    ap.add_argument("--holds", default="500,3000,6000", help="comma-sep hold durations (ms)")
    ap.add_argument("--post-obs", type=float, default=3000.0, help="post-release observation (ms)")
    ap.add_argument("--baseline", default=True, action=argparse.BooleanOptionalAction,
                    help="exit_atlas: include an unperturbed bounded-reference cell (--no-baseline to skip)")
    # ---- Stage-2 arms mode: persistence-field calibration (from Stage-1) ----
    ap.add_argument("--T", type=float, default=15000.0, help="arms mode: spontaneous window (ms)")
    ap.add_argument("--arms", default="A_slow_off,B_m4_anchor,C_sensor_on,D_full",
                    help="arms mode: comma list of A_slow_off,B_m4_anchor,C_sensor_on,D_full,E1_no_qI,E2_no_SG,E4_clamp_p")
    ap.add_argument("--d-sweep", dest="d_sweep", default=None,
                    help="arms mode: D-only 'tau_p:eta_r' grid (comma-sep), one build shared, e.g. '5000:30,8000:50'")
    ap.add_argument("--include-anchor", dest="include_anchor", default=False, action=argparse.BooleanOptionalAction,
                    help="d-sweep: also run B_m4_anchor as the un-terminated reference")
    ap.add_argument("--tau-p", dest="tau_p", type=float, default=5000.0)
    ap.add_argument("--tau-p-down", dest="tau_p_down", type=float, default=None,
                    help="asymmetric p decay time (ms); None -> symmetric. Fast charge (tau_p) + slow decay = long hold")
    ap.add_argument("--persist-onset-ms", dest="persist_onset_ms", type=float, default=0.0,
                    help="established-state fork: p inactive until this t (ms) so the M4 state forms first, then engages")
    ap.add_argument("--theta-p", dest="theta_p", type=float, default=0.0)
    ap.add_argument("--a50-p", dest="a50_p", type=float, default=1.0)
    ap.add_argument("--sigma-p", dest="sigma_p", type=float, default=1.5)
    ap.add_argument("--eta-r", dest="eta_r", type=float, default=15.0)
    ap.add_argument("--p50-r", dest="p50_r", type=float, default=0.0, help="Phi(p) Hill half-point; 0 -> linear")
    ap.add_argument("--n-r", dest="n_r", type=float, default=2.0, help="Phi(p) Hill exponent (if p50_r>0)")
    ap.add_argument("--clamp-val", dest="clamp_val", type=float, default=0.8, help="E4: frozen p value")
    ap.add_argument("--early-stop", dest="early_stop", default=True, action=argparse.BooleanOptionalAction,
                    help="exit_atlas: early-stop genuine runaway for speed (won't cut the bounded state or an exit)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="arms mode: skip arms whose per-arm JSON already completed (crash-safe re-run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="s1")
    a = ap.parse_args()
    if not a.confirm_run:
        print("REFUSED: exit sim gate. Re-run with --confirm-run.")
        return
    if a.out is None:
        sub = {"arms": "stage2_arms", "frozen_atlas": "phase2_exit_atlas"}.get(a.mode, "stage1_exit_atlas")
        a.out = os.path.join(PP.ROOT, "results", "topic4_sef_hfo", "m4_snn_native_exit", sub)
    if a.mode == "arms":
        _run_arms(a)
        return
    if a.mode == "frozen_atlas":
        _run_atlas(a)
        return
    os.makedirs(a.out, exist_ok=True)
    holds = [float(x) for x in a.holds.split(",")]

    # early-stop truncates only genuine runaway (>=120Hz sustained 100ms); the bounded state maxes ~64-97Hz
    # and an exit stays low -> neither is cut. It still sets runaway_ms so a rebound_runaway is DETECTED,
    # just not simulated in full -> big speed win on rebound cells. Default on; --no-early-stop for full traces.
    M4._EARLY_STOP["on"] = a.early_stop
    t_build = time.time()
    S = PP.build_substrate(a.seed)
    M4._S["S"] = S
    with open(os.path.join(a.out, f"pids_{a.tag}_seed{a.seed}.txt"), "w") as f:
        f.write(f"{os.getpid()}\n")       # manifest for the resource monitor (parent; workers COW-fork)

    cells = []
    if a.baseline:                        # optional: unperturbed bounded reference (== each hold's pre_af)
        cells.append(("baseline", None, a.t0, a.t0 + a.post_obs, a.t0 + a.post_obs))
    for h in holds:
        t1 = a.t0 + h
        T = t1 + a.post_obs
        cells.append((f"hold{int(h)}", dict(kind="inhibitory_pulse", t0=a.t0, t1=t1, val=a.dvth), a.t0, t1, T))

    print(f"[exit-atlas] substrate E1146 {PP.MONTAGE} N={S['N']} seed={a.seed} t0={a.t0} dvth={a.dvth} "
          f"holds={holds} n_cells={len(cells)} workers={a.workers} build={time.time()-t_build:.0f}s", flush=True)
    t_run = time.time()
    with mp.Pool(min(a.workers, len(cells))) as pool:
        results = pool.map(_cell_worker, cells)

    rows = [r for r, _ in results]
    out_tag = f"{a.tag}_seed{a.seed}"
    np.savez_compressed(os.path.join(a.out, f"exit_atlas_{out_tag}.npz"),   # npz FIRST (survives a json failure)
                        posE=S["posE"].astype(np.float32), src_xy=S["src_xy"], snk_xy=S["snk_xy"],
                        L=float(S["L"]),
                        **{f"{r['label']}__{k}": arr for (r, arrs) in results if arrs
                           for k, arr in arrs.items()})
    json.dump(_sanitize(dict(meta=dict(seed=a.seed, t0=a.t0, dvth=a.dvth, holds=holds, post_obs=a.post_obs,
                                       base_kq=BASE_KQ, base_ag=BASE_AG, subject=PP.SUBJECT, montage=PP.MONTAGE,
                                       N=int(S["N"]), axis_unit=S["axis_unit"].tolist(),
                                       wall_s=round(time.time() - t_run, 1), argv=" ".join(sys.argv)),
                             rows=rows)),
              open(os.path.join(a.out, f"exit_atlas_{out_tag}.json"), "w"), indent=2, allow_nan=False)

    print("\n===== Stage-1a exit-boundary atlas =====", flush=True)
    for r in rows:
        if "error" in r:
            print(f"  {r['label']:12s} ERROR {r['error']}", flush=True)
            continue
        print(f"  {r['label']:12s} hold={r['hold_ms']:6.0f}ms verdict={r['verdict']:16s} "
              f"cls={r['termination_class']:15s} pre_af={r['pre_af']} post_af={r['post_af']} "
              f"qI:{r['qI_t0']:.2f}->{r['qI_t1']:.2f} qI_fin={r['qI_final']:.2f} runaway={r['runaway_ms']}", flush=True)
    print(f"wrote exit_atlas_{out_tag}.json + .npz to {a.out}", flush=True)


if __name__ == "__main__":
    main()
