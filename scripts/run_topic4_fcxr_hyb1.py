"""FCXR-HYB1 runner — Z / activity-excess-K / X lifecycle sprint.

Plan of record: docs/superpowers/plans/2026-07-29-topic4-fcxr-hyb1.md
Nothing runs on import; every simulation requires --confirm-run.  Fixed dt=0.05, RC1 substrate,
NO kick anywhere (KICK_BOOST=0, t_kick=1e9) -- that is gate 1 and it is enforced by construction.

Scaffolding (flock, run ids, blessed-engine check, resource log, meminfo) is reused from
run_topic4_mz_fcxr; the run/reduce/sentinel pattern is reused from run_topic4_mz_fcxr_lifecycle.
The only new engine-side pieces are the non-blessed asymmetric Z and the excess-K adapter.

Stages:
  topology   H0   component sign/order audit from existing artifacts + a synthetic stream (no 40k)
  zaxis      H0b  ONE 3 s probe -> the p-weighted GABA survival curve -> three I_th_EI levels
  baseline   H1   dK off/on x seed{1,3}, 8 s, no kick -> baseline-preservation gate
  screen     H2   12 cells: 3 hazards x dK{off,on} x X{off,on}, 14 s
  lifecycle  H3   <=2 survivors at 24 s, all seven gates
  mshape     H4   candidate x M{off,on} (only after a candidate exists)
  manifest        STATUS.md + run_manifest.json
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-hyb1")

import argparse
import gc
import json
import resource
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP                          # noqa: E402
import run_topic4_mz_slowvars as OLD                    # noqa: E402
import run_topic4_mz_fcxr as FCXR                       # noqa: E402
import run_topic4_mz_fcxr_lifecycle as LC               # noqa: E402
from kick_probe import simulate_kick                    # noqa: E402
from mz_slow_vars import MZSlowVars, MZSlowVarsConfig   # noqa: E402
import src.snn_engine.activity_excess_k as AK           # noqa: E402
import src.topic4_fcxr_hyb1 as H                        # noqa: E402
from src.topic4_mz_fcxr_dynamics import (               # noqa: E402
    rolling_rate_upper, load_onset_depletion_pi, assert_field_substrate_aligned,
)
from src.topic4_mz_fcxr_lifecycle import (              # noqa: E402
    build_windows, classify_lifecycle, depletion_coordinate,
)

DT = LC.DT_LC                       # 0.05 ms
N_GRID, DX_MM, L_MM = 32, 0.625, 20.0
DT_ION_MS = 0.5
OUT = os.path.join(FCXR.OUT_ROOT, "hyb1_lifecycle")
LC1 = os.path.join(FCXR.OUT_ROOT, "lifecycle_closure")
LC1_XWT = "/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-lc1/results/topic4_sef_hfo/" \
          "mz_full_conductance_spatial_relay/lifecycle_closure"      # read-only upstream artifacts
ION_XWT = "/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-ion/results/topic4_sef_hfo/" \
          "mz_full_conductance_spatial_relay/ion_homeostasis"        # read-only B2.1 artifacts
SEED_DEV, SEEDS_BASE = 1, (1, 3)
T_PROBE_MS, T_BASE_MS, T_SCREEN_MS, T_LIFE_MS = 3000.0, 8000.0, 14000.0, 24000.0
X_CFG = dict(x_min=0.1, tau_x_down=1000.0, tau_x_up=10000.0)          # LC1's terminating config
SENSOR = dict(tau_y=LC.SENSOR_TAU_Y, K_y=LC.SENSOR_K_Y, hill_n=LC.SENSOR_HILL_N)
D_X_SNAP_MS = 100.0
M_CFG = dict(tau_adp=250.0, m_force_frac=0.10)          # H4 only, fixed, never swept


# ------------------------------------------------------------------ io / sentinels
def _jw(path, payload):
    LC._atomic_json(path, payload)


def _sent(run_dir, name, **kw):
    LC._sentinel(run_dir, name, **kw)


def _stage_sentinels(stage):
    return (os.path.join(OUT, f"RUNNING_{stage}.json"), os.path.join(OUT, f"DONE_{stage}.json"),
            os.path.join(OUT, f"FAILED_{stage}.json"))


def _begin(stage, **kw):
    os.makedirs(OUT, exist_ok=True)
    run, done, fail = _stage_sentinels(stage)
    for p in (done, fail):
        if os.path.exists(p):
            os.remove(p)
    _jw(run, dict(stage=stage, pid=os.getpid(), t=datetime.now(timezone.utc).isoformat(), **kw))
    with open(os.path.join(OUT, f"launcher_{stage}.pid"), "w") as f:
        f.write(str(os.getpid()))
    return run


def _end(stage, ok, **kw):
    run, done, fail = _stage_sentinels(stage)
    if os.path.exists(run):
        os.remove(run)
    _jw(done if ok else fail, dict(stage=stage, t=datetime.now(timezone.utc).isoformat(), **kw))


import contextlib     # noqa: E402
import fcntl           # noqa: E402


@contextlib.contextmanager
def _stage_lock(name):
    """Per-STAGE singleton, not a global build lock.

    The FCXR launcher lock is one exclusive flock for the whole results root, so it also refuses
    two DIFFERENT HYB1 cells.  The contract here is 'do not submit the same stage twice', which is
    what a stage-scoped lock enforces -- and it is what lets the two baseline seeds, or two screen
    cells, run as the two workers the resource rules allow for T < 20 s.
    """
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f".hyb1_{name}.lock")
    with open(path, "a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"stage {name} is already running; refusing duplicate") from exc
        fh.seek(0); fh.truncate()
        fh.write(f"pid={os.getpid()} t={datetime.now(timezone.utc).isoformat()}\n")
        fh.flush()
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _guard(tag, swap_base_mb):
    state, info = LC._resource_state(swap_base_mb)
    FCXR._resource_log(OUT, tag, info)
    return state, info


# ------------------------------------------------------------------ configs
def _cfg_base():
    return LC._slowoff_cfg()


def _cfg_run(*, I_th_EI=None, use_x=False, use_m=False, y_gate=None):
    c = _cfg_base()
    if I_th_EI is not None:
        c.update(use_z=True, tau_z=H.TAU_Z_DOWN_MS, I_th_EI=float(I_th_EI),
                 tau_z_down=H.TAU_Z_DOWN_MS, tau_z_up=H.TAU_Z_UP_MS)
    if use_x:
        c.update(use_x=True, y_gate=float(y_gate), **X_CFG, **SENSOR)
    if use_m:
        c.update(use_m=True, tau_adp_E=M_CFG["tau_adp"])
    return c


def _voxel_map(S):
    pos = np.vstack([np.asarray(S["posE"], float), np.asarray(S["posI"], float)])
    return AK.cell_to_voxel(pos, L_MM, N_GRID)


def _make_dk(S, b_v, *, enabled, record_load=False, snapshots=(), eps=None):
    b = np.asarray(b_v, float)
    if eps is None:
        finite = b[np.isfinite(b) & (b > 0)]
        eps = float(H.EPS_FRAC * np.median(finite)) if finite.size else 1.0
    cfg = AK.ActivityExcessKConfig(
        b_v=b, eps=float(eps), q_K=H.Q_K, n_grid=N_GRID, dx_mm=DX_MM, dt_ion_ms=DT_ION_MS,
        g_dK=H.G_DELTA_K, enabled=bool(enabled), record_load=bool(record_load),
        snapshot_blocks=tuple(snapshots))
    return AK.ActivityExcessK(S["N"], _voxel_map(S), cfg)


class _SensorSampler(AK.ExcessKMZAdapter):
    """zaxis probe only: pool the per-cell GABA sensor at a fixed cadence.

    The hazard identity h_Z = a_p(I_th)/tau_z is exact at the state the run STARTS from, so the
    survival curve must describe the pre-onset baseline -- a single final-step snapshot would be
    one noisy instant of it.  Pooling over the probe gives the same quantity with far less
    sampling noise and costs nothing (a few 32k vectors).
    """

    def __init__(self, mz, dk, *, every_steps):
        super().__init__(mz, dk)
        object.__setattr__(self, "_every", int(every_steps))
        object.__setattr__(self, "_i", 0)
        object.__setattr__(self, "sensor_samples", [])

    def step(self, spk, labels, dt):
        super().step(spk, labels, dt)
        object.__setattr__(self, "_i", self._i + 1)
        if self._i % self._every == 0:
            self.sensor_samples.append(np.asarray(self.mz._z_sensor_last_E, np.float32).copy())


def _run(S, cfg_dict, T_ms, *, seed, dk=None, snap_ms=D_X_SNAP_MS, sensor_every_ms=None):
    """One continuous slow-driven run.  NO kick, ever: KICK_BOOST=0 and t_kick=1e9 (gate 1)."""
    import dataclasses
    p = dataclasses.replace(S["p"], T=float(T_ms), dt=DT)
    # snapshot_steps is a {step_index: label} map, not a stride -- build the ladder explicitly so
    # D_Z / D_X are sampled on a fixed grid rather than at whatever the engine defaults to.
    every = max(1, int(round(snap_ms / DT)))
    snaps = {k: f"t{int(round(k * DT))}ms" for k in range(every, int(T_ms / DT) + 1, every)}
    mz = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg_dict), NE=S["NE"],
                    core_mask_E=OLD.build_core_masks(S), snapshot_steps=snaps)
    slow = (AK.ExcessKMZAdapter(mz, dk) if sensor_every_ms is None
            else _SensorSampler(mz, dk, every_steps=max(1, int(round(sensor_every_ms / DT)))))
    S["net"]["rng"] = np.random.default_rng(int(seed))
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"],
                        early_stop_runaway=False)
    return res, mz, dk, slow


# ------------------------------------------------------------------ H0 topology audit
def cmd_topology(args):
    """Component-grounded sign / order audit.  No toy ODE detached from the SNN: every sign is
    either read out of an existing accepted artifact or exercised on the real slow-variable code
    with a small deterministic synthetic stream."""
    _begin("topology")
    ev, missing = {}, []

    def _load(p):
        if not os.path.exists(p):
            missing.append(p)
            return None
        return json.load(open(p))

    z75 = _load(os.path.join(LC1_XWT, "z_only_summary_seed1_q75.json"))
    z50 = _load(os.path.join(LC1_XWT, "z_only_summary_seed1_q50.json"))
    sens = _load(os.path.join(LC1_XWT, "sensor_separation_seed1_ty120.json"))
    xctl = _load(os.path.join(LC1_XWT, "xcontrol_seed1.json"))
    mc = _load(os.path.join(ION_XWT, "b2_1_matched_control.json"))

    if z75 and z50:
        ev["high_activity_depletes_Z"] = dict(
            ok=bool(z75["D_Z_end"] > 0 and z50["D_Z_end"] > z75["D_Z_end"]),
            q75_D_Z_end=z75["D_Z_end"], q50_D_Z_end=z50["D_Z_end"],
            claim="sustained activity drives D_Z up, and the stronger regime drives it further")
    if sens:
        ev["sustained_activity_depletes_X"] = dict(
            ok=bool(sens.get("separated")), base_occ=sens.get("base_occ"),
            dense_occ=sens.get("dense_occ"),
            claim="the persistence sensor separates interictal from dense by >100x occupancy")
    if xctl:
        rows = sorted(xctl["rows"], key=lambda r: -float(r["x_min"]))
        occ = [float(r["roll_occ"]) for r in rows]
        ev["X_depletion_suppresses_activity"] = dict(
            ok=bool(len(occ) >= 2 and occ[0] > min(occ[1:])),
            x_min=[float(r["x_min"]) for r in rows], roll_occ=occ,
            neutral_roll_occ=xctl["neutral_roll_occ"],
            claim=("lowering the relay floor lowers dense occupancy relative to the NEUTRAL "
                   "x_min=1 arm, i.e. X has authority over the high state"))
    if mc:
        w2 = (mc.get("spatial_extent") or {}).get("closed", [{}, {}])[1]
        w2o = (mc.get("spatial_extent") or {}).get("open", [{}, {}])[1]
        ev["excess_activity_recruits_via_K"] = dict(
            ok=bool(w2 and w2o and w2.get("recruit_radius_mm", 0) > w2o.get("recruit_radius_mm", 1e9)),
            closed_radius_mm=w2.get("recruit_radius_mm"), open_radius_mm=w2o.get("recruit_radius_mm"),
            claim="live K feedback widened the later window's recruitment (B2.1 matched control)")

    # the ONE new component: asymmetric Z, exercised on the real code with a deterministic stream
    NE_, NI_ = 40, 10
    mz = MZSlowVars(NE_ + NI_, 18.0, MZSlowVarsConfig(use_z=True, tau_z=H.TAU_Z_DOWN_MS,
                                                      I_th_EI=5.0, tau_z_down=500.0,
                                                      tau_z_up=20000.0), NE=NE_)
    seq = []
    for sensor, n in ((9.0, 4000), (0.0, 4000)):
        for _ in range(n):
            mz._z_sensor_last_E = np.full(NE_, sensor)
            mz.step(np.zeros(NE_ + NI_, bool), None, DT)
        seq.append(float(mz.z[mz.is_E].mean()))
    ev["low_activity_recovers_Z"] = dict(
        ok=bool(seq[0] < 1.0 and seq[1] > seq[0]),
        z_after_load=seq[0], z_after_release=seq[1],
        claim="Z depletes under load then recovers when the load falls -- no instantaneous reset")
    ev["dK_clears_when_activity_stops"] = dict(
        ok=bool(AK.d_dK_dt(np.full((4, 4), 1.0), np.zeros((4, 4)),
                           AK.ActivityExcessKConfig(b_v=np.full(16, 1.0), eps=0.1, q_K=H.Q_K,
                                                    n_grid=4))[0, 0] < 0),
        claim="with the load below background the excess field relaxes toward zero")

    ok = bool(ev) and all(v["ok"] for v in ev.values())
    status = ("TOPOLOGY_OK" if ok else
              ("TOPOLOGY_INPUT_UNRESOLVED" if missing else "TOPOLOGY_SIGN_FAILED"))
    payload = dict(status=status, generated=datetime.now(timezone.utc).isoformat(),
                   code_commit=FCXR._git_sha(), evidence=ev, missing_inputs=missing,
                   note=("TOPOLOGY_INPUT_UNRESOLVED means an input was absent, NOT that the "
                         "mechanism was refuted"))
    _jw(os.path.join(OUT, "component_topology.json"), payload)
    _end("topology", ok, status=status)
    print(f"[topology] {status}: " + " ".join(f"{k}={'ok' if v['ok'] else 'FAIL'}"
                                              for k, v in ev.items()), flush=True)
    return 0 if ok else 2


# ------------------------------------------------------------------ H0b Z axis + background
def cmd_zaxis(args):
    """ONE 3 s slow-off probe gives BOTH the p-weighted GABA survival curve (the Z axis) and the
    per-voxel background envelope b_v (the excess-K deadband).  One run, two locks."""
    with _stage_lock("zaxis"):
        FCXR._assert_engine_blessed()
        _begin("zaxis", T=T_PROBE_MS)
        run_dir = os.path.join(OUT, "runs", FCXR._run_id("zaxis_probe"))
        swap_base = LC._launch_baseline(run_dir, "zaxis")
        try:
            S = PP.build_substrate(SEED_DEV)
            pk = load_onset_depletion_pi(LC.SNAP_ZA_FMT.format(seed=SEED_DEV))
            assert_field_substrate_aligned(pk, S)
            t0 = time.time()
            dk = _make_dk(S, np.full(N_GRID * N_GRID, np.inf), enabled=False, record_load=True)
            cfg = _cfg_base()
            cfg["record_calib"] = True
            res, mz, dk, slow = _run(S, cfg, T_PROBE_MS, seed=SEED_DEV, dk=dk,
                                     sensor_every_ms=100.0)
            wall = round(time.time() - t0, 1)

            samples = np.stack(slow.sensor_samples).astype(float)       # (n_sample, NE)
            p_i = np.asarray(pk["p_i"], float)[:S["NE"]]
            sensor = samples.ravel()
            p_w = np.tile(p_i, samples.shape[0])
            verdict = H.adjudicate_z_axis(sensor, p_w)
            verdict["survival_probe"] = dict(n_samples=int(samples.shape[0]),
                                             sample_every_ms=100.0, n_cells=int(samples.shape[1]),
                                             pooled_over="time x cell, p-weighted")

            load = np.stack(dk.load_trace)                       # (n_block, n_voxel)
            b_v = AK.background_envelope(load, H.Q_BG)
            eps = float(H.EPS_FRAC * np.median(b_v[b_v > 0])) if np.any(b_v > 0) else 1e-3
            np.savez_compressed(os.path.join(OUT, "background_envelope.npz"),
                                b_v=b_v.astype(np.float64), eps=np.float64(eps),
                                q_bg=np.float64(H.Q_BG), n_blocks=np.int64(load.shape[0]),
                                load_mean=load.mean(axis=0).astype(np.float32))
            del res, load
            gc.collect()

            verdict.update(generated=datetime.now(timezone.utc).isoformat(),
                           code_commit=FCXR._git_sha(), T_ms=T_PROBE_MS, seed=SEED_DEV,
                           wall_s=wall, peak_rss_gb=round(_rss_gb(), 2),
                           background=dict(path="background_envelope.npz", Q_BG=H.Q_BG, eps=eps,
                                           b_v_median=float(np.median(b_v)),
                                           b_v_max=float(b_v.max()),
                                           n_occupied=int(np.count_nonzero(b_v > 0))))
            th = np.linspace(float(np.quantile(sensor, 0.50)), float(sensor.max()), 400)
            a_curve = [H.hazard_from_survival(sensor, p_w, x, 1000.0)[1] for x in th]
            np.savez_compressed(os.path.join(OUT, "z_survival_curve.npz"),
                                theta=th, a_p=np.asarray(a_curve, float),
                                sensor_q=np.quantile(sensor, np.linspace(0, 1, 1001)))
            verdict["survival_curve"] = "z_survival_curve.npz"
            _jw(os.path.join(OUT, "z_axis_calibration.json"), verdict)
            ok = verdict["status"] == "Z_AXIS_LOCKED"
            if not ok:
                _jw(os.path.join(OUT, "DESIGN_BLOCKED_Z_AXIS.json"), verdict)
            _end("zaxis", ok, status=verdict["status"], wall_s=wall)
            print(f"[zaxis] {verdict['status']}  eps={eps:.4g}  " +
                  " ".join(f"{k}={v['I_th_EI']:.4g}(h={v['h_Z_realised']:.4g})"
                           for k, v in verdict["levels"].items()) +
                  "  anchors " + " ".join(f"{k}:{v['rel_err']:+.2f}"
                                          for k, v in verdict["anchor_prediction"].items()) +
                  f"  ({wall}s, peak {round(_rss_gb(),1)}GB)", flush=True)
            return 0 if ok else 2
        except BaseException as e:
            _end("zaxis", False, error=repr(e))
            raise


# ------------------------------------------------------------------ reductions
def _reduce(res, mz, dk, S, band=None, frozen_bar=None):
    rate = np.asarray(res["rate_E"], float)
    bar = float(frozen_bar) if frozen_bar is not None else float(OLD.slowoff_event_bar(res, DT))
    events, af, af_bin, floor, _ = OLD._events_from_res(res, DT, event_bar=bar)
    af = np.asarray(af, float)
    ret = [e for e in events if e["returned"]]
    roll_hi = rolling_rate_upper(rate, DT)
    num = LC._numerical(S, res, mz, DT)
    win = build_windows(rate, DT, af, af_bin, roll_hi, ret, LC.LC_WIN_MS,
                        event_lookback_ms=LC.LC_LOOKBACK_MS, finite=num["finite"])
    out = dict(frozen_bar=bar, roll_hi=roll_hi, numerical=num, windows=win, rate=rate,
               n_returning=len(ret), af_bin_ms=float(af_bin),
               event_times_ms=[float(e["t_on"]) for e in ret],
               event_dur_ms=[float(e["dur_ms"]) for e in ret],
               event_part=[float(e["peak_ext"]) for e in ret],
               end_rate_hz=float(rate[-int(1000.0 / DT):].mean()))
    if band is None:
        recs = [w["recruit_frac"] for w in win]
        ers = [w["event_rate_hz"] for w in win]
        band = dict(win_ms=LC.LC_WIN_MS, event_lookback_ms=LC.LC_LOOKBACK_MS, roll_hi=roll_hi,
                    recruit_p90=LC._pct(recs, 90, 0.0),
                    event_rate_lo=max(0.05, 0.3 * LC._pct(ers, 10, 0.0)),
                    event_rate_hi=1.8 * LC._pct(ers, 90, 0.0) if ers else 0.0)
    out["band"] = band
    out["lifecycle"] = classify_lifecycle(win, band, runaway=bool(out["end_rate_hz"] >
                                                                 H.RUNAWAY_RATE_HZ))
    if dk is not None:
        tr = np.asarray(dk.trace, float) if dk.trace else np.zeros((1, 4))
        out["dk"] = dict(duty=dk.duty_cycle(), frac_over=dk.frac_over_amplitude(),
                         running_max=dk.dK_running_max,
                         mean_series=tr[:, 1].tolist(), max_series=tr[:, 2].tolist(),
                         extent_series=tr[:, 3].tolist(),
                         q99=float(np.quantile(tr[:, 2], 0.99)) if tr.shape[0] > 4 else 0.0)
    return out


def _iei_cv(times_ms):
    t = np.asarray(times_ms, float)
    if t.size < 3:
        return 0.0
    d = np.diff(np.sort(t))
    return float(d.std() / d.mean()) if d.mean() > 0 else 0.0


# ------------------------------------------------------------------ H1 baseline preservation
def cmd_baseline(args):
    seeds0 = (int(args.seed),) if getattr(args, "seed", None) else SEEDS_BASE
    with _stage_lock(f"baseline_s{seeds0[0]}" if len(seeds0) == 1 else "baseline"):
        FCXR._assert_engine_blessed()
        seeds = (int(args.seed),) if getattr(args, "seed", None) else SEEDS_BASE
        tag = f"baseline_s{seeds[0]}" if len(seeds) == 1 else "baseline"
        _begin(tag, T=T_BASE_MS, seeds=list(seeds))
        bg = np.load(os.path.join(OUT, "background_envelope.npz"))
        rows = []
        try:
            for seed in seeds:
                S = PP.build_substrate(seed)
                for on in (False, True):
                    st, info = _guard(f"baseline_s{seed}_dk{int(on)}", LC._swap_used_mb())
                    if st == "hard":
                        raise SystemExit(f"resource hard-stop: {info}")
                    t0 = time.time()
                    dk = _make_dk(S, bg["b_v"], enabled=on)
                    res, mz, dk, _ = _run(S, _cfg_base(), T_BASE_MS, seed=seed, dk=dk)
                    r = _reduce(res, mz, dk, S)
                    del res
                    gc.collect()
                    rows.append(dict(seed=seed, dk_on=on, wall_s=round(time.time() - t0, 1),
                                     peak_rss_gb=round(_rss_gb(), 2), **{
                                         k: r[k] for k in ("frozen_bar", "roll_hi", "n_returning",
                                                           "end_rate_hz", "band")},
                                     label=r["lifecycle"]["label"], numerical=r["numerical"],
                                     dk=r.get("dk", {}), iei_cv=_iei_cv(r["event_times_ms"]),
                                     event_rate_hz=len(r["event_times_ms"]) / (T_BASE_MS / 1000.0),
                                     duration_ms_median=float(np.median(r["event_dur_ms"]))
                                     if r["event_dur_ms"] else 0.0,
                                     participation_median=float(np.median(r["event_part"]))
                                     if r["event_part"] else 0.0))
                    print(f"[baseline] seed{seed} dK={'on' if on else 'off'}: "
                          f"label={rows[-1]['label']} n_ret={rows[-1]['n_returning']} "
                          f"iei_cv={rows[-1]['iei_cv']:.3f} duty={rows[-1]['dk'].get('duty',0):.4f} "
                          f"fracOver={rows[-1]['dk'].get('frac_over',0):.5f} "
                          f"maxq99={rows[-1]['dk'].get('q99',0):.4g} "
                          f"({rows[-1]['wall_s']}s)", flush=True)
            verdicts = {}
            for seed in seeds:
                off = next(r for r in rows if r["seed"] == seed and not r["dk_on"])
                on = next(r for r in rows if r["seed"] == seed and r["dk_on"])
                lo, hi = off["band"]["event_rate_lo"], off["band"]["event_rate_hi"]
                verdicts[f"seed{seed}"] = H.adjudicate_baseline_preservation(dict(
                    dk_duty=on["dk"].get("duty", 0.0),
                    dk_frac_over=on["dk"].get("frac_over", 0.0),
                    dk_spatial_max_q99_mM=on["dk"].get("q99", 0.0),
                    event_rate_in_band=bool(lo <= on["event_rate_hz"] <= max(hi, lo + 1e-9)),
                    iei_cv_in_band=bool(abs(on["iei_cv"] - off["iei_cv"]) <= 0.5 * max(off["iei_cv"], 1e-9)),
                    iei_cv=on["iei_cv"],
                    duration_in_band=bool(abs(on["duration_ms_median"] - off["duration_ms_median"])
                                          <= 0.5 * max(off["duration_ms_median"], 1e-9)),
                    participation_in_band=bool(abs(on["participation_median"] - off["participation_median"])
                                               <= 0.5 * max(off["participation_median"], 1e-9)),
                    clip_frac_max=on["numerical"]["clip_frac_max"],
                    numerical_unsafe=on["numerical"]["numerical_unsafe"]))
            ok = all(v["status"] == "BASELINE_PRESERVED" for v in verdicts.values())
            payload = dict(status="BASELINE_PRESERVED" if ok else "STOP_BASELINE_DISTURBED",
                           generated=datetime.now(timezone.utc).isoformat(),
                           code_commit=FCXR._git_sha(), T_ms=T_BASE_MS, rows=rows,
                           per_seed=verdicts, seeds=list(seeds))
            _jw(os.path.join(OUT, f"baseline_preservation_seed{seeds[0]}.json"
                             if len(seeds) == 1 else "baseline_preservation.json"), payload)
            if not ok:
                _jw(os.path.join(OUT, "STOP_BASELINE_DISTURBED.json"), payload)
            _end(tag, ok, status=payload["status"])
            print(f"[baseline] {payload['status']}", flush=True)
            return 0 if ok else 2
        except BaseException as e:
            _end(tag, False, error=repr(e))
            raise


# ------------------------------------------------------------------ H2 development screen
def _cells():
    """12 cells: 3 Z hazards x dK{off,on} x X{off,on}.  Order puts the two most informative
    dimensions first so a partial run still spans the grid."""
    lv = json.load(open(os.path.join(OUT, "z_axis_calibration.json")))["levels"]
    out = []
    for hz in ("H_LO", "H_MID", "H_HI"):
        for dk_on in (False, True):
            for x_on in (False, True):
                out.append(dict(cell=f"{hz}_dk{int(dk_on)}_x{int(x_on)}", hazard=hz,
                                I_th_EI=float(lv[hz]["I_th_EI"]),
                                h_Z=float(lv[hz]["h_Z_realised"]), dk_on=dk_on, x_on=x_on))
    return out


def _y_gate():
    """Persistence-sensor gate: reuse LC1's accepted seed-1 value; do NOT re-derive it here."""
    return float(json.load(open(os.path.join(LC1_XWT,
                                             "sensor_separation_seed1_ty120.json")))["y_gate"])


def _screen_metrics(r, mz, dk, T_ms):
    """One cell's reduced summary, in the shape adjudicate_lifecycle expects."""
    lc = r["lifecycle"]
    snaps = sorted(mz.snapshots.items(), key=lambda kv: kv[1]["step"])
    t_ms = np.array([kv[1]["step"] * DT for kv in snaps], float)
    pk = _screen_metrics.p_i
    D_Z = np.array([depletion_coordinate(kv[1]["z_E"], pk) for kv in snaps], float) if snaps \
        else np.zeros(0)
    x_key = "x_E" if snaps and "x_E" in snaps[0][1] else None
    D_X = np.array([depletion_coordinate(kv[1][x_key], pk) for kv in snaps], float) \
        if x_key else np.zeros(0)
    bout = lc.get("bout")
    onset_ms = float(bout[0] * LC.LC_WIN_MS) if bout else None
    x_delay = None
    if D_X.size and onset_ms is not None:
        hit = np.nonzero(D_X >= H.X_ACTIVE_D_X)[0]
        if hit.size:
            x_delay = float(t_ms[hit[0]] - onset_ms)
    return dict(
        kick_boost=0.0, t_kick_ms=1e9, onset_detected=bool(bout is not None),
        pre_interictal_ms=float(lc.get("pre_ms") or 0.0),
        bout_ms=(float(lc["bout_ms"]) if lc.get("bout_ms") else None),
        bounded=bool(r["end_rate_hz"] < H.RUNAWAY_RATE_HZ),
        clip_frac_max=r["numerical"]["clip_frac_max"], finite=r["numerical"]["finite"],
        numerical_unsafe=r["numerical"]["numerical_unsafe"], end_rate_hz=r["end_rate_hz"],
        recruit_contacts=None, onset_gradient_r2=None,
        x_activation_delay_ms=x_delay, post_return_ms=float(lc.get("post_return_ms") or 0.0),
        label=lc["label"], post_iei_cv=None, band_event_rate=None, band_duration=None,
        band_participation=None,
        D_Z_end=float(D_Z[-1]) if D_Z.size else None, D_Z_max=float(D_Z.max()) if D_Z.size else None,
        D_X_max=float(D_X.max()) if D_X.size else None,
        dk_duty=(r.get("dk") or {}).get("duty"), dk_max=(r.get("dk") or {}).get("running_max"),
        n_returning=r["n_returning"], iei_cv=_iei_cv(r["event_times_ms"]),
        regimes=lc.get("regimes"), T_ms=T_ms)


def cmd_screen(args):
    """H2: the 12-cell development screen.  Each cell writes its own DONE sentinel, so a resumed
    launcher never re-submits work that already finished."""
    with _stage_lock(f"screen_{args.shard}"):
        FCXR._assert_engine_blessed()
        _begin(f"screen_{args.shard}", T=T_SCREEN_MS, shard=args.shard)
        rd = os.path.join(OUT, "runs", "screen")
        os.makedirs(rd, exist_ok=True)
        bg = np.load(os.path.join(OUT, "background_envelope.npz"))
        cells = [c for i, c in enumerate(_cells()) if i % args.shards == args.shard]
        S = PP.build_substrate(SEED_DEV)
        pk = load_onset_depletion_pi(LC.SNAP_ZA_FMT.format(seed=SEED_DEV))
        assert_field_substrate_aligned(pk, S)
        _screen_metrics.p_i = np.asarray(pk["p_i"], float)
        yg = _y_gate()
        base = json.load(open(os.path.join(OUT, f"baseline_preservation_seed{SEED_DEV}.json")))
        off = next(r for r in base["rows"] if r["seed"] == SEED_DEV and not r["dk_on"])
        band, bar = off["band"], off["frozen_bar"]
        swap_base = LC._swap_used_mb()
        try:
            for c in cells:
                dst = os.path.join(rd, c["cell"] + ".json")
                if os.path.exists(dst):
                    print(f"[screen] {c['cell']}: already DONE, skipping", flush=True)
                    continue
                st, info = _guard(f"screen_{c['cell']}", swap_base)
                if st == "hard":
                    raise SystemExit(f"resource hard-stop before {c['cell']}: {info}")
                t0 = time.time()
                dk = _make_dk(S, bg["b_v"], enabled=c["dk_on"], eps=float(bg["eps"]))
                cfg = _cfg_run(I_th_EI=c["I_th_EI"], use_x=c["x_on"], y_gate=yg)
                res, mz, dk, _ = _run(S, cfg, T_SCREEN_MS, seed=SEED_DEV, dk=dk)
                r = _reduce(res, mz, dk, S, band=band, frozen_bar=bar)
                m = _screen_metrics(r, mz, dk, T_SCREEN_MS)
                del res
                gc.collect()
                v = H.adjudicate_lifecycle(m, spatial_leg="UNRESOLVED")
                row = dict(**c, wall_s=round(time.time() - t0, 1),
                           peak_rss_gb=round(_rss_gb(), 2), metrics=m,
                           screen_verdict=v["status"], failed_gates=v["failed"],
                           failure_layer=v["failure_layer"],
                           note=("SHORT screen: recovery is NOT claimable at T=14 s; gate 6 and "
                                 "the spatial leg are judged at H3 only"))
                LC._atomic_json(dst, row)
                print(f"[screen] {c['cell']}: label={m['label']} pre={m['pre_interictal_ms']:.0f}ms "
                      f"bout={m['bout_ms']} end={m['end_rate_hz']:.1f}Hz "
                      f"D_Z={m['D_Z_end']} D_Xmax={m['D_X_max']} "
                      f"dkmax={m['dk_max']} layers={v['failure_layer']} "
                      f"({row['wall_s']}s)", flush=True)
            _end(f"screen_{args.shard}", True, n_cells=len(cells))
            return 0
        except BaseException as e:
            _end(f"screen_{args.shard}", False, error=repr(e))
            raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="FCXR-HYB1 lifecycle sprint runner (dt=0.05, no kick)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("topology", "zaxis", "baseline", "screen"):
        p = sub.add_parser(name)
        p.add_argument("--confirm-run", action="store_true")
        if name == "baseline":
            p.add_argument("--seed", type=int, default=None,
                           help="run ONE seed (both dK arms) so the two seeds can go in parallel")
        if name == "screen":
            p.add_argument("--shard", type=int, default=0)
            p.add_argument("--shards", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd != "topology" and not args.confirm_run:
        raise SystemExit("REFUSING: simulations require --confirm-run")
    return {"topology": cmd_topology, "zaxis": cmd_zaxis, "baseline": cmd_baseline,
            "screen": cmd_screen}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
