"""FCXR-HYB2 runner — event-limited recruitment sprint.

Plan of record: docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md
Nothing runs on import; every simulation requires --confirm-run.  dt=0.05, RC1 substrate, and NO
kick in the calibration / B0 / screen stages (KICK_BOOST=0, t_kick=1e9).

Stages
  preflight     artifact existence + sha256, blessed-engine check, resource snapshot.  No sim.
  calibration   seed1 then seed3, 24 s sensor-only.  Locks b_v / GAP / tau_R / Q_on / Q_scale and
                stores the per-cell GABA sensor for the S_Z replay.  Each seed also IS its own
                Gate B0 ELR-off arm.  THE ONLY 40k RUNS AUTHORISED BEFORE calibration_lock.json.
  finalize      global tau_R from the MIN cross-seed GAP_05 + per-seed Q_on re-derivation.  No sim.
  zaxis         offline S_Z replay from the stored sensor.  No sim.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-hyb2")

import argparse
import contextlib
import dataclasses
import fcntl
import gc
import hashlib
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
import src.snn_engine.event_limited_recruitment as ELR  # noqa: E402
import src.topic4_fcxr_hyb2 as H2                       # noqa: E402
from src.topic4_fcxr_hyb1 import IEI_CV_MIN as H1_IEI_CV_MIN   # noqa: E402  reuse, do not redefine
from src.topic4_mz_fcxr_dynamics import (               # noqa: E402
    rolling_rate_upper, load_onset_depletion_pi, assert_field_substrate_aligned,
)
from src.topic4_mz_fcxr_lifecycle import build_windows, classify_lifecycle   # noqa: E402

DT = LC.DT_LC                       # 0.05 ms
N_GRID, DX_MM, L_MM = 32, 0.625, 20.0
OUT = os.path.join(FCXR.OUT_ROOT, "hyb2_event_limited_recruitment")
LC1 = ("/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-mz-fcxr-lc1/results/topic4_sef_hfo/"
       "mz_full_conductance_spatial_relay/lifecycle_closure")
ION = ("/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-ion/results/topic4_sef_hfo/"
       "mz_full_conductance_spatial_relay/ion_homeostasis")
SEEDS = (1, 3)
T_CAL_MS = 24000.0
SENSOR_EVERY_MS = 100.0             # per-cell GABA sensor cadence for the S_Z replay
SOFT_SWAP_MB, HARD_SWAP_MB = 256.0, 512.0
WALL_KILL_S = 3600.0

INPUTS = {
    "lc1_baseline_seed1": f"{LC1}/baseline_contract_seed1.json",
    "lc1_baseline_seed3": f"{LC1}/baseline_contract_seed3.json",
    "lc1_zonly_seed1_q75": f"{LC1}/z_only_summary_seed1_q75.json",
    "lc1_zonly_seed1_q50": f"{LC1}/z_only_summary_seed1_q50.json",
    "lc1_sensor_seed1": f"{LC1}/sensor_separation_seed1_ty120.json",
    "lc1_xcontrol_seed1": f"{LC1}/xcontrol_seed1.json",
    "b2_1_matched_control": f"{ION}/b2_1_matched_control.json",
    "pi_snapshot_seed1": LC.SNAP_ZA_FMT.format(seed=1),
    "pi_snapshot_seed3": LC.SNAP_ZA_FMT.format(seed=3),
}


# ------------------------------------------------------------------ scaffolding
def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _jw(path, payload):
    LC._atomic_json(path, payload)


@contextlib.contextmanager
def _stage_lock(name):
    """Per-STAGE singleton, not the FCXR global build lock (that one also refuses two DIFFERENT
    cells, which would forbid the two workers the resource rules allow below 20 s)."""
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f".hyb2_{name}.lock"), "a+") as fh:
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


def _begin(stage, **kw):
    os.makedirs(OUT, exist_ok=True)
    for suf in ("DONE", "FAILED"):
        p = os.path.join(OUT, f"{suf}_{stage}.json")
        if os.path.exists(p):
            os.remove(p)
    _jw(os.path.join(OUT, f"RUNNING_{stage}.json"),
        dict(stage=stage, pid=os.getpid(), t=datetime.now(timezone.utc).isoformat(), **kw))
    with open(os.path.join(OUT, f"launcher_{stage}.pid"), "w") as f:
        f.write(str(os.getpid()))


def _end(stage, ok, **kw):
    p = os.path.join(OUT, f"RUNNING_{stage}.json")
    if os.path.exists(p):
        os.remove(p)
    _jw(os.path.join(OUT, f"{'DONE' if ok else 'FAILED'}_{stage}.json"),
        dict(stage=stage, t=datetime.now(timezone.utc).isoformat(), **kw))


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


def _guard(tag, swap_base_mb):
    state, info = LC._resource_state(swap_base_mb)
    FCXR._resource_log(OUT, tag, info)
    return state, info


# ------------------------------------------------------------------ preflight
def cmd_preflight(args):
    _begin("preflight")
    rows, ok = {}, True
    for k, p in INPUTS.items():
        e = os.path.exists(p)
        ok &= e
        rows[k] = dict(path=p, exists=e, sha256=_sha(p) if e else None)
    blessed = {f"src/snn_engine/{n}.py": _sha(os.path.join(ROOT, "src", "snn_engine", f"{n}.py"))
               for n in ("kick_probe", "lfp", "params", "model", "connectivity",
                         "connectivity_rot")}
    avail, swap = FCXR._meminfo()
    payload = dict(status="PREFLIGHT_PASS" if ok else "PREFLIGHT_MISSING_ARTIFACT",
                   generated=datetime.now(timezone.utc).isoformat(),
                   code_commit=FCXR._git_sha(), inputs=rows, blessed_engine_sha256=blessed,
                   resources=dict(mem_available_gb=round(avail, 2),
                                  swap_used_baseline_mb=round(swap * 1024.0, 1),
                                  nproc=os.cpu_count()))
    _jw(os.path.join(OUT, "preflight.json"), payload)
    _end("preflight", ok, status=payload["status"])
    print(f"[preflight] {payload['status']}  MemAvailable={avail:.0f}GB "
          f"swap_baseline={swap*1024:.1f}MB", flush=True)
    return 0 if ok else 2


# ------------------------------------------------------------------ calibration
class _SensorSampler(ELR.ELRMZAdapter):
    """Pool the per-cell GABA sensor at a fixed cadence for the S_Z replay (plan 5.1)."""

    def __init__(self, mz, elr, *, every_steps):
        super().__init__(mz, elr)
        object.__setattr__(self, "_every", int(every_steps))
        object.__setattr__(self, "_i", 0)
        object.__setattr__(self, "sensor", [])

    def step(self, spk, labels, dt):
        super().step(spk, labels, dt)
        object.__setattr__(self, "_i", self._i + 1)
        if self._i % self._every == 0:
            self.sensor.append(np.asarray(self.mz._z_sensor_last_E, np.float32).copy())


def _voxel_map(S):
    pos = np.vstack([np.asarray(S["posE"], float), np.asarray(S["posI"], float)])
    return ELR.cell_to_voxel(pos, L_MM, N_GRID)


def _sensor_only_cfg():
    c = LC._slowoff_cfg()
    c["record_calib"] = True
    return c


def _run_sensor_only(S, seed, T_ms, elr):
    p = dataclasses.replace(S["p"], T=float(T_ms), dt=DT)
    mz = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**_sensor_only_cfg()), NE=S["NE"],
                    core_mask_E=OLD.build_core_masks(S))
    slow = _SensorSampler(mz, elr, every_steps=max(1, int(round(SENSOR_EVERY_MS / DT))))
    S["net"]["rng"] = np.random.default_rng(int(seed))
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"],
                        early_stop_runaway=False)
    return res, mz, slow


def _events_with_offsets(res, bar):
    """Canonical onsets AND offsets under the frozen bar (plan 3.1)."""
    events, af, af_bin, floor, _ = OLD._events_from_res(res, DT, event_bar=bar)
    ret = [e for e in events if e["returned"]]
    on = np.array([float(e["t_on"]) for e in ret])
    off = np.array([float(e["t_on"]) + float(e["dur_ms"]) for e in ret])
    return ret, on, off, np.asarray(af, float), float(af_bin), floor


def _replay_q(load_tv, b_v, eps_s, tau_R_ms, occupied):
    """pass 2: offline replay of the envelope on the RECORDED load.  No simulation."""
    L = np.asarray(load_tv, float)
    q = np.zeros(L.shape[1])
    a = float(np.exp(-H2.DT_R_MS / float(tau_R_ms)))
    out = np.empty_like(L)
    for t in range(L.shape[0]):
        e = ELR.deadband_positive(L[t] - b_v, eps_s)
        e[~occupied] = 0.0
        q = q * a + e * (1.0 - a)
        out[t] = q
    return out


def cmd_calibration(args):
    """seed1 then seed3.  seed1 failing the timescale gate or CALIBRATION_INVALID -> no seed3."""
    with _stage_lock(f"calibration_s{args.seed}"):
        FCXR._assert_engine_blessed()
        _begin(f"calibration_s{args.seed}", T=T_CAL_MS, seed=args.seed)
        swap_base = LC._swap_used_mb()
        try:
            st, info = _guard(f"calibration_s{args.seed}_start", swap_base)
            if st == "hard":
                raise SystemExit(f"resource hard-stop before start: {info}")
            S = PP.build_substrate(args.seed)
            pk = load_onset_depletion_pi(LC.SNAP_ZA_FMT.format(seed=args.seed))
            assert_field_substrate_aligned(pk, S)
            voxel = _voxel_map(S)
            nv = N_GRID * N_GRID
            # +inf background during the probe -> the source is identically zero, so this run is a
            # pure sensor: q_v stays 0 and cannot feed back even by accident.
            cfg0 = ELR.ELRConfig(b_v=np.full(nv, np.inf), eps_s=1.0, tau_R_ms=1.0, Q_on=1.0,
                                 Q_scale=1.0, eps_q=0.1, n_grid=N_GRID, dt_R_ms=H2.DT_R_MS,
                                 enabled=False, record_load=True, record_q_trace=False)
            elr = ELR.EventLimitedRecruitment(S["N"], voxel, cfg0)

            t0 = time.time()
            res, mz, slow = _run_sensor_only(S, args.seed, T_CAL_MS, elr)
            wall = round(time.time() - t0, 1)

            bar = float(OLD.slowoff_event_bar(res, DT))
            ret, on, off, af, af_bin, floor = _events_with_offsets(res, bar)
            rate = np.asarray(res["rate_E"], float)
            num = LC._numerical(S, res, mz, DT)
            roll_hi = rolling_rate_upper(rate, DT)
            win = build_windows(rate, DT, af, af_bin, roll_hi, ret, LC.LC_WIN_MS,
                                event_lookback_ms=LC.LC_LOOKBACK_MS, finite=num["finite"])
            lc = classify_lifecycle(win, json.load(open(INPUTS[f"lc1_baseline_seed{args.seed}"]))
                                    ["band"])
            sensor = np.stack(slow.sensor).astype(np.float32)
            load = np.stack(elr.load_trace)                       # (n_block, n_voxel)
            del res
            gc.collect()

            # ---- pass 1: b_v from the CALIBRATION half only
            n_half = int(load.shape[0] * H2.CAL_SPLIT_FRAC)
            occupied = elr.occupied
            b_v = H2.background_envelope(load[:n_half], occupied)
            eps_s = H2.eps_s_from_background(b_v)

            # ---- timescale from canonical onsets AND offsets
            gaps = H2.event_gaps(on, off)
            g05 = float(np.quantile(gaps, 0.05)) if gaps.size else 0.0
            g01 = float(np.quantile(gaps, 0.01)) if gaps.size else 0.0
            gmin = float(gaps.min()) if gaps.size else 0.0
            tr = H2.tau_R_from_timescale(H2.T_EVENT_GUARD_MS, g05)
            if not tr["feasible"]:
                v = dict(status="DESIGN_BLOCKED_EVENT_TIMESCALE", tau=tr, gap_05_ms=g05,
                         n_events=len(ret), seed=args.seed)
                _jw(os.path.join(OUT, "DESIGN_BLOCKED_EVENT_TIMESCALE.json"), v)
                _end(f"calibration_s{args.seed}", False, **v)
                print(f"[calibration] seed{args.seed} DESIGN_BLOCKED_EVENT_TIMESCALE {tr}",
                      flush=True)
                return 2
            tau_R = tr["tau_R_ms"]

            # ---- pass 2: offline envelope replay -> per-event peaks -> Q_on
            q_tv = _replay_q(load, b_v, eps_s, tau_R, occupied)
            peaks = H2.event_peak_values(q_tv, on, occupied, tau_R, H2.DT_R_MS)
            q_on = H2.q_on_from_event_peaks(peaks) if peaks else 0.0
            cal = H2.adjudicate_calibration(dict(T_event_guard_ms=H2.T_EVENT_GUARD_MS,
                                                 gap_05_ms=g05, gap_01_ms=g01, gap_min_ms=gmin,
                                                 Q_on=q_on, Q_scale=q_on))

            np.save(os.path.join(OUT, f"load_seed{args.seed}.npy"),
                    load.astype(np.float32))
            np.savez_compressed(
                os.path.join(OUT, f"calibration_seed{args.seed}.npz"),
                b_v=b_v, eps_s=np.float64(eps_s), occupied=occupied,
                sensor=sensor, sensor_dt_ms=np.float64(SENSOR_EVERY_MS),
                p_i=np.asarray(pk["p_i"], np.float64),
                onsets_ms=on, offsets_ms=off, gaps_ms=gaps,
                q_peaks_cal=np.asarray(peaks, float),
                q_max_series=q_tv[:, occupied].max(axis=1).astype(np.float32))

            payload = dict(
                seed=args.seed, status=cal["status"], generated=datetime.now(timezone.utc).isoformat(),
                code_commit=FCXR._git_sha(), T_ms=T_CAL_MS, wall_s=wall,
                peak_rss_gb=round(_rss_gb(), 2), frozen_event_bar=bar, n_events=len(ret),
                event_durations_ms=[round(float(e["dur_ms"]), 3) for e in ret],
                gap_ms=dict(n=int(gaps.size), q05=g05, q01=g01, min=gmin,
                            median=float(np.median(gaps)) if gaps.size else 0.0),
                T_event_guard_ms=H2.T_EVENT_GUARD_MS, tau=cal["tau"],
                tau_R_ms=cal.get("tau_R_ms"), residual_tail=cal.get("residual_tail"),
                b_v=dict(median=float(np.median(b_v[occupied])), max=float(b_v[occupied].max()),
                         n_occupied=int(occupied.sum())), eps_s=eps_s,
                Q_on=q_on, Q_scale=q_on, eps_q=H2.EPS_Q_FRAC * q_on, I_R_max=H2.I_R_MAX,
                n_cal_event_peaks=len(peaks),
                b0_off_arm=dict(label=lc["label"], numerical=num, roll_hi_hz=roll_hi,
                                note="this run IS the Gate B0 ELR-off arm for this seed"),
                artifact=f"calibration_seed{args.seed}.npz")
            _jw(os.path.join(OUT, f"calibration_seed{args.seed}.json"), payload)
            ok = cal["status"] == "CALIBRATION_LOCKED"
            if not ok:
                _jw(os.path.join(OUT, f"{cal['status']}.json"), payload)
            _end(f"calibration_s{args.seed}", ok, status=cal["status"], wall_s=wall)
            print(f"[calibration] seed{args.seed} {cal['status']}  n_ev={len(ret)} "
                  f"GAP05={g05:.1f} GAP01={g01:.1f} GAPmin={gmin:.1f} ms  "
                  f"tau_R={cal.get('tau_R_ms', float('nan')):.2f} ms  "
                  f"b_v_med={np.median(b_v[occupied]):.1f} Hz  Q_on={q_on:.3f}  "
                  f"off-arm={lc['label']}  ({wall}s, peak {round(_rss_gb(),1)}GB)", flush=True)
            return 0 if ok else 2
        except BaseException as e:
            _end(f"calibration_s{args.seed}", False, error=repr(e))
            raise


# ------------------------------------------------------------------ finalize (offline)
def cmd_finalize(args):
    """Global tau_R from the MINIMUM cross-seed GAP_05 (plan 3.6), then re-derive each seed's
    Q_on at that tau_R by replaying the RECORDED load.  Fully offline: no simulation."""
    _begin("finalize")
    try:
        per = {}
        for seed in SEEDS:
            jp = os.path.join(OUT, f"calibration_seed{seed}.json")
            if not os.path.exists(jp):
                raise SystemExit(f"missing {jp}: run calibration --seed {seed} first")
            j = json.load(open(jp))
            if j["status"] != "CALIBRATION_LOCKED":
                raise SystemExit(f"seed{seed} is {j['status']}: finalize refuses to proceed")
            per[seed] = j

        gap05 = {s: per[s]["gap_ms"]["q05"] for s in SEEDS}
        gap05_lock = min(gap05.values())
        tr = H2.tau_R_from_timescale(H2.T_EVENT_GUARD_MS, gap05_lock)
        if not tr["feasible"]:
            v = dict(status="DESIGN_BLOCKED_EVENT_TIMESCALE", tau=tr, gap_05_per_seed=gap05)
            _jw(os.path.join(OUT, "DESIGN_BLOCKED_EVENT_TIMESCALE.json"), v)
            _end("finalize", False, **v)
            return 2
        tau_R = tr["tau_R_ms"]

        seeds_out, ok = {}, True
        for seed in SEEDS:
            z = np.load(os.path.join(OUT, f"calibration_seed{seed}.npz"))
            b_v, eps_s, occ = z["b_v"], float(z["eps_s"]), z["occupied"].astype(bool)
            on = z["onsets_ms"]
            # the recorded load is not re-stored (it is large); replay from the q_max series is not
            # enough, so the per-seed npz keeps what IS needed: we recompute Q_on from the stored
            # per-event peaks ONLY when tau_R is unchanged, otherwise we must reload the load.
            load_p = os.path.join(OUT, f"load_seed{seed}.npy")
            if os.path.exists(load_p):
                q_tv = _replay_q(np.load(load_p, mmap_mode="r"), b_v, eps_s, tau_R, occ)
                peaks = H2.event_peak_values(q_tv, on, occ, tau_R, H2.DT_R_MS)
                q_on = H2.q_on_from_event_peaks(peaks)
                src = "re-derived at the global tau_R from the recorded load"
            elif abs(per[seed]["tau_R_ms"] - tau_R) < 1e-9:
                # No recorded load, but `q_max_series` IS the full-record max over occupied voxels
                # at this seed's own tau_R -- and that tau_R equals the global one (this seed set
                # it, being the min-GAP_05 seed).  So max over event windows of that series is
                # exactly max_{event, occupied voxel}: an exact re-derivation, not a carried-over
                # value.  It matters because the stored per-seed Q_on came from the superseded
                # first-half-only loop; here it happens to agree, but agreeing by luck is not a
                # provenance any downstream gate should rest on.
                q_on = H2.q_on_from_event_peaks(
                    H2.event_peak_values(np.asarray(z["q_max_series"], float)[:, None], on,
                                         np.array([True]), tau_R, H2.DT_R_MS))
                src = ("re-derived from the stored q_max_series (no recorded load; this seed's own "
                       "tau_R already equals the global tau_R, so the series is at the right tau_R)")
            else:
                q_on = float(per[seed]["Q_on"])
                src = (f"per-seed value at that seed's own tau_R "
                       f"({per[seed]['tau_R_ms']:.2f} ms); the recorded load was not retained AND "
                       f"that tau_R differs from the global one, so it could NOT be re-derived")
                ok = False
            # b_v is quantised: one spike in a voxel-block is n_cells_per_voxel / dt_R Hz, so b_v
            # can only be an integer multiple of that.  About half of all occupied voxels sit just
            # below the 99th-percentile boundary and get b_v = 0, which makes the MEDIAN an
            # uninformative summary (it flips between 0 and one quantum on a ~50/50 split).  Report
            # the zero fraction instead -- and note what it MEANS: in those voxels the deadband does
            # no work and Q_on is the only gate, i.e. the two-stage gate collapses to one stage.
            bo = b_v[occ]
            quantum = float(np.min(bo[bo > 0])) if np.any(bo > 0) else float("nan")
            seeds_out[f"seed{seed}"] = dict(
                Q_on=q_on, Q_scale=q_on, eps_q=H2.EPS_Q_FRAC * q_on, eps_s=eps_s,
                b_v_median=float(np.median(bo)), b_v_zero_frac=float(np.mean(bo == 0.0)),
                b_v_quantum_hz=quantum, b_v_mean=float(bo.mean()), b_v_max=float(bo.max()),
                n_occupied=int(occ.sum()),
                gap_05_ms=gap05[seed], gap_01_ms=per[seed]["gap_ms"]["q01"],
                gap_min_ms=per[seed]["gap_ms"]["min"], n_events=per[seed]["n_events"],
                Q_on_source=src, artifact=f"calibration_seed{seed}.npz")

        lock = dict(
            status="CALIBRATION_LOCK" if ok else "CALIBRATION_LOCK_INCOMPLETE",
            generated=datetime.now(timezone.utc).isoformat(), code_commit=FCXR._git_sha(),
            plan="docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md",
            T_event_guard_ms=H2.T_EVENT_GUARD_MS, gap_05_per_seed=gap05,
            gap_05_lock_ms=gap05_lock, gap_05_lock_rule="min over seeds (plan 3.6)",
            tau_R_ms=tau_R, tau_interval=tr["interval"], headroom_ms=tr["headroom_ms"],
            residual={k: H2.residual(v, tau_R) for k, v in
                      dict(gap_05=gap05_lock,
                           gap_01=min(per[s]["gap_ms"]["q01"] for s in SEEDS),
                           gap_min=min(per[s]["gap_ms"]["min"] for s in SEEDS)).items()},
            I_R_max=H2.I_R_MAX, dt_R_ms=H2.DT_R_MS, Q_BG=H2.Q_BG,
            eps_s_rule="0.10 * median_v(b_v), per seed",
            eps_q_rule="0.10 * Q_on, per seed", Q_scale_rule="Q_scale := Q_on (plan 4.3)",
            seeds=seeds_out,
            b0_off_arm={f"seed{s}": per[s]["b0_off_arm"] for s in SEEDS},
            inputs_sha256=json.load(open(os.path.join(OUT, "preflight.json")))["inputs"],
            note=("tau_R is GLOBAL (min cross-seed GAP_05); b_v / Q_on / Q_scale are PER SEED "
                  "under the identical frozen rule, so the rule is out-of-sample on seed3"),
            deadband_coverage_note=(
                "About half of all occupied voxels have b_v = 0 (they fire in under 1% of 0.5 ms "
                "blocks), so in that half the deadband provides no protection and Q_on is the only "
                "gate -- the two-stage gate collapses to one stage there. b_v is also quantised to "
                "multiples of one spike per voxel-block, which is why its MEDIAN is uninformative "
                "and flips between 0 and one quantum across seeds on a ~50/50 split. Recorded "
                "BEFORE Gate B0 and Gate A0 run; Q_BG is NOT adjusted."))
        _jw(os.path.join(OUT, "calibration_lock.json"), lock)
        _end("finalize", ok, status=lock["status"])
        print(f"[finalize] {lock['status']}  GAP05 per seed {gap05}  lock={gap05_lock:.1f} ms  "
              f"tau_R={tau_R:.2f} ms  headroom={tr['headroom_ms']:.2f}  "
              f"residual={ {k: round(v,5) for k,v in lock['residual'].items()} }  "
              + "  ".join(f"Q_on[s{s}]={seeds_out[f'seed{s}']['Q_on']:.2f}" for s in SEEDS),
              flush=True)
        return 0 if ok else 2
    except BaseException as e:
        _end("finalize", False, error=repr(e))
        raise


# ------------------------------------------------------------------ Gate B0 (ELR-on arm)
def _b0_metrics(*, active_occupancy, q_stats, on_events, on_onsets, off_durations_ms,
                off_onsets_ms, band, T_ms, numerical, label):
    """Assemble the Gate B0 measurement dict.  Pure: no simulation, no file IO, so the reduction
    of a 40 minute run is unit-testable without running it."""
    def _cv(t):
        t = np.sort(np.asarray(t, float))
        d = np.diff(t)
        return float(d.std() / d.mean()) if d.size and d.mean() > 0 else 0.0

    dur = np.array([e["dur_ms"] for e in on_events], float)
    par = np.array([e["peak_ext"] for e in on_events], float)
    cv, off_cv = _cv(on_onsets), _cv(off_onsets_ms)
    lo, hi = band["event_rate_lo"], band["event_rate_hi"]
    rate_hz = len(on_events) / (float(T_ms) / 1000.0)
    off_rate_hz = off_durations_ms.size / (float(T_ms) / 1000.0)
    off_dur_med = float(np.median(off_durations_ms)) if off_durations_ms.size else 0.0
    clauses = dict(
        event_rate=bool(lo <= rate_hz <= max(hi, lo + 1e-9)),
        iei_cv=bool(cv >= H1_IEI_CV_MIN and abs(cv - off_cv) <= 0.5 * max(off_cv, 1e-9)),
        duration=bool(dur.size > 0 and abs(float(np.median(dur)) - off_dur_med)
                      <= 0.5 * max(off_dur_med, 1e-9)),
        participation=bool(par.size > 0),
        not_silent=bool(len(on_events) > 0))
    return dict(
        active_occupancy=active_occupancy,
        pre_onset_residual_frac=q_stats["pre_onset_residual_frac"],
        q_floor_drift=q_stats["q_floor_drift"],
        event_stats_in_band=all(clauses.values()),
        event_stats_detail=dict(
            clauses=clauses, event_rate_hz=rate_hz, off_event_rate_hz=off_rate_hz,
            iei_cv=cv, off_iei_cv=off_cv,
            duration_median_ms=float(np.median(dur)) if dur.size else 0.0,
            off_duration_median_ms=off_dur_med,
            participation_median=float(np.median(par)) if par.size else 0.0,
            n_events=len(on_events), off_n_events=int(off_durations_ms.size),
            band=[lo, hi], label=label),
        clip_frac_max=numerical["clip_frac_max"], numerical_unsafe=numerical["numerical_unsafe"])


def cmd_gate_b0(args):
    """One ELR-ON 24 s run per seed; the ELR-off arm is the calibration run of the same seed."""
    with _stage_lock(f"gateB0_s{args.seed}"):
        FCXR._assert_engine_blessed()
        _begin(f"gateB0_s{args.seed}", T=T_CAL_MS, seed=args.seed)
        swap_base = LC._swap_used_mb()
        try:
            lock = json.load(open(os.path.join(OUT, "calibration_lock.json")))
            if lock["status"] != "CALIBRATION_LOCK":
                raise SystemExit(f"calibration is {lock['status']}: Gate B0 may not run")
            k = lock["seeds"][f"seed{args.seed}"]
            off = json.load(open(os.path.join(OUT, f"calibration_seed{args.seed}.json")))
            base = json.load(open(INPUTS[f"lc1_baseline_seed{args.seed}"]))
            z = np.load(os.path.join(OUT, f"calibration_seed{args.seed}.npz"))

            st, info = _guard(f"gateB0_s{args.seed}_start", swap_base)
            if st == "hard":
                raise SystemExit(f"resource hard-stop before start: {info}")
            S = PP.build_substrate(args.seed)
            cfg = ELR.ELRConfig(b_v=z["b_v"], eps_s=k["eps_s"], tau_R_ms=lock["tau_R_ms"],
                                Q_on=k["Q_on"], Q_scale=k["Q_scale"], eps_q=k["eps_q"],
                                I_R_max=lock["I_R_max"], n_grid=N_GRID, dt_R_ms=H2.DT_R_MS,
                                enabled=True, record_load=True, record_q_trace=True)
            elr = ELR.EventLimitedRecruitment(S["N"], _voxel_map(S), cfg)

            t0 = time.time()
            res, mz, _ = _run_sensor_only(S, args.seed, T_CAL_MS, elr)
            wall = round(time.time() - t0, 1)

            bar = float(off["frozen_event_bar"])          # the SAME frozen bar as the off arm
            ret, on, offs, af, af_bin, floor = _events_with_offsets(res, bar)
            rate = np.asarray(res["rate_E"], float)
            num = LC._numerical(S, res, mz, DT)
            roll_hi = rolling_rate_upper(rate, DT)
            win = build_windows(rate, DT, af, af_bin, roll_hi, ret, LC.LC_WIN_MS,
                                event_lookback_ms=LC.LC_LOOKBACK_MS, finite=num["finite"])
            lc = classify_lifecycle(win, base["band"])
            del res
            gc.collect()

            # Clauses 2 and 3 want PER-VOXEL q inside each pre-onset window ("q99 across events x
            # occupied voxels").  The scalar q_trace cannot supply that, so replay the recorded load
            # offline exactly as calibration pass 2 does.
            q_tv = _replay_q(np.stack(elr.load_trace), z["b_v"], k["eps_s"],
                             lock["tau_R_ms"], elr.occupied)
            w = int(round(H2.B0_PRE_ONSET_WINDOW_MS / H2.DT_R_MS))
            pre_windows = [q_tv[int(t_on / H2.DT_R_MS) - w:int(t_on / H2.DT_R_MS)][:, elr.occupied]
                           for t_on in on if int(t_on / H2.DT_R_MS) - w >= 0]
            qs = H2.b0_envelope_statistics(pre_windows, k["Q_on"])
            del q_tv, pre_windows
            elr.load_trace = None
            gc.collect()
            m = _b0_metrics(
                active_occupancy=elr.active_occupancy(), q_stats=qs,
                on_events=ret, on_onsets=on,
                off_durations_ms=np.asarray(off["event_durations_ms"], float),
                off_onsets_ms=np.asarray(z["onsets_ms"], float),
                band=base["band"], T_ms=T_CAL_MS, numerical=num, label=lc["label"])
            v = H2.adjudicate_gate_B0(m)
            d = m["event_stats_detail"]
            v.update(seed=args.seed, generated=datetime.now(timezone.utc).isoformat(),
                     code_commit=FCXR._git_sha(), T_ms=T_CAL_MS, wall_s=wall,
                     peak_rss_gb=round(_rss_gb(), 2), measured=m, q_stats=qs,
                     elr_config=dict(tau_R_ms=lock["tau_R_ms"], Q_on=k["Q_on"],
                                     Q_scale=k["Q_scale"], eps_s=k["eps_s"], eps_q=k["eps_q"],
                                     I_R_max=lock["I_R_max"]),
                     off_arm=dict(label=off["b0_off_arm"]["label"],
                                  n_events=d["off_n_events"],
                                  source=f"calibration_seed{args.seed}.json"),
                     q_max_running=elr.q_running_max, t_gate_ms=elr.t_gate_ms())
            _jw(os.path.join(OUT, f"gate_b0_seed{args.seed}.json"), v)
            ok = v["status"] == "BASELINE_INVISIBLE"
            if not ok:
                _jw(os.path.join(OUT, f"STOP_ELR_BASELINE_VISIBLE_seed{args.seed}.json"), v)
            _end(f"gateB0_s{args.seed}", ok, status=v["status"], wall_s=wall)
            print(f"[gateB0] seed{args.seed} {v['status']}  "
                  f"occ={m['active_occupancy']:.5f} resid={m['pre_onset_residual_frac']:.5f} "
                  f"drift={m['q_floor_drift']:+.5f}  "
                  f"n_ev {d['off_n_events']}->{d['n_events']}  "
                  f"IEI_CV {d['off_iei_cv']:.3f}->{d['iei_cv']:.3f}  "
                  f"dur {d['off_duration_median_ms']:.1f}->{d['duration_median_ms']:.1f} ms  "
                  f"label={d['label']}  qmax={elr.q_running_max:.2f} (Q_on {k['Q_on']:.1f})  "
                  f"({wall}s)", flush=True)
            return 0 if ok else 2
        except BaseException as e:
            _end(f"gateB0_s{args.seed}", False, error=repr(e))
            raise


# ------------------------------------------------------------------ Gate A0 (actuator efficacy)
A0_T_MS = 9000.0                      # plan 5.2: 9 s, not 6 -- q50 reaches dense ~3 s, ictal ~6 s
A0_Z = dict(I_th_EI=1.6652801609959704, tau_z=10000.0)     # LC1 q50, the ONLY authorised input


def _a0_window_readout(res, S, elr, t_gate_ms, window_ms):
    """Recruitment extent inside [t_gate, t_gate+window].  Same three measures as B2.1."""
    spk = res["E_spk_bool"]
    a = int(round(t_gate_ms / DT))
    b = min(spk.shape[0], int(round((t_gate_ms + window_ms) / DT)))
    part = spk[a:b].any(axis=0)
    posE = np.asarray(S["posE"], float)
    sel = posE[part]
    rad = float(np.sqrt(np.mean(np.sum((sel - np.asarray(S["src_xy"], float)) ** 2, axis=1)))) \
        if sel.shape[0] else float("nan")
    vox = _voxel_map(S)[:S["NE"]]
    return dict(window_participants=int(part.sum()), recruitment_radius_mm=rad,
                participant_voxels=int(np.unique(vox[part]).size),
                window_ms=[t_gate_ms, t_gate_ms + window_ms])


def cmd_gate_a0(args):
    """One arm per invocation so the two can occupy two workers.  ELR-ON and ELR-OFF are the SAME
    deterministic simulation up to t_gate; the off arm zeroes only the membrane current and keeps
    q_v evolving, so t_gate is a counterfactual sensor tracked identically in both."""
    arm = args.arm
    with _stage_lock(f"gateA0_{arm}"):
        FCXR._assert_engine_blessed()
        _begin(f"gateA0_{arm}", T=A0_T_MS, arm=arm)
        swap_base = LC._swap_used_mb()
        try:
            for s_ in SEEDS:
                gp = os.path.join(OUT, f"gate_b0_seed{s_}.json")
                if not os.path.exists(gp):
                    raise SystemExit(f"missing {gp}: Gate B0 must pass on both seeds first")
                if json.load(open(gp))["status"] != "BASELINE_INVISIBLE":
                    raise SystemExit(f"Gate B0 seed{s_} did not pass: A0 may not run")
            lock = json.load(open(os.path.join(OUT, "calibration_lock.json")))
            k = lock["seeds"]["seed1"]
            z = np.load(os.path.join(OUT, "calibration_seed1.npz"))

            st, info = _guard(f"gateA0_{arm}_start", swap_base)
            if st == "hard":
                raise SystemExit(f"resource hard-stop before start: {info}")
            S = PP.build_substrate(1)
            cfg_elr = ELR.ELRConfig(b_v=z["b_v"], eps_s=k["eps_s"], tau_R_ms=lock["tau_R_ms"],
                                    Q_on=k["Q_on"], Q_scale=k["Q_scale"], eps_q=k["eps_q"],
                                    I_R_max=lock["I_R_max"], n_grid=N_GRID, dt_R_ms=H2.DT_R_MS,
                                    enabled=(arm == "on"), record_q_trace=True)
            elr = ELR.EventLimitedRecruitment(S["N"], _voxel_map(S), cfg_elr)
            cfg = LC._slowoff_cfg()
            cfg.update(use_z=True, tau_z=A0_Z["tau_z"], I_th_EI=A0_Z["I_th_EI"], record_calib=True)

            import dataclasses as _dc
            p = _dc.replace(S["p"], T=A0_T_MS, dt=DT)
            mz = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**cfg), NE=S["NE"],
                            core_mask_E=OLD.build_core_masks(S))
            S["net"]["rng"] = np.random.default_rng(1)
            t0 = time.time()
            res = simulate_kick(p, S["net"], 0.0, slow=ELR.ELRMZAdapter(mz, elr),
                                kick_center=list(S["src_xy"]), r_kick=PP.R_KICK, t_kick=1e9,
                                V_th_per_neuron=S["vth"], early_stop_runaway=False)
            wall = round(time.time() - t0, 1)

            num = LC._numerical(S, res, mz, DT)
            rate = np.asarray(res["rate_E"], float)
            end_rate = float(rate[-int(1000.0 / DT):].mean())
            tg = elr.t_gate_ms()
            read = _a0_window_readout(res, S, elr, tg, H2.A0_WINDOW_MS) if tg is not None else {}
            del res
            gc.collect()

            payload = dict(arm=arm, generated=datetime.now(timezone.utc).isoformat(),
                           code_commit=FCXR._git_sha(), T_ms=A0_T_MS, wall_s=wall,
                           peak_rss_gb=round(_rss_gb(), 2), z_config=A0_Z,
                           elr_enabled=(arm == "on"), t_gate_ms=tg,
                           ms_after_t_gate=(A0_T_MS - tg) if tg is not None else 0.0,
                           crossed_Q_on=bool(tg is not None), q_max=elr.q_running_max,
                           Q_on=k["Q_on"], max_R_evt=float(
                               ELR.recruit_current(np.array([elr.q_running_max]), cfg_elr)[0]),
                           end_rate_hz=end_rate,
                           early_stopped=bool(end_rate >= H2.A0_RUNAWAY_HZ),
                           clip_frac_max=num["clip_frac_max"], finite=num["finite"],
                           numerical=num, n_E=int(S["NE"]),
                           n_occupied_voxels=int(elr.occupied.sum()), **read)
            _jw(os.path.join(OUT, f"gate_a0_arm_{arm}.json"), payload)
            _end(f"gateA0_{arm}", True, arm=arm, t_gate_ms=tg, wall_s=wall)
            print(f"[gateA0:{arm}] t_gate={tg} ms  q_max={elr.q_running_max:.1f} "
                  f"(Q_on {k['Q_on']:.1f})  R_evt_max={payload['max_R_evt']:.3f} "
                  f"(I_R_max {lock['I_R_max']:.3f})  end_rate={end_rate:.1f} Hz  "
                  f"parts={read.get('window_participants')} rad={read.get('recruitment_radius_mm')} "
                  f"vox={read.get('participant_voxels')}  ({wall}s)", flush=True)
            return 0
        except BaseException as e:
            _end(f"gateA0_{arm}", False, error=repr(e))
            raise


def cmd_gate_a0_adjudicate(args):
    """Offline: combine the two arms under the three-way verdict.  No simulation."""
    _begin("gateA0_adjudicate")
    try:
        arms = {a: json.load(open(os.path.join(OUT, f"gate_a0_arm_{a}.json")))
                for a in ("off", "on")}
        if arms["off"]["t_gate_ms"] != arms["on"]["t_gate_ms"]:
            raise SystemExit(f"t_gate differs between arms "
                             f"({arms['off']['t_gate_ms']} vs {arms['on']['t_gate_ms']}): the "
                             f"counterfactual sensor is NOT identical, the comparison is void")
        keys = ("window_participants", "recruitment_radius_mm", "participant_voxels")
        m = dict(crossed_Q_on=arms["on"]["crossed_Q_on"],
                 ms_after_t_gate=arms["on"]["ms_after_t_gate"],
                 n_E=arms["on"]["n_E"], n_occupied_voxels=arms["on"]["n_occupied_voxels"],
                 off=dict(**{k: arms["off"].get(k) for k in keys},
                          end_rate_hz=arms["off"]["end_rate_hz"],
                          early_stopped=arms["off"]["early_stopped"]),
                 on={k: arms["on"].get(k) for k in keys},
                 max_R_evt=arms["on"]["max_R_evt"], clip_frac_max=arms["on"]["clip_frac_max"],
                 finite=arms["on"]["finite"])
        v = H2.adjudicate_gate_A0(m)
        v.update(generated=datetime.now(timezone.utc).isoformat(), code_commit=FCXR._git_sha(),
                 measured=m, arms={a: dict(t_gate_ms=arms[a]["t_gate_ms"], wall_s=arms[a]["wall_s"],
                                           end_rate_hz=arms[a]["end_rate_hz"],
                                           q_max=arms[a]["q_max"]) for a in ("off", "on")},
                 t_gate_identical=True)
        _jw(os.path.join(OUT, "gate_a0.json"), v)
        ok = v["status"] == "A0_RECRUITMENT_EFFECTIVE"
        if not ok:
            _jw(os.path.join(OUT, f"{v['status']}.json"), v)
        _end("gateA0_adjudicate", ok, status=v["status"])
        print(f"[gateA0] {v['status']}" + (f"  relative={ {k: round(x,4) for k,x in v['relative'].items()} }"
                                           f"  {v['n_measures_up']}/{len(v['relative'])} up"
                                           if "relative" in v else f"  {v.get('note','')}"),
              flush=True)
        return 0 if ok else 2
    except BaseException as e:
        _end("gateA0_adjudicate", False, error=repr(e))
        raise


# ------------------------------------------------------------------ S_Z response axis (offline)
def cmd_zaxis(args):
    """plan 6: open-loop cumulative-depletion coordinate, replayed offline on the STORED per-cell
    GABA sensor from the seed-1 calibration run.  No simulation.

    Scope, restated where it is computed: a frozen replay cannot express self-limitation, and if a
    cell's above/below-threshold status never changes then S_Z is strictly proportional to the t=0
    hazard.  What it adds is time-occupancy above threshold.  It is a parameter coordinate for
    spacing three Z points; it predicts nothing about closed-loop branching.
    """
    _begin("zaxis")
    try:
        lock_p = os.path.join(OUT, "calibration_lock.json")
        if not os.path.exists(lock_p):
            raise SystemExit("calibration_lock.json missing: run finalize first (plan section 1)")
        z = np.load(os.path.join(OUT, "calibration_seed1.npz"))
        sensor = np.asarray(z["sensor"], float)              # (n_frame, NE)
        p_i = np.asarray(z["p_i"], float)[:sensor.shape[1]]
        dt = float(z["sensor_dt_ms"])
        T_cal = sensor.shape[0] * dt
        v = H2.adjudicate_z_response_axis(sensor, p_i, dt_ms=dt)
        v.update(generated=datetime.now(timezone.utc).isoformat(), code_commit=FCXR._git_sha(),
                 seed=1, sensor_frames=int(sensor.shape[0]), sensor_dt_ms=dt, T_cal_ms=T_cal,
                 C_analytic=H2.c_analytic(T_cal, H2.TAU_Z_DOWN_MS),
                 hyb1_levels_for_comparison={"H_LO": 96.30, "H_MID": 72.35, "H_HI": 46.80},
                 hyb1_caveat=("HYB1's H_LO landed ON q75 (ratio 0.996) and was therefore NOT an "
                              "interior level; state explicitly whether the new 25% point does too"))
        _jw(os.path.join(OUT, "z_response_axis.json"), v)
        ok = v["status"] == "Z_RESPONSE_AXIS_LOCKED"
        if not ok:
            _jw(os.path.join(OUT, "DESIGN_BLOCKED_Z_RESPONSE_AXIS.json"), v)
        _end("zaxis", ok, status=v["status"])
        msg = f"[zaxis] {v['status']}  T_cal={T_cal:.0f} ms  C={v['C_analytic']:.3f}"
        if ok:
            msg += ("  S_Z q75=%.4f q50=%.4f  " % (v["S_Z_q75"], v["S_Z_q50"]) +
                    "  ".join(f"{k}: I_th={x['I_th_EI']:.2f}" for k, x in v["levels"].items()))
        print(msg, flush=True)
        return 0 if ok else 2
    except BaseException as e:
        _end("zaxis", False, error=repr(e))
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description="FCXR-HYB2 runner (dt=0.05, no kick)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    sub.add_parser("finalize")
    sub.add_parser("zaxis")
    sub.add_parser("gate-a0-adjudicate")
    a = sub.add_parser("gate-a0")
    a.add_argument("--arm", choices=["off", "on"], required=True)
    a.add_argument("--confirm-run", action="store_true")
    for nm in ("calibration", "gate-b0"):
        c = sub.add_parser(nm)
        c.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
        c.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd not in ("preflight", "finalize", "zaxis", "gate-a0-adjudicate") \
            and not args.confirm_run:
        raise SystemExit("REFUSING: simulations require --confirm-run")
    return {"preflight": cmd_preflight, "calibration": cmd_calibration,
            "finalize": cmd_finalize, "zaxis": cmd_zaxis,
            "gate-b0": cmd_gate_b0, "gate-a0": cmd_gate_a0,
            "gate-a0-adjudicate": cmd_gate_a0_adjudicate}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
