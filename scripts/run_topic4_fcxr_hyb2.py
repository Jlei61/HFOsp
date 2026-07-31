"""FCXR-HYB2 runner — event-limited recruitment sprint.

Plan of record: docs/superpowers/plans/2026-07-31-topic4-fcxr-hyb2.md
Nothing runs on import; every simulation requires --confirm-run.  dt=0.05, RC1 substrate, and NO
kick in the calibration / B0 / screen stages (KICK_BOOST=0, t_kick=1e9).

Stages
  preflight     artifact existence + sha256, blessed-engine check, resource snapshot.  No sim.
  calibration   seed1 then seed3, 24 s sensor-only.  Locks b_v / GAP / tau_R / Q_on / Q_scale and
                stores the per-cell GABA sensor for the S_Z replay.  Each seed also IS its own
                Gate B0 ELR-off arm.  THE ONLY 40k RUNS AUTHORISED BEFORE calibration_lock.json.
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
            blk = H2.DT_R_MS
            peaks = []
            for t_on in on[on < n_half * blk]:
                a0 = int(t_on / blk)
                a1 = min(q_tv.shape[0], a0 + int(round(3.0 * tau_R / blk)) + 1)
                if a1 > a0:
                    peaks.append(float(q_tv[a0:a1][:, occupied].max()))
            q_on = H2.q_on_from_event_peaks(peaks) if peaks else 0.0
            cal = H2.adjudicate_calibration(dict(T_event_guard_ms=H2.T_EVENT_GUARD_MS,
                                                 gap_05_ms=g05, gap_01_ms=g01, gap_min_ms=gmin,
                                                 Q_on=q_on, Q_scale=q_on))

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


def main(argv=None):
    ap = argparse.ArgumentParser(description="FCXR-HYB2 runner (dt=0.05, no kick)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("preflight")
    c = sub.add_parser("calibration")
    c.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    c.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd != "preflight" and not args.confirm_run:
        raise SystemExit("REFUSING: simulations require --confirm-run")
    return {"preflight": cmd_preflight, "calibration": cmd_calibration}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
