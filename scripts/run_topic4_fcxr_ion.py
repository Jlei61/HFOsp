"""FCXR-ION runner -- Phase B0-B2 only (spec §12).

Nothing runs on import; every simulation stage requires --confirm-run.

Stages
  b0-units            engine voltage-unit audit (no simulation)          -> b0_voltage_unit_audit.json
  b0-provenance       parameter table + analytic feasibility (no sim)    -> b0_parameter_provenance.json
                                                                            b0_analytic_feasibility.json
  b0-direction-power  ONE 40k / 11 s arm-C pump-off trajectory: the      -> b0_direction_power.json
                      initiation-site power precondition, the                b0_baseline_rate_field.npz
                      mean-rate reproduction check, and the per-cell
                      baseline rate field the heterogeneous initializer
                      needs (spec §4.2c)
  b1-smallnet         ion layer on N~1000 / N~4000 (occupancy-matched)
  b1-gate-h           Gate H adjudication                               -> gate_H.json
  b1-select-f         f' in {0.5, 1.0, 2.0} against the five gates      -> b1_f_selection.json
  b2-bias             40k bias recalibration + one closure iteration    -> b2_bias_calibration.json
  b2-validate         11 s validation run at the frozen bias            -> b2_closure_iteration.json
  b2-confirm          confirmatory conn {1,3} x unseen noise {402,403,404}
  b2-adjudicate       Gate B verdict                                     -> gate_B.json / candidate_verdict.json

Design: docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md (rev4)
Plan:   docs/superpowers/plans/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-B0-B2.md (rev3)
Outputs: results/topic4_sef_hfo/mz_full_conductance_spatial_relay/ion_homeostasis/
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-topic4-fcxr-ion")

import argparse
import dataclasses
import fcntl
import hashlib
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import src.topic4_fcxr_ion as ION                        # noqa: E402

OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "mz_full_conductance_spatial_relay",
                   "ion_homeostasis")

# ------------------------------------------------------------------ locked run constants
DT = 0.05                    # ms, engine step (arm-C pump-off trajectory)
CONN_SEED_DEV = 1
CONN_SEEDS_CONFIRM = (1, 3)
NOISE_DIRECTION_POWER = 202  # reproduces the accepted pump-off arm
NOISE_DEV = 401
NOISE_CONFIRM = (402, 403, 404)
BURN_IN_MS = 1000.0
BLOCK_MS = 2000.0
N_BLOCKS = 5
T_MS = BURN_IN_MS + N_BLOCKS * BLOCK_MS      # 11000 ms
G_SAT = 21.6
N_GRID_40K = 32              # spec §8 primary grid; L=20 mm -> dx = 0.625 mm
DX_MM_40K = 20.0 / N_GRID_40K
DT_ION_MS = 0.5              # spec §8 primary ion sub-step

# ------------------------------------------------------------------ artifact roots (plan §1.3)
WORKTREES = os.path.dirname(ROOT)
MAIN_ROOT = os.path.dirname(WORKTREES)
PUMP_ROOT = os.path.join(WORKTREES, "topic4-mz-fcxr-pump-lifecycle")
HEO1_ROOT = os.path.join(WORKTREES, "topic4-mz-fcxr-heo1")
ARTIFACT_ROOTS = [("worktree", ROOT), ("pump_lifecycle", PUMP_ROOT),
                  ("heo1", HEO1_ROOT), ("main", MAIN_ROOT)]

_PUMP_DIR = "results/topic4_sef_hfo/mz_full_conductance_spatial_relay/pump_lifecycle"
_HEO1_DIR = "results/topic4_sef_hfo/mz_full_conductance_spatial_relay/high_energy_oscillatory_branch"


def _schema_pump_equiv(d):
    return (isinstance(d["per_arm"]["pump_off"]["pooled"]["mean_rate_hz"], float)
            and len(d["per_arm"]["pump_off"]["block_metrics"]) == N_BLOCKS
            and len(d["blocks"]) == N_BLOCKS)


def _schema_pump_calib(d):
    return isinstance(d["event_bar"], float) and d["provenance"]["noise_seed"] == 201


def _schema_e1146(d):
    ac = d["adaptive_cluster"]
    return (ac["stable_k"] == 2 and isinstance(ac["candidate_forward_reverse_pairs"], list)
            and "propagation_stereotypy" in d and "temporal_dynamics" in d)


def _schema_geometry(d):
    return any(c.get("typical_rank") is not None for c in d["channels"])


def _schema_heo1(d):
    return isinstance(d, dict) and len(d) > 0


REQUIRED_ARTIFACTS = {
    "pump_off_baseline": dict(
        rel=f"{_PUMP_DIR}/pump_baseline_equivalence.json", kind="json", schema=_schema_pump_equiv,
        purpose="r0 lock, block metrics and inter-block tolerances of the accepted arm-C pump-off arm"),
    "pump_event_bar": dict(
        rel=f"{_PUMP_DIR}/pump_baseline_calibration.json", kind="json", schema=_schema_pump_calib,
        purpose="frozen canonical event bar (noise 201) reused by the accepted equivalence run"),
    "e1146_propagation": dict(
        rel="results/interictal_propagation_masked/per_subject/epilepsiae_1146.json",
        kind="json", schema=_schema_e1146,
        purpose="template-layer target: two stable templates (Gate B B-real)"),
    "placement_geometry_t_a": dict(
        rel="results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/"
            "epilepsiae_1146_t_a.json", kind="json", schema=_schema_geometry,
        purpose="core_A geometry (template a source foci) for the registered sheet placement"),
    "placement_geometry_t_b": dict(
        rel="results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/"
            "epilepsiae_1146_t_b.json", kind="json", schema=_schema_geometry,
        purpose="core_B geometry (template b source foci)"),
    "heo1_slow_off_contract": dict(
        rel=f"{_HEO1_DIR}/baseline_spectral_contract_seed1.json", kind="json", schema=_schema_heo1,
        purpose="HEO1 slow-off baseline contract (interictal band reference, consulted in T8)"),
    "arm_c_config_source": dict(
        rel="scripts/run_topic4_mz_fcxr.py", kind="source", schema=None,
        purpose="the accepted arm-C FCXR config builder _fc_cfg (e_gaba/e_k/v_match/E_E)"),
}

BLESSED_ENGINE = ("kick_probe.py", "lfp.py", "params.py", "model.py",
                  "connectivity.py", "connectivity_rot.py")


# ================================================================== io / provenance helpers
def _git_sha():
    return subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x).__name__)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=_jsonable)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def blessed_engine_hashes():
    return {f"src/snn_engine/{f}": _sha(os.path.join(ROOT, "src", "snn_engine", f))
            for f in BLESSED_ENGINE}


# ================================================================== resource contract
def _meminfo():
    with open("/proc/meminfo") as f:
        v = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return v["MemAvailable"] / 1048576.0, (v["SwapTotal"] - v["SwapFree"]) / 1048576.0


def _self_rss_gb():
    import resource as _r
    return _r.getrusage(_r.RUSAGE_SELF).ru_maxrss / 1048576.0


def sibling_40k_tasks():
    """Count L=20 Topic-4 SNN workers outside this process tree.  Deliberately broader than the
    older marker list, which predates the run_topic4_zm_* family and would report 0 siblings while
    two 40k jobs are running."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,ppid,args"], capture_output=True, text=True).stdout
    except Exception:
        return 99                                   # fail-safe: assume heavy contention
    me = {os.getpid(), os.getppid()}
    markers = ("run_topic4_mz_", "run_topic4_zm_", "run_topic4_sef", "run_topic4_heo",
               "run_m4_", "topic4_mz_", "run_topic4_fcxr_ion")
    n = 0
    for line in out.splitlines()[1:]:
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, ppid, args = int(parts[0]), int(parts[1]), parts[2]
        if pid in me or ppid in me or "python" not in args.lower():
            continue
        if any(m in args for m in markers):
            n += 1
    return n


def resource_log(tag, extra=None):
    avail, swap = _meminfo()
    row = dict(t=datetime.now(timezone.utc).isoformat(), tag=tag,
               mem_available_gb=round(avail, 2), swap_used_gb=round(swap, 3),
               self_rss_gb=round(_self_rss_gb(), 3), load1=os.getloadavg()[0],
               siblings=sibling_40k_tasks())
    if extra:
        row.update(extra)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "resource_log.jsonl"), "a") as f:
        f.write(json.dumps(row, default=_jsonable) + "\n")
    return row


def _baseline_swap_path():
    return os.path.join(OUT, ".swap_baseline.json")


def record_swap_baseline():
    avail, swap = _meminfo()
    _write_json(_baseline_swap_path(), dict(swap_used_gb=swap, mem_available_gb=avail,
                                            t=datetime.now(timezone.utc).isoformat()))
    return swap


def check_resource_gate(stage, *, single_run_peak_gb=15.0):
    """Re-checked before EVERY 40k launch (prompt §7).  swap delta > 256 MiB -> stop submitting;
    > 512 MiB and rising, or MemAvailable < 2x measured single-run peak -> pause."""
    avail, swap = _meminfo()
    base = swap
    if os.path.exists(_baseline_swap_path()):
        base = json.load(open(_baseline_swap_path()))["swap_used_gb"]
    delta_mib = (swap - base) * 1024.0
    status = "OK"
    if delta_mib > 512.0 or avail < 2.0 * single_run_peak_gb:
        status = "PAUSE"
    elif delta_mib > 256.0:
        status = "NO_NEW_SUBMISSIONS"
    rep = dict(stage=stage, status=status, swap_used_gb=swap, swap_baseline_gb=base,
               swap_delta_mib=round(delta_mib, 1), mem_available_gb=round(avail, 2),
               single_run_peak_gb=single_run_peak_gb, siblings=sibling_40k_tasks())
    if status == "PAUSE":
        _write_json(os.path.join(OUT, "RESOURCE_PAUSED.json"), rep)
    return rep


MAX_WORKERS = 6              # user-authorised ceiling for this sprint (2026-07-28)


def plan_workers(requested, *, per_worker_gb=15.0):
    avail, swap = _meminfo()
    slots = int((avail - 24.0) // per_worker_gb)          # keep 24 GB machine reserve
    sib = sibling_40k_tasks()
    cap = MAX_WORKERS if sib == 0 else max(1, MAX_WORKERS - min(sib, MAX_WORKERS - 1))
    n = max(1, min(int(requested), cap, max(slots, 1)))
    return dict(workers=n, slots=slots, cap=cap, siblings=sib,
                mem_available_gb=round(avail, 2), swap_used_gb=round(swap, 3),
                per_worker_gb=per_worker_gb)


# ================================================================== sentinels / locks
def sentinel(name, payload):
    _write_json(os.path.join(OUT, name), payload)


@contextmanager
def stage_lock(stage):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f".{stage}.lock")
    with open(path, "a+") as lk:
        try:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"stage {stage} is already running (lock {path}); refusing duplicate "
                             f"submission") from exc
        lk.seek(0)
        lk.truncate()
        lk.write(f"pid={os.getpid()} t={datetime.now(timezone.utc).isoformat()}\n")
        lk.flush()
        try:
            yield
        finally:
            fcntl.flock(lk.fileno(), fcntl.LOCK_UN)


@contextmanager
def staged(stage, extra=None):
    """RUNNING/DONE/FAILED sentinels + pid file + resource log around one stage."""
    with stage_lock(stage):
        t0 = time.time()
        sentinel(f"RUNNING_{stage}.json",
                 dict(stage=stage, pid=os.getpid(), started=datetime.now(timezone.utc).isoformat(),
                      argv=sys.argv, **(extra or {})))
        with open(os.path.join(OUT, f"{stage}.pid"), "w") as f:
            f.write(str(os.getpid()))
        resource_log(f"{stage}_start", extra)
        try:
            yield
        except BaseException as exc:
            sentinel(f"FAILED_{stage}.json",
                     dict(stage=stage, error=f"{type(exc).__name__}: {exc}",
                          t=datetime.now(timezone.utc).isoformat(), wall_s=round(time.time() - t0, 1)))
            resource_log(f"{stage}_failed")
            raise
        else:
            sentinel(f"DONE_{stage}.json",
                     dict(stage=stage, wall_s=round(time.time() - t0, 1),
                          finished=datetime.now(timezone.utc).isoformat()))
            resource_log(f"{stage}_done")
        finally:
            p = os.path.join(OUT, f"RUNNING_{stage}.json")
            if os.path.exists(p):
                os.remove(p)


# ================================================================== artifact preflight (plan §1.3)
def resolve_artifact(rel):
    """Search [worktree, pump_lifecycle, heo1, main] in order.  Loud failure when nothing matches --
    never fall back to a relative path and never substitute a default."""
    tried = []
    for label, root in ARTIFACT_ROOTS:
        p = os.path.join(root, rel)
        tried.append(p)
        if os.path.exists(p):
            return dict(resolved_abs_path=p, root_used=label, sha256=_sha(p),
                        mtime=datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat(),
                        size_bytes=os.path.getsize(p))
    raise SystemExit("FCXR-ION preflight: required artifact not found in any root\n"
                     f"  rel  = {rel}\n  tried:\n    " + "\n    ".join(tried))


def preflight(*, write=True):
    entries, failures = {}, []
    for key, spec in REQUIRED_ARTIFACTS.items():
        try:
            info = resolve_artifact(spec["rel"])
        except SystemExit as exc:
            failures.append(dict(key=key, rel=spec["rel"], error=str(exc)))
            entries[key] = dict(rel=spec["rel"], resolved_abs_path=None, schema_ok=False,
                                purpose=spec["purpose"])
            continue
        ok, detail = True, "not a json schema check"
        if spec["kind"] == "json" and spec["schema"] is not None:
            try:
                ok = bool(spec["schema"](json.load(open(info["resolved_abs_path"]))))
                detail = "schema fields present and consistent" if ok else "schema mismatch"
            except Exception as exc:
                ok, detail = False, f"{type(exc).__name__}: {exc}"
        info.update(rel=spec["rel"], schema_ok=ok, schema_detail=detail, purpose=spec["purpose"])
        entries[key] = info
        if not ok:
            failures.append(dict(key=key, rel=spec["rel"], error=detail))

    payload = dict(
        generated=datetime.now(timezone.utc).isoformat(),
        code_commit=_git_sha(),
        search_order=[lbl for lbl, _ in ARTIFACT_ROOTS],
        roots={lbl: root for lbl, root in ARTIFACT_ROOTS},
        artifacts=entries,
        blessed_engine_sha256=blessed_engine_hashes(),
        status="PASS" if not failures else "FAIL",
        failures=failures,
    )
    if write:
        _write_json(os.path.join(OUT, "b0_artifact_preflight.json"), payload)
    if failures:
        if write:
            _write_json(os.path.join(OUT, "FAILED_preflight.json"), payload)
        raise SystemExit("FCXR-ION preflight FAILED:\n" +
                         "\n".join(f"  {f['key']}: {f['error']}" for f in failures))
    return payload


def load_artifact(key):
    return json.load(open(resolve_artifact(REQUIRED_ARTIFACTS[key]["rel"])["resolved_abs_path"]))


# ================================================================== substrate / arm-C trajectory
def _substrate(seed=CONN_SEED_DEV):
    import run_m4_phaseplane as PP                    # noqa: E402  (heavy import; stage-local)
    return PP.build_substrate(int(seed)), PP


def arm_c_pump_off_cfg():
    """The accepted arm-C FCXR config, taken from the same builder the pump sprint used."""
    import run_topic4_mz_fcxr as FCXR                 # noqa: E402
    return FCXR._fc_cfg(1.0, ff_conductance=False, rec_conductance=True,
                        fail_on_clip=False, rec_sat_g=G_SAT)


def cell_voxel_index(pos, L, n_grid):
    """Flat finite-volume voxel index of every neuron (E then I, engine ordering)."""
    p = np.asarray(pos, float)
    ix = np.clip((p[:, 0] / L * n_grid).astype(int), 0, n_grid - 1)
    iy = np.clip((p[:, 1] / L * n_grid).astype(int), 0, n_grid - 1)
    return (iy * n_grid + ix).astype(np.int32)


def run_arm_c(S, *, noise_seed, T_ms=T_MS, slow=None, dump_i=True, verbose=False):
    """One spontaneous (no-kick) arm-C trajectory, parameter-for-parameter identical to the
    accepted pump-off arm: dt=0.05, conn seed from S, kick disabled (t_kick=1e9), per-neuron V_th
    substrate, no early stop.  slow=None -> the caller supplies the plain arm-C MZSlowVars."""
    import run_m4_phaseplane as PP                    # noqa: E402
    import run_topic4_mz_slowvars as OLD              # noqa: E402
    from kick_probe import simulate_kick              # noqa: E402
    from mz_slow_vars import MZSlowVars, MZSlowVarsConfig  # noqa: E402

    p = dataclasses.replace(S["p"], T=float(T_ms), dt=DT)
    if slow is None:
        slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**arm_c_pump_off_cfg()),
                          NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
    S["net"]["rng"] = np.random.default_rng(int(noise_seed))
    res = simulate_kick(p, S["net"], 0.0, slow=slow, kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=1e9, V_th_per_neuron=S["vth"],
                        early_stop_runaway=False, dump_i_spikes=dump_i, verbose=verbose)
    return res, slow


def block_edges(n_steps):
    b0 = int(round(BURN_IN_MS / DT))
    w = int(round(BLOCK_MS / DT))
    return [(b0 + i * w, min(b0 + (i + 1) * w, n_steps)) for i in range(N_BLOCKS)]


def detect_events(res, event_bar, lo, hi):
    """Self-terminating events inside one block, using the FROZEN canonical bar (the accepted
    pipeline: the bar comes from the noise-201 calibration trajectory, not from this run's own max)."""
    import run_topic4_mz_slowvars as OLD              # noqa: E402
    sub = dict(E_spk_bool=res["E_spk_bool"][lo:hi],
               rate_E=np.asarray(res["rate_E"], float)[lo:hi])
    events, _, _, _, _ = OLD._events_from_res(sub, DT, event_bar=event_bar)
    return [e for e in events if e["returned"]]


# ================================================================== stages
def cmd_b0_units(args):
    with staged("b0-units"):
        pre = preflight()
        audit = ION.audit_voltage_units()
        audit["preflight_status"] = pre["status"]
        audit["code_commit"] = pre["code_commit"]
        audit["blessed_engine_sha256"] = pre["blessed_engine_sha256"]
        audit["params_source"] = resolve_artifact("src/snn_engine/params.py")
        audit["arm_c_config_source"] = pre["artifacts"]["arm_c_config_source"]
        _write_json(os.path.join(OUT, "b0_voltage_unit_audit.json"), audit)
        print(f"[b0-units] status={audit['status']}")
        for c in audit["chain"]:
            print(f"   {'OK ' if c['ok'] else 'FAIL'}  {c['step']}\n         {c['evidence']}")
        if audit["status"] != "CONFIRMED":
            raise SystemExit("T1 NOT_CONFIRMED -- stop in B0; do not enter B1 with an unconfirmed "
                             "voltage unit chain (plan §15)")
    return 0


def cmd_b0_provenance(args):
    with staged("b0-provenance"):
        pre = preflight()
        feas = ION.analytic_feasibility()
        feas["code_commit"] = pre["code_commit"]
        prov = dict(generated=datetime.now(timezone.utc).isoformat(),
                    code_commit=pre["code_commit"],
                    allowed_sources=list(ION.ALLOWED_SOURCES),
                    param_table=ION.PARAM_TABLE,
                    r0_source=dict(
                        artifact=pre["artifacts"]["pump_off_baseline"]["resolved_abs_path"],
                        sha256=pre["artifacts"]["pump_off_baseline"]["sha256"],
                        field="per_arm.pump_off.pooled.mean_rate_hz",
                        value=ION.R0_HZ,
                        cross_check_heo1_hz=ION.R0_HEO1_HZ),
                    locks=dict(eta_pump=ION.ETA_PUMP_B0_B2,
                               g_K_ion=ION.G_K_ION_REFERENCE,
                               g_K_ion_kind="effective reference normalization (calibrated in B3)",
                               rho_locked=ION.RHO,
                               f_prime_candidates=list(ION.F_PRIME_CANDIDATES),
                               f_prime_primary=ION.F_PRIME_PRIMARY,
                               n_grid=N_GRID_40K, dx_mm=DX_MM_40K, dt_ion_ms=DT_ION_MS))
        # cross-check the locked r0 against the artifact it claims to come from
        eq = load_artifact("pump_off_baseline")
        got = float(eq["per_arm"]["pump_off"]["pooled"]["mean_rate_hz"])
        prov["r0_source"]["artifact_value"] = got
        if abs(got - ION.R0_HZ) > 1e-6:
            raise SystemExit(f"r0 lock mismatch: module {ION.R0_HZ} vs artifact {got}")
        _write_json(os.path.join(OUT, "b0_parameter_provenance.json"), prov)
        _write_json(os.path.join(OUT, "b0_analytic_feasibility.json"), feas)
        print(f"[b0-provenance] feasibility={feas['status']}  gates={feas['gates']}")
        print(f"   tau_Na={feas['tau_Na_s']:.2f}s  tau_Ko={feas['tau_Ko_s']:.4f}s  "
              f"ratio={feas['tau_ratio']:.1f}x")
        for r in feas["rows"]:
            tag = "primary" if r["is_primary"] else ("candidate" if r["in_candidate_set"]
                                                     else "reference-only")
            print(f"   f'={r['f_prime']:<5} q_ion={r['q_ion']:.5f}  Na*={r['Na_star']:.2f} "
                  f"K_o*={r['K_o_star']:.2f}  dE_K={r['dE_K_interictal_mV']:+.2f} mV "
                  f"({r['dE_K_interictal_pct_Vth']:.0f}% Vth)  50Hz dE_K="
                  f"{r['dE_K_50hz_mV']:+.2f} ({r['dE_K_50hz_pct_Vth']:.0f}%)   [{tag}]")
        if feas["status"] != "PASS":
            raise SystemExit("analytic feasibility gate FAILED -- stop in B0")
    return 0


def cmd_b0_direction_power(args):
    if not args.confirm_run:
        raise SystemExit("b0-direction-power runs a 40k / 11 s trajectory; pass --confirm-run")
    with staged("b0-direction-power", dict(noise_seed=NOISE_DIRECTION_POWER, T_ms=T_MS)):
        pre = preflight()
        gate = check_resource_gate("b0-direction-power")
        resource_log("b0_direction_power_gate", gate)
        if gate["status"] == "PAUSE":
            raise SystemExit(f"resource gate PAUSE before launch: {gate}")
        calib = load_artifact("pump_event_bar")
        eq = load_artifact("pump_off_baseline")
        event_bar = float(calib["event_bar"])
        r0_ref = float(eq["per_arm"]["pump_off"]["pooled"]["mean_rate_hz"])

        t0 = time.time()
        S, PP = _substrate(CONN_SEED_DEV)
        print(f"[b0-direction-power] substrate ready N={S['N']} NE={S['NE']} "
              f"core_A={S['src_xy']} core_B={S['snk_xy']} ({time.time()-t0:.0f}s)", flush=True)
        resource_log("substrate_built")
        res, _slow = run_arm_c(S, noise_seed=NOISE_DIRECTION_POWER, T_ms=T_MS, dump_i=True,
                               verbose=True)
        resource_log("simulation_done", dict(wall_s=round(res["wall_s"], 1)))

        n_steps = res["E_spk_bool"].shape[0]
        blocks = block_edges(n_steps)
        # --- mean-rate reproduction against the accepted arm ---------------------------------
        rate = np.asarray(res["rate_E"], float)
        blk_rates = [float(rate[lo:hi].mean()) for lo, hi in blocks]
        r0_new = float(np.mean(blk_rates))

        # --- events + both direction readouts -------------------------------------------------
        posE = np.asarray(S["posE"], float)
        A, B = np.asarray(S["src_xy"], float), np.asarray(S["snk_xy"], float)
        per_block, all_ev, all_legacy = [], [], dict(n=0, fwd=0)
        for (lo, hi) in blocks:
            ret = detect_events(res, event_bar, lo, hi)
            spk = res["E_spk_bool"][lo:hi]
            init = ION.initiation_site_readout(spk, posE, A, B, ret, dt=DT, core_r=PP.CORE_R)
            leg = ION.two_sided_forward_fraction(spk, posE, A, B, ret, dt=DT, core_r=PP.CORE_R)
            per_block.append(dict(lo=lo, hi=hi, n_events=len(ret), mean_rate_hz=float(rate[lo:hi].mean()),
                                  initiation=dict(n_scoreable=init["n_scoreable"], n_A=init["n_A"],
                                                  n_B=init["n_B"], n_ambiguous=init["n_ambiguous"]),
                                  legacy=leg))
            all_ev.extend(init["per_event"])
            all_legacy["n"] += leg["n_direction_events"]
            if leg["n_direction_events"]:
                all_legacy["fwd"] += int(round(leg["forward_event_fraction"] * leg["n_direction_events"]))

        nA = sum(1 for e in all_ev if e["core"] == "A")
        nB = sum(1 for e in all_ev if e["core"] == "B")
        namb = sum(1 for e in all_ev if e["core"] == "ambiguous")
        n_sc = nA + nB
        pooled = dict(n_events=len(all_ev), n_scoreable=n_sc, n_A=nA, n_B=nB, n_ambiguous=namb,
                      frac_A=nA / max(n_sc, 1), frac_B=nB / max(n_sc, 1),
                      frac_ambiguous=namb / max(len(all_ev), 1))
        power = ION.direction_power_gate(pooled)

        # --- per-cell baseline rate field for the heterogeneous initializer (spec §4.2c) -------
        win_s = sum(hi - lo for lo, hi in blocks) * DT / 1000.0
        cntE = np.zeros(S["NE"], np.int64)
        cntI = np.zeros(S["N"] - S["NE"], np.int64)
        for lo, hi in blocks:
            cntE += res["E_spk_bool"][lo:hi].sum(axis=0)
            cntI += res["I_spk_bool"][lo:hi].sum(axis=0)
        rate_E = (cntE / win_s).astype(np.float32)
        rate_I = (cntI / win_s).astype(np.float32)
        voxel = cell_voxel_index(S["net"]["pos"], S["L"], N_GRID_40K)
        n_per_voxel = np.bincount(voxel, minlength=N_GRID_40K ** 2).astype(np.int32)
        field_path = os.path.join(OUT, "b0_baseline_rate_field.npz")
        _write_npz(field_path,
                   rate_E=rate_E, rate_I=rate_I, cell_voxel=voxel.astype(np.int32),
                   n_cells_per_voxel=n_per_voxel, n_grid=np.int32(N_GRID_40K),
                   dx_mm=np.float64(DX_MM_40K), L=np.float64(S["L"]),
                   noise_seed=np.int32(NOISE_DIRECTION_POWER),
                   conn_seed=np.int32(CONN_SEED_DEV), dt=np.float64(DT),
                   window_blocks=np.asarray(blocks, np.int64), window_s=np.float64(win_s),
                   core_A=np.asarray(A, float), core_B=np.asarray(B, float),
                   core_r=np.float64(PP.CORE_R))
        field_sha = _sha(field_path)

        payload = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            provenance=dict(noise_seed=NOISE_DIRECTION_POWER, conn_seed=CONN_SEED_DEV,
                            T_ms=T_MS, dt=DT, arm="arm-C pump-off (no ion layer)",
                            event_bar=event_bar,
                            event_bar_source=pre["artifacts"]["pump_event_bar"]["resolved_abs_path"],
                            blessed_engine_sha256=pre["blessed_engine_sha256"],
                            wall_s=round(res["wall_s"], 1)),
            reproduction=dict(mean_rate_hz=r0_new, accepted_mean_rate_hz=r0_ref,
                              abs_diff=abs(r0_new - r0_ref),
                              rel_diff=abs(r0_new - r0_ref) / r0_ref,
                              block_rates_hz=blk_rates,
                              reproduced=bool(abs(r0_new - r0_ref) / r0_ref < 1e-6)),
            initiation_site=dict(pooled=pooled, per_block=per_block,
                                 frac_earliest=0.05, ambiguous_frac=0.20, core_r=float(PP.CORE_R),
                                 core_A=[float(A[0]), float(A[1])],
                                 core_B=[float(B[0]), float(B[1])]),
            legacy_two_sided_readout=dict(
                n_direction_events=all_legacy["n"],
                forward_event_fraction=(all_legacy["fwd"] / all_legacy["n"]
                                        if all_legacy["n"] else None),
                note="rev1 readout, kept only as the contrast case (spec §9): it needs BOTH cores "
                     "to participate, so most events are unscoreable on this substrate"),
            power_gate=power,
            baseline_rate_field=dict(path=field_path, sha256=field_sha,
                                     n_E=int(rate_E.size), n_I=int(rate_I.size),
                                     n_grid=N_GRID_40K, dx_mm=DX_MM_40K,
                                     n_empty_voxels=int((n_per_voxel == 0).sum()),
                                     cells_per_voxel_mean=float(n_per_voxel.mean()),
                                     cells_per_voxel_min=int(n_per_voxel.min()),
                                     mean_rate_E_hz=float(rate_E.mean()),
                                     mean_rate_I_hz=float(rate_I.mean()),
                                     window_s=win_s),
        )
        _write_json(os.path.join(OUT, "b0_direction_power.json"), payload)
        print(f"[b0-direction-power] mean_rate {r0_new:.6f} Hz vs accepted {r0_ref:.6f} Hz "
              f"(rel {payload['reproduction']['rel_diff']:.2e})")
        print(f"   events={pooled['n_events']}  scoreable={n_sc}  A={nA} B={nB} amb={namb}  "
              f"-> {power['status']}")
        print(f"   legacy two-sided readout scoreable: {all_legacy['n']}")
        print(f"   rate field: E {rate_E.mean():.3f} Hz, I {rate_I.mean():.3f} Hz, "
              f"empty voxels {int((n_per_voxel == 0).sum())}/{N_GRID_40K**2}")
        if power["status"] != "PASS":
            _write_json(os.path.join(OUT, "INSUFFICIENT_POWER.json"), power)
            raise SystemExit("B0-2 initiation readout has insufficient power on the accepted "
                             "baseline -- stop; do not enter B1/B2 (plan §15)")
    return 0


# ================================================================== small networks (plan §9.1)
# Occupancy is matched to the 40k sheet (39.1 cells/voxel) because that is what the ion layer
# actually sees: the K source is a per-cell average inside a voxel and diffusion is inert at this
# scale (spec §8), so the local statistics -- not the global network size -- set the ion signal.
# Absolute local geometry (density, core radius, kick radius, dx) is therefore kept identical to
# the 40k substrate rather than rescaled.
SMALL_NETS = {
    "n1000": dict(L=3.16228, n_grid=5, T_baseline_ms=3000.0),
    "n4000": dict(L=6.32456, n_grid=10, T_baseline_ms=3000.0),
}
CORE_MEAN, CORE_STD, CORE_R, BASE_MEAN = 17.5, 1.0, 1.5, 18.0   # Stage-5 blessed core field
R_KICK_SMALL = 0.3
DUR_KICK_MS = 18.0
DRIVE = 0.6
G_INH = 3.6


def build_small_net(tag, seed=CONN_SEED_DEV):
    """Small occupancy-matched substrate with one low-V_th core at the sheet centre."""
    from params import Params                                    # noqa: E402
    from connectivity import place_neurons                       # noqa: E402
    from connectivity_rot import build_connectivity_rot          # noqa: E402
    from src.sef_hfo_heterogeneity import sample_core_field      # noqa: E402

    spec = SMALL_NETS[tag]
    p = Params(g=G_INH, L=spec["L"], density=100.0, T=spec["T_baseline_ms"], dt=DT,
               nu_ext_ratio=DRIVE, seed=seed)
    rng = np.random.default_rng(seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng, theta_EE=0.0, AR=2.0, verbose=False)
    is_E = np.zeros(len(net["pos"]), bool)
    is_E[:NE] = True
    centre = np.array([spec["L"] / 2.0, spec["L"] / 2.0])
    vth = sample_core_field(net["pos"], is_E, centre, CORE_R, np.random.default_rng(seed + 7),
                            core_mean=CORE_MEAN, core_std=CORE_STD, base_mean=BASE_MEAN)["vth"]
    from src.snn_engine.ion_homeostasis import cell_to_voxel      # noqa: E402
    return dict(tag=tag, p=p, net=net, NE=NE, NI=NI, N=NE + NI, L=spec["L"],
                n_grid=spec["n_grid"], dx_mm=spec["L"] / spec["n_grid"],
                posE=net["pos"][:NE], vth=vth, core_xy=centre, seed=seed,
                cell_voxel=cell_to_voxel(net["pos"], spec["L"], spec["n_grid"]))


def run_small(S, *, T_ms, noise_seed, slow=None, kick_boost=0.0, t_kick=1e9, kick_center=None,
              dump_i=True):
    from kick_probe import simulate_kick                          # noqa: E402
    from mz_slow_vars import MZSlowVars, MZSlowVarsConfig         # noqa: E402
    import run_topic4_mz_slowvars as OLD                          # noqa: E402
    p = dataclasses.replace(S["p"], T=float(T_ms), dt=DT)
    if slow is None:
        slow = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**arm_c_pump_off_cfg()), NE=S["NE"])
    S["net"]["rng"] = np.random.default_rng(int(noise_seed))
    res = simulate_kick(p, S["net"], float(kick_boost), slow=slow,
                        kick_center=list(kick_center if kick_center is not None else S["core_xy"]),
                        r_kick=R_KICK_SMALL, t_kick=float(t_kick), V_th_per_neuron=S["vth"],
                        early_stop_runaway=False, dump_i_spikes=dump_i)
    return res, slow


def small_rate_field(S, res, *, burn_in_ms=1000.0):
    lo = int(round(burn_in_ms / DT))
    n = res["E_spk_bool"].shape[0]
    win_s = (n - lo) * DT / 1000.0
    rate_E = (res["E_spk_bool"][lo:].sum(axis=0) / win_s).astype(np.float64)
    rate_I = (res["I_spk_bool"][lo:].sum(axis=0) / win_s).astype(np.float64)
    return rate_E, rate_I, win_s


def cmd_b1_smallnet(args):
    if not args.confirm_run:
        raise SystemExit("b1-smallnet runs short small-network trajectories; pass --confirm-run")
    with staged("b1-smallnet"):
        preflight()
        out = {}
        arrays = {}
        for tag in SMALL_NETS:
            S = build_small_net(tag)
            res, _ = run_small(S, T_ms=SMALL_NETS[tag]["T_baseline_ms"], noise_seed=NOISE_DEV)
            rE, rI, win_s = small_rate_field(S, res)
            n_per_vox = np.bincount(S["cell_voxel"], minlength=S["n_grid"] ** 2)
            out[tag] = dict(N=S["N"], NE=S["NE"], NI=S["NI"], L=S["L"], n_grid=S["n_grid"],
                            dx_mm=S["dx_mm"], window_s=win_s,
                            mean_rate_E_hz=float(rE.mean()), mean_rate_I_hz=float(rI.mean()),
                            cells_per_voxel_mean=float(n_per_vox.mean()),
                            cells_per_voxel_min=int(n_per_vox.min()),
                            n_empty_voxels=int((n_per_vox == 0).sum()),
                            E_in_degree_over_pool=Params_C_EE / max(S["NE"], 1),
                            wall_s=round(res["wall_s"], 1))
            arrays[f"{tag}_rate_E"] = rE.astype(np.float32)
            arrays[f"{tag}_rate_I"] = rI.astype(np.float32)
            arrays[f"{tag}_cell_voxel"] = S["cell_voxel"]
            arrays[f"{tag}_NE"] = np.int32(S["NE"])
            arrays[f"{tag}_n_grid"] = np.int32(S["n_grid"])
            arrays[f"{tag}_dx_mm"] = np.float64(S["dx_mm"])
            print(f"[b1-smallnet] {tag}: N={S['N']} grid={S['n_grid']}x{S['n_grid']} "
                  f"cells/voxel={n_per_vox.mean():.1f} rate_E={rE.mean():.3f} Hz "
                  f"rate_I={rI.mean():.3f} Hz  E-in-degree/pool="
                  f"{out[tag]['E_in_degree_over_pool']:.2f}", flush=True)
        path = os.path.join(OUT, "b1_smallnet_rate_fields.npz")
        _write_npz(path, **arrays)
        out["_meta"] = dict(path=path, sha256=_sha(path), noise_seed=NOISE_DEV,
                            conn_seed=CONN_SEED_DEV, dt=DT,
                            note="each network supplies its OWN per-cell rate field; the 40k field "
                                 "must not be pushed onto a small network (spec §4.2c)")
        _write_json(os.path.join(OUT, "b1_smallnet.json"), out)
    return 0


Params_C_EE = 800          # engine in-degree (params.py); recorded to expose small-net degeneracy


# ================================================================== Gate H (plan §8)
def _gate_h_network_items(S, rE, rI, q_ion):
    """The items that need this network's own rate field / dynamics."""
    from src.snn_engine.ion_homeostasis import (                  # noqa: E402
        IonHomeostasisConfig, IonHomeostaticMZAdapter, build_from_rate_field, resting_state)
    from mz_slow_vars import MZSlowVars, MZSlowVarsConfig         # noqa: E402
    import hashlib as _h

    ng, dx = S["n_grid"], S["dx_mm"]
    cfg = IonHomeostasisConfig(q_ion=q_ion, n_grid=ng, dx_mm=dx, dt_ion_ms=DT_ION_MS)
    ions, rep = build_from_rate_field(S["N"], S["NE"], S["cell_voxel"], cfg, rE, rI,
                                      return_report=True)
    rates = np.concatenate([rE, rI])
    dNa, dK = ions.derivatives(rates)
    lim = ION.GATE_H_RESIDUAL_MAX_MM_S
    resid = dict(
        q95_abs_dNa_dt=float(np.quantile(np.abs(dNa), 0.95)),
        q99_abs_dNa_dt=float(np.quantile(np.abs(dNa), 0.99)),
        max_abs_dNa_dt=float(np.abs(dNa).max()),
        q95_abs_dKo_dt=float(np.quantile(np.abs(dK), 0.95)),
        q99_abs_dKo_dt=float(np.quantile(np.abs(dK), 0.99)),
        max_abs_dKo_dt=float(np.abs(dK).max()),
        mean_abs_dNa_dt=float(np.abs(dNa).mean()),      # reported, never a pass criterion
        mean_abs_dKo_dt=float(np.abs(dK).mean()),
        threshold=lim, n_iter=rep["n_iter"], converged=bool(rep["converged"]),
        n_empty_voxels=int(rep["n_empty_voxels"]),
        n_voxels_interpolated=int(rep["n_voxels_interpolated"]))
    resid["ok"] = bool(max(resid[f"{s}_abs_{v}"] for s in ("q95", "q99", "max")
                           for v in ("dNa_dt", "dKo_dt")) < lim and resid["converged"])

    # reverse regression: the scalar initializer must NOT pass the same gate
    sca = ION.scalar_steady_state_init(rE, rI, S["cell_voxel"][:S["NE"]],
                                       S["cell_voxel"][S["NE"]:], n_grid=ng, q_ion=q_ion, dx_mm=dx)
    resid["scalar_init_q99_abs_dNa_dt"] = sca["q99_abs_dNa_dt"]
    resid["scalar_init_would_pass"] = bool(sca["q99_abs_dNa_dt"] < lim)

    # baseline pump is constitutive, not silently off
    pump_mean = float(ions.pump_flux_all.mean())
    pump = dict(ok=bool(pump_mean > ION.GATE_H_PUMP_MIN_FRAC * ION.I_PUMP_0),
                mean_I_pump=pump_mean, I_pump_0=ION.I_PUMP_0,
                ratio_to_rest=pump_mean / ION.I_PUMP_0,
                saturation_frac_of_rho=pump_mean / ION.RHO,
                threshold=ION.GATE_H_PUMP_MIN_FRAC * ION.I_PUMP_0)

    # local perturbation recovers within 3 s (~4.6 tau_Ko)
    probe = build_from_rate_field(S["N"], S["NE"], S["cell_voxel"], cfg, rE, rI)
    tgt = (ng // 2, ng // 2)
    K_ref = float(probe.K_o_grid[tgt])
    probe.K_o_grid[tgt] = K_ref + 0.5
    probe._refresh_membrane_state()
    for _ in range(int(round(3000.0 / DT_ION_MS))):
        probe._cell_spikes[:] = np.round(rates * DT_ION_MS * 1e-3).astype(np.int32)
        probe.update()
    settled = build_from_rate_field(S["N"], S["NE"], S["cell_voxel"], cfg, rE, rI)
    for _ in range(int(round(3000.0 / DT_ION_MS))):
        settled._cell_spikes[:] = np.round(rates * DT_ION_MS * 1e-3).astype(np.int32)
        settled.update()
    resid_rel = abs(float(probe.K_o_grid[tgt]) - float(settled.K_o_grid[tgt])) / max(K_ref, 1e-9)
    recov = dict(ok=bool(resid_rel < ION.GATE_H_RECOVERY_REL_TOL), residual_rel=resid_rel,
                 threshold=ION.GATE_H_RECOVERY_REL_TOL, perturbation_mM=0.5, window_s=3.0)

    # ions-off byte parity through the REAL engine
    def _fp(slow):
        r, _ = run_small(S, T_ms=400.0, noise_seed=NOISE_DEV, slow=slow, dump_i=False)
        return (_h.sha1(r["E_spk_bool"].tobytes()).hexdigest(),
                _h.sha1(np.asarray(r["rate_E"]).tobytes()).hexdigest())

    def _mz():
        return MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**arm_c_pump_off_cfg()), NE=S["NE"])

    off_cfg = IonHomeostasisConfig(q_ion=q_ion, n_grid=ng, dx_mm=dx, dt_ion_ms=DT_ION_MS,
                                   enabled=False)
    bare = _fp(_mz())
    adapt_none = _fp(IonHomeostaticMZAdapter(_mz(), None))
    adapt_off = _fp(IonHomeostaticMZAdapter(_mz(), resting_state(S["N"], S["NE"],
                                                                 S["cell_voxel"], off_cfg)))
    parity = dict(ok=bool(bare == adapt_none == adapt_off), bare_sha=bare[0][:16],
                  adapter_none_sha=adapt_none[0][:16], adapter_disabled_sha=adapt_off[0][:16])

    # no negative concentrations / no guard-band collision on a live ion run
    live = build_from_rate_field(S["N"], S["NE"], S["cell_voxel"], cfg, rE, rI)
    res_live, _ = run_small(S, T_ms=1000.0, noise_seed=NOISE_DEV,
                            slow=IonHomeostaticMZAdapter(_mz(), live), dump_i=False)
    bounds = dict(ok=bool(live.Na_i_all.min() > 0 and live.K_o_grid.min() > 0
                          and live.Na_i_all.max() < live.cfg.na_bounds[1]
                          and live.K_o_grid.max() < live.cfg.ko_bounds[1]),
                  Na_min=float(live.Na_i_all.min()), Na_max=float(live.Na_i_all.max()),
                  K_min=float(live.K_o_grid.min()), K_max=float(live.K_o_grid.max()),
                  na_bounds=list(live.cfg.na_bounds), ko_bounds=list(live.cfg.ko_bounds),
                  n_ion_updates=live.n_updates, mean_rate_E_hz=float(res_live["rate_E"].mean()))
    return dict(heterogeneous_init_residual=resid, baseline_pump_nonzero=pump,
                local_perturbation_recovers=recov, ions_off_byte_parity=parity,
                no_negative_or_bound_collision=bounds)


def _gate_h_numeric_items(S, rE, rI, q_ion):
    """Items that are pure numerics: they do not depend on network dynamics."""
    from src.snn_engine.ion_homeostasis import IonHomeostasisConfig, resting_state  # noqa: E402
    ng, dx = S["n_grid"], S["dx_mm"]
    cfg = IonHomeostasisConfig(q_ion=q_ion, n_grid=ng, dx_mm=dx, dt_ion_ms=DT_ION_MS)

    rest = resting_state(S["N"], S["NE"], S["cell_voxel"], cfg)
    for _ in range(400):
        rest.update()
    fixed = dict(ok=bool(np.max(np.abs(rest.Na_i_all - ION.NA_I0)) < ION.GATE_H_FIXED_POINT_TOL
                         and np.max(np.abs(rest.K_o_grid - ION.K_O0)) < ION.GATE_H_FIXED_POINT_TOL),
                 max_abs_Na_drift=float(np.max(np.abs(rest.Na_i_all - ION.NA_I0))),
                 max_abs_K_drift=float(np.max(np.abs(rest.K_o_grid - ION.K_O0))),
                 n_updates=400, threshold=ION.GATE_H_FIXED_POINT_TOL)

    vox = S["cell_voxel"].copy()
    victim = int(np.bincount(vox, minlength=ng * ng).argmax())
    donor = (victim + 1) % (ng * ng)
    vox2 = np.where(vox == victim, donor, vox)
    empty = resting_state(S["N"], S["NE"], vox2, cfg)
    for _ in range(400):
        empty.update()
    ev = dict(ok=bool(int(empty.n_per_grid[victim]) == 0
                      and np.max(np.abs(empty.K_o_grid - ION.K_O0)) < ION.GATE_H_FIXED_POINT_TOL),
              emptied_voxel=victim, n_cells_in_emptied=int(empty.n_per_grid[victim]),
              max_abs_K_drift=float(np.max(np.abs(empty.K_o_grid - ION.K_O0))),
              K_at_empty_voxel=float(empty.K_o_grid.ravel()[victim]),
              broken_form_would_give_mM_s=2.0 * ION.BETA * ION.I_PUMP_0,
              threshold=ION.GATE_H_FIXED_POINT_TOL)

    bud = ION.k_budget_closure(rE, rI, S["cell_voxel"][:S["NE"]], S["cell_voxel"][S["NE"]:],
                               n_grid=ng, q_ion=q_ion, dx_mm=dx, dt_ion_ms=DT_ION_MS, n_steps=400)
    budget = dict(ok=bool(bud["relative_error"] < ION.GATE_H_BUDGET_REL_TOL),
                  relative_error=bud["relative_error"], delta_total_K=bud["delta_total_K"],
                  budget=bud["budget"], terms=bud["terms"], threshold=ION.GATE_H_BUDGET_REL_TOL)
    zf = dict(ok=bool(abs(bud["diffusion_net_flux"]) < ION.GATE_H_DIFFUSION_ABS_TOL),
              diffusion_net_flux=bud["diffusion_net_flux"], threshold=ION.GATE_H_DIFFUSION_ABS_TOL)

    st = resting_state(S["N"], S["NE"], S["cell_voxel"], cfg)
    st.Na_i_all[:] = 21.0
    st._refresh_membrane_state()
    dNa, dK = st.derivatives(np.zeros(S["N"]))
    Ip = float(ION.pump_flux(21.0, ION.K_O0))
    stoich = dict(ok=bool(abs(float(dNa[0]) + 3.0 * (Ip - ION.I_PUMP_0)) < 1e-12
                          and abs(float(dK.ravel()[0]) + 2.0 * ION.BETA * (Ip - ION.I_PUMP_0)) < 1e-12),
                  Na_coefficient=float(-dNa[0] / (Ip - ION.I_PUMP_0)),
                  K_coefficient_over_beta=float(-dK.ravel()[0] / (Ip - ION.I_PUMP_0) / ION.BETA),
                  expected=(3.0, 2.0))

    finals = {}
    for dti in (2.0, 1.0, 0.5, 0.25):
        c2 = IonHomeostasisConfig(q_ion=q_ion, n_grid=ng, dx_mm=dx, dt_ion_ms=dti)
        s2 = resting_state(S["N"], S["NE"], S["cell_voxel"], c2)
        s2.K_o_grid[ng // 2, ng // 2] = ION.K_O0 + 1.0
        s2._refresh_membrane_state()
        for _ in range(int(round(200.0 / dti))):
            s2.update()
        finals[dti] = s2.K_o_grid.copy()
    d = {k: float(np.max(np.abs(finals[a] - finals[b])))
         for k, (a, b) in dict(coarse=(2.0, 1.0), mid=(1.0, 0.5), fine=(0.5, 0.25)).items()}
    conv = dict(ok=bool(d["fine"] < d["mid"] < d["coarse"] and d["fine"] < 1e-4), **d,
                note="numerics only: dt-halving agreement says nothing about whether the "
                     "equations are the right ones")

    totals, fields = {}, {}
    for g2 in (ng, 2 * ng):
        from src.snn_engine.ion_homeostasis import cell_to_voxel as _c2v      # noqa: E402
        v2 = _c2v(S["net"]["pos"], S["L"], g2)
        c2 = IonHomeostasisConfig(q_ion=q_ion, n_grid=g2, dx_mm=S["L"] / g2, dt_ion_ms=DT_ION_MS)
        from src.snn_engine.ion_homeostasis import build_from_rate_field as _b  # noqa: E402
        io2 = _b(S["N"], S["NE"], v2, c2, rE, rI)
        totals[g2] = io2.total_extracellular_K()
        fields[g2] = io2.K_o_grid.copy()
    rel = abs(totals[2 * ng] - totals[ng]) / totals[ng]
    coarse = fields[2 * ng].reshape(ng, 2, ng, 2).mean(axis=(1, 3))
    field_dev = float(np.max(np.abs(coarse - fields[ng])) / np.mean(fields[ng]))
    grid = dict(ok=bool(rel < 1e-3 and field_dev < 1e-2), total_K_rel_diff=rel,
                coarse_grained_max_rel_dev=field_dev, grids=[ng, 2 * ng],
                note="per-voxel equality is NOT expected; the invariants are the total content "
                     "and the coarse-grained field")

    a = resting_state(S["N"], S["NE"], S["cell_voxel"], cfg)
    b = resting_state(S["N"], S["NE"], S["cell_voxel"], cfg)
    rng = np.random.default_rng(17)
    trains = [rng.random(S["N"]) < 0.03 for _ in range(60)]
    for spk in trains:
        a.accumulate(spk)
        a.update()
    for spk in trains[:30]:
        b.accumulate(spk)
        b.update()
    c = resting_state(S["N"], S["NE"], S["cell_voxel"], cfg)
    c.load_state_dict(b.state_dict())
    for spk in trains[30:]:
        c.accumulate(spk)
        c.update()
    ckpt = dict(ok=bool(np.array_equal(a.Na_i_all, c.Na_i_all)
                        and np.array_equal(a.K_o_grid, c.K_o_grid)),
                max_abs_Na_diff=float(np.max(np.abs(a.Na_i_all - c.Na_i_all))),
                max_abs_K_diff=float(np.max(np.abs(a.K_o_grid - c.K_o_grid))))

    return dict(resting_fixed_point=fixed, empty_voxel_fixed_point=ev, k_budget_closure=budget,
                zero_flux_boundary=zf, pump_stoichiometry=stoich, dt_ion_convergence=conv,
                grid_consistency=grid, checkpoint_restart_identity=ckpt)


def cmd_b1_gate_h(args):
    if not args.confirm_run:
        raise SystemExit("b1-gate-h runs short small-network trajectories; pass --confirm-run")
    with staged("b1-gate-h"):
        pre = preflight()
        fields_path = os.path.join(OUT, "b1_smallnet_rate_fields.npz")
        if not os.path.exists(fields_path):
            raise SystemExit("run --stage b1-smallnet first (each network needs its OWN rate field)")
        z = np.load(fields_path)
        q_ion = ION.q_ion_from_fprime(ION.F_PRIME_PRIMARY)
        blessed_ok = pre["blessed_engine_sha256"] == blessed_engine_hashes()

        per_net, verdicts = {}, {}
        for tag in SMALL_NETS:
            t0 = time.time()
            S = build_small_net(tag)
            rE = np.asarray(z[f"{tag}_rate_E"], float)
            rI = np.asarray(z[f"{tag}_rate_I"], float)
            checks = _gate_h_numeric_items(S, rE, rI, q_ion)
            checks.update(_gate_h_network_items(S, rE, rI, q_ion))
            checks["blessed_engine_unmodified"] = dict(
                ok=bool(blessed_ok), sha256=blessed_engine_hashes())
            v = ION.adjudicate_gate_H(checks)
            v["network"] = dict(tag=tag, N=S["N"], NE=S["NE"], n_grid=S["n_grid"],
                                dx_mm=S["dx_mm"], L=S["L"],
                                E_in_degree_over_pool=Params_C_EE / S["NE"],
                                wall_s=round(time.time() - t0, 1))
            per_net[tag] = v
            verdicts[tag] = v["status"]
            print(f"[b1-gate-h] {tag}: {v['status']}  "
                  f"(init residual q99 dNa={checks['heterogeneous_init_residual']['q99_abs_dNa_dt']:.2e}, "
                  f"dKo={checks['heterogeneous_init_residual']['q99_abs_dKo_dt']:.2e})", flush=True)
            for name, code in ION._GATE_H_ORDER:
                if not checks[name].get("ok"):
                    print(f"    FAIL {name} -> {code}: {checks[name]}", flush=True)
            resource_log(f"gate_h_{tag}")

        # N~1000 is degenerate for DYNAMICS (see note): N~4000 is the primary tier.
        primary = "n4000"
        status = per_net[primary]["status"]
        payload = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            status=status, primary_network=primary, per_network=per_net,
            per_network_status=verdicts,
            f_prime_used=ION.F_PRIME_PRIMARY, q_ion=q_ion, dt_ion_ms=DT_ION_MS,
            thresholds=dict(residual_max_mM_s=ION.GATE_H_RESIDUAL_MAX_MM_S,
                            budget_rel_tol=ION.GATE_H_BUDGET_REL_TOL,
                            diffusion_abs_tol=ION.GATE_H_DIFFUSION_ABS_TOL,
                            fixed_point_tol=ION.GATE_H_FIXED_POINT_TOL,
                            recovery_rel_tol=ION.GATE_H_RECOVERY_REL_TOL,
                            residual_rationale=(
                                "1e-6 mM/s integrates to 1.1e-5 mM over the 11 s window, four "
                                "orders below the interictal excursion the layer should produce, "
                                "and four orders below what the scalar initializer leaves")),
            small_network_caveat=(
                "N~1000 is DEGENERATE for network dynamics: the engine's fixed E->E in-degree "
                "C_EE=800 equals its whole 800-cell E pool (ratio 1.00) and it settles at "
                "17.7 Hz against the 40k substrate's 4.16 Hz. The plan's occupancy scaling fixes "
                "the voxel occupancy but cannot fix the absolute in-degree. Its numeric items are "
                "still informative (they do not depend on network dynamics); its dynamics-dependent "
                "items are reported but NOT used as the verdict. N~4000 (in-degree/pool = 0.25, "
                "3.28 Hz) is the primary tier."),
            blessed_engine_sha256=blessed_engine_hashes(),
        )
        _write_json(os.path.join(OUT, "gate_H.json"), payload)
        print(f"[b1-gate-h] VERDICT ({primary}) = {status}   (n1000 tier: {verdicts['n1000']})")
        if status != "PASS":
            raise SystemExit(f"Gate H = {status} -- do NOT enter any 40k stage (plan §8/§15)")
    return 0


# ================================================================== T7: f' selection (plan §9)
# Measurement mode: the ion layer runs as a SENSOR (g_K_ion = 0), i.e. its state is a pure
# function of the recorded raster.  Three reasons, all pre-declared:
#   1. it is the accepted device in this lineage -- the pump sprint calibrated its per-cell load
#      from one sensor-only trajectory plus offline replay, and that was reviewed and accepted;
#   2. the five gates ask "what does ONE interictal event do to the ions".  Measuring that with
#      the feedback live is circular: a larger f' would change the network, change the event and
#      change dK_o, so the quantity being gated would depend on the answer being chosen;
#   3. the integration gate's linear reference (2.973 = sum of exp(-0.2k/tau_Ko), k=0..4) assumes
#      five IDENTICAL deposits; only a fixed raster satisfies that premise.
# The feedback-on behaviour is reported separately as a diagnostic, never as a gate.
# Executed on the 40k substrate with real spontaneous events -- see cmd_b1_select_f's docstring
# for the deviation from the plan's small-network protocol and the instrument failure behind it.
def _blk(ms):
    return int(round(ms / DT_ION_MS))


def _t7_replay(io_factory, cap, kernel, bg, n_bg_blocks, *, hot, participants, with_event=True,
               settle_blocks=4000, clamp_K=None):
    """Offline sensor replay: settle on background, optionally inject ONE recorded event kernel,
    then continue on background only.  Returns the K trace at `hot` and the median Na of the
    event's participants, sampled every block-group.

    The gates' analytic references (K back inside 1 sigma in 3 s = 4.6 tau_Ko; Na excess decaying
    30.8% in 20 s = 1-exp(-20/54.42)) are PURE-RELAXATION predictions.  They are only meaningful
    for an isolated event, which never occurs on a substrate firing an event every ~455 ms -- so
    the replay is not a convenience, it is what makes the gate well posed.
    """
    io = io_factory()
    nb = len(bg)

    def _step(block):
        io.replay_block(block)
        if clamp_K is not None:                 # ruling 2: hold K at THIS f''s working point,
            io.K_o_grid[:] = clamp_K            # removing the event-induced K excursion only
            io._refresh_membrane_state()        # (the baseline working point is unchanged)

    for i in range(settle_blocks):
        _step(cap[bg[i % nb]])
    K0 = float(io.K_o_grid.ravel()[hot])
    Na0 = io.Na_i_all[participants].copy()
    Kt, Nat, Pt = [], [], []

    def _sample():
        Kt.append(float(io.K_o_grid.ravel()[hot]))
        Nat.append(float(np.median(io.Na_i_all[participants] - Na0)))
        Pt.append(float(np.mean(io.pump_flux_all[participants])))

    if with_event:
        for c in kernel:
            _step(c)
            _sample()
    for i in range(n_bg_blocks):
        _step(cap[bg[(settle_blocks + i) % nb]])
        _sample()
    return np.array(Kt), np.array(Nat), K0, np.array(Pt)


def _t7_measure_40k(io_factory, io_live, cap, kernel, bg, events, hot, participants,
                    sigma_rest, f_prime):
    """The five pre-registered quantities, measured on REAL spontaneous interictal events."""
    dt_s = DT_ION_MS * 1e-3
    n20 = int(round(20.0 / dt_s))
    n3 = int(round(3.0 / dt_s))

    # ---- measurable / safe: peak dK_o of a typical single event, on the live sensor trace ----
    K = np.asarray(io_live.k_trace)
    tK = np.asarray(io_live.k_trace_blocks) * DT_ION_MS
    flat = K.reshape(K.shape[0], -1)
    peaks = []
    for ev in events:
        pre = (tK >= ev["t_on"] - 100.0) & (tK < ev["t_on"] - 10.0)
        win = (tK >= ev["t_on"]) & (tK < ev["t_on"] + 250.0)
        if not (pre.any() and win.any()):
            continue
        d = flat[win] - flat[pre].mean(axis=0)
        peaks.append(float(d.max()))
    dK_med = float(np.median(peaks))

    # ---- recovery K + recovery Na: isolated-event replay against a background-only control ----
    Ke, Nae, K0, Pe = _t7_replay(io_factory, cap, kernel, bg, n20, hot=hot,
                                 participants=participants, with_event=True)
    Kc, Nac, _, Pc = _t7_replay(io_factory, cap, kernel, bg, n20, hot=hot,
                                participants=participants, with_event=False)
    nk = len(kernel)
    # the event's own contribution = event replay minus the matched background-only control
    dK_ev = Ke[:len(Kc) + nk][nk:] - Kc[:len(Ke) - nk]
    i_pk = int(np.argmax(Ke[:nk + n3]))
    k_after_3s = abs(float(Ke[min(i_pk + n3, len(Ke) - 1)] - Kc[min(i_pk + n3 - nk, len(Kc) - 1)]))
    k_resid_sigma = k_after_3s / max(sigma_rest, 1e-12)

    exc = Nae[:len(Nac) + nk][nk:] - Nac[:len(Nae) - nk]
    i_peak = int(np.argmax(exc[:max(1, nk + 200)]))       # the peak must occur near the event
    peak_val = float(exc[i_peak])
    i_end = min(i_peak + n20, len(exc) - 1)
    tail = exc[i_peak:i_end + 1]
    still_rising = bool(i_end > i_peak and float(exc[i_end]) > peak_val)
    measurable = bool(peak_val > 0 and (i_end - i_peak) >= int(0.9 * n20) and not still_rising)
    decay = float((peak_val - exc[i_end]) / peak_val) if peak_val > 0 else float("nan")
    slack = ION.F_GATES["na_monotone_sigma_slack"] * float(np.std(np.diff(tail))) if tail.size > 2 else 0.0
    monotone = bool(np.all(np.diff(tail) <= slack + 1e-12))

    # ---- integration: the SAME recorded event replayed 5x at 200 ms ----
    io2 = io_factory()
    nb = len(bg)
    for i in range(4000):
        io2.replay_block(cap[bg[i % nb]])
    pre = float(io2.K_o_grid.ravel()[hot])
    cpeaks, cluster_trace = [], []
    for _ in range(5):
        seg = []
        for c in kernel:
            io2.replay_block(c)
            seg.append(float(io2.K_o_grid.ravel()[hot]))
        cluster_trace.extend(seg)
        cpeaks.append(max(seg) - pre)
    cluster_trace = np.asarray(cluster_trace) - pre
    ratio = cpeaks[4] / cpeaks[0] if cpeaks[0] > 0 else float("nan")

    # ---------------- T7.1 additions (user rulings 2026-07-28) ----------------
    dt_s = DT_ION_MS * 1e-3
    tail = exc[i_peak:i_end + 1]
    net_decay = float((peak_val - exc[i_end]) / peak_val) if peak_val > 0 else float("nan")
    slope, se, tstat = ION._tail_slope(tail, dt_s, ION.F_GATES_V2["na_tail_window_s"])

    # ruling 2: K-clamp control at THIS f''s own working point K_o*(f'), NOT at rest 4.0
    Jm, Na_star, Ko_star = ION.coupled_working_point_jacobian(f_prime)
    Kec, Naec, _, Pec = _t7_replay(io_factory, cap, kernel, bg, n20, hot=hot,
                                   participants=participants, with_event=True, clamp_K=Ko_star)
    Kcc, Nacc, _, Pcc = _t7_replay(io_factory, cap, kernel, bg, n20, hot=hot,
                                   participants=participants, with_event=False, clamp_K=Ko_star)
    exc_c = Naec[:len(Nacc) + nk][nk:] - Nacc[:len(Naec) - nk]
    ip_c = int(np.argmax(exc_c[:max(1, nk + 200)]))
    pkc = float(exc_c[ip_c])
    iec = min(ip_c + n20, len(exc_c) - 1)
    decay_clamped = float((pkc - exc_c[iec]) / pkc) if pkc > 0 else float("nan")

    # coupled-Jacobian prediction from the MEASURED initial perturbation
    dK0_event = float(np.max(Ke[:nk]) - K0)
    pred_coupled, _J, _Na, _Ko = ION.coupled_na_decay_prediction(f_prime, peak_val, dK0_event)
    pred_frozen = float(1.0 - np.exp(20.0 * Jm[0, 0]))

    tau_ko_wp = 1.0 / abs(Jm[1, 1])
    lin_wp = float(sum(np.exp(-0.2 * k / tau_ko_wp) for k in range(5)))
    numerically_valid = bool(np.all(np.isfinite(Ke)) and np.all(np.isfinite(Nae))
                             and np.all(np.isfinite(Kec)) and np.all(np.isfinite(Naec))
                             and float(np.min(Ke)) > 0.0)

    st = 5      # keep every 5th ion block (2.5 ms) -- ample for tau_Ko = 0.57 s
    traces = dict(K_event=Ke[::st].astype(np.float32), K_control=Kc[::st].astype(np.float32),
                  na_excess_k_clamped=exc_c[::st].astype(np.float32),
                  pump_event=Pe[::st].astype(np.float32),
                  pump_control=Pc[::st].astype(np.float32),
                  pump_event_k_clamped=Pec[::st].astype(np.float32),
                  na_excess=exc[::st].astype(np.float32),
                  na_event=Nae[::st].astype(np.float32), na_control=Nac[::st].astype(np.float32),
                  cluster_K=np.asarray(cluster_trace, np.float32),
                  trace_stride_blocks=np.int32(st), dt_ion_ms=np.float64(DT_ION_MS),
                  n_kernel_blocks=np.int32(nk), i_peak=np.int32(i_peak), i_end=np.int32(i_end))
    return traces, dict(dK_peak_single_mM=dK_med, dK_peak_per_event_mM=peaks,
                sigma_rest_K_mM=sigma_rest, n_events_measured=len(peaks),
                k_returns_within_1sigma_3s=bool(k_resid_sigma <= ION.F_GATES["k_recovery_sigma"]),
                k_residual_after_3s_in_sigma=k_resid_sigma,
                k_event_peak_replay_mM=float(np.max(dK_ev)) if dK_ev.size else float("nan"),
                na_excess_peak_mM=peak_val, na_excess_at_20s_mM=float(exc[i_end]),
                na_excess_decay_frac_20s=decay,
                na_excess_monotone_nonincreasing=monotone,
                na_excess_measurable=measurable, na_excess_still_rising_at_20s=still_rising,
                na_analytic_decay_frac_20s=float(1.0 - np.exp(-20.0 / 54.4188)),
                na_n_participants=int(participants.size),
                integration_peaks_mM=[float(p) for p in cpeaks],
                integration_ratio_5th_over_1st=float(ratio),
                cluster_spacing_ms=200.0, cluster_n=5,
                # ---- T7.1 (user rulings 2026-07-28) ----
                na_net_decay_frac=net_decay,
                na_tail_slope=slope, na_tail_slope_se=se, na_tail_slope_t=tstat,
                na_decay_pred_coupled=pred_coupled,
                na_decay_pred_frozen_K=pred_frozen,
                na_decay_frac_k_clamped=decay_clamped,
                na_excess_peak_k_clamped_mM=pkc,
                dK0_event_mM=dK0_event,
                pump_mean_event_peak=float(np.max(Pe[:nk])),
                pump_mean_control=float(np.mean(Pc)),
                pump_mean_event_peak_k_clamped=float(np.max(Pec[:nk])),
                working_point_Na_star=float(Na_star), working_point_K_o_star=float(Ko_star),
                jacobian=[[float(v) for v in row] for row in Jm],
                tau_Ko_at_workpoint_s=float(tau_ko_wp),
                integration_linear_at_workpoint=lin_wp,
                matched_background_snr=float(dK_med / max(sigma_rest, 1e-12)),
                numerically_valid=numerically_valid,
                numerical_detail=(f"K_min replay {float(np.min(Ke)):.4f} mM; all replay traces "
                                  f"finite"))


def cmd_b1_select_f(args):
    """T7 on the REAL 40k substrate with REAL spontaneous interictal events (see the deviation note).

    The plan put T7 on a small network to avoid paying for 40k before Gate H.  Gate H has passed,
    so 40k is permitted, and the first small-network attempt failed as an INSTRUMENT, not as
    physics: its pre-equilibrium was built from a 3.28 Hz rate field while the protocol run went at
    5.48 Hz, so the entire 26 s was one monotonic Na transient and the decay came out identically
    0.000 for all three candidates by construction; and the kick recruited 2652 of 3200 E cells
    (83%), which is not the "single interictal event" the gates name.  That run is preserved under
    superseded/.  Running the same five pre-registered gates on the real substrate removes both
    defects: the rate field comes from the very trajectory being measured, and the events are the
    substrate's own spontaneous interictal events.
    """
    if not args.confirm_run:
        raise SystemExit("b1-select-f runs a 40k trajectory; pass --confirm-run")
    with staged("b1-select-f"):
        pre = preflight()
        gh = json.load(open(os.path.join(OUT, "gate_H.json")))
        if gh["status"] != "PASS":
            raise SystemExit(f"Gate H is {gh['status']}: T7 may not run (plan §8)")
        gate = check_resource_gate("b1-select-f")
        resource_log("b1_select_f_gate", gate)
        if gate["status"] == "PAUSE":
            raise SystemExit(f"resource gate PAUSE: {gate}")

        from src.snn_engine.ion_homeostasis import (                # noqa: E402
            IonHomeostasisConfig, build_from_rate_field)
        from mz_slow_vars import MZSlowVars, MZSlowVarsConfig       # noqa: E402
        import run_topic4_mz_slowvars as OLD                        # noqa: E402

        bar = float(load_artifact("pump_event_bar")["event_bar"])
        rE, rI, voxel, field_sha = _load_40k_rate_field()
        S, PP = _substrate(CONN_SEED_DEV)
        b_lo = _blk(BURN_IN_MS)
        b_hi = _blk(T_MS)

        def _cfg(fp, capture):
            return IonHomeostasisConfig(
                q_ion=ION.q_ion_from_fprime(fp), n_grid=N_GRID_40K, dx_mm=DX_MM_40K,
                dt_ion_ms=DT_ION_MS, g_K_ion=0.0,          # SENSOR mode
                k_trace_stride=2, capture_counts_range=(b_lo, b_hi) if capture else ())

        def _mk(fp, capture=False):
            return build_from_rate_field(S["N"], S["NE"], voxel, _cfg(fp, capture), rE, rI)

        instances = {fp: _mk(fp, capture=(fp == ION.F_PRIME_PRIMARY))
                     for fp in ION.F_PRIME_CANDIDATES}

        class _Fan:
            def __init__(self, mz, ions_list):
                object.__setattr__(self, "mz", mz)
                object.__setattr__(self, "_ions", ions_list)

            def __getattr__(self, name):
                if name in ("mz", "_ions"):
                    raise AttributeError(name)
                return getattr(self.mz, name)

            def membrane_terms(self, *a, **k):
                return self.mz.membrane_terms(*a, **k)     # sensor mode: membrane untouched

            def apply_currents(self, *a, **k):
                return self.mz.apply_currents(*a, **k)

            def step(self, spk, labels, dt):
                self.mz.step(spk, labels, dt)
                for io in self._ions:
                    io.accumulate(spk)
                    io.maybe_update(dt)

        mz = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**arm_c_pump_off_cfg()),
                        NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
        t0 = time.time()
        res, _ = run_arm_c(S, noise_seed=NOISE_DIRECTION_POWER, T_ms=T_MS,
                           slow=_Fan(mz, list(instances.values())), dump_i=False, verbose=True)
        print(f"[b1-select-f] sensor run {time.time()-t0:.0f}s  "
              f"mean_rate_E={float(np.asarray(res['rate_E'])[_blk(BURN_IN_MS)*0:].mean()):.3f} Hz",
              flush=True)
        resource_log("b1_select_f_sim_done")

        n_steps = res["E_spk_bool"].shape[0]
        events = []
        for lo, hi in block_edges(n_steps):
            for e in detect_events(res, bar, lo, hi):
                events.append(dict(e, t_on=e["t_on"] + lo * DT, t_off=e["t_off"] + lo * DT))
        print(f"[b1-select-f] {len(events)} self-terminating events on the real substrate",
              flush=True)

        cap = np.asarray(instances[ION.F_PRIME_PRIMARY].captured_counts)   # (blocks, N) int8
        ev_mask = np.zeros(cap.shape[0], bool)
        for e in events:
            ev_mask[max(0, _blk(e["t_on"] - 30.0) - b_lo):
                    min(cap.shape[0], _blk(e["t_off"] + 300.0) - b_lo)] = True
        bg = np.where(~ev_mask)[0]
        print(f"[b1-select-f] background blocks: {bg.size}/{cap.shape[0]}", flush=True)
        if bg.size < 2000:
            raise SystemExit("not enough event-free blocks to build the background replay")

        # typical event = the one whose K excursion is the median (primary f' as the reference)
        ref = instances[ION.F_PRIME_PRIMARY]
        Kref = np.asarray(ref.k_trace).reshape(len(ref.k_trace), -1)
        tK = np.asarray(ref.k_trace_blocks) * DT_ION_MS
        pk, hots = [], []
        for e in events:
            p = (tK >= e["t_on"] - 100.0) & (tK < e["t_on"] - 10.0)
            w = (tK >= e["t_on"]) & (tK < e["t_on"] + 250.0)
            if not (p.any() and w.any()):
                pk.append(-1.0), hots.append(0)
                continue
            d = Kref[w] - Kref[p].mean(axis=0)
            hots.append(int(np.argmax(d.max(axis=0))))
            pk.append(float(d.max()))
        order = np.argsort(pk)
        i_typ = int(order[len(order) // 2])
        typ, hot = events[i_typ], hots[i_typ]
        k0 = _blk(typ["t_on"] - 20.0) - b_lo
        kernel = cap[k0:k0 + int(round(200.0 / DT_ION_MS))]
        s0, s1 = int(round(typ["t_on"] / DT)), int(round((typ["t_off"] + 50.0) / DT))
        participants = np.where(res["E_spk_bool"][s0:s1].any(axis=0))[0]
        sigma_rest = float(np.median(Kref[np.isin(np.round(tK / DT_ION_MS).astype(int) - b_lo, bg)]
                                     .std(axis=0)))
        print(f"[b1-select-f] typical event t_on={typ['t_on']:.0f} ms hot voxel={hot} "
              f"participants={participants.size} sigma_rest={sigma_rest:.5f} mM", flush=True)

        rows, all_traces = [], {}
        for fp, io in instances.items():
            tr, m = _t7_measure_40k(lambda fp=fp: _mk(fp), io, cap, kernel, bg, events, hot,
                                    participants, sigma_rest, fp)
            all_traces.update({f"f{fp}_{k}": v for k, v in tr.items()})
            if not m["na_excess_measurable"]:
                ev = dict(admissible=False, gates=dict(recovery_Na=dict(
                    ok=False, reason="the event-induced Na excess never peaked inside the "
                                     "measurement window -- reported as UNMEASURABLE rather than "
                                     "as a decay of 0")), measured=m)
            else:
                ev = ION.evaluate_f_prime_gates(m)
            ev["f_prime"] = fp
            ev["q_ion"] = ION.q_ion_from_fprime(fp)
            rows.append(ev)
            print(f"[b1-select-f] f'={fp}: admissible={ev['admissible']}  "
                  f"dK={m['dK_peak_single_mM']:.4f}  K_rec={m['k_residual_after_3s_in_sigma']:.2f}sig "
                  f"Na_decay={m['na_excess_decay_frac_20s']:.3f} "
                  f"(analytic {m['na_analytic_decay_frac_20s']:.3f})  "
                  f"ratio={m['integration_ratio_5th_over_1st']:.3f}", flush=True)
            for name, gg in ev["gates"].items():
                if not gg["ok"]:
                    print(f"      FAIL {name}: {gg}", flush=True)
            resource_log(f"b1_select_f_{fp}")

        # The pre-registered T7 contract is an occupancy-matched SMALL network with the ion layer
        # LIVE (plan §9.1/§9.2).  This run changed the experimental object: 40k substrate,
        # g_K_ion = 0, sensor replay on a fixed raster.  The change is defensible, but a result
        # from a different object may NOT inherit the canonical verdict of the contract it
        # replaced, so `select_f_prime` (which still implements the original contract, untouched)
        # is wrapped and its label withheld.
        sel = ION.withhold_canonical_verdict(
            ION.select_f_prime(rows),
            protocol_deviation=(
                "registered on an occupancy-matched small network with the ion feedback live; "
                "executed on the 40k substrate in sensor mode (g_K_ion = 0) with an offline "
                "isolated-event replay. Network scale, event source, closed-vs-open loop and the "
                "statistical object of the gates all changed."),
            blocking_gates_are_open_loop=(
                "The integration gate is measured with K -> E_K -> firing deliberately cut, so it "
                "can only characterise the CLEARANCE side. Using it to block B2 -- the phase whose "
                "purpose is to test the closed loop -- would be self-sealing, and this artifact "
                "must not be read as licensing that block on mechanism grounds."))
        _write_npz(os.path.join(OUT, "b1_f_selection_traces.npz"), **all_traces)
        payload = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            status=sel["status"], selected_f_prime=sel["selected"],
            contract_verdict_if_protocol_had_matched=sel["contract_verdict_if_protocol_had_matched"],
            verdict_withheld_because=sel["verdict_withheld_because"],
            blocking_gates_measured_open_loop=sel["blocking_gates_measured_open_loop"],
            semantics=sel["semantics"], b2_entry=sel["b2_entry"],
            tie_break=sel.get("tie_break"),
            traces=os.path.join(OUT, "b1_f_selection_traces.npz"),
            rows=rows, gates_definition=ION.F_GATES,
            measurement_mode=dict(
                substrate="40k E1146-informed sheet (the real one)", g_K_ion=0.0,
                mode="sensor: the ion state is a pure function of the recorded raster",
                events="the substrate's own spontaneous self-terminating interictal events",
                rationale="the five gates ask what ONE interictal event does to the ions. Measuring "
                          "with the feedback live is circular (a larger f' changes the network, the "
                          "event and dK_o), and the gates' analytic references are PURE-RELAXATION "
                          "predictions that only exist for an isolated event -- which never occurs "
                          "on a substrate firing every ~455 ms. Hence: sensor mode plus an "
                          "isolated-event replay against a matched background-only control.",
                same_device_as="the accepted pump-sprint sensor-only load calibration"),
            deviation_from_plan=dict(
                planned="T7 on a small occupancy-matched network (plan §9.1/§9.2)",
                executed="T7 on the 40k substrate with real spontaneous events",
                reason="the small-network attempt failed as an instrument, not as physics: its "
                       "pre-equilibrium was built from a 3.28 Hz field while the protocol ran at "
                       "5.48 Hz, making the whole 26 s one monotonic Na transient (decay came out "
                       "identically 0.000 for all three candidates BY CONSTRUCTION), and the kick "
                       "recruited 83% of the E population. Gate H had already passed, so 40k was "
                       "permitted.",
                superseded_evidence="superseded/b1_f_selection_smallnet_instrument_failure.json",
                precedent="the pump sprint likewise found an event-detection instrument bug, fixed "
                          "it, re-ran once and kept the first run as evidence"),
            protocol=dict(N=S["N"], n_grid=N_GRID_40K, dx_mm=DX_MM_40K, T_ms=T_MS,
                          noise_seed=NOISE_DIRECTION_POWER, conn_seed=CONN_SEED_DEV,
                          dt=DT, dt_ion_ms=DT_ION_MS, event_bar=bar,
                          n_events=len(events), n_background_blocks=int(bg.size),
                          typical_event_t_on_ms=typ["t_on"], hot_voxel=hot,
                          n_participants=int(participants.size), sigma_rest_mM=sigma_rest,
                          rate_field_sha256=field_sha),
            small_network_caveat=gh["small_network_caveat"])
        # ---------------- T7.1: re-adjudicate the SAME measurements under the repaired gates ----
        rows_v2 = []
        for r in rows:
            e2 = ION.evaluate_f_prime_gates_v2(r["measured"])
            e2["f_prime"] = r["f_prime"]
            e2["q_ion"] = r["q_ion"]
            rows_v2.append(e2)
            m = r["measured"]
            print(f"[T7.1] f'={r['f_prime']}: admissible={e2['admissible']}  "
                  f"dK={m['dK_peak_single_mM']:.4f} (floor 0.15, ceiling 0.90)  "
                  f"net_decay={m['na_net_decay_frac']:.3f} "
                  f"(coupled pred {m['na_decay_pred_coupled']:.3f}, "
                  f"K-clamped {m['na_decay_frac_k_clamped']:.3f})  "
                  f"tail_slope_t={m['na_tail_slope_t']:+.2f}  "
                  f"ratio={m['integration_ratio_5th_over_1st']:.3f} "
                  f"(linear@wp {m['integration_linear_at_workpoint']:.3f}, non-blocking)",
                  flush=True)
            for name, gg in e2["gates"].items():
                if not gg["ok"]:
                    print(f"      HARD-GATE FAIL {name}: {gg}", flush=True)
        sel2 = ION.select_f_prime_v2(rows_v2)
        _write_json(os.path.join(OUT, "b1_f_selection_v2.json"), dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            contract="T7.1 (docs/superpowers/specs/2026-07-28-topic4-fcxr-ion-T7_1-lock.md)",
            status=sel2["status"], selected_f_prime=sel2["selected"],
            semantics=sel2.get("semantics"), tie_break=sel2.get("tie_break"),
            rows=rows_v2, hard_gates=ION.F_GATES_V2,
            rulings=dict(
                r1="5*sigma_rest removed; absolute 0.15 mM floor only; background is diagnostic",
                r2="Na reference from the COUPLED working-point Jacobian; K-clamp control at "
                   "K_o*(f'), not at rest",
                r3="net decay from peak to 20 s + tail-window trend; per-sample and smoothed "
                   "monotonicity both dropped",
                r4="integration is NON-BLOCKING and becomes a B2 pre-condition risk register",
                r5="the small-network contract is abolished for dynamic f' selection; its "
                   "numerical tests are retained"),
            b2_risk_register=dict(
                item="open-loop K accumulation across 200 ms-spaced events is sub-linear",
                measured={r["f_prime"]: r["measured"]["integration_ratio_5th_over_1st"]
                          for r in rows},
                linear_at_workpoint={r["f_prime"]: r["measured"]["integration_linear_at_workpoint"]
                                     for r in rows},
                b2_must_report="whether closed-loop K -> E_K -> firing overcomes this "
                               "load-dependent clearance",
                blocks_b2_entry=False),
            small_net_contract=dict(retained=list(ION.SMALL_NET_CONTRACT_RETAINED),
                                    abolished=list(ION.SMALL_NET_CONTRACT_ABOLISHED)),
            protocol=payload["protocol"], measurement_mode=payload["measurement_mode"],
            traces=os.path.join(OUT, "b1_f_selection_traces.npz")))
        print(f"[T7.1] {sel2['status']}  selected f' = {sel2['selected']}")

        _write_json(os.path.join(OUT, "b1_f_selection.json"), payload)
        print(f"[b1-select-f] {sel['status']}  selected f' = {sel['selected']}  "
              f"(contract label had the protocol matched: "
              f"{sel['contract_verdict_if_protocol_had_matched']})")
        if sel2["selected"] is None:
            _write_json(os.path.join(OUT, "NO_ADMISSIBLE_SCALE.json"), payload)
            raise SystemExit(
                "T7.1 = NO_ADMISSIBLE_SCALE -- no candidate passed the four repaired hard gates, "
                "so B2 stays closed. Still NOT a mechanism NO-GO.")
    return 0


# ================================================================== T8: 40k bias calibration
BIAS_BOUNDS = (-2.0, 2.0)      # engine units = mV; V_th = 18, so +-11% of threshold (plan §10)
BIAS_DELTA = 0.5               # finite-difference step
PROBE_T_MS = 4000.0          # overridable: --probe-t-ms (see cmd_b2_bias docstring)
PROBE_BURN_MS = 1000.0
MAX_PROBES = 12
R_E_TARGET = ION.R0_HZ
R_I_TARGET = 10.367974281311035   # accepted arm-C pump-off I rate, reproduced bit-exactly in T2
CLOSURE_REL_TOL = 0.10


def _load_40k_rate_field():
    z = np.load(os.path.join(OUT, "b0_baseline_rate_field.npz"))
    return (np.asarray(z["rate_E"], float), np.asarray(z["rate_I"], float),
            np.asarray(z["cell_voxel"], np.int32),
            _sha(os.path.join(OUT, "b0_baseline_rate_field.npz")))


def probe_task(job):
    """One 40k probe at a fixed (I_bias_E, I_bias_I).  Runs in its own process.

    The initial ion state is the HETEROGENEOUS pre-equilibrium of the T2 rate field, scaled by the
    predicted rate response at this bias (spec §4.2c).  A 4 s probe estimates the FAST rate
    response only -- it cannot and does not claim that Na has reached steady state (tau_Na=54.4 s).
    """
    m = sys.modules[__name__]          # under spawn this is __mp_main__, not a third copy
    from src.snn_engine.ion_homeostasis import (                    # noqa: E402
        IonHomeostasisConfig, IonHomeostaticMZAdapter, build_from_rate_field)
    from mz_slow_vars import MZSlowVars, MZSlowVarsConfig           # noqa: E402
    import run_topic4_mz_slowvars as OLD                            # noqa: E402

    t0 = time.time()
    rE, rI, voxel, sha = m._load_40k_rate_field()
    rE = rE * float(job["rate_scale_E"])
    rI = rI * float(job["rate_scale_I"])
    S, _PP = m._substrate(int(job["conn_seed"]))
    cfg = IonHomeostasisConfig(q_ion=float(job["q_ion"]), n_grid=m.N_GRID_40K,
                               dx_mm=m.DX_MM_40K, dt_ion_ms=m.DT_ION_MS,
                               g_K_ion=ION.G_K_ION_REFERENCE,
                               I_bias_E=float(job["I_bias_E"]), I_bias_I=float(job["I_bias_I"]))
    ions, rep = build_from_rate_field(S["N"], S["NE"], voxel, cfg, rE, rI, return_report=True)
    Na0, K0 = ions.Na_i_all.copy(), ions.K_o_grid.copy()
    mz = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**m.arm_c_pump_off_cfg()),
                    NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
    try:
        res, _ = m.run_arm_c(S, noise_seed=int(job["noise_seed"]), T_ms=float(job["T_ms"]),
                             slow=IonHomeostaticMZAdapter(mz, ions), dump_i=False, verbose=False)
    except FloatingPointError as exc:
        # A bias probe CAN destabilise the substrate; that is information, not a crash. Report it
        # so the bounded Newton can avoid the region and the artifact records where it happened.
        return dict(job=job, r_E=float("nan"), r_I=float("nan"),
                    failed=f"{type(exc).__name__}: {exc}",
                    Na_mean_start=float(Na0.mean()), K_mean_start=float(K0.mean()),
                    Na_max_end=float(ions.Na_i_all.max()), K_max_end=float(ions.K_o_grid.max()),
                    n_ion_updates=int(ions.n_updates), rate_field_sha256=sha,
                    wall_s=round(time.time() - t0, 1))
    lo = int(round(m.PROBE_BURN_MS / m.DT))
    if lo >= np.asarray(res["rate_E"]).size:
        raise ValueError(f"probe T_ms={job['T_ms']} does not exceed the {m.PROBE_BURN_MS} ms "
                         f"burn-in: the measurement window would be empty and the rate would come "
                         f"back as NaN")
    out = dict(job=job,
               r_E=float(np.asarray(res["rate_E"], float)[lo:].mean()),
               r_I=float(np.asarray(res["rate_I"], float)[lo:].mean()),
               Na_mean_start=float(Na0.mean()), Na_mean_end=float(ions.Na_i_all.mean()),
               K_mean_start=float(K0.mean()), K_mean_end=float(ions.K_o_grid.mean()),
               K_max_end=float(ions.K_o_grid.max()), Na_max_end=float(ions.Na_i_all.max()),
               pump_mean_end=float(ions.pump_flux_all.mean()),
               init_residual_q99_dNa=rep["q99_abs_dNa_dt"],
               init_residual_q99_dKo=rep["q99_abs_dKo_dt"],
               rate_field_sha256=sha, wall_s=round(time.time() - t0, 1))
    # per-cell counts over the SAME post-burn-in window, for the B2.1 self-consistent loop
    lo0, hi0 = blocks[0][0], blocks[-1][1]
    out["per_cell_counts_E"] = res["E_spk_bool"][lo0:hi0].sum(axis=0).astype(int).tolist()
    out["per_cell_counts_I"] = res["I_spk_bool"][lo0:hi0].sum(axis=0).astype(int).tolist()
    return out


def _run_probes(jobs, workers):
    import multiprocessing as mp
    if len(jobs) == 1:
        return [probe_task(jobs[0])]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(jobs))) as pool:
        return pool.map(probe_task, jobs)


def cmd_b2_bias(args):
    if not args.confirm_run:
        raise SystemExit("b2-bias runs 40k probes; pass --confirm-run")
    with staged("b2-bias"):
        pre = preflight()
        gh = json.load(open(os.path.join(OUT, "gate_H.json")))
        fs = json.load(open(os.path.join(OUT, "b1_f_selection_v2.json")))
        if gh["status"] != "PASS":
            raise SystemExit(f"Gate H is {gh['status']}: no 40k stage may run")
        if fs["status"] != "PROVISIONAL_CANDIDATE":
            raise SystemExit(f"T7.1 selection is {fs['status']}: B2 may not run")
        gate = check_resource_gate("b2-bias")
        resource_log("b2_bias_gate", gate)
        if gate["status"] == "PAUSE":
            raise SystemExit(f"resource gate PAUSE: {gate}")

        fp = float(fs["selected_f_prime"])
        q_ion = ION.q_ion_from_fprime(fp)
        plan = plan_workers(args.workers or 4)
        print(f"[b2-bias] f'={fp} q_ion={q_ion:.5f}  workers={plan['workers']} "
              f"(siblings={plan['siblings']}, MemAvailable={plan['mem_available_gb']} GB)",
              flush=True)

        probe_T = float(args.probe_t_ms or PROBE_T_MS)

        def job(be, bi, sE=1.0, sI=1.0, tag=""):
            return dict(I_bias_E=be, I_bias_I=bi, q_ion=q_ion, conn_seed=CONN_SEED_DEV,
                        noise_seed=NOISE_DEV, T_ms=probe_T, rate_scale_E=sE, rate_scale_I=sI,
                        tag=tag)

        probes, history = [], []
        # ---- 4 common-noise probes: base + two single-axis steps + a linearity check ----
        d = BIAS_DELTA
        batch = [job(0.0, 0.0, tag="base"), job(d, 0.0, tag="dE"), job(0.0, d, tag="dI"),
                 job(d, d, tag="dEdI")]
        res = _run_probes(batch, plan["workers"])
        probes += res
        resource_log("b2_bias_jacobian_done")
        base, pE, pI, pEI = res
        bad = [p["job"]["tag"] for p in res if not np.isfinite(p["r_E"])]
        if bad:
            payload = dict(status="UNRESOLVED_CALIBRATION", reason=
                           f"probe(s) {bad} destabilised the substrate (ion guard band); the "
                           f"Jacobian cannot be estimated. This is a calibrator failure and a "
                           f"substrate-stability finding, NOT a mechanism NO-GO.",
                           probes=res, generated=datetime.now(timezone.utc).isoformat(),
                           f_prime=fp, q_ion=q_ion, bias_bounds=list(BIAS_BOUNDS))
            _write_json(os.path.join(OUT, "b2_bias_calibration.json"), payload)
            raise SystemExit(payload["reason"])
        J = np.array([[(pE["r_E"] - base["r_E"]) / d, (pI["r_E"] - base["r_E"]) / d],
                      [(pE["r_I"] - base["r_I"]) / d, (pI["r_I"] - base["r_I"]) / d]])
        lin_pred = np.array([base["r_E"], base["r_I"]]) + J @ np.array([d, d])
        lin_err = float(np.max(np.abs(lin_pred - np.array([pEI["r_E"], pEI["r_I"]]))))
        print(f"[b2-bias] base r_E={base['r_E']:.3f} r_I={base['r_I']:.3f} | target "
              f"{R_E_TARGET:.3f}/{R_I_TARGET:.3f}\n          J=\n{J}\n          "
              f"linearity check |err| = {lin_err:.3f} Hz", flush=True)
        history.append(dict(stage="jacobian", J=J.tolist(), linearity_max_err_hz=lin_err))

        target = np.array([R_E_TARGET, R_I_TARGET])
        cur = np.array([0.0, 0.0])
        cur_rates = np.array([base["r_E"], base["r_I"]])
        verdict, best = None, base
        for rnd in range(2):
            resid = target - cur_rates
            try:
                step = np.linalg.solve(J, resid)
            except np.linalg.LinAlgError:
                verdict = "UNRESOLVED_CALIBRATION"
                history.append(dict(stage=f"round{rnd}", error="singular Jacobian"))
                break
            trust = 1.5
            if np.max(np.abs(step)) > trust:                 # trust-region clip
                step = step * trust / np.max(np.abs(step))
            nxt = np.clip(cur + step, *BIAS_BOUNDS)
            hit_bound = bool(np.any(np.isclose(nxt, BIAS_BOUNDS[0]))
                             or np.any(np.isclose(nxt, BIAS_BOUNDS[1])))
            pred = cur_rates + J @ (nxt - cur)
            sE = float(np.clip(pred[0] / R_E_TARGET, 0.2, 5.0))
            sI = float(np.clip(pred[1] / R_I_TARGET, 0.2, 5.0))
            jobs = [job(float(nxt[0]), float(nxt[1]), sE, sI, tag=f"round{rnd}")]
            if rnd == 0:                                     # second probe of the round: a bracket
                alt = np.clip(cur + 0.5 * step, *BIAS_BOUNDS)
                predA = cur_rates + J @ (alt - cur)
                jobs.append(job(float(alt[0]), float(alt[1]),
                                float(np.clip(predA[0] / R_E_TARGET, 0.2, 5.0)),
                                float(np.clip(predA[1] / R_I_TARGET, 0.2, 5.0)),
                                tag=f"round{rnd}_half"))
            got = _run_probes(jobs, plan["workers"])
            probes += got
            resource_log(f"b2_bias_round{rnd}_done")
            for g in got:
                err = max(abs(g["r_E"] - R_E_TARGET) / R_E_TARGET,
                          abs(g["r_I"] - R_I_TARGET) / R_I_TARGET)
                print(f"[b2-bias] round{rnd} bias=({g['job']['I_bias_E']:+.3f},"
                      f"{g['job']['I_bias_I']:+.3f}) -> r_E={g['r_E']:.3f} r_I={g['r_I']:.3f} "
                      f"rel_err={err:.3f}", flush=True)
            best = min(probes, key=lambda p: max(abs(p["r_E"] - R_E_TARGET) / R_E_TARGET,
                                                 abs(p["r_I"] - R_I_TARGET) / R_I_TARGET))
            history.append(dict(stage=f"round{rnd}", bias=[float(nxt[0]), float(nxt[1])],
                                probes=[dict(bias=[g["job"]["I_bias_E"], g["job"]["I_bias_I"]],
                                             r_E=g["r_E"], r_I=g["r_I"]) for g in got],
                                hit_bound=hit_bound))
            b0 = got[0]
            if max(abs(b0["r_E"] - R_E_TARGET) / R_E_TARGET,
                   abs(b0["r_I"] - R_I_TARGET) / R_I_TARGET) <= CLOSURE_REL_TOL:
                verdict = "CALIBRATED"
                best = b0
                break
            # re-anchor for the next round using the measured point
            cur = np.array([b0["job"]["I_bias_E"], b0["job"]["I_bias_I"]])
            cur_rates = np.array([b0["r_E"], b0["r_I"]])
            if len(probes) >= MAX_PROBES:
                break

        # Contract (spec 7.1): the closure hard gate is |r0' - r0|/r0 on the E rate. r_I is
        # recorded and handed to Gate B as a check item (spec 7.1 line 207), NOT used as a gate
        # here. max(E, I) stays the SEARCH objective -- it finds a better joint point -- but a
        # verdict stricter than the contract would wrongly read as a failure.
        best_err_E = abs(best["r_E"] - R_E_TARGET) / R_E_TARGET
        best_err_I = abs(best["r_I"] - R_I_TARGET) / R_I_TARGET
        best_err = max(best_err_E, best_err_I)
        if verdict is None:
            verdict = "CALIBRATED" if best_err_E <= CLOSURE_REL_TOL else "UNRESOLVED_CALIBRATION"
        # NO_GO_BASELINE requires the legal box to be BRACKETED and shown to contain no solution.
        # A probe budget that ran out is a calibrator failure, never a mechanism conclusion.
        bracketed = False
        payload = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            status=verdict, f_prime=fp, q_ion=q_ion,
            targets=dict(r_E_hz=R_E_TARGET, r_I_hz=R_I_TARGET, rel_tol=CLOSURE_REL_TOL),
            best=dict(I_bias_E=best["job"]["I_bias_E"], I_bias_I=best["job"]["I_bias_I"],
                      r_E=best["r_E"], r_I=best["r_I"], rel_err=best_err,
                      rel_err_E=best_err_E, rel_err_I=best_err_I),
            gating_rule=("spec 7.1: the closure hard gate is the E-rate residual "
                         "|r0'-r0|/r0 <= 10%. r_I is recorded and handed to Gate B as a check "
                         "item (spec 7.1), not gated here. max(E,I) remains the search objective."),
            jacobian=J.tolist(), linearity_max_err_hz=lin_err,
            n_probes=len(probes), max_probes=MAX_PROBES, probes=probes, history=history,
            bias_bounds=list(BIAS_BOUNDS), bias_delta=BIAS_DELTA,
            legal_box_bracketed=bracketed,
            nuisance_parameters=["I_bias_E", "I_bias_I"],
            verdict_semantics=dict(
                CALIBRATED="the two biases recovered the accepted rates within 10%",
                UNRESOLVED_CALIBRATION="the calibrator did not converge inside its probe budget; "
                                       "the feasible region was NOT excluded, so this is NOT a "
                                       "mechanism conclusion",
                NO_GO_BASELINE="only when the legal box is fully bracketed AND contains no "
                               "solution"),
            probe_T_ms=probe_T,
            probe_caveat=("A probe shorter than the validation window estimates the FAST rate "
                          "response only. tau_Na = 54.4 s, so no probe here demonstrates that Na "
                          "has reached steady state; even the 11 s validation only shows stability "
                          "near the initialization point. EXECUTION FINDING (2026-07-28): with the "
                          "plan's 4 s probes the calibrator converged to 4.054 Hz but the 11 s "
                          "validation at the SAME bias came back at 5.159 Hz, so the closure gate "
                          "failed at 33%. The probe window must match the validation window; "
                          "probe_T_ms records which was used."),
            workers=plan)
        _write_json(os.path.join(OUT, "b2_bias_calibration.json"), payload)
        print(f"[b2-bias] {verdict}  best bias=({best['job']['I_bias_E']:+.3f},"
              f"{best['job']['I_bias_I']:+.3f}) r_E={best['r_E']:.3f} r_I={best['r_I']:.3f} "
              f"rel_err={best_err:.3f}  probes={len(probes)}")
        if verdict != "CALIBRATED":
            raise SystemExit(f"b2-bias = {verdict}: archive and stop (plan §10/§15)")
    return 0


# ================================================================== T9: full 11 s trajectory
K_TRACE_STRIDE = 20            # ion blocks -> 10 ms K_o frames
NA_SNAPSHOT_MS = 100.0         # B2.1 spec 2.1: dense Na sampling for the signed-slope fit
FAR_FIELD_MM = 5.0             # "far from the event" for the whole-sheet K-wave check


def trajectory_task(job):
    """One full 11 s trajectory with the ion layer live, at the FROZEN bias and q_ion.

    Returns the accepted block metrics, the initiation-site readout and the ion diagnostics.
    Used for both the development validation run and the six confirmatory runs.
    """
    m = sys.modules[__name__]          # under spawn this is __mp_main__, not a third copy
    from src.snn_engine.ion_homeostasis import (                    # noqa: E402
        IonHomeostasisConfig, IonHomeostaticMZAdapter, build_from_rate_field)
    from mz_slow_vars import MZSlowVars, MZSlowVarsConfig           # noqa: E402
    import run_topic4_mz_slowvars as OLD                            # noqa: E402

    t0 = time.time()
    rE, rI, voxel, sha = m._load_40k_rate_field()
    ov = job.get("rate_field_override")
    if ov is not None:               # B2.1: a self-consistent ions-ON field replaces the scaling
        rE = np.asarray(ov["rate_E"], float)
        rI = np.asarray(ov["rate_I"], float)
    else:
        rE = rE * float(job["rate_scale_E"])
        rI = rI * float(job["rate_scale_I"])
    S, PP = m._substrate(int(job["conn_seed"]))
    n_steps = int(round(float(job["T_ms"]) / m.DT))
    blocks = m.block_edges(n_steps)
    b_lo = int(round(blocks[0][0] * m.DT / m.DT_ION_MS))
    b_hi = int(round(blocks[-1][1] * m.DT / m.DT_ION_MS)) - 1
    na_stride = int(round(m.NA_SNAPSHOT_MS / m.DT_ION_MS))     # B2.1 spec 2.1: dense sampling
    na_blocks = tuple(range(b_lo, b_hi + 1, na_stride))
    cfg = IonHomeostasisConfig(q_ion=float(job["q_ion"]), n_grid=m.N_GRID_40K, dx_mm=m.DX_MM_40K,
                               dt_ion_ms=m.DT_ION_MS, g_K_ion=ION.G_K_ION_REFERENCE,
                               I_bias_E=float(job["I_bias_E"]), I_bias_I=float(job["I_bias_I"]),
                               k_trace_stride=m.K_TRACE_STRIDE, na_snapshot_blocks=na_blocks)
    ions, rep = build_from_rate_field(S["N"], S["NE"], voxel, cfg, rE, rI, return_report=True)
    mz = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**m.arm_c_pump_off_cfg()),
                    NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
    res, _ = m.run_arm_c(S, noise_seed=int(job["noise_seed"]), T_ms=float(job["T_ms"]),
                         slow=IonHomeostaticMZAdapter(mz, ions), dump_i=True, verbose=False)

    posE = np.asarray(S["posE"], float)
    A, B = np.asarray(S["src_xy"], float), np.asarray(S["snk_xy"], float)
    srcm = np.linalg.norm(posE - A, axis=1) <= PP.CORE_R
    snkm = np.linalg.norm(posE - B, axis=1) <= PP.CORE_R
    offm = ~(srcm | snkm)
    rate = np.asarray(res["rate_E"], float)
    bar = float(job["event_bar"])

    blk, all_ev = [], []
    for lo, hi in blocks:
        ret = m.detect_events(res, bar, lo, hi)
        spk = res["E_spk_bool"][lo:hi]
        onsets = np.array([e["t_on"] for e in ret], float)
        iei = np.diff(onsets) if onsets.size >= 2 else np.array([])
        tot = max(1, int(spk.sum()))
        win_s = (hi - lo) * m.DT / 1000.0
        init = ION.initiation_site_readout(spk, posE, A, B, ret, dt=m.DT, core_r=PP.CORE_R)
        blk.append(dict(
            ied_rate_hz=len(ret) / win_s,
            iei_median_ms=float(np.median(iei)) if iei.size else float("nan"),
            iei_cv=float(np.std(iei) / np.mean(iei)) if iei.size and np.mean(iei) > 0 else float("nan"),
            duration_median_ms=float(np.median([e["dur_ms"] for e in ret])) if ret else float("nan"),
            participation_median=float(np.median([e["peak_ext"] for e in ret])) if ret else float("nan"),
            mean_rate_hz=float(rate[lo:hi].mean()), n_events=len(ret),
            source_spike_share=float(spk[:, srcm].sum()) / tot,
            sink_spike_share=float(spk[:, snkm].sum()) / tot,
            offaxis_spike_share=float(spk[:, offm].sum()) / tot,
            n_scoreable=init["n_scoreable"], n_A=init["n_A"], n_B=init["n_B"],
            n_ambiguous=init["n_ambiguous"]))
        all_ev += [dict(e, _lo=lo) for e in ret]

    n_sc = sum(b["n_scoreable"] for b in blk)
    nA = sum(b["n_A"] for b in blk)
    nB = sum(b["n_B"] for b in blk)
    pooled = {k: float(np.nanmean([b[k] for b in blk]))
              for k in ("ied_rate_hz", "iei_median_ms", "iei_cv", "duration_median_ms",
                        "participation_median", "mean_rate_hz", "n_events",
                        "source_spike_share", "sink_spike_share", "offaxis_spike_share")}
    pooled.update(n_scoreable=n_sc, n_A=nA, n_B=nB,
                  frac_A=nA / max(n_sc, 1), frac_B=nB / max(n_sc, 1),
                  n_ambiguous=sum(b["n_ambiguous"] for b in blk))

    # ---- ion diagnostics ----
    snaps = sorted(ions.na_snapshots)
    Na = np.stack([ions.na_snapshots[b] for b in snaps])
    dt_blk = (snaps[1] - snaps[0]) * m.DT_ION_MS * 1e-3
    dNa = np.abs(np.diff(Na, axis=0)) / dt_blk
    slope_Na = ION.slope_stats(ION.signed_secular_slope(Na, dt_blk))
    Kt = np.asarray(ions.k_trace)
    kb = np.asarray(ions.k_trace_blocks) * m.DT_ION_MS
    keep = kb >= m.BURN_IN_MS
    Kp = Kt.reshape(Kt.shape[0], -1)[keep]
    dK = np.abs(np.diff(Kp[::int(max(1, Kp.shape[0] // 6))], axis=0)) / \
        (float(job["T_ms"]) * 1e-3 / 6.0)
    slope_K = ION.slope_stats(ION.signed_secular_slope(Kp, m.K_TRACE_STRIDE * m.DT_ION_MS * 1e-3))

    # whole-sheet K wave check: for the largest event, event voxel vs far field
    ng = m.N_GRID_40K
    gx, gy = np.meshgrid(np.arange(ng), np.arange(ng))
    cen = np.stack([(gx.ravel() + 0.5) * m.DX_MM_40K, (gy.ravel() + 0.5) * m.DX_MM_40K], axis=1)
    base_K = Kp[:max(1, int(0.1 * Kp.shape[0]))].mean(axis=0)
    exc = Kp - base_K
    hot = int(np.argmax(exc.max(axis=0)))
    far = np.linalg.norm(cen - cen[hot], axis=1) > m.FAR_FIELD_MM
    peak_hot = float(exc[:, hot].max())
    peak_far = float(exc[:, far].max()) if far.any() else 0.0

    out = dict(job=job, pooled=pooled, block_metrics=blk,
               n_events_total=len(all_ev),
               ion=dict(
                   # ---- B2.1 GATE quantity: signed secular slope (spec 2.1) ----
                   slope_Na=slope_Na, slope_K=slope_K,
                   slope_q99_Na=slope_Na["q99_abs"], slope_q99_K=slope_K["q99_abs"],
                   slope_bound_Na=ION.B2_1_SLOPE_BOUND_NA, slope_bound_K=ION.B2_1_SLOPE_BOUND_K,
                   n_na_snapshots=int(Na.shape[0]), na_snapshot_dt_s=float(dt_blk),
                   # ---- superseded first-difference statistic, kept as a DIAGNOSTIC only ----
                   q95_abs_dNa_dt=float(np.quantile(dNa, 0.95)),
                   q99_abs_dNa_dt=float(np.quantile(dNa, 0.99)),
                   max_abs_dNa_dt=float(dNa.max()),
                   mean_abs_dNa_dt=float(dNa.mean()),
                   q95_abs_dKo_dt=float(np.quantile(dK, 0.95)),
                   q99_abs_dKo_dt=float(np.quantile(dK, 0.99)),
                   max_abs_dKo_dt=float(dK.max()),
                   Na_mean_first=float(Na[0].mean()), Na_mean_last=float(Na[-1].mean()),
                   Na_max=float(Na.max()), K_mean=float(Kp.mean()), K_max=float(Kp.max()),
                   K_min=float(Kp.min()),
                   pump_mean=float(ions.pump_flux_all.mean()),
                   pump_saturation_frac_of_rho=float(ions.pump_flux_all.mean() / ION.RHO),
                   pump_max_frac_of_rho=float(ions.pump_flux_all.max() / ION.RHO),
                   k_wave_peak_event_voxel_mM=peak_hot,
                   k_wave_peak_far_field_mM=peak_far,
                   k_wave_far_over_event=(peak_far / peak_hot) if peak_hot > 0 else float("nan"),
                   k_wave_far_over_event_is_confounded=True,
                   k_wave_caveat=(
                       "NOT a valid whole-sheet-K-wave test while the ion state is still drifting. "
                       "peak_far is the max over ALL far voxels across the WHOLE post-burn-in "
                       "window against an early-window baseline, so a global upward drift alone "
                       "drives it toward 1.0 with no wave present. Answering spec 11 stop "
                       "condition 3 needs an EVENT-LOCKED comparison: dK at the event voxel vs far "
                       "voxels in the same short window around that event, each against its own "
                       "pre-event baseline. Not implemented; the stop condition is therefore "
                       "UNANSWERED, not triggered."),
                   far_field_mm=m.FAR_FIELD_MM,
                   init_residual_q99_dNa=rep["q99_abs_dNa_dt"],
                   init_residual_q99_dKo=rep["q99_abs_dKo_dt"]),
               rate_field_sha256=sha, wall_s=round(time.time() - t0, 1))
    # per-cell counts over the SAME post-burn-in window, for the B2.1 self-consistent loop
    lo0, hi0 = blocks[0][0], blocks[-1][1]
    out["per_cell_counts_E"] = res["E_spk_bool"][lo0:hi0].sum(axis=0).astype(int).tolist()
    out["per_cell_counts_I"] = res["I_spk_bool"][lo0:hi0].sum(axis=0).astype(int).tolist()
    return out


CL_PROBE_T_MS = 4000.0          # overridable: --probe-t-ms (see cmd_b2_bias docstring)
CL_PROBE_T_KICK_MS = 2500.0
CL_PROBE_SPACING_MS = 200.0
CL_PROBE_KICK_BOOST = 3.0


def closed_loop_integration_probe(job):
    """Ruling 4: B2 must REPORT whether the closed loop overcomes the load-dependent clearance.

    Matched counterpart to the open-loop integration diagnostic: two identical kicks 200 ms apart
    on the 40k substrate with the ion layer LIVE at the frozen bias.  The open-loop diagnostic's
    2nd/1st ratio is the comparison point; the linear-superposition reference uses tau_Ko at the
    same working point.  Two kicks, not five, because the blessed engine exposes exactly two kick
    times and it may not be modified.
    """
    m = sys.modules[__name__]
    from src.snn_engine.ion_homeostasis import (                    # noqa: E402
        IonHomeostasisConfig, IonHomeostaticMZAdapter, build_from_rate_field)
    from mz_slow_vars import MZSlowVars, MZSlowVarsConfig           # noqa: E402
    from kick_probe import simulate_kick                            # noqa: E402
    import run_m4_phaseplane as PP                                  # noqa: E402
    import run_topic4_mz_slowvars as OLD                            # noqa: E402

    t0 = time.time()
    rE, rI, voxel, sha = m._load_40k_rate_field()
    ov = job.get("rate_field_override")
    if ov is not None:
        rE, rI = np.asarray(ov["rate_E"], float), np.asarray(ov["rate_I"], float)
    else:
        rE, rI = rE * float(job["rate_scale_E"]), rI * float(job["rate_scale_I"])
    S, _ = m._substrate(int(job["conn_seed"]))
    cfg = IonHomeostasisConfig(q_ion=float(job["q_ion"]), n_grid=m.N_GRID_40K, dx_mm=m.DX_MM_40K,
                               dt_ion_ms=m.DT_ION_MS, g_K_ion=ION.G_K_ION_REFERENCE,
                               I_bias_E=float(job["I_bias_E"]), I_bias_I=float(job["I_bias_I"]),
                               k_trace_stride=2,
                               freeze_membrane_from_block=int(job.get(
                                   "freeze_membrane_from_block", 0)))
    ions = build_from_rate_field(S["N"], S["NE"], voxel, cfg, rE, rI)
    mz = MZSlowVars(S["N"], 18.0, MZSlowVarsConfig(**m.arm_c_pump_off_cfg()),
                    NE=S["NE"], core_mask_E=OLD.build_core_masks(S))
    p = dataclasses.replace(S["p"], T=m.CL_PROBE_T_MS, dt=m.DT)
    S["net"]["rng"] = np.random.default_rng(int(job["noise_seed"]))
    res = simulate_kick(p, S["net"], m.CL_PROBE_KICK_BOOST,
                        slow=IonHomeostaticMZAdapter(mz, ions), kick_center=list(S["src_xy"]),
                        r_kick=PP.R_KICK, t_kick=m.CL_PROBE_T_KICK_MS,
                        t_kick2=m.CL_PROBE_T_KICK_MS + m.CL_PROBE_SPACING_MS,
                        KICK_BOOST2=m.CL_PROBE_KICK_BOOST,
                        V_th_per_neuron=S["vth"], early_stop_runaway=False, dump_i_spikes=False)
    # drive readout: without it the two arms cannot be shown to be comparable (spec 2.3)
    spk = res["E_spk_bool"]
    kick_spikes, kick_parts = [], []
    for j in range(2):
        a = int(round((m.CL_PROBE_T_KICK_MS + j * m.CL_PROBE_SPACING_MS) / m.DT))
        b = int(round((m.CL_PROBE_T_KICK_MS + (j + 1) * m.CL_PROBE_SPACING_MS) / m.DT))
        seg = spk[a:b]
        kick_spikes.append(int(seg.sum()))
        kick_parts.append(int(seg.any(axis=0).sum()))

    K = np.asarray(ions.k_trace).reshape(len(ions.k_trace), -1)
    t = np.asarray(ions.k_trace_blocks) * m.DT_ION_MS
    base = K[(t >= m.CL_PROBE_T_KICK_MS - 300.0) & (t < m.CL_PROBE_T_KICK_MS - 20.0)].mean(axis=0)
    w1 = (t >= m.CL_PROBE_T_KICK_MS) & (t < m.CL_PROBE_T_KICK_MS + m.CL_PROBE_SPACING_MS)
    w2 = (t >= m.CL_PROBE_T_KICK_MS + m.CL_PROBE_SPACING_MS) & \
         (t < m.CL_PROBE_T_KICK_MS + 2 * m.CL_PROBE_SPACING_MS)
    hot = int(np.argmax((K[w1] - base).max(axis=0)))
    p1 = float((K[w1] - base)[:, hot].max())
    p2 = float((K[w2] - base)[:, hot].max())
    # spatial readout: participant COUNT alone cannot distinguish "more cells in the same place"
    # from "the same activity spread wider", so record extent explicitly (descriptive, no null)
    posE = np.asarray(S["posE"], float)
    dk_maps, spatial = [], []
    for j, w in enumerate((w1, w2)):
        a = int(round((m.CL_PROBE_T_KICK_MS + j * m.CL_PROBE_SPACING_MS) / m.DT))
        b = int(round((m.CL_PROBE_T_KICK_MS + (j + 1) * m.CL_PROBE_SPACING_MS) / m.DT))
        part = spk[a:b].any(axis=0)
        dk = (K[w] - base).max(axis=0)
        dk_maps.append([round(float(x), 6) for x in dk])
        spatial.append(dict(
            active_voxels_25pct=ION.active_voxel_count(dk, frac=0.25),
            active_voxels_50pct=ION.active_voxel_count(dk, frac=0.50),
            participant_voxels=int(np.unique(voxel[:S["NE"]][part]).size),
            recruit_radius_mm=ION.recruitment_radius_mm(posE, part, center=S["src_xy"])))
    tau = float(job["tau_Ko_at_workpoint_s"])
    linear = 1.0 + float(np.exp(-m.CL_PROBE_SPACING_MS * 1e-3 / tau))
    return dict(job=job, hot_voxel=hot, peak1_mM=p1, peak2_mM=p2,
                kick_spikes=kick_spikes, kick_participants=kick_parts,
                ratio_2nd_over_1st=(p2 / p1 if p1 > 0 else float("nan")),
                linear_prediction_at_workpoint=linear,
                mean_rate_E_hz=float(np.asarray(res["rate_E"], float).mean()),
                spatial=spatial, dk_map_per_kick=dk_maps,
                k_trace_hot=[round(float(x), 6) for x in (K[:, hot] - base[hot])],
                k_trace_t_ms=[float(x) for x in t],
                K_max=float(K.max()), wall_s=round(time.time() - t0, 1))


def cmd_b2_validate(args):
    if not args.confirm_run:
        raise SystemExit("b2-validate runs 40k trajectories; pass --confirm-run")
    with staged("b2-validate"):
        pre = preflight()
        cal = json.load(open(os.path.join(OUT, "b2_bias_calibration.json")))
        if cal["status"] != "CALIBRATED":
            raise SystemExit(f"b2-bias is {cal['status']}: validation may not run")
        gate = check_resource_gate("b2-validate")
        if gate["status"] == "PAUSE":
            raise SystemExit(f"resource gate PAUSE: {gate}")
        bar = float(load_artifact("pump_event_bar")["event_bar"])
        bE, bI = cal["best"]["I_bias_E"], cal["best"]["I_bias_I"]
        # Reproduce the SELECTED PROBE's condition exactly. The closure gate asks whether the
        # calibrated point is stable; introducing a different pre-equilibrium scaling here would
        # test a different condition, and with tau_Na = 54.4 s the initial ion state matters over
        # an 11 s window.
        sel_probe = min(cal["probes"], key=lambda p: abs(p["job"]["I_bias_E"] - bE)
                        + abs(p["job"]["I_bias_I"] - bI))
        sE = float(sel_probe["job"]["rate_scale_E"])
        sI = float(sel_probe["job"]["rate_scale_I"])
        print(f"[b2-validate] bias=({bE:+.4f},{bI:+.4f}) reproducing probe "
              f"'{sel_probe['job']['tag']}' scales ({sE:.4f},{sI:.4f})", flush=True)

        def job(q, tag):
            return dict(I_bias_E=bE, I_bias_I=bI, q_ion=q, conn_seed=CONN_SEED_DEV,
                        noise_seed=NOISE_DEV, T_ms=T_MS, rate_scale_E=sE, rate_scale_I=sI,
                        event_bar=bar, tag=tag)

        q0 = ION.q_ion_from_fprime(float(cal["f_prime"]))
        first = trajectory_task(job(q0, "pre_closure"))
        r0p = first["pooled"]["mean_rate_hz"]
        # EXACTLY ONE closure iteration, then freeze (spec §7.1)
        q1 = ION.q_ion_from_fprime(float(cal["f_prime"]), r0=r0p)
        print(f"[b2-validate] r0'={r0p:.4f} Hz -> q_ion {q0:.5f} -> {q1:.5f}", flush=True)
        # q_ion scales as 1/r0', so a suppressed r0' AMPLIFIES it and can leave the region where a
        # bounded Na steady state exists at all. Check before spending the run, and report it as a
        # closure failure with the reason rather than crashing in the initializer.
        rEf, rIf, voxf, _ = _load_40k_rate_field()
        try:
            ION.heterogeneous_steady_state(rEf * sE, rIf * sI, voxf[:32000], voxf[32000:],
                                           n_grid=N_GRID_40K, q_ion=q1, dx_mm=DX_MM_40K,
                                           max_iter=2)
        except ValueError as exc:
            payload = dict(
                generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
                frozen=dict(I_bias_E=bE, I_bias_I=bI, f_prime=cal["f_prime"], q_ion_final=None),
                closure=dict(r0_locked=ION.R0_HZ, r0_measured=r0p, q_ion_before=q0,
                             q_ion_after=q1, passed=False, status="UNRESOLVED_CALIBRATION",
                             infeasible=str(exc),
                             reason=("the one closure iteration recomputes q_ion as J_Na_0*f'/r0', "
                                     "so a suppressed r0' amplifies it; at this r0' the amplified "
                                     "q_ion admits NO bounded Na steady state for a large fraction "
                                     "of cells. That is a limit of the closure FORMULA at a "
                                     "suppressed working point, not a mechanism result.")),
                runs=[first])
            _write_json(os.path.join(OUT, "b2_closure_iteration.json"), payload)
            _write_json(os.path.join(OUT, "UNRESOLVED_CALIBRATION.json"), payload)
            raise SystemExit(payload["closure"]["reason"])
        second = trajectory_task(job(q1, "post_closure"))
        r0pp = second["pooled"]["mean_rate_hz"]
        rel = abs(r0pp - ION.R0_HZ) / ION.R0_HZ
        # spec 7.1: the closure gate is the E-rate residual AND a non-significant inter-block ion
        # trend. The first implementation checked only the residual; both are contract.
        # B2.1 spec 2.1: the gate quantity is the SIGNED SECULAR SLOPE, each against its own
        # bound; the first-difference q99 is retained only as a diagnostic.
        ion_ok = bool(second["ion"]["slope_q99_Na"] < ION.B2_1_SLOPE_BOUND_NA
                      and second["ion"]["slope_q99_K"] < ION.B2_1_SLOPE_BOUND_K)
        ion_q99 = max(second["ion"]["slope_q99_Na"] / ION.B2_1_SLOPE_BOUND_NA,
                      second["ion"]["slope_q99_K"] / ION.B2_1_SLOPE_BOUND_K)
        closure_ok = bool(rel <= CLOSURE_REL_TOL and ion_ok)

        # ruling 4: the closed-loop counterpart of the open-loop integration diagnostic
        fs = json.load(open(os.path.join(OUT, "b1_f_selection_v2.json")))
        row = next(r for r in fs["rows"] if r["f_prime"] == float(cal["f_prime"]))
        cl = closed_loop_integration_probe(dict(
            I_bias_E=bE, I_bias_I=bI, q_ion=q1, conn_seed=CONN_SEED_DEV, noise_seed=NOISE_DEV,
            rate_scale_E=sE, rate_scale_I=sI,
            tau_Ko_at_workpoint_s=row["measured"]["tau_Ko_at_workpoint_s"], tag="closed_loop"))
        ol = row["measured"]["integration_peaks_mM"]
        cl["open_loop_ratio_2nd_over_1st"] = float(ol[1] / ol[0]) if ol[0] > 0 else float("nan")
        cl["closed_loop_exceeds_open_loop"] = bool(
            cl["ratio_2nd_over_1st"] > cl["open_loop_ratio_2nd_over_1st"])
        cl["interpretation"] = (
            "matched 2-kick comparison at 200 ms. open-loop ratio comes from the SAME kernel "
            "replayed with g_K_ion = 0; closed-loop is the live layer at the frozen bias. A "
            "closed-loop ratio ABOVE the open-loop one means K -> E_K -> firing recovers some of "
            "the accumulation the load-dependent clearance removes; it does NOT by itself mean "
            "the loop is regenerative.")
        print(f"[b2-validate] closed-loop 2nd/1st = {cl['ratio_2nd_over_1st']:.3f} vs open-loop "
              f"{cl['open_loop_ratio_2nd_over_1st']:.3f} (linear@wp "
              f"{cl['linear_prediction_at_workpoint']:.3f})", flush=True)

        payload = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            frozen=dict(I_bias_E=bE, I_bias_I=bI, f_prime=cal["f_prime"], q_ion_final=q1),
            closure=dict(r0_locked=ION.R0_HZ, r0_measured=r0p, q_ion_before=q0, q_ion_after=q1,
                         r0_after_closure=r0pp, rel_err=rel, tol=CLOSURE_REL_TOL,
                         passed=closure_ok, iterations=1,
                         status=("PASS" if closure_ok else "UNRESOLVED_CALIBRATION"),
                         rate_residual_ok=bool(rel <= CLOSURE_REL_TOL),
                         ion_trend_ok=ion_ok, ion_trend_worst_ratio_to_bound=ion_q99,
                         slope_q99_Na=second["ion"]["slope_q99_Na"],
                         slope_q99_K=second["ion"]["slope_q99_K"],
                         slope_bound_Na=ION.B2_1_SLOPE_BOUND_NA,
                         slope_bound_K=ION.B2_1_SLOPE_BOUND_K,
                         gate_definition=("spec 7.1: E-rate residual <= 10% AND a non-significant "
                                          "inter-block ion trend. BOTH are contract."),
                         rule="recompute q_ion exactly once, rerun, then FREEZE -- convergence may "
                              "not be pursued by further iteration (spec §7.1)"),
            runs=[first, second],
            closed_loop_integration=cl,
            b2_risk_register_answer=dict(
                question="does closed-loop K -> E_K -> firing overcome the load-dependent "
                         "clearance measured open loop? (T7.1 ruling 4)",
                closed_loop_ratio=cl["ratio_2nd_over_1st"],
                open_loop_ratio=cl["open_loop_ratio_2nd_over_1st"],
                linear_at_workpoint=cl["linear_prediction_at_workpoint"],
                answer=cl["interpretation"]),
            window_caveat=("11 s = 0.2 tau_Na. This run can only show that the ion state is stable "
                           "NEAR the initialization point; it does NOT show the system has reached "
                           "steady state. Inter-block trends are reported per cell and per voxel "
                           "(q95/q99/max); a flat population mean is not evidence."))
        _write_json(os.path.join(OUT, "b2_closure_iteration.json"), payload)
        print(f"[b2-validate] r0'={r0p:.4f} -> q_ion {q0:.5f}->{q1:.5f} -> r0''={r0pp:.4f} "
              f"(rel {rel:.4f}, tol {CLOSURE_REL_TOL}; ion q99 {ion_q99:.4f} vs "
              f"{ION.GATE_B_ION_BLOCK_DRIFT_MAX})  closure "
              f"{'PASS' if closure_ok else 'UNRESOLVED_CALIBRATION'}")
        if not closure_ok:
            failed = ([] if rel <= CLOSURE_REL_TOL else [f"E-rate residual {rel:.4f} > "
                                                        f"{CLOSURE_REL_TOL}"]) + \
                     ([] if ion_ok else [f"signed secular slope q99: Na "
                                f"{second['ion']['slope_q99_Na']:.3e} vs "
                                f"{ION.B2_1_SLOPE_BOUND_NA:.3e}, K "
                                f"{second['ion']['slope_q99_K']:.3e} vs "
                                f"{ION.B2_1_SLOPE_BOUND_K:.3e}"])
            reason = ("closure -> UNRESOLVED_CALIBRATION. Failing clause(s): " + "; ".join(failed)
                      + ". Per plan section 10 this is a CALIBRATOR failure, not a mechanism "
                      "NO-GO: NO_GO_BASELINE would require the legal [-2,+2] box to be bracketed "
                      "and shown to contain no solution, and it was not. Do not iterate the "
                      "closure further -- fix the calibrator.")
            payload["closure"]["failing_clauses"] = failed
            payload["closure"]["reason"] = reason
            # ONE final payload feeds every canonical file, so they cannot disagree.
            _write_json(os.path.join(OUT, "b2_closure_iteration.json"), payload)
            _write_json(os.path.join(OUT, "UNRESOLVED_CALIBRATION.json"), payload)
            raise SystemExit(reason)
    return 0


def cmd_b2_confirm(args):
    if not args.confirm_run:
        raise SystemExit("b2-confirm runs six 40k trajectories; pass --confirm-run")
    with staged("b2-confirm"):
        pre = preflight()
        clo = json.load(open(os.path.join(OUT, "b2_closure_iteration.json")))
        if not clo["closure"]["passed"]:
            raise SystemExit("closure did not pass: confirmatory runs may not proceed")
        gate = check_resource_gate("b2-confirm")
        resource_log("b2_confirm_gate", gate)
        if gate["status"] == "PAUSE":
            raise SystemExit(f"resource gate PAUSE: {gate}")
        fz = clo["frozen"]
        last = clo["runs"][-1]["job"]
        bar = float(load_artifact("pump_event_bar")["event_bar"])
        jobs = [dict(I_bias_E=fz["I_bias_E"], I_bias_I=fz["I_bias_I"], q_ion=fz["q_ion_final"],
                     conn_seed=cs, noise_seed=ns, T_ms=T_MS,
                     rate_scale_E=last["rate_scale_E"], rate_scale_I=last["rate_scale_I"],
                     event_bar=bar, tag=f"conn{cs}_noise{ns}")
                for cs in CONN_SEEDS_CONFIRM for ns in NOISE_CONFIRM]
        plan = plan_workers(args.workers or 4, per_worker_gb=16.0)
        print(f"[b2-confirm] {len(jobs)} trajectories, workers={plan['workers']} "
              f"(siblings={plan['siblings']}, MemAvailable={plan['mem_available_gb']} GB)",
              flush=True)
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=plan["workers"]) as pool:
            runs = pool.map(trajectory_task, jobs)
        resource_log("b2_confirm_done")
        for r in runs:
            p = r["pooled"]
            print(f"[b2-confirm] {r['job']['tag']}: rate={p['mean_rate_hz']:.3f} "
                  f"events={p['n_events']:.1f}/blk scoreable={p['n_scoreable']} "
                  f"A={p['n_A']} B={p['n_B']} minfrac={min(p['frac_A'], p['frac_B']):.3f}",
                  flush=True)
        _write_json(os.path.join(OUT, "b2_confirmatory.json"),
                    dict(generated=datetime.now(timezone.utc).isoformat(),
                         code_commit=pre["code_commit"], frozen=fz, runs=runs,
                         note="bias and q_ion were frozen on the development seed; these six "
                              "trajectories were NOT used for tuning", workers=plan))
    return 0


# ================================================================== B2.1 (spec 2026-07-28 lock)
def cmd_b2_ionsoff_control(args):
    """B2.1 spec 3: the SAME-SEED ions-off control, so 'the ion layer raises the rate by X%'
    becomes a matched comparison instead of one across noise202 and noise401."""
    if not args.confirm_run:
        raise SystemExit("b2-ionsoff-control runs a 40k trajectory; pass --confirm-run")
    with staged("b2-ionsoff-control", dict(noise_seed=NOISE_DEV)):
        pre = preflight()
        gate = check_resource_gate("b2-ionsoff-control")
        if gate["status"] == "PAUSE":
            raise SystemExit(f"resource gate PAUSE: {gate}")
        S, _ = _substrate(CONN_SEED_DEV)
        res, _ = run_arm_c(S, noise_seed=NOISE_DEV, T_ms=T_MS, dump_i=True, verbose=False)
        n_steps = res["E_spk_bool"].shape[0]
        blocks = block_edges(n_steps)
        rate = np.asarray(res["rate_E"], float)
        rI = np.asarray(res["rate_I"], float)
        rE_m = float(np.mean([rate[lo:hi].mean() for lo, hi in blocks]))
        rI_m = float(np.mean([rI[lo:hi].mean() for lo, hi in blocks]))
        cal = json.load(open(os.path.join(OUT, "b2_bias_calibration.json")))
        base = next(p for p in cal["probes"] if p["job"]["tag"] == "base")
        payload = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            noise_seed=NOISE_DEV, conn_seed=CONN_SEED_DEV, T_ms=T_MS, ions="OFF",
            r_E=rE_m, r_I=rI_m,
            matched_ions_on_zero_bias=dict(r_E=base["r_E"], r_I=base["r_I"],
                                           tag=base["job"]["tag"]),
            ion_layer_rate_change=dict(
                r_E_rel=(base["r_E"] - rE_m) / rE_m, r_I_rel=(base["r_I"] - rI_m) / rI_m,
                note="SAME noise seed and connectivity, ions off vs ions on at zero bias -- this "
                     "is the matched causal comparison the earlier 6.960-vs-4.158 number was not"),
            locked_target_r0=ION.R0_HZ,
            ions_off_vs_locked_target=dict(rel=(rE_m - ION.R0_HZ) / ION.R0_HZ,
                                           note="noise401 ions-off vs the noise202 locked r0; a "
                                                "seed-to-seed difference, not an ion effect"))
        _write_json(os.path.join(OUT, "b2_1_ionsoff_control.json"), payload)
        print(f"[b2-ionsoff-control] ions OFF noise401: r_E={rE_m:.4f} r_I={rI_m:.4f}; "
              f"ions ON zero bias: r_E={base['r_E']:.4f} r_I={base['r_I']:.4f}  -> "
              f"ion effect on r_E = {100*payload['ion_layer_rate_change']['r_E_rel']:+.1f}%")
    return 0


def cmd_b2_selfconsistent(args):
    """B2.1 spec 2.2: damped self-consistent ions-on rate-field initialization.

    r -> ionic equilibrium -> live network -> r'.  Shrink, damp, iterate at most
    B2_1_MAX_UPDATES times, then judge the ion slope on a trajectory that was NOT used to derive
    its own initialization (true by construction: each run is initialized from the PREVIOUS one).
    q_ion and both biases are FROZEN throughout.
    """
    if not args.confirm_run:
        raise SystemExit("b2-selfconsistent runs up to 4 40k trajectories; pass --confirm-run")
    with staged("b2-selfconsistent"):
        pre = preflight()
        cal = json.load(open(os.path.join(OUT, "b2_bias_calibration.json")))
        if cal["status"] != "CALIBRATED":
            raise SystemExit(f"b2-bias is {cal['status']}: B2.1 self-consistency may not run")
        gate = check_resource_gate("b2-selfconsistent")
        if gate["status"] == "PAUSE":
            raise SystemExit(f"resource gate PAUSE: {gate}")

        bE, bI = cal["best"]["I_bias_E"], cal["best"]["I_bias_I"]
        q_ion = ION.q_ion_from_fprime(float(cal["f_prime"]))
        rE0, rI0, voxel, sha = _load_40k_rate_field()
        NE = int(rE0.size)
        nv = N_GRID_40K ** 2
        cur_E, cur_I = rE0.copy(), rI0.copy()          # iteration 0 = the pump-off field
        win_s = N_BLOCKS * BLOCK_MS / 1000.0
        history, prev_field_source = [], "pump-off (spec 4.2c original)"

        for k in range(ION.B2_1_MAX_UPDATES + 1):
            job = dict(I_bias_E=bE, I_bias_I=bI, q_ion=q_ion, conn_seed=CONN_SEED_DEV,
                       noise_seed=NOISE_DEV, T_ms=T_MS, rate_scale_E=1.0, rate_scale_I=1.0,
                       event_bar=float(load_artifact("pump_event_bar")["event_bar"]),
                       tag=f"selfconsistent_iter{k}",
                       rate_field_override=dict(rate_E=cur_E.tolist(), rate_I=cur_I.tolist()))
            r = trajectory_task(job)
            meas_E = ION.shrink_rate_field(np.asarray(r["per_cell_counts_E"], float),
                                           voxel[:NE], window_s=win_s, n_voxels=nv)
            meas_I = ION.shrink_rate_field(np.asarray(r["per_cell_counts_I"], float),
                                           voxel[NE:], window_s=win_s, n_voxels=nv)
            chg = ION.damped_field_change(np.concatenate([cur_E, cur_I]),
                                          np.concatenate([meas_E, meas_I]))
            rel = chg["max_rel"]          # spec 2.2 gate statistic, contract-locked to the max
            history.append(dict(
                iteration=k, field_source=prev_field_source, rate_rel_change=rel,
                change=chg, mean_rate_I_hz=float(np.mean(meas_I)),
                slope_Na=r["ion"]["slope_Na"], slope_K=r["ion"]["slope_K"],
                mean_rate_E_hz=r["pooled"]["mean_rate_hz"],
                slope_q99_Na=r["ion"]["slope_q99_Na"], slope_q99_K=r["ion"]["slope_q99_K"],
                slope_mean_signed_Na=r["ion"]["slope_Na"]["mean_signed"],
                slope_mean_signed_K=r["ion"]["slope_K"]["mean_signed"],
                first_difference_q99_Na=r["ion"]["q99_abs_dNa_dt"],
                n_events=r["pooled"]["n_events"], n_scoreable=r["pooled"]["n_scoreable"]))
            print(f"[b2-selfconsistent] iter{k}: damped max={rel:.4f} "
                  f"(q99={chg['q99_rel']:.4f} q95={chg['q95_rel']:.4f}; "
                  f"undamped max={chg['undamped_max_rel']:.4f}) "
                  f"r_E={r['pooled']['mean_rate_hz']:.4f} "
                  f"slope_q99 Na={r['ion']['slope_q99_Na']:.3e} "
                  f"(bound {ION.B2_1_SLOPE_BOUND_NA:.3e}) "
                  f"K={r['ion']['slope_q99_K']:.3e} (bound {ION.B2_1_SLOPE_BOUND_K:.3e})",
                  flush=True)
            resource_log(f"selfconsistent_iter{k}")
            verdict = ION.adjudicate_b2_1_selfconsistency(dict(
                rate_rel_change=rel, slope_q99_Na=r["ion"]["slope_q99_Na"],
                slope_q99_K=r["ion"]["slope_q99_K"], n_updates=k,
                independent_window=bool(k > 0)))
            if verdict["status"] == "CONVERGED" and k > 0:
                break
            if k == ION.B2_1_MAX_UPDATES:
                break
            cur_E = ION.damped_rate_update(cur_E, meas_E)
            cur_I = ION.damped_rate_update(cur_I, meas_I)
            prev_field_source = f"damped update from iter{k} (alpha={ION.B2_1_ALPHA})"

        payload = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            contract="docs/superpowers/specs/2026-07-28-topic4-fcxr-ion-B2_1-lock.md",
            status=verdict["status"], verdict=verdict, history=history,
            frozen=dict(I_bias_E=bE, I_bias_I=bI, q_ion=q_ion, f_prime=cal["f_prime"],
                        note="q_ion and both biases were frozen for the whole loop (spec 2.2)"),
            shrinkage=dict(n0=ION.B2_1_SHRINK_N0, alpha=ION.B2_1_ALPHA,
                           max_updates=ION.B2_1_MAX_UPDATES),
            source_rate_field_sha256=sha,
            converged_rate_field=dict(rate_E=cur_E.tolist(), rate_I=cur_I.tolist()),
            converged_rate_field_note=("the field the LAST iteration ran on. Named for the "
                                       "converged case; when NOT_CONVERGED it is simply the final "
                                       "iterate, and carries no fixed-point claim."))
        _write_json(os.path.join(OUT, "b2_1_selfconsistent.json"), payload)
        print(f"[b2-selfconsistent] {verdict['status']} after {len(history)} trajectories")
        if verdict["status"] != "CONVERGED":
            raise SystemExit(
                "B2.1 self-consistency NOT_CONVERGED -- the calibration instrument is still not "
                "sound, so the B2 closure may not be re-run and Gate B stays closed. Do not widen "
                "the slope bound or the update cap.")
    return 0


def cmd_b2_matched_control(args):
    """B2.1 spec 2.3: the two arms differ ONLY in whether the event-induced dE_K reaches the
    membrane.  The open arm freezes membrane_current() at its pre-kick value, which both cuts the
    event-induced feedback and holds the pre-kick working point exactly."""
    if not args.confirm_run:
        raise SystemExit("b2-matched-control runs two 40k probes; pass --confirm-run")
    with staged("b2-matched-control"):
        pre = preflight()
        sc = json.load(open(os.path.join(OUT, "b2_1_selfconsistent.json")))
        if sc["status"] != "CONVERGED" and not args.allow_nonconverged_field:
            raise SystemExit(f"B2.1 self-consistency is {sc['status']}: matched control may not run "
                             f"on a converged field. Pass --allow-nonconverged-field to run it on "
                             f"the SHARED non-fixed-point field instead; the within-arm comparison "
                             f"stays valid, only its generalization is limited.")
        field_note = ("CONVERGED self-consistent fixed point" if sc["status"] == "CONVERGED" else
                      "NOT a fixed point: B2.1 self-consistency was " + sc["status"] + ". The two "
                      "arms still share the IDENTICAL initial ion field, so the within-comparison "
                      "(arms differ only in whether the event-induced dE_K reaches the membrane) "
                      "is valid; what is limited is generalization beyond this operating point.")
        fz = sc["frozen"]
        base = dict(I_bias_E=fz["I_bias_E"], I_bias_I=fz["I_bias_I"], q_ion=fz["q_ion"],
                    conn_seed=CONN_SEED_DEV, noise_seed=NOISE_DEV,
                    # NOT the iter-0 field when non-converged: the loop updates cur_* at the end
                    # of every iteration except the last, so this is the field the LAST iteration
                    # ran on.  Either way both arms share it bit-for-bit, which is what 2.3 needs.
                    rate_field_override=sc["converged_rate_field"],
                    tau_Ko_at_workpoint_s=float(next(
                        r for r in json.load(open(os.path.join(OUT, "b1_f_selection_v2.json")))
                        ["rows"] if r["f_prime"] == float(fz["f_prime"])
                    )["measured"]["tau_Ko_at_workpoint_s"]))
        freeze_block = int(round((CL_PROBE_T_KICK_MS - 50.0) / DT_ION_MS))
        arms = {}
        for tag, fb in (("closed", 0), ("open", freeze_block)):
            arms[tag] = closed_loop_integration_probe(
                dict(base, tag=tag, freeze_membrane_from_block=fb))
            a = arms[tag]
            print(f"[b2-matched-control] {tag}: peaks {a['peak1_mM']:.4f}/{a['peak2_mM']:.4f} "
                  f"spikes {a.get('kick_spikes')} participants {a.get('kick_participants')} "
                  f"active_vox25 {[x['active_voxels_25pct'] for x in a['spatial']]} "
                  f"radius_mm {[round(x['recruit_radius_mm'], 3) for x in a['spatial']]}",
                  flush=True)
        v = ION.adjudicate_matched_control(
            closed=dict(spikes=arms["closed"]["kick_spikes"],
                        participants=arms["closed"]["kick_participants"],
                        peaks=[arms["closed"]["peak1_mM"], arms["closed"]["peak2_mM"]]),
            open_=dict(spikes=arms["open"]["kick_spikes"],
                       participants=arms["open"]["kick_participants"],
                       peaks=[arms["open"]["peak1_mM"], arms["open"]["peak2_mM"]]),
            # bit-identical simulations until the freeze block: same seed, same config, the ONLY
            # difference is the pinned membrane current applied strictly before the first kick
            structurally_identical_until_freeze=True)
        v.update(generated=datetime.now(timezone.utc).isoformat(),
                 code_commit=pre["code_commit"],
                 contract="docs/superpowers/specs/2026-07-28-topic4-fcxr-ion-B2_1-lock.md",
                 initial_field=field_note,
                 spatial_extent={t: arms[t]["spatial"] for t in ("closed", "open")},
                 spatial_note=("descriptive, single seed, no null. active_voxels_* counts voxels "
                               "reaching that fraction of THAT kick's own peak excess K; "
                               "recruit_radius_mm is the RMS distance of participating E cells "
                               "from the kick centre. Recorded so that a rise in participant "
                               "COUNT is not read as spatial spread without evidence."),
                 selfconsistency_status=sc["status"],
                 design=("both arms share q_ion, bias, initial ion field, noise seed, kick times, "
                         "strength and radius; the OPEN arm freezes membrane_current() at the "
                         "block before the first kick, which cuts the event-induced dE_K while "
                         "holding the pre-kick working point"),
                 arms=arms)
        _write_json(os.path.join(OUT, "b2_1_matched_control.json"), v)
        print(f"[b2-matched-control] {v['status']}"
              + (f"  closed {v['ratio']['closed_2nd_over_1st']:.3f} vs open "
                 f"{v['ratio']['open_2nd_over_1st']:.3f}" if "ratio" in v else ""))
    return 0


def cmd_b2_adjudicate(args):
    with staged("b2-adjudicate"):
        pre = preflight()
        conf = json.load(open(os.path.join(OUT, "b2_confirmatory.json")))
        clo = json.load(open(os.path.join(OUT, "b2_closure_iteration.json")))
        cal = json.load(open(os.path.join(OUT, "b2_bias_calibration.json")))
        eq = load_artifact("pump_off_baseline")
        prop = load_artifact("e1146_propagation")
        tol = {k: dict(off=v["off"], margin=v["margin"], underpowered=v["underpowered"])
               for k, v in eq["equivalence"]["per_metric"].items()}
        ac = prop["adaptive_cluster"]
        template_layer = dict(
            subject="epilepsiae_1146", stable_k=ac["stable_k"],
            inter_cluster_corr=ac["inter_cluster_corr_matrix"][0][1],
            candidate_forward_reverse_pairs=ac["candidate_forward_reverse_pairs"],
            layer="template layer (PR-2 adaptive cluster)",
            supports="two stable propagation templates exist for this subject",
            does_not_support="a confirmed forward/reverse template pair "
                             "(candidate_forward_reverse_pairs is the empty list) or established "
                             "bidirectional propagation")
        gb = ION.adjudicate_gate_B(conf["runs"], tol, template_layer=template_layer)
        gb.update(generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
                  frozen=clo["frozen"], closure=clo["closure"],
                  calibration_status=cal["status"],
                  tolerance_source=dict(
                      artifact=resolve_artifact(
                          REQUIRED_ARTIFACTS["pump_off_baseline"]["rel"])["resolved_abs_path"],
                      field="equivalence.per_metric[*].{off, margin, underpowered}",
                      note="margins were pre-locked by the accepted pump sprint; they are read, "
                           "not recomputed"),
                  window_caveat=clo["window_caveat"])
        _write_json(os.path.join(OUT, "gate_B.json"), gb)

        gh = json.load(open(os.path.join(OUT, "gate_H.json")))
        fs = json.load(open(os.path.join(OUT, "b1_f_selection_v2.json")))
        verdict = dict(
            generated=datetime.now(timezone.utc).isoformat(), code_commit=pre["code_commit"],
            sprint="FCXR-ION Phase B0-B2",
            layers=dict(
                engineering="green" if gh["status"] == "PASS" else "see gate_H",
                gate_H=gh["status"], f_prime_selection=fs["status"],
                selected_f_prime=fs.get("selected_f_prime"),
                bias_calibration=cal["status"], gate_B=gb["status"],
                lifecycle="not tested (B3/B4 not authorised)"),
            allowed_statement=gb["allowed_statement"],
            forbidden_statements=gb["forbidden_statements"],
            locks=dict(eta_pump=ION.ETA_PUMP_B0_B2, g_K_ion=ION.G_K_ION_REFERENCE,
                       g_K_ion_kind="effective reference normalization (B3 would calibrate it)",
                       rho=ION.RHO, nuisance_parameters=["I_bias_E", "I_bias_I"]),
            not_executed=["B3 (local K perturbation -> g_K_ion)",
                          "B4 (frozen high branch -> eta_pump)",
                          "three-dimensional phase map", "dynamic lifecycle",
                          "causal decomposition", "any Cl / Ca / two-compartment extension"])
        _write_json(os.path.join(OUT, "candidate_verdict.json"), verdict)
        print(f"[b2-adjudicate] Gate B = {gb['status']}  "
              f"(B-real {gb['b_real']['n_direction_passing']}/{gb['b_real']['n_trajectories']} "
              f"trajectories, B-model ok={gb['b_model']['ok']})")
        print(f"   allowed: {gb['allowed_statement']}")
    return 0


def write_manifest():
    """run_manifest.json -- every field the plan §12 lists."""
    pre = preflight(write=False)
    fld = os.path.join(OUT, "b0_baseline_rate_field.npz")

    def _maybe(name):
        p = os.path.join(OUT, name)
        return json.load(open(p)) if os.path.exists(p) else None

    fs, cal, clo, gh, gb = (_maybe(n) for n in ("b1_f_selection_v2.json", "b2_bias_calibration.json",
                                                "b2_closure_iteration.json", "gate_H.json",
                                                "gate_B.json"))
    cfg_hash = hashlib.sha256(json.dumps(arm_c_pump_off_cfg(), sort_keys=True).encode()).hexdigest()
    avail, swap = _meminfo()
    man = dict(
        sprint="FCXR-ION Phase B0-B2", generated=datetime.now(timezone.utc).isoformat(),
        code_commit=pre["code_commit"], branch="codex/topic4-fcxr-ion",
        spec="docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md",
        plan="docs/superpowers/plans/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-B0-B2.md",
        artifacts=pre["artifacts"],
        blessed_engine_sha256=blessed_engine_hashes(),
        selected_f_prime=(fs or {}).get("selected_f_prime"),
        q_ion=(clo or {}).get("frozen", {}).get("q_ion_final"),
        slow_variable_config=dict(
            arm="arm-C FCXR (ff additive, recurrent conductance, g_sat=21.6)",
            config=arm_c_pump_off_cfg(), config_sha256=cfg_hash,
            use_z=False, use_m=False, use_x=False, use_h=False,
            note="Z / M / X / H are all OFF on this substrate; the ion layer is the only "
                 "slow mechanism added, and eta_pump is locked to 0"),
        seeds=dict(connectivity_dev=CONN_SEED_DEV, connectivity_confirm=list(CONN_SEEDS_CONFIRM),
                   noise_direction_power=NOISE_DIRECTION_POWER, noise_dev=NOISE_DEV,
                   noise_confirm=list(NOISE_CONFIRM), initial_state="deterministic reset"),
        grid=dict(n_grid=N_GRID_40K, dx_mm=DX_MM_40K, L_mm=20.0, dt_ion_ms=DT_ION_MS, dt_ms=DT,
                  cells_per_voxel_mean=39.0625),
        locks=dict(eta_pump=ION.ETA_PUMP_B0_B2, g_K_ion=ION.G_K_ION_REFERENCE,
                   g_K_ion_kind="effective reference normalization",
                   ions_on_both_E_and_I=True,
                   virtual_electrode_readout="E-only (engine recorder limitation, spec §5)",
                   rho=ION.RHO, r0_hz=ION.R0_HZ),
        baseline_rate_field=dict(path=fld, sha256=_sha(fld) if os.path.exists(fld) else None),
        gates=dict(gate_H=(gh or {}).get("status"), gate_B=(gb or {}).get("status"),
                   f_selection=(fs or {}).get("status"),
                   bias_calibration=(cal or {}).get("status"),
                   closure=((clo or {}).get("closure") or {}).get("status"),
                   closure_failing_clauses=((clo or {}).get("closure") or {}).get(
                       "failing_clauses")),
        known_metric_limits=dict(
            ion_trend_q99=("q99 of |first differences| -- a FLUCTUATION MAGNITUDE, not a secular "
                           "trend; does not approach zero while interictal events continue"),
            closed_vs_open_loop=("NOT a matched control: the arms differ in q_ion, drive and "
                                 "recruitment, so no causal reading of the feedback is licensed"),
            k_wave_far_over_event=("confounded by global drift; spec 11 stop condition 3 is "
                                   "UNANSWERED, not triggered")),
        resources=dict(mem_available_gb_now=round(avail, 2), swap_used_gb_now=round(swap, 3),
                       baseline=json.load(open(_baseline_swap_path()))
                       if os.path.exists(_baseline_swap_path()) else None,
                       max_workers_authorised=MAX_WORKERS))
    _write_json(os.path.join(OUT, "run_manifest.json"), man)
    print(f"[manifest] written: gate_H={man['gates']['gate_H']} gate_B={man['gates']['gate_B']} "
          f"f'={man['selected_f_prime']}")
    return man


def write_status():
    """STATUS.md -- layered, so an upstream PASS never silently propagates downstream."""
    def _m(n):
        p = os.path.join(OUT, n)
        return json.load(open(p)) if os.path.exists(p) else None

    pf, units, feas = _m("b0_artifact_preflight.json"), _m("b0_voltage_unit_audit.json"), \
        _m("b0_analytic_feasibility.json")
    dp, gh = _m("b0_direction_power.json"), _m("gate_H.json")
    fs = _m("b1_f_selection_v2.json") or _m("b1_f_selection.json")
    cal, clo, gb = _m("b2_bias_calibration.json"), _m("b2_closure_iteration.json"), _m("gate_B.json")

    def row(name, obj, key="status", extra=""):
        v = "NOT RUN" if obj is None else obj.get(key, "?")
        return f"| {name} | `{v}` | {extra} |"

    L = ["# FCXR-ION Phase B0-B2 — STATUS", "",
         "Spec: `docs/superpowers/specs/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-design.md` (rev4)",
         "Plan: `docs/superpowers/plans/2026-07-27-topic4-fcxr-constitutive-na-k-homeostasis-B0-B2.md` (rev3)",
         f"Branch: `codex/topic4-fcxr-ion`  ·  commit `{_git_sha()[:8]}`", "",
         "**Layers are reported separately. An upstream PASS does not propagate downstream.**", "",
         "| layer | verdict | note |", "|---|---|---|",
         row("artifact preflight", pf, extra="7 inputs resolved across worktree / pump / heo1 / main"),
         row("T1 engine voltage units", units,
             extra="dimension only; `g_K_ion = 1` stays a declared normalization"),
         row("T4 analytic feasibility", feas,
             extra=f"tau_Na={feas['tau_Na_s']:.1f}s vs tau_Ko={feas['tau_Ko_s']:.3f}s "
                   f"({feas['tau_ratio']:.0f}x)" if feas else ""),
         row("T2 direction-readout power", (dp or {}).get("power_gate") if dp else None,
             extra=(f"{dp['initiation_site']['pooled']['n_scoreable']} of "
                    f"{dp['initiation_site']['pooled']['n_events']} events scoreable; "
                    f"mean-rate reproduction rel {dp['reproduction']['rel_diff']:.1e}")
             if dp else ""),
         row("Gate H (homeostasis + numerics)", gh,
             extra=f"primary tier {gh['primary_network']}" if gh else ""),
         row("T7.1 f' selection", fs,
             extra=(f"selected f' = {fs.get('selected_f_prime')} (PROVISIONAL: chosen on an "
                    f"open-loop scale diagnostic under the repaired gates; not a closed-loop "
                    f"claim)" if fs else "")),
         row("T8 bias calibration", cal,
             extra=(f"bias=({cal['best']['I_bias_E']:+.3f}, {cal['best']['I_bias_I']:+.3f}), "
                    f"{cal['n_probes']} probes") if cal else ""),
         row("closure iteration", (clo or {}).get("closure") if clo else None, key="status",
             extra=(f"rate residual {clo['closure']['rel_err']:.4f} <= 0.10 OK; ion inter-block "
                    f"trend q99 {clo['closure'].get('ion_trend_q99', float('nan')):.4f} vs bound "
                    f"{clo['closure'].get('ion_trend_bound')} FAILS -- spec 7.1 requires BOTH"
                    if clo else "")),
         row("Gate B (interictal substrate)", gb,
             extra=(f"B-real {gb['b_real']['n_direction_passing']}/"
                    f"{gb['b_real']['n_trajectories']}, B-model ok={gb['b_model']['ok']}")
             if gb else ""),
         "| lifecycle | `NOT TESTED` | B3/B4 not authorised |", "",
         "### B2.1 calibration-instrument repair "
         "(`docs/superpowers/specs/2026-07-28-topic4-fcxr-ion-B2_1-lock.md`)", "",
         "| item | verdict | note |", "|---|---|---|",
         "| 2.1 signed secular trend estimator | `FIXED` | replaces the first-difference q99; a "
         "synthetic stationary-but-eventful trace now reads slope < 1e-3 while its first-difference "
         "q99 is 100x larger. **With the corrected estimator the ion gate STILL fails**: per-cell "
         "q99 7.5e-2 vs bound 2.07e-2 (Na), 7.4e-2 vs 1.09e-3 (K) |",
         "| 2.2 self-consistent ions-on initialization | `NOT_CONVERGED` | oscillates, not slow: "
         "r_E 4.44 -> 2.64 -> 0.21 -> 4.53 Hz, relative change 2.47 -> 11.01. At frozen bias the "
         "map has no attracting fixed point, so pre-equilibrium and bias are NOT separable |",
         "| 2.3 matched open/closed control | `COMPARABLE` | arms bit-identical until the freeze; "
         "kick1 agrees to 1.7%/2.7%, kick2 diverges 26% in recruitment. Closed 2nd/1st = 0.991 vs "
         "open 1.273, while the closed arm recruits 42% more cells on kick 2 |",
         "| 3 same-seed ions-off control | `DONE` | ions off 3.716 Hz vs ions on at zero bias "
         "6.960 Hz -> **+87.3%** (the earlier +64% spanned noise202/noise401). noise401's own "
         "ions-off baseline is 10.6% BELOW the locked r0, so the biases absorb a seed difference "
         "too |", "",
         "## Why Gate B was not adjudicated", "",
         "Because the **closure iteration did not pass**, and confirmatory trajectories at a "
         "working point whose ion state is still moving would produce Gate B numbers that do not "
         "mean what Gate B claims. The closure has two clauses (spec 7.1) and they split:", "",
         "- **rate residual: PASS.** The two nuisance biases brought the E-POPULATION MEAN rate "
         "to within 1.8% of the locked target (4.085 Hz against 4.158 Hz), on ONE development "
         "window. Not yet accepted: r_I (still 10.3% high), event rate / IEI / duration / "
         "participation, the two-core initiation split, and seed robustness -- so this is NOT "
         "'the interictal working point is recovered'. The zero-bias 6.96 Hz is from noise401 "
         "while the 4.158 Hz target is from noise202, so the +64% is not yet a matched causal "
         "comparison either.",
         "- **ion inter-block statistic: FAIL.** per-cell / per-voxel q99 of |first differences| "
         "= 0.084 / 0.259 mM/s against a 0.05 bound. NOTE what this number is and is not: it is a "
         "FLUCTUATION MAGNITUDE, not a secular trend, and ordinary interictal events keep feeding "
         "it even in a statistically stationary state. The Na population mean moved only "
         "-0.0026 mM/s net over the same window, about 33x smaller. The safe statement is 'this "
         "trajectory did not pass the existing q99 first-difference gate', NOT 'the ion field is "
         "drifting at 0.26 mM/s'.", "",
         "Per plan section 15 an `UNRESOLVED_CALIBRATION` means the calibrator must be improved "
         "**separately** -- it is explicitly NOT a mechanism NO-GO, and the legal bias box was "
         "never bracketed (7 of 12 probes over +-0.5 mV inside +-2 mV). The repair is scoped in "
         "`docs/superpowers/specs/2026-07-28-topic4-fcxr-ion-B2_1-lock.md`.", "",
         "The closed-vs-open-loop comparison this sprint owed was NOT answerable at the time of "
         "that adjudication (the two arms differed in q_ion, drive and recruitment). B2.1 section "
         "2.3 replaced it with a structurally matched pair and ANSWERED it -- see the B2.1 table "
         "above and `figures/b2_1_calibration_repair.png`.", "",
         "## Allowed statement", "",
         f"> {gb['allowed_statement']}" if gb else "> (Gate B has not been adjudicated.)", "",
         "## Not claimed", ""]
    for f in (gb or {}).get("forbidden_statements", [
            "reproduced bidirectional propagation (not established for E1146)",
            "the ion mechanism is refuted",
            "the electrogenic pump pathway was tested (eta_pump locked to 0)",
            "a seizure lifecycle was obtained",
            "patient ion concentrations were reconstructed"]):
        L.append(f"- {f}")
    L += ["",
          "## Window caveat", "",
          "`tau_Na` = 54.4 s and every trajectory here is 11 s = 0.2 `tau_Na`. These runs can show "
          "that the ion state is stable NEAR its initialization point; they cannot show that the "
          "system has reached steady state. All ion stationarity is therefore reported per cell "
          "and per voxel (q95 / q99 / max) — a flat population mean is not evidence.", ""]
    with open(os.path.join(OUT, "STATUS.md"), "w") as f:
        f.write("\n".join(L))
    print("[status] STATUS.md written")


def cmd_manifest(args):
    write_manifest()
    write_status()
    return 0


STAGES = {
    "b0-units": cmd_b0_units,
    "b0-provenance": cmd_b0_provenance,
    "b0-direction-power": cmd_b0_direction_power,
    "b1-smallnet": cmd_b1_smallnet,
    "b1-gate-h": cmd_b1_gate_h,
    "b1-select-f": cmd_b1_select_f,
    "b2-bias": cmd_b2_bias,
    "b2-validate": cmd_b2_validate,
    "b2-confirm": cmd_b2_confirm,
    "b2-ionsoff-control": cmd_b2_ionsoff_control,
    "b2-selfconsistent": cmd_b2_selfconsistent,
    "b2-matched-control": cmd_b2_matched_control,
    "b2-adjudicate": cmd_b2_adjudicate,
    "manifest": cmd_manifest,
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=sorted(STAGES) + ["preflight"])
    ap.add_argument("--confirm-run", action="store_true",
                    help="required for any stage that starts a simulation")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--probe-t-ms", type=float, default=None,
                    help="bias-probe trajectory length; must match the validation window, "
                         "otherwise the calibrator cannot see the tau_Na drift")
    ap.add_argument("--record-swap-baseline", action="store_true")
    ap.add_argument("--allow-nonconverged-field", action="store_true",
                    help="run the matched control on the shared NON-fixed-point ion field; the "
                         "within-arm comparison stays valid, generalization is limited")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    if a.record_swap_baseline or not os.path.exists(_baseline_swap_path()):
        record_swap_baseline()
    if a.stage == "preflight":
        p = preflight()
        print(json.dumps({k: dict(root=v["root_used"], schema_ok=v["schema_ok"],
                                  sha256=v["sha256"][:12])
                          for k, v in p["artifacts"].items()}, indent=2))
        print(f"preflight: {p['status']}")
        return 0
    return STAGES[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
