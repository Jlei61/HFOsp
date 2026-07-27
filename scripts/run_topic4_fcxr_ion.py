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


def _not_yet(name):
    def _f(args):
        raise SystemExit(f"stage {name} is not implemented yet in this worktree")
    return _f


STAGES = {
    "b0-units": cmd_b0_units,
    "b0-provenance": cmd_b0_provenance,
    "b0-direction-power": cmd_b0_direction_power,
    "b1-smallnet": cmd_b1_smallnet,
    "b1-gate-h": cmd_b1_gate_h,
    "b1-select-f": _not_yet("b1-select-f"),
    "b2-bias": _not_yet("b2-bias"),
    "b2-validate": _not_yet("b2-validate"),
    "b2-confirm": _not_yet("b2-confirm"),
    "b2-adjudicate": _not_yet("b2-adjudicate"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=sorted(STAGES) + ["preflight"])
    ap.add_argument("--confirm-run", action="store_true",
                    help="required for any stage that starts a simulation")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--record-swap-baseline", action="store_true")
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
