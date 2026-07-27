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


def _not_yet(name):
    def _f(args):
        raise SystemExit(f"stage {name} is not implemented yet in this worktree")
    return _f


STAGES = {
    "b0-units": cmd_b0_units,
    "b0-provenance": cmd_b0_provenance,
    "b0-direction-power": cmd_b0_direction_power,
    "b1-smallnet": _not_yet("b1-smallnet"),
    "b1-gate-h": _not_yet("b1-gate-h"),
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
