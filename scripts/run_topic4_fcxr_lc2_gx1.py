#!/usr/bin/env python3
"""FCXR-LC2-GX1 frozen entry/offset diagnostics."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import resource
import subprocess
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import run_topic4_fcxr_lc2_forks as F  # noqa: E402
import run_topic4_mz_fcxr as FCXR  # noqa: E402


OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core",
                   "gx1_entry_offset_diagnostics")
LC2 = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core",
                   "closed_loop_exploration")
SCRIPT_REL = "scripts/run_topic4_fcxr_lc2_gx1.py"
CONTRACT_VERSION = 2
G_SAT = 21.6
FAMILIES = {
    "H1": dict(tau_ms=522.0314431365073, theta_base=1.2594716548919684,
               false_latch_fraction=0.0),
    "H6": dict(tau_ms=632.4555320336759, theta_base=1.1122742295265198,
               false_latch_fraction=2.0 / 9.0),
}
THETA_SCALES = (1.0, 1.25)
RHO_FRACS = (0.025, 0.05, 0.075)
K_RATIO = 0.05
STRIP_ARMS = (
    ("healthy_low", 0.0, 0.0),
    ("susceptible_low", 0.15, 0.0),
    ("susceptible_high", 0.15, 2.0),
)
X_AVAILABILITIES = (1.0, 0.5, 0.1, 0.0)
RSS_SINGLE_FALLBACK_GIB = 7.214
MECHANISM_MODULES = ("src/snn_engine/mz_slow_vars.py",)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _source_sha():
    # The frozen-fork runner performs the actual full-resolution return-window
    # classification, so it is part of the executable scientific contract.
    h = hashlib.sha256()
    for rel in (SCRIPT_REL, "scripts/run_topic4_fcxr_lc2_forks.py"):
        with open(os.path.join(ROOT, rel), "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def _load(path):
    with open(path) as f:
        return json.load(f)


def _write(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    FCXR._write_json(path, payload)


def _meminfo():
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(mem_available_gib=d["MemAvailable"] / 1024.0 / 1024.0,
                swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0)


def _rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0


@contextmanager
def _stage_lock(stage):
    os.makedirs(OUT, exist_ok=True)
    fp = open(os.path.join(OUT, f".{stage}.lock"), "a+")
    try:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        fp.close()
        raise SystemExit(f"GX1 stage already running: {stage}") from exc
    try:
        yield
    finally:
        fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
        fp.close()


def build_strip_rows(noise_seed=401):
    rows = []
    for rho_frac in RHO_FRACS:
        for theta_scale in THETA_SCALES:
            for family in ("H1", "H6"):
                f = FAMILIES[family]
                theta = f["theta_base"] * theta_scale
                point_id = f"{family}_ts{theta_scale:g}_r{int(round(1000*rho_frac)):03d}"
                for arm, depletion, h_scale in STRIP_ARMS:
                    rows.append(dict(
                        index=len(rows), contract_version=CONTRACT_VERSION, stage="S1",
                        point_id=point_id, candidate_id=family, candidate_run_id=point_id,
                        family=family, arm=arm, tau_ms=float(f["tau_ms"]),
                        theta_base=float(f["theta_base"]), theta_scale=float(theta_scale),
                        theta=float(theta), k_ratio=K_RATIO, k=float(K_RATIO * theta),
                        rho_fraction=float(rho_frac), rho=float(G_SAT * rho_frac),
                        false_latch_fraction=float(f["false_latch_fraction"]),
                        D=float(depletion), h_init_scale=float(h_scale),
                        x_depletion=0.0, x_availability=1.0, T_ms=4000.0,
                        connection_seed=1, noise_seed=int(noise_seed), no_kick=True,
                        M=False, K=False, A=False, ELR=False, coop_A=0.0,
                    ))
    return rows


def _strip_point_pass(arms):
    by = {r["arm"]: r for r in arms}
    required = {x[0] for x in STRIP_ARMS}
    if set(by) != required:
        return dict(label="INCOMPLETE_POINT", pass_point=False)
    if any(r.get("numerical_failure", True) for r in arms):
        return dict(label="SELECTIVITY_STRIP_NUMERICAL_FAILURE", pass_point=False)
    healthy = by["healthy_low"].get("workpoint_label") == "INTERICTAL_WORKPOINT"
    susceptible_low = by["susceptible_low"].get("workpoint_label") == "INTERICTAL_WORKPOINT"
    hi = by["susceptible_high"]
    high_label = hi.get("workpoint_label") in ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
    state = hi.get("state_tail_1s", {})
    slope_ok = float(state.get("h_slope_per_s", -np.inf)) >= -0.05 * max(
        float(state.get("h_mean", 0.0)), float(hi.get("theta", 0.0)))
    high_ok = bool(high_label and float(state.get("ceiling_fraction", 1.0)) < 0.05 and slope_ok)
    passed = bool(healthy and susceptible_low and high_ok)
    if passed:
        label = "SELECTIVITY_POINT_PASS"
    elif not healthy:
        label = "HEALTHY_LOW_IGNITES"
    elif not susceptible_low:
        label = "SUSCEPTIBLE_LOW_IGNITES"
    elif not high_ok:
        label = "SUSCEPTIBLE_HIGH_NOT_MAINTAINED"
    else:
        label = "SELECTIVITY_POINT_UNRESOLVED"
    return dict(label=label, pass_point=passed, healthy_low_ok=healthy,
                susceptible_low_ok=susceptible_low, susceptible_high_ok=high_ok)


def _measured_single_rss_gib():
    """Spec section 7 sizes the second worker from the REMEASURED 4 s RSS, not the 3.16 s LC2 estimate."""
    path = os.path.join(OUT, "resource_log.jsonl")
    peaks = []
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                try:
                    peaks.append(float(json.loads(line)["peak_rss_gib"]))
                except (ValueError, KeyError, TypeError):
                    continue
    return max(peaks + [RSS_SINGLE_FALLBACK_GIB])


def _adjacent(a, b):
    if a["family"] != b["family"]:
        return False
    ri, rj = RHO_FRACS.index(a["rho_fraction"]), RHO_FRACS.index(b["rho_fraction"])
    ti, tj = THETA_SCALES.index(a["theta_scale"]), THETA_SCALES.index(b["theta_scale"])
    return abs(ri - rj) + abs(ti - tj) == 1


def aggregate_strip(rows):
    point_rows = []
    for pid in sorted({r["point_id"] for r in rows}):
        arms = [r for r in rows if r["point_id"] == pid]
        meta = arms[0]
        point_rows.append(dict(point_id=pid, family=meta["family"],
                               rho_fraction=float(meta["rho_fraction"]),
                               theta_scale=float(meta["theta_scale"]),
                               theta=float(meta["theta"]), rho=float(meta["rho"]),
                               arms=arms, **_strip_point_pass(arms)))
    passed = [p for p in point_rows if p["pass_point"]]
    window_ids = set()
    for i, a in enumerate(passed):
        for b in passed[i + 1:]:
            if _adjacent(a, b):
                window_ids.update((a["point_id"], b["point_id"]))
    for p in point_rows:
        p["in_adjacent_window"] = p["point_id"] in window_ids
    n_failed = sum(p["label"] in ("SELECTIVITY_STRIP_NUMERICAL_FAILURE", "INCOMPLETE_POINT")
                   for p in point_rows)
    if window_ids:
        verdict = "NATURAL_SELECTIVITY_WINDOW_CANDIDATE"
    elif passed:
        verdict = "ISOLATED_SELECTIVITY_POINT"
    elif n_failed:
        verdict = "SELECTIVITY_STRIP_NUMERICAL_FAILURE"
    else:
        verdict = "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP"
    return dict(verdict=verdict, n_points=len(point_rows), n_pass=len(passed),
                n_window_points=len(window_ids), n_numerical_failure_points=n_failed,
                point_rows=point_rows)


def select_strip_anchor(aggregate):
    pts = [p for p in aggregate["point_rows"] if p.get("in_adjacent_window")]
    if not pts:
        return None
    order = {"H1": 0, "H6": 1}
    pts.sort(key=lambda p: (p["rho_fraction"], -p["theta_scale"], order[p["family"]], p["point_id"]))
    return pts[0]


def build_x_rows(strip_aggregate, noise_seed=401):
    anchor = select_strip_anchor(strip_aggregate)
    if anchor is None:
        family = "H6"; tau = FAMILIES[family]["tau_ms"]
        theta = FAMILIES[family]["theta_base"]; theta_scale = 1.0
        rho_fraction = 0.10; anchor_source = "archived_H6_k05_r10_no_strip_window"
        point_id = "H6_k05_r10"
    else:
        family = anchor["family"]; tau = FAMILIES[family]["tau_ms"]
        theta = anchor["theta"]; theta_scale = anchor["theta_scale"]
        rho_fraction = anchor["rho_fraction"]; anchor_source = "natural_selectivity_window_anchor"
        point_id = anchor["point_id"]
    rows = []
    for x in X_AVAILABILITIES:
        rows.append(dict(
            index=len(rows), contract_version=CONTRACT_VERSION, stage="X1",
            point_id=point_id, candidate_id=family, candidate_run_id=f"X_{point_id}",
            family=family, arm=f"x_{x:g}", anchor_source=anchor_source,
            tau_ms=float(tau), theta_base=float(FAMILIES[family]["theta_base"]),
            theta_scale=float(theta_scale), theta=float(theta), k_ratio=K_RATIO,
            k=float(K_RATIO * theta), rho_fraction=float(rho_fraction),
            rho=float(G_SAT * rho_fraction), D=0.15, h_init_scale=2.0,
            x_depletion=float(1.0 - x), x_availability=float(x),
            required_low_min_ms=2000.0,
            T_ms=float(max(5000.0, 8.0 * tau)), connection_seed=1,
            noise_seed=int(noise_seed), no_kick=True, M=False, K=False, A=False,
            ELR=False, coop_A=0.0,
        ))
    return rows


def archived_relay_availabilities(frozen_map, point_id):
    """Read the comparison range from the same-anchor archived fork evidence."""
    values = sorted({float(r["x_availability"]) for r in frozen_map.get("rows", [])
                     if r.get("candidate_run_id") == point_id
                     and 0.0 < float(r.get("x_availability", 1.0)) < 1.0}, reverse=True)
    if not values:
        raise ValueError(f"no archived relay load at anchor {point_id}")
    return values


def classify_x_authority(rows, archived_availabilities):
    by = {float(r["x_availability"]): r for r in rows}
    if set(by) != set(X_AVAILABILITIES):
        return dict(verdict="X_AUTHORITY_UNRESOLVED", reason="incomplete_manifest")
    if any(r.get("numerical_failure", True) for r in rows):
        return dict(verdict="X_AUTHORITY_UNRESOLVED", reason="numerical_failure")
    high_labels = ("FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
    if by[1.0].get("workpoint_label") not in high_labels:
        return dict(verdict="X_AUTHORITY_UNRESOLVED", reason="anchor_high_not_established")
    returning = sorted(x for x, r in by.items()
                       if r.get("required_low_workpoint_label") == "INTERICTAL_WORKPOINT")
    nonzero = [x for x in returning if x > 0.0]
    archived_availabilities = [float(v) for v in archived_availabilities]
    if not archived_availabilities or any(not (0.0 < v < 1.0) for v in archived_availabilities):
        return dict(verdict="X_AUTHORITY_UNRESOLVED", reason="invalid_archived_load_range")
    physiological_floor = min(archived_availabilities)
    if 0.0 not in returning:
        verdict = "H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN"
    elif any(x >= physiological_floor for x in returning):
        verdict = "X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH"
    else:
        # x=0 returns but every returning arm sits below the archived physiological loads: the path is
        # reachable and the observed dynamic range is what is missing (spec 4.3 bullet 1).
        verdict = "X_PATH_REACHABLE_RANGE_INSUFFICIENT"
    return dict(verdict=verdict, returning_availabilities=returning,
                smallest_tested_availability_returning=min(returning) if returning else None,
                largest_tested_availability_returning=max(returning) if returning else None,
                smallest_nonzero_availability_returning=min(nonzero) if nonzero else None,
                archived_physiological_availabilities=archived_availabilities)


def _archived_x_range(rows):
    lock = _load(os.path.join(OUT, "execution_lock.json"))
    frozen = _load(lock["artifacts"]["frozen_map"]["path"])
    point_ids = {str(r["point_id"]) for r in rows}
    if len(point_ids) != 1:
        raise ValueError(f"X rows do not share one anchor: {sorted(point_ids)}")
    return archived_relay_availabilities(frozen, next(iter(point_ids)))


def _cell_path(stage, row):
    folder = "strip_cells" if stage == "S1" else "x_cells"
    return os.path.join(OUT, folder,
                        f"{row['point_id']}__{row['arm']}__n{row['noise_seed']}.json")


def _run_row(stage, row):
    path = _cell_path(stage, row)
    if os.path.isfile(path):
        prior = _load(path)
        if (prior.get("gx1_contract_version") == CONTRACT_VERSION and
                prior.get("gx1_source_sha256") == row.get("gx1_source_sha256")):
            return prior
    out = F._run_row(row)
    out["gx1_contract_version"] = CONTRACT_VERSION
    out["gx1_source_sha256"] = row["gx1_source_sha256"]
    _write(path, out)
    return out


def _append_resource(stage, row):
    payload = dict(stage=stage, point_id=row["point_id"], arm=row["arm"],
                   peak_rss_gib=row["peak_rss_gib"], wall_s=row["wall_s"],
                   mem=_meminfo(), finished=_now())
    with open(os.path.join(OUT, "resource_log.jsonl"), "a") as f:
        f.write(json.dumps(payload) + "\n")


def _run_all(stage, rows, workers):
    before = _meminfo()
    if before["mem_available_gib"] < 96.0:
        raise SystemExit(f"OOM safety stop: {before}")
    rss_single = _measured_single_rss_gib()
    if workers == 2 and before["mem_available_gib"] < 96.0 + 2 * 1.35 * rss_single:
        raise SystemExit(f"OOM safety stop for second worker (rss_single={rss_single:.3f} GiB): {before}")
    running = os.path.join(OUT, f"{stage}_RUNNING.json")
    done = os.path.join(OUT, f"{stage}_DONE.json")
    failed = os.path.join(OUT, f"{stage}_FAILED.json")
    _write(running, dict(stage=stage, pid=os.getpid(), workers=workers,
                         resource_before=before, started=_now()))
    results = []
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_run_row, stage, row): row for row in rows}
            for j, fut in enumerate(as_completed(futs), 1):
                out = fut.result(); results.append(out); _append_resource(stage, out)
                print(f"[{stage}] {j}/{len(rows)} {out['point_id']} {out['arm']} "
                      f"wp={out['workpoint_label']} post={out['required_low_workpoint_label']} "
                      f"tail={out['state_tail_1s']['rate_mean_hz']:.1f}Hz "
                      f"RSS={out['peak_rss_gib']:.2f}GiB", flush=True)
                now = _meminfo()
                if now["swap_used_mib"] - before["swap_used_mib"] >= 512.0:
                    raise MemoryError(f"swap hard stop: before={before}, now={now}")
        results.sort(key=lambda r: int(r["index"]))
        _write(done, dict(stage=stage, status="COMPLETE", n_rows=len(results),
                          resource_before=before, resource_after=_meminfo(), finished=_now()))
        if os.path.exists(running):
            os.remove(running)
        return results
    except Exception as exc:
        _write(failed, dict(stage=stage, error=repr(exc), traceback=traceback.format_exc(), failed=_now()))
        raise


def cmd_lock(_args):
    os.makedirs(OUT, exist_ok=True)
    closeout = os.path.join(LC2, "candidate_verdict.json")
    r1 = os.path.join(LC2, "r1_resegmentation_summary.json")
    frozen = os.path.join(LC2, "frozen_fork_map.json")
    for path in (closeout, r1, frozen, F.P_FIELD, F.BASELINE_CONTRACT):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    old_lock = _load(os.path.join(LC2, "execution_lock.json"))
    engine = {}
    for rel, expected in old_lock["engine_hashes"].items():
        got = _sha256(os.path.join(ROOT, rel))
        if got != expected:
            raise RuntimeError(f"blessed engine drift: {rel}")
        engine[rel] = got
    payload = dict(status="LOCKED", contract_version=CONTRACT_VERSION, code_head=_git_head(),
                   source_sha256=_source_sha(), artifacts={
                       "lc2_closeout": dict(path=closeout, sha256=_sha256(closeout)),
                       "r1_candidates": dict(path=r1, sha256=_sha256(r1)),
                       "frozen_map": dict(path=frozen, sha256=_sha256(frozen)),
                       "p_field": dict(path=F.P_FIELD, sha256=_sha256(F.P_FIELD)),
                       "baseline_contract": dict(path=F.BASELINE_CONTRACT,
                                                 sha256=_sha256(F.BASELINE_CONTRACT)),
                   }, engine_hashes=engine,
                   # The blessed set does not cover the module that implements the H gate and the frozen
                   # relay, i.e. the mechanism actually under test.  Pin it here too.
                   mechanism_module_hashes={rel: _sha256(os.path.join(ROOT, rel))
                                            for rel in MECHANISM_MODULES},
                   strip=dict(families=FAMILIES,
                       theta_scales=THETA_SCALES, rho_fracs=RHO_FRACS, k_ratio=K_RATIO,
                       arms=STRIP_ARMS, T_ms=4000.0, noise_seed=401),
                   x_probe=dict(availabilities=X_AVAILABILITIES, noise_seed=401,
                                duration_rule="max(5000,8*tau_H)"),
                   locked_out=["dynamic_Z", "dynamic_X", "lifecycle", "M", "K", "A", "ELR",
                               "kick", "new_EE_edges"], resource_at_lock=_meminfo(), locked_at=_now())
    _write(os.path.join(OUT, "execution_lock.json"), payload)
    print(json.dumps(payload, indent=2))


def _manifest_rows(name):
    p = os.path.join(OUT, name)
    if not os.path.isfile(p):
        raise SystemExit(f"missing manifest: {p}")
    m = _load(p)
    if m.get("source_sha256") != _source_sha():
        raise SystemExit("runner source drifted after manifest lock")
    return m["rows"]


def _aggregate_rows(name):
    """Read-only re-derivation over already simulated cells.

    Unlike `_manifest_rows` this does not refuse a drifted runner: re-classifying archived cells after a
    post-hoc classifier correction must stay possible.  The drift is recorded in the payload instead of
    being hidden, and no simulation can start from this path.
    """
    m = _load(os.path.join(OUT, name))
    return m["rows"], dict(manifest_source_sha256=m.get("source_sha256"),
                           current_source_sha256=_source_sha())


def cmd_strip_manifest(_args):
    lock = _load(os.path.join(OUT, "execution_lock.json"))
    if lock.get("source_sha256") != _source_sha():
        raise SystemExit("runner source drifted after execution lock")
    rows = build_strip_rows()
    for row in rows:
        row["gx1_source_sha256"] = lock["source_sha256"]
    payload = dict(stage="S1", status="LOCKED", contract_version=CONTRACT_VERSION,
                   code_head=_git_head(), source_sha256=lock["source_sha256"], n_rows=len(rows),
                   n_points=len({r["point_id"] for r in rows}), rows=rows, created=_now())
    _write(os.path.join(OUT, "selectivity_strip_manifest.json"), payload)
    print(json.dumps(dict(status="LOCKED", n_rows=len(rows), n_points=payload["n_points"]), indent=2))


def cmd_one(stage, manifest, args):
    if not args.confirm_run:
        raise SystemExit("--confirm-run is required")
    FCXR._assert_engine_blessed()
    rows = _manifest_rows(manifest)
    print(json.dumps(_run_row(stage, rows[int(args.index)]), indent=2))


def cmd_strip_all(args):
    if not args.confirm_run:
        raise SystemExit("--confirm-run is required")
    FCXR._assert_engine_blessed()
    rows = _manifest_rows("selectivity_strip_manifest.json")
    with _stage_lock("S1"):
        results = _run_all("S1", rows, int(args.workers))
        aggregate = aggregate_strip(results)
        payload = dict(stage="S1", status="COMPLETE", n_rows=len(results), **aggregate,
                       finished=_now())
        _write(os.path.join(OUT, "selectivity_strip.json"), payload)


def _reload_cells(stage, manifest):
    rows, provenance = _aggregate_rows(manifest)
    results = []
    for row in rows:
        path = _cell_path(stage, row)
        if not os.path.isfile(path):
            raise SystemExit(f"missing {stage} cell: {path}")
        results.append(_load(path))
    return results, provenance


def cmd_strip_aggregate(_args):
    results, provenance = _reload_cells("S1", "selectivity_strip_manifest.json")
    payload = dict(stage="S1", status="COMPLETE", n_rows=len(results),
                   **aggregate_strip(results), source_provenance=provenance, finished=_now())
    _write(os.path.join(OUT, "selectivity_strip.json"), payload)
    print(json.dumps({k: payload[k] for k in ("verdict", "n_points", "n_pass", "n_window_points",
                                              "n_numerical_failure_points")}, indent=2))


def cmd_x_aggregate(_args):
    results, provenance = _reload_cells("X1", "x_authority_manifest.json")
    results.sort(key=lambda r: int(r["index"]))
    verdict = classify_x_authority(results, _archived_x_range(results))
    payload = dict(stage="X1", status="COMPLETE", n_rows=len(results), rows=results,
                   **verdict, source_provenance=provenance, finished=_now())
    _write(os.path.join(OUT, "x_authority_map.json"), payload)
    print(json.dumps({k: v for k, v in verdict.items() if k != "rows"}, indent=2))


def cmd_x_manifest(_args):
    lock = _load(os.path.join(OUT, "execution_lock.json"))
    if lock.get("source_sha256") != _source_sha():
        raise SystemExit("runner source drifted after execution lock")
    strip = _load(os.path.join(OUT, "selectivity_strip.json"))
    rows = build_x_rows(strip)
    source = lock["source_sha256"]
    for row in rows:
        row["gx1_source_sha256"] = source
    payload = dict(stage="X1", status="LOCKED", contract_version=CONTRACT_VERSION,
                   code_head=_git_head(), source_sha256=source, strip_verdict=strip["verdict"],
                   n_rows=len(rows), rows=rows, created=_now())
    _write(os.path.join(OUT, "x_authority_manifest.json"), payload)
    print(json.dumps(dict(status="LOCKED", anchor=rows[0]["point_id"], n_rows=len(rows)), indent=2))


def cmd_x_all(args):
    if not args.confirm_run:
        raise SystemExit("--confirm-run is required")
    FCXR._assert_engine_blessed()
    rows = _manifest_rows("x_authority_manifest.json")
    with _stage_lock("X1"):
        results = _run_all("X1", rows, int(args.workers))
        verdict = classify_x_authority(results, _archived_x_range(results))
        payload = dict(stage="X1", status="COMPLETE", n_rows=len(results), rows=results,
                       **verdict, finished=_now())
        _write(os.path.join(OUT, "x_authority_map.json"), payload)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    sub.add_parser("strip-manifest")
    p = sub.add_parser("strip-one"); p.add_argument("--index", type=int, required=True); p.add_argument("--confirm-run", action="store_true")
    p = sub.add_parser("strip-all"); p.add_argument("--workers", type=int, choices=(1, 2), default=1); p.add_argument("--confirm-run", action="store_true")
    sub.add_parser("strip-aggregate")
    sub.add_parser("x-aggregate")
    sub.add_parser("x-manifest")
    p = sub.add_parser("x-one"); p.add_argument("--index", type=int, required=True); p.add_argument("--confirm-run", action="store_true")
    p = sub.add_parser("x-all"); p.add_argument("--workers", type=int, choices=(1, 2), default=1); p.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "lock": cmd_lock(args)
    elif args.cmd == "strip-manifest": cmd_strip_manifest(args)
    elif args.cmd == "strip-one": cmd_one("S1", "selectivity_strip_manifest.json", args)
    elif args.cmd == "strip-all": cmd_strip_all(args)
    elif args.cmd == "strip-aggregate": cmd_strip_aggregate(args)
    elif args.cmd == "x-aggregate": cmd_x_aggregate(args)
    elif args.cmd == "x-manifest": cmd_x_manifest(args)
    elif args.cmd == "x-one": cmd_one("X1", "x_authority_manifest.json", args)
    else: cmd_x_all(args)


if __name__ == "__main__":
    main()
