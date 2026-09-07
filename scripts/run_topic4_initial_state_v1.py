#!/usr/bin/env python3
"""Resumable physical controller for initial-state v1: I1 screen -> gate -> I2.

Admission (plan §4): at most --max-branch-workers of this branch, at most
--max-combined SNN workers including the mainline, a memory floor plus a
per-worker reserve, and >= --disk-gib free. Never kills other jobs. Failed
units are retried on the same manifest unit (no seed redraw); runaway is a
physical outcome, not a failure.
"""
from __future__ import annotations

import argparse
import fcntl
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import topic4_initial_state_runtime as rt  # noqa: E402

WORKER = ROOT / "scripts/run_topic4_initial_state_worker.py"
ANALYZE = ROOT / "scripts/analyze_topic4_initial_state_v1.py"
GATE_PASS = "PERSISTENT_INITIAL_STATE_EFFECT_SINGLE_GRAPH"


def unit_complete(job, frozen_hashes):
    path = Path(job["output_json"])
    if not path.exists():
        return False
    try:
        record = rt.read(path)
    except ValueError:
        return False
    arrays = Path(record.get("arrays", {}).get("path", ""))
    return bool(
        record.get("status") == "INITIAL_STATE_WORKER_COMPLETE"
        and record.get("qualification_tag") is None
        and record.get("stage") == job["stage"]
        and record.get("dynamics_seed") == job["dynamics_seed"]
        and record.get("arm") == job["arm"]
        and record.get("topology_seed") == job["topology_seed"]
        and float(record.get("simulation", {}).get("duration_ms", -1)) == float(job["duration_ms"])
        and record.get("initial_voltage", {}).get("sha256") == job["initial_voltage_sha256"]
        and record.get("provenance", {}).get("frozen_manifest_sha256") in frozen_hashes
        and arrays.exists() and rt.sha(arrays) == record["arrays"]["sha256"])


def write_status(out, name, **payload):
    rt.write(out / "status.json", {"status": name, "updated_unix": time.time(),
                                   "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **payload})


def ensure_replication_network(design, frozen_path, out):
    """Build (once) the corrected graph for the replication topology seed."""
    frozen = rt.read(frozen_path)
    seed = int(design["stages"]["replication"]["topology_seed"])
    if str(seed) in frozen.get("replication_networks", {}):
        record = frozen["replication_networks"][str(seed)]
        if rt.sha(record["path"]) == record["sha256"]:
            return record
    import dataclasses
    for _path in (str(ROOT), str(ROOT / "src" / "snn_engine")):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from params import Params
    from src.topic4_core_field_runner import cache_key, connectivity_config, get_network
    from scripts.rebuild_topic4_autapse_corrected_reference import summarize
    reference = rt.network_record(design, design["stages"]["screen"]["topology_seed"])
    cfg = dict(reference["config"])
    kwargs = {k: v for k, v in cfg.items() if k in {f.name for f in dataclasses.fields(Params)}}
    kwargs["seed"] = seed
    params = Params(**kwargs)
    cache = out / "replication" / "network_cache"
    cache.mkdir(parents=True, exist_ok=True)
    started = time.time()
    net, n_e, n_i, hit = get_network(params, cfg["theta_EE_deg"], cfg["AR"], str(cache))
    stats = summarize(net)
    if any(r["self_edges"] or not r["exact_expected_degree"] for r in stats.values()):
        raise RuntimeError("replication topology failed structural validation")
    config = connectivity_config(params, cfg["theta_EE_deg"], cfg["AR"])
    path = cache / (cache_key(config) + ".pkl")
    record = {"path": str(path), "sha256": rt.sha(path), "topology_seed": seed,
              "status": "CORRECTED_GRAPH_VALIDATED", "pathways": stats, "config": config,
              "cache_hit": hit, "build_seconds": time.time() - started,
              "screen_reference_config": {k: v for k, v in cfg.items() if k != "seed"}}
    del net
    frozen.setdefault("replication_networks", {})[str(seed)] = record
    history = out / "frozen_manifest_history.json"
    rows = rt.read(history)["sha256"] if history.exists() else []
    rows.append(rt.sha(frozen_path))
    rt.write(history, {"sha256": rows})
    rt.write(frozen_path, frozen)
    rt.write(out / "replication" / "network_record.json", record)
    return record


def run_analysis(stage, out, log_name):
    log = open(out / "run_logs" / log_name, "w")
    code = subprocess.call([rt.PYTHON, str(ANALYZE), "--stage", stage], cwd=ROOT, env=rt.ENV,
                           stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if code:
        raise RuntimeError(f"analysis --stage {stage} failed with exit code {code}")


def run_script(script, arguments, out, log_name):
    log = open(out / "run_logs" / log_name, "w")
    code = subprocess.call([rt.PYTHON, str(ROOT / "scripts" / script), *arguments], cwd=ROOT, env=rt.ENV,
                           stdout=log, stderr=subprocess.STDOUT)
    log.close()
    if code:
        raise RuntimeError(f"{script} {' '.join(arguments)} failed with exit code {code}")


def run_stage(stage, design, out, frozen_path, args, jobs):
    logs = out / "run_logs"
    logs.mkdir(parents=True, exist_ok=True)
    history = out / "frozen_manifest_history.json"
    frozen_hashes = {rt.sha(frozen_path)} | set(rt.read(history)["sha256"] if history.exists() else [])
    stage_jobs = [j for j in jobs if j["stage"] == stage]
    pending = [j for j in stage_jobs if not unit_complete(j, frozen_hashes)]
    complete = len(stage_jobs) - len(pending)
    active, attempts, failed = {}, {}, {}
    frozen = rt.read(frozen_path)
    while pending or active:
        for stem, (proc, log) in list(active.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            job = active.pop(stem)[0].job
            if code == 0 and unit_complete(job, frozen_hashes):
                complete += 1
                continue
            attempts[stem] = attempts.get(stem, 0) + 1
            row = {"stem": stem, "exit_code": code, "attempt": attempts[stem], "time": time.time(),
                   "kind": "engineering_failure"}
            with open(out / "failures.jsonl", "a") as stream:
                stream.write(__import__("json").dumps(row) + "\n")
            if attempts[stem] < args.max_attempts:
                pending.insert(0, job)
            else:
                failed[stem] = row
        try:
            rt.verify_source_snapshot(frozen)
        except RuntimeError as exc:
            write_status(out, "FAILURE_FROZEN_SOURCE_CHANGED_NO_NEW_DISPATCH", stage=stage,
                         error=repr(exc), running=len(active))
            if not active:
                raise
            time.sleep(args.poll)
            continue
        counts = rt.running_worker_pids()
        n_mainline = len(counts["mainline"])
        n_branch = len(active)
        memory = rt.available_gib()
        disk = shutil.disk_usage(out).free / 1024 ** 3
        allowed = min(
            args.max_branch_workers - n_branch,
            args.max_combined - n_mainline - n_branch,
            int(max(0.0, (memory - args.floor_gib - args.reserve_gib * n_branch)) // args.reserve_gib),
            len(pending),
        )
        if disk < args.disk_gib:
            allowed = 0
        for _ in range(max(0, allowed)):
            job = pending.pop(0)
            log = open(logs / f"{job['stem']}.log", "a")
            command = [
                "/usr/bin/prlimit", f"--as={int(args.address_space_gib * 1024 ** 3)}", "--",
                rt.PYTHON, str(WORKER), "--frozen-manifest", str(frozen_path),
                "--stage", stage, "--dynamics-seed", str(job["dynamics_seed"]),
                "--arm", job["arm"], "--out-json", job["output_json"], "--out-npz", job["output_npz"],
            ]
            proc = subprocess.Popen(command, cwd=ROOT, env=rt.ENV, stdout=log, stderr=subprocess.STDOUT)
            proc.job = job
            active[job["stem"]] = (proc, log)
        name = f"RUNNING_{stage.upper()}" if active else "WAITING_RESOURCE_ADMISSION"
        write_status(out, name, stage=stage, total=len(stage_jobs), complete=complete,
                     running=len(active), pending=len(pending), failed=list(failed),
                     mainline_workers=n_mainline, memory_available_gib=memory, disk_free_gib=disk,
                     active=[{"stem": s, "pid": p.pid} for s, (p, _) in active.items()])
        if pending or active:
            time.sleep(args.poll)
    return {"total": len(stage_jobs), "complete": complete, "failed": failed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=rt.DESIGN_PATH)
    parser.add_argument("--max-branch-workers", type=int, default=2)
    parser.add_argument("--max-combined", type=int, default=8)
    parser.add_argument("--reserve-gib", type=float, default=20.0)
    parser.add_argument("--floor-gib", type=float, default=40.0)
    parser.add_argument("--disk-gib", type=float, default=30.0)
    parser.add_argument("--address-space-gib", type=float, default=26.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll", type=float, default=15.0)
    parser.add_argument("--stage", choices=("auto", "screen", "replication"), default="auto")
    args = parser.parse_args()
    if not 1 <= args.max_branch_workers <= 2:
        raise ValueError("the branch allows at most two physical workers")
    design = rt.load_design(args.design)
    out = rt.output_root(design)
    frozen_path = out / "frozen_manifest.json"
    guard = open(out / "controller.lock", "a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    qualification = rt.read(out / "qualification.json")
    if qualification["status"] != "I0_QUALIFICATION_PASS":
        raise RuntimeError("I0 qualification has not passed; no formal unit is dispatched")
    frozen = rt.read(frozen_path)
    if frozen["design_sha256"] != rt.sha(args.design):
        raise RuntimeError("frozen manifest does not match the design file")
    rt.verify_source_snapshot(frozen)
    jobs = rt.read(out / "jobs.json")["jobs"]
    (out / "run_logs").mkdir(parents=True, exist_ok=True)

    stages = ["screen", "replication"] if args.stage == "auto" else [args.stage]
    summary = {}
    if "screen" in stages:
        summary["screen"] = run_stage("screen", design, out, frozen_path, args, jobs)
        if summary["screen"]["failed"] or summary["screen"]["complete"] != summary["screen"]["total"]:
            write_status(out, "SCREEN_INCOMPLETE_ENGINEERING_FAILURES", **summary["screen"])
            raise RuntimeError(f"screen incomplete: {summary['screen']}")
        write_status(out, "ANALYZING_SCREEN", **summary["screen"])
        run_analysis("screen", out, "analysis_screen.log")
    effect = rt.read(out / "screen" / "primary_effect.json")
    verdict = effect["verdict"]
    if "replication" in stages:
        if verdict == GATE_PASS:
            write_status(out, "BUILDING_REPLICATION_GRAPH", screen_verdict=verdict)
            ensure_replication_network(design, frozen_path, out)
            summary["replication"] = run_stage("replication", design, out, frozen_path, args, jobs)
            if summary["replication"]["failed"]:
                write_status(out, "REPLICATION_INCOMPLETE_ENGINEERING_FAILURES", **summary["replication"])
                raise RuntimeError(f"replication incomplete: {summary['replication']}")
            write_status(out, "ANALYZING_REPLICATION", **summary["replication"])
            run_analysis("replication", out, "analysis_replication.log")
        else:
            summary["replication"] = {"skipped": True, "reason": f"screen verdict {verdict} does not open I2"}
    write_status(out, "ANALYZING_FINAL", screen_verdict=verdict, **{k: v for k, v in summary.items()})
    run_analysis("final", out, "analysis_final.log")
    rendered = []
    for stage in ("screen", "replication"):
        if (out / stage / "primary_effect.json").exists():
            write_status(out, f"RENDERING_{stage.upper()}", screen_verdict=verdict)
            run_script("render_topic4_initial_state_events_v1.py", ["--stage", stage], out, f"render_{stage}.log")
            rendered.append(stage)
    run_script("report_topic4_initial_state_v1.py", [], out, "report.log")
    final = rt.read(out / "primary_effect.json")["final"]
    write_status(out, "INITIAL_STATE_ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW",
                 initial_state_effect=final["initial_state_effect"],
                 patient_conditional_propagation=final["patient_conditional_propagation"],
                 stage_verdicts=final["stage_verdicts"], rendered_stages=rendered,
                 figures_human_review="PENDING", summary=summary)
    print({"status": "INITIAL_STATE_ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW", "screen_verdict": verdict,
           "initial_state_effect": final["initial_state_effect"], "summary": summary})


if __name__ == "__main__":
    main()
