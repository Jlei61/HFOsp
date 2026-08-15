#!/usr/bin/env python
"""Passive monitor for the already-running formal data-driven SNN cohort.

This process NEVER launches, kills, restarts or re-queues a formal worker.  Its
whole job is to watch, at a 450 s cadence, and to leave a machine-readable trail
so the next agent (or the next shift) can see what happened without re-deriving
it:

  * the supervisor pid is alive
  * `controller.status` stage / completed / active / pending
  * per-unit SUCCESS / FAILED / RUNNING counts from `run_logs/*.status`
  * free disk and available memory against the run's own floors
  * whether the 29 runtime modules the worker hashes against commit 96618174 are
    still clean (a dirty module aborts every newly launched worker, so a drift
    here is the single most damaging thing that can happen to the run)

It exits when `controller.status` reaches COMPLETE / FAILED, when the supervisor
pid disappears, or when the deadline passes.  Desktop notification is sent on
completion, on supervisor death and on frozen-module drift.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FORMAL = ROOT / "results/topic4_sef_hfo/data_driven_snn_cohort_v1/formal"
EXPECTED_COMMIT = "96618174e9768f39a5e04ea699df1a43ba078a23"
FROZEN_MODULES = [
    "scripts/freeze_topic4_rev10_sa_spectral_field_candidates.py",
    "scripts/run_topic4_core_field_stage3_fit.py",
    "scripts/run_topic4_core_field_stage3_profile_round1.py",
    "scripts/run_topic4_data_driven_snn_cohort_formal_worker.py",
    "scripts/run_topic4_rev10_sa_spectral_field_worker.py",
    "scripts/run_topic4_rev9_factorial_worker.py",
    "scripts/run_topic4_rev9_node_kick_canary.py",
    "scripts/run_topic4_rev9l_forced_source_worker.py",
    "src/__init__.py", "src/interictal_propagation.py", "src/sef_hfo_observation.py",
    "src/topic4_cohort_fast_readout.py", "src/topic4_component_pair_edge.py",
    "src/topic4_continuous_field.py", "src/topic4_core_connectivity.py",
    "src/topic4_core_field.py", "src/topic4_core_field_cmaes.py",
    "src/topic4_core_field_profile.py", "src/topic4_core_field_rev9.py",
    "src/topic4_core_field_runner.py", "src/topic4_core_field_stage3.py",
    "src/topic4_forced_source_capacity.py", "src/topic4_graph_edge_flow.py",
    "src/topic4_local_connectivity.py", "src/topic4_rev9_edge_structure.py",
    "src/topic4_rev9_factorial.py", "src/topic4_rev9_local_response.py",
    "src/topic4_spatial_ou_drive.py", "src/topic4_spectral_field.py",
    "config/topic4_data_driven_snn_cohort_formal_v1.json",
]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _notify(title: str, body: str) -> None:
    try:
        subprocess.run(["notify-send", title, body], timeout=10, check=False)
    except Exception:
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError) as exc:
        return isinstance(exc, PermissionError)
    except Exception:
        return False


def _supervisor_pid() -> int | None:
    p = FORMAL / "supervisor.status"
    if not p.exists():
        return None
    for token in p.read_text().split():
        if token.startswith("pid="):
            return int(token.split("=", 1)[1])
    return None


def _frozen_drift() -> list[str]:
    """Any frozen runtime module that is dirty or no longer matches the commit."""
    drift = []
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", *FROZEN_MODULES],
        cwd=ROOT, text=True, capture_output=True).stdout.strip()
    if dirty:
        drift.extend(f"dirty:{line[3:]}" for line in dirty.splitlines())
    import hashlib
    for rel in FROZEN_MODULES:
        try:
            want = subprocess.run(["git", "show", f"{EXPECTED_COMMIT}:{rel}"],
                                  cwd=ROOT, capture_output=True, check=True).stdout
            have = (ROOT / rel).read_bytes()
            if hashlib.sha256(want).hexdigest() != hashlib.sha256(have).hexdigest():
                drift.append(f"content:{rel}")
        except Exception as exc:
            drift.append(f"unreadable:{rel}:{exc!r}")
    return drift


def _snapshot() -> dict:
    ctrl = (FORMAL / "controller.status").read_text().strip() if (
        FORMAL / "controller.status").exists() else "MISSING"
    counts: dict[str, int] = {}
    failed: list[str] = []
    for st in sorted((FORMAL / "run_logs").glob("*.status")):
        state = st.read_text().split()[0] if st.read_text().strip() else "EMPTY"
        counts[state] = counts.get(state, 0) + 1
        if state == "FAILED":
            failed.append(st.stem)
    du = os.statvfs(ROOT)
    free_gib = du.f_bavail * du.f_frsize / 2**30
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemAvailable", "MemTotal"):
            mem[k] = int(v.split()[0]) / 2**20
    pid = _supervisor_pid()
    return {
        "timestamp": _now(),
        "controller_status": ctrl,
        "supervisor_pid": pid,
        "supervisor_alive": bool(pid and _alive(pid)),
        "unit_status_counts": counts,
        "failed_units": failed,
        "n_worker_json": len(list((FORMAL / "workers").glob("*.json"))),
        "free_disk_gib": round(free_gib, 1),
        "mem_available_gib": round(mem.get("MemAvailable", 0), 1),
        "active_formal_workers": int(subprocess.run(
            ["pgrep", "-fc", "run_topic4_data_driven_snn_cohort_formal_worker.py"],
            capture_output=True, text=True).stdout.strip() or 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=450, help="300-600 s cadence")
    ap.add_argument("--deadline-hours", type=float, default=11.0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent /
                                        "formal_cohort_watch"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    trail = out / "watch_trail.jsonl"
    latest = out / "latest.json"
    deadline = time.time() + args.deadline_hours * 3600
    notified_drift = False

    while True:
        snap = _snapshot()
        drift = _frozen_drift()
        snap["frozen_module_drift"] = drift
        with open(trail, "a") as f:
            f.write(json.dumps(snap) + "\n")
        latest.write_text(json.dumps(snap, indent=2))
        print(json.dumps(snap), flush=True)

        if drift and not notified_drift:
            notified_drift = True
            _notify("Topic 4 formal cohort: FROZEN MODULE DRIFT",
                    f"{len(drift)} module(s) no longer match commit 96618174; "
                    "newly launched workers will abort.")
        state = snap["controller_status"].split()[0] if snap["controller_status"] else ""
        if state in {"COMPLETE", "FAILED"}:
            _notify("Topic 4 formal cohort finished", snap["controller_status"][:180])
            (out / "FINISHED").write_text(json.dumps(snap, indent=2))
            return
        if not snap["supervisor_alive"]:
            _notify("Topic 4 formal cohort: supervisor gone",
                    f"controller: {snap['controller_status'][:120]}")
            (out / "SUPERVISOR_GONE").write_text(json.dumps(snap, indent=2))
            return
        if time.time() > deadline:
            (out / "DEADLINE").write_text(json.dumps(snap, indent=2))
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
