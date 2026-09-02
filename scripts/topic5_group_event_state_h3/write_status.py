#!/usr/bin/env python3
"""One file that says what has finished, what is running, and what failed.

Deliberately derived from the artefacts on disk rather than from a counter kept in
memory: a status that is only true while the process lives is not a status.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state_h3.io import write_json_atomic  # noqa: E402

OUT_ROOT = ROOT / "results/epi_prssm/group_event_state/v0_2/h3"
LEASE = ROOT / "results/epi_prssm/group_event_state/v0_2/shared/resource_leases/agent_c.json"


def _stage(subdir: str) -> dict:
    path = OUT_ROOT / "machine" / subdir
    if not path.exists():
        return {"status": "not_started", "n_ok": 0, "n_files": 0}
    files = sorted(path.glob("*.json"))
    ok, bad = [], []
    for f in files:
        try:
            payload = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            bad.append(f.name)
            continue
        (ok if payload.get("status") == "ok" else bad).append(f.name)
    return {
        "status": "ok" if ok and not bad else ("partial" if ok else "not_started"),
        "n_ok": len(ok),
        "n_files": len(files),
        "unreadable_or_failed": bad[:10],
        "directory": str(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    lease = {}
    if LEASE.exists():
        try:
            lease = json.loads(LEASE.read_text())
        except (json.JSONDecodeError, OSError):
            lease = {}
    owner_pid = lease.get("pid")
    owner_alive = False
    if isinstance(owner_pid, int):
        try:
            os.kill(owner_pid, 0)
            owner_alive = True
        except (ProcessLookupError, PermissionError):
            owner_alive = False

    payload = {
        "agent": "agent_c",
        "role": "H3 event feedback",
        "tag": args.tag,
        "written_epoch": time.time(),
        "queue_owner": {
            "pid": owner_pid,
            "pgid": lease.get("pgid"),
            "alive": owner_alive,
            "phase": lease.get("phase"),
            "n_pending": lease.get("n_pending"),
            "n_running": lease.get("n_running"),
            "n_failed": lease.get("n_failed"),
            "n_resource_failed": lease.get("n_resource_failed"),
            "heartbeat_epoch": lease.get("heartbeat_epoch"),
        },
        "stages": {
            "models": _stage(args.tag),
            "impulse": _stage(f"impulse_{args.tag}"),
            "perturbation": _stage(f"perturbation_{args.tag}"),
            "innovation": _stage(f"innovation_{args.tag}"),
        },
        "support_artifacts": {
            name: (OUT_ROOT / "support" / name).exists()
            for name in (
                "coverage_support_primary.json",
                "coverage_support_postictal0.json",
                "background_table.json",
                "background_coverage.json",
                "event_features.json",
                "training_protocol_freeze.md",
                "seed_null_preregistration.md",
            )
        },
        "process_management": "exact PID/PGID from this lease only; never pkill -f",
    }
    write_json_atomic(payload, OUT_ROOT / "STATUS.json")
    print(json.dumps(payload["stages"], indent=2))
    print(f"queue owner alive: {owner_alive} (pid {owner_pid})")


if __name__ == "__main__":
    main()
