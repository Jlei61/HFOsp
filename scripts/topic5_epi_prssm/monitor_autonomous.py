#!/usr/bin/env python3
"""Read-only monitor: controller liveness, task states and resource headroom."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT  # noqa: E402

LOGS = OUTPUT_ROOT / "logs"
TASKS = OUTPUT_ROOT / "jobs"


def snapshot(tag: str = "main") -> dict:
    suffix = "" if tag == "main" else f".{tag}"
    status_path = LOGS / f"controller{suffix}.status"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    alive = False
    pid = status.get("pid")
    if pid:
        alive = Path(f"/proc/{pid}").exists()
    stale = bool(status) and (time.time() - status.get("heartbeat", 0) > 300)
    states: dict[str, int] = {}
    failures = []
    for path in sorted(TASKS.glob("task_*.task.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        states[record.get("state", "?")] = states.get(record.get("state", "?"), 0) + 1
        if record.get("state") in ("FAILED", "OOM", "NAN", "INVALID_INPUT"):
            failures.append({"label": record.get("label"), "state": record.get("state"),
                             "returncode": record.get("returncode")})
    jobs: dict[str, int] = {}
    for path in sorted(TASKS.glob("*.status.json")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        jobs[record.get("state", "?")] = jobs.get(record.get("state", "?"), 0) + 1
    return {"controller_state": status.get("state"), "controller_pid": pid,
            "controller_alive": alive,
            "controller_stale": stale and not alive,
            "heartbeat_iso": status.get("heartbeat_iso"),
            "worker_limit": status.get("worker_limit"),
            "active": status.get("active", []),
            "task_states": states, "job_states": jobs, "failures": failures[:20],
            "system": status.get("system", {})}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    data = snapshot(args.tag)
    if args.json:
        print(json.dumps(data, indent=2))
        return
    print(f"controller : {data['controller_state']} pid={data['controller_pid']} "
          f"alive={data['controller_alive']} stale={data['controller_stale']} "
          f"heartbeat={data['heartbeat_iso']}")
    print(f"tasks      : {data['task_states']}")
    print(f"jobs       : {data['job_states']}")
    print(f"workers    : limit={data['worker_limit']} active={len(data['active'])}")
    system = data.get("system", {})
    if system:
        print(f"resources  : mem_avail={system.get('mem_available_gib', 0):.0f} GiB  "
              f"load={system.get('loadavg', [0])[0]:.1f}  "
              f"disk_free={system.get('disk_free_gib', 0):.0f} GiB")
    for item in data["active"][:40]:
        print(f"   running  {item['label']}  pid={item['pid']}")
    for failure in data["failures"]:
        print(f"   FAILED   {failure['label']} rc={failure['returncode']}")


if __name__ == "__main__":
    main()
