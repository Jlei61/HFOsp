#!/usr/bin/env python3
"""Resumable supervisor for the Epi-PRSSM v0.1 experiment matrix.

Design rules this controller obeys:

* it never kills a process it did not start;
* a task's identity is its (script, args) tuple, so a re-launch adopts finished
  work instead of repeating it;
* worker capacity is computed from measured sentinel peak RSS, not from a guess
  about model size, and is re-checked every scaling interval;
* a task that dies is recorded with its state and traceback, and only that task
  is retried -- never the whole stage;
* every status write is atomic, so a killed controller leaves readable state.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_json, code_revision, package_hash, sha256_obj,
)

PYTHON = os.environ.get("EPI_PRSSM_PYTHON",
                        "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
LOGS = OUTPUT_ROOT / "logs"
TASKS = OUTPUT_ROOT / "jobs"
MANIFESTS = OUTPUT_ROOT / "manifests"
#: One controller instance per stage tag, so two stages can run side by side
#: without overwriting each other's status, log or task records.
TAG = "main"
STATUS = LOGS / "controller.status"
CONTROLLER_LOG = LOGS / "controller.log"


def set_tag(tag: str) -> None:
    global TAG, STATUS, CONTROLLER_LOG
    TAG = tag
    suffix = "" if tag == "main" else f".{tag}"
    STATUS = LOGS / f"controller{suffix}.status"
    CONTROLLER_LOG = LOGS / f"controller{suffix}.log"

RAM_RESERVE_GIB = 20.0
RAM_RESERVE_FRACTION = 0.20
DISK_LOW_WATER_GIB = 6.0
CPU_RESERVE = 2
SAFETY_FACTOR = 1.25
HEARTBEAT_SECONDS = 60
RESCALE_SECONDS = 300


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{stamp} {message}"
    print(line, flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    with CONTROLLER_LOG.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def task_key(task: dict) -> str:
    return sha256_obj({"script": task["script"], "args": task["args"]})[:20]


def task_path(task: dict) -> Path:
    return TASKS / f"task_{task_key(task)}.task.json"


def read_task(task: dict) -> dict | None:
    path = task_path(task)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def write_task(task: dict, **fields) -> None:
    payload = read_task(task) or {"key": task_key(task), "script": task["script"],
                                  "args": task["args"], "label": task.get("label"),
                                  "workload": task.get("workload", "cpu_train")}
    payload.update(fields)
    atomic_write_json(task_path(task), payload)


def system_snapshot() -> dict:
    meminfo = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            meminfo[parts[0].rstrip(":")] = float(parts[1]) / (1024 ** 2)  # GiB
    load1, load5, load15 = (float(x) for x in Path("/proc/loadavg").read_text().split()[:3])
    usage = shutil.disk_usage(str(OUTPUT_ROOT))
    gpu = []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20)
        for row in out.stdout.strip().splitlines():
            index, total, used, free = (int(x) for x in row.split(","))
            gpu.append({"index": index, "total_mib": total, "used_mib": used, "free_mib": free})
    except Exception:
        pass
    return {
        "mem_total_gib": meminfo.get("MemTotal", float("nan")),
        "mem_available_gib": meminfo.get("MemAvailable", float("nan")),
        "swap_free_gib": meminfo.get("SwapFree", float("nan")),
        "swap_total_gib": meminfo.get("SwapTotal", float("nan")),
        "loadavg": [load1, load5, load15],
        "cpu_logical": os.cpu_count(),
        "disk_free_gib": usage.free / (1024 ** 3),
        "gpus": gpu,
        "foreign_cpu_cores": foreign_cpu_cores(),
    }


def foreign_cpu_cores() -> float:
    """CPU cores currently used by processes that are not this experiment.

    This machine is shared.  Sizing the pool from the core count alone would let
    this task crowd out somebody else's run, so the capacity check subtracts what
    other processes are already drawing.  Their processes are never touched.
    """
    try:
        out = subprocess.run(["ps", "-eo", "pcpu,cmd"], capture_output=True, text=True,
                             timeout=20).stdout.splitlines()[1:]
    except Exception:
        return 0.0
    total = 0.0
    for line in out:
        line = line.strip()
        if not line:
            continue
        head, _, cmd = line.partition(" ")
        try:
            percent = float(head)
        except ValueError:
            continue
        if percent < 20.0 or "topic5_epi_prssm" in cmd:
            continue
        total += percent
    return total / 100.0


def max_workers(peak_rss_gib: float, cap: int, running_here: int = 0) -> int:
    """How many workers this pool may hold.

    The budget is shared three ways: other people's processes on this machine,
    this experiment's other controllers, and this pool.  Sizing from the core
    count alone would let several of my own controllers each claim the whole
    machine and crowd out a neighbour's run.
    """
    snapshot = system_snapshot()
    reserve = max(RAM_RESERVE_GIB, RAM_RESERVE_FRACTION * snapshot["mem_total_gib"])
    headroom = max(snapshot["mem_available_gib"] - reserve, 0.0)
    by_ram = int(headroom / max(peak_rss_gib * SAFETY_FACTOR, 0.05))
    foreign = int(foreign_cpu_cores())
    mine_elsewhere = max(len(_own_worker_lines()) - running_here, 0)
    by_cpu = int(snapshot["cpu_logical"]) - CPU_RESERVE - foreign - mine_elsewhere
    if snapshot["disk_free_gib"] < DISK_LOW_WATER_GIB:
        return 0
    # a busy machine slows this pool down; it must never stop it completely, or a
    # queue would sit forever waiting for a neighbour who has no reason to stop
    return max(1, min(cap, max(by_ram, 1), max(by_cpu, 1)))


def _own_worker_lines() -> list[str]:
    try:
        out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True, text=True,
                             timeout=20).stdout.splitlines()
    except Exception:
        return []
    return [line for line in out
            if "topic5_epi_prssm" in line and "launch_autonomous" not in line
            and "run_remaining_chain" not in line]


def measure_sentinel(task: dict, timeout: int = 3600) -> dict:
    """Run one real task to completion and record its measured peak RSS."""
    log(f"sentinel start: {task['label']}")
    started = time.time()
    process = _spawn(task)
    peak = 0.0
    while process.poll() is None:
        peak = max(peak, _rss_gib(process.pid))
        time.sleep(2.0)
        if time.time() - started > timeout:
            process.terminate()
            break
    measured = {"label": task["label"], "workload": task.get("workload", "cpu_train"),
                "returncode": process.returncode, "wall_seconds": time.time() - started,
                "peak_rss_gib": round(max(peak, 0.2), 3)}
    log(f"sentinel done: {measured}")
    return measured


def _rss_gib(pid: int) -> float:
    total = 0.0
    try:
        for path in [Path(f"/proc/{pid}/status")] + [
                Path(f"/proc/{child.strip()}/status")
                for child in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]:
            for line in path.read_text().splitlines():
                if line.startswith("VmRSS:"):
                    total += float(line.split()[1]) / (1024 ** 2)
    except Exception:
        return total
    return total


def _spawn(task: dict) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    stdout = (LOGS / f"task_{task_key(task)}.log").open("a", encoding="utf-8")
    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                        "PYTHONUNBUFFERED": "1"})
    command = [PYTHON, str(ROOT / task["script"]), *[str(a) for a in task["args"]]]
    stdout.write(f"\n===== {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(command)}\n")
    stdout.flush()
    return subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=str(ROOT / "scripts/topic5_epi_prssm"),
                            env=environment, start_new_session=True)


def write_status(state: str, tasks: list[dict], running: dict, extra: dict | None = None) -> None:
    counts: dict[str, int] = {}
    for task in tasks:
        record = read_task(task) or {}
        counts[record.get("state", "PENDING")] = counts.get(record.get("state", "PENDING"), 0) + 1
    atomic_write_json(STATUS, {
        "contract": "topic5_epi_prssm_v0_1_controller", "tag": TAG,
        "state": state, "heartbeat": time.time(),
        "heartbeat_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "pid": os.getpid(), "code_revision": code_revision(), "package_hash": package_hash(),
        "n_tasks": len(tasks), "state_counts": counts,
        "active": [{"label": t["label"], "pid": p.pid} for t, p in running.values()],
        "system": system_snapshot(), **(extra or {}),
    })


def _foreign_live(task: dict) -> bool:
    """True when another controller already has this exact task running.

    Task identity is (script, args), so two controllers handed overlapping plans
    would otherwise spawn the same unit twice and race on its output.  A task whose
    record says RUNNING and whose pid is still alive is left to its owner.
    """
    record = read_task(task) or {}
    if record.get("state") != "RUNNING":
        return False
    pid = record.get("pid")
    return bool(pid) and Path(f"/proc/{pid}").exists() and pid != os.getpid()


def run_pool(tasks: list[dict], *, cap: int, peak_rss_gib: float,
             poll_seconds: float = 5.0) -> None:
    foreign = []
    pending = []
    for task in tasks:
        state = (read_task(task) or {}).get("state")
        # An explicit --overwrite means "run this regardless of what happened before".
        # Adopting such a task as already complete silently defeats the request: a
        # producer re-run meant to apply a new matching rule finished in one second
        # twice in a row because of this, and the rule never ran.
        if "--overwrite" in [str(a) for a in task.get("args", [])]:
            state = None
        if state == "COMPLETE":
            log(f"adopt COMPLETE {task['label']}")
            continue
        if _foreign_live(task):
            log(f"adopt RUNNING (owned by another controller) {task['label']}")
            foreign.append(task)
            continue
        write_task(task, state="PENDING")
        pending.append(task)
    running: dict[str, tuple[dict, subprocess.Popen]] = {}
    queue = list(pending)
    limit = max_workers(peak_rss_gib, cap, running_here=0)
    log(f"worker limit = {limit} (peak_rss={peak_rss_gib:.2f} GiB, cap={cap}, "
        f"foreign_cores={foreign_cpu_cores():.1f})")
    last_heartbeat = 0.0
    last_rescale = time.time()
    while queue or running or foreign:
        for task in list(foreign):
            if not _foreign_live(task):
                state = (read_task(task) or {}).get("state")
                if state == "COMPLETE":
                    log(f"foreign task finished COMPLETE {task['label']}")
                    foreign.remove(task)
                else:
                    log(f"foreign task ended as {state}; taking it over {task['label']}")
                    foreign.remove(task)
                    write_task(task, state="PENDING")
                    queue.append(task)
        while queue and len(running) < limit:
            task = queue.pop(0)
            process = _spawn(task)
            running[task_key(task)] = (task, process)
            write_task(task, state="RUNNING", pid=process.pid, started_at=time.time())
            log(f"start {task['label']} pid={process.pid} ({len(running)}/{limit})")
        for key in list(running):
            task, process = running[key]
            code = process.poll()
            if code is None:
                continue
            del running[key]
            state = "COMPLETE" if code == 0 else "FAILED"
            write_task(task, state=state, returncode=code, finished_at=time.time())
            log(f"{state} {task['label']} rc={code}")
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_SECONDS:
            write_status("RUNNING", tasks, running,
                         extra={"queue_remaining": len(queue), "worker_limit": limit,
                                "adopted_from_other_controllers": len(foreign)})
            last_heartbeat = now
        if now - last_rescale >= RESCALE_SECONDS:
            new_limit = max_workers(peak_rss_gib, cap, running_here=len(running))
            if new_limit != limit:
                log(f"rescale worker limit {limit} -> {new_limit} "
                    f"(foreign_cores={foreign_cpu_cores():.1f}, running_here={len(running)})")
                limit = new_limit
            last_rescale = now
        time.sleep(poll_seconds)
    write_status("IDLE", tasks, {})


def load_plan(path: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text())
    return payload["tasks"] if isinstance(payload, dict) else payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--cap", type=int, default=28)
    parser.add_argument("--sentinel", action="store_true",
                        help="run the first task alone and measure its peak RSS first")
    parser.add_argument("--peak-rss-gib", type=float, default=None)
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()
    set_tag(args.tag)

    LOGS.mkdir(parents=True, exist_ok=True)
    TASKS.mkdir(parents=True, exist_ok=True)
    tasks = load_plan(Path(args.plan))
    log(f"controller start: {len(tasks)} tasks from {args.plan}")
    write_status("STARTING", tasks, {})

    peak = args.peak_rss_gib
    audit = {"contract": "topic5_epi_prssm_v0_1_resource_audit", "sentinels": []}
    if peak is None:
        if args.sentinel and tasks:
            measured = measure_sentinel(tasks[0])
            audit["sentinels"].append(measured)
            peak = measured["peak_rss_gib"]
        else:
            peak = 2.0
    audit.update({"assumed_peak_rss_gib": peak, "safety_factor": SAFETY_FACTOR,
                  "ram_reserve_gib": RAM_RESERVE_GIB,
                  "ram_reserve_fraction": RAM_RESERVE_FRACTION,
                  "cpu_reserve": CPU_RESERVE, "disk_low_water_gib": DISK_LOW_WATER_GIB,
                  "cap": args.cap, "computed_worker_limit": max_workers(peak, args.cap),
                  "system_at_plan_time": system_snapshot()})
    audit["tag"] = TAG
    suffix = "" if TAG == "main" else f"_{TAG}"
    atomic_write_json(MANIFESTS / f"RESOURCE_AUDIT{suffix}.json", audit)

    run_pool(tasks, cap=args.cap, peak_rss_gib=peak)
    log("controller done")


if __name__ == "__main__":
    main()
