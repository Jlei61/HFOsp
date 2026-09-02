"""Resource snapshot, workload sentinels, concurrency planning and leases (design §9).

Contract clauses (plan Task 8):
  [R1] concurrency = minimum over pending / GPU / RAM / CPU / disk / iowait / lease / ceiling, binding named;
  [R2] GPU reserve 4 GiB per card and demand x1.25; RAM reserve max(20 %, 20 GiB) and demand x1.25;
       CPU reserve 2 logical cores plus the load that is not ours;
  [R3] free disk < 10 GiB or sustained iowait above the threshold -> zero new jobs;
  [R4] missing supervisor grant -> conservative default, labelled as such;
  [R5] a sentinel records peak allocated / reserved VRAM, host RSS, I/O bytes, wall time and effective batch.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import resource
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from .paths import V033_ROOT, atomic_write_json

GPU_RESERVE_GIB = 4.0
RAM_RESERVE_FRACTION = 0.20
RAM_RESERVE_MIN_GIB = 20.0
CPU_RESERVE_CORES = 2
DISK_MIN_GIB = 10.0
IOWAIT_MAX_PCT = 30.0
SAFETY_FACTOR = 1.25
GIB = float(1 << 30)
DEFAULT_LEASE: dict[str, Any] = {"max_workers": 0, "gpu_ids": [], "max_gpu_workers": 0, "threads_per_worker": 1,
                                 "lease_source": "fail_closed_no_active_grant", "active": False}
SUPERVISOR_GRANT_NAME = "supervisor_grant_agent_b.json"
AGENT_LEASE_NAME = "agent_b.json"


# ---------------------------------------------------------------- snapshot
def _nvidia_smi() -> list[dict[str, Any]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=20)
    except Exception:
        return []
    gpus = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpus.append({"index": int(parts[0]), "total_gib": float(parts[1]) / 1024.0, "used_gib": float(parts[2]) / 1024.0,
                     "free_gib": float(parts[3]) / 1024.0, "utilization_pct": float(parts[4])})
    return gpus


def _meminfo() -> tuple[float, float]:
    total = available = 0.0
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemTotal:"):
                total = float(line.split()[1]) * 1024.0 / GIB
            elif line.startswith("MemAvailable:"):
                available = float(line.split()[1]) * 1024.0 / GIB
    return total, available


def _cpu_times() -> tuple[float, float]:
    with open("/proc/stat") as handle:
        fields = handle.readline().split()[1:]
    values = [float(v) for v in fields]
    return values[4], sum(values)


def iowait_percent(interval: float = 0.2) -> float:
    io0, tot0 = _cpu_times()
    time.sleep(interval)
    io1, tot1 = _cpu_times()
    return 100.0 * (io1 - io0) / max(tot1 - tot0, 1e-9)


def snapshot(disk_path: Path | str | None = None) -> dict[str, Any]:
    total, available = _meminfo()
    path = Path(disk_path) if disk_path is not None else (V033_ROOT if V033_ROOT.exists() else Path("/data") if Path("/data").exists() else Path("/"))
    usage = shutil.disk_usage(str(path))
    return {"snapshot_epoch": time.time(), "gpus": _nvidia_smi(), "mem_total_gib": total, "mem_available_gib": available,
            "load1": float(os.getloadavg()[0]), "cores": int(os.cpu_count() or 1), "iowait_pct": iowait_percent(),
            "disk_free_gib": usage.free / GIB, "disk_path": str(path)}


# ---------------------------------------------------------------- sentinel
def _proc_io() -> tuple[int, int]:
    try:
        with open("/proc/self/io") as handle:
            data = dict(line.strip().split(":") for line in handle if ":" in line)
        return int(data.get("read_bytes", 0)), int(data.get("write_bytes", 0))
    except OSError:
        return 0, 0


def run_sentinel(workload_class: str, fn: Callable[[], Mapping[str, Any] | None], *, out_path: Path | None = None,
                 device: str | None = None) -> dict[str, Any]:
    """[R5] Run one non-empty workload and measure what a worker of this class costs."""

    uses_gpu = device is not None and str(device).startswith("cuda")
    torch = None
    if uses_gpu:
        import torch  # noqa: WPS433
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        torch.cuda.synchronize(torch.device(device))
    r0, w0 = _proc_io()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0 / GIB
    started = time.time()
    extra = fn() or {}
    if uses_gpu:
        torch.cuda.synchronize(torch.device(device))
    wall = time.time() - started
    r1, w1 = _proc_io()
    rss_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024.0 / GIB
    peak_alloc = float(torch.cuda.max_memory_allocated(torch.device(device))) / GIB if uses_gpu else 0.0
    peak_reserved = float(torch.cuda.max_memory_reserved(torch.device(device))) / GIB if uses_gpu else 0.0
    try:
        import torch as _t
        threads = int(_t.get_num_threads())
    except Exception:  # pragma: no cover
        threads = int(os.environ.get("OMP_NUM_THREADS", "1"))
    report = {"workload_class": workload_class, "uses_gpu": bool(uses_gpu), "device": device,
              "peak_allocated_gib": peak_alloc, "peak_reserved_gib": peak_reserved, "rss_peak_gib": rss_peak,
              "rss_before_gib": rss_before, "wall_seconds": wall, "io_read_bytes": r1 - r0, "io_write_bytes": w1 - w0,
              "threads": threads, "effective_batch": extra.get("effective_batch"), "measured_epoch": time.time(),
              "extra": {k: v for k, v in extra.items() if k != "effective_batch"}}
    if out_path is not None:
        atomic_write_json(Path(out_path), report)
    return report


# ----------------------------------------------------------------- planner
def plan_concurrency(snap: Mapping[str, Any], sentinel: Mapping[str, Any], lease: Mapping[str, Any], *, pending: int,
                     threads: int = 1, ceiling: int | None = None, my_running_threads: float = 0.0) -> dict[str, Any]:
    """[R1]-[R3] Slots = min over every limit; the binding limit is named."""

    limits: dict[str, int | None] = {"pending": int(pending)}
    if bool(sentinel.get("uses_gpu")) and float(sentinel.get("peak_reserved_gib") or 0.0) > 0.0:
        need = SAFETY_FACTOR * float(sentinel["peak_reserved_gib"])
        allowed = set(int(i) for i in lease.get("gpu_ids", []))
        total = 0
        for gpu in snap.get("gpus", []):
            if int(gpu["index"]) in allowed:
                total += max(int(math.floor((float(gpu["free_gib"]) - GPU_RESERVE_GIB) / need)), 0)
        limits["gpu"] = min(total, int(lease.get("max_gpu_workers", total)))
    else:
        limits["gpu"] = None
    reserve = max(RAM_RESERVE_FRACTION * float(snap["mem_total_gib"]), RAM_RESERVE_MIN_GIB)
    rss_need = SAFETY_FACTOR * max(float(sentinel.get("rss_peak_gib") or 0.0), 1e-3)
    limits["ram"] = max(int(math.floor((float(snap["mem_available_gib"]) - reserve) / rss_need)), 0)
    other_load = max(float(snap["load1"]) - float(my_running_threads), 0.0)
    limits["cpu"] = max(int(math.floor((int(snap["cores"]) - CPU_RESERVE_CORES - other_load) / max(int(threads), 1))), 0)
    limits["disk"] = 0 if float(snap["disk_free_gib"]) < DISK_MIN_GIB else None
    limits["iowait"] = 0 if float(snap["iowait_pct"]) > IOWAIT_MAX_PCT else None
    limits["lease"] = int(lease.get("max_workers", DEFAULT_LEASE["max_workers"]))
    limits["ceiling"] = None if ceiling is None else int(ceiling)
    order = ("disk", "iowait", "pending", "gpu", "ram", "cpu", "lease", "ceiling")
    slots, binding = None, None
    for name in order:
        value = limits[name]
        if value is None:
            continue
        if slots is None or value < slots:
            slots, binding = int(value), name
    return {"slots": max(int(slots or 0), 0), "binding": binding, "limits": limits,
            "reserves": {"gpu_gib_per_card": GPU_RESERVE_GIB, "ram_gib": reserve, "cpu_cores": CPU_RESERVE_CORES,
                         "disk_min_gib": DISK_MIN_GIB, "iowait_max_pct": IOWAIT_MAX_PCT, "safety_factor": SAFETY_FACTOR},
            "threads_per_worker": int(threads)}


# ------------------------------------------------------------------- leases
def read_supervisor_lease(shared_root: Path) -> dict[str, Any]:
    """[R4] Return launch capacity only for a valid, active, unexpired supervisor grant."""

    path = Path(shared_root) / "resource_leases" / SUPERVISOR_GRANT_NAME
    if not path.exists():
        return {**DEFAULT_LEASE, "reason": "missing supervisor grant"}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {**DEFAULT_LEASE, "lease_source": "fail_closed_unreadable_grant",
                "reason": "unreadable supervisor grant"}
    status = str(payload.get("status", ""))
    expires = payload.get("expires_at")
    try:
        expires_epoch = datetime.fromisoformat(str(expires)).astimezone(timezone.utc).timestamp()
    except (TypeError, ValueError):
        expires_epoch = float("-inf")
    active = status.startswith("ACTIVE") and expires_epoch > time.time()
    if not active:
        return {**DEFAULT_LEASE, "lease_source": str(path), "grant": payload,
                "reason": f"inactive or expired supervisor grant: {status or 'MISSING_STATUS'}"}
    gpu_ids = [int(v) for v in payload.get("gpu_ids", [])]
    lease = dict(DEFAULT_LEASE)
    lease.update({
        "max_workers": max(int(payload.get("max_workers", 0)), 0),
        "gpu_ids": gpu_ids,
        "max_gpu_workers": max(int(payload.get("max_jobs_per_gpu_before_sentinel_review", 1)), 0) * len(gpu_ids),
        "threads_per_worker": max(int(payload.get("threads_per_worker", 1)), 1),
        "active": True,
    })
    lease["lease_source"] = str(path)
    lease["grant"] = payload
    return lease


def write_agent_lease(shared_root: Path, payload: Mapping[str, Any]) -> Path:
    path = Path(shared_root) / "resource_leases" / AGENT_LEASE_NAME
    atomic_write_json(path, {"agent": "agent_b", "pid": os.getpid(), "pgid": os.getpgid(0), "heartbeat_epoch": time.time(),
                             **dict(payload)})
    return path
