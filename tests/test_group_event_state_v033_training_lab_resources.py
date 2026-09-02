"""Task 8: resource snapshot, sentinel, concurrency planner, leases (R1-R5)."""

from __future__ import annotations

import json
import math

import pytest

from src.topic5_group_event_state.v033_training_lab.resources import (
    CPU_RESERVE_CORES,
    DEFAULT_LEASE,
    DISK_MIN_GIB,
    GPU_RESERVE_GIB,
    IOWAIT_MAX_PCT,
    RAM_RESERVE_FRACTION,
    RAM_RESERVE_MIN_GIB,
    SAFETY_FACTOR,
    plan_concurrency,
    read_supervisor_lease,
    run_sentinel,
    snapshot,
    write_agent_lease,
)


def _snap(**over):
    base = {"gpus": [{"index": 0, "total_gib": 24.0, "used_gib": 0.0, "free_gib": 24.0, "utilization_pct": 0},
                     {"index": 1, "total_gib": 24.0, "used_gib": 0.0, "free_gib": 24.0, "utilization_pct": 0}],
            "mem_total_gib": 250.0, "mem_available_gib": 218.0, "load1": 12.0, "cores": 80, "iowait_pct": 1.0,
            "disk_free_gib": 3000.0}
    base.update(over)
    return base


def _sentinel(**over):
    base = {"workload_class": "gpu_train_fixed_leaky", "uses_gpu": True, "peak_reserved_gib": 2.0,
            "peak_allocated_gib": 1.5, "rss_peak_gib": 3.0, "wall_seconds": 10.0, "threads": 1}
    base.update(over)
    return base


LEASE = {"max_workers": 100, "gpu_ids": [0, 1], "max_gpu_workers": 100, "threads_per_worker": 1, "lease_source": "test"}


def test_r1_r2_planner_takes_the_minimum_with_the_documented_reserves_and_names_the_binding_limit():
    plan = plan_concurrency(_snap(), _sentinel(), LEASE, pending=1000, threads=1)
    limits = plan["limits"]
    assert limits["gpu"] == 2 * math.floor((24.0 - GPU_RESERVE_GIB) / (SAFETY_FACTOR * 2.0)) == 16
    reserve = max(RAM_RESERVE_FRACTION * 250.0, RAM_RESERVE_MIN_GIB)
    assert limits["ram"] == math.floor((218.0 - reserve) / (SAFETY_FACTOR * 3.0)) == 44
    assert limits["cpu"] == math.floor((80 - CPU_RESERVE_CORES - 12.0) / 1) == 66
    assert plan["slots"] == 16 and plan["binding"] == "gpu"
    ram_bound = plan_concurrency(_snap(mem_available_gib=60.0), _sentinel(), LEASE, pending=1000, threads=1)
    assert ram_bound["binding"] == "ram" and ram_bound["slots"] == math.floor((60.0 - reserve) / 3.75)
    cpu_bound = plan_concurrency(_snap(load1=75.0), _sentinel(), LEASE, pending=1000, threads=1)
    assert cpu_bound["binding"] == "cpu" and cpu_bound["slots"] == 3
    lease_bound = plan_concurrency(_snap(), _sentinel(), {**LEASE, "max_workers": 2}, pending=1000, threads=1)
    assert lease_bound["binding"] == "lease" and lease_bound["slots"] == 2
    pending_bound = plan_concurrency(_snap(), _sentinel(), LEASE, pending=1, threads=1)
    assert pending_bound["slots"] == 1 and pending_bound["binding"] == "pending"
    cpu_class = plan_concurrency(_snap(), _sentinel(uses_gpu=False, peak_reserved_gib=0.0), LEASE, pending=1000)
    assert cpu_class["limits"]["gpu"] is None and cpu_class["binding"] in ("ram", "cpu")
    ceiling = plan_concurrency(_snap(), _sentinel(), LEASE, pending=1000, ceiling=3)
    assert ceiling["slots"] == 3 and ceiling["binding"] == "ceiling"
    own = plan_concurrency(_snap(load1=75.0), _sentinel(), LEASE, pending=1000, threads=1, my_running_threads=10)
    assert own["limits"]["cpu"] == 13


def test_r3_low_disk_or_high_iowait_forbids_new_jobs():
    disk = plan_concurrency(_snap(disk_free_gib=DISK_MIN_GIB - 1), _sentinel(), LEASE, pending=10)
    assert disk["slots"] == 0 and disk["binding"] == "disk"
    io = plan_concurrency(_snap(iowait_pct=IOWAIT_MAX_PCT + 5), _sentinel(), LEASE, pending=10)
    assert io["slots"] == 0 and io["binding"] == "iowait"


def test_r4_missing_supervisor_lease_falls_back_to_the_conservative_default(tmp_path):
    lease = read_supervisor_lease(tmp_path)
    assert lease["lease_source"] == "default_conservative" and lease["max_workers"] == DEFAULT_LEASE["max_workers"] == 2
    grant = tmp_path / "resource_leases" / "supervisor_grant_agent_b.json"
    grant.parent.mkdir(parents=True)
    grant.write_text(json.dumps({"max_workers": 6, "gpu_ids": [1], "max_gpu_workers": 3, "threads_per_worker": 2}))
    lease = read_supervisor_lease(tmp_path)
    assert lease["lease_source"] == str(grant) and lease["max_workers"] == 6 and lease["gpu_ids"] == [1]
    path = write_agent_lease(tmp_path, {"running_units": 1, "gpu_workers": {"1": 1}})
    payload = json.loads(path.read_text())
    assert payload["agent"] == "agent_b" and payload["pid"] > 0 and "heartbeat_epoch" in payload
    assert path == tmp_path / "resource_leases" / "agent_b.json"


def test_r5_sentinel_measures_a_cpu_workload_and_snapshot_reads_the_machine(tmp_path):
    def work():
        total = sum(i * i for i in range(200_000))
        return {"effective_batch": 123, "checksum": total}

    report = run_sentinel("cpu_t0", work, out_path=tmp_path / "sentinel.json", device="cpu")
    for key in ("workload_class", "uses_gpu", "peak_allocated_gib", "peak_reserved_gib", "rss_peak_gib",
                "wall_seconds", "io_read_bytes", "io_write_bytes", "threads", "effective_batch", "measured_epoch"):
        assert key in report, key
    assert report["uses_gpu"] is False and report["peak_allocated_gib"] == 0.0
    assert report["wall_seconds"] > 0 and report["rss_peak_gib"] > 0 and report["effective_batch"] == 123
    assert json.loads((tmp_path / "sentinel.json").read_text())["workload_class"] == "cpu_t0"
    snap = snapshot()
    for key in ("gpus", "mem_total_gib", "mem_available_gib", "load1", "cores", "iowait_pct", "disk_free_gib",
                "snapshot_epoch"):
        assert key in snap, key
    assert snap["cores"] >= 1 and snap["mem_total_gib"] > 0
