"""Execution-layer contract for the patient-specific cohort supervisor.

The v2 run used one batch-level gate and processed subjects strictly in series,
so a ten-candidate generation could never use more than ten workers and the
memory check only ran between batches. v2.1 admits workers one at a time under a
global cap and a live memory floor, which is what lets several subjects run at
once without risking an out-of-memory kill.
"""
import json
import threading
from pathlib import Path

import pytest

from scripts.launch_topic4_patient_specific_field_cohort_v2 import WorkerAdmission

ROOT = Path(__file__).resolve().parents[1]
CONFIG_V2P1 = ROOT / "config/topic4_patient_specific_field_connectivity_cohort_v2p1.json"


def test_admission_never_exceeds_the_global_worker_cap():
    admission = WorkerAdmission(3, 0.0, 0.0, memory_reader=lambda: 1000.0)
    peak = 0
    live = 0
    condition = threading.Condition()
    barrier = threading.Event()

    def body():
        nonlocal peak, live
        with admission.slot():
            with condition:
                live += 1
                peak = max(peak, live)
                condition.notify_all()
            barrier.wait(timeout=10.0)
            with condition:
                live -= 1

    threads = [threading.Thread(target=body) for _ in range(12)]
    for thread in threads:
        thread.start()
    with condition:
        assert condition.wait_for(lambda: live == 3, timeout=10.0)
    barrier.set()
    for thread in threads:
        thread.join(timeout=15.0)
    assert peak == 3


def test_admission_waits_while_memory_sits_under_the_floor():
    """A worker must not be launched into a host that is already tight."""
    readings = [10.0, 12.0, 45.0]
    slept = []
    admission = WorkerAdmission(
        4, 30.0, 7.0,
        memory_reader=lambda: readings.pop(0) if readings else 45.0,
        sleeper=slept.append,
    )
    with admission.slot():
        pass
    assert slept == [7.0, 7.0]
    assert admission.launched == 1


def test_admission_reports_the_floor_it_enforces():
    payload = json.loads(CONFIG_V2P1.read_text())["execution"]
    admission = WorkerAdmission.from_config(payload)
    assert admission.max_workers == payload["max_workers"]
    assert admission.floor_gib == payload["worker_admission_memory_floor_gib"]
    assert admission.floor_gib >= 30.0
    assert payload["subject_concurrency"] >= 1


def test_slot_is_released_even_when_the_worker_raises():
    admission = WorkerAdmission(1, 0.0, 0.0, memory_reader=lambda: 1000.0)
    with pytest.raises(RuntimeError):
        with admission.slot():
            raise RuntimeError("worker failed")
    with admission.slot():
        pass
    assert admission.launched == 2


def test_admission_keeps_one_worker_of_headroom_and_staggers_launches():
    """Memory only drops after a launched worker allocates, so leave room."""
    readings = [35.0, 36.0, 45.0]
    slept = []
    admission = WorkerAdmission(
        8, 30.0, 5.0, headroom_gib=8.0, stagger_seconds=3.0,
        memory_reader=lambda: readings.pop(0) if readings else 60.0,
        sleeper=slept.append,
    )
    with admission.slot():
        pass
    # 35 GiB clears the 30 GiB floor but not floor plus one worker of headroom
    assert slept[:2] == [5.0, 5.0]
    assert slept[-1] == 3.0


def test_config_headroom_matches_the_measured_worker_footprint():
    payload = json.loads(CONFIG_V2P1.read_text())["execution"]
    admission = WorkerAdmission.from_config(payload)
    assert admission.headroom_gib == payload["estimated_memory_gib_per_worker"]
    assert admission.stagger_seconds == payload["worker_admission_stagger_seconds"]
