"""Task 9: request ingest, units, search driver, controller, worker, OOM/NaN/stale (K1-K9)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from src.topic5_group_event_state.v033_training_lab.queue import (
    EXIT_NAN,
    EXIT_OK,
    EXIT_OOM,
    Controller,
    SearchDriver,
    Unit,
    classify_exit,
    derive_nan_diagnostic_unit,
    detect_stale,
    execute_unit,
    ingest_requests,
    oom_backoff,
    write_agent_status,
)
from src.topic5_group_event_state.v033_training_lab.request import JobStatus

REGISTERED = ("count_profile",)
HEAD = "d" * 40


def _request(request_id="req_toy", kind="toy", **over):
    base = {
        "request_id": request_id,
        "schema_version": "v2", "sealed": False,
        "scientific_target": {"family": "S_N", "predictive_view": "S_N", "objective": "count_profile",
                              "bin_convention": "left_closed_right_open_[t+a,t+b)",
                              "bins_seconds": [[0, 300], [300, 900], [900, 1800]]},
        "input_view": {"kind": kind, "seed": 3, "subject": "epilepsiae_1146", "data_registry_key": "dev"} if kind in ("R0", "R1")
        else {"kind": kind, "seed": 3, "synthetic": {"beta": 1.0, "dispersion_r": 8.0, "generator_seed": 1, "noise_seed": 2}},
        "state_architecture": "fixed_leaky",
        "split_hash": "auto",
        "baseline_H": {"name": "H_mark", "hash": "b" * 64, "source": "provisional_local"},
        "endpoint_and_reduction": {"selection_phase": "inner_val", "metric": "nb_nll", "reduction": "mean_per_anchor"},
        "search_budget": {"n_configs": 2, "max_steps": 40, "rung_steps": [20, 40], "eta": 2, "seeds_low": 1,
                          "seeds_mid": 2, "seeds_final": 2, "n_final": 1, "validate_every": 10,
                          "space_restrict": {"schedule": ["constant"], "warmup_fraction": [0.0], "depth": [1],
                                             "width": [32], "dropout": [0.0], "scaling": ["zscore"]},
                          "max_batches": 1, "t0_tiny_steps": 60, "t0_probe_steps": 20},
        "seed_policy": {"base_seed": 11},
        "resource_ceiling": {"max_workers": 1, "gpu_ids": [], "vram_gib": 0, "ram_gib": 8, "threads": 1},
        "science_code_commit": HEAD,
        "input_hash": "auto",
        "requested_by": "agent_c",
    }
    base.update(over)
    return base


def _roots(tmp_path):
    shared = tmp_path / "shared"
    agent = tmp_path / "agent_b"
    (shared / "job_requests").mkdir(parents=True)
    return shared, agent


def test_k1_ingest_writes_one_status_per_request_and_is_idempotent(tmp_path):
    shared, agent = _roots(tmp_path)
    (shared / "job_requests" / "science_toy.json").write_text(json.dumps(_request()))
    (shared / "job_requests" / "science_human.json").write_text(json.dumps(_request("req_human", kind="R1")))
    (shared / "job_requests" / "science_bad.json").write_text(json.dumps({"request_id": "req_bad"}))
    seen = ingest_requests(shared, agent, registered=REGISTERED, release_present=False, head_commit=HEAD)
    by_id = {s["request_id"]: s["status"] for s in seen}
    assert by_id == {"req_toy": "PENDING", "req_human": "HELD_NO_RELEASE", "req_bad": "INVALID_REQUEST"}
    status_path = shared / "job_status" / "training_req_toy.json"
    first = json.loads(status_path.read_text())
    assert first["status"] == "PENDING" and first["owner"] == "agent_b" and "request_hash" in first
    again = ingest_requests(shared, agent, registered=REGISTERED, release_present=False, head_commit=HEAD)
    assert json.loads(status_path.read_text())["ingested_epoch"] == first["ingested_epoch"]
    assert {s["request_id"]: s["status"] for s in again} == by_id
    released = ingest_requests(shared, agent, registered=REGISTERED, release_present=True, head_commit=HEAD)
    human = [s for s in released if s["request_id"] == "req_human"][0]
    assert human["status"] == "INVALID_REQUEST" and any("auto" in r for r in human["reasons"])
    assert (agent / "requests" / "req_toy" / "request.json").exists()


def test_k3_exit_classification_and_oom_backoff_ladder():
    assert classify_exit(EXIT_OK, "") == JobStatus.COMPLETE.value
    assert classify_exit(EXIT_OOM, "") == JobStatus.OOM_RETRYABLE.value
    assert classify_exit(1, "RuntimeError: CUDA out of memory. Tried to allocate") == JobStatus.OOM_RETRYABLE.value
    assert classify_exit(EXIT_NAN, "") == JobStatus.NAN.value
    assert classify_exit(1, "boom") == JobStatus.FAILED.value
    params = {"recipe": {"arch": {"chunk_seconds": 3600.0}}}
    one = oom_backoff(params, attempt=1)
    assert one["recipe"]["arch"]["chunk_seconds"] == 1800.0 and one["checkpointing"] is False
    two = oom_backoff(one, attempt=2)
    assert two["checkpointing"] is True and two["recipe"]["checkpointing"] is True
    assert two["recipe"]["arch"]["chunk_seconds"] == 1800.0
    three = oom_backoff(two, attempt=3)
    assert three["checkpointing"] is True and three["recipe"]["arch"]["chunk_seconds"] == 450.0
    assert three["gradient_accumulation"].startswith("not_applicable")
    assert oom_backoff(three, attempt=4) is None


def test_k4_nan_derives_a_diagnostic_unit_with_halved_lr_and_amp_off():
    unit = Unit(unit_id="u1", unit_type="train", request_id="r", job_key="k",
                params={"recipe": {"lr": {"adapter_w": 1e-3, "encoder_weights": 2e-3}, "amp_encoder": True},
                        "seed": 1, "steps_budget": 20}, out_dir="/tmp/x")
    diag = derive_nan_diagnostic_unit(unit)
    assert diag.diagnostic_rerun is True and diag.derived_from == "u1" and diag.unit_id != "u1"
    assert diag.params["recipe"]["lr"] == {"adapter_w": 5e-4, "encoder_weights": 1e-3}
    assert diag.params["recipe"]["amp_encoder"] is False
    assert diag.out_dir != unit.out_dir


def test_k5_stale_requires_dead_pid_and_old_heartbeat():
    now = time.time()
    alive = {"status": "RUNNING", "pid": os.getpid(), "heartbeat_epoch": now - 5000}
    assert detect_stale(alive, now=now) is False
    dead_fresh = {"status": "RUNNING", "pid": 2 ** 22 + 12345, "heartbeat_epoch": now - 10}
    assert detect_stale(dead_fresh, now=now) is False
    dead_old = {"status": "RUNNING", "pid": 2 ** 22 + 12345, "heartbeat_epoch": now - 2000}
    assert detect_stale(dead_old, now=now) is True
    assert detect_stale({**dead_old, "status": "COMPLETE"}, now=now) is False


def test_k9_search_driver_walks_t0_rungs_seeds_and_card_over_in_process_units(tmp_path):
    shared, agent = _roots(tmp_path)
    (shared / "job_requests" / "science_toy.json").write_text(json.dumps(_request()))
    ingest_requests(shared, agent, registered=REGISTERED, release_present=False, head_commit=HEAD)
    request = json.loads((agent / "requests" / "req_toy" / "request.json").read_text())
    driver = SearchDriver(request, agent)
    executed: list[str] = []
    for _ in range(60):
        units = driver.next_units()
        if not units:
            break
        for unit in units:
            code = execute_unit(unit, device="cpu")
            assert code == EXIT_OK, unit.unit_type
            executed.append(unit.unit_type)
    state = driver.state
    assert state["phase"] == "done"
    assert executed[0] == "t0" and executed[-1] == "card"
    train_units = [u for u in executed if u == "train"]
    assert len(train_units) == 2 + 2                     # rung0: 2 configs x 1 seed; final: 1 config x 2 seeds
    card = json.loads(Path(state["card_path"]).read_text())
    assert card["evidence_label"] in ("TRAINING-ADEQUATE", "DIAGNOSTIC")
    assert card["request"]["request_id"] == "req_toy"
    assert state["incumbent"]["config_id"] and state["stop_reason"] == "max_batches"
    status = json.loads((shared / "job_status" / "training_req_toy.json").read_text())
    assert status["status"] == "COMPLETE" and status["card_path"] == state["card_path"]
    assert status["evidence_label"] == card["evidence_label"]
    assert "failure_classification" in status


def test_k2_k6_k7_controller_respects_slots_release_and_writes_the_status_page(tmp_path):
    shared, agent = _roots(tmp_path)
    (shared / "job_requests" / "science_toy.json").write_text(json.dumps(_request()))
    (shared / "job_requests" / "science_human.json").write_text(json.dumps(_request("req_human", kind="R1")))
    spawned: list[dict] = []

    def fake_spawner(unit_path, *, gpu, log_path):
        spawned.append({"unit": str(unit_path), "gpu": gpu})
        return {"pid": os.getpid(), "pgid": os.getpgid(0)}

    ctl = Controller(shared, agent, registered=REGISTERED, head_commit=HEAD, release_present=lambda: False,
                     spawner=fake_spawner, lease={"max_workers": 1, "gpu_ids": [], "max_gpu_workers": 0,
                                                  "threads_per_worker": 1, "lease_source": "test"},
                     results_index=tmp_path / "index")
    report = ctl.step()
    assert report["ingested"]["req_toy"] == "PENDING" and report["ingested"]["req_human"] == "HELD_NO_RELEASE"
    assert len(spawned) == 1 and report["spawned"] == 1 and report["slots"] == 1
    assert all("req_human" not in s["unit"] for s in spawned)
    second = ctl.step()
    assert second["spawned"] == 0 and second["running"] == 1          # lease of 1 worker is honoured
    page = json.loads((agent / "agent_b.status.json").read_text())
    for key in ("updated_epoch", "commit", "sealed_partition_opened", "release_present", "heartbeat_epoch",
                "counts", "resources", "lease", "requests", "next_batch_rationale", "controller_pid"):
        assert key in page, key
    assert page["sealed_partition_opened"] is False and page["counts"]["running"] == 1
    assert (tmp_path / "index" / "agent_b.status.json").exists()
    assert (shared / "resource_leases" / "agent_b.json").exists()


def test_k2_completed_unit_with_same_job_key_is_skipped(tmp_path):
    out = tmp_path / "unit_out"
    out.mkdir()
    (out / "unit_result.json").write_text(json.dumps({"job_key": "k", "status": "COMPLETE"}))
    unit = Unit(unit_id="u", unit_type="train", request_id="r", job_key="k", params={}, out_dir=str(out))
    assert execute_unit(unit, device="cpu") == EXIT_OK
    assert json.loads((out / "unit_result.json").read_text())["status"] == "COMPLETE"
    assert json.loads((out / "unit_status.json").read_text())["status"] == JobStatus.SKIPPED_EXISTING.value


def test_k8_no_pkill_anywhere_in_the_training_lab():
    root = Path(__file__).resolve().parents[1] / "src" / "topic5_group_event_state" / "v033_training_lab"
    for path in root.glob("*.py"):
        assert "pkill" not in path.read_text(), path


def test_k7_agent_status_page_is_atomic_and_mirrored(tmp_path):
    path = write_agent_status(tmp_path / "agent", tmp_path / "index", commit=HEAD, release_present=False,
                              counts={"pending": 1}, resources={}, lease={}, requests={}, next_batch_rationale="idle")
    page = json.loads(path.read_text())
    assert page["sealed_partition_opened"] is False and page["agent"] == "agent_b"
    assert (tmp_path / "index" / "agent_b.status.json").exists()
