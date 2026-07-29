import hashlib
import json
import pytest

import src.topic4_zm_phasec_resources as R


def test_parse_process_swap_kb():
    assert R.parse_process_swap_kb(
        "Name:\tpython\nVmRSS:\t100 kB\nVmSwap:\t0 kB\n"
    ) == 0
    assert R.parse_process_swap_kb("VmSwap:\t65 kB\n") == 65
    with pytest.raises(RuntimeError, match="VmSwap"):
        R.parse_process_swap_kb("VmRSS:\t100 kB\n")


def test_worker_swap_snapshot_and_zero_allowance(monkeypatch):
    values = {11: 0, 12: 4}
    monkeypatch.setattr(R, "process_swap_kb", lambda pid: values[pid])
    snapshot = R.worker_swap_snapshot([12, 11, 12])
    assert snapshot == {
        "worker_swap_kb_by_pid": {"11": 0, "12": 4},
        "worker_swap_unavailable_pids": [],
        "worker_swap_total_kb": 4,
        "worker_swap_max_kb": 4,
    }
    assert R.worker_swap_exceeded([11], allowed_bytes=0) is False
    assert R.worker_swap_exceeded([11, 12], allowed_bytes=0) is True
    assert R.worker_swap_exceeded(
        [11, 12], allowed_bytes=4 * 1024
    ) is False


def test_worker_swap_audit_retains_per_pid_samples_and_maxima():
    audit = {}
    R.update_worker_swap_audit(audit, {
        "worker_swap_kb_by_pid": {"11": 0, "12": 2},
        "worker_swap_total_kb": 2,
        "worker_swap_max_kb": 2,
    }, sampled_at=1.0)
    R.update_worker_swap_audit(audit, {
        "worker_swap_kb_by_pid": {"11": 1},
        "worker_swap_total_kb": 1,
        "worker_swap_max_kb": 1,
    }, sampled_at=2.0)
    assert audit["11"] == {
        "n_samples": 2,
        "first_sample_at": 1.0,
        "last_sample_at": 2.0,
        "observed_max_kb": 1,
    }
    assert audit["12"]["n_samples"] == 1
    assert audit["12"]["observed_max_kb"] == 2
    R.record_final_worker_swap(
        audit, pid=11, value_kb=3, sampled_at=3.0
    )
    assert audit["11"]["final_publish_swap_kb"] == 3
    assert audit["11"]["observed_max_kb"] == 3


def test_launch_token_audit_does_not_conflate_reused_pid():
    audit = {}
    for token, task in (("token-a", "task-a"), ("token-b", "task-b")):
        R.register_worker_swap_audit(
            audit,
            pid=11,
            task_key=task,
            run_id="run",
            launch_token=token,
            launched_at=0.5,
        )
        R.update_worker_swap_audit(
            audit,
            {
                "worker_swap_kb_by_pid": {"11": 0},
                "worker_swap_unavailable_pids": [],
                "worker_swap_total_kb": 0,
                "worker_swap_max_kb": 0,
            },
            sampled_at=1.0,
            audit_key_by_pid={"11": token},
        )
        R.record_final_worker_swap(
            audit,
            pid=11,
            launch_token=token,
            value_kb=0,
            sampled_at=2.0,
        )
    assert set(audit) == {"token-a", "token-b"}
    assert audit["token-a"]["task_key"] == "task-a"
    assert audit["token-b"]["task_key"] == "task-b"
    assert audit["token-a"]["n_samples"] == audit["token-b"]["n_samples"] == 1


def test_reused_pid_new_launch_cannot_inherit_old_live_sample():
    audit = {}
    R.register_worker_swap_audit(
        audit, pid=11, task_key="old", run_id="run",
        launch_token="old-token", launched_at=0.0,
    )
    R.update_worker_swap_audit(
        audit,
        {
            "worker_swap_kb_by_pid": {"11": 0},
            "worker_swap_unavailable_pids": [],
            "worker_swap_total_kb": 0,
            "worker_swap_max_kb": 0,
        },
        sampled_at=1.0,
        audit_key_by_pid={"11": "old-token"},
    )
    R.record_final_worker_swap(
        audit, pid=11, launch_token="old-token",
        value_kb=0, sampled_at=2.0,
    )
    R.register_worker_swap_audit(
        audit, pid=11, task_key="new", run_id="run",
        launch_token="new-token", launched_at=3.0,
    )
    R.record_final_worker_swap(
        audit, pid=11, launch_token="new-token",
        value_kb=0, sampled_at=4.0,
    )
    with pytest.raises(RuntimeError, match="live VmSwap sample"):
        R.build_resource_receipt(
            artifact_path="/tmp/missing",
            artifact_root="/tmp",
            artifact_sha256="a" * 64,
            manifest_sha256="m" * 64,
            task_key="new",
            run_id="run",
            launch_token="new-token",
            pid=11,
            audit_row=audit["new-token"],
            sampled_allowed_bytes=0,
        )


def test_exited_pid_is_unavailable_not_a_zero_live_sample(monkeypatch):
    monkeypatch.setattr(
        R, "process_swap_kb", lambda pid: None if pid == 12 else 0
    )
    snapshot = R.worker_swap_snapshot([11, 12])
    assert snapshot["worker_swap_kb_by_pid"] == {"11": 0}
    assert snapshot["worker_swap_unavailable_pids"] == ["12"]
    audit = {}
    R.update_worker_swap_audit(audit, snapshot, sampled_at=1.0)
    assert set(audit) == {"11"}


def test_process_swap_tolerates_proc_teardown_between_read_and_parse(
    monkeypatch,
):
    class VanishingStatus:
        def __init__(self):
            self.reads = 0

        def read_text(self, **_kwargs):
            self.reads += 1
            if self.reads == 1:
                return ""
            raise FileNotFoundError

        def exists(self):
            return self.reads < 2

    status = VanishingStatus()
    monkeypatch.setattr(R, "Path", lambda _value: status)
    assert R.process_swap_kb(12) is None
    assert status.reads == 2


@pytest.mark.parametrize("state", ("Z (zombie)", "X (dead)", "x (dead)"))
def test_process_swap_treats_terminal_task_as_exited_unavailable(
    monkeypatch, state,
):
    class TerminalStatus:
        def read_text(self, **_kwargs):
            return f"Name:\tpython\nState:\t{state}\n"

        def exists(self):
            return True

    monkeypatch.setattr(R, "Path", lambda _value: TerminalStatus())
    assert R.process_swap_kb(12) is None


def test_process_swap_still_fails_for_malformed_live_status(monkeypatch):
    class MalformedLiveStatus:
        def read_text(self, **_kwargs):
            return "Name:\tpython\nVmRSS:\t100 kB\n"

        def exists(self):
            return True

    monkeypatch.setattr(R, "Path", lambda _value: MalformedLiveStatus())
    with pytest.raises(RuntimeError, match="live process 12"):
        R.process_swap_kb(12)


def test_coordinator_identity_requires_both_environment_fields(monkeypatch):
    monkeypatch.delenv(R.COORDINATOR_RUN_ENV, raising=False)
    monkeypatch.delenv(R.COORDINATOR_TOKEN_ENV, raising=False)
    assert R.coordinator_identity_from_env() == {
        "coordinator_run_id": None,
        "coordinator_launch_token": None,
    }
    monkeypatch.setenv(R.COORDINATOR_RUN_ENV, "run")
    with pytest.raises(RuntimeError, match="incomplete"):
        R.coordinator_identity_from_env()
    monkeypatch.setenv(R.COORDINATOR_TOKEN_ENV, "token")
    assert R.coordinator_identity_from_env()["coordinator_launch_token"] == "token"


def test_resource_receipt_binds_artifact_and_resume_audit(tmp_path):
    artifact = tmp_path / "part.json"
    artifact.write_text(json.dumps({
        "status": "complete",
        "runtime_provenance": {
            "coordinator_run_id": "run",
            "coordinator_launch_token": "token",
            "self_pid_at_publish": 11,
            "self_vm_swap_kb_at_publish": 0,
        },
    }) + "\n", encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt = R.build_resource_receipt(
        artifact_path=artifact,
        artifact_root=tmp_path,
        artifact_sha256=artifact_sha,
        manifest_sha256="m" * 64,
        task_key="identity|s1",
        run_id="run",
        launch_token="token",
        pid=11,
        audit_row={
            "pid": 11,
            "task_key": "identity|s1",
            "coordinator_run_id": "run",
            "coordinator_launch_token": "token",
            "n_samples": 2,
            "first_sample_at": 1.0,
            "last_sample_at": 2.0,
            "observed_max_kb": 0,
            "final_publish_swap_kb": 0,
        },
        sampled_allowed_bytes=0,
    )
    path = R.resource_receipt_path(artifact)
    R.publish_resource_receipt_once(path, receipt)
    valid, reason, loaded = R.validate_resource_receipt(
        path,
        artifact_path=artifact,
        artifact_root=tmp_path,
        manifest_sha256="m" * 64,
        task_key="identity|s1",
    )
    assert valid and reason == "valid_resource_audit_receipt"
    assert loaded["evidence_scope"].startswith("periodic live")

    artifact.write_text(json.dumps({
        "status": "changed",
        "runtime_provenance": {
            "coordinator_run_id": "run",
            "coordinator_launch_token": "token",
            "self_pid_at_publish": 11,
            "self_vm_swap_kb_at_publish": 0,
        },
    }) + "\n", encoding="utf-8")
    valid, reason, _ = R.validate_resource_receipt(
        path,
        artifact_path=artifact,
        artifact_root=tmp_path,
        manifest_sha256="m" * 64,
        task_key="identity|s1",
    )
    assert not valid
    assert reason == "resource_audit_receipt_contract_mismatch"


def test_resource_receipt_rejects_artifact_coordinator_mismatch(tmp_path):
    artifact = tmp_path / "part.json"
    artifact.write_text(json.dumps({
        "status": "complete",
        "runtime_provenance": {
            "coordinator_run_id": "other-run",
            "coordinator_launch_token": "token",
            "self_pid_at_publish": 11,
            "self_vm_swap_kb_at_publish": 0,
        },
    }), encoding="utf-8")
    artifact_sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    receipt = R.build_resource_receipt(
        artifact_path=artifact,
        artifact_root=tmp_path,
        artifact_sha256=artifact_sha,
        manifest_sha256="m" * 64,
        task_key="task",
        run_id="run",
        launch_token="token",
        pid=11,
        audit_row={
            "pid": 11,
            "task_key": "task",
            "coordinator_run_id": "run",
            "coordinator_launch_token": "token",
            "n_samples": 1,
            "first_sample_at": 1.0,
            "last_sample_at": 1.0,
            "observed_max_kb": 0,
            "final_publish_swap_kb": 0,
        },
        sampled_allowed_bytes=0,
    )
    receipt_path = R.resource_receipt_path(artifact)
    R.publish_resource_receipt_once(receipt_path, receipt)
    assert R.validate_resource_receipt(
        receipt_path,
        artifact_path=artifact,
        artifact_root=tmp_path,
        manifest_sha256="m" * 64,
        task_key="task",
    )[:2] == (False, "resource_audit_receipt_contract_mismatch")


def test_resource_receipt_refuses_no_live_sample():
    with pytest.raises(RuntimeError, match="live VmSwap sample"):
        R.build_resource_receipt(
            artifact_path="/tmp/missing",
            artifact_root="/tmp",
            artifact_sha256="a" * 64,
            manifest_sha256="m" * 64,
            task_key="x",
            run_id="run",
            launch_token="token",
            pid=1,
            audit_row={
                "pid": 1,
                "task_key": "x",
                "coordinator_run_id": "run",
                "coordinator_launch_token": "token",
                "n_samples": 0,
                "observed_max_kb": 0,
                "final_publish_swap_kb": 0,
            },
            sampled_allowed_bytes=0,
        )


def test_signal_mask_helpers_block_and_restore(monkeypatch):
    calls = []
    monkeypatch.setattr(
        R.signal,
        "pthread_sigmask",
        lambda operation, values: calls.append((operation, values)) or {"old"},
    )
    previous = R.block_coordinator_termination_signals()
    R.restore_coordinator_signal_mask(previous)
    assert previous == {"old"}
    assert calls[0][0] == R.signal.SIG_BLOCK
    assert calls[0][1] == {R.signal.SIGHUP, R.signal.SIGTERM}
    assert calls[1] == (R.signal.SIG_SETMASK, {"old"})


def test_cleanup_terminates_only_live_owned_workers():
    class Process:
        def __init__(self, code=None):
            self.code = code
            self.terminated = False
            self.killed = False

        def poll(self):
            return self.code

        def terminate(self):
            self.terminated = True
            self.code = 0

        def wait(self, timeout=None):
            return self.code

        def kill(self):
            self.killed = True
            self.code = -9

    class Handle:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    live, exited = Process(), Process(0)
    first, second = Handle(), Handle()
    R.terminate_owned_workers([
        {"process": live, "handle": first},
        {"proc": exited, "handle": second},
    ])
    assert live.terminated and not live.killed
    assert not exited.terminated
    assert first.closed and second.closed
