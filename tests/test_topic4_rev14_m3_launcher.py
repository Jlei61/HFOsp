import json
from pathlib import Path

from scripts import launch_topic4_rev14_m3_canary as launcher
from scripts import monitor_topic4_rev14_m3_canary as monitor


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_rev14_m3_canary.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def test_launcher_is_dry_run_by_default():
    args = launcher._parse_args([
        "--config", str(CONFIG), "--expected-commit", "a" * 40,
    ])
    assert args.execute is False
    assert args.worker_cap == 9
    assert args.unit_prefix == monitor.DEFAULT_UNIT_PREFIX


def test_controller_command_is_systemd_nohup_single_threaded_and_executing(tmp_path):
    unit, command = launcher._monitor_command(
        config_path=CONFIG,
        expected_commit="b" * 40,
        artifact_root=ARTIFACT_ROOT,
        unit_prefix=monitor.DEFAULT_UNIT_PREFIX,
        worker_cap=9,
        log_path=tmp_path / "controller.log",
    )
    assert unit == "codex-t4-r14-m3-controller-bbbbbbbb"
    assert command[:2] == ["systemd-run", "--user"]
    assert "/usr/bin/nohup" in command
    assert str(launcher.MONITOR) in command
    assert command[command.index("--worker-cap") + 1] == "9"
    assert command[-1] == "--execute"
    for name in launcher.NUMERIC_ENV:
        assert f"--setenv={name}=1" in command
    assert not any("topic4-ps-cohort-v2p1" in token for token in command)


def test_launcher_dry_run_never_invokes_systemd(monkeypatch, capsys):
    config = json.loads(CONFIG.read_text())
    monkeypatch.setattr(monitor, "_resolve_commit", lambda value: "c" * 40)
    monkeypatch.setattr(monitor, "_require_clean_commit", lambda value: None)
    monkeypatch.setattr(
        monitor, "_load_contract", lambda *args: (config, {"candidates": []}),
    )
    calls = []
    monkeypatch.setattr(
        launcher.subprocess, "run", lambda *args, **kwargs: calls.append(args),
    )
    launcher.main([
        "--config", str(CONFIG), "--expected-commit", "HEAD",
        "--artifact-root", str(ARTIFACT_ROOT),
    ])
    assert calls == []
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "REV14_M3_CONTROLLER_DRY_RUN"
    assert payload["worker_cap"] == 9


def test_launcher_execute_invokes_only_the_controller(monkeypatch, tmp_path):
    config = json.loads(CONFIG.read_text())
    config["output_root"] = "output"
    monkeypatch.setattr(monitor, "_resolve_commit", lambda value: "d" * 40)
    monkeypatch.setattr(monitor, "_require_clean_commit", lambda value: None)
    monkeypatch.setattr(
        monitor, "_load_contract", lambda *args: (config, {"candidates": []}),
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append(command)

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    launcher.main([
        "--config", str(CONFIG), "--expected-commit", "HEAD",
        "--artifact-root", str(tmp_path), "--execute",
    ])
    assert len(calls) == 1
    assert calls[0][:2] == ["systemd-run", "--user"]
    assert str(launcher.MONITOR) in calls[0]
    assert str(monitor.WORKER) not in calls[0]


def test_launcher_rejects_non_rev14_or_protected_prefix():
    for prefix in ("topic4", "topic4-ps-cohort-v2p1", "codex-t4-r13"):
        try:
            monitor._validate_unit_prefix(prefix)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe prefix accepted: {prefix}")

