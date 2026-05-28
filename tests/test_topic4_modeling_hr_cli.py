"""Tests for scripts/run_topic4_phase4_stage1_hr.py exit-code contract."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "scripts/run_topic4_phase4_stage1_hr.py", *args],
        cwd=str(cwd),
        env={"PYTHONPATH": str(cwd), **dict(__import__("os").environ)},
        capture_output=True, text=True, timeout=600,
    )


def test_cli_help_works():
    """CLI --help returns 0 and mentions key flags."""
    repo_root = Path(__file__).resolve().parents[1]
    proc = _run_cli(["--help"], repo_root)
    assert proc.returncode == 0
    assert "--mode" in proc.stdout
    assert "--output-dir" in proc.stdout


@pytest.mark.slow
def test_cli_no_baseline_exits_one(tmp_path):
    """CLI exits 1 when sweep produces no excitable baseline.

    Trigger by feeding a sweep grid that's all silent (very deep I).
    Outputs regime_summary.json with stage1_exit_contract_passed=false.
    """
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "stage1_no_baseline"
    proc = _run_cli(
        ["--mode", "synthetic-allsilent",
         "--output-dir", str(output_dir)],
        repo_root,
    )
    assert proc.returncode == 1, (
        f"Expected exit 1, got {proc.returncode}; stdout={proc.stdout}"
    )
    summary_path = output_dir / "regime_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert summary["stage1_exit_contract_passed"] is False
    assert summary["baseline"] is None
